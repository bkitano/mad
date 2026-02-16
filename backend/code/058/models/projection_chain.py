"""
Projection chain implementations for GLA-style linear RNN layers.

This module implements the input projection chain and output epilogue chain
in both unfused (separate kernels) and fused (single kernel) variants.

The GLA layer projection chain:
  Input:  x -> [W_Q, W_K, W_V, W_g, W_alpha] -> activations -> [Q, K', V, gate, alpha]
  Output: scan_output -> gate * scan_output -> W_O -> + residual

Proposal 058 hypothesis: Fusing these chains reduces HBM round-trips by ~50%
and achieves >30% wall-clock speedup for the projection chain.

Math (from proposal):
  Unfused:
    Q_raw = x @ W_Q          # Kernel 1 (GEMM)
    K_raw = x @ W_K          # Kernel 2 (GEMM)
    V     = x @ W_V          # Kernel 3 (GEMM)
    g_raw = x @ W_g          # Kernel 4 (GEMM)
    alpha_raw = x @ W_alpha  # Kernel 5 (GEMM)
    K' = normalize(K_raw)    # Kernel 6 (elementwise)
    gate = SiLU(g_raw)       # Kernel 7 (elementwise)
    alpha = sigmoid(alpha_raw)# Kernel 8 (elementwise)

  Fused:
    [Q_raw; K_raw; V; g_raw; alpha_raw] = x @ [W_Q; W_K; W_V; W_g; W_alpha]  # 1 GEMM
    Q, K', V, gate, alpha = fused_activations(...)                              # 1 kernel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnfusedProjectionChain(nn.Module):
    """
    Unfused input projection chain — 8 separate kernel launches.

    This mimics the current fla-org implementation where each projection
    is a separate GEMM kernel and each activation is a separate elementwise kernel.

    Each GEMM writes its output to HBM, then the activation kernel reads it back,
    applies the activation, and writes the result to HBM again.
    Total HBM traffic: ~2x the intermediate tensor sizes.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, n_state: int):
        """
        Args:
            d_model: Input/model dimension
            d_k: Key dimension per head
            d_v: Value dimension per head
            n_heads: Number of attention heads
            n_state: SSM state dimension (for alpha/decay)
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.n_state = n_state

        # Separate projection matrices (each is a separate GEMM kernel launch)
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_g = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_alpha = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor [B, T, d_model]

        Returns:
            Q: [B, T, H*d_k]
            K_norm: [B, T, H*d_k] (L2 normalized)
            V: [B, T, H*d_v]
            gate: [B, T, H*d_v] (SiLU activated)
            alpha: [B, T, H] (sigmoid activated)
        """
        # 5 separate GEMM kernel launches
        Q_raw = self.W_Q(x)        # Kernel 1: x @ W_Q -> HBM
        K_raw = self.W_K(x)        # Kernel 2: x @ W_K -> HBM
        V = self.W_V(x)            # Kernel 3: x @ W_V -> HBM
        g_raw = self.W_g(x)        # Kernel 4: x @ W_g -> HBM
        alpha_raw = self.W_alpha(x) # Kernel 5: x @ W_alpha -> HBM

        # 3 separate activation kernel launches (read from HBM, write back to HBM)
        K_norm = F.normalize(K_raw, dim=-1)  # Kernel 6: L2 normalize
        gate = F.silu(g_raw)                  # Kernel 7: SiLU activation
        alpha = torch.sigmoid(alpha_raw)      # Kernel 8: Sigmoid activation

        return Q_raw, K_norm, V, gate, alpha


class FusedProjectionChain(nn.Module):
    """
    Fused input projection chain — 2 kernel launches.

    Uses a single wide GEMM x @ [W_Q; W_K; W_V; W_g; W_alpha] to compute
    all projections at once, followed by a single fused activation kernel.

    This is what EVT epilogue fusion (CUTLASS 3.x) achieves:
    the activations are applied in the GEMM epilogue while results are
    still in registers, before writing to HBM.

    In PyTorch, we approximate this with a single Linear + single activation pass.
    HBM traffic: ~1x the intermediate tensor sizes (write once, no read-back for activations).
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, n_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.n_state = n_state

        # Output dimensions for each projection
        self.q_dim = n_heads * d_k
        self.k_dim = n_heads * d_k
        self.v_dim = n_heads * d_v
        self.g_dim = n_heads * d_v
        self.alpha_dim = n_heads

        self.total_out = self.q_dim + self.k_dim + self.v_dim + self.g_dim + self.alpha_dim

        # Single wide projection matrix — 1 GEMM kernel launch
        self.W_proj = nn.Linear(d_model, self.total_out, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor [B, T, d_model]

        Returns:
            Same as UnfusedProjectionChain
        """
        # Single GEMM: x @ [W_Q; W_K; W_V; W_g; W_alpha] (Kernel 1)
        proj = self.W_proj(x)  # [B, T, total_out]

        # Split into individual projections (no kernel launch — just pointer arithmetic)
        Q_raw = proj[..., :self.q_dim]
        K_raw = proj[..., self.q_dim:self.q_dim + self.k_dim]
        V = proj[..., self.q_dim + self.k_dim:self.q_dim + self.k_dim + self.v_dim]
        g_raw = proj[..., self.q_dim + self.k_dim + self.v_dim:self.q_dim + self.k_dim + self.v_dim + self.g_dim]
        alpha_raw = proj[..., -self.alpha_dim:]

        # Fused activations — ideally 1 kernel with EVT, but in PyTorch
        # these are still separate elementwise kernels unless torch.compile fuses them
        K_norm = F.normalize(K_raw, dim=-1)
        gate = F.silu(g_raw)
        alpha = torch.sigmoid(alpha_raw)

        return Q_raw, K_norm, V, gate, alpha


class FullyFusedProjectionChain(nn.Module):
    """
    Fully fused projection chain using torch.compile for automatic kernel fusion.

    torch.compile can fuse the elementwise operations (normalize, SiLU, sigmoid)
    into a single kernel, approximating what CUTLASS EVT epilogue fusion achieves.

    This represents the best-case PyTorch-level fusion without custom CUDA.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, n_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.n_state = n_state

        self.q_dim = n_heads * d_k
        self.k_dim = n_heads * d_k
        self.v_dim = n_heads * d_v
        self.g_dim = n_heads * d_v
        self.alpha_dim = n_heads

        self.total_out = self.q_dim + self.k_dim + self.v_dim + self.g_dim + self.alpha_dim

        # Single wide projection
        self.W_proj = nn.Linear(d_model, self.total_out, bias=False)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def _fused_activations(self, proj: torch.Tensor) -> tuple:
        """
        Fused activation kernel — torch.compile should merge these into 1 kernel.
        """
        Q_raw = proj[..., :self.q_dim]
        K_raw = proj[..., self.q_dim:self.q_dim + self.k_dim]
        V = proj[..., self.q_dim + self.k_dim:self.q_dim + self.k_dim + self.v_dim]
        g_raw = proj[..., self.q_dim + self.k_dim + self.v_dim:self.q_dim + self.k_dim + self.v_dim + self.g_dim]
        alpha_raw = proj[..., -self.alpha_dim:]

        K_norm = F.normalize(K_raw, dim=-1)
        gate = F.silu(g_raw)
        alpha = torch.sigmoid(alpha_raw)

        return Q_raw, K_norm, V, gate, alpha

    def forward(self, x: torch.Tensor) -> tuple:
        proj = self.W_proj(x)
        return self._fused_activations(proj)


class UnfusedOutputChain(nn.Module):
    """
    Unfused output epilogue chain — 3 separate kernel launches.

    Output chain: scan_output * gate -> W_O -> + residual

    Each step reads from / writes to HBM separately.
    """

    def __init__(self, d_model: int, d_v: int, n_heads: int):
        super().__init__()
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scan_output: torch.Tensor, gate: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scan_output: [B, T, H*d_v] — output from chunkwise scan
            gate: [B, T, H*d_v] — SiLU(g_raw) from input chain
            residual: [B, T, d_model] — input for skip connection

        Returns:
            output: [B, T, d_model]
        """
        # Kernel 1: Elementwise gating (read scan_output + gate from HBM, write to HBM)
        gated = scan_output * gate

        # Kernel 2: Output projection GEMM (read gated from HBM, write to HBM)
        projected = self.W_O(gated)

        # Kernel 3: Residual + LayerNorm (read projected + residual, write output)
        output = self.norm(projected + residual)

        return output


class FusedOutputChain(nn.Module):
    """
    Fused output epilogue chain — ideally 1 kernel launch.

    With EVT/DSM fusion: gate is loaded from DSM (or registers),
    gating is applied as GEMM prologue, residual+norm is EVT epilogue.

    In PyTorch, torch.compile may fuse the gating + residual + norm.
    """

    def __init__(self, d_model: int, d_v: int, n_heads: int):
        super().__init__()
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scan_output: torch.Tensor, gate: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        # Same computation, but structured for fusion
        # In a true CUTLASS implementation, the gating would be a GEMM prologue
        # and residual+norm would be the EVT epilogue
        gated = scan_output * gate
        projected = self.W_O(gated)
        output = self.norm(projected + residual)
        return output

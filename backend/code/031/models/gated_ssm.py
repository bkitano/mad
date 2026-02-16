"""
Gated SSM Model (Mamba-2 / GLA style) with VNM-Sparse Projections

Implements the gated chunkwise SSM from proposal 031. The key insight is that
projection matrices (W_Q, W_K, W_V, W_gate, W_O) dominate ~75% of FLOPs,
so applying VNM sparsity to these gives real speedup.

Mathematical formulation (proposal):
  Q_t = x_t W_Q,  K_t = x_t W_K,  V_t = x_t W_V
  g_t = Swish(x_t W_gate)
  h_t = diag(alpha_t) h_{t-1} + K_t^T V_t   (state update via scan)
  o_t = Q_t h_t                               (readout)
  y_t = (o_t * g_t) W_O                       (gated output projection)

For MVE, we use a simplified recurrent scan (not chunked) since
we're testing projection sparsity quality, not kernel speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .vnm_sparse_linear import VNMSparseLinear


class GatedSSMLayer(nn.Module):
    """
    Single Gated SSM layer with VNM-sparse projections.

    Args:
        d_model: Model dimension
        d_head: Per-head dimension (d_k = d_v)
        n_heads: Number of attention heads
        state_dim: State dimension per head
        vnm_M: VNM block width (0=dense, 4=2:4, 6=V:2:6, 8=V:2:8)
        use_gate: Whether to use SwiGLU gating (True) or no gating
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        d_head: int = 32,
        n_heads: int = 4,
        state_dim: int = 16,
        vnm_M: int = 8,
        use_gate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.state_dim = state_dim
        self.d_inner = d_head * n_heads  # Total inner dimension
        self.use_gate = use_gate

        # Adjust V for small matrices (V should be <= out_features)
        V = min(64, d_model)

        # Projection matrices (proposal: these are the targets for VNM sparsity)
        # W_Q, W_K: d_model -> d_inner (for multi-head Q, K)
        # W_V: d_model -> d_inner
        # W_gate: d_model -> d_inner (for SwiGLU gating)
        # W_O: d_inner -> d_model (output projection)
        self.W_Q = VNMSparseLinear(d_model, self.d_inner, M=vnm_M, V=V)
        self.W_K = VNMSparseLinear(d_model, self.d_inner, M=vnm_M, V=V)
        self.W_V = VNMSparseLinear(d_model, self.d_inner, M=vnm_M, V=V)
        self.W_O = VNMSparseLinear(self.d_inner, d_model, M=vnm_M, V=V)

        if use_gate:
            self.W_gate = VNMSparseLinear(d_model, self.d_inner, M=vnm_M, V=V)

        # State transition: learnable decay per head per state dim
        # alpha_t = sigmoid(alpha_logit) for stability
        self.alpha_logit = nn.Parameter(torch.randn(n_heads, state_dim) * 0.1)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with recurrent state scan.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # Project Q, K, V (proposal eqs. 1-3)
        Q = self.W_Q(x)  # (B, T, d_inner)
        K = self.W_K(x)  # (B, T, d_inner)
        V = self.W_V(x)  # (B, T, d_inner)

        # Reshape to multi-head: (B, T, n_heads, d_head)
        Q = Q.view(batch, seq_len, self.n_heads, self.d_head)
        K = K.view(batch, seq_len, self.n_heads, self.d_head)
        V = V.view(batch, seq_len, self.n_heads, self.d_head)

        # Compute gating (proposal eq. 4): g_t = Swish(x_t W_gate)
        if self.use_gate:
            gate = F.silu(self.W_gate(self.norm(residual)))  # (B, T, d_inner)
            gate = gate.view(batch, seq_len, self.n_heads, self.d_head)

        # State decay factor
        alpha = torch.sigmoid(self.alpha_logit)  # (n_heads, state_dim)

        # Recurrent scan (proposal eq. 5):
        # h_t = diag(alpha) h_{t-1} + K_t^T V_t
        # For efficiency in MVE, we use a simplified version:
        # h[head] is (state_dim, d_head) — maps from state to output per head
        # We treat K as state_dim-dimensional (project d_head -> state_dim)
        # For simplicity, use K directly truncated/padded to state_dim

        # Simplified: use K[:state_dim] as key into state, V as value
        # This is a simplification — full Mamba-2 uses more complex state update
        h = torch.zeros(batch, self.n_heads, self.state_dim, self.d_head,
                        device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # Get current Q, K, V for this timestep
            q_t = Q[:, t]  # (B, n_heads, d_head)
            k_t = K[:, t]  # (B, n_heads, d_head)
            v_t = V[:, t]  # (B, n_heads, d_head)

            # Project K to state dimension (use first state_dim dims or linear proj)
            # Simplified: use first state_dim elements of k as "key"
            # k_state: (B, n_heads, state_dim)
            if self.d_head >= self.state_dim:
                k_state = k_t[:, :, :self.state_dim]
            else:
                # Pad if d_head < state_dim
                k_state = F.pad(k_t, (0, self.state_dim - self.d_head))

            # State update: h_t = alpha * h_{t-1} + k_state^T @ v_t
            # alpha: (n_heads, state_dim) -> broadcast to (1, n_heads, state_dim, 1)
            alpha_expanded = alpha.unsqueeze(0).unsqueeze(-1)  # (1, n_heads, state_dim, 1)
            h = alpha_expanded * h + torch.einsum('bhi,bhj->bhij', k_state, v_t)

            # Readout: o_t = Q_t @ h_t
            # q_state: (B, n_heads, state_dim)
            if self.d_head >= self.state_dim:
                q_state = q_t[:, :, :self.state_dim]
            else:
                q_state = F.pad(q_t, (0, self.state_dim - self.d_head))

            o_t = torch.einsum('bhi,bhij->bhj', q_state, h)  # (B, n_heads, d_head)
            outputs.append(o_t)

        # Stack outputs: (B, T, n_heads, d_head)
        o = torch.stack(outputs, dim=1)

        # Apply gating (proposal eq. 7): y_t = (o_t * g_t) W_O
        if self.use_gate:
            o = o * gate

        # Reshape back to (B, T, d_inner) and project output
        o = o.reshape(batch, seq_len, self.d_inner)
        y = self.W_O(o)  # (B, T, d_model)
        y = self.dropout(y)

        # Residual connection
        return residual + y

    def get_sparse_projections(self) -> dict:
        """Return all VNM-sparse projection layers for analysis."""
        projs = {
            'W_Q': self.W_Q,
            'W_K': self.W_K,
            'W_V': self.W_V,
            'W_O': self.W_O,
        }
        if self.use_gate:
            projs['W_gate'] = self.W_gate
        return projs


class GatedSSMModel(nn.Module):
    """
    Full model: Embedding + N GatedSSM layers + Classification head.

    For MQAR task: embeds tokens, processes with SSM layers, classifies output.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        d_head: Per-head dimension
        n_heads: Number of heads
        n_layers: Number of SSM layers
        state_dim: State dimension per head
        vnm_M: VNM sparsity config (0=dense, 4=2:4, 6=V:2:6, 8=V:2:8)
        use_gate: Whether to use SwiGLU gating
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 16,
        d_model: int = 128,
        d_head: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        state_dim: int = 16,
        vnm_M: int = 8,
        use_gate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (simple learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # SSM layers
        self.layers = nn.ModuleList([
            GatedSSMLayer(
                d_model=d_model,
                d_head=d_head,
                n_heads=n_heads,
                state_dim=state_dim,
                vnm_M=vnm_M,
                use_gate=use_gate,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Linear(d_model, vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practices."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape

        # Embed
        x = self.embedding(tokens)  # (B, T, d_model)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Process through SSM layers
        for layer in self.layers:
            x = layer(x)

        # Classify
        x = self.final_norm(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    def count_parameters(self) -> dict:
        """Count total and effective parameters."""
        total = sum(p.numel() for p in self.parameters())

        # Count effective (non-zero) params in sparse layers
        effective = 0
        for module in self.modules():
            if isinstance(module, VNMSparseLinear):
                with torch.no_grad():
                    W_sparse = module.get_sparse_weight()
                    effective += (W_sparse != 0).sum().item()
                if module.bias is not None:
                    effective += module.bias.numel()
            elif isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm)):
                effective += sum(p.numel() for p in module.parameters())
            # Skip nn.Parameter from parent modules (handled by specific modules)

        return {
            'total': total,
            'effective': effective,
            'sparsity': 1.0 - effective / total if total > 0 else 0.0,
        }

    def get_all_mask_flip_rates(self) -> dict:
        """Get mask flip rates for all sparse projection layers."""
        rates = {}
        for i, layer in enumerate(self.layers):
            for name, proj in layer.get_sparse_projections().items():
                if not proj.is_dense:
                    proj.update_mask_stats()
                    rates[f'layer{i}/{name}'] = proj.mask_flip_rate
        return rates

    def get_all_sparsities(self) -> dict:
        """Get actual sparsity levels for all sparse projection layers."""
        sparsities = {}
        for i, layer in enumerate(self.layers):
            for name, proj in layer.get_sparse_projections().items():
                sparsities[f'layer{i}/{name}'] = proj.actual_sparsity
        return sparsities

"""
Gated Linear Attention (GLA) with chunkwise forward pass.

Implements the GLA model from Yang et al. (ICML 2024) with:
- Chunkwise parallel computation (primary chunks)
- Secondary sub-chunking for inter/intra sub-chunk split
- Support for BF16 baseline and INT4+FP8 quantized variants

References:
- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training"
- Proposal 054: SageAttention2-Style INT4 Smoothing for Chunkwise Linear RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 8 for tensor core alignment
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def gla_chunkwise_forward(
    Q: torch.Tensor,      # (B, H, L, dk)
    K: torch.Tensor,      # (B, H, L, dk)
    V: torch.Tensor,      # (B, H, L, dv)
    gate: torch.Tensor,   # (B, H, L) — log gate values
    chunk_size: int = 128,
    sub_chunk_size: int = 16,
    mode: str = "bf16",    # "bf16", "int4_no_smooth", "int4_smooth", "int4_fp8"
) -> torch.Tensor:
    """
    GLA chunkwise forward pass with optional INT4/FP8 quantization.

    This implements the two-level chunking from GLA (Yang et al., 2024):
    1. Primary chunks of size C (chunk_size)
    2. Secondary sub-chunks of size c (sub_chunk_size) within each chunk

    For the intra-chunk computation:
    - Inter-sub-chunk matmuls (i > j): QK^T computed via matmul (quantizable)
    - Intra-sub-chunk blocks (i == j): Computed in FP32 with causal mask

    Args:
        Q, K, V: Query, Key, Value tensors
        gate: Log-space gate values (alpha_t = sigmoid(gate_t))
        chunk_size: Primary chunk size C
        sub_chunk_size: Secondary sub-chunk size c
        mode: Precision mode for inter-sub-chunk matmuls
    """
    from .quantization import int4_matmul_with_smoothing, fp8_matmul

    B, H, L, dk = Q.shape
    dv = V.shape[-1]
    C = chunk_size
    c = sub_chunk_size
    Ns = C // c  # Number of sub-chunks per chunk

    # Pad sequence length to multiple of chunk_size
    pad_len = (C - L % C) % C
    if pad_len > 0:
        Q = F.pad(Q, (0, 0, 0, pad_len))
        K = F.pad(K, (0, 0, 0, pad_len))
        V = F.pad(V, (0, 0, 0, pad_len))
        gate = F.pad(gate, (0, pad_len), value=0.0)

    L_padded = Q.shape[2]
    n_chunks = L_padded // C

    # Reshape into chunks: (B, H, n_chunks, C, d)
    Q_chunks = Q.reshape(B, H, n_chunks, C, dk)
    K_chunks = K.reshape(B, H, n_chunks, C, dk)
    V_chunks = V.reshape(B, H, n_chunks, C, dv)
    gate_chunks = gate.reshape(B, H, n_chunks, C)

    # Compute cumulative gate for causal masking within chunks
    # gate_cum[t] = sum_{s=0}^{t} gate[s] (log space)
    gate_cum = torch.cumsum(gate_chunks, dim=-1)  # (B, H, n_chunks, C)

    # Output accumulator
    O = torch.zeros(B, H, n_chunks, C, dv, device=Q.device, dtype=Q.dtype)

    # Process each chunk
    for chunk_idx in range(n_chunks):
        q_c = Q_chunks[:, :, chunk_idx]  # (B, H, C, dk)
        k_c = K_chunks[:, :, chunk_idx]  # (B, H, C, dk)
        v_c = V_chunks[:, :, chunk_idx]  # (B, H, C, dv)
        g_c = gate_cum[:, :, chunk_idx]  # (B, H, C)

        # Reshape into sub-chunks: (B, H, Ns, c, d)
        q_sc = q_c.reshape(B, H, Ns, c, dk)
        k_sc = k_c.reshape(B, H, Ns, c, dk)
        v_sc = v_c.reshape(B, H, Ns, c, dv)
        g_sc = g_c.reshape(B, H, Ns, c)

        # Compute gated Q and K for inter-sub-chunk matmuls
        # Q_tilde = Q * exp(gate_cum)  (data-dependent gating)
        # K_tilde = K * exp(-gate_cum)
        # This ensures Q_tilde @ K_tilde^T captures the causal gate decay
        g_last = g_sc[:, :, :, -1:]  # Last gate value in each sub-chunk
        g_first = g_sc[:, :, :, :1]  # First gate value in each sub-chunk

        # For inter-sub-chunk: apply gate absorption
        # Lambda_i = exp(g_cum[i*c:(i+1)*c] - g_cum[(i+1)*c-1])
        # Gamma_j = exp(g_cum[j*c:(j+1)*c] - g_cum[j*c])
        Lambda = torch.exp(g_sc - g_last)  # (B, H, Ns, c) — decay within sub-chunk from end
        Gamma = torch.exp(g_sc - g_first)  # (B, H, Ns, c) — decay within sub-chunk from start

        # Gated Q and K: Q_s = Q * Lambda, K_s = K * Gamma
        Q_s = q_sc * Lambda.unsqueeze(-1)  # (B, H, Ns, c, dk)
        K_s = k_sc * Gamma.unsqueeze(-1)   # (B, H, Ns, c, dk)

        # ===== INTRA-CHUNK COMPUTATION =====
        # Two parts:
        # 1. Inter-sub-chunk blocks (i > j): Q_s[i] @ K_s[j]^T — the target for INT4
        # 2. Intra-sub-chunk blocks (i == j): causal masked, computed in FP32

        o_chunk = torch.zeros(B, H, Ns, c, dv, device=Q.device, dtype=Q.dtype)

        for i in range(Ns):
            # === Inter-sub-chunk blocks (i > j) ===
            for j in range(i):
                Qi = Q_s[:, :, i]   # (B, H, c, dk)
                Kj = K_s[:, :, j]   # (B, H, c, dk)
                Vj = v_sc[:, :, j]  # (B, H, c, dv)

                # Inter-sub-chunk gate decay: exp(g_last[i] - g_last[j])
                inter_gate = torch.exp(
                    g_sc[:, :, i, -1:] - g_sc[:, :, j, -1:]
                ).unsqueeze(-1)  # (B, H, 1, 1)

                # P[i][j] = Q_s[i] @ K_s[j]^T — THIS IS THE INT4 TARGET
                if mode == "bf16":
                    P_ij = torch.matmul(Qi, Kj.transpose(-2, -1))  # (B, H, c, c)
                elif mode == "int4_no_smooth":
                    P_ij = int4_matmul_with_smoothing(Qi, Kj, smooth=False)
                elif mode in ("int4_smooth", "int4_fp8"):
                    P_ij = int4_matmul_with_smoothing(Qi, Kj, smooth=True)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                P_ij = P_ij * inter_gate

                # O[i] += P[i][j] @ V[j] — THIS IS THE FP8 TARGET
                if mode == "int4_fp8":
                    o_chunk[:, :, i] += fp8_matmul(P_ij, Vj)
                else:
                    o_chunk[:, :, i] += torch.matmul(P_ij, Vj)

            # === Intra-sub-chunk block (i == i): causal, computed in FP32 ===
            Qi = q_sc[:, :, i]   # (B, H, c, dk)
            Ki = k_sc[:, :, i]   # (B, H, c, dk)
            Vi = v_sc[:, :, i]   # (B, H, c, dv)

            # Causal mask within sub-chunk
            causal_mask = torch.tril(torch.ones(c, c, device=Q.device, dtype=Q.dtype))

            # Intra-sub-chunk gate mask
            g_intra = g_sc[:, :, i]  # (B, H, c)
            # D[m,n] = exp(g[m] - g[n]) for m >= n
            D = torch.exp(g_intra.unsqueeze(-1) - g_intra.unsqueeze(-2))  # (B, H, c, c)
            D = D * causal_mask  # Zero out non-causal entries

            # Attention: always FP32 for intra-sub-chunk (precision-sensitive)
            P_ii = torch.matmul(Qi.float(), Ki.float().transpose(-2, -1))  # (B, H, c, c)
            P_ii = P_ii * D.float()
            o_chunk[:, :, i] += torch.matmul(P_ii, Vi.float()).to(Q.dtype)

        # Reshape sub-chunks back to chunk
        O[:, :, chunk_idx] = o_chunk.reshape(B, H, C, dv)

    # Reshape back to sequence
    O = O.reshape(B, H, L_padded, dv)
    if pad_len > 0:
        O = O[:, :, :L, :]

    return O


class GLALayer(nn.Module):
    """
    Single GLA (Gated Linear Attention) layer.

    Implements the GLA attention mechanism with:
    - Q, K, V projections
    - Data-dependent gating (log-space)
    - Chunkwise parallel forward pass
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dk_per_head: int,
        dv_per_head: int,
        chunk_size: int = 128,
        sub_chunk_size: int = 16,
        mode: str = "bf16",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = dk_per_head
        self.dv = dv_per_head
        self.chunk_size = chunk_size
        self.sub_chunk_size = sub_chunk_size
        self.mode = mode

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * dk_per_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * dk_per_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * dv_per_head, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(n_heads * dv_per_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            mode: Override the default precision mode
        """
        B, L, _ = x.shape
        mode = mode or self.mode

        # Project
        Q = self.q_proj(x).reshape(B, L, self.n_heads, self.dk).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, L, self.n_heads, self.dk).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, L, self.n_heads, self.dv).permute(0, 2, 1, 3)

        # L2-normalize Q and K (standard in GLA)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Data-dependent gate (log-space)
        gate = F.logsigmoid(self.gate_proj(x))  # (B, L, H)
        gate = gate.permute(0, 2, 1)  # (B, H, L)

        # Chunkwise forward
        O = gla_chunkwise_forward(
            Q, K, V, gate,
            chunk_size=self.chunk_size,
            sub_chunk_size=self.sub_chunk_size,
            mode=mode,
        )

        # Output projection
        O = O.permute(0, 2, 1, 3).reshape(B, L, self.n_heads * self.dv)
        return self.out_proj(O)


class GLABlock(nn.Module):
    """GLA block with pre-norm, residual connections, and FFN."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dk_per_head: int,
        dv_per_head: int,
        chunk_size: int = 128,
        sub_chunk_size: int = 16,
        mode: str = "bf16",
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GLALayer(d_model, n_heads, dk_per_head, dv_per_head,
                             chunk_size, sub_chunk_size, mode)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mode=mode)
        x = x + self.ffn(self.norm2(x))
        return x


class GLAModel(nn.Module):
    """
    GLA Language Model for the copying task MVE.

    Architecture (from proposal 054 MVE):
    - 2 layers, d=256, dk=128, dv=256, 2 heads (~1M params)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 2,
        dk_per_head: int = 128,
        dv_per_head: int = 256,
        chunk_size: int = 128,
        sub_chunk_size: int = 16,
        mode: str = "bf16",
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = mode

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GLABlock(d_model, n_heads, dk_per_head, dv_per_head,
                     chunk_size, sub_chunk_size, mode)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token indices
            mode: Precision mode override
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, mode=mode)
        x = self.norm(x)
        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""
Gated Second-Order Linear Attention (GHLA)

From proposal 064: Combines HLA's second-order key metric S_t^K with
GLA's data-dependent diagonal gating.

Recurrence (proposal eq.):
  S_t^K = Diag(alpha_t^K) S_{t-1}^K Diag(alpha_t^K) + k_t k_t^T    [d_k x d_k]
  C_t^{QV} = Diag(alpha_t^C) C_{t-1}^{QV} + q_t v_t^T              [d_k x d_v]
  G_t = Diag(alpha_t^K) G_{t-1} + k_t (k_t^T Diag(alpha_t^C) C_{t-1}^{QV})  [d_k x d_v]
  o_t = q_t^T (S_t^K C_t^{QV} - G_t)                                [d_v]

Gates: alpha_t^K, alpha_t^C in (0,1)^{d_k} — data-dependent via sigmoid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GHLALayer(nn.Module):
    """Single GHLA attention layer with multi-head support."""

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        # Gate projections for alpha_t^K and alpha_t^C (proposal eq.)
        # Using single linear for simplicity in MVE (low-rank not needed at this scale)
        self.W_gate_K = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_gate_C = nn.Linear(d_model, n_heads * d_k, bias=True)

        # Output projection
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        # Gate temperature
        self.tau = 1.0

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stability."""
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        # Initialize gate biases to produce gates near 1.0 (slow forgetting initially)
        for m in [self.W_gate_K, self.W_gate_C]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 3.0)  # sigmoid(3) ~ 0.95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_k, self.d_v

        # Project Q, K, V: (B, T, H*dk) -> (B, H, T, dk)
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)

        # Compute data-dependent gates: (B, T, H*dk) -> (B, H, T, dk)
        alpha_K = torch.sigmoid(self.W_gate_K(x) / self.tau).view(B, T, H, dk).transpose(1, 2)
        alpha_C = torch.sigmoid(self.W_gate_C(x) / self.tau).view(B, T, H, dk).transpose(1, 2)

        # Recurrent computation (sequential — no chunkwise needed at MVE scale)
        S_K = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        C_QV = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        G = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]        # (B, H, dk)
            k_t = k[:, :, t, :]        # (B, H, dk)
            v_t = v[:, :, t, :]        # (B, H, dv)
            aK_t = alpha_K[:, :, t, :]  # (B, H, dk)
            aC_t = alpha_C[:, :, t, :]  # (B, H, dk)

            # 1. Gate existing states
            # Diag(a) S Diag(a) => element-wise: (a_i * a_j) * S_ij
            gate_outer = aK_t.unsqueeze(-1) * aK_t.unsqueeze(-2)  # (B, H, dk, dk)
            S_K_gated = gate_outer * S_K
            C_QV_prev = aC_t.unsqueeze(-1) * C_QV  # gated C_{t-1}

            # 2. Update G_t (uses C_{t-1}^{QV} before adding current q_t v_t^T)
            # G_t = Diag(aK_t) G_{t-1} + k_t (k_t^T C_QV_prev)
            k_t_C = torch.einsum('bhk,bhkv->bhv', k_t, C_QV_prev)  # (B, H, dv)
            G = aK_t.unsqueeze(-1) * G + k_t.unsqueeze(-1) * k_t_C.unsqueeze(-2)  # (B, H, dk, dv)

            # 3. Update S_K and C_QV
            S_K = S_K_gated + k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            C_QV = C_QV_prev + q_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # 4. Output: o_t = q_t^T (S_K C_QV - G)
            S_C = torch.matmul(S_K, C_QV)  # (B, H, dk, dv)
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, S_C - G)  # (B, H, dv)
            outputs.append(o_t)

        # Stack outputs: (B, H, T, dv) -> (B, T, H*dv)
        out = torch.stack(outputs, dim=2)  # (B, H, T, dv)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dv)

        return self.W_o(out)


class GHLABlock(nn.Module):
    """GHLA block with attention + FFN + LayerNorm."""

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, ffn_mult: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GHLALayer(d_model, d_k, d_v, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GHLAModel(nn.Module):
    """Full GHLA model for sequence-to-sequence tasks (e.g., MQAR)."""

    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int,
                 n_heads: int, n_layers: int, max_seq_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GHLABlock(d_model, d_k, d_v, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)

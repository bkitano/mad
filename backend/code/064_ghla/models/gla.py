"""
First-Order Gated Linear Attention (GLA) — Baseline 1

Standard GLA recurrence:
  S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T    [d_k x d_v]
  o_t = q_t^T S_t                              [d_v]

This is the first-order baseline. It has no second-order key metric S_t^K,
so the attention kernel is a fixed linear function of queries, not data-adaptive.
"""

import torch
import torch.nn as nn
import math


class GLALayer(nn.Module):
    """Single GLA (first-order) attention layer."""

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        # Data-dependent gate (same as GHLA but only one gate)
        self.W_gate = nn.Linear(d_model, n_heads * d_k, bias=True)

        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_gate.weight, gain=0.01)
        nn.init.constant_(self.W_gate.bias, 3.0)  # sigmoid(3) ~ 0.95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_k, self.d_v

        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)  # (B, H, T, dk)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        alpha = torch.sigmoid(self.W_gate(x)).view(B, T, H, dk).transpose(1, 2)

        # Recurrent: S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T
        S = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            a_t = alpha[:, :, t, :]

            S = a_t.unsqueeze(-1) * S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, S)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dv)
        return self.W_o(out)


class GLABlock(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, ffn_mult: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GLALayer(d_model, d_k, d_v, n_heads)
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


class GLAModel(nn.Module):
    """First-order GLA model for MQAR."""

    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int,
                 n_heads: int, n_layers: int, max_seq_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GLABlock(d_model, d_k, d_v, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

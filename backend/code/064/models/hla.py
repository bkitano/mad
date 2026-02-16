"""
Higher-Order Linear Attention (HLA) — Baselines 2 & 3

Baseline 2 - HLA (ungated):
  S_t^K = S_{t-1}^K + k_t k_t^T               [d_k x d_k]
  C_t^{QV} = C_{t-1}^{QV} + q_t v_t^T         [d_k x d_v]
  G_t = G_{t-1} + k_t (k_t^T C_{t-1}^{QV})    [d_k x d_v]
  o_t = q_t^T (S_t^K C_t^{QV} - G_t)          [d_v]

Baseline 3 - HLA with fixed decay gamma=0.99:
  S_t^K = gamma * S_{t-1}^K + k_t k_t^T
  C_t^{QV} = gamma * C_{t-1}^{QV} + q_t v_t^T
  G_t = gamma * G_{t-1} + k_t (k_t^T (gamma * C_{t-1}^{QV}))
  o_t = q_t^T (S_t^K C_t^{QV} - G_t)
"""

import torch
import torch.nn as nn
import math


class HLALayer(nn.Module):
    """Second-order linear attention WITHOUT gating (ungated HLA)."""

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_k, self.d_v

        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)

        S_K = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        C_QV = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        G = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            # Update G first (uses C_{t-1})
            k_C = torch.einsum('bhk,bhkv->bhv', k_t, C_QV)  # (B, H, dv)
            G = G + k_t.unsqueeze(-1) * k_C.unsqueeze(-2)

            # Update S_K and C_QV
            S_K = S_K + k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            C_QV = C_QV + q_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Output
            S_C = torch.matmul(S_K, C_QV)
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, S_C - G)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dv)
        return self.W_o(out)


class HLADecayLayer(nn.Module):
    """Second-order linear attention WITH fixed decay gamma=0.99."""

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, gamma: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.gamma = gamma

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_k, self.d_v
        gamma = self.gamma

        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)

        S_K = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        C_QV = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        G = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            # Update G first (uses C_{t-1} with decay)
            C_QV_prev = gamma * C_QV  # gated previous C_QV
            k_C = torch.einsum('bhk,bhkv->bhv', k_t, C_QV_prev)
            G = gamma * G + k_t.unsqueeze(-1) * k_C.unsqueeze(-2)

            # Update S_K and C_QV with fixed decay
            S_K = gamma * S_K + k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            C_QV = C_QV_prev + q_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Output
            S_C = torch.matmul(S_K, C_QV)
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, S_C - G)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dv)
        return self.W_o(out)


class HLABlock(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int,
                 ffn_mult: float = 2.0, use_decay: bool = False, gamma: float = 0.99):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if use_decay:
            self.attn = HLADecayLayer(d_model, d_k, d_v, n_heads, gamma=gamma)
        else:
            self.attn = HLALayer(d_model, d_k, d_v, n_heads)
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


class HLAModel(nn.Module):
    """Ungated HLA model."""

    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int,
                 n_heads: int, n_layers: int, max_seq_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            HLABlock(d_model, d_k, d_v, n_heads, use_decay=False) for _ in range(n_layers)
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


class HLADecayModel(nn.Module):
    """HLA with fixed decay gamma=0.99."""

    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int,
                 n_heads: int, n_layers: int, max_seq_len: int = 256, gamma: float = 0.99):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            HLABlock(d_model, d_k, d_v, n_heads, use_decay=True, gamma=gamma)
            for _ in range(n_layers)
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

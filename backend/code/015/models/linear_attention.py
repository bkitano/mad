"""
Linear Attention baseline for Experiment 006.

Standard linear attention (RetNet-style) using the (R, +, x) semiring.
This should achieve < 80% on MQAR due to state dilution from additive aggregation.

Linear attention: S_t = gamma * S_{t-1} + phi(k_t) v_t^T  (additive, lossy)
vs Tropical SSM: l_t = max(a_t + l_{t-1}, b_t)            (hard max, lossless)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LinearAttentionLayer(nn.Module):
    """
    Standard causal linear attention with ELU+1 feature map and learnable decay.
    
    S_t = gamma * S_{t-1} + phi(k_t) outer v_t
    z_t = gamma * z_{t-1} + phi(k_t)
    y_t = S_t^T phi(q_t) / (z_t^T phi(q_t) + eps)
    """
    def __init__(self, d_model, n_heads=4, d_k=16, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.decay_log = nn.Parameter(torch.zeros(n_heads))
        self.W_g = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def feature_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        batch, seq_len, d = x.shape
        H, dk = self.n_heads, self.d_k

        q = self.W_Q(x).view(batch, seq_len, H, dk)
        k = self.W_K(x).view(batch, seq_len, H, dk)
        v = self.W_V(x).view(batch, seq_len, H, dk)

        q = self.feature_map(q)
        k = self.feature_map(k)
        gamma = torch.sigmoid(self.decay_log)

        S = torch.zeros(batch, H, dk, dk, device=x.device)
        z = torch.zeros(batch, H, dk, device=x.device)
        outputs = torch.zeros(batch, seq_len, H, dk, device=x.device)

        for t in range(seq_len):
            kt = k[:, t, :, :]
            vt = v[:, t, :, :]
            qt = q[:, t, :, :]

            g = gamma.view(1, H, 1, 1)
            gz = gamma.view(1, H, 1)

            S = g * S + torch.einsum('bhi,bhj->bhij', kt, vt)
            z = gz * z + kt

            numerator = torch.einsum('bhij,bhi->bhj', S, qt)
            denominator = (z * qt).sum(dim=-1, keepdim=True) + 1e-6
            outputs[:, t, :, :] = numerator / denominator

        gate = torch.sigmoid(self.W_g(x)).view(batch, seq_len, H, dk)
        y = outputs * gate
        y = y.reshape(batch, seq_len, H * dk)
        y = self.W_O(y)
        y = self.dropout(y)
        return y


class LinearAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, d_k=16, d_ff=None, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LinearAttentionLayer(d_model, n_heads, d_k, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_k=16,
                 num_layers=2, num_classes=None, max_seq_len=256,
                 d_ff=None, dropout=0.1):
        super().__init__()
        if num_classes is None:
            num_classes = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            LinearAttentionBlock(d_model, n_heads, d_k, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, num_classes),
        )

    def forward(self, x):
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)

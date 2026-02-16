"""
Tropical-Gated SSM (TG-SSM) from Proposal 015.

Core idea: Replace the standard (R, +, x) semiring in SSM state recurrences
with the tropical semiring (R u {-inf}, max, +), using input-dependent gating
and a smooth annealing schedule from log-semiring to hard tropical.

Key equations:
  Scalar tropical recurrence:
    l_t = max(a_t + l_{t-1}, b_t)

  Annealed log-to-tropical:
    l_t^(mu) = (1/mu) * log(exp(mu(a_t + l_{t-1})) + exp(mu * b_t))

  Where:
    a_t = -softplus(W_a x_t + c_a) in (-inf, 0]  (input-dependent decay)
    b_t = q_t^T k_t / sqrt(d_k)                    (input bid / score)

  Parallel scan operator:
    (a1, l1) x (a2, l2) = (a1 + a2, max(a2 + l1, l2))
    Identity: (0, -inf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
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


def tropical_scan_sequential(a, b, mu=1.0):
    """
    Sequential tropical scan with log-semiring annealing.

    Computes:
      l_t = (1/mu) * log(exp(mu(a_t + l_{t-1})) + exp(mu * b_t))
    As mu -> inf, converges to max(a_t + l_{t-1}, b_t).

    Args:
        a: Decay values, shape (batch, seq_len, heads). Must be <= 0.
        b: Bid/score values, shape (batch, seq_len, heads).
        mu: Temperature. Higher = harder max.

    Returns:
        ell: Hidden states, shape (batch, seq_len, heads).
        weights: Attention weights, shape (batch, seq_len, heads, seq_len).
    """
    batch, seq_len, heads = a.shape
    device = a.device

    ell = torch.zeros(batch, seq_len, heads, device=device)
    # Track cumulative decayed scores for value retrieval
    scores = torch.full((batch, seq_len, heads, seq_len), -1e9, device=device)

    # Position 0
    ell[:, 0, :] = b[:, 0, :]
    scores[:, 0, :, 0] = b[:, 0, :]

    for t in range(1, seq_len):
        # Decay all previous scores by a_t
        scores[:, t, :, :t] = scores[:, t - 1, :, :t] + a[:, t, :].unsqueeze(-1)
        # Current bid
        scores[:, t, :, t] = b[:, t, :]
        # Compute l_t via log-semiring
        active_scores = scores[:, t, :, :t + 1]
        ell[:, t, :] = (1.0 / mu) * torch.logsumexp(mu * active_scores, dim=-1)

    # Compute attention weights: softmax_mu over scores
    weights = torch.zeros(batch, seq_len, heads, seq_len, device=device)
    for t in range(seq_len):
        active_scores = scores[:, t, :, :t + 1]
        w = F.softmax(mu * active_scores, dim=-1)
        weights[:, t, :, :t + 1] = w

    return ell, weights


class TropicalSSMLayer(nn.Module):
    """
    Single Tropical-Gated SSM layer.

    Per-head computation:
      1. Project: q, k, v = x W_Q, x W_K, x W_V
      2. Bid: b_t = q_t^T k_t / sqrt(d_k)
      3. Decay: a_t = -softplus(x_t W_a + c_a) <= 0
      4. Tropical scan (annealed)
      5. Retrieve: y_t = softmax_mu(scores) . V
      6. Gate: y_t = y_t * sigmoid(x_t W_g)
    """
    def __init__(self, d_model, n_heads=4, d_k=16, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_a = nn.Linear(d_model, n_heads, bias=True)
        self.W_g = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_a.bias)

    def forward(self, x, mu=1.0):
        batch, seq_len, d = x.shape
        H, dk = self.n_heads, self.d_k

        q = self.W_Q(x).view(batch, seq_len, H, dk)
        k = self.W_K(x).view(batch, seq_len, H, dk)
        v = self.W_V(x).view(batch, seq_len, H, dk)

        # Bid: q^T k / sqrt(d_k)
        b = (q * k).sum(dim=-1) / math.sqrt(dk)
        # Decay: -softplus ensures a_t <= 0
        a = -F.softplus(self.W_a(x))

        # Tropical scan
        _, weights = tropical_scan_sequential(a, b, mu=mu)

        # Retrieve values: weights (B, T, H, T) @ v (B, T, H, dk)
        weights_bh = weights.permute(0, 2, 1, 3).reshape(batch * H, seq_len, seq_len)
        v_bh = v.permute(0, 2, 1, 3).reshape(batch * H, seq_len, dk)
        y = torch.bmm(weights_bh, v_bh)
        y = y.view(batch, H, seq_len, dk).permute(0, 2, 1, 3)

        # Gate output
        gate = torch.sigmoid(self.W_g(x)).view(batch, seq_len, H, dk)
        y = y * gate

        y = y.reshape(batch, seq_len, H * dk)
        y = self.W_O(y)
        y = self.dropout(y)
        return y


class TropicalSSMBlock(nn.Module):
    """Pre-norm residual block: TropicalSSM + SwiGLU FFN."""
    def __init__(self, d_model, n_heads=4, d_k=16, d_ff=None, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.tropical_ssm = TropicalSSMLayer(d_model, n_heads, d_k, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, mu=1.0):
        x = x + self.tropical_ssm(self.norm1(x), mu=mu)
        x = x + self.ffn(self.norm2(x))
        return x


class TropicalSSMClassifier(nn.Module):
    """Full TG-SSM model for MQAR classification."""
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_k=16,
                 num_layers=2, num_classes=None, max_seq_len=256,
                 d_ff=None, dropout=0.1):
        super().__init__()
        if num_classes is None:
            num_classes = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TropicalSSMBlock(d_model, n_heads, d_k, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, num_classes),
        )

    def forward(self, x, mu=1.0):
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h, mu=mu)
        h = self.norm(h)
        return self.head(h)

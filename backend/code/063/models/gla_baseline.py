"""
Standard GLA (Gated Linear Attention) Baseline

Uses independent per-head Q, K, V projections (no sharing).
This is the control for comparing against MFA-factored projections.

State update: S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
Output:       o_t = S_t^T @ q_t

Reference: Yang et al. (2024) "Gated Linear Attention Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLALayer(nn.Module):
    """Single GLA layer with standard independent per-head projections.

    Per-head projection cost: T * d * n * (2*d_k + d_v)
    For d=128, n=2, d_k=d_v=32: T * 128 * 2 * 96 = T * 24,576 FLOPs
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head

        # Independent per-head projections (standard approach)
        # Fused into single matrices: W_q in R^{d x (n*d_k)}
        self.W_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_v = nn.Linear(d_model, self.d_inner, bias=False)

        # Per-channel decay gate: alpha_t = sigmoid(W_alpha @ x_t) in [0,1]^{d_k}
        self.W_alpha = nn.Linear(d_model, self.d_inner, bias=True)

        # Output projection
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(self.W_alpha.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )

        output = self._recurrent_scan(q, k, v, alpha)

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_inner)
        output = self.W_o(output)
        output = self.dropout(output)

        return output + residual

    def _recurrent_scan(self, q, k, v, alpha):
        """
        GLA recurrent scan (no delta rule).
        S_t = diag(alpha_t) * S_{t-1} + k_norm_t * v_t^T
        o_t = S_t^T @ q_t
        """
        B, H, T, d = q.shape
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        for t in range(T):
            S = alpha[:, :, t, :].unsqueeze(-1) * S
            k_t = F.normalize(k[:, :, t, :], dim=-1)
            v_t = v[:, :, t, :]
            S = S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            q_t = q[:, :, t, :]
            o_t = torch.einsum("bhji,bhj->bhi", S, q_t)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2)


class GLAModel(nn.Module):
    """Full GLA model with standard projections (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            GLALayer(d_model, n_heads, d_head, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.head(h)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

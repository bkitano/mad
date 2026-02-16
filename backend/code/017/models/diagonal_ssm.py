"""
Diagonal SSM Baseline.

State transition: A_t = diag(alpha(x_t)) where alpha_i ∈ (0, 1) via sigmoid.

This is the simplest SSM baseline — purely diagonal, abelian, cannot represent
non-commutative groups like B_3. Expected to score < 60% on B3 composition.

h_t = alpha(x_t) ⊙ h_{t-1} + B x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal SSM layer: h_t = alpha_t ⊙ h_{t-1} + B x_t

    alpha_t = sigmoid(W_a x_t + b_a) ∈ (0, 1)^n
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_heads = num_heads

        n = state_dim

        # Diagonal decay: alpha(x_t) = sigmoid(W_a x_t + b_a)
        self.alpha_proj = nn.Linear(d_model, num_heads * n)

        # Input injection
        self.input_proj = nn.Linear(d_model, num_heads * n)

        # Output projection
        self.output_proj = nn.Linear(num_heads * n, d_model)

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.alpha_proj.weight, std=0.01)
        # Init bias positive so alpha starts near 1 (slow decay)
        nn.init.constant_(self.alpha_proj.bias, 2.0)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d = x.shape
        n = self.state_dim
        H = self.num_heads

        residual = x
        x = self.norm(x)

        alpha_all = torch.sigmoid(self.alpha_proj(x).view(batch, seq_len, H, n))
        b_all = self.input_proj(x).view(batch, seq_len, H, n)

        h = torch.zeros(batch, H, n, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            alpha_t = alpha_all[:, t]
            b_t = b_all[:, t]
            h = alpha_t * h + b_t
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape(batch, seq_len, H * n)
        outputs = self.output_proj(outputs)
        outputs = self.dropout(outputs)

        return residual + outputs


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return residual + self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DiagonalSSM(nn.Module):
    """Full Diagonal SSM model for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        state_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DiagonalSSMLayer(
                d_model=d_model,
                state_dim=state_dim,
                num_heads=num_heads,
                dropout=dropout,
            ))
            self.layers.append(SwiGLU(d_model, dropout=dropout))

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.output_norm(h)
        logits = self.output_head(h)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Permutation-Only SSM Baseline (No Signs).

State transition: A_t = P(x_t) ∈ S_n (permutation group only)

This baseline tests whether the Z_2^n sign component of the hyperoctahedral
group B_n adds value. If this baseline matches HyperSSM, then the sign
component is unnecessary.

h_t = gamma(x_t) ⊙ (P(x_t) h_{t-1}) + (1 - gamma(x_t)) ⊙ B x_t

Uses the same Gumbel-Sinkhorn permutation mechanism as HyperSSM,
but WITHOUT the sign/diagonal component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .hyper_ssm import sinkhorn_normalize, hungarian_hard, SwiGLU


class PermOnlySSMLayer(nn.Module):
    """
    Permutation-only SSM layer (no sign flips).

    Same as HyperSSMLayer but without the sign component D(x_t).
    Tests the value of the Z_2^n factor in B_n = Z_2^n ⋊ S_n.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 8,
        num_heads: int = 4,
        sinkhorn_iters: int = 5,
        tau: float = 1.0,
        use_hard_perm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau
        self.use_hard_perm = use_hard_perm

        n = state_dim

        # Permutation cost matrix (no sign projection!)
        self.perm_proj = nn.Linear(d_model, num_heads * n * n)

        # Forget gate
        self.gate_proj = nn.Linear(d_model, num_heads * n)

        # Input injection
        self.input_proj = nn.Linear(d_model, num_heads * n)

        # Output projection
        self.output_proj = nn.Linear(num_heads * n, d_model)

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.perm_proj.weight, std=0.01)
        nn.init.zeros_(self.perm_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.constant_(self.gate_proj.bias, 2.0)
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

        perm_logits = self.perm_proj(x).view(batch, seq_len, H, n, n)
        perm_logits = perm_logits / self.tau

        gamma = torch.sigmoid(self.gate_proj(x).view(batch, seq_len, H, n))
        b_input = self.input_proj(x).view(batch, seq_len, H, n)

        h = torch.zeros(batch, H, n, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            perm_t = perm_logits[:, t]
            perm_flat = perm_t.reshape(batch * H, n, n)
            soft_perm = sinkhorn_normalize(perm_flat, num_iters=self.sinkhorn_iters)

            if self.use_hard_perm:
                perm_matrix = hungarian_hard(soft_perm)
            else:
                perm_matrix = soft_perm

            perm_matrix = perm_matrix.view(batch, H, n, n)

            # Apply permutation ONLY (no signs): A_t h = P h
            h_perm = torch.einsum('bhij,bhj->bhi', perm_matrix, h)

            # Gated update (same as HyperSSM but without signs)
            gamma_t = gamma[:, t]
            b_t = b_input[:, t]
            h = gamma_t * h_perm + (1.0 - gamma_t) * b_t

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape(batch, seq_len, H * n)
        outputs = self.output_proj(outputs)
        outputs = self.dropout(outputs)

        return residual + outputs


class PermOnlySSM(nn.Module):
    """Full permutation-only SSM model for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        state_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 1,
        sinkhorn_iters: int = 5,
        tau: float = 1.0,
        use_hard_perm: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PermOnlySSMLayer(
                d_model=d_model,
                state_dim=state_dim,
                num_heads=num_heads,
                sinkhorn_iters=sinkhorn_iters,
                tau=tau,
                use_hard_perm=use_hard_perm,
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

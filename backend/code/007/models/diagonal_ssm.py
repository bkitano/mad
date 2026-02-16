"""
Diagonal SSM Baseline (Mamba-style diagonal gating).

A simple diagonal SSM with input-dependent scalar gates:
  x_t = α(u_t) ⊙ x_{t-1} + B(u_t) · u_t
  y_t = C · x_t + D · u_t

where α(u_t) = σ(W_α u_t + b_α) ∈ (0, 1)^n.

This is used ONLY for speed comparison (success criterion 4):
"Forward pass wall-clock time is < 3× that of a diagonal SSM of equal state dimension."

This model has the same state dimension n as OscGate-SSM to make the
speed comparison fair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSM(nn.Module):
    """
    Input-dependent diagonal SSM (Mamba-style).

    Args:
        d_model: Input/output dimension
        n: State dimension (should match OscGate-SSM's n = 2m for fair comparison)
    """

    def __init__(self, d_model: int, n: int):
        super().__init__()
        self.d_model = d_model
        self.n = n

        # Input-dependent decay α(u_t) = σ(W_α u_t + b_α)
        self.W_alpha = nn.Linear(d_model, n)

        # Input projection
        self.B = nn.Linear(d_model, n)

        # Output projection
        self.C = nn.Parameter(torch.randn(d_model, n) * 0.01)
        self.D = nn.Parameter(torch.zeros(d_model, d_model))

        nn.init.xavier_normal_(self.B.weight, gain=0.1)
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, 0.0)  # sigmoid(0) = 0.5

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        x = torch.zeros(batch_size, self.n, device=device, dtype=dtype)
        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]

            # Input-dependent decay
            alpha = torch.sigmoid(self.W_alpha(u_t))  # (batch, n)

            # State update: x_t = α_t ⊙ x_{t-1} + B u_t
            x = alpha * x + self.B(u_t)

            # Output
            y_t = F.linear(x, self.C) + F.linear(u_t, self.D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class DiagonalSSMClassifier(nn.Module):
    """DiagonalSSM wrapped for classification (speed comparison baseline)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DiagonalSSM(d_model, n) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.dropout(x)
        for ssm, norm in zip(self.layers, self.norms):
            x = x + ssm(norm(x))
        return self.head(x)

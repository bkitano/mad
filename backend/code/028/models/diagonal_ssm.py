"""
Diagonal SSM baseline for Experiment 028.

Standard diagonal SSM with input-dependent gating (Mamba-style).
Used as baseline: should achieve < 30% on S5 permutation composition
because diagonal transitions can only represent abelian (commutative)
state transformations, not the non-abelian S5 group.

State update:
  h_t = diag(lambda_t) * h_{t-1} + B * x_t
  lambda_t = sigmoid(W_lambda * x_t + b_lambda)   [element-wise, in (0,1)]

This is the simplest input-dependent SSM -- each state dimension evolves
independently. Cannot represent permutation composition because S5 is
non-abelian and requires state mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSM(nn.Module):
    """
    Diagonal SSM with input-dependent decay.

    Args:
        d_model: Input/output dimension
        n: State dimension
    """

    def __init__(self, d_model: int, n: int):
        super().__init__()
        self.d_model = d_model
        self.n = n

        # Input-dependent decay: lambda_t = sigmoid(W_lambda * x_t)
        self.W_lambda = nn.Linear(d_model, n)

        # Input/output projections
        self.B = nn.Linear(d_model, n, bias=False)
        self.C = nn.Linear(n, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))

        self._init_params()

    def _init_params(self):
        """Initialize for stable training with moderate decay."""
        nn.init.zeros_(self.W_lambda.weight)
        # sigmoid(2.0) ~ 0.88 -- moderate initial retention
        nn.init.constant_(self.W_lambda.bias, 2.0)
        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Sequential recurrence: h_t = lambda_t * h_{t-1} + B * x_t

        Args:
            u: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = u.shape
        h = torch.zeros(batch_size, self.n, device=u.device, dtype=u.dtype)

        outputs = []
        for t in range(seq_len):
            u_t = u[:, t, :]
            lam = torch.sigmoid(self.W_lambda(u_t))  # (batch, n), in (0, 1)
            h = lam * h + self.B(u_t)
            y_t = self.C(h) + u_t * self.D
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class DiagonalSSMClassifier(nn.Module):
    """
    Diagonal SSM wrapped for classification.

    Architecture:
      Embedding -> [DiagSSM -> LayerNorm + Residual] x num_layers -> MLP -> logits
    """

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
        """
        Args:
            tokens: (batch, seq_len) integer token indices

        Returns:
            logits: (batch, seq_len, num_classes)
        """
        x = self.embedding(tokens)
        x = self.dropout(x)
        for ssm, norm in zip(self.layers, self.norms):
            x = x + ssm(norm(x))
        logits = self.head(x)
        return logits

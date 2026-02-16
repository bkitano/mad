"""
Diagonal SSM baseline for group state tracking.

A diagonal SSM uses h_t = diag(lambda) * h_{t-1} + B * x_t,
where lambda are learned eigenvalues. Since diagonal matrices commute,
this model can only represent abelian (commutative) dynamics.

Expected: < 30% accuracy on S3/D4 state tracking (near random chance)
because S3 (6 elements) and D4 (8 elements) are non-abelian groups.

This serves as the lower bound / negative control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSMStep(nn.Module):
    """Single-step diagonal SSM: h_{t+1} = diag(lambda) * h_t + B * x_t.

    Args:
        n: State dimension
        d_in: Input feature dimension
    """

    def __init__(self, n: int = 4, d_in: int = 32):
        super().__init__()
        self.n = n

        # Learnable eigenvalues, initialized near 0.95 (stable, slow decay)
        # Parameterized as sigmoid(raw) to ensure |lambda| in (0, 1)
        self.log_lambda = nn.Parameter(torch.ones(n) * 2.0)  # sigmoid(2) ≈ 0.88

        # Input projection
        self.B_proj = nn.Linear(d_in, n)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, n) current hidden state
            x: (batch, d_in) current input features

        Returns:
            h_new: (batch, n) updated hidden state
        """
        # lambda in (0, 1) for stability
        lam = torch.sigmoid(self.log_lambda)

        # Diagonal update: elementwise multiply
        h_new = lam.unsqueeze(0) * h + self.B_proj(x)

        return h_new


class DiagonalSSMLayer(nn.Module):
    """Full diagonal SSM sequence layer."""

    def __init__(self, d_model: int, n: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.step = DiagonalSSMStep(n=n, d_in=d_model)
        self.C_proj = nn.Linear(n, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.n, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            h = self.step(h, x[:, t])
            y_t = self.C_proj(h)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return self.norm(x + y)


class DiagonalSSMClassifier(nn.Module):
    """Diagonal SSM classifier for group state tracking.

    Architecture:
        Embedding -> [Diagonal SSM Layer] x L -> MLP Head -> Logits

    Args:
        vocab_size: Number of input tokens
        num_classes: Number of output classes
        d_model: Hidden dimension
        n: State dimension
        num_layers: Number of layers
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        n: int = 4,
        num_layers: int = 1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [DiagonalSSMLayer(d_model=d_model, n=n) for _ in range(num_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) input token indices

        Returns:
            logits: (batch, seq_len, num_classes)
        """
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        logits = self.head(x)
        return logits

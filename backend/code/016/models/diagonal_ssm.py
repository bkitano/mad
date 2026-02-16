"""
Diagonal SSM Baseline

Standard diagonal SSM with input-dependent gating.
State transition: h_t = diag(lambda_t) * h_{t-1} + B @ x_t
where lambda_t = sigmoid(W_lambda @ x_t) ∈ (0, 1)^n

This is the standard Mamba/S4D-style diagonal SSM.
It is provably UNABLE to represent non-solvable groups like S5,
because diagonal matrices commute (form an abelian group).

Expected result on S5 composition: < 30% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.gs_monomial_ssm import RMSNorm, SwiGLU


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal SSM layer with input-dependent decay.

    h_t = lambda_t * h_{t-1} + B_t * x_t
    y_t = C @ h_t
    """

    def __init__(self, d_model: int, state_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input-dependent decay: lambda_t = sigmoid(W @ x_t)
        self.lambda_proj = nn.Linear(d_model, state_dim, bias=True)
        # Initialize bias so lambda starts near 0.9 (good for long-range)
        nn.init.constant_(self.lambda_proj.bias, 2.0)

        # Input projection
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)

        # Output projection
        self.C_proj = nn.Linear(state_dim, d_model, bias=False)

        # Output gate
        self.out_gate = nn.Linear(d_model, d_model, bias=False)

        # Output normalization
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len)

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d_model)

            # Input-dependent decay
            lam = torch.sigmoid(self.lambda_proj(x_t))  # (batch, state_dim) in (0, 1)

            # State update
            h = lam * h + self.B_proj(x_t)

            if mask is not None:
                # Zero out updates at padding positions
                h = h * mask[:, t].unsqueeze(-1)

            # Output
            y_t = self.C_proj(h)
            gate = torch.sigmoid(self.out_gate(x_t))
            y_t = gate * y_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = self.out_norm(y)
        return y


class DiagonalSSMBlock(nn.Module):
    """Diagonal SSM block with pre-norm residual + SwiGLU FFN."""

    def __init__(self, d_model: int, state_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.ssm_norm = RMSNorm(d_model)
        self.ssm = DiagonalSSMLayer(d_model, state_dim)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.ssm(self.ssm_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DiagonalSSM(nn.Module):
    """
    Diagonal SSM model for group composition tasks.

    Baseline: expected to FAIL on S5 composition (< 30% accuracy)
    because diagonal matrices commute.
    """

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        max_seq_len: int = 64,
        d_model: int = 32,
        state_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(num_tokens, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            DiagonalSSMBlock(d_model, state_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.head_fc1 = nn.Linear(d_model, d_model * 2)
        self.head_fc2 = nn.Linear(d_model * 2, num_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        x = F.gelu(self.head_fc1(x))
        logits = self.head_fc2(x)
        return logits


if __name__ == "__main__":
    batch = 4
    seq_len = 22
    num_tokens = 5
    num_classes = 120

    model = DiagonalSSM(
        num_tokens=num_tokens,
        num_classes=num_classes,
        max_seq_len=seq_len,
        d_model=32,
        state_dim=16,
        num_layers=2,
    )

    tokens = torch.randint(0, num_tokens, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    logits = model(tokens, mask)

    print(f"DiagonalSSM:")
    print(f"  Output: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

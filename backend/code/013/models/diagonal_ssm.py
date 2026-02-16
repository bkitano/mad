"""
Diagonal SSM Baseline (Mamba-2 style)

Standard diagonal SSM for comparison with Circulant SSM.
No coordinate mixing — each state dimension evolves independently.

Recurrence:
    h_t = diag(alpha_t) h_{t-1} + B_t x_t
    y_t = C_t^T h_t

where alpha_t = sigmoid(W_alpha x_t) in (0, 1) — input-dependent decay.

This is the baseline that should achieve <60% on Z8 composition
because diagonal transitions cannot represent the cyclic shift
structure needed for modular arithmetic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal SSM layer (no coordinate mixing).

    Each state dimension is an independent scalar recurrence:
        h_{t,i} = alpha_{t,i} * h_{t-1,i} + u_{t,i}

    Args:
        d_model: input/output dimension
        state_dim: SSM state dimension
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input-dependent decay: alpha_t = sigmoid(W_alpha x_t + b_alpha)
        self.W_alpha = nn.Linear(d_model, state_dim)

        # Input projection
        self.W_B = nn.Linear(d_model, state_dim)

        # Output projection
        self.W_C = nn.Linear(state_dim, d_model)

        # Skip connection
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.W_alpha.bias, 2.0)  # sigmoid(2) ~ 0.88
        nn.init.normal_(self.W_alpha.weight, std=0.01)
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Input-dependent decay
        alpha = torch.sigmoid(self.W_alpha(x))  # (B, T, n)

        # Input to state
        u = self.W_B(x)  # (B, T, n)

        # Sequential scan (diagonal — no mixing)
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            h = alpha[:, t] * h + u[:, t]  # Element-wise (no mixing!)
            outputs.append(h)

        h_all = torch.stack(outputs, dim=1)  # (B, T, n)

        # Output projection
        y = self.W_C(h_all) + self.D * residual

        return y


class DiagonalSSMModel(nn.Module):
    """
    Full Diagonal SSM model for sequence classification (baseline).

    Same architecture as CirculantSSMModel but with DiagonalSSMLayer
    instead of CirculantSSMLayer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        state_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # SSM layers with FFN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'ssm': DiagonalSSMLayer(d_model, state_dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = x + layer['ssm'](x)
            x = x + layer['ffn'](x)

        x = self.out_norm(x)
        logits = self.out_head(x)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

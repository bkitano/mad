"""
LinOSS Baseline: Fixed-parameter oscillatory SSM (LTI).

This is the LinOSS model from Rusch & Rus 2025 — an LTI oscillatory SSM
with FIXED ω and ζ parameters (not input-dependent).

Used as a baseline to show that input-dependence (selectivity) is necessary
for the selective copying task. LinOSS should FAIL (<40% accuracy) because
LTI models cannot perform content-dependent operations.

The architecture is identical to OscGate-SSM except:
- ω and ζ are nn.Parameter (learned but fixed across time)
- NOT functions of x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinOSS(nn.Module):
    """
    Fixed-parameter (LTI) oscillatory SSM.

    Same transition structure as OscGate-SSM but with fixed ω, ζ.

    Args:
        d_model: Input/output dimension
        m: Number of oscillators (state dim n = 2m)
        dt: Discretization timestep
    """

    def __init__(self, d_model: int, m: int, dt: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.n = 2 * m
        self.dt = dt

        # --- FIXED oscillatory parameters (LTI — NOT input-dependent) ---
        # log(ω) parameterization ensures ω > 0
        self.log_omega = nn.Parameter(torch.randn(m) * 0.1)
        # ζ via sigmoid of logit ensures ζ ∈ (0, 1)
        self.zeta_logit = nn.Parameter(torch.zeros(m))

        # --- I/O projections ---
        self.B = nn.Linear(d_model, self.n)
        self.C = nn.Parameter(torch.randn(d_model, self.n) * 0.01)
        self.D = nn.Parameter(torch.zeros(d_model, d_model))

        nn.init.xavier_normal_(self.B.weight, gain=0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FIXED (time-invariant) transition matrix.

        Args:
            u: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        # Fixed parameters — same for all timesteps (LTI property)
        omega = torch.exp(self.log_omega).clamp(min=1e-4, max=100.0)  # (m,)
        zeta = torch.sigmoid(self.zeta_logit)  # (m,)

        A_diag = omega ** 2  # (m,)
        S_diag = 1.0 / (1.0 + self.dt ** 2 * A_diag)  # (m,)
        d = 1.0 - zeta  # (m,)

        # Pre-compute fixed transition coefficients
        # M = [[d*S, -d*Δt*A*S], [d*Δt*S, d*S]]
        coeff_zz = d * S_diag  # z→z
        coeff_zy = -d * self.dt * A_diag * S_diag  # y→z
        coeff_yz = d * self.dt * S_diag  # z→y
        coeff_yy = d * S_diag  # y→y

        x = torch.zeros(batch_size, self.n, device=device, dtype=dtype)
        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]

            z = x[:, :self.m]
            y_state = x[:, self.m:]

            z_new = coeff_zz * z + coeff_zy * y_state
            y_new = coeff_yz * z + coeff_yy * y_state

            F_t = self.B(u_t)
            x = torch.cat([z_new, y_new], dim=-1) + F_t

            y_t = F.linear(x, self.C) + F.linear(u_t, self.D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class LinOSSClassifier(nn.Module):
    """LinOSS wrapped for classification (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        m: int,
        num_classes: int,
        num_layers: int = 2,
        dt: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LinOSS(d_model, m, dt) for _ in range(num_layers)
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

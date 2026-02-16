"""
Oscillatory-Gated Selective SSM (OscGate-SSM)

From proposal 007-oscillatory-gated-selective-ssm.

Key idea: Make oscillatory parameters ω(x_t) and ζ(x_t) input-dependent
while preserving stability-by-construction from the physics of damped
harmonic oscillators.

The state transition matrix M_t has 2×2 block-diagonal structure:
  M_t = [[S_t, -Δt A(x_t) S_t],
         [Δt S_t, S_t]]

where S_t = (I + Δt² A(x_t))^{-1} and A(x_t) = diag(ω(x_t)²).

Damping applied as: M_t^damped = diag(d_t) · M_t
where d_t = 1 - ζ(x_t) ∈ (0, 1).

Stability guarantee: ||M_t^damped||_2 ≤ max_k (1 - ζ_k) · √S_kk < 1
for ANY input x_t and ANY learned parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OscGateSSM(nn.Module):
    """
    Single-layer OscGate-SSM.

    Args:
        d_model: Input/output dimension
        m: Number of oscillators (state dim n = 2m)
        dt: Discretization timestep
        omega_max: Maximum frequency (for numerical stability, proposal Risk 4)
    """

    def __init__(self, d_model: int, m: int, dt: float = 0.01, omega_max: float = 100.0):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.n = 2 * m  # State dimension
        self.dt = dt
        self.omega_max = omega_max

        # --- Input-dependent parameter projections (proposal Step 1) ---
        # ω(x_t) = softplus(W_ω x_t + b_ω) > 0
        self.W_omega = nn.Linear(d_model, m)
        # ζ(x_t) = σ(W_ζ x_t + b_ζ) ∈ (0, 1)
        self.W_zeta = nn.Linear(d_model, m)

        # --- Input projection B(x_t) and output projection C ---
        # B maps input to state space; C maps state to output
        self.B_proj = nn.Linear(d_model, self.n)
        self.C = nn.Parameter(torch.randn(d_model, self.n) * 0.01)

        # --- Direct feedthrough ---
        self.D = nn.Parameter(torch.zeros(d_model, d_model))

        # Initialize projections
        self._init_params()

    def _init_params(self):
        """Initialize parameters for stable training."""
        # Initialize W_omega bias so initial frequencies are moderate
        nn.init.zeros_(self.W_omega.weight)
        # softplus(1.0) ≈ 1.3 — moderate initial frequency
        nn.init.constant_(self.W_omega.bias, 1.0)

        # Initialize W_zeta bias so initial damping ζ ≈ 0.5 (moderate)
        nn.init.zeros_(self.W_zeta.weight)
        nn.init.constant_(self.W_zeta.bias, 0.0)  # sigmoid(0) = 0.5

        # Small init for B projection
        nn.init.xavier_normal_(self.B_proj.weight, gain=0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequential recurrence.

        For the MVE, we use sequential scan (not parallel) for simplicity.
        The parallel scan version would use the associative operator
        (M_i, F_i) ⊕ (M_j, F_j) = (M_j M_i, M_j F_i + F_j).

        Args:
            u: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        # Initialize state x = [z, y]^T ∈ R^{2m}
        x = torch.zeros(batch_size, self.n, device=device, dtype=dtype)

        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)

            # --- Step 1: Compute input-dependent ω and ζ (proposal eq. for Step 1) ---
            # ω(x_t) = softplus(W_ω x_t + b_ω), clamped for numerical stability
            omega = F.softplus(self.W_omega(u_t))  # (batch, m), > 0
            omega = torch.clamp(omega, min=1e-4, max=self.omega_max)

            # ζ(x_t) = σ(W_ζ x_t + b_ζ), ∈ (0, 1)
            zeta = torch.sigmoid(self.W_zeta(u_t))  # (batch, m), ∈ (0, 1)

            # --- Step 2: Construct A(x_t) = diag(ω²) (proposal Step 2) ---
            A_diag = omega ** 2  # (batch, m), ≥ 0 guaranteed

            # --- Step 3: Compute S_t = (I + Δt² A(x_t))^{-1} (proposal Step 3) ---
            # S is diagonal, so inversion is element-wise
            S_diag = 1.0 / (1.0 + self.dt ** 2 * A_diag)  # (batch, m), ∈ (0, 1]

            # --- Step 4: Apply damping d_t = 1 - ζ(x_t) (proposal Step 4) ---
            d = 1.0 - zeta  # (batch, m), ∈ (0, 1)

            # --- Step 5: Compute M_t^damped · x_{t-1} + F_t (proposal Step 5) ---
            # Split state into z (velocity-like) and y (position-like)
            z = x[:, :self.m]  # (batch, m)
            y_state = x[:, self.m:]  # (batch, m)

            # M_t = [[S, -Δt A S], [Δt S, S]]
            # With damping: M_t^damped = diag(d) · M_t
            # New z = d * (S * z - Δt * A * S * y)
            # New y = d * (Δt * S * z + S * y)
            z_new = d * (S_diag * z - self.dt * A_diag * S_diag * y_state)
            y_new = d * (self.dt * S_diag * z + S_diag * y_state)

            # Input forcing F_t = B(x_t) · u_t
            F_t = self.B_proj(u_t)  # (batch, 2m)

            # State update
            x = torch.cat([z_new, y_new], dim=-1) + F_t  # (batch, 2m)

            # Output: y_t = C · x_t + D · u_t
            y_t = F.linear(x, self.C) + F.linear(u_t, self.D)  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class OscGateSSMClassifier(nn.Module):
    """
    OscGate-SSM wrapped for classification (selective copying task).

    Architecture:
      Embedding → [OscGateSSM → LayerNorm + Residual] × num_layers → MLP → logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        m: int,
        num_classes: int,
        num_layers: int = 2,
        dt: float = 0.01,
        omega_max: float = 100.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            OscGateSSM(d_model, m, dt, omega_max) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # MLP head for more expressive output mapping
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
        x = self.embedding(tokens)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        for ssm, norm in zip(self.layers, self.norms):
            x = x + ssm(norm(x))  # Pre-norm residual
        logits = self.head(x)  # (batch, seq_len, num_classes)
        return logits

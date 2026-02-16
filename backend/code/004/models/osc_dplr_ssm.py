"""
Oscillatory-DPLR SSM Implementation

This module implements the Oscillatory Diagonal-Plus-Low-Rank State Space Model
from proposal 004-oscillatory-dplr-ssm.md.

Key features:
1. Oscillatory eigenvalues from damped harmonic oscillator: λ = -ζω + iω√(1-ζ²)
2. DPLR structure: A = Λ + PQ^T where Λ is diagonal with oscillatory eigenvalues
3. Bilinear (Tustin) discretization for stability guarantee
4. Efficient convolution via SSM convolution kernel

Mathematical formulation (from proposal):
- Second-order ODE: ÿ(t) + 2ζω ẏ(t) + ω²y(t) = u(t)
- Eigenvalues: λ = -ζω ± iω√(1-ζ²) (guaranteed stable for ζ ≥ 0)
- Discrete: A_d = (I - Δ/2 A)^{-1}(I + Δ/2 A) (bilinear transform)
- DPLR: A = Λ + PQ^T where Λ = diag(λ_1, ..., λ_n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OscillatoryDPLRSSM(nn.Module):
    """
    Oscillatory-DPLR State Space Model.

    Args:
        n: State dimension (number of oscillators)
        r: Low-rank component rank (r << n)
        d_input: Input dimension
        d_output: Output dimension
        dt: Discretization step size (Δ in the proposal)
        init_omega_range: (min, max) for omega initialization (natural frequency)
    """

    def __init__(
        self,
        n: int = 16,
        r: int = 2,
        d_input: int = 1,
        d_output: int = 1,
        dt: float = 0.01,
        init_omega_range: Tuple[float, float] = (0.01, 0.1),
    ):
        super().__init__()
        self.n = n
        self.r = r
        self.d_input = d_input
        self.d_output = d_output
        self.dt = dt

        # Oscillatory parameters (proposal eq. 3)
        # ω > 0: natural frequency (oscillation rate)
        # Initialize log-uniform in [init_omega_range]
        log_omega_min = torch.log(torch.tensor(init_omega_range[0]))
        log_omega_max = torch.log(torch.tensor(init_omega_range[1]))
        log_omega_init = torch.rand(n) * (log_omega_max - log_omega_min) + log_omega_min
        self.log_omega = nn.Parameter(log_omega_init)

        # ζ ∈ [0, 1]: damping ratio (0=undamped, 1=critically damped)
        # Use sigmoid parameterization to enforce [0, 1] constraint
        # Initialize uniformly
        zeta_logit_init = torch.randn(n) * 0.5  # Start around 0.5 after sigmoid
        self.zeta_logit = nn.Parameter(zeta_logit_init)

        # Low-rank components (proposal eq. 4): A = Λ + PQ^T
        # Initialize small to avoid destabilizing oscillatory eigenvalues
        self.P = nn.Parameter(torch.randn(n, r) * 0.1)
        self.Q = nn.Parameter(torch.randn(n, r) * 0.1)

        # Input/output projections (B, C matrices)
        self.B = nn.Parameter(torch.randn(n, d_input) * 0.1)
        self.C = nn.Parameter(torch.randn(d_output, n) * 0.1)

        # D matrix (direct feedthrough)
        self.D = nn.Parameter(torch.zeros(d_output, d_input))

    def get_omega(self) -> torch.Tensor:
        """Get natural frequencies ω > 0."""
        return torch.exp(self.log_omega)

    def get_zeta(self) -> torch.Tensor:
        """Get damping ratios ζ ∈ [0, 1]."""
        return torch.sigmoid(self.zeta_logit)

    def get_continuous_eigenvalues(self) -> torch.Tensor:
        """
        Compute oscillatory eigenvalues (proposal eq. 3).

        λ = -ζω + iω√(1-ζ²)

        Returns:
            Complex tensor of shape (n,) with eigenvalues
        """
        omega = self.get_omega()
        zeta = self.get_zeta()

        # Real part: -ζω (damping)
        real = -zeta * omega

        # Imaginary part: ω√(1-ζ²) (oscillation)
        # Clamp to avoid numerical issues when ζ ≈ 1
        imag = omega * torch.sqrt(torch.clamp(1 - zeta**2, min=1e-8))

        # Combine into complex eigenvalues
        eigenvalues = torch.complex(real, imag)

        return eigenvalues

    def bilinear_discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bilinear (Tustin) discretization to A and B.

        A_d = (I - Δ/2 A)^{-1}(I + Δ/2 A)
        B_d = (I - Δ/2 A)^{-1} B √Δ

        For DPLR structure A = Λ + PQ^T, we use:
        (I - Δ/2 A)^{-1} = (I - Δ/2 Λ - Δ/2 PQ^T)^{-1}

        We apply Woodbury identity for efficiency, but for MVE (small n=16),
        we can directly invert for simplicity.

        Returns:
            A_d: Discrete state matrix (n, n)
            B_d: Discrete input matrix (n, d_input)
        """
        dt = self.dt
        n = self.n

        # Build continuous A = Λ + PQ^T
        Lambda = torch.diag(self.get_continuous_eigenvalues())  # (n, n)
        A_continuous = Lambda + self.P @ self.Q.T  # (n, n)

        # Bilinear transform
        I = torch.eye(n, dtype=A_continuous.dtype, device=A_continuous.device)

        # (I - Δ/2 A)
        I_minus = I - (dt / 2) * A_continuous

        # (I + Δ/2 A)
        I_plus = I + (dt / 2) * A_continuous

        # A_d = (I - Δ/2 A)^{-1}(I + Δ/2 A)
        A_d = torch.linalg.solve(I_minus, I_plus)

        # B_d = (I - Δ/2 A)^{-1} B √Δ
        # Convert B to complex dtype to match A_continuous
        B_complex = self.B.to(A_continuous.dtype)
        B_d = torch.linalg.solve(I_minus, B_complex * torch.sqrt(torch.tensor(dt)))

        return A_d, B_d

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Run SSM in recurrent mode (scan through sequence).

        Args:
            u: Input sequence (batch, seq_len, d_input)

        Returns:
            y: Output sequence (batch, seq_len, d_output)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device

        # Discretize
        A_d, B_d = self.bilinear_discretize()

        # Initialize state
        x = torch.zeros(batch_size, self.n, dtype=A_d.dtype, device=device)

        # Output buffer
        outputs = []

        # Convert C and D to complex dtype
        C_complex = self.C.to(A_d.dtype)
        D_complex = self.D.to(A_d.dtype)

        # Scan through sequence
        for t in range(seq_len):
            # State update: x_{t+1} = A_d x_t + B_d u_t
            u_t = u[:, t].to(A_d.dtype)  # Convert input to complex
            x = (A_d @ x.unsqueeze(-1)).squeeze(-1) + (B_d @ u_t.unsqueeze(-1)).squeeze(-1)

            # Output: y_t = C x_t + D u_t
            y = (C_complex @ x.unsqueeze(-1)).squeeze(-1) + (D_complex @ u_t.unsqueeze(-1)).squeeze(-1)

            outputs.append(y)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_output)

        return y.real  # Return real part only

    def get_learned_frequencies(self) -> torch.Tensor:
        """
        Get learned oscillator frequencies for analysis.

        Returns:
            omega: Natural frequencies (n,)
        """
        return self.get_omega().detach()

    def get_learned_damping(self) -> torch.Tensor:
        """
        Get learned damping ratios for analysis.

        Returns:
            zeta: Damping ratios (n,)
        """
        return self.get_zeta().detach()

"""
DPLR SSM Model for Experiment 018: Hutchinson Adaptive Rank.

Architecture: A = Lambda + P @ Q^* (Diagonal Plus Low-Rank)
- Lambda: diagonal eigenvalues (complex, negative real parts for stability)
- P, Q: low-rank factors of shape (n, r) where r is per-layer rank
- B: input projection (n, d)
- C: output projection (d, n)

The model processes sequences via discretized SSM recurrence:
  x_k = A_bar * x_{k-1} + B_bar * u_k
  y_k = C * x_k

where A_bar, B_bar are the ZOH-discretized versions.

References:
- Proposal 018: Hutchinson Trace-Guided Adaptive Rank for DPLR SSMs
- S4: Efficiently Modeling Long Sequences (Gu et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DPLRSSMLayer(nn.Module):
    """
    Single DPLR SSM layer.

    State transition: A = Lambda + P @ Q^* (complex-valued)
    Uses ZOH discretization and parallel scan for efficient training.

    Args:
        n: State dimension
        d: Input/output dimension (model dimension)
        r: Rank of the low-rank correction P @ Q^*
        dt_min: Minimum timestep for discretization
        dt_max: Maximum timestep for discretization
    """

    def __init__(self, n: int, d: int, r: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.n = n
        self.d = d
        self.r = r

        # Diagonal part Lambda: parameterized as log(-Re(lambda)) and Im(lambda)
        # Re(lambda) < 0 for stability
        # (proposal eq: Lambda diagonal eigenvalues)
        self.log_neg_real = nn.Parameter(torch.randn(n) * 0.5 + 0.5)  # log(-Re(lambda))
        self.imag = nn.Parameter(torch.randn(n) * math.pi)  # Im(lambda)

        # Low-rank factors P, Q in C^{n x r}
        # (proposal: A = Lambda + P @ Q^*)
        self.P_real = nn.Parameter(torch.randn(n, r) * 0.1)
        self.P_imag = nn.Parameter(torch.randn(n, r) * 0.1)
        self.Q_real = nn.Parameter(torch.randn(n, r) * 0.1)
        self.Q_imag = nn.Parameter(torch.randn(n, r) * 0.1)

        # Input projection B: R^d -> C^n
        self.B_real = nn.Parameter(torch.randn(n, d) * (1.0 / math.sqrt(d)))
        self.B_imag = nn.Parameter(torch.randn(n, d) * (1.0 / math.sqrt(d)))

        # Output projection C: C^n -> R^d
        self.C_real = nn.Parameter(torch.randn(d, n) * (1.0 / math.sqrt(n)))
        self.C_imag = nn.Parameter(torch.randn(d, n) * (1.0 / math.sqrt(n)))

        # Timestep (learnable, per-layer)
        self.log_dt = nn.Parameter(
            torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )

        # Skip connection and normalization
        self.D = nn.Parameter(torch.randn(d) * 0.01)  # skip connection

    @property
    def Lambda(self) -> torch.Tensor:
        """Get complex diagonal eigenvalues. Shape: (n,)"""
        return -torch.exp(self.log_neg_real) + 1j * self.imag

    @property
    def P(self) -> torch.Tensor:
        """Get complex P matrix. Shape: (n, r)"""
        return self.P_real + 1j * self.P_imag

    @property
    def Q(self) -> torch.Tensor:
        """Get complex Q matrix. Shape: (n, r)"""
        return self.Q_real + 1j * self.Q_imag

    @property
    def B(self) -> torch.Tensor:
        """Get complex B matrix. Shape: (n, d)"""
        return self.B_real + 1j * self.B_imag

    @property
    def C(self) -> torch.Tensor:
        """Get complex C matrix. Shape: (d, n)"""
        return self.C_real + 1j * self.C_imag

    def _discretize(self):
        """
        ZOH discretization of the continuous SSM.

        A_bar = exp(A * dt) ≈ (I + A*dt/2) / (I - A*dt/2) (bilinear)
        B_bar = (A_bar - I) * A^{-1} * B ≈ dt * B (for small dt)

        For the DPLR structure, we use the Woodbury identity to compute
        the discretized A efficiently.

        Returns: (A_bar_diag, A_bar_P, A_bar_Q, B_bar) for efficient recurrence
        """
        dt = torch.exp(self.log_dt)  # (1,)
        Lambda = self.Lambda  # (n,)

        # For the diagonal part, exact discretization:
        # exp(Lambda * dt) element-wise
        A_bar_diag = torch.exp(Lambda * dt)  # (n,)

        # For the low-rank correction, use first-order approximation:
        # exp((Lambda + PQ^*) * dt) ≈ exp(Lambda*dt) * (I + dt * exp(Lambda*dt)^{-1} * (exp(Lambda*dt)-I) * PQ^*/Lambda)
        # Simplified: we use the diagonal discretization and apply low-rank correction
        # through the recurrence directly.

        # B_bar = (exp(Lambda*dt) - I) / Lambda * B (diagonal part)
        # For stability, compute element-wise
        B_bar = (A_bar_diag - 1.0) / (Lambda + 1e-8) * dt  # correction factor
        B_bar = B_bar.unsqueeze(-1) * self.B  # (n, d)

        return A_bar_diag, dt, B_bar

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using scan-based recurrence.

        Args:
            u: Input tensor of shape (batch, seq_len, d)

        Returns:
            y: Output tensor of shape (batch, seq_len, d)
        """
        batch, seq_len, d = u.shape
        A_bar_diag, dt, B_bar = self._discretize()

        # Compute the low-rank discretized correction
        P = self.P  # (n, r)
        Q = self.Q  # (n, r)
        C = self.C  # (d, n)

        # Input projection: u @ B^T -> (batch, seq_len, n)
        # B_bar is (n, d), so we need u @ B_bar^T
        x_input = torch.einsum('bld,nd->bln', u.to(torch.complex64), B_bar)  # (batch, seq_len, n)

        # Run the recurrence with DPLR structure:
        # x_k = A_bar_diag * x_{k-1} + dt * P @ Q^* @ x_{k-1} + B_bar @ u_k
        # We compute this step by step for clarity (scan version below)

        # For efficiency, use the parallel scan formulation
        # But for MVE, sequential recurrence is fine for seq_len=1024
        x = torch.zeros(batch, self.n, dtype=torch.complex64, device=u.device)  # (batch, n)
        ys = []

        for t in range(seq_len):
            # Low-rank correction: PQ^* @ x = P @ (Q^* @ x)
            # Q^* @ x: (r,) per batch
            Qx = torch.einsum('nr,bn->br', Q.conj(), x)  # (batch, r)
            PQx = torch.einsum('nr,br->bn', P, Qx)  # (batch, n)

            # State update: x = A_bar_diag * x + dt * PQ^* x + B_bar @ u
            x = A_bar_diag * x + dt * PQx + x_input[:, t, :]  # (batch, n)

            # Output: y = Re(C @ x)
            y_t = torch.einsum('dn,bn->bd', C, x).real  # (batch, d)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d)

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def compute_importance_logdet(self, num_freqs: int = 16, k_max: int = 4) -> float:
        """
        Compute low-rank importance score via power-series log-det method.

        (Proposal eq): I_logdet = E_omega[|sum_{k=1}^{4} (-1)^{k+1}/k * tr((Q^*(iw I - Lambda)^{-1} P)^k)|]

        This measures how much the low-rank correction PQ^* contributes
        to the SSM's transfer function. Higher = more important.

        Cost: O(k_max * r^3 * num_freqs) per layer — negligible.

        Args:
            num_freqs: Number of sampled frequencies |Omega|
            k_max: Number of power series terms

        Returns:
            Importance score (float)
        """
        with torch.no_grad():
            Lambda = self.Lambda  # (n,)
            P = self.P  # (n, r)
            Q = self.Q  # (n, r)

            # Sample frequencies uniformly in [0.01, 10] (log-spaced)
            omegas = torch.logspace(-2, 1, num_freqs, device=Lambda.device)

            total_importance = 0.0

            for omega in omegas:
                # Resolvent diagonal: (i*omega - Lambda)^{-1}
                z = 1j * omega
                resolvent_diag = 1.0 / (z - Lambda)  # (n,)

                # M = Q^* @ diag(resolvent) @ P  (r x r matrix)
                # = (Q.conj().T @ diag(resolvent_diag)) @ P
                Q_scaled = Q.conj() * resolvent_diag.unsqueeze(-1)  # (n, r) * (n, 1)
                M = Q_scaled.T @ P  # (r, r)

                # Power series: sum_{k=1}^{k_max} (-1)^{k+1}/k * tr(M^k)
                log_det_approx = 0.0
                M_power = torch.eye(self.r, dtype=M.dtype, device=M.device)  # M^0 = I
                for k in range(1, k_max + 1):
                    M_power = M_power @ M  # M^k
                    sign = (-1.0) ** (k + 1)
                    log_det_approx += sign / k * torch.trace(M_power)

                total_importance += torch.abs(log_det_approx).item()

            return total_importance / num_freqs

    def compute_importance_hutchinson(self, num_freqs: int = 16, num_probes: int = 1) -> float:
        """
        Compute low-rank importance via Hutchinson trace estimate of Frobenius norm.

        (Proposal eq): I = E_omega[||R_LR(omega)||_F^2]
        ≈ E_omega[g^* R_LR^* R_LR g], g ~ N(0, I)

        Args:
            num_freqs: Number of sampled frequencies
            num_probes: Number of random probe vectors (m)

        Returns:
            Importance score (float)
        """
        with torch.no_grad():
            Lambda = self.Lambda  # (n,)
            P = self.P  # (n, r)
            Q = self.Q  # (n, r)

            omegas = torch.logspace(-2, 1, num_freqs, device=Lambda.device)
            total = 0.0

            for omega in omegas:
                z = 1j * omega
                resolvent_diag = 1.0 / (z - Lambda)  # (n,)

                for _ in range(num_probes):
                    # Random probe vector g ~ N(0, I) in C^n
                    g = torch.randn(self.n, dtype=torch.complex64, device=Lambda.device)

                    # Step 1: v1 = diag(resolvent) @ g  — O(n)
                    v1 = resolvent_diag * g  # (n,)

                    # Step 2: v2 = Q^* @ v1  — O(nr)
                    v2 = Q.conj().T @ v1  # (r,)

                    # Step 3: solve (I + Q^* diag(res) P) @ w = v2  — O(r^3)
                    Q_res_P = (Q.conj() * resolvent_diag.unsqueeze(-1)).T @ P  # (r, r)
                    system = torch.eye(self.r, dtype=Q_res_P.dtype, device=Q_res_P.device) + Q_res_P
                    w = torch.linalg.solve(system, v2)  # (r,)

                    # Step 4: v3 = P @ w  — O(nr)
                    v3 = P @ w  # (n,)

                    # Step 5: result = diag(resolvent) @ v3  — O(n)
                    result = resolvent_diag * v3  # (n,)

                    # Hutchinson estimate of ||R_LR||_F^2 = g^* R^* R g
                    total += torch.abs(torch.dot(result.conj(), result)).item()

            return total / (num_freqs * num_probes)

    def truncate_rank(self, new_r: int):
        """
        Truncate the low-rank factors to a new (lower) rank via SVD.

        (Proposal eq): PQ^* = U Sigma V^* ≈ U_{:r} Sigma_{:r} V_{:r}^*
        New P = U_{:r} Sigma_{:r}^{1/2}, New Q = V_{:r} Sigma_{:r}^{1/2}

        Args:
            new_r: New rank (must be <= current rank)
        """
        if new_r >= self.r:
            return  # No truncation needed

        with torch.no_grad():
            P = self.P  # (n, r)
            Q = self.Q  # (n, r)

            # Compute PQ^* = (n, n) — but we only need top new_r singular values
            # Use SVD of the r x r matrix Q^* P for efficiency
            # PQ^* = P @ Q^* has rank at most r
            # SVD: PQ^* = U @ S @ V^*

            # Actually compute P @ Q^T (n x n is expensive for large n)
            # Better: use the thin SVD via the r x r Gram matrices
            # But for n=32, direct SVD is fine
            M = P @ Q.conj().T  # (n, n) — acceptable for n=32

            U, S, Vh = torch.linalg.svd(M, full_matrices=False)

            # Truncate to new_r
            U_trunc = U[:, :new_r]  # (n, new_r)
            S_trunc = S[:new_r]  # (new_r,)
            Vh_trunc = Vh[:new_r, :]  # (new_r, n)

            # New factors: P_new = U @ S^{1/2}, Q_new = V @ S^{1/2}
            S_sqrt = torch.sqrt(S_trunc + 1e-10)
            P_new = U_trunc * S_sqrt.unsqueeze(0)  # (n, new_r)
            Q_new = (Vh_trunc.conj().T * S_sqrt.unsqueeze(0))  # (n, new_r)

            # Update parameters (resize)
            self.r = new_r
            self.P_real = nn.Parameter(P_new.real)
            self.P_imag = nn.Parameter(P_new.imag)
            self.Q_real = nn.Parameter(Q_new.real)
            self.Q_imag = nn.Parameter(Q_new.imag)


class DPLRSSMBlock(nn.Module):
    """
    A single block: DPLR SSM + LayerNorm + MLP (GLU variant).

    Architecture:
        x -> LayerNorm -> DPLR SSM -> + -> LayerNorm -> MLP -> + -> out
             |_________________________|   |__________________|
    """

    def __init__(self, d: int, n: int, r: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.ssm = DPLRSSMLayer(n=n, d=d, r=r)
        self.norm2 = nn.LayerNorm(d)

        mlp_hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d)
        Returns:
            (batch, seq_len, d)
        """
        # SSM path with residual
        h = self.norm1(x)
        h = self.ssm(h)
        x = x + self.dropout(h)

        # MLP path with residual
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class DPLRSSMModel(nn.Module):
    """
    Full DPLR SSM model for sequence classification.

    Architecture:
        Input -> Linear (input_dim -> d) -> [DPLRSSMBlock] x L -> Pool -> Linear -> classes

    Args:
        input_dim: Input feature dimension (e.g., 1 for sCIFAR pixel values)
        d: Model hidden dimension
        n: SSM state dimension
        r: Low-rank correction rank (can be per-layer via r_per_layer)
        num_layers: Number of DPLR SSM blocks
        num_classes: Number of output classes
        dropout: Dropout rate
        r_per_layer: Optional list of per-layer ranks (overrides r)
    """

    def __init__(
        self,
        input_dim: int = 1,
        d: int = 64,
        n: int = 32,
        r: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
        r_per_layer: Optional[list] = None,
    ):
        super().__init__()
        self.d = d
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d)

        # SSM blocks
        if r_per_layer is not None:
            assert len(r_per_layer) == num_layers
            ranks = r_per_layer
        else:
            ranks = [r] * num_layers

        self.blocks = nn.ModuleList([
            DPLRSSMBlock(d=d, n=n, r=ranks[i], dropout=dropout)
            for i in range(num_layers)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        x = self.input_proj(x)  # (batch, seq_len, d)

        for block in self.blocks:
            x = block(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d)
        x = self.norm_out(x)
        logits = self.head(x)  # (batch, num_classes)

        return logits

    def get_ssm_layers(self) -> list:
        """Get all SSM layers for importance computation."""
        return [block.ssm for block in self.blocks]

    def get_ranks(self) -> list:
        """Get current ranks of all SSM layers."""
        return [block.ssm.r for block in self.blocks]

    def count_lr_params(self) -> int:
        """Count parameters in low-rank factors across all layers."""
        total = 0
        for block in self.blocks:
            ssm = block.ssm
            # P and Q: each has 2 * n * r real parameters (real + imag)
            total += 4 * ssm.n * ssm.r  # P_real, P_imag, Q_real, Q_imag
        return total

    def count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

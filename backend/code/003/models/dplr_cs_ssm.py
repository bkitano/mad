"""
DPLR Column-Sparse SSM (DPLR-CS-SSM)

Implements A = P (Lambda + p q^T) P^T where:
- P: fixed permutation matrix (identity, cyclic, bit-reversal, or learned)
- Lambda: diagonal matrix with eigenvalues on/inside unit circle
- p, q: rank-1 correction vectors (DPLR structure)

Reference: proposals/003-dplr-column-sparse-cauchy-kernel.md

Key insight: The permutation P creates coupling between state dimensions
that diagonal SSMs cannot represent. For tasks requiring inter-dimension
communication (like parity), this coupling should help.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bit_reversal_permutation(N: int) -> torch.Tensor:
    """Create bit-reversal permutation matrix of size N.

    Bit-reversal maximizes mixing between state dimensions by mapping
    index i to the index formed by reversing i's binary representation.

    N must be a power of 2.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"N must be a power of 2, got {N}"
    n_bits = int(math.log2(N))

    P = torch.zeros(N, N)
    for i in range(N):
        # Reverse the bits of i
        rev = 0
        val = i
        for _ in range(n_bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        P[rev, i] = 1.0
    return P


def cyclic_shift_permutation(N: int) -> torch.Tensor:
    """Create cyclic shift permutation: maps i -> (i+1) mod N.

    Simplest non-trivial permutation that rotates state dimensions.
    """
    P = torch.zeros(N, N)
    for i in range(N):
        P[(i + 1) % N, i] = 1.0
    return P


class DPLR_CS_SSM(nn.Module):
    """
    DPLR Column-Sparse SSM layer.

    State transition: A = P (Lambda + p q^T) P^T  [proposal eq. in Math Formulation]
    State update: h_t = A h_{t-1} + B x_t
    Output: y_t = C h_t

    Uses bilinear (Tustin) discretization for numerical stability.

    Args:
        state_dim: State dimension N
        input_dim: Input feature dimension d
        output_dim: Output feature dimension
        P_type: Permutation type ('identity', 'cyclic', 'bit_reversal', 'learned')
        dt: Discretization time step
    """

    def __init__(
        self,
        state_dim: int = 16,
        input_dim: int = 32,
        output_dim: int = 32,
        P_type: str = 'identity',
        dt: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.P_type = P_type
        self.dt = dt

        # --- DPLR core parameters ---
        # Lambda: diagonal eigenvalues (proposal: on/inside unit circle)
        # Parameterize in log-space for stability, init near -0.5
        self.log_lambda = nn.Parameter(torch.randn(state_dim) * 0.1 - 0.5)

        # p, q: rank-1 correction vectors (proposal: small init)
        self.p = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.q = nn.Parameter(torch.randn(state_dim) * 0.01)

        # --- Permutation matrix ---
        if P_type == 'identity':
            P = torch.eye(state_dim)
        elif P_type == 'cyclic':
            P = cyclic_shift_permutation(state_dim)
        elif P_type == 'bit_reversal':
            P = bit_reversal_permutation(state_dim)
        elif P_type == 'learned':
            # Learned permutation via doubly-stochastic relaxation
            # Initialize with random permutation
            perm = torch.randperm(state_dim)
            P = torch.zeros(state_dim, state_dim)
            for i, j in enumerate(perm):
                P[i, j] = 1.0
        else:
            raise ValueError(f"Unknown P_type: {P_type}")

        if P_type == 'learned':
            # For learned permutation, use soft doubly-stochastic matrix
            # Initialize logits from the permutation matrix
            self.P_logits = nn.Parameter(P * 5.0 + torch.randn_like(P) * 0.1)
        else:
            self.register_buffer('P', P)

        # --- Input/output projections ---
        # B: input → state
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * (1.0 / math.sqrt(input_dim)))
        # C: state → output
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * (1.0 / math.sqrt(state_dim)))
        # D: skip connection
        self.D = nn.Parameter(torch.zeros(output_dim, input_dim))

    def _get_permutation(self) -> torch.Tensor:
        """Get the permutation matrix P.

        For learned permutations, apply Sinkhorn normalization to get
        a doubly-stochastic matrix, then use straight-through estimator
        to get a hard permutation during forward pass.
        """
        if self.P_type == 'learned':
            # Sinkhorn normalization (5 iterations for fast convergence)
            log_P = self.P_logits
            for _ in range(5):
                log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
                log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
            P_soft = torch.exp(log_P)

            # Straight-through: hard permutation in forward, soft gradients in backward
            # Hungarian algorithm approximation: just take argmax per row
            with torch.no_grad():
                # Greedy assignment (not optimal but fast)
                P_hard = torch.zeros_like(P_soft)
                cols_used = set()
                for i in range(self.state_dim):
                    row = P_soft[i].clone()
                    for c in cols_used:
                        row[c] = -float('inf')
                    j = row.argmax().item()
                    P_hard[i, j] = 1.0
                    cols_used.add(j)

            return P_hard + P_soft - P_soft.detach()  # Straight-through
        else:
            return self.P

    def _get_A_continuous(self) -> torch.Tensor:
        """Compute continuous-time state matrix A.

        A = P (Lambda + p q^T) P^T  [proposal eq.]

        where Lambda = diag(lambda_i) with lambda_i = -exp(log_lambda_i) < 0
        for stability (eigenvalues in left half-plane).
        """
        P = self._get_permutation()

        # Lambda: negative real eigenvalues for stability
        Lambda = -torch.exp(self.log_lambda)  # All negative → stable
        Lambda_diag = torch.diag(Lambda)

        # Low-rank correction: p q^T
        low_rank = torch.outer(self.p, self.q)

        # DPLR core: Lambda + p q^T
        core = Lambda_diag + low_rank

        # Apply permutation: A = P @ core @ P^T
        A = P @ core @ P.t()

        return A

    def _discretize(self, A_c: torch.Tensor) -> tuple:
        """Bilinear (Tustin) discretization: continuous → discrete.

        A_d = (I + dt/2 * A_c) @ inv(I - dt/2 * A_c)
        B_d = dt * inv(I - dt/2 * A_c) @ B

        This ensures |eigenvalues(A_d)| <= 1 when eigenvalues(A_c) have
        negative real parts → numerical stability.
        """
        dt = self.dt
        N = self.state_dim
        I = torch.eye(N, device=A_c.device, dtype=A_c.dtype)

        I_minus = I - (dt / 2) * A_c
        I_plus = I + (dt / 2) * A_c

        # A_d = inv(I_minus) @ I_plus
        A_d = torch.linalg.solve(I_minus, I_plus)
        # B_d = dt * inv(I_minus) @ B
        B_d = dt * torch.linalg.solve(I_minus, self.B)

        return A_d, B_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process sequence through SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            y: Output tensor of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get discrete-time matrices
        A_c = self._get_A_continuous()
        A_d, B_d = self._discretize(A_c)

        # Sequential scan (sufficient for MVE scale)
        h = torch.zeros(batch_size, self.state_dim, device=device)
        outputs = []

        for t in range(seq_len):
            # h_t = A_d @ h_{t-1} + B_d @ x_t
            h = torch.einsum('ij,bj->bi', A_d, h) + torch.einsum('ij,bj->bi', B_d, x[:, t, :])
            # y_t = C @ h_t + D @ x_t
            y_t = torch.einsum('ij,bj->bi', self.C, h) + torch.einsum('ij,bj->bi', self.D, x[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)


class DPLRCSModel(nn.Module):
    """
    Full model wrapping DPLR-CS SSM for parity classification.

    Architecture:
    1. Input projection: d_input → d_model
    2. DPLR-CS SSM layer
    3. Global average pooling over sequence
    4. Classification head → 2 classes (parity 0 or 1)

    Args:
        d_input: Input dimension (1 for binary parity)
        d_model: Hidden/model dimension
        state_dim: SSM state dimension N
        P_type: Permutation type for SSM
        dt: Discretization step
    """

    def __init__(
        self,
        d_input: int = 1,
        d_model: int = 32,
        state_dim: int = 16,
        P_type: str = 'identity',
        dt: float = 1.0,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # SSM layer
        self.ssm = DPLR_CS_SSM(
            state_dim=state_dim,
            input_dim=d_model,
            output_dim=d_model,
            P_type=P_type,
            dt=dt,
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Classification head (parity: 2 classes)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len) or (batch, seq_len, 1)

        Returns:
            logits: Shape (batch, 2) for binary parity classification
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # SSM processing
        x = self.ssm(x)  # (batch, seq_len, d_model)

        # Layer norm
        x = self.norm(x)

        # Global average pooling → classification
        x = x.mean(dim=1)  # (batch, d_model)
        logits = self.classifier(x)  # (batch, 2)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

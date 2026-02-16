"""
Displacement-Rank SSM (DR-SSM): Cauchy-Like State Transitions.

From proposal 022-displacement-rank-ssm-state-transitions.

The state transition at each time step is a Cauchy-like matrix with
displacement rank α:

    (A_t)_{ij} = δ_{ij} · d_i(x_t) + Σ_{k=1}^{α} G_{ik}(x_t) · H_{jk}(x_t) / (s_i - s_j)

where:
    - d_i(x_t) ∈ ℝ: input-dependent diagonal (decay/gate)
    - G(x_t), H(x_t) ∈ ℝ^{n×α}: input-dependent generators
    - s ∈ ℝ^n: fixed displacement nodes (Chebyshev)
    - α: displacement rank (the capacity parameter)

The recurrence is:
    h_t = A_t h_{t-1} + B_t x_t

The Cauchy-like matvec A_t h decomposes as:
    A_t h = d(x_t) ⊙ h + Σ_k G[:,k] · Cauchy_s(H[:,k] ⊙ h)

where Cauchy_s(v)_i = Σ_j v_j / (s_i - s_j) is a Cauchy matrix-vector product.

For the MVE, we use the naive O(n²) Cauchy matvec (proposal suggests
optimizing to O(n log n) via FFT later, but for n=16 the naive version
is fine and avoids FFT overhead).

Key insight: displacement rank α is the capacity knob:
  - α = 0: diagonal SSM (Mamba-equivalent)
  - α = 1: DPLR (S4-equivalent)
  - α = n: dense SSM (full mixing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def chebyshev_nodes(n: int) -> torch.Tensor:
    """
    Generate Chebyshev nodes on [-1, 1] (proposal: displacement nodes).

    s_i = cos(π(2i-1)/(2n)) for i = 1, ..., n

    These are well-separated and provide good numerical conditioning
    for Cauchy matrices (proposal Section: Initialization, item 1).
    """
    i = torch.arange(1, n + 1, dtype=torch.float32)
    return torch.cos(math.pi * (2 * i - 1) / (2 * n))


def cauchy_matvec_naive(s: torch.Tensor, d: torch.Tensor, G: torch.Tensor,
                         H: torch.Tensor, h: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Compute A @ h where A is Cauchy-like with displacement rank α.

    A_{ij} = d_i δ_{ij} + Σ_k G_{ik} H_{jk} / (s_i - s_j)

    This is the naive O(αn²) implementation. For n=16, this is fine.
    For larger n, use FFT-based O(αn log n) version.

    Args:
        s: (n,) displacement nodes
        d: (batch, n) diagonal gates
        G: (batch, n, alpha) row generators
        H: (batch, n, alpha) col generators
        h: (batch, n) input vector
        eps: small constant to avoid division by zero on diagonal

    Returns:
        result: (batch, n) output vector A @ h

    Complexity: O(α n²) per sample
    """
    # Diagonal part: O(n)
    result = d * h  # (batch, n)

    # Cauchy part: build (n, n) Cauchy kernel (fixed, independent of batch)
    # C_{ij} = 1 / (s_i - s_j) for i ≠ j, 0 for i = j
    n = s.shape[0]
    diffs = s.unsqueeze(1) - s.unsqueeze(0)  # (n, n)
    # Avoid division by zero on diagonal
    diffs = diffs + torch.eye(n, device=s.device)  # Add 1 on diagonal
    cauchy_kernel = 1.0 / diffs  # (n, n)
    cauchy_kernel = cauchy_kernel * (1.0 - torch.eye(n, device=s.device))  # Zero diagonal

    # Vectorized: compute all alpha generators at once
    # weighted[b, j, k] = H[b, j, k] * h[b, j]
    weighted = H * h.unsqueeze(-1)  # (batch, n, alpha)
    # cauchy_result[b, i, k] = Σ_j cauchy_kernel[i, j] * weighted[b, j, k]
    cauchy_result = torch.einsum('ij,bjk->bik', cauchy_kernel, weighted)  # (batch, n, alpha)
    # result += Σ_k G[b, i, k] * cauchy_result[b, i, k]
    result = result + (G * cauchy_result).sum(dim=-1)  # (batch, n)

    return result


def dense_matvec(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Dense matrix-vector product for the Dense SSM baseline.

    Args:
        A: (batch, n, n) full transition matrix
        h: (batch, n) state vector

    Returns:
        (batch, n) = A @ h
    """
    return torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)


class DRSSMLayer(nn.Module):
    """
    Single Displacement-Rank SSM layer.

    Parameterizes state transitions as Cauchy-like matrices with
    displacement rank α. Input-dependent parameters following
    proposal eq. (97-100):
        d(x_t) = σ(x_t W_d) ∈ (0,1)^n       (diagonal gate)
        G(x_t) = x_t W_G ∈ ℝ^{n × α}         (row generators)
        H(x_t) = x_t W_H ∈ ℝ^{n × α}         (col generators)

    Args:
        d_model: input/output dimension
        n: state dimension
        alpha: displacement rank (capacity parameter)
    """

    def __init__(self, d_model: int, n: int, alpha: int):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.alpha = alpha

        # Fixed displacement nodes — Chebyshev (proposal Initialization §1)
        self.register_buffer('s', chebyshev_nodes(n))

        # Input-dependent diagonal gate: d(x_t) = σ(x_t W_d) ∈ (0, 1)^n
        self.W_d = nn.Linear(d_model, n)

        # Input-dependent generators (only if alpha > 0)
        if alpha > 0:
            self.W_G = nn.Linear(d_model, n * alpha)
            self.W_H = nn.Linear(d_model, n * alpha)

        # Input projection: B(x_t) = x_t W_B ∈ ℝ^n
        self.W_B = nn.Linear(d_model, n)

        # Output projection: C ∈ ℝ^{d_model × n}
        self.C = nn.Linear(n, d_model, bias=False)

        # Direct feedthrough
        self.D = nn.Parameter(torch.zeros(d_model))

        self._init_params()

    def _init_params(self):
        """Initialize for stable training (proposal Initialization §2-3)."""
        # Diagonal gate: initial d ≈ 0.9 (moderate decay)
        # sigmoid(2.2) ≈ 0.9
        nn.init.zeros_(self.W_d.weight)
        nn.init.constant_(self.W_d.bias, 2.2)

        # Generators: small perturbation, N(0, 1/√(nα))
        if self.alpha > 0:
            std = 1.0 / math.sqrt(self.n * self.alpha)
            nn.init.normal_(self.W_G.weight, 0, 0.01)
            nn.init.normal_(self.W_G.bias, 0, std)
            nn.init.normal_(self.W_H.weight, 0, 0.01)
            nn.init.normal_(self.W_H.bias, 0, std)

        # Input projection: small init
        nn.init.xavier_normal_(self.W_B.weight, gain=0.1)

        # Output: small init
        nn.init.xavier_normal_(self.C.weight, gain=0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Sequential recurrence forward pass.

        h_t = A_t h_{t-1} + B_t x_t
        y_t = C h_t + D ⊙ x_t

        Args:
            u: (batch, seq_len, d_model) input

        Returns:
            y: (batch, seq_len, d_model) output
        """
        batch, T, _ = u.shape
        device = u.device

        h = torch.zeros(batch, self.n, device=device, dtype=u.dtype)
        outputs = []

        for t in range(T):
            u_t = u[:, t, :]  # (batch, d_model)

            # Compute input-dependent parameters
            d_t = torch.sigmoid(self.W_d(u_t))  # (batch, n), ∈ (0, 1)
            B_t = self.W_B(u_t)  # (batch, n)

            if self.alpha > 0:
                G_t = self.W_G(u_t).view(batch, self.n, self.alpha)  # (batch, n, α)
                H_t = self.W_H(u_t).view(batch, self.n, self.alpha)  # (batch, n, α)

                # Normalize generators for stability (proposal Risk 4)
                G_norm = torch.norm(G_t, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                H_norm = torch.norm(H_t, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                scale = math.sqrt(self.alpha)
                G_t = G_t / G_norm * scale
                H_t = H_t / H_norm * scale

                # Cauchy-like matvec: A_t @ h
                h = cauchy_matvec_naive(self.s, d_t, G_t, H_t, h)
            else:
                # α = 0: pure diagonal SSM (Mamba-equivalent)
                h = d_t * h

            # Add input forcing
            h = h + B_t

            # Output projection
            y_t = self.C(h) + self.D * u_t  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, T, d_model)


class DenseSSMLayer(nn.Module):
    """
    Dense SSM layer — upper bound on expressivity (α = n).

    Uses a full n×n input-dependent transition matrix.
    Cost: O(n²) per step.

    This serves as the "gold standard" baseline — if DR-SSM can match
    its accuracy at α < n with better speed, the proposal is validated.
    """

    def __init__(self, d_model: int, n: int):
        super().__init__()
        self.d_model = d_model
        self.n = n

        # Input-dependent transition: A(x_t) = reshape(x_t W_A, [n, n])
        # then apply stability: A_stable = diag(σ(d)) + tanh(A_offdiag) * scale
        self.W_d = nn.Linear(d_model, n)  # diagonal
        self.W_A = nn.Linear(d_model, n * n)  # full off-diagonal

        # Input/output projections
        self.W_B = nn.Linear(d_model, n)
        self.C = nn.Linear(n, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))

        self._init_params()

    def _init_params(self):
        nn.init.zeros_(self.W_d.weight)
        nn.init.constant_(self.W_d.bias, 2.2)
        nn.init.normal_(self.W_A.weight, 0, 0.01)
        nn.init.normal_(self.W_A.bias, 0, 1.0 / self.n)
        nn.init.xavier_normal_(self.W_B.weight, gain=0.1)
        nn.init.xavier_normal_(self.C.weight, gain=0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, T, _ = u.shape
        device = u.device

        h = torch.zeros(batch, self.n, device=device, dtype=u.dtype)
        outputs = []

        for t in range(T):
            u_t = u[:, t, :]

            # Build full transition matrix
            d_t = torch.sigmoid(self.W_d(u_t))  # (batch, n)
            A_flat = self.W_A(u_t)  # (batch, n*n)
            A_full = A_flat.view(batch, self.n, self.n)  # (batch, n, n)

            # Stabilize: scale off-diagonal to prevent explosion
            A_full = torch.tanh(A_full) * (1.0 / math.sqrt(self.n))

            # Add diagonal gate
            A_diag = torch.diag_embed(d_t)  # (batch, n, n)
            A_t = A_diag + A_full - torch.diag_embed(
                A_full.diagonal(dim1=-2, dim2=-1)
            )  # Replace diagonal of A_full with d_t

            # State update
            B_t = self.W_B(u_t)
            h = torch.bmm(A_t, h.unsqueeze(-1)).squeeze(-1) + B_t

            y_t = self.C(h) + self.D * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class DRSSMClassifier(nn.Module):
    """
    DR-SSM wrapped for S5 classification.

    Architecture:
      Embedding → [DRSSMLayer → LayerNorm + Residual] × num_layers → MLP → logits

    The final output takes the LAST hidden state (since S5 composition
    requires processing the full sequence before producing an answer).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n: int,
        alpha: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_dense: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n = n
        self.alpha = alpha
        self.use_dense = use_dense

        self.embedding = nn.Embedding(vocab_size, d_model)

        if use_dense:
            self.layers = nn.ModuleList([
                DenseSSMLayer(d_model, n) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DRSSMLayer(d_model, n, alpha) for _ in range(num_layers)
            ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # MLP head for classification
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) integer token indices ∈ {0, 1}

        Returns:
            logits: (batch, num_classes) — classification at final position
        """
        x = self.embedding(tokens)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))  # Pre-norm residual

        # Take final time step for classification
        x_final = x[:, -1, :]  # (batch, d_model)
        logits = self.head(x_final)  # (batch, num_classes)
        return logits

    def get_model_name(self) -> str:
        if self.use_dense:
            return f"Dense-SSM (α=n={self.n})"
        elif self.alpha == 0:
            return f"Diagonal-SSM (α=0)"
        else:
            return f"DR-SSM (α={self.alpha})"

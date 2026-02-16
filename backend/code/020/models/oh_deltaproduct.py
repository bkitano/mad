"""
Oscillatory Householder DeltaProduct (OH-DeltaProduct)

From proposal 020-oscillatory-householder-deltaproduct.

Key idea: Decompose the state-transition matrix into:
  Ã_t = H_t · R_t
where:
  R_t = oscillatory rotation-contraction (LinOSS-style block-diagonal 2×2 blocks)
  H_t = product of n_h Householder reflections (DeltaProduct-style)

Stability guarantee (Proposition from proposal):
  ||Ã_t||_2 = ||H_t R_t||_2 ≤ ||H_t||_2 · ||R_t||_2 ≤ 1 · 1 = 1
since:
  - ||H_t||_2 ≤ 1 when β ∈ [0, 2] (each reflection has eigenvalues in [-1, 1])
  - ||R_t||_2 ≤ 1 (LinOSS contraction by construction)

State update (proposal eq.):
  h_t = H_t · R_t · h_{t-1} + B_t · x_t

This model supports 3 modes via flags:
  1. OH-DeltaProduct: Full model (oscillatory + Householder, β ∈ (0,2))
  2. LinOSS-only: n_h=0, oscillatory only (should fail on non-abelian tasks)
  3. DeltaProduct-only: use_oscillatory=False, Householder only
  4. β-restricted: beta_range='half' for β ∈ (0,1) ablation (no negative eigenvalues)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OHDeltaProductSSM(nn.Module):
    """
    Single-layer OH-DeltaProduct SSM.

    Args:
        d_model: Input/output dimension
        m: Number of oscillators (state dim = 2m)
        n_h: Number of Householder reflections per step
        dt: Discretization timestep
        omega_max: Maximum frequency (clamp for stability)
        use_oscillatory: If False, skip R_t (DeltaProduct-only mode)
        beta_range: 'full' for β ∈ (0,2), 'half' for β ∈ (0,1)
    """

    def __init__(
        self,
        d_model: int,
        m: int,
        n_h: int = 2,
        dt: float = 0.1,
        omega_max: float = 100.0,
        use_oscillatory: bool = True,
        beta_range: str = 'full',  # 'full' → (0,2), 'half' → (0,1)
    ):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.n = 2 * m  # State dimension
        self.n_h = n_h
        self.dt = dt
        self.omega_max = omega_max
        self.use_oscillatory = use_oscillatory
        self.beta_range = beta_range

        # --- Oscillatory component (proposal: LinOSS rotation-contraction) ---
        if use_oscillatory:
            # A_t = ReLU(Â(x_t)) ≥ 0, input-dependent (selective)
            # Proposal: A_t = ReLU(W_A x_t)
            self.W_A = nn.Linear(d_model, m)

        # --- Householder reflective component ---
        if n_h > 0:
            # k_{t,j} = normalize(W_k^{(j)} x_t) ∈ R^{2m}
            self.W_k = nn.ModuleList([
                nn.Linear(d_model, self.n) for _ in range(n_h)
            ])
            # β_{t,j} = 2σ(w_β^{(j)} · x_t) ∈ (0,2) [or σ(...) ∈ (0,1)]
            self.w_beta = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(n_h)
            ])

        # --- Input projection B_t = W_B x_t ---
        self.B_proj = nn.Linear(d_model, self.n)

        # --- Output projection C ---
        self.C = nn.Parameter(torch.randn(d_model, self.n) * 0.01)

        # --- Direct feedthrough D ---
        self.D = nn.Parameter(torch.zeros(d_model))

        self._init_params()

    def _init_params(self):
        """Initialize parameters for stable training."""
        if self.use_oscillatory:
            # Initialize W_A so initial A ≈ small (R_t ≈ I initially)
            # This follows Risk 1 mitigation: let Householder dominate early
            nn.init.xavier_normal_(self.W_A.weight, gain=0.1)
            nn.init.constant_(self.W_A.bias, 0.0)

        if self.n_h > 0:
            for j in range(self.n_h):
                nn.init.xavier_normal_(self.W_k[j].weight, gain=0.5)
                nn.init.zeros_(self.W_k[j].bias)
                # Initialize beta bias so β starts moderate
                nn.init.constant_(self.w_beta[j].bias, 0.0)

        nn.init.xavier_normal_(self.B_proj.weight, gain=0.5)

    def _compute_oscillatory_transition(self, u_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute R_t · h (oscillatory rotation-contraction applied to state).

        R_t is block-diagonal with m 2×2 blocks (proposal eq.):
          [[S_k, -Δt·A_k·S_k],
           [Δt·S_k, S_k]]
        where S_k = 1/(1 + Δt²·A_k), A_k = ReLU(W_A x_t)_k

        Args:
            u_t: (batch, d_model) input at time t
            h: (batch, 2m) state vector [z_1,...,z_m, y_1,...,y_m]

        Returns:
            R_t · h: (batch, 2m)
        """
        # A_t = ReLU(W_A x_t) ≥ 0 (input-dependent, selective)
        A = F.relu(self.W_A(u_t))  # (batch, m), ≥ 0
        A = torch.clamp(A, max=self.omega_max)

        # S_k = (1 + Δt² A_k)^{-1} ∈ (0, 1]
        S = 1.0 / (1.0 + self.dt ** 2 * A)  # (batch, m)

        # Split state: h = [z, y] where z = positions 0..m-1, y = positions m..2m-1
        z = h[:, :self.m]      # (batch, m)
        y = h[:, self.m:]      # (batch, m)

        # Apply 2×2 block rotation-contraction (proposal eq.):
        # z_new = S_k · z_k - Δt · A_k · S_k · y_k
        # y_new = Δt · S_k · z_k + S_k · y_k
        z_new = S * z - self.dt * A * S * y
        y_new = self.dt * S * z + S * y

        return torch.cat([z_new, y_new], dim=-1)  # (batch, 2m)

    def _compute_householder_product(self, u_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute H_t · h where H_t = Π_{j=1}^{n_h} (I - β_{t,j} k_{t,j} k_{t,j}^T).

        Each Householder reflection is applied sequentially:
          h ← h - β_{t,j} · k_{t,j} · (k_{t,j}^T · h)

        This is O(n_h · 2m) per step — efficient rank-1 updates.

        Args:
            u_t: (batch, d_model) input at time t
            h: (batch, 2m) state vector

        Returns:
            H_t · h: (batch, 2m)
        """
        for j in range(self.n_h):
            # k_{t,j} = normalize(W_k^{(j)} x_t) (proposal eq.)
            k = self.W_k[j](u_t)  # (batch, 2m)
            k = F.normalize(k, dim=-1)  # Unit vector

            # β_{t,j} ∈ (0, 2) or (0, 1) depending on mode
            if self.beta_range == 'full':
                # β = 2σ(w_β · x_t) ∈ (0, 2) — enables negative eigenvalues
                beta = 2.0 * torch.sigmoid(self.w_beta[j](u_t))  # (batch, 1)
            else:
                # β = σ(w_β · x_t) ∈ (0, 1) — ablation: no negative eigenvalues
                beta = torch.sigmoid(self.w_beta[j](u_t))  # (batch, 1)

            # Apply reflection: h ← h - β · k · (k^T · h)
            # k^T · h: inner product → (batch, 1)
            dot = (k * h).sum(dim=-1, keepdim=True)
            h = h - beta * k * dot

        return h

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequential recurrence.

        State update (proposal eq.):
          h_t = H_t · R_t · h_{t-1} + B_t · x_t

        Args:
            u: Input tensor (batch, seq_len, d_model)

        Returns:
            y: Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        # Initialize state h_0 = 0
        h = torch.zeros(batch_size, self.n, device=device, dtype=dtype)

        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)

            # Step 1: Apply oscillatory rotation-contraction R_t
            if self.use_oscillatory:
                h_rot = self._compute_oscillatory_transition(u_t, h)  # R_t · h_{t-1}
            else:
                h_rot = h  # Skip oscillatory (DeltaProduct-only mode)

            # Step 2: Apply Householder reflections H_t
            if self.n_h > 0:
                h_ref = self._compute_householder_product(u_t, h_rot)  # H_t · R_t · h_{t-1}
            else:
                h_ref = h_rot  # Skip Householder (LinOSS-only mode)

            # Step 3: Add input forcing
            B_t = self.B_proj(u_t)  # (batch, 2m)
            h = h_ref + B_t

            # Step 4: Output projection
            y_t = F.linear(h, self.C) + u_t * self.D  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class OHDeltaProductClassifier(nn.Module):
    """
    OH-DeltaProduct wrapped for classification (S3 composition task).

    Architecture:
      Embedding → [OHDeltaProductSSM → LayerNorm + Residual] × num_layers → MLP head → logits

    This mirrors the pattern from code/003 (OscGateSSMClassifier).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        m: int,
        num_classes: int,
        n_h: int = 2,
        num_layers: int = 1,
        dt: float = 0.1,
        omega_max: float = 100.0,
        use_oscillatory: bool = True,
        beta_range: str = 'full',
        dropout: float = 0.05,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            OHDeltaProductSSM(
                d_model=d_model, m=m, n_h=n_h, dt=dt,
                omega_max=omega_max, use_oscillatory=use_oscillatory,
                beta_range=beta_range,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # MLP classification head
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

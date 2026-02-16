"""
Capacitance-Coupled Multi-Scale SSM (MC-SSM)
From Proposal 019: capacitance-coupled-multi-scale-ssm

Core idea:
  Partition state space into k blocks at different timescales.
  Couple them via a small k×k capacitance matrix C_t.

  h_t = A_diag * h_{t-1} + U @ C_t @ V^T @ h_{t-1} + B @ x_t
  y_t = sum_i alpha_i * C_i @ h_t^(i)

  where A_diag is block-diagonal with per-scale transitions,
  C_t is input-dependent capacitance coupling (k×k),
  and alpha is input-dependent scale-selection gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm + self.eps)


class CapacitanceMatrix(nn.Module):
    """
    Input-dependent capacitance matrix C_t ∈ R^{k×k}.

    C_t = Diag(c_1, ..., c_k) + C_off(x_t)

    Diagonal: c_i = sigmoid(w_c^i · x_t) — self-coupling gate per scale
    Off-diagonal: C_off_{ij} = -softplus(w_{ij} · x_t) for i≠j
      (non-positive, ensuring diagonal dominance)

    Proposal eq: C_t = Diag(c^(1)_t, ..., c^(k)_t) + C_off(x_t)
    """
    def __init__(self, d_input: int, k: int):
        super().__init__()
        self.k = k
        # Diagonal self-coupling: d_input -> k
        self.diag_proj = nn.Linear(d_input, k)
        # Off-diagonal coupling: d_input -> k*(k-1) (only off-diag entries)
        self.off_diag_proj = nn.Linear(d_input, k * (k - 1))

        # Initialize near zero for warm-start (proposal: initialize C_off ≈ 0)
        nn.init.zeros_(self.off_diag_proj.weight)
        nn.init.constant_(self.off_diag_proj.bias, -3.0)  # softplus(-3) ≈ 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_input) — input at current timestep
        Returns:
            C_t: (batch, k, k) — capacitance matrix
        """
        batch = x.shape[0]

        # Diagonal entries: sigmoid for (0, 1) range
        c_diag = torch.sigmoid(self.diag_proj(x))  # (batch, k)

        # Off-diagonal entries: -softplus for non-positive values
        off_diag_raw = self.off_diag_proj(x)  # (batch, k*(k-1))
        off_diag_vals = -F.softplus(off_diag_raw)  # Non-positive

        # Build full k×k matrix
        C = torch.zeros(batch, self.k, self.k, device=x.device, dtype=x.dtype)

        # Fill diagonal
        C[:, range(self.k), range(self.k)] = c_diag

        # Fill off-diagonal
        idx = 0
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    C[:, i, j] = off_diag_vals[:, idx]
                    idx += 1

        return C


class MultiScaleSSMLayer(nn.Module):
    """
    Single layer of the Multi-Scale Capacitance SSM.

    State transition:
      h_t = A_diag_t * h_{t-1} + U @ C_t @ V^T @ h_{t-1} + B_t @ x_t
      y_t = sum_i alpha_t^(i) * C_i @ h_t^(i)

    where A_diag_t is block-diagonal with per-scale diagonal transitions.
    """
    def __init__(self, d_model: int, n_state: int, k_scales: int,
                 dt_min: float = 0.001, dt_max: float = 1.0,
                 use_coupling: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state
        self.k_scales = k_scales
        self.n_per_scale = n_state // k_scales
        self.use_coupling = use_coupling

        assert n_state % k_scales == 0, "n_state must be divisible by k_scales"

        # --- Per-scale timescale parameters ---
        # Geometric spacing: dt_i = dt_min * rho^(i-1)
        # Proposal eq: Δt_i = Δt_min · ρ^{i-1}, ρ = (Δt_max/Δt_min)^{1/(k-1)}
        rho = (dt_max / dt_min) ** (1.0 / max(k_scales - 1, 1))
        dt_values = torch.tensor([dt_min * (rho ** i) for i in range(k_scales)])
        self.register_buffer('dt_scales', dt_values)  # (k,)

        # Learnable eigenvalues per scale (log-parameterized for stability)
        # Proposal eq: A^(i)_t = exp(-Δt_i · Λ^(i))
        self.log_lambda = nn.Parameter(
            torch.randn(k_scales, self.n_per_scale) * 0.1 + 1.0
        )  # (k, n_per_scale) — will be positive after exp

        # --- Input/output projections ---
        # B: input -> state, C: state -> output
        self.B_proj = nn.Linear(d_model, n_state)
        self.C_projs = nn.ModuleList([
            nn.Linear(self.n_per_scale, d_model) for _ in range(k_scales)
        ])

        # --- Capacitance coupling ---
        if use_coupling:
            # Capacitance matrix: input-dependent k×k coupling
            self.capacitance = CapacitanceMatrix(d_model, k_scales)

            # Interface projections U, V ∈ R^{n × k}
            # Each column selects/projects onto one scale's "boundary"
            self.U = nn.Parameter(torch.randn(n_state, k_scales) * 0.01)
            self.V = nn.Parameter(torch.randn(n_state, k_scales) * 0.01)

        # --- Scale-selection gate ---
        # alpha_t = softmax(w_alpha · x_t) — input-dependent scale weighting
        self.alpha_proj = nn.Linear(d_model, k_scales)

    def _get_A_diag(self):
        """
        Compute block-diagonal transition matrix entries.

        A^(i) = exp(-Δt_i · Λ^(i)) where Λ^(i) > 0 (via softplus).
        Returns (n_state,) vector of diagonal entries.
        """
        # Ensure positive eigenvalues
        lambdas = F.softplus(self.log_lambda)  # (k, n_per_scale), positive

        # Scale by timescale: A_i = exp(-dt_i * lambda_i)
        # This gives decay rates: fast scales (small dt) decay slowly per step,
        # slow scales (large dt) decay faster per step
        dt_expanded = self.dt_scales.unsqueeze(1)  # (k, 1)
        A_diag = torch.exp(-dt_expanded * lambdas)  # (k, n_per_scale)

        return A_diag.reshape(-1)  # (n_state,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        # Get diagonal transition entries
        A_diag = self._get_A_diag()  # (n_state,)

        # Project input to state space
        B_x = self.B_proj(x)  # (batch, seq_len, n_state)

        # Initialize hidden state
        h = torch.zeros(batch, self.n_state, device=device, dtype=x.dtype)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)

            # Block-diagonal transition: h = A_diag * h
            h_new = A_diag.unsqueeze(0) * h  # (batch, n_state)

            # Capacitance coupling: h += U @ C_t @ V^T @ h
            if self.use_coupling:
                C_t = self.capacitance(x_t)  # (batch, k, k)
                # V^T @ h: project state to k-dim interface
                Vh = torch.bmm(
                    self.V.T.unsqueeze(0).expand(batch, -1, -1),  # (batch, k, n)
                    h.unsqueeze(2)  # (batch, n, 1)
                ).squeeze(2)  # (batch, k)
                # C_t @ (V^T @ h): apply capacitance coupling
                CVh = torch.bmm(C_t, Vh.unsqueeze(2)).squeeze(2)  # (batch, k)
                # U @ (C_t @ V^T @ h): project back to full state
                coupling = torch.mm(CVh, self.U.T)  # (batch, n_state)
                h_new = h_new + coupling

            # Add input
            h_new = h_new + B_x[:, t, :]  # (batch, n_state)
            h = h_new

            # Output: weighted sum across scales
            # alpha_t = softmax(w_alpha · x_t)
            alpha = F.softmax(self.alpha_proj(x_t), dim=-1)  # (batch, k)

            # y_t = sum_i alpha_i * C_i @ h^(i)
            h_scales = h.view(batch, self.k_scales, self.n_per_scale)
            y_t = torch.zeros(batch, d_model, device=device, dtype=x.dtype)
            for i in range(self.k_scales):
                y_i = self.C_projs[i](h_scales[:, i, :])  # (batch, d_model)
                y_t = y_t + alpha[:, i:i+1] * y_i

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class MultiScaleCapacitanceSSM(nn.Module):
    """
    Full MC-SSM model for sequence classification.

    Architecture: input_proj -> [MC-SSM layer + RMSNorm + residual] x L -> pool -> classifier
    """
    def __init__(self, d_input: int, d_model: int, n_state: int, k_scales: int,
                 n_layers: int, n_classes: int, dt_min: float = 0.001,
                 dt_max: float = 1.0, dropout: float = 0.1,
                 use_coupling: bool = True):
        super().__init__()
        self.d_model = d_model
        self.model_type = "MC-SSM" if use_coupling else "Uncoupled-MS-SSM"

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # SSM layers with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                MultiScaleSSMLayer(d_model, n_state, k_scales,
                                   dt_min=dt_min, dt_max=dt_max,
                                   use_coupling=use_coupling)
            )
            self.norms.append(RMSNorm(d_model))

        self.final_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_input) — raw signal
        Returns:
            logits: (batch, n_classes)
        """
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.dropout(h)

        # SSM layers with pre-norm residual
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = self.dropout(h)
            h = h + residual

        h = self.final_norm(h)

        # Global average pooling over sequence
        h = h.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(h)  # (batch, n_classes)
        return logits

    def get_capacitance_stats(self):
        """Get statistics about the learned capacitance matrices for analysis."""
        stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'capacitance') and layer.use_coupling:
                # Get A_diag to check timescale separation
                A_diag = layer._get_A_diag()
                A_scales = A_diag.view(layer.k_scales, layer.n_per_scale)

                stats[f'layer_{i}'] = {
                    'dt_scales': layer.dt_scales.detach().cpu().tolist(),
                    'A_diag_mean_per_scale': A_scales.mean(dim=1).detach().cpu().tolist(),
                    'A_diag_std_per_scale': A_scales.std(dim=1).detach().cpu().tolist(),
                }
        return stats


class MonolithicSSMLayer(nn.Module):
    """
    Monolithic SSM baseline — single diagonal transition, same total state dim.
    No multi-scale structure, no coupling.

    h_t = A * h_{t-1} + B @ x_t
    y_t = C @ h_t
    """
    def __init__(self, d_model: int, n_state: int):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state

        # Single diagonal transition (log-parameterized)
        self.log_A = nn.Parameter(torch.randn(n_state) * 0.1 - 1.0)

        # Input/output projections
        self.B_proj = nn.Linear(d_model, n_state)
        self.C_proj = nn.Linear(n_state, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        device = x.device

        # Diagonal transition: A = sigmoid(log_A) for stability in (0, 1)
        A = torch.sigmoid(self.log_A)  # (n_state,)

        B_x = self.B_proj(x)  # (batch, seq_len, n_state)

        h = torch.zeros(batch, self.n_state, device=device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = A.unsqueeze(0) * h + B_x[:, t, :]
            y_t = self.C_proj(h)  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MonolithicSSM(nn.Module):
    """
    Monolithic SSM baseline for sequence classification.
    Same total state dimension n=32, but no multi-scale structure.
    """
    def __init__(self, d_input: int, d_model: int, n_state: int,
                 n_layers: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.model_type = "Monolithic-SSM"

        self.input_proj = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MonolithicSSMLayer(d_model, n_state))
            self.norms.append(RMSNorm(d_model))

        self.final_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.dropout(h)

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = self.dropout(h)
            h = h + residual

        h = self.final_norm(h)
        h = h.mean(dim=1)
        logits = self.classifier(h)
        return logits


def UncoupledMultiScaleSSM(d_input: int, d_model: int, n_state: int, k_scales: int,
                           n_layers: int, n_classes: int, dt_min: float = 0.001,
                           dt_max: float = 1.0, dropout: float = 0.1):
    """
    Uncoupled multi-scale SSM — same block-diagonal structure but C=0.
    This is the MC-SSM with use_coupling=False.
    """
    return MultiScaleCapacitanceSSM(
        d_input=d_input, d_model=d_model, n_state=n_state, k_scales=k_scales,
        n_layers=n_layers, n_classes=n_classes, dt_min=dt_min, dt_max=dt_max,
        dropout=dropout, use_coupling=False
    )

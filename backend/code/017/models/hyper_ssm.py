"""
Hyperoctahedral Signed-Permutation SSM (HyperSSM).

State transition: A_t = D(x_t) * P(x_t) ∈ B_n (hyperoctahedral group)

where:
  - D(x_t) = diag(s_1, ..., s_n): sign matrix, s_i ∈ [-1, +1]
    Relaxation: s_i = 2*sigmoid(W_s @ x_t + b_s)_i - 1  (Proposal eq. for sign component)
  - P(x_t): permutation matrix from Gumbel-Sinkhorn
    Relaxation: Sinkhorn^L(W_P @ x_t / tau) with ST hardening  (Proposal eq. for perm component)

Gating for forgetting (Krohn-Rhodes aperiodic component):
  h_t = gamma(x_t) ⊙ (A_t h_{t-1}) + (1 - gamma(x_t)) ⊙ B x_t

where gamma(x_t) = sigmoid(W_gamma @ x_t + b_gamma) ∈ (0, 1)^n

References:
  - Proposal 017: Hyperoctahedral Signed-Permutation SSM State Transitions
  - Mena et al., 2018: Learning latent permutations with Gumbel-Sinkhorn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def sinkhorn_normalize(log_alpha: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Sinkhorn normalization to produce doubly stochastic matrix.

    Args:
        log_alpha: (batch, n, n) log-score matrix
        num_iters: number of Sinkhorn iterations

    Returns:
        (batch, n, n) doubly stochastic matrix
    """
    for _ in range(num_iters):
        # Row normalization (log-space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization (log-space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def hungarian_hard(soft_perm: torch.Tensor) -> torch.Tensor:
    """
    Hardening via greedy row-wise argmax (approximation to Hungarian algorithm).
    For small n (e.g., 8), greedy argmax with conflict resolution works well.

    Uses straight-through estimator: forward uses hard, backward uses soft gradient.

    Args:
        soft_perm: (batch, n, n) doubly stochastic matrix

    Returns:
        (batch, n, n) permutation matrix (hard), with ST gradient
    """
    batch, n, _ = soft_perm.shape
    hard_perm = torch.zeros_like(soft_perm)

    # Greedy assignment: for each row, pick the best available column
    soft_np = soft_perm.detach()
    for b in range(batch):
        available = torch.ones(n, device=soft_perm.device, dtype=torch.bool)
        # Process rows in order of "confidence" (max value in row, descending)
        row_max_vals = soft_np[b].max(dim=-1).values
        row_order = row_max_vals.argsort(descending=True)

        for row in row_order:
            # Among available columns, pick the one with highest score
            scores = soft_np[b, row].clone()
            scores[~available] = -float('inf')
            col = scores.argmax()
            hard_perm[b, row, col] = 1.0
            available[col] = False

    # Straight-through estimator: hard in forward, soft gradient in backward
    return hard_perm + soft_perm - soft_perm.detach()


class HyperSSMLayer(nn.Module):
    """
    Single HyperSSM layer with signed-permutation state transitions.

    For each head:
      1. Compute signs s(x_t) = 2*sigmoid(W_s x_t) - 1 ∈ [-1, 1]^n
      2. Compute soft permutation P(x_t) = Sinkhorn(W_P x_t / tau)
      3. Hard permutation via ST: P_hard = Hungarian(P_soft) + (P_soft - P_soft.detach())
      4. State update: h_t = gamma ⊙ (D(s) P h_{t-1}) + (1 - gamma) ⊙ B x_t

    where gamma = sigmoid(W_gamma x_t + b_gamma) is the forget gate.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 8,
        num_heads: int = 4,
        sinkhorn_iters: int = 5,
        tau: float = 1.0,
        use_hard_perm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau
        self.use_hard_perm = use_hard_perm

        n = state_dim  # state dimension per head

        # --- Sign component: W_s projects input to n sign logits per head ---
        # s_i(x_t) = 2*sigmoid(W_s x_t + b_s)_i - 1  (Proposal: Sign component eq.)
        self.sign_proj = nn.Linear(d_model, num_heads * n)

        # --- Permutation component: W_P projects input to n x n cost matrix per head ---
        # P(x_t) = Sinkhorn^L(W_P x_t / tau)  (Proposal: Permutation component eq.)
        self.perm_proj = nn.Linear(d_model, num_heads * n * n)

        # --- Forget gate: gamma(x_t) = sigmoid(W_gamma x_t + b_gamma) ---
        # (Proposal: Gating for Forgetting eq.)
        self.gate_proj = nn.Linear(d_model, num_heads * n)

        # --- Input projection: B x_t ---
        self.input_proj = nn.Linear(d_model, num_heads * n)

        # --- Output projection: C h_t ---
        self.output_proj = nn.Linear(num_heads * n, d_model)

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        # Sign projection: small init so signs start near 0 (undecided)
        nn.init.normal_(self.sign_proj.weight, std=0.01)
        nn.init.zeros_(self.sign_proj.bias)

        # Perm projection: small init so Sinkhorn starts near uniform
        nn.init.normal_(self.perm_proj.weight, std=0.01)
        nn.init.zeros_(self.perm_proj.bias)

        # Gate: init bias positive so gamma starts near 1 (keep state)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Input/output projections: Xavier
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape
        n = self.state_dim
        H = self.num_heads

        residual = x
        x = self.norm(x)

        # Project input to per-head components
        # Signs: (batch, seq_len, H, n)
        sign_logits = self.sign_proj(x).view(batch, seq_len, H, n)
        signs = 2.0 * torch.sigmoid(sign_logits) - 1.0  # ∈ [-1, 1]

        # Permutation cost matrices: (batch, seq_len, H, n, n)
        perm_logits = self.perm_proj(x).view(batch, seq_len, H, n, n)
        perm_logits = perm_logits / self.tau

        # Gate: (batch, seq_len, H, n)
        gamma = torch.sigmoid(self.gate_proj(x).view(batch, seq_len, H, n))

        # Input injection: (batch, seq_len, H, n)
        b_input = self.input_proj(x).view(batch, seq_len, H, n)

        # Sequential scan over time steps
        h = torch.zeros(batch, H, n, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Compute permutation matrix for this timestep
            # Flatten batch and heads for Sinkhorn
            perm_t = perm_logits[:, t]  # (batch, H, n, n)
            perm_flat = perm_t.reshape(batch * H, n, n)

            # Sinkhorn normalization -> doubly stochastic
            soft_perm = sinkhorn_normalize(perm_flat, num_iters=self.sinkhorn_iters)

            # Hardening via ST estimator
            if self.use_hard_perm:
                perm_matrix = hungarian_hard(soft_perm)  # (batch*H, n, n)
            else:
                perm_matrix = soft_perm

            perm_matrix = perm_matrix.view(batch, H, n, n)

            # Apply signed permutation: A_t h = D(s) P h
            # Step 1: P h  (permutation)
            h_perm = torch.einsum('bhij,bhj->bhi', perm_matrix, h)

            # Step 2: D(s) (P h) = signs ⊙ (P h)
            signs_t = signs[:, t]  # (batch, H, n)
            h_signed_perm = signs_t * h_perm

            # Gated update: h_t = gamma ⊙ (A_t h_{t-1}) + (1 - gamma) ⊙ B x_t
            gamma_t = gamma[:, t]  # (batch, H, n)
            b_t = b_input[:, t]  # (batch, H, n)
            h = gamma_t * h_signed_perm + (1.0 - gamma_t) * b_t

            outputs.append(h)

        # Stack outputs: (batch, seq_len, H, n)
        outputs = torch.stack(outputs, dim=1)

        # Flatten heads and project to d_model
        outputs = outputs.reshape(batch, seq_len, H * n)
        outputs = self.output_proj(outputs)
        outputs = self.dropout(outputs)

        return residual + outputs


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return residual + self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class HyperSSM(nn.Module):
    """
    Full HyperSSM model for sequence classification.

    Architecture: Embedding -> L x (HyperSSMLayer + SwiGLU) -> Output Head

    Matches proposal: 1-layer HyperSSM, n=8 state dim, d=32 input dim, ~50K params
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        state_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 1,
        sinkhorn_iters: int = 5,
        tau: float = 1.0,
        use_hard_perm: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HyperSSMLayer(
                d_model=d_model,
                state_dim=state_dim,
                num_heads=num_heads,
                sinkhorn_iters=sinkhorn_iters,
                tau=tau,
                use_hard_perm=use_hard_perm,
                dropout=dropout,
            ))
            self.layers.append(SwiGLU(d_model, dropout=dropout))

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, num_classes) logits
        """
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            h = layer(h)

        h = self.output_norm(h)
        logits = self.output_head(h)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

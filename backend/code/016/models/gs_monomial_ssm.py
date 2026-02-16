"""
Group-and-Shuffle Monomial SSM (GS-Monomial SSM) — Vectorized Implementation

State transition at each timestep:
    A_t = L_t @ P_shuffle @ R_t

where:
- R_t = block-diag(pi_R^(1) D_R^(1), ..., pi_R^(r) D_R^(r))
- L_t = block-diag(pi_L^(1) D_L^(1), ..., pi_L^(r) D_L^(r))
- P_shuffle = fixed stride permutation (deinterleave)

All block operations are batched/vectorized for efficiency:
- Permutations computed for all r blocks simultaneously
- Diagonal scaling vectorized across all blocks

For MVE: n=16, b=4, r=4 blocks, d_model=64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation for channel mixing."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        d_ff = int(8 / 3 * d_model)
        d_ff = max(d_ff, 16)

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def sinkhorn_normalize(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """
    Sinkhorn normalization to produce doubly-stochastic matrices.

    Args:
        log_alpha: (batch, r, b, b) log-space assignment matrices for all blocks
        n_iters: number of Sinkhorn iterations

    Returns:
        (batch, r, b, b) doubly-stochastic matrices
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def hungarian_hard(soft_perm: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator: hard permutation in forward, soft gradient in backward.

    Args:
        soft_perm: (batch, r, b, b) doubly-stochastic matrices

    Returns:
        (batch, r, b, b) hard permutation matrices (STE)
    """
    hard = torch.zeros_like(soft_perm)
    indices = soft_perm.argmax(dim=-2)  # (batch, r, b) — which row for each col
    hard.scatter_(-2, indices.unsqueeze(-2), 1.0)
    return (hard - soft_perm).detach() + soft_perm


class BatchedMonomialFactor(nn.Module):
    """
    Vectorized block-diagonal monomial factor.

    Processes all r blocks simultaneously using batched operations.
    Each block applies: h' = D * (P @ h) where P is a permutation and D is diagonal.

    Input-dependent via projections from x_t.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,  # r
        block_size: int,  # b
        sinkhorn_iters: int = 5,
        tau: float = 0.5,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau

        n = num_blocks * block_size  # total state dim

        # Permutation logits: d_model -> r * b * b (all blocks at once)
        self.perm_proj = nn.Linear(d_model, num_blocks * block_size * block_size, bias=False)

        # Diagonal values: d_model -> r * b (all blocks at once)
        self.diag_proj = nn.Linear(d_model, n, bias=False)

        # Contraction gate: d_model -> r * b (all blocks at once)
        self.alpha_proj = nn.Linear(d_model, n, bias=False)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Apply block-diagonal monomial factor to full state vector.

        Args:
            x_t: (batch, d_model) — input at current timestep
            h: (batch, n) — full state vector (n = r * b)

        Returns:
            h_new: (batch, n) — transformed state vector
        """
        batch = x_t.shape[0]
        r = self.num_blocks
        b = self.block_size

        # Reshape state into blocks: (batch, r, b)
        h_blocks = h.view(batch, r, b)

        # --- Permutations for all blocks ---
        perm_logits = self.perm_proj(x_t).view(batch, r, b, b)  # (batch, r, b, b)
        perm_logits = perm_logits / self.tau

        # Sinkhorn on all blocks simultaneously
        soft_perm = sinkhorn_normalize(perm_logits, self.sinkhorn_iters)  # (batch, r, b, b)
        hard_perm = hungarian_hard(soft_perm)  # (batch, r, b, b)

        # Apply permutation to all blocks: h' = P @ h for each block
        # (batch, r, b, b) @ (batch, r, b, 1) -> (batch, r, b, 1)
        h_permuted = torch.matmul(hard_perm, h_blocks.unsqueeze(-1)).squeeze(-1)  # (batch, r, b)

        # --- Diagonal scaling for all blocks ---
        alpha = torch.sigmoid(self.alpha_proj(x_t)).view(batch, r, b)  # (batch, r, b)
        diag_vals = alpha * torch.tanh(self.diag_proj(x_t).view(batch, r, b))  # (batch, r, b)

        # Apply diagonal: h'' = D * h'
        h_new = diag_vals * h_permuted  # (batch, r, b)

        return h_new.view(batch, r * b)  # (batch, n)


class GSMonomialSSMLayer(nn.Module):
    """
    GS-Monomial SSM Layer (vectorized).

    State transition: h_t = A_t @ h_{t-1} + B_t @ x_t
    where A_t = L_t @ P_shuffle @ R_t

    All block operations are vectorized across r blocks.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        block_size: int = 4,
        sinkhorn_iters: int = 5,
        tau: float = 0.5,
        use_shuffle: bool = True,
    ):
        super().__init__()

        assert state_dim % block_size == 0

        self.d_model = d_model
        self.state_dim = state_dim
        self.block_size = block_size
        self.num_blocks = state_dim // block_size
        self.use_shuffle = use_shuffle

        # R factor: block-diagonal monomial (pre-shuffle)
        self.R_factor = BatchedMonomialFactor(
            d_model, self.num_blocks, block_size, sinkhorn_iters, tau
        )

        # L factor: block-diagonal monomial (post-shuffle)
        self.L_factor = BatchedMonomialFactor(
            d_model, self.num_blocks, block_size, sinkhorn_iters, tau
        )

        # Fixed shuffle permutation
        if use_shuffle:
            self.register_buffer('shuffle_perm', self._build_stride_permutation())

        # Input/output projections
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(state_dim, d_model, bias=False)
        self.out_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(d_model)

    def _build_stride_permutation(self) -> torch.Tensor:
        """
        Build stride (deinterleave) permutation.

        For n=16, b=4, r=4:
        [0,1,2,3 | 4,5,6,7 | 8,9,10,11 | 12,13,14,15]
        -> [0,4,8,12 | 1,5,9,13 | 2,6,10,14 | 3,7,11,15]
        """
        n = self.state_dim
        b = self.block_size
        r = self.num_blocks
        indices = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            indices[i] = (i % r) * b + (i // r)
        return indices

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sequential scan through input sequence.

        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d_model)

            # Step 1: R factor (block-diagonal monomial)
            h = self.R_factor(x_t, h)

            # Step 2: P_shuffle (fixed permutation)
            if self.use_shuffle:
                h = h[:, self.shuffle_perm]

            # Step 3: L factor (block-diagonal monomial)
            h = self.L_factor(x_t, h)

            # Step 4: Input contribution
            b_t = self.B_proj(x_t)
            if mask is not None:
                b_t = b_t * mask[:, t].unsqueeze(-1)
            h = h + b_t

            # Step 5: Output
            y_t = self.C_proj(h)
            gate = torch.sigmoid(self.out_gate(x_t))
            y_t = gate * y_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = self.out_norm(y)
        return y


class GSMonomialSSMBlock(nn.Module):
    """GS-Monomial SSM block with pre-norm residual + SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        block_size: int = 4,
        sinkhorn_iters: int = 5,
        tau: float = 0.5,
        dropout: float = 0.1,
        use_shuffle: bool = True,
    ):
        super().__init__()

        self.ssm_norm = RMSNorm(d_model)
        self.ssm = GSMonomialSSMLayer(
            d_model=d_model,
            state_dim=state_dim,
            block_size=block_size,
            sinkhorn_iters=sinkhorn_iters,
            tau=tau,
            use_shuffle=use_shuffle,
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.ssm(self.ssm_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GSMonomialSSM(nn.Module):
    """
    Full GS-Monomial SSM model for group composition tasks.

    Architecture:
    - Token + position embeddings
    - L x (GS-Monomial SSM + SwiGLU) blocks with pre-norm residuals
    - Final norm + MLP classifier
    """

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        max_seq_len: int = 64,
        d_model: int = 32,
        state_dim: int = 16,
        block_size: int = 4,
        num_layers: int = 2,
        sinkhorn_iters: int = 5,
        tau: float = 0.5,
        dropout: float = 0.1,
        use_shuffle: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_shuffle = use_shuffle

        self.token_embed = nn.Embedding(num_tokens, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            GSMonomialSSMBlock(
                d_model=d_model,
                state_dim=state_dim,
                block_size=block_size,
                sinkhorn_iters=sinkhorn_iters,
                tau=tau,
                dropout=dropout,
                use_shuffle=use_shuffle,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.head_fc1 = nn.Linear(d_model, d_model * 2)
        self.head_fc2 = nn.Linear(d_model * 2, num_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        x = F.gelu(self.head_fc1(x))
        logits = self.head_fc2(x)
        return logits


if __name__ == "__main__":
    batch = 4
    seq_len = 22
    num_tokens = 5
    num_classes = 120

    model = GSMonomialSSM(
        num_tokens=num_tokens, num_classes=num_classes, max_seq_len=seq_len,
        d_model=32, state_dim=16, block_size=4, num_layers=2, use_shuffle=True,
    )
    tokens = torch.randint(0, num_tokens, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    logits = model(tokens, mask)
    print(f"GS-Monomial (shuffle): {logits.shape}, params={sum(p.numel() for p in model.parameters()):,}")

    model2 = GSMonomialSSM(
        num_tokens=num_tokens, num_classes=num_classes, max_seq_len=seq_len,
        d_model=32, state_dim=16, block_size=4, num_layers=2, use_shuffle=False,
    )
    logits2 = model2(tokens, mask)
    print(f"GS-Monomial (no shuffle): {logits2.shape}, params={sum(p.numel() for p in model2.parameters()):,}")

    loss = logits.sum()
    loss.backward()
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
    print(f"NaN grads: {has_nan}")

"""
Column-Sparse DeltaNet (CS-DeltaNet)

DeltaNet with input-dependent permutation routing, inspired by PD-SSM.

Key innovation: Adds an input-dependent permutation P(x_t) to the state transition:
    S_t = P(x_t) · S_{t-1} + β_t · k_t (v_t - S_{t-1}^T k_t)^T

This enables:
1. Permutation routing between state dimensions (like PD-SSM)
2. Optional negative eigenvalues for sign-flipping dynamics (like NEG-DeltaNet)

The permutation P(x_t) is learned via Gumbel-softmax during training.

Variants:
- CS-DeltaNet: Column-sparse permutation with β ∈ (0, 1)
- CS-NEG-DeltaNet: Column-sparse permutation with β ∈ (0, 2)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        d_ff = ((d_ff + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def gumbel_softmax_permutation(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """
    Generate a column-sparse permutation matrix via Gumbel-softmax.

    For each column j, sample a one-hot vector indicating which row is 1.
    This creates a permutation matrix P where each column has exactly one 1.

    Args:
        logits: (batch, nhead, state_dim, state_dim) - logits for each column's row selection
        tau: temperature for Gumbel-softmax (lower = more discrete)
        hard: if True, use straight-through estimator for discrete samples

    Returns:
        P: (batch, nhead, state_dim, state_dim) - permutation matrices
    """
    # For each column, apply Gumbel-softmax to select which row has the 1
    # logits[..., i, j] = logit for row i, column j
    # We want to sample one row per column, so we apply along dim=-2

    batch, nhead, dim, _ = logits.shape

    # Reshape to apply gumbel_softmax per column
    # (batch, nhead, dim, dim) -> (batch * nhead * dim, dim)
    logits_flat = logits.permute(0, 1, 3, 2).reshape(-1, dim)  # (B*H*D, D)

    # Gumbel-softmax: sample one-hot vectors
    soft_samples = F.gumbel_softmax(logits_flat, tau=tau, hard=hard, dim=-1)

    # Reshape back: (B*H*D, D) -> (batch, nhead, dim, dim)
    P = soft_samples.reshape(batch, nhead, dim, dim).permute(0, 1, 3, 2)

    return P


class CSDeltaNetLayer(nn.Module):
    """
    Column-Sparse DeltaNet layer with input-dependent permutation routing.

    State update:
        S_t = P(x_t) · S_{t-1} + β_t · k_t ⊗ (v_t - S_{t-1}^T k_t)

    Where P(x_t) is an input-dependent permutation matrix.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        allow_neg_eigval: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.allow_neg_eigval = allow_neg_eigval
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Beta (learning rate for delta update)
        self.beta_proj = nn.Linear(d_model, nhead, bias=False)

        # Permutation logits projection: input -> permutation logits per head
        # For each head, produce (head_dim x head_dim) logits for the permutation
        self.perm_proj = nn.Linear(d_model, nhead * self.head_dim * self.head_dim, bias=False)

        # Output normalization
        self.out_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def _get_permutation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute input-dependent permutation matrix P(x_t).

        Args:
            x: (batch, d_model) - input at time t

        Returns:
            P: (batch, nhead, head_dim, head_dim) - permutation matrices
        """
        batch_size = x.shape[0]

        # Project to permutation logits
        perm_logits = self.perm_proj(x)  # (batch, nhead * head_dim * head_dim)
        perm_logits = perm_logits.view(batch_size, self.nhead, self.head_dim, self.head_dim)

        # Sample permutation via Gumbel-softmax
        P = gumbel_softmax_permutation(perm_logits, tau=self.gumbel_tau, hard=self.gumbel_hard)

        return P

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) with 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape for multi-head
        q = self.q_proj(x)
        q = F.silu(q)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        q = self._l2_normalize(q, dim=-1)

        k = self.k_proj(x)
        k = F.silu(k)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self._l2_normalize(k, dim=-1)

        v = self.v_proj(x)
        v = F.silu(v)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

        # Beta: sigmoid -> (0, 1), optionally scaled to (0, 2)
        beta = torch.sigmoid(self.beta_proj(x))  # (batch, seq_len, nhead)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Sequential update with permutation routing
        outputs = []
        M = torch.zeros(batch_size, self.nhead, self.head_dim, self.head_dim,
                       device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d_model)
            q_t = q[:, t]  # (batch, nhead, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t].unsqueeze(-1)  # (batch, nhead, 1)

            # Get input-dependent permutation
            P_t = self._get_permutation(x_t)  # (batch, nhead, head_dim, head_dim)

            # Apply permutation routing: M = P · M
            # M is (batch, nhead, head_dim, head_dim), P_t is (batch, nhead, head_dim, head_dim)
            # We want to permute the "state dimensions" of M
            # Applying P from the left: P · M permutes rows
            M = torch.einsum('bnij,bnjk->bnik', P_t, M)

            # Read: out = q^T M
            out_t = torch.einsum('bnh,bnhd->bnd', q_t, M)
            outputs.append(out_t)

            # Compute delta: δ = v - M^T k
            retrieved = torch.einsum('bnhd,bnd->bnh', M, k_t)
            delta = v_t - retrieved

            # Update: M = M + β * k ⊗ δ
            update = beta_t.unsqueeze(-1) * torch.einsum('bnh,bnd->bnhd', k_t, delta)

            if mask is not None:
                update = update * mask[:, t].view(batch_size, 1, 1, 1)

            M = M + update

        # Reshape output
        out = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = self.out_norm(out)
        out = self.o_proj(out)
        out = self.dropout(out)

        return out


class CSDeltaNetBlock(nn.Module):
    """CS-DeltaNet block with residual connections."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        allow_neg_eigval: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = CSDeltaNetLayer(
            d_model, nhead, dropout, allow_neg_eigval, gumbel_tau, gumbel_hard
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CSDeltaNet(nn.Module):
    """
    Column-Sparse DeltaNet for group composition tasks.

    Combines:
    1. Input-dependent permutation routing (column-sparse, PD-SSM inspired)
    2. Optional negative eigenvalue extension (NEG-DeltaNet)

    Configurations:
    - CS-DeltaNet: allow_neg_eigval=False, β ∈ (0, 1)
    - CS-NEG-DeltaNet: allow_neg_eigval=True, β ∈ (0, 2)
    """

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        eos_idx: int,
        max_seq_len: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        allow_neg_eigval: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Embeddings
        self.token_embed = nn.Embedding(num_tokens, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # CS-DeltaNet layers
        self.layers = nn.ModuleList([
            CSDeltaNetBlock(d_model, nhead, dropout, allow_neg_eigval, gumbel_tau, gumbel_hard)
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, max_seq_len)
            mask: (batch, max_seq_len)

        Returns:
            logits: (batch, max_seq_len, num_classes)
        """
        batch_size, seq_len = tokens.shape

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embed(tokens) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 12
    num_tokens = 11  # D4: 8 elements + BOS + EOS + PAD
    num_classes = 8   # D4: 8 elements

    print("Testing CS-DeltaNet (column-sparse, β ∈ (0, 1))")
    model_cs = CSDeltaNet(
        num_tokens=num_tokens,
        num_classes=num_classes,
        eos_idx=9,
        max_seq_len=seq_len,
        d_model=32,
        nhead=4,
        num_layers=1,
        allow_neg_eigval=False,
    )

    tokens = torch.randint(0, num_tokens, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    logits = model_cs(tokens, mask)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_cs.parameters()):,}")

    print("\nTesting CS-NEG-DeltaNet (column-sparse, β ∈ (0, 2))")
    model_cs_neg = CSDeltaNet(
        num_tokens=num_tokens,
        num_classes=num_classes,
        eos_idx=9,
        max_seq_len=seq_len,
        d_model=32,
        nhead=4,
        num_layers=1,
        allow_neg_eigval=True,
    )

    logits_neg = model_cs_neg(tokens, mask)
    print(f"Output shape: {logits_neg.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_cs_neg.parameters()):,}")

    # Compare parameter counts
    print("\nParameter comparison:")
    from models.deltanet import GroupDeltaNet
    model_std = GroupDeltaNet(
        num_tokens=num_tokens,
        num_classes=num_classes,
        eos_idx=9,
        max_seq_len=seq_len,
        d_model=32,
        nhead=4,
        num_layers=1,
        allow_neg_eigval=False,
    )
    print(f"Standard DeltaNet: {sum(p.numel() for p in model_std.parameters()):,}")
    print(f"CS-DeltaNet: {sum(p.numel() for p in model_cs.parameters()):,}")
    print(f"Extra params for permutation: {sum(p.numel() for p in model_cs.parameters()) - sum(p.numel() for p in model_std.parameters()):,}")

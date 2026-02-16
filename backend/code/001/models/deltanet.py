"""
Standard DeltaNet Model for Group State Tracking

This is a simplified DeltaNet implementation for the MVE experiment.
It supports the negative eigenvalue extension (allow_neg_eigval=True)
which extends beta from (0,1) to (0,2), enabling NC^1 expressivity.

Key difference from CS-DeltaNet:
- No input-dependent permutation routing
- State matrix updates via standard delta rule only
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
        d_ff = ((d_ff + 63) // 64) * 64  # Round to multiple of 64

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DeltaNetLayer(nn.Module):
    """
    DeltaNet layer with optional negative eigenvalue extension.

    Delta rule update:
        M_t = M_{t-1} + β_t * k_t ⊗ (v_t - M_{t-1}^T k_t)

    When allow_neg_eigval=True:
        β_t ∈ (0, 2) instead of (0, 1)
        This enables negative eigenvalues in the effective state transition.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.allow_neg_eigval = allow_neg_eigval

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Beta (learning rate for delta update)
        self.beta_proj = nn.Linear(d_model, nhead, bias=False)

        # Output normalization
        self.out_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True) + eps)

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

        # Delta rule (sequential)
        outputs = []
        M = torch.zeros(batch_size, self.nhead, self.head_dim, self.head_dim,
                       device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            q_t = q[:, t]  # (batch, nhead, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t].unsqueeze(-1)  # (batch, nhead, 1)

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


class DeltaNetBlock(nn.Module):
    """DeltaNet block with residual connections."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = DeltaNetLayer(d_model, nhead, dropout, allow_neg_eigval)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GroupDeltaNet(nn.Module):
    """
    DeltaNet model for group composition tasks.

    Supports two configurations:
    - Standard DeltaNet: allow_neg_eigval=False, β ∈ (0, 1)
    - NEG-DeltaNet: allow_neg_eigval=True, β ∈ (0, 2)
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

        # DeltaNet layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(d_model, nhead, dropout, allow_neg_eigval)
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

    model = GroupDeltaNet(
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

    logits = model(tokens, mask)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with negative eigenvalue extension
    model_neg = GroupDeltaNet(
        num_tokens=num_tokens,
        num_classes=num_classes,
        eos_idx=9,
        max_seq_len=seq_len,
        d_model=32,
        nhead=4,
        num_layers=1,
        allow_neg_eigval=True,
    )

    logits_neg = model_neg(tokens, mask)
    print(f"NEG-DeltaNet output shape: {logits_neg.shape}")

"""
V:N:M Sparse Linear Layer with S-STE Training

Implements the two-level hierarchical sparsity from proposal 031:
  Level 1 (Column Selection): For each V x M block, retain top-4 columns by importance
  Level 2 (2:4 S-STE):       Within retained columns, apply S-STE soft-thresholding

Mathematical formulation (proposal eq.):
  Step 1 — Column importance: Imp_j = sum_i |W_ij|
  Step 2 — S-STE: (S_soft(a))_i = sign(a_i) * max(|a_i| - |a_(2)|, 0)
  Step 3 — Sparse output: W_tilde = beta * S_soft(W * mask_col)

Sparsity configurations:
  - 2:4 only (50%):  No column pruning, just 2:4 within groups of 4
  - V:2:6 (67%):     Keep 4 of 6 columns per block, then 2:4 within retained
  - V:2:8 (75%):     Keep 4 of 8 columns per block, then 2:4 within retained
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VNMSparseLinear(nn.Module):
    """
    Linear layer with V:N:M structured sparsity trained via S-STE.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        M: Block width for column selection (5->60%, 6->67%, 8->75% sparsity)
            Set M=4 for 2:4-only (no column pruning), M=0 for dense
        V: Block height for column importance computation (default 64, adjusted if needed)
        bias: Whether to include bias
    """

    def __init__(self, in_features: int, out_features: int, M: int = 8, V: int = 64, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.M = M  # Block width (0 = dense, 4 = 2:4 only, 6 = V:2:6, 8 = V:2:8)
        self.V = V

        # Dense weight (always maintained during training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # S-STE frozen scaling factor (computed on first forward pass)
        self.register_buffer('beta', torch.tensor(1.0))
        self.beta_frozen = False

        # Mask flip tracking
        self.register_buffer('prev_mask', torch.zeros(out_features, in_features, dtype=torch.bool))
        self.register_buffer('mask_flip_count', torch.tensor(0.0))
        self.register_buffer('mask_total_count', torch.tensor(0.0))

    @property
    def is_dense(self) -> bool:
        return self.M == 0

    @property
    def is_two_four_only(self) -> bool:
        return self.M == 4

    def _column_mask(self, W: torch.Tensor) -> torch.Tensor:
        """
        Step 1: VNM column selection (proposal Step 1).

        For each group of M columns, compute importance and retain top-4.
        Uses STE (straight-through estimator) for gradient flow.

        Args:
            W: Weight matrix (out_features, in_features)

        Returns:
            col_mask: Binary mask (out_features, in_features), 1 = retained
        """
        out_f, in_f = W.shape

        if self.M <= 4:
            # No column pruning needed (2:4 only or dense)
            return torch.ones_like(W)

        # Pad input dim to be divisible by M
        pad_size = (self.M - in_f % self.M) % self.M
        if pad_size > 0:
            W_padded = F.pad(W, (0, pad_size))
        else:
            W_padded = W

        padded_in_f = W_padded.shape[1]
        n_blocks = padded_in_f // self.M

        # Reshape into blocks of M columns: (out_f, n_blocks, M)
        W_blocks = W_padded.view(out_f, n_blocks, self.M)

        # Column importance: L1 norm across rows (proposal eq.)
        # Imp_j = sum_i |W_ij| for each column within each block
        col_imp = W_blocks.abs().sum(dim=0)  # (n_blocks, M)

        # Top-4 selection per block
        _, top_idx = col_imp.topk(4, dim=-1)  # (n_blocks, 4)
        col_mask_blocks = torch.zeros_like(col_imp)
        col_mask_blocks.scatter_(-1, top_idx, 1.0)

        # Expand mask to full weight shape
        col_mask_padded = col_mask_blocks.unsqueeze(0).expand(out_f, -1, -1)
        col_mask_padded = col_mask_padded.reshape(out_f, padded_in_f)

        # Remove padding
        col_mask = col_mask_padded[:, :in_f]

        return col_mask

    def _sste_two_four(self, W: torch.Tensor) -> torch.Tensor:
        """
        Step 2: S-STE 2:4 soft-thresholding (proposal Step 2).

        Within each group of 4 elements, apply soft-thresholding:
          (S_soft(a))_i = sign(a_i) * max(|a_i| - |a_(2)|, 0)

        This keeps the top 2 elements per group of 4 (2:4 pattern).
        The soft-thresholding provides continuous gradients unlike hard masking.

        Args:
            W: Weight matrix (arbitrary shape, last dim must be divisible by 4)

        Returns:
            W_soft: Soft-thresholded weight (same shape)
        """
        original_shape = W.shape
        total_elements = W.numel()

        # Reshape to groups of 4
        pad_size = (4 - total_elements % 4) % 4
        if pad_size > 0:
            W_flat = torch.cat([W.reshape(-1), torch.zeros(pad_size, device=W.device)])
        else:
            W_flat = W.reshape(-1)

        W_groups = W_flat.view(-1, 4)  # (n_groups, 4)

        # Sort by absolute value
        abs_vals = W_groups.abs()
        sorted_abs, _ = abs_vals.sort(dim=-1)  # ascending

        # Threshold = 2nd largest = sorted_abs[:, 1] (0-indexed ascending, so index 1)
        # In a group of 4, sorted ascending: [smallest, 2nd smallest, 2nd largest, largest]
        # We keep top 2, so threshold is the 2nd smallest (index 1)
        threshold = sorted_abs[:, 1:2]  # (n_groups, 1)

        # Soft-thresholding: sign(a) * max(|a| - threshold, 0)
        W_soft = torch.sign(W_groups) * F.relu(abs_vals - threshold)

        # Reshape back
        W_soft = W_soft.reshape(-1)[:total_elements].view(original_shape)

        return W_soft

    def get_sparse_weight(self) -> torch.Tensor:
        """
        Apply full VNM sparsification pipeline.

        For VNM (M > 4): Two-level sparsity
          Level 1: Column selection — keep 4 of M columns per block
          Level 2: 2:4 S-STE — keep 2 of 4 within retained columns
          Combined: keep 2 of M elements → sparsity = 1 - 2/M

        For 2:4 (M = 4): Single-level
          Just 2:4 S-STE on all elements → 50% sparsity

        Returns:
            W_sparse: Sparsified weight matrix
        """
        W = self.weight

        if self.is_dense:
            return W

        if self.is_two_four_only:
            # 2:4 only — apply S-STE directly on the full weight
            W_soft = self._sste_two_four(W)
        else:
            # VNM two-level sparsity:
            # We need to apply 2:4 only within the 4 retained columns per M-group.
            # Strategy: reshape into M-groups, select top-4 columns, apply 2:4 on those 4,
            # then scatter back.
            out_f, in_f = W.shape

            # Pad input dim to be divisible by M
            pad_size = (self.M - in_f % self.M) % self.M
            if pad_size > 0:
                W_padded = F.pad(W, (0, pad_size))
            else:
                W_padded = W

            padded_in_f = W_padded.shape[1]
            n_blocks = padded_in_f // self.M

            # Reshape into blocks: (out_f, n_blocks, M)
            W_blocks = W_padded.view(out_f, n_blocks, self.M)

            # Step 1: Column importance — L1 norm across rows
            col_imp = W_blocks.abs().sum(dim=0)  # (n_blocks, M)
            _, top_idx = col_imp.topk(4, dim=-1)  # (n_blocks, 4)

            # Gather the 4 retained columns per block: (out_f, n_blocks, 4)
            top_idx_expanded = top_idx.unsqueeze(0).expand(out_f, -1, -1)
            W_retained = torch.gather(W_blocks, dim=2, index=top_idx_expanded)

            # Step 2: Apply 2:4 S-STE on the retained columns
            # W_retained is (out_f, n_blocks, 4) — each group of 4 is one "retained block"
            W_retained_flat = W_retained.reshape(-1, 4)  # (out_f * n_blocks, 4)
            abs_vals = W_retained_flat.abs()
            sorted_abs, _ = abs_vals.sort(dim=-1)  # ascending
            threshold = sorted_abs[:, 1:2]  # 2nd smallest
            W_soft_flat = torch.sign(W_retained_flat) * F.relu(abs_vals - threshold)
            W_soft_retained = W_soft_flat.view(out_f, n_blocks, 4)

            # Scatter back into full M-width blocks (zeros for pruned columns)
            W_soft_blocks = torch.zeros_like(W_blocks)
            W_soft_blocks.scatter_(dim=2, index=top_idx_expanded, src=W_soft_retained)

            # Reshape back to (out_f, padded_in_f) and remove padding
            W_soft = W_soft_blocks.view(out_f, padded_in_f)[:, :in_f]

        # Compute frozen beta on first call (proposal: freeze after iteration 1)
        if not self.beta_frozen and self.training:
            with torch.no_grad():
                numerator = (W * W_soft).sum()
                denominator = (W_soft * W_soft).sum()
                if denominator > 0:
                    self.beta = numerator / denominator
                else:
                    self.beta = torch.tensor(1.0, device=W.device)
            self.beta_frozen = True

        # Apply scaling
        W_sparse = self.beta * W_soft

        # STE: use sparse weight in forward, but gradient flows to dense weight
        # This is automatically handled by PyTorch's autograd since W_soft
        # is computed from W with differentiable operations
        return W_sparse

    def update_mask_stats(self):
        """Track mask flip rate for convergence monitoring."""
        with torch.no_grad():
            W_sparse = self.get_sparse_weight()
            current_mask = (W_sparse.abs() > 0)

            if self.mask_total_count > 0:
                flips = (current_mask != self.prev_mask).sum().float()
                total = current_mask.numel()
                self.mask_flip_count = flips
                self.mask_total_count = torch.tensor(float(total), device=flips.device)

            self.prev_mask = current_mask.clone()

    @property
    def mask_flip_rate(self) -> float:
        """Current mask flip rate (fraction of mask bits that changed)."""
        if self.mask_total_count > 0:
            return (self.mask_flip_count / self.mask_total_count).item()
        return 0.0

    @property
    def actual_sparsity(self) -> float:
        """Measure actual sparsity of the sparse weight."""
        with torch.no_grad():
            W_sparse = self.get_sparse_weight()
            return (W_sparse == 0).float().mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with VNM-sparse weight.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            output: (..., out_features)
        """
        W_sparse = self.get_sparse_weight()
        output = F.linear(x, W_sparse, self.bias)
        return output

    def extra_repr(self) -> str:
        s = f'in_features={self.in_features}, out_features={self.out_features}'
        if self.is_dense:
            s += ', mode=dense'
        elif self.is_two_four_only:
            s += ', mode=2:4'
        else:
            s += f', mode=V:2:{self.M} ({100 * (1 - 4/self.M * 0.5):.0f}% sparse)'
        if self.bias is not None:
            s += ', bias=True'
        return s

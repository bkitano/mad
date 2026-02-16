"""
HSS Linear Attention Layer for MVE 005.

Implements a single linear attention layer where the state matrix S_t
is stored in HSS (Hierarchically Semi-Separable) format.

Key idea from proposal:
  - Standard linear attention maintains S_t = S_{t-1} + phi(k_t) phi(v_t)^T  (dense d x d)
  - HSS-LinAttn stores S_t in HSS form: O(r*d*log(d)) instead of O(d^2)
  - Queries via HSS matvec: o_t = S_t @ phi(q_t)  in O(r*d*log(d))

This implementation uses a FLAT tensor representation of the HSS tree
(no recursive Python objects) for much better performance.

Architecture matches proposal Section "Minimum Viable Experiment":
  - d=64, r=8, ~10K params
  - Feature map: phi(x) = elu(x) + 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Feature map phi(x) = elu(x) + 1. Ensures non-negative values."""
    return F.elu(x) + 1.0


class FlatHSSState:
    """
    Flat (non-recursive) HSS state matrix representation.

    For a d x d matrix with L = log2(d) levels, we store:
      - leaf_blocks: (batch, num_leaves, leaf_size, leaf_size) - diagonal leaf blocks
      - W: list of (batch, num_nodes_at_level, r, r) - coupling matrices at each level
      - U_L, V_L, U_R, V_R: basis matrices (fixed, initialized once)

    The tree has L levels (0 = leaf, L-1 = root):
      Level 0: d/leaf_size leaf blocks (dense)
      Level l: d/(leaf_size * 2^l) internal nodes with coupling matrices

    For d=64, leaf_size=4:
      Level 0: 16 leaf blocks (4x4 each)
      Level 1: 8 nodes (coupling W: r x r)
      Level 2: 4 nodes
      Level 3: 2 nodes
      Level 4: 1 node (root)
    """

    def __init__(self, d: int, r: int, batch_size: int, device: torch.device, dtype: torch.dtype,
                 leaf_size: int = 4):
        assert d & (d - 1) == 0, f"d must be power of 2, got {d}"
        assert d >= leaf_size

        self.d = d
        self.r = r
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.leaf_size = leaf_size

        self.num_leaves = d // leaf_size
        self.num_levels = int(math.log2(d // leaf_size))  # Number of internal levels

        # Leaf blocks: (batch, num_leaves, leaf_size, leaf_size)
        self.leaf_blocks = torch.zeros(batch_size, self.num_leaves, leaf_size, leaf_size,
                                       device=device, dtype=dtype)

        # Coupling matrices at each level: W[l] has shape (batch, num_nodes, r_l, r_l)
        # where num_nodes = num_leaves / 2^(l+1) and r_l = min(r, block_size_at_level)
        self.W = []
        self.ranks = []  # effective rank at each level
        self.block_sizes = []  # block size at each level

        for l in range(self.num_levels):
            num_nodes = self.num_leaves // (2 ** (l + 1))
            block_size = leaf_size * (2 ** l)  # half-size of the block at this level
            rl = min(r, block_size)
            self.ranks.append(rl)
            self.block_sizes.append(block_size)
            self.W.append(torch.zeros(batch_size, max(num_nodes, 1), rl, rl,
                                      device=device, dtype=dtype))

        # Fixed basis matrices: U_L, V_R at each level
        # These define how to project vectors into the low-rank subspace
        # U_L[l]: (block_size, r_l) - basis for left half
        # V_R[l]: (block_size, r_l) - basis for right half
        self.U_L = []
        self.V_R = []
        self.U_R = []
        self.V_L = []

        for l in range(self.num_levels):
            block_size = self.block_sizes[l]
            rl = self.ranks[l]
            # Initialize with orthonormal bases
            Q1, _ = torch.linalg.qr(torch.randn(block_size, rl, device=device, dtype=dtype))
            Q2, _ = torch.linalg.qr(torch.randn(block_size, rl, device=device, dtype=dtype))
            Q3, _ = torch.linalg.qr(torch.randn(block_size, rl, device=device, dtype=dtype))
            Q4, _ = torch.linalg.qr(torch.randn(block_size, rl, device=device, dtype=dtype))
            self.U_L.append(Q1)
            self.V_R.append(Q2)
            self.U_R.append(Q3)
            self.V_L.append(Q4)

    def rank1_update(self, u: torch.Tensor, v: torch.Tensor):
        """
        Update HSS state with rank-1 outer product: S += u @ v^T

        From proposal: S_t = S_{t-1} + phi(k_t) phi(v_t)^T

        Args:
            u: (batch, d) - e.g., phi(k_t)
            v: (batch, d) - e.g., phi(v_t)
        """
        B = self.batch_size
        ls = self.leaf_size

        # 1. Update leaf blocks (exact)
        # Reshape u, v into leaf chunks
        u_leaves = u.view(B, self.num_leaves, ls)  # (B, num_leaves, ls)
        v_leaves = v.view(B, self.num_leaves, ls)  # (B, num_leaves, ls)

        # leaf_blocks += u_leaf @ v_leaf^T for each leaf
        self.leaf_blocks = self.leaf_blocks + torch.einsum('bni,bnj->bnij', u_leaves, v_leaves)

        # 2. Update coupling matrices at each level
        for l in range(self.num_levels):
            block_size = self.block_sizes[l]
            rl = self.ranks[l]
            stride = block_size * 2  # Full block size at this level
            num_nodes = self.num_leaves // (2 ** (l + 1))

            if num_nodes == 0:
                break

            # Extract left and right halves of u, v at this level
            # Each node at level l covers a range of size `stride`
            # Left half: [node_start, node_start + block_size)
            # Right half: [node_start + block_size, node_start + stride)

            u_reshaped = u.view(B, num_nodes, stride)  # (B, num_nodes, stride)
            v_reshaped = v.view(B, num_nodes, stride)

            u_left = u_reshaped[:, :, :block_size]   # (B, num_nodes, block_size)
            u_right = u_reshaped[:, :, block_size:]   # (B, num_nodes, block_size)
            v_left = v_reshaped[:, :, :block_size]
            v_right = v_reshaped[:, :, block_size:]

            # Project onto basis: u_proj_L = U_L^T @ u_left, v_proj_R = V_R^T @ v_right
            # U_L[l]: (block_size, rl)
            u_proj_L = torch.einsum('dr,bnd->bnr', self.U_L[l], u_left)   # (B, num_nodes, rl)
            v_proj_R = torch.einsum('dr,bnd->bnr', self.V_R[l], v_right)  # (B, num_nodes, rl)

            # Update coupling: W += u_proj_L @ v_proj_R^T
            self.W[l] = self.W[l] + torch.einsum('bni,bnj->bnij', u_proj_L, v_proj_R)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        HSS matrix-vector multiply: y = S @ x

        Args:
            x: (batch, d)
        Returns:
            y: (batch, d)
        """
        B = self.batch_size
        ls = self.leaf_size

        # 1. Leaf block contributions
        x_leaves = x.view(B, self.num_leaves, ls)  # (B, num_leaves, ls)
        y = torch.einsum('bnij,bnj->bni', self.leaf_blocks, x_leaves)  # (B, num_leaves, ls)
        y = y.view(B, self.d)  # (B, d)

        # 2. Off-diagonal contributions from each level
        for l in range(self.num_levels):
            block_size = self.block_sizes[l]
            rl = self.ranks[l]
            stride = block_size * 2
            num_nodes = self.num_leaves // (2 ** (l + 1))

            if num_nodes == 0:
                break

            x_reshaped = x.view(B, num_nodes, stride)
            x_left = x_reshaped[:, :, :block_size]   # (B, num_nodes, block_size)
            x_right = x_reshaped[:, :, block_size:]   # (B, num_nodes, block_size)

            # Top-right block contribution: U_L @ W @ V_R^T @ x_right
            # Step 1: V_R^T @ x_right -> (B, num_nodes, rl)
            vr_x = torch.einsum('dr,bnd->bnr', self.V_R[l], x_right)
            # Step 2: W @ (V_R^T @ x_right) -> (B, num_nodes, rl)
            w_vr_x = torch.einsum('bnij,bnj->bni', self.W[l], vr_x)
            # Step 3: U_L @ result -> (B, num_nodes, block_size)
            y_top = torch.einsum('dr,bnr->bnd', self.U_L[l], w_vr_x)

            # Bottom-left block contribution: U_R @ W^T @ V_L^T @ x_left
            vl_x = torch.einsum('dr,bnd->bnr', self.V_L[l], x_left)
            wt_vl_x = torch.einsum('bnji,bnj->bni', self.W[l], vl_x)  # W^T
            y_bot = torch.einsum('dr,bnr->bnd', self.U_R[l], wt_vl_x)

            # Add contributions back
            y_contrib = torch.cat([y_top, y_bot], dim=-1)  # (B, num_nodes, stride)
            y = y + y_contrib.view(B, self.d)

        return y

    def to_dense(self) -> torch.Tensor:
        """Convert to full dense matrix for analysis. Returns (batch, d, d)."""
        B = self.batch_size
        result = torch.zeros(B, self.d, self.d, device=self.device, dtype=self.dtype)

        # Add leaf blocks
        ls = self.leaf_size
        for i in range(self.num_leaves):
            start = i * ls
            end = start + ls
            result[:, start:end, start:end] = self.leaf_blocks[:, i]

        # Add off-diagonal contributions
        for l in range(self.num_levels):
            block_size = self.block_sizes[l]
            stride = block_size * 2
            num_nodes = self.num_leaves // (2 ** (l + 1))

            for n in range(num_nodes):
                node_start = n * stride
                left_start = node_start
                left_end = node_start + block_size
                right_start = node_start + block_size
                right_end = node_start + stride

                # Top-right: U_L @ W @ V_R^T
                W_n = self.W[l][:, n]  # (B, rl, rl)
                off_diag_tr = torch.einsum('dr,brj,jd->brd',
                    self.U_L[l], W_n, self.V_R[l].T)  # (B, block_size, block_size)

                # Correct: should be U_L @ W @ V_R^T with proper dims
                # U_L: (block_size, rl), W: (B, rl, rl), V_R: (block_size, rl)
                off_diag_tr = self.U_L[l] @ W_n @ self.V_R[l].T  # (B, block_size, block_size)

                result[:, left_start:left_end, right_start:right_end] = off_diag_tr

                # Bottom-left: U_R @ W^T @ V_L^T
                off_diag_bl = self.U_R[l] @ W_n.transpose(-1, -2) @ self.V_L[l].T
                result[:, right_start:right_end, left_start:left_end] = off_diag_bl

        return result

    def memory_usage(self) -> int:
        """Count total number of stored floats (per batch element)."""
        count = self.num_leaves * self.leaf_size * self.leaf_size  # leaf blocks

        for l in range(self.num_levels):
            num_nodes = self.num_leaves // (2 ** (l + 1))
            rl = self.ranks[l]
            count += num_nodes * rl * rl  # W matrices

        # Basis matrices (shared across batch, but counted per element for comparison)
        for l in range(self.num_levels):
            block_size = self.block_sizes[l]
            rl = self.ranks[l]
            count += 4 * block_size * rl  # U_L, V_R, U_R, V_L

        return count

    def reset(self, mask: Optional[torch.Tensor] = None):
        """Reset state to zero. If mask provided, only reset masked elements."""
        if mask is None:
            self.leaf_blocks.zero_()
            for w in self.W:
                w.zero_()
        else:
            # mask: (batch,) bool tensor
            self.leaf_blocks[mask] = 0.0
            for w in self.W:
                w[mask] = 0.0


class DenseLinearAttention(nn.Module):
    """
    Standard Dense Linear Attention (baseline).

    Maintains S_t = S_{t-1} + phi(k_t) phi(v_t)^T as dense d x d matrix.
    """

    def __init__(self, d_model: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.W_o = nn.Linear(d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, segment_flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = elu_feature_map(self.W_q(x))
        k = elu_feature_map(self.W_k(x))
        v = self.W_v(x)

        S = torch.zeros(batch_size, self.d_head, self.d_head, device=x.device, dtype=x.dtype)
        z = torch.zeros(batch_size, self.d_head, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            if segment_flags is not None:
                reset_mask = segment_flags[:, t].bool()
                if reset_mask.any():
                    S[reset_mask] = 0.0
                    z[reset_mask] = 0.0

            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]

            S = S + torch.einsum('bi,bj->bij', k_t, v_t)
            z = z + k_t

            o_t = torch.einsum('bij,bj->bi', S, q_t)
            denom = torch.einsum('bi,bi->b', z, q_t).unsqueeze(-1).clamp(min=1e-6)
            o_t = o_t / denom
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)
        output = self.W_o(self.dropout(output))
        return output

    def state_memory(self) -> int:
        return self.d_head * self.d_head


class HSSLinearAttention(nn.Module):
    """
    HSS Linear Attention Layer using flat tensor representation.
    """

    def __init__(self, d_model: int, d_head: int, hss_rank: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.hss_rank = hss_rank

        assert d_head & (d_head - 1) == 0, f"d_head must be power of 2, got {d_head}"

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.W_o = nn.Linear(d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, segment_flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = elu_feature_map(self.W_q(x))
        k = elu_feature_map(self.W_k(x))
        v = self.W_v(x)

        hss_state = FlatHSSState(
            d=self.d_head, r=self.hss_rank, batch_size=batch_size,
            device=x.device, dtype=x.dtype
        )
        z = torch.zeros(batch_size, self.d_head, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]

            if segment_flags is not None:
                reset_mask = segment_flags[:, t].bool()
                if reset_mask.any():
                    z[reset_mask] = 0.0
                    hss_state.reset(reset_mask)

            hss_state.rank1_update(k_t, v_t)
            z = z + k_t

            o_t = hss_state.matvec(q_t)
            denom = torch.einsum('bi,bi->b', z, q_t).unsqueeze(-1).clamp(min=1e-6)
            o_t = o_t / denom
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)
        output = self.W_o(self.dropout(output))
        return output

    def state_memory(self) -> int:
        state = FlatHSSState(self.d_head, self.hss_rank, 1, torch.device('cpu'), torch.float32)
        return state.memory_usage()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class HSSLinearAttentionModel(nn.Module):
    """
    Full model wrapping HSS or Dense Linear Attention.

    MVE config from proposal:
      - d=64, r=8, ~10K params
      - Single layer
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_head: int = 64,
        max_seq_len: int = 64,
        hss_rank: int = 8,
        use_hss: bool = True,
        num_output_classes: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.use_hss = use_hss
        self.vocab_size = vocab_size
        self.num_output_classes = num_output_classes or vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        if use_hss:
            self.attention = HSSLinearAttention(d_model, d_head, hss_rank, dropout)
        else:
            self.attention = DenseLinearAttention(d_model, d_head, dropout)

        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.num_output_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, segment_flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = x + self.attention(x, segment_flags=segment_flags)
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_state_memory_ratio(self) -> float:
        dense_mem = self.d_head * self.d_head
        if self.use_hss:
            hss_mem = self.attention.state_memory()
        else:
            hss_mem = dense_mem
        return hss_mem / dense_mem

"""
Attention variants for the cos-LogLinear MVE.

Implements 4 attention mechanisms:
1. VanillaLinearAttention - ELU+1 kernel, single state
2. CosFormerAttention - Cosine reweighted, single state (Ptolemy decomposition)
3. LogLinearAttention - ELU+1 kernel, hierarchical Fenwick tree states
4. CosLogLinearAttention - Cosine reweighted + hierarchical states (proposed)

All share the same model wrapper (MQARModel) for fair comparison.

Implementation note: The Fenwick tree variants use a precomputed causal mask
approach to avoid in-place operations during the forward pass, making them
fully differentiable with autograd.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import lru_cache


# =============================================================================
# Feature maps (kernel functions)
# =============================================================================


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map: ensures non-negativity for linear attention."""
    return F.elu(x) + 1.0


# =============================================================================
# Fenwick tree utilities for log-linear attention
# =============================================================================


def lowest_set_bit(n: int) -> int:
    """Return the lowest set bit of n."""
    return n & (-n)


def get_fenwick_level(idx: int) -> int:
    """Return the Fenwick tree level for a 1-indexed position."""
    return int(math.log2(lowest_set_bit(idx))) if idx > 0 else 0


@lru_cache(maxsize=32)
def build_fenwick_masks(T: int, num_levels: int):
    """
    Precompute Fenwick tree structure as a set of level-specific causal masks.

    For each level l, builds a (T, T) binary mask where mask[i, j] = 1 iff
    position j contributes to position i's output via a level-l Fenwick node.

    This replaces the sequential in-place accumulation with a differentiable
    masked sum.

    Returns: level_masks - list of (T, T) binary masks, one per level.
    """
    # For each query position t (0-indexed), find which source positions j
    # contribute at each level via the Fenwick tree prefix query.
    #
    # Fenwick tree structure:
    # - Position j (1-indexed) is assigned to level log2(lowbit(j))
    # - Prefix query for t1 = t+1 collects from nodes by walking t1 -= lowbit(t1)
    # - Each node at index idx covers the positions that update it via
    #   the upward propagation: the set of positions whose update path includes idx.

    # Approach: For each (query_pos t, source_pos j) pair, determine which
    # Fenwick node(s) contain j and are included in the prefix query for t.
    # Then assign a level based on the Fenwick node's level.

    # Simpler approach: directly compute the "effective level" mask.
    # For each position j (1-indexed), it updates Fenwick nodes at:
    #   j, j + lowbit(j), j + lowbit(j+lowbit(j)), etc.
    # For each query position t (1-indexed), the prefix query visits:
    #   t, t - lowbit(t), etc.
    # Position j contributes to query t at level l if there exists a
    # Fenwick node idx such that:
    #   1. j's update propagation reaches idx (j updates idx)
    #   2. t's prefix query visits idx
    #   3. The Fenwick level of idx is l

    # Pre-build: for each position j, which nodes does it update?
    j_to_nodes = {}  # j -> list of (node_idx, level)
    for j in range(1, T + 1):
        nodes = []
        idx = j
        while idx <= T:
            level = get_fenwick_level(idx)
            nodes.append((idx, level))
            idx += lowest_set_bit(idx)
        j_to_nodes[j] = nodes

    # For each query position t, which nodes does prefix query visit?
    t_to_query_nodes = {}
    for t in range(1, T + 1):
        nodes = set()
        idx = t
        while idx > 0:
            nodes.add(idx)
            idx -= lowest_set_bit(idx)
        t_to_query_nodes[t] = nodes

    # Build level masks: level_masks[l][t-1, j-1] = 1 iff position j
    # contributes to query t via a level-l Fenwick node
    level_masks = [torch.zeros(T, T) for _ in range(num_levels)]

    for t in range(1, T + 1):
        query_nodes = t_to_query_nodes[t]
        for j in range(1, t + 1):  # Causal: j <= t
            for (node_idx, node_level) in j_to_nodes[j]:
                if node_idx in query_nodes:
                    # j contributes to t at this level
                    lvl = min(node_level, num_levels - 1)
                    level_masks[lvl][t - 1, j - 1] = 1.0
                    break  # Each j contributes via exactly one node per query

    return level_masks


# =============================================================================
# 1. Vanilla Linear Attention
# =============================================================================


class VanillaLinearAttention(nn.Module):
    """
    Standard linear attention with ELU+1 kernel.

    O_i = phi(Q_i)^T @ S_i / (phi(Q_i) . z_i)
    S_i = sum_{j<=i} phi(K_j) V_j^T  (d x d state matrix)
    """

    def __init__(self, d_model: int, nhead: int, head_dim: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, nhead * head_dim)
        self.k_proj = nn.Linear(d_model, nhead * head_dim)
        self.v_proj = nn.Linear(d_model, nhead * head_dim)
        self.out_proj = nn.Linear(nhead * head_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        Q_phi = elu_plus_one(Q)  # (B, H, T, D)
        K_phi = elu_plus_one(K)

        # Causal linear attention via cumulative state (vectorized)
        KV = torch.einsum('bhti,bhtj->bhtij', K_phi, V)  # (B, H, T, D, D)
        S = torch.cumsum(KV, dim=2)  # cumulative state
        z = torch.cumsum(K_phi, dim=2)  # normalization

        num = torch.einsum('bhti,bhtij->bhtj', Q_phi, S)
        den = torch.einsum('bhti,bhti->bht', Q_phi, z).unsqueeze(-1).clamp(min=1e-6)

        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# =============================================================================
# 2. CosFormer Attention (Ptolemy decomposition)
# =============================================================================


class CosFormerAttention(nn.Module):
    """
    cosFormer: Rethinking Softmax in Attention (Qin et al., 2022).

    s(Q'_i, K'_j) = Q'^T_i K'_j * cos(pi/2 * (i-j)/M)

    Ptolemy decomposition: cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
    O_i = (Q_cos @ S_cos + Q_sin @ S_sin) / (Q_cos . z_cos + Q_sin . z_sin)
    """

    def __init__(self, d_model: int, nhead: int, head_dim: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, nhead * head_dim)
        self.k_proj = nn.Linear(d_model, nhead * head_dim)
        self.v_proj = nn.Linear(d_model, nhead * head_dim)
        self.out_proj = nn.Linear(nhead * head_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        Q_prime = F.relu(Q)
        K_prime = F.relu(K)

        # Position-dependent cos/sin modulation
        M = T
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        angles = (math.pi / (2 * M)) * positions
        cos_pos = torch.cos(angles).view(1, 1, T, 1)
        sin_pos = torch.sin(angles).view(1, 1, T, 1)

        Q_cos = Q_prime * cos_pos
        Q_sin = Q_prime * sin_pos
        K_cos = K_prime * cos_pos
        K_sin = K_prime * sin_pos

        # Cumulative states (vectorized, no in-place ops)
        KV_cos = torch.einsum('bhti,bhtj->bhtij', K_cos, V)
        KV_sin = torch.einsum('bhti,bhtj->bhtij', K_sin, V)
        S_cos = torch.cumsum(KV_cos, dim=2)
        S_sin = torch.cumsum(KV_sin, dim=2)

        z_cos = torch.cumsum(K_cos, dim=2)
        z_sin = torch.cumsum(K_sin, dim=2)

        num = (torch.einsum('bhti,bhtij->bhtj', Q_cos, S_cos) +
               torch.einsum('bhti,bhtij->bhtj', Q_sin, S_sin))

        den = (torch.einsum('bhti,bhti->bht', Q_cos, z_cos) +
               torch.einsum('bhti,bhti->bht', Q_sin, z_sin))
        den = den.unsqueeze(-1).clamp(min=1e-6)

        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# =============================================================================
# 3. Log-Linear Attention (ELU+1 kernel + Fenwick hierarchy)
# =============================================================================


class LogLinearAttention(nn.Module):
    """
    Log-linear attention with hierarchical Fenwick tree states.

    Uses precomputed level masks instead of sequential in-place updates
    to maintain autograd compatibility.

    At each query position t, the output combines contributions from
    O(log T) Fenwick tree nodes, each at a different temporal resolution.
    Learned per-level weights control the mixing.
    """

    def __init__(self, d_model: int, nhead: int, head_dim: int, max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.num_levels = int(math.ceil(math.log2(max_seq_len + 1))) + 1

        self.q_proj = nn.Linear(d_model, nhead * head_dim)
        self.k_proj = nn.Linear(d_model, nhead * head_dim)
        self.v_proj = nn.Linear(d_model, nhead * head_dim)
        self.out_proj = nn.Linear(nhead * head_dim, d_model)

        # Per-level weight predictor
        self.level_weight_proj = nn.Linear(d_model, nhead * self.num_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        Q_phi = elu_plus_one(Q)  # (B, H, T, D)
        K_phi = elu_plus_one(K)

        # Level weights (softmax-normalized)
        level_logits = self.level_weight_proj(x).view(B, T, self.nhead, self.num_levels)
        level_logits = level_logits.permute(0, 2, 1, 3)  # (B, H, T, L)
        level_weights = F.softmax(level_logits, dim=-1)

        # Get precomputed Fenwick level masks
        level_masks = build_fenwick_masks(T, self.num_levels)

        # For each level, compute the masked attention:
        # At level l: output_t = w_l * Q_t @ (sum_{j: mask_l[t,j]=1} K_j V_j^T)
        #           / w_l * Q_t . (sum_{j: mask_l[t,j]=1} K_j)

        # Approach: For each level l, use the level mask as a causal attention mask.
        # attn_l[t,j] = mask_l[t,j] (binary)
        # num_l_t = sum_j mask_l[t,j] * (Q_t . K_j) * V_j  (linear attention form)
        # But we want to keep the linear attention form (via state matrices).
        #
        # Actually, the most efficient way is:
        # For each level, compute QK attention scores masked by level mask, then apply to V.
        # This is O(T^2) per level, but since T=128 and L~8, total is ~8*128^2 = 131K ops.
        # Perfectly fine for MVE.

        num_total = torch.zeros(B, self.nhead, T, self.head_dim,
                              device=x.device, dtype=x.dtype)
        den_total = torch.zeros(B, self.nhead, T, 1,
                              device=x.device, dtype=x.dtype)

        for l in range(self.num_levels):
            mask_l = level_masks[l].to(x.device)  # (T, T)

            # Linear attention scores: Q_phi @ K_phi^T, masked by level mask
            # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
            attn_scores = torch.einsum('bhid,bhjd->bhij', Q_phi, K_phi)
            attn_scores = attn_scores * mask_l.unsqueeze(0).unsqueeze(0)  # Apply mask

            # Weighted output
            # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
            level_out = torch.einsum('bhij,bhjd->bhid', attn_scores, V)

            # Level weight
            w_l = level_weights[:, :, :, l].unsqueeze(-1)  # (B, H, T, 1)

            num_total = num_total + w_l * level_out

            # Normalization
            level_den = attn_scores.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
            den_total = den_total + w_l * level_den

        den_total = den_total.clamp(min=1e-6)
        out = num_total / den_total

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# =============================================================================
# 4. cos-LogLinear Attention (PROPOSED - cosine + Fenwick hierarchy)
# =============================================================================


class CosLogLinearAttention(nn.Module):
    """
    Cosine-Reweighted Log-Linear Attention (proposed).

    Combines cosFormer's Ptolemy decomposition with log-linear's Fenwick tree:

    O_i = sum_l w_l * [Q_cos @ S_cos^l + Q_sin @ S_sin^l]
          / sum_l w_l * [Q_cos . z_cos^l + Q_sin . z_sin^l]

    Uses precomputed Fenwick level masks for autograd compatibility.
    """

    def __init__(self, d_model: int, nhead: int, head_dim: int, max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.num_levels = int(math.ceil(math.log2(max_seq_len + 1))) + 1

        self.q_proj = nn.Linear(d_model, nhead * head_dim)
        self.k_proj = nn.Linear(d_model, nhead * head_dim)
        self.v_proj = nn.Linear(d_model, nhead * head_dim)
        self.out_proj = nn.Linear(nhead * head_dim, d_model)

        self.level_weight_proj = nn.Linear(d_model, nhead * self.num_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # cosFormer: ReLU + positional cos/sin modulation
        Q_prime = F.relu(Q)
        K_prime = F.relu(K)

        M = T
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        angles = (math.pi / (2 * M)) * positions
        cos_pos = torch.cos(angles).view(1, 1, T, 1)
        sin_pos = torch.sin(angles).view(1, 1, T, 1)

        Q_cos = Q_prime * cos_pos
        Q_sin = Q_prime * sin_pos
        K_cos = K_prime * cos_pos
        K_sin = K_prime * sin_pos

        # Level weights
        level_logits = self.level_weight_proj(x).view(B, T, self.nhead, self.num_levels)
        level_logits = level_logits.permute(0, 2, 1, 3)
        level_weights = F.softmax(level_logits, dim=-1)

        # Get precomputed Fenwick level masks
        level_masks = build_fenwick_masks(T, self.num_levels)

        num_total = torch.zeros(B, self.nhead, T, self.head_dim,
                              device=x.device, dtype=x.dtype)
        den_total = torch.zeros(B, self.nhead, T, 1,
                              device=x.device, dtype=x.dtype)

        for l in range(self.num_levels):
            mask_l = level_masks[l].to(x.device)  # (T, T)
            mask_l_expanded = mask_l.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

            # Cos stream attention: Q_cos @ K_cos^T, masked
            attn_cos = torch.einsum('bhid,bhjd->bhij', Q_cos, K_cos)
            attn_cos = attn_cos * mask_l_expanded

            # Sin stream attention: Q_sin @ K_sin^T, masked
            attn_sin = torch.einsum('bhid,bhjd->bhij', Q_sin, K_sin)
            attn_sin = attn_sin * mask_l_expanded

            # Combined attention (cosFormer Ptolemy identity)
            attn_combined = attn_cos + attn_sin  # (B, H, T, T)

            # Output: attention-weighted values
            level_out = torch.einsum('bhij,bhjd->bhid', attn_combined, V)

            # Level weight
            w_l = level_weights[:, :, :, l].unsqueeze(-1)  # (B, H, T, 1)
            num_total = num_total + w_l * level_out

            # Denominator
            level_den = attn_combined.sum(dim=-1, keepdim=True)
            den_total = den_total + w_l * level_den

        den_total = den_total.clamp(min=1e-6)
        out = num_total / den_total

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# =============================================================================
# Model wrapper for MQAR task
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AttentionBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, attn_module: nn.Module, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = attn_module
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MQARModel(nn.Module):
    """
    Full model for MQAR task.

    Embedding -> N x AttentionBlock -> RMSNorm -> Linear -> Logits
    """

    ATTENTION_TYPES = {
        'vanilla_linear': VanillaLinearAttention,
        'cosformer': CosFormerAttention,
        'log_linear': LogLinearAttention,
        'cos_log_linear': CosLogLinearAttention,
    }

    def __init__(
        self,
        attention_type: str,
        vocab_size: int,
        d_model: int = 32,
        nhead: int = 2,
        head_dim: int = 16,
        num_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention_type = attention_type

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        attn_cls = self.ATTENTION_TYPES[attention_type]
        blocks = []
        for _ in range(num_layers):
            attn = attn_cls(d_model=d_model, nhead=nhead, head_dim=head_dim,
                          max_seq_len=max_seq_len)
            blocks.append(AttentionBlock(attn, d_model, dropout))

        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.embedding(x) + self.pos_embedding(positions)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        return self.output_head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

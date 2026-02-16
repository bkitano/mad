"""
GLA Intra-Chunk Computation: Baseline and FlashMask Tile-Skip Variants.

This implements the intra-chunk attention computation for Gated Linear Attention (GLA):
    P_[n] = Q_[n] @ K_[n]^T * D_[n]      (attention scores with causal gate mask)
    O_[n]^intra = P_[n] @ V_[n]           (intra-chunk output)

where D_[n] is the causal gate mask with D_ij = prod(alpha_l, l=j+1..i) for i>=j, 0 otherwise.

For document-packed sequences, alpha_t = 0 at document boundaries, making D block-diagonal.

We implement two Triton kernels:
1. Baseline: standard causal skip (j > i => skip)
2. FlashMask tile-skip: additionally skips tiles where ALL keys are in a different document

Both kernels use precomputed log-cumulative-sum of alpha for efficient gate mask computation.

Reference: Proposal 056-flashmask-tile-skip-chunkwise-linear-rnn.md
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# PyTorch Reference Implementation (for correctness checking)
# ============================================================================

def gla_intra_chunk_reference(
    Q: torch.Tensor,      # (B, H, T, dk)
    K: torch.Tensor,      # (B, H, T, dk)
    V: torch.Tensor,      # (B, H, T, dv)
    alpha: torch.Tensor,  # (B, H, T) - per-position gates
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of GLA intra-chunk computation.
    Processes each chunk independently. No tile-skipping.

    Returns: O (B, H, T, dv) - the intra-chunk output
    """
    B, H, T, dk = Q.shape
    dv = V.shape[-1]
    num_chunks = T // chunk_size
    O = torch.zeros(B, H, T, dv, device=Q.device, dtype=Q.dtype)

    for n in range(num_chunks):
        s = n * chunk_size
        e = (n + 1) * chunk_size

        Qn = Q[:, :, s:e, :]   # (B, H, C, dk)
        Kn = K[:, :, s:e, :]   # (B, H, C, dk)
        Vn = V[:, :, s:e, :]   # (B, H, C, dv)
        alpha_n = alpha[:, :, s:e]  # (B, H, C)

        C = chunk_size

        # Build causal gate mask D using log-cumsum
        # D_ij = prod(alpha_l, l=j+1..i) for i>=j, 0 for i<j
        # = exp(sum(log(alpha_l), l=j+1..i))
        # = exp(log_cumsum[i] - log_cumsum[j])
        log_alpha = torch.log(alpha_n.clamp(min=1e-10))  # (B, H, C)
        log_cumsum = torch.cumsum(log_alpha, dim=-1)  # (B, H, C)

        # D_ij = exp(log_cumsum[i] - log_cumsum[j])
        D = torch.exp(log_cumsum[:, :, :, None] - log_cumsum[:, :, None, :])  # (B,H,C,C)

        # Causal mask
        causal = torch.tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))
        D = D * causal

        # Handle NaN from log(0) at boundaries
        D = torch.nan_to_num(D, nan=0.0)

        # P = Q @ K^T * D, O = P @ V
        P = torch.matmul(Qn, Kn.transpose(-2, -1)) * D
        O[:, :, s:e, :] = torch.matmul(P, Vn)

    return O


# ============================================================================
# Precomputation utilities
# ============================================================================

def precompute_log_cumsum(alpha: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Precompute per-chunk log cumulative sum of alpha.

    Args:
        alpha: (B, H, T) per-position gates
        chunk_size: size of each chunk

    Returns:
        log_cumsum: (B, H, T) per-chunk log cumulative sum
    """
    B, H, T = alpha.shape
    num_chunks = T // chunk_size
    alpha_chunks = alpha.reshape(B, H, num_chunks, chunk_size)
    log_alpha = torch.log(alpha_chunks.clamp(min=1e-10))
    log_cumsum = torch.cumsum(log_alpha, dim=-1)
    return log_cumsum.reshape(B, H, T).contiguous()


def compute_column_sparse_mask(
    alpha: torch.Tensor,
    chunk_size: int = 128,
    sub_chunk_size: int = 16,
) -> dict:
    """
    Compute FlashMask's column-sparse mask vectors from per-position gates alpha.

    For each key position j, LTE_j = the next position where alpha=0 after j
    (i.e., the end of j's document within this chunk).
    If no boundary exists after j in the chunk, LTE_j = chunk_end.

    Then computes per-tile min/max of LTE for tile classification.
    """
    B, H, T = alpha.shape
    device = alpha.device
    num_chunks = T // chunk_size
    Ns = chunk_size // sub_chunk_size

    # For each position, find the next boundary (alpha~0) after it within the chunk
    LTE = torch.full((B, H, T), 0, dtype=torch.int32, device=device)

    alpha_chunks = alpha.reshape(B, H, num_chunks, chunk_size)
    is_boundary = (alpha_chunks.abs() < 1e-6)  # (B, H, num_chunks, C)

    for n in range(num_chunks):
        chunk_start = n * chunk_size
        chunk_end = (n + 1) * chunk_size

        # Scan from right to find next boundary for each position
        # LTE[j] = position of next boundary after j, or chunk_end if none
        for j in range(chunk_size - 1, -1, -1):
            if j == chunk_size - 1:
                LTE[:, :, chunk_start + j] = torch.where(
                    is_boundary[:, :, n, j],
                    chunk_start + j,
                    chunk_end,
                ).int()
            else:
                LTE[:, :, chunk_start + j] = torch.where(
                    is_boundary[:, :, n, j],
                    chunk_start + j,
                    LTE[:, :, chunk_start + j + 1],
                ).int()

    # Per-tile min/max of LTE
    LTE_tile_min = torch.zeros(B, H, num_chunks, Ns, dtype=torch.int32, device=device)
    LTE_tile_max = torch.zeros(B, H, num_chunks, Ns, dtype=torch.int32, device=device)

    for n in range(num_chunks):
        chunk_start = n * chunk_size
        for tj in range(Ns):
            col_start = chunk_start + tj * sub_chunk_size
            col_end = chunk_start + (tj + 1) * sub_chunk_size
            tile_lte = LTE[:, :, col_start:col_end]
            LTE_tile_min[:, :, n, tj] = tile_lte.min(dim=-1).values
            LTE_tile_max[:, :, n, tj] = tile_lte.max(dim=-1).values

    return {
        'LTE': LTE,
        'LTE_tile_min': LTE_tile_min.contiguous(),
        'LTE_tile_max': LTE_tile_max.contiguous(),
    }


# ============================================================================
# Triton Kernel: Unified GLA Intra-Chunk (with optional tile-skip)
# ============================================================================

@triton.jit
def _gla_intra_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    log_cumsum_ptr,
    LTE_tile_max_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    dk: tl.constexpr, dv: tl.constexpr,
    chunk_size: tl.constexpr, sub_chunk_size: tl.constexpr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lcb, stride_lch, stride_lct,
    stride_ltb, stride_lth, stride_ltc, stride_lts,
    SKIP_TILES: tl.constexpr,
):
    """
    GLA intra-chunk kernel.

    Each program instance handles one (batch, head, chunk, query_sub_chunk).

    When SKIP_TILES=True, checks LTE_tile_max to skip tiles where
    all key positions' documents end before the query sub-chunk starts.
    """
    pid_bhc = tl.program_id(0)
    pid_qi = tl.program_id(1)

    num_chunks = T // chunk_size
    Ns = chunk_size // sub_chunk_size

    pid_c = pid_bhc % num_chunks
    pid_bh = pid_bhc // num_chunks
    pid_h = pid_bh % H
    pid_b = pid_bh // H

    chunk_start = pid_c * chunk_size
    qi_start = chunk_start + pid_qi * sub_chunk_size

    # Offsets
    qi_offsets = qi_start + tl.arange(0, sub_chunk_size)
    dk_offsets = tl.arange(0, dk)
    dv_offsets = tl.arange(0, dv)

    # Load Q: (c, dk)
    q_ptrs = (Q_ptr + pid_b * stride_qb + pid_h * stride_qh
              + qi_offsets[:, None] * stride_qt + dk_offsets[None, :] * stride_qd)
    q = tl.load(q_ptrs)

    # Load log_cumsum for query positions: (c,)
    lcs_q_ptrs = (log_cumsum_ptr + pid_b * stride_lcb + pid_h * stride_lch
                  + qi_offsets * stride_lct)
    log_cs_q = tl.load(lcs_q_ptrs)

    # Local indices for causal mask
    qi_local = pid_qi * sub_chunk_size + tl.arange(0, sub_chunk_size)

    acc = tl.zeros([sub_chunk_size, dv], dtype=tl.float32)

    for kj in range(Ns):
        # === CAUSAL SKIP ===
        if kj > pid_qi:
            continue

        # === FLASHMASK TILE-SKIP ===
        if SKIP_TILES:
            if kj < pid_qi:
                # Load LTE_tile_max for this key sub-chunk
                lte_max_ptr = (LTE_tile_max_ptr
                               + pid_b * stride_ltb + pid_h * stride_lth
                               + pid_c * stride_ltc + kj * stride_lts)
                lte_max = tl.load(lte_max_ptr)

                # If ALL keys' documents end before query sub-chunk starts,
                # the entire tile is zero (cross-document) -> skip
                if lte_max <= qi_start:
                    continue

        kj_start = chunk_start + kj * sub_chunk_size
        kj_offsets = kj_start + tl.arange(0, sub_chunk_size)

        # Load K: (c, dk)
        k_ptrs = (K_ptr + pid_b * stride_kb + pid_h * stride_kh
                  + kj_offsets[:, None] * stride_kt + dk_offsets[None, :] * stride_kd)
        k = tl.load(k_ptrs)

        # Load V: (c, dv)
        v_ptrs = (V_ptr + pid_b * stride_vb + pid_h * stride_vh
                  + kj_offsets[:, None] * stride_vt + dv_offsets[None, :] * stride_vd)
        v = tl.load(v_ptrs)

        # QK^T: (c, c)
        qk = tl.dot(q, tl.trans(k))

        # Gate mask from log cumulative sums
        lcs_k_ptrs = (log_cumsum_ptr + pid_b * stride_lcb + pid_h * stride_lch
                      + kj_offsets * stride_lct)
        log_cs_k = tl.load(lcs_k_ptrs)

        # D[i,j] = exp(log_cs[i] - log_cs[j]) for i >= j, else 0
        log_diff = log_cs_q[:, None] - log_cs_k[None, :]
        gate_mask = tl.exp(log_diff)

        # Causal mask within tile
        kj_local = kj * sub_chunk_size + tl.arange(0, sub_chunk_size)
        causal = qi_local[:, None] >= kj_local[None, :]
        gate_mask = tl.where(causal, gate_mask, 0.0)

        # NaN -> 0 (from log(0) at boundaries: exp(-inf - (-inf)) = exp(nan))
        gate_mask = tl.where(gate_mask != gate_mask, 0.0, gate_mask)
        # Inf -> 0 (safety)
        gate_mask = tl.where(gate_mask > 1e6, 0.0, gate_mask)

        # Apply mask and accumulate
        qk = qk * gate_mask
        acc += tl.dot(qk.to(v.dtype), v)

    # Store output
    o_ptrs = (O_ptr + pid_b * stride_ob + pid_h * stride_oh
              + qi_offsets[:, None] * stride_ot + dv_offsets[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty))


# ============================================================================
# Python Wrapper
# ============================================================================

def gla_intra_chunk_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: torch.Tensor,
    chunk_size: int = 128,
    sub_chunk_size: int = 16,
    use_tile_skip: bool = False,
    log_cumsum: torch.Tensor = None,
    LTE_tile_max: torch.Tensor = None,
) -> torch.Tensor:
    """
    Triton-based GLA intra-chunk computation.

    Args:
        Q: (B, H, T, dk) queries
        K: (B, H, T, dk) keys
        V: (B, H, T, dv) values
        alpha: (B, H, T) per-position gates
        chunk_size: primary chunk size C
        sub_chunk_size: secondary sub-chunk size c
        use_tile_skip: whether to use FlashMask tile-skip
        log_cumsum: precomputed log cumulative sum (optional, computed if None)
        LTE_tile_max: precomputed per-tile LTE max (optional, computed if None)

    Returns:
        O: (B, H, T, dv) output
    """
    B, H, T, dk = Q.shape
    dv = V.shape[-1]
    num_chunks = T // chunk_size
    Ns = chunk_size // sub_chunk_size

    # Ensure contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Precompute log cumsum if not provided
    if log_cumsum is None:
        log_cumsum = precompute_log_cumsum(alpha, chunk_size)
    log_cumsum = log_cumsum.contiguous()

    # Compute tile skip bounds if needed
    if use_tile_skip and LTE_tile_max is None:
        mask_info = compute_column_sparse_mask(alpha, chunk_size, sub_chunk_size)
        LTE_tile_max = mask_info['LTE_tile_max']

    # Dummy if not using tile skip
    if LTE_tile_max is None:
        LTE_tile_max = torch.zeros(B, H, num_chunks, Ns, dtype=torch.int32, device=Q.device).contiguous()
    LTE_tile_max = LTE_tile_max.contiguous()

    # Allocate output
    O = torch.zeros(B, H, T, dv, device=Q.device, dtype=Q.dtype).contiguous()

    # Grid: (B * H * num_chunks, Ns)
    grid = (B * H * num_chunks, Ns)

    _gla_intra_chunk_kernel[grid](
        Q, K, V, O,
        log_cumsum,
        LTE_tile_max,
        B, H, T, dk, dv,
        chunk_size, sub_chunk_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        log_cumsum.stride(0), log_cumsum.stride(1), log_cumsum.stride(2),
        LTE_tile_max.stride(0), LTE_tile_max.stride(1), LTE_tile_max.stride(2), LTE_tile_max.stride(3),
        SKIP_TILES=use_tile_skip,
    )

    return O

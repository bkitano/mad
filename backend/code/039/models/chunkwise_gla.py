"""
MVE 039: Chunkwise GLA Forward Pass Kernels

Implements the chunkwise Gated Linear Attention (GLA) intra-chunk computation:

For each chunk i, inner tile j:
    O_i += (Q_i @ K_j^T * D_ij) @ V_j      (Proposal eq. 1: intra-chunk)

where D_ij is the causal decay mask.

Three implementations:
1. PyTorch reference: Pure PyTorch for correctness
2. Triton baseline: Standard sequential kernel (load -> QK^T -> decay -> SV -> accumulate)
3. Triton pipelined: Software-pipelined with double-buffering that overlaps
   next tile loading with current tile's decay computation

The pipelined version tests the core hypothesis: can we overlap the sequential
element-wise decay masking with compute (GEMM) to improve throughput?

Key variables (from proposal):
- BLOCK_M, BLOCK_N: inner tile sizes (typically 64)
- d: head dimension (typically 64-128)
- chunk_size: outer chunk size
- gamma_t in (0, 1): per-token decay
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# PyTorch Reference Implementation (intra-chunk only)
# =============================================================================

def pytorch_chunkwise_gla_forward(Q, K, V, gamma, chunk_size=64):
    """
    Pure PyTorch chunkwise GLA intra-chunk forward pass.

    Computes only the intra-chunk attention (no cross-chunk state),
    matching what the Triton kernels compute.

    Args:
        Q: [B, H, T, d] - queries
        K: [B, H, T, d] - keys
        V: [B, H, T, d_v] - values
        gamma: [B, H, T] - per-token decay factors in (0, 1)
        chunk_size: L - chunk size for tiling

    Returns:
        O: [B, H, T, d_v] - output (intra-chunk only)
    """
    B, H, T, d = Q.shape
    d_v = V.shape[-1]
    assert T % chunk_size == 0, f"T={T} must be divisible by chunk_size={chunk_size}"

    num_chunks = T // chunk_size
    O = torch.zeros(B, H, T, d_v, device=Q.device, dtype=torch.float32)

    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size

        Q_c = Q[:, :, start:end, :].float()  # [B, H, L, d]
        K_c = K[:, :, start:end, :].float()  # [B, H, L, d]
        V_c = V[:, :, start:end, :].float()  # [B, H, L, d_v]
        g_c = gamma[:, :, start:end].float()  # [B, H, L]

        # Compute causal decay mask D[a,b] = prod_{t=b+1}^{a} gamma_t for a >= b
        log_g = torch.log(g_c.clamp(min=1e-6))  # [B, H, L]
        cum_log_g = torch.cumsum(log_g, dim=-1)  # [B, H, L]

        # D[a, b] = exp(cum_log_g[a] - cum_log_g[b]) for a >= b
        D = torch.exp(cum_log_g.unsqueeze(-1) - cum_log_g.unsqueeze(-2))  # [B, H, L, L]
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=Q.device))
        D = D * causal_mask  # [B, H, L, L]

        # Intra-chunk: O = (Q @ K^T * D) @ V
        S = torch.matmul(Q_c, K_c.transpose(-2, -1))  # [B, H, L, L]
        S = S * D  # element-wise decay masking
        O[:, :, start:end, :] = torch.matmul(S, V_c)  # [B, H, L, d_v]

    return O


# =============================================================================
# Triton Baseline Kernel: Sequential QK^T -> Decay -> SV
# =============================================================================

@triton.jit
def _chunkwise_gla_baseline_kernel(
    Q_ptr, K_ptr, V_ptr, cum_log_gamma_ptr, O_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vdv,
    stride_gb, stride_gh, stride_gt,
    stride_ob, stride_oh, stride_ot, stride_odv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    d: tl.constexpr, d_v: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Baseline chunkwise GLA: processes intra-chunk computation.
    Each program instance handles one (batch, head, chunk, query_tile_m) block.

    Sequential pipeline: for each N tile: load K,V -> QK^T -> decay -> SV -> accumulate
    """
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_m = tl.program_id(2)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    chunk_start = pid_chunk * chunk_size
    m_start = pid_m * BLOCK_M

    # Output accumulator: [BLOCK_M, d_v] in float32
    acc = tl.zeros((BLOCK_M, d_v), dtype=tl.float32)

    # Ranges
    m_range = tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, d)
    dv_range = tl.arange(0, d_v)

    # Load Q tile: [BLOCK_M, d]
    q_base = pid_b * stride_qb + pid_h * stride_qh
    q_row = chunk_start + m_start + m_range
    q_ptrs = Q_ptr + q_base + q_row[:, None] * stride_qt + d_range[None, :] * stride_qd
    q_mask = (q_row[:, None] < T) & (d_range[None, :] < d)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Load cum_log_gamma for Q positions
    g_base = pid_b * stride_gb + pid_h * stride_gh
    q_g_ptrs = cum_log_gamma_ptr + g_base + q_row * stride_gt
    q_g_mask = q_row < T
    q_cum_log = tl.load(q_g_ptrs, mask=q_g_mask, other=0.0).to(tl.float32)

    # Number of N tiles within this chunk
    num_n_tiles = chunk_size // BLOCK_N

    for n_idx in range(num_n_tiles):
        n_start = n_idx * BLOCK_N
        n_range = tl.arange(0, BLOCK_N)
        k_row = chunk_start + n_start + n_range

        # Step 1: Load K tile [BLOCK_N, d]
        k_base = pid_b * stride_kb + pid_h * stride_kh
        k_ptrs = K_ptr + k_base + k_row[:, None] * stride_kt + d_range[None, :] * stride_kd
        k_mask = (k_row[:, None] < T) & (d_range[None, :] < d)
        K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Step 2: GEMM 1 - S = Q @ K^T  [BLOCK_M, BLOCK_N]
        S = tl.dot(Q_tile, tl.trans(K_tile))

        # Step 3: Decay mask
        k_g_ptrs = cum_log_gamma_ptr + g_base + k_row * stride_gt
        k_g_mask = k_row < T
        k_cum_log = tl.load(k_g_ptrs, mask=k_g_mask, other=0.0).to(tl.float32)

        # Causal: q_pos >= k_pos (within chunk: m_start+m >= n_start+n)
        q_pos = m_start + m_range
        k_pos = n_start + n_range
        causal = q_pos[:, None] >= k_pos[None, :]
        decay = tl.exp(q_cum_log[:, None] - k_cum_log[None, :])
        decay = tl.where(causal, decay, 0.0)

        S = S * decay

        # Step 4: Load V tile [BLOCK_N, d_v]
        v_base = pid_b * stride_vb + pid_h * stride_vh
        v_ptrs = V_ptr + v_base + k_row[:, None] * stride_vt + dv_range[None, :] * stride_vdv
        v_mask = (k_row[:, None] < T) & (dv_range[None, :] < d_v)
        V_tile = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # Step 5: GEMM 2 - O += S @ V  [BLOCK_M, d_v]
        acc += tl.dot(S.to(tl.float32), V_tile)

    # Store output
    o_base = pid_b * stride_ob + pid_h * stride_oh
    o_row = chunk_start + m_start + m_range
    o_ptrs = O_ptr + o_base + o_row[:, None] * stride_ot + dv_range[None, :] * stride_odv
    o_mask = (o_row[:, None] < T) & (dv_range[None, :] < d_v)
    tl.store(o_ptrs, acc, mask=o_mask)


# =============================================================================
# Triton Pipelined Kernel: Overlapped loading + decay with GEMM
# =============================================================================

@triton.jit
def _chunkwise_gla_pipelined_kernel(
    Q_ptr, K_ptr, V_ptr, cum_log_gamma_ptr, O_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vdv,
    stride_gb, stride_gh, stride_gt,
    stride_ob, stride_oh, stride_ot, stride_odv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    d: tl.constexpr, d_v: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Pipelined chunkwise GLA: overlaps next tile loading with current computation.

    Pipeline schedule (adapted from proposal):
    - Pre-load first K,V tile
    - For each tile: issue loads for NEXT tile, then compute current tile's
      GEMM + decay + GEMM. The loads can overlap with compute.
    - Process last tile without pre-loading
    """
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_m = tl.program_id(2)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    chunk_start = pid_chunk * chunk_size
    m_start = pid_m * BLOCK_M

    # Output accumulator
    acc = tl.zeros((BLOCK_M, d_v), dtype=tl.float32)

    # Ranges
    m_range = tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, d)
    dv_range = tl.arange(0, d_v)
    n_range = tl.arange(0, BLOCK_N)

    # Load Q tile once [BLOCK_M, d]
    q_base = pid_b * stride_qb + pid_h * stride_qh
    q_row = chunk_start + m_start + m_range
    q_ptrs = Q_ptr + q_base + q_row[:, None] * stride_qt + d_range[None, :] * stride_qd
    q_mask = (q_row[:, None] < T) & (d_range[None, :] < d)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Pre-compute Q cumulative log-gamma
    g_base = pid_b * stride_gb + pid_h * stride_gh
    q_g_ptrs = cum_log_gamma_ptr + g_base + q_row * stride_gt
    q_g_mask = q_row < T
    q_cum_log = tl.load(q_g_ptrs, mask=q_g_mask, other=0.0).to(tl.float32)

    num_n_tiles = chunk_size // BLOCK_N

    q_pos = m_start + m_range

    # ===== PRE-LOAD first K, V, gamma tiles =====
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh

    k_row_0 = chunk_start + n_range
    k_ptrs_0 = K_ptr + k_base + k_row_0[:, None] * stride_kt + d_range[None, :] * stride_kd
    k_mask_0 = (k_row_0[:, None] < T) & (d_range[None, :] < d)
    K_cur = tl.load(k_ptrs_0, mask=k_mask_0, other=0.0).to(tl.float32)

    v_ptrs_0 = V_ptr + v_base + k_row_0[:, None] * stride_vt + dv_range[None, :] * stride_vdv
    v_mask_0 = (k_row_0[:, None] < T) & (dv_range[None, :] < d_v)
    V_cur = tl.load(v_ptrs_0, mask=v_mask_0, other=0.0).to(tl.float32)

    kg_ptrs_0 = cum_log_gamma_ptr + g_base + k_row_0 * stride_gt
    k_cum_log_cur = tl.load(kg_ptrs_0, mask=k_row_0 < T, other=0.0).to(tl.float32)

    # ===== MAIN PIPELINE LOOP =====
    # Process each N tile. For all but the last, pre-load the NEXT tile
    # while computing the current tile. This overlaps memory loads with compute.
    for n_idx in range(num_n_tiles):
        n_start = n_idx * BLOCK_N

        # Pre-fetch NEXT tile's K, V, gamma (issues loads that overlap with compute below)
        # For the last tile, these loads go to valid memory (clamped) but results are unused
        next_n_start = tl.minimum((n_idx + 1) * BLOCK_N, chunk_size - BLOCK_N)
        k_row_next = chunk_start + next_n_start + n_range
        k_ptrs_next = K_ptr + k_base + k_row_next[:, None] * stride_kt + d_range[None, :] * stride_kd
        k_mask_next = (k_row_next[:, None] < T) & (d_range[None, :] < d)
        K_next = tl.load(k_ptrs_next, mask=k_mask_next, other=0.0).to(tl.float32)

        v_ptrs_next = V_ptr + v_base + k_row_next[:, None] * stride_vt + dv_range[None, :] * stride_vdv
        v_mask_next = (k_row_next[:, None] < T) & (dv_range[None, :] < d_v)
        V_next = tl.load(v_ptrs_next, mask=v_mask_next, other=0.0).to(tl.float32)

        kg_ptrs_next = cum_log_gamma_ptr + g_base + k_row_next * stride_gt
        k_cum_log_next = tl.load(kg_ptrs_next, mask=k_row_next < T, other=0.0).to(tl.float32)

        # ---- GEMM 1: S = Q @ K^T ---- (tensor cores)
        S = tl.dot(Q_tile, tl.trans(K_cur))

        # ---- Decay mask (element-wise, overlaps with prefetch) ----
        k_pos = n_start + n_range
        causal = q_pos[:, None] >= k_pos[None, :]
        decay = tl.exp(q_cum_log[:, None] - k_cum_log_cur[None, :])
        decay = tl.where(causal, decay, 0.0)
        S = S * decay

        # ---- GEMM 2: O += S @ V ---- (tensor cores)
        acc += tl.dot(S.to(tl.float32), V_cur)

        # Rotate buffers: next tile becomes current tile
        K_cur = K_next
        V_cur = V_next
        k_cum_log_cur = k_cum_log_next

    # Store output
    o_base = pid_b * stride_ob + pid_h * stride_oh
    o_row = chunk_start + m_start + m_range
    o_ptrs = O_ptr + o_base + o_row[:, None] * stride_ot + dv_range[None, :] * stride_odv
    o_mask = (o_row[:, None] < T) & (dv_range[None, :] < d_v)
    tl.store(o_ptrs, acc, mask=o_mask)


# =============================================================================
# Wrapper Functions
# =============================================================================

def _precompute_chunk_cum_log_gamma(gamma, chunk_size):
    """
    Precompute per-chunk cumulative log-gamma for decay mask.

    Args:
        gamma: [B, H, T] raw decay factors in (0, 1)
        chunk_size: chunk size

    Returns:
        cum_log_gamma: [B, H, T] per-chunk cumulative log-gamma
    """
    B, H, T = gamma.shape
    num_chunks = T // chunk_size
    gamma_reshaped = gamma.reshape(B, H, num_chunks, chunk_size)
    log_g = torch.log(gamma_reshaped.float().clamp(min=1e-6))
    cum_log_g = torch.cumsum(log_g, dim=-1)  # cumsum within each chunk
    return cum_log_g.reshape(B, H, T).contiguous()


def triton_chunkwise_gla_baseline(Q, K, V, gamma, chunk_size=64):
    """
    Baseline chunkwise GLA forward pass using Triton.

    Sequential pipeline: load tiles -> QK^T -> decay mask -> SV -> accumulate

    Args:
        Q: [B, H, T, d] queries
        K: [B, H, T, d] keys
        V: [B, H, T, d_v] values
        gamma: [B, H, T] per-token decay factors in (0, 1)
        chunk_size: chunk size for tiling

    Returns:
        O: [B, H, T, d_v] output
    """
    B, H, T, d = Q.shape
    d_v = V.shape[-1]
    assert T % chunk_size == 0
    assert chunk_size % 64 == 0 or chunk_size == 32, f"chunk_size must be multiple of 64 (or 32)"
    num_chunks = T // chunk_size

    BLOCK_M = min(64, chunk_size)
    BLOCK_N = min(64, chunk_size)
    num_m_tiles = chunk_size // BLOCK_M

    # Precompute per-chunk cumulative log-gamma
    cum_log_gamma = _precompute_chunk_cum_log_gamma(gamma, chunk_size)

    O = torch.zeros(B, H, T, d_v, device=Q.device, dtype=torch.float32)

    # Make sure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    cum_log_gamma = cum_log_gamma.contiguous()

    grid = (B * H, num_chunks, num_m_tiles)

    _chunkwise_gla_baseline_kernel[grid](
        Q, K, V, cum_log_gamma, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        cum_log_gamma.stride(0), cum_log_gamma.stride(1), cum_log_gamma.stride(2),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, T, d, d_v, chunk_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return O


def triton_chunkwise_gla_pipelined(Q, K, V, gamma, chunk_size=64):
    """
    Pipelined chunkwise GLA forward pass using Triton.

    Software-pipelined: pre-loads next tile while computing current tile,
    mimicking warp-specialized pingpong scheduling.

    Args:
        Q: [B, H, T, d] queries
        K: [B, H, T, d] keys
        V: [B, H, T, d_v] values
        gamma: [B, H, T] per-token decay factors in (0, 1)
        chunk_size: chunk size for tiling

    Returns:
        O: [B, H, T, d_v] output
    """
    B, H, T, d = Q.shape
    d_v = V.shape[-1]
    assert T % chunk_size == 0
    num_chunks = T // chunk_size

    BLOCK_M = min(64, chunk_size)
    BLOCK_N = min(64, chunk_size)
    num_m_tiles = chunk_size // BLOCK_M

    # Precompute per-chunk cumulative log-gamma
    cum_log_gamma = _precompute_chunk_cum_log_gamma(gamma, chunk_size)

    O = torch.zeros(B, H, T, d_v, device=Q.device, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    cum_log_gamma = cum_log_gamma.contiguous()

    grid = (B * H, num_chunks, num_m_tiles)

    _chunkwise_gla_pipelined_kernel[grid](
        Q, K, V, cum_log_gamma, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        cum_log_gamma.stride(0), cum_log_gamma.stride(1), cum_log_gamma.stride(2),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, T, d, d_v, chunk_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return O

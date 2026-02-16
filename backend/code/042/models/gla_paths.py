"""
GLA Intra-Chunk Computation: Two Contraction Paths

From Proposal 042: Contraction-Ordered Multi-Operand Chunkwise GLA Fusion

The intra-chunk output of GLA for chunk j is:
  O_hat_j = (Q_j @ K_j^T * M_j) @ V_j + Q_j @ h_jC

where:
  Q_j, K_j in R^{C x d_k}  (query, key)
  V_j in R^{C x d_v}        (value)
  M_j in R^{C x C}          (causal decay mask, lower triangular)
  h_jC in R^{d_k x d_v}     (boundary state from inter-chunk scan)

Path 1 (Standard): Left-to-right evaluation
  1. S = Q @ K^T              [C x C]  -- O(C^2 d_k)
  2. S_tilde = S * M           [C x C]  -- O(C^2)
  3. O_intra = S_tilde @ V     [C x d_v] -- O(C^2 d_v)
  4. O_state = Q @ h           [C x d_v] -- O(C d_k d_v)
  5. O_hat = O_intra + O_state [C x d_v] -- O(C d_v)
  Total: 2C^2(d_k + d_v) + 2C*d_k*d_v

Path 2 (Right-associated with rank-r mask correction):
  Decompose M = L - Delta where L is all-ones lower triangular
  and Delta = M_lowertri - M (the deviation from identity-like mask)

  Then: O_hat = Q @ cumsum(K^T @ V) - correction(Q, K, V, Delta) + Q @ h

  If Delta has low effective rank r:
    Delta ≈ U @ Sigma @ W^T  (rank-r SVD)

  Then correction uses modified Q, K with rank-r factors:
    Q_tilde = Q * (U @ Sigma)    -- broadcast multiply
    K_tilde = K * W              -- broadcast multiply
    correction = Q_tilde @ K_tilde^T @ V  -- two GEMMs with effective dim r

  Total (if r << C): 5C*d_k*d_v + 2C^2*r*d_v (much cheaper when r is small)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def build_causal_decay_mask(alpha: torch.Tensor) -> torch.Tensor:
    """
    Build the causal decay mask M_j from per-timestep decay rates alpha.

    M_j[i, k] = prod_{t=k+1}^{i} alpha_t   for k < i
    M_j[i, i] = 1
    M_j[i, k] = 0                           for k > i

    Args:
        alpha: (C,) per-timestep decay rates in (0, 1)

    Returns:
        M: (C, C) causal decay mask (lower triangular)
    """
    C = alpha.shape[0]
    # log_alpha[t] = log(alpha[t])
    log_alpha = torch.log(alpha + 1e-8)
    # cumsum_log[i] = sum_{t=0}^{i} log(alpha[t])
    cumsum_log = torch.cumsum(log_alpha, dim=0)
    # M[i, k] = exp(cumsum_log[i] - cumsum_log[k]) for k <= i
    # This gives prod_{t=k+1}^{i} alpha_t
    M = torch.exp(cumsum_log.unsqueeze(1) - cumsum_log.unsqueeze(0))
    # Apply causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(C, C, device=alpha.device, dtype=alpha.dtype))
    M = M * causal_mask
    return M


def path1_standard(
    Q: torch.Tensor,  # (C, d_k)
    K: torch.Tensor,  # (C, d_k)
    V: torch.Tensor,  # (C, d_v)
    M: torch.Tensor,  # (C, C)
    h: torch.Tensor,  # (d_k, d_v)
) -> torch.Tensor:
    """
    Path 1: Standard left-to-right evaluation.

    O_hat = (Q @ K^T * M) @ V + Q @ h

    Steps:
      1. S = Q @ K^T              -- GEMM: O(C^2 d_k)
      2. S_tilde = S * M          -- Elementwise: O(C^2)
      3. O_intra = S_tilde @ V    -- GEMM: O(C^2 d_v)
      4. O_state = Q @ h          -- GEMM: O(C d_k d_v)
      5. O_hat = O_intra + O_state -- Add: O(C d_v)

    Returns: O_hat (C, d_v)
    """
    # Step 1: Q @ K^T -> (C, C)
    S = Q @ K.T
    # Step 2: Apply causal decay mask
    S_tilde = S * M
    # Step 3: Attention @ V -> (C, d_v)
    O_intra = S_tilde @ V
    # Step 4: State correction -> (C, d_v)
    O_state = Q @ h
    # Step 5: Combine
    O_hat = O_intra + O_state
    return O_hat


def decompose_mask_low_rank(
    M: torch.Tensor,  # (C, C)
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose the deviation Delta = L - M into a rank-r approximation.

    L is the all-ones lower triangular matrix.
    Delta = L - M captures how much the mask deviates from pure causal attention.

    When alpha_t ≈ 1 (slow decay), M ≈ L, so Delta has low effective rank.

    Returns:
        U: (C, r)
        S_diag: (r,)
        W: (C, r)
    """
    C = M.shape[0]
    L = torch.tril(torch.ones(C, C, device=M.device, dtype=M.dtype))
    Delta = L - M

    # SVD of Delta
    # Use float32 for SVD stability, then cast back
    Delta_f32 = Delta.float()
    U_full, S_full, Vh_full = torch.linalg.svd(Delta_f32, full_matrices=False)

    # Truncate to rank r
    r = min(rank, C)
    U = U_full[:, :r].to(M.dtype)
    S_diag = S_full[:r].to(M.dtype)
    W = Vh_full[:r, :].T.to(M.dtype)  # (C, r)

    return U, S_diag, W


def path2_right_associated(
    Q: torch.Tensor,      # (C, d_k)
    K: torch.Tensor,      # (C, d_k)
    V: torch.Tensor,      # (C, d_v)
    M: torch.Tensor,      # (C, C)  -- only used for mask decomposition
    h: torch.Tensor,      # (d_k, d_v)
    rank: int,            # rank for mask approximation
    U: torch.Tensor = None,  # precomputed (C, r)
    S_diag: torch.Tensor = None,  # precomputed (r,)
    W: torch.Tensor = None,  # precomputed (C, r)
) -> torch.Tensor:
    """
    Path 2: Right-associated evaluation with rank-r mask correction.

    O_hat = Q @ cumsum(K^T V) - correction + Q @ h

    where cumsum computes the causal prefix sum of outer products:
      cumsum(K^T V)[i] = sum_{k=0}^{i} k_k @ v_k^T  (cumulative along C dim)

    And correction uses rank-r approximation of (L - M):
      Delta ≈ U @ diag(S) @ W^T
      correction = (Q * (U @ diag(S))) @ (K * W)^T @ V

    Steps:
      1. P = K^T @ V cumulative sum along C  -- O(C d_k d_v)
      2. O_right = Q @ P_cumsum               -- O(C d_k d_v)
      3. Q_tilde = Q * (U @ S)                -- O(C r d_k)
      4. K_tilde = K * W                      -- O(C r d_k)
      5. correction = (Q_tilde @ K_tilde^T) @ V  -- O(C^2 r + C^2 d_v) or O(C r d_k d_v)
      6. O_state = Q @ h                      -- O(C d_k d_v)
      7. O_hat = O_right - correction + O_state

    Returns: O_hat (C, d_v)
    """
    C, d_k = Q.shape
    d_v = V.shape[1]

    if U is None or S_diag is None or W is None:
        U, S_diag, W = decompose_mask_low_rank(M, rank)

    r = U.shape[1]

    # Step 1: Compute causal cumulative sum of outer products
    # KtV[i] = K[i]^T @ V[i] is per-row outer product, then cumsum
    # Actually: we want cumsum along the C dimension of K[:i]^T @ V[:i]
    # This is: for each position i, sum_{k=0}^{i} K[k,:].unsqueeze(1) @ V[k,:].unsqueeze(0)
    # = cumsum of (K.unsqueeze(2) * V.unsqueeze(1)) along dim 0
    # Shape: (C, d_k, d_v) cumulative

    # Efficient: KtV_cum[i] = sum_{k=0}^{i} k_k v_k^T
    KV_outer = K.unsqueeze(2) * V.unsqueeze(1)  # (C, d_k, d_v)
    KV_cumsum = torch.cumsum(KV_outer, dim=0)    # (C, d_k, d_v)

    # Step 2: O_right[i] = Q[i] @ KV_cumsum[i]
    # = sum over d_k: Q[i, q] * KV_cumsum[i, q, v]
    O_right = torch.einsum('cq,cqv->cv', Q, KV_cumsum)  # (C, d_v)

    # Step 3-5: Correction term using low-rank mask decomposition
    # Delta ≈ U @ diag(S) @ W^T
    # The correction is: (Q @ K^T * Delta) @ V
    # With low-rank Delta: sum_r [ (Q * U[:,r]*S[r]) @ (K * W[:,r])^T ] @ V
    # But we can batch this more efficiently:

    # Q_tilde = Q * (U @ diag(S)) -> (C, d_k) * (C, r) needs broadcasting
    # For each rank component r_i:
    #   correction_r = (Q * U[:,r_i] * S[r_i]) @ (K * W[:,r_i])^T @ V

    US = U * S_diag.unsqueeze(0)  # (C, r)

    # For small r, loop is fine. For larger r, batch.
    if r <= 8:
        # Loop approach (lower memory, fine for small r)
        correction = torch.zeros(C, d_v, device=Q.device, dtype=Q.dtype)
        for i in range(r):
            Q_mod = Q * US[:, i:i+1]     # (C, d_k)
            K_mod = K * W[:, i:i+1]      # (C, d_k)
            # (Q_mod @ K_mod^T) @ V -> (C, C) @ (C, d_v)
            attn_r = Q_mod @ K_mod.T     # (C, C)
            correction += attn_r @ V     # (C, d_v)
        # Apply causal mask to correction
        # Actually, the low-rank approximation of Delta is only valid for lower-triangular part
        # We need: sum_r (Q * US_r) @ (K * W_r)^T -> needs causal masking
        # Let's redo with causal mask
        correction = torch.zeros(C, d_v, device=Q.device, dtype=Q.dtype)
        causal = torch.tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))
        for i in range(r):
            Q_mod = Q * US[:, i:i+1]
            K_mod = K * W[:, i:i+1]
            attn_r = (Q_mod @ K_mod.T) * causal
            correction += attn_r @ V
    else:
        # Batched approach for larger r
        # Reshape for batched matmul
        Q_tilde = Q.unsqueeze(0) * US.T.unsqueeze(2)  # (r, C, d_k)
        K_tilde = K.unsqueeze(0) * W.T.unsqueeze(2)   # (r, C, d_k)
        attn_r = Q_tilde @ K_tilde.transpose(1, 2)    # (r, C, C)
        causal = torch.tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))
        attn_r = attn_r * causal.unsqueeze(0)
        correction = (attn_r @ V.unsqueeze(0)).sum(0)  # (C, d_v)

    # Step 6: State correction
    O_state = Q @ h  # (C, d_v)

    # Step 7: Combine
    # O_hat = O_right - correction + O_state
    # O_right gives Q @ L_lower_tri @ (K^T V), which is the all-ones causal attention
    # Subtracting correction gives Q @ M @ (K^T V) approximately
    O_hat = O_right - correction + O_state

    return O_hat


def path2_right_associated_exact(
    Q: torch.Tensor,      # (C, d_k)
    K: torch.Tensor,      # (C, d_k)
    V: torch.Tensor,      # (C, d_v)
    M: torch.Tensor,      # (C, C)
    h: torch.Tensor,      # (d_k, d_v)
) -> torch.Tensor:
    """
    Path 2 with exact mask (rank = C). Used to verify correctness
    independent of rank approximation quality.
    """
    return path2_right_associated(Q, K, V, M, h, rank=Q.shape[0])


def compute_flops(C: int, d_k: int, d_v: int, r: int) -> Dict[str, int]:
    """
    Compute theoretical FLOPs for both paths.

    Path 1: 2C^2*d_k + C^2 + 2C^2*d_v + 2C*d_k*d_v
    Path 2: C*d_k*d_v (outer products) + C*d_k*d_v (cumsum reads)
            + C*d_k*d_v (Q @ cumsum)
            + r * (2C^2*d_k + 2C^2*d_v) (correction via r rank-1 attentions)
            + 2C*d_k*d_v (state correction)
    """
    # Path 1
    p1_qkt = 2 * C * C * d_k        # Q @ K^T
    p1_mask = C * C                   # elementwise mask
    p1_sv = 2 * C * C * d_v          # S_tilde @ V
    p1_state = 2 * C * d_k * d_v     # Q @ h
    p1_total = p1_qkt + p1_mask + p1_sv + p1_state

    # Path 2
    p2_kv_outer = 2 * C * d_k * d_v   # K outer V per position
    p2_cumsum = C * d_k * d_v          # cumulative sum (adds)
    p2_q_cum = 2 * C * d_k * d_v      # Q @ cumsum result (einsum)
    p2_correction = r * (2 * C * C * d_k + 2 * C * C * d_v)  # r rank-1 corrections
    p2_state = 2 * C * d_k * d_v      # Q @ h
    p2_total = p2_kv_outer + p2_cumsum + p2_q_cum + p2_correction + p2_state

    return {
        "path1_flops": p1_total,
        "path2_flops": p2_total,
        "ratio": p2_total / p1_total if p1_total > 0 else float('inf'),
        "path1_breakdown": {
            "QKt": p1_qkt,
            "mask": p1_mask,
            "SV": p1_sv,
            "state": p1_state,
        },
        "path2_breakdown": {
            "KV_outer": p2_kv_outer,
            "cumsum": p2_cumsum,
            "Q_cumsum": p2_q_cum,
            "correction": p2_correction,
            "state": p2_state,
        },
    }


def measure_effective_rank(M: torch.Tensor, eps: float = 0.01) -> int:
    """
    Measure the effective rank of the mask deviation Delta = L - M.

    Effective rank is the number of singular values > eps * max_singular_value.

    This is the key quantity that determines whether Path 2 is beneficial.
    """
    C = M.shape[0]
    L = torch.tril(torch.ones(C, C, device=M.device, dtype=torch.float32))
    Delta = L - M.float()

    _, S, _ = torch.linalg.svd(Delta, full_matrices=False)
    threshold = eps * S[0]
    effective_rank = (S > threshold).sum().item()

    return effective_rank

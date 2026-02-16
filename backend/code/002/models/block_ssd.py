"""
Block-SSD Forward Pass for DeltaNet

SSD-style block decomposition: split sequence into sub-blocks of size Q,
compute inter-block state contribution via matmul, run intra-block delta
rule with corrected values.

DeltaNet recurrence (S_0 = I):
    S_t = S_{t-1} + beta_t * k_t * (v_t - S_{t-1}^T k_t)^T

Decomposition for sub-block starting at state S_init:
    o_t = S_init @ q_t + M_t @ q_t

where M_t evolves as:
    M_t = (I - beta_t k_t k_t^T) M_{t-1} + beta_t k_t (v_t - S_init^T k_t)^T

Key optimizations:
1. S_init @ Q^T: one large matmul instead of Q separate matvecs (tensor core friendly)
2. V_corrected = V - K @ S_init: one large matmul (tensor core friendly)
3. Intra-block runs for Q << T steps (reduced sequential depth)
"""

import torch


def block_ssd_forward(K: torch.Tensor, V: torch.Tensor, beta: torch.Tensor,
                      Q_in: torch.Tensor = None, chunk_size: int = 64,
                      sub_block: int = 16) -> torch.Tensor:
    """
    Block-SSD forward: inter-block matmul + sequential intra-block.

    Args:
        K: (T, d) — key vectors (L2-normalized)
        V: (T, d) — value vectors
        beta: (T,) — learning rates in (0, 1)
        Q_in: (T, d) — query vectors (defaults to V)
        chunk_size: C — outer chunk size (for future inter-chunk optimization)
        sub_block: Q — sub-block size

    Returns:
        outputs: (T, d)
    """
    T, d = K.shape
    if Q_in is None:
        Q_in = V

    S = torch.eye(d, device=K.device, dtype=K.dtype)
    outputs = torch.empty(T, d, device=K.device, dtype=K.dtype)

    for sb_start in range(0, T, sub_block):
        sb_end = min(sb_start + sub_block, T)
        Q_sz = sb_end - sb_start

        Ksb = K[sb_start:sb_end]     # (Q, d)
        Vsb = V[sb_start:sb_end]     # (Q, d)
        Qsb = Q_in[sb_start:sb_end]  # (Q, d)
        bsb = beta[sb_start:sb_end]  # (Q,)

        # ═══════════════════════════════════════════
        # Inter-block: batched matmul (tensor core)
        # Replaces Q separate matvecs with ONE (Q,d)×(d,d) matmul
        # ═══════════════════════════════════════════
        inter = Qsb @ S.T       # (Q, d) [MATMUL: Q*d*d FLOPs]

        # ═══════════════════════════════════════════
        # Value correction: batched matmul (tensor core)
        # v'_t = v_t - S^T @ k_t for all t in sub-block at once
        # ═══════════════════════════════════════════
        Vc = Vsb - Ksb @ S      # (Q, d) [MATMUL: Q*d*d FLOPs]

        # ═══════════════════════════════════════════
        # Intra-block: sequential delta rule on deviation M
        # M_t evolves from 0 with corrected values v'
        # Only Q steps (not T steps!)
        # ═══════════════════════════════════════════
        M = torch.zeros(d, d, device=K.device, dtype=K.dtype)
        for t in range(Q_sz):
            outputs[sb_start + t] = inter[t] + M @ Qsb[t]
            delta_M = Vc[t] - M.T @ Ksb[t]
            M.add_(bsb[t] * torch.outer(Ksb[t], delta_M))

        # Update inter-block state
        S.add_(M)

    return outputs


def block_ssd_forward_v2(K: torch.Tensor, V: torch.Tensor, beta: torch.Tensor,
                          Q_in: torch.Tensor = None, chunk_size: int = 64,
                          sub_block: int = 16) -> torch.Tensor:
    """
    Alias for block_ssd_forward (v2 uses same algorithm for now).
    The UT transform for fully-matmul intra-block requires further research
    on the correct WY decomposition for DeltaNet's state recurrence.
    """
    return block_ssd_forward(K, V, beta, Q_in, chunk_size, sub_block)

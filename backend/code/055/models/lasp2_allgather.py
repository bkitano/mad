"""
LASP-2 AllGather baseline for Gated DeltaNet Sequence Parallelism.

LASP-2 uses AllGather to collect the full state matrix and WY factors
from ALL P devices, then computes the prefix scan locally.

Communication per device: O(P * (d_v*d_k + d_k^2))
  - Grows linearly with P (number of GPUs)

This is the baseline that our WY-All-Scan aims to beat.
"""

import torch
import torch.distributed as dist
import time
from typing import Tuple, Dict


def lasp2_allgather_scan(
    S_local: torch.Tensor,     # (H, d_v, d_k) - local states per head
    gamma_dev: torch.Tensor,   # (H, d_k) - cumulative gating per head
    W_dev: torch.Tensor,       # (H, d_k, d_k) - WY factor W per head
    K_dev: torch.Tensor,       # (H, d_k, d_k) - WY factor K per head
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    LASP-2 AllGather + local prefix scan.

    Steps:
    1. AllGather S_local, gamma_dev, W_dev, K_dev from all P devices
    2. Each device locally computes the prefix scan over all gathered data
    3. Extract this device's corrected state

    Communication: AllGather of (d_v*d_k + d_k + d_k*d_k + d_k*d_k) * H elements
                   = O(P * H * (d_v*d_k + 2*d_k^2 + d_k)) total
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    H, d_v, d_k = S_local.shape

    timing = {}

    # --- Phase 1: AllGather all states and factors ---
    t_comm_start = time.perf_counter()

    # AllGather S_local: each device contributes (H, d_v, d_k)
    S_all = [torch.empty_like(S_local) for _ in range(world_size)]
    gamma_all = [torch.empty_like(gamma_dev) for _ in range(world_size)]
    W_all = [torch.empty_like(W_dev) for _ in range(world_size)]
    K_all = [torch.empty_like(K_dev) for _ in range(world_size)]

    dist.all_gather(S_all, S_local)
    dist.all_gather(gamma_all, gamma_dev)
    dist.all_gather(W_all, W_dev)
    dist.all_gather(K_all, K_dev)

    torch.cuda.synchronize()
    t_comm_end = time.perf_counter()

    # --- Phase 2: Local prefix scan ---
    t_compute_start = time.perf_counter()

    # Compute prefix scan: S_global[0] = S[0]
    # S_global[p] = gamma[p] * (S_global[p-1] @ (I - W[p].T @ K[p])) + S[p]
    S_globals = [S_all[0].clone()]  # Device 0 has no predecessor

    for p in range(1, world_size):
        S_prev = S_globals[p - 1]  # (H, d_v, d_k)
        gamma_p = gamma_all[p]     # (H, d_k)
        W_p = W_all[p]             # (H, d_k, d_k)
        K_p = K_all[p]             # (H, d_k, d_k)
        S_p = S_all[p]             # (H, d_v, d_k)

        # Per-head WY correction
        S_global_p = torch.zeros_like(S_p)
        for h in range(H):
            # WY correction: S_prev @ (I - W.T @ K) = S_prev - S_prev @ W.T @ K
            WtK = W_p[h].T @ K_p[h]  # (d_k, d_k)
            S_corrected = S_prev[h] - S_prev[h] @ WtK  # (d_v, d_k)

            # Apply gating
            S_corrected = gamma_p[h].unsqueeze(0) * S_corrected  # (d_v, d_k)

            # Add local contribution
            S_global_p[h] = S_corrected + S_p[h]

        S_globals.append(S_global_p)

    torch.cuda.synchronize()
    t_compute_end = time.perf_counter()

    # This device's corrected global state
    S_result = S_globals[rank]

    timing["comm_ms"] = (t_comm_end - t_comm_start) * 1000
    timing["compute_ms"] = (t_compute_end - t_compute_start) * 1000
    timing["total_ms"] = timing["comm_ms"] + timing["compute_ms"]

    return S_result, timing

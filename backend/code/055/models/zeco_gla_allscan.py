"""
ZeCO-GLA All-Scan baseline (diagonal-only transitions).

This implements ZeCO's All-Scan for GLA/Mamba-2 models where the
state transition is purely diagonal (elementwise gating).

Communication per device: O(H * d_v * d_k) - just the state matrix
  - P-independent (no WY factors needed since transition is diagonal)

The scan operation is just elementwise multiply by gamma:
    S_send = gamma_dev * S_recv + S_local

This is the communication efficiency ceiling for diagonal models.
Our WY-All-Scan should be within 2x of this latency (3x message size
but similar pipeline structure).
"""

import torch
import torch.distributed as dist
import time
from typing import Tuple, Dict


def zeco_gla_allscan(
    S_local: torch.Tensor,     # (H, d_v, d_k) - local states per head
    gamma_dev: torch.Tensor,   # (H, d_k) - cumulative gating per head
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ZeCO All-Scan for diagonal-only transitions (GLA/Mamba-2).

    P2P pipeline where each device sends only the state matrix.
    The scan operator is: S_out = gamma * S_recv + S_local

    Communication: P2P of (H * d_v * d_k) elements per step
    No WY factors needed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    H, d_v, d_k = S_local.shape

    timing = {}

    # Buffer for state matrix only (no WY factors)
    buf_size = H * d_v * d_k
    send_buf = torch.empty(buf_size, device=S_local.device, dtype=S_local.dtype)
    recv_buf = torch.empty(buf_size, device=S_local.device, dtype=S_local.dtype)

    # Start with local state for device 0
    S_global = S_local.clone() if rank == 0 else torch.zeros_like(S_local)

    total_comm_time = 0.0
    total_compute_time = 0.0

    for step in range(world_size - 1):
        sender = step
        receiver = step + 1

        if rank == sender:
            send_buf[:] = S_global.reshape(-1)

            t0 = time.perf_counter()
            dist.send(send_buf, dst=receiver)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_comm_time += (t1 - t0)

        elif rank == receiver:
            t0 = time.perf_counter()
            dist.recv(recv_buf, src=sender)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_comm_time += (t1 - t0)

            S_recv = recv_buf.reshape(H, d_v, d_k)

            # Diagonal-only scan: elementwise multiply + add
            t2 = time.perf_counter()
            S_global = gamma_dev.unsqueeze(1) * S_recv + S_local  # (H, d_v, d_k)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            total_compute_time += (t3 - t2)

        dist.barrier()

    timing["comm_ms"] = total_comm_time * 1000
    timing["compute_ms"] = total_compute_time * 1000
    timing["total_ms"] = (total_comm_time + total_compute_time) * 1000

    return S_global, timing

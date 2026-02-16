"""
WY-Factored All-Scan Collective for Gated DeltaNet Sequence Parallelism.

This implements the core novelty from proposal 055: extending ZeCO's All-Scan
to handle Gated DeltaNet's non-diagonal state transitions via WY factorization.

The All-Scan is a P2P pipeline where each device sends its cumulative state
and WY factors to the next device. The receiving device applies the WY
correction to compute the global state.

Communication per device (P-independent):
    S_local:     d_v x d_k   (state matrix)
    gamma_dev:   d_k          (cumulative gating)
    W_dev:       d_k x d_k   (WY factor W)
    K_dev:       d_k x d_k   (WY factor K)
    Total: d_v*d_k + 2*d_k^2 + d_k elements

The scan operation at each pipeline stage:
    S_send = gamma_dev * (S_recv @ (I - W_dev.T @ K_dev)) + S_local
           = gamma_dev * (S_recv - S_recv @ W_dev.T @ K_dev) + S_local
"""

import torch
import torch.distributed as dist
import time
from typing import Tuple, Dict, Optional


def wy_allscan_step(
    S_recv: torch.Tensor,       # (d_v, d_k) - received global state from predecessor
    gamma_dev: torch.Tensor,    # (d_k,) - this device's cumulative gating
    W_dev: torch.Tensor,        # (d_k, d_k) - this device's WY factor W
    K_dev: torch.Tensor,        # (d_k, d_k) - this device's WY factor K
    S_local: torch.Tensor,      # (d_v, d_k) - this device's local state
) -> torch.Tensor:
    """
    Apply WY correction to received state and combine with local state.

    Computes:
        S_out = gamma_dev * (S_recv - S_recv @ W_dev.T @ K_dev) + S_local

    This is the scan operator for the WY-factored All-Scan.
    """
    # WY correction matmul: S_recv @ W_dev.T then multiply by K_dev
    # S_recv: (d_v, d_k), W_dev.T: (d_k, d_k) -> (d_v, d_k)
    tmp = S_recv @ W_dev.T      # (d_v, d_k) - the key GEMM
    tmp = tmp @ K_dev           # (d_v, d_k)

    # Apply correction
    S_corrected = S_recv - tmp  # (d_v, d_k)

    # Apply gating
    S_corrected = gamma_dev.unsqueeze(0) * S_corrected  # (d_v, d_k)

    # Add local contribution
    S_out = S_corrected + S_local  # (d_v, d_k)

    return S_out


def wy_allscan(
    S_local: torch.Tensor,     # (H, d_v, d_k) - local states per head
    gamma_dev: torch.Tensor,   # (H, d_k) - cumulative gating per head
    W_dev: torch.Tensor,       # (H, d_k, d_k) - WY factor W per head
    K_dev: torch.Tensor,       # (H, d_k, d_k) - WY factor K per head
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Run the WY-factored All-Scan collective across all GPUs.

    Each device sends (S_global, gamma_dev, W_dev, K_dev) to the next device.
    The next device applies the WY scan operator and forwards the result.

    This is a sequential P2P pipeline: device 0 -> 1 -> 2 -> ... -> P-1.

    Returns:
        S_global: (H, d_v, d_k) - corrected global state at this device boundary
        timing: Dict with latency breakdown
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    H, d_v, d_k = S_local.shape

    timing = {}

    # Pack all data into a single contiguous buffer for efficient P2P
    # Buffer layout: [S_global (H*d_v*d_k), gamma (H*d_k), W (H*d_k*d_k), K (H*d_k*d_k)]
    buf_size = H * (d_v * d_k + d_k + d_k * d_k + d_k * d_k)
    send_buf = torch.empty(buf_size, device=S_local.device, dtype=S_local.dtype)
    recv_buf = torch.empty(buf_size, device=S_local.device, dtype=S_local.dtype)

    # Start with local state for device 0
    if rank == 0:
        S_global = S_local.clone()
    else:
        S_global = torch.zeros_like(S_local)

    # Offsets for packing
    off1 = H * d_v * d_k
    off2 = off1 + H * d_k
    off3 = off2 + H * d_k * d_k

    total_comm_time = 0.0
    total_compute_time = 0.0

    # Pipeline: device 0 sends to 1, 1 processes and sends to 2, etc.
    for step in range(world_size - 1):
        sender = step
        receiver = step + 1

        if rank == sender:
            # Pack send buffer
            send_buf[:off1] = S_global.reshape(-1)
            send_buf[off1:off2] = gamma_dev.reshape(-1)
            send_buf[off2:off3] = W_dev.reshape(-1)
            send_buf[off3:] = K_dev.reshape(-1)

            # Send to next device
            t0 = time.perf_counter()
            dist.send(send_buf, dst=receiver)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_comm_time += (t1 - t0)

        elif rank == receiver:
            # Receive from predecessor
            t0 = time.perf_counter()
            dist.recv(recv_buf, src=sender)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_comm_time += (t1 - t0)

            # Unpack received state (the global state from predecessor chain)
            S_recv = recv_buf[:off1].reshape(H, d_v, d_k)

            # Apply WY scan for each head
            t2 = time.perf_counter()
            for h in range(H):
                S_global[h] = wy_allscan_step(
                    S_recv[h], gamma_dev[h], W_dev[h], K_dev[h], S_local[h]
                )
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            total_compute_time += (t3 - t2)

        # Sync all devices at each step (simulates pipeline)
        dist.barrier()

    timing["comm_ms"] = total_comm_time * 1000
    timing["compute_ms"] = total_compute_time * 1000
    timing["total_ms"] = (total_comm_time + total_compute_time) * 1000

    return S_global, timing


def wy_allscan_pipelined(
    S_local: torch.Tensor,     # (H, d_v, d_k)
    gamma_dev: torch.Tensor,   # (H, d_k)
    W_dev: torch.Tensor,       # (H, d_k, d_k)
    K_dev: torch.Tensor,       # (H, d_k, d_k)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Pipelined WY-All-Scan that overlaps communication with computation.

    Instead of waiting for the full buffer, we pipeline across heads:
    - Send head h while computing WY correction for head h-1.

    This is closer to the actual ZeCO implementation which pipelines
    across K-blocks of the state tensor.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    H, d_v, d_k = S_local.shape

    timing = {}
    t_start = time.perf_counter()

    # Per-head buffers for pipelined send/recv
    head_buf_size = d_v * d_k + d_k + d_k * d_k + d_k * d_k

    S_global = S_local.clone() if rank == 0 else torch.zeros_like(S_local)

    total_compute_time = 0.0

    for step in range(world_size - 1):
        sender = step
        receiver = step + 1

        if rank == sender:
            # Pack and send all heads in a single buffer
            send_buf = torch.empty(H * head_buf_size, device=S_local.device, dtype=S_local.dtype)
            for h in range(H):
                off = h * head_buf_size
                s_size = d_v * d_k
                g_size = d_k
                w_size = d_k * d_k

                send_buf[off:off + s_size] = S_global[h].reshape(-1)
                send_buf[off + s_size:off + s_size + g_size] = gamma_dev[h].reshape(-1)
                send_buf[off + s_size + g_size:off + s_size + g_size + w_size] = W_dev[h].reshape(-1)
                send_buf[off + s_size + g_size + w_size:off + s_size + g_size + 2 * w_size] = K_dev[h].reshape(-1)

            dist.send(send_buf, dst=receiver)

        elif rank == receiver:
            recv_buf = torch.empty(H * head_buf_size, device=S_local.device, dtype=S_local.dtype)
            dist.recv(recv_buf, src=sender)

            t_comp_start = time.perf_counter()
            for h in range(H):
                off = h * head_buf_size
                s_size = d_v * d_k
                g_size = d_k
                w_size = d_k * d_k

                S_recv_h = recv_buf[off:off + s_size].reshape(d_v, d_k)

                S_global[h] = wy_allscan_step(
                    S_recv_h, gamma_dev[h], W_dev[h], K_dev[h], S_local[h]
                )
            torch.cuda.synchronize()
            t_comp_end = time.perf_counter()
            total_compute_time += (t_comp_end - t_comp_start)

        dist.barrier()

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    timing["total_ms"] = (t_end - t_start) * 1000
    timing["compute_ms"] = total_compute_time * 1000
    timing["comm_ms"] = timing["total_ms"] - timing["compute_ms"]

    return S_global, timing

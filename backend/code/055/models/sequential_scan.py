"""
Sequential (single-device) prefix scan for Gated DeltaNet state transitions.

This serves as the ground-truth reference for numerical verification.
It computes the global state at each device boundary by sequentially
applying each device's transition.

From the proposal:
  S_{pL} = F_dev^(p) * S_{(p-1)L} + S_local^(p)

where F_dev^(p) = diag(gamma_dev^(p)) * (I - W_dev^(p).T @ K_dev^(p))
"""

import torch
from typing import List, Tuple


def sequential_prefix_scan(
    states_local: List[torch.Tensor],   # [P] x (d_v, d_k) - local final states per device
    gamma_devs: List[torch.Tensor],     # [P] x (d_k,) - cumulative gating per device
    W_devs: List[torch.Tensor],         # [P] x (d_k, d_k) - WY factor W per device
    K_devs: List[torch.Tensor],         # [P] x (d_k, d_k) - WY factor K per device
) -> List[torch.Tensor]:
    """
    Compute global prefix scan states sequentially on a single device.

    Returns the corrected state at each device boundary.
    This is the ground truth for verifying distributed implementations.

    The recurrence is:
        S_global[0] = S_local[0]
        S_global[p] = gamma_dev[p] * (S_global[p-1] @ P_dev[p]) + S_local[p]

    where P_dev[p] = I - W_dev[p].T @ K_dev[p]
    """
    P = len(states_local)
    d_v, d_k = states_local[0].shape

    # Device 0: no predecessor, so global state = local state
    global_states = [states_local[0].clone()]

    for p in range(1, P):
        S_prev = global_states[p - 1]  # (d_v, d_k)
        gamma = gamma_devs[p]          # (d_k,)
        W = W_devs[p]                  # (d_k, d_k)
        K = K_devs[p]                  # (d_k, d_k)
        S_local = states_local[p]      # (d_v, d_k)

        # Apply WY correction: S_prev @ (I - W.T @ K)
        # = S_prev - S_prev @ W.T @ K
        # = S_prev - (S_prev @ W.T) @ K
        WtK = W.T @ K                   # (d_k, d_k)
        S_corrected = S_prev - S_prev @ WtK  # (d_v, d_k)

        # Apply gating: gamma * S_corrected
        S_corrected = gamma.unsqueeze(0) * S_corrected  # (d_v, d_k)

        # Add local contribution
        S_global = S_corrected + S_local  # (d_v, d_k)

        global_states.append(S_global)

    return global_states


def sequential_prefix_scan_diagonal(
    states_local: List[torch.Tensor],   # [P] x (d_v, d_k) - local final states per device
    gamma_devs: List[torch.Tensor],     # [P] x (d_k,) - cumulative gating per device
) -> List[torch.Tensor]:
    """
    Compute global prefix scan for diagonal-only transitions (GLA/Mamba-2).

    The recurrence is:
        S_global[0] = S_local[0]
        S_global[p] = gamma_dev[p] * S_global[p-1] + S_local[p]

    No WY correction needed - just elementwise multiplication by gamma.
    This is the ZeCO case.
    """
    P = len(states_local)
    global_states = [states_local[0].clone()]

    for p in range(1, P):
        S_prev = global_states[p - 1]
        gamma = gamma_devs[p]
        S_local = states_local[p]

        # Diagonal-only: elementwise multiply
        S_corrected = gamma.unsqueeze(0) * S_prev  # (d_v, d_k)
        S_global = S_corrected + S_local

        global_states.append(S_global)

    return global_states

"""
Naive Sequential DeltaNet Forward Pass

DeltaNet recurrence (S_0 = I):
    S_t = S_{t-1} + beta_t * k_t * (v_t - S_{t-1}^T k_t)^T

Output: o_t = S_t @ q_t (before update at step t)

This processes one token at a time: O(T * d^2) sequential work.
"""

import torch


def naive_wy_forward(K: torch.Tensor, V: torch.Tensor, beta: torch.Tensor,
                     Q: torch.Tensor = None) -> torch.Tensor:
    """
    Standard sequential DeltaNet forward pass.

    Args:
        K: (T, d) — key vectors (L2-normalized)
        V: (T, d) — value vectors
        beta: (T,) — learning rates in (0, 1)
        Q: (T, d) — query vectors (defaults to V)

    Returns:
        outputs: (T, d)
    """
    T, d = K.shape
    if Q is None:
        Q = V

    S = torch.eye(d, device=K.device, dtype=K.dtype)
    outputs = torch.empty(T, d, device=K.device, dtype=K.dtype)

    for t in range(T):
        outputs[t] = S @ Q[t]
        delta = V[t] - S.T @ K[t]
        S.add_(beta[t] * torch.outer(K[t], delta))

    return outputs

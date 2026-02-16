"""
Prefix Scan (Hillis-Steele inclusive) for dense SSM recurrence.

Computes h_t = A_t h_{t-1} + b_t via the associative operator:
    (A_i, b_i) . (A_j, b_j) = (A_i @ A_j, A_i @ b_j + b_i)

Work: O(T n^3 log T)  |  Depth: O(log T)

Reference: Proposal 026, Background.
"""
import torch


def sequential_scan(A, b, h0=None):
    """Ground truth sequential scan. Work: O(Tn^2), Depth: O(T)."""
    T, n, _ = A.shape
    h = torch.zeros(T, n, device=A.device, dtype=A.dtype)
    h_prev = h0 if h0 is not None else torch.zeros(n, device=A.device, dtype=A.dtype)
    for t in range(T):
        h[t] = A[t] @ h_prev + b[t]
        h_prev = h[t]
    return h


def prefix_scan(A, b, h0=None):
    """
    Hillis-Steele inclusive prefix scan.
    Returns (h, gemm_count).
    """
    T, n, _ = A.shape
    gemm_count = 0

    A_s = A.clone()
    b_s = b.clone()

    if h0 is not None:
        b_s[0] = A_s[0] @ h0 + b_s[0]
        A_s[0] = torch.eye(n, device=A.device, dtype=A.dtype)
        gemm_count += 1

    stride = 1
    while stride < T:
        count = T - stride
        if count <= 0:
            stride *= 2
            continue

        idx = torch.arange(stride, T, device=A.device)
        src = idx - stride

        Ar = A_s[idx]
        Al = A_s[src]
        br = b_s[idx]
        bl = b_s[src]

        A_new = torch.bmm(Ar, Al)
        gemm_count += count
        b_new = torch.bmm(Ar, bl.unsqueeze(-1)).squeeze(-1) + br
        gemm_count += count

        A_s = A_s.clone()
        b_s = b_s.clone()
        A_s[idx] = A_new
        b_s[idx] = b_new

        stride *= 2

    return b_s[:T], gemm_count

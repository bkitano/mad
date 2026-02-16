"""
Cyclic Reduction for dense SSM recurrence (optimized, no Python loops).

Computes h_t = A_t h_{t-1} + b_t by solving the block-bidiagonal system
via recursive even/odd elimination.

Forward elimination:
    A_tilde[j] = A[2j+1] @ A[2j]
    b_tilde[j] = A[2j+1] @ b[2j] + b[2j+1]

Back-substitution:
    h_{2j} = A_{2j} @ h_{2j-1} + b_{2j}

Work: O(T n^3) -- geometric series  |  Depth: O(log T)

Reference: Proposal 026, Mathematical Formulation.
"""
import torch


def cyclic_reduction(A, b, h0=None):
    """
    Cyclic reduction for dense SSM recurrence.
    Fully vectorized — no Python loops except over log(T) levels.
    Returns (h, gemm_count).
    """
    T, n, _ = A.shape
    device, dtype = A.device, A.dtype
    gemm_count = 0

    A_w = A.clone()
    b_w = b.clone()

    if h0 is not None:
        b_w[0] = A_w[0] @ h0 + b_w[0]
        A_w[0] = torch.eye(n, device=device, dtype=dtype)
        gemm_count += 1

    # Phase 1: Forward Elimination
    levels = []
    cur_idx = torch.arange(T, device=device)
    cur_A = A_w
    cur_b = b_w

    while len(cur_idx) > 1:
        Tc = len(cur_idx)
        Th = Tc // 2
        has_rem = (Tc % 2 == 1)

        ep = torch.arange(0, 2 * Th, 2, device=device)
        op = torch.arange(1, 2 * Th + 1, 2, device=device)

        Ae = cur_A[ep]
        be = cur_b[ep]
        Ao = cur_A[op]
        bo = cur_b[op]

        # Store for back-substitution: we need the original A and b at even
        # positions, and the predecessor index mapping.
        # Predecessor of even[j] in the original sequence is at position ep[j]-1
        # in the current level's index array.
        # We store the original indices directly.
        pred_orig = torch.zeros(Th, device=device, dtype=torch.long)
        # For ep[j] == 0, predecessor doesn't exist (use h0=0)
        # For ep[j] > 0, predecessor is cur_idx[ep[j] - 1]
        mask = ep > 0
        pred_orig[mask] = cur_idx[ep[mask] - 1]
        # For ep[j] == 0, pred_orig stays 0 (will use zero state)

        levels.append({
            'Ae': Ae, 'be': be,
            'e_orig': cur_idx[ep],
            'pred_orig': pred_orig,
            'first_is_zero': (ep[0] == 0).item() if Th > 0 else False,
        })

        # Elimination: A_red = A_odd @ A_even, b_red = A_odd @ b_even + b_odd
        A_red = torch.bmm(Ao, Ae)
        gemm_count += Th
        b_red = torch.bmm(Ao, be.unsqueeze(-1)).squeeze(-1) + bo
        gemm_count += Th

        if has_rem:
            cur_A = torch.cat([A_red, cur_A[Tc - 1:Tc]], dim=0)
            cur_b = torch.cat([b_red, cur_b[Tc - 1:Tc]], dim=0)
            cur_idx = torch.cat([cur_idx[op], cur_idx[Tc - 1:Tc]])
        else:
            cur_A = A_red
            cur_b = b_red
            cur_idx = cur_idx[op]

    # Base case
    h = torch.zeros(T, n, device=device, dtype=dtype)
    h[cur_idx[0]] = cur_b[0]

    # Phase 2: Back-Substitution (fully vectorized)
    for lev in reversed(levels):
        Ae = lev['Ae']
        be = lev['be']
        e_orig = lev['e_orig']
        pred_orig = lev['pred_orig']
        first_zero = lev['first_is_zero']
        Th = len(e_orig)

        # Gather predecessor hidden states using vectorized indexing
        h_prev = h[pred_orig]  # (Th, n)
        # Zero out the first element if its predecessor is h0=0
        if first_zero:
            h_prev[0] = 0.0

        # h_even = A_even @ h_prev + b_even  (batched matmul)
        h_even = torch.bmm(Ae, h_prev.unsqueeze(-1)).squeeze(-1) + be
        gemm_count += Th

        # Scatter results back (vectorized)
        h[e_orig] = h_even

    return h, gemm_count

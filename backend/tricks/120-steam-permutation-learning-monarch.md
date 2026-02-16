# 120: STEAM: STE-based Permutation Learning in Monarch Factorization

**Category**: decomposition
**Gain type**: expressivity
**Source**: Mohamed, Emiya, Chaux (ICASSP 2025)
**Paper**: [papers/learning-permutations-monarch-factorization.pdf]
**Documented**: 2026-02-15

## Description

Standard Monarch matrices factorize a dense $N \times N$ matrix as $M = P_2 L P_1 R P_0$, where $L, R$ are block-diagonal and $P_0, P_1, P_2$ are **fixed** permutations (typically the bit-reversal stride permutation $\bar{P}$). This trick makes the outer permutations $P_0$ and $P_2$ **learnable** while keeping $P_1 = \bar{P}$ fixed, increasing the expressivity of the factorization without changing its inference-time structure.

The key challenge is that optimizing over the symmetric group $\mathcal{P}_N$ is a combinatorial problem. STEAM (STE Alternate minimization for Monarch) solves this by introducing dense "cost" matrices $\mathcal{X}_0, \mathcal{X}_2 \in \mathbb{R}^{N \times N}$ and a surjective mapping $h: \mathbb{R}^{N \times N} \to \mathcal{P}_N$ (the auction/linear assignment solver). Gradients flow through the non-differentiable $h$ via the Straight-Through Estimator (STE): during backpropagation, $\frac{\partial h}{\partial \mathcal{X}} \approx I$, so the update rule accumulates true gradients $\frac{\partial F}{\partial P_i}$ directly into the cost matrices $\mathcal{X}_i$. This accumulated gradient history enables exploration of the permutation space and escape from local minima.

The connection to permutation-augmented structured sparsity (PA-DST) is direct: both methods recognize that **fixed coordinate alignments** in structured matrices limit expressivity, and that a single learned permutation per factor can recover it. STEAM provides an alternative to Sinkhorn/Birkhoff relaxation: instead of softening the permutation constraint, it maintains hard permutations throughout and navigates the discrete space via STE-guided gradient accumulation.

## Mathematical Form

**Core Factorization:**

$$
M = P_2 L \bar{P} R P_0
$$

where:
- $L, R \in \mathcal{BD}^{(N)}$ are block-diagonal matrices, each composed of $n$ blocks of size $n \times n$, with $n^2 = N$
- $\bar{P}$ is the fixed stride (bit-reversal) permutation
- $P_0, P_2 \in \mathcal{P}_N$ are **learned** permutation matrices

**Optimization Problem:**

$$
\min_{L, R \in \mathcal{BD}^{(N)},\; P_0, P_2 \in \mathcal{P}_N} \frac{1}{2} \|P_2 L \bar{P} R P_0 - A\|_F^2
$$

**Key Reformulation (Proposition 1):** Since $P_0, P_2, \bar{P}$ are orthogonal, the problem decouples into:

$$
\min_{P_0, P_2 \in \mathcal{P}_N} F(P_0, P_2), \quad F(P_0, P_2) = \frac{1}{2} \|p_{\mathcal{M}}(\bar{P} P_2^T A P_0^T)\|_F^2
$$

where $p_{\mathcal{M}}(\cdot)$ is the closed-form Monarch projection (block SVDs). This separates permutation search from block-diagonal fitting.

**STE Gradient (used in backpropagation):**

$$
\frac{\partial F}{\partial P_i} = \begin{cases} (P_2 L \bar{P} R)^T (P_2 L \bar{P} R P_0 - A) & \text{if } i = 0 \\ (P_2 L \bar{P} R P_0 - A)(L \bar{P} R P_0)^T & \text{if } i = 2 \end{cases}
$$

**STEAM Update Rule:**

For each $i \in \{0, 2\}$:
1. Compute step size: $\eta = \frac{1}{\alpha \|L \bar{P} R\|_2^2}$ (Lipschitz-controlled)
2. Accumulate gradient: $\mathcal{X}_i \leftarrow \mathcal{X}_i - \eta \frac{\partial F}{\partial P_i}$
3. Solve assignment: $P_i \leftarrow h(\mathcal{X}_i)$ (auction algorithm on cost matrix)
4. Update block-diagonals: $(L, R) \leftarrow \pi_{\mathcal{M}}(\bar{P} P_2^T A P_0^T)$

The algorithm returns the best $(P_0, P_2, L, R)$ encountered across all $T$ iterations.

**End-to-End Neural Network Training:**

When used as a learnable layer (no target matrix $A$), the STE principle backpropagates task loss gradients through $h$:

$$
\text{input} \xrightarrow{\times P_0} \xrightarrow{\times R} \xrightarrow{\times \bar{P}} \xrightarrow{\times L} \xrightarrow{\times P_2} \text{output}
$$

The cost matrices $\mathcal{X}_0, \mathcal{X}_2$ accumulate task-loss gradients, and the auction algorithm extracts hard permutations at each step.

## Complexity

| Operation | Fixed Monarch | STEAM (per iteration) |
|-----------|--------------|----------------------|
| Mat-vec (inference) | $O(N^{3/2})$ | $O(N^{3/2})$ (same structure) |
| Monarch projection $\pi_{\mathcal{M}}$ | $O(N^{5/2})$ | $O(N^{5/2})$ |
| Gradient w.r.t. $P_i$ | N/A | $O(N^{5/2})$ (block-diagonal rearrangement) |
| Assignment solver $h$ | N/A | $O(N^2)$ best-case, $O(N^{5/2} \log NC)$ worst-case |
| Parameters (inference) | $2N^{3/2}$ | $2N^{3/2} + N$ (permutation index array) |
| Parameters (training) | $2N^{3/2}$ | $2N^{3/2} + 2N^2$ (cost matrices $\mathcal{X}_i$) |

**Memory:** Training adds two $N \times N$ dense cost matrices (the $\mathcal{X}_i$). At inference, only an integer index array of size $N$ is stored per learned permutation.

**Empirical:** On MNIST with $N = 784$, dense layer has 614,656 params; Monarch has 43,904; STEAM has 43,904 at inference + 44,688 during training. STEAM achieves 97.49% accuracy vs. Monarch's 92.92% (AdamW) — recovering most of the dense model's 96.35%.

## Applicability

- **Monarch-based layers in transformers and MLPs:** Drop-in enhancement for any Monarch linear layer. The learned permutations increase expressivity from the fixed stride permutation, recovering accuracy lost to the structured factorization.
- **Sparse matrix factorization / approximation:** When approximating a dense matrix with Monarch structure, STEAM reduces relative Frobenius error by 10-40% compared to fixed-permutation Monarch projection, especially in lower dimensions ($N \leq 100$).
- **Structured SSM transitions:** State transition matrices parameterized as Monarch (e.g., in Monarch Mixer) can benefit from learned permutations to expand the reachable set of linear maps.
- **Connection to PA-DST:** STEAM provides a complementary approach to the Birkhoff/Sinkhorn relaxation used in PA-DST. While PA-DST learns soft doubly-stochastic matrices with a penalty driving them to permutations, STEAM maintains hard permutations and uses STE + auction to explore the discrete space. This avoids the $O(N^2)$ soft matrix overhead during forward pass at the cost of solving an assignment problem.

## Limitations

- The cost matrices $\mathcal{X}_i$ introduce $O(N^2)$ training memory, partially negating the sub-quadratic advantage of Monarch
- The objective is non-monotone: the best solution is tracked across iterations, and the algorithm may not converge to a fixed point
- Learning two permutations simultaneously ($P_0$ and $P_2$) is significantly harder than learning one — the joint combinatorial space is $|\mathcal{P}_N|^2 = (N!)^2$
- Auction algorithm warm-starting helps empirically but has no theoretical convergence guarantees for the outer loop
- Tested only on moderate dimensions ($N \leq 100$ for factorization, $N = 784$ for end-to-end); scalability to LLM-scale dimensions is an open question
- Higher-order Monarch ($p > 2$ factors) with learned permutations remains unexplored

## Implementation Notes

```python
import torch
from scipy.optimize import linear_sum_assignment

def auction_assignment(cost_matrix):
    """
    Solve linear assignment problem (surjective h mapping).
    Uses scipy as fallback; production code uses auction algorithm
    with warm-starting from previous iteration.
    """
    # Negate because linear_sum_assignment minimizes, but STEAM
    # accumulates gradients (we want argmin of cost)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    P = torch.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1.0
    return P

def steam_iteration(A, X0, X2, P0, P2, P_bar, alpha=1.001):
    """One iteration of STEAM algorithm."""
    for i in [0, 2]:
        # Monarch projection: get L, R from rotated target
        A_rot = P_bar @ P2.T @ A @ P0.T
        L, R = monarch_projection(A_rot)  # closed-form block SVDs

        # Compute residual
        M_hat = P2 @ L @ P_bar @ R @ P0
        residual = M_hat - A

        # Gradient w.r.t. P_i (STE: pass through h)
        if i == 0:
            grad = (P2 @ L @ P_bar @ R).T @ residual
        else:  # i == 2
            grad = residual @ (L @ P_bar @ R @ P0).T

        # Lipschitz step size
        eta = 1.0 / (alpha * torch.linalg.norm(L @ P_bar @ R, ord=2) ** 2)

        # Accumulate gradient into cost matrix (STE update)
        if i == 0:
            X0 = X0 - eta * grad
            P0 = auction_assignment(X0)
        else:
            X2 = X2 - eta * grad
            P2 = auction_assignment(X2)

    return X0, X2, P0, P2, L, R

def steam_forward_train(L_blocks, R_blocks, P0, P2, P_bar, x):
    """Forward pass with learned permutations (training)."""
    x = P0 @ x           # Learned input permutation
    x = blockdiag_mv(R_blocks, x)  # Block-diagonal R
    x = P_bar @ x        # Fixed stride permutation
    x = blockdiag_mv(L_blocks, x)  # Block-diagonal L
    x = P2 @ x           # Learned output permutation
    return x

def steam_forward_infer(L_blocks, R_blocks, perm0, perm2, x, p, q):
    """Inference: permutations absorbed as index remapping."""
    x = x[..., perm0]    # Free re-indexing (P0)
    x = x.reshape(-1, q, p)
    x = torch.bmm(R_blocks, x.unsqueeze(-1)).squeeze(-1)
    x = x.reshape(-1, p, q).transpose(-2, -1).reshape(-1, p * q)
    x = torch.bmm(L_blocks, x.reshape(-1, p, q).unsqueeze(-1)).squeeze(-1)
    x = x.reshape(-1, q, p).transpose(-2, -1).reshape(-1, p * q)
    x = x[..., perm2]    # Free re-indexing (P2)
    return x
```

## References

- Mohamed, M., Emiya, V., and Chaux, C. (2025). Learning Permutations in Monarch Factorization. ICASSP 2025. DOI: 10.1109/ICASSP49660.2025.10889798. Code: https://gitlab.lis-lab.fr/valentin.emiya/STEAM
- Mohamed, M., Malgouyres, F., Emiya, V., and Chaux, C. (2024). Straight-Through Meets Sparse Recovery: The Support Exploration Algorithm. ICML 2024.
- Dao, T., et al. (2022). Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML 2022.
- Fu, S., et al. (2024). Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture. NeurIPS 2024.
- Bertsekas, D. P. (1988). The Auction Algorithm: A Distributed Relaxation Method for the Assignment Problem. Annals of Operations Research.

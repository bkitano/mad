# 034: Cyclic Reduction for Block-Tridiagonal Systems

**Category**: parallelization
**Gain type**: efficiency
**Source**: Hockney (1965); Golub & Hockney (1965); Buneman (1969); reviewed by Gander & Golub (1997), Neuenhofen (2018)
**Paper**: [papers/cyclic-reduction-block-tridiagonal.pdf]
**Documented**: 2026-02-15

## Description

Cyclic reduction is a divide-and-conquer algorithm for solving block-tridiagonal linear systems that achieves $O(\log N)$ parallel depth by recursively eliminating alternating (odd/even) block rows via Schur complements. The core insight is that a block-tridiagonal system of $N$ block equations can be reordered into odd and even indices, yielding two independent half-sized block-tridiagonal systems whose coefficient matrices are themselves Schur complements of the original. Since each reduced system retains block-tridiagonal structure, the process recurses until the system is small enough for direct solution (e.g., Cholesky). All block eliminations at a given recursion level are independent and can be computed in parallel.

This is the block-matrix generalization of the scalar cyclic reduction used in Poisson solvers, and it is closely related to the parallel prefix (parallel scan) approach: both achieve logarithmic depth, but cyclic reduction typically requires fewer total operations. For Hermitian positive definite (HPD) systems, the reduced systems are guaranteed to also be HPD (since Schur complements of HPD matrices are HPD), ensuring numerical stability without pivoting. The connection to neural network sequence models is direct: the recurrence $h_t = A_t h_{t-1} + B_t x_t$ in linear state-space models and linear RNNs, when written as a global system, yields a block-bidiagonal (or block-tridiagonal for boundary conditions) matrix that cyclic reduction can solve in $O(\log T)$ parallel depth.

## Mathematical Form

**Block-Tridiagonal System:**

$$
\underline{A} \cdot \mathbf{x} = \mathbf{y}, \qquad \underline{A} = \begin{pmatrix} A_1 & B_1^H \\ B_1 & A_2 & B_2^H \\ & B_2 & A_3 & \ddots \\ & & \ddots & \ddots & B_{N-1}^H \\ & & & B_{N-1} & A_N \end{pmatrix}
$$

where $A_j \in \mathbb{C}^{m \times m}$ are diagonal blocks, $B_j \in \mathbb{C}^{m \times m}$ are sub-diagonal blocks, and $\underline{A} \in \mathbb{C}^{Nm \times Nm}$ is Hermitian positive definite.

**Odd-Even Reordering:**

Permute block rows/columns into odd indices $(1, 3, 5, \ldots, N-1)$ then even indices $(2, 4, 6, \ldots, N)$:

$$
\underline{A}_\pi = \begin{pmatrix} \underline{D}_1 & \underline{C}^H \\ \underline{C} & \underline{D}_2 \end{pmatrix}
$$

where $\underline{D}_1$ contains the odd-indexed diagonal blocks, $\underline{D}_2$ contains the even-indexed diagonal blocks, and $\underline{C}$ captures the coupling.

**Schur Complement Reduction:**

Eliminating the even unknowns yields the odd system:

$$
\underline{U} \cdot \mathbf{x}_o = \mathbf{u}, \qquad \underline{U} = \underline{D}_1 - \underline{C}^H \underline{D}_2^{-1} \underline{C}
$$

Eliminating the odd unknowns yields the even system:

$$
\underline{V} \cdot \mathbf{x}_e = \mathbf{v}, \qquad \underline{V} = \underline{D}_2 - \underline{C} \underline{D}_1^{-1} \underline{C}^H
$$

Both $\underline{U}$ and $\underline{V}$ are block-tridiagonal of dimension $(N/2) \cdot m$.

**Explicit Block Formulas (for the odd system $\underline{U}$):**

The blocks of the reduced system $\underline{U}$ are, for $j = 1, 2, \ldots, N/2$:

$$
U_j = A_{2j-1} - B_{2j-2} A_{2j-2}^{-1} B_{2j-2}^H - B_{2j-1}^H A_{2j}^{-1} B_{2j-1}
$$

$$
E_j = -B_{2j} A_{2j}^{-1} B_{2j-1}
$$

$$
\mathbf{u}_j = \mathbf{y}_{2j-1} - B_{2j-2} A_{2j-2}^{-1} \mathbf{y}_{2j-2} - B_{2j-1}^H A_{2j}^{-1} \mathbf{y}_{2j}
$$

with boundary conventions $B_0 := \mathbf{0}$, $A_0 := I$, $B_N := \mathbf{0}$, $A_{N+1} := I$.

**Key Property (HPD preservation):**

If $\underline{A}$ is Hermitian positive definite, then both $\underline{U}$ and $\underline{V}$ are Hermitian positive definite (they are Schur complements of a positive definite matrix).

**Recursive Structure:**

At recursion level $l$ (starting from $l = 0$):
- System size: $N / 2^l$ blocks of size $m \times m$
- After $\lceil \log_2 N \rceil$ levels: a single $m \times m$ system solved directly

## Complexity

| Operation | Sequential (Thomas) | Cyclic Reduction (Parallel) |
|-----------|--------------------|-----------------------------|
| Solve $Nm \times Nm$ block-tridiagonal | $O(N m^3)$ sequential | $O(N m^3)$ total work |
| Parallel depth | $O(N)$ steps | $O(\log N)$ steps |
| Per-level work | — | $O(N/2^l)$ independent $m \times m$ inversions |
| Per-level parallel time | — | $O(m^3)$ per level |
| Total parallel time | $O(Nm^3)$ | $O(m^3 \log N)$ |

**Memory:** $O(Nm^2)$ — at each recursion level, the $N/2^l$ new blocks $U_j, E_j, \mathbf{u}_j$ must be stored. Total storage across all levels is $O(Nm^2 \sum_{l=0}^{\log N} 2^{-l}) = O(Nm^2)$.

**Communication:** Each block reduction at level $l$ requires data from 3 neighboring blocks at level $l-1$. On distributed systems, this involves nearest-neighbor communication with stride $2^l$, yielding $O(\log N)$ communication rounds.

## Applicability

- **Parallel linear RNN / SSM inference and training**: The recurrence $h_t = A_t h_{t-1} + b_t$ written as a block-bidiagonal system $\underline{A} \mathbf{h} = \mathbf{b}$ can be solved via cyclic reduction in $O(\log T)$ depth — this is the same parallelization exploited by parallel scan (prefix sum), but cyclic reduction uses fewer total operations when $m$ is large
- **Chunkwise parallel algorithms**: Modern SSM training (e.g., Mamba, DeltaNet) uses chunkwise computation where intra-chunk is quadratic attention and inter-chunk is a linear recurrence — cyclic reduction provides an alternative to prefix scan for the inter-chunk recurrence
- **Implicit time-stepping in physics-informed networks**: Solving the block-tridiagonal system arising from implicit Euler or Crank-Nicolson discretization of PDEs in time
- **Batched block-tridiagonal solves on GPU**: Recent work (arXiv:2509.03015) shows that recursive Schur-complement reduction combined with batched BLAS achieves high GPU utilization for block-tridiagonal SPD systems
- **Poisson solvers and spectral methods**: The original application — solving discretized Poisson/Helmholtz equations where the coefficient matrix is block-tridiagonal with Toeplitz structure

## Limitations

- **Requires block-tridiagonal structure**: Does not apply to general sparse or dense systems; the system must have bandwidth 1 in block terms (or be reducible to such form)
- **Communication overhead**: On distributed systems, the communication stride doubles at each level, leading to non-local data movement that can dominate runtime on networks with high latency
- **Fill-in for non-HPD systems**: For indefinite or non-symmetric systems, the Schur complement blocks can have larger norms than the originals, leading to numerical instability; pivoting strategies are needed
- **Block size sensitivity**: When $m$ is very small (scalar tridiagonal, $m = 1$), the parallel cyclic reduction (PCR) or hybrid Thomas-PCR algorithms are preferred due to lower overhead; cyclic reduction excels when $m$ is moderately large
- **Compared to parallel scan**: Parallel prefix scan achieves the same $O(\log N)$ depth with simpler implementation (associative binary operator) but uses $O(N \log N)$ total work vs. $O(N)$ for cyclic reduction

## Implementation Notes

```python
import numpy as np

def cyclic_reduction_solve(A_blocks, B_blocks, y_blocks):
    """
    Solve a Hermitian positive definite block-tridiagonal system
    using cyclic reduction via Schur complement elimination.

    A_blocks: list of N diagonal blocks, each m x m
    B_blocks: list of N-1 sub-diagonal blocks, each m x m
    y_blocks: list of N right-hand-side blocks, each m x k

    Returns: list of N solution blocks x_j, each m x k
    """
    N = len(A_blocks)
    m = A_blocks[0].shape[0]

    if N == 1:
        # Base case: direct solve
        return [np.linalg.solve(A_blocks[0], y_blocks[0])]

    if N % 2 != 0:
        raise ValueError("N must be a power of 2 for simple cyclic reduction")

    # Phase 1: Compute reduced odd and even systems (all j independent, parallelizable)
    # Odd system blocks: indices 0, 2, 4, ... (original 1, 3, 5, ...)
    N_half = N // 2

    # Precompute inverses of even-indexed diagonal blocks (parallel)
    # A_0^{-1} = I (boundary), A_{N+1}^{-1} = I (boundary)
    A_inv = {}
    for j in range(N):
        A_inv[j] = np.linalg.inv(A_blocks[j])

    # Build odd system: U_j, E_j, u_j for j = 1, ..., N/2
    U_diag = []  # diagonal blocks of reduced system
    U_sub = []   # sub-diagonal blocks of reduced system
    u_rhs = []   # right-hand sides of reduced system

    for j in range(N_half):
        idx = 2 * j  # odd index in 0-based: 0, 2, 4, ...
        # U_j = A_{2j} - B_{2j-1}^H A_{2j-1}^{-1} B_{2j-1} - B_{2j}^H A_{2j+1}^{-1} B_{2j}
        Uj = A_blocks[idx].copy()
        if idx > 0:
            Uj -= B_blocks[idx - 1].conj().T @ A_inv[idx - 1] @ B_blocks[idx - 1]
        if idx < N - 1:
            Uj -= B_blocks[idx].conj().T @ A_inv[idx + 1] @ B_blocks[idx]
        U_diag.append(Uj)

        # E_j = -B_{2j+1} A_{2j+1}^{-1} B_{2j} (sub-diagonal of reduced system)
        if j < N_half - 1:
            idx_next = idx + 2
            Ej = -B_blocks[idx + 1] @ A_inv[idx + 1] @ B_blocks[idx]
            U_sub.append(Ej)

        # u_j = y_{2j} - B_{2j-1}^H A_{2j-1}^{-1} y_{2j-1} - B_{2j}^H A_{2j+1}^{-1} y_{2j+1}
        uj = y_blocks[idx].copy()
        if idx > 0:
            uj -= B_blocks[idx - 1].conj().T @ A_inv[idx - 1] @ y_blocks[idx - 1]
        if idx < N - 1:
            uj -= B_blocks[idx].conj().T @ A_inv[idx + 1] @ y_blocks[idx + 1]
        u_rhs.append(uj)

    # Phase 2: Recursively solve the reduced (N/2)-sized system
    x_odd = cyclic_reduction_solve(U_diag, U_sub, u_rhs)

    # Phase 3: Back-substitute to recover even-indexed unknowns
    # x_{2j+1} = A_{2j+1}^{-1} (y_{2j+1} - B_{2j}^H x_{2j} - B_{2j+1} x_{2j+2})
    x_all = [None] * N
    for j in range(N_half):
        x_all[2 * j] = x_odd[j]

    for j in range(N_half):
        idx = 2 * j + 1  # even original index (1-based), 0-based: 1, 3, 5, ...
        if idx < N:
            rhs = y_blocks[idx].copy()
            if idx > 0:
                rhs -= B_blocks[idx - 1] @ x_all[idx - 1]
            if idx < N - 1:
                rhs -= B_blocks[idx].conj().T @ x_all[idx + 1]
            x_all[idx] = A_inv[idx] @ rhs

    return x_all


# Connection to parallel scan for linear SSM recurrence:
# h_t = A_t h_{t-1} + b_t  =>  block-bidiagonal system
#   [[I,        ],   [h_1]   [b_1        ]
#    [-A_2, I,  ],   [h_2] = [b_2        ]
#    [    -A_3, I]]  [h_3]   [b_3        ]
#
# This is lower block-bidiagonal (bandwidth 1), solvable by
# cyclic reduction in O(log T) parallel depth with O(T m^2) work.
# Compare: parallel scan uses O(T m^3 log T) work due to
# associative operator requiring m x m matrix multiplications.
```

## References

- Hockney, R. W. (1965). A fast direct solution of Poisson's equation using Fourier analysis. *JACM*, 12(1), 95–113.
- Buneman, O. (1969). A compact non-iterative Poisson solver. *Report 294*, Stanford University Institute for Plasma Research.
- Gander, W. & Golub, G. H. (1997). Cyclic Reduction — History and Applications. *Proceedings of the Workshop on Scientific Computing*, Springer.
- Neuenhofen, M. P. (2018). Review of Cyclic Reduction for Parallel Solution of Hermitian Positive Definite Block-Tridiagonal Linear Systems. arXiv:1807.00370.
- Polizzi, E. & Sameh, A. (2006). A parallel hybrid banded system solver: the SPIKE algorithm. *Parallel Computing*, 32(2), 177–194.
- Bini, D. A., Meini, B. (2009). The cyclic reduction algorithm: from Poisson equation to stochastic processes and beyond. *Numerical Algorithms*, 51, 23–60.
- Khabou, A. et al. (2025). Harnessing Batched BLAS/LAPACK Kernels on GPUs for Parallel Solutions of Block Tridiagonal Systems. arXiv:2509.03015.

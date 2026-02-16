# 159: Recursive WY Merge for Blocked Householder Accumulation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Elmroth & Gustavson (2000), IBM J. Res. Develop. 44(4)
**Paper**: [papers/recursive-blocked-qr-elmroth-gustavson.pdf]
**Documented**: 2026-02-15

## Description

When accumulating a product of $k$ Householder reflections into the compact WY form $Q = I - YTY^\top$, the standard LAPACK approach (DLARFT) builds the upper-triangular $T$ matrix column-by-column via a **bordering** procedure — each new column requires a matrix-vector product (Level-2 BLAS), creating a sequential bottleneck. The recursive WY merge trick replaces this with a **binary-tree divide-and-conquer** that builds $T$ entirely from **matrix-matrix multiplications** (Level-3 BLAS / tensor-core-friendly GEMMs).

The key insight: given two compact WY representations $Q_1 = I - Y_1 T_1 Y_1^\top$ and $Q_2 = I - Y_2 T_2 Y_2^\top$, their product $Q = Q_1 Q_2$ has the compact WY form $Q = I - Y T Y^\top$ where $Y = (Y_1, Y_2)$ and $T$ is a $2 \times 2$ block upper-triangular matrix with a **single off-diagonal block** computed via one GEMM. Applying this recursively over $\log_2 k$ levels converts all of the $T$-construction into GEMMs.

This is the algorithm behind LAPACK's `DGEQRT3` routine and is directly applicable to constructing orthogonal transition matrices in neural networks (orthogonal RNNs, DeltaNet, DeltaProduct) where $Q$ is parameterized as a product of Householder reflections.

## Mathematical Form

**Core Merge Formula:**

Given two compact WY factors:

$$
Q_1 = I - Y_1 T_1 Y_1^\top, \quad Q_2 = I - Y_2 T_2 Y_2^\top
$$

their product is:

$$
Q = Q_1 Q_2 = I - Y T Y^\top
$$

where $Y = (Y_1, Y_2)$ is formed by concatenation and:

$$
T = \begin{pmatrix} T_1 & T_3 \\ 0 & T_2 \end{pmatrix}
$$

with the off-diagonal block:

$$
T_3 = -T_1 (Y_1^\top Y_2) T_2
$$

**Key Definitions:**

- $Y_1 \in \mathbb{R}^{m \times k_1}$, $Y_2 \in \mathbb{R}^{m \times k_2}$ — unit lower-trapezoidal Householder vector matrices
- $T_1 \in \mathbb{R}^{k_1 \times k_1}$, $T_2 \in \mathbb{R}^{k_2 \times k_2}$ — upper-triangular $T$ factors
- $k = k_1 + k_2$ — total number of reflections
- $m$ — dimension of the Householder vectors (state dimension)

**Derivation:**

Starting from $Q = Q_1 Q_2$:

$$
Q = (I - Y_1 T_1 Y_1^\top)(I - Y_2 T_2 Y_2^\top)
$$

$$
= I - Y_1 T_1 Y_1^\top - Y_2 T_2 Y_2^\top + Y_1 T_1 Y_1^\top Y_2 T_2 Y_2^\top
$$

Setting $T_3 = -T_1(Y_1^\top Y_2)T_2$, we can verify:

$$
I - \begin{pmatrix} Y_1 & Y_2 \end{pmatrix} \begin{pmatrix} T_1 & T_3 \\ 0 & T_2 \end{pmatrix} \begin{pmatrix} Y_1 & Y_2 \end{pmatrix}^\top
$$

$$
= I - Y_1 T_1 Y_1^\top - Y_1 T_3 Y_2^\top - Y_2 T_2 Y_2^\top
$$

$$
= I - Y_1 T_1 Y_1^\top + Y_1 T_1 (Y_1^\top Y_2) T_2 Y_2^\top - Y_2 T_2 Y_2^\top = Q_1 Q_2 \quad \checkmark
$$

**Recursive Algorithm (RGEQR3):**

```
RGEQR3(A[1:m, 1:n]) -> (Y, R, T):
  if n == 1:
    compute Householder H = I - τ·u·u^T such that H·A = (x, 0)^T
    return (u, x, τ)
  else:
    n1 = ⌊n/2⌋, j1 = n1 + 1
    (Y1, R1, T1) = RGEQR3(A[1:m, 1:n1])         // left half
    A[1:m, j1:n] ← Q1^T · A[1:m, j1:n]          // DGEMM + DTRMM
    (Y2, R2, T2) = RGEQR3(A[j1:m, j1:n])         // right half (on updated trailing matrix)
    T3 = -T1 · (Y1^T · Y2) · T2                   // DGEMM (the merge!)
    Y = (Y1, Y2), T = [[T1, T3], [0, T2]]
    return (Y, R, T)
```

**For $k = 2$ (base case of the merge):**

$$
T = \begin{pmatrix} \tau_1 & -\tau_1 u_1^\top u_2 \tau_2 \\ 0 & \tau_2 \end{pmatrix}
$$

**Standard LAPACK bordering (DLARFT, for comparison):**

At each step $j$, compute column $j$ of $T$ via:

$$
w = -\tau_j T(1{:}j{-}1, 1{:}j{-}1) \cdot Y(:, 1{:}j{-}1)^\top \cdot v_j
$$

This requires one matrix-vector product (DGEMV) per column — **Level-2 BLAS, sequential**.

## Complexity

| Operation | DLARFT (Bordering) | Recursive Merge (RGEQR3) |
|-----------|-------------------|--------------------------|
| $T$ construction | $O(mk^2)$ via $k$ sequential DGEMV | $O(mk^2)$ via $\log_2 k$ levels of DGEMM |
| BLAS level | Level-2 (bandwidth-bound) | Level-3 (compute-bound) ✓ |
| Parallelism in $T$ build | $O(k)$ sequential steps | $O(\log_2 k)$ sequential levels |
| Trailing matrix update | Level-3 DGEMM | Level-3 DGEMM (same) |
| Tuning parameters | Block size $nb$, crossover $nx$ | Block size $nb$ only |
| Tall-thin speedup | 1× baseline | **1.5–3× over DGEQRF** |
| Square matrix speedup | 1× baseline | **1.15–1.20× over DGEQRF** |

**Memory:** Same $O(mk + k^2)$ for $Y$ and $T$ matrices. The recursive approach requires $O(\log_2 k)$ stack depth but no additional workspace beyond what DLARFT needs.

**FLOP count:** RGEQR3 has FLOP count $R(m,k) = 3k^2 m - k(5k^2 + 3k - 38)/6 - \Delta W(k)$, where $\Delta W(k)$ is a small correction. The FLOP ratio of RGEQR3 to DGEQR2 is approximately 7/6 (17% higher FLOPs), but the 3:1 FLOP-rate ratio of Level-3 vs Level-2 BLAS means RGEQR3 executes **~2.4× faster** despite the extra FLOPs.

**The core trade-off:** More FLOPs, but all in GEMMs (tensor cores), yielding large wall-clock speedups. On IBM POWER2: 90.3% of peak (415.4 MFLOP/s) vs 76.8% for DGEQRF. On modern GPUs with tensor cores, the Level-3 advantage is even larger (10–16×).

## Applicability

- **Orthogonal RNN weight construction:** When parameterizing a transition matrix as $Q = H_1 H_2 \cdots H_L$ (as in CWY/Householder RNNs), the recursive merge builds the compact WY form $I - YTY^\top$ using GEMMs at every level of the recursion tree. This is the optimal way to precompute the CWY representation on GPUs.

- **DeltaNet / DeltaProduct training:** Within each chunk of $C$ tokens, the UT transform already converts WY accumulation to matmuls. The recursive merge provides the **inter-chunk** analog: merging compact WY representations from adjacent chunks into larger blocks, enabling hierarchical parallelism.

- **Blocked QR for neural network layers:** Any layer requiring QR factorization (orthogonal constraints, weight re-parameterization, Jacobian computation in neural ODEs) benefits from the recursive approach. The cache-oblivious nature means no block-size tuning is needed.

- **Parallel scan on Householder products:** The merge formula $T_3 = -T_1(Y_1^\top Y_2)T_2$ is an **associative binary operation** on compact WY pairs $(Y, T)$. This means products of Householder reflections can be accumulated via **parallel prefix scan** over the recursion tree, with each node performing one GEMM. This is the mathematical foundation for parallelizing orthogonal state transitions across time steps.

## Limitations

- **$O(k^3)$ cubic cost at large $k$:** The recursive merge cost grows as $O(mk^2 + k^3)$ due to the $T_1 (Y_1^\top Y_2) T_2$ computation involving $k \times k$ triangular matrices. For $k = m$ (full QR), the cubic term $(13/6)m^3$ dominates and is worse than the $(4/3)m^3$ of standard blocked QR. The hybrid algorithm (RGEQRF) mitigates this by using recursion only up to $k = nb$ (the block size, typically 32–64).

- **Recursive overhead for small problems:** Function call overhead and the inability to use register-blocking at the leaves of the recursion tree mean a crossover parameter $rb$ is needed (use direct DGEQR2 for $k \leq rb$). In practice $rb = 4$ works well.

- **Not directly applicable to generalized Householder ($\beta \neq 2$):** The compact WY form $I - YTY^\top$ assumes standard Householder reflections with $\tau = 2/\|v\|^2$. For DeltaNet's generalized form $I - \beta k k^\top$ with arbitrary $\beta$, one must use the UT transform variant (trick 139) instead.

- **Sequential trailing-matrix dependency:** In QR factorization, the right half of the matrix depends on the left half being factored first (the trailing matrix update). This limits the parallelism to within each panel. For the pure WY construction use case (no trailing matrix), the recursion is fully parallel.

## Implementation Notes

```python
import torch

def recursive_wy_merge(Y1, T1, Y2, T2):
    """
    Merge two compact WY representations into one.

    Q1 = I - Y1 T1 Y1^T,  Q2 = I - Y2 T2 Y2^T
    Q = Q1 Q2 = I - Y T Y^T

    Args:
        Y1: (m, k1) - Householder vectors for Q1
        T1: (k1, k1) - upper triangular T for Q1
        Y2: (m, k2) - Householder vectors for Q2
        T2: (k2, k2) - upper triangular T for Q2

    Returns:
        Y: (m, k1+k2) - concatenated Householder vectors
        T: (k1+k2, k1+k2) - merged upper triangular T
    """
    k1 = T1.shape[0]
    k2 = T2.shape[0]

    # The merge: one GEMM for Y1^T Y2, then triangular multiplies
    # Y1^T Y2: (k1, m) @ (m, k2) -> (k1, k2)  [TENSOR CORE GEMM]
    G = Y1.T @ Y2  # (k1, k2)

    # T3 = -T1 @ G @ T2: two triangular matrix multiplies
    # In practice, fuse as: T3 = -T1 @ (G @ T2)
    T3 = -T1 @ (G @ T2)  # (k1, k2)

    # Assemble block upper triangular T
    T = torch.zeros(k1 + k2, k1 + k2, dtype=T1.dtype, device=T1.device)
    T[:k1, :k1] = T1
    T[:k1, k1:] = T3
    T[k1:, k1:] = T2

    # Concatenate Y
    Y = torch.cat([Y1, Y2], dim=1)

    return Y, T


def recursive_wy_build(v_list, tau_list):
    """
    Build compact WY representation from a list of Householder vectors
    using recursive binary merging (RGEQR3 strategy).

    All T-construction is done via GEMMs (Level-3 BLAS).

    Args:
        v_list: list of k vectors, each (m,) — Householder vectors
        tau_list: list of k scalars — Householder coefficients

    Returns:
        Y: (m, k) — Householder vector matrix
        T: (k, k) — upper triangular T matrix

    Complexity: O(mk^2) via log2(k) levels of GEMMs
    """
    k = len(v_list)

    if k == 1:
        # Base case: single reflector
        Y = v_list[0].unsqueeze(1)  # (m, 1)
        T = tau_list[0].reshape(1, 1)  # (1, 1)
        return Y, T

    # Recursive split
    mid = k // 2
    Y1, T1 = recursive_wy_build(v_list[:mid], tau_list[:mid])
    Y2, T2 = recursive_wy_build(v_list[mid:], tau_list[mid:])

    # Merge via GEMM
    return recursive_wy_merge(Y1, T1, Y2, T2)


def parallel_householder_scan(v_list, tau_list):
    """
    Parallel prefix scan over Householder reflections using the
    merge formula as the associative binary operator.

    Each merge is a GEMM — maps to tensor cores.
    Parallel depth: O(log k), work: O(mk^2).

    This is the key enabler for parallelizing orthogonal
    state transitions across time steps in sequence models.
    """
    k = len(v_list)

    # Initialize leaves: each reflector as a (Y, T) pair
    pairs = [(v.unsqueeze(1), tau.reshape(1, 1))
             for v, tau in zip(v_list, tau_list)]

    # Binary tree reduction (up-sweep)
    while len(pairs) > 1:
        new_pairs = []
        for i in range(0, len(pairs), 2):
            if i + 1 < len(pairs):
                Y, T = recursive_wy_merge(*pairs[i], *pairs[i+1])
                new_pairs.append((Y, T))
            else:
                new_pairs.append(pairs[i])
        pairs = new_pairs

    return pairs[0]  # Final (Y, T)
```

**GPU efficiency analysis:**

1. **$Y_1^\top Y_2$ GEMM:** $(k_1 \times m) \times (m \times k_2) \to (k_1 \times k_2)$. When $m \gg k$, this is a wide matmul with high arithmetic intensity — perfect for tensor cores.

2. **Triangular multiplies $T_1 \cdot G$ and $G \cdot T_2$:** These are DTRMM operations, which map to tensor cores as masked GEMMs. For small $k$ (typical: 32–128), they fit entirely in shared memory.

3. **Memory access:** The recursive structure is cache-oblivious — data is accessed in blocks that naturally fit the cache/shared memory hierarchy. No block-size tuning needed.

4. **Parallelism:** At each level of the recursion tree, all merges at that level are independent and can execute as a batched GEMM. With $k$ reflections, there are $\log_2 k$ levels, each with $k/2^l$ independent merge operations.

5. **Comparison to DLARFT (bordering):** DLARFT does $k$ sequential DGEMV operations (Level-2 BLAS, bandwidth-bound). On A100/H100, Level-3 BLAS achieves 10–16× higher throughput than Level-2, so the recursive approach should give similar speedups for the $T$-construction phase.

## References

- Elmroth, E. & Gustavson, F. G. (2000). Applying Recursion to Serial and Parallel QR Factorization Leads to Better Performance. *IBM J. Res. Develop.*, 44(4), 605–624.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. *SIAM J. Sci. Stat. Comput.*, 10(1), 53–57.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Dongarra, J., Kaufman, L., & Hammarling, S. (1988). Squeezing the Most out of Eigenvalue Solvers on High-Performance Computers. *Linear Algebra Appl.*, 77, 113–136.
- LAPACK DGEQRT3: Recursive QR factorization producing compact WY form. https://www.netlib.org/lapack/
- Likhosherstov, V., Davis, J., Choromanski, K., & Weller, A. (2021). CWY Parametrization: a Solution for Parallelized Optimization of Orthogonal and Stiefel Matrices. AISTATS 2021.

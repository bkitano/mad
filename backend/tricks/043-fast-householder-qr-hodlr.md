# 043: Fast Householder-Based QR Decomposition for HODLR Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Kressner & Susnjara (2018), EPFL
**Paper**: [papers/fast-qr-hodlr-matrices.pdf]
**Documented**: 2026-02-15

## Description

This trick provides a numerically stable, fast QR decomposition for HODLR (Hierarchically Off-Diagonal Low-Rank) matrices — the simplest hierarchical low-rank format and a special case of HSS matrices. The key innovation is combining the recursive block QR decomposition by Elmroth & Gustavson with the compact WY representation of Householder reflectors, all performed in the HODLR format. This achieves $O(k^2 n \log^2 n)$ complexity while maintaining numerical orthogonality down to roundoff error, even for highly ill-conditioned matrices ($\kappa(A) > 10^{12}$).

Existing fast QR methods for hierarchical matrices all suffer from numerical instability: Cholesky-based QR breaks down when $A^T A$ loses positive definiteness (at $\kappa(A)^2$), LU-based QR requires well-conditioned leading blocks, and block Gram-Schmidt loses orthogonality for ill-conditioned matrices. This algorithm avoids all these issues by directly using Householder reflectors — the gold standard for numerical stability — while exploiting HODLR structure for efficiency. The orthogonal factor $Q$ is represented implicitly as $Q = I - YTY^T$, where $Y$ is a unit lower triangular HODLR matrix and $T$ is an upper triangular HODLR matrix, enabling $O(kn \log n)$ application of $Q$ or $Q^T$ to a vector.

## Mathematical Form

**HODLR Matrix Structure:**

A matrix $A \in \mathbb{R}^{n \times n}$ is HODLR of level $\ell$ and rank $k$ if it admits the recursive $2 \times 2$ block partition:

$$
A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
$$

where diagonal blocks $A_{11}, A_{22}$ are again HODLR matrices (of level $\ell - 1$), and off-diagonal blocks have low rank:

$$
A|_{\text{off}} = A_L A_R, \quad A_L \in \mathbb{R}^{n_L \times k}, \quad A_R \in \mathbb{R}^{k \times n_R}
$$

**Storage:** $O(kn \log n)$ vs $O(n^2)$ dense.

**Compact WY Representation:**

The orthogonal factor from Householder QR is represented as:

$$
Q = I_m - YTY^T
$$

where $T \in \mathbb{R}^{n \times n}$ is upper triangular and $Y \in \mathbb{R}^{m \times n}$ is unit lower triangular (with first $n$ rows forming a unit lower triangular matrix). The QR decomposition is:

$$
A = Q \begin{bmatrix} R \\ 0 \end{bmatrix}
$$

**Recursive Block QR (Elmroth-Gustavson):**

For a dense matrix $A = [A_1 \mid A_2]$ partitioned into two block columns:

1. Recursively compute $A_1 = Q_1 \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$ with $Q_1 = I_m - Y_1 T_1 Y_1^T$

2. Update $A_2 \leftarrow Q_1^T A_2 = A_2 - Y_1 T_1 (Y_1^T A_2)$

3. Partition the updated block as $\begin{bmatrix} A_{12} \\ A_{22} \end{bmatrix}$ and recursively compute $A_{22} = Q_2 \begin{bmatrix} R_2 \\ 0 \end{bmatrix}$

4. Combine: $Y = [Y_1 \mid \tilde{Y}_2]$, $T = \begin{bmatrix} T_1 & T_{12} \\ 0 & T_2 \end{bmatrix}$ where $T_{12} = -T_1 Y_1^T \tilde{Y}_2 T_2$

**HODLR Block Column Structure:**

At each recursion level, the algorithm processes a "block column" of the form:

$$
H = \begin{bmatrix} \bar{A} \\ B \\ C \end{bmatrix}
$$

where:
- $\bar{A} \in \mathbb{R}^{m \times m}$ is a HODLR matrix of level $\tilde{\ell} \leq \ell$
- $B \in \mathbb{R}^{p \times m}$ is given in factored form $B = B_L B_R$ with $B_L \in \mathbb{R}^{p \times r_1}$, $B_R \in \mathbb{R}^{r_1 \times m}$
- $C \in \mathbb{R}^{r_2 \times m}$ is a small dense matrix

This structure arises naturally from the HODLR format: $\bar{A}$ is the diagonal HODLR block, $B$ captures the off-diagonal low-rank interaction within the current block column, and $C$ captures the low-rank interaction extending to other block columns.

**Recursive HODLR QR (Algorithm 2):**

At the highest level ($\tilde{\ell} = \ell$), $B$ and $C$ are void and $H = A$. The algorithm proceeds:

1. **Preprocessing**: Orthonormalize $B_L$ via economy QR, update $B_R \leftarrow R B_R$

2. **Compress**: Form $\tilde{H} = \begin{bmatrix} \bar{A} \\ B_R \\ C \end{bmatrix}$ (size $(m + r_1 + r_2) \times m$)

3. **Base case** ($\tilde{\ell} = 0$): Dense QR via Algorithm 1

4. **Recursion** ($\tilde{\ell} \geq 1$): Repartition $\tilde{H}$ according to the HODLR format of $\bar{A}$:

$$
\tilde{H} = \begin{bmatrix} \bar{A}_{11} & \bar{A}_{12} \\ \bar{A}_{21} & \bar{A}_{22} \\ B_{R,1} & B_{R,2} \\ C_1 & C_2 \end{bmatrix}
$$

5. **First block column**: Recursively QR-decompose $\begin{bmatrix} \bar{A}_{11} \\ \bar{A}_{21} \\ B_{R,1} \\ C_1 \end{bmatrix}$ — this has the form (11) with level $\tilde{\ell} - 1$

6. **Update second block column**: Compute $S = T_1^T Y_1^T \begin{bmatrix} \bar{A}_{12} \\ \bar{A}_{22} \\ B_{R,2} \\ C_2 \end{bmatrix}$

The key insight is that $S$ is a sum of low-rank matrices:

$$
S = T_1^T (Y_{A,11}^T \bar{A}_{12} + Y_{A,21}^T \bar{A}_{22} + Y_{B_{R,1}}^T B_{R,2} + Y_{C,1}^T C_2)
$$

Each term is low-rank (at most rank $k$), so $S$ has rank at most $(\ell + 1)k$. To prevent quadratic rank growth, $(\ell + 1)k$ separate rank-$k$ additions are performed, each followed by recompression to rank $O(k)$.

7. **Second block column**: Recursively QR-decompose the updated unreduced part

8. **Combine**: Merge the two WY representations into a single one

**Output Structure:**

$$
A = Q \begin{bmatrix} R \\ 0 \end{bmatrix}, \quad Q = I - YTY^T
$$

where:
- $Y_A \in \mathbb{R}^{m \times m}$ — unit lower triangular HODLR matrix of level $\tilde{\ell}$
- $\tilde{Y}_B = [Y_{B_{R,1}} \mid Y_{B_{R,2}}]$ — factored low-rank
- $Y_C = [Y_{C,1} \mid Y_{C,2}]$ — small dense
- $R, T \in \mathbb{R}^{m \times m}$ — upper triangular HODLR matrices of level $\tilde{\ell}$

**Key Definitions:**

- $A \in \mathcal{H}_{n \times n}(\ell, k)$ — HODLR matrix with level $\ell = O(\log n)$ and rank $k$
- $n_{\min}$ — Minimal leaf block size (default 250)
- $\epsilon$ — Truncation tolerance for recompression (default $10^{-10}$)
- $\mathcal{T}_\epsilon$ — Recompression operator: truncate to rank $k_\epsilon$ maintaining $\epsilon$-accuracy

## Complexity

| Operation | Dense QR | CholQR (HODLR) | hQR (This Trick) |
|-----------|----------|-----------------|-------------------|
| QR factorization | $O(n^3)$ | $O(k^2 n \log^2 n)$ | $O(k^2 n \log^2 n)$ |
| Numerical orthogonality | $O(\mathbf{u})$ | $O(\kappa(A)^2 \mathbf{u})$ | $O(\mathbf{u})$ |
| Robustness to ill-conditioning | Excellent | Fails at $\kappa > 10^5$ | Excellent |
| Applicable to non-SPD | Yes | Requires $A^T A$ SPD | Yes |

**Detailed Complexity Breakdown:**

- **Line 3 (left-orthogonalization)**: $O(k^2 n \log n)$ — orthonormalize all off-diagonal $B_L$ factors
- **Line 7 (base case dense QR)**: $O(kn \log n)$ — $2^\ell$ dense QR decompositions of size $O(\ell k) \times O(1)$
- **Lines 11-12 (compute $S$)**: $O(k^2 n \log^2 n)$ — HODLR matrix-vector products + low-rank additions with recompression
- **Line 13 (update second block column)**: $O(k^2 n \log^2 n)$ — HODLR matrix multiplications + recompression
- **Lines 15-16 (compute $T_{12}$)**: $O(k^2 n \log^2 n)$ — same cost as $S$

**Total:** $O(k^2 n \log^2 n)$

**Memory:** $O(kn \log n)$ for storing $Y$, $T$, and $R$ in HODLR format

**Practical Performance:**

| $n$ | hQR time (s) | CholQR time (s) | MATLAB QR time (s) |
|-----|-------------|-----------------|-------------------|
| 4000 | 0.3 | 0.15 | 0.4 |
| 16000 | 2.5 | 1.3 | 20 |
| 64000 | 20 | 10 | 1000+ |
| 256000 | 160 | — (fails) | — (OOM) |

The hQR method is approximately 2$\times$ slower than CholQR but does not fail for ill-conditioned matrices, and is orders of magnitude faster than dense QR for large $n$.

## Applicability

1. **Orthogonalization of Spectral Projectors**: Computing QR decompositions of matrices like $\Pi_{<0}(:, C)$ where $\Pi_{<0}$ is a spectral projector of a banded symmetric matrix — the projector has HODLR structure, and its orthogonalization is needed for spectral divide-and-conquer eigensolvers

2. **Stable Linear Solves with HODLR/HSS Matrices**: The QR decomposition provides a numerically stable alternative to LU decomposition for solving $Ax = b$ when $A$ is HODLR — especially important when $A$ is non-symmetric or non-positive-definite, where Cholesky-based methods fail

3. **Least-Squares Problems with Structured Matrices**: Solving $\min_x \|Ax - b\|_2$ via QR when $A$ is a rectangular HODLR matrix — arises in structured regression, kernel interpolation, and attention-based linear least squares

4. **State Space Model Computations**: Orthogonalizing structured basis matrices that arise in SSM discretization and spectral methods, where the matrices naturally have off-diagonal low-rank structure

5. **Gaussian Process Computations**: Computing the QR factorization of kernel matrices (which have HODLR structure from exponentially decaying kernels) for numerically stable GP regression — more stable than the Cholesky-based approach commonly used

6. **Structured Eigensolvers**: The QR decomposition of HODLR matrices is a key subroutine in the QDWH-based polar decomposition for fast symmetric eigenvalue computation

## Limitations

1. **HODLR Only (Not Full HSS)**: The algorithm is designed for HODLR matrices (weak admissibility, off-diagonal blocks are low-rank at each level). Extension to the broader HSS class (nested bases) or full $\mathcal{H}$-matrices requires additional work — the 2024 paper by Griem & Le Borne addresses the $\mathcal{H}$-matrix case

2. **$Q$ and $R$ Are Not HODLR**: The factors $Q$ and $R$ do not inherit the HODLR property from $A$ — their off-diagonal ranks can grow (logarithmically in practice). $Q$ is represented implicitly via the compact WY form $(I - YTY^T)$ where $Y, T$ are HODLR

3. **Rank Growth During Updates**: The update $\hat{A}_{12} = \bar{A}_{12} - Y_{A,11} S$ introduces rank growth in the off-diagonal blocks. Separate rank-$k$ additions with recompression (rather than a single rank-$(\ell+1)k$ addition) are needed to maintain $O(k)$ ranks, adding implementation complexity

4. **Approximate**: Like all HODLR arithmetic, the algorithm introduces approximation errors proportional to the truncation tolerance $\epsilon$ at each recompression step. Errors accumulate through $O(\ell) = O(\log n)$ recursion levels

5. **Not GPU-Optimized**: The reference MATLAB implementation is sequential; the recursive structure with variable-size blocks makes GPU parallelization challenging

6. **Constant Factor**: Approximately 2$\times$ slower than the unstable CholQR method for well-conditioned matrices — the stability guarantee comes at a moderate constant-factor cost

## Implementation Notes

```python
# Pseudocode for Householder-based QR of HODLR matrices
# (Algorithm 2 from Kressner & Susnjara)

def hodlr_qr(A_hodlr, B_L=None, B_R=None, C=None, level=None):
    """
    Compute QR decomposition A = Q[R; 0] for HODLR matrix A.

    The orthogonal factor Q = I - Y T Y^T is stored in compact WY form
    where Y is unit lower triangular HODLR and T is upper triangular HODLR.

    Args:
        A_hodlr: HODLR matrix of level ell
        B_L, B_R: factored low-rank matrix B = B_L @ B_R (off-diagonal)
        C: dense matrix (small, from outer recursion)
        level: current HODLR level (ell_tilde)

    Returns:
        Y: compact WY factor (HODLR lower triangular)
        T: compact WY factor (HODLR upper triangular)
        R: upper triangular HODLR matrix

    Complexity: O(k^2 n log^2 n)
    """
    if level is None:
        level = A_hodlr.level

    m = A_hodlr.rows

    # Preprocessing: orthonormalize B_L if needed
    if B_L is not None and not is_orthonormal(B_L):
        Q_B, R_B = np.linalg.qr(B_L, mode='economic')
        B_L = Q_B
        B_R = R_B @ B_R

    # Form compressed block column
    # H_tilde = [A_hodlr; B_R; C]  (size (m + r1 + r2) x m)

    if level == 0:
        # Base case: dense QR
        H_dense = assemble_dense(A_hodlr, B_R, C)
        Y, T, R = dense_recursive_block_qr(H_dense)
        return Y, T, R

    # Recursive case: split according to HODLR format
    A11, A12_lr, A21_lr, A22 = A_hodlr.split()
    # A12_lr, A21_lr are in factored low-rank form

    # First block column: [A11; A21; B_R1; C1]
    # This has structure (11) with level = level - 1
    Y1, T1, R1 = hodlr_qr(
        A11,
        B_L=A21_lr.left,   # left factor of A21
        B_R=A21_lr.right,   # right factor of A21
        C=vstack(B_R[:, :m1], C[:, :m1]),
        level=level - 1
    )

    # Compute S = T1^T @ Y1^T @ [A12; A22; B_R2; C2]
    # Key insight: each of the 4 terms is low-rank
    #   Y_{A,11}^T @ A12  (rank k from A12 low-rank factor)
    #   Y_{A,21}^T @ A22  (rank k from A21 low-rank factor)
    #   Y_{B,1}^T @ B_{R,2}  (rank r1)
    #   Y_{C,1}^T @ C2  (rank r2)
    S = compute_low_rank_update(Y1, T1, A12_lr, A22, B_R, C)

    # Recompression: add each rank-k term separately
    # to prevent rank growth to O(ell * k)
    # After each addition, recompress to tolerance epsilon
    S_compressed = incremental_low_rank_add_recompress(
        terms=[term1, term2, term3, term4],
        tolerance=epsilon * norm(A_hodlr)
    )

    # Update second block column
    A12_updated = hodlr_subtract_low_rank(A12_lr, Y1.A11 @ S)
    A22_updated = hodlr_subtract_low_rank(A22, Y1.A21 @ S)
    B_R2_updated = B_R[:, m1:] - Y1.B @ S
    C2_updated = C[:, m1:] - Y1.C @ S

    # QR of updated second block column
    Y2, T2, R2 = hodlr_qr(
        A22_updated,
        B_L=None,  # already processed
        B_R=B_R2_updated,
        C=C2_updated,
        level=level - 1
    )

    # Compute coupling block T12 = -T1 @ Y1^T @ Y_tilde_2 @ T2
    T12 = compute_t12_block(T1, Y1, Y2, T2)  # same structure as S

    # Assemble output
    T = hodlr_upper_triangular(T1, T12, T2)
    R = hodlr_upper_triangular(R1, A12_updated[:m1, :], R2)
    Y = hodlr_unit_lower_triangular(Y1, Y2)  # with B_L @ Y_B appended

    return Y, T, R


def apply_Q(Y, T, x, transpose=False):
    """
    Apply Q = I - Y T Y^T (or Q^T) to vector x.

    Uses HODLR matrix-vector multiplication.
    Complexity: O(k n log n) per vector
    """
    if transpose:
        # Q^T x = x - Y T^T Y^T x
        temp = hodlr_matvec(Y, x, transpose=True)   # Y^T x: O(kn log n)
        temp = hodlr_solve_triangular(T, temp, transpose=True)  # T^T \ temp
        return x - hodlr_matvec(Y, temp)              # Y @ temp: O(kn log n)
    else:
        # Q x = x - Y T Y^T x
        temp = hodlr_matvec(Y, x, transpose=True)   # Y^T x
        temp = hodlr_solve_triangular(T, temp)        # T \ temp
        return x - hodlr_matvec(Y, temp)              # Y @ temp
```

**Key Implementation Insights:**

1. **Compact WY representation is essential**: Storing $Q$ as $I - YTY^T$ with HODLR $Y, T$ enables $O(kn \log n)$ application — much cheaper than storing $Q$ explicitly (which would be dense $O(n^2)$). This is the same WY representation used in LAPACK but specialized to the HODLR format

2. **Incremental recompression prevents rank explosion**: The update $S$ is a sum of $(\ell + 1)$ rank-$k$ matrices. Adding them all at once would create a rank-$(\ell + 1)k$ matrix, causing $O(\ell^2 k^2)$ recompression cost. Instead, performing $(\ell + 1)$ separate rank-$k$ additions, each followed by recompression to rank $O(k)$, keeps the cost at $O(\ell k^2)$ per step — crucial for the overall $O(k^2 n \log^2 n)$ bound

3. **Numerical orthogonality**: Unlike CholQR or Gram-Schmidt approaches, hQR achieves $e_{\text{orth}} = \|Q^T Q - I\|_2 \sim O(\mathbf{u})$ (machine precision) regardless of $\kappa(A)$. CholQR achieves only $O(\kappa(A)^2 \mathbf{u})$, failing entirely when $\kappa(A) > 10^5$

4. **Off-diagonal ranks of factors grow slowly**: While $Q$ and $R$ do not inherit HODLR structure exactly, their off-diagonal ranks grow only logarithmically in $n$ (bounded by $\sim 10$–$40$ in experiments), keeping memory reasonable

5. **Rectangular extension**: For rectangular HODLR matrices $A \in \mathbb{R}^{m \times n}$ with $m > n$, the algorithm uses *permuted triangular form* to avoid mixing dense and low-rank blocks, processing block columns sequentially with appropriate row permutations

## References

- Kressner, D. & Susnjara, A. (2018). Fast QR decomposition of HODLR matrices. arXiv:1809.10585. Published in *SIAM Journal on Matrix Analysis and Applications*.
- Elmroth, E. & Gustavson, F. (2000). Applying recursion to serial and parallel QR factorization leads to better performance. *IBM J. Research & Development*, 44(4), 605-624.
- Schreiber, R. & Van Loan, C. F. (1989). A storage-efficient WY representation for products of Householder transformations. *SIAM J. Sci. Statist. Comput.*, 10(1), 53-57.
- Benner, P. & Mach, T. (2010). On the QR decomposition of $\mathcal{H}$-matrices. *Computing*, 88(3-4), 111-129.
- Griem, V. & Le Borne, S. (2024). A Block Householder-Based Algorithm for the QR Decomposition of Hierarchical Matrices. *SIAM J. Matrix Anal. Appl.*, 45(2), 847-874.

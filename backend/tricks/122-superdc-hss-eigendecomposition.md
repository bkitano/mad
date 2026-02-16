# 122: SuperDC: Stable Superfast Divide-and-Conquer Eigendecomposition for HSS Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ou & Xia (2021), Purdue University
**Paper**: [papers/superdc-hss-eigensolver.pdf]
**Documented**: 2026-02-15

## Description

SuperDC is a stable, superfast divide-and-conquer eigensolver for dense Hermitian matrices with small off-diagonal (numerical) ranks, given in hierarchically semiseparable (HSS) form. It computes the full eigenvalue decomposition $A \approx Q \Lambda Q^T$ in $O(r^2 n \log^2 n)$ flops with $O(rn \log n)$ storage, compared to $O(n^3)$ flops and $O(n^2)$ storage for standard eigensolvers like LAPACK's `eig`. At $n = 32{,}768$, SuperDC is **136× faster** than MATLAB's `eig` using only **1/15 of the memory**.

The algorithm generalizes the classical tridiagonal divide-and-conquer eigensolver to HSS matrices. It operates in two stages: a **dividing stage** that recursively splits the HSS matrix into block-diagonal plus low-rank updates, and a **conquering stage** that solves a sequence of rank-1 update eigenproblems via secular equations accelerated by the fast multipole method (FMM). The key innovations are:

1. **Balanced dividing strategy** with norm-controlled low-rank updates that eliminates exponential norm growth in HSS generators
2. **Triangular FMM** that avoids cancellation when accelerating secular equation function evaluations
3. **Local shifting** that integrates per-eigenvalue shifts into the FMM structure for clustered eigenvalues, maintaining both stability and $O(n)$ complexity per conquering step

## Mathematical Form

**Core Operation:**

Given a symmetric HSS matrix $A$ with HSS rank $r$, compute:

$$
A \approx Q \Lambda Q^T
$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains eigenvalues and $Q$ is the orthogonal eigenmatrix.

**Dividing Stage — HSS Splitting:**

For an HSS node $p$ with children $i, j$, the diagonal block is:

$$
D_p = \begin{pmatrix} D_i & U_i B_i U_j^T \\ U_j B_i^T U_i^T & D_j \end{pmatrix}
$$

This is rewritten as a block-diagonal matrix plus a rank-$2r$ update:

$$
D_p = \text{diag}(\hat{D}_i, \hat{D}_j) + Z_p Z_p^T
$$

where:

$$
\hat{D}_i = D_i - \frac{1}{\|B_i\|_2} U_i B_i B_i^T U_i^T, \quad \hat{D}_j = D_j - \|B_i\|_2 U_j U_j^T
$$

$$
Z_p = \begin{pmatrix} \frac{1}{\sqrt{\|B_i\|_2}} U_i B_i \\ \sqrt{\|B_i\|_2} U_j \end{pmatrix}
$$

The **balancing** by $\|B_i\|_2$ is the key stability innovation — it ensures generator norms grow only linearly $O(n\beta)$ instead of exponentially $O(\beta^{n/4})$.

**Conquering Stage — Secular Equations:**

After dividing to leaves and solving small dense eigenproblems, the conquering stage merges solutions upward. At each node, the rank-$r$ update is decomposed into $r$ rank-1 updates:

$$
\bar{\Lambda} + \mathbf{v}\mathbf{v}^T = \tilde{Q} \Lambda \tilde{Q}^T
$$

where $\bar{\Lambda} = \text{diag}(d_1, \ldots, d_n)$ with $d_1 \leq \cdots \leq d_n$. Finding eigenvalues $\lambda_k$ requires solving the **secular equation**:

$$
f(x) = 1 + \sum_{k=1}^{n} \frac{v_k^2}{d_k - x} = 0
$$

**Shifted Secular Equation (for stability):**

To handle clustered eigenvalues, a local shift $d_k$ is used:

$$
g_k(y) \equiv f(d_k + y) = 1 + \sum_{j=1}^{n} \frac{v_j^2}{\delta_{jk} - y} = 0
$$

where $\delta_{jk} = d_j - d_k$ and the eigenvalue gap $\eta_k = \lambda_k - d_k$ is found by solving $g_k(\eta_k) = 0$.

**Triangular FMM Acceleration:**

Function evaluations are split into lower and upper triangular parts:

$$
\mathbf{f} = \mathbf{e} + \boldsymbol{\psi} + \boldsymbol{\phi} = \mathbf{e} + C_L \mathbf{w} + C_U \mathbf{w}
$$

where $C_L$ and $C_U$ are the lower and strictly upper triangular parts of the Cauchy matrix $C = \left(\frac{1}{d_j - x_i}\right)_{n \times n}$. The triangular FMM applies the FMM separately to each triangular part, avoiding the cancellation that occurs when positive and negative terms are mixed.

**Structured Eigenvectors via Löwner's Formula:**

$$
\mathbf{q}_k = \left(\frac{\hat{v}_1}{d_1 - \lambda_k} \cdots \frac{\hat{v}_k}{d_k - \lambda_k} \cdots \frac{\hat{v}_n}{d_n - \lambda_k}\right)^T
$$

where $\hat{v}_i = \sqrt{\frac{\prod_j (\lambda_j - d_i)}{\prod_{j \neq i} (d_j - d_i)}}$

The eigenmatrix $\hat{Q} = \left(\frac{\hat{v}_i b_j}{d_i - \lambda_j}\right)_{n \times n}$ is a **Cauchy-like matrix** that can be applied to vectors in $O(n \log n)$ via FMM.

**Overall Eigenmatrix Structure:**

$$
Q = Q^{(l_{\max})} \prod_{l=l_{\max}-1}^{0} \left(P^{(l)} Q^{(l)}\right)
$$

where $Q^{(l)} = \text{diag}(\hat{Q}_i)$ are block-diagonal intermediate eigenmatrices (products of Cauchy-like matrices) and $P^{(l)}$ are permutation matrices at each tree level.

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — Symmetric HSS matrix
- $r$ — HSS rank (maximum off-diagonal rank)
- $\mathcal{T}$ — Postordered full binary tree for HSS structure
- $D_i, U_i, B_i, R_i$ — HSS generators at node $i$
- $\beta = \max_k \|B_k\|_2$ — Maximum coupling matrix norm
- $\tau$ — User-supplied deflation tolerance
- $l_{\max}$ — Leaf level of the HSS tree ($O(\log n)$ for balanced trees)

## Complexity

| Operation | Standard `eig` | SuperDC |
|-----------|---------------|---------|
| Eigendecomposition | $O(n^3)$ | $O(r^2 n \log^2 n)$ |
| Eigenvalues only | $O(n^2)$ | $O(r^2 n \log^2 n)$ |
| Apply $Q$ or $Q^T$ to vector | $O(n^2)$ | $O(rn \log n)$ |
| Storage for $Q$ | $O(n^2)$ | $O(rn \log n)$ |

**Detailed Breakdown:**

- **Dividing stage**: $O(r^2 n)$ — dominated by HSS generator updates at each tree level
- **Conquering stage**: $O(r^2 n \log^2 n)$ — $O(\log n)$ tree levels, each with $O(r)$ rank-1 eigenproblems, each solved via $O(n)$ FMM-accelerated Newton iterations
- **Secular equation per root**: $O(n)$ via FMM (vs $O(n^2)$ naive), with typically 2–3 Newton iterations to full machine precision

**Memory:** $O(rn \log n)$ for the structured eigenmatrix vs $O(n^2)$ for dense $Q$

**Practical speedup:** At $n = 32{,}768$: SuperDC is 136× faster and uses 1/15 the memory of MATLAB `eig`. At $n = 262{,}144$: SuperDC completes in ~383 seconds while `eig` runs out of 80GB memory.

## Applicability

1. **State Space Model Eigenanalysis**: Computing eigenvalues/eigenvectors of structured transition matrices $A$ in SSMs, where $A$ often has HSS structure from discretized continuous operators — enables efficient spectral initialization or eigenvalue-based discretization

2. **Structured Weight Initialization**: Initializing neural network weight matrices with prescribed spectral properties (eigenvalues) using the structured eigenmatrix $Q$, applied via FMM in $O(n \log n)$ rather than $O(n^2)$

3. **Spectral Normalization of Structured Layers**: Computing the spectral norm $\sigma_1(W)$ or full spectrum of HSS-structured weight matrices for Lipschitz-constrained networks

4. **Kernel Matrix Analysis**: Eigendecomposing kernel/attention matrices that have hierarchical low-rank off-diagonal structure (e.g., from exponentially decaying or spatially local kernels)

5. **Gaussian Process Inference**: Fast eigendecomposition of structured covariance matrices for GP posterior computation, where the kernel matrix often has HSS structure

6. **Matrix Functions via Eigendecomposition**: Computing $f(A) = Q f(\Lambda) Q^T$ for matrix exponentials, inverse square roots, etc. in continuous-time models — complements the telescopic decomposition approach with a spectral method

7. **SVD of Non-Hermitian Matrices**: Computing SVD of a non-Hermitian $A$ by eigendecomposing the Hermitian matrix $A^T A$ or $\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix}$, both of which inherit HSS structure

## Limitations

1. **Symmetric/Hermitian Only**: Directly applies only to symmetric (Hermitian) matrices; non-symmetric eigenproblems require reformulation
2. **HSS Form Required**: Matrix must first be in HSS form; construction cost is $O(r^2 n)$ or $O(n^2)$ depending on access pattern
3. **Approximate**: The eigendecomposition is approximate; accuracy is controlled by HSS compression tolerance and deflation parameter $\tau$
4. **Sequential Tree Depth**: The $O(\log n)$ tree levels must be processed sequentially in both dividing and conquering stages
5. **FMM Implementation Complexity**: Requires a stable triangular FMM implementation, which is non-trivial
6. **Rank Sensitivity**: Complexity is $O(r^2 n \log^2 n)$; for large HSS rank $r$, may not outperform dense solvers
7. **Deflation Sensitivity**: Very small deflation tolerance $\tau$ increases accuracy but reduces deflation opportunities, potentially increasing cost
8. **Currently CPU-only**: The reference Matlab implementation is not GPU-optimized; porting the hierarchical FMM structure to GPUs is challenging

## Implementation Notes

```python
# Pseudocode for SuperDC eigensolver
def superdc_eigendecompose(A_hss, tau=1e-10):
    """
    Compute eigendecomposition A ≈ Q Λ Q^T for symmetric HSS matrix.

    Args:
        A_hss: HSS representation {D_i, U_i, B_i, R_i} for all nodes
        tau: deflation tolerance

    Returns:
        eigenvalues: array of n eigenvalues
        Q_structured: structured eigenmatrix (product of Cauchy-like matrices)

    Complexity: O(r^2 n log^2 n) flops, O(rn log n) storage
    """
    tree = A_hss.tree

    # ====== DIVIDING STAGE ======
    # Recursively split HSS into block-diagonal + low-rank updates
    # Process top-down from root to one level above leaves
    for node_p in preorder(tree.internal_nodes):
        i, j = node_p.children
        B_i = A_hss.B[i]
        U_i, U_j = A_hss.U[i], A_hss.U[j]
        norm_B = norm(B_i, 2)

        # Balanced dividing: scale by ||B_i|| to prevent exponential norm growth
        # Key stability innovation: norms grow O(n*beta) instead of O(beta^(n/4))
        D_hat_i = A_hss.D[i] - (1/norm_B) * U_i @ B_i @ B_i.T @ U_i.T
        D_hat_j = A_hss.D[j] - norm_B * U_j @ U_j.T

        # Low-rank update factor (rank = min(rowsize, colsize) of B_i)
        Z_p = vstack([U_i @ B_i / sqrt(norm_B),
                       sqrt(norm_B) * U_j])

        # Update HSS generators for children using Lemma 2.1
        update_hss_generators(A_hss, node_p, D_hat_i, D_hat_j)

        # Store for conquering stage
        node_p.Z = Z_p
        node_p.D_hat = (D_hat_i, D_hat_j)

    # ====== CONQUERING STAGE ======
    # Solve dense eigenproblems at leaves
    for leaf in tree.leaves:
        leaf.eigenvalues, leaf.Q_local = dense_eig(leaf.D_hat)

    # Merge upward through tree
    for node_p in postorder(tree.internal_nodes):
        i, j = node_p.children
        Lambda_i, Q_i = i.eigenvalues, i.Q_local
        Lambda_j, Q_j = j.eigenvalues, j.Q_local

        # Form merged diagonal + rank-r update
        Lambda_merged = concatenate(Lambda_i, Lambda_j)
        Z_hat = block_diag(Q_i, Q_j).T @ node_p.Z  # O(rn) via structured mult

        # Permute so diagonal entries are sorted (required for FMM)
        P, Lambda_sorted = sort_with_permutation(Lambda_merged)
        Z_hat_sorted = P @ Z_hat

        # Solve r rank-1 update eigenproblems sequentially
        Q_hat = eye(len(Lambda_sorted))
        for col in range(Z_hat_sorted.shape[1]):
            v = Z_hat_sorted[:, col]

            # Apply deflation
            v, Lambda_sorted, perm = deflate(v, Lambda_sorted, tau)

            # Solve secular equations with triangular FMM + local shifting
            eigenvalues_new, Q_cauchy = solve_secular_fmm(
                Lambda_sorted, v, tau
            )

            Q_hat = Q_hat @ perm @ Q_cauchy
            Lambda_sorted = eigenvalues_new

        node_p.eigenvalues = Lambda_sorted
        node_p.Q_local = block_diag(Q_i, Q_j) @ P.T @ Q_hat
        # Q_local stored as structured product of Cauchy-like matrices

    return tree.root.eigenvalues, tree.root.Q_local


def solve_secular_fmm(d, v, tau):
    """
    Solve secular equation f(x) = 1 + sum(v_k^2 / (d_k - x)) = 0
    for all n roots using FMM-accelerated modified Newton's method.

    Uses triangular FMM + local shifting for stability.

    Complexity: O(n) per Newton iteration, O(1) iterations typical → O(n) total
    """
    n = len(d)
    eigenvalues = zeros(n)

    for k in range(n - 1):
        # Use shifted secular equation for stability
        delta = d - d[k]  # shifts: delta_jk = d_j - d_k
        y_k = initial_guess(d, v, k)  # initial gap estimate

        for iteration in range(max_iter):
            # Evaluate g_k(y) and g_k'(y) via triangular FMM
            # Split into lower/upper triangular Cauchy matrices
            # C_L w and C_U w evaluated separately to avoid cancellation
            g_val, g_deriv = triangular_fmm_evaluate(delta, v, y_k, k)

            # Modified Newton update
            dy = -g_val / g_deriv
            y_k += dy

            # Check convergence
            if abs(g_val) < tau * n * (1 + abs(psi(y_k)) + abs(phi(y_k))):
                break

        eigenvalues[k] = d[k] + y_k  # recover eigenvalue from gap

    # Last eigenvalue: simple rational interpolation
    eigenvalues[-1] = solve_last_eigenvalue(d, v)

    # Compute structured eigenvectors via Löwner's formula + FMM
    v_hat = lowner_formula(eigenvalues, d)  # O(n log n) via FMM
    Q_cauchy = cauchy_eigenvector_matrix(v_hat, d, eigenvalues)

    return eigenvalues, Q_cauchy
```

**Key Implementation Insights:**

1. **Balancing is critical**: Without the $\|B_i\|_2$ balancing in the dividing stage, HSS generator norms grow as $O(\beta^{2^l})$ — exponentially in tree depth — causing overflow. With balancing, growth is bounded by $O(n\beta)$

2. **Triangular FMM**: Standard FMM for the Cauchy kernel $\kappa(s,t) = \frac{1}{s-t}$ evaluates the full matrix-vector product. The triangular variant evaluates $C_L \mathbf{w}$ and $C_U \mathbf{w}$ separately, which is essential because the secular equation splitting $f = 1 + \psi_k + \phi_k$ separates positive and negative terms

3. **Local shifting**: The key insight is that for far-field interactions (well-separated eigenvalues), no shift is needed since $x_k - d_j$ can be computed accurately. Shifts are only applied locally (near-field), where $\delta_{jk} - y_k$ replaces $d_j - x_k$. This preserves the FMM's hierarchical structure

4. **Cauchy-like eigenmatrix**: Each intermediate eigenmatrix $\hat{Q}_i$ is stored implicitly as the product of $r$ Cauchy-like matrices (5 vectors each), enabling $O(rn \log n)$ application via FMM instead of $O(n^2)$ dense multiplication

5. **Deflation**: When $|v_k| < \tau$ or eigenvalues are too close, deflation skips costly secular equation solves; a user-tunable $\tau$ trades accuracy for speed

6. **Available implementation**: Matlab code at https://www.math.purdue.edu/~xiaj

## References

- Ou, X. & Xia, J. (2021). SuperDC: Stable superfast divide-and-conquer eigenvalue decomposition. arXiv:2108.04209. Published in *SIAM Journal on Scientific Computing*, 44(4), A2283-A2311, 2022.
- Vogel, J., Xia, J., Cauley, S., & Balakrishnan, V. (2016). Superfast divide-and-conquer method and perturbation analysis for structured eigenvalue solutions. *SIAM Journal on Scientific Computing*, 38(3), A1358-A1382.
- Gu, M. & Eisenstat, S. C. (1995). A divide-and-conquer algorithm for the symmetric tridiagonal eigenproblem. *SIAM Journal on Matrix Analysis and Applications*, 16(1), 172-191.
- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices. *Numerical Linear Algebra with Applications*, 17(6), 953-976.

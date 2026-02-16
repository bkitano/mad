# 118: Sparsified Nested Dissection (spaND)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Cambier, Chen, Boman, Rajamanickam, Tuminaro, & Darve (2020), SIAM J. Matrix Anal. Appl.
**Paper**: [papers/sparsified-nested-dissection.pdf]
**Documented**: 2026-02-15

## Description

Sparsified Nested Dissection (spaND) is an approximate factorization algorithm for large, sparse, symmetric positive definite (SPD) linear systems that achieves near-linear $O(N \log N)$ factorization time — a dramatic improvement over the $O(N^2)$ cost of classical nested dissection (ND) in 3D. The key idea is to interleave standard ND elimination with a **sparsification** step: after eliminating interiors at each level of the elimination tree, the algorithm uses low-rank approximations to shrink the size of all separator interfaces *without introducing any fill-in*.

The central insight is that separators between well-separated clusters have low-rank off-diagonal coupling. Rather than storing these couplings in a hierarchical low-rank format (like HSS or $\mathcal{H}$-matrices) and performing expensive dense algebra on large fronts, spaND directly *eliminates* the redundant degrees of freedom from the separators early on, keeping all fronts small and sparse throughout the factorization.

Three operations alternate at each tree level:
1. **Eliminate** interiors via block Cholesky factorization
2. **Scale** each interface by its diagonal block Cholesky factor (block-diagonal preconditioning)
3. **Sparsify** each interface using low-rank approximation via orthogonal transformations, splitting DOFs into "coarse" (retained) and "fine" (eliminated) — crucially, this creates no fill-in

The orthogonal variant (OrthS) is provably guaranteed to preserve SPD structure, meaning the preconditioner never breaks down.

## Mathematical Form

**Core Factorization:**

The algorithm produces an approximate factorization:

$$
M^\top A M \approx I
$$

where $M$ is assembled as a product of elementary transformations:

$$
M = \prod_{\ell=1}^{L} \left( \prod_{s \in S_\ell} E_s^\top \prod_{p \in C_\ell} S_p^\top \prod_{p \in C_\ell} Q_p \right)
$$

**Key Definitions:**

- $A \in \mathbb{R}^{N \times N}$ — SPD sparse matrix
- $S_\ell$ — Set of ND separators at level $\ell$
- $C_\ell$ — Set of interface clusters at level $\ell$
- $E_s$ — Elimination operator for interior/separator $s$
- $S_p$ — Block-diagonal scaling operator for interface $p$
- $Q_p$ — Orthogonal sparsification operator for interface $p$
- $\varepsilon$ — Approximation tolerance (controls accuracy vs. cost)

**Elimination Step:**

For a separator $s$ with neighbors $n$ and disconnected nodes $w$, the block-arrowhead form is:

$$
A = \begin{bmatrix} A_{ss} & & A_{sn} \\ & A_{ww} & A_{wn} \\ A_{ns} & A_{nw} & A_{nn} \end{bmatrix}
$$

Compute $L_s L_s^\top = A_{ss}$ (Cholesky) and define:

$$
E_s = \begin{bmatrix} L_s^{-1} & & \\ -A_{ns} A_{ss}^{-1} & I & \\ & & I \end{bmatrix}
$$

Then $E_s A E_s^\top$ eliminates $s$ with Schur complement $B_{nn} = A_{nn} - A_{ns} A_{ss}^{-1} A_{sn}$.

**Interface Scaling Step:**

For interface $p$ with neighbors $n$, scale by Cholesky factor of diagonal block:

$$
S_p = \begin{bmatrix} L_p^{-1} & \\ & I \end{bmatrix}, \quad S_p A S_p^\top = \begin{bmatrix} I & C_{pn} \\ C_{np} & A_{nn} \end{bmatrix}
$$

This normalizes $A_{pp} = I$, which is critical for controlling approximation error.

**Sparsification Step (Orthogonal variant):**

Compute a rank-revealing decomposition of $A_{pn}$ (after scaling, so $A_{pp} = I$):

$$
A_{pn} = Q_{pc} W_{cn} + Q_{pf} W_{fn}, \quad \|W_{fn}\|_2 = O(\varepsilon)
$$

where $Q_{pp} = [Q_{pf} \quad Q_{pc}]$ is a square orthogonal matrix splitting DOFs into:
- $c$ ("coarse") — retained skeleton DOFs with $O(1)$ connections to neighbors
- $f$ ("fine") — redundant DOFs with only $O(\varepsilon)$ connections

Apply the orthogonal change of basis:

$$
Q_p = \begin{bmatrix} Q_{pp} & \\ & I \end{bmatrix}
$$

$$
Q_p^\top A Q_p = \begin{bmatrix} I & & O(\varepsilon) \\ & I & W_{cn} \\ O(\varepsilon) & W_{cn}^\top & A_{nn} \end{bmatrix}
$$

Dropping the $O(\varepsilon)$ entries effectively eliminates $f$ without fill-in. The separator size decreases from $|p|$ to $|c|$.

**SPD Preservation Theorem:**

For SPD matrix $A$ and any orthogonal low-rank approximation $A_{pn} = Q_{pf} W_{fn} + Q_{pc} W_{cn}$:

$$
B_p = \begin{bmatrix} I & W_{cn} \\ W_{cn}^\top & A_{nn} \end{bmatrix} \succeq 0
$$

The sparsified matrix remains SPD regardless of $\varepsilon$, because the approximation error $S_B = S_A + W_{fn}^\top W_{fn} \succeq S_A$ (the Schur complement only increases).

**Error Bound:**

For the orthogonal+scaling variant (OrthS):

$$
\|E_{nn}\|_2 \leq \|Y_2\|_2^2 = O(\varepsilon^2)
$$

This quadratic error improvement (vs. $O(\varepsilon^2 \|C_{ff}^{-1}\|_2)$ for interpolative without scaling) is key to the algorithm's practical effectiveness.

## Complexity

**3D PDE problems ($N$ unknowns from $n \times n \times n$ grid, $N = n^3$):**

| Operation | Classical ND | spaND |
|-----------|-------------|-------|
| Factorization | $O(N^2)$ | $O(N \log N)$ |
| Solve (apply) | $O(N^{4/3})$ | $O(N)$ |
| Memory | $O(N \log N)$ to $O(N^{4/3})$ | $O(N)$ |
| Top separator size | $O(N^{2/3})$ | $O(N^{1/3})$ |

**2D PDE problems ($N = n^2$):**

| Operation | Classical ND | spaND |
|-----------|-------------|-------|
| Factorization | $O(N^{3/2})$ | $O(N \log N)$ |
| Solve (apply) | $O(N \log N)$ | $O(N)$ |
| Memory | $O(N \log N)$ | $O(N)$ |

**Key complexity assumption:** The rank of separator interfaces (off-diagonal blocks for well-separated clusters) scales as $s_\ell \in O(2^{-\ell/3} N^{1/3})$ — comparable to the diameter of the separator, consistent with the fast multipole method and Green's function decay.

**Factorization cost derivation:**

$$
t_{\text{spaND,fact}} \in O\left(\sum_{\ell=1}^{L} 2^\ell \cdot 2^{-\ell} N \right) = O(N \log N)
$$

vs. classical ND: $t_{\text{ND,fact}} \in O\left(\sum_{\ell=0}^{L} 2^{-\ell} N^2 \right) = O(N^2)$

## Applicability

1. **Large Sparse Systems in Neural Network Training**: When second-order optimization methods (Newton, natural gradient) produce large sparse linear systems, spaND provides a near-linear preconditioner. This is especially relevant for physics-informed neural networks (PINNs) where the Hessian inherits PDE structure.

2. **Gaussian Process Inference**: GP kernel matrices from spatial data have HSS-like structure. spaND applied to the sparse precision matrix (inverse covariance) enables $O(N \log N)$ inference, critical for GP layers in deep learning.

3. **Graph Neural Networks**: The Laplacian and related operators on graphs produce sparse SPD systems. spaND can serve as a fast preconditioner for spectral graph convolutions.

4. **State Space Models**: Discretized PDE operators that serve as state transition matrices in SSMs have the nested dissection structure. spaND enables fast implicit solves.

5. **Attention as Sparse Solve**: When attention kernels are sparse and SPD (e.g., from local + low-rank approximations), spaND can be used for fast inversion in the linear attention framework.

6. **Scientific Machine Learning**: Any application coupling neural networks with PDE solvers benefits from spaND's near-linear sparse solve capability.

## Limitations

1. **SPD Requirement**: The OrthS variant (with provable stability) requires SPD matrices. The interpolative variant can handle more general cases but may break down.

2. **Approximate, Not Exact**: spaND produces an approximate factorization controlled by tolerance $\varepsilon$. It is primarily a preconditioner, not a direct solver — CG iterations are still needed.

3. **Geometry Helps**: While purely algebraic (only needs the sparse matrix), performance improves with geometric information for clustering and separator computation.

4. **Sequential Bottleneck**: The tree-based elimination has $O(\log N)$ sequential levels, limiting parallelism to within-level operations.

5. **Rank Growth in 3D**: For highly oscillatory problems (e.g., Helmholtz at high frequency), off-diagonal ranks may grow faster than $O(N^{1/3})$, degrading the complexity advantage.

6. **Constant Factors**: While asymptotically near-linear, the hidden constants are larger than simple preconditioners like ILU(0), making spaND most beneficial for large-scale problems.

## Implementation Notes

```python
# Pseudocode: spaND Algorithm (OrthS variant)
def spaND(A, L, epsilon):
    """
    Sparsified Nested Dissection for SPD matrix A.

    Args:
        A: sparse SPD matrix (N x N)
        L: number of ND levels
        epsilon: approximation tolerance

    Returns:
        M: product of elementary transformations s.t. M^T A M ≈ I
           (can be used as preconditioner: x ≈ M M^T b)

    Complexity: O(N log N) factorization, O(N) solve
    """
    # Step 1: Modified ND ordering with interface tracking
    C = modified_nd_ordering(A, L)
    # C[v] = (separator, left_neighbor, right_neighbor) for each vertex v

    M = []  # accumulate transformations

    for level in range(L, 0, -1):  # bottom-up through ND tree
        # Step 2: Eliminate all interiors at this level
        for s in separators_at_level(level):
            L_s = cholesky(A[s, s])
            E_s = build_elimination(L_s, A, s)
            A = E_s @ A @ E_s.T  # Schur complement update
            M.append(E_s)

        # Step 3: Scale each interface
        for p in interfaces_at_level(level):
            L_p = cholesky(A[p, p])
            S_p = build_scaling(L_p, p)
            A = S_p @ A @ S_p.T  # now A[p,p] = I
            M.append(S_p)

        # Step 4: Sparsify each interface (key step!)
        for p in interfaces_at_level(level):
            n = neighbors(p)  # indices of neighboring DOFs

            # Low-rank approximation of A[p, n]
            # (which equals A_pn after scaling)
            Q_pp, rank = rank_revealing_qr(A[p, n], epsilon)
            # Q_pp is orthogonal, splitting p into:
            #   f = fine (first |p|-rank cols) -> weak connections
            #   c = coarse (last rank cols) -> strong connections

            Q_p = build_sparsification(Q_pp, p)
            A = Q_p.T @ A @ Q_p

            # Drop O(epsilon) entries connecting f to n
            # This eliminates f without fill-in!
            f_indices = p[:len(p) - rank]
            A[f_indices, n] = 0
            A[n, f_indices] = 0

            M.append(Q_p)

        # Step 5: Merge clusters for next level
        merge_clusters(level)

    return M


def spaND_solve(M, b):
    """
    Apply spaND preconditioner: x ≈ M @ M^T @ b
    Complexity: O(N) since M is a product of sparse/structured ops.
    """
    # Forward pass: apply M^T
    y = b.copy()
    for transform in reversed(M):
        y = transform.T @ y

    # Backward pass: apply M
    x = y.copy()
    for transform in M:
        x = transform @ x

    return x


def preconditioned_cg(A, b, M, tol=1e-12):
    """
    CG with spaND preconditioner.
    Typically converges in O(1) iterations for smooth problems.
    """
    x = zeros_like(b)
    r = b - A @ x
    z = spaND_solve(M, r)
    p = z.copy()

    for k in range(max_iter):
        Ap = A @ p
        alpha = dot(r, z) / dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap

        if norm(r_new) / norm(b) < tol:
            break

        z_new = spaND_solve(M, r_new)
        beta = dot(r_new, z_new) / dot(r, z)
        p = z_new + beta * p
        r, z = r_new, z_new

    return x
```

**Key Implementation Insights:**

1. **Orthogonal vs. Interpolative**: The orthogonal variant (OrthS) is preferred because it guarantees SPD preservation and achieves $O(\varepsilon^2)$ error vs. $O(\varepsilon^2 \|C_{ff}^{-1}\|)$ for interpolative — a significant practical difference.

2. **Block Diagonal Scaling**: The scaling step $S_p$ normalizes each interface to $A_{pp} = I$, which is critical for controlling the approximation error and reducing CG iterations by 2-10x.

3. **RRQR for Low-Rank**: The rank-revealing QR factorization is used for the low-rank approximation, with rank determined by the tolerance $\varepsilon$ on the diagonal of R: $|R_{ii}|/|R_{11}| \geq \varepsilon$.

4. **Fill-In Free**: The sparsification step creates no fill-in because the fine DOFs $f$ are only connected to the parent interface $p$, not to any neighbors — this is ensured by the nested dissection ordering.

5. **Practical CG Counts**: For $\varepsilon = 10^{-4}$, typical CG iteration counts are 5-15 for 2D problems and 10-30 for 3D problems, roughly independent of $N$.

## References

- Cambier, L., Chen, C., Boman, E. G., Rajamanickam, S., Tuminaro, R. S., & Darve, E. (2020). An algebraic sparsified nested dissection algorithm using low-rank approximations. *SIAM Journal on Matrix Analysis and Applications*, 41(2), 715-746. arXiv:1901.02971
- Ho, K. L., & Ying, L. (2016). Hierarchical interpolative factorization for elliptic operators: differential equations. *Communications on Pure and Applied Mathematics*, 69(8), 1415-1451.
- George, A. (1973). Nested dissection of a regular finite element mesh. *SIAM Journal on Numerical Analysis*, 10(2), 345-363.
- Xia, J., & Xin, Z. (2018). Effective and robust preconditioning of general SPD matrices via structured incomplete factorization. *SIAM Journal on Matrix Analysis and Applications*, 38(4), 1298-1322.

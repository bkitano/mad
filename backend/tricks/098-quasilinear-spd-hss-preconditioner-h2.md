# 098: Quasilinear SPD HSS Preconditioner from H² Representation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Xing, Huang, Chow — "Efficient construction of an HSS preconditioner for symmetric positive definite $\mathcal{H}^2$ matrices" (2021)
**Paper**: [papers/hss-preconditioner-h2-construction.pdf]
**Documented**: 2026-02-15

## Description

This trick constructs a symmetric positive definite (SPD) HSS preconditioner for kernel matrices in quasilinear $O(rN \log N)$ time by leveraging an existing $\mathcal{H}^2$ representation of the matrix. The standard approach to building an SPD HSS approximation requires quadratic $O(N^2 r)$ cost because it must access all matrix entries for the "scaling-and-compression" technique that guarantees positive definiteness. The key insight is that if the matrix is already available in $\mathcal{H}^2$ form (which provides linear-cost matrix-vector products), the expensive scaling and compression operations at each level of the HSS hierarchy can be reduced to operations on small $O(r) \times O(r)$ matrices by reusing the $\mathcal{H}^2$ block structure. Specifically, the HSS coefficient matrices $B_{ij}$ that dominate the cost can be computed directly from the $\mathcal{H}^2$ representation's Type-3 blocks using $B_{ij} = (\Phi_i U_i^{\mathcal{H}^2}) B_{ij}^{\mathcal{H}^2} (\Phi_j U_j^{\mathcal{H}^2})^T$, avoiding the recursive descent to leaf level. The resulting preconditioner preserves positive definiteness — crucial for use with conjugate gradient (CG) methods — and reduces CG iterations by orders of magnitude compared to diagonal preconditioning.

## Mathematical Form

**Scaling-and-Compression Technique:**

At each level $k$ of the partition tree, off-diagonal blocks $A_{ij}^{(k-1)}$ are compressed into low-rank form via:

1. **Scale:** Factor diagonal blocks $A_{ii}^{(k-1)} = S_i S_i^T$ (Cholesky), then scale:

$$
A_{ij}^{(k-1)} \xrightarrow{\text{scale}} C_{ij}^{(k-1)} = S_i^{-1} A_{ij}^{(k-1)} S_j^{-T}
$$

This makes diagonal blocks identity, enabling SPD-preserving compression.

2. **Compress:** Find orthonormal $V_i$ to approximate the off-diagonal block row:

$$
C_{ii^c}^{(k-1)} \xrightarrow{\text{compress}} V_i V_i^T C_{ii^c}^{(k-1)} V_j V_j^T
$$

3. **Scale back:** The final low-rank approximation is:

$$
A_{ij}^{(k-1)} \approx A_{ij}^{(k)} = S_i V_i (V_i^T C_{ij}^{(k-1)} V_j) V_j^T S_j^T = U_i B_{ij} U_j^T
$$

where $U_i = S_i V_i$ is the basis matrix and $B_{ij} = V_i^T C_{ij}^{(k-1)} V_j$ is the coefficient matrix.

**SPD Guarantee:**

The level-$k$ approximation satisfies:

$$
A^{(k)} = \operatorname{diag}(U_i W_i) A^{(k-1)} \operatorname{diag}(U_i W_i)^T + \operatorname{diag}(S_i (I - V_i V_i^T) S_i^T)
$$

where $W_i = V_i^T S_i^{-1}$. Since $A^{(k-1)}$ is SPD (by induction) and $(I - V_i V_i^T) \succeq 0$, $A^{(k)}$ is SPD.

**Generalized Symmetric Factorization for Nonbinary Trees:**

For a node $i$ at level $k$ with children $i_1, \ldots, i_m$, the diagonal block decomposes as:

$$
A_{ii}^{(k-1)} = \begin{bmatrix} S_{i_1} & & \\ & \ddots & \\ & & S_{i_m} \end{bmatrix} (I + \mathbf{V}_i \mathbf{B}_{ii} \mathbf{V}_i^T) \begin{bmatrix} S_{i_1} & & \\ & \ddots & \\ & & S_{i_m} \end{bmatrix}^T
$$

where $\mathbf{B}_{ii}$ is the block matrix of children's coefficient matrices and $\mathbf{V}_i = \operatorname{diag}(V_{i_1}, \ldots, V_{i_m})$.

The symmetric factorization $A_{ii}^{(k-1)} = S_i S_i^T$ uses:

$$
\bar{S}_i = I + \mathbf{V}_i ((I + \mathbf{B}_{ii})^{1/2} - I) \mathbf{V}_i^T
$$

$$
S_i = \begin{bmatrix} S_{i_1} & & \\ & \ddots & \\ & & S_{i_m} \end{bmatrix} \bar{S}_i
$$

with $\bar{S}_i^{-1} = I + \mathbf{V}_i ((I + \mathbf{B}_{ii})^{-1/2} - I) \mathbf{V}_i^T$.

**Quasilinear Acceleration via $\mathcal{H}^2$ Representation:**

The $\mathcal{H}^2$ representation compresses blocks as:

$$
A_{ij} = U_i^{\mathcal{H}^2} B_{ij}^{\mathcal{H}^2} (U_j^{\mathcal{H}^2})^T
$$

Define the nested scaling operator:

$$
\Phi_i = \begin{cases} V_i^T S_i^{-1} & \text{if } i \text{ is a leaf node} \\ \bar{V}_i^T (I + \mathbf{B}_{ii})^{-1/2} \begin{bmatrix} \Phi_{i_1} & & \\ & \ddots & \\ & & \Phi_{i_m} \end{bmatrix} & \text{if } i \text{ has children } i_1, \ldots, i_m \end{cases}
$$

**Key Shortcut:** For Type-3 blocks (blocks already compressed in $\mathcal{H}^2$ form), the HSS coefficient matrix can be computed directly:

$$
B_{ij} = (\Phi_i U_i^{\mathcal{H}^2}) B_{ij}^{\mathcal{H}^2} (\Phi_j U_j^{\mathcal{H}^2})^T
$$

The quantity $\Phi_i U_i^{\mathcal{H}^2}$ is computed recursively:

$$
\Phi_i U_i^{\mathcal{H}^2} = \bar{V}_i^T (I + \mathbf{B}_{ii})^{-1/2} \begin{bmatrix} \Phi_{i_1} & & \\ & \ddots & \\ & & \Phi_{i_m} \end{bmatrix} \begin{bmatrix} U_{i_1}^{\mathcal{H}^2} & & \\ & \ddots & \\ & & U_{i_m}^{\mathcal{H}^2} \end{bmatrix} R_i^{\mathcal{H}^2}
$$

All matrices in this product are small ($O(r) \times O(r)$), so each multiplication costs $O(r^3)$.

**Randomized Basis Computation:**

The basis matrices $V_i$ and $\bar{V}_i$ are computed via a randomized algorithm. Instead of forming $\Lambda_{ii^c}$ explicitly, the algorithm uses $\mathcal{H}^2$ matrix-vector products:

$$
Y^{(k)} = (A - \operatorname{diag}(\{A_{ii}\}_{i \in \text{lvl}(k)})) \Omega, \quad k = 1, 2, \ldots, L-1
$$

where $\Omega \in \mathbb{R}^{N \times (r+p)}$ is a random Gaussian matrix. Each $Y^{(k)}$ costs $O((r+p)N)$ via the $\mathcal{H}^2$ representation. Then $\Lambda_{ii^c} \Omega_{i^c}$ is extracted from row subsets of $Y^{(k)}$ and transformed level-by-level using Algorithm 5.2.

**Key Definitions:**

- $A \in \mathbb{R}^{N \times N}$ — SPD kernel matrix
- $r$ — maximum HSS approximation rank
- $L$ — number of levels in the partition tree $\mathcal{T}$
- $m$ — branching factor of the partition tree
- $\mathcal{F}_i$ — far-field set of node $i$ (nodes in $\text{lvl}(k) \setminus \{i\}$ for HSS)
- $U_i, R_i, B_{ij}$ — HSS generators (basis, transfer, coefficient matrices)
- $S_i$ — Cholesky-like factor satisfying $A_{ii}^{(k-1)} = S_i S_i^T$
- $\Phi_i$ — nested scaling operator combining compression and scaling across levels

## Complexity

| Operation | Standard SPD HSS | Quasilinear SPD HSS (this trick) |
|-----------|-----------------|----------------------------------|
| HSS construction | $O(N^2 r)$ | $O(r N \log N)$ |
| Per-level computation | $O(N \cdot r^2)$ | $O(r^3 \cdot |\text{lvl}(k)|)$ |
| $V_i$ computation (randomized) | $O(N^2 r)$ | $O(r N \log N)$ |
| $B_{ij}$ computation | $O(N r^2)$ per level | $O(r^3)$ per block |
| Storage (HSS representation) | $O(Nr)$ | $O(Nr)$ |

**Memory:** $O(Nr)$ for the HSS representation, same as standard.

**Matrix-vector products needed:** $O(\log N)$ products with $A$ (one per level), each costing $O(rN)$ via $\mathcal{H}^2$ representation.

**Preconditioner application cost:** $O(Nr^2)$ per CG iteration via ULV factorization of the HSS approximation.

## Applicability

- **Kernel attention layers**: SPD kernel matrices (Gaussian, Matérn, polynomial kernels) arising in attention-like mechanisms can be preconditioned for fast iterative solves; the preconditioner construction is quasilinear instead of quadratic
- **Gaussian process layers**: Neural networks with GP layers require solving $K^{-1} y$ and computing $\log \det K$; an SPD HSS preconditioner accelerates CG for the solve and enables stochastic trace estimation for the log-determinant
- **Graph neural networks**: Laplacian and kernel matrices on graphs with geometric structure (meshes, point clouds) naturally have $\mathcal{H}^2$ structure; this trick enables efficient preconditioning for spectral methods
- **Physics-informed neural networks (PINNs)**: Discretized PDE operators (elliptic, Helmholtz) produce kernel matrices with hierarchical low-rank structure; fast preconditioner construction accelerates the linear system solves in PINN training
- **Neural PDE solvers**: Architectures like Neural-HSS that parameterize solution operators using HSS structure benefit from fast preconditioner construction for iterative refinement
- **Large-scale Nyström attention**: When approximating attention via Nyström with $m$ landmarks, the $m \times m$ kernel matrix is SPD; if $m$ is large, this trick provides an efficient preconditioner

## Limitations

- Requires an existing $\mathcal{H}^2$ representation of the matrix; constructing $\mathcal{H}^2$ from scratch still costs $O(Nr)$ or $O(Nr \log N)$ but uses different algorithms
- The approximation accuracy of the preconditioner depends on the HSS rank $r$; for ill-conditioned matrices, larger $r$ may be needed, increasing the $O(r^3)$ per-block cost
- The scaling-and-compression technique uses $(I + \mathbf{B}_{ii})^{\pm 1/2}$ which requires eigendecomposition of the $mr \times mr$ matrix $\mathbf{B}_{ii}$; this costs $O(m^3 r^3)$ per nonleaf node
- The SPD guarantee relies on exact arithmetic; in floating point, near-singular $\mathbf{B}_{ii}$ matrices can cause loss of positive definiteness
- Currently limited to SPD matrices; non-symmetric or indefinite matrices require different preconditioning strategies
- The $\mathcal{H}^2$ representation must use the same partition tree as the target HSS representation; converting between different tree structures adds overhead

## Implementation Notes

```python
# Pseudocode for quasilinear SPD HSS preconditioner construction

import numpy as np
from scipy.linalg import cholesky, sqrtm

def build_spd_hss_preconditioner(A_h2, rank_r, oversample_p=5):
    """
    Construct SPD HSS preconditioner from H^2 representation.

    Args:
        A_h2: H^2 matrix representation with {U_i^H2, B_ij^H2, R_i^H2}
        rank_r: target HSS rank
        oversample_p: oversampling parameter for randomized compression

    Returns:
        HSS preconditioner {A_ii, U_i, B_ij, R_i} guaranteed SPD
    """
    N = A_h2.size
    tree = A_h2.partition_tree
    L = tree.num_levels

    # Step 1: Compute random sketches Y^(k) for all levels
    # Y^(k) = (A - diag(A_ii at level k)) * Omega
    Omega = np.random.randn(N, rank_r + oversample_p)
    Y = {}
    for k in range(1, L):
        # Fast H^2 matvec: O(rN) per product
        Y[k] = A_h2.matvec_off_diagonal(Omega, level=k)

    # Step 2: Level-by-level HSS construction
    Phi = {}  # nested scaling operators
    S = {}    # Cholesky-like factors
    V = {}    # compression bases
    V_bar = {}  # small bases
    B = {}    # coefficient matrices
    U = {}    # basis matrices
    R = {}    # transfer matrices

    # Leaf level (k=1): direct computation
    for i in tree.leaves():
        # Compute S_i from Cholesky of A_ii
        S[i] = cholesky(A_h2.diagonal_block(i))
        Phi[i] = np.linalg.solve(S[i], np.eye(S[i].shape[0]))  # S_i^{-1}

        # Compute V_i via randomized algorithm
        # Extract relevant rows of Y^(1)
        T_i = Phi[i] @ Y[1][tree.indices(i), :]

        # Pivoted QR for rank-r approximation
        V[i], _ = np.linalg.qr(T_i)
        V[i] = V[i][:, :rank_r]

        # Set basis matrix
        U[i] = S[i] @ V[i]

        # Compute Phi_i = V_i^T @ S_i^{-1}
        Phi[i] = V[i].T @ np.linalg.solve(S[i], np.eye(S[i].shape[0]))

    # Non-leaf levels (k=2 to L-1)
    for k in range(2, L):
        for i in tree.nodes_at_level(k):
            children = tree.children(i)
            m = len(children)

            # Assemble B_ii from children's coefficient matrices
            B_ii = assemble_children_B(B, children)

            # Compute (I + B_ii)^{+/- 1/2} via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(B_ii)
            sqrt_factor = eigvecs @ np.diag(np.sqrt(1 + eigvals)) @ eigvecs.T
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(1 + eigvals)) @ eigvecs.T

            # Compute V_bar_i via randomized algorithm
            # (using level-by-level transformed sketches)
            T_i = level_by_level_transform(Y, k, i, Phi, children, inv_sqrt)
            V_bar[i], _ = np.linalg.qr(T_i)
            V_bar[i] = V_bar[i][:, :rank_r]

            # KEY SPEEDUP: Compute B_ij directly from H^2 Type-3 blocks
            Phi_U = compute_phi_u_recursive(Phi, A_h2, i, children, inv_sqrt)
            for j in tree.siblings(i):
                if A_h2.is_type3_block(i, j):
                    # O(r^3) instead of recursive descent!
                    B[i, j] = Phi_U[i] @ A_h2.B_h2[i, j] @ Phi_U[j].T

            # Compute transfer matrix R_i
            R[i] = inv_sqrt @ np.diag([Phi[c] for c in children]) @ V_bar[i]

    return HSS_Preconditioner(U=U, B=B, R=R, A_diag=A_h2.diag_blocks)
```

## References

- Xing, Huang, Chow, "Efficient construction of an HSS preconditioner for symmetric positive definite $\mathcal{H}^2$ matrices," arXiv:2011.07632v2, 2021
- Xia, Xi, Gu, "A superfast structured solver for Toeplitz linear systems via randomized sampling," SIAM J. Matrix Anal. Appl. 33(3), pp. 837–858, 2012
- Aminfar, Ambikasaran, Darve, "A fast block low-rank dense solver with applications to finite-element matrices," J. Comput. Phys. 304, pp. 170–188, 2016
- Martinsson, "A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix," SIAM J. Matrix Anal. Appl. 32(4), pp. 1251–1274, 2011
- Ghysels, Li, Rouet, Williams, Napov, "An efficient multicore implementation of a novel HSS-structured multifrontal solver using randomized sampling," SIAM J. Sci. Comput. 38(5), pp. S358–S384, 2016

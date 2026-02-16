# 102: Recursive Skeletonization Fast Direct Solver

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ho & Greengard (2012), SIAM J. Sci. Comput.
**Paper**: [papers/recursive-skeletonization.pdf]
**Documented**: 2026-02-15

## Description

Recursive skeletonization is a fast direct solver for structured linear systems that embeds a hierarchically compressed matrix into a larger but highly structured *sparse* system, enabling the use of existing sparse direct solvers (e.g., UMFPACK) for both factorization and inverse application. The key idea is a two-phase process: (1) multilevel matrix compression via the interpolative decomposition (ID), which identifies "skeleton" degrees of freedom at each level that capture the low-rank off-diagonal interactions, and (2) introduction of auxiliary variables that embed the compressed representation into a sparse system amenable to standard sparse LU factorization.

Unlike traditional HSS solvers that implement custom recursive block elimination, recursive skeletonization leverages well-optimized sparse solver infrastructure, inheriting numerical stability features like pivoting. The method achieves $O(N)$ complexity for 2D problems and $O(N^{3/2})$ for 3D precomputation, with $O(N)$ and $O(N \log N)$ solve-phase costs respectively. The compressed solve is $\sim$100$\times$ faster than FMM-accelerated iterative methods, making it ideal for problems requiring multiple right-hand sides or ill-conditioned systems.

## Mathematical Form

**Block Separability (Core Structure):**

A matrix $A \in \mathbb{C}^{N \times N}$ is *block separable* with $p$ blocks if each off-diagonal submatrix admits a low-rank factorization:

$$
A_{ij} = L_i S_{ij} R_j, \quad i \neq j
$$

where $L_i \in \mathbb{C}^{n_i \times k_i^r}$ depends only on row index $i$, $R_j \in \mathbb{C}^{k_j^c \times n_j}$ depends only on column index $j$, and $k_i^r, k_j^c \ll n_i$. This gives the decomposition:

$$
A = D + LSR
$$

where $D = \text{diag}(A_{11}, \ldots, A_{pp})$ is block diagonal, $L = \text{diag}(L_1, \ldots, L_p)$ and $R = \text{diag}(R_1, \ldots, R_p)$ are block-diagonal projection matrices, and $S \in \mathbb{C}^{K_r \times K_c}$ is dense with zero diagonal blocks.

**Sparse Embedding (Key Innovation):**

Introducing auxiliary variables $\mathbf{z} = R\mathbf{x}$ and $\mathbf{y} = S\mathbf{z}$, the system $A\mathbf{x} = \mathbf{b}$ is rewritten as:

$$
\begin{bmatrix} D & L & \\ R & & -I \\ & -I & S \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \mathbf{y} \\ \mathbf{z} \end{bmatrix} = \begin{bmatrix} \mathbf{b} \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
$$

This system is highly structured and sparse, and can be efficiently factored using standard sparse direct solver technology.

**Interpolative Decomposition (ID):**

The compression uses the rank-$k$ ID: for a numerically low-rank matrix $A \in \mathbb{C}^{m \times n}$,

$$
A \approx BP
$$

where $B \in \mathbb{C}^{m \times k}$ is a subset of columns of $A$ (the *skeleton*), and $P \in \mathbb{C}^{k \times n}$ is a well-conditioned projection matrix containing a $k \times k$ identity submatrix. The ID is rank-revealing and can be computed in $O(mn \log k + k^2 n)$ operations via randomized sampling.

**Multilevel Telescoping Representation:**

Applying recursive skeletonization over $\lambda$ levels of the hierarchy yields:

$$
A \approx D^{(1)} + L^{(1)} \left[ D^{(2)} + L^{(2)} \left( \cdots D^{(\lambda)} + L^{(\lambda)} S R^{(\lambda)} \cdots \right) R^{(2)} \right] R^{(1)}
$$

**Multilevel Sparse Embedding:**

The telescoping representation embeds into the sparse system:

$$
\begin{bmatrix} D^{(1)} & L^{(1)} & & & \\ R^{(1)} & & -I & & \\ & & D^{(2)} & L^{(2)} & \\ & & R^{(2)} & & \ddots \\ & & & \ddots & D^{(\lambda)} & -I \\ & & & & -I & S \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \mathbf{y}^{(1)} \\ \mathbf{z}^{(1)} \\ \vdots \\ \mathbf{y}^{(\lambda)} \\ \mathbf{z}^{(\lambda)} \end{bmatrix} = \begin{bmatrix} \mathbf{b} \\ \mathbf{0} \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \\ \mathbf{0} \end{bmatrix}
$$

**Compressed Inverse:**

The inverse has the telescoping form:

$$
A^{-1} \approx \mathcal{D}^{(1)} + \mathcal{L}^{(1)} \left[ \mathcal{D}^{(2)} + \mathcal{L}^{(2)} \left( \cdots \mathcal{D}^{(\lambda)} + \mathcal{L}^{(\lambda)} \mathcal{S}^{-1} \mathcal{R}^{(\lambda)} \cdots \right) \mathcal{R}^{(2)} \right] \mathcal{R}^{(1)}
$$

where $\mathcal{D} = D^{-1} - D^{-1}L\Lambda R D^{-1}$, $\mathcal{L} = D^{-1}L\Lambda$, $\mathcal{R} = \Lambda R D^{-1}$, $\Lambda = (RD^{-1}L)^{-1}$, and $\mathcal{S} = \Lambda + S$.

## Complexity

| Operation | Naive ($d$-dim) | Recursive Skeletonization |
|-----------|-----------------|---------------------------|
| Matrix compression (2D) | $O(N^3)$ | $O(N)$ |
| Matrix compression (3D) | $O(N^3)$ | $O(N^{3/2})$ |
| Matrix-vector multiply (2D) | $O(N^2)$ | $O(N)$ |
| Matrix-vector multiply (3D) | $O(N^2)$ | $O(N \log N)$ |
| Factorization + solve (2D) | $O(N^3)$ | $O(N)$ |
| Factorization + solve (3D) | $O(N^3)$ | $O(N^{3/2})$ |
| Inverse application (2D) | $O(N^2)$ | $O(N)$ |
| Inverse application (3D) | $O(N^2)$ | $O(N \log N)$ |

**Memory:** $O(N)$ in 2D; $O(N \log N)$ in 3D for the compressed sparse representation.

**Key dimension-dependent scaling:** The interaction rank between adjacent blocks at level $l$ scales as:

$$
k_l \sim \begin{cases} \log n_l & \text{if } d = 1 \\ n_l^{1-1/d} & \text{if } d > 1 \end{cases}
$$

where $d$ is the spatial dimension and $n_l$ is the block size at level $l$.

## Applicability

1. **Structured Attention Solvers**: For attention matrices arising from kernel functions with decay properties (e.g., RBF attention, Gaussian process kernels), recursive skeletonization provides a direct solver that avoids iteration entirely — critical for ill-conditioned attention matrices
2. **Preconditioner Construction**: The compressed inverse serves as an extremely effective preconditioner for iterative methods, with $O(N)$ setup and apply costs in 2D
3. **Multiple Right-Hand-Side Problems**: Once factored, solving for new right-hand sides costs only $O(N)$ in 2D — ideal for batched inference or multi-head attention where the same kernel matrix is applied to different value vectors
4. **State Space Model Discretization**: Computing $\mathbf{x}(t) = \exp(\Delta A)^{-1} \mathbf{b}$ for structured transition matrices arising in SSMs
5. **Generalized FMM**: Acts as a kernel-independent fast multipole method, computing matrix-vector products without requiring knowledge of the underlying kernel
6. **Low-Rank Update Problems**: Direct methods can exploit Sherman-Morrison-Woodbury to update factorizations when the system matrix undergoes low-rank modifications

## Limitations

1. **Geometric Structure Required**: The proxy surface acceleration assumes the matrix arises from evaluating a potential field satisfying Green's identities — purely algebraic matrices may not benefit from the proxy compression
2. **3D Scaling**: In 3D, the $O(N^{3/2})$ precomputation cost is higher than $O(N)$ FMM, making it less competitive for single-solve problems
3. **Memory Overhead**: The sparse embedding introduces auxiliary variables, roughly tripling the system dimension (though sparsity keeps the memory manageable)
4. **Non-Oscillatory Kernels**: Compression efficiency degrades for highly oscillatory kernels (e.g., high-frequency Helmholtz), as the off-diagonal ranks grow with frequency
5. **Pivoting Complexity**: While the sparse solver handles pivoting, analyzing the fill-in and complexity with pivoting is more involved than the no-pivoting case

## Implementation Notes

```python
# Recursive Skeletonization (Ho & Greengard, Algorithm)
def recursive_skeletonize(A, tree, epsilon):
    """
    Compress dense matrix A into sparse embedding via recursive ID.

    Phase 1: Multilevel matrix compression
    Phase 2: Sparse system factorization via UMFPACK

    Args:
        A: dense matrix (N x N)
        tree: hierarchical index tree (quadtree/octree)
        epsilon: target relative precision

    Returns:
        sparse_factor: factored sparse embedding for fast solves
    """
    # Phase 1: Compress level by level (leaves to root)
    for level in range(1, tree.depth + 1):
        blocks = tree.blocks_at_level(level)

        for i in range(len(blocks)):
            # Compute row ID of off-diagonal block row
            # (optionally accelerated via proxy surface)
            off_diag_row = get_off_diagonal_row(A, blocks, i)
            L_i, row_skeletons = interpolative_decomposition(
                off_diag_row.T, epsilon  # column ID = row ID of transpose
            )

            # Compute column ID of off-diagonal block column
            off_diag_col = get_off_diagonal_col(A, blocks, i)
            R_i, col_skeletons = interpolative_decomposition(
                off_diag_col, epsilon
            )

            # Extract skeleton submatrix S_ij
            # S_ij = A[row_skeletons_i, col_skeletons_j]

        # Regroup: skeleton indices become new blocks at next level
        # Compress the skeleton matrix S in the same form

    # Phase 2: Build sparse embedding
    # Stack [D, L, -I; R, 0, 0; 0, -I, S] for each level
    sparse_system = build_multilevel_sparse_embedding(
        D_list, L_list, R_list, S
    )

    # Factor using UMFPACK (handles pivoting automatically)
    sparse_factor = sparse_lu(sparse_system)

    return sparse_factor


def solve_with_skeletonization(sparse_factor, b):
    """
    Apply compressed inverse to right-hand side.

    Cost: O(N) in 2D, O(N log N) in 3D
    """
    # Pad b with zeros for auxiliary variables
    b_padded = pad_with_zeros(b, sparse_factor.n_aux)

    # Solve using pre-computed sparse LU
    x_padded = sparse_factor.solve(b_padded)

    # Extract original unknowns
    return x_padded[:len(b)]


# Proxy surface acceleration for compression
def compress_with_proxy(A, block_i, neighbors, proxy_points):
    """
    Accelerate ID computation using proxy surface.

    Instead of compressing against ALL far-field blocks (global),
    compress against neighbors + proxy surface charges (local).

    Reduces compression cost from O(N * n_i) to O(n_neighbors * n_i).
    """
    # Near-field: actual matrix entries with neighbors
    near_rows = get_neighbor_indices(block_i, neighbors)
    A_near = A[near_rows, block_i.indices]

    # Far-field: proxy charges on enclosing surface Gamma
    A_proxy = kernel(proxy_points, block_i.points)

    # Stack and compute ID
    A_compress = vstack([A_near, A_proxy])
    return interpolative_decomposition(A_compress, epsilon)
```

**Key Implementation Insights:**

1. **Sparse Solver Delegation**: By embedding into a sparse system, the method inherits pivoting and numerical stability from mature sparse solvers like UMFPACK, avoiding custom block-elimination code
2. **Proxy Surface Trick**: Replaces global off-diagonal block access with local neighbor + proxy evaluation, reducing per-block compression cost from $O(N)$ to $O(1)$ (constant depending only on precision)
3. **Sparsification Visualization**: At each level, compression leaves surviving "skeletons" along block boundaries — the matrix progressively sparsifies from a dense cloud to boundary-concentrated points
4. **Solve-Phase Speedup**: The solve phase is $\sim$100$\times$ faster than a single FMM call, enabling massive speedups for problems with multiple right-hand sides ($T_\text{FMM}/T_\text{sv} \sim 100$–$2500$)
5. **FLAM Implementation**: The FLAM (Fast Linear Algebra in MATLAB) package provides a reference implementation at https://github.com/klho/FLAM

## References

- Ho, K. L. & Greengard, L. (2012). A fast direct solver for structured linear systems by recursive skeletonization. *SIAM Journal on Scientific Computing*, 34(5), A2507-A2532. arXiv:1110.3105.
- Martinsson, P. G. & Rokhlin, V. (2005). A fast direct solver for boundary integral equations in two dimensions. *Journal of Computational Physics*, 205, 1-23.
- Gillman, A., Young, P. M., & Martinsson, P. G. (2012). A direct solver with $O(N)$ complexity for integral equations on one-dimensional domains. *Frontiers of Mathematics in China*, 7(2), 217-247.
- Cheng, H., Gimbutas, Z., Martinsson, P. G., & Rokhlin, V. (2005). On the compression of low rank matrices. *SIAM Journal on Scientific Computing*, 26(4), 1389-1404.

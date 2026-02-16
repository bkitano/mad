# 138: ULV Factorization HSS Solver

**Category**: decomposition
**Gain type**: efficiency
**Source**: Chandrasekaran, Gu, & Pals (2006), SIAM J. Matrix Anal. Appl.
**Paper**: [papers/ulv-hss-solver.pdf]
**Documented**: 2026-02-15

## Description

The ULV factorization is a fast, numerically backward-stable direct solver for linear systems $Ax = b$ where $A$ is in Hierarchically Semiseparable (HSS) form. The algorithm computes an implicit $ULV^H$ decomposition — where $U$ and $V$ are unitary and $L$ is lower-triangular — by recursively compressing off-diagonal blocks via unitary transformations, solving small diagonal subsystems, and merging sibling nodes up the HSS tree. The entire factorization and solve achieve $O(r^2 n)$ complexity (linear in $n$ for fixed HSS rank $r$), compared to $O(n^3)$ for standard Gaussian elimination.

The key insight is a two-phase recursive procedure at each tree level: (1) **compress** — apply unitary rotations $q_{k,i}$ from the left and $w_{k,i}$ from the right to zero out rows/columns in the off-diagonal generators, reducing the system size; (2) **merge** — combine sibling block rows/columns into a parent node with a smaller HSS representation. After recursing to the root, a small dense system remains. The solution is then recovered by back-substituting through the unitary transformations.

## Mathematical Form

**HSS Representation (Multi-level):**

For a matrix $A$ with $K+1$ levels in its HSS tree, each leaf node $\text{Node}(K, i)$ has:
- Diagonal block $D_{K;i}$
- Row generators $U_{K;i}$, column generators $V_{K;i}$
- Translation operators $R_{k;i}$, $W_{k;i}$ connecting levels
- Coupling matrices $B_{k;2i-1,2i}$, $B_{k;2i,2i-1}$ between siblings

with the nested basis property:

$$
U_{k-1;j} = \begin{pmatrix} U_{k;2j} R_{k;2j-1} \\ U_{k;2j} R_{k;2j} \end{pmatrix}, \quad V_{k-1;j} = \begin{pmatrix} V_{k;2j} W_{k;2j-1} \\ V_{k;2j} W_{k;2j} \end{pmatrix}
$$

**Compression Step (Leaf Level):**

For each block row $i$ at the finest level $K$, find a unitary $q_{K;i}$ such that:

$$
\bar{U}_{K;i} = q_{K;i}^H U_{K;i} = \begin{pmatrix} 0 \\ \hat{U}_{K;i} \end{pmatrix} \quad \begin{matrix} m_i - n_{K;i} \\ n_{K;i} \end{matrix}
$$

This zeros out the top $m_i - n_{K;i}$ rows of the generator, decoupling those rows from off-diagonal interactions.

**Diagonal Triangularization:**

Pick a unitary $w_{K;i}$ to lower-triangularize the transformed diagonal block:

$$
\bar{D}_{K;i} = (q_{K;i}^H D_{K;i}) w_{K;i}^H = \begin{pmatrix} D_{K;i;1,1} & 0 \\ D_{K;i;2,1} & D_{K;i;2,2} \end{pmatrix} \quad \begin{matrix} m_i - n_{K;i} & n_{K;i} \\ \end{matrix}
$$

**Partial Solve and Elimination:**

The first $m_i - n_{K;i}$ equations decouple:

$$
D_{K;i;1,1} z_{K;i} = \beta_{K;i}
$$

Solve for $z_{K;i}$ via forward substitution in $O(m_i^2)$ time, then subtract to obtain a reduced system $\hat{A}\hat{x} = \hat{b}$ that has the same HSS structure but with one fewer level.

**Merge Step (Internal Nodes):**

When no compression is possible, merge sibling nodes. For siblings $(2i-1, 2i)$:

$$
\hat{D}_{K-1;i} = \begin{pmatrix} D_{K;2i-1;2,2} & U_{K;2i-1} B_{K;2i,2i-1} V_{K;2i-1}^H \\ U_{K;2i} B_{K;2i,2i-1} V_{K;2i-1}^H & D_{K;2i;2,2} \end{pmatrix}
$$

$$
\hat{U}_{K-1;i} = \begin{pmatrix} U_{K;2i-1} R_{K;2i-1} \\ U_{K;2i} R_{K;2i} \end{pmatrix}, \quad \hat{V}_{K-1;i} = \begin{pmatrix} V_{K;2i-1} W_{K;2i-1} \\ V_{K;2i} W_{K;2i} \end{pmatrix}
$$

The merged system $\hat{A}\hat{x} = \hat{b}$ has an HSS tree with $K$ levels — one fewer than the original.

**Recovery:**

After solving the reduced system, recover the original unknowns:

$$
x_{K;i} = w_{K;i}^H \begin{pmatrix} z_{K;i} \\ \hat{x}_{K;i} \end{pmatrix}
$$

## Complexity

| Operation | Naive | ULV HSS Solver |
|-----------|-------|----------------|
| System solve $Ax = b$ | $O(n^3)$ | $O(r^2 n)$ |
| Factorization | $O(n^3)$ | $O(r^2 n)$ |
| Back-substitution | $O(n^2)$ | $O(rn)$ |
| Fast matvec (up/down sweep) | $O(n^2)$ | $O(rn)$ |
| HSS construction | $O(n^3)$ or $O(n^2)$ | $O(n^2)$ general, $O(n)$ smooth kernels |

**Memory:** $O(rn)$ for the HSS representation vs $O(n^2)$ dense

**Detailed Flop Count (uniform rank $r = p$, leaf size $m_K = 2p$):**

$$
46Np^2 + 37Npr
$$

where $N = n$ is the matrix order. For 2D problems with $n_0 = \alpha N^{1/2}$ and rank growth $\gamma = 1/\sqrt{2}$:

$$
98N^{3/2}\alpha^3 + 70N\alpha^4 + N\alpha^2 r(4\log_2^2 N + 11\log_2 N + 28)
$$

## Applicability

1. **State Space Models (SSMs)**: Solving discretized linear systems $(I - \Delta A)x = b$ where the transition matrix has HSS structure (exponentially decaying off-diagonal ranks)
2. **Structured Attention**: Inverting kernel matrices that arise in linear attention when the kernel has hierarchical low-rank off-diagonal structure
3. **Preconditioning**: Computing approximate factorizations of large structured matrices for iterative solvers in neural network training
4. **Fast Inverse Multipole Method**: The algorithm is a stable version of the fast inverse multipole method, applicable to integral equation solvers used in physics-informed neural networks
5. **Sparse Direct Solvers**: STRUMPACK and similar libraries use HSS-ULV factorization as a key component for solving fill-in blocks that arise during sparse LU factorization
6. **Continuous-Time Models**: Solving linear ODEs with structured transition matrices

## Limitations

1. **HSS Form Required**: The matrix must first be converted to HSS representation, which costs $O(n^2)$ in general (though $O(n)$ for smooth kernels)
2. **Rank Sensitivity**: Complexity depends quadratically on the HSS rank $r$; if $r$ is large, the advantage over dense solvers diminishes
3. **Stability Condition**: Backward stability requires "proper form" — the translation operators must satisfy $\|R_{k;i}\| \leq 1$ and $\|W_{k;i}\| \leq 1$ in a submultiplicative norm
4. **Symmetric Preference**: While the algorithm handles general matrices, symmetric positive definite matrices allow simpler and more efficient variants
5. **Static Structure**: The HSS tree structure must be determined in advance; adaptive refinement requires reconstruction
6. **Sequential Nature**: The recursive compress-merge procedure is inherently sequential along the tree depth ($O(\log n)$ sequential steps)

## Implementation Notes

```python
# Pseudocode for ULV HSS Solver
def ulv_hss_solve(hss_tree, b):
    """
    Solve Ax = b where A is in HSS form via ULV factorization.
    Complexity: O(r^2 * n) where r is HSS rank.

    Three cases at each level:
    1. Compressible off-diagonal blocks -> compress & partial solve
    2. Incompressible blocks -> merge siblings
    3. No off-diagonal blocks (root) -> dense solve
    """
    K = hss_tree.depth

    # Forward elimination: compress and merge bottom-up
    for level in range(K, 0, -1):
        for node_i in hss_tree.nodes_at_level(level):
            m_i = node_i.block_size
            n_ki = node_i.rank  # rank of U_{k;i}

            if n_ki < m_i:  # Case 1: Compressible
                # Compute unitary q to zero out top of U
                q = compute_ql_factorization(node_i.U)  # O(m_i * n_ki^2)

                # Apply q to compress off-diagonal
                node_i.b = q.H @ node_i.b
                node_i.D = q.H @ node_i.D

                # Compute unitary w to triangularize diagonal
                w = compute_lq_factorization(node_i.D[:m_i - n_ki, :])
                node_i.D = node_i.D @ w.H
                node_i.V = w @ node_i.V

                # Partial forward substitution
                z = solve_triangular(node_i.D[:m_i - n_ki, :m_i - n_ki],
                                     node_i.b[:m_i - n_ki])

                # Update RHS: subtract known part
                node_i.b_reduced = node_i.b[m_i - n_ki:] - \
                    node_i.D[m_i - n_ki:, :m_i - n_ki] @ z

                node_i.z = z
                node_i.q = q
                node_i.w = w

            else:  # Case 2: Incompressible -> merge with sibling
                sibling = hss_tree.sibling(node_i)
                parent = hss_tree.parent(node_i)

                # Form merged diagonal block
                parent.D = [[node_i.D, node_i.U @ node_i.B @ sibling.V.H],
                            [sibling.U @ sibling.B @ node_i.V.H, sibling.D]]

                # Form merged generators
                parent.U = [[node_i.U @ node_i.R],
                            [sibling.U @ sibling.R]]
                parent.V = [[node_i.V @ node_i.W],
                            [sibling.V @ sibling.W]]

    # Solve root (dense, small)
    x_root = dense_solve(root.D, root.b)

    # Back-substitution: recover x top-down
    for level in range(1, K + 1):
        for node_i in hss_tree.nodes_at_level(level):
            # Recover original variables
            x_i = node_i.w.H @ concatenate([node_i.z, node_i.x_hat])

    return x
```

**Key Implementation Insights:**

1. **QL not QR**: Use QL factorization (not QR) to zero out the *top* rows of $U_{K;i}$, preserving the bottom rows that carry the rank information
2. **Unitary Stability**: All transformations are unitary (orthogonal), ensuring numerical backward stability — this is the key advantage over non-unitary approaches like Gaussian elimination on the HSS structure
3. **In-Place Updates**: The compressed HSS representation after applying $q$ and $w$ retains HSS structure, enabling recursive application
4. **Parallelism**: At each tree level, all nodes can be processed independently (within-level parallelism), though levels must be processed sequentially
5. **Proper Form**: Ensure $\|R_{k;i}\| \leq 1$ and $\|W_{k;i}\| \leq 1$ during HSS construction for guaranteed backward stability

## References

- Chandrasekaran, S., Gu, M., & Pals, T. (2006). A fast ULV decomposition solver for hierarchically semiseparable representations. *SIAM Journal on Matrix Analysis and Applications*, 28(3), 603-622.
- Chandrasekaran, S., Dewilde, P., Gu, M., Pals, T., & van der Veen, A. (2003). Fast stable solvers for sequentially semi-separable linear systems of equations. *Technical report*, UC Berkeley.
- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices. *Numerical Linear Algebra with Applications*, 17(6), 953-976.
- Gorman, C., Chavez, G., Ghysels, P., Mary, T., Rouet, F.-H., & Li, X. S. (2018). Matrix-free construction of HSS representation using adaptive randomized sampling. *arXiv:1810.04125*.

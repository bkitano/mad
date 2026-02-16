# 131: Tree Quasi-Separable (TQS) Matrices

**Category**: decomposition
**Gain type**: flexibility
**Source**: Govindarajan, Chandrasekaran, Dewilde (2024) — arXiv:2402.13381
**Paper**: [papers/tree-quasi-separable-matrices.pdf]
**Documented**: 2026-02-15

## Description

Tree Quasi-Separable (TQS) matrices are a new class of rank-structured matrices that simultaneously unify and generalize both Sequentially Semi-Separable (SSS) and Hierarchically Semi-Separable (HSS) representations. While SSS matrices correspond to line graphs (chain-like sequential structure) and HSS matrices correspond to binary trees, TQS matrices generalize to arbitrary connected acyclic graphs (trees). This flexibility enables compact representations of inverses of sparse matrices whose adjacency graph is a tree — structures that neither SSS nor HSS can compactly represent in general.

The key insight is that TQS matrices are defined by running an input-driven Linear Time-Varying (LTV) dynamical system on the edges of a tree graph. Each block entry $\text{T}\{i,j\}$ is determined by chaining small "spinner" matrices along the unique path from node $j$ to node $i$ in the tree. TQS matrices inherit all favorable algebraic closure properties: the sum, product, and inverse of TQS matrices are again TQS matrices with controlled rank growth. This makes TQS a strictly more general framework that subsumes SSS and HSS as special cases.

## Mathematical Form

**Core Definition:**

Given a connected acyclic graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$ with nodes $\{1, \ldots, K\}$ and rank-profile $\{\rho_e\}_{e \in \mathbb{E}}$, a TQS matrix $\text{T} \in \mathbb{F}^{M \times N}$ has block entries:

$$
\text{T}\{i,j\} = \begin{cases} \text{D}^i & i = j \\ \text{Out}^i_{p_{\nu-1}} \text{Trans}^{p_{\nu-1}}_{i,p_{\nu-2}} \cdots \text{Trans}^{p_2}_{p_3,p_1} \text{Trans}^{p_1}_{p_2,j} \text{Inp}^j_{p_1} & i \neq j \end{cases}
$$

where $j = p_0 - p_1 - \cdots - p_\nu = i$ is the unique path from node $j$ to node $i$ in the tree.

**Generator Matrices (at each node $k$):**

- $\text{D}^k \in \mathbb{F}^{m_k \times n_k}$ — input-to-output operator
- $\text{Inp}^k_j \in \mathbb{F}^{\rho_{(k,j)} \times n_k}$ — input-to-edge operator (from input $x_k$ to state $h_{(k,j)}$)
- $\text{Trans}^k_{i,j} \in \mathbb{F}^{\rho_{(j,k)} \times \rho_{(k,i)}}$ — edge-to-edge transfer operator (state propagation through node $k$)
- $\text{Out}^k_i \in \mathbb{F}^{m_k \times \rho_{(i,k)}}$ — edge-to-output operator

**Spinner Matrix (transition map at node $k$):**

All generators at node $k$ are organized into the spinner matrix:

$$
\text{S}^k := \begin{pmatrix}
0 & \text{V}^k_{i_1,i_2} & \cdots & \text{V}^k_{i_1,i_p} & \text{W}^k_{i_1,j} & \text{P}^k_{i_1} \\
\text{V}^k_{i_2,i_1} & 0 & \cdots & \text{V}^k_{i_2,i_p} & \text{W}^k_{i_2,j} & \text{P}^k_{i_2} \\
\vdots & & \ddots & & \vdots & \vdots \\
\text{V}^k_{i_p,i_1} & \text{V}^k_{i_p,i_2} & \cdots & 0 & \text{W}^k_{i_p,j} & \text{P}^k_{i_p} \\
\text{U}^k_{j,i_1} & \text{U}^k_{j,i_2} & \cdots & \text{U}^k_{j,i_p} & 0 & \text{B}^k_j \\
\text{C}^k_{i_1} & \text{C}^k_{i_2} & \cdots & \text{C}^k_{i_p} & \text{Q}^k_j & \text{D}^k
\end{pmatrix}
$$

where $i_1, \ldots, i_p \in \mathcal{C}(k)$ are children and $j = \mathcal{P}(k)$ is the parent of node $k$.

**Graph-Induced Rank Structure (GIRS):**

$$
\text{rank}\,\text{T}\{\bar{\mathbb{A}}, \mathbb{A}\} \leqslant c \cdot \mathcal{E}(\mathbb{A}) \quad \forall \mathbb{A} \subset \mathbb{V}
$$

where $\mathcal{E}(\mathbb{A})$ is the number of border edges (edges crossing the partition) and $c = \max_{e \in \mathbb{E}} \rho_e$.

**Key Definitions:**

- $\rho_e$ — rank associated with edge $e \in \mathbb{E}$ (rank-profile)
- $\rho_{\max} := \max_{e \in \mathbb{E}} \rho_e$ — maximum edge rank
- $\text{deg}_{\max} := \max_{i \in \mathbb{V}} \text{deg}(i)$ — maximum node degree
- $\text{H}_e$ — Hankel block associated with edge $e$, whose rank equals $\rho_e$

**Relationship to SSS and HSS:**

- **SSS matrices** are TQS matrices on a *line graph* (chain $1 - 2 - \cdots - K$) with the last node as root
- **HSS matrices** are TQS matrices on a *post-ordered binary tree* where all non-leaf nodes have zero input/output dimensions

## Complexity

| Operation | Dense | TQS (rank $\rho_{\max}$, degree $d_{\max}$) |
|-----------|-------|----------------------------------------------|
| Matrix-vector product | $O(N^2)$ | $O\left(\sum_{i \in \mathbb{V}} (m_i + \sum_j r_{(i,j)})(n_i + \sum_j r_{(j,i)})\right)$ |
| Linear system solve | $O(N^3)$ | $O\left(\sum_{i \in \mathbb{V}} (n_i + \sum_j r_{(i,j)})^3\right)$ |
| Construction (realization) | $O(N^3)$ | $O(K \cdot \rho_{\max}^3)$ (per Hankel block compression) |
| Sum $\text{T}_1 + \text{T}_2$ | $O(N^2)$ | TQS with rank $\{\rho_{1,e} + \rho_{2,e}\}$ |
| Product $\text{T}_1 \text{T}_2$ | $O(N^3)$ | TQS with rank $\{\rho_{1,e} + \rho_{2,e}\}$ |
| Inverse $\text{T}^{-1}$ | $O(N^3)$ | TQS with rank $\{\rho_e\}$ (same!) |

**Memory:** $O\left(\sum_{i \in \mathbb{V}} (m_i + \sum_j r_{(i,j)})(n_i + \sum_j r_{(j,i)})\right)$ — linear in $N$ when $\rho_{\max}, d_{\max} \ll K$ and $m_i, n_i \sim N/K$

**Key Property:** Inverse preserves the exact same rank-profile (no rank growth), unlike sums and products which double the rank.

## Applicability

1. **Networked dynamical systems**: TQS naturally represents transfer operators of LTV systems on tree-structured networks — directly applicable to graph neural networks with tree connectivity
2. **Sparse matrix inverses**: Inverses of banded/sparse matrices whose adjacency graph is a tree (e.g., tridiagonal, arrowhead, tree-banded) have exact TQS representations — important for SSM/RNN Jacobian computations
3. **Non-binary hierarchical models**: HSS requires binary tree partitioning; TQS supports arbitrary tree topologies, enabling more natural decompositions of multi-scale architectures
4. **Nested dissection linear solvers**: Sparse matrices from PDE discretizations with nested dissection ordering produce TQS representations that are more natural than HSS
5. **Sequence models with branching**: Tree-structured attention or recurrence patterns (e.g., parsing trees, hierarchical document models) can exploit TQS structure

## Limitations

1. **Tree topology restriction**: The graph $\mathbb{G}$ must be acyclic; matrices with cyclic interaction patterns don't admit compact TQS representations
2. **Arrowhead matrices**: A canonical $K$-by-$K$ arrowhead matrix has a TQS representation with $O(K)$ parameters (same as dense), so not all tree-structured representations are efficient
3. **Efficiency conditions**: Linear-time algorithms require $\rho_{\max} \ll K$ and $\text{deg}_{\max} \ll K$; high-degree nodes or high-rank edges destroy efficiency
4. **Construction cost**: The realization algorithm requires computing low-rank factorizations of Hankel blocks at each level via upsweep/downsweep passes
5. **Maturity**: TQS is a new framework (2024); GPU-optimized implementations and integration with deep learning frameworks are not yet available
6. **No LU/Cholesky factorization yet**: The paper does not derive LU or Cholesky factorizations for general TQS; this is listed as future work

## Implementation Notes

```python
# TQS Construction Algorithm (Algorithm 1 from paper)
# Two-phase: upsweep (leaves→root) + downsweep (root→leaves)

def construct_tqs(T_dense, tree, root):
    """
    Construct TQS representation from dense matrix T and tree graph.
    Returns generator matrices {D, B, C, U, V, W, P, Q} at each node.

    Phase 1 (Upsweep): Extract B, C, U generators
    Phase 2 (Downsweep): Extract P, Q, V, W generators
    """
    generators = {}

    # Phase 0: Diagonal stage
    for node in tree.nodes:
        generators[node].D = T_dense[node.rows, node.cols]

    # Phase 1: Upsweep (leaves to root)
    for level in reversed(range(1, tree.depth + 1)):
        for node_i in tree.nodes_at_level(level):
            parent_j = tree.parent(node_i)

            # Form Hankel block H_{(i,j)} using previously computed factors
            H = form_hankel_block(T_dense, node_i, parent_j,
                                  previously_computed_Y)

            # Low-rank compression: H = X * Z
            rho = numerical_rank(H)
            X, Z = low_rank_factorize(H, rho)

            # Extract generators from factorization
            generators[node_i].B_parent = Z[node_i, :]  # B^i_j
            generators[parent_j].C_child = X[parent_j, :]  # C^j_i
            # Extract U generators for sibling-to-parent transfers
            for sibling in tree.children(parent_j):
                if sibling != node_i:
                    generators[parent_j].U[node_i, sibling] = Z[sibling, :]

    # Phase 2: Downsweep (root to leaves) — symmetric procedure for P, Q, V, W
    for level in range(1, tree.depth + 1):
        for node_i in tree.nodes_at_level(level):
            parent_j = tree.parent(node_i)
            H_down = form_hankel_block_down(T_dense, parent_j, node_i,
                                             previously_computed_Y)
            X, Z = low_rank_factorize(H_down, rho)
            generators[node_i].P = Z  # P^i_j
            generators[parent_j].Q = X  # Q^j_i
            # Extract V, W generators similarly

    return generators

# TQS Matrix-Vector Product (Algorithm 2 from paper)
def tqs_matvec(generators, tree, x):
    """
    Compute b = T*x in O(sum (m_i + sum r)(n_i + sum r)) time
    using upsweep/downsweep on the tree.
    """
    # Initialize: b_i = D^i * x_i
    b = {i: generators[i].D @ x[i] for i in tree.nodes}
    h = {}  # edge state vectors

    # Upsweep: leaves to root
    for level in reversed(range(1, tree.depth + 1)):
        for node_i in tree.nodes_at_level(level):
            parent_j = tree.parent(node_i)
            children = tree.children(node_i)

            # State equation: h_{(i,j)} = sum U * h_{(w,i)} + B * x_i
            h[(node_i, parent_j)] = generators[node_i].B[parent_j] @ x[node_i]
            for child_w in children:
                h[(node_i, parent_j)] += (
                    generators[node_i].U[parent_j, child_w] @ h[(child_w, node_i)]
                )

            # Output: b_j += C * h_{(i,j)}
            b[parent_j] += generators[parent_j].C[node_i] @ h[(node_i, parent_j)]

    # Downsweep: root to leaves
    for level in range(1, tree.depth + 1):
        for node_i in tree.nodes_at_level(level):
            parent_j = tree.parent(node_i)
            grandparent_k = tree.parent(parent_j) if parent_j else None
            siblings = tree.siblings(node_i)

            # State equation: h_{(j,i)} = W * h_{(k,j)} + sum V * h_{(s,j)} + P * x_j
            h[(parent_j, node_i)] = generators[node_i].P[parent_j] @ x[parent_j]
            if grandparent_k:
                h[(parent_j, node_i)] += (
                    generators[node_i].W[parent_j] @ h[(grandparent_k, parent_j)]
                )
            for sib in siblings:
                h[(parent_j, node_i)] += (
                    generators[node_i].V[parent_j, sib] @ h[(sib, parent_j)]
                )

            # Output: b_i += Q * h_{(j,i)}
            b[node_i] += generators[node_i].Q[parent_j] @ h[(parent_j, node_i)]

    return b

# TQS Linear System Solve
def tqs_solve(generators, tree, b):
    """
    Solve T*x = b by lifting to a sparse block system.
    The adjacency graph of the lifted system equals the tree,
    which is chordal → perfect elimination without fill-in.

    Complexity: O(sum (n_i + sum r_{(i,j)})^3)
    """
    # Form sparse lifted system: Xi * theta = beta
    # where theta = [x_1, h_{(i1,j1)}, h_{(i2,j2)}, ...]
    # Xi has the same sparsity pattern as the tree → no fill-in
    theta = sparse_chordal_solve(Xi, beta)
    x = extract_solution(theta)
    return x
```

**Key Implementation Insights:**

1. **Tree topology determines structure**: SSS = line graph, HSS = binary tree, TQS = arbitrary tree. Choose the tree matching the matrix's natural interaction pattern
2. **Inverse preserves rank**: Unlike sums/products which double the rank, inversion preserves the exact rank-profile — crucial for iterative methods
3. **Sparse lifting trick**: Linear systems $\text{T}x = b$ are solved by reformulating as a larger sparse system whose adjacency graph matches the tree, enabling fill-free elimination
4. **Julia implementation available**: Reference implementation at https://github.com/nithingovindarajan/TQSmatrices

## References

- Govindarajan, N., Chandrasekaran, S., & Dewilde, P. (2024). Tree quasi-separable matrices: a simultaneous generalization of sequentially and hierarchically semi-separable representations. *arXiv:2402.13381*.
- Chandrasekaran, S., Dewilde, P., Gu, M., Lyons, W., & Pals, T. (2007). A fast solver for HSS representations via sparse matrices. *SIAM Journal on Matrix Analysis and Applications*, 29(1), 67-81.
- Chandrasekaran, S., Dewilde, P., Gu, M., Pals, T., Sun, X., van der Veen, A.-J., & White, D. (2005). Some fast algorithms for sequentially semiseparable representations. *SIAM Journal on Matrix Analysis and Applications*, 27(2), 341-364.

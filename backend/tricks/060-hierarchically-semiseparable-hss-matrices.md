# 060: Hierarchically Semiseparable (HSS) Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Numerical linear algebra, structured matrices
**Paper**: [papers/fast-hss-algorithms.pdf] (Xia et al., 2010)
**Documented**: 2026-02-15

## Description

Hierarchically Semiseparable (HSS) matrices are a generalization of semiseparable matrices that exploit hierarchical low-rank structure in off-diagonal blocks. They enable fast $O(n)$ to $O(rn)$ algorithms for operations that would normally take $O(n^2)$ or $O(n^3)$ time, where $r$ is the HSS rank (typically much smaller than $n$). HSS matrices arise naturally in discretized PDEs, integral equations, and covariance matrices, making them highly applicable to neural network computations involving structured linear transformations.

The key insight is that off-diagonal blocks at multiple hierarchical levels have small numerical rank, which can be captured using a binary tree structure. This allows matrix operations to be performed via tree traversals with linear or near-linear complexity.

## Mathematical Form

**HSS Matrix Structure:**

An $n \times n$ matrix $H$ has an HSS representation with rank $r$ if it can be recursively partitioned using a binary tree $\mathcal{T}$ such that:

$$
H = \begin{pmatrix}
D_1 & U_1 B_1 V_2^T \\
U_2 B_2 V_1^T & D_2
\end{pmatrix}
$$

where at each level:
- $D_i$ are diagonal blocks (recursively HSS matrices)
- $U_i, V_i \in \mathbb{R}^{m \times r}$ are orthonormal column matrices (generators)
- $B_i \in \mathbb{R}^{r \times r}$ are small coupling matrices
- $r \ll m$ is the HSS rank

**Postordering HSS Representation:**

Using postordering notation (simplified), a 4×4 block HSS matrix looks like:

$$
\begin{pmatrix}
D_1 & U_1 B_1 V_2^T & U_1 R_1 B_3 W_4^T V_4^T & U_1 R_1 B_3 W_5^T V_5^T \\
U_2 B_2 V_1^T & D_2 & U_2 R_2 B_3 W_4^T V_4^T & U_2 R_2 B_3 W_5^T V_5^T \\
U_4 R_4 B_6 W_1^T V_1^T & U_4 R_4 B_6 W_2^T V_2^T & D_4 & U_4 B_4 V_5^T \\
U_5 R_5 B_6 W_1^T V_1^T & U_5 R_5 B_6 W_2^T V_2^T & U_5 B_5 V_4^T & D_5
\end{pmatrix}
$$

**Key Definitions:**

- $H \in \mathbb{R}^{n \times n}$ — Original matrix with low-rank off-diagonal blocks
- $r$ — HSS rank (maximum rank of off-diagonal blocks at all levels)
- $U_i, V_i \in \mathbb{R}^{m_i \times r}$ — Column basis matrices (orthonormal)
- $R_i, W_i \in \mathbb{R}^{r \times r}$ — Translation operators (nested structure)
- $B_i \in \mathbb{R}^{r \times r}$ — Coupling matrices
- $D_i$ — Diagonal blocks (can be dense or recursively HSS)

**Data-Sparse Representation:**

For each node in the HSS tree $\mathcal{T}$, store:
$$
\{U_\tau, V_\tau : \tau \in \mathcal{T}\}, \quad \{\tilde{A}_{\tau,\tau'} : \tau,\tau' \text{ siblings}\}, \quad \{A_{\alpha,\alpha} : \alpha \text{ leaf node}\}
$$

where $\tilde{A}_{\tau,\tau'}$ are small $r \times r$ matrices for sibling interactions.

## Complexity

| Operation | Naive | HSS (rank $r$) |
|-----------|-------|----------------|
| Matrix-vector product | $O(n^2)$ | $O(rn)$ |
| HSS construction | $O(n^3)$ | $O(r^2 n)$ |
| ULV factorization | $O(n^3)$ | $O(r^2 n)$ |
| System solve | $O(n^3)$ | $O(r^2 n)$ |
| Matrix inversion | $O(n^3)$ | $O(r^2 n)$ |
| Compression (QR) | N/A | $O(r^2 n)$ |

**Memory:** $O(rn)$ vs $O(n^2)$ for storing the full matrix

**Tree depth:** $O(\log n)$ for balanced binary tree

## Applicability

HSS matrices are highly applicable to:

1. **State Space Models (SSMs)**: Discretized linear recurrences often produce HSS-structured transition matrices
2. **Linear Attention**: Kernel matrices with exponential decay or local correlation structure
3. **Structured Linear Layers**: Any linear transformation $y = Wx$ where $W$ has hierarchical low-rank off-diagonal blocks
4. **Fast Convolutions**: Toeplitz and circulant matrices are special cases
5. **Preconditioning**: HSS approximations can accelerate iterative solvers
6. **Neural ODEs**: Discretized differential operators in continuous-depth models

## Limitations

1. **Construction Cost**: Building the HSS representation requires $O(r^2 n)$ time, which may be expensive if done frequently
2. **Rank Requirement**: Only beneficial when HSS rank $r \ll n$; dense matrices with no low-rank structure don't benefit
3. **Dynamic Updates**: Modifying the matrix structure after HSS construction can be expensive
4. **Numerical Rank**: Requires off-diagonal blocks to have small numerical rank (not just exact rank)
5. **Tree Structure**: Optimal tree structure depends on matrix block structure; poor choices reduce efficiency
6. **Forward-Only**: Most algorithms are designed for forward operations; backward pass requires additional consideration

## Implementation Notes

```python
# Pseudocode for HSS matrix-vector product
def hss_matvec(hss_tree, x):
    """
    Compute y = H*x where H is in HSS form
    Complexity: O(rn) where r is HSS rank
    """
    # Upward pass: compute partial results at leaf nodes
    for leaf in postorder(hss_tree.leaves):
        leaf.result = leaf.D @ x[leaf.indices]

    # Upward pass: aggregate via U generators
    for node in postorder(hss_tree.internal_nodes):
        node.temp = node.U.T @ x[node.indices]

    # Downward pass: distribute via V generators
    for node in preorder(hss_tree.internal_nodes):
        for child in node.children:
            child.result += node.V @ (node.B @ node.temp)

    # Collect results
    y = concatenate([leaf.result for leaf in hss_tree.leaves])
    return y

# Fast HSS construction via randomized compression
def construct_hss(A, tree, rank, oversampling=10):
    """
    Construct HSS representation from dense or implicit matrix
    Uses randomized QR for compression
    """
    r = rank + oversampling

    # Bottom-up: compress off-diagonal blocks at each level
    for node in postorder(tree):
        if node.is_leaf:
            node.D = A[node.rows, node.cols]
        else:
            # Compress off-diagonal interactions
            X_off = A[node.rows, sibling(node).cols]
            Y_off = A[sibling(node).rows, node.cols]

            # Randomized QR compression
            node.U, node.B = compress_cols(X_off, r)
            node.V, _ = compress_cols(Y_off.T, r)

            # Extract translation operators for nested structure
            if not node.parent.is_root:
                node.R = extract_nested_basis(node.U, node.parent.U)
                node.W = extract_nested_basis(node.V, node.parent.V)

    return hss_tree
```

**Key Implementation Insights:**

1. **Postordering**: Process tree nodes in postorder (children before parents) for upward passes
2. **Basis Reuse**: Ignore previously computed bases in QR factorizations to save cost
3. **Orthogonality**: Use orthogonal transformations throughout for numerical stability
4. **Partial Trees**: Can use partial binary trees (not necessarily complete) for irregular block structures

## References

- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices. *Numerical Linear Algebra with Applications*, 17(6), 953-976.
- Chandrasekaran, S., Gu, M., & Pals, T. (2006). A fast ULV decomposition solver for hierarchically semiseparable representations. *SIAM Journal on Matrix Analysis and Applications*, 28(3), 603-622.
- Martinsson, P. G. (2011). A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix. *SIAM Journal on Matrix Analysis and Applications*, 32(4), 1251-1274.

# 052: Generalized HSS Cholesky Factorization with Postordering

**Category**: decomposition
**Gain type**: efficiency
**Source**: Xia, Chandrasekaran, Gu, & Li (2010), Numer. Linear Algebra Appl.
**Paper**: [papers/hss-sparse-embedding-solver.pdf]
**Documented**: 2026-02-15

## Description

The generalized HSS Cholesky factorization is an explicit, linear-complexity factorization scheme for symmetric positive definite (SPD) matrices in HSS form. Unlike the implicit ULV factorization that stores factors as sequences of unitary transformations, this algorithm produces an explicit generalized Cholesky factor $L_H$ such that $H = L_H L_H^T$, where $L_H$ itself has HSS-like structure composed of lower-triangular blocks $\{L_i\}$, orthogonal transformations $\{Q_i\}$, and permutations $\{P_i\}$.

The two key innovations are:

1. **Postordering HSS notation**: Instead of the traditional level-wise (global) tree traversal, nodes are relabeled with a single index following the postorder traversal of the HSS tree. This simplifies notation, improves data locality, limits communication to parent-child pairs only, and naturally supports general (partial) binary trees — not just perfect binary trees.

2. **Compression reuse**: When compressing off-diagonal blocks at a non-leaf node, portions that were already compressed at child nodes (the $U, V$ bases) can be ignored, dramatically reducing the cost of QR factorizations from $O(N^2)$ per block to $O(mr^2)$.

The factorization traverses the postordered HSS tree bottom-up. At each node, three operations are performed: (a) compress off-diagonal generators by introducing zeros via orthogonal transformations, (b) partially factorize the transformed diagonal block, and (c) merge the Schur complement with sibling information to form the parent node's generators. The total cost is $O(r^2 N)$ — linear in $N$.

## Mathematical Form

**Core Operation:**

For an SPD HSS matrix $H$ with HSS rank $r$, the generalized Cholesky factorization computes:

$$
H = L_H L_H^T
$$

where $L_H = Q_3 \hat{L}_3 P_3 \begin{pmatrix} I & 0 \\ 0 & L_3 \end{pmatrix}$ is assembled recursively from the HSS tree.

**Postordering HSS Representation:**

With single-index postordering, the $4 \times 4$ block HSS matrix becomes:

$$
H = \begin{pmatrix} D_1 & U_1 B_1 V_2^T & U_1 R_1 B_3 W_4^T V_4^T & U_1 R_1 B_3 W_5^T V_5^T \\ U_2 B_2 V_1^T & D_2 & U_2 R_2 B_3 W_4^T V_4^T & U_2 R_2 B_3 W_5^T V_5^T \\ U_4 R_4 B_6 W_1^T V_1^T & U_4 R_4 B_6 W_2^T V_2^T & D_4 & U_4 B_4 V_5^T \\ U_5 R_5 B_6 W_1^T V_1^T & U_5 R_5 B_6 W_2^T V_2^T & U_5 B_5 V_4^T & D_5 \end{pmatrix}
$$

**Key Definitions:**

- $H \in \mathbb{R}^{N \times N}$ — SPD matrix with HSS representation
- $D_i \in \mathbb{R}^{m_i \times m_i}$ — Diagonal blocks at leaf nodes
- $U_i \in \mathbb{R}^{m_i \times k_i}$ — Row basis generators (column basis for off-diagonal block rows)
- $V_i \in \mathbb{R}^{m_i \times k_i}$ — Column basis generators (for symmetric case, $U_i = V_i$, $R_i = W_i$)
- $R_i, W_i$ — Translation operators connecting child to parent levels
- $B_i$ — Coupling matrices between siblings
- $r$ — HSS rank (maximum off-diagonal numerical rank)
- $k_i$ — Rank at node $i$ (satisfies $k_i \leq r$)

**Step 1: Introducing Zeros (Compression):**

At leaf node $i$ with generator $U_i$ of size $m_i \times k_i$ (where $m_i > k_i$), compute a QL factorization:

$$
U_i = Q_i \begin{pmatrix} 0 \\ \tilde{U}_i \end{pmatrix}, \quad \tilde{U}_i : k_i \times k_i
$$

Apply $Q_i^T$ from left and $Q_i$ from right to the diagonal block:

$$
\hat{D}_i = Q_i^T D_i Q_i
$$

This zeros out the first $m_i - k_i$ rows and columns of the off-diagonal blocks.

**Step 2: Partial Factorization:**

Partition the transformed diagonal block conformally:

$$
\hat{D}_i = \begin{pmatrix} D_{i;1,1} & D_{i;1,2} \\ D_{i;2,1} & D_{i;2,2} \end{pmatrix} = \begin{pmatrix} L_i & 0 \\ D_{i;2,1} L_i^{-T} & I \end{pmatrix} \begin{pmatrix} L_i^T & L_i^{-1} D_{i;1,2} \\ 0 & \tilde{D}_i \end{pmatrix}
$$

where $L_i$ is the Cholesky factor of $D_{i;1,1} = L_i L_i^T$ and $\tilde{D}_i$ is the Schur complement:

$$
\tilde{D}_i = D_{i;2,2} - D_{i;2,1} L_i^{-T} L_i^{-1} D_{i;1,2}
$$

**Step 3: Merging Siblings:**

For siblings $c_1, c_2$ with parent $i$, form the merged generators:

$$
D_i = \begin{pmatrix} \tilde{D}_{c_1} & \tilde{U}_{c_1} B_{c_1} \tilde{U}_{c_2}^T \\ \tilde{U}_{c_2} B_{c_1}^T \tilde{U}_{c_1}^T & \tilde{D}_{c_2} \end{pmatrix}, \quad U_i = \begin{pmatrix} \tilde{U}_{c_1} R_{c_1} \\ \tilde{U}_{c_2} R_{c_2} \end{pmatrix}
$$

Repeat compression and factorization at parent. At root, compute final Cholesky: $D_n = L_n L_n^T$.

**Assembled Factor:**

$$
L_H = Q_3 \hat{L}_3 P_3 \begin{pmatrix} I & 0 \\ 0 & L_3 \end{pmatrix}
$$

where $\hat{L}_3 = \text{diag}\left(\begin{pmatrix} L_1 & 0 \\ T_1 & I \end{pmatrix}, \begin{pmatrix} L_2 & 0 \\ T_2 & I \end{pmatrix}\right)$ with $T_i = D_{i;2,1} L_i^{-T}$.

## Complexity

| Operation | Naive (Dense) | Standard HSS Cholesky | Generalized HSS Cholesky |
|-----------|--------------|----------------------|--------------------------|
| Factorization | $O(N^3)$ | $O(N^2)$ | $O(r^2 N)$ |
| System solve | $O(N^2)$ | $O(rN)$ | $O(rN)$ |
| Storage | $O(N^2)$ | $O(rN)$ | $O(rN)$ |

**Detailed Cost Breakdown (uniform leaf size $m$, HSS rank $r$):**

| Node Type | Operation | Cost | Count |
|-----------|-----------|------|-------|
| Leaf | Compression (QR) | $O(mr^2)$ | $N/m$ |
| Leaf | Diagonal update ($Q_i$) | $O(m^2 r)$ | $N/m$ |
| Leaf | Partial factorization | $O((m-r)^3)$ | $N/m$ |
| Non-leaf | Merge step | $O(r^3)$ | $N/m - 1$ |
| Non-leaf | Compression + factorization | $O(r^3)$ | $N/m - 1$ |

**Total:** $[O(mr^2) + O(m^2 r) + O((m-r)^3)] \times \frac{N}{m} + O(r^3) \times \frac{N}{m} = O(r^2 N)$ since $m = O(r)$.

**Memory:** $O(rN)$ for the HSS representation and its Cholesky factors.

**Compared to previous methods:**
- Construction algorithm in the original HSS work needed SVDs of eight $m \times 3m$ matrices — the new method's QR-based approach costs roughly $3rN^2 + 6r^2N$ vs. significantly more for SVD-based methods.

## Applicability

1. **State Space Models**: Fast factorization of structured transition matrices arising in SSM discretization, where $A$ has HSS structure from exponentially decaying interactions.

2. **Preconditioning for Neural Network Training**: The explicit Cholesky factor serves as a high-quality preconditioner for iterative solvers — useful for second-order optimization methods (natural gradient, K-FAC) where the Fisher information matrix has HSS structure.

3. **Structured Attention Kernels**: When attention weight matrices exhibit hierarchical low-rank off-diagonal structure (e.g., from locality bias), this factorization enables $O(N)$ solves.

4. **Gaussian Process Inference**: SPD kernel matrices from GPs (e.g., Matérn kernels) are HSS; this provides fast Cholesky for log-likelihood computation and sampling.

5. **Sparse Direct Solvers**: The dense frontal matrices in multifrontal sparse solvers (e.g., STRUMPACK) are compressed using HSS, and this generalized Cholesky is used for the factorization step.

6. **Parallel Computing**: The postordering traversal keeps data locality by limiting communication to parent-child node pairs, making it well-suited for distributed-memory architectures.

## Limitations

1. **SPD Requirement**: The factorization requires the matrix to be symmetric positive definite; indefinite or nonsymmetric matrices need the more general ULV approach.

2. **HSS Rank Sensitivity**: Cost scales as $O(r^2 N)$; if the off-diagonal rank $r$ grows with $N$ (e.g., 3D problems), the linear advantage diminishes.

3. **Stability Condition**: Backward stability requires $\|R_i\| < 1$ in a submultiplicative norm; the orthogonal construction algorithm in the paper ensures this.

4. **Static Tree**: The HSS tree structure must be fixed a priori; adaptive refinement is not supported within a single factorization.

5. **Approximate for Numerical Ranks**: When using numerical (approximate) ranks with tolerance $\tau$, the factorization is an approximation to the true Cholesky factor.

## Implementation Notes

```python
# Pseudocode: Generalized HSS Cholesky Factorization (Postordering)
def generalized_hss_cholesky(hss_tree):
    """
    Compute H = L_H @ L_H.T for SPD HSS matrix H.

    Traverses the postordered HSS tree bottom-up.
    Complexity: O(r^2 * N) where r is HSS rank, N is matrix size.
    """
    stack = []  # stack for passing Schur complements up the tree
    n = hss_tree.num_nodes

    for i in range(1, n):  # postorder traversal (leaves first)
        if hss_tree.is_leaf(i):
            # Step 1: Compress U_i via QL factorization
            U_i = hss_tree.U[i]  # m_i x k_i
            Q_i, U_tilde_i = ql_factorization(U_i)
            # U_tilde_i is k_i x k_i (bottom block)

            # Step 2: Apply orthogonal transform to diagonal
            D_hat_i = Q_i.T @ hss_tree.D[i] @ Q_i  # O(m_i^2 * k_i)

            # Step 3: Partial Cholesky of top-left block
            m_i, k_i = U_i.shape
            D_11 = D_hat_i[:m_i-k_i, :m_i-k_i]
            D_21 = D_hat_i[m_i-k_i:, :m_i-k_i]
            D_22 = D_hat_i[m_i-k_i:, m_i-k_i:]

            L_i = cholesky(D_11)  # O((m_i - k_i)^3)
            T_i = D_21 @ solve_triangular(L_i, eye(m_i-k_i))

            # Schur complement
            D_tilde_i = D_22 - T_i @ T_i.T

            # Store factors and push Schur complement
            factors[i] = (L_i, T_i, Q_i)
            stack.append((D_tilde_i, U_tilde_i))

        else:
            # Non-leaf: pop children's Schur complements
            c1, c2 = hss_tree.children(i)
            D_tilde_c2, U_tilde_c2 = stack.pop()
            D_tilde_c1, U_tilde_c1 = stack.pop()

            # Step 4: Merge siblings into parent diagonal
            B_c1 = hss_tree.B[c1]
            D_i = block_matrix([
                [D_tilde_c1, U_tilde_c1 @ B_c1 @ U_tilde_c2.T],
                [U_tilde_c2 @ B_c1.T @ U_tilde_c1.T, D_tilde_c2]
            ])

            # Form merged generator for parent
            U_i = vstack([
                U_tilde_c1 @ hss_tree.R[c1],
                U_tilde_c2 @ hss_tree.R[c2]
            ])

            # Repeat compression + partial factorization
            Q_i, U_tilde_i = ql_factorization(U_i)
            D_hat_i = Q_i.T @ D_i @ Q_i
            # ... (same as leaf case)
            stack.append((D_tilde_i, U_tilde_i))

    # Root node: final dense Cholesky
    D_root = stack.pop()[0]
    L_root = cholesky(D_root)

    return factors, L_root


def hss_cholesky_solve(factors, L_root, b):
    """
    Solve H @ x = b using generalized HSS Cholesky factors.
    Forward substitution (postorder) + backward (reverse-postorder).
    Complexity: O(r * N).
    """
    # Forward substitution: L_H @ y = b
    # Traverse postorder (bottom-up)
    for i in postorder(tree):
        L_i, T_i, Q_i = factors[i]
        y_i = Q_i.T @ b_i
        y_i_1 = solve_triangular(L_i, y_i[:m_i-k_i])  # partial solve
        y_i_2 = y_i[m_i-k_i:] - T_i @ y_i_1  # update for parent
        # Pass y_i_2 to parent

    # Backward substitution: L_H.T @ x = y
    # Traverse reverse-postorder (top-down)
    x_root = solve_triangular(L_root.T, y_root)
    for i in reverse_postorder(tree):
        # Recover x_i from parent's contribution
        x_i = Q_i @ vstack([
            solve_triangular(L_i.T, y_i_1 - T_i.T @ x_hat_i),
            x_hat_i
        ])

    return x
```

**Key Implementation Insights:**

1. **QL not QR**: Use QL factorization (zeros at the top) to preserve the bottom rows of $U_i$ that carry coupling information to the parent.
2. **Compression Reuse**: At non-leaf nodes, previously compressed $U, V$ bases from children can be ignored during QR factorizations — this is the key to the cost reduction from $O(N^2)$ to $O(r^2 N)$.
3. **Stack-Based Traversal**: The postordering allows a simple stack-based implementation where each node only communicates with its parent, ensuring excellent data locality.
4. **In-Place**: Solution vectors can reuse storage of the right-hand side $b$, requiring no extra memory.
5. **Parallelism**: Sibling pairs at the same depth can be processed independently; the sequential bottleneck is the tree depth $O(\log N)$.

## References

- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices. *Numerical Linear Algebra with Applications*, 17(6), 953-976. DOI: 10.1002/nla.691
- Xia, J. (2013). Efficient structured multifrontal factorization for general large sparse matrices. *SIAM Journal on Scientific Computing*, 35(2), A832-A860.
- Chandrasekaran, S., Gu, M., & Pals, T. (2006). A fast ULV decomposition solver for hierarchically semiseparable representations. *SIAM J. Matrix Anal. Appl.*, 28(3), 603-622.
- Ghysels, P., Li, X. S., Rouet, F.-H., Williams, S., & Napov, A. (2016). An efficient multicore implementation of a novel HSS-structured multifrontal solver using randomized sampling. *SIAM J. Sci. Comput.*, 38(5), S358-S384.

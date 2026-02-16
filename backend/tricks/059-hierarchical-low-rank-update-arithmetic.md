# 059: Hierarchical Low-Rank Update Arithmetic for $\mathcal{H}^2$-Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Börm & Reimer (2014/2018), Kiel University
**Paper**: [papers/hierarchical-low-rank-update-arithmetic.pdf]
**Documented**: 2026-02-15

## Description

This trick provides an efficient algorithm for performing algebraic operations (matrix multiplication, LR factorization, inversion, Cholesky factorization) on $\mathcal{H}^2$-matrices — the class that includes HSS matrices as a special case — by reducing all operations to a sequence of **local low-rank updates**. Each local update adds a low-rank matrix $XY^*$ to a single submatrix block of the $\mathcal{H}^2$-matrix, then **recompresses** to maintain the $\mathcal{H}^2$-structure, all in $O(k^2(\#\hat{t}_0 + \#\hat{s}_0))$ operations (linear in the affected block dimensions).

The key innovation over prior $\mathcal{H}^2$-arithmetic is that the recompression after a low-rank update preserves the **nested cluster basis** property — the defining feature of $\mathcal{H}^2$/HSS matrices that enables $O(nk)$ storage and $O(nk)$ matrix-vector products. Previous methods either required $O(nk^2 \log^2 n)$ operations for matrix multiplication or lost the nested basis property. This approach achieves $O(nk^2 \log n)$ for multiplication and $O(nk^2 \log n)$ for inversion/factorization, with only $O(nk)$ storage throughout.

## Mathematical Form

**$\mathcal{H}^2$-Matrix Representation:**

An $\mathcal{H}^2$-matrix $G \in \mathbb{R}^{\mathcal{I} \times \mathcal{J}}$ is defined by:
- A block tree $\mathcal{T}_{\mathcal{I} \times \mathcal{J}}$ partitioning the index set
- **Cluster bases** $(V_t)_{t \in \mathcal{T}_{\mathcal{I}}}$ and $(W_s)_{s \in \mathcal{T}_{\mathcal{J}}}$ with $V_t \in \mathbb{R}^{\hat{t} \times k}$, $W_s \in \mathbb{R}^{\hat{s} \times k}$
- **Coupling matrices** $S_b \in \mathbb{R}^{k \times k}$ for each admissible block $b = (t,s) \in \mathcal{L}^+_{\mathcal{I} \times \mathcal{J}}$
- **Nearfield matrices** $G|_{\hat{t} \times \hat{s}}$ for inadmissible blocks $(t,s) \in \mathcal{L}^-_{\mathcal{I} \times \mathcal{J}}$

so that:

$$
G|_{\hat{t} \times \hat{s}} = V_t S_b W_s^* \quad \text{for all } b = (t,s) \in \mathcal{L}^+_{\mathcal{I} \times \mathcal{J}}
$$

The **nested basis property** requires transfer matrices $E_t \in \mathbb{R}^{k \times k}$ such that:

$$
V_t|_{\hat{t}' \times k} = V_{t'} E_{t'} \quad \text{for all } t' \in \text{sons}(t)
$$

**Storage:** $O((n_{\mathcal{I}} + n_{\mathcal{J}})k)$ — optimal.

**Core Operation — Global Low-Rank Update:**

Given an $\mathcal{H}^2$-matrix $Z$ and a rank-$k$ matrix $XY^*$ with $X \in \mathbb{R}^{\mathcal{I} \times k}$, $Y \in \mathbb{R}^{\mathcal{J} \times k}$, compute:

$$
\widetilde{Z} := Z + XY^*
$$

as an $\mathcal{H}^2$-matrix of the same rank. For each admissible block $b = (t,s)$:

$$
\widetilde{Z}|_{\hat{t} \times \hat{s}} = V_t S_b W_s^* + X|_{\hat{t} \times k} Y|_{\hat{s} \times k}^*
$$

By introducing expanded cluster bases:

$$
\widetilde{V}_t := \begin{pmatrix} V_t & X|_{\hat{t} \times k} \end{pmatrix}, \quad \widetilde{W}_s := \begin{pmatrix} W_s & Y|_{\hat{s} \times k} \end{pmatrix}
$$

$$
\widetilde{S}_b := \begin{pmatrix} S_b \\ & I \end{pmatrix}
$$

we obtain an **exact** $\mathcal{H}^2$-representation of $\widetilde{Z}$ with rank doubled to $2k$.

**Recompression — The Critical Step:**

To prevent rank doubling from cascading through repeated updates, a recompression algorithm finds new orthogonal cluster bases $(Q_t)_{t \in \mathcal{T}_{\mathcal{I}}}$ of rank $k$ such that:

$$
Q_t Q_t^* \widetilde{Z}|_{\hat{t} \times \hat{s}} \approx \widetilde{Z}|_{\hat{t} \times \hat{s}} \quad \text{for all } s \in \text{row}^*(t)
$$

The recompression exploits the nested structure: for node $t$ with $\text{sons}(t) = \{t_1, t_2\}$, the weight matrix that captures all block interactions is:

$$
B_t := \begin{pmatrix} B_{t,s_1} \\ \vdots \\ B_{t,s_\varrho} \\ P_{t^+} \hat{B}_{t^+} E_t^* \end{pmatrix}
$$

where $B_{t,s_i} = \widetilde{W}_{s_i} \widetilde{S}_b^*$ are coupling-weighted interactions, and the last row inherits from the parent via transfer matrices. The QR decomposition of $B_t$ yields:

$$
P_t \hat{B}_t = B_t
$$

where $P_t$ is the orthogonal projector. The basis $Q_t$ is then constructed from the SVD of $\widetilde{V}_t \hat{B}_t^*$:

$$
Q_t Q_t^* \widetilde{V}_t \hat{B}_t^* \approx \widetilde{V}_t \hat{B}_t^*
$$

**Local Low-Rank Update:**

For updating a single block $Z|_{\hat{t}_0 \times \hat{s}_0} \leftarrow Z|_{\hat{t}_0 \times \hat{s}_0} + XY^*$ where $b_0 = (t_0, s_0)$:

1. Apply recompression to subtrees $\mathcal{T}_{\hat{t}_0}$ and $\mathcal{T}_{\hat{s}_0}$ only
2. Update coupling matrices:
   - If $t \in \mathcal{T}_{\hat{t}_0}$ and $s \in \mathcal{T}_{\hat{s}_0}$: $S_b \leftarrow R_t \begin{pmatrix} S_b & 0 \\ 0 & I \end{pmatrix} R_s^*$
   - If $t \in \mathcal{T}_{\hat{t}_0}$ and $s \notin \mathcal{T}_{\hat{s}_0}$: $S_b \leftarrow R_t \begin{pmatrix} S_b \\ 0 \end{pmatrix}$
   - If $t \notin \mathcal{T}_{\hat{t}_0}$ and $s \in \mathcal{T}_{\hat{s}_0}$: $S_b \leftarrow \begin{pmatrix} S_b & 0 \end{pmatrix} R_s^*$
3. Update transfer matrices at subtree roots:
   - $E_{t_0} \leftarrow R_{t_0} \begin{pmatrix} E_{t_0} \\ 0 \end{pmatrix}$, similarly for $E_{s_0}$

**Matrix Multiplication via Local Updates:**

The product $Z \leftarrow Z + \alpha X|_{\hat{t} \times \hat{s}} Y|_{\hat{s} \times \hat{r}}$ for blocks $(t,s), (s,r)$ in the block tree is computed recursively. When one of $(t,s)$ or $(s,r)$ is a leaf block:

$$
X|_{\hat{t} \times \hat{s}} Y|_{\hat{s} \times \hat{r}} = X|_{\hat{t} \times \hat{s}} V_s S_{s,r} W_r^* = \underbrace{(X|_{\hat{t} \times \hat{s}} V_s)}_{\widetilde{X}_{t,s}} S_{s,r} W_r^*
$$

which is a rank-$k$ matrix, so a local low-rank update can be applied.

**Key Definitions:**

- $k$ — Cluster basis rank (analogous to HSS rank $r$)
- $n_{\mathcal{I}} = \#\mathcal{I}$ — Matrix dimension
- $p_{\mathcal{I}}$ — Depth of cluster tree $\mathcal{T}_{\mathcal{I}}$ ($O(\log n)$ for balanced trees)
- $C_{\text{sp}}$ — Sparsity constant of the block tree (bounded for practical problems)
- $\mathcal{L}^+_{\mathcal{I} \times \mathcal{J}}$ — Set of admissible (far-field) leaf blocks
- $\mathcal{L}^-_{\mathcal{I} \times \mathcal{J}}$ — Set of inadmissible (near-field) leaf blocks
- $E_t$ — Transfer matrix connecting cluster basis at $t$ to its children
- $R_t = Q_t^* \widetilde{V}_t$ — Rotation matrix from recompression

## Complexity

| Operation | Naive | $\mathcal{H}$-matrix | $\mathcal{H}^2$ (this trick) |
|-----------|-------|----------------------|------------------------------|
| Matrix-vector product | $O(n^2)$ | $O(nk \log n)$ | $O(nk)$ |
| Global low-rank update + recompression | N/A | N/A | $O(nk^2)$ |
| Local low-rank update | N/A | N/A | $O((\#\hat{t}_0 + \#\hat{s}_0) k^2)$ |
| Matrix multiplication | $O(n^3)$ | $O(nk^2 \log^2 n)$ | $O(nk^2 \log n)$ |
| LR factorization | $O(n^3)$ | $O(nk^2 \log^2 n)$ | $O(nk^2 \log n)$ |
| Matrix inversion | $O(n^3)$ | $O(nk^2 \log^2 n)$ | $O(nk^2 \log n)$ |
| Forward substitution | $O(n^2)$ | $O(nk \log n)$ | $O(nk \log n)$ |

**Detailed complexity for multiplication (Theorem 6):**

$$
W_{\text{mm}}(t_0, s_0, r_0) \leq C_{\text{mm}} k^2 (p_{\mathcal{I}} + 1)(\#\hat{t}_0 + \#\hat{s}_0 + \#\hat{r}_0)
$$

where $p_{\mathcal{I}} = O(\log n)$ is the tree depth and $C_{\text{mm}} = C_{\text{sp}}^2 C_{\text{mb}}$ depends on the sparsity constant.

**Memory:** $O(nk)$ throughout all operations (optimal)

**Practical performance:** Numerical experiments show $O(n \log n)$ setup time for preconditioners requiring $O(n)$ storage and $O(n)$ application cost.

## Applicability

1. **Structured Linear Layer Arithmetic**: When neural network weight matrices are parameterized as $\mathcal{H}^2$/HSS matrices, this trick enables efficient matrix-matrix operations (e.g., computing $W_1 W_2$ for layer fusion, or $W^{-1}$ for invertible networks) in near-linear time

2. **Fast Preconditioning for Iterative Attention**: Building $\mathcal{H}^2$-preconditioners for iterative solvers applied to attention-like kernel matrices — the LR factorization provides an approximate inverse that can be applied in $O(nk)$ time

3. **Gradient Computation in Structured Layers**: During backpropagation through HSS-structured layers, the gradient involves matrix products $\frac{\partial L}{\partial W} = \delta x^T$ which can be accumulated as low-rank updates and recompressed to maintain structure

4. **Online/Streaming Matrix Updates**: When the weight matrix evolves during training via low-rank updates (e.g., LoRA-style adaptation), this trick maintains the $\mathcal{H}^2$ structure through arbitrary sequences of rank-$k$ updates

5. **Sparse Direct Solvers**: The core subroutine in multifrontal sparse solvers (STRUMPACK, MUMPS) where frontal matrices are compressed in HSS/$\mathcal{H}^2$ form and the Schur complement involves matrix products

6. **Newton-Schulz Iteration for Matrix Inverse**: Computing $W^{-1}$ via $X_{k+1} = X_k(2I - WX_k)$ where all operations are $\mathcal{H}^2$-matrix arithmetic, converging in $O(\log(\kappa))$ iterations with $O(nk^2 \log n)$ per iteration

7. **Kernel Matrix Operations in GP/Attention**: Performing algebraic operations on kernel matrices (inversion, Cholesky, determinant) that arise in Gaussian processes or attention mechanisms with hierarchical structure

## Limitations

1. **Approximate**: Recompression introduces approximation error at each step; errors can accumulate through sequences of operations (though bounded by SVD truncation)
2. **Rank Growth**: If the true rank of the result exceeds $k$, accuracy degrades; adaptive rank selection adds complexity
3. **Binary Cluster Tree Required**: The algorithm assumes a binary cluster tree; more general tree structures require modification
4. **Sparsity Constant**: Complexity depends on the block tree sparsity constant $C_{\text{sp}}$; pathological partitions can make this large
5. **Implementation Complexity**: Maintaining nested cluster bases through updates requires careful bookkeeping of transfer matrices, coupling matrices, and basis changes
6. **Non-Symmetric Generality**: Full generality for non-symmetric matrices requires separate row and column cluster bases
7. **Sequential Recompression**: The recompression algorithm proceeds level-by-level through the cluster tree, introducing $O(\log n)$ sequential steps

## Implementation Notes

```python
# Pseudocode for local low-rank update of H^2-matrix
def h2_local_lowrank_update(Z_h2, t0, s0, X, Y, k_target):
    """
    Update Z|_{t0 x s0} <- Z|_{t0 x s0} + X @ Y^*
    while maintaining H^2 structure.

    Args:
        Z_h2: H^2-matrix representation
        t0, s0: root clusters of the subtrees to update
        X: R^{#t0 x k} low-rank factor
        Y: R^{#s0 x k} low-rank factor
        k_target: target rank after recompression

    Complexity: O((#t0 + #s0) * k^2)
    """
    tree_t = Z_h2.subtree(t0)
    tree_s = Z_h2.subtree(s0)

    # Step 1: Expand cluster bases to accommodate the update
    for t in tree_t.nodes:
        V_old = Z_h2.row_basis[t]
        V_new = hstack([V_old, X[t.indices, :]])  # rank 2k
        Z_h2.row_basis[t] = V_new

    for s in tree_s.nodes:
        W_old = Z_h2.col_basis[s]
        W_new = hstack([W_old, Y[s.indices, :]])  # rank 2k
        Z_h2.col_basis[s] = W_new

    # Expand coupling matrices
    for (t, s) in Z_h2.admissible_blocks_in(tree_t, tree_s):
        S_old = Z_h2.coupling[t, s]
        Z_h2.coupling[t, s] = block_diag(S_old, eye(k_target))

    # Step 2: Recompression — restore rank k from rank 2k
    # Process bottom-up through both subtrees
    for t in postorder(tree_t):
        # Compute weight matrix B_t capturing all block interactions
        B_t = compute_weight_matrix(Z_h2, t)

        # QR decomposition of B_t for efficient processing
        P_t, B_hat_t = thin_qr(B_t)

        # SVD-based recompression of V_tilde * B_hat_t^*
        V_tilde = Z_h2.row_basis[t]
        U, sigma, Vt = truncated_svd(V_tilde @ B_hat_t.conj().T, k_target)

        # New orthogonal basis
        Q_t = U[:, :k_target]

        # Rotation matrix for updating coupling matrices
        R_t = Q_t.conj().T @ V_tilde

        # Update basis
        Z_h2.row_basis[t] = Q_t

        # Store R_t for coupling matrix updates
        t.rotation = R_t

    # Similar recompression for column subtree
    for s in postorder(tree_s):
        # ... analogous to row recompression ...
        pass

    # Step 3: Update coupling matrices with rotation matrices
    for (t, s) in Z_h2.all_admissible_blocks():
        S = Z_h2.coupling[t, s]

        if t in tree_t and s in tree_s:
            S = t.rotation @ block_diag(S, eye(k_target)) @ s.rotation.conj().T
        elif t in tree_t:
            S = t.rotation @ vstack([S, zeros(k_target, S.shape[1])])
        elif s in tree_s:
            S = hstack([S, zeros(S.shape[0], k_target)]) @ s.rotation.conj().T

        Z_h2.coupling[t, s] = S

    # Step 4: Update transfer matrices at subtree roots
    Z_h2.transfer[t0] = t0.rotation @ vstack([Z_h2.transfer[t0],
                                                zeros(k_target, k_target)])
    Z_h2.transfer[s0] = s0.rotation @ vstack([Z_h2.transfer[s0],
                                                zeros(k_target, k_target)])

    return Z_h2


def h2_matrix_multiply(A_h2, B_h2):
    """
    Compute C = A * B where A, B are H^2-matrices.

    Reduces to a sequence of local low-rank updates.
    Complexity: O(n k^2 log n)
    """
    C_h2 = zero_h2_matrix(A_h2.row_tree, B_h2.col_tree, k=A_h2.rank)

    def multiply_recursive(t, s, r):
        """Compute C|_{t x r} += A|_{t x s} * B|_{s x r}"""
        block_ts = A_h2.block_type(t, s)
        block_sr = B_h2.block_type(s, r)

        if block_ts == 'admissible' or block_sr == 'admissible':
            # At least one factor is low-rank → result is low-rank
            if block_ts == 'admissible':
                # A|_{t,s} = V_t S_{t,s} W_s^*
                # Product = V_t S_{t,s} (W_s^* B|_{s,r}) = V_t * (S_{t,s} @ X_tilde)
                X_tilde = h2_matvec_block(B_h2, s, r, A_h2.col_basis[s])
                X = A_h2.row_basis[t] @ A_h2.coupling[t,s] @ X_tilde
                Y = identity_block(r)  # simplified
            else:
                # Similar for B admissible
                pass

            # Apply as local low-rank update to C
            h2_local_lowrank_update(C_h2, t, r, X, Y, k_target=A_h2.rank)

        else:
            # Both are subdivided → recurse
            for t_child in sons(t):
                for s_child in sons(s):
                    for r_child in sons(r):
                        multiply_recursive(t_child, s_child, r_child)

    multiply_recursive(A_h2.row_tree.root,
                       A_h2.col_tree.root,
                       B_h2.col_tree.root)
    return C_h2


def h2_lr_factorization(A_h2):
    """
    Compute A = L * R (LR/LU factorization) in H^2-matrix arithmetic.

    Reduces to matrix multiplication + forward substitution,
    both using local low-rank updates.

    Complexity: O(n k^2 log n)
    """
    tree = A_h2.row_tree

    if tree.is_leaf:
        # Small dense factorization
        return dense_lu(A_h2.to_dense())

    t1, t2 = tree.root.children

    # Recursive block LR factorization:
    # A = [[A11, A12], [A21, A22]]
    # L11 R11 = A11                  (recurse)
    # L11 R12 = A12                  (forward substitution)
    # L21 R11 = A21                  (forward substitution)
    # L22 R22 = A22 - L21 R12       (recurse after update)

    L11, R11 = h2_lr_factorization(A_h2.subblock(t1, t1))
    R12 = h2_forward_sub(L11, A_h2.subblock(t1, t2))
    L21 = h2_backward_sub(R11, A_h2.subblock(t2, t1))

    # Schur complement update: A22 <- A22 - L21 * R12
    # This is an H^2 matrix product followed by subtraction
    update = h2_matrix_multiply(L21, R12)
    A22_updated = h2_subtract(A_h2.subblock(t2, t2), update)

    L22, R22 = h2_lr_factorization(A22_updated)

    return assemble_block_lr(L11, L21, L22, R11, R12, R22)
```

**Key Implementation Insights:**

1. **Recompression is the bottleneck**: Every local low-rank update triggers a recompression of the affected subtrees. The efficiency comes from only recompressing the local subtrees (not the entire matrix), costing $O((\#\hat{t}_0 + \#\hat{s}_0)k^2)$ per update

2. **Weight matrix trick**: The weight matrix $B_t$ collects all coupling interactions visible from cluster $t$, allowing a single SVD to determine the optimal basis $Q_t$. The QR pre-factorization of $B_t$ reduces the SVD to a $(2k) \times (2k)$ problem — $O(k^3)$ per node

3. **Nested basis preservation**: When updating transfer matrices $E_{t_0} \leftarrow R_{t_0} \begin{pmatrix} E_{t_0} \\ 0 \end{pmatrix}$, the nested property is automatically maintained for all ancestors, since they inherit the change through the transfer matrices. Only the root of each affected subtree needs explicit transfer matrix updates

4. **Factorization reduces to multiplication**: The LR factorization complexity matches the multiplication complexity $O(nk^2 \log n)$ because forward/backward substitution costs at most as much as multiplication (Theorem 7)

5. **Practical rank $k$**: For FEM/BEM applications, $k$ is typically 10–50 (depending on accuracy tolerance), making $k^2$ a moderate constant

## References

- Börm, S. & Reimer, K. (2018). Efficient arithmetic operations for rank-structured matrices based on hierarchical low-rank updates. *Computing and Visualization in Science*, 16(6), 247-258. arXiv:1402.5056.
- Börm, S. (2017). Hierarchical matrix arithmetic with accumulated updates. arXiv:1703.09085.
- Börm, S. (2010). *Efficient Numerical Methods for Non-local Operators: $\mathcal{H}^2$-Matrix Compression, Algorithms and Analysis*. EMS Tracts in Mathematics.
- Hackbusch, W. (2015). *Hierarchical Matrices: Algorithms and Analysis*. Springer.
- Börm, S. & Grasedyck, L. (2005). Hybrid cross approximation of integral operators. *Numerische Mathematik*, 101(2), 221-249.

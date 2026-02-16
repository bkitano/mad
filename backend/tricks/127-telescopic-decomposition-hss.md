# 127: Telescopic Decomposition for HSS Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Rational Krylov approximation theory
**Paper**: [papers/hss-matrix-functions.pdf] (Casulli, Kressner, Robol, 2024)
**Documented**: 2026-02-15

## Description

Telescopic decomposition is an unconventional representation of hierarchically semiseparable (HSS) matrices that enables fast computation of matrix functions $f(A)$ through recursive low-rank updates with rational Krylov subspaces. Instead of the standard HSS representation, telescopic decomposition represents a matrix as:

$$A = U \Lambda V^T + D$$

where $U, V$ are block diagonal orthonormal matrices and $D$ is recursively decomposed. This representation allows computing $f(A)x$ or $f(A)$ approximations with reduced execution times compared to divide-and-conquer methods, particularly for matrix functions like exponentials, inverse square roots, and sign functions.

The key advantage is that matrix functions can be approximated by recursively performing low-rank updates while keeping the matrices involved small, avoiding large-scale linear system solves.

## Mathematical Form

**Core Operation:**

$$
A = U^{(L)} A^{(L-1)} (V^{(L)})^T
$$

where recursively:

$$
A^{(L-1)} := (U^{(L)})^T (A - D^{(L)}) V^{(L)}
$$

**General Telescopic Decomposition:**

For a cluster tree $\mathcal{T}$ of depth $L$, a matrix $A \in \mathbb{R}^{n \times n}$ is in telescopic decomposition of rank $r$ if:

$$
A = D^{(L)} + U^{(L)} A^{(L-1)} (V^{(L)})^T
$$

with:
- $D_\tau \in \mathbb{R}^{|\tau| \times |\tau|}$ — Diagonal blocks (size $|\tau| \times |\tau|$ for node $\tau$)
- $U_\tau, V_\tau \in \mathbb{R}^{|\tau| \times r}$ — Orthonormal columns (if $\text{depth}(\tau) = L$ and $2r \times r$ otherwise)
- $U^{(L)}, V^{(L)}$ — Block diagonal matrices defined by $U_\tau, V_\tau$ as in standard form

**Standard Telescopic Decomposition:**

For each nonleaf node $\tau$ with children $\alpha, \beta$:

$$
D_\tau = \begin{pmatrix}
0 & \tilde{A}_{\alpha,\beta} \\
\tilde{A}_{\beta,\alpha} & 0
\end{pmatrix}
$$

This structure admits a one-to-one correspondence with HSS matrices.

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — Symmetric HSS matrix (can be generalized to nonsymmetric)
- $r$ — Telescopic rank (equals HSS rank)
- $L$ — Depth of cluster tree ($O(\log n)$ for balanced trees)
- $\mathcal{T}_{2r}^{(L-1)}$ — Balanced cluster tree of depth $L-1$ with leaves of size $2r$
- $f: \mathbb{C} \to \mathbb{C}$ — Scalar function to be applied element-wise to eigenvalues

**Recovery from Telescopic Decomposition:**

```
Algorithm: Recovering A from telescopic decomposition
Input: {U_τ, V_τ, D_τ} for all τ ∈ T
Output: Matrix A

A ← D_γ  (γ is the root)
for ℓ = 1, ..., L do
    A ← D^(ℓ) + U^(ℓ) A (V^(ℓ))^T
end for
```

## Complexity

| Operation | Naive Matrix Function | Telescopic HSS |
|-----------|----------------------|----------------|
| Computing $f(A)$ | $O(n^3)$ | $O(r^2 n)$ to $O(r^3 n)$ |
| Matrix function $f(A)x$ | $O(n^3)$ then $O(n^2)$ | $O(kr^2 n)$ where $k$ is Krylov iterations |
| Storage | $O(n^2)$ | $O(rn)$ |
| Construction from HSS | N/A | $O(r^3 L)$ per node, $O(r^3 n)$ total |

**Detailed Complexity for Matrix Functions:**

For computing $f(A)$ where $A$ is symmetric HSS:
- **Divide-and-conquer approach**: Requires solving shifted linear systems $(A - \xi_j I)^{-1}$ which are expensive even with HSS structure
- **Telescopic approach**: Recursively approximates via rational Krylov with only small matrices ($2r \times 2r$) at each level
- **Practical speedup**: 5-50× faster than divide-and-conquer for typical matrix functions

**Key insight**: No large-scale linear system needs to be solved; only small $2r \times 2r$ subproblems at each tree level.

## Applicability

Telescopic decomposition is particularly powerful for:

1. **Matrix Functions in Neural Networks**:
   - Computing $\exp(W)$ for continuous-time models
   - Inverse square roots $W^{-1/2}$ for whitening/normalization
   - Sign function $\text{sign}(W)$ for weight initialization or constraints

2. **State Space Models**:
   - Computing transition matrix functions $\exp(\Delta A)$ in discretization
   - Efficient evaluation of $f(A)x$ for state propagation

3. **Attention Mechanisms**:
   - Computing softmax-like functions via matrix exponentials
   - Kernel approximations requiring matrix functions

4. **Eigenvalue-Based Operations**:
   - Any operation expressible as $f(A) = V f(\Lambda) V^T$ for symmetric $A$
   - Fractional powers, logarithms, etc.

5. **Preconditioning**:
   - Computing approximate inverses or square roots for preconditioning

## Limitations

1. **HSS Structure Required**: Matrix must first be in HSS form; not applicable to general dense matrices
2. **Symmetric Preference**: Algorithm described primarily for symmetric matrices; nonsymmetric case requires additional considerations
3. **Function Regularity**: Works best for functions with good rational approximations (exponentials, inverse square roots, sign function)
4. **Construction Overhead**: Converting from standard HSS to telescopic form has $O(r^3 n)$ cost
5. **Approximation Quality**: Quality depends on rational Krylov approximation; may need many terms for oscillatory functions
6. **Tree Depth Dependency**: Complexity grows with tree depth $L = O(\log n)$

## Implementation Notes

```python
# Pseudocode for computing f(A)x via telescopic decomposition
def telescopic_matvec_function(A_telescopic, f, x):
    """
    Compute y ≈ f(A)x using telescopic decomposition
    A_telescopic: telescopic decomposition {U, V, D, tree}
    f: scalar function to apply
    x: vector
    Returns: y ≈ f(A)x

    Complexity: O(k*r^2*n) where k is number of Krylov iterations
    """
    tree = A_telescopic.tree
    L = tree.depth

    # Initialize at root
    y = f(A_telescopic.D_root) @ x

    # Recursive descent through tree
    for level in range(1, L+1):
        # At each level, perform low-rank update via rational Krylov
        for node in tree.nodes_at_depth(level):
            # Build rational Krylov subspace for f(A^(level-1))
            # where A^(level-1) is small (2r × 2r)
            A_small = node.A_reduced  # 2r × 2r matrix
            Q, T = rational_arnoldi(A_small, poles, k_iters)

            # Approximate f(A_small) ≈ Q @ f(T) @ Q.T
            f_T = apply_function_to_triangular(f, T)

            # Update y with telescopic contribution
            U_node = A_telescopic.U[node]
            V_node = A_telescopic.V[node]
            y += U_node @ (Q @ f_T @ Q.T @ (V_node.T @ x))

    return y

# Converting HSS to telescopic decomposition
def hss_to_telescopic(hss_matrix, tree):
    """
    Convert standard HSS representation to telescopic form

    Complexity: O(r^3 * n) where r is HSS rank
    """
    telescopic = {}

    # Compute principal submatrices A_α,α for leaf nodes
    for leaf in tree.leaves:
        telescopic.C_alpha = hss_matrix.D_alpha

    # Bottom-up: compute matrices for internal nodes
    for level in range(tree.depth - 1, 0, -1):
        for node in tree.nodes_at_depth(level):
            alpha, beta = node.children

            # Extract coupling structure
            U_alpha, V_alpha = hss_matrix.U[alpha], hss_matrix.V[alpha]
            U_beta, V_beta = hss_matrix.U[beta], hss_matrix.V[beta]

            # Build small coupling matrices
            D_alpha = telescopic.C_alpha
            D_beta = telescopic.C_beta

            # Standard telescopic form for siblings
            telescopic.D[node] = block_matrix([
                [0, U_alpha @ hss_matrix.B_alphabeta @ V_beta.T],
                [U_beta @ hss_matrix.B_betaalpha @ V_alpha.T, 0]
            ])

            # Update for next level
            A_reduced = (U_node.T @ (A - D_node) @ V_node)
            telescopic.C[node] = A_reduced

    return telescopic

# Fast matrix function computation
def compute_matrix_function(A_hss, f):
    """
    Compute f(A) for symmetric HSS matrix A

    Returns: Approximation to f(A) in HSS form
    """
    # Convert to telescopic form
    A_tele = hss_to_telescopic(A_hss, A_hss.tree)

    # Recursively approximate f(A) via small matrix functions
    result = A_tele.D_root.copy()

    for level in range(1, A_hss.tree.depth + 1):
        for node in A_hss.tree.nodes_at_depth(level):
            # Only need to compute f() of small 2r × 2r matrices
            A_small = A_tele.get_reduced_matrix(node)
            f_A_small = matrix_function_small(f, A_small)  # O(r^3)

            # Accumulate contribution
            U, V = A_tele.U[node], A_tele.V[node]
            result = result + U @ f_A_small @ V.T

    return result
```

**Key Implementation Insights:**

1. **Small Matrix Functions**: At each tree level, only compute $f()$ of $2r \times 2r$ matrices, not $n \times n$
2. **Rational Krylov**: Use pole selection strategies (e.g., Zolotarev points) for optimal approximation
3. **Avoid Linear Solves**: Unlike divide-and-conquer, no shifted system solves $(A - \xi I)^{-1}$ required
4. **Orthogonality**: Maintain orthonormal $U, V$ bases throughout for numerical stability
5. **Function-Specific**: Pole selection and iteration count depend on target function $f$

## References

- Casulli, A. A., Kressner, D., & Robol, L. (2024). Computing Functions of Symmetric Hierarchically Semiseparable Matrices. *SIAM Journal on Matrix Analysis and Applications*, 45(2), 1019-1043. arXiv:2402.17369
- Levitt, J., & Martinsson, P. G. (2024). Fast matrix function evaluations via rational Krylov methods. *SIAM Journal on Scientific Computing*.
- Robol, L., Vandebril, R., & Van Barel, M. (2016). A framework for structured linearizations of matrix polynomials in various bases. *SIAM Journal on Matrix Analysis and Applications*, 38(1), 188-216.

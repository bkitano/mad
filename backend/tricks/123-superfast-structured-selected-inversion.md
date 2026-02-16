# 123: Superfast Structured Selected Inversion

**Category**: decomposition
**Gain type**: efficiency
**Source**: Xia, Xi, Cauley, & Balakrishnan (2013), "Superfast Structured Selected Inversion for Large Sparse Matrices"
**Paper**: [papers/superfast-structured-selected-inversion.pdf]
**Documented**: 2026-02-15

## Description

Selected inversion extracts only the diagonal (and certain off-diagonal) blocks of $A^{-1}$ without computing the full inverse — a critical operation when only $\text{diag}(A^{-1})$ is needed (e.g., variance estimation, uncertainty quantification, Gaussian process posterior covariance, Fisher information diagonals). The naive approach computes $A^{-1}$ fully in $O(n^3)$, then reads the diagonal. This trick combines three ideas: (1) the **multifrontal method** organizes sparse elimination into a tree of dense "frontal matrices" connected by Schur complements; (2) **HSS (hierarchically semiseparable) compression** approximates the frontal matrices in structured low-rank form; and (3) a **fast Schur complement formula** (Theorem 3.1) shows that the product $U_k^T F_{\mathbf{i},\mathbf{i}}^{-1} U_k$ simplifies to $\tilde{U}_k^T \tilde{D}_k^{-1} \tilde{U}_k$ using shared HSS generators, reducing the cost from $O(r^2 N)$ (full HSS inversion) to $O(r^3)$ per node. The two-pass algorithm — forward LDL factorization (postorder traversal) then backward selected inversion (reverse postorder) — achieves $O(n)$ total complexity for 2D problems and $O(n^{4/3})$ for 3D problems, compared to $O(n^{1.5})$ and $O(n^2)$ for exact methods. Memory is also $O(n)$.

## Mathematical Form

**Problem Setting:**

Given a sparse symmetric matrix $A \in \mathbb{R}^{n \times n}$, compute $\text{diag}(A^{-1})$ (the diagonal entries of the inverse) without forming the full inverse.

**Step 1: Nested Dissection Reordering**

Reorder $A$ via nested dissection into a block structure:

$$
A = \begin{pmatrix} A_{11} & & A_{13} \\ & A_{22} & A_{23} \\ A_{31} & A_{32} & A_{33} \end{pmatrix}
$$

where $A_{11}$, $A_{22}$ are subdomain blocks and $A_{33}$ is the separator block.

**Step 2: Block LDL Factorization via Schur Complements**

$$
A = L D L^T
$$

where:

$$
L = \begin{pmatrix} I & & \\ & I & \\ L_{31} & L_{32} & I \end{pmatrix}, \quad D = \begin{pmatrix} A_{11} & & \\ & A_{22} & \\ & & \mathbf{F}_3 \end{pmatrix}
$$

with:

$$
L_{31} = A_{31} A_{11}^{-1}, \quad L_{32} = A_{32} A_{22}^{-1}
$$

$$
\mathbf{F}_3 = A_{33} - L_{31} A_{11} L_{31}^T - L_{32} A_{22} L_{32}^T
$$

Here $\mathbf{F}_3$ is the **Schur complement** — the "reduced" interface problem.

**Step 3: HSS Compression of Frontal Matrices**

Each frontal matrix $\mathbf{F}_\mathbf{i}$ at node $\mathbf{i}$ of the assembly tree is approximated in HSS form. An HSS matrix $F$ of order $N$ with HSS rank $r$ is defined recursively:

$$
D_i = F|_{t_i \times t_i} = \begin{pmatrix} D_{c_1} & U_{c_1} B_{c_1} V_{c_2}^T \\ U_{c_2} B_{c_2} V_{c_1}^T & D_{c_2} \end{pmatrix}
$$

with generators $D_i, U_i, V_i, R_i, B_i, W_i$ at each node $i$.

**Step 4: Fast Schur Complement Computation (Theorem 3.1)**

The key mathematical insight: for an HSS frontal matrix $\mathbf{F}_\mathbf{i}$ with generators $D_i, U_i$, and HSS inverse generators $\tilde{D}_i, \tilde{U}_i$:

$$
U_k^T F_{\mathbf{i},\mathbf{i}}^{-1} U_k = \tilde{U}_k^T \tilde{D}_k^{-1} \tilde{U}_k
$$

This means the Schur complement product reuses the **same** basis matrices between $F_{\mathbf{i},\mathbf{i}}$ and $F_{\mathbf{i},\mathbf{i}}^{-1}$, reducing computation from $O(r^2 N)$ to $O(r^3)$.

**Step 5: Selected Inversion (backward pass)**

For $C = A^{-1}$, using the block LDL factorization:

$$
C = \begin{pmatrix} I & & -L_{31}^T \\ & I & -L_{32}^T \\ & & I \end{pmatrix} \begin{pmatrix} A_{11}^{-1} & & \\ & A_{22}^{-1} & \\ & & \mathbf{F}_3^{-1} \end{pmatrix} \begin{pmatrix} I & & \\ & I & \\ -L_{31} & -L_{32} & I \end{pmatrix}
$$

The diagonal blocks of $C$ are:

$$
\text{diag}(C) = \left( A_{11}^{-1} + L_{31}^T \mathbf{F}_3^{-1} L_{31}, \quad A_{22}^{-1} + L_{32}^T \mathbf{F}_3^{-1} L_{32}, \quad \mathbf{F}_3^{-1} \right)
$$

Each correction term $L_{3i}^T \mathbf{F}_3^{-1} L_{3i}$ is computed in structured form using Theorem 3.1.

At each node in the backward traversal:

$$
C_{\mathbf{i},\mathbf{i}} = F_{\mathbf{i},\mathbf{i}}^{-1} - L_{\mathcal{N}_i, \mathbf{i}}^T C_{\mathcal{N}_i, \mathbf{i}}
$$

stored in HSS-plus-low-rank form:

$$
C_{\mathbf{i},\mathbf{i}} = F_{\mathbf{i},\mathbf{i}}^{-1} - P_k \Theta P_k^T, \quad \text{where } \Theta = B_k U_{k+1}^T P_{k+1} B_k^T
$$

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — sparse symmetric matrix (e.g., from PDE discretization)
- $\mathbf{F}_\mathbf{i}$ — frontal matrix at node $\mathbf{i}$ of the assembly tree
- $F_{\mathbf{i},\mathbf{i}}$ — diagonal block of frontal matrix (separator portion)
- $\mathbf{U}_\mathbf{i}$ — update matrix (Schur complement contribution): $\mathbf{U}_\mathbf{i} = F_{\mathcal{N}_i, \mathcal{N}_i} - L_{\mathcal{N}_i, \mathbf{i}} F_{\mathbf{i},\mathbf{i}}^{-1} F_{\mathcal{N}_i, \mathbf{i}}^T$
- $r$ — maximum HSS rank across all frontal matrices
- $\mathcal{T}$ — assembly tree from nested dissection; $\mathcal{N}_i$ — ancestor set of node $\mathbf{i}$
- $l_s$ — switching level: exact factorization below, HSS approximation above

## Complexity

| Operation | Exact Multifrontal (2D) | Structured (2D) | Exact Multifrontal (3D) | Structured (3D) |
|-----------|------------------------|-----------------|------------------------|-----------------|
| Factorization $\xi_\text{fact}$ | $O(n^{1.5})$ | $O(rn \log n)$ | $O(n^2)$ | $O(rn^{4/3})$ |
| Selected inversion $\xi_\text{inv}$ | $O(n^{1.5})$ | $O(rn)$ | $O(n^2)$ | $O(r^{3/2}n)$ |
| Memory $\sigma_\text{mem}$ | $O(n \log n)$ | $O(n)$ | $O(n^{4/3})$ | $O(r^{1/2}n)$ |

For constant HSS rank $r$: the selected inversion is **$O(n)$** for both 2D and 3D — truly superfast.

For growing rank patterns $r_l = O(N_l^{1/p})$:

| Rank pattern $r_l$ | Max rank $r$ | Inversion $\xi_\text{inv}$ | Memory |
|---------------------|-------------|---------------------------|--------|
| $O(1)$ | $O(1)$ | $O(n)$ | $O(n)$ |
| $O((\log N_l)^p)$ | $O((\log N)^p)$ | $O(n)$ | $O(n)$ |
| $O(N_l^{1/3})$ | $O(N^{1/3})$ | $O(n \log n)$ | $O(n \log^{1/2} n)$ |

**Parallel advantage:** The assembly tree structure exposes massive parallelism — all nodes at the same level can be processed simultaneously, giving $O(\log n)$ parallel depth.

## Applicability

- **Bayesian neural network uncertainty**: Computing $\text{diag}(\mathcal{F}^{-1})$ of the Fisher information matrix for posterior variance estimates — avoids full $O(n^3)$ inversion when the Fisher matrix has sparse/structured block form
- **Gaussian process inference**: GP posterior covariance requires $\text{diag}(K^{-1})$ where $K$ is the kernel matrix — if $K$ admits HSS structure (e.g., from spatial/temporal kernels), this trick extracts posterior variances in near-linear time
- **State space model conditioning**: When the state transition matrix $A$ in an SSM has block-sparse structure (e.g., multi-scale dynamics), selected inversion computes the needed diagonal blocks of $(zI - A)^{-1}$ for resolvent-based computations
- **Structured preconditioners**: Building preconditioners for iterative attention solvers where only the diagonal/block-diagonal of an approximate inverse is needed
- **Electronic structure / physics-informed networks**: The original application domain — extracting diagonal entries of Green's functions for quantum mechanical simulations
- **Sparse attention pattern analysis**: When attention matrices have known sparsity patterns (e.g., block-sparse, banded), the multifrontal + HSS framework can exploit this structure

## Limitations

- Requires the matrix to be **symmetric** (or symmetrizable) — not directly applicable to non-symmetric attention matrices without modification
- HSS compression quality depends on the **off-diagonal rank** of the frontal matrices — for some problems (e.g., high-frequency Helmholtz), ranks grow as $O(N)$ and the method loses its advantage
- The nested dissection + assembly tree construction is a non-trivial preprocessing step with its own $O(n \log n)$ cost
- Implementation complexity is high: requires HSS compression, structured LDL factorization, and careful management of generators at each tree node
- The **switching level** $l_s$ must be tuned to balance exact (below $l_s$) and structured (above $l_s$) operations for optimal performance
- Accuracy is controlled by the HSS compression tolerance $\tau$: smaller $\tau$ gives higher accuracy but larger ranks and higher cost (empirically, $\tau = 10^{-5}$ gives relative error $\sim 10^{-6}$)
- Currently demonstrated primarily on sparse matrices from PDE discretizations — applicability to the dense-but-structured matrices in neural networks (e.g., attention) requires further investigation

## Implementation Notes

```python
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

def selected_inversion_sketch(A_sparse, block_size=64):
    """Sketch of superfast structured selected inversion.

    In practice, this requires a full HSS library. Here we illustrate
    the key ideas with block LDL + Schur complement approach.

    The real algorithm uses:
    1. Nested dissection ordering (e.g., via METIS)
    2. Multifrontal LDL factorization
    3. HSS compression of frontal matrices
    4. Fast Schur complement formula (Theorem 3.1)
    5. Backward traversal for selected inversion
    """
    n = A_sparse.shape[0]

    # Step 1: Nested dissection reordering (simplified)
    # In practice: use METIS or similar graph partitioner
    # perm = nested_dissection(A_sparse)
    # A_reordered = A_sparse[perm][:, perm]

    # Step 2: Block LDL factorization with Schur complements
    # For illustration, consider a 3-block case:
    # A = [[A11, 0, A13], [0, A22, A23], [A31, A32, A33]]
    # L31 = A31 @ inv(A11)
    # L32 = A32 @ inv(A22)
    # F3 = A33 - L31 @ A11 @ L31.T - L32 @ A22 @ L32.T  (Schur complement)

    # Step 3: HSS compression of frontal matrices
    # F_hss = hss_compress(F3, tolerance=1e-5)

    # Step 4: Fast selected inversion (backward pass)
    # diag(A^{-1}) at separator block: just diag(F3^{-1})
    # diag(A^{-1}) at subdomain blocks:
    #   diag(C_11) = diag(A11^{-1}) + diag(L31.T @ F3^{-1} @ L31)
    #   The correction term uses Theorem 3.1:
    #   U_k^T F^{-1} U_k = tilde_U_k^T tilde_D_k^{-1} tilde_U_k
    #   which costs O(r^3) instead of O(r^2 * N)

    pass

def block_selected_inversion_dense(A, B, C, D):
    """Selected inversion for a 2x2 block matrix.

    Given M = [[A, B], [C, D]], compute diag(M^{-1}) using
    Schur complement-based selected inversion.

    Returns: (diag_block_11, diag_block_22) of M^{-1}
    """
    # Forward: compute Schur complement
    A_inv = np.linalg.inv(A)
    S = D - C @ A_inv @ B    # Schur complement of A

    # Backward: selected inversion
    S_inv = np.linalg.inv(S)

    # Block (2,2) of M^{-1} is S^{-1}
    diag_22 = np.diag(S_inv)

    # Block (1,1) of M^{-1} is A^{-1} + A^{-1} B S^{-1} C A^{-1}
    # For diagonal only, we compute:
    # diag(A^{-1}) + diag(A^{-1} B S^{-1} C A^{-1})
    A_inv_B = A_inv @ B
    correction = A_inv_B @ S_inv @ C @ A_inv
    diag_11 = np.diag(A_inv) + np.diag(correction)

    return diag_11, diag_22

def recursive_selected_inversion(blocks, tree_structure):
    """Recursive selected inversion following the assembly tree.

    Key insight: at each level, the Schur complement captures
    the coupling between subdomains. The backward pass propagates
    corrections down the tree using the fast formula:
        C_{i,i} = F_{i,i}^{-1} - P_k * Theta * P_k^T
    where Theta is a small r x r matrix.
    """
    # Forward pass (postorder): compute block LDL factors
    # For each node from leaves to root:
    #   - Form frontal matrix F_i (extend-add of children's updates)
    #   - Compress F_i in HSS form
    #   - Compute HSS inversion of F_{i,i}
    #   - Compute update matrix U_i (Schur complement)

    # Backward pass (reverse postorder): selected inversion
    # At root: C_{k,k} = F_k^{-1}
    # For each node from root to leaves:
    #   - Compute C_{N_i, i} = low-rank correction
    #   - Compute C_{i,i} = F_{i,i}^{-1} - correction
    #   - Extract diag(C_{i,i}) from HSS generators

    pass
```

## References

- Xia, J., Xi, Y., Cauley, S., & Balakrishnan, V. (2013). Superfast Structured Selected Inversion for Large Sparse Matrices. *Proceedings of the Project Review, Geo-Mathematical Imaging Group*, Purdue University, Vol. 1, pp. 139-156.
- Xia, J. (2013). Efficient Structured Multifrontal Factorization for General Large Sparse Matrices. *SIAM J. Sci. Comput.*, 35, A832-A860.
- Xia, J. (2013). Randomized Sparse Direct Solvers. *SIAM J. Matrix Anal. Appl.*, 34, 197-227.
- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2009). Superfast Multifrontal Method for Large Structured Linear Systems of Equations. *SIAM J. Matrix Anal. Appl.*, 31, 1382-1411.
- Chandrasekaran, S., Dewilde, P., Gu, M., & Pals, T. (2006). A Fast ULV Decomposition Solver for Hierarchically Semiseparable Representations. *SIAM J. Matrix Anal. Appl.*, 28, 603-622.
- Lin, L., Yang, C., Lu, J., Ying, L., & E, W. (2009). A Fast Algorithm for Extracting the Diagonal of the Inverse Matrix. *Commun. Math. Sci.*, 7, 755-777.
- Gillman, A., Young, P., & Martinsson, P. G. (2012). A Direct Solver with $O(N)$ Complexity for Integral Equations in One-Dimensional Domains. *Front. Math. China*, 7, 217-247.

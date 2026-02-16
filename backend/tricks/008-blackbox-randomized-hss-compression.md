# 008: Black-Box Randomized HSS Compression

**Category**: decomposition
**Gain type**: efficiency
**Source**: Levitt & Martinsson (2022), SIAM J. Sci. Comput.
**Paper**: [papers/blackbox-hss-compression.pdf]
**Documented**: 2026-02-15

## Description

Black-box randomized HSS compression is a linear-complexity algorithm for computing a Hierarchically Block Separable (HBS/HSS) representation of a matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ using *only* matrix-vector products — no individual matrix entries are needed. The algorithm draws two tall thin Gaussian random test matrices $\boldsymbol{\Omega}, \boldsymbol{\Psi} \in \mathbb{R}^{N \times s}$ with $s = O(k)$ columns (where $k$ is the block rank), computes sample matrices $\mathbf{Y} = \mathbf{A}\boldsymbol{\Omega}$ and $\mathbf{Z} = \mathbf{A}^*\boldsymbol{\Psi}$, and then reconstructs the full telescoping factorization of $\mathbf{A}$ from these samples alone.

The key innovation over prior "peeling" algorithms is achieving **true linear complexity** $O(k^2 N)$ with only $s = O(k)$ samples (typically $s \approx 3k$), compared to peeling algorithms that require $O(k \log N)$ samples and $O(k^2 N \log N)$ operations. The approach is fully "black-box" — no matrix entry evaluation is required — and all samples can be extracted in parallel, enabling streaming/single-view operation.

The core idea exploits the telescoping factorization structure: at each level, the algorithm isolates the off-diagonal information by nullifying contributions from the local diagonal block using the structure of the random test matrix, then compresses the resulting randomized samples via QR factorization.

## Mathematical Form

**Telescoping Factorization:**

The algorithm constructs a factorization of the form:

$$
\mathbf{A} = \mathbf{U}^{(L)} \tilde{\mathbf{A}}^{(L)} (\mathbf{V}^{(L)})^* + \mathbf{D}^{(L)}
$$

where recursively:

$$
\tilde{\mathbf{A}}^{(\ell+1)} = \mathbf{U}^{(\ell)} \tilde{\mathbf{A}}^{(\ell)} (\mathbf{V}^{(\ell)})^* + \mathbf{D}^{(\ell)}, \quad \ell = L-1, L-2, \ldots, 1
$$

with:
- $\mathbf{U}^{(\ell)} = \text{diag}(\mathbf{U}_\tau : \tau \text{ is a node on level } \ell)$ — block-diagonal basis matrices
- $\mathbf{V}^{(\ell)} = \text{diag}(\mathbf{V}_\tau : \tau \text{ is a node on level } \ell)$ — block-diagonal basis matrices
- $\mathbf{D}^{(\ell)} = \text{diag}(\mathbf{D}_\tau : \tau \text{ is a node on level } \ell)$ — discrepancy (remainder) matrices
- $\mathbf{D}^{(0)} = \tilde{\mathbf{A}}^{(\ell+1)}$ at the root

**Discrepancy Matrices (Key Innovation):**

Unlike standard HSS, the discrepancy is defined as:

$$
\mathbf{D}_\tau = \mathbf{A}_{\tau,\tau} - \mathbf{U}_\tau \mathbf{U}_\tau^* \mathbf{A}_{\tau,\tau} \mathbf{V}_\tau \mathbf{V}_\tau^*
$$

This "remainder after projection" definition is critical — it avoids needing to access individual matrix entries.

**Computing Basis Matrices at Leaf Level:**

For each leaf node $\tau$ with index set $I_\tau$, extract the local block of test/sample matrices:

$$
\boldsymbol{\Omega}_\tau = \boldsymbol{\Omega}(I_\tau, :), \quad \mathbf{Y}_\tau = \mathbf{Y}(I_\tau, :)
$$

Compute the nullspace projector that annihilates the diagonal contribution:

$$
\mathbf{P}_\tau = \text{null}(\boldsymbol{\Omega}_\tau, r)
$$

Then the product $\mathbf{Y}_\tau \mathbf{P}_\tau$ is a randomized sample of $\mathbf{A}(I_\tau, I_\tau^c)$ — the off-diagonal row block. The basis is found by orthonormalization:

$$
\mathbf{U}_\tau = \text{col}(\mathbf{Y}_\tau \mathbf{P}_\tau, r)
$$

Similarly for column bases using $\boldsymbol{\Psi}$ and $\mathbf{Z}$:

$$
\mathbf{Q}_\tau = \text{null}(\boldsymbol{\Psi}_\tau, r), \quad \mathbf{V}_\tau = \text{col}(\mathbf{Z}_\tau \mathbf{Q}_\tau, r)
$$

**Computing Discrepancy at Leaf Level:**

$$
\mathbf{D}_\tau = (\mathbf{I} - \mathbf{U}_\tau \mathbf{U}_\tau^*) \mathbf{Y}_\tau \boldsymbol{\Omega}_\tau^\dagger + \mathbf{U}_\tau \mathbf{U}_\tau^* \left( (\mathbf{I} - \mathbf{V}_\tau \mathbf{V}_\tau^*) \mathbf{Z}_\tau \boldsymbol{\Psi}_\tau^\dagger \right)^*
$$

**Recursive Compression (Coarser Levels):**

After compressing level $L$, extract samples of the reduced matrix $\tilde{\mathbf{A}}^{(L)}$ using:

$$
\underbrace{(\mathbf{U}^{(L)})^* (\mathbf{Y} - \mathbf{D}^{(L)} \boldsymbol{\Omega})}_{\text{sample matrix}} = \tilde{\mathbf{A}}^{(L)} \underbrace{(\mathbf{V}^{(L)})^* \boldsymbol{\Omega}}_{\text{test matrix}}
$$

For a non-leaf node $\tau$ at level $\ell$ with children $\alpha, \beta$:

$$
\boldsymbol{\Omega}_\tau = \begin{bmatrix} \mathbf{V}_\alpha^* \boldsymbol{\Omega}_\alpha \\ \mathbf{V}_\beta^* \boldsymbol{\Omega}_\beta \end{bmatrix}, \quad \mathbf{Y}_\tau = \begin{bmatrix} \mathbf{U}_\alpha^* (\mathbf{Y}_\alpha - \mathbf{D}_\alpha \boldsymbol{\Omega}_\alpha) \\ \mathbf{U}_\beta^* (\mathbf{Y}_\beta - \mathbf{D}_\beta \boldsymbol{\Omega}_\beta) \end{bmatrix}
$$

Then apply the same nullspace-projection and orthonormalization to find $\mathbf{U}_\tau, \mathbf{V}_\tau, \mathbf{D}_\tau$.

## Complexity

| Operation | Peeling Algorithm | Black-Box Compression |
|-----------|-------------------|----------------------|
| Matrix-vector products | $O(k \log N)$ | $O(k)$ |
| Floating point ops | $O(k^2 N \log N)$ | $O(k^2 N)$ |
| Matrix entry access | $O(kN)$ (partial) | **None** (fully black-box) |
| Total information | $\sim 5kN$ values | $\sim 6kN$ values |

**Detailed Complexity:**

$$
T_{\text{compress}} = 6rN \times T_{\text{rand}} + 6r \times T_{\text{mult}} + O(r^2 N) \times T_{\text{flop}}
$$

where $r = k + p$ (rank plus oversampling), $T_{\text{rand}}$ is the time to generate a random number, $T_{\text{mult}}$ is the time for a matrix-vector product with $\mathbf{A}$, and $T_{\text{flop}}$ is the time for a floating point operation.

**Per-node cost:** $O(r^3)$ for QR and nullspace computations, with $O(N/m)$ leaf nodes, giving $O(r^3 \cdot N/m) = O(r^2 N)$ total (since $m = 2r$).

**Memory:** $O(rN)$ for the HSS representation plus $O(sN)$ for the sample matrices.

**Sample count:** $s \geq \max(r + m, 3r)$ where $m$ is the leaf node size and $r = k + p$. In practice $s \approx 3k$ suffices.

## Applicability

1. **Implicit Matrix Compression**: When $\mathbf{A}$ is defined implicitly via a fast matrix-vector product (e.g., Fast Multipole Method, FFT-based convolution), the black-box approach avoids ever forming $\mathbf{A}$ explicitly
2. **Neural Network Weight Compression**: Compressing large weight matrices that have hierarchical low-rank structure into HSS form for efficient inference, using only forward/backward passes as the "matvec" oracle
3. **Attention Kernel Compression**: Compressing attention matrices $\text{softmax}(QK^T/\sqrt{d})$ that exhibit hierarchical low-rank structure (e.g., due to spatial/temporal locality)
4. **Schur Complement Computation**: When the Schur complement $\mathbf{S}_{22} = \mathbf{A}_{21}\mathbf{A}_{11}^{-1}\mathbf{A}_{12}$ inherits rank structure, the black-box approach compresses it without explicit formation
5. **Sparse Direct Solvers**: Compressing dense frontal matrices that arise during multifrontal sparse LU factorization (as implemented in STRUMPACK)
6. **State Space Model Discretization**: Compressing transition matrices $\exp(\Delta A)$ when a fast matrix exponential-vector product is available
7. **Streaming/Online Settings**: All $s$ random test vectors and matrix-vector products can be generated in a single batch, enabling pipelined GPU execution

## Limitations

1. **Known Rank Required**: The block rank $k$ must be known or estimated in advance (adaptive variants exist but are more complex)
2. **Oversampling Sensitivity**: Requires $s \approx 3k$ samples for reliability; insufficient oversampling leads to poor approximation
3. **Non-Adaptive**: Unlike the Gorman et al. (2018) adaptive algorithm, this fixed-sample approach cannot discover the rank during compression
4. **Gaussian Test Matrices**: Analysis assumes Gaussian random test matrices; structured random matrices (e.g., SRHT) may not work without modification
5. **Square Matrices**: Described for $N \times N$ matrices; rectangular generalizations require additional consideration
6. **Accuracy Depends on $k$**: If the true off-diagonal ranks vary significantly across levels, using a uniform $k$ may over- or under-approximate different parts of the matrix

## Implementation Notes

```python
# Black-box HSS compression (Algorithm 4.1 from Levitt & Martinsson)
def blackbox_hss_compress(matvec, matvec_adj, N, tree, k, p=10):
    """
    Compress matrix A into HSS form using only matvec access.

    Args:
        matvec: function computing A @ x
        matvec_adj: function computing A^* @ x
        N: matrix dimension
        tree: binary tree defining HSS structure
        k: block rank
        p: oversampling parameter (default 10)

    Returns:
        HSS representation {U_tau, V_tau, D_tau} for all nodes tau

    Complexity: O(k^2 * N) flops + O(k) matvecs
    """
    r = k + p
    m = 2 * r  # leaf node size
    s = 3 * r  # number of samples

    # Step 1: Generate random test matrices and compute samples
    Omega = randn(N, s)  # Gaussian test matrix
    Psi = randn(N, s)    # Gaussian test matrix

    Y = matvec(Omega)       # Y = A @ Omega,  O(s) matvecs
    Z = matvec_adj(Psi)     # Z = A^* @ Psi,  O(s) matvecs

    # Step 2: Compress level by level, finest to coarsest
    for level in range(tree.depth, -1, -1):  # L, L-1, ..., 0
        for tau in tree.nodes_at_level(level):
            # Extract local test and sample blocks
            if tau.is_leaf:
                Omega_tau = Omega[tau.indices, :]
                Psi_tau = Psi[tau.indices, :]
                Y_tau = Y[tau.indices, :]
                Z_tau = Z[tau.indices, :]
            else:
                alpha, beta = tau.children
                # Reduce via previously computed bases
                Omega_tau = vstack([V_alpha.T @ Omega_alpha,
                                    V_beta.T @ Omega_beta])
                Psi_tau = vstack([U_alpha.T @ Psi_alpha,
                                  U_beta.T @ Psi_beta])
                Y_tau = vstack([U_alpha.T @ (Y_alpha - D_alpha @ Omega_alpha),
                                U_beta.T @ (Y_beta - D_beta @ Omega_beta)])
                Z_tau = vstack([V_alpha.T @ (Z_alpha - D_alpha.T @ Psi_alpha),
                                V_beta.T @ (Z_beta - D_beta.T @ Psi_beta)])

            if level > 0:
                # Compute nullspace projectors
                P_tau = null(Omega_tau, r)  # annihilate diagonal contribution
                Q_tau = null(Psi_tau, r)

                # Extract off-diagonal samples and compress
                U_tau = col(Y_tau @ P_tau, r)    # orthonormal basis for row space
                V_tau = col(Z_tau @ Q_tau, r)     # orthonormal basis for col space

                # Compute discrepancy (remainder) matrix
                D_tau = (I - U_tau @ U_tau.T) @ Y_tau @ pinv(Omega_tau) \
                      + U_tau @ U_tau.T @ ((I - V_tau @ V_tau.T) @ Z_tau @ pinv(Psi_tau)).T

                # Store for recursive use
                tau.U, tau.V, tau.D = U_tau, V_tau, D_tau
            else:
                # Root node: just solve for D^(0) directly
                D_tau = Y_tau @ pinv(Omega_tau)
                tau.D = D_tau

    return tree  # contains {U_tau, V_tau, D_tau} for all nodes


def apply_hss_matvec(tree, q):
    """
    Apply compressed HBS matrix to vector: u = A @ q
    Uses upward pass + downward pass through the tree.

    Complexity: O(k * N)
    """
    # Upward pass: compress q through V bases
    for level in range(tree.depth, 0, -1):
        for tau in tree.nodes_at_level(level):
            if tau.is_leaf:
                tau.q_hat = V_tau.T @ q[tau.indices]
            else:
                alpha, beta = tau.children
                tau.q_hat = V_tau.T @ vstack([alpha.q_hat, beta.q_hat])

    # Downward pass: expand through U bases and add D contributions
    for level in range(0, tree.depth + 1):
        for tau in tree.nodes_at_level(level):
            if tau.is_root:
                alpha, beta = tau.children
                u_alpha, u_beta = D_tau @ vstack([alpha.q_hat, beta.q_hat])
                alpha.u_hat, beta.u_hat = u_alpha, u_beta
            elif tau.is_parent:
                alpha, beta = tau.children
                combined = U_tau @ tau.u_hat + D_tau @ vstack([alpha.q_hat,
                                                                beta.q_hat])
                alpha.u_hat, beta.u_hat = split(combined)
            else:  # leaf
                u[tau.indices] = U_tau @ tau.u_hat + D_tau @ q[tau.indices]

    return u
```

**Key Implementation Insights:**

1. **Null Space Trick**: The nullspace of $\boldsymbol{\Omega}_\tau$ (local test matrix block) annihilates the contribution from columns indexed by $I_\tau$, isolating the off-diagonal information without accessing matrix entries
2. **All Samples in Parallel**: The $s$ matrix-vector products $\mathbf{Y} = \mathbf{A}\boldsymbol{\Omega}$ can all be computed simultaneously, making this ideal for GPU batched operations
3. **Modified Discrepancy**: The discrepancy $\mathbf{D}_\tau$ includes both the projected and unprojected remainders, avoiding the need for element access
4. **Information Efficiency**: Requires $\sim 6kN$ values total (vs $\sim 5kN$ for semi-matrix-free methods), a modest 20% overhead for full black-box capability
5. **Streaming Mode**: Since all samples are drawn upfront, this works in settings where you can only observe $\mathbf{A}$ acting on vectors once (streaming/single-pass)

## References

- Levitt, J. & Martinsson, P. G. (2024). Linear-complexity black-box randomized compression of rank-structured matrices. *SIAM Journal on Scientific Computing*, 46(2), A1157-A1180. arXiv:2205.02990.
- Martinsson, P. G. (2011). A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix. *SIAM Journal on Matrix Analysis and Applications*, 32(4), 1251-1274.
- Gorman, C., Chavez, G., Ghysels, P., Mary, T., Rouet, F.-H., & Li, X. S. (2018). Matrix-free construction of HSS representation using adaptive randomized sampling. *arXiv:1810.04125*.
- Lin, L., Lu, J., & Ying, L. (2011). Fast construction of hierarchical matrix representation from matrix-vector multiplication. *Journal of Computational Physics*, 230(10), 4071-4087.

# 015: Block-Diagonal Schur Complement Saddle-Point Preconditioning

**Category**: decomposition
**Gain type**: efficiency
**Source**: Bergamaschi, Mart&iacute;nez & Pilotto (2026); extends Bergamaschi et al. (2024, 2026) on spectral bounds for block-diagonal preconditioners of multiple saddle-point systems
**Paper**: [papers/schur-complement-saddle-point-preconditioning.pdf]
**Documented**: 2026-02-15

## Description

Multiple saddle-point systems are symmetric block-tridiagonal indefinite linear systems of the form $\mathcal{A}x = b$ where diagonal blocks alternate in sign (positive, negative, positive, ...) and off-diagonal blocks couple adjacent fields. They arise naturally in constrained optimization (KKT systems), mixed finite element discretizations, and multi-physics coupling. The key preconditioning idea is to use a block-diagonal matrix $\mathcal{P}_D = \text{blkdiag}(S_0, S_1, \ldots, S_N)$ where each $S_k$ is a **recursively-defined Schur complement**:

$$S_0 = A_0, \qquad S_k = A_k + B_k S_{k-1}^{-1} B_k^T, \quad k = 1, \ldots, N$$

When the exact Schur complements are used, the preconditioned matrix $\mathcal{P}_D^{-1}\mathcal{A}$ has eigenvalues that are exactly the zeros of a sequence of Chebyshev-like polynomials, yielding at most $2(N+1)$ distinct eigenvalues. This means Krylov solvers (MINRES) converge in at most $N+1$ iterations — independent of the matrix dimension! In practice, exact Schur complements are too expensive, so inexact approximations $\widehat{S}_k \approx S_k$ are used (e.g., via incomplete Cholesky). The paper establishes tight eigenvalue bounds for the inexact case via parametric polynomial root analysis, showing that the spectral interval of $\mathcal{P}^{-1}\mathcal{A}$ is controlled by the extremal eigenvalues of the ratios $\widehat{S}_k^{-1} A_k$ and $\widehat{S}_k^{-1}(\widetilde{S}_k - A_k)$ where $\widetilde{S}_k = A_k + B_k \widehat{S}_{k-1}^{-1} B_k^T$.

## Mathematical Form

**Multiple Saddle-Point System:**

$$
\mathcal{A} = \begin{pmatrix} A_0 & B_1^T & & \\ B_1 & -A_1 & B_2^T & \\ & B_2 & A_2 & \ddots \\ & & \ddots & \ddots & B_N^T \\ & & & B_N & (-1)^N A_N \end{pmatrix}
$$

where $A_0 \in \mathbb{R}^{n_0 \times n_0}$ is symmetric positive definite, $A_k \in \mathbb{R}^{n_k \times n_k}$ ($k \geq 1$) are symmetric positive semi-definite, and $B_k \in \mathbb{R}^{n_k \times n_{k-1}}$ have full rank. The sign pattern on diagonal blocks alternates: $+, -, +, -, \ldots$

**Exact Block-Diagonal Preconditioner:**

$$
\mathcal{P}_D = \text{blkdiag}(S_0, S_1, \ldots, S_N)
$$

with recursive Schur complements:

$$
S_0 = A_0, \qquad S_k = A_k + B_k S_{k-1}^{-1} B_k^T, \quad k = 1, \ldots, N
$$

**Preconditioned System:**

$$
\mathcal{Q} = \mathcal{P}^{-1/2} \mathcal{A} \mathcal{P}^{-1/2} = \begin{pmatrix} E_0 & R_1^T & \\ R_1 & -E_1 & R_2^T \\ & R_2 & E_2 & \ddots \\ & & \ddots & \ddots \\ & & & R_N & (-1)^N E_N \end{pmatrix}
$$

where $R_k = \widehat{S}_k^{-1/2} B_k \widehat{S}_{k-1}^{-1/2}$ and $E_k = \widehat{S}_k^{-1/2} A_k \widehat{S}_k^{1/2}$, satisfying the identity:

$$
R_k R_k^T + E_k = \widehat{S}_k^{-1/2} \widetilde{S}_k \widehat{S}_k^{-1/2} \equiv \overline{S}_k
$$

**Spectral Characterization via Parametric Polynomials:**

The eigenvalues of $\mathcal{P}^{-1}\mathcal{A}$ are zeros of the parametric polynomial sequence:

$$
U_0(\lambda, \boldsymbol{\gamma}_R, \boldsymbol{\gamma}_E) = 1
$$

$$
U_1(\lambda, \boldsymbol{\gamma}_R, \boldsymbol{\gamma}_E) = \lambda - \gamma_E^{(0)}
$$

$$
U_{k+1}(\lambda) = (\lambda + (-1)^{k+1} \gamma_E^{(k)}) U_k(\lambda) - \gamma_R^{(k)} U_{k-1}(\lambda), \quad k \geq 1
$$

where $\gamma_E^{(k)} \in [\alpha_E^{(k)}, \beta_E^{(k)}]$ are Rayleigh quotients of $E_k$ and $\gamma_R^{(k)} \in [\alpha_R^{(k)}, \beta_R^{(k)}]$ are Rayleigh quotients of $R_k R_k^T$.

**Key Definitions:**

- $\alpha_E^{(k)} = \lambda_{\min}(E_k)$, $\beta_E^{(k)} = \lambda_{\max}(E_k)$ — extremal eigenvalues of preconditioned diagonal blocks
- $\alpha_R^{(k)} = \lambda_{\min}(R_k R_k^T)$, $\beta_R^{(k)} = \lambda_{\max}(R_k R_k^T)$ — extremal eigenvalues of preconditioned coupling
- $\mathcal{I}_k$ — interval containing zeros of $U_k$ for all admissible parameter combinations

**Eigenvalue Bounds (Theorem 2):**

$$
\sigma(\mathcal{P}^{-1}\mathcal{A}) \subseteq [\xi_{-,LB}^{(N+1)},\; b_{N+1}] \cup [a_{N+1},\; \xi_{+,UB}^{(N+1)}]
$$

where $b_{N+1} = \max\{\xi_{-,UB}^{(k)} : k \text{ even}\}$ and $a_{N+1} = \min\{\xi_{+,LB}^{(k)} : k \text{ odd}\}$.

**MINRES Convergence Bound:**

If eigenvalues lie in $[\rho_l^-, \rho_u^-] \cup [\rho_l^+, \rho_u^+]$ with $\rho_l^- < \rho_u^- < 0 < \rho_l^+ < \rho_u^+$ and $\rho_u^+ - \rho_l^+ = \rho_u^- - \rho_l^-$:

$$
\frac{\|r_k\|}{\|r_0\|} \leq 2 \left( \frac{\sqrt{|\rho_l^- \rho_u^+|} - \sqrt{|\rho_u^- \rho_l^+|}}{\sqrt{|\rho_l^- \rho_u^+|} + \sqrt{|\rho_u^- \rho_l^+|}} \right)^{\lfloor k/2 \rfloor}
$$

## Complexity

| Operation | Unpreconditioned | With Block-Diagonal Schur Preconditioner |
|-----------|-----------------|------------------------------------------|
| MINRES iterations | $O(n)$ | $O(N+1)$ with exact $S_k$; $O(\sqrt{\kappa})$ with inexact |
| Preconditioner setup | — | $O(\sum_k \text{IC}(A_k)) + O(N \cdot \text{solve}(S_{k-1}))$ |
| Per-iteration cost | $O(\text{nnz}(\mathcal{A}))$ | $O(\text{nnz}(\mathcal{A})) + O(\sum_k \text{solve}(\widehat{S}_k))$ |
| Biot poroelasticity ($N=2$) | 248 iterations ($\delta=10^{-3}$) | 64 iterations ($\delta=10^{-6}$) |

**Memory:** $O(\sum_k \text{nnz}(\widehat{S}_k))$ for the incomplete factorizations of the approximate Schur complements. The block-diagonal structure means each block can be stored independently.

**Parallelism:** Each block $\widehat{S}_k$ is independent in the application phase (block-diagonal solve), enabling $N+1$-way parallelism in the preconditioner application. The recursive Schur complement construction is sequential in $k$, but each individual factorization can use internal parallelism.

## Applicability

- **KKT systems in neural network optimization**: Second-order optimizers (natural gradient, K-FAC) produce KKT/saddle-point systems when equality constraints are present; the block-diagonal Schur preconditioner provides dimension-independent iteration counts
- **Constrained optimization layers**: Differentiable optimization layers (OptNet, differentiable convex optimization) require solving KKT systems in the forward/backward pass; this preconditioner enables efficient iterative solution for large-scale problems
- **Multi-physics coupling in scientific ML**: Physics-informed networks for coupled systems (fluid-structure, poroelasticity, electromagnetics) produce multiple saddle-point systems from mixed finite element discretizations
- **Block-structured attention with constraints**: Attention mechanisms with linear constraints (e.g., doubly stochastic attention) lead to augmented systems with saddle-point structure; the Schur complement preconditioner provides fast convergence
- **Equilibrium models and implicit layers**: Deep equilibrium networks (DEQs) require solving fixed-point equations; when the Jacobian has saddle-point structure (common in min-max formulations like GANs), the block-diagonal Schur preconditioner accelerates the Anderson/Broyden solver
- **Stokes-Darcy coupled problems**: Arising in computational fluid dynamics and used as benchmarks for Schur complement preconditioners

## Limitations

- **Recursive Schur complement cost**: Computing $S_k = A_k + B_k S_{k-1}^{-1} B_k^T$ requires applying $S_{k-1}^{-1}$, which is itself expensive — motivating the use of inexact approximations $\widehat{S}_k$ that introduce spectral perturbation
- **Quality of approximation matters**: The eigenvalue bounds depend on the extremal eigenvalues of $\widehat{S}_k^{-1} A_k$ and $\widehat{S}_k^{-1}(\widetilde{S}_k - A_k)$; poor approximations (large $\beta_E / \alpha_E$ ratio) lead to wide spectral intervals and slow convergence
- **Assumption $n_0 \geq n_1 \geq \cdots \geq n_N$**: The theory requires non-increasing block sizes; when violated (e.g., $n_2 > n_1$ in the Biot problem), the coupling matrix $R_k R_k^T$ becomes singular, requiring a modified eigenvalue analysis
- **Only for symmetric saddle-point systems**: The alternating sign structure and symmetry are essential; non-symmetric KKT systems (e.g., from non-symmetric constraints) require different preconditioners
- **Incomplete Cholesky fill-in trade-off**: The drop tolerance $\delta$ in IC factorization controls the trade-off between preconditioner quality and cost; too aggressive dropping ($\delta = 10^{-3}$) leads to 248 MINRES iterations vs. 64 for $\delta = 10^{-6}$, but the latter is more expensive per setup

## Implementation Notes

```python
import numpy as np
from scipy.sparse import block_diag, bmat
from scipy.sparse.linalg import LinearOperator, minres
from scipy.linalg import cho_factor, cho_solve

def build_schur_complement_preconditioner(A_blocks, B_blocks, approx='exact'):
    """
    Build block-diagonal Schur complement preconditioner for a
    multiple saddle-point system.

    A_blocks: list of N+1 symmetric (semi-)definite matrices [A_0, A_1, ..., A_N]
    B_blocks: list of N coupling matrices [B_1, B_2, ..., B_N]
    approx: 'exact' for exact Schur complements, 'ic' for incomplete Cholesky

    Returns: LinearOperator for the block-diagonal preconditioner P_D
    """
    N = len(B_blocks)
    S_factors = []

    # S_0 = A_0
    if approx == 'exact':
        S_factors.append(cho_factor(A_blocks[0].toarray()
                                     if hasattr(A_blocks[0], 'toarray')
                                     else A_blocks[0]))
    else:
        from scipy.sparse.linalg import spilu
        S_factors.append(spilu(A_blocks[0]))

    # Recursively compute S_k = A_k + B_k S_{k-1}^{-1} B_k^T
    for k in range(1, N + 1):
        B_k = B_blocks[k - 1]

        # Compute B_k S_{k-1}^{-1} B_k^T (or approximate it)
        if approx == 'exact':
            # Solve S_{k-1} Z = B_k^T => Z = S_{k-1}^{-1} B_k^T
            Z = cho_solve(S_factors[k - 1], B_k.T)
            S_k = A_blocks[k] + B_k @ Z
            S_factors.append(cho_factor(S_k))
        else:
            # For inexact: use IC approximation of S_k
            # S_tilde_k = A_k + B_k * S_hat_{k-1}^{-1} * B_k^T
            # Then approximate S_tilde_k with IC
            S_hat_prev = S_factors[k - 1]

            def apply_S_inv_BT(v):
                return S_hat_prev.solve(B_k.T @ v)

            # Approximate S_k via probing or direct assembly
            S_factors.append(spilu(A_blocks[k]))  # simplified

    # Build block-diagonal preconditioner application
    n_total = sum(A.shape[0] for A in A_blocks)
    block_sizes = [A.shape[0] for A in A_blocks]
    offsets = np.cumsum([0] + block_sizes)

    def apply_preconditioner(v):
        result = np.zeros_like(v)
        for k in range(N + 1):
            vk = v[offsets[k]:offsets[k + 1]]
            if approx == 'exact':
                result[offsets[k]:offsets[k + 1]] = cho_solve(S_factors[k], vk)
            else:
                result[offsets[k]:offsets[k + 1]] = S_factors[k].solve(vk)
        return result

    return LinearOperator((n_total, n_total), matvec=apply_preconditioner)


def solve_multiple_saddle_point(A_blocks, B_blocks, rhs, tol=1e-8):
    """
    Solve a symmetric multiple saddle-point system using MINRES
    with block-diagonal Schur complement preconditioning.

    System structure:
    [[ A_0,  B_1^T,              ]   [x_0]   [b_0]
     [ B_1, -A_1,   B_2^T,      ] * [x_1] = [b_1]
     [       B_2,   A_2,  ...   ]   [x_2]   [b_2]
     [              ...    B_N^T ]   [...]   [...]
     [                B_N, ±A_N  ]]  [x_N]   [b_N]

    Uses MINRES which is optimal for symmetric indefinite systems.
    With exact Schur complements: converges in N+1 iterations.
    With inexact approximations: converges based on spectral bounds.
    """
    # Build the full system matrix
    N = len(B_blocks)
    blocks = [[None] * (N + 1) for _ in range(N + 1)]
    for k in range(N + 1):
        sign = (-1)**k
        blocks[k][k] = sign * A_blocks[k] if k > 0 else A_blocks[k]
    for k in range(N):
        blocks[k + 1][k] = B_blocks[k]
        blocks[k][k + 1] = B_blocks[k].T

    A_full = bmat(blocks, format='csr')

    # Build preconditioner
    P = build_schur_complement_preconditioner(A_blocks, B_blocks, approx='exact')

    # Solve with MINRES
    x, info = minres(A_full, rhs, M=P, tol=tol)

    return x, info
```

## References

- Bergamaschi, L., Mart&iacute;nez, A. & Pilotto, M. (2026). Spectral Analysis of Block Diagonally Preconditioned Multiple Saddle-Point Matrices with Inexact Schur Complements. arXiv:2602.05952.
- Bergamaschi, L., Mart&iacute;nez, A., Pearson, J. W. & Potschka, A. (2024). Spectral analysis of block preconditioners for double saddle-point linear systems with application to PDE-constrained optimization. *Computational Optimization with Applications*, 91, 423–455.
- Bergamaschi, L. & Bergamaschi, M. (2026). Eigenvalue bounds for preconditioned symmetric multiple saddle-point matrices with SPD preconditioners. *Linear Algebra and its Applications*.
- Beik, F. P. A. & Benzi, M. (2018). Iterative methods for double saddle point systems. *SIAM J. Matrix Anal. Appl.*, 39, 902–921.
- Pearson, J. W. & Potschka, A. (2024). Double saddle-point preconditioning for Krylov methods in the inexact sequential homotopy method. *Numerical Linear Algebra with Applications*, 31, e2553.
- Beigl, A., Sogn, J. & Zulehner, W. (2020). Robust preconditioners for multiple saddle point problems and applications to optimal control problems. *SIAM J. Matrix Anal. Appl.*, 41, 1590–1615.
- Murphy, M. F., Golub, G. H. & Wathen, A. J. (2000). A note on preconditioning for indefinite linear systems. *SIAM J. Sci. Comput.*, 21(6), 1969–1972.

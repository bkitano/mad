# 095: Power Schur Complement Low-Rank Correction (PSLR)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Zheng, Xi & Saad (2020); extends Neumann series preconditioning with Sherman-Morrison-Woodbury low-rank deflation
**Paper**: [papers/power-schur-complement-low-rank.pdf]
**Documented**: 2026-02-15

## Description

The Power Schur complement Low-Rank (PSLR) preconditioner combines a Neumann-style power series expansion of the Schur complement inverse with a low-rank correction via the Sherman-Morrison-Woodbury formula. Given a block-reordered sparse matrix $A = \begin{pmatrix} B & E \\ F & C \end{pmatrix}$ (obtained via graph partitioning into $s$ subdomains), the bottleneck is solving systems with the Schur complement $S = C - FB^{-1}E$. Rather than forming $S$ explicitly or using ILU on it, PSLR approximates $S^{-1}$ by: (1) splitting $S = C_0 - E_s$ where $C_0$ is the block-diagonal of $C$, (2) expanding $(I - C_0^{-1}E_s)^{-1}$ as an $m$-term power series, and (3) correcting residual eigenvalues outside the unit circle with a rank-$r_k$ deflation via Arnoldi + Woodbury.

The crucial advantage over pure Neumann series is **robustness to indefiniteness**: when $\rho(C_0^{-1}E_s) > 1$ (divergent Neumann series), the low-rank correction deflates the offending eigenvalues, making the combined approximation converge. This enables preconditioning of highly indefinite systems — such as shifted Laplacians arising in Helmholtz equations — where standard ILU and AMG methods fail.

## Mathematical Form

**Block Structure (after graph partitioning):**

$$
Az = \begin{pmatrix} B & E \\ F & C \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} f \\ g \end{pmatrix}
$$

where $B \in \mathbb{R}^{p \times p}$ is block-diagonal (subdomain interiors), $C \in \mathbb{R}^{q \times q}$ is the interface block, and $E, F$ couple interiors to interfaces. The Schur complement is:

$$
S = C - FB^{-1}E
$$

**Schur Complement Splitting:**

$$
S = C_0 - E_s
$$

where $C_0 = \text{diag}(C_1, C_2, \ldots, C_s)$ is the block-diagonal of $C$ and $E_s = C_0 - S = (C_0 - C) + FB^{-1}E$.

**Power Series Expansion:**

$$
S^{-1} = (I - C_0^{-1}E_s)^{-1}C_0^{-1} \approx \sum_{i=0}^{m} (C_0^{-1}E_s)^i C_0^{-1}
$$

This converges when $\rho(C_0^{-1}E_s) < 1$ (guaranteed for SPD matrices with diagonally dominant $C$).

**Error Matrix:**

$$
E_{rr}(m) = (E_s C_0^{-1})^{m+1}
$$

The eigenvalue decay rate of $E_{rr}(m)$ is $m+1$ times faster than that of $E_{rr}(0)$.

**Low-Rank Correction via Arnoldi + Woodbury:**

Compute a rank-$r_k$ approximation of the error via Arnoldi iteration:

$$
E_{rr}(m) \approx V_{r_k} H_{r_k} V_{r_k}^T
$$

where $V_{r_k} \in \mathbb{R}^{q \times r_k}$ has orthonormal columns and $H_{r_k} = V_{r_k}^T E_{rr}(m) V_{r_k}$ is upper Hessenberg.

Apply the Sherman-Morrison-Woodbury formula to correct:

$$
S_{\text{app}}^{-1} = \left[\sum_{i=0}^{m} (C_0^{-1}E_s)^i C_0^{-1}\right] (I + V_{r_k} G_{r_k} V_{r_k}^T)
$$

where $G_{r_k} = (I - H_{r_k})^{-1} - I \in \mathbb{R}^{r_k \times r_k}$ is a small dense matrix.

**Key Definitions:**

- $B = \text{diag}(B_1, \ldots, B_s) \in \mathbb{R}^{p \times p}$ — block-diagonal subdomain interior matrices
- $C \in \mathbb{R}^{q \times q}$ — interface coupling matrix with block structure $C_{ij}$
- $C_0 = \text{diag}(C_1, \ldots, C_s)$ — block-diagonal of $C$ (local interface blocks)
- $E_s = C_0 - S$ — off-diagonal + fill-in correction
- $m$ — number of power series terms (controls approximation vs. cost)
- $r_k$ — rank of low-rank correction (controls deflation accuracy)
- $V_{r_k} \in \mathbb{R}^{q \times r_k}$ — Arnoldi basis vectors
- $G_{r_k} \in \mathbb{R}^{r_k \times r_k}$ — small dense Woodbury correction matrix

**Approximation Accuracy:**

$$
\frac{\|S^{-1} - S_{\text{app}}^{-1}\|}{\|S^{-1}\|} \leq \|X(m, r_k)\| \cdot \|Z(r_k)^{-1}\|
$$

where $X(m, r_k) = E_{rr}(m) - V_{r_k}H_{r_k}V_{r_k}^T$ and $Z(r_k) = I - V_{r_k}H_{r_k}V_{r_k}^T$.

**Spectral Clustering Property:**

$$
\lambda(S_{\text{app}}^{-1}S) = 1 - \lambda(Z(r_k)^{-1}X(m, r_k))
$$

When $X(m, r_k)$ is small (good power series + good low-rank approximation), the eigenvalues of the preconditioned system cluster near 1, ensuring fast Krylov convergence.

## Complexity

| Operation | Naive | With PSLR |
|-----------|-------|-----------|
| Solve $Sz = g'$ directly | $O(q^3)$ | — |
| ILU of $S$ | $O(q \cdot \text{nnz}(S))$, sequential | $O(\sum_i \text{nnz}(B_i) + \text{nnz}(C_i))$, parallel |
| Apply $S_{\text{app}}^{-1}$ to vector | — | $O(m \cdot q \cdot \text{nnz}_{\text{block}}) + O(q \cdot r_k)$ |
| Preconditioner construction | $O(q^2)$ ILU | $O(\sum_i \text{ilu}(B_i, C_i)) + O(m \cdot r_k \cdot q)$ Arnoldi |
| Krylov iterations (GMRES) | 171 (no precond.) | 78–86 (with PSLR, $m=3$, $r_k=15$) |

**Memory:** $O(\text{nnz}(ILU) + q \cdot r_k + r_k^2)$ where the ILU fill is on block-diagonal factors (small) and the low-rank correction adds $q \cdot r_k + r_k^2$ dense entries.

**Parallelism:** Highly parallel — the ILU factorizations of $B_i$ and $C_i$ are independent across subdomains; the power series application involves block-diagonal solves and sparse matrix-vector products; the Arnoldi procedure and Woodbury correction are sequential but operate on small ($r_k \times r_k$) systems.

## Applicability

- **Preconditioning large sparse linear systems in neural network training**: Natural gradient, Fisher information matrix inversion, and Hessian preconditioning for second-order optimizers where the matrix has block structure from layer/parameter grouping
- **Domain decomposition for distributed training**: When model parameters or data are partitioned across devices, the PSLR preconditioner enables efficient interface coupling solves
- **Shifted Laplacian systems (Helmholtz)**: Indefinite systems where standard preconditioners (ILU, AMG) fail — relevant to wave-equation-based neural architectures and physics-informed networks
- **Block-structured state space models**: SSMs with multi-scale or multi-compartment state matrices that induce natural block partitions; the Schur complement captures inter-compartment coupling
- **Graph neural networks**: Message passing on partitioned graphs where the interaction matrix is block-sparse; PSLR provides an efficient preconditioner for the implicit message passing (equilibrium GNN) formulation
- **Iterative refinement in structured attention**: When attention matrices have block-sparse + low-rank structure, PSLR-style preconditioning can accelerate iterative attention computation

## Limitations

- **Requires graph partitioning**: The matrix must first be reordered via a graph partitioner (e.g., METIS), adding a preprocessing step
- **Power series convergence**: When $\rho(C_0^{-1}E_s)$ is close to or slightly above 1, many terms $m$ may be needed, increasing cost
- **Low-rank correction rank selection**: The optimal $r_k$ depends on the number of eigenvalues of $E_{rr}(m)$ with magnitude $> 1$; if this number is large, the correction becomes expensive
- **Not a direct solver**: PSLR is a preconditioner for iterative methods (GMRES/CG), not a direct solver — convergence still depends on the problem
- **Symmetric vs. general**: Originally motivated by SPD systems; for highly nonsymmetric or strongly indefinite matrices, the spectral radius condition may require large $m$ and $r_k$

## Implementation Notes

```python
import numpy as np
from scipy.sparse.linalg import spilu, LinearOperator, gmres

def build_pslr_preconditioner(B_blocks, C_blocks, E_blocks, F_blocks, m=3, r_k=15):
    """
    Build the PSLR preconditioner.

    B_blocks: list of s subdomain interior matrices
    C_blocks: list of s local interface matrices (diagonal blocks of C)
    E_blocks, F_blocks: coupling matrices
    m: number of power series terms
    r_k: rank of low-rank correction

    Returns: a LinearOperator that applies S_app^{-1}
    """
    s = len(B_blocks)

    # Step 1: ILU factorization of each B_i and C_i (parallel across subdomains)
    B_ilu = [spilu(B_i) for B_i in B_blocks]
    C_ilu = [spilu(C_i) for C_i in C_blocks]

    # Step 2: Arnoldi procedure to approximate E_rr(m)
    # Apply E_rr(m) operator = (E_s C_0^{-1})^{m+1} via repeated matvecs
    def apply_E_rr(v):
        """Apply E_rr(m) = (E_s C_0^{-1})^{m+1} to a vector."""
        for _ in range(m + 1):
            # Apply C_0^{-1}: solve block-diagonal system
            w = block_diagonal_solve(C_ilu, v)
            # Apply E_s = (C_0 - C) + F B^{-1} E
            v = apply_E_s(w, B_ilu, C_blocks, E_blocks, F_blocks)
        return v

    # Arnoldi iteration to get V_{r_k}, H_{r_k}
    V_rk, H_rk = arnoldi(apply_E_rr, q, r_k)

    # Compute G_{r_k} = (I - H_{r_k})^{-1} - I
    G_rk = np.linalg.inv(np.eye(r_k) - H_rk) - np.eye(r_k)

    # Step 3: Build the preconditioner application
    def apply_pslr(v):
        """Apply S_app^{-1} = [sum (C_0^{-1} E_s)^i C_0^{-1}] (I + V G V^T)"""
        # Low-rank correction: (I + V G V^T) v
        v_corrected = v + V_rk @ (G_rk @ (V_rk.T @ v))

        # Power series: sum_{i=0}^{m} (C_0^{-1} E_s)^i C_0^{-1} v
        y = block_diagonal_solve(C_ilu, v_corrected)  # i=0 term
        result = y.copy()
        for i in range(1, m + 1):
            y = apply_C0inv_Es(y, C_ilu, E_blocks, F_blocks, B_ilu)
            result += y

        return result

    return LinearOperator((q, q), matvec=apply_pslr)

def solve_with_pslr(A, b, s=35, m=3, r_k=15):
    """
    Solve Ax = b using GMRES with PSLR preconditioner.

    1. Partition A via graph partitioning into s subdomains
    2. Build PSLR preconditioner for the Schur complement
    3. Solve interface system with preconditioned GMRES
    4. Back-substitute to recover interior unknowns
    """
    # Reorder A into block form
    B, E, F, C, perm = graph_partition_reorder(A, s)

    # Build PSLR preconditioner
    M = build_pslr_preconditioner(B, C, E, F, m, r_k)

    # Solve Schur complement system: S y = g - F B^{-1} f
    rhs = compute_schur_rhs(B, E, F, b, perm)
    y, info = gmres(S_operator, rhs, M=M, tol=1e-8)

    # Recover interior unknowns: x = B^{-1}(f - E y)
    x = back_substitute(B, E, y, b, perm)

    return combine_solution(x, y, perm)
```

## References

- Zheng, Q., Xi, Y., & Saad, Y. (2020). A Power Schur Complement Low-Rank Correction Preconditioner for General Sparse Linear Systems. arXiv:2002.00917.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
- Xi, Y., Li, R., & Saad, Y. (2016). Schur complement based domain decomposition preconditioners with low-rank corrections. arXiv:1505.04340.
- Li, R. & Saad, Y. (2013). Multilevel Schur complement Low-Rank (MSLR) preconditioner. *SIAM J. Sci. Comput.*, 35(6), A2697–A2720.
- Zheng, Q., Xi, Y., & Saad, Y. (2022). parGeMSLR: A Parallel Multilevel Schur Complement Low-Rank Preconditioning and Solution Package. arXiv:2205.03224.

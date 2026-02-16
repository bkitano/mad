# 180: Superoptimal Circulant Preconditioner

**Category**: approximation
**Gain type**: efficiency
**Source**: Tyrtyshnikov "Optimal and Superoptimal Circulant Preconditioners" (SIAM J. Matrix Anal. Appl., 1992)
**Paper**: [papers/tyrtyshnikov-superoptimal-circulant-1992.pdf]
**Documented**: 2026-02-15

## Description

While the **optimal circulant preconditioner** (T. Chan, trick 084) minimizes $\|C - A\|_F$ over all circulant matrices $C$, the **superoptimal circulant preconditioner** (Tyrtyshnikov 1992) minimizes a fundamentally different objective:

$$
\hat{C} = \arg\min_{C \in \mathscr{F}_n} \|I - CA\|_F
$$

where $\mathscr{F}_n$ is the set of real nonsingular circulant matrices of order $n$. This minimizes the **residual** $I - CA$ directly, rather than the approximation error $C - A$. The distinction matters because a good approximation to $A$ need not be a good preconditioner — what matters for iterative solver convergence is how close $C^{-1}A$ is to the identity.

The key mathematical insight is that the superoptimal preconditioner $\hat{C}$ satisfies $\hat{C} = U^{-1}F$ where $U$ is the optimal circulant approximation to $AA^T$ and $F$ is the optimal circulant approximation to $A$. In other words, $\hat{C}^{-1} = FU^{-1}$, so applying the superoptimal preconditioner is equivalent to first applying the optimal circulant approximation of $A$ and then correcting by the (circulant approximation of the) Gram matrix $AA^T$.

Critical properties inherited from the original matrix $A$:
- If $A$ is nonsingular and positive definite, then $\hat{C}$ is also nonsingular and positive definite
- If $A = A^T$, then $\hat{C} = \hat{C}^T$ (symmetry is preserved)
- The eigenvalues of $\hat{C}^{-1}A$ cluster around 1 (often more tightly than $C_{\text{opt}}^{-1}A$)

For **Toeplitz** matrices, the superoptimal preconditioner can be computed in $O(n \log n)$ operations, the same cost as computing the optimal preconditioner. For **general** matrices, it costs $O(n^2 \log n)$. Application of the preconditioner (matrix-vector product) costs only $O(n \log n)$ via FFT regardless of structure.

## Mathematical Form

**The Optimization Problem:**

Let $\hat{C} = C^{-1}$ where $C$ is the superoptimal preconditioner. We seek to minimize:

$$
\|I - \hat{C}A\|_F, \quad \hat{C} \in \mathscr{F}_n
$$

Writing $\hat{C} = \sum_{j=0}^{n-1} \hat{c}_j Q^j$ where $Q$ is the backward cyclic permutation matrix:

$$
Q = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \\ 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{bmatrix}
$$

**Minimization via Trace Expansion (Eq. 4.5):**

$$
\mathscr{F}(\hat{c}_0, \ldots, \hat{c}_{n-1}) = \left\|I - \sum_{j=0}^{n-1} \hat{c}_j Q^j A\right\|_F^2 = n - \sum_{i=0}^{n-1} \hat{c}_i f_i + \sum_{i=0}^{n-1}\sum_{j=0}^{n-1} \hat{c}_i \hat{c}_j u_{ij}
$$

where:

$$
f_i = \text{tr}(A^T Q^{-i} + Q^i A), \quad u_{ij} = \text{tr}(A^T Q^{j-i} A)
$$

Setting $\partial \mathscr{F}/\partial \hat{c}_i = 0$ yields the linear system:

$$
(U + U^T) \hat{\mathbf{c}} = \mathbf{f}
$$

where $U = [u_{ij}]$ is a **circulant matrix** with entries $u_{ij} = \text{tr}(A^T Q^{j-i} A)$.

**Closed-Form Solution (Eq. 4.14-4.16):**

Since $U + U^T = 2U$ (because $U$ is symmetric positive definite when $A$ is nonsingular), and denoting $F = [f_0, \ldots, f_{n-1}]^T = [\text{tr}(Q^{1-j}A)]_{j=0}^{n-1}$:

$$
U\hat{\mathbf{c}} = F
$$

The circulant $\tilde{U} = \frac{1}{n}U$ is the **optimal circulant approximation of $AA^T$** (by Theorem 3.1), and the circulant $\tilde{F} = \frac{1}{n}F$ is the **optimal circulant approximation of $A$** (equation 4.16-4.17). Therefore:

$$
\hat{C}^{-1} = \tilde{F} \cdot \tilde{U}^{-1}
$$

or equivalently, $\hat{C} = \tilde{U} \cdot \tilde{F}^{-1}$, where both $\tilde{U}$ and $\tilde{F}$ are circulant matrices.

**Eigenvalue Spectrum of $\hat{C}^{-1}A$:**

The preconditioned matrix $\hat{C}^{-1}A$ has eigenvalues that cluster around 1. The key improvement over the optimal preconditioner:

- Optimal: eigenvalues of $C_{\text{opt}}^{-1}A$ cluster around 1 but can spread over $[0, \kappa(A)]$
- Superoptimal: eigenvalues of $\hat{C}^{-1}A$ cluster more tightly, with $\|I - \hat{C}^{-1}A\|_F \leq \|I - C_{\text{opt}}^{-1}A\|_F$ by construction

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — the matrix to be preconditioned
- $\hat{C}$ — superoptimal circulant preconditioner (the inverse is circulant)
- $\tilde{F}$ — optimal circulant approximation of $A$ (T. Chan's preconditioner)
- $\tilde{U}$ — optimal circulant approximation of $AA^T$
- $Q$ — backward cyclic permutation matrix
- $\mathscr{F}_n$ — set of real nonsingular circulant matrices of order $n$
- $u_{ij} = \text{tr}(A^T Q^{j-i} A)$ — entries of the circulant Gram matrix
- $f_i = \text{tr}(A^T Q^{-i} + Q^i A)$ — right-hand side vector

## Complexity

| Operation | Optimal (Chan) | Superoptimal (Tyrtyshnikov) | Dense |
|-----------|---------------|---------------------------|-------|
| Compute preconditioner (general $A$) | $O(n^2)$ | $O(n^2 \log n)$ | — |
| Compute preconditioner (Toeplitz $A$) | $O(n)$ | $O(n \log n)$ | — |
| Compute preconditioner (doubly Toeplitz $A$) | $O(n \log n)$ | $O(n \log n)$ | — |
| Apply preconditioner (matvec) | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ |
| Storage | $O(n)$ | $O(n)$ | $O(n^2)$ |

**Fast Algorithm for Toeplitz $A$ (Corollary 4, Section 5):**

For a Toeplitz matrix $A$, the diagonal sums $s_k = \sum_{i-j \equiv k \pmod{n}} a_{ij}$ (needed for both $\tilde{F}$ and $\tilde{U}$) reduce to Toeplitz-by-vector multiplications via Theorem 5.1. Writing $A = L + L^T$ (lower + upper triangular Toeplitz parts), the values $s_k(AA^T)$ decompose as:

$$
\begin{bmatrix} s_0 \\ s_1 \\ \vdots \\ s_{n-1} \end{bmatrix} = P \begin{bmatrix} r_{n-1} \\ r_{n-2} \\ \vdots \\ r_0 \end{bmatrix} + Q \begin{bmatrix} 0 \\ r_{n-1} \\ 2r_{n-2} \\ \vdots \\ (n-1)r_1 \end{bmatrix}
$$

where $P$ and $Q$ are Toeplitz matrices and $r_j$ are the Toeplitz coefficients. Each Toeplitz-by-vector multiplication costs $O(n \log n)$ via circulant embedding + FFT.

**For symmetric real Toeplitz $A$ of order $n = 2^L$ (Corollary 5):** The superoptimal preconditioner requires only **4 complex FFTs of order $n$** and **3 complex FFTs of order $n$** (total: 7 FFTs), reducible to FFTs of complex vectors of order $n/4$.

**Memory:** $O(n)$ — two circulant defining vectors ($\tilde{F}$ and $\tilde{U}$), each of length $n$.

## Applicability

- **Toeplitz system preconditioning for sequence models**: For learned Toeplitz token mixers (TNN, S4-style convolution kernels), the superoptimal preconditioner provides tighter spectral clustering of $\hat{C}^{-1}A$ around unity compared to Chan's optimal preconditioner, meaning fewer CG/GMRES iterations. Since both preconditioners have the same $O(n \log n)$ application cost, the superoptimal version yields faster total convergence at negligible extra construction cost
- **Implicit inverse approximation**: The superoptimal preconditioner directly approximates $A^{-1}$ rather than $A$. In normalizing flows or invertible sequence layers, this provides a better starting point for computing $A^{-1}x$ via iterative refinement
- **Circulant-cycle preconditioning refinement**: When combined with trick 028 (circulant cycle decomposition), the superoptimal preconditioner for the dominant $k$-cycle submatrix provides a refined preconditioner that accounts for both the cycle structure and the inverse structure simultaneously
- **Two-level/doubly Toeplitz systems**: For block-Toeplitz-with-Toeplitz-blocks matrices (arising in 2D convolution, image processing), the superoptimal doubly circulant preconditioner can be computed in $O(n \log n)$ and preserves positive-definiteness — critical for SPD-constrained iterative solvers
- **Gradient preconditioning**: In second-order optimization for structured layers (Toeplitz, circulant), the superoptimal circulant preconditioner of the Hessian provides a better-conditioned search direction than the simple circulant (Chan) preconditioner

## Limitations

- **Higher construction cost for general matrices**: For arbitrary $A$, computing the superoptimal preconditioner costs $O(n^2 \log n)$ vs $O(n^2)$ for the optimal one — the extra $\log n$ factor comes from needing to compute $s_k(AA^T)$ via FFT-based convolutions
- **Requires nonsingularity**: Both $A$ and the resulting preconditioner must be nonsingular. Unlike Chan's optimal preconditioner (which always exists), the superoptimal preconditioner's existence requires that $A$ be nonsingular
- **Not always better spectral clustering**: Tyrtyshnikov's numerical examples (Section 6) show that while $\hat{C}^{-1}A$ has smaller $\|I - \hat{C}^{-1}A\|_F$, the superoptimal preconditioner can sometimes produce a wider eigenvalue spread than the optimal one (e.g., $C^{-1}A$ eigenvalues in $[0.58, 3]$ vs $T^{-1}A$ in $[0.58, 2.4]$ for the $1/\sqrt{1+|i-j|}$ Toeplitz)
- **Complex arithmetic for non-symmetric $A$**: When $A \neq A^T$, the circulant preconditioner $\hat{C}$ need not be symmetric, requiring complex eigenvalue handling even though the matrix is real
- **GPU memory access pattern**: Computing the preconditioner requires forming $AA^T$ (or its Toeplitz analog), which involves a non-trivial memory access pattern. However, **applying** the preconditioner is simply two FFT-based circulant matvecs — highly GPU-friendly with coalesced memory access and existing cuFFT support

## Implementation Notes

```python
import torch
import torch.fft as fft

def superoptimal_circulant_preconditioner(A):
    """Compute Tyrtyshnikov's superoptimal circulant preconditioner.

    Minimizes ||I - C_hat * A||_F over all circulant C_hat.
    The preconditioner inverse C_hat^{-1} = F_tilde * U_tilde^{-1}
    where F_tilde = optimal circulant of A, U_tilde = optimal circulant of AA^T.

    Args:
        A: (n, n) real nonsingular matrix

    Returns:
        f_eig: (n,) FFT eigenvalues of F_tilde (optimal circulant of A)
        u_eig: (n,) FFT eigenvalues of U_tilde (optimal circulant of AA^T)
    """
    n = A.shape[0]

    # Step 1: Compute optimal circulant of A (Chan's preconditioner)
    # c_k = (1/n) * sum of entries on k-th wrapped diagonal
    f_row = torch.zeros(n, dtype=A.dtype, device=A.device)
    for k in range(n):
        idx = torch.arange(n, device=A.device)
        f_row[k] = A[idx, (idx + k) % n].mean()

    # Step 2: Compute optimal circulant of AA^T
    AAT = A @ A.T
    u_row = torch.zeros(n, dtype=A.dtype, device=A.device)
    for k in range(n):
        idx = torch.arange(n, device=A.device)
        u_row[k] = AAT[idx, (idx + k) % n].mean()

    # Step 3: FFT eigenvalues
    f_eig = fft.fft(f_row)
    u_eig = fft.fft(u_row)

    return f_eig, u_eig


def superoptimal_circulant_toeplitz(t_row, t_col):
    """Compute superoptimal circulant preconditioner for Toeplitz A.

    Uses the fast algorithm: O(n log n) instead of O(n^2 log n).

    Args:
        t_row: (n,) first row of Toeplitz matrix [t_0, t_{-1}, ..., t_{-(n-1)}]
        t_col: (n,) first column of Toeplitz matrix [t_0, t_1, ..., t_{n-1}]

    Returns:
        f_eig: (n,) FFT eigenvalues of F_tilde
        u_eig: (n,) FFT eigenvalues of U_tilde
    """
    n = len(t_row)

    # Optimal circulant of A (Chan's formula)
    f_row = torch.zeros(n, dtype=t_row.dtype, device=t_row.device)
    f_row[0] = t_row[0]
    for k in range(1, n):
        f_row[k] = ((n - k) * t_row[k] + k * t_col[n - k]) / n

    # For AA^T of Toeplitz A, use the Toeplitz structure:
    # AA^T is also Toeplitz when A is symmetric; otherwise use
    # Theorem 5.1 to compute diagonal sums via Toeplitz matvecs.

    # For symmetric Toeplitz (t_row == t_col):
    # Compute s_k(AA^T) via circulant embedding
    # Embed Toeplitz coefficients in circulant of size 2n
    c_embed = torch.zeros(2 * n, dtype=t_row.dtype, device=t_row.device)
    c_embed[:n] = t_col
    c_embed[n+1:] = torch.flip(t_row[1:], [0])

    c_fft = fft.fft(c_embed)
    # AA^T diagonal sums = circular convolution of Toeplitz coefficients
    aat_fft = c_fft * c_fft.conj()
    aat_circ = fft.ifft(aat_fft).real[:n]

    # Optimal circulant of AA^T
    u_row = torch.zeros(n, dtype=t_row.dtype, device=t_row.device)
    u_row[0] = aat_circ[0]
    for k in range(1, n):
        u_row[k] = ((n - k) * aat_circ[k] + k * aat_circ[n - k]) / n

    f_eig = fft.fft(f_row)
    u_eig = fft.fft(u_row)

    return f_eig, u_eig


def apply_superoptimal_precond(f_eig, u_eig, b):
    """Apply superoptimal preconditioner: x = C_hat^{-1} b = F_tilde * U_tilde^{-1} * b.

    Two FFT-based circulant matvecs: O(n log n) total.

    Args:
        f_eig: (n,) FFT eigenvalues of F_tilde
        u_eig: (n,) FFT eigenvalues of U_tilde
        b: (..., n) right-hand side vector(s)

    Returns:
        x: (..., n) preconditioned vector(s)
    """
    # Step 1: z = U_tilde^{-1} b (circulant solve via FFT)
    b_fft = fft.fft(b.to(torch.complex64), dim=-1)
    z_fft = b_fft / u_eig
    z = fft.ifft(z_fft, dim=-1)

    # Step 2: x = F_tilde z (circulant matvec via FFT)
    z_fft2 = fft.fft(z, dim=-1)
    x_fft = f_eig * z_fft2
    x = fft.ifft(x_fft, dim=-1).real

    return x


# Example: Compare optimal vs superoptimal preconditioning quality
def compare_preconditioners(A):
    """Compare spectral clustering of optimal vs superoptimal."""
    n = A.shape[0]

    # Optimal circulant (Chan)
    c_opt = torch.zeros(n)
    for k in range(n):
        idx = torch.arange(n)
        c_opt[k] = A[idx, (idx + k) % n].mean()
    c_opt_eig = fft.fft(c_opt)

    # Superoptimal
    f_eig, u_eig = superoptimal_circulant_preconditioner(A)

    # Form preconditioned matrices
    C_opt_inv_A = torch.zeros(n, n, dtype=torch.complex64)
    C_sup_inv_A = torch.zeros(n, n, dtype=torch.complex64)

    for j in range(n):
        e_j = torch.zeros(n)
        e_j[j] = 1.0
        Ae_j = A @ e_j

        # Optimal: C_opt^{-1} A e_j
        ae_fft = fft.fft(Ae_j.to(torch.complex64))
        C_opt_inv_A[:, j] = fft.ifft(ae_fft / c_opt_eig)

        # Superoptimal: (F U^{-1}) A e_j
        C_sup_inv_A[:, j] = apply_superoptimal_precond(
            f_eig, u_eig, Ae_j
        ).to(torch.complex64)

    # Residual norms
    I = torch.eye(n, dtype=torch.complex64)
    res_opt = torch.norm(I - C_opt_inv_A, p='fro').item()
    res_sup = torch.norm(I - C_sup_inv_A, p='fro').item()

    print(f"||I - C_opt^{{-1}} A||_F = {res_opt:.4f}")
    print(f"||I - C_sup^{{-1}} A||_F = {res_sup:.4f}")
    print(f"Superoptimal improvement: {res_opt/res_sup:.2f}x")

    return res_opt, res_sup
```

## References

- Tyrtyshnikov, E.E. "Optimal and Superoptimal Circulant Preconditioners" SIAM J. Matrix Anal. Appl., 13(2):459-473, 1992
- Chan, T.F. "An Optimal Circulant Preconditioner for Toeplitz Systems" SIAM J. Sci. Stat. Comput., 9(4):766-771, 1988
- Strang, G. "A Proposal for Toeplitz Matrix Calculations" Studies in Applied Mathematics, 74:171-176, 1986
- Chan, R.H. & Strang, G. "The Asymptotic Toeplitz-Circulant Eigenvalue Problem" Numerical Analysis Report 87-5, MIT, 1987
- Di Benedetto, F. & Fiorentino, G. & Serra, S. "Superoptimal Preconditioners for Functions of Matrices" Numer. Math. Theory Methods Appl., 2018
- Reichel, L. & Sadok, H. "A Note on Superoptimal Generalized Circulant Preconditioners" Applied Numerical Mathematics, 2013

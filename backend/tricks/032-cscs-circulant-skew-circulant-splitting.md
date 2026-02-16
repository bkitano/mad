# 032: CSCS: Circulant and Skew-Circulant Splitting

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ng "Circulant and skew-circulant splitting methods for Toeplitz systems" (J. Comput. Applied Math., 2003); Liu et al. "The eigen-structures of real (skew) circulant matrices with some applications" (arXiv 2018)
**Paper**: [papers/cscs-circulant-skew-circulant-splitting.pdf]
**Documented**: 2026-02-15

## Description

Any Toeplitz matrix $T$ can be uniquely decomposed as the sum of a circulant matrix $C$ and a skew-circulant matrix $S$: $T = C + S$. Both $C$ and $S$ are diagonalizable in $O(n \log n)$ — circulant matrices by the standard DFT matrix $F$, and skew-circulant matrices by a "twisted" DFT matrix $\tilde{F} = DF^*$ where $D = \text{diag}(1, e^{i\pi/n}, e^{2i\pi/n}, \ldots)$. This splitting enables an **alternating iterative solver** (the CSCS iteration) for Toeplitz systems $Tx = b$ that converges unconditionally when both $C$ and $S$ are positive definite, with each iteration costing only $O(n \log n)$ via FFT.

The key advance in Liu et al. (2018) is replacing the complex FFT with **real Schur decompositions** of $C$ and $S$, factored through DCT and DST matrices. The resulting **DCT-DST version of CSCS** operates entirely in real arithmetic, saving half the storage and approximately half the FLOPs compared to the FFT version ($\frac{25}{2}n \log n$ real FLOPs per iteration vs $30n \log n$ for FFT-CSCS). Numerically, the DCT-DST version runs nearly 2× faster at large $n$.

This is directly relevant to neural network sequence models because Toeplitz matrices naturally arise in relative-position token mixing (TNN), state-space model convolution kernels (S4, DSS), and causal linear attention. The CSCS splitting provides a way to decompose any learned Toeplitz token mixer into two FFT-diagonalizable components, each admitting efficient inversion, preconditioning, and spectral analysis.

## Mathematical Form

**Toeplitz Matrix:**

A Toeplitz matrix $T = (t_{jk})$ with $t_{jk} = t_{j-k}$ for $j, k = 0, \ldots, n-1$.

**Circulant and Skew-Circulant Splitting (CSCS):**

$$
T = C + S
$$

where $C = (c_{jk})$ is a circulant matrix and $S = (s_{jk})$ is a skew-circulant matrix, defined by:

$$
c_{jk} = \begin{cases} \frac{1}{2}t_0, & \text{if } j = k \\ \frac{t_{j-k} + t_{j-k-n}}{2}, & \text{otherwise} \end{cases}
$$

$$
s_{jk} = \begin{cases} \frac{1}{2}t_0, & \text{if } j = k \\ \frac{t_{j-k} - t_{j-k-n}}{2}, & \text{otherwise} \end{cases}
$$

**Schur Canonical Forms:**

Circulant: $C = F \Lambda F^*$ where $\Lambda$ holds eigenvalues of $C$.

Skew-circulant: $S = \tilde{F} \tilde{\Lambda} \tilde{F}^*$ where $\tilde{F} = DF^*$ with $D = \text{diag}(1, e^{i\pi/n}, \ldots, e^{(n-1)i\pi/n})$ and $\tilde{\Lambda}$ holds eigenvalues of $S$.

Both $\Lambda$ and $\tilde{\Lambda}$ can be obtained in $O(n \log n)$ via FFTs of the first rows of $C$ and $S$.

**Real Schur Form of Circulant Matrix $C$ (even case, $n = 2m$):**

$$
U^T C U = \Omega = \begin{bmatrix} \alpha_0 & & & & & \beta_1 \\ & \alpha_1 & & & \ddots & \\ & & \ddots & & & \\ & & & \alpha_m & & \\ & & & -\beta_{m-1} & \alpha_{m-1} & \\ & -\beta_1 & & & & \alpha_1 \end{bmatrix}
$$

where eigenvalues $\lambda_k = \alpha_k + i\beta_k$ and $U$ is a real orthogonal matrix constructed from real/imaginary parts of DFT eigenvectors.

**Real Schur Form of Skew-Circulant Matrix $S$:**

$$
\tilde{U}^T S \tilde{U} = \Sigma
$$

with analogous block-diagonal structure using eigenvalues $\tilde{\lambda}_k = \tilde{\alpha}_k + i\tilde{\beta}_k$.

**DCT-DST Factorization:**

The orthogonal matrices $U$ and $\tilde{U}$ can be factored through DCT and DST matrices via an auxiliary orthogonal matrix $Q$:

$$
QU = \begin{bmatrix} \mathcal{C}^I_{m+1} & \\ & J_{m-1}\mathcal{S}^I_{m-1}J_{m-1} \end{bmatrix} \quad (n = 2m)
$$

$$
Q^T\tilde{U} = \begin{bmatrix} \mathcal{C}^{II}_m & \\ & J_m \mathcal{S}^{II}_m J_m \end{bmatrix} \quad (n = 2m)
$$

where $\mathcal{C}^I, \mathcal{C}^{II}$ are DCT Type I and II matrices, $\mathcal{S}^I, \mathcal{S}^{II}$ are DST Type I and II matrices, and $J$ is the reversal matrix.

**CSCS Iteration (FFT version):**

Given initial guess $x^{(0)}$, for $k = 0, 1, \ldots$ until convergence:

$$
\begin{cases} F(\theta I + \Lambda)F^* x^{(k+\frac{1}{2})} = \tilde{F}(\theta I - \tilde{\Lambda})\tilde{F}^* x^{(k)} + b \\ \tilde{F}(\theta I + \tilde{\Lambda})\tilde{F}^* x^{(k+1)} = F(\theta I - \Lambda)F^* x^{(k+\frac{1}{2})} + b \end{cases}
$$

where $\theta > 0$ is a positive constant.

**CSCS Iteration (DCT-DST version, $n = 2m$):**

$$
\begin{cases} Q^T \begin{bmatrix} \mathcal{C}^I_{m+1} \\ & \mathcal{S}^I_{m-1} \end{bmatrix} (\theta I + \Omega) \begin{bmatrix} \mathcal{C}^I_{m+1} \\ & \mathcal{S}^I_{m-1} \end{bmatrix}^T (Qx^{(k+\frac{1}{2})}) = Q \begin{bmatrix} \mathcal{C}^{II}_m \\ & \mathcal{S}^{II}_m \end{bmatrix} (\theta I - \Sigma) \begin{bmatrix} \mathcal{C}^{II}_m \\ & \mathcal{S}^{II}_m \end{bmatrix}^T (Q^T x^{(k)}) + b \\ Q \begin{bmatrix} \mathcal{C}^{II}_m \\ & \mathcal{S}^{II}_m \end{bmatrix} (\theta I + \Sigma) \begin{bmatrix} \mathcal{C}^{II}_m \\ & \mathcal{S}^{II}_m \end{bmatrix}^T (Q^T x^{(k+1)}) = Q^T \begin{bmatrix} \mathcal{C}^I_{m+1} \\ & \mathcal{S}^I_{m-1} \end{bmatrix} (\theta I - \Omega) \begin{bmatrix} \mathcal{C}^I_{m+1} \\ & \mathcal{S}^I_{m-1} \end{bmatrix}^T (Qx^{(k+\frac{1}{2})}) + b \end{cases}
$$

**Key Definitions:**

- $T \in \mathbb{R}^{n \times n}$ — Toeplitz matrix
- $C$ — circulant part of $T$: entries are averages of opposite diagonals
- $S$ — skew-circulant part of $T$: entries are half-differences of opposite diagonals
- $\Omega, \Sigma$ — real Schur forms (block-diagonal with $2 \times 2$ rotation blocks)
- $\mathcal{C}^I, \mathcal{S}^I$ — DCT-I and DST-I matrices
- $\mathcal{C}^{II}, \mathcal{S}^{II}$ — DCT-II and DST-II matrices
- $\theta > 0$ — iteration parameter (controls convergence rate)
- $J$ — reversal (anti-identity) matrix

## Complexity

| Operation | FFT-CSCS (per iter) | DCT-DST CSCS (per iter) | Direct Toeplitz solve |
|-----------|--------------------|-----------------------|----------------------|
| FLOPs | $30n \log n$ (complex) | $\frac{25}{2}n \log n$ (real) | $O(n \log^2 n)$ |
| Storage | $4n$ real values | $2n$ real values | $O(n)$ |
| Arithmetic | Complex | **Real only** | Complex |
| FFTs per iter | 6 FFTs of $n$-vectors | 6 DCTs + 6 DSTs of $\sim n/2$-vectors | — |

**Convergence:** The CSCS iteration converges unconditionally when both $C$ and $S$ are positive definite (guaranteed for symmetric positive definite Toeplitz $T$). The convergence rate depends on $\theta$ and the spectral radius $\rho$ of the iteration matrix.

**Practical speedup (from paper, $n = 8000$):**
- Symmetric PD Toeplitz: DCT-DST version ~1.7× faster than FFT version (83.6s vs 152.9s)
- Non-symmetric PD Toeplitz: DCT-DST version ~2× faster than FFT version (49.8s vs 95.7s)
- DCT-DST CSCS: 10–25× fewer iterations than AHSS iteration at comparable cost per iteration

**Memory:** $O(n)$ — only the defining vectors of $C$ and $S$ (each $n$ entries) plus workspace for DCT/DST transforms.

## Applicability

- **Toeplitz system solving in sequence models**: When a learned Toeplitz token mixer needs to be inverted (e.g., for bidirectional inference, normalizing flows, or Newton steps), the CSCS iteration provides an efficient $O(n \log n)$-per-step iterative solver that avoids the $O(n \log^2 n)$ complexity of superfast direct Toeplitz solvers
- **Preconditioning for Toeplitz-structured attention**: In models with Toeplitz-structured attention (TNN, relative position encoders), the circulant component $C$ serves as a natural preconditioner for conjugate gradient methods, with the CSCS decomposition providing a more refined two-level preconditioner
- **Real-arithmetic Toeplitz matrix-vector products**: The DCT-DST algorithms for $Cx$ and $Sx$ enable computing $Tx = Cx + Sx$ entirely in real arithmetic, avoiding complex FFT overhead — useful on hardware with optimized DCT/DST (video codecs, DSP)
- **State-space model analysis**: SSM convolution kernels produce causal Toeplitz matrices; the CSCS splitting decomposes these into circulant + skew-circulant parts, each with closed-form spectral characterization, enabling spectral analysis and regularization
- **Fractional diffusion equations**: The CSCS iteration has been applied as a fast solver for Toeplitz systems arising from discretization of fractional diffusion operators, with polynomial CSCS preconditioners achieving mesh-independent convergence

## Limitations

- **Convergence requires PD splitting**: The unconditional convergence guarantee requires both $C$ and $S$ to be positive definite. For general Toeplitz matrices (e.g., non-symmetric or indefinite), convergence is not guaranteed without shifting ($\theta$ must be chosen carefully)
- **Iteration count**: While each iteration is $O(n \log n)$, the total iteration count depends on the condition number. For ill-conditioned Toeplitz systems, many iterations may be needed, and direct superfast solvers ($O(n \log^2 n)$ total) may be preferable
- **Parameter $\theta$ sensitivity**: The optimal $\theta$ is difficult to compute analytically; the paper notes it must be chosen "experimentally approximately optimal." Poor $\theta$ choices can significantly slow convergence
- **Not a direct solver**: Unlike LU-based or displacement-rank-based direct Toeplitz solvers, CSCS is iterative and does not provide exact solutions in finite steps
- **DCT/DST library maturity**: On GPUs, FFT libraries (cuFFT) are more optimized than DCT/DST implementations, potentially negating the theoretical real-arithmetic advantage

## Implementation Notes

```python
import torch
import torch.fft as fft

def toeplitz_cscs_split(t):
    """Split Toeplitz coefficients into circulant + skew-circulant parts.

    Given Toeplitz matrix T with T_{jk} = t_{j-k},
    the CSCS decomposition is T = C + S where:
    - C is circulant: c_{j-k} = (t_{j-k} + t_{j-k-n}) / 2
    - S is skew-circulant: s_{j-k} = (t_{j-k} - t_{j-k-n}) / 2

    Args:
        t: (2n-1,) Toeplitz coefficients [t_{-(n-1)}, ..., t_0, ..., t_{n-1}]
           where t[n-1] = t_0 (the main diagonal)

    Returns:
        c_row: (n,) first row of circulant C
        s_row: (n,) first row of skew-circulant S
    """
    n = (len(t) + 1) // 2

    # First row of T: [t_0, t_{-1}, t_{-2}, ..., t_{-(n-1)}]
    t_row = torch.flip(t[:n], [0])  # [t_0, t_{-1}, ..., t_{-(n-1)}]

    # First column of T: [t_0, t_1, t_2, ..., t_{n-1}]
    t_col = t[n-1:]  # [t_0, t_1, ..., t_{n-1}]

    # Circulant first row: c_k = (t_{-k} + t_{n-k}) / 2 for k=0,...,n-1
    # c_0 = t_0/2 + t_0/2 = t_0 (special case: we split diagonal evenly)
    c_row = torch.zeros(n, dtype=t.dtype, device=t.device)
    c_row[0] = t_row[0]  # t_0 (full, since both C and S get t_0/2 on diagonal)
    for k in range(1, n):
        c_row[k] = (t_row[k] + t_col[n - k]) / 2

    # Skew-circulant first row: s_k = (t_{-k} - t_{n-k}) / 2
    s_row = torch.zeros(n, dtype=t.dtype, device=t.device)
    s_row[0] = 0  # s_0 = 0 (diagonal already in C)
    for k in range(1, n):
        s_row[k] = (t_row[k] - t_col[n - k]) / 2

    return c_row, s_row


def circulant_matvec(c_row, x):
    """Compute circulant matrix-vector product via FFT: C @ x."""
    c_fft = fft.fft(c_row)
    x_fft = fft.fft(x)
    return fft.ifft(c_fft * x_fft).real


def skew_circulant_matvec(s_row, x):
    """Compute skew-circulant matrix-vector product via twisted FFT.

    S = F_tilde * diag(eigenvalues) * F_tilde^*
    where F_tilde = D * F^* with D = diag(1, e^{iπ/n}, ..., e^{i(n-1)π/n})
    """
    n = len(x)
    # Twist factor: D = diag(e^{ikπ/n}) for k=0,...,n-1
    k = torch.arange(n, device=x.device, dtype=torch.float32)
    twist = torch.exp(1j * torch.pi * k / n)

    # Eigenvalues of S via twisted FFT of first row
    s_twisted = s_row * twist.real  # for real s_row
    s_eig = fft.fft(s_row * twist)

    # Multiply: x -> twist*x -> FFT -> diag multiply -> IFFT -> un-twist
    x_twisted = x * twist
    x_fft = fft.fft(x_twisted)
    y_fft = s_eig * x_fft
    y_twisted = fft.ifft(y_fft)
    y = (y_twisted * twist.conj()).real

    return y


def cscs_solve(t, b, theta=None, max_iter=100, tol=1e-7):
    """Solve Tx = b via CSCS iteration.

    T = C + S (circulant + skew-circulant splitting).
    Each iteration:
      (θI + C) x^{k+1/2} = (θI - S) x^k + b
      (θI + S) x^{k+1}   = (θI - C) x^{k+1/2} + b

    Both sub-steps are diagonal in their respective Fourier bases,
    so each costs O(n log n).

    Args:
        t: (2n-1,) Toeplitz coefficients
        b: (n,) right-hand side
        theta: iteration parameter (auto-selected if None)
        max_iter: maximum iterations
        tol: convergence tolerance (relative residual)

    Returns:
        x: (n,) approximate solution
    """
    n = len(b)
    c_row, s_row = toeplitz_cscs_split(t)

    # Eigenvalues of C and S
    c_eig = fft.fft(c_row)  # circulant eigenvalues

    # Skew-circulant eigenvalues via twisted FFT
    k = torch.arange(n, device=b.device, dtype=torch.float32)
    twist = torch.exp(1j * torch.pi * k / n)
    s_eig = fft.fft(s_row.to(torch.complex64) * twist)

    # Auto-select theta if not provided
    if theta is None:
        theta = max(c_eig.real.max().item(), s_eig.real.max().item()) * 0.5

    # Iteration
    x = torch.zeros_like(b)
    r0_norm = b.norm()

    for iteration in range(max_iter):
        # Half-step: solve (θI + C) x_half = (θI - S) x + b
        rhs1 = theta * x - skew_circulant_matvec(s_row, x) + b
        rhs1_fft = fft.fft(rhs1.to(torch.complex64))
        x_half = fft.ifft(rhs1_fft / (theta + c_eig)).real

        # Full step: solve (θI + S) x_new = (θI - C) x_half + b
        rhs2 = theta * x_half - circulant_matvec(c_row, x_half) + b
        rhs2_twisted = rhs2.to(torch.complex64) * twist
        rhs2_fft = fft.fft(rhs2_twisted)
        x_new_twisted = fft.ifft(rhs2_fft / (theta + s_eig))
        x = (x_new_twisted * twist.conj()).real.float()

        # Check convergence
        residual = b - circulant_matvec(c_row, x) - skew_circulant_matvec(s_row, x)
        if residual.norm() / r0_norm < tol:
            break

    return x
```

## References

- Ng, M.K. "Circulant and skew-circulant splitting methods for Toeplitz systems" J. Comput. Applied Math., 159:101-108, 2003
- Liu, Z.Y., Chen, S., Xu, W. & Zhang, Y. "The eigen-structures of real (skew) circulant matrices with some applications" Comp. Appl. Math. 38:1-13, 2019. arXiv:1806.05652
- Chen, F. & Jiang, Y.-L. "On HSS and AHSS iteration methods for nonsymmetric positive definite Toeplitz systems" J. Comput. Appl. Math. 234:2432-2440, 2010
- Bai, Z.-Z., Golub, G. & Ng, M.K. "Hermitian and Skew-Hermitian Splitting Methods for Non-Hermitian Positive Definite Linear Systems" SIAM J. Matrix Anal. Appl. 24:603-626, 2003
- Huckle, T. "Circulant and skewcirculant matrices for solving Toeplitz matrix problems" SIAM J. Matrix Anal. Appl. 13(3):767-777, 1992
- Chan, R. & Ng, M. "Conjugate gradient methods for Toeplitz systems" SIAM Review 38:427-482, 1996

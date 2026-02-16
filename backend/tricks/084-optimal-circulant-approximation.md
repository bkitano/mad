# 084: Optimal Circulant Approximation (T. Chan's Preconditioner)

**Category**: approximation
**Gain type**: efficiency
**Source**: T. Chan "An Optimal Circulant Preconditioner for Toeplitz Systems" (SIAM J. Sci. Stat. Comput., 1988); Salahub "Approximating real symmetric Toeplitz matrices using the nearest circulant" (arXiv 2022)
**Paper**: [papers/optimal-circulant-approximation.pdf], [papers/nearest-circulant-toeplitz.pdf]
**Documented**: 2026-02-15

## Description

Given any $n \times n$ matrix $A$ (typically Toeplitz), the **optimal circulant approximation** $C^*$ is the circulant matrix that minimizes $\|C - A\|_F$ over all circulant matrices $C$. This has a remarkably simple closed-form solution: the $m$-th entry of $C^*$'s defining vector is the **weighted average of the $m$-th and $(n-m)$-th diagonals** of $A$. For a general matrix, it is equivalent to averaging the entries along each "wrapped diagonal" — the set of entries $(i,j)$ where $(i - j) \bmod n = m$.

The key insight is that the projection from the space of all $n \times n$ matrices onto the $n$-dimensional subspace of circulant matrices has a trivially computable closed form: **diagonal averaging with wrap-around**. Since circulant matrix-vector products can be computed in $O(n \log n)$ via FFT, this provides a principled way to convert any structured or unstructured matrix into an FFT-compatible form.

For Toeplitz matrices $A$ with entries $a_{ij} = \rho_{|i-j|}$, the optimal circulant has defining entries:

$$
c_m = \rho_m + \frac{m}{M}(\rho_{M-m} - \rho_m)
$$

This is a weighted interpolation between $\rho_m$ (the $m$-th Toeplitz coefficient) and $\rho_{M-m}$ (the "wrap-around" coefficient), with the weight $m/M$ controlling how much the circulant deviates from the Toeplitz structure. For small $m$ relative to $M$, $c_m \approx \rho_m$ — the circulant closely matches the Toeplitz matrix near its central diagonals, with errors concentrated in the corners.

This is distinct from **Strang's preconditioner**, which copies the central diagonals and wraps them around directly. Chan's optimal preconditioner generally achieves a lower condition number for $C^{-1}A$ and better spectral clustering around unity.

## Mathematical Form

**Optimal Circulant for General Matrix:**

Given any $A \in \mathbb{R}^{n \times n}$, define the circulant $C^* = \text{circ}(c_0, c_1, \ldots, c_{n-1})$ where:

$$
c_m = \frac{1}{n} \sum_{k=0}^{n-1} A_{k, (k+m) \bmod n}, \quad m = 0, 1, \ldots, n-1
$$

This is the average of entries along the $m$-th "wrapped diagonal" of $A$.

**Optimal Circulant for Toeplitz Matrix (T. Chan's Formula):**

For Toeplitz matrix $A$ with $A_{ij} = a_{-(n-1)}, \ldots, a_0, \ldots, a_{n-1}$ where $A_{ij} = a_{i-j}$, the optimal circulant $C$ has entries:

$$
c_i = \frac{i \cdot a_{-(n-i)} + (n-i) \cdot a_i}{n}, \quad i = -(n-1), \ldots, 0, \ldots, (n-1)
$$

**For Symmetric Toeplitz (Salahub's formulation):**

For symmetric Toeplitz $\Sigma$ with $\Sigma_{ij} = \rho_{|i-j|}$:

$$
c_m = \begin{cases} \rho_0 & \text{for } m = 0 \\ \rho_m + \frac{m}{M}(\rho_{M-m} - \rho_m) & \text{otherwise} \end{cases}
$$

The resulting $\mathbf{C}_\Sigma$ is a **symmetric circulant** (i.e., $c_m = c_{M-m}$), which guarantees all eigenvalues are real.

**Eigenvalues of Optimal Circulant:**

$$
\lambda_k(\mathbf{C}_\Sigma) = \rho_0 + 2 \sum_{m=1}^{M-1} \frac{M-m}{M} \rho_m \cos\frac{2\pi m k}{M}
$$

**Residual Norm (approximation error):**

$$
\min_C \|C - A\|_F^2 = \sum_{i=0}^{n-1} (a_i - a_{-(n-i)})^2 \cdot \frac{i(n-i)}{n} \leq \frac{n}{4} \sum_{i=0}^{n-1} (a_i - a_{-(n-i)})^2
$$

For symmetric Toeplitz ($a_i = a_{-i}$), the residual is **zero** — the optimal circulant perfectly matches a symmetric Toeplitz matrix when it is itself circulant.

**Comparison: Strang vs Chan Preconditioner (n=4 example):**

For symmetric Toeplitz $A$ with entries $(a_0, a_1, a_2, a_3)$:

$$
S = \text{circ}(a_0, a_1, a_2, a_1), \quad C = \text{circ}(a_0, \alpha, a_2, \alpha)
$$

where $\alpha = (3a_1 + a_3)/4$. Strang discards $a_3$; Chan incorporates it via averaging.

**Key Definitions:**

- $\text{circ}(c_0, \ldots, c_{n-1})$ — circulant matrix with first row $(c_0, \ldots, c_{n-1})$
- $\rho_m$ — the $m$-th Toeplitz coefficient ($\rho_{|i-j|}$ for symmetric case)
- $M$ — matrix dimension
- $\omega = e^{2\pi i / M}$ — $M$-th root of unity
- $\|\cdot\|_F$ — Frobenius norm

## Complexity

| Operation | Toeplitz MV (direct) | Circulant MV (via FFT) | Computing $C^*$ |
|-----------|---------------------|----------------------|----------------|
| Matrix-vector product | $O(n^2)$ | $O(n \log n)$ | — |
| Computing optimal circulant | — | — | $O(n)$ |
| Solving $Ax = b$ via CG with $C^*$ preconditioner | $O(n^2 \cdot k)$ | $O(n \log n \cdot k')$ | — |

where $k$ is the number of CG iterations without preconditioning and $k' \ll k$ is the number with preconditioning (spectrum of $C^{*-1}A$ clusters around 1).

**Memory:** $O(n)$ for storing the circulant defining vector.

**Key speedup:** The eigenvalues of $C^{*-1}A$ cluster tightly around 1, meaning preconditioned conjugate gradient converges in very few iterations. For $a_k = t^k$ with $|t| < 1$ and large $n$, only $O(1)$ distinct eigenvalue clusters exist.

## Applicability

- **Toeplitz system solving**: The primary application — preconditioning $Ax = b$ where $A$ is Toeplitz. Widely used in signal processing, time series analysis, and image processing where Toeplitz structure arises naturally
- **Attention matrix approximation**: In Transformers, the attention matrix $\text{softmax}(QK^\top / \sqrt{d})$ can be projected onto the nearest circulant to enable $O(n \log n)$ token mixing — this is exactly the BCCB circulant attention trick, and this formula provides the optimal projection
- **Sequence model initialization**: For Toeplitz-based sequence models (TNN, S4), the optimal circulant approximation provides a principled way to convert Toeplitz coefficients to circulant form, avoiding the $2\times$ padding overhead of standard circulant embedding when approximate computation is acceptable
- **Structured weight compression**: Any dense weight matrix can be projected onto the nearest circulant, providing a $k$-fold compression (from $n^2$ to $n$ parameters) with minimal Frobenius-norm error
- **Covariance matrix approximation**: In statistics, Toeplitz covariance matrices of stationary processes can be approximated by circulant matrices, enabling efficient eigenvalue computation and sampling via FFT

## Limitations

- The circulant approximation is **exact only when $A$ is itself circulant**. For non-symmetric Toeplitz matrices, the residual $\|A - C^*\|_F$ can be significant, especially when the corner entries $a_i$ and $a_{-(n-i)}$ differ substantially
- **Positive definiteness not guaranteed**: Even if $A$ is positive definite, the optimal circulant $C^*$ may not be — eigenvalues of $C^*$ can be negative, requiring post-processing (e.g., clipping negative eigenvalues to a small positive value)
- **Loses non-stationarity**: Circulant structure enforces translation invariance (all rows are cyclic shifts), so content-dependent or position-dependent interactions are lost
- **Approximation degrades for small $M$**: For small matrix sizes, the wrap-around averaging introduces noticeable artifacts. The approximation quality improves as $M \to \infty$ (asymptotic equivalence in weak norm)
- **Not a drop-in replacement for asymmetric structure**: For causal (lower-triangular) Toeplitz matrices arising in autoregressive models, the circulant approximation wraps future information into the past, breaking causality

## Implementation Notes

```python
import torch
import torch.fft as fft

def optimal_circulant_approximation(A):
    """Compute the nearest circulant matrix to A in Frobenius norm.

    The m-th entry of the circulant's defining vector is the average
    of all entries on the m-th wrapped diagonal of A.

    Args:
        A: (n, n) matrix (any structure)

    Returns:
        c: (n,) defining vector of the optimal circulant
    """
    n = A.shape[0]
    c = torch.zeros(n, device=A.device, dtype=A.dtype)

    for m in range(n):
        # Average entries along the m-th wrapped diagonal
        # These are entries (k, (k+m) % n) for k = 0, ..., n-1
        indices = torch.arange(n, device=A.device)
        c[m] = A[indices, (indices + m) % n].mean()

    return c


def optimal_circulant_toeplitz(rho):
    """Compute optimal circulant for symmetric Toeplitz matrix.

    Given Toeplitz coefficients rho = [rho_0, rho_1, ..., rho_{M-1}],
    compute the optimal circulant defining vector using Chan's formula.

    Args:
        rho: (M,) Toeplitz coefficients where Sigma_{ij} = rho_{|i-j|}

    Returns:
        c: (M,) defining vector of the optimal circulant
    """
    M = len(rho)
    m = torch.arange(M, device=rho.device, dtype=rho.dtype)

    # Chan's formula: c_m = rho_m + (m/M) * (rho_{M-m} - rho_m)
    # For m=0: c_0 = rho_0
    rho_mirror = torch.roll(torch.flip(rho, [0]), 1)  # rho_{M-m}
    rho_mirror[0] = rho[0]

    c = rho + (m / M) * (rho_mirror - rho)
    c[0] = rho[0]

    return c


def circulant_matvec(c, x):
    """Multiply circulant matrix (defined by c) with vector x via FFT.

    Args:
        c: (n,) defining vector of the circulant
        x: (..., n) input vector(s)

    Returns:
        y: (..., n) output = C @ x
    """
    c_fft = fft.fft(c)
    x_fft = fft.fft(x, dim=-1)
    return fft.ifft(c_fft * x_fft, dim=-1).real


class CirculantApproxTokenMixer(torch.nn.Module):
    """Token mixer that projects a learned Toeplitz matrix
    onto its optimal circulant approximation.

    Avoids the 2x padding of standard circulant embedding
    at the cost of approximation error.
    """

    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # Learn Toeplitz coefficients directly
        self.rho = torch.nn.Parameter(
            torch.randn(max_seq_len) * 0.01
        )
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        n = x.shape[1]
        rho = self.rho[:n]

        # Compute optimal circulant (no 2x padding needed!)
        c = optimal_circulant_toeplitz(rho)

        # Apply via FFT: O(n log n) per feature dimension
        # vs O(2n log 2n) with standard circulant embedding
        return circulant_matvec(c, x)
```

## References

- Chan, T.F. "An Optimal Circulant Preconditioner for Toeplitz Systems" SIAM J. Sci. Stat. Comput., 9(4):766-771, 1988
- Salahub, C. "Approximating real symmetric Toeplitz matrices using the nearest circulant" arXiv:2208.05771, 2022
- Strang, G. "A Proposal for Toeplitz Matrix Calculations" Studies in Applied Mathematics, 74:171-176, 1986
- Tyrtyshnikov, E.E. "Optimal and Superoptimal Circulant Preconditioners" SIAM J. Matrix Anal. Appl., 13(2):459-473, 1992
- Gray, R.M. "Toeplitz and Circulant Matrices: A Review" Foundations and Trends in Communications and Information Theory, 2006

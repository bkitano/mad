# 037: Displacement Rank for Cauchy-Like Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Kailath, Kung & Morf (1979); Pan (1990, 2017); Bostan, Jeannerod, Mouilleron & Schost (2017); Thomas, Gu, Dao, Rudra & Ré (NeurIPS 2018)
**Paper**: [papers/displacement-rank-cauchy-like.pdf] (Bostan et al. 2017), [papers/ldr-compressed-transforms.pdf] (Thomas, Gu, Dao et al. 2019)
**Documented**: 2026-02-15

## Description

The displacement rank framework provides a unified way to represent and compute with structured matrices — including Toeplitz-like, Hankel-like, Vandermonde-like, and Cauchy-like matrices — through compact "generators" and displacement operators. Instead of storing an $n \times n$ matrix explicitly ($n^2$ entries), a matrix with displacement rank $\alpha$ is fully determined by its two displacement operators and a rank-$\alpha$ residual ($O(\alpha n)$ parameters). This enables fast matrix-vector multiplication in $O(\alpha n \log n)$ time and matrix inversion in $O(\alpha^2 n \log^2 n)$ time, compared to $O(n^2)$ and $O(n^3)$ respectively for dense matrices.

The key insight for neural networks is that **Cauchy-like matrices** — matrices whose displacement operators are both diagonal — arise naturally when SSMs evaluate their generating functions at roots of unity (the Cauchy kernel trick). The displacement rank framework explains *why* these computations are efficient and provides a path to learning even richer classes of structured weight matrices.

## Mathematical Form

**Core Displacement Equation (Sylvester type):**

$$
\nabla_{\mathbf{M},\mathbf{N}}(\mathbf{A}) = \mathbf{M}\mathbf{A} - \mathbf{A}\mathbf{N} = \mathbf{G}\mathbf{H}^T
$$

where $\mathbf{G} \in \mathbb{F}^{m \times \alpha}$ and $\mathbf{H} \in \mathbb{F}^{n \times \alpha}$ are the **generators** of length $\alpha$.

**Stein-type displacement:**

$$
\Delta_{\mathbf{M},\mathbf{N}}(\mathbf{A}) = \mathbf{A} - \mathbf{M}\mathbf{A}\mathbf{N} = \mathbf{G}\mathbf{H}^T
$$

**Key Definitions:**

- $\mathbf{A} \in \mathbb{F}^{m \times n}$ — the structured matrix to represent
- $\mathbf{M} \in \mathbb{F}^{m \times m}$, $\mathbf{N} \in \mathbb{F}^{n \times n}$ — displacement operators (sparse, structured)
- $\alpha = \text{rank}(\nabla_{\mathbf{M},\mathbf{N}}(\mathbf{A}))$ — the displacement rank
- $(\mathbf{G}, \mathbf{H})$ — generator pair; compact representation of $\mathbf{A}$

**Cauchy-Like Specialization:**

When $\mathbf{M} = \mathbb{D}(\mathbf{x}) = \text{diag}(x_1, \ldots, x_m)$ and $\mathbf{N} = \mathbb{D}(\mathbf{y}) = \text{diag}(y_1, \ldots, y_n)$ are both diagonal, the displacement equation yields a **Cauchy-like** matrix:

$$
A_{ij} = \frac{\sum_{k=1}^{\alpha} G_{ik} H_{jk}}{x_i - y_j}
$$

For $\alpha = 1$, this reduces to the classical Cauchy matrix $C_{\mathbf{s},\mathbf{t}} = \left(\frac{1}{s_i - t_j}\right)_{i,j}$.

**Unified Structured Matrix Taxonomy:**

| $\mathbf{M}$, $\mathbf{N}$ choice | Structure |
|---|---|
| Both shift matrices $\mathbb{Z}_{m,\varphi}$, $\mathbb{Z}_{n,\psi}$ | Toeplitz-like / Hankel-like |
| One shift, one diagonal | Vandermonde-like |
| Both diagonal | **Cauchy-like** |

**Recovery via Krylov Matrices (LDR approach):**

Given displacement operators $\mathbf{A}, \mathbf{B}$ and generator vectors $\mathbf{g}_i, \mathbf{h}_i$, the weight matrix is:

$$
\mathbf{W} = \sum_{i=1}^{r} \mathcal{K}(\mathbf{A}, \mathbf{g}_i) \mathcal{K}(\mathbf{B}^T, \mathbf{h}_i)^T
$$

where $\mathcal{K}(\mathbf{A}, \mathbf{v}) = [\mathbf{v}, \mathbf{A}\mathbf{v}, \mathbf{A}^2\mathbf{v}, \ldots, \mathbf{A}^{n-1}\mathbf{v}]$ is the Krylov matrix. This has displacement rank at most $2r$ with respect to $(\mathbf{A}^{-1}, \mathbf{B})$.

## Complexity

| Operation | Dense | With Displacement Rank $\alpha$ |
|-----------|-------|------------|
| Storage | $O(n^2)$ | $O(\alpha n)$ |
| Matrix-vector product (Toeplitz/Hankel) | $O(n^2)$ | $O(\alpha n \log n)$ |
| Matrix-vector product (Cauchy/Vandermonde) | $O(n^2)$ | $O(\alpha n \log n)$ |
| Matrix inversion | $O(n^3)$ | $O(\alpha^2 n \log^2 n)$ |
| Linear system solve | $O(n^3)$ | $O(\alpha^2 n \log^2 n)$ |
| Krylov-based MVM (LDR-SD) | $O(n^2)$ | $O(r \cdot n \log^2 n)$ |

**Memory:** $O(\alpha n)$ vs $O(n^2)$

**Closure Properties (crucial for deep networks):**
- Transpose/Inverse preserves displacement rank $\alpha$
- Sum of two LDR matrices has rank $\leq \alpha_1 + \alpha_2$
- Product of two LDR matrices has rank $\leq \alpha_1 + \alpha_2$
- Block matrices composed of LDR blocks remain LDR

## Applicability

- **S4 / SSM kernel computation**: The Cauchy kernel trick in S4 is a direct instance — evaluating $\hat{K}(\omega) = \sum_i v_i / (\omega - \lambda_i)$ is multiplication by a rank-1 Cauchy matrix, and the DPLR correction via Woodbury adds a low-rank perturbation, keeping displacement rank small.
- **Compressed weight layers**: LDR matrices replace dense $n \times n$ weight matrices in FC, convolutional, and recurrent layers with $O(rn)$ parameters and $O(rn \log^2 n)$ MVM, achieving 20x+ compression with equal or better accuracy (Thomas, Gu, Dao et al., NeurIPS 2018).
- **Learnable equivariance**: The displacement operators $\mathbf{A}, \mathbf{B}$ define transformations to which the weight matrix is *approximately equivariant*. Learning these operators discovers latent structure in data (e.g., 2D spatial structure from flattened images).
- **Monarch matrices and butterfly factorizations**: These can be seen through the displacement rank lens as specific instances of structured matrices with block-diagonal displacement operators.

## Limitations

- Fast algorithms for Cauchy-like MVM require eigenvalues of the displacement operators to be well-separated; clustering causes numerical instability.
- The "superfast" $O(n \log^2 n)$ algorithms rely on polynomial arithmetic (FFT, Chinese remaindering) which may not map efficiently to GPU hardware without careful implementation.
- For small matrices ($n < 1000$), the overhead of structured algorithms can exceed the cost of dense operations on modern GPUs.
- The LDR-TD class (tridiagonal operators) is more expressive but lacks efficient near-linear time algorithms in practice — the current implementation falls back to $O(n^2)$ Krylov expansion.
- Numerical stability of superfast solvers for Cauchy-like systems requires extended precision or careful pivoting strategies.

## Implementation Notes

```python
# LDR matrix-vector multiplication via Krylov matrices (LDR-SD)
# Following Thomas, Gu, Dao et al. (2019)
import torch
import torch.fft

def ldr_sd_multiply(A_diag, B_diag, G, H, x):
    """
    Multiply LDR-SD matrix by vector x.
    A_diag: (n,) subdiagonal of operator A (+ corner element)
    B_diag: (n,) subdiagonal of operator B
    G: (n, r) generator matrix
    H: (n, r) generator matrix
    x: (n,) input vector

    Returns W @ x where W = sum_i K(A, g_i) K(B^T, h_i)^T
    Total cost: O(r * n * log^2(n)) via batched FFTs
    """
    n, r = G.shape
    result = torch.zeros(n, dtype=x.dtype)

    for i in range(r):
        # Compute K(B^T, h_i)^T @ x via FFT-based method
        # This is equivalent to polynomial multiplication
        # when B is a subdiagonal (companion) matrix
        Kbt_h_x = krylov_transpose_multiply(B_diag, H[:, i], x)

        # Compute K(A, g_i) @ (K(B^T, h_i)^T @ x)
        result += krylov_multiply(A_diag, G[:, i], Kbt_h_x)

    return result

def krylov_multiply(subdiag, v, x):
    """Multiply Krylov matrix K(A, v) by x using batched FFTs.
    When A is subdiagonal, K(A,v) has special structure
    enabling O(n log^2 n) multiplication."""
    n = len(v)
    # Use doubling trick:
    # K(A, v, 2n) can be computed from K(A, v, n) via FFT
    # Total: O(log(n)) levels × O(n log n) FFT = O(n log^2 n)
    ...

# Cauchy-like matrix as displacement rank structure
def cauchy_like_matvec(s, t, G, H, x):
    """
    Multiply Cauchy-like matrix by vector.
    A_{ij} = sum_k G[i,k]*H[j,k] / (s[i] - t[j])

    For rank-1 (standard Cauchy), G and H are vectors.
    Uses fast multipole or HSS for O(n log n) complexity.
    """
    # Naive O(alpha * n * m) implementation
    # (practical on GPU via broadcasting)
    alpha = G.shape[1]
    m, n = len(s), len(t)

    # Compute sum over displacement rank components
    result = torch.zeros(m, dtype=x.dtype)
    diffs = s[:, None] - t[None, :]  # (m, n)
    for k in range(alpha):
        numerator = G[:, k:k+1] * H[:, k:k+1].T  # (m, n)
        result += (numerator / diffs) @ x

    return result
```

## References

- Kailath, T., Kung, S., & Morf, M. (1979). Displacement ranks of matrices and linear equations. Journal of Mathematical Analysis and Applications.
- Pan, V. Y. (1990). On computations with dense structured matrices. Mathematics of Computation, 55(191), 179-190.
- Bostan, A., Jeannerod, C.-P., Mouilleron, C., & Schost, É. (2017). On Matrices With Displacement Structure: Generalized Operators and Faster Algorithms. SIAM Journal on Matrix Analysis and Applications, 38(3), 733-775.
- Thomas, A. T., Gu, A., Dao, T., Rudra, A., & Ré, C. (2019). Learning Compressed Transforms with Low Displacement Rank. NeurIPS 2018.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022.
- Pan, V. Y. (2017). Fast Approximate Computations with Cauchy Matrices and Polynomials. arXiv:1506.02285.

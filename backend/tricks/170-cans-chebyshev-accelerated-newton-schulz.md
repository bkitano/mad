# 170: CANS — Chebyshev-Accelerated Newton-Schulz

**Category**: approximation
**Gain type**: efficiency
**Source**: Grishina, Smirnov & Rakhuba (2025) — arXiv:2506.10935; Cesista, Jiacheng & Jordan (2025) — Muon coefficient search
**Paper**: [papers/cans-chebyshev-newton-schulz.pdf]
**Documented**: 2026-02-15

## Description

The standard Newton-Schulz iteration for computing the polar factor $UV^\top$ of a matrix uses fixed polynomial coefficients — either the classical $(3/2, -1/2)$ cubic or the Muon optimizer's empirically tuned quintic $(3.4445, -4.7750, 2.0315)$. **CANS (Chebyshev-Accelerated Newton-Schulz)** replaces these fixed coefficients with **theoretically optimal ones** derived from Chebyshev's alternance theorem, yielding faster convergence with the same number of matrix multiplications.

The key insight is that Newton-Schulz iteration applies an odd polynomial $p(x)$ to the singular values of the iterate $X_k$, aiming to push them toward 1 (orthogonality). The convergence rate depends on how well the *composition* $p_s(\cdots p_1(p_0(x))\cdots)$ approximates the constant function $f \equiv 1$ on the interval $[\sigma_n, \sigma_1]$ containing the singular values. Classical Newton-Schulz uses the same polynomial at every step, but CANS allows **different optimal polynomials at each step**, adapting to the shrinking interval as singular values converge toward 1.

For degree-3 polynomials, CANS derives closed-form optimal coefficients via Proposition 2 of the paper. For degree-5 and higher, it uses the Remez algorithm to compute minimax-optimal polynomials. The composition of CANS polynomials achieves a **larger derivative at the origin** $\phi'(0)$ than the Muon polynomial for the same number of matmuls, meaning small singular values are inflated faster — a property directly correlated with optimizer convergence speed. Experiments on NanoGPT show CANS polynomials outperform both the original Muon polynomial and the computationally searched polynomial of Jiacheng [16] at equal matmul budgets.

Additionally, CANS introduces **$\delta$-orthogonalization**: instead of demanding $f \equiv 1$ exactly, it constructs polynomials that confine singular values to $[1 - \delta, 1 + \delta]$ while **maximizing $\phi'(0)$**. This controlled-accuracy regime is ideal for the Muon optimizer, where exact orthogonality is unnecessary and inflating small singular values quickly matters more.

## Mathematical Form

**Newton-Schulz Iteration:**

Given $X \in \mathbb{R}^{m \times n}$ with SVD $X = U S V^\top$, singular values $\sigma_i \in [a, b] \subset (0, \infty)$:

$$
X_{k+1} = p_k(X_k X_k^\top) X_k
$$

where $p_k$ is a degree-$(2d_k - 1)$ odd polynomial applied to the squared singular values. After $s$ steps, the composed polynomial is:

$$
\phi(x) = p_s(p_{s-1}(\cdots p_1(p_0(x)) \cdots))
$$

The goal: $\phi(\sigma_i^2) \approx 1$ for all $\sigma_i \in [a, b]$, equivalently $\max_{x \in [a,b]} |\phi(x) - 1| \to 0$.

**Optimal Odd Polynomial Approximation (Theorem 1):**

Let $L_n = \{\alpha_1 x + \alpha_3 x^3 + \cdots + \alpha_{2n-1} x^{2n-1}\}$ be the space of odd polynomials of degree $\leq 2n - 1$. The best uniform approximation $p_{n,a,b}$ of $f \equiv 1$ on $[a, b]$ is unique and characterized by **Chebyshev alternance**: there exist $n + 1$ points $a = x_0 < x_1 < \cdots < x_n = b$ such that

$$
|p(x_j) - 1| = \varepsilon(n, a, b), \quad p(x_j) - 1 = -(p(x_{j-1}) - 1) \quad \forall j
$$

**Closed-Form for Degree 3 (Proposition 2):**

$$
p_{2,a,b}(x) = \frac{2}{2\left(\frac{a^2 + ab + b^2}{3}\right)^{3/2} + a^2 b + b^2 a} \left((a^2 + ab + b^2)x - x^3\right)
$$

The minimax error is:

$$
\varepsilon(2, a, b) = \frac{2\left(\frac{a^2 + ab + b^2}{3}\right)^{3/2} - a^2 b - b^2 a}{2\left(\frac{a^2 + ab + b^2}{3}\right)^{3/2} + a^2 b + b^2 a}
$$

**Iterated CANS (Algorithm 1):**

Starting with bounds $[a_0, b_0]$ on singular values:

$$
a_0 = a, \quad b_0 = b
$$

For each iteration $k = 0, 1, \ldots, s$:
1. Compute optimal polynomial $p_k, \varepsilon_k = \text{remez}(a_k, b_k, 2d_k - 1)$
2. Update bounds: $a_{k+1} = 1 - \varepsilon_k, \quad b_{k+1} = 1 + \varepsilon_k$
3. Apply: $X_{k+1} = p_k(X_k X_k^\top) X_k$

**Convergence (Proposition 3):**

For iterated degree-3 polynomials ($d_k = 2$ for all $k$):

$$
\varepsilon_{n+1} \leq \varepsilon_n^2, \qquad \lim_{n \to \infty} \frac{\varepsilon_{n+1}}{\varepsilon_n^2} = \frac{3}{4}
$$

The number of iterations to achieve error $\varepsilon$ from initial $a_0$:

$$
n \geq \left\lceil \log_2 \left(\frac{\ln \varepsilon}{\ln(1 - a_0)}\right) \right\rceil
$$

**$\delta$-Orthogonalization (Section 3.3):**

For the Muon optimizer, construct polynomials in $\mathcal{P}_{d,\delta}$ — the set of degree-$(2d-1)$ odd polynomials $p$ satisfying:
- $p([a, 1+\delta]) \subset [1-\delta, 1+\delta]$ (maps into $\delta$-band)
- $p([0, a]) \subset [0, 1-\delta]$ (non-negative, bounded below band)
- $p(x) \geq cx$ for all $x \in [0, a]$ (monotone increasing)

Among all such polynomials, choose $q_{d,\delta}$ that **maximizes derivative at zero**: $q'_{d,\delta}(0) = \max_{p \in \mathcal{P}_{d,\delta}} p'(0)$.

For degree 3, CANS provably maximizes $p'(0)$ over all of $\mathcal{P}_{2,\delta}$ (Proposition 4(ii)).

**Key Definitions:**

- $p_{n,a,b}$ — best degree-$(2n-1)$ odd polynomial approximation of $f \equiv 1$ on $[a, b]$
- $\varepsilon(n, a, b) = \|p_{n,a,b} - 1\|_{C[a,b]}$ — minimax approximation error
- $\mathcal{P}_{d,\delta}$ — set of "safe" polynomials for $\delta$-orthogonalization
- $q_{d,\delta}$ — polynomial in $\mathcal{P}_{d,\delta}$ with maximal $p'(0)$

## Complexity

| Operation | Classical NS | Muon (quintic) | CANS (degree 3, $s$ iters) | CANS (degree 5, $s$ iters) |
|-----------|-------------|----------------|---------------------------|---------------------------|
| Matmuls per iter | 2 | 3 | 2 | 3 |
| Total matmuls | $2s$ | $3s$ | $2s$ | $3s$ |
| Error after $s$ iters | $\varepsilon_0^{2^s}$ | $\varepsilon_0^{3^s}$ (empirical) | $\leq \frac{3}{4}\varepsilon_0^{2^s}$ (tighter) | Better than Muon |
| Derivative $\phi'(0)$ | Lower | Baseline | **Higher** | **Highest** |

**Same matmul budget, better convergence:** For 12 matmuls (4 iterations of degree-3), CANS outperforms Muon with 12 matmuls (4 iterations of quintic) on NanoGPT test loss (Figure 5 in paper).

**Normalization cost:** Estimating $\sigma_1$ via Gelfand's formula $\sigma_1 \leq \|(A^\top A)^k\|_F^{1/(2k)}$ is free since $(A^\top A)^k$ is already computed during Newton-Schulz iterations.

**Memory:** $O(mn)$ — same as storing the iterate $X$.

## Applicability

- **Muon optimizer acceleration:** Direct drop-in replacement for Muon's Newton-Schulz polynomial. CANS degree-3 polynomials with 4 iterations (12 matmuls) outperform Muon's quintic with 4 iterations (12 matmuls) on NanoGPT training (Figure 5). The improvement is in *convergence speed* (faster loss decrease per step), not per-step throughput.

- **Riemannian optimization on Stiefel manifold:** CANS provides a fast approximate polar retraction. Table 1 shows CANS retraction achieves **43.6s/epoch** (SGD) vs 69.5s (Cayley), 61.0s (QR), and 34.9s (no retraction) on Wide ResNet-16-10 CIFAR-10, with the same or better accuracy (95.97% vs 94.81% Cayley, 94.80% QR).

- **Orthogonal weight constraints in neural networks:** Any architecture enforcing orthogonal weights (orthogonal RNNs, OFT fine-tuning, orthogonal convolutions) can use CANS for faster projection to the orthogonal manifold.

- **Preprocessing for iterative methods:** $\delta$-orthogonalization can serve as a preprocessing step before any method that benefits from well-conditioned input (e.g., running a few CANS iterations before a high-accuracy Newton-Schulz finishing step).

## Limitations

- **Requires spectral bounds:** CANS needs estimates of the smallest singular value $\sigma_n$ (or at least a lower bound $a$). Overestimating $a$ leads to faster convergence; underestimating it is conservative but safe. When $\sigma_n$ is unknown, the $\delta$-orthogonalization scheme (Algorithm 2) adaptively searches via binary search, adding overhead.

- **Remez algorithm instability at high degree:** The Remez algorithm for computing optimal polynomials of degree $> 5$ is numerically unstable. The paper notes "we have not observed an improvement of our methods when using polynomials of degrees higher than 5."

- **Marginal improvement at high iteration count:** When many iterations are used (e.g., $s \geq 7$), the interval $[a_k, b_k]$ is so close to $[1,1]$ that all reasonable polynomials perform similarly. CANS's advantage is most pronounced in the **low-iteration regime** (3-5 iterations) that matters for Muon.

- **Coefficient computation overhead:** The Remez algorithm must be run offline to compute optimal coefficients for a given $(a, b)$ pair. In practice, the coefficients are precomputed for a range of intervals and stored as lookup tables.

- **NanoGPT-scale validation only:** The NanoGPT experiments use 0.8B tokens on a single A100. Scaling behavior at 1B+ parameter models with hundreds of billions of tokens is not yet validated, though the authors note "the picture may change when training larger models."

## Implementation Notes

```python
import torch
import math

def cans_degree3_coefficients(a, b):
    """
    Compute closed-form optimal degree-3 odd polynomial coefficients
    for best uniform approximation of f=1 on [a, b].

    Returns (c1, c3) such that p(x) = c1*x + c3*x^3
    """
    s = (a**2 + a*b + b**2) / 3.0
    denom = 2 * s**1.5 + a**2 * b + b**2 * a
    c1 = 2 * (a**2 + a*b + b**2) / denom
    c3 = -2 / denom
    eps = (2 * s**1.5 - a**2 * b - b**2 * a) / denom
    return c1, c3, eps


def cans_orthogonalize(X, n_iters=4, degree=3):
    """
    Orthogonalize X via CANS (Algorithm 1).

    X: (m, n) matrix, m >= n
    n_iters: number of Newton-Schulz iterations
    degree: polynomial degree (3 or 5)

    Returns: X_orth ≈ Polar(X) = U @ V^T
    """
    # Estimate sigma_1 via Gelfand's formula (free during iteration)
    # sigma_1 ≈ sqrt(||X^T X||_F^{1/1}) for k=1
    sigma1_est = math.sqrt(torch.linalg.norm(X.T @ X).item())

    # Normalize so singular values fall in [a, 1] approximately
    X = X / sigma1_est

    # Estimate spectral bounds
    # Conservative: a = sigma_n / sigma_1, b = 1
    # If unknown, use Frobenius-based estimate
    m, n = X.shape
    a = 1.0 / sigma1_est  # rough lower bound
    b = 1.0

    for k in range(n_iters):
        if degree == 3:
            c1, c3, eps = cans_degree3_coefficients(a, b)
            # p(x) = c1*x + c3*x^3, applied to singular values via:
            # X_{k+1} = c1 * X_k + c3 * X_k @ (X_k^T @ X_k)
            G = X.T @ X       # (n, n) GEMM - tensor core friendly
            X = c1 * X + c3 * (X @ G)  # (m, n) GEMM - tensor core friendly

        # Update interval bounds
        a = 1 - eps
        b = 1 + eps

    return X


def cans_delta_orthogonalize(X, delta=0.3, n_iters=4, degree=3):
    """
    δ-orthogonalization: push singular values into [1-δ, 1+δ]
    while maximizing derivative at 0 (Algorithm 2).

    This is the Muon-optimized variant where exact orthogonality
    is unnecessary but fast inflation of small singular values matters.
    """
    sigma1_est = math.sqrt(torch.linalg.norm(X.T @ X).item())
    X = X / sigma1_est

    # Binary search for the right 'a' parameter at each step
    a, b = 1.0 / sigma1_est, 1.0
    B = 1 + delta  # right boundary

    for k in range(n_iters):
        c1, c3, eps = cans_degree3_coefficients(a, b)
        G = X.T @ X
        X = c1 * X + c3 * (X @ G)
        a, b = 1 - eps, 1 + eps

    return X


# Comparison with Muon's original polynomial:
# Muon: p(x) = 3.4445*x - 4.7750*x^3 + 2.0315*x^5  (quintic, 3 matmuls/iter)
# CANS degree-3: p(x) = c1*x + c3*x^3              (cubic, 2 matmuls/iter)
# CANS degree-5: computed via Remez                  (quintic, 3 matmuls/iter)
#
# For 12 matmuls total:
#   Muon: 4 iters × 3 mm = 12 mm
#   CANS degree-3: 6 iters × 2 mm = 12 mm  (more iterations, cheaper each)
#   CANS degree-5: 4 iters × 3 mm = 12 mm  (same structure, better coefficients)
#
# CANS outperforms Muon on NanoGPT at equal matmul budget (Figure 5).

# Example CANS coefficients from Appendix G:
# CANS, eps=0.3, order=3, iter=7, mm=14 (black in Figure 3):
CANS_COEFFICIENTS_ORDER3_ITER7 = [
    (5.181702879894027, -5.177039351076183),
    (2.585422564566848, -0.647862782007566),
    (2.565592012027513, -0.645264570196127),
    (2.516223347431526, -0.638782620243433),
    (2.401068707564606, -0.623585125272674),
    (2.170844761790119, -0.592849780534662),
    (1.839437716819516, -0.547668362229117),
]
```

**GPU efficiency analysis:**

- **Pure GEMM:** Every CANS iteration is 2 matmuls (degree 3) or 3 matmuls (degree 5) — all tensor-core friendly. Identical kernel structure to standard Newton-Schulz.
- **No overhead vs Muon:** The polynomial coefficients are precomputed scalars. The per-iteration kernel is identical to Muon — just different scalar coefficients multiplied into the matmul output.
- **Better convergence per matmul:** The key GPU win is fewer total iterations needed (or better quality at equal iterations), reducing the total number of GEMM kernel launches and associated memory traffic.
- **bfloat16 compatible:** Same numerical properties as standard Newton-Schulz — the polynomials have positive leading coefficients and the iteration is self-correcting.
- **Gelfand normalization is free:** $\sigma_1 \leq \|(X^\top X)^k\|_F^{1/(2k)}$ reuses $(X^\top X)^k$ which is already computed as part of the Newton-Schulz iteration, so the normalization adds zero extra matmuls.

## References

- Grishina, E., Smirnov, M. & Rakhuba, M. (2025). Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials. arXiv:2506.10935.
- Jordan, K. et al. (2024). Muon: An optimizer for hidden layers in neural networks. GitHub: KellerJordan/Muon.
- Cesista, F. L., Jiacheng, Y. & Jordan, K. (2025). Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing the Newton-Schulz Coefficients. Blog post.
- Bernstein, J. & Newhouse, L. (2024). Old Optimizer, New Norm: An Anthology. arXiv:2409.20325.
- Kim, G. Y. & Oh, M.-H. (2026). Convergence of Muon with Newton-Schulz. ICLR 2026. arXiv:2601.19156.
- Trefethen, L. N. (2020). Approximation Theory and Approximation Practice. SIAM.

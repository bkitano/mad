# 064: Hutchinson Stochastic Trace Estimator

**Category**: approximation
**Gain type**: efficiency
**Source**: Hutchinson (1990); Girard (1987); Meyer et al. (Hutch++, 2021)
**Paper**: [papers/hutchinson-trace-estimator.pdf]
**Documented**: 2026-02-15

## Description

Computing the trace of a $d \times d$ matrix $A$ exactly requires $d$ matrix-vector products (one per standard basis vector $e_i$). The Hutchinson stochastic trace estimator replaces this with a randomized approximation: draw random vectors $g_i$ with i.i.d. entries of mean 0 and variance 1, then $\mathbb{E}[g^T A g] = \text{tr}(A)$. Averaging over $m$ samples gives $\text{tr}(A) \approx \frac{1}{m} \sum_{i=1}^m g_i^T A g_i$, requiring only $m \ll d$ matrix-vector products. This is the foundational "matrix-free" trace estimation trick used throughout deep learning: it enables efficient computation of Jacobian traces in FFJORD/Neural ODEs, log-determinants in invertible networks (via the power series identity $\ln \det(I+A) = \text{tr}(\ln(I+A))$), Hessian traces for second-order optimization, and Laplacian traces in physics-informed neural networks. The Hutch++ improvement (Meyer et al., 2021) reduces the query complexity from $O(1/\varepsilon^2)$ to $O(1/\varepsilon)$ for PSD matrices by first projecting off the top eigenspace, making the residual easier to estimate.

## Mathematical Form

**Core Operation (Hutchinson's Estimator):**

$$
\text{H}_m(A) = \frac{1}{m} \sum_{i=1}^{m} g_i^T A g_i = \frac{1}{m} \text{tr}(G^T A G)
$$

where $G = [g_1, \ldots, g_m] \in \mathbb{R}^{d \times m}$ with i.i.d. entries satisfying $\mathbb{E}[g_{ij}] = 0$, $\text{Var}(g_{ij}) = 1$.

**Key Definitions:**

- $A \in \mathbb{R}^{d \times d}$ — matrix accessed only via matrix-vector products $Ax$
- $g_i \in \mathbb{R}^d$ — random probe vectors (Rademacher $\pm 1$ or Gaussian $\mathcal{N}(0, I)$)
- $m$ — number of probe vectors (controls accuracy)

**Why It Works:**

$$
\mathbb{E}[g^T A g] = \mathbb{E}\left[\sum_{i,j} g_i A_{ij} g_j\right] = \sum_{i,j} A_{ij} \mathbb{E}[g_i g_j] = \sum_{i} A_{ii} = \text{tr}(A)
$$

since $\mathbb{E}[g_i g_j] = \delta_{ij}$ (uncorrelated, unit variance).

**Variance (for Rademacher vectors):**

$$
\text{Var}[\text{H}_1(A)] = 2 \|A\|_F^2 - 2 \sum_i A_{ii}^2 = 2 \sum_{i \neq j} A_{ij}^2
$$

For Gaussian vectors: $\text{Var}[\text{H}_1(A)] = 2\|A\|_F^2$. Rademacher vectors have lower variance when $A$ has large diagonal entries.

**Accuracy Bound (Hutchinson):**

For sub-Gaussian random vectors and PSD $A$, with $m = O(\log(1/\delta)/\varepsilon^2)$ queries:

$$
\mathbb{P}\left[(1 - \varepsilon)\text{tr}(A) \leq \text{H}_m(A) \leq (1 + \varepsilon)\text{tr}(A)\right] \geq 1 - \delta
$$

**Hutch++ (variance-reduced, optimal):**

Split the $m$ query budget into thirds:

1. Sample $S \in \mathbb{R}^{d \times m/3}$ with random $\pm 1$ entries
2. Compute $Q$ = orthonormal basis for column span of $AS$ (via QR)
3. Return:

$$
\text{Hutch++}(A) = \text{tr}(Q^T A Q) + \frac{3}{m} \text{tr}\left(G^T (I - QQ^T) A (I - QQ^T) G\right)
$$

The first term captures the top eigenspace exactly; the second applies Hutchinson to the deflated residual.

**Hutch++ Accuracy Bound:**

With $m = O(\sqrt{\log(1/\delta)/\varepsilon} + \log(1/\delta))$ queries, for PSD $A$:

$$
(1 - \varepsilon)\text{tr}(A) \leq \text{Hutch++}(A) \leq (1 + \varepsilon)\text{tr}(A)
$$

with probability $\geq 1 - \delta$. This is $O(1/\varepsilon)$ vs Hutchinson's $O(1/\varepsilon^2)$ — a quadratic improvement.

## Complexity

| Operation | Exact | Hutchinson | Hutch++ |
|-----------|-------|------------|---------|
| Trace of $d \times d$ matrix | $d$ mat-vec products | $O(1/\varepsilon^2)$ mat-vec products | $O(1/\varepsilon)$ mat-vec products |
| Per query cost (implicit $A$) | $O(d)$ to $O(d^2)$ | same | same + $O(dm^2)$ for QR |
| Jacobian trace (via autodiff) | $O(d^2)$ (full Jacobian) | $O(md)$ ($m$ VJPs) | $O(md)$ + QR overhead |

**Memory:** $O(d)$ per probe vector — no need to store or compute the full matrix $A$

**Typical usage:** $m = 1$ probe vector often suffices for stochastic training (noise is absorbed into SGD), making the per-step cost of a Jacobian trace equal to a single backward pass.

## Applicability

- **FFJORD / Continuous Normalizing Flows**: The instantaneous change of density $\frac{\partial \ln p}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$ requires $\text{tr}(J_f)$ at each ODE step. Hutchinson reduces this from $O(d^2)$ to $O(d)$ using one VJP
- **Invertible Residual Networks / Residual Flows**: Each term $\text{tr}(J_g^k)$ in the power series log-determinant $\ln \det(I + J_g) = \sum (-1)^{k+1}/k \cdot \text{tr}(J_g^k)$ is estimated via Hutchinson
- **Physics-Informed Neural Networks (PINNs)**: Computing $\text{tr}(\nabla^2 u)$ (the Laplacian) for PDEs like Fokker-Planck or Poisson equations, avoiding full Hessian computation
- **Score matching**: The denoising score matching objective involves $\text{tr}(\nabla^2 \log p_\theta)$, estimated via Hutchinson (Sliced Score Matching)
- **Second-order optimization**: Estimating $\text{tr}(H)$ for the Hessian $H$ to compute adaptive learning rates or natural gradient approximations
- **Gaussian processes**: Stochastic estimation of $\text{tr}(K^{-1})$ and log-determinants $\ln \det(K)$ for marginal likelihood computation with large kernel matrices
- **Relation to Woodbury resolvent**: When combined with the power series log-det identity, Hutchinson estimation converts the resolvent-based determinant computation $\det(zI - A)$ into a series of efficient stochastic queries — complementary to the exact Woodbury approach for structured matrices

## Limitations

- Variance scales with $\|A\|_F^2$ — for matrices with large off-diagonal entries, many samples may be needed
- Only provides a $(1 \pm \varepsilon)$ multiplicative guarantee for PSD matrices; for general matrices, the error is additive: $|\text{H}_m(A) - \text{tr}(A)| \leq \varepsilon \|A\|_F$
- Hutch++ requires adaptive queries (later queries depend on earlier results), which may not parallelize as easily as non-adaptive Hutchinson
- For small $d$ (say $d < 100$), exact trace computation is faster than the overhead of random sampling
- The single-sample ($m=1$) estimator used in training has high variance, relying on SGD averaging over minibatches for convergence
- Does not exploit matrix structure (sparsity, low-rank, etc.) — specialized methods may be faster when structure is known (cf. Woodbury for diagonal + low-rank)

## Implementation Notes

```python
import torch

def hutchinson_trace(matvec_fn, d, m=1, estimator='rademacher'):
    """Estimate tr(A) using Hutchinson's stochastic estimator.

    Args:
        matvec_fn: function computing A @ v for a vector v
        d: dimension of the matrix
        m: number of probe vectors
        estimator: 'rademacher' (±1) or 'gaussian'

    Returns:
        Scalar estimate of tr(A)
    """
    trace_est = 0.0
    for _ in range(m):
        if estimator == 'rademacher':
            v = torch.randint(0, 2, (d,)).float() * 2 - 1  # ±1
        else:
            v = torch.randn(d)
        Av = matvec_fn(v)
        trace_est += (v * Av).sum()
    return trace_est / m


def hutchinson_jacobian_trace(f, x, m=1):
    """Estimate tr(J_f(x)) using Hutchinson + reverse-mode autodiff.

    This is the key subroutine for FFJORD / Neural ODE density estimation.

    Args:
        f: function R^d -> R^d (e.g., neural network)
        x: input point, shape (d,)
        m: number of probe vectors

    Returns:
        Scalar estimate of tr(df/dx)
    """
    x = x.requires_grad_(True)
    fx = f(x)
    trace_est = 0.0
    for _ in range(m):
        v = torch.randn_like(x)
        # Vector-Jacobian product: v^T @ J_f
        vjp = torch.autograd.grad(fx, x, v, retain_graph=True)[0]
        trace_est += (vjp * v).sum()
    return trace_est / m


def hutch_plus_plus(matvec_fn, d, m):
    """Hutch++ trace estimator (variance-reduced).

    Splits m queries: m/3 for subspace, m/3 for basis, m/3 for Hutchinson.
    Optimal O(1/eps) convergence for PSD matrices.
    """
    k = m // 3

    # Step 1: Random sketching to find top eigenspace
    S = torch.randn(d, k)
    AS = torch.stack([matvec_fn(S[:, i]) for i in range(k)], dim=1)

    # Step 2: QR factorization for orthonormal basis
    Q, _ = torch.linalg.qr(AS)

    # Step 3: Exact trace of projected component
    AQ = torch.stack([matvec_fn(Q[:, i]) for i in range(k)], dim=1)
    trace_top = (Q * AQ).sum()  # tr(Q^T A Q)

    # Step 4: Hutchinson on deflated residual
    trace_residual = 0.0
    for _ in range(k):
        g = torch.randn(d)
        g_perp = g - Q @ (Q.T @ g)  # Project out top subspace
        Ag_perp = matvec_fn(g_perp) - Q @ (Q.T @ matvec_fn(g_perp))
        trace_residual += (g_perp * Ag_perp).sum()
    trace_residual /= k

    return trace_top + trace_residual
```

## References

- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, 19(2), 433–450.
- Girard, D. A. (1987). Un algorithme simple et rapide pour la validation croisée généralisée sur des problèmes de grande taille. *RR 669-M, IMAG, Grenoble*.
- Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021). Hutch++: Optimal Stochastic Trace Estimation. *SODA*. arXiv:2010.09649.
- Avron, H., & Toledo, S. (2011). Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. *Journal of the ACM*, 58(2), 1–34.
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*. arXiv:1810.01367.
- Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2019). Invertible Residual Networks. *ICML*. arXiv:1811.00995.
- Skilling, J. (1989). The eigenvalues of mega-dimensional matrices. *Maximum Entropy and Bayesian Methods*, 455–466.
- Song, Y., Garg, S., Shi, J., & Ermon, S. (2020). Sliced Score Matching: A Scalable Approach to Density and Score Estimation. *UAI*. arXiv:1905.07088.

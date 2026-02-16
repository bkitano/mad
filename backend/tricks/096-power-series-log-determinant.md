# 096: Power Series Log-Determinant Estimation

**Category**: approximation
**Gain type**: efficiency
**Source**: Behrmann et al. (i-ResNet, 2019); Chen et al. (Residual Flows, 2019)
**Paper**: [papers/invertible-residual-networks.pdf], [papers/residual-flows-log-det.pdf]
**Documented**: 2026-02-15

## Description

Computing $\ln |\det J_F(x)|$ for a general $d \times d$ Jacobian costs $O(d^3)$, making normalizing flows with free-form architectures intractable in high dimensions. The power series log-determinant trick exploits the identity $\ln \det(I + A) = \text{tr}(\ln(I + A))$ and the matrix logarithm's Taylor expansion to express the log-determinant as an infinite series of traces. For contractive residual blocks $F(x) = x + g(x)$ with $\text{Lip}(g) < 1$, the Jacobian $I + J_g$ has all eigenvalues with positive real part, guaranteeing convergence. Each trace term $\text{tr}(J_g^k)$ is estimated via the Hutchinson stochastic trace estimator using a single vector-Jacobian product (computable via reverse-mode autodiff), avoiding any explicit Jacobian materialization. The series can be truncated for a biased estimator, or randomly truncated via a Russian roulette scheme for an unbiased estimator — the key innovation of Residual Flows that enables principled maximum likelihood training of free-form invertible networks.

## Mathematical Form

**Core Identity:**

For $F(x) = x + g(x)$ with $J_g(x) = \frac{\partial g}{\partial x}$ and $\|J_g\|_2 < 1$ (spectral radius less than 1):

$$
\ln |\det J_F(x)| = \ln \det(I + J_g(x)) = \text{tr}\left(\ln(I + J_g(x))\right) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \text{tr}(J_g^k)
$$

**Key Definitions:**

- $F: \mathbb{R}^d \to \mathbb{R}^d$ — invertible residual transformation $F(x) = x + g(x)$
- $J_g(x) = \frac{\partial g(x)}{\partial x} \in \mathbb{R}^{d \times d}$ — Jacobian of the residual block
- $\text{Lip}(g) < 1$ — Lipschitz constant ensuring contractivity and convergence
- The series converges because the eigenvalues of $J_g$ have magnitude less than 1

**Hutchinson Trace Estimation (for each term):**

$$
\text{tr}(J_g^k) = \mathbb{E}_{v \sim \mathcal{N}(0, I)}[v^T J_g^k v]
$$

where $v^T J_g^k v$ is computed iteratively: $w^T := v^T$, then $w^T := w^T J_g$ repeated $k$ times (each step is one vector-Jacobian product via autodiff), yielding $v^T J_g^k v = w^T v$.

**Biased Estimator (i-ResNet, fixed truncation at $n$ terms):**

$$
PS(J_g, n) := \sum_{k=1}^{n} (-1)^{k+1} \frac{\text{tr}(J_g^k)}{k}
$$

**Unbiased Estimator (Residual Flows, Russian roulette):**

$$
\ln \det(I + J_g) = \mathbb{E}_{n, v}\left[\sum_{k=1}^{n} \frac{(-1)^{k+1}}{k} \frac{v^T J_g^k v}{\mathbb{P}(N \geq k)}\right]
$$

where $n \sim p(N)$ is a random truncation point and $\mathbb{P}(N \geq k)$ is the survival function of $p(N)$. The reweighting by $1/\mathbb{P}(N \geq k)$ ensures unbiasedness.

**Neumann Series Gradient (memory-efficient):**

$$
\frac{\partial}{\partial \theta} \ln \det(I + J_g(x, \theta)) = \mathbb{E}_{n, v}\left[\left(\sum_{k=0}^{n} \frac{(-1)^k}{\mathbb{P}(N \geq k)} v^T J(x, \theta)^k\right) \frac{\partial(J_g(x, \theta))}{\partial \theta} v\right]
$$

This avoids backpropagating through the power series, reducing memory from $O(n \cdot m)$ to $O(m)$ where $m$ is the number of residual blocks.

**Truncation Error Bound:**

$$
|PS(J_g, n) - \ln \det(I + J_g)| \leq -d\left(\ln(1 - \text{Lip}(g)) + \sum_{k=1}^{n} \frac{\text{Lip}(g)^k}{k}\right)
$$

## Complexity

| Operation | Naive | With Trick |
|-----------|-------|------------|
| Log-determinant of $d \times d$ Jacobian | $O(d^3)$ | $O(nd)$ |
| Per term $\text{tr}(J_g^k)$ | $O(d^2)$ (explicit) | $O(d)$ (one VJP) |
| Gradient of log-det (naive backprop) | $O(nd)$ compute, $O(nd)$ memory | $O(nd)$ compute, $O(d)$ memory |
| Full normalizing flow ($m$ blocks, $n$ terms) | $O(md^3)$ | $O(mnd)$ |

**Memory (Neumann gradient):** $O(d)$ per block vs $O(nd)$ for naive backprop through the series

**Expected cost with Russian roulette:** In practice, $p(N) = \text{Geom}(0.5)$ with 2 deterministic terms gives expected $n \approx 4$ terms per sample, less than the 5--10 terms needed for the biased estimator.

## Applicability

- **Invertible Residual Networks (i-ResNets)**: The enabling trick that allows standard ResNets (with spectral normalization) to be used as normalizing flows for density estimation
- **Residual Flows**: State-of-the-art flow-based generative models using unbiased log-density estimation with free-form Jacobians
- **FFJORD and Continuous Normalizing Flows**: The Hutchinson trace estimator is used for $\text{tr}(J_g)$ (the $k=1$ case) in the instantaneous change of variables formula $\frac{\partial \ln p}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$
- **Neural ODEs**: Any ODE-based generative model where density evaluation requires Jacobian traces
- **Connections to Woodbury resolvent**: For DPLR state matrices $A = \Lambda - PQ^*$, the resolvent $(zI - A)^{-1}$ computed via Woodbury gives exact log-determinants via the matrix determinant lemma. The power series approach provides an alternative when the matrix lacks explicit low-rank structure but is contractive

## Limitations

- Requires $\text{Lip}(g) < 1$ strictly — the spectral norm of each weight matrix must be controlled (via spectral normalization), which adds overhead and may limit expressivity
- Convergence rate is $O(\text{Lip}(g)^n)$ — if $\text{Lip}(g)$ is close to 1, many terms are needed for accuracy. In practice, $\text{Lip}(g) \leq 0.98$ with 5--10 terms suffices
- The Russian roulette estimator has variable compute cost per sample, which can cause GPU utilization issues with batched training
- Stochastic trace estimation adds variance — typically 1 random vector per term suffices, but high-precision applications may need more
- Not applicable to non-residual architectures (coupling layers, autoregressive flows) which have analytically tractable determinants via structure
- The Neumann gradient series requires the Jacobian to be non-symmetric, unlike the original power series which works for any contractive $J_g$

## Implementation Notes

```python
import torch

def power_series_log_det_unbiased(g, x, n_exact=2, p_geom=0.5):
    """Unbiased log-det estimator via Russian roulette + Hutchinson.

    Args:
        g: residual function, g(x) maps R^d -> R^d
        x: input point, shape (d,)
        n_exact: number of terms computed exactly (no reweighting)
        p_geom: parameter of geometric distribution for random truncation

    Returns:
        Unbiased estimate of ln det(I + J_g(x))
    """
    x = x.requires_grad_(True)
    gx = g(x)
    d = x.shape[-1]

    # Sample random probe vector for Hutchinson estimator
    v = torch.randn_like(x)

    # Sample truncation point: n_exact deterministic + geometric tail
    n_extra = torch.distributions.Geometric(p_geom).sample().int().item()
    n_total = n_exact + n_extra

    log_det_est = 0.0
    w = v.clone()  # w will accumulate v^T J_g^k

    for k in range(1, n_total + 1):
        # Compute w := w^T @ J_g via vector-Jacobian product
        w = torch.autograd.grad(gx, x, w, retain_graph=True)[0]

        # Contribution: (-1)^{k+1} / k * (v^T J_g^k v)
        trace_est_k = (w * v).sum()
        sign = (-1) ** (k + 1)

        # Reweight: exact terms get weight 1, random terms get 1/P(N >= k)
        if k <= n_exact:
            weight = 1.0
        else:
            # P(N >= k) for shifted geometric
            weight = 1.0 / ((1 - p_geom) ** (k - n_exact))

        log_det_est += sign * trace_est_k * weight / k

    return log_det_est


def power_series_log_det_biased(g, x, n_terms=10):
    """Biased (truncated) log-det estimator with Hutchinson trace.

    Simpler version from i-ResNet. Bias bounded by O(Lip(g)^n).
    """
    x = x.requires_grad_(True)
    gx = g(x)

    v = torch.randn_like(x)
    log_det = 0.0
    w = v.clone()

    for k in range(1, n_terms + 1):
        w = torch.autograd.grad(gx, x, w, retain_graph=True)[0]
        trace_k = (w * v).sum()
        log_det += (-1) ** (k + 1) * trace_k / k

    return log_det
```

## References

- Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2019). Invertible Residual Networks. *ICML*. arXiv:1811.00995.
- Chen, R. T. Q., Behrmann, J., Duvenaud, D., & Jacobsen, J.-H. (2019). Residual Flows for Invertible Generative Modeling. *NeurIPS*. arXiv:1906.02735.
- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, 19(2), 433–450.
- Skilling, J. (1989). The eigenvalues of mega-dimensional matrices. *Maximum Entropy and Bayesian Methods*, 455–466.
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*. arXiv:1810.01367.
- Kahn, H. (1955). Use of different Monte Carlo sampling techniques. *RAND Corporation*.
- Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021). Hutch++: Optimal Stochastic Trace Estimation. *SODA*. arXiv:2010.09649.

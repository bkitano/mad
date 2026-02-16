# 165: Neumann Log-Determinant with Russian Roulette

**Category**: approximation
**Gain type**: efficiency
**Source**: Chen, Behrmann, Duvenaud & Jacobsen (NeurIPS 2019) — Residual Flows; Behrmann et al. (ICML 2019) — Invertible Residual Networks
**Paper**: [papers/residual-flows-neumann-logdet.pdf]
**Documented**: 2026-02-15

## Description

Computing $\log \det(I + J_g)$ for the Jacobian $J_g$ of a residual block $f(x) = x + g(x)$ is the central bottleneck in training invertible residual networks (i-ResNets) and Residual Flows. The naive approach requires $O(d^3)$ via LU decomposition. By combining three tricks — the **Neumann-derived power series** for $\log(I + A)$, **Hutchinson's stochastic trace estimator**, and a **Russian roulette unbiased truncation** — the cost is reduced to $O(Kd)$ per training step where $K \sim 4$ is the expected number of series terms, while maintaining an **unbiased** gradient estimator.

The mathematical foundation is the identity $\log \det(I + A) = \text{tr}(\log(I + A)) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \text{tr}(A^k)$, which converges when $\|A\|_{\text{op}} < 1$ (i.e., the Lipschitz constant of $g$ is less than 1). Each trace term $\text{tr}(A^k)$ is estimated stochastically as $v^\top A^k v$ where $v \sim \mathcal{N}(0, I)$ (Hutchinson estimator), and each power $A^k v$ is computed via $k$ sequential vector-Jacobian products (VJPs) without materializing the Jacobian. The Russian roulette estimator makes the truncation unbiased: instead of truncating at a fixed $n$ (which introduces bias), the truncation point $N$ is drawn randomly and each term is reweighted by $1/\mathbb{P}(N \geq k)$, yielding $\mathbb{E}[\hat{L}] = \log \det(I + J_g)$ exactly.

A separate Neumann-series-based gradient estimator (Theorem 2 in the paper) avoids differentiating through the forward power series, reducing memory from $O(n \cdot m)$ (where $n$ = series terms, $m$ = residual blocks) to $O(m)$ — constant regardless of the number of series terms.

## Mathematical Form

**Core Identity (Neumann-derived log-determinant):**

For $f(x) = x + g(x)$ with $\text{Lip}(g) < 1$ and Jacobian $J_g(x) = \frac{dg(x)}{dx}$:

$$
\log \det(I + J_g) = \text{tr}\!\left(\log(I + J_g)\right) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \text{tr}(J_g^k)
$$

This follows from the matrix identity $\log(I + A) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} A^k$ for $\|A\|_{\text{op}} < 1$.

**Hutchinson Stochastic Trace Estimator (Trick 064):**

$$
\text{tr}(J_g^k) = \mathbb{E}_{v \sim \mathcal{N}(0,I)}\!\left[v^\top J_g^k v\right]
$$

The product $v^\top J_g^k$ is computed iteratively via $k$ vector-Jacobian products (VJPs), each costing $O(d)$:

$$
w_0 = v, \quad w_{i+1} = w_i^\top J_g \quad (\text{one VJP}), \quad v^\top J_g^k v = w_k^\top v
$$

**Biased Truncated Estimator (Behrmann et al. 2019):**

$$
\log p(x) \approx \log p(f(x)) + \sum_{k=1}^{n} \frac{(-1)^{k+1}}{k} v^\top J_g^k v
$$

Bias: $O(\text{Lip}(g)^{n+1})$, which grows with dimension and Lipschitz constant.

**Unbiased Russian Roulette Estimator (Theorem 1, Chen et al. 2019):**

Let $N$ be a random variable with $p(N) = \text{Geom}(1-q)$ (support on all positive integers). Then:

$$
\log p(x) = \log p(f(x)) + \mathbb{E}_{N, v}\!\left[\sum_{k=1}^{N} \frac{(-1)^{k+1}}{k} \cdot \frac{v^\top J_g^k v}{\mathbb{P}(N \geq k)}\right]
$$

This is **exactly unbiased**: $\mathbb{E}[\hat{L}] = \log \det(I + J_g)$, regardless of the choice of $p(N)$, as long as $p(N)$ has support on all positive integers.

In practice: compute 2 terms exactly, then draw one sample from $\text{Geom}(0.5)$ for the remaining terms. Expected cost: ~4 VJPs per sample.

**Memory-Efficient Gradient via Neumann Series (Theorem 2):**

The naive gradient requires backpropagating through each power series term, costing $O(n)$ memory. Instead, express the gradient directly as a power series:

$$
\frac{\partial}{\partial \theta} \log \det(I + J_g) = \mathbb{E}_{N, v}\!\left[\left(\sum_{k=0}^{N} \frac{(-1)^k}{\mathbb{P}(N \geq k)} \, v^\top J(x,\theta)^k\right) \frac{\partial J_g(x,\theta)}{\partial \theta} v\right]
$$

This computes the gradient in a single backward pass with $O(1)$ memory overhead (independent of $n$), because the power series for the gradient does not require storing intermediate activations.

**Backward-in-Forward (Eq. 9):**

Since $\log \det(I + J_g)$ is a scalar, compute its gradient during the forward pass:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = \underbrace{\frac{\partial \mathcal{L}}{\partial \log \det(I + J_g)}}_{\text{scalar}} \cdot \underbrace{\frac{\partial \log \det(I + J_g)}{\partial \theta}}_{\text{vector, computed in forward}}
$$

This reduces memory by a factor of $m$ (number of residual blocks).

**Key Definitions:**

- $g: \mathbb{R}^d \to \mathbb{R}^d$ — residual function with $\text{Lip}(g) < 1$
- $J_g(x) = \frac{dg(x)}{dx} \in \mathbb{R}^{d \times d}$ — Jacobian of $g$ (never materialized)
- $v \sim \mathcal{N}(0, I_d)$ — Hutchinson probe vector
- $N \sim p(N)$ — Russian roulette truncation variable
- $\text{Lip}(g)$ — Lipschitz constant of $g$ (enforced via spectral normalization)

## Complexity

| Operation | Exact LU | Biased Truncation ($n$ terms) | Russian Roulette ($K$ expected terms) |
|-----------|----------|-------------------------------|---------------------------------------|
| Forward log-det | $O(d^3)$ | $O(nd)$ per sample | $O(Kd)$ per sample, $K \approx 4$ |
| Memory (forward) | $O(d^2)$ | $O(nd)$ | $O(d)$ |
| Memory (gradient, naive) | $O(d^2)$ | $O(n \cdot m \cdot d)$ | $O(n \cdot m \cdot d)$ |
| Memory (Neumann gradient) | — | — | $O(m \cdot d)$ ✓ |
| Bias | None | $O(\text{Lip}(g)^{n+1})$ | **None** (exactly unbiased) ✓ |

where $d$ = dimension, $m$ = number of residual blocks, $K$ = expected series terms.

**Memory savings from Neumann gradient series (Figure 3 in paper):**

| Dataset | Naive Backprop | + Neumann Series | + Backward-in-Forward | Both Combined |
|---------|---------------|-----------------|----------------------|---------------|
| MNIST | 192.1 GB | 31.2 GB | 19.8 GB | 13.6 GB |
| CIFAR10-small | 11.3 GB | 7.4 GB | 5.9 GB | — |
| CIFAR10-large | 263.5 GB | 40.8 GB | 26.1 GB | 18.0 GB |

(Per minibatch of 64 samples, $n=10$ power series terms)

## Applicability

- **Residual Flows for density estimation:** The enabling trick for training free-form invertible residual networks as generative models. Achieves state-of-the-art density estimation: 0.97 bits/dim on MNIST, 3.28 bits/dim on CIFAR-10, outperforming Glow, FFJORD, and i-ResNets.

- **Invertible ResNets (i-ResNets):** Any ResNet constrained to have Lipschitz constant $< 1$ (via spectral normalization) can be used as a generative model with this log-det estimator. Enables joint generative and discriminative training (hybrid modeling).

- **Continuous normalizing flows (FFJORD):** The continuous-time analog uses Hutchinson's estimator for $\text{tr}(\partial f / \partial z)$ directly. The Russian roulette trick can be applied to reduce bias in the time-discretized ODE solver.

- **Normalizing flows with residual blocks:** Any flow model using $f(x) = x + g(x)$ architecture benefits — avoids the structured Jacobian constraints needed by coupling layers (NICE, RealNVP, Glow).

- **Potential for SSM log-det computation:** When SSM transition matrices are near-identity ($A = I + \epsilon B$ with $\|\epsilon B\|_{\text{op}} < 1$), the log-determinant of the state transition can be estimated with this technique for variational inference or normalizing flow layers built from SSM blocks.

## Limitations

- **Requires Lipschitz constraint $\text{Lip}(g) < 1$:** The power series only converges when the Jacobian spectral radius is $< 1$. This must be enforced architecturally via spectral normalization, which adds overhead and constrains expressivity.

- **Variance of the estimator:** The Russian roulette estimator has higher variance than fixed truncation. The variance depends on the distribution $p(N)$ and the Lipschitz constant — for $\text{Lip}(g)$ close to 1, the variance can be large.

- **Sequential VJPs:** Each term $v^\top J_g^k v$ requires $k$ sequential VJPs, which are backward passes through the network $g$. These are inherently sequential and memory-bandwidth-bound — not compute-bound like matmuls.

- **Not tensor-core friendly:** The VJP computation is general autograd backpropagation, not structured as GEMM. This limits GPU utilization compared to matmul-heavy architectures.

- **Training overhead:** Despite $O(Kd)$ per sample, the constant factor is large: each VJP is a full backward pass through $g$, and multiple probe vectors may be needed for variance reduction. In practice, Residual Flows train slower per epoch than coupling-based flows, but achieve better density estimation.

- **Not applicable when $\text{Lip}(g) \geq 1$:** Standard ResNets have unbounded Lipschitz constant. The technique only applies to the constrained i-ResNet setting.

## Implementation Notes

```python
import torch
import torch.autograd as autograd

def log_det_russian_roulette(g_func, x, n_exact=2, p_geom=0.5, n_samples=1):
    """
    Unbiased estimate of log det(I + J_g(x)) using Russian roulette.

    g_func: callable, the residual function g such that f(x) = x + g(x)
    x: (batch, d) input
    n_exact: number of terms computed exactly (no reweighting)
    p_geom: parameter of geometric distribution for remaining terms
    n_samples: number of probe vectors for Hutchinson estimator

    Returns: (batch,) unbiased estimate of log det(I + J_g(x))
    """
    d = x.shape[-1]
    log_det = torch.zeros(x.shape[0], device=x.device)

    for _ in range(n_samples):
        # Hutchinson probe vector
        v = torch.randn_like(x)  # (batch, d)

        # Compute g(x) with gradient tracking
        x.requires_grad_(True)
        gx = g_func(x)

        # Iteratively compute v^T J_g^k v via VJPs
        w = v  # w_0 = v
        for k in range(1, n_exact + 1):
            # w = w^T J_g  (one VJP, cost = one backward pass through g)
            w = autograd.grad(gx, x, grad_outputs=w,
                              create_graph=True, retain_graph=True)[0]
            # Accumulate: (-1)^{k+1}/k * w^T v
            log_det += ((-1) ** (k + 1)) / k * (w * v).sum(dim=-1)

        # Russian roulette for remaining terms
        # Draw N ~ Geom(1 - p_geom) + n_exact
        n_extra = torch.distributions.Geometric(1 - p_geom).sample().int().item()

        for k in range(n_exact + 1, n_exact + n_extra + 1):
            w = autograd.grad(gx, x, grad_outputs=w,
                              create_graph=True, retain_graph=True)[0]
            # Reweight by 1/P(N >= k) where P(N >= k) = p_geom^(k - n_exact)
            weight = 1.0 / (p_geom ** (k - n_exact))
            log_det += ((-1) ** (k + 1)) / k * weight * (w * v).sum(dim=-1)

    return log_det / n_samples


def neumann_gradient_series(g_func, x, theta, n_terms=5, p_geom=0.5):
    """
    Memory-efficient gradient of log det(I + J_g) via Neumann series.

    Instead of backpropagating through the power series (O(n*m) memory),
    compute the gradient directly as a separate power series (O(m) memory).

    This is Theorem 2 from the Residual Flows paper.
    """
    v = torch.randn_like(x)

    # Compute: sum_{k=0}^{N} (-1)^k / P(N>=k) * v^T J^k
    # Then multiply by dJ/dtheta * v
    gx = g_func(x)

    # Build the "left vector": sum of (-1)^k * v^T J^k
    left = v.clone()  # k=0 term
    w = v
    for k in range(1, n_terms):
        w = autograd.grad(gx, x, grad_outputs=w,
                          retain_graph=True, create_graph=False)[0]
        left = left + ((-1) ** k) * w

    # The gradient is: left^T @ (dJ/dtheta @ v)
    # This is computed as: autograd.grad(gx, theta, grad_outputs=left)
    # Memory: O(d) regardless of n_terms
    grad = autograd.grad(gx, theta, grad_outputs=left)
    return grad


# Practical settings from the paper:
# - n_exact = 2 (compute first 2 terms exactly)
# - p_geom = 0.5 (geometric distribution parameter)
# - Expected total terms: 2 + E[Geom(0.5)] = 2 + 2 = 4
# - n_samples = 1 (single probe vector suffices)
# - Lipschitz constant enforced at 0.98 via spectral normalization
```

**GPU efficiency analysis:**

- **Memory is the key win:** The Neumann gradient series + backward-in-forward reduces memory from 263 GB to 18 GB for CIFAR10-large (14.6x reduction), enabling training of much larger models on a single GPU.
- **Compute is VJP-dominated:** Each VJP is a full backward pass through the residual network $g$, which is memory-bandwidth-bound (not compute-bound). This means the technique doesn't benefit from tensor cores but does benefit from memory-efficient backward passes.
- **Amortization:** The expected 4 VJPs per sample are amortized across the minibatch and across residual blocks. The main overhead is $O(K \cdot m)$ backward passes per training step.
- **Wall-clock:** Residual Flows train on a single GPU with batch size 64, converging in 300-350 epochs — competitive with Glow which requires 40 GPUs.

## References

- Chen, R. T. Q., Behrmann, J., Duvenaud, D. & Jacobsen, J.-H. (2019). Residual Flows for Invertible Generative Modeling. NeurIPS 2019. arXiv:1906.02735.
- Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D. & Jacobsen, J.-H. (2019). Invertible Residual Networks. ICML 2019. arXiv:1811.00995.
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I. & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR 2019. arXiv:1810.01367.
- Kahn, H. (1955). Use of Different Monte Carlo Sampling Techniques. RAND Corporation.
- Hutchinson, M. F. (1990). A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines. Comm. Stat.—Simulation and Computation, 19(2), 433–450.
- Skilling, J. (1989). The Eigenvalues of Mega-dimensional Matrices. Maximum Entropy and Bayesian Methods, 455–466.

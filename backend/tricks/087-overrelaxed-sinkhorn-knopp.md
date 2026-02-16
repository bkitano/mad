# 087: Overrelaxed Sinkhorn-Knopp Algorithm

**Category**: approximation
**Gain type**: efficiency
**Source**: Thibault, Chizat, Dossal & Papadakis (Inria, CNRS, 2021); Lehmann, von Renesse, Sambale & Uschmajew (2020)
**Paper**: [papers/overrelaxed-sinkhorn.pdf]
**Documented**: 2026-02-15

## Description

The Sinkhorn-Knopp (SK) algorithm converges linearly to the optimal transport plan, but the convergence rate degrades severely as the entropic regularization parameter $\varepsilon \to 0$ (equivalently, as temperature $\tau \to 0$ in permutation learning). In the blockwise Sinkhorn channel permutation setting, low temperature is needed to produce sharp, near-permutation doubly stochastic matrices — precisely the regime where standard Sinkhorn is slowest.

The **overrelaxed Sinkhorn-Knopp** algorithm replaces the standard alternating Bregman projections with **overrelaxed projections** using a parameter $\omega \in [1, 2)$. Instead of fully projecting onto each marginal constraint, the algorithm overshoots — taking a weighted geometric mean between the current iterate and the standard projection. This is the OT analog of the classical **Successive Over-Relaxation (SOR)** method for linear systems.

The key result is that with the **optimal overrelaxation parameter** $\theta^* = 2/(1 + \sqrt{\eta})$, the local linear convergence rate improves from $1 - \eta$ (standard SK) to $(1 - \sqrt{\eta})/(1 + \sqrt{\eta})$, where $1 - \eta$ is the second-largest eigenvalue of the linearized SK iteration matrix. This is exactly the SOR speedup formula from numerical linear algebra, yielding up to **$20\times$** faster convergence in the challenging low-$\varepsilon$ regime.

Crucially, the algorithm maintains all the desirable properties of standard Sinkhorn: it is **first-order** (only matrix-vector products), **parallelizable** (same row/column scaling structure), and **simple to implement** (a 3-line modification to log-domain Sinkhorn). The overrelaxation parameter $\omega$ is chosen adaptively at each step using a Lyapunov function descent criterion, guaranteeing **global convergence** — unlike Anderson acceleration or RNA methods which can diverge.

For blockwise Sinkhorn channel permutation with $B = 64$ blocks at temperature $\tau = 0.1$, standard Sinkhorn may require 15-20 iterations while overrelaxed Sinkhorn achieves the same precision in 3-5 iterations at identical per-iteration cost.

## Mathematical Form

**Standard Sinkhorn as Dual Coordinate Ascent:**

The dual of the entropy-regularized OT problem is:

$$
\max_{\alpha \in \mathbb{R}^{n_1}, \beta \in \mathbb{R}^{n_2}} E(\alpha, \beta) := \langle \alpha, \mu_1 \rangle + \langle \beta, \mu_2 \rangle - \varepsilon \sum_{i,j} e^{(\alpha_i + \beta_j - c_{i,j})/\varepsilon}
$$

Standard Sinkhorn performs alternating maximization:

$$
\alpha^{\ell+1} = \arg\max_\alpha E(\alpha, \beta^\ell)
$$

$$
\beta^{\ell+1} = \arg\max_\beta E(\alpha^{\ell+1}, \beta)
$$

with explicit solutions:

$$
\alpha_i^{\ell+1} = \varepsilon \log\left(\sum_j \exp\left(\log(\mu_1)_i - (\beta_j^\ell - c_{i,j})/\varepsilon\right)\right)
$$

$$
\beta_j^{\ell+1} = \varepsilon \log\left(\sum_i \exp\left(\log(\mu_2)_j - (\alpha_i^{\ell+1} - c_{i,j})/\varepsilon\right)\right)
$$

**Overrelaxed Bregman Projection:**

For $\omega \geq 0$, the $\omega$-relaxed projection operator $P_{\mathcal{C}_k}^\omega$ is defined in log-domain as:

$$
\log P_{\mathcal{C}_k}^\omega(\gamma) = (1 - \omega) \log \gamma + \omega \log P_{\mathcal{C}_k}(\gamma)
$$

where $P_{\mathcal{C}_k}$ is the standard Bregman projection onto marginal constraint $\mathcal{C}_k$. Note:
- $\omega = 0$: identity (no update)
- $\omega = 1$: standard Sinkhorn
- $\omega \in (1, 2)$: overrelaxation (overshooting past the projection)

**SOR-Form Update (dual variables):**

$$
\alpha^{\ell+1} = (1 - \omega)\alpha^\ell + \omega \arg\max_\alpha E(\alpha, \beta^\ell)
$$

$$
\beta^{\ell+1} = (1 - \omega)\beta^\ell + \omega \arg\max_\beta E(\alpha^{\ell+1}, \beta)
$$

Equivalently, in the scaling variable form ($u_i = e^{\alpha_i/\varepsilon}$, $v_j = e^{\beta_j/\varepsilon}$):

$$
\tilde{u} = \mu_1 \oslash (\gamma^0 v), \quad u = u^{1-\omega} \otimes \tilde{u}^\omega
$$

$$
\tilde{v} = \mu_2 \oslash ({{}^t\gamma^0} u), \quad v = v^{1-\omega} \otimes \tilde{v}^\omega
$$

where $\gamma^0 = e^{-c/\varepsilon}$, $\oslash$ is pointwise division, and $\otimes$ is pointwise multiplication.

**Lyapunov Function for Global Convergence:**

$$
F(\gamma) = \text{KL}(\gamma^*, \gamma)
$$

The decrease per step is:

$$
F(\gamma) - F(P_{\mathcal{C}_k}^\omega(\gamma)) = \langle \mu_k, \varphi_\omega((A_k \gamma) \oslash \mu_k) \rangle
$$

where $\varphi_\omega(x) = x(1 - x^{-\omega}) - \omega \log x$.

**Adaptive $\omega$ Selection:**

Define:

$$
\Theta^*(w) = \sup \{\omega \in [1, 2] \mid \varphi_\omega(\min w) \geq 0\}
$$

$$
\Theta(w) = \min(\max(1, \Theta^*(w) - \delta), \theta_0)
$$

where $\delta > 0$ is a safety margin and $\theta_0 \in [1, 2)$ is a target upper bound. The overrelaxation parameter at each half-step is:

$$
\omega_k = \Theta((A_k \gamma) \oslash \mu_k)
$$

This ensures $F$ decreases at each step, guaranteeing convergence.

**Optimal Overrelaxation Parameter (local analysis):**

Let $1 - \eta$ be the second-largest eigenvalue of the linearized SK iteration matrix:

$$
M_1 = \text{diag}(\mathbf{1} \oslash \mu_1) \gamma^* \, \text{diag}(\mathbf{1} \oslash \mu_2)^\dagger \gamma^*
$$

The optimal overrelaxation parameter is:

$$
\theta^* = \frac{2}{1 + \sqrt{\eta}}
$$

The overrelaxed algorithm converges locally at rate:

$$
f(\theta^*, \eta) = \frac{1 - \sqrt{\eta}}{1 + \sqrt{\eta}}
$$

compared to the standard SK rate of $1 - \eta$. This is a quadratic improvement: e.g., if $\eta = 0.01$, the SK rate is $0.99$ while the overrelaxed rate is $0.818$ — requiring roughly $\frac{\log(0.99)}{\log(0.818)} \approx 20\times$ fewer iterations.

## Complexity

| Operation | Standard Sinkhorn | Overrelaxed Sinkhorn |
|-----------|-------------------|----------------------|
| Per iteration | $O(n^2)$ | $O(n^2)$ (identical) |
| Convergence rate (local) | $1 - \eta$ | $(1 - \sqrt{\eta})/(1 + \sqrt{\eta})$ |
| Iterations to precision $\delta$ | $O\left(\frac{\log(1/\delta)}{-\log(1 - \eta)}\right) \approx O\left(\frac{\log(1/\delta)}{\eta}\right)$ | $O\left(\frac{\log(1/\delta)}{-\log\frac{1-\sqrt{\eta}}{1+\sqrt{\eta}}}\right) \approx O\left(\frac{\log(1/\delta)}{\sqrt{\eta}}\right)$ |
| Speedup ratio | 1$\times$ | $\approx 1/\sqrt{\eta}$ $\times$ |

**Practical speedups (from paper, Figure 7):**

| Setting | Regularization $\varepsilon$ | Speedup (iterations) |
|---------|------------------------------|----------------------|
| Quadratic cost, random marginals | $10^{-4}$ | $> 20\times$ |
| Quadratic cost, random marginals | $10^{-2}$ | $\sim 5\times$ |
| Random cost, uniform marginals | $5 \times 10^{-3}$ | $\sim 20\times$ |
| Random cost, uniform marginals | $5 \times 10^{-2}$ | $\sim 3\times$ |

**Memory:** Identical to standard Sinkhorn — $O(n^2)$ for the Gibbs kernel $\gamma^0 = e^{-c/\varepsilon}$ and $O(n_1 + n_2)$ for the dual variables. The adaptive $\omega$ computation requires only the marginal ratio vector $(A_k \gamma) \oslash \mu_k$, which is already computed in the standard Sinkhorn step.

## Applicability

- **Blockwise Sinkhorn channel permutation:** Direct drop-in replacement for the Sinkhorn normalization in PermLLM. Each block's log-domain Sinkhorn iteration gains overrelaxation at zero extra memory cost. At temperature $\tau = 0.1$ (the final annealed value in PermLLM), $\eta$ is small and overrelaxation provides the largest gains — exactly when it is most needed.
- **Sinkhorn permutation relaxation:** The Gumbel-Sinkhorn operator benefits from overrelaxation, especially at low temperatures where the doubly stochastic output must be sharp. Reducing Sinkhorn iterations from 20 to 5 per forward pass significantly reduces computational cost in permutation learning training loops.
- **Temperature-annealed Sinkhorn:** PermLLM anneals temperature from $\tau = 1.0$ to $\tau = 0.1$ during training. Overrelaxation provides increasing benefits as $\tau$ decreases, automatically accelerating the Sinkhorn convergence in the difficult late-training regime.
- **Expert-choice routing:** Sinkhorn-based token-to-expert assignment in Mixture-of-Experts benefits from faster convergence, reducing the routing overhead per forward pass.
- **Differentiable sorting and ranking:** SoftSort, NeuralSort, and other differentiable sorting algorithms that use Sinkhorn normalization as a subroutine.
- **Composition with FlashSinkhorn and SNS:** Overrelaxation is orthogonal to both IO-aware tiling (FlashSinkhorn) and Newton acceleration (SNS). All three can be combined: overrelaxed Sinkhorn warm-up with FlashSinkhorn tiling, followed by SNS Newton steps.

## Limitations

- The optimal $\theta^*$ depends on $\eta$, the second-largest eigenvalue of the SK iteration matrix, which is generally unknown a priori — the adaptive heuristic based on the Lyapunov function adds a small computational overhead (a few Newton iterations on a 1D function per step)
- For $\omega$ too close to 2, the algorithm can oscillate — the adaptive scheme with safety margin $\delta$ prevents this but may be conservative
- The convergence analysis is local: the quadratic rate improvement $1/\sqrt{\eta}$ applies only near convergence. In early iterations, the adaptive $\omega$ may select values close to 1 (no acceleration)
- For well-conditioned problems (large $\varepsilon$, $\eta$ close to 1), the speedup is negligible since standard Sinkhorn already converges quickly
- The geometric mean operation $u^{1-\omega} \otimes \tilde{u}^\omega$ in log-domain is a weighted average $(1-\omega)\log u + \omega \log \tilde{u}$ — this is numerically stable but introduces an additional elementwise operation per step
- The overrelaxation parameter must satisfy $\omega < 2$ strictly; approaching 2 risks divergence

## Implementation Notes

```python
import torch

def overrelaxed_sinkhorn(log_alpha, n_iters=10, tau=1.0, theta_0=1.9, delta=0.05):
    """
    Overrelaxed Sinkhorn normalization in log-domain.

    Drop-in replacement for standard Sinkhorn with faster convergence.

    Args:
        log_alpha: (n, n) log-scores matrix (e.g., learnable W_P / tau)
        n_iters: number of Sinkhorn iterations (can use fewer than standard)
        tau: temperature parameter
        theta_0: target upper bound for overrelaxation (1.0 = standard SK)
        delta: safety margin for adaptive omega

    Returns:
        (n, n) doubly stochastic matrix
    """
    log_alpha = log_alpha / tau
    n = log_alpha.shape[0]

    for _ in range(n_iters):
        # Row normalization with overrelaxation
        log_alpha_tilde = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        omega = _adaptive_omega(log_alpha, log_alpha_tilde, dim=-1, theta_0=theta_0, delta=delta)
        log_alpha = (1 - omega) * log_alpha + omega * log_alpha_tilde

        # Column normalization with overrelaxation
        log_alpha_tilde = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        omega = _adaptive_omega(log_alpha, log_alpha_tilde, dim=-2, theta_0=theta_0, delta=delta)
        log_alpha = (1 - omega) * log_alpha + omega * log_alpha_tilde

    return torch.exp(log_alpha)


def _adaptive_omega(log_curr, log_proj, dim, theta_0=1.9, delta=0.05):
    """
    Adaptively choose overrelaxation parameter omega.

    Uses Lyapunov-based criterion: find largest omega in [1, theta_0]
    such that phi_omega(min(marginal_ratio)) >= 0.

    Args:
        log_curr: current log-iterate
        log_proj: log-iterate after standard projection
        dim: dimension of the marginal constraint
        theta_0: target upper bound
        delta: safety margin

    Returns:
        omega: scalar overrelaxation parameter
    """
    # Marginal ratio: how far current iterate is from satisfying the constraint
    # r_i = (sum_j gamma_{ij}) / mu_i
    # In log domain: log_r = logsumexp(log_curr, dim) - log(mu)
    # For uniform marginals: log(mu) = -log(n)
    log_marginal = torch.logsumexp(log_curr, dim=dim)
    r_min = torch.exp(log_marginal).min().item()

    if r_min <= 0 or r_min >= 1.0:
        return 1.0  # Fallback to standard Sinkhorn

    # Find largest omega such that phi_omega(r_min) >= 0
    # phi_omega(x) = x(1 - x^{-omega}) - omega * log(x)
    # Use bisection on omega in [1, theta_0]
    omega_lo, omega_hi = 1.0, theta_0
    for _ in range(8):  # 8 bisection steps
        omega_mid = (omega_lo + omega_hi) / 2
        phi = r_min * (1 - r_min ** (-omega_mid)) - omega_mid * torch.log(torch.tensor(r_min))
        if phi >= 0:
            omega_lo = omega_mid
        else:
            omega_hi = omega_mid

    return max(1.0, omega_lo - delta)


def overrelaxed_sinkhorn_fixed(log_alpha, n_iters=10, tau=1.0, omega=1.5):
    """
    Simplified overrelaxed Sinkhorn with fixed omega.

    For quick experiments; use omega ~= 1.5 as a safe default.
    """
    log_alpha = log_alpha / tau

    for _ in range(n_iters):
        # Overrelaxed row normalization
        log_alpha_proj = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = (1 - omega) * log_alpha + omega * log_alpha_proj

        # Overrelaxed column normalization
        log_alpha_proj = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = (1 - omega) * log_alpha + omega * log_alpha_proj

    return torch.exp(log_alpha)


# Usage in blockwise Sinkhorn channel permutation:
# Replace:
#   P_soft = sinkhorn_block(W_P, tau=0.1, n_iters=20)
# With:
#   P_soft = overrelaxed_sinkhorn(W_P, tau=0.1, n_iters=5, theta_0=1.9)
#
# Same per-iteration cost, ~4x fewer iterations needed.
```

## References

- Thibault, A., Chizat, L., Dossal, C. & Papadakis, N. (2021). Overrelaxed Sinkhorn-Knopp Algorithm for Regularized Optimal Transport. Algorithms, 14(5), 143. arXiv:1711.01851.
- Lehmann, T., von Renesse, M.-K., Sambale, A. & Uschmajew, A. (2020). A Note on Overrelaxation in the Sinkhorn Algorithm. Optimization Letters. arXiv:2012.12562.
- Knight, P. A. (2008). The Sinkhorn-Knopp Algorithm: Convergence and Applications. SIAM J. Matrix Anal. Appl.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.
- Chizat, L. (2017). Unbalanced Optimal Transport: Models, Numerical Methods, Applications. PhD thesis, Universite Paris Dauphine.
- Anderson, D. G. (1965). Iterative Procedures for Nonlinear Integral Equations. J. ACM, 12(4), 547-560.

# 040: ε-Scaling Log-Stabilized Sinkhorn Algorithm

**Category**: stability
**Gain type**: efficiency
**Source**: Schmitzer (TU München; SIAM J. Sci. Comput. 2019)
**Paper**: [papers/stabilized-sparse-sinkhorn-scaling.pdf]
**Documented**: 2026-02-15

## Description

The Sinkhorn-Knopp algorithm is the standard method for solving entropy-regularized optimal transport and for projecting matrices onto the Birkhoff polytope (doubly stochastic matrices). In blockwise Sinkhorn channel permutation (PermLLM), each $B \times B$ block's learnable parameter matrix is Sinkhorn-normalized at temperature $\tau$ (equivalently, regularization $\varepsilon = \tau$). As temperature is annealed from $\tau = 1.0$ toward $\tau = 0.1$ during training, standard Sinkhorn encounters three compounding numerical failures:

1. **Diverging scaling factors:** The diagonal scaling vectors $u, v$ grow as $\exp(C/\varepsilon)$ where $C = \max c$. At $\varepsilon = 0.1$, entries can exceed $10^{40}$, causing float32 overflow.
2. **Slow convergence:** The number of iterations to reach accuracy $q_\text{target}$ scales as $O(C / \varepsilon(1 - q_\text{target}))$ — the iteration count grows inversely with $\varepsilon$.
3. **Dense kernel matrix:** The Gibbs kernel $K_{ij} = \exp(-c_{ij}/\varepsilon)$ is dense $n \times n$, requiring $O(n^2)$ memory and $O(n^2)$ per iteration even when the optimal coupling is sparse.

This trick combines four modifications that address all three issues simultaneously:

**1. Log-domain stabilization (Section 3.1):** Instead of maintaining scaling factors $u = \exp(\alpha/\varepsilon)$, $v = \exp(\beta/\varepsilon)$ that diverge, introduce a redundant parametrization $u = \tilde{u} \odot \exp(\hat{\alpha}/\varepsilon)$, $v = \tilde{v} \odot \exp(\hat{\beta}/\varepsilon)$ where $(\hat{\alpha}, \hat{\beta})$ are "absorbed" dual variables and $(\tilde{u}, \tilde{v})$ are "relative" scaling factors. The algorithm alternates between stabilized iterations (updating $\tilde{u}, \tilde{v}$ using the stabilized kernel $\bar{K}$) and absorption iterations (absorbing large values of $\tilde{u}, \tilde{v}$ into $\hat{\alpha}, \hat{\beta}$ and resetting $\tilde{u}, \tilde{v} \leftarrow 1$). The extreme values in $\exp(\cdot/\varepsilon)$ cancel between the kernel and the dual variables, preventing overflow while retaining the simple matrix-scaling structure.

**2. ε-scaling (Section 3.2):** Instead of solving directly at the target $\varepsilon$, solve a sequence of problems with decreasing regularization $\varepsilon_1 > \varepsilon_2 > \cdots > \varepsilon_n$, using the dual solution at $\varepsilon_k$ as initialization for $\varepsilon_{k+1}$. The key insight (from auction algorithm theory) is that the iteration count per stage is bounded by $O(\varepsilon_1 N \cdot (4\log N + 24\log M) / \varepsilon_2(1 - q_\text{target}))$, which does not grow as the overall $\varepsilon$ decreases. With geometric scheduling $\varepsilon_k = \varepsilon_0 \cdot \lambda^k$ ($\lambda \in [0.5, 0.75]$), the total iterations scale as $O(\log(C/\varepsilon))$ rather than $O(C/\varepsilon)$.

**3. Kernel truncation (Section 3.3):** For small $\varepsilon$, most entries of the optimal coupling $\pi^\dagger$ are exponentially close to zero. Truncate the stabilized kernel to a sparse support $\mathcal{N} = \{(x,y) : \exp(-[c(x,y) - \hat{\alpha}(x) - \hat{\beta}(y)]/\varepsilon) \geq \theta\}$. The truncation error on the primal-dual gap is bounded by $\|\tilde{u}\|_\infty \cdot \|\tilde{v}\|_\infty \cdot \theta \cdot \rho(X \times Y)$, which is negligible when absorption is performed regularly. Sparse kernel iterations cost $O(|\mathcal{N}|)$ instead of $O(n^2)$.

**4. Multi-scale scheme (Section 3.4):** Combine a hierarchical coarse-to-fine partition of the source and target spaces with ε-scaling. At coarse levels (large $\varepsilon$), use coarsened measures with few variables; at fine levels (small $\varepsilon$), the dual solution from the coarse level provides a warm start that makes the fine-level truncated kernel already concentrated on the correct support. This enables solving large problems ($n > 10^5$) that are infeasible for flat Sinkhorn.

For the blockwise Sinkhorn channel permutation setting, modifications 1 and 2 are most relevant: log-domain stabilization prevents overflow during temperature annealing ($\tau: 1.0 \to 0.1$), and ε-scaling accelerates convergence at low temperature by solving through a sequence of intermediate temperatures.

## Mathematical Form

**Standard Sinkhorn (Scaling Form):**

Given kernel $K \in \mathbb{R}_+^{X \times Y}$ with $K(x,y) = \exp(-c(x,y)/\varepsilon) \cdot \rho(x,y)$, the scaling iterations are:

$$
u^{(\ell+1)} = \mu \oslash (K v^{(\ell)}), \qquad v^{(\ell+1)} = \nu \oslash (K^\top u^{(\ell+1)})
$$

The optimal coupling is $\pi^\dagger = \text{diag}(u^\dagger) K \text{diag}(v^\dagger)$.

**Log-Domain Stabilized Form (Algorithm 2):**

Decompose scaling factors as:

$$
u = \tilde{u} \odot \exp(\hat{\alpha}/\varepsilon), \qquad v = \tilde{v} \odot \exp(\hat{\beta}/\varepsilon)
$$

Define the stabilized kernel:

$$
[\bar{K}(\hat{\alpha}, \hat{\beta}, \varepsilon)](x,y) = \exp\left(-\frac{1}{\varepsilon}[c(x,y) - \hat{\alpha}(x) - \hat{\beta}(y)]\right) \cdot \rho(x,y)
$$

**Stabilized iterations** (while $\|\tilde{u}\|_\infty, \|\tilde{v}\|_\infty \leq \tau$):

$$
\tilde{u} \leftarrow \text{proxdiv}_\varepsilon F_X(\bar{K} \tilde{v}, \hat{\alpha}), \qquad \tilde{v} \leftarrow \text{proxdiv}_\varepsilon F_Y(\bar{K}^\top \tilde{u}, \hat{\beta})
$$

For standard OT with marginals $\mu, \nu$, these reduce to:

$$
\tilde{u} \leftarrow \mu \oslash (\bar{K} \tilde{v}), \qquad \tilde{v} \leftarrow \nu \oslash (\bar{K}^\top \tilde{u})
$$

**Absorption iteration** (when $\|\tilde{u}\|_\infty$ or $\|\tilde{v}\|_\infty > \tau$):

$$
(\hat{\alpha}, \hat{\beta}) \leftarrow (\hat{\alpha}, \hat{\beta}) + \varepsilon \cdot \log(\tilde{u}, \tilde{v}); \quad (\tilde{u}, \tilde{v}) \leftarrow (1_X, 1_Y); \quad \bar{K} \leftarrow \text{get}\bar{K}(\hat{\alpha}, \hat{\beta}, \varepsilon)
$$

The absorption step shifts the extreme values from the scaling factors into the dual variables and recomputes the stabilized kernel. Since $\exp(\cdot/\varepsilon)$ terms in $\hat{\alpha}, \hat{\beta}$ cancel with those in $\bar{K}$, the numerical range of $\tilde{u}, \tilde{v}$ stays bounded.

**ε-Scaling (Algorithm 3):**

Given a decreasing schedule $\mathcal{E} = (\varepsilon_1, \ldots, \varepsilon_n)$:

$$
\text{for } \varepsilon \in \mathcal{E}: \quad (\alpha, \beta) \leftarrow \text{ScalingAlgorithmStabilized}(\varepsilon, \theta, \alpha, \beta)
$$

The dual variables $(\alpha, \beta)$ remain stable under ε-changes (Theorem 20), while the scaling factors $(u, v)$ would diverge. Recommended schedule: $\varepsilon_k = \varepsilon_0 \cdot \lambda^k$ with $\lambda \in [0.5, 0.75]$, $\varepsilon_0 \approx \max c$.

**Iteration Bound (Proposition 18, Asymmetric Sinkhorn):**

For fixed $\varepsilon > 0$ with $C = \max c$, the number of iterations to achieve assigned mass fraction $q^{(n)} \geq q_\text{target}$ is:

$$
n \leq 2 + \frac{C}{\varepsilon(1 - q_\text{target})}
$$

**ε-Scaling Iteration Bound (Proposition 24):**

For a single ε-scaling step from $\varepsilon_1$ to $\varepsilon_2$:

$$
n \leq 2 + \frac{\varepsilon_1}{\varepsilon_2} \cdot \frac{N \cdot (4\log N + 24\log M) + \log M}{1 - q_\text{target}}
$$

The crucial difference: the bound depends on $\varepsilon_1 / \varepsilon_2$ (the ratio between consecutive scales, typically $\sim 2$), not on $C/\varepsilon_2$ (which grows as $\varepsilon_2 \to 0$).

**Kernel Truncation:**

Truncate the kernel to its significant entries:

$$
\mathcal{N}(\hat{\alpha}, \hat{\beta}, \varepsilon, \theta) := \{(x,y) : \exp(-[c(x,y) - \hat{\alpha}(x) - \hat{\beta}(y)]/\varepsilon) \geq \theta\}
$$

$$
[\hat{K}(\hat{\alpha}, \hat{\beta}, \varepsilon, \theta)](x,y) := \begin{cases} \bar{K}(x,y) & \text{if } (x,y) \in \mathcal{N} \\ 0 & \text{else} \end{cases}
$$

**Truncation Error Bound (Proposition 11):**

$$
E(\pi) - J(\alpha, \beta) \leq \hat{E}(\pi) - \hat{J}(\alpha, \beta) + \|\tilde{u}\|_\infty \cdot \|\tilde{v}\|_\infty \cdot \theta \cdot \rho(X \times Y)
$$

**Stability of Dual Solutions (Theorem 20):**

For two regularization parameters $\varepsilon_1 > \varepsilon_2 > 0$ with optimal duals $(\alpha_1, \beta_1)$ and $(\alpha_2, \beta_2)$:

$$
\max \Delta\alpha - \min \Delta\alpha \leq \varepsilon_1 \cdot N \cdot (4\log N + 24\log M)
$$

$$
\max \Delta\beta - \min \Delta\beta \leq \varepsilon_1 \cdot N \cdot (4\log N + 24\log M)
$$

where $\Delta\alpha = \alpha_2 - \alpha_1$. This stability result — crucially independent of the cost function $c$ — justifies warm-starting: the dual variables change only mildly between ε-scales.

## Complexity

| Operation | Standard Sinkhorn | Log-Stabilized | + ε-Scaling | + Truncation |
|-----------|-------------------|---------------|-------------|-------------|
| Numerical range | $u_i \in [e^{-C/\varepsilon}, e^{C/\varepsilon}]$ | $\tilde{u}_i \in [1/\tau, \tau]$ | Same | Same |
| Iterations to precision $q$ | $O(C/\varepsilon(1-q))$ | Same (just stable) | $O(\log(C/\varepsilon))$ | $O(\log(C/\varepsilon))$ |
| Cost per iteration | $O(n^2)$ | $O(n^2)$ + periodic $O(n^2)$ absorption | $O(n^2)$ per scale | $O(|\mathcal{N}|) \ll O(n^2)$ |
| Memory | $O(n^2)$ dense kernel | $O(n^2)$ + $O(n)$ dual vars | $O(n^2)$ | $O(|\mathcal{N}|)$ sparse |
| Works at $\varepsilon \to 0$? | No (overflow) | Yes | Yes (fast) | Yes (fast + sparse) |

**Concrete example ($n = 256$, $C = 1$, target $\varepsilon = 0.01$):**

- Standard Sinkhorn: $\sim C/\varepsilon = 100$ iterations, but scaling factors $\sim e^{100}$ cause float32 overflow
- Log-stabilized: same $\sim 100$ iterations, but numerically stable
- + ε-scaling ($\lambda = 0.7$, 7 stages): $\sim 7 \times 15 = 105$ total iterations, but each stage converges quickly with warm start
- + truncation ($\theta = 10^{-10}$): sparse kernels at low ε reduce per-iteration cost

**For blockwise Sinkhorn channel permutation ($B = 64$, $\tau: 1.0 \to 0.1$):**

- At $\tau = 1.0$: standard Sinkhorn works fine, 5 iterations sufficient
- At $\tau = 0.3$: scaling factors start growing, log-stabilization needed
- At $\tau = 0.1$: $C/\varepsilon \approx 10$ per entry, stabilization essential; ε-scaling reduces iterations from $\sim 15$ to $\sim 5$ per block

## Applicability

- **Blockwise Sinkhorn channel permutation (PermLLM):** The temperature annealing schedule ($\tau: 1.0 \to 0.1$) during training creates exactly the regime where standard Sinkhorn fails numerically. Log-domain stabilization prevents overflow at low $\tau$, and ε-scaling accelerates convergence by warm-starting from the previous temperature's dual solution. The block structure ($N_B$ independent $B \times B$ blocks) makes this embarrassingly parallel.
- **Learnable permutation cost bipartite matching:** The Sinkhorn solver in the differentiable bipartite matching framework (arXiv 2601.22980) uses entropy-regularized OT with temperature annealing. Log-stabilization and ε-scaling directly improve the solver's numerical robustness and convergence.
- **FlashSinkhorn integration:** FlashSinkhorn already uses log-domain Sinkhorn updates internally (the attention-form rewriting is inherently in log-domain). The ε-scaling heuristic can be layered on top by running FlashSinkhorn at a sequence of decreasing $\varepsilon$ values, warm-starting dual potentials between scales.
- **Overrelaxed Sinkhorn:** The overrelaxation technique (accelerating convergence at each $\varepsilon$) composes naturally with ε-scaling (reducing the number of $\varepsilon$ stages). The stabilized formulation ensures both tricks remain numerically stable.
- **Sinkhorn-Newton-Sparse (SNS):** The log-stabilized Sinkhorn provides a better warm-up phase for the Newton stage of SNS, since the dual variables $(\hat{\alpha}, \hat{\beta})$ are already in a numerically stable range.
- **General entropic OT:** Applies to Wasserstein barycenters, multi-marginal OT, unbalanced transport, and Wasserstein gradient flows — all use Sinkhorn-type scaling algorithms.

## Limitations

- **Log-domain overhead:** The stabilized kernel $\bar{K}$ must be recomputed at each absorption iteration, costing $O(n^2)$. In practice, absorption is infrequent (every 10-50 stabilized iterations) so the amortized cost is low, but it adds implementation complexity.
- **ε-scheduling is heuristic:** While the theory proves stability of dual solutions under ε-changes (Theorem 20), the optimal schedule $\mathcal{E} = (\varepsilon_0 \lambda^0, \varepsilon_0 \lambda^1, \ldots)$ depends on the problem structure. Too aggressive ($\lambda < 0.5$) causes numerical instability; too conservative ($\lambda > 0.8$) wastes iterations.
- **Kernel truncation requires dual variable quality:** The sparse support $\mathcal{N}$ is determined from the current duals $(\hat{\alpha}, \hat{\beta})$. If these are far from optimal (early in optimization), the truncation may discard important entries. In practice, truncation should only be applied after a few ε-scaling stages.
- **Not directly differentiable through ε-scaling:** If the Sinkhorn solver is inside a training loop, the ε-scaling schedule adds control flow that complicates automatic differentiation. For blockwise channel permutation (where Sinkhorn runs in the inner loop of training), a fixed ε (the current temperature $\tau$) with log-stabilization may be more practical than full ε-scaling.
- **Multi-scale scheme requires spatial structure:** The coarse-to-fine approach (Section 3.4) requires hierarchical partitioning of the source and target spaces, which is natural for point clouds but less so for the abstract cost matrices in channel permutation.

## Implementation Notes

```python
import torch

def stabilized_sinkhorn(C, mu, nu, eps, n_iters=100, tau=1e3,
                        eps_schedule=None):
    """
    Log-domain stabilized Sinkhorn with optional ε-scaling.

    Args:
        C: (n, m) cost matrix
        mu: (n,) source marginal
        nu: (m,) target marginal
        eps: target regularization parameter
        n_iters: max iterations per ε-level
        tau: absorption threshold for scaling factors
        eps_schedule: list of decreasing ε values (ε-scaling)
                      If None, solve at target eps directly.

    Returns:
        alpha: (n,) dual variable (in log-domain)
        beta: (m,) dual variable (in log-domain)
    """
    n, m = C.shape

    # Initialize dual variables
    alpha_hat = torch.zeros(n, dtype=C.dtype, device=C.device)
    beta_hat = torch.zeros(m, dtype=C.dtype, device=C.device)

    if eps_schedule is None:
        eps_schedule = [eps]

    for eps_k in eps_schedule:
        # Compute stabilized kernel
        # K_bar[i,j] = exp(-(C[i,j] - alpha_hat[i] - beta_hat[j]) / eps_k)
        # For numerical evaluation, compute in log-domain

        # Initialize relative scaling factors
        u_tilde = torch.ones(n, dtype=C.dtype, device=C.device)
        v_tilde = torch.ones(m, dtype=C.dtype, device=C.device)

        for it in range(n_iters):
            # Compute stabilized kernel-vector products in log-domain
            # log(K_bar @ v_tilde) for each i:
            log_Kv = torch.logsumexp(
                -(C - alpha_hat.unsqueeze(1) - beta_hat.unsqueeze(0)) / eps_k
                + torch.log(v_tilde).unsqueeze(0),
                dim=1
            )
            u_tilde = mu / torch.exp(log_Kv)

            # log(K_bar^T @ u_tilde) for each j:
            log_KTu = torch.logsumexp(
                -(C - alpha_hat.unsqueeze(1) - beta_hat.unsqueeze(0)) / eps_k
                + torch.log(u_tilde).unsqueeze(1),
                dim=0
            )
            v_tilde = nu / torch.exp(log_KTu)

            # Absorption: if scaling factors grow too large
            if u_tilde.abs().max() > tau or v_tilde.abs().max() > tau:
                alpha_hat = alpha_hat + eps_k * torch.log(u_tilde)
                beta_hat = beta_hat + eps_k * torch.log(v_tilde)
                u_tilde = torch.ones_like(u_tilde)
                v_tilde = torch.ones_like(v_tilde)

            # Check convergence (marginal error)
            # In practice, check primal-dual gap

        # Final absorption before next ε-level
        alpha_hat = alpha_hat + eps_k * torch.log(u_tilde)
        beta_hat = beta_hat + eps_k * torch.log(v_tilde)

    return alpha_hat, beta_hat


def eps_scaling_schedule(eps_target, C_max, lam=0.65):
    """
    Generate geometric ε-scaling schedule.

    Args:
        eps_target: final target regularization
        C_max: maximum cost value
        lam: geometric ratio (0.5-0.75 recommended)

    Returns:
        schedule: list of decreasing ε values
    """
    eps_0 = C_max  # start at order of max cost
    schedule = []
    eps = eps_0
    while eps > eps_target:
        schedule.append(eps)
        eps *= lam
    schedule.append(eps_target)
    return schedule


def stabilized_sinkhorn_block_permutation(W_P, tau, n_iters=5):
    """
    Log-stabilized Sinkhorn for a single block's permutation matrix.
    Drop-in replacement for standard Sinkhorn in PermLLM.

    Args:
        W_P: (B, B) learnable parameter matrix
        tau: temperature (annealed from 1.0 to 0.1)
        n_iters: Sinkhorn iterations

    Returns:
        P_soft: (B, B) doubly stochastic matrix
    """
    # Use log-domain Sinkhorn (numerically stable at low tau)
    log_alpha = W_P / tau

    # Log-domain Sinkhorn iterations
    for _ in range(n_iters):
        # Row normalization in log-domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization in log-domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

    return torch.exp(log_alpha)


# For PermLLM with ε-scaling during temperature annealing:
# Instead of jumping directly to tau_current, solve through
# intermediate temperatures:
#
# tau_schedule = eps_scaling_schedule(tau_current, C_max=10.0, lam=0.7)
# alpha, beta = stabilized_sinkhorn(cost_matrix, mu, nu,
#                                    eps=tau_current,
#                                    eps_schedule=tau_schedule)
#
# This warm-starts each temperature level from the previous solution,
# dramatically improving convergence at low tau.
```

## References

- Schmitzer, B. (2019). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM J. Sci. Comput., 41(3), A1443-A1481. arXiv:1610.06519.
- Chizat, L. et al. (2018). Scaling Algorithms for Unbalanced Optimal Transport Problems. Math. Comp., 87(314):2563-2609.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.
- Bertsekas, D. (1981). A new algorithm for the assignment problem. Math. Programming, 21:152-171.
- Peyré, G. & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in Machine Learning.
- Knight, P. A. (2008). The Sinkhorn-Knopp Algorithm: Convergence and Applications. SIAM J. Matrix Anal. Appl.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Zou, L. et al. (2025). PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models. NeurIPS 2025.

# 171: Neumann Series Implicit Hessian Inverse Optimizer

**Category**: approximation
**Gain type**: efficiency
**Source**: Krishnan, Xiao & Saurous (2017) — Google Research; arXiv:1712.03298
**Paper**: [papers/neumann-optimizer.pdf]
**Documented**: 2026-02-15

## Description

The **Neumann Optimizer** is a second-order stochastic optimization algorithm that implicitly inverts the Hessian of individual mini-batches using the Neumann series expansion, without ever forming the Hessian matrix or computing Hessian-vector products. The key insight combines two ideas: (1) the Neumann series $A^{-1} = \sum_{i=0}^{\infty} (I - A)^i$ can be applied iteratively as a Richardson iteration $z_{t+1} = (I_n - A)z_t + b$ to solve $Az = b$; and (2) for the specific system $[\nabla^2 \hat{f}](w - w_t) = -\nabla \hat{f}$ arising from a quadratic approximation of the mini-batch loss, the Hessian-vector product $\nabla^2 \hat{f} \cdot m$ can be approximated by a finite difference of gradients: $\nabla \hat{f}(w_t + \eta m) \approx \nabla \hat{f}(w_t) + \eta \nabla^2 \hat{f} \cdot m$.

This double approximation eliminates the Hessian entirely: each inner-loop step becomes $m_k = m_{k-1} - \nabla \hat{f}(w_t + \eta m_{k-1})$, requiring only a single gradient evaluation (one forward + backward pass). The result is a second-order optimizer whose per-step cost is identical to SGD — one gradient per inner iteration — yet it implicitly incorporates curvature information from the mini-batch Hessian.

The algorithm scales to large batch sizes (up to 32,000 on ImageNet) without quality degradation, achieving linear speedup. At fixed batch sizes, it improves top-1 validation error by 0.8–0.9% across Inception-V3, ResNet-50, ResNet-101, and Inception-ResNet-V2 on ImageNet, compared to well-tuned baselines (RMSProp/SGD+momentum).

## Mathematical Form

**Setup: Quadratic Approximation of Mini-Batch Loss**

Given weights $w_t$ and mini-batch loss $\hat{f}(w) = \frac{1}{B}\sum_{i=1}^{B} f_{t_i}(w)$, form the quadratic:

$$
\hat{f}(w) \approx \hat{f}(w_t) + \nabla\hat{f}^\top(w - w_t) + \frac{1}{2}(w - w_t)^\top [\nabla^2 \hat{f}](w - w_t)
$$

Minimizing gives the Newton step, which requires solving:

$$
[\nabla^2 \hat{f}](w - w_t) = -\nabla\hat{f}(w_t)
$$

**Neumann Series for the Hessian Inverse:**

Setting $A = \eta \nabla^2 \hat{f}$ with $\eta < 1/\lambda_{\max}$ (where $\lambda_{\max}$ is the largest eigenvalue of $\nabla^2 \hat{f}$), the Richardson iteration:

$$
z_0 = b, \qquad z_{t+1} = (I_n - A)z_t + b
$$

converges to $A^{-1}b$. With $b = -\nabla\hat{f}(w_t)$:

$$
m_{t+1} = (I_n - \eta\nabla^2\hat{f})m_t - \nabla\hat{f}(w_t)
$$

$$
= m_t - (\nabla\hat{f}(w_t) + \eta\nabla^2\hat{f} \cdot m_t)
$$

**Gradient Approximation of Hessian-Vector Product:**

The crucial observation: via Taylor expansion,

$$
\nabla\hat{f}(w_t + \eta m_t) = \nabla\hat{f}(w_t) + \eta\nabla^2\hat{f} \cdot m_t + O(\|\eta m_t\|^2)
$$

For sufficiently small $\|\eta m_t\|$, the bold terms combine:

$$
m_{t+1} \approx m_t - \nabla\hat{f}(w_t + \eta m_t)
$$

This replaces the Hessian-vector product with a **single gradient evaluation at a shifted point**. No Hessian is ever formed or stored.

**Convexification for Non-Convex Mini-Batch Losses:**

Mini-batch Hessians may have negative eigenvalues. Add a cubic regularizer $\frac{\alpha}{3}\|w - w_t\|^3$ and a repulsive term $\beta/\|w - v_t\|$ to the objective (where $v_t$ is an exponential moving average of $w_t$):

$$
\hat{g}(w) = \hat{f}(w) + \frac{\alpha}{3}\|w - w_t\|^3 + \frac{\beta}{\|w - v_t\|}
$$

With eigenvalues of $\nabla^2\hat{g}$ satisfying $\lambda_{\min} < \lambda(\nabla^2\hat{g}) < \lambda_{\max}$, define:

$$
\mu = \frac{\lambda_{\max}}{|\lambda_{\min}| + \lambda_{\max}}, \qquad \eta = \frac{1}{\lambda_{\max}}
$$

The modified matrix $\hat{B} = (1 - \mu)I_n + \mu\eta\nabla^2\hat{g}$ is positive definite. The Neumann iteration becomes:

$$
m_k \approx \mu m_{k-1} - \eta\nabla\hat{g}(w_t + \mu m_{k-1})
$$

This adds momentum-like weighting $\mu$ that increases from 0.5 to 0.9 during training as negative eigenvalues shrink.

**Practical Algorithm (Algorithm 2):**

$$
\begin{aligned}
d_t &= \nabla\hat{f}(w_t) + \left(\alpha\|w_t - v_t\|^2 - \frac{\beta}{\|w_t - v_t\|^2}\right)\frac{w_t - v_t}{\|w_t - v_t\|} \\
m_t &= \mu(t) m_{t-1} - \eta(t) d_t \\
w_t &= w_{t-1} + \mu(t) m_{t-1} - \eta(t) d_t \\
v_t &= v_t + \gamma(v_{t-1} - w_t)
\end{aligned}
$$

where $\mu(t) \propto 1 - \frac{1}{1+t}$ ramps from 0.5 to 0.9, $\eta(t) \propto 1/t$ decays, and $\gamma = 0.99$.

**Key Definitions:**

- $\hat{f}(w)$ — mini-batch loss function
- $\nabla^2\hat{f}$ — mini-batch Hessian (never explicitly computed)
- $\lambda_{\max}, \lambda_{\min}$ — extreme eigenvalues of mini-batch Hessian
- $\mu$ — momentum/convexification parameter
- $K$ — number of inner-loop iterations (starts at 10 epochs, doubles periodically)
- $\alpha = 10^{-7}$ — cubic regularizer weight
- $\beta = 10^{-5} \times n$ — repulsive regularizer weight ($n$ = number of parameters)

## Complexity

| Operation | SGD/Adam | Newton (explicit) | Neumann Optimizer |
|-----------|----------|-------------------|-------------------|
| Per outer step | 1 gradient | 1 gradient + $O(n^2)$ Hessian | $K$ gradients |
| Per inner step | — | Hessian-vector product $O(n^2)$ | 1 gradient $O(n)$ |
| Memory | $O(n)$ | $O(n^2)$ for Hessian | $O(n)$ (gradient + momentum) |
| Hyperparameters | 3+ (lr, $\beta_1$, $\beta_2$, $\epsilon$) | Many | **1** (learning rate only) |

**Effective cost:** Each outer step computes $K$ gradient evaluations (forward + backward passes), where $K$ starts at a small value and doubles periodically. Since each gradient evaluation has the same cost as one SGD step, the Neumann optimizer costs $K\times$ SGD per outer step. However, $K$ is kept small (the inner and outer loops use stochastic gradients from different mini-batches, making the inner loop a stochastic optimization itself with diminishing returns per iteration).

**Memory:** $O(n)$ — stores only $w_t, m_t, v_t$ (3 copies of the parameter vector), same as SGD with momentum plus EMA.

## Applicability

- **Large-batch distributed training:** The optimizer's primary strength is efficient large-batch training. It scales linearly up to batch size 32,000 on ImageNet without quality loss — a 4× improvement over contemporary baselines (Goyal et al., 2017 scaled to 8,192). At batch 32,000 with ResNet-50, it achieves 24.0% top-1 error (vs 23.9% baseline at batch 1,600).

- **Improved generalization:** At fixed batch sizes, the Neumann optimizer consistently improves top-1 validation error by 0.8–0.9% across architectures:
  - Inception-V3: 21.7% → 20.8% (+0.91%)
  - ResNet-50: 23.9% → 23.0% (+0.94%)
  - ResNet-101: 22.6% → 21.7% (+0.86%)
  - Inception-ResNet-V2: 20.3% → 19.5% (+0.84%)

- **Implicit curvature for structured layers:** The Neumann series approximation to the Hessian inverse is particularly well-conditioned when the loss landscape is locally near-quadratic — a regime that holds for well-initialized large models with batch normalization.

- **Connection to modern optimizers:** The "evaluate gradient at shifted point" idea reappears in lookahead optimizers and is conceptually related to Shampoo's preconditioning. The Neumann optimizer can be seen as a precursor to the implicit curvature exploitation in Muon (trick 164) and SOAP.

## Limitations

- **Sequential inner-loop gradient evaluations:** Each inner-loop step requires a full forward + backward pass through the network with a different mini-batch. These are inherently sequential (each depends on the previous $m_k$), creating a latency bottleneck. On modern GPUs, this means $K\times$ the wall-clock time per optimizer step compared to Adam, unless pipelined.

- **Not tensor-core optimized:** The optimizer's bottleneck is gradient evaluations (standard backprop), not matrix multiplications. There are no additional matmul-heavy operations that would benefit from tensor cores beyond what's already in the forward/backward pass.

- **Negative result on sequence models:** The authors report the optimizer fails on Tacotron (sequence-to-sequence speech synthesis), likely because aggressive gradient clipping invalidates the Taylor approximation $\nabla\hat{f}(w + \eta m) \approx \nabla\hat{f}(w) + \eta\nabla^2\hat{f}\cdot m$.

- **Dated baseline comparisons:** Results are compared against 2017-era optimizers (RMSProp, SGD+momentum) on Tesla P100 GPUs. Modern optimizers (AdamW with warmup + cosine decay, Muon, SOAP) and hardware (A100/H100) may narrow or eliminate the advantage.

- **Inner-loop schedule is heuristic:** The doubling schedule for $K$ and the convexification parameters ($\alpha, \beta$) are set empirically. The paper provides a single set of hyperparameters that works across architectures, but the optimality of these choices is not theoretically justified.

- **Not validated for LLM pretraining:** All experiments are image classification on ImageNet. The optimizer's behavior on language modeling, which has different loss landscape properties (less locally quadratic, more heterogeneous curvature), is unknown.

## Implementation Notes

```python
import torch

class NeumannOptimizer(torch.optim.Optimizer):
    """
    Neumann Optimizer: Implicit Hessian inverse via Neumann series.

    The key trick: replace Hessian-vector product H*m with
    (grad(w + eta*m) - grad(w)) / eta ≈ H*m (Taylor expansion).

    Then further simplify: m_{k+1} = mu*m_k - eta*grad(w + mu*m_k)
    which needs only ONE gradient evaluation per inner step.
    """

    def __init__(self, params, lr=0.01, alpha=1e-7, beta_scale=1e-5,
                 gamma=0.99, mu_start=0.5, mu_end=0.9):
        defaults = dict(lr=lr, alpha=alpha, beta_scale=beta_scale,
                        gamma=gamma, mu_start=mu_start, mu_end=mu_end)
        super().__init__(params, defaults)

    def step(self, closure, K=1):
        """
        One outer step of the Neumann optimizer.

        closure: callable that returns loss and computes gradients
                 Must accept 'perturbed_params' argument for shifted evaluation.
        K: number of inner loop iterations
        """
        # This is the idealized algorithm.
        # In practice, the inner loop evaluates stochastic gradients
        # at w_t + mu * m_t, not deterministic gradients.

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu_start']  # ramps to mu_end during training
            alpha = group['alpha']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['ema'] = p.data.clone()  # v_t: exponential moving avg

                m = state['momentum']
                v = state['ema']

                # Compute regularized gradient direction
                diff = p.data - v
                diff_norm = diff.norm()
                if diff_norm > 1e-8:
                    # Cubic regularizer + repulsive term
                    beta = group['beta_scale'] * p.data.numel()
                    reg = (alpha * diff_norm**2 - beta / diff_norm**2)
                    d = p.grad.data + reg * diff / diff_norm
                else:
                    d = p.grad.data

                # Neumann iteration: m = mu * m - lr * d
                # This is the simplified form where inner/outer loops
                # are "flattened" (Section 3.2)
                m.mul_(mu).add_(d, alpha=-lr)

                # Weight update
                p.data.add_(m)

                # EMA update
                v.mul_(gamma).add_(p.data, alpha=1 - gamma)

        return None


# The CORE TRICK in 3 lines:
# Instead of computing H^{-1} @ grad (costs O(n^2) or O(n) Hessian-vector products),
# use the Neumann series iteratively:
#
#   m_{k+1} = m_k - grad(w + eta * m_k)        # shifted gradient evaluation
#           ≈ m_k - (grad(w) + eta * H * m_k)   # Taylor expansion
#           = (I - eta*H) m_k - grad(w)          # Richardson iteration
#           → H^{-1} @ (-grad(w)) as k → ∞      # Neumann series convergence
#
# Cost per inner step: ONE gradient evaluation (same as SGD)
# No Hessian needed. No Hessian-vector products needed.
# Curvature information comes from evaluating the gradient at a shifted point.
```

**GPU efficiency analysis:**

- **Same per-step cost as SGD:** Each inner iteration is a single forward + backward pass (gradient evaluation at a shifted point). No additional matmuls or special operations.
- **Memory efficient:** $O(n)$ — only stores momentum $m$ and EMA $v$ alongside parameters. Same as SGD+momentum.
- **The tradeoff is wall-clock time:** $K$ inner iterations per outer step means $K\times$ the forward+backward passes. This is the fundamental cost of incorporating second-order information.
- **Batch-size scaling is the GPU win:** The optimizer enables 4× larger batch sizes without quality loss, meaning 4× more data parallelism → linear speedup to 250 GPUs. For distributed training, this is a significant wall-clock improvement.
- **No kernel changes needed:** The optimizer uses only standard autograd operations. No custom CUDA kernels required.

## References

- Krishnan, S., Xiao, Y. & Saurous, R. A. (2017). Neumann Optimizer: A Practical Optimization Algorithm for Deep Neural Networks. arXiv:1712.03298.
- Goyal, P. et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677.
- Martens, J. & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. ICML 2015.
- Gupta, V., Koren, T. & Singer, Y. (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018.
- Agarwal, N., Bullins, B. & Hazan, E. (2016). Second Order Stochastic Optimization in Linear Time. OPT Workshop, ICML 2016.

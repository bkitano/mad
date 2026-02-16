# 057: Gumbel-Softmax Reparameterization

**Category**: approximation
**Gain type**: flexibility
**Source**: Jang et al. (2017), "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017); Maddison et al. (2017), "The Concrete Distribution" (ICLR 2017)
**Paper**: [papers/gumbel-softmax-categorical-reparameterization.pdf]
**Documented**: 2026-02-15

## Description

Enable gradient-based optimization through **discrete categorical gating decisions** by replacing the non-differentiable $\arg\max$ with a differentiable continuous relaxation. The trick combines two ideas: (1) the **Gumbel-Max trick** for sampling from categorical distributions, and (2) a **softmax temperature** that smoothly interpolates between continuous (differentiable) and discrete (one-hot) samples. This allows backpropagation through discrete routing decisions (e.g., which expert to use, which path to take, which tokens to select) that would otherwise require high-variance REINFORCE estimators. The Straight-Through (ST) variant uses $\arg\max$ in the forward pass but the softmax gradient in the backward pass, giving the best of both worlds: discrete decisions at inference, differentiable training.

## Mathematical Form

**Step 1: The Gumbel-Max Trick (exact discrete sampling)**

To draw a sample $z$ from a categorical distribution with class probabilities $\pi_1, \ldots, \pi_k$:

$$
z = \text{one\_hot}\left(\arg\max_i \left[g_i + \log \pi_i\right]\right)
$$

where $g_1, \ldots, g_k$ are i.i.d. samples from $\text{Gumbel}(0, 1)$, obtained by:

$$
g = -\log(-\log(u)), \quad u \sim \text{Uniform}(0, 1)
$$

This is exact but non-differentiable due to $\arg\max$.

**Step 2: The Gumbel-Softmax Relaxation (differentiable approximation)**

Replace $\arg\max$ with $\text{softmax}$ at temperature $\tau > 0$:

$$
y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_{j=1}^{k} \exp((\log \pi_j + g_j) / \tau)} \quad \text{for } i = 1, \ldots, k
$$

**Key Definitions:**

- $\pi = (\pi_1, \ldots, \pi_k)$ — class probabilities (unnormalized logits from a gating network)
- $g_i \sim \text{Gumbel}(0, 1)$ — i.i.d. Gumbel noise for stochastic sampling
- $\tau > 0$ — temperature parameter controlling the sharpness of the relaxation
- $y \in \Delta^{k-1}$ — continuous sample on the $(k-1)$-simplex (approximates one-hot)

**Temperature Behavior:**

$$
\tau \to 0: \quad y \to \text{one\_hot}(\arg\max_i [\log \pi_i + g_i]) \quad \text{(discrete, high variance gradients)}
$$

$$
\tau \to \infty: \quad y \to \text{Uniform}(1/k, \ldots, 1/k) \quad \text{(uniform, low variance gradients)}
$$

**Step 3: Straight-Through (ST) Gumbel-Softmax Estimator**

For scenarios requiring discrete decisions in the forward pass:

$$
\text{Forward:} \quad z = \text{one\_hot}(\arg\max_i [g_i + \log \pi_i])
$$

$$
\text{Backward:} \quad \nabla_\theta z \approx \nabla_\theta y \quad \text{(use softmax gradient as proxy)}
$$

This is implemented via the stop-gradient trick:

$$
z_{\text{ST}} = \text{stopgrad}(z - y) + y
$$

In the forward pass, $z_{\text{ST}} = z$ (discrete). In the backward pass, $\nabla z_{\text{ST}} = \nabla y$ (continuous).

**Density of the Gumbel-Softmax Distribution:**

$$
p_{\pi, \tau}(y_1, \ldots, y_k) = \Gamma(k) \tau^{k-1} \left(\sum_{i=1}^{k} \pi_i / y_i^\tau \right)^{-k} \prod_{i=1}^{k} \left(\pi_i / y_i^{\tau + 1}\right)
$$

**Reparameterization Gradient (the key efficiency gain):**

Since $y = g(\theta, \epsilon)$ where $\epsilon = (g_1, \ldots, g_k)$ is independent noise:

$$
\frac{\partial}{\partial \theta} \mathbb{E}_{z \sim p_\theta}[f(z)] = \frac{\partial}{\partial \theta} \mathbb{E}_\epsilon [f(g(\theta, \epsilon))] = \mathbb{E}_\epsilon \left[\frac{\partial f}{\partial g} \frac{\partial g}{\partial \theta}\right]
$$

This is a **low-variance, single-sample** gradient estimate, compared to the high-variance REINFORCE estimator: $\mathbb{E}_z[f(z) \nabla_\theta \log p_\theta(z)]$.

## Complexity

| Operation | Marginalization | REINFORCE | Gumbel-Softmax |
|-----------|----------------|-----------|----------------|
| Forward passes per sample | $k$ (enumerate all classes) | $1$ | $1$ |
| Gradient variance | $0$ (exact) | High ($\propto k$) | Low |
| Gradient bias | None | None | $O(\tau)$ (anneals to 0) |
| Training speed ($k = 10$) | $1\times$ | $1\times$ | $2\times$ faster |
| Training speed ($k = 100$) | $1\times$ | $1\times$ | $9.9\times$ faster |

**Memory:** $O(k)$ per sample for Gumbel noise and softmax computation — negligible compared to model parameters.

**Key efficiency insight:** Marginalization requires $O(k)$ forward passes per step (one per class). Gumbel-Softmax requires only $O(1)$ forward passes with single-sample gradient estimation, yielding a $k\times$ speedup for models with $k$ discrete categories.

## Applicability

- **Mixture of Experts (MoE) routing:** Differentiable expert selection in sparse MoE models (e.g., Switch Transformer, GShard). Enables end-to-end training of discrete routing decisions
- **Neural Architecture Search (NAS):** DARTS and related methods use Gumbel-Softmax to select operations in a differentiable architecture search
- **Discrete latent variable models:** VAEs with categorical latents (VQ-VAE precursors), discrete autoencoders
- **Sparse attention patterns:** Learning which tokens to attend to via differentiable subset selection
- **Conditional computation:** Input-dependent selection of model sub-networks, layers, or feature subsets
- **Token selection/merging:** Differentiable token pruning in vision transformers
- **Quantization-aware training:** Differentiable approximation of discrete quantization levels

## Limitations

- **Bias-variance tradeoff:** Low temperature $\tau$ gives near-discrete samples but high-variance gradients; high $\tau$ gives smooth gradients but biased (non-discrete) samples
- **Temperature scheduling:** Requires annealing $\tau$ from high to low during training; schedule is a hyperparameter (e.g., $\tau = \max(0.5, \exp(-rt))$)
- **ST estimator is biased:** The Straight-Through gradient $\nabla_\theta z \approx \nabla_\theta y$ introduces a systematic bias between forward and backward pass
- **Softmax bottleneck:** For very large $k$ (thousands of experts), computing softmax over all classes becomes expensive. Top-k pre-filtering is often used in practice
- **Not suitable for hard combinatorial constraints:** The relaxation lives on the simplex, not on the set of valid combinatorial solutions (e.g., permutations require Sinkhorn, not Gumbel-Softmax)

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1.0, hard=False):
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: (*, k) unnormalized log-probabilities
        tau: temperature (lower = more discrete)
        hard: if True, use Straight-Through estimator
    Returns:
        (*, k) sample from the Gumbel-Softmax distribution
    """
    # Sample Gumbel noise: g = -log(-log(u))
    gumbels = -torch.log(-torch.log(
        torch.rand_like(logits).clamp(min=1e-10)
    ))

    # Gumbel-Softmax sample
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)

    if hard:
        # Straight-Through: discrete forward, continuous backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # stopgrad(y_hard - y_soft) + y_soft
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft

# Example: differentiable expert routing
class DifferentiableRouter(torch.nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, n_experts)

    def forward(self, x, tau=1.0, hard=True):
        # x: (batch, d_model)
        logits = self.gate(x)  # (batch, n_experts)
        # Differentiable discrete routing
        routing_weights = gumbel_softmax(logits, tau=tau, hard=hard)
        return routing_weights  # (batch, n_experts) — one-hot if hard

# NOTE: PyTorch provides torch.nn.functional.gumbel_softmax()
# with the same interface
```

## References

- Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017. arXiv:1611.01144.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. ICLR 2017. arXiv:1611.00712.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432.
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 8(3-4), 229–256.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.

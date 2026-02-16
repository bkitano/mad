# 256: WeSaR — Weight Scaling as Reparameterization for Uniform Update Ratios

**Category**: stability
**Gain type**: efficiency
**Source**: Nishida, Nishida & Saito (2024) — "Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes", EMNLP 2024. arXiv:2410.05052
**Paper**: [papers/wesar-weight-scaling-reparameterization.pdf]
**Documented**: 2026-02-16

## Description

WeSaR identifies a root cause of loss spikes in LLM pretraining: **non-uniform parameter norms** across weight matrices. Standard initialization methods (He, Small Init) must set different standard deviations for different weight matrices to satisfy gradient propagation requirements (e.g., $W_o, W_d$ get scaled by $1/\sqrt{2N}$ for residual scaling). This creates matrices with very different norms, leading to **uneven update ratios** $\|\Delta W\| / \|W\|$ — parameters with smaller norms receive disproportionately large relative updates, destabilizing training.

The fix is elegant: introduce a **scalar gate parameter** $\alpha$ per weight matrix, use $\bar{W} = \alpha W$ in the model instead of $W$, and initialize all actual parameters $W$ with the **same small standard deviation** $\sigma$. The gate $\alpha$ absorbs the required per-matrix scaling (e.g., $1/\sqrt{d_{in}}$, $1/\sqrt{2Nd}$) so that gradient propagation requirements are satisfied. Because Adam normalizes gradients by their running variance, the gate reparameterization **does not change the parameter update** — it only changes the norm of $W$, making all update ratios uniform.

This is simpler and more effective than Weight Normalization (no per-step normalization needed), $\sigma$Reparam (no spectral norm computation), and Residual Scaling (reparameterizes all parameters, not just $W_o, W_d$). WeSaR achieves the best perplexity across 130M, 1.3B, and 13B models while eliminating loss spikes, and enables higher learning rates (1e-3 vs. conventional 5e-4) with smaller batch sizes (1M vs. 4M tokens).

## Mathematical Form

**Core Reparameterization:**

For each parameter matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$
W \sim \mathcal{N}(0, \sigma^2), \quad \bar{W} = \alpha W = \frac{\sigma_\cdot}{\sigma} W
$$

where:
- $\sigma$ is a common small standard deviation for all parameters (hyperparameter, e.g., $\sigma^2 = 4 \times 10^{-5}$)
- $\sigma_\cdot$ is the standard deviation required by the initialization method (e.g., He: $\sqrt{1/d_{\text{in}}}$)
- $\alpha = \sigma_\cdot / \sigma$ is the gate parameter (trainable scalar per matrix)
- $\bar{W}$ is the virtual parameter used inside the model

**Key Insight — Adam Invariance:**

The gradient through the gate:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \bar{W}} \cdot \frac{\partial \bar{W}}{\partial W} = \frac{\sigma_\cdot}{\sigma} \cdot \frac{\partial \mathcal{L}}{\partial \bar{W}}
$$

With Adam's update $\Delta W_t = \mu_t \frac{M_t}{\sqrt{V_t}}$, both $M_t$ and $\sqrt{V_t}$ are multiplied by $\sigma_\cdot / \sigma$ equally, so:

$$
\Delta W_t = \mu_t \frac{M_t}{\sqrt{V_t}} \quad \text{(independent of } \sigma_\cdot \text{)}
$$

The reparameterization does not change the parameter update direction or magnitude — it only changes $\|W\|$, making the update ratio $\|\Delta W\| / \|W\|$ uniform across all matrices.

**Gradient Propagation Requirement:**

For a linear layer $\boldsymbol{y} = W\boldsymbol{x}$, stable back-propagation requires:

$$
E\left[\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}}\right\|^2\right] = E\left[\|W^\top \boldsymbol{\delta}\|^2\right] = d_{\text{in}} \text{Var}[W] \cdot E[\|\boldsymbol{\delta}\|^2] = E[\|\boldsymbol{\delta}\|^2]
$$

This requires $\text{Var}[W] = 1/d_{\text{in}}$, i.e., $\sigma_\cdot = 1/\sqrt{d_{\text{in}}}$ (He initialization). WeSaR satisfies this via $\bar{W} = \alpha W$ while keeping $W$ itself at uniform small scale.

**Per-Matrix Gate Values (with He backbone):**

| Matrix | Gate $\alpha = \sigma_\cdot / \sigma$ | Effective $\sigma_\cdot$ |
|--------|---------------------------------------|--------------------------|
| $W_e$ (embedding) | $1$ | $\sigma$ |
| $W_k, W_q$ | $\sqrt{1/d} / \sigma$ | $\sqrt{1/d}$ |
| $W_v$ | $\sqrt{1/d} / \sigma$ | $\sqrt{1/d}$ |
| $W_o$ | $\sqrt{1/(2Nd)} / \sigma$ | $\sqrt{1/(2Nd)}$ |
| $W_u$ (FFN up) | $\sqrt{1/d} / \sigma$ | $\sqrt{1/d}$ |
| $W_d$ (FFN down) | $\sqrt{2/(8Nd)} / \sigma$ | $\sqrt{2/(8Nd)}$ |
| $W_p$ (prediction) | $\sqrt{1/d} / \sigma$ | $\sqrt{1/d}$ |

## Complexity

| Operation | Baseline (Small Init) | With WeSaR |
|-----------|----------------------|------------|
| Forward pass | Same | Same (gate is fused into weight) |
| Backward pass | Same | Same (Adam invariance) |
| Extra parameters | $0$ | $1$ scalar per weight matrix |
| Training steps to target loss | $S$ | $< S$ (faster convergence) |
| Loss spikes | Frequent | Eliminated |

**Memory:** +1 scalar per weight matrix (~10–20 extra scalars total). Negligible.

**Inference:** Zero overhead — gate $\alpha$ can be folded into $W$ after training: $\bar{W}_{\text{final}} = \alpha_{\text{final}} W_{\text{final}}$.

**Training cost:** Essentially zero overhead. No per-step normalization, no spectral norm computation, no extra matrix operations.

## Applicability

- **LLM pretraining** at any scale (validated 130M to 13B). Eliminates loss spikes that waste compute.
- **Drop-in replacement** for any initialization method — WeSaR adds a gate on top of any backbone init (He, Small Init, Xavier).
- Enables **higher learning rates** (1e-3 vs. 5e-4) and **smaller batch sizes** (1M vs. 4M tokens) for faster convergence.
- Compatible with standard training techniques (gradient clipping, warmup, z-loss, weight decay).
- Applicable to **any architecture** with linear layers (Transformers, SSMs, linear attention) — the gate reparameterization is architecture-agnostic.
- Particularly valuable when **scaling up model size**, where the non-uniformity problem worsens (e.g., $1/\sqrt{2N}$ scaling shrinks $W_o, W_d$ norms as depth $N$ increases).

## Limitations

- **Requires Adam optimizer**: The invariance property relies on Adam's per-element gradient normalization. Does not hold for SGD.
- **One cause of many**: The authors acknowledge WeSaR addresses non-uniform norms but not all causes of loss spikes. Other techniques (warmup, gradient clipping, z-loss) remain complementary.
- **Hyperparameter $\sigma$**: While the method is robust ($\sigma^2 \in [1\text{e-5}, 4\text{e-5}]$ all work well), the optimal $\sigma$ is not necessarily proportional to $d$.
- **Not validated with SwiGLU**: Experiments use GeLU FFN. The interaction with gated activation functions (used in LLaMA, Mistral) is not studied.
- **30K GPU-hours on H100**: The experimental cost is significant (~$149K on AWS), limiting reproducibility.

## Implementation Notes

```python
import torch
import torch.nn as nn

class WeSaRLinear(nn.Module):
    """Linear layer with WeSaR gate reparameterization."""

    def __init__(self, d_in, d_out, sigma=0.00632, target_sigma=None):
        super().__init__()
        # All actual parameters initialized with same small sigma
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * sigma)

        # Gate parameter: alpha = target_sigma / sigma
        if target_sigma is None:
            target_sigma = 1.0 / (d_in ** 0.5)  # He init default
        alpha_init = target_sigma / sigma
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        # Virtual parameter: W_bar = alpha * W
        return x @ (self.alpha * self.weight).T

    def merge_gate(self):
        """Merge gate into weight for inference (zero overhead)."""
        self.weight.data *= self.alpha.data
        self.alpha.data.fill_(1.0)

# Usage for Transformer with He backbone + residual scaling:
# sigma^2 = 4e-5 (common for all parameters)
sigma = 0.00632  # sqrt(4e-5)

# Each layer's matrices get appropriate target_sigma:
# W_q, W_k: target = 1/sqrt(d)
# W_o: target = 1/sqrt(2*N*d)  (residual scaling)
# W_u: target = 1/sqrt(d)
# W_d: target = sqrt(2/(8*N*d))  (residual scaling + gain)

# Key insight: after training, just do model.merge_gate()
# for all WeSaR layers => zero inference overhead
```

## GPU Efficiency Analysis

**Memory Access Pattern:**
- The gate multiplication $\alpha W$ can be fused into the GEMM kernel (multiply $\alpha$ into one operand before matmul) — no extra memory access
- Alternatively, $\alpha$ can be applied as a scalar multiply on the output — single extra multiply per matmul
- Fully coalesced, no irregular access

**Parallelism:**
- No additional synchronization points
- No sequential operations added
- Gate gradients are trivial scalar reductions

**Tensor Core Utilization:**
- Identical to baseline — the core computation is the same GEMM
- Gate can be folded into the weight or output scaling with no tensor core impact

**Arithmetic Intensity:**
- Identical to baseline (one extra scalar multiply per forward pass per layer — negligible)
- No additional HBM traffic

**Verdict:** This is essentially a **zero-cost** stability improvement. No wall-clock overhead during training, zero overhead at inference (gate merges into weight). Pure win for any LLM pretraining pipeline.

## References

- Nishida, Nishida & Saito. "Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes." EMNLP 2024. arXiv:2410.05052
- Salimans & Kingma (2016). Weight Normalization: A Simple Reparameterization.
- Zhai et al. (2023). σReparam: Stabilizing Transformers via spectral normalization.
- Noci et al. (2022). Residual Scaling as Reparameterization.
- Takase et al. (2024). Spike No More — Scaled Embed and Embed LN.
- Nguyen & Salazar (2019). Small Initialization for Transformers.
- He et al. (2015). Delving Deep into Rectifiers (He initialization).

# 233: StableSSM — Gradient-Balanced Reparameterization for State-Space Models

**Category**: stability
**Gain type**: expressivity
**Source**: Wang & Li (2024) — "StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization", ICML 2024 (PMLR 235). arXiv:2311.14495
**Paper**: [papers/stablessm-hurwitz-reparameterization.pdf]
**Documented**: 2026-02-15

## Description

State-space models (SSMs) without reparameterization suffer from the **curse of memory**: they can only stably approximate target functionals with **exponentially decaying** memory, even though many real-world sequence dependencies (e.g., power-law decay in language) require longer memory. This paper proves this limitation is fundamental — not an optimization failure — and introduces a principled reparameterization framework that lifts it.

The root cause is that as SSMs learn longer-range dependencies, their recurrent weights $\Lambda = \text{Diag}(\lambda_1, \ldots, \lambda_m)$ converge toward the **stability boundary** ($|\lambda_i| \to 1^-$ in discrete time, $\text{Re}(\lambda_i) \to 0^-$ in continuous time). Near this boundary, two pathologies emerge:

1. **Approximation instability:** Small weight perturbations cause exponentially large changes in the output trajectory, making the learned model fragile.
2. **Gradient explosion:** The gradient $|\partial \text{Loss}/\partial \lambda_i|$ scales as $c/(1 - \lambda_i)^2$ in discrete time — it diverges as $\lambda_i \to 1$.

A **stable reparameterization** $f : \mathbb{R} \to (-\infty, 0)$ maps a trainable parameter $w$ to an eigenvalue $\lambda = f(w)$ such that $\lambda$ is automatically in the stable region ($\text{Re}(\lambda) < 0$) for all $w$. Common choices include:

- **Exponential**: $f(w) = -e^w$ (used in S4, S4D)
- **Softplus**: $f(w) = -\log(1 + e^w)$ (used in S5)

These are all **stable** in the paper's formal sense (Definition 3.4), meaning the perturbation error is bounded. But they are **not equally good** for optimization. The paper derives the **optimal** reparameterization in the gradient-boundedness sense:

$$
f^*(w) = -\frac{1}{aw^2 + b}, \quad a > 0, b \geq 0
$$

This "best" reparameterization achieves the **smallest gradient-over-weight ratio** across all eigenvalues, meaning the optimization landscape is maximally balanced — no eigenvalue mode has a disproportionately large gradient relative to its weight magnitude.

**GPU efficiency:** The reparameterization is a cheap elementwise function applied to the $m$ eigenvalue parameters per layer (typically $m = 64$–$256$). It adds negligible compute. The benefit is indirect: stable reparameterization enables training with **larger learning rates** and avoids NaN divergence that would otherwise require expensive restarts. The "best" reparameterization consistently trains stably at learning rates where exponential and softplus reparameterizations diverge to NaN.

## Mathematical Form

**SSM continuous-time dynamics (diagonal case):**

$$
\frac{dh_t}{dt} = \Lambda h_t + U x_t + b
$$

where $\Lambda = \text{Diag}(\lambda_1, \ldots, \lambda_m)$ with $\lambda_i < 0$ (stability requirement), $U \in \mathbb{R}^{m \times d}$, $b \in \mathbb{R}^m$.

**Discrete-time recurrence:**

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = c^\top \sigma(h_t)
$$

where $\bar{A} = e^{\Lambda \Delta t}$ gives discrete eigenvalues $\bar{\lambda}_i = e^{\lambda_i \Delta t} \in (0, 1)$ for $\lambda_i < 0$.

---

**Curse of Memory (Theorem 3.3):**

For any bounded, causal, continuous, regular, time-homogeneous functional sequence $\mathbf{H}$ approximated by SSMs $\{\hat{\mathbf{H}}(\cdot, \hat{\theta}_m)\}_{m=1}^{\infty}$ that $\beta_0$-stably approximate $\mathbf{H}$, the memory function decays exponentially:

$$
\mathcal{M}(\mathbf{H})(t) \leq (d+1) L_0 \theta_{\max}^2 e^{-\beta t}, \quad t \geq 0, \; \beta < \beta_0
$$

This means: without reparameterization, SSMs can only stably learn targets whose influence from past inputs decays exponentially. Targets with polynomial decay (e.g., $\mathcal{M}(t) \sim t^{-\alpha}$) cannot be stably approximated.

---

**Stable reparameterization (Definition 3.4):**

A function $f : \mathbb{R} \to \mathbb{R}$ mapping trainable weight $w$ to eigenvalue $\lambda = f(w)$ is **stable** if there exists a continuous function $g : [0, \infty) \to [0, \infty)$ with $g(0) = 0$ such that:

$$
\sup_w \left[ |f(w)| \sup_{|\tilde{w} - w| \leq \beta} \int_0^\infty \left| e^{f(\tilde{w})t} - e^{f(w)t} \right| dt \right] \leq g(\beta)
$$

The key property: the perturbation error is bounded uniformly over all weights $w$. Without reparameterization ($f(w) = w$), this integral diverges as $w \to 0^-$.

**Theorem 3.5 (Stable approximation universality):** With stable reparameterization, SSMs can stably approximate **any** bounded, causal, continuous, regular, time-homogeneous linear functional — including those with polynomial memory decay.

---

**Gradient norm bound (Theorem 3.6):**

For eigenvalue parameterized via $f(w) = \lambda$, the gradient norm is bounded by:

$$
G_f(w) := \left|\frac{\partial \text{Loss}}{\partial w}\right| \leq C_{\mathbf{H}, \hat{\mathbf{H}}_m} \cdot \frac{|f'(w)|}{f(w)^2}
$$

where $C_{\mathbf{H}, \hat{\mathbf{H}}_m}$ is a constant independent of $f$. The **gradient scale function** $|f'(w)| / f(w)^2$ determines how the gradient scales across different eigenvalues.

**Discrete-time version:**

$$
G_f^D(w) \leq C_{\mathbf{H}, \hat{\mathbf{H}}_m} \cdot \frac{|f'(w)|}{(1 - f(w))^2}
$$

---

**Optimal ("best") reparameterization:**

Minimizing the worst-case gradient-over-weight ratio $|f'(w)| / f(w)^2 = L |w|$ (Lipschitz condition) leads to:

$$
\frac{f'(w)}{f(w)^2} = 2aw \implies \frac{1}{f(w)} = -(aw^2 + b) \implies f^*(w) = -\frac{1}{aw^2 + b}
$$

with $a > 0, b \geq 0$. The paper uses $a = 1, b = 0.5$, giving:

$$
f^*(w) = -\frac{1}{w^2 + 0.5}
$$

**Discrete-time version:** $f^*(w) = 1 - \frac{1}{w^2 + 0.5}$, ensuring $\lambda \in (-1, 1)$ with $\lim_{w \to 0} f^*(w) = -1$ (not crossing the stability boundary $\lambda = \pm 1$).

---

**Comparison of gradient scale functions:**

| Reparameterization | $f(w)$ | $\frac{\|f'(w)\|}{f(w)^2}$ | Behavior near boundary |
|---|---|---|---|
| Direct (none) | $w$ | $\frac{1}{w^2}$ | $\to \infty$ as $w \to 0$ |
| Exponential | $-e^w$ | $\frac{1}{e^w}$ | Bounded but varies exponentially |
| Softplus | $-\log(1+e^w)$ | $\frac{e^w}{(1+e^w)\log^2(1+e^w)}$ | Bounded but uneven |
| **Best** | $-\frac{1}{w^2+0.5}$ | $\frac{2|w|}{(w^2+0.5)^{-2}} \cdot (w^2+0.5)^2 = 2|w|$ | Linear in $|w|$ — maximally balanced |

## Complexity

| Operation | Without reparam | With "best" reparam |
|-----------|----------------|---------------------|
| Eigenvalue computation | $O(m)$ per layer | $O(m)$ per layer (elementwise) |
| Gradient computation | $O(m)$ per layer | $O(m)$ per layer (chain rule) |
| Training stability (MNIST) | NaN at LR $\geq 5 \times 10^{-4}$ | Stable up to LR $= 5 \times 10^{0}$ |
| Training stability (CIFAR-10) | NaN at all LRs | Stable up to LR $= 5 \times 10^{0}$ |

**Memory:** Zero additional memory. The reparameterization replaces the existing eigenvalue parameterization — the number of trainable parameters is unchanged ($m$ per layer).

**FLOPs:** Negligible — $m$ elementwise operations per layer (division + addition). For $m = 256$ and $d = 1024$, this is $< 0.001\%$ of a single layer's FLOP budget.

**Key quantitative results (Table 2, MNIST):**

| LR | Direct | Softplus | Exp | Best |
|----|--------|----------|-----|------|
| $5 \times 10^{-4}$ | 0.094 | 0.094 | 0.093 | **0.092** |
| $5 \times 10^{-3}$ | NaN | 0.024 | 0.024 | **0.023** |
| $5 \times 10^{-2}$ | NaN | 0.803 | 0.868 | **0.089** |
| $5 \times 10^{-1}$ | NaN | 2.314 | 2.314 | **2.186** |
| $5 \times 10^{0}$ | NaN | NaN | NaN | **199.0** (not NaN) |

**Long Range Arena (Table 4):**

| | Listops | Text | Retrieval | Image | Pathfinder | PathX | Avg |
|---|---|---|---|---|---|---|---|
| Exp (S4) | 59.60 | 86.82 | 90.90 | **88.65** | 94.2 | **96.35** | 86.09 |
| **Best** | **60.80** | **88.5** | **91.3** | 87.39 | **94.8** | 96.1 | **86.48** |

## Applicability

- **All diagonal SSMs:** S4D, S5, LRU, DSS, and any model parameterizing a diagonal recurrence matrix $\Lambda$. The reparameterization is a drop-in replacement for the eigenvalue mapping function.

- **Mamba and selective SSMs:** Mamba uses input-dependent discretization of eigenvalues. The "best" reparameterization can replace the softplus used for the discretization step $\Delta = \text{softplus}(\text{param})$ to achieve more balanced gradients.

- **Linear attention with decay:** Models like RetNet, GLA, and RWKV that use decay-gated recurrences benefit from stable reparameterization of their decay parameters.

- **Any recurrent model with eigenvalue constraints:** LSTM/GRU-style models with learnable forget gates could use this reparameterization to achieve more balanced gradient flow across gates with different time constants.

- **Large-scale SSM training:** The stability benefit is most pronounced at scale, where aggressive learning rates and long training runs amplify the gradient imbalance between fast-decaying and slow-decaying modes.

## Limitations

- **Theoretical optimality is for gradient boundedness, not loss landscape:** The "best" reparameterization minimizes gradient-over-weight ratio, but this doesn't guarantee faster convergence or lower final loss — it guarantees more stable optimization. In practice, the improvement over exp/softplus is modest at low learning rates.

- **Only addresses eigenvalue parameterization:** The gradient imbalance from input/output matrices ($U, c$) is not addressed by this reparameterization. The paper focuses on the diagonal recurrence weights.

- **The "best" function has a singularity at $w = 0$:** $f^*(0) = -1/(0 + 0.5) = -2$, which is fine, but the gradient scale $2|w|$ vanishes at $w = 0$, meaning modes initialized exactly at $w = 0$ have zero gradient. In practice, random initialization avoids this.

- **Validated on relatively small models:** Experiments use the Hyena architecture (∼ 153M params) on WikiText-103, plus smaller models on MNIST, CIFAR-10, and LRA. The theory predicts benefits at scale, but large-scale SSM training validation (e.g., Mamba at 1B+) is not provided.

- **Not applicable to dense (non-diagonal) recurrence matrices:** The analysis assumes diagonal $\Lambda$. Models with full recurrence matrices (e.g., some variants of linear RNNs) would need a different reparameterization strategy.

- **Negligible wall-clock impact of the reparameterization itself:** The benefit is entirely indirect — enabling higher learning rates and avoiding divergence. The "best" reparameterization adds no measurable throughput overhead.

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StableSSMReparameterization:
    """
    Reparameterization functions mapping trainable weights w
    to stable eigenvalues λ = f(w) with Re(λ) < 0.

    The 'best' reparameterization achieves the most balanced
    gradient scale across all eigenvalue modes, enabling
    training at higher learning rates without NaN divergence.
    """

    @staticmethod
    def exponential(w):
        """S4/S4D-style: λ = -exp(w). Gradient scale: 1/exp(w)."""
        return -torch.exp(w)

    @staticmethod
    def softplus(w):
        """S5-style: λ = -log(1 + exp(w)). Gradient scale: bounded but uneven."""
        return -F.softplus(w)

    @staticmethod
    def best(w, a=1.0, b=0.5):
        """
        Optimal reparameterization: λ = -1/(a*w² + b).

        Gradient scale: 2*a*|w| — LINEAR in |w|, maximally balanced.
        This is the unique solution (up to a, b) that minimizes the
        worst-case gradient-over-weight ratio.

        Properties:
        - λ ∈ (-1/b, 0) for all w ∈ ℝ (automatically stable)
        - Gradient ∂L/∂w ≤ C * 2a|w| (Lipschitz in w)
        - No eigenvalue crosses stability boundary
        - Smallest max(|gradient|/|weight|) across all modes

        Parameters:
        - a=1, b=0.5: default from paper
          → λ ranges from -2 (at w=0) to 0⁻ (as |w|→∞)
          → Discrete: λ_discrete = 1 - 1/(w²+0.5), range (-1, 1)
        """
        return -1.0 / (a * w**2 + b)

    @staticmethod
    def best_discrete(w, a=1.0, b=0.5):
        """
        Discrete-time version: λ = 1 - 1/(a*w² + b).
        Range: (-1, 1) for all w, with λ(0) = 1 - 1/b = -1 (for b=0.5).
        """
        return 1.0 - 1.0 / (a * w**2 + b)


class DiagonalSSMLayer(nn.Module):
    """
    Diagonal SSM layer with configurable reparameterization.

    Usage:
        layer = DiagonalSSMLayer(d_input=256, d_state=64, reparam='best')
    """
    def __init__(self, d_input, d_state, reparam='best', dt=0.01):
        super().__init__()
        self.d_state = d_state
        self.dt = dt
        self.reparam = reparam

        # Trainable eigenvalue parameters (real-valued)
        self.w = nn.Parameter(torch.randn(d_state) * 0.5)

        # Input/output projections
        self.B = nn.Parameter(torch.randn(d_state, d_input) * (2 / d_input)**0.5)
        self.C = nn.Parameter(torch.randn(d_input, d_state) * (2 / d_state)**0.5)

    def get_eigenvalues(self):
        """Map trainable w to stable eigenvalues via reparameterization."""
        if self.reparam == 'best':
            return StableSSMReparameterization.best(self.w)
        elif self.reparam == 'exp':
            return StableSSMReparameterization.exponential(self.w)
        elif self.reparam == 'softplus':
            return StableSSMReparameterization.softplus(self.w)
        else:
            return self.w  # Direct (unstable at high LR)

    def forward(self, x):
        """
        x: (B, T, d_input) input sequence
        Returns: (B, T, d_input) output sequence
        """
        B_sz, T, D = x.shape
        lambd = self.get_eigenvalues()  # (d_state,)

        # Discretize: A_bar = exp(λ * dt)
        A_bar = torch.exp(lambd * self.dt)  # (d_state,)

        # Recurrence (or use parallel scan for efficiency)
        h = torch.zeros(B_sz, self.d_state, device=x.device)
        outputs = []
        for t in range(T):
            h = A_bar * h + x[:, t] @ self.B.T  # (B, d_state)
            y = h @ self.C.T  # (B, d_input)
            outputs.append(y)

        return torch.stack(outputs, dim=1)

# GPU efficiency analysis:
# - Reparameterization: m elementwise ops (division + addition)
#   For m=256: 256 FLOPs per forward pass (negligible)
# - No extra memory: replaces existing eigenvalue computation
# - No new kernel launches: fused into eigenvalue computation
# - Coalesced memory access: eigenvalues are a contiguous vector
# - Key benefit: enables 10-1000x higher learning rates
#   without NaN, avoiding expensive training restarts
#
# Gradient comparison at different learning rates:
# Direct:   NaN at LR ≥ 5e-4 (gradient ∝ 1/w² → explodes)
# Exp:      NaN at LR ≥ 5e+0 (gradient varies exponentially)
# Softplus: NaN at LR ≥ 5e+0 (gradient bounded but uneven)
# Best:     Stable at LR = 5e+0 (gradient ∝ |w| — balanced)
```

## References

- Wang, S. & Li, Q. (2024). StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization. ICML 2024 (PMLR 235). arXiv:2311.14495.
- Gu, A., Goel, K., & Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. (S4 — uses exponential reparameterization)
- Smith, J.T.H. et al. (2023). Simplified State Space Layers for Sequence Modeling. ICML 2023. (S5 — uses softplus reparameterization)
- Gu, A., Gupta, A., & Re, C. (2022). On the Parameterization and Initialization of Diagonal State Space Models. NeurIPS 2022. (S4D — diagonal SSMs with exp reparameterization)
- Orvieto, A. et al. (2023). Resurrecting Recurrent Neural Networks for Long Sequences. ICML 2023. (LRU — uses exponential parameterization with magnitude constraint)
- Wang, S. & Xue, Q. (2023). On the Curse of Memory in Recurrent Neural Networks. (Curse of memory theory for nonlinear RNNs)
- Li, Z. et al. (2022). Approximation and Optimization Theory for Linear Continuous-Time Recurrent Neural Networks. JMLR 2022. (Theoretical foundations for memory/approximation in linear RNNs)

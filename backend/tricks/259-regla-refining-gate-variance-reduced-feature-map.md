# 259: ReGLA — Refining Gate and Variance-Reduced Exponential Feature Map for Gated Linear Attention

**Category**: stability
**Gain type**: expressivity
**Source**: Lu, Kobyzev, Rezagholizadeh, Chen & Langlais (2025) — ReGLA: Refining Gated Linear Attention. NAACL 2025. arXiv:2502.01578.
**Paper**: [papers/regla-refining-gated-linear-attention.pdf]
**Documented**: 2026-02-16

## Description

Gated Linear Attention (GLA) uses a sigmoid forget gate $g_t = \sigma(W_g x_t)$ to control memory decay. A critical but under-explored problem is **gate saturation**: when $g_t$ approaches 0 or 1, the gradient $\nabla g_t = g_t \odot (1 - g_t)$ vanishes, making it impossible for the model to adjust the gating behavior. This is especially problematic during training because:

1. Random initialization can push many gate activations into saturated regions
2. Once saturated, the gate cannot escape — creating a **dead gate** problem analogous to dead ReLU neurons
3. The model loses the ability to fine-tune its memory management, degrading performance

ReGLA introduces two synergistic tricks to address this:

**Trick 1 — Refining Gate:** Add a second sigmoid gate $r_t = \sigma(W_r x_t + b_r)$ that **interpolates between the squared gate and its complement**, creating a refined forget gate:

$$
F_t = \left((1 - r_t) \odot g_t^2 + r_t \odot (1 - (1-g_t)^2)\right) \mathbf{1}^\top
$$

This reparameterization has a crucial property: **the gradient never vanishes at the boundaries**. When $g_t \to 0$, $\nabla F_t \approx 2r_t g_t$ (non-zero for $r_t > 0$); when $g_t \to 1$, $\nabla F_t \approx 2r_t(1-g_t)$ (non-zero). The refining gate $r_t$ controls how much gradient flows near saturation, with larger $r_t$ providing stronger gradient at the boundaries.

**Trick 2 — Normalized Exponential Feature Map with Variance Reduction:** Replace commonly-used feature maps (ReLU, ELU+1, FAVOR+) with a **bounded, non-negative exponential** map:

$$
\phi_q(x)_{i,l} = \exp\!\left((W_q x)_{i,l} - \max_j (W_q x)_{j,l}\right)
$$

$$
\phi_k(x)_{i,l} = \exp\!\left((W_k x)_{i,l} - \max_{j,s} (W_k x)_{j,s}\right)
$$

The max-subtraction ensures $\phi \in (0, 1]$, guaranteeing both **boundedness** (no explosion) and **non-negativity** (no sign cancellation in the recurrent state). Additionally, a **variance reduction scaling factor** $\frac{1}{e\sqrt{d(e^2-1)}}$ is applied to the inner product to stabilize variance to $O(1)$ regardless of feature dimension:

$$
\text{Var}[\phi_q(x)^\top \phi_k(x)] \approx d \cdot (e^2 - 1) \cdot e^2
$$

vs. $\text{Var}[x^\top y] = d$ for identity features. The scaling factor corrects this $e^2(e^2-1)$-fold variance inflation.

## Mathematical Form

**GLA Recurrence with Refining Gate:**

$$
S_t = F_t \odot S_{t-1} + v_t \, \phi(k_t)^\top
$$

where $F_t \in \mathbb{R}^{d \times d}$ is the gating matrix (outer-product form as in GLA):

$$
F_t = \left((1 - r_t) \odot g_t^2 + r_t \odot \left(1 - (1 - g_t)^2\right)\right) \mathbf{1}^\top
$$

**Refining Gate Components:**

$$
g_t = \sigma(W_g x_t + b_g) \in (0, 1)^d
$$

$$
r_t = \sigma(W_r x_t + b_r) \in (0, 1)^d
$$

**Properties of the Refined Gate $F_t$:**

The effective forget rate for dimension $i$ is:

$$
F_{t,i} = (1 - r_{t,i}) \cdot g_{t,i}^2 + r_{t,i} \cdot (1 - (1 - g_{t,i})^2)
$$

Note that:
- $F_{t,i}$ ranges from $g_{t,i}^2$ (when $r_{t,i} = 0$) to $1 - (1 - g_{t,i})^2$ (when $r_{t,i} = 1$)
- Both bounds are in $[0, 1]$, so $F_{t,i} \in [0, 1]$ always holds
- The lower bound $g_t^2$ and upper bound $1 - (1-g_t)^2 = g_t(2 - g_t)$ create an **effective activation range** around the saturation region

**Gradient Analysis:**

$$
\frac{\partial F_t}{\partial g_t} = 2(1 - r_t) \cdot g_t + 2r_t \cdot (1 - g_t)
$$

At $g_t = 0$: $\frac{\partial F}{\partial g} = 2r_t$ (non-zero when $r_t > 0$)

At $g_t = 1$: $\frac{\partial F}{\partial g} = 2(1 - r_t)$ (non-zero when $r_t < 1$)

Compare to vanilla sigmoid: $\frac{\partial \sigma}{\partial x} = \sigma(1 - \sigma) = 0$ at both boundaries.

**Normalized Exponential Feature Map:**

$$
\phi_q(x)_{i,l} = \exp\!\left((W_q x)_{i,l} - \max_{1 \leq j \leq d} (W_q x)_{j,l}\right)
$$

$$
\phi_k(x)_{i,l} = \exp\!\left((W_k x)_{i,l} - \max_{\substack{1 \leq j \leq d \\ 1 \leq s \leq L}} (W_k x)_{j,s}\right)
$$

**Variance Reduction Factor (Theorem 3.1 in the paper):**

For $x_i, y_i \sim \mathcal{N}(0, 1)$:
- Identity inner product: $\text{Var}[\sum x_i y_i] = d$
- Exponential inner product: $\text{Var}[\sum \exp(x_i) \exp(y_i)] = d \cdot e^2(e^2 - 1)$

The scaling factor $\frac{1}{e\sqrt{d(e^2 - 1)}}$ normalizes the variance to $O(1)$.

**Key Definitions:**

- $g_t \in (0, 1)^d$ — base sigmoid forget gate
- $r_t \in (0, 1)^d$ — refining gate (controls gradient flow at saturation)
- $F_t \in [0, 1]^{d \times d}$ — refined gating matrix (outer-product form)
- $\phi_q, \phi_k: \mathbb{R}^d \to \mathbb{R}^d_+$ — normalized exponential feature maps
- $W_g, W_r \in \mathbb{R}^{d \times d}$ — gate projection matrices
- $W_q, W_k \in \mathbb{R}^{d \times d}$ — feature map projection matrices

## Complexity

| Component | GLA (baseline) | ReGLA |
|-----------|---------------|-------|
| Gate computation | $O(d^2)$ — one projection | $O(2d^2)$ — two projections ($W_g, W_r$) |
| Gate activation | $O(d)$ — sigmoid | $O(d)$ — 2 sigmoids + arithmetic |
| Feature map | $O(d)$ — ReLU/ELU | $O(d)$ — exp + max |
| Total per-step overhead | baseline | $+O(d^2)$ for $W_r$ projection |
| Extra parameters | 0 | $d^2 + d$ (one linear layer) |

**Memory:** $+O(d^2)$ parameters for $W_r$. No extra activation memory since $r_t$ can be computed in-place.

**The overhead is small:** The extra $W_r$ projection is a single $d \times d$ matmul, which is $<5\%$ of the total per-layer compute (dominated by QKV projections at $3d^2$ and the output projection at $d^2$).

**Perplexity results (WikiText-103, 160M params, Table 4):**

| Method | Feature Map | Gate | PPL |
|--------|------------|------|-----|
| LA w/ ReLU | ReLU | none | 28.5 |
| LA w/ ELU | ELU+1 | none | 31.3 |
| HedgeHog | learned exp | none | 22.4 |
| LA w/ Fast Decay | identity | $G_t = g_z g_f^\top$ | 20.8 |
| **ReGLA** | **norm exp** | **refining** | **19.0** |
| **Hybrid ReGLA** | **norm exp** | **refining + softmax layers** | **17.8** |
| Transformer (softmax) | — | — | 18.5 |

ReGLA closes the gap to softmax attention (19.0 vs 18.5), and the hybrid variant surpasses it (17.8).

## Applicability

- **GLA and variants (primary):** ReGLA directly extends GLA's gating mechanism. Since it preserves the outer-product gate structure $F_t = f_t \mathbf{1}^\top$, it is compatible with GLA's chunkwise-parallel algorithm (trick 177) and secondary chunking. The refining gate adds only elementwise operations to the gate computation, not affecting the matmul structure.

- **Post-linearization:** ReGLA can replace softmax attention in pre-trained transformers via continual pre-training. The paper demonstrates this on Pythia-160M, achieving competitive performance after only 50K steps of continual training.

- **Linear attention with any gating scheme:** The refining gate idea applies to any sigmoid-gated linear recurrence: HGRN, HGRN2, RWKV, RetNet. Any model suffering from gate saturation can benefit.

- **Feature map choice for chunkwise training:** The normalized exponential feature map $\phi(x) = \exp(x - \max(x))$ is compatible with GLA's secondary chunking (trick 177) because it maintains bounded, non-negative features. The max-subtraction is a local operation that doesn't break the chunkwise-parallel structure.

- **Mamba/SSM extension:** While Mamba uses a different parameterization (data-dependent $\Delta_t$ via softplus), the gate saturation problem exists there too. The refining gate concept could be adapted to SSM discretization parameters.

## Limitations

- **Extra parameters:** The refining gate adds $d^2 + d$ parameters per layer (the $W_r$ projection and bias). At scale (e.g., $d = 4096$), this is $\sim$16M parameters per layer, or about 1% overhead for a 7B model.

- **Only validated at small scale:** The paper trains models up to 160M parameters on WikiText-103 and SlimPajama (8M tokens). Scaling behavior to 1B+ parameters and 100B+ tokens is unknown.

- **Interaction with log-space computation unclear:** GLA's secondary chunking (trick 177) computes gating in log-space for numerical stability. The refined gate $F_t = (1-r_t)g_t^2 + r_t(1-(1-g_t)^2)$ involves squared terms that may complicate log-space conversion. The standard approach would be to compute $\log F_t$ directly, but this requires $\log$ of a sum which doesn't simplify as cleanly as $\log \sigma(x)$.

- **Gate distribution shift:** The paper observes that after training, ReGLA's gate activations concentrate around values significantly different from 1.0, suggesting a propensity to favor local (recent) information. Whether this bias is beneficial or limiting at larger scale is unclear.

- **No custom kernel:** The paper uses the FLA library's existing kernels. A fused kernel incorporating the refining gate computation could further improve throughput by avoiding an extra elementwise pass.

- **Feature map is not matmul-friendly on its own:** The $\exp(x - \max(x))$ operation is elementwise, not a matmul. However, it produces features that are then used in standard matmuls ($\phi(Q)\phi(K)^\top$), so it doesn't break tensor core utilization of the subsequent attention computation.

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReGLA(nn.Module):
    """
    ReGLA: Refining Gated Linear Attention
    Combines refined gating + normalized exponential feature map.
    """
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d = d_model
        self.n_heads = n_heads

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Gate projections
        self.W_g = nn.Linear(d_model, d_model)  # base forget gate
        self.W_r = nn.Linear(d_model, d_model)  # refining gate

        # Stable normalization layer
        self.norm = nn.LayerNorm(d_model)

    def safe_exp_feature_map(self, x, dim=-1):
        """
        Normalized exponential feature map: exp(x - max(x))
        Guarantees: bounded [0, 1], non-negative, stable.
        """
        return torch.exp(x - x.max(dim=dim, keepdim=True).values)

    def variance_reduction_scale(self, d):
        """
        Scaling factor to normalize variance of exp inner products.
        Var[sum exp(x_i)exp(y_i)] = d * e^2 * (e^2 - 1)
        We want Var ≈ 1, so divide by e * sqrt(d * (e^2 - 1))
        """
        import math
        e = math.e
        return 1.0 / (e * math.sqrt(d * (e**2 - 1)))

    def refined_gate(self, x):
        """
        Refining gate mechanism.
        F = (1 - r) * g^2 + r * (1 - (1-g)^2)
        = (1 - r) * g^2 + r * g * (2 - g)

        Key property: gradient never vanishes at g=0 or g=1.
        dF/dg = 2(1-r)*g + 2r*(1-g)
        At g=0: dF/dg = 2r  (non-zero!)
        At g=1: dF/dg = 2(1-r)  (non-zero!)
        """
        g = torch.sigmoid(self.W_g(x))  # base gate (B, T, D)
        r = torch.sigmoid(self.W_r(x))  # refining gate (B, T, D)

        # Refined forget gate
        F = (1 - r) * g**2 + r * (1 - (1 - g)**2)
        # Equivalently: F = (1 - r) * g**2 + r * g * (2 - g)
        return F  # (B, T, D), all values in [0, 1]

    def forward(self, x):
        """
        Args:
            x: (B, T, D) — input sequence
        Returns:
            y: (B, T, D) — output sequence
        """
        B, T, D = x.shape

        # Feature maps (bounded, non-negative)
        q = self.safe_exp_feature_map(self.W_q(x), dim=-1)  # (B, T, D)
        k = self.safe_exp_feature_map(self.W_k(x), dim=-1)  # (B, T, D)
        v = self.W_v(x)  # (B, T, D)

        # Variance reduction scaling
        scale = self.variance_reduction_scale(D)

        # Refined forget gate
        F = self.refined_gate(x)  # (B, T, D), in [0, 1]

        # Recurrent computation (can be parallelized via chunkwise form)
        S = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            # Gate the state (outer-product gate as in GLA)
            S = F[:, t, :].unsqueeze(-1) * S  # (B, D, D) * broadcast

            # Add new association
            S = S + torch.einsum('bd,be->bde', v[:, t], k[:, t])

            # Query the state
            h = torch.einsum('bde,be->bd', S, q[:, t]) * scale

            outputs.append(h)

        y = torch.stack(outputs, dim=1)  # (B, T, D)

        # Stable normalization + output projection
        y = self.norm(y)
        y = self.W_o(y)

        return y


# KEY INSIGHTS:
#
# 1. REFINING GATE solves gate saturation:
#    - Vanilla sigmoid: gradient = g*(1-g) → 0 at boundaries
#    - Refined gate: gradient = 2(1-r)*g + 2r*(1-g) → 2r or 2(1-r) at boundaries
#    - The refining gate r controls the gradient magnitude at saturation
#    - Cost: one extra d×d linear projection (< 5% overhead)
#
# 2. NORMALIZED EXP FEATURE MAP:
#    - exp(x - max(x)) ∈ (0, 1] — bounded AND non-negative
#    - Both properties required for stable linear attention:
#      * Bounded: prevents inner product explosion
#      * Non-negative: prevents sign cancellation in recurrent state
#    - Only exp(x - max(x)) satisfies BOTH (Table 1 in paper)
#    - ReLU: unbounded, non-negative ✗✓
#    - cos/sin: bounded, can be negative ✓✗
#    - FAVOR+: bounded, non-negative, but requires resampling ✓✓*
#
# 3. VARIANCE REDUCTION:
#    - Var[exp(x)·exp(y)] = e²(e²-1) per dimension (Theorem 3.1)
#    - vs. Var[x·y] = 1 for identity
#    - Factor of ~40× variance inflation!
#    - Scale by 1/(e·√(d(e²-1))) to normalize
#
# 4. COMPATIBILITY with chunkwise GLA (trick 177):
#    - The refined gate F_t ∈ [0,1] has the same structure as GLA's gate
#    - Secondary chunking applies: inter-sub-chunk matmuls use TC,
#      intra-sub-chunk uses log-space full precision
#    - The only change is computing log(F_t) = log((1-r)g² + r(1-(1-g)²))
#      instead of log(σ(x)), which is slightly more expensive but still O(d)
```

## References

- Lu, P., Kobyzev, I., Rezagholizadeh, M., Chen, B., & Langlais, P. (2025). ReGLA: Refining Gated Linear Attention. NAACL 2025. arXiv:2502.01578.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Mao, H. H. (2022). Fine-Tuning Pre-Trained Transformers into Decaying Fast Weights. EMNLP 2022.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Choromanski, K. M., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Gu, A., Goel, K., & Ré, C. (2020). Improving the Gating Mechanism of Recurrent Neural Networks. ICML 2020.
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.

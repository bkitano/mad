# 235: Unit Scaling for Low-Precision Training

**Category**: stability
**Gain type**: efficiency
**Source**: Blake, Orr & Luschi (2023) — "Unit Scaling: Out-of-the-Box Low-Precision Training" (ICML 2023)
**Paper**: [papers/unit-scaling-low-precision-training.pdf]
**Documented**: 2026-02-15

## Description

Unit scaling is a model design paradigm that enables out-of-the-box training in FP16 or FP8 **without loss scaling**. The core idea is to insert fixed, analytically-derived scaling factors into every operation in the forward and backward passes so that all activations, weights, and gradients have approximately **unit variance** ($\sigma \approx 1$) at initialization. This places all tensor values near the center of the representable range of low-precision formats, maximizing signal-to-noise ratio and avoiding both overflow and underflow without any dynamic rescaling mechanism.

Unlike loss scaling (which applies a single global scale to all gradients) or per-tensor automatic scaling (which requires runtime statistics collection and extra memory), unit scaling uses **compile-time-constant** scaling factors derived from operation dimensions. This means zero runtime overhead when fused into preceding ops via `torch.compile` or `jax.jit`.

## Mathematical Form

**Core Principle:** For a model represented as a computational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, replace every operation $f$ with a *scaled op* $f^*$ having scaling factors $\alpha, \beta_1, \ldots, \beta_k \in \mathbb{R}^+$:

$$
f^*(x_1, \ldots, x_k) \triangleq \alpha \cdot f(x_1, \ldots, x_k)
$$

$$
f^*_{\text{grad}}(x_1, \ldots, x_k, g)_i \triangleq \beta_i \cdot f_{\text{grad}}(x_1, \ldots, x_k, g)_i, \quad \forall i \in [1..k]
$$

where $\alpha$ scales the forward output and $\beta_i$ scales the gradient with respect to input $i$.

**Selecting Scaling Factors:** Given unit-scaled inputs, derive the output scale $\sigma_Y$ and set:

$$
\alpha = \frac{1}{\sigma_Y}
$$

For each gradient path, compute $\sigma_{x'_i}$ and set:

$$
\beta_i = \frac{1}{\sigma_{x'_i}}
$$

**Key Scaling Factors (Table A.2):**

| Operation | Forward scale $\alpha$ | Backward scales $\beta$ |
|-----------|----------------------|------------------------|
| $\text{matmul}(X^{b \times m}, W^{m \times n})$ | $m^{-1/2}$ | $\beta_X = n^{-1/2}$, $\beta_W = b^{-1/2}$ |
| $\text{sum}(x) = \sum_{i=1}^n x_i$ | $n^{-1/2}$ | $\beta = 1$ |
| $\text{relu}(x) = \max(x, 0)$ | $\sqrt{2/(1 - 1/\pi)}$ | $\beta = \sqrt{2}$ |
| $\text{gelu}(x) = x \cdot \Phi(x)$ | $1.701$ | $\beta = 1.481$ |
| $\text{tanh}(x)$ | $1.593$ | $\beta = 1.467$ |
| $\text{sigmoid}(x)$ | $4.802$ | $\beta = 4.722$ |
| $\text{softmax}(x)_i$ | $s$ | $\beta = s$ |
| $\text{softmax\_xent}(x, t)$ | $1$ | $\beta = s / \sqrt{s-1}$ |
| $\text{layer\_norm}(X, w, c)$ | $1$ | $\beta_x = 1$, $\beta_w = 1$, $\beta_c = b^{-1/2}$ |

where $b$ = batch size, $m$ = reduction dimension, $n$ = output dimension, $s$ = softmax input size.

**Constraint Resolution:** When forward and backward scaling factors are constrained (non-cut-edges in the graph), resolve by taking the geometric mean:

$$
\alpha = \beta_1 = (\alpha_{\text{unconstrained}} \cdot \beta_{1,\text{unconstrained}})^{1/2}
$$

**Example — Scaled Matmul Projection:**

For $X \in \mathbb{R}^{b \times m}$, $W \in \mathbb{R}^{m \times n}$, incoming gradient $G \in \mathbb{R}^{b \times n}$:

$$
\text{matmul}^*(X, W) = \alpha \cdot X W
$$

$$
\text{matmul}^*_{\text{grad}}(X, W, G)_1 = \beta_1 \cdot G W^\top
$$

$$
\text{matmul}^*_{\text{grad}}(X, W, G)_2 = \beta_2 \cdot X^\top G
$$

With constrained $\alpha = \beta_1 = (m \cdot n)^{-1/4}$ and unconstrained $\beta_2 = b^{-1/2}$.

**Residual Weighted Addition:**

$$
x_{l+1} = \sqrt{1 - \tau} \cdot x_l + \sqrt{\tau} \cdot f(x_l)
$$

where $\tau$ controls the residual-to-skip ratio, ensuring unit variance is maintained through depth.

## Complexity

| Operation | Naive (with loss scaling) | With Unit Scaling |
|-----------|-------------------------|-------------------|
| Forward pass | $O(1)$ extra FLOPs + loss scale search | $O(1)$ extra FLOPs, no search |
| Backward pass | $O(1)$ extra FLOPs + overflow checking | $O(1)$ extra FLOPs, no checking |
| Hyperparameter tuning | Multiple runs to find loss scale | Zero — works out-of-the-box |
| Runtime overhead | Dynamic scale tracking + batch skipping | None (factors fused at compile time) |

**Memory:** Identical to baseline — scaling factors are constants, not stored tensors. Eliminates need for loss-scale state and gradient-overflow detection buffers.

**FLOPs:** <0.2% overhead (scalar multiplications fused into preceding ops).

## GPU Efficiency Analysis

**Memory Access Pattern:**
- Scaling factors are compile-time constants fused into preceding operations via `torch.compile` or `jax.jit`
- No additional memory loads or stores beyond baseline
- All operations remain coalesced — just scalar multiply before write

**Parallelism:**
- Zero sequential bottlenecks — each scaling factor is independent per operation
- Maps perfectly to tensor cores: matmul inputs are simply pre-scaled
- No warp divergence or load imbalance introduced

**Arithmetic Intensity:**
- Adding a scalar multiply to an existing op is essentially free (already compute-bound or memory-bound)
- The real gain: enables FP8 matmuls on H100 tensor cores → 2× theoretical throughput over BF16

**HBM Bandwidth Reduction:**
- FP8 activations/gradients: 2× reduction in memory footprint vs FP16, 4× vs FP32
- FP8 weights: 2× reduction in memory bandwidth for weight loads
- Enables larger batch sizes within the same GPU memory budget

## Applicability

- **Transformers**: Full support demonstrated on BERT (Base & Large) in FP16 and FP8 with no accuracy loss
- **RNNs and CNNs**: Validated across Conv, RNN, and Attention sequence layers in character language modeling
- **Any architecture**: The method is generic — any model can be unit-scaled by applying the recipe to its computational graph
- **Combined with u-μP**: The follow-up work (u-μP, arXiv 2407.17465) combines unit scaling with maximal update parameterization for width-independent hyperparameter transfer + low-precision training simultaneously
- **SSMs**: Applicable to SSM architectures where recurrence multiplications benefit from controlled dynamic range

## Limitations

- **Does not adapt during training**: Fixed scaling factors are set at initialization. If tensor distributions shift significantly during training (e.g., emergence of activation outliers at >1B parameters), unit scaling alone may not be sufficient
- **Requires model redesign**: Cannot be applied as a drop-in to existing models without modifying the architecture code (residual connections, initialization, activation functions all change)
- **Residual weighting hyperparameter**: The $\tau$ parameter for residual connections introduces a new design choice (though less sensitive than loss scale)
- **Outlier sensitivity**: At very large model scales (>6.7B parameters), activation outliers can emerge that challenge the unit-variance assumption, though FP8's logarithmic spacing handles outliers better than INT8
- **Not a substitute for mixed precision**: Still requires FP32 master weights — unit scaling addresses the dynamic range problem, not the precision problem

## Implementation Notes

```python
# Core scaled op abstraction
def scaled(X, alpha, beta):
    """Forward: multiply by alpha. Backward: multiply grad by beta."""
    # In forward: Y = X * alpha
    # In backward: grad_X = grad_Y * beta
    # Implemented via custom autograd function
    return ScaledOp.apply(X, alpha, beta)

def scaled_projection(X, W):
    """Unit-scaled matmul with constrained scaling."""
    b, m, n = X.shape[0], X.shape[1], W.shape[1]
    alpha = (m * n) ** (-1/4)       # constrained fwd scale
    beta_X = (m * n) ** (-1/4)      # constrained grad scale for X
    beta_W = b ** (-1/2)            # unconstrained grad scale for W
    X = scaled(X, beta=beta_X)      # scale on input edge
    W = scaled(W, beta=beta_W)      # scale on weight edge
    return scaled(torch.matmul(X, W), alpha=alpha)

class ScaledFFN(nn.Module):
    def __init__(self, d, h, tau):
        super().__init__()
        self.norm = ScaledLayerNorm(d)
        self.W1 = nn.Parameter(torch.randn(d, h))   # unit variance init
        self.W2 = nn.Parameter(torch.randn(h, d))   # unit variance init
        self.tau = tau  # residual weight

    def forward(self, X):
        a = (1 - self.tau) ** (1/2)
        b = self.tau ** (1/2)
        Z = self.norm(scaled(X, beta=b))
        Z = scaled_projection(Z, self.W1)
        Z = scaled_gelu(Z)
        Z = scaled_projection(Z, self.W2)
        return a * X + b * Z  # weighted residual add

# Key insight: all scaling factors are compile-time constants
# torch.compile / jax.jit will fuse γ·X into the preceding write,
# making the overhead negligible (measured <0.2% FLOPs increase)
```

## References

- Blake, C., Orr, D. & Luschi, C. (2023). "Unit Scaling: Out-of-the-Box Low-Precision Training." ICML 2023. arXiv:2303.11257.
- Blake, C., Sherrington, D., Sherrington, E. & Sherrington, R. (2024). "u-μP: The Unit-Scaled Maximal Update Parametrization." arXiv:2407.17465.
- Micikevicius, P. et al. (2018). "Mixed Precision Training." ICLR 2018. arXiv:1710.03740.
- Noune, B. et al. (2022). "8-bit numerical formats for deep neural networks." arXiv:2206.02915.
- Graphcore unit-scaling library: https://graphcore-research.github.io/unit-scaling

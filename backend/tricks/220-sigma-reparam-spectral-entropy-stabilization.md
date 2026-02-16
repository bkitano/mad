# 220: Sigma Reparameterization (σReparam)

**Category**: stability
**Gain type**: efficiency
**Source**: Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse" (ICML 2023)
**Paper**: papers/sigma-reparam-attention-entropy.pdf
**Documented**: 2026-02-15

## Description

σReparam is a weight reparameterization technique that prevents **attention entropy collapse** — a failure mode where attention scores become pathologically concentrated (near-zero entropy), causing training instability (oscillating loss or divergence). The key insight is that attention entropy has a tight lower bound that decreases *exponentially fast* with the spectral norm of the attention logit matrix $\boldsymbol{\sigma} = \sigma(W_K W_Q^\top) \cdot \|XX^\top\|_2$. By controlling spectral norms through reparameterization, entropy collapse is prevented without constraining model capacity.

The technique replaces every linear layer's weight $W$ with $\hat{W} = \frac{\gamma}{\sigma(W)} W$, where $\sigma(W)$ is the spectral norm (computed via one-step power iteration) and $\gamma$ is a learnable scalar initialized to 1. This decouples the spectral norm update rate from the weight matrix dimensionality, providing uniform control across layers of varying sizes.

**Why it matters for GPU pretraining**: σReparam eliminates the need for learning rate warmup, weight decay, cosine scheduling, and even LayerNorm in some configurations — reducing training time by up to 16% while matching or exceeding baseline accuracy. It enables stable training of 100L-100L deep encoder-decoder Transformers where baselines diverge.

## Mathematical Form

**Core Operation — Weight Reparameterization:**

$$
\hat{W} = \frac{\gamma}{\sigma(W)} W
$$

where $\sigma(W) = \|W\|_2$ is the spectral norm (largest singular value) of $W \in \mathbb{R}^{d \times c}$, and $\gamma \in \mathbb{R}$ is a learnable parameter initialized to 1.

**Key Definitions:**

- $W \in \mathbb{R}^{d \times c}$ — weight matrix of any linear layer
- $\sigma(W) \in \mathbb{R}$ — spectral norm, computed via power iteration
- $\gamma \in \mathbb{R}$ — learned scalar controlling effective spectral norm of $\hat{W}$
- $W_K, W_Q \in \mathbb{R}^{d \times n_a}$ — key and query projection matrices
- $X \in \mathbb{R}^{T \times d}$ — input sequence ($T$ tokens, $d$ dimensions)

**Attention Entropy Lower Bound (Theorem 3.1):**

Let $\sigma = \|W_K W_Q^\top\|_2$, $\sigma_x = \|XX^\top\|_2$, $\boldsymbol{\sigma} = \sigma \sigma_x$, and $\beta = \exp\!\left(-\boldsymbol{\sigma}\sqrt{\frac{T}{T-1}}\right)$. Then:

$$
\mathrm{Ent}(A_i) \geq \log\!\left(1 + (T-1)\beta\right) + \frac{\boldsymbol{\sigma}\sqrt{T(T-1)}\,\beta}{1 + (T-1)\beta}
$$

This bound is **tight**: the minimum attainable entropy behaves like $\Omega(T\boldsymbol{\sigma} e^{-\boldsymbol{\sigma}})$, hence decreasing exponentially fast with $\boldsymbol{\sigma}$.

**Spectral Norm via Power Iteration (one step per training step):**

$$
u \leftarrow Wv, \quad u \leftarrow \frac{u}{\|u\|}, \quad v \leftarrow W^\top u, \quad v \leftarrow \frac{v}{\|v\|}, \quad \sigma(W) \approx u^\top W v
$$

**Spectral Norm of Ideal Update (Proposition 3.2):**

For stochastic gradient $g = \mu + \epsilon$ with $\mathbb{E}[\epsilon] = 0$, $\mathbb{E}[\epsilon^2] = n^2$, the Adam ideal update $\Delta = \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}$ has spectral norm:

$$
\sigma(\Delta) \geq \sqrt{w}\sqrt{1 - \frac{1}{w^2}\sum_{i,j=1}^{w} \frac{n_{i,j}^2}{\mu_{i,j}^2 + n_{i,j}^2}}
$$

This shows the naive spectral norm of weight updates grows as $\sim\sqrt{w}$ (width), while σReparam's $\gamma$ provides dimensionality-independent control.

## Complexity

| Operation | Naive (with LN + warmup) | With σReparam |
|-----------|--------------------------|---------------|
| Forward pass | $O(Td^2)$ per layer | $O(Td^2)$ per layer (same) |
| Spectral norm (power iter) | N/A | $O(dc)$ per layer (2 matvecs) |
| Training overhead | — | ~0% wall-clock (ASR), ~22% (deep MT) |

**Memory:** Two auxiliary buffers $u \in \mathbb{R}^d$, $v \in \mathbb{R}^c$ per linear layer + one scalar $\gamma$. Negligible compared to weights.

**Wall-clock timing (Table 5 from paper):**

| Configuration | ASR (ms/step) | MT 8L-18L (ms/step) |
|---------------|---------------|----------------------|
| post-LN | 450 | 1700 |
| pre-LN | 450 | 1800 |
| σReparam | 450 | 2200 |
| σReparam + post-LN | 510 | 2300 |

For ASR: zero overhead. For deep MT: ~22-29% overhead due to FP32 precision required on attention + σReparam operands (rest stays mixed precision). Net training time can still decrease because warmup/scheduling epochs are eliminated.

## Applicability

- **All transformer architectures**: Vision Transformers (ViT-B/L/H), encoder-decoder (MT), decoder-only (LM), speech (ASR)
- **Self-supervised learning**: SimCLR + ViT stability improvements
- **Very deep models**: Enables stable training of 100L-100L post-LN encoder-decoders where DeepNorm fails
- **Simplified training recipes**: Removes need for LR warmup, weight decay, cosine schedule, LayerNorm, and adaptive optimizers (enables SGD with LARS)
- **Complementary to existing stabilizers**: Works alongside pre-LN, DeepNorm, QK-norm (trick 215)

## Limitations

- **FP32 requirement for attention + σReparam**: The spectral norm computation and attention logits must use FP32 precision for numerical stability; cannot fully run in FP16/BF16
- **Wall-clock overhead on deep models**: 22-29% overhead for deep MT models due to FP32 attention requirement (though this is offset by eliminating warmup epochs)
- **Not a panacea**: Best performance on SimCLR ViTs is σReparam + pre-LN, suggesting it complements rather than fully replaces normalization
- **Power iteration is sequential per layer**: One matvec pair per layer per step — low arithmetic intensity, memory-bound operation
- **Causal connection unclear**: The paper shows correlation between entropy collapse and instability, but acknowledges the causal direction is not definitively established

## Implementation Notes

```python
# σReparam in PyTorch-like pseudocode (from Appendix C)
# Parameters: W (d, c), gamma (1,)
# Buffers: u (d,), v (c,) — left/right singular vectors

# Initialization
u = randn(d); u = u / u.norm(dim=0)
v = randn(c); v = v / v.norm(dim=0)
gamma = ones(1)

# Training: one power iteration step per gradient update
if training:
    with torch.no_grad():
        u = W.mv(v)
        u = u / u.norm(dim=0)
        v = W.T.mv(u)
        v = v / v.norm(dim=0)

# Compute spectral norm and reparameterized weight
sigma = einsum('d,dc,c->', u, W, v)
W_hat = gamma / sigma * W  # effective spectral norm = gamma

# Key GPU consideration: power iteration is 2 matvecs (memory-bound)
# but only O(d+c) work per layer — negligible vs O(Td^2) forward pass
# Critical: use FP32 for sigma computation and attention logits
```

**Practical notes:**
- Apply to ALL linear layers (including patch embedding) for best results
- Initialize $\gamma = 1$; no special initialization needed for weights (trunc_normal(0.02) works)
- One power iteration step per gradient step is sufficient (more steps show no improvement)
- During inference: compute $\hat{W}$ once and freeze — zero inference overhead
- Code: https://github.com/apple/ml-sigma-reparam

## References

- Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., Susskind, J. "Stabilizing Transformer Training by Preventing Attention Entropy Collapse." ICML 2023. arXiv:2303.06296
- Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. "Spectral Normalization for Generative Adversarial Networks." ICLR 2018.
- Cohen, J.M. et al. "Adaptive gradient methods at the edge of stability." arXiv:2207.14484, 2022.
- Wang, H. et al. "DeepNet: Scaling Transformers to 1,000 Layers." arXiv:2203.00555, 2022.

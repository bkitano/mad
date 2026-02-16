# 241: Dynamic Tanh (DyT) — Normalization-Free Transformer Training

**Category**: stability
**Gain type**: efficiency
**Source**: Zhu, Chen, He, LeCun & Liu (2025) — "Transformers without Normalization", CVPR 2025. arXiv:2503.10622
**Paper**: [papers/dynamic-tanh-normalization-free.pdf]
**Documented**: 2026-02-15

## Description

Layer normalization (LN) and RMSNorm are ubiquitous in transformers, but they require computing **per-token statistics** (mean, variance) via reduction operations across the embedding dimension. This creates synchronization points, limits kernel fusion opportunities, and adds non-trivial overhead especially for long sequences and small batch sizes.

Dynamic Tanh (DyT) replaces normalization layers entirely with a simple **elementwise** operation: $\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$. The key insight is that in trained transformers, the input-output mapping of LN layers empirically resembles a **tanh-shaped S-curve** — it acts approximately linearly on the majority of values (which cluster near zero) while **squashing extreme outliers** nonlinearly. DyT captures this behavior directly without computing any activation statistics.

The learnable scalar $\alpha$ tracks $1/\text{std}$ of the input activations throughout training, effectively learning a global normalization scale. This eliminates the per-token reduction entirely — every element is processed independently, making DyT a purely **elementwise** operation with no cross-token or cross-channel dependencies.

**GPU efficiency:** DyT eliminates per-token reductions (mean/variance computation across $C$ channels), replacing them with a scalar multiply + tanh + elementwise affine. This is a strict improvement in arithmetic intensity: no reduction across the $C$-dimension, no synchronization barrier, and trivial fusion into any preceding or following kernel. However, the authors note that when LN is properly compiled/optimized (e.g., via torch.compile), DyT shows **no measurable wall-clock speedup** in their benchmarks. The primary benefit is architectural simplicity and removing the dependency on activation statistics, which can help with kernel fusion in custom CUDA implementations and avoids the need for online statistics computation in inference engines.

## Mathematical Form

**Standard Layer Normalization:**

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where $\mu = \frac{1}{C}\sum_k x_k$ and $\sigma^2 = \frac{1}{C}\sum_k (x_k - \mu)^2$ require **reductions over the channel dimension** per token.

---

**Dynamic Tanh (DyT):**

$$
\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta
$$

where:
- $\alpha \in \mathbb{R}$ — learnable scalar, initialized to $\alpha_0$ (default 0.5). Controls the "width" of the S-curve
- $\gamma \in \mathbb{R}^C$ — learnable per-channel scale vector, initialized to $\mathbf{1}$
- $\beta \in \mathbb{R}^C$ — learnable per-channel shift vector, initialized to $\mathbf{0}$
- $x \in \mathbb{R}^{B \times T \times C}$ — input tensor (batch × tokens × channels)

**No reduction operations.** Each element $x_{b,t,c}$ is processed independently:

$$
\text{DyT}(x)_{b,t,c} = \gamma_c \cdot \tanh(\alpha \cdot x_{b,t,c}) + \beta_c
$$

---

**Why it works — the squashing hypothesis:**

For most inputs ($\sim$99%), $|\alpha x| \ll 1$, so $\tanh(\alpha x) \approx \alpha x$ (linear regime). For extreme outliers, $\tanh$ saturates to $\pm 1$, squashing them. This mirrors what LN does: it linearly transforms each token's activations (via mean subtraction and variance division) while disproportionately compressing outlier channels.

The learned $\alpha$ tracks $1/\text{std}(x)$ of the input activations:

$$
\alpha \approx \frac{1}{\text{std}(x)} \quad \text{(empirically observed, see Figure 8 of paper)}
$$

This means $\alpha$ learns a **global** normalization scale, unlike LN which computes it per-token. The key tradeoff: DyT cannot adapt to per-token variance differences, but in practice this doesn't hurt performance.

---

**Initialization of $\alpha$ for LLMs:**

For LLMs, $\alpha_0$ should be tuned based on model width. The paper finds:

| Model Width | Attention $\alpha_0$ | Other (FFN) $\alpha_0$ |
|-------------|---------------------|----------------------|
| 1024 | 1.0 | 1.0 |
| 2048 | 1.0 | 0.5 |
| 4096 | 0.8 | 0.2 |
| 8192 | 0.2 | 0.05 |

Wider models require smaller $\alpha_0$, and attention blocks benefit from larger $\alpha_0$ than FFN blocks.

## Complexity

| Operation | LayerNorm/RMSNorm | DyT |
|-----------|-------------------|-----|
| Forward pass | $O(BTC)$ with reduction over $C$ | $O(BTC)$ elementwise only |
| Per-token statistics | $2T$ reductions (mean + var) | **None** |
| Learnable parameters | $2C$ ($\gamma, \beta$) | $2C + 1$ ($\gamma, \beta, \alpha$) |
| Cross-token dependency | None | None |
| Cross-channel dependency | **Yes** (reduction over $C$) | **None** |

**Memory:** Identical to LN — $O(BTC)$ for activations. Saves the $O(BT)$ buffer for per-token mean/variance (negligible).

**Wall-clock:** The paper acknowledges that with torch.compile or optimized LN kernels, **DyT does not offer measurable throughput gains** (Appendix C). The benefit is primarily in:
1. Removing a synchronization point (enables deeper kernel fusion in custom implementations)
2. Simpler implementation (no running statistics, no epsilon for numerical stability)
3. Better compatibility with quantization/low-precision pipelines (no variance computation in reduced precision)

## Applicability

- **Transformer LLMs (7B–70B):** Validated on LLaMA 7B/13B/34B/70B with 200B tokens training. Matches RMSNorm across all scales (Table 4 of paper). Identical zero-shot benchmark scores.

- **Vision Transformers:** ViT-B/L, ConvNeXt-B/L on ImageNet-1K. DyT slightly outperforms LN (+0.2–0.5% accuracy).

- **Diffusion models:** DiT-B/L/XL for image generation. Comparable or better FID scores.

- **Speech models:** wav2vec 2.0 Base/Large. Matches LN validation loss.

- **DNA sequence models:** HyenaDNA, Caduceus. Identical performance to LN.

- **SSMs and non-transformer architectures:** The paper validates on HyenaDNA (a long-convolution model) and Caduceus (bidirectional SSM), showing DyT works beyond standard transformers. The tanh squashing replaces LN in any architecture where normalization is used for activation stability.

## Limitations

- **No wall-clock speedup with optimized LN:** When using torch.compile or fused LN kernels, DyT provides no measurable throughput improvement. The benefit is architectural, not computational.

- **Struggles with Batch Normalization:** Preliminary experiments show DyT cannot replace BatchNorm in classic ResNets (Appendix D). It's designed specifically for per-token normalization (LN/RMSNorm), not per-channel batch statistics.

- **$\alpha_0$ tuning for LLMs:** While non-LLM models are insensitive to $\alpha_0$ (0.5 works broadly), LLMs require width-dependent tuning of $\alpha_0$ (Table 10). Wider models need smaller $\alpha_0$.

- **Larger models are more stability-sensitive:** Training stability with DyT follows the same pattern as LN — larger models and higher learning rates require care. DyT with $\alpha_0 = 0.5$ matches LN's stability profile (Figure 10).

- **No per-token adaptation:** LN adapts normalization to each token's statistics. DyT uses a single global $\alpha$, which means it cannot handle cases where different tokens have wildly different activation scales within the same layer. In practice this doesn't matter for transformers, but could be limiting for architectures with highly heterogeneous activations.

## Implementation Notes

```python
import torch
import torch.nn as nn

class DyT(nn.Module):
    """
    Dynamic Tanh — drop-in replacement for LayerNorm/RMSNorm.

    Key advantage: purely elementwise, no per-token reductions.
    Replaces: nn.LayerNorm(C) or RMSNorm(C)

    GPU notes:
    - No reduction across C dimension → no warp-level sync needed
    - tanh is a fast elementwise op on all GPUs
    - Trivially fuses into preceding/following linear layers
    - α is a single scalar → broadcast multiply, negligible cost
    """
    def __init__(self, C, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        # x: (B, T, C) — purely elementwise, no reductions
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


# Usage: drop-in replacement in any transformer block
# Before:
#   self.norm1 = nn.LayerNorm(d_model)
#   self.norm2 = nn.LayerNorm(d_model)
# After:
#   self.norm1 = DyT(d_model, init_alpha=0.5)
#   self.norm2 = DyT(d_model, init_alpha=0.5)

# For LLMs, use width-dependent alpha initialization:
def get_llm_alpha(width, block_type):
    """
    Recommended alpha_0 from Table 10/11 of the paper.
    Attention blocks use higher alpha than FFN blocks.
    """
    if block_type == "attention":
        if width <= 1024: return 1.0
        elif width <= 2048: return 1.0
        elif width <= 4096: return 0.8
        else: return 0.2
    else:  # "ffn" or "final"
        if width <= 1024: return 1.0
        elif width <= 2048: return 0.5
        elif width <= 4096: return 0.2
        else: return 0.05

# Also add a learnable scalar after embeddings for LLMs:
# embed_scale = nn.Parameter(torch.ones(1))
# x = embed_scale * embedding(tokens)
```

## References

- Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). Transformers without Normalization. CVPR 2025. arXiv:2503.10622.
- Ba, J.L., Kiros, J.R., & Hinton, G.E. (2016). Layer Normalization. arXiv:1607.06450.
- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.
- Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. ICML 2021.
- Ni, R. et al. (2024). Unlocking the Power of Normalization Layers. arXiv preprint.
- Klambauer, G. et al. (2017). Self-Normalizing Neural Networks. NeurIPS 2017.

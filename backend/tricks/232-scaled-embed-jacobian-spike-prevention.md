# 232: Scaled Embed — Jacobian Spectral Norm Bound for Loss Spike Prevention

**Category**: stability
**Gain type**: efficiency
**Source**: Takase, Kiyono, Kobayashi & Suzuki (2024) — "Spike No More: Stabilizing the Pre-training of Large Language Models", COLM 2025. arXiv:2312.16903
**Paper**: [papers/spike-no-more-scaled-embed.pdf]
**Documented**: 2026-02-15

## Description

Loss spikes during LLM pre-training — sudden, catastrophic jumps in training loss — waste enormous compute budgets and sometimes irrecoverably ruin training runs. This paper provides a **theoretical framework** explaining why spikes occur and identifies two concrete conditions that prevent them, leading to a trivially simple fix: multiply embeddings by $\sqrt{d}$.

The core insight is that loss spikes are caused by **sudden growth of gradient norms**, which in turn are controlled by the **spectral norms of the Jacobian matrices** of each Transformer sub-layer. For a Pre-LN Transformer with $N$ layers, the gradient norm is bounded by a product of per-layer Jacobian spectral norms. Each layer's Jacobian has the form $I + J_{\text{sub-layer}}$ (due to the residual/shortcut connection), so the gradient norm bound depends on two factors:

1. **Small sub-layers**: The spectral norms of the FFN and attention Jacobians ($\sigma_1 \sigma_2$, $\sigma_O$) should be small relative to the shortcut standard deviation.
2. **Large shortcut**: The standard deviation $\sigma_x$ of the residual stream (shortcut) should be large, so that the ratio $\sigma_{\text{sub-layer}} / \sigma_x$ remains small.

The widely-used initialization for LLMs (Megatron-style: $\sigma = \sqrt{2/(5d)}$, with $W_2, W_O$ scaled by $\sqrt{1/(2N)}$) satisfies the **small sub-layers** condition. However, it does **not** satisfy the **large shortcut** condition because embeddings are initialized with the same small standard deviation $\sqrt{2/(5d)}$, which shrinks as $d$ grows. This explains why larger models are more prone to loss spikes.

The fix is the **Scaled Embed** method: multiply embeddings by $\sqrt{d}$, which rescales the standard deviation from $\sqrt{2/(5d)}$ to $\sqrt{2/5} \approx 0.632$, close to 1. This was actually present in the original Transformer (Vaswani et al., 2017) but was dropped in modern LLM implementations. An equivalent alternative is **Embed LN**: applying LayerNorm to the embedding layer output, which normalizes the standard deviation to exactly 1.

**GPU efficiency:** The Scaled Embed is a single elementwise multiply on the embedding output — effectively zero overhead. No new parameters, no new kernel launches, no change to any attention or FFN computation. The stability gain enables training with **1.5× higher learning rates** (at 13B scale), directly translating to faster convergence in wall-clock time.

## Mathematical Form

**Gradient norm upper bound for $N$-layer Pre-LN Transformer:**

For loss $\mathcal{L}$ and input at layer 1:

$$
\left\|\frac{\partial \mathcal{L}}{\partial x_1}\right\|_2 \leq \left\|\frac{\partial \mathcal{L}}{\partial y_N}\right\|_2 \prod_{n=1}^{N-1} \left\|\frac{\partial y_n}{\partial x'_n}\right\|_2 \left\|\frac{\partial x'_n}{\partial x_n}\right\|_2
$$

where $y_n = x'_n + \text{FFN}(\text{LN}(x'_n))$ and $x'_n = x_n + \text{Attn}(\text{LN}(x_n))$.

---

**FFN Jacobian bound (Eq. 15 in paper):**

$$
\left\|\frac{\partial y}{\partial x'}\right\|_2 \leq 1 + \frac{\sigma_1 \sigma_2}{\sigma_{x'}} C_{\text{ffn}}
$$

where:
- $\sigma_1, \sigma_2$ — standard deviations of weight matrices $W_1 \in \mathbb{R}^{d_{\text{ffn}} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{\text{ffn}}}$
- $\sigma_{x'}$ — standard deviation of the shortcut (intermediate residual stream)
- $C_{\text{ffn}} = (\sqrt{d} + \sqrt{d_{\text{ffn}}})^2$ — dimension-dependent constant

The derivation uses:
1. $\|W_i\|_2 \approx \sigma_i(\sqrt{d} + \sqrt{d_{\text{ffn}}})$ for Gaussian random matrices (Vershynin, 2018)
2. LayerNorm Jacobian: $\frac{\partial \text{LN}(x')}{\partial x'} \approx \frac{1}{\sigma_{x'}} I$ (since $zz^\top/d \approx 0$ for $d \gg 1$)

---

**Attention Jacobian bound (Eq. 20 in paper):**

$$
\left\|\frac{\partial x'}{\partial x}\right\|_2 \leq 1 + \frac{\sigma_O}{\sigma_x} C_{\text{attn}}
$$

where:
- $\sigma_O$ — standard deviation of the output projection $W_O$
- $\sigma_x$ — standard deviation of the shortcut (input residual stream)
- $C_{\text{attn}} = 2\sqrt{d} \cdot \|J^Z\|_2$ where $J^Z$ is the Jacobian of the multi-head concatenation

---

**Conditions to suppress gradient norm growth:**

For the upper bounds to remain small (close to 1), we need:

$$
\sigma_1 \sigma_2 \ll \sigma_{x'} \quad \text{(small FFN sub-layer)}
$$

$$
\sigma_O \ll \sigma_x \quad \text{(small attention sub-layer)}
$$

**Condition 1 — Small sub-layers:** Satisfied by Megatron-style initialization:
- $\sigma = \sqrt{\frac{2}{5d}}$ for all weight matrices
- $W_2, W_O$ scaled by $\sqrt{\frac{1}{2N}}$
- This gives $\sigma_1 \sigma_2 \approx \frac{2}{5d} \cdot \sqrt{\frac{1}{2N}}$, which is small for large $d$ and $N$

**Condition 2 — Large shortcut:** The problem is the **embedding** layer. With standard initialization:

$$
\sigma_x = \sigma_{\text{embed}} = \sqrt{\frac{2}{5d}} \xrightarrow{d \to \infty} 0
$$

So $\sigma_{x}$ shrinks as $d$ grows, making $\sigma_O / \sigma_x$ large and violating the condition.

---

**Fix: Scaled Embed:**

$$
\text{Embed}_{\text{scaled}}(x) = \sqrt{d} \cdot \text{Embed}(x)
$$

This rescales:

$$
\sigma_x = \sqrt{d} \cdot \sqrt{\frac{2}{5d}} = \sqrt{\frac{2}{5}} \approx 0.632
$$

which is close to 1 and independent of $d$. The gradient norm upper bound becomes bounded regardless of model dimension.

**Alternative: Embed LN:**

$$
\text{Embed}_{\text{LN}}(x) = \text{LayerNorm}(\text{Embed}(x))
$$

This sets $\sigma_x = 1$ exactly.

## Complexity

| Operation | Baseline (Vanilla) | With Scaled Embed |
|-----------|-------------------|-------------------|
| Embedding | $O(T \cdot d)$ lookup | $O(T \cdot d)$ lookup + $O(T \cdot d)$ multiply |
| Extra FLOPs | — | $T \cdot d$ multiplications (negligible) |
| Extra parameters | — | None (0 parameters) |
| Extra memory | — | None (in-place multiply) |

**Memory:** Zero overhead. The scaling is an in-place elementwise operation.

**Wall-clock cost:** Immeasurably small — a single elementwise multiply on a $[B, T, d]$ tensor, fully parallelizable and memory-coalesced. In practice, this can be folded into the embedding lookup kernel.

**Training speedup via stability:**

| Model | Method | Max stable LR | WikiText PPL | LAMBADA PPL |
|-------|--------|--------------|-------------|-------------|
| 1.7B | Vanilla | $5 \times 10^{-4}$ (spikes) | 22.58 | 15.22 |
| 1.7B | Scaled Embed | $5 \times 10^{-4}$ (no spikes) | **21.29** | **12.53** |
| 13B | Vanilla | $1 \times 10^{-4}$ only | N/A at $3\times10^{-4}$ | 6.50 |
| 13B | Scaled Embed | $3 \times 10^{-4}$ (stable) | **14.47** | **5.97** |

At 13B scale, Scaled Embed enables 3× higher learning rate, with perplexity improvements of 0.71 (WikiText) and 0.53 (LAMBADA).

## Applicability

- **All Pre-LN Transformer LLMs:** Directly applicable to GPT-style, LLaMA-style, and any Pre-LN Transformer. The theory applies whenever the residual stream standard deviation at the embedding layer is too small relative to the sub-layer weight standard deviations.

- **Models using Megatron-style initialization:** Specifically targets the mismatch between the small embedding variance ($\sigma^2 = 2/(5d)$) and the residual stream signal scale expected by downstream layers.

- **Large models ($d > 1024$):** The issue becomes more severe as $d$ grows because $\sigma_{\text{embed}} = \sqrt{2/(5d)} \to 0$. Models with $d = 4096$ or larger benefit the most.

- **Higher learning rates:** The theoretical framework explains why larger learning rates trigger spikes (they amplify the gradient norm growth), so the fix enables more aggressive learning rate schedules.

- **RMSNorm variants:** The paper extends the analysis to RMSNorm (Appendix B.4), showing essentially the same conditions apply with slightly different constants.

- **Post-LN Transformers:** The analysis extends to Post-LN (Appendix H), where the conditions are harder to satisfy, explaining why Post-LN is known to be less stable.

## Limitations

- **Theory assumes Gaussian distributions:** The analysis relies on Assumption 1 (inputs and weights follow $\mathcal{N}(0, \sigma^2)$). While empirically validated, the bounds are approximate — the actual spectral norms fluctuate during training.

- **Only addresses embedding-layer instability:** Loss spikes can also originate from attention logit explosion (trick 215), optimizer state corruption (trick 226), or output logit divergence. Scaled Embed specifically fixes the embedding-scale mismatch.

- **Scaling factor is initialization-dependent:** The $\sqrt{d}$ factor is optimal for $\sigma_{\text{embed}} = \sqrt{2/(5d)}$. Different initialization schemes would require a different scaling factor (the general condition is $\sigma_x \approx 1$).

- **Embed LN is slightly more robust but adds cost:** LayerNorm adds a reduction operation per token position. For very long sequences, this is still negligible vs. attention cost, but Scaled Embed is strictly cheaper (a single constant multiply).

- **Validated up to 13B parameters:** The paper tests at 350M, 1.7B, and 13B scales. The theoretical analysis suggests it should generalize, but empirical validation at 100B+ scale is not provided.

## Implementation Notes

```python
import torch
import torch.nn as nn

class ScaledEmbedding(nn.Module):
    """
    Embedding with sqrt(d) scaling to prevent loss spikes.

    The original Transformer (Vasnwani et al., 2017) used this scaling,
    but modern LLM implementations (GPT-2, GPT-3, LLaMA) dropped it.
    This trick restores it with theoretical justification.

    GPU efficiency: Single elementwise multiply, zero parameters added.
    Can be fused into embedding lookup kernel.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale = d_model ** 0.5  # sqrt(d)

        # Standard initialization: N(0, 2/(5d))
        # After scaling: std dev becomes sqrt(2/5) ≈ 0.632
        nn.init.normal_(self.embed.weight, mean=0, std=(2 / (5 * d_model)) ** 0.5)

    def forward(self, x):
        return self.embed(x) * self.scale  # Single multiply


class EmbedLN(nn.Module):
    """
    Alternative: LayerNorm on embedding output.
    Sets std dev to exactly 1 (slightly more robust than Scaled Embed).
    Used by Le Scao et al. (2022) in BLOOM training.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(self.embed(x))


# WHY VANILLA FAILS:
#
# With d=4096, Megatron init: σ_embed = sqrt(2/(5*4096)) = 0.0099
# Upper bound per FFN layer (Eq. 15):
#   || ∂y/∂x' || ≤ 1 + (σ1 * σ2 / σ_x') * C_ffn
#
# With σ_x' ≈ σ_embed ≈ 0.01 at shallow layers:
#   ratio = σ1 * σ2 / 0.01 >> 1 → gradient norm explodes
#
# With Scaled Embed, σ_x' ≈ sqrt(2/5) ≈ 0.63:
#   ratio = σ1 * σ2 / 0.63 << 1 → gradient norm stays bounded
#
# Empirically (Figure 2 in paper):
#   Vanilla:       upper bound > 10 at layer 1 (explodes)
#   Scaled Embed:  upper bound < 0.1 at all layers (stable)

# GPU efficiency:
# - Scaled Embed: 1 elementwise multiply on [B, T, d] tensor
# - Embed LN: 1 reduction + 1 elementwise on [B, T, d] tensor
# - Both: zero extra kernel launches (fuse into embedding kernel)
# - Both: zero extra HBM round-trips
# - Both: enable higher learning rates → faster convergence
```

## References

- Takase, S., Kiyono, S., Kobayashi, S., & Suzuki, J. (2024). Spike No More: Stabilizing the Pre-training of Large Language Models. COLM 2025. arXiv:2312.16903.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017. (Original Transformer used $\sqrt{d}$ embedding scaling)
- Le Scao, T. et al. (2022). BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. (Used Embed LN for stability)
- Shoeybi, M. et al. (2020). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. (Standard LLM initialization)
- Zeng, A. et al. (2023). GLM-130B: An Open Bilingual Pre-Trained Model. (Used Embed Detach for stability, but doesn't satisfy large shortcut condition)
- Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML 2020. (Pre-LN stability analysis)
- Nguyen, T.Q. & Salazar, J. (2019). Transformers without Tears: Improving the Normalization of Self-Attention. (Small initialization for stability)

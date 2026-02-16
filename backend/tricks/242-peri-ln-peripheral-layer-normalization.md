# 242: Peri-LN — Peripheral Layer Normalization for Stable Transformer Training

**Category**: stability
**Gain type**: efficiency
**Source**: Kim, Lee, Park, Oh, Kim, Yoo, Shin, Han, Shin & Yoo (2025) — "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture", ICML 2025. arXiv:2502.02732
**Paper**: [papers/peri-ln-peripheral-normalization.pdf]
**Documented**: 2026-02-15

## Description

The placement of layer normalization (LN) in transformers is a critical design choice that governs both training stability and final model quality. The two dominant strategies — **Post-LN** and **Pre-LN** — each have well-known failure modes:

- **Post-LN** ($y_l = \text{Norm}(x_l + \text{Module}(x_l))$): Constrains hidden-state variance at constant scale, but places normalization **directly on the residual path** (position C in the paper's diagram), which can cause **gradient vanishing** and slow convergence in deeper models.

- **Pre-LN** ($y_l = x_l + \text{Module}(\text{Norm}(x_l))$): Preserves gradients better by keeping an unimpeded identity path, but **leaves module outputs unnormalized**. This allows hidden-state variance to grow **exponentially** across layers during training, producing "massive activations" that can exceed FP16 range and trigger numerical overflow/divergence.

**Peri-LN** applies normalization **peripherally around** each sub-layer — both before the module input (like Pre-LN) AND after the module output (Output-LN):

$$
y_l = x_l + \text{Norm}\!\Big(\text{Module}\!\big(\text{Norm}(x_l)\big)\Big)
$$

This design achieves **linear or sub-exponential variance growth** (like Post-LN) while maintaining **strong gradient flow** (like Pre-LN). The Output-LN acts as a self-regularizing mechanism that damps variance growth without degrading the gradient highway.

**GPU efficiency:** Peri-LN adds one extra LN per sub-layer (the Output-LN). LN is a cheap elementwise reduction over the channel dimension — on H100 with $d = 4096$, this adds $\sim$2µs per call ($O(d)$ reduction), negligible compared to the matmul cost of attention/FFN. The stability gain is the real payoff: Peri-LN eliminates gradient spikes and early-stage divergence, meaning **no wasted training runs**. The paper shows Pre-LN diverges on multiple random seeds even at optimal learning rates, while Peri-LN trains stably in every case. At scale, avoiding a single divergent run saves thousands of GPU-hours.

**Production adoption:** Peri-LN (or its Output-LN component) is used in **Gemma 2/3** (Google, 2024–2025), **OLMo 2** (AI2, 2024), and **Rivière et al.** (Meta, 2024), though these adoptions appeared independently without a unifying framework.

## Mathematical Form

**Three normalization placement strategies** (Figure 2 of paper shows positions A, B, C around each sub-layer):

| Strategy | Position A (before module) | Position B (after module) | Position C (after residual add) |
|----------|:---:|:---:|:---:|
| Post-LN | ✗ | ✗ | ✓ |
| Pre-LN | ✓ | ✗ | ✗ |
| **Peri-LN** | **✓** | **✓** | ✗ |

---

**Post-LN:**

$$
y_l = \text{Norm}\big(x_l + \text{Module}(x_l)\big) \tag{1}
$$

**Pre-LN:**

$$
y_l = x_l + \text{Module}\big(\text{Norm}(x_l)\big) \tag{2}
$$

**Peri-LN (proposed):**

$$
y_l = x_l + \text{Norm}\!\Big(\text{Module}\!\big(\text{Norm}(x_l)\big)\Big) \tag{3}
$$

with additional optional normalizations:
- **(Optional) Initial embedding normalization:** $y_0 = \text{Norm}(x_{\text{embed}})$
- **Final normalization:** $y_L = \text{Norm}(x_L)$ (applied to the last layer's output before the language model head)

---

**Variance propagation analysis:**

Under Peri-LN, if $\text{Norm}(\text{Module}(\text{Norm}(x_l)))$ produces outputs with near-constant variance $\beta_0$:

$$
\text{Var}(x_{l+1}) \approx \text{Var}(x_l) + \beta_0 \tag{4}
$$

This gives **linear variance growth** with depth, compared to Pre-LN's **exponential** growth when module outputs amplify the signal.

---

**Gradient stability (Proposition 3.1 of paper):**

Consider a single MLP sub-layer with weight $W^{(2)}$ and pre-activation $h := \text{ReLU}(\tilde{x} W^{(1)} + b^{(1)})$.

**(1) Pre-LN (exploding gradient):**

$$
\tilde{x} = \text{Norm}(x), \quad a = \text{MLP}(\tilde{x}), \quad o = x + a
$$

$$
\left\|\frac{\partial \mathcal{L}(o)}{\partial W^{(2)}_{i,j}}\right\| \propto \|h_i\|
$$

When a massive activation occurs ($\|h\| \gg 1$), the gradient **scales with $\|h\|$**, causing gradient explosion.

**(2) Peri-LN (self-regularizing gradient):**

$$
\tilde{x} = \text{Norm}(x), \quad a = \text{MLP}(\tilde{x}), \quad \tilde{a} = \text{Norm}(a), \quad o = x + \tilde{a}
$$

$$
\left\|\frac{\partial \mathcal{L}(o)}{\partial W^{(2)}_{i,j}}\right\| \leq \frac{4\gamma\sqrt{D}\,\|h\|}{\|a\|} \tag{8}
$$

Even when $\|h\|$ is massive, the Output-Norm introduces a **damping factor** $\|a\|$ in the denominator. Since $\|a\|$ grows with $\|h\|$, the ratio $\|h\|/\|a\|$ remains bounded, preventing gradient explosion.

---

**Gradient norm hierarchy (empirically verified):**

$$
\|\nabla W\|_{\text{Post-LN}} > \|\nabla W\|_{\text{Pre-LN}} > \|\nabla W\|_{\text{Peri-LN}}
$$

Peri-LN has the smallest and most **uniform** gradient norms across layers at both initialization and end of training (Figure 7).

## Complexity

| Operation | Pre-LN | Peri-LN | Overhead |
|-----------|--------|---------|----------|
| LN calls per transformer block | 2 (attn + FFN) | **4** (2× pre + 2× post) | +2 LN calls |
| LN on embeddings | 0 or 1 | 1 (optional) | +0–1 |
| LN on final output | 1 | 1 | 0 |
| Parameters per block | $4C$ ($\gamma, \beta$ for 2 LNs) | $8C$ ($\gamma, \beta$ for 4 LNs) | $+4C$ |

**Per-LN-call cost:** $O(C)$ reduction (mean + variance) + $O(C)$ elementwise (normalize + affine). For $C = 4096$, this is $\sim$16KB of data, fitting comfortably in L1/shared memory.

**Wall-clock overhead:** The paper reports negligible overhead (< 1% of step time) because LN is bandwidth-limited and dwarfed by the matmul cost of attention and FFN projections. For a transformer with $d = 4096$, the LN cost is $\sim O(BTC)$ vs. $\sim O(BT \cdot 4d^2)$ for the FFN, a ratio of $\sim 1 : 4d \approx 1 : 16384$.

**Memory:** +$O(BTC)$ for storing the extra LN output (needed for backprop). In practice this is small compared to attention maps ($O(BHT^2)$) and KV cache.

**Training stability gain (the real efficiency):**

| Metric | Pre-LN | Peri-LN |
|--------|--------|---------|
| Divergent runs (5 seeds) | 1–2 out of 5 | **0 out of 5** |
| Loss std across seeds (400M) | ±1.63 (ARC-Easy) | **±0.81** |
| Max stable LR | $2 \times 10^{-3}$ | $2 \times 10^{-3}$ (same) |
| Avg benchmark (400M) | 49.69 | **51.57** |
| Avg benchmark (1.5B) | 53.71 | **56.55** |
| Avg benchmark (3.2B) | 56.69 | **58.56** |

Peri-LN consistently outperforms Pre-LN by **+2–3 points** on average benchmarks with **half the variance**, meaning more reliable training outcomes.

## Applicability

- **All transformer-based LLMs:** Validated on 400M, 1.5B, and 3.2B parameter models trained on 30B tokens (DCLM dataset). Consistent improvements across all scales. Works with both LayerNorm and RMSNorm.

- **Production models already using it:**
  - **Gemma 2/3** (Google, 2024–2025): Uses output normalization on attention and MLP sub-layers
  - **OLMo 2** (AI2, 2024): Applies output normalization following the Peri-LN pattern
  - **Rivière et al.** (Meta, 2024): Uses peripheral normalization in their model design

- **Instruction-tuned models:** Peri-LN shows stronger gains after SFT (supervised fine-tuning), with +5 points on SFT benchmarks compared to Pre-LN at 400M scale.

- **Mixed-precision training:** Particularly important for FP16/BF16 training. Figure 11 of the paper shows Pre-LN hidden states exceed FP16 max ($6.5 \times 10^4$) within 0.5B tokens, while Peri-LN stays well within BF16 range throughout training. This makes Peri-LN essential for stable FP16 pretraining.

- **Deep models:** The variance-damping property becomes more important with depth. The paper shows consistent benefits from 24-layer (400M) to 48-layer (3.2B) architectures.

- **SSMs and hybrid architectures:** While the paper focuses on transformers, the principle of wrapping sub-layers with input+output normalization applies to any residual architecture, including Mamba-style SSM blocks and hybrid attention-SSM models.

## Limitations

- **+2 LayerNorm calls per block:** While cheap, this does add some overhead. For very latency-sensitive inference (e.g., single-token generation), the extra LN calls add $\sim$4µs per block on H100, totaling $\sim$128µs for a 32-layer model. This is small but non-zero.

- **Not validated beyond 3.2B parameters:** The paper's largest model is 3.2B (30B tokens). The production adoption in Gemma-2 (27B), OLMo-2 (7B/13B), and similar models provides indirect evidence at larger scales, but systematic ablations at 7B+ are not reported.

- **Same maximum stable learning rate as Pre-LN:** Peri-LN does not enable higher learning rates (both optimal at $2 \times 10^{-3}$). The benefit is more reliable convergence and better final quality, not faster training per step.

- **Output-LN γ needs to be learnable:** Freezing the $\gamma$ parameter of the output normalization degrades performance (Figure 8). The learned $\gamma$ starts near 1 and adjusts during training, providing an adaptive scaling mechanism. This adds a minor hyperparameter concern.

- **Post-LN can sometimes match at small scale:** At 400M parameters, Post-LN with carefully tuned lower learning rates can match Pre-LN performance. The gap widens at larger scales (1.5B, 3.2B).

## Implementation Notes

```python
import torch
import torch.nn as nn

class PeriLNTransformerBlock(nn.Module):
    """
    Transformer block with Peri-LN (peripheral normalization).

    Key change from Pre-LN: add Output-LN after each sub-layer's
    module output, before the residual addition.

    y_l = x_l + Norm(Module(Norm(x_l)))
         ^              ^        ^
         |              |        |
       residual    Output-LN  Pre-LN (same as before)
       (identity)  (NEW)

    GPU notes:
    - Extra LN is a cheap O(C) reduction — negligible vs matmul cost
    - No new matmuls, no irregular memory access
    - Fuses naturally with existing kernels (e.g., fused bias+LN)
    - The stability gain prevents divergent runs → saves GPU-hours
    """
    def __init__(self, d_model, n_heads, d_ff=None, norm_type="rmsnorm"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        Norm = nn.RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm

        # Attention sub-layer
        self.attn_pre_norm = Norm(d_model)     # Pre-LN (position A)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_post_norm = Norm(d_model)    # Output-LN (position B) ← NEW

        # FFN sub-layer
        self.ffn_pre_norm = Norm(d_model)      # Pre-LN (position A)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_post_norm = Norm(d_model)     # Output-LN (position B) ← NEW

    def forward(self, x, mask=None):
        # Attention with Peri-LN: x + Norm(Attn(Norm(x)))
        residual = x
        x_norm = self.attn_pre_norm(x)         # Pre-normalize input
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        attn_out = self.attn_post_norm(attn_out)  # Post-normalize output ← NEW
        x = residual + attn_out

        # FFN with Peri-LN: x + Norm(FFN(Norm(x)))
        residual = x
        x_norm = self.ffn_pre_norm(x)          # Pre-normalize input
        ffn_out = self.ffn(x_norm)
        ffn_out = self.ffn_post_norm(ffn_out)  # Post-normalize output ← NEW
        x = residual + ffn_out

        return x


class PeriLNTransformer(nn.Module):
    """
    Full transformer with Peri-LN strategy.

    Additional normalizations:
    1. (Optional) Normalize embeddings
    2. Final normalization before LM head
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_norm = nn.RMSNorm(d_model)  # Optional embedding norm

        self.layers = nn.ModuleList([
            PeriLNTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model)  # Final norm before head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.embed_norm(x)  # Normalize embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.lm_head(x)


# Migration from Pre-LN to Peri-LN:
# 1. Keep all existing Pre-LN normalization layers
# 2. Add one RMSNorm/LayerNorm AFTER each attention output (before residual add)
# 3. Add one RMSNorm/LayerNorm AFTER each FFN output (before residual add)
# 4. (Optional) Add embedding normalization
# 5. Keep all other hyperparameters the same
#
# That's it. No learning rate changes, no initialization changes needed.
```

## References

- Kim, J., Lee, B., Park, C., Oh, Y., Kim, B., Yoo, T., Shin, S., Han, D., Shin, J., & Yoo, K.M. (2025). Peri-LN: Revisiting Normalization Layer in the Transformer Architecture. ICML 2025. arXiv:2502.02732.
- Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML 2020.
- Kedia, A. et al. (2024). Signal propagation in Transformers. arXiv preprint.
- Sun, Z. et al. (2024). Massive Activations in Large Language Models. ICML 2024.
- Rivière, M. et al. (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- OLMo Team (2024). OLMo 2: An Open Language Model. arXiv preprint.
- Team, G. et al. (2025). Gemma 3 Technical Report. arXiv preprint.
- Wortsman, M. et al. (2024). Small-Scale Proxies for Large-Scale Transformer Training Instabilities. ICLR 2024.
- De, S. & Smith, S.L. (2020). Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. NeurIPS 2020.

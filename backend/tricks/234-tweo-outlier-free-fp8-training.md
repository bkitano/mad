# 234: TWEO — Outlier-Preventing Loss for FP8 Training

**Category**: stability
**Gain type**: efficiency
**Source**: Liang et al., "TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies" (arXiv:2511.23225, Nov 2025)
**Paper**: papers/tweo-outlier-free-fp8-training.pdf
**Documented**: 2026-02-15

## Description

Extreme activation outliers (magnitudes > 1000) are the primary obstacle to FP8 training and post-training quantization of transformers. Existing solutions either bypass outliers by keeping sensitive layers in BF16 (sacrificing throughput), or require invasive architectural modifications (Smooth-SwiGLU, register tokens, clipped softmax). TWEO takes a fundamentally different approach: rather than working around outliers, it **prevents them from forming** via a simple regularization loss added to the training objective.

The key insight is that extreme outliers are **not data-dependent** — they are a mechanical artifact of weight matrix structure that emerges during training. Specifically, outliers arise when a row $w^T$ of weight matrix $B$ becomes collinear with a left singular vector $u_i$ of weight matrix $A$ in an MLP layer $y = BAx$. This structural alignment amplifies the product $s_i (w^T u_i)(v_i^T x)$, causing extreme outputs. Since the root cause is structural (weight colinearity), a data-independent regularizer suffices.

TWEO adds a scaled $L_p$ penalty on layer output activations that is tolerant of normal magnitudes but aggressively penalizes extreme values. This reduces peak activations from >30,000 to <20 across all layers, enabling the simplest possible FP8 configuration: true per-tensor scaling with `DelayedScaling` and an extremely short `amax_history_len=16` — a configuration where standard training catastrophically collapses.

## Mathematical Form

**Core Operation — TWEO Loss:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot \mathcal{L}_{\text{TWEO}}
$$

where the TWEO regularizer is:

$$
\mathcal{L}_{\text{TWEO}} = \frac{1}{L} \sum_{l=1}^{L} \mathbb{E}\left[\left(\frac{|A^{(l)}|}{\tau + \epsilon}\right)^p\right]
$$

**Key Definitions:**

- $A^{(l)}$ — output activation tensor of the $l$-th Transformer block (i.e., $y = x + \text{MLP}(\text{LN}(x))$)
- $\tau > 0$ — magnitude scaling factor (soft threshold), default $\tau = 3$
- $p$ — penalty power, default $p = 4$
- $\epsilon$ — small constant for numerical stability (1e-6)
- $\lambda(t)$ — time-varying loss weight, default $\lambda = 0.01$, optionally with cosine annealing
- $L$ — number of Transformer blocks
- $\mathbb{E}[\cdot]$ — mean over batch, sequence length, and hidden dimension

**Why $p = 4$ is critical — Nonlinear Selectivity:**

The interaction of $\tau$ and $p$ creates selective suppression:

- **Normal activations** ($|A| < \tau$): penalty term $\approx (0.5)^4 = 0.0625$ — negligible
- **At threshold** ($|A| = \tau$): penalty $= 1$ — moderate
- **Extreme outliers** ($|A| = 10\tau$): penalty $= (10)^4 = 10000$ — aggressive suppression

**Root Cause — Colinearity-Driven Outliers (Eq. 1):**

For MLP output $y_k = w^T A x = \sum_{i=1}^{d_1} s_i (w^T u_i)(v_i^T x)$, extreme outliers occur when:
1. Row $w$ of $B$ is collinear with a left singular vector $u_i$ of $A$ ($|w^T u_i|$ large)
2. Input $x$ is aligned with corresponding right singular vector $v_i$ ($|v_i^T x|$ large)
3. Singular value $s_i$ is large

## Complexity

| Operation | Without TWEO | With TWEO |
|-----------|-------------|-----------|
| Forward pass | Standard | + $O(LBTd)$ elementwise ops per step |
| Backward pass | Standard | + $O(LBTd)$ gradient through penalty |
| FP8 scaling strategy | Fine-grained (per-block/tile-wise) | Simplest per-tensor scaling |
| Quantization (PTQ) | SmoothQuant / per-token / per-channel | Simple AbsMax per-tensor static |

**Training overhead**: The TWEO loss adds negligible compute — it's an elementwise absolute-value, power, and mean over activations that are already computed. The net effect is a **36% training throughput increase** from using aggressive FP8 vs. BF16.

**Memory:** Identical to baseline — no extra buffers, parameters, or architectural changes required.

## Applicability

- **FP8 pre-training** of LLMs (GPT-2 124M → 7B validated) and ViTs (Swin-T/S/B, ViT-S/B)
- **Post-training quantization**: TWEO-trained models support simple W8A8 per-tensor static quantization (AbsMax) — previously impossible due to outliers. Even the **residual stream** can be quantized for the first time
- **Universal**: Works across language and vision, classic MLPs and GLU variants, without architectural modification
- Adopted FP8 config: NVIDIA Transformer Engine, `Format.HYBRID` (E4M3/E5M2), `DelayedScaling`, `amax_history_len=16`

## Limitations

- Not validated at >7B scale (authors note resource constraints for 700B+ models)
- Adds two hyperparameters ($\tau$, $p$) though defaults ($\tau=3$, $p=4$) are robust across all tested settings
- The $\lambda(t)$ weighting requires mild tuning; cosine annealing optional but not essential
- Only applied from scratch — not yet tested for removing outliers from already-trained models via fine-tuning
- The penalty is on block-level outputs; it doesn't directly regulate internal attention or MLP hidden activations (though empirically those are controlled too)

## Implementation Notes

```python
# TWEO loss — add to training loop
def tweo_loss(block_outputs, tau=3.0, p=4, eps=1e-6):
    """
    block_outputs: list of L tensors, each (B, T, d)
    Returns scalar loss to add to task loss.
    """
    loss = 0.0
    for act in block_outputs:
        # Scaled Lp penalty
        loss += ((act.abs() / (tau + eps)) ** p).mean()
    return loss / len(block_outputs)

# Training loop
loss = task_loss + lambda_weight * tweo_loss(block_outputs)
loss.backward()

# FP8 config — simplest possible, no engineering tricks needed
# Uses NVIDIA Transformer Engine:
#   format = Format.HYBRID  (E4M3 fwd, E5M2 bwd)
#   DelayedScaling with amax_history_len = 16
#   ALL linear layers + LayerNorm under FP8 autocast
#   No layers excluded (including embeddings and LM head!)
```

**GPU Efficiency Analysis:**

- **Memory access**: TWEO loss is computed on activations already in registers/SRAM from the forward pass. The elementwise power operation is trivially fused into the existing backward kernel
- **Arithmetic intensity**: Extremely high — just elementwise ops on existing tensors, no new memory loads
- **Tensor core utilization**: The real gain is enabling per-tensor FP8 GEMM with tensor cores. Per-tensor scaling means a single scaling factor per matrix multiply, maximizing throughput vs. per-token/per-channel schemes that require extra scaling/descaling operations
- **HBM bandwidth**: Major win — FP8 weights and activations halve memory traffic vs. BF16. The residual stream quantization (newly enabled by TWEO) further reduces memory bandwidth
- **No kernel launches added**: The TWEO loss is fused into existing backward pass computation

## References

- Liang et al., "TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies" (arXiv:2511.23225, 2025)
- Fishman et al., "Scaling FP8 Training to Trillion-Token LLMs" (arXiv:2409.12517, 2024) — Smooth-SwiGLU approach (trick 227)
- Hernández-Cano et al., "Towards Fully FP8 GEMM LLM Training at Scale" (arXiv:2505.20524, 2025)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs" (PMLR 2023)
- Bondarenko et al., "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing" (NeurIPS 2023)

# 227: Smooth-SwiGLU — Per-Channel Activation Rescaling for FP8 Stability

**Category**: stability
**Gain type**: efficiency
**Source**: Fishman et al., "Scaling FP8 Training to Trillion-Token LLMs" (ICLR 2025)
**Paper**: papers/scaling-fp8-smooth-swiglu.pdf
**Documented**: 2026-02-15

## Description

SwiGLU is the standard FFN activation in modern LLMs (LLaMA, PaLM), but it has a hidden instability: its output is **quadratic** in the input when the two weight vectors $\mathbf{w}_1, \mathbf{w}_2$ align, causing extreme activation outliers that only emerge after $\sim 200$B tokens of training. Under FP8 precision, these outliers exceed the representable range and cause training divergence. Smooth-SwiGLU applies a **per-channel scaling factor** to the linear (gating) branch before quantization, then inverts it after the downstream linear layer. This bounds the activation dynamic range within FP8's representable range while being **mathematically equivalent** to standard SwiGLU — at inference the scaling factors are absorbed into the adjacent weight matrices with zero overhead.

Combined with FP8 quantization of both Adam optimizer moments (first moment in E4M3, second in E5M2), this enables stable FP8 training of 7B-parameter LLMs on 2 trillion tokens — a 20$\times$ increase over prior FP8 limits — with ~34% throughput improvement and ~30% memory reduction vs. BF16.

## Mathematical Form

**Standard SwiGLU:**

$$
\text{SwiGLU}_{\mathbf{w}_1, \mathbf{w}_2}(\mathbf{x}) = (\mathbf{x}^\top \mathbf{w}_1) \cdot \text{Swish}(\mathbf{x}^\top \mathbf{w}_2)
$$

where $\text{Swish}(z) = z \cdot \sigma(z)$ and $\sigma(z) = 1/(1+e^{-z})$.

**The Instability (Theorem 1):**

When $\mathbf{w}_1 \approx \mathbf{w}_2$ (which happens due to $\ell_2$ regularization driving both to eigenvectors of $A = \sum_n \lambda_n \mathbf{x}_n \mathbf{x}_n^\top$), the SwiGLU output becomes approximately:

$$
\text{SwiGLU} \approx (\mathbf{x}^\top \mathbf{w})^2 \cdot \sigma(\mathbf{x}^\top \mathbf{w}) / c^2
$$

This is **quadratic** in the input, unlike ReLU/GeLU/Swish which are at most linear for large inputs ($\lim_{u \to \pm\infty} |f(u)/u| \leq 1$). The quadratic behavior amplifies outliers.

**Smooth-SwiGLU (per-channel rescaling):**

For each output channel $i$, compute a scaling factor $s_i$ from the per-channel maximum of the linear branch:

$$
\text{Smooth-SwiGLU}_{\hat{\mathbf{w}}_{1,i}, \hat{\mathbf{w}}_{2,i}}(\mathbf{x}) = s_i^{-1} \cdot Q\!\left(s_i \cdot (\hat{\mathbf{w}}_{1,i}^\top Q(\mathbf{x})) \cdot \text{Swish}(\hat{\mathbf{w}}_{2,i}^\top Q(\mathbf{x}))\right)
$$

where $Q(\cdot)$ denotes FP8 quantization and $\hat{\mathbf{w}} = Q(\mathbf{w})$.

**Scaling Factor Computation:**

1. Split activation tensor into chunks (one per channel)
2. Compute per-channel maximum: $s_i = \max_j |(\mathbf{w}_{1,i}^\top \mathbf{x}_j)|$
3. Scale the linear branch by $s_i$ before the elementwise product

**Inference-Time Absorption (zero overhead):**

The scaling factors are absorbed into adjacent weight matrices:

$$
\tilde{\mathbf{w}}_{1,i} \triangleq Q(s_i \cdot \mathbf{w}_{1,i}), \quad \tilde{\mathbf{w}}_{3,i} \triangleq Q(s_i^{-1} \cdot \mathbf{w}_{3,i})
$$

where $\mathbf{w}_3$ is the down-projection weight. After absorption, the computation is identical to standard SwiGLU with no additional operations.

**Key Definitions:**

- $\mathbf{x} \in \mathbb{R}^d$ — input from previous layer
- $\mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^d$ — gate and up-projection weight vectors (per channel)
- $\mathbf{w}_3 \in \mathbb{R}^{d_\text{ffn} \times d}$ — down-projection weight matrix
- $s_i \in \mathbb{R}$ — per-channel scaling factor
- $Q: \mathbb{R} \to \text{FP8}$ — quantization function (E4M3 for forward, E5M2 for backward)

**FP8 Optimizer Moments:**

Additionally, the paper shows both Adam moments can be stored in FP8:
- First moment ($m_t$): E4M3 format (4 exponent, 3 mantissa) — sufficient precision for mean estimation
- Second moment ($v_t$): E5M2 format (5 exponent, 2 mantissa) — needs wider dynamic range due to inverse square root in Adam update

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{Q_{E4M3}(m_t)}{\sqrt{Q_{E5M2}(v_t)} + \epsilon}
$$

## Complexity

| Operation | BF16 Baseline | With Smooth-SwiGLU FP8 |
|-----------|--------------|----------------------|
| FFN matmul (per token) | $O(d \cdot d_\text{ffn})$ in BF16 | $O(d \cdot d_\text{ffn})$ in FP8 (~2$\times$ throughput) |
| Scaling overhead | — | $O(d_\text{ffn})$ per-channel max + scale (negligible) |
| Inference overhead | — | Zero (absorbed into weights) |

**Memory:**
- Optimizer: FP32 moments ($2 \times 4$ bytes/param) → FP8 moments ($2 \times 1$ byte/param) = ~30% total memory reduction
- Activations: FP8 vs BF16 = 2$\times$ reduction in activation memory

**Throughput:** ~34% improvement (16.89 vs 12.65 samples/sec on 8$\times$ Intel Gaudi2 for LLaMA2-7B)

## GPU Efficiency Analysis

**Memory Access Pattern:** Fully coalesced. The per-channel scaling is an elementwise multiply on contiguous memory. The core computation remains standard matmul (GEMM), which is the most optimized GPU primitive.

**Parallelism:** Per-channel max reduction is embarrassingly parallel across channels. The scaling multiply is elementwise. All heavy compute remains in GEMMs that saturate tensor cores.

**Arithmetic Intensity:** Improved vs. BF16 — FP8 GEMMs have 2$\times$ the arithmetic intensity (same FLOPs, half the bytes loaded). The per-channel scaling adds negligible FLOPs relative to the matmuls.

**Tensor Core Usage:** FP8 GEMMs directly use FP8 tensor cores (available on H100, Intel Gaudi2). The scaling operation does not affect tensor core utilization of the main matmuls.

**Integration:**
- Training: Requires modifying the SwiGLU forward pass to insert per-channel scaling before the FP8 quantization point. Fits naturally into existing mixed-precision training frameworks.
- Inference: Zero modification — scaling factors pre-absorbed into weights at checkpoint conversion time.

## Applicability

- **All SwiGLU-based transformers**: LLaMA, LLaMA2, PaLM, Gemma, Mistral, and any model using SwiGLU/GeGLU/ReGLU FFNs
- **FP8 training at scale**: Critical for training beyond ~100B tokens in FP8 precision
- **Long training runs**: The weight alignment effect is progressive — only manifests after sufficient training. Short runs ($<100$B tokens) may not need this fix
- **Extends to other GLU variants**: Theorem 1 holds for any GLU variant (the proof doesn't rely on specific properties of Swish), so the same instability and fix apply to GeGLU, ReGLU, etc.

## Limitations

- **Training-only overhead**: During training, the per-channel max and scaling add a small overhead (though negligible vs. GEMM cost). At inference, zero overhead after weight absorption.
- **FP8 hardware required**: The throughput gains require FP8-capable hardware (H100, Gaudi2). On older GPUs, the stability insight still applies but without the speed benefit.
- **Per-channel granularity**: The scaling uses per-channel (per-output-neuron) granularity. Finer granularity (e.g., per-token-per-channel as in DeepSeek V3) might be needed for extreme cases.
- **Delayed scaling assumption**: The paper uses delayed scaling (scale factors from previous iteration). Just-in-time scaling would be more robust but adds overhead.
- **Complementary to other fixes**: Smooth-SwiGLU addresses activation outliers specifically from the SwiGLU structure. Other sources of instability (attention logit growth, embedding scale) require separate techniques (see tricks 215, 220).

## Implementation Notes

```python
# Smooth-SwiGLU: per-channel rescaling for FP8 stability
def smooth_swiglu_forward(x, w1, w2, w3, quantize_fn):
    """
    x: [batch, seq, d]
    w1: [d, d_ffn] (gate projection)
    w2: [d, d_ffn] (up projection)
    w3: [d_ffn, d] (down projection)
    """
    # Standard projections (in FP8)
    gate = quantize_fn(x) @ quantize_fn(w1)  # [B, S, d_ffn]
    up   = quantize_fn(x) @ quantize_fn(w2)  # [B, S, d_ffn]

    # Per-channel scaling factor from gate branch
    s = gate.abs().amax(dim=(0, 1))  # [d_ffn], per-channel max

    # Scale gate, apply swish to up, multiply, then unscale
    # This bounds the product's dynamic range for FP8 quantization
    out = quantize_fn((gate / s) * swish(up))  # safe for FP8

    # Down projection with inverse scaling absorbed
    return out @ quantize_fn(w3 * s.unsqueeze(1))  # absorb s^{-1} into w3


# At inference checkpoint conversion (zero runtime cost):
def absorb_smooth_swiglu_scales(w1, w3, s):
    """Pre-multiply scales into weights for zero-overhead inference."""
    w1_absorbed = w1 * s.unsqueeze(0)     # scale into gate weights
    w3_absorbed = w3 / s.unsqueeze(1)     # inverse scale into down-proj
    return w1_absorbed, w3_absorbed
```

## References

- Fishman et al., "Scaling FP8 Training to Trillion-Token LLMs", ICLR 2025. arXiv:2409.12517
- Shazeer, "GLU Variants Improve Transformer", 2020. arXiv:2002.05202
- Micikevicius et al., "FP8 Formats for Deep Learning", 2022. arXiv:2209.05433
- Peng et al., "FP8-LM: Training FP8 Large Language Models", 2023. arXiv:2310.18313
- Code: https://github.com/Anonymous1252022/Megatron-DeepSpeed

# 247: Stochastic Rounding for Low-Precision Training

**Category**: stability
**Gain type**: efficiency
**Source**: Gupta et al., "Deep Learning with Limited Numerical Precision" (ICML 2015); Xia et al., "A Simple and Efficient Stochastic Rounding Method" (arXiv 2021); Connolly et al., "Direct Quantized Training with Stochastic Rounding" (arXiv 2024)
**Paper**: papers/gupta-stochastic-rounding-2015.pdf, papers/stochastic-rounding-low-precision.pdf
**Documented**: 2026-02-15

## Description

Stochastic rounding (SR) is a probabilistic rounding scheme that preserves gradient information during low-precision training by maintaining a **non-zero probability** for small weight updates to affect parameters, even when the update magnitude is below the minimum representable precision $\epsilon$. Unlike round-to-nearest (RTN), which deterministically rounds all values in $[-\epsilon/2, \epsilon/2]$ to zero — silently killing small gradients — stochastic rounding rounds up or down with probability proportional to proximity, making the expected rounded value equal to the true value: $\mathbb{E}[\text{SR}(x)] = x$.

This is the foundational technique that enables training in FP8, FP4, and aggressive fixed-point formats. Without SR, low-precision training diverges or hits irreducible convergence floors because the majority of gradient updates are smaller than the precision threshold and are systematically eliminated. SR converts deterministic truncation into unbiased noise, which SGD can average out over steps.

**Why it matters for GPU pretraining**: SR enables aggressive quantization (FP8/FP4) for both forward and backward passes, unlocking 2-4x throughput gains from reduced memory bandwidth and higher tensor core utilization. NVIDIA Blackwell GPUs have native hardware SR support for FP4 conversion, making this technique directly hardware-accelerated. The key GPU benefit is **reduced HBM bandwidth** — the dominant bottleneck — via smaller data types, with SR ensuring training quality is preserved.

## Mathematical Form

**Core Operation — Stochastic Rounding:**

Given a real value $x$ and target precision with resolution $\epsilon$ (smallest representable step), let $\lfloor x \rfloor$ denote the largest representable value $\leq x$:

$$
\text{SR}(x) = \begin{cases} \lfloor x \rfloor & \text{with probability } 1 - \frac{x - \lfloor x \rfloor}{\epsilon} \\[6pt] \lfloor x \rfloor + \epsilon & \text{with probability } \frac{x - \lfloor x \rfloor}{\epsilon} \end{cases}
$$

**Key Property — Unbiasedness:**

$$
\mathbb{E}[\text{SR}(x)] = \lfloor x \rfloor \left(1 - \frac{x - \lfloor x \rfloor}{\epsilon}\right) + (\lfloor x \rfloor + \epsilon) \cdot \frac{x - \lfloor x \rfloor}{\epsilon} = x
$$

The rounding error has zero mean: $\mathbb{E}[\text{SR}(x) - x] = 0$.

**Variance of Stochastic Rounding:**

$$
\text{Var}[\text{SR}(x)] = \epsilon^2 \cdot p(1-p), \quad \text{where } p = \frac{x - \lfloor x \rfloor}{\epsilon}
$$

Maximum variance is $\epsilon^2/4$ (when $x$ is exactly halfway between representable values).

**Contrast with Round-to-Nearest (RTN):**

$$
\text{RTN}(x) = \begin{cases} \lfloor x \rfloor & \text{if } x - \lfloor x \rfloor \leq \epsilon/2 \\ \lfloor x \rfloor + \epsilon & \text{if } x - \lfloor x \rfloor > \epsilon/2 \end{cases}
$$

RTN has zero variance but non-zero bias: any update in $(-\epsilon/2, \epsilon/2)$ is deterministically zeroed. For weight updates where $|\alpha \nabla_w L| < \epsilon/2$, the gradient signal is **permanently lost**.

**Composability — Sums Commute with SR:**

For $N$ values $x_1, \ldots, x_N$:

$$
\text{SR}(x_1) + \text{SR}(x_2) + \cdots + \text{SR}(x_N) = \text{SR}(x_1 + x_2 + \cdots + x_N)
$$

in the sense that both sides have the same expected value $\sum x_i$. This means SR applied to individual products in matrix multiplication preserves the expected dot product.

**Application to Weight Updates (SGD):**

$$
w_{t+1} = Q_{\text{SR}}\!\left(w_t - \alpha \nabla f(w_t)\right)
$$

where $Q_{\text{SR}}$ applies stochastic rounding to the target precision. Since $\mathbb{E}[Q_{\text{SR}}(w_{t+1})] = w_t - \alpha \nabla f(w_t)$, convergence guarantees of SGD are preserved (up to increased variance).

**Improved Variant — Random Rounding (RR, Xia et al. 2021):**

Uses constant probability $p = 0.5$ for all values:

$$
\text{RR}(x) = \begin{cases} \lfloor x \rfloor & \text{with probability } 0.5 \\ \lfloor x \rfloor + \epsilon & \text{with probability } 0.5 \end{cases}
$$

This has higher bias ($|B|_{\max} = \epsilon/2$) but fewer zeros in the rounded output. The constant probability simplifies hardware implementation (no proportional random number generation needed). RR achieves faster convergence than both RTN and standard CSR in experiments with 8-bit fractional precision.

## Complexity

| Operation | FP32 Baseline | FP8 + Stochastic Rounding |
|-----------|--------------|---------------------------|
| Weight memory | $32n$ bits | $8n$ bits (4x reduction) |
| Activation memory | $32n$ bits | $8n$ bits (4x reduction) |
| Matmul throughput (H100) | 989 TFLOPS (BF16) | 1979 TFLOPS (FP8, 2x) |
| HBM bandwidth pressure | Baseline | 2-4x reduction |

**Memory**: $O(n)$ in both cases, but constant factor is 2-4x smaller with FP8/FP4.

**Compute overhead of SR**: One random number generation per rounding operation. On GPU, this is a single `curand` call fused into the quantization kernel — negligible compared to matmul cost.

**Convergence**: SR preserves the $O(1/\sqrt{T})$ convergence rate of SGD for convex objectives (same asymptotic rate as FP32), but with increased variance proportional to $\epsilon^2$.

## Applicability

- **FP8 LLM pretraining**: SR on gradient quantization enables end-to-end FP8 training (forward in E4M3, backward in E5M2). Used in NVIDIA Transformer Engine and DeepSpeed FP8 pipelines
- **FP4/MXFP4 training**: Stochastic rounding is essential for sub-8-bit training. NVIDIA Blackwell has native SR hardware for FP4 conversion instructions
- **SSM state updates**: SSMs accumulate state over long sequences; SR prevents systematic drift in low-precision state representations that would compound over time steps
- **Gradient accumulation**: When accumulating microbatch gradients in low precision, SR prevents small gradient contributions from being silently zeroed
- **Distributed training**: SR on gradient all-reduce enables FP8 communication (2x bandwidth reduction) while maintaining convergence — critical for scaling to thousands of GPUs
- **Complementary to trick 221 (Kahan summation)**: SR is the probabilistic alternative to Kahan's deterministic error compensation. SR requires hardware RNG support but no auxiliary buffer; Kahan requires extra memory but no RNG. Both solve the same fundamental problem of precision loss in weight updates

## Limitations

- **Requires random number generation**: Each rounding operation needs a uniform random sample. This is cheap on GPUs ($\sim$1 cycle per `curand` call) but adds hardware complexity for custom accelerators
- **Increased variance**: SR trades zero-bias for non-zero variance ($\leq \epsilon^2/4$ per element). Over many accumulations, the variance can grow as $O(N\epsilon^2)$ for sums of $N$ terms, potentially requiring higher-precision accumulators for long reductions
- **Not helpful for large values**: When $|x| \gg \epsilon$, both SR and RTN behave identically. SR only helps when updates are near or below the precision threshold
- **Hardware support varies**: As of 2025, native SR is available on Graphcore IPUs, NVIDIA Blackwell (FP4), and some FPGAs, but not on A100/H100 for general FP8 operations (must be emulated in software)
- **Forward pass impact is small**: Gupta et al. and Zamirai et al. both show that rounding in forward/backward computation has minimal impact on accuracy — the critical application is in **weight updates and gradient accumulation**, not general matmuls
- **Accumulator precision still matters**: SR on individual products doesn't help if the accumulator itself overflows or loses precision. FP8 matmuls on tensor cores use FP32 accumulators internally, which is essential

## Implementation Notes

```python
# Stochastic Rounding for FP8 weight updates
# Key: apply SR only at the quantization boundary (weight update step)

import torch

def stochastic_round_to_fp8(x: torch.Tensor, dtype=torch.float8_e4m3fn) -> torch.Tensor:
    """
    Stochastic rounding to FP8.
    x: input tensor in higher precision (BF16/FP32)
    Returns: tensor rounded to FP8 with unbiased rounding
    """
    # Quantize down (floor) and up (ceil) to nearest FP8 values
    x_floor = x.to(dtype).to(x.dtype)  # round-to-nearest then floor approximation
    x_ceil = x_floor + torch.sign(x - x_floor) * fp8_epsilon(x_floor)

    # Probability of rounding up = fractional position between floor and ceil
    eps = (x_ceil - x_floor).clamp(min=1e-30)
    prob_up = ((x - x_floor) / eps).clamp(0, 1)

    # Stochastic decision
    rand = torch.rand_like(prob_up)
    result = torch.where(rand < prob_up, x_ceil, x_floor)

    return result.to(dtype)


# In practice, SR is fused into the optimizer kernel:
# 1. Compute Adam/SGD update in FP32 accumulator
# 2. Add update to FP32 copy of weights (or use Kahan summation in BF16)
# 3. Quantize weights back to FP8 for next forward pass using SR
# 4. Quantize gradients to FP8 for all-reduce using SR
#
# The SR operation is ~3 instructions per element:
#   curand() -> compare -> select
# This is negligible compared to the matmul FLOPs.

# For NVIDIA Blackwell (native SR support):
# Use the FP4 conversion instruction with SR rounding mode flag
# No software emulation needed — handled in hardware pipeline
```

**GPU implementation considerations:**
- **Fuse SR into quantization kernels**: The random number generation and comparison should be part of the same kernel that performs the type conversion, avoiding extra HBM reads/writes
- **Use fast PRNG**: `curand_uniform()` in device code, or even simpler PRNGs like xorshift for non-cryptographic use. The quality of randomness matters less than the unbiasedness property
- **Per-tensor vs per-element SR**: Per-element SR gives the best theoretical properties, but per-block SR (one random offset per 128-element block) is cheaper and often sufficient in practice
- **Combine with block scaling**: In MXFP4/FP8 block formats, apply SR after block-wise scaling to maximize the effective dynamic range within each block

## References

- Gupta, S., Agrawal, A., Gopalakrishnan, K., Narayanan, P. "Deep Learning with Limited Numerical Precision." ICML, 2015. arXiv:1502.02551
- Xia, L., Anthonissen, M., Hochstenbach, M., Koren, B. "A Simple and Efficient Stochastic Rounding Method for Training Neural Networks in Low Precision." arXiv:2103.13445, 2021.
- Connolly, N., Bréhard, F., Sherard-Smith, T., Sherard-Smith, M. "Direct Quantized Training of Language Models with Stochastic Rounding." arXiv:2412.04787, 2024.
- Zamirai, P., Zhang, J., Aberger, C.R., De Sa, C. "Revisiting BFloat16 Training." arXiv:2010.06192, 2020.
- NVIDIA. "Pretraining Large Language Models with NVFP4." arXiv:2509.25149, 2025.

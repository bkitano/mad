# 263: Deferred Normalization–Linear Fusion — Hiding Collective Ops Behind Matmul

**Category**: kernel
**Gain type**: efficiency
**Source**: Salmani & Soloveychik (2025) — "LLM Inference Acceleration via Efficient Operation Fusion", d-Matrix. arXiv:2502.17728
**Paper**: [papers/deferred-normalization-fusion-inference.pdf]
**Documented**: 2026-02-16

## Description

In every transformer decoder block, normalization operations (LayerNorm, RMSNorm) and Softmax require **collective reductions** — aggregating all elements of a vector to compute a mean, variance, or sum-of-exponentials. These reductions are the primary latency bottleneck on distributed and in-memory compute architectures, accounting for approximately **20% of total inference latency** because they require spatial data aggregation across processing units that would otherwise operate independently.

The key observation is purely algebraic: **every normalization operation in a transformer is immediately followed by a matrix multiplication** (attention projections, MLP layers, or value aggregation). Since matrix multiplication commutes with scalar scaling, the normalization's reduction (which produces a scalar denominator) can be **deferred** — computed concurrently with the matmul rather than sequentially before it.

**The decomposition:** Any normalization followed by a linear layer can be split into:
1. An **elementwise sub-operation** (mean subtraction, exponentiation, etc.) — fused into the matmul
2. A **collective sub-operation** (computing the denominator: $\sigma$, $r(h)$, $\sum e^{x_i}$) — computed in parallel on a separate compute unit (e.g., SIMD) while the matmul runs on the main compute engine (e.g., DIMC)

The result is then a simple scalar division of the matmul output by the separately-computed denominator. Since division happens **after** the matmul, and the denominator computation runs **concurrently** with the matmul, the collective operation's latency is completely hidden.

**This is exact** — no approximation. The algebraic equivalence guarantees bit-identical results to the unfused computation, unlike DyT/Derf/TaperNorm which approximate or replace normalization.

**GPU relevance:** While the paper targets d-Matrix's Corsair in-memory compute architecture, the same algebraic decomposition applies to any system where reductions and matmuls can run concurrently — including GPU architectures with separate SIMD units, or systems with asynchronous compute capabilities (e.g., TMA + warp specialization on H100).

## Mathematical Form

**Fused LayerNorm + Linear:**

Standard computation (sequential):

$$
\mathbf{y} = \underbrace{\left(\frac{\mathbf{x} - \bar{x}\mathbf{1}}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta\right)}_{\text{LayerNorm}} \mathbf{F}
$$

where $\mathbf{x} \in \mathbb{R}^{1 \times n}$, $\mathbf{F} \in \mathbb{R}^{n \times m}$, $\bar{x} = \frac{1}{n}\sum_i x_i$, $\sigma^2 = \frac{1}{n}\|\mathbf{x} - \bar{x}\mathbf{1}\|^2$.

Rewrite LayerNorm in matrix form:

$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\sigma^2 + \epsilon}} \left(\mathbf{I} - \frac{1}{n}\mathbf{E}\right) \Gamma + \beta
$$

where $\Gamma = \text{Diag}(\gamma)$ and $\mathbf{E}$ is the all-ones matrix. Note $(\mathbf{I} - \frac{1}{n}\mathbf{E})\Gamma$ is a **static matrix** — precomputed once.

**Deferred fusion:**

$$
\mathbf{y}\mathbf{F} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \cdot \underbrace{\left(\mathbf{x}\left(\mathbf{I} - \frac{1}{n}\mathbf{E}\right)\Gamma\mathbf{F}\right)}_{\text{elementwise + matmul (main engine)}} + \underbrace{\beta\mathbf{F}}_{\text{precomputed bias}}
$$

The denominator $\sqrt{\sigma^2 + \epsilon}$ is computed **concurrently** on a separate unit while the matmul $\mathbf{x}(\mathbf{I} - \frac{1}{n}\mathbf{E})\Gamma\mathbf{F}$ runs on the main compute engine.

---

**Fused RMSNorm + Linear:**

Standard:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{n}\mathbf{x}\mathbf{x}^T + \varepsilon}} \odot \gamma
$$

Deferred fusion:

$$
\text{RMSNorm}(\mathbf{x}) \cdot \mathbf{F} = \frac{1}{\sqrt{\frac{1}{n}\mathbf{x}\mathbf{x}^T + \varepsilon}} \cdot \underbrace{(\mathbf{x} \, \Gamma \, \mathbf{F})}_{\text{matmul}}
$$

The elementwise scaling $\mathbf{x} \odot \gamma$ is absorbed into the matmul as $\mathbf{x} \, \Gamma \, \mathbf{F}$ where $\Gamma\mathbf{F}$ can be precomputed. The RMS denominator $r(\mathbf{x}) = \sqrt{\frac{1}{n}\|\mathbf{x}\|_2^2 + \varepsilon}$ runs concurrently.

---

**Fused Softmax + Value Matmul:**

Standard attention: $\mathbf{y} = \text{softmax}(\mathbf{x}) \, \mathbf{V}$

$$
\mathbf{y} = \left(\frac{[e^{x_1}, e^{x_2}, \ldots, e^{x_n}]}{\sum_i e^{x_i}}\right) \mathbf{V}
$$

Deferred fusion:

$$
\mathbf{y} = \frac{1}{\sum_i e^{x_i}} \cdot \underbrace{[e^{x_1}, e^{x_2}, \ldots, e^{x_n}] \, \mathbf{V}}_{\text{elementwise exp + matmul (main engine)}}
$$

The sum of exponentials $\sum_i e^{x_i}$ is computed concurrently while the unnormalized attention–value matmul runs on the main engine. Division is applied as a final scalar operation.

**Note:** In practice, the numerically stable form uses $e^{x_i - \max_j x_j}$ (online softmax, trick #83). The max computation requires a separate reduction but does not change the fusion structure — the sum-of-exp denominator is still deferred.

## Complexity

| Operation | Conventional (sequential) | Deferred fusion |
|-----------|--------------------------|-----------------|
| LayerNorm + Linear | $O(n)$ reduction **then** $O(nm)$ matmul | $O(nm)$ matmul **with** $O(n)$ reduction concurrent |
| RMSNorm + Linear | $O(n)$ reduction **then** $O(nm)$ matmul | $O(nm)$ matmul **with** $O(n)$ reduction concurrent |
| Softmax + V matmul | $O(n)$ reduction **then** $O(nm)$ matmul | $O(nm)$ matmul **with** $O(n)$ reduction concurrent |
| Total per decoder block | 3 sequential reductions on critical path | **0** reductions on critical path |

**Latency reduction:** ~15–20% total inference latency reduction on Corsair (d-Matrix DIMC architecture) for LLaMA 2/3 models.

**FLOPs:** Identical — no additional computation. The same operations execute, just reordered for concurrency.

**Accuracy:** Exact — algebraically equivalent. No approximation error.

**Memory:** The precomputed matrices $(\mathbf{I} - \frac{1}{n}\mathbf{E})\Gamma\mathbf{F}$ and $\Gamma\mathbf{F}$ replace the original $\mathbf{F}$ with same-shaped matrices. One additional scalar output (the denominator) is stored temporarily. Net memory overhead: negligible.

## Applicability

- **LLM inference acceleration:** Primary use case. Every decoder block has 2 normalization layers and 1 softmax, each followed by a matmul. Deferring all three removes ~20% latency on in-memory compute hardware.

- **Any transformer architecture:** The algebraic decomposition applies universally — GPT, LLaMA, Mistral, etc. Pre-norm and post-norm configurations both benefit (the paper describes the pre-norm case where the residual branches before normalization).

- **In-memory computing (DIMC) architectures:** Particularly impactful because collective operations require spatial aggregation across distributed memory banks, which is the primary bottleneck of analog/digital in-memory compute.

- **GPU inference with warp specialization:** On H100, one warp group could compute the reduction while another runs the matmul via warp specialization. This maps naturally to the two-pipeline architecture of modern GPU kernels.

- **Asynchronous compute engines:** Any system with multiple independent compute units (e.g., CPU SIMD + GPU, or multi-chiplet architectures) can exploit this concurrency.

- **Extends to SwiGLU / gated MLP:** The paper notes this fusion works with LLaMA's gated MLP (up-projection with SwiGLU gating), since the elementwise gating is absorbed into the fused elementwise sub-operation.

## Limitations

- **Requires concurrent compute capability:** The speedup comes from running the reduction in parallel with the matmul. On a system where these must execute sequentially (e.g., a single-issue scalar processor), there is no benefit.

- **GPU applicability is architecture-dependent:** On current NVIDIA GPUs, the matmul (tensor cores) and reduction (CUDA cores/SFU) share the same SM. True concurrency requires warp specialization or separate compute units. The 15–20% gain demonstrated on d-Matrix Corsair may not directly translate to GPU.

- **LayerNorm requires precomputing $(\mathbf{I} - \frac{1}{n}\mathbf{E})\Gamma\mathbf{F}$:** This is a dense $n \times m$ matrix that must be recomputed if $\gamma$ changes (fine-tuning). For RMSNorm, $\Gamma\mathbf{F}$ is simpler.

- **Online softmax max-computation:** The stable softmax requires a max reduction first, then a sum-of-exp reduction. The max cannot be deferred in the same way — it must complete before computing exponentials. The paper implicitly assumes the max is already known or uses a two-pass approach.

- **Primarily validated on inference:** The fusion is described for inference. During training, the backward pass requires the per-token statistics for gradient computation, so the fusion structure is more complex.

- **Paper is brief (5 pages):** Limited experimental details — latency reduction is reported as "approximately 15–20%" without per-layer breakdowns or comparison to optimized GPU baselines.

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeferredRMSNormLinear(nn.Module):
    """
    Fused RMSNorm + Linear layer with deferred normalization.

    Instead of: y = RMSNorm(x) @ W
    Computes:   y = (x @ (diag(gamma) @ W)) / rms(x)

    The denominator rms(x) can be computed concurrently with the matmul.
    On a GPU, this enables overlapping the reduction with the matmul
    via warp specialization or async compute.

    GPU notes:
    - Precompute gamma_W = diag(gamma) @ W once (or when gamma changes)
    - Main matmul: x @ gamma_W — full tensor core utilization
    - Concurrent reduction: rms = sqrt(mean(x^2) + eps) — on separate unit
    - Final: y = matmul_result / rms — elementwise scalar division
    - Eliminates the sequential dependency: norm -> matmul
    """
    def __init__(self, d_in, d_out, eps=1e-8):
        super().__init__()
        self.eps = eps

        # RMSNorm parameters
        self.gamma = nn.Parameter(torch.ones(d_in))

        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * (d_in ** -0.5))
        self.bias = nn.Parameter(torch.zeros(d_out))

        # Precomputed fused weight
        self.register_buffer('gamma_W', torch.zeros(d_in, d_out))
        self._update_fused_weight()

    def _update_fused_weight(self):
        """Precompute diag(gamma) @ W = gamma.unsqueeze(1) * W"""
        self.gamma_W = self.gamma.unsqueeze(1) * self.weight

    def forward(self, x):
        """
        x: (B, T, d_in)
        Returns: (B, T, d_out)
        """
        # In a real implementation, these two would run concurrently:

        # 1. Matmul (main compute engine / tensor cores)
        # Uses precomputed gamma_W = diag(gamma) @ W
        matmul_result = x @ self.gamma_W + self.bias  # (B, T, d_out)

        # 2. RMS reduction (concurrent on separate unit / warp)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 3. Final division (after both complete)
        return matmul_result / rms


class DeferredSoftmaxValue(nn.Module):
    """
    Fused Softmax + Value matmul with deferred normalization.

    Instead of: y = softmax(scores) @ V
    Computes:   y = (exp(scores - max) @ V) / sum(exp(scores - max))

    The sum-of-exp denominator is computed concurrently with the
    unnormalized exp(scores) @ V matmul.
    """
    @staticmethod
    def forward(scores, V):
        """
        scores: (B, H, T, S) attention logits (after QK^T/sqrt(d))
        V: (B, H, S, d_v) value matrix

        In a real implementation, steps 2 and 3 would run concurrently.
        """
        # 1. Max for numerical stability (must complete first)
        max_scores = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)  # (B, H, T, S)

        # 2. Unnormalized attention @ V (main compute engine)
        unnorm_attn_V = exp_scores @ V  # (B, H, T, d_v)

        # 3. Sum of exponentials (concurrent reduction)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # (B, H, T, 1)

        # 4. Final normalization (after both complete)
        return unnorm_attn_V / sum_exp


# Example: applying to a full transformer block
def fuse_rmsnorm_into_projection(rmsnorm_gamma, projection_weight):
    """
    Precompute the fused weight matrix for deferred RMSNorm-Linear fusion.

    gamma: (d,) RMSNorm scale parameters
    W: (d, d_out) projection weight matrix

    Returns: gamma_W = diag(gamma) @ W, shape (d, d_out)
    """
    return rmsnorm_gamma.unsqueeze(1) * projection_weight
```

## GPU Efficiency Analysis

**Memory Access Pattern:**
- The fused matmul $\mathbf{x} \, \Gamma \mathbf{F}$ has identical memory access to a standard matmul — coalesced reads of $\mathbf{x}$ and $\Gamma\mathbf{F}$
- The concurrent reduction reads $\mathbf{x}$ from the same memory (potentially cached)
- No additional HBM round-trips vs. sequential execution
- One extra scalar write (the denominator) — negligible

**Parallelism:**
- The matmul fully utilizes tensor cores — no change from standard execution
- The reduction is a simple $O(n)$ sum, easily parallelizable across a few warps
- No warp divergence in either sub-operation
- The concurrency between matmul and reduction is the source of speedup

**Arithmetic Intensity:**
- Matmul: $O(nm)$ FLOPs / $O(nm + n)$ bytes — unchanged, compute-bound for large $m$
- Reduction: $O(n)$ FLOPs / $O(n)$ bytes — memory-bound, but hidden behind matmul
- Net effect: same total FLOPs, higher utilization (no idle cycles waiting for reduction)

**Hardware:**
- Maps to warp-specialized kernels on H100 (one warp group for matmul, one for reduction)
- On d-Matrix Corsair: DIMC handles matmul, SIMD handles reduction — natural hardware concurrency
- $\Gamma\mathbf{F}$ precomputation is a one-time cost — negligible
- Compatible with all precision formats (FP16, BF16, FP8) — the scalar division is in higher precision

## References

- Salmani, M. & Soloveychik, I. (2025). LLM Inference Acceleration via Efficient Operation Fusion. arXiv:2502.17728.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.
- Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867.
- d-Matrix (2024). d-Matrix Corsair: Performance and efficiency for AI inference at scale. White Paper.

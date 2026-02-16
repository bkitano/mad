# 221: Kahan Compensated Summation for Low-Precision Training

**Category**: stability
**Gain type**: efficiency
**Source**: Zamirai et al., "Revisiting BFloat16 Training" (arXiv 2020); Kahan, "Further remarks on reducing truncation errors" (1965)
**Paper**: papers/revisiting-bfloat16-kahan-summation.pdf
**Documented**: 2026-02-15

## Description

Kahan compensated summation is a numerical analysis technique that enables **pure 16-bit floating-point training** to match 32-bit training accuracy by tracking and compensating rounding errors in model weight updates. The key insight is that **nearest rounding during weight updates is the primary bottleneck** for 16-bit training accuracy — not rounding during forward/backward compute. When gradient updates are small relative to weight magnitudes, nearest rounding silently cancels them, creating an irreducible convergence floor.

Kahan summation maintains a 16-bit auxiliary "error compensation" buffer $c_t$ alongside each weight. At each step, the compensation from previous rounding errors is injected into the current update before accumulation. A reverse subtraction then captures the new rounding error for the next step. Small updates that would individually be rounded away instead accumulate in $c_t$ until they're large enough to affect the weights.

**Why it matters for GPU pretraining**: Enables pure BF16 training (weights, activations, gradients, optimizer states all in 16-bit) while matching FP32 accuracy. This eliminates the need for 32-bit FPUs entirely, providing up to 2x memory reduction for weights+optimizer vs mixed precision, 10-30% throughput improvement (depending on GPU count), and reduced hardware complexity for custom accelerators.

## Mathematical Form

**The Problem — Nearest Rounding Convergence Floor:**

Let $Q(\cdot)$ denote nearest rounding to BF16. The weight update with nearest rounding is:

$$
w_{t+1} = Q\!\left(w_t - \alpha \nabla f_{\sigma(t)}(w_t)\right)
$$

**Theorem 1 (Convergence floor).** For loss functions with $L$-Lipschitz continuous gradients and optimal weights $w^*$, if the model update $|\alpha \nabla f_{\sigma(t)}(w_t)_j|$ is smaller than half the spacing between $w_t$ and its nearest representable BF16 neighbor, the update is **entirely cancelled**. The distance to optimum is bounded below by:

$$
\|w_t - w^*\| \geq \min\!\left(\frac{\epsilon(1 - \alpha L)}{\alpha L + \epsilon} \cdot \min_j |w_j^*|, \; \|w_0 - w^*\|\right)
$$

where $\epsilon$ is the machine epsilon of the floating-point format. This bound is $\mathcal{O}(\epsilon)$ and gets *worse* as step size decreases — the opposite of exact arithmetic.

**Solution 1 — Stochastic Rounding:**

$$
Q_{\mathrm{SR}}(a) = \begin{cases} a_u & \text{with probability } \frac{a - a_l}{a_u - a_l} \\ a_l & \text{otherwise} \end{cases}
$$

where $a_u = \min_{x \geq a, x \in \mathbb{S}} x$ and $a_l = \max_{x \leq a, x \in \mathbb{S}} x$. This makes rounded weights an **unbiased estimator** of the true weights: $\mathbb{E}[Q_{\mathrm{SR}}(a)] = a$.

**Solution 2 — Kahan Summation (Algorithm 1 from paper):**

Given gradient update $u_{t+1} = -\alpha \nabla f_{\sigma(t)}(w_t)$ and compensation buffer $c_t \in \mathbb{R}^d$ (initialized to $0$):

$$
\begin{aligned}
y_{t+1} &= u_{t+1} - c_t & \quad \triangleright \text{ compensate update with accumulated error} \\
s_{t+1} &= w_t + y_{t+1} & \quad \triangleright \text{ accumulate into weights (rounded by hardware)} \\
c_{t+1} &= (s_{t+1} - w_t) - y_{t+1} & \quad \triangleright \text{ measure new rounding error} \\
w_{t+1} &= s_{t+1} & \quad \triangleright \text{ update weights}
\end{aligned}
$$

**Key Definitions:**

- $w_t \in \mathbb{R}^d$ — model weights in BF16
- $c_t \in \mathbb{R}^d$ — compensation buffer in BF16 (tracks accumulated rounding errors)
- $u_{t+1} \in \mathbb{R}^d$ — raw gradient update (e.g., $-\alpha g_t$ for SGD, or Adam update)
- $\epsilon$ — machine epsilon (BF16: $\epsilon \approx 2^{-8} \approx 3.9 \times 10^{-3}$; FP16: $\epsilon \approx 2^{-11} \approx 4.9 \times 10^{-4}$)
- $Q(\cdot)$ — nearest rounding to the target precision format
- $\mathbb{S}$ — set of representable values in the floating-point format

**Effective precision increase:** The compensation buffer effectively doubles the mantissa bits for the accumulation, increasing effective summation precision from 8 bits (BF16) to approximately 16 bits.

## Complexity

| Operation | Mixed Precision (BF16+FP32) | Pure BF16 + Kahan |
|-----------|----------------------------|-------------------|
| Weight memory | $2 \times$ (BF16 activations + FP32 master weights) | $2 \times$ BF16 (weights + compensation) |
| Optimizer state memory | FP32 | BF16 |
| FPU requirement | Both 16-bit and 32-bit | 16-bit only |
| Per-step compute | Baseline | +3 elementwise BF16 ops per weight |

**Memory comparison (Table 2):**

| Component | 32-bit | Mixed Precision | 16-bit + Kahan |
|-----------|--------|-----------------|----------------|
| Weights | 32-bit | 32-bit (master copy) | 16-bit |
| Optimizer state | 32-bit | 32-bit | 16-bit |
| Activations & grads | 32-bit | 16-bit | 16-bit |
| Requires 32-bit FPU | Yes | Yes | **No** |
| Compensation buffer | N/A | N/A | 16-bit ($1\times$ weight size) |

Net weight+optimizer memory: Kahan uses $2\times$ 16-bit for weights (weights + compensation) vs mixed precision's 32-bit master + 16-bit copy. This is a **2x reduction** in weight storage.

## Applicability

- **All model architectures**: Validated on ResNet-18/50, BERT-Base, DLRM, DeepSpeech2 across classification, NLI, recommendation, and speech
- **Transformer pretraining**: Directly applicable to LLM pretraining where optimizer states dominate memory (Adam has 2 states per parameter)
- **Custom accelerator design**: Enables hardware with only 16-bit FPUs (no 32-bit units needed), reducing chip area, power, and latency by $1.5\text{-}3\times$
- **SSM training**: Applicable to any model using SGD/Adam — the technique is architecture-agnostic, operating only on the weight update step
- **Gradient accumulation**: Particularly beneficial when accumulating many small microbatch gradients that would individually be rounded away
- **Complementary to trick 190 (SageAttention2)**: Kahan summation stabilizes weight updates while SageAttention2 stabilizes attention computation — they target different precision bottlenecks

## Limitations

- **2x weight memory overhead**: The compensation buffer $c_t$ doubles the memory for weights (though optimizer states and activations are unaffected, and total memory is still less than mixed precision)
- **Requires hardware support**: The error measurement step $(s_{t+1} - w_t) - y_{t+1}$ relies on the specific rounding behavior of FMAC units; this works correctly on standard hardware but must be verified on novel accelerators
- **Sequential dependency**: The compensation must be applied before the next weight update — cannot overlap compensation with gradient computation (though this is a single elementwise pass, not a bottleneck)
- **Stochastic rounding is simpler**: If hardware supports stochastic rounding natively, it requires no auxiliary buffer and achieves comparable (though slightly lower) accuracy; Kahan summation is the deterministic alternative
- **Not applicable to forward/backward compute**: The paper shows rounding in forward/backward has negligible impact — Kahan summation is specifically for the weight update step where rounding is catastrophic

## Implementation Notes

```python
# Kahan Summation for SGD weight updates in pure BF16
# All tensors stored in BF16; FMAC units use 32-bit accumulators internally

class KahanSGD:
    def __init__(self, params, lr):
        self.lr = lr
        # Initialize compensation buffers to zero
        self.compensation = {p: torch.zeros_like(p) for p in params}

    def step(self, params, grads):
        for p, g in zip(params, grads):
            c = self.compensation[p]

            # Raw update (in BF16)
            u = -self.lr * g

            # Compensate: inject accumulated rounding error
            y = u - c                    # BF16 subtraction

            # Accumulate: add compensated update to weights
            s = p + y                    # BF16 addition (rounded by hardware)

            # Measure: capture new rounding error
            # Key insight: (s - p) recovers what was actually added,
            # subtracting y gives the rounding error
            c_new = (s - p) - y          # BF16: exact if |s-p| >> |y-actual|

            # Update
            p.copy_(s)
            self.compensation[p] = c_new

# For AdamW, apply Kahan summation to the final weight update step:
# u = -lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
# Then apply the same y, s, c_new pattern above

# GPU kernel fusion opportunity:
# The 3 elementwise ops (y, s, c_new) can be fused into a single
# kernel that reads (w, c, update) and writes (w_new, c_new),
# costing 2 loads + 2 stores — same memory traffic as a standard
# weight update kernel that reads (w, update) and writes (w_new),
# since the compensation buffer is hot in L2 cache.
```

**Practical notes for GPU implementation:**
- The 3 elementwise operations (compensate, accumulate, measure) should be **fused into a single CUDA kernel** with the optimizer step to avoid extra HBM round-trips
- The compensation buffer $c_t$ has good temporal locality (accessed every step) and will stay in L2 cache
- On current GPUs (A100/H100), the bottleneck is typically the Adam state reads, not the extra BF16 ops
- For gradient accumulation across microbatches, apply Kahan summation to the gradient accumulator as well
- The `optimi` library (https://optimi.benjaminwarner.dev/kahan_summation/) provides production PyTorch implementations

## References

- Zamirai, P., Zhang, J., Aberger, C.R., De Sa, C. "Revisiting BFloat16 Training." arXiv:2010.06192, 2020.
- Kahan, W. "Further remarks on reducing truncation errors." Communications of the ACM, 8:40, 1965.
- Micikevicius, P. et al. "Mixed Precision Training." arXiv:1710.03740, 2017.
- Hopkins, M. et al. "Stochastic rounding and reduced-precision fixed-point arithmetic for solving neural ODEs." Phil. Trans. Royal Society A, 2020.

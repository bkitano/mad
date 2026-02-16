# 236: FP8 Precision Decoupling and Auto-Scaling

**Category**: stability
**Gain type**: efficiency
**Source**: Peng, Wu, Wei et al. (2023) — "FP8-LM: Training FP8 Large Language Models" (arXiv:2310.18313)
**Paper**: [papers/fp8-lm-training.pdf]
**Documented**: 2026-02-15

## Description

FP8-LM introduces two complementary numerical stability techniques that enable **full FP8 training** of large language models (7B–175B parameters) across compute, gradients, optimizer states, and distributed communication — achieving 75% faster training and 39% memory reduction compared to BF16 on H100 GPUs.

The two key techniques are:

1. **Precision Decoupling**: Strategically assigns different precisions to different components of the Adam optimizer based on their sensitivity to quantization error. Master weights stay in FP16 (with tensor scaling), first-order moments go to FP8, while second-order moments remain in FP16 to avoid underflow from squaring small gradients.

2. **Automatic Scaling (Auto-Scale)**: A dynamic scaling factor $\mu$ applied to gradients before FP8 all-reduce communication, which adaptively adjusts based on overflow/underflow statistics to keep gradient values within the FP8 representable range. Unlike delayed scaling (which uses historical amax), auto-scaling uses a threshold-based decision rule on the current batch.

Together these reduce optimizer memory from 16 bytes/parameter (standard Adam) to 6 bytes/parameter, and communication volume by 63–65%.

## Mathematical Form

### Precision Decoupling

**Standard Adam memory layout** (16 bytes/param):

$$
\underbrace{4}_{\text{master weights (FP32)}} + \underbrace{4}_{\text{gradients (FP32)}} + \underbrace{4 + 4}_{\text{Adam states } m, v \text{ (FP32)}} = 16 \text{ bytes}
$$

**FP8-LM decoupled layout** (6 bytes/param):

$$
\underbrace{2}_{\text{master weights (FP16+scale)}} + \underbrace{1}_{\text{gradients (FP8)}} + \underbrace{1 + 2}_{\text{Adam: } m \text{ (FP8)}, v \text{ (FP16)}} = 6 \text{ bytes}
$$

**Key insight**: The first-order moment $m_t$ can tolerate FP8 quantization because Adam's update direction is more important than its magnitude. The second-order moment $v_t$ requires higher precision because:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

The squaring operation $g_t^2$ can produce values below the FP8 minimum representable value ($\sim 2^{-14}$ for E4M3), causing catastrophic underflow.

### Auto-Scaling for FP8 Gradient Communication

**Problem**: During distributed data-parallel training, gradients must be aggregated across $N$ GPUs. Two naive approaches fail:

*Pre-scaling* (divide before sum):

$$
g = g_1/N + g_2/N + \cdots + g_N/N
$$

This causes **underflow** because dividing small FP8 gradients by large $N$ pushes values below representable range.

*Post-scaling* (sum before divide):

$$
g = (g_1 + g_2 + \cdots + g_N) / N
$$

This causes **overflow** because summing many FP8 values exceeds the maximum representable value.

**Auto-scaling solution**: Introduce a dynamic scaling factor $\mu$ applied before all-reduce:

$$
g'_i = \mu \cdot g_i
$$

**Adaptation rule**: Compute the proportion of values in $g'_i$ that saturate the FP8 maximum representable value. If this proportion exceeds a threshold $\delta = 0.001$ (0.1%):

$$
\mu \leftarrow \mu / 2 \quad \text{(halve scale to prevent overflow)}
$$

If the proportion stays below $\delta$ for a sustained period:

$$
\mu \leftarrow \mu \times 2^{1/1000} \quad \text{(exponentially increase over 1000 steps)}
$$

### Distributed Scaling Factor Unification

For tensor-parallel all-reduce, each GPU $i$ has gradient tensor $g'_i$ with local scaling factor $s'_i$. To enable standard NCCL all-reduce (which doesn't support per-tensor scaling), compute a global minimum:

$$
s'_g = \min(s'_1, s'_2, \ldots, s'_N)
$$

Requantize all tensors to this shared scale:

$$
g''_i = \text{FP8}\left(s'_g \cdot (g'_i / s'_i)\right)
$$

Then standard all-reduce proceeds:

$$
g = g''_1 + g''_2 + \cdots + g''_N, \qquad s = N \cdot s'_g
$$

The final collected gradient is $g / s$, which is equivalent to dividing the sum by $N$ in theory.

### FP8 Tensor Scaling

Each FP8 tensor is stored as a pair $(t, s)$ where $t$ is the FP8 data and $s$ is an FP32 scaling factor. The actual value is $t/s$. The scaling factor is chosen to map the tensor's maximum absolute value to the FP8 maximum:

$$
s = \frac{\text{FP8\_MAX}}{\max(|T|)}
$$

where $\text{FP8\_MAX} = 448$ for E4M3 format and $\text{FP8\_MAX} = 57344$ for E5M2 format.

## Complexity

| Operation | BF16 Baseline | FP8-LM |
|-----------|--------------|--------|
| GEMM compute | BF16 tensor cores | FP8 tensor cores (2× throughput) |
| Optimizer memory | 16 bytes/param | 6 bytes/param (2.6× reduction) |
| Gradient communication | 16-bit all-reduce | 8-bit all-reduce (63–65% reduction) |
| Activation communication | 16-bit all-gather | 8-bit all-gather (34% reduction) |
| Training throughput (175B) | 22.4 samples/s | 39.3 samples/s (75% faster) |
| GPU memory (175B) | 66.1 GB | 57.7 GB (13% reduction) |

**Memory per parameter:**
- Standard Adam (FP32): $O(16)$ bytes
- BF16 mixed precision: $O(16)$ bytes (FP32 master + FP32 optimizer)
- FP8-LM: $O(6)$ bytes

## GPU Efficiency Analysis

**Memory Access Pattern:**
- FP8 tensors are 2× denser than FP16, improving cache utilization for bandwidth-bound ops
- Scaling factors are per-tensor scalars (single FP32 value per tensor) — negligible memory overhead
- All access patterns remain coalesced; FP8 quantization/dequantization is elementwise

**Parallelism:**
- FP8 GEMMs directly leverage H100 FP8 tensor cores (WGMMA instructions)
- Auto-scaling decision is a single reduction (max over tensor) + scalar comparison — fully parallel
- Global scale unification requires one scalar all-reduce per tensor (negligible vs. gradient all-reduce)
- No sequential bottlenecks; all operations are embarrassingly parallel

**Tensor Core Utilization:**
- H100 FP8 tensor cores: 1978 TFLOPS (E4M3×E5M2) vs. 989 TFLOPS (BF16)
- Measured MFU: 34.2% for FP8 vs. 24.9% for TE (37.4% improvement over NVIDIA Transformer Engine)
- The key insight: FP8-LM applies FP8 to **all** GEMMs (not just compute, but also communication), maximizing the benefit

**HBM Bandwidth Reduction:**
- Activation storage: 50% reduction (FP8 vs FP16)
- Weight loads: 50% reduction for FP8 weights during forward/backward
- Gradient communication: 63–65% reduction in all-reduce volume
- Enables training 175B model on 32 H100s (80GB) with 4K context — impossible with BF16 TE

## Applicability

- **LLM pretraining**: Validated on GPT-7B, GPT-13B, GPT-175B with equivalent accuracy to BF16
- **Instruction tuning (SFT)**: 27% training speed improvement, comparable performance to BF16 Vicuna
- **RLHF**: 32% weight memory reduction, 62% optimizer memory reduction with comparable alignment performance
- **Transformers**: Direct applicability to any transformer-based architecture using standard Adam optimizer
- **SSMs**: The precision decoupling principle applies to any model using Adam — the key insight that $m_t$ tolerates FP8 while $v_t$ needs FP16 is architecture-agnostic
- **Distributed training**: Benefits scale with number of GPUs due to communication volume reduction

## Limitations

- **Hardware requirement**: Requires H100 (Hopper) or newer GPUs with FP8 tensor core support. Not applicable to A100 or older hardware
- **FP8 format sensitivity**: E4M3 (4 exponent, 3 mantissa bits) is used for forward/weights; E5M2 (5 exponent, 2 mantissa bits) is better for gradients due to wider dynamic range. Format selection matters
- **Second-order moment precision**: $v_t$ cannot be reduced to FP8 without divergence — this is a hard constraint. Attempting FP8 for $v_t$ causes training loss divergence
- **Master weight precision**: FP8 master weights also cause divergence. FP16 with tensor scaling is the minimum viable precision for master weights
- **Scaling factor overhead**: Per-tensor scaling factors add a small amount of metadata. For ZeRO-style sharding, tensors must be distributed whole (not split) to keep scaling factors consistent, which can cause mild memory imbalance across GPUs
- **Not composable with all optimizers**: Validated primarily with AdamW. Other optimizers with different state structures may have different precision sensitivity profiles
- **Auto-scaling latency**: The threshold-based adaptation has a 1000-step increase period, meaning recovery from underflow is slow

## Implementation Notes

```python
import torch

class FP8Tensor:
    """FP8 tensor with per-tensor scaling factor."""
    def __init__(self, data_fp8, scale_fp32):
        self.data = data_fp8     # uint8 storage, FP8 E4M3 or E5M2
        self.scale = scale_fp32  # single FP32 scalar

    @staticmethod
    def quantize(tensor_fp32):
        FP8_MAX = 448.0  # E4M3 max
        amax = tensor_fp32.abs().max()
        scale = FP8_MAX / amax.clamp(min=1e-12)
        data = (tensor_fp32 * scale).to(torch.float8_e4m3fn)
        return FP8Tensor(data, scale)

    def dequantize(self):
        return self.data.float() / self.scale

class FP8AutoScaler:
    """Auto-scaling for FP8 gradient all-reduce."""
    def __init__(self, threshold=0.001, warmup=25):
        self.mu = 1.0
        self.threshold = threshold
        self.steps_below = 0

    def scale_gradients(self, grad):
        scaled = grad * self.mu
        # Check overflow ratio
        overflow_ratio = (scaled.abs() >= FP8_MAX).float().mean()
        if overflow_ratio > self.threshold:
            self.mu /= 2.0
            self.steps_below = 0
        else:
            self.steps_below += 1
            if self.steps_below >= 1000:
                self.mu *= 2.0 ** (1/1000)
        return FP8Tensor.quantize(scaled)

class FP8Adam:
    """Adam with precision-decoupled FP8 states."""
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95)):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.state = {}
        for p in params:
            self.state[p] = {
                'master_weight': p.data.to(torch.float16),  # FP16
                'm': torch.zeros_like(p, dtype=torch.float8_e4m3fn),  # FP8
                'v': torch.zeros_like(p, dtype=torch.float16),  # FP16
                'm_scale': torch.tensor(1.0),  # FP32 scaling factor
            }

    def step(self, param, grad_fp8):
        s = self.state[param]
        grad = grad_fp8.dequantize()

        # Update moments with precision-appropriate formats
        m = s['m'].float() / s['m_scale']
        m = self.beta1 * m + (1 - self.beta1) * grad
        s['m'] = FP8Tensor.quantize(m)  # back to FP8

        v = s['v'].float()
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        s['v'] = v.to(torch.float16)  # stays FP16 (squaring!)

        # Update master weights in FP16
        update = m / (v.sqrt() + 1e-8)
        s['master_weight'] -= self.lr * update.to(torch.float16)
```

## References

- Peng, H., Wu, K., Wei, Y. et al. (2023). "FP8-LM: Training FP8 Large Language Models." arXiv:2310.18313.
- Micikevicius, P. et al. (2022). "FP8 formats for deep neural networks." arXiv:2209.05433.
- Micikevicius, P. et al. (2018). "Mixed Precision Training." ICLR 2018. arXiv:1710.03740.
- NVIDIA (2022). "Nvidia H100 Tensor Core GPU Architecture."
- Sun, X. et al. (2019). "Hybrid 8-bit floating point (HFP8) training and inference for deep neural networks." NeurIPS 2019.
- FP8-LM open-source framework: https://aka.ms/MS.AMP

# 252: RG-LRU Real-Gated Linear Recurrent Unit

**Category**: parallelization
**Gain type**: efficiency
**Source**: De, Smith, Fernando et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models" (Google DeepMind, 2024)
**Paper**: papers/griffin-rg-lru.pdf
**Documented**: 2026-02-16

## Description

The Real-Gated Linear Recurrent Unit (RG-LRU) is a diagonal linear recurrence with two input-dependent gates (recurrence gate and input gate) that achieves training efficiency competitive with Transformers while providing $O(1)$ per-step inference with a fixed-size state. The key design insight is that **neither gate depends on the previous hidden state $h_{t-1}$**, only on the current input $x_t$. This means all gate values can be pre-computed in parallel across the entire sequence before executing the recurrence, making the recurrence itself a simple element-wise linear scan amenable to either:

1. **Linear scan kernel**: A custom Pallas/CUDA kernel that processes the recurrence sequentially but keeps $h$ in VMEM/SRAM throughout, minimizing memory transfers. This is 3x faster than a naive implementation because the RG-LRU is **memory-bound** (only 0.75 FLOPs/byte), so minimizing HBM round-trips dominates.

2. **Parallel scan**: Theoretically applicable since the recurrence is linear ($h_t = a_t \odot h_{t-1} + b_t$), but empirically slower on TPU-v3 due to the random-access memory pattern of the prefix-sum tree recombination.

The RG-LRU is embedded in a **recurrent block** (Conv1D + GeLU + RG-LRU on one branch, gated by another branch), forming the temporal mixing layer of Griffin. The critical architectural trick is **interleaving recurrent blocks with local sliding-window attention blocks** (ratio 2:1), combining the recurrence's constant-memory state with attention's precise short-range lookup. The recurrent weight matrices use **block-diagonal structure** for model-parallel sharding, avoiding inter-device communication.

## Mathematical Form

**Core RG-LRU Equations:**

$$
r_t = \sigma(W_a x_t + b_a) \quad \text{(recurrence gate)}
$$

$$
i_t = \sigma(W_x x_t + b_x) \quad \text{(input gate)}
$$

$$
a_t = a^{c \cdot r_t} \quad \text{(input-dependent decay)}
$$

$$
h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t)
$$

where $a = \sigma(\Lambda)$ is a learnable parameter per channel with $0 \leq a \leq 1$, $c = 8$ is a fixed constant that amplifies the gating range, and $\sigma$ is the sigmoid function. The $\sqrt{1 - a_t^2}$ normalization ensures the recurrence preserves signal magnitude.

**Key Definitions:**

- $x_t \in \mathbb{R}^{D_{\text{RNN}}}$ — input at time $t$ (from Conv1D + GeLU branch)
- $h_t \in \mathbb{R}^{D_{\text{RNN}}}$ — hidden state (element-wise / diagonal)
- $W_a, W_x \in \mathbb{R}^{D_{\text{RNN}} \times D_{\text{RNN}}}$ — gate weight matrices (block-diagonal for sharding)
- $a \in (0, 1)^{D_{\text{RNN}}}$ — learnable base decay, parameterized as $\sigma(\Lambda)$
- $r_t \in (0, 1)^{D_{\text{RNN}}}$ — recurrence gate (modulates decay strength)
- $i_t \in (0, 1)^{D_{\text{RNN}}}$ — input gate (scales incoming signal)
- $c = 8$ — constant amplifying the gating range ($a^{cr_t}$ maps $a$ from $(0,1)$ to a wider effective range)

**Parallel scan formulation:** Since $a_t$ and $i_t$ depend only on $x_t$, they can be pre-computed for all $t$ in parallel. The recurrence becomes:

$$
h_t = \alpha_t \odot h_{t-1} + \beta_t, \quad \alpha_t = a_t, \quad \beta_t = \sqrt{1 - a_t^2} \odot (i_t \odot x_t)
$$

This is a first-order linear recurrence solvable via parallel prefix scan with associative operator:

$$
(\alpha_i, \beta_i) \bullet (\alpha_j, \beta_j) = (\alpha_i \odot \alpha_j, \; \alpha_j \odot \beta_i + \beta_j)
$$

**Log-space computation for stability:**

In practice, $a_t = a^{c \cdot r_t}$ is computed in log-space as $\exp(c \cdot r_t \cdot \log(a))$ to avoid numerical underflow for large $c \cdot r_t$ products.

**Recurrent block architecture:**

$$
y_t = \text{Linear}(\text{RG-LRU}(\text{Conv1D}(\text{Linear}(x_t))) \odot \text{GeLU}(\text{Linear}(x_t)))
$$

The two branches (Conv1D + RG-LRU vs. GeLU gate) are computed in parallel and merged by element-wise multiplication, analogous to the gated architecture of Mamba.

**Griffin hybrid pattern (interleaved layers):**

$$
[\text{Recurrent} \to \text{MLP}] \to [\text{Recurrent} \to \text{MLP}] \to [\text{LocalMQA} \to \text{MLP}] \to \text{repeat}
$$

Two recurrent blocks per one local sliding-window MQA block (window size 1024).

**Initialization:**

$\Lambda$ is initialized such that $a^c = \sigma(\Lambda)^c$ is uniformly distributed between $0.9$ and $0.999$, following LRU initialization practices. This ensures a range of time constants across channels.

## Complexity

| Operation | Global MQA (Transformer) | RG-LRU (Griffin) |
|-----------|------------------------|-----------------|
| Training (per step) | $O(T^2 D)$ | $O(T D_{\text{RNN}})$ |
| Inference (per step) | $O(T)$ (KV cache read) | $O(D_{\text{RNN}})$ constant |
| Inference memory | $O(T \cdot D)$ (KV cache grows) | $O(D_{\text{RNN}})$ (fixed state) |
| Arithmetic intensity | High (matmul-dominated) | Low (0.75 FLOPs/byte, memory-bound) |

**Memory:** The RG-LRU state is diagonal ($D_{\text{RNN}}$ elements), so inference memory is $O(D_{\text{RNN}})$ constant regardless of sequence length. For local attention blocks with window $w = 1024$, the KV cache is bounded at $O(w \cdot d)$.

**Measured training speed (TPU-v3, tokens/step relative to MQA Transformer at 2K):**

| Model Scale | MQA (2K) | Griffin (2K) | Griffin (4K) | Griffin (8K) |
|-------------|----------|--------------|--------------|--------------|
| 400M | 1.00x | 0.93x | 0.98x | 1.00x (Transformer slows to 0.68x) |
| 1B | 1.00x | 0.97x | 0.99x | 1.01x |
| 7B | 1.00x | 1.05x | 1.05x | 1.07x |

Griffin matches or exceeds Transformer training throughput, with the advantage growing at longer sequences because the Transformer's $O(T^2)$ attention dominates while Griffin stays $O(T)$.

**Measured inference throughput (1B model, max tokens/s decoded):**

| Tokens Decoded | MQA | Griffin | Hawk | Griffin/MQA Speedup |
|----------------|-----|---------|------|---------------------|
| 512 | ~3000 | ~3300 | ~3500 | 1.1x |
| 1024 | ~3000 | ~4000 | ~5500 | 1.3x |
| 2048 | ~2800 | ~9000 | ~11000 | 3.2x |
| 4096 | ~2000 | ~13000 | ~15000 | 6.5x |

## Applicability

- **Language modeling at scale**: Griffin matches Llama-2 (13B, 2T tokens) performance at 7B/14B with only 300B tokens — 7x fewer tokens needed.
- **Drop-in attention replacement**: The recurrent block is designed as a direct replacement for MQA blocks, using the same residual + RMSNorm structure.
- **Long-context extrapolation**: Griffin extrapolates to sequences significantly longer than training length, unlike Transformers which degrade rapidly.
- **Efficient distributed training**: Block-diagonal gate weights ($W_a$, $W_x$ split into 16 blocks) enable Megatron-style sharding with no inter-device communication for the recurrent block.
- **Memory-efficient inference**: Fixed $O(D_{\text{RNN}})$ state enables larger batch sizes on a single device vs. growing KV cache, directly improving throughput.

## Limitations

- **Memory-bound training**: The RG-LRU has only 0.75 FLOPs/byte arithmetic intensity, far below GPU tensor core capacity (~312 TFLOPs on H100). Training is dominated by HBM bandwidth, not compute. Custom fused kernels (Pallas) are essential — naive implementation is 3x slower.
- **Linear scan preferred over parallel scan**: Empirically on TPU-v3, the sequential linear scan with VMEM-resident state is faster than parallel prefix scan due to the random-access memory pattern of tree recombination. This means the recurrence is still fundamentally sequential over time — the trick is making each step very fast via memory optimization, not parallelizing across time.
- **Diagonal state limits expressivity**: The element-wise recurrence $h_t = a_t \odot h_{t-1} + b_t$ means channels don't interact in the recurrence, limiting the model's ability to perform cross-channel state mixing. This is why the hybrid with local attention is important.
- **Local attention still needed**: Pure Hawk (recurrent only) underperforms Griffin (recurrent + local attention) on retrieval and copying tasks. The fixed-size recurrent state cannot perfectly recall arbitrary past tokens.
- **TPU-specific kernel**: The Pallas linear scan kernel is optimized for TPU-v3 VMEM. Porting to NVIDIA GPUs requires different optimization strategies (shared memory tiling, warp-level primitives).
- **Not compatible with convolution-based parallelization**: The input-dependent gating $a_t = a^{c \cdot r_t}$ makes the system non-LTI (Linear Time Invariant), so it cannot be computed via FFT-based convolution like S4.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

class RGLRU(torch.nn.Module):
    """Real-Gated Linear Recurrent Unit."""

    def __init__(self, d_rnn, num_blocks=16):
        super().__init__()
        self.d_rnn = d_rnn
        # Block-diagonal gate weights for model parallelism
        self.W_a = torch.nn.Linear(d_rnn, d_rnn, bias=True)
        self.W_x = torch.nn.Linear(d_rnn, d_rnn, bias=True)
        # Learnable base decay (parameterized in log-sigmoid space)
        # Initialize so a^c is uniform in [0.9, 0.999]
        self.log_a = torch.nn.Parameter(
            torch.log(-torch.log(torch.linspace(0.9, 0.999, d_rnn)))
        )
        self.c = 8.0  # fixed amplification constant

    def forward(self, x, h_prev=None):
        """
        x: [B, T, D] input sequence
        h_prev: [B, D] previous hidden state (None for training)
        Returns: [B, T, D] output, [B, D] final hidden state
        """
        B, T, D = x.shape

        # Pre-compute all gates in parallel (no h_{t-1} dependency!)
        r = torch.sigmoid(self.W_a(x))   # [B, T, D] recurrence gate
        i = torch.sigmoid(self.W_x(x))   # [B, T, D] input gate

        # Compute input-dependent decay in log-space for stability
        a = torch.sigmoid(self.log_a)     # [D] base decay
        log_a = torch.log(a)
        # a_t = a^{c * r_t} = exp(c * r_t * log(a))
        a_t = torch.exp(self.c * r * log_a.unsqueeze(0).unsqueeze(0))  # [B, T, D]

        # Input contribution with magnitude normalization
        b_t = torch.sqrt(1 - a_t ** 2) * (i * x)  # [B, T, D]

        # Sequential linear scan (memory-efficient)
        # In practice: custom kernel keeps h in SRAM/VMEM
        if h_prev is None:
            h_prev = torch.zeros(B, D, device=x.device)

        outputs = []
        h = h_prev
        for t in range(T):
            h = a_t[:, t] * h + b_t[:, t]
            outputs.append(h)

        output = torch.stack(outputs, dim=1)  # [B, T, D]
        return output, h

class RecurrentBlock(torch.nn.Module):
    """Griffin recurrent block — replaces MQA."""

    def __init__(self, d_model, d_rnn=None):
        super().__init__()
        d_rnn = d_rnn or (4 * d_model // 3)

        # Branch 1: Conv1D + RG-LRU
        self.linear_y = torch.nn.Linear(d_model, d_rnn)
        self.conv1d = torch.nn.Conv1d(d_rnn, d_rnn, kernel_size=4,
                                       padding=3, groups=d_rnn)
        self.rg_lru = RGLRU(d_rnn)

        # Branch 2: Gating
        self.linear_x = torch.nn.Linear(d_model, d_rnn)

        # Output projection
        self.linear_out = torch.nn.Linear(d_rnn, d_model)

    def forward(self, x):
        # Two parallel branches
        y = self.conv1d(self.linear_y(x).transpose(1, 2)).transpose(1, 2)
        y = y[:, :x.shape[1], :]  # trim padding
        y, _ = self.rg_lru(y)

        gate = F.gelu(self.linear_x(x))

        # Merge branches
        return self.linear_out(y * gate)

# Griffin architecture: interleave recurrent and local attention
# [RecurrentBlock, RecurrentBlock, LocalMQA] repeated N/3 times
# Local attention uses sliding window of size 1024
```

## References

- De, Smith, Fernando et al. "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models." arXiv:2402.19427, 2024.
- Orvieto et al. "Resurrecting Recurrent Neural Networks for Long Sequences." ICML 2023. (LRU — basis for RG-LRU)
- Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023. (similar gated recurrent block design)
- Feng et al. "Were RNNs All We Needed?" arXiv:2410.01201, 2024. (minGRU/minLSTM — same input-only gate insight)
- Pöppel et al. "FlashRNN: I/O-Aware Optimization of Traditional RNNs on Modern Hardware." arXiv:2412.07752, 2024. (fused kernel approach for dense recurrent weights)
- Yang et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. (extends retention/RG-LRU with matrix-valued state via chunkwise parallelism)

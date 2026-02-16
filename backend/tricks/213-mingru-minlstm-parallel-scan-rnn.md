# 213: minGRU/minLSTM Parallel-Scan-Compatible RNNs

**Category**: parallelization
**Gain type**: efficiency
**Source**: Feng, Tung, Ahmed, Bengio & Hajimirsadeghi, "Were RNNs All We Needed?" (2024)
**Paper**: papers/mingru-minlstm-parallel-rnn.pdf
**Documented**: 2026-02-15

## Description

minGRU and minLSTM are minimal variants of classical GRU and LSTM that achieve full parallelizability over the sequence dimension during training by removing hidden-state dependencies from their gates. The key insight is two-fold:

1. **Drop previous-state dependencies from gates**: Classical LSTM/GRU compute gates as $\sigma(\text{Linear}([x_t, h_{t-1}]))$, creating a sequential dependency. By simplifying to $\sigma(\text{Linear}(x_t))$, the gate values $a_t, b_t$ for the parallel scan $v_t = a_t \odot v_{t-1} + b_t$ can be precomputed for all time steps simultaneously.

2. **Remove tanh range restriction**: Since gates no longer depend on previous states, the tanh activation on candidate states becomes unnecessary and can be removed, further simplifying the architecture.

The resulting models use 75-87% fewer parameters than their traditional counterparts, train with $O(T \log T)$ parallel scan instead of $O(T)$ sequential BPTT, and achieve competitive performance with Mamba and Transformers on language modeling, selective copying, and reinforcement learning tasks.

## Mathematical Form

### minGRU

**Traditional GRU:**
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$
$$
z_t = \sigma(\text{Linear}_{d_h}([x_t, h_{t-1}]))
$$
$$
r_t = \sigma(\text{Linear}_{d_h}([x_t, h_{t-1}]))
$$
$$
\tilde{h}_t = \tanh(\text{Linear}_{d_h}([x_t, r_t \odot h_{t-1}]))
$$

**minGRU (simplified):**
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$
$$
z_t = \sigma(\text{Linear}_{d_h}(x_t))
$$
$$
\tilde{h}_t = \text{Linear}_{d_h}(x_t)
$$

**Parallel scan form:** Setting $a_t = (1 - z_t)$ and $b_t = z_t \odot \tilde{h}_t$:
$$
h_t = a_t \odot h_{t-1} + b_t
$$

This is a linear first-order recurrence computable via parallel prefix scan with the associative operator:
$$
(a_i, b_i) \bullet (a_j, b_j) = (a_i \odot a_j, \; a_j \odot b_i + b_j)
$$

### minLSTM

**Traditional LSTM:**
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)
$$
$$
f_t = \sigma(\text{Linear}_{d_h}([x_t, h_{t-1}])), \quad i_t = \sigma(\text{Linear}_{d_h}([x_t, h_{t-1}]))
$$
$$
o_t = \sigma(\text{Linear}_{d_h}([x_t, h_{t-1}])), \quad \tilde{c}_t = \tanh(\text{Linear}_{d_h}([x_t, h_{t-1}]))
$$

**minLSTM (simplified):**
$$
h_t = f'_t \odot h_{t-1} + i'_t \odot \tilde{h}_t
$$
$$
f_t = \sigma(\text{Linear}_{d_h}(x_t)), \quad i_t = \sigma(\text{Linear}_{d_h}(x_t))
$$
$$
\tilde{h}_t = \text{Linear}_{d_h}(x_t)
$$

where gates are normalized for time-independent scale:
$$
f'_t = \frac{f_t}{f_t + i_t}, \quad i'_t = \frac{i_t}{f_t + i_t}
$$

**Parallel scan form:** Setting $a_t = f'_t$ and $b_t = i'_t \odot \tilde{h}_t$:
$$
h_t = a_t \odot h_{t-1} + b_t
$$

**Key Definitions:**

- $h_t \in \mathbb{R}^{d_h}$ — hidden state at time $t$
- $z_t, f_t, i_t \in (0, 1)^{d_h}$ — gate activations (input-dependent only, no $h_{t-1}$)
- $\tilde{h}_t \in \mathbb{R}^{d_h}$ — candidate hidden state (linear projection of $x_t$, no tanh)
- $\odot$ — element-wise (Hadamard) product
- $d_h$ — hidden dimension, $d_x$ — input dimension

## Complexity

| Operation | Traditional LSTM/GRU | minLSTM/minGRU |
|-----------|---------------------|----------------|
| Training (sequential) | $O(T \cdot d_h(d_x + d_h))$ — BPTT | $O(T \log T \cdot d_h)$ — parallel scan |
| Gate computation | $O(T \cdot d_h(d_x + d_h))$ sequential | $O(T \cdot d_h \cdot d_x)$ fully parallel |
| Inference (per step) | $O(d_h(d_x + d_h))$ | $O(d_h \cdot d_x)$ |
| Parameters (GRU, $d_h = \alpha d_x$) | $O(3d_h(d_x + d_h))$ | $O(2d_h \cdot d_x)$ — 67-87% fewer |
| Parameters (LSTM, $d_h = \alpha d_x$) | $O(4d_h(d_x + d_h))$ | $O(3d_h \cdot d_x)$ — 62-85% fewer |

**Memory:** minGRU/minLSTM use ~88% more memory than traditional counterparts during training (parallel scan materializes the full computation graph), but Mamba uses 56% more than minGRU. Runtime is the practical bottleneck, not memory.

**Measured speedups (T4 GPU, batch=64):**

| Sequence Length | minGRU vs GRU | minLSTM vs LSTM |
|----------------|---------------|-----------------|
| 512 | 175× | 235× |
| 4096 | 1324× | 1361× |

## Applicability

- **Drop-in replacement for LSTM/GRU**: Anywhere traditional recurrent models are used (time series, RL, NLP) but training speed is the bottleneck
- **Competitive with modern SSMs**: Comparable test loss to Mamba on language modeling (1.548 vs 1.575 on Shakespeare nanoGPT)
- **Selective copying tasks**: Solves the Mamba selective copying benchmark (99.5% accuracy), matching S6 and outperforming S4, H3, Hyena
- **Reinforcement learning**: Competitive with Decision Transformer and Decision Mamba on D4RL benchmarks
- **Multi-layer stacking recovers expressivity**: While single-layer gates are time-independent, stacking 2-3 layers reintroduces time-dependent behavior through inter-layer dependencies ($x_{1:n}^{(2)} \leftarrow h_{1:n}^{(1)}$)
- **Efficient inference**: $O(1)$ memory per step in recurrent mode (same as traditional RNNs)

## Limitations

- **Gates are input-independent (no $h_{t-1}$)**: Single-layer minGRU/minLSTM cannot perform content-aware state-dependent gating. This is mitigated by stacking multiple layers but represents a fundamental expressivity trade-off.
- **Parallel scan memory overhead**: Training uses ~88% more memory than sequential RNNs due to materialized scan tree. Not ideal for memory-constrained settings.
- **Scale validation limited**: Paper validates at nanoGPT scale (character-level Shakespeare). Unclear how minGRU/minLSTM perform at 1B+ parameter, 100B+ token scales compared to Mamba-2 or Transformers.
- **No custom GPU kernels**: Reference implementation uses plain PyTorch `torch.cumsum` or `parallel_scan` utilities. Lacks FlashRNN-level kernel optimization; actual GPU efficiency may be suboptimal without fused kernels.
- **minLSTM training stability concerns**: minLSTM shows higher variance across seeds than minGRU on selective copying, due to competing forget/input gate optimization.
- **Linear recurrence only**: The simplification specifically requires the recurrence to be a linear first-order form $h_t = a_t \odot h_{t-1} + b_t$. Cannot accommodate more complex nonlinear state mixing.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

# ===== minGRU =====
def min_gru(x, h_0, W_z, W_h):
    """
    x: [B, T, d_x] input sequence
    h_0: [B, d_h] initial hidden state
    W_z: Linear(d_x, d_h) — gate projection
    W_h: Linear(d_x, d_h) — candidate projection
    """
    # All gates computed in parallel (no h_{t-1} dependency!)
    z = torch.sigmoid(W_z(x))        # [B, T, d_h]
    h_tilde = W_h(x)                  # [B, T, d_h] — no tanh!

    # Parallel scan formulation: h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t
    # Use log-space for numerical stability
    log_coeffs = torch.log(1 - z)     # log(a_t) for scan
    log_values = torch.log(z) + torch.log(torch.abs(h_tilde))  # log(b_t)

    # Parallel prefix sum in log-space (equivalent to cumulative product)
    h = parallel_scan(log_coeffs, h_tilde * z, h_0)  # [B, T, d_h]
    return h

# ===== minLSTM =====
def min_lstm(x, h_0, W_f, W_i, W_h):
    """
    x: [B, T, d_x] input sequence
    h_0: [B, d_h] initial hidden state
    W_f, W_i: Linear(d_x, d_h) — forget/input gate projections
    W_h: Linear(d_x, d_h) — candidate projection
    """
    f = torch.sigmoid(W_f(x))  # [B, T, d_h]
    i = torch.sigmoid(W_i(x))  # [B, T, d_h]
    h_tilde = W_h(x)           # [B, T, d_h] — no tanh!

    # Normalize gates for time-independent scale
    f_prime = f / (f + i)
    i_prime = i / (f + i)

    # Parallel scan: h_t = f'_t * h_{t-1} + i'_t * h_tilde_t
    h = parallel_scan(f_prime, i_prime * h_tilde, h_0)
    return h

# The parallel_scan uses the associative operator:
# (a_i, b_i) ⊕ (a_j, b_j) = (a_i * a_j, a_j * b_i + b_j)
# This can be computed in O(T log T) work with O(log T) depth.
# In practice, use chunkwise scan for better GPU utilization.
```

## References

- Feng, Tung, Ahmed, Bengio & Hajimirsadeghi. "Were RNNs All We Needed?" arXiv:2410.01201, 2024.
- GitHub: https://github.com/BorealisAI/minRNNs
- Martin & Cundy. "Parallelizing Linear Recurrent Neural Nets Over Sequence Length." ICLR 2018.
- Blelloch. "Prefix Sums and Their Applications." 1990.
- Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
- Beck et al. "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517, 2024.

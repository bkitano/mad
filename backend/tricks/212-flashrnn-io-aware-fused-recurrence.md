# 212: FlashRNN I/O-Aware Fused Recurrence

**Category**: kernel
**Gain type**: efficiency
**Source**: Pöppel, Beck & Hochreiter, "FlashRNN: I/O-Aware Optimization of Traditional RNNs on Modern Hardware" (2024)
**Paper**: papers/flashrnn-io-aware-rnn.pdf
**Documented**: 2026-02-15

## Description

FlashRNN fuses the recurrent matrix multiplication and pointwise nonlinearity of traditional RNNs (LSTM, GRU, sLSTM, Elman) into a single persistent GPU kernel, caching the recurrent weight matrix $R$ in registers across all time steps. This eliminates repeated HBM round-trips for $R$ at every step of the sequential recurrence. The approach extends RNNs with multi-head parallelism (block-diagonal $R$), enabling independent heads to run on separate thread blocks/SMs, analogous to multi-head attention in Transformers.

The key insight is that while RNNs are inherently sequential over the time dimension, the dominant cost is *memory-bound*: each time step alternates between a matrix multiplication ($R \cdot h_{t-1}$) and pointwise activations (sigmoid, tanh), requiring separate kernel launches and HBM reads/writes in naive implementations. By fusing these into a single persistent kernel that keeps $R$ in registers/SRAM, FlashRNN converts the workload from memory-bound to compute-bound.

## Mathematical Form

**Generic RNN formulation:**

A generic RNN with $N_s$ states $s^{(i)} \in \mathbb{R}^d$ and $N_g$ gates $g^{(j)} \in \mathbb{R}^d$:

$$
g_t^{(j)} = x_t^{(j)} + R^{(j)} s_{t-1}^{(0)} + b^{(j)}
$$

$$
s_t^{(i)} = \mathcal{P}^{(i)}\left(\left(s_{t-1}^{(i')}\right)_{i' \in \{1..N_s\}}, \left(g_t^{(j)}\right)_{j \in \{1..N_g\}}\right)
$$

where $\mathcal{P}^{(i)}$ is a pointwise function (element-wise, no cross-cell mixing), $R^{(j)} \in \mathbb{R}^{d \times d}$ is the recurrent weight matrix for gate $j$, and $x_t^{(j)}$ are pre-computed input projections.

**Key Definitions:**

- $R^{(j)} \in \mathbb{R}^{d \times d}$ — recurrent weight matrix for gate $j$ (cached in registers)
- $s_t^{(i)} \in \mathbb{R}^d$ — state $i$ at time $t$
- $g_t^{(j)} \in \mathbb{R}^d$ — gate $j$ pre-activation at time $t$
- $\mathcal{P}^{(i)}$ — pointwise recurrence function (e.g., LSTM cell update)

**Head-wise parallelization:**

The embedding dimension $d$ is split into $N_{\text{heads}}$ heads of dimension $d_{\text{head}} = d / N_{\text{heads}}$. The recurrent matrix becomes block-diagonal:

$$
R = \text{diag}(R_{\text{head}}^{(1)}, R_{\text{head}}^{(2)}, \ldots, R_{\text{head}}^{(N_{\text{heads}})})
$$

where each $R_{\text{head}}^{(h)} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$ is processed independently, enabling parallel execution across SMs.

**Backward pass:**

$$
\delta g_t^{(j)} = \frac{\partial \mathcal{P}^{(l)}\left(\left(s_{t-1}^{(k)}\right), \left(g_t^{(j)}\right)\right)}{\partial g_t^{(j)}} \delta s_t^{(l)}
$$

$$
\delta s_{t-1}^{(i)} = \frac{\partial \mathcal{P}^{(l)}\left(\left(s_{t-1}^{(k)}\right), \left(g_t^{(j)}\right)\right)}{\partial s_{t-1}^{(i)}} \delta s_t^{(l)} + \left(\sum_{j} R^{(j)T} \delta g_{t-1}^{(j)}\right) \quad \text{if } i = 0
$$

$$
\delta R^{(j)} = \sum_t \delta g_t^{(j)} s_t^{(0)T}
$$

## Complexity

| Operation | Naive (PyTorch) | FlashRNN Fused |
|-----------|----------------|----------------|
| Per-step matmul | $O(d^2)$ + kernel launch + HBM read $R$ | $O(d_{\text{head}}^2)$, $R$ in registers |
| Per-step pointwise | Separate kernel + HBM read/write states | Fused, states in SRAM/registers |
| Total HBM reads for $R$ | $O(T \cdot N_g \cdot d^2)$ | $O(N_g \cdot d^2)$ (loaded once) |
| Kernel launches | $O(T \cdot 2)$ (matmul + pointwise per step) | $O(1)$ (single persistent kernel) |

**Memory:** States stored in SRAM during recurrence; only final states and gates written to HBM. Register file caches $R$ (256 KB SRAM per SM on H100).

**Measured speedups (H100, LSTM, $d=768$, $T=1024$):**

| Variant | vs. Vanilla PyTorch | vs. cuDNN (nn.LSTM) |
|---------|-------------------|---------------------|
| CUDA fused ($d_h$=64, 12 heads) | ~50× | ~2× faster train time |
| CUDA alternating | ~30× | Comparable |
| Triton fused | ~20× | — |

## Applicability

- **Traditional RNNs**: LSTM, GRU, Elman networks with recurrent connections
- **Modern variants**: sLSTM (from xLSTM), any architecture with nonlinear state-to-state dependencies
- **State-tracking tasks**: Parity, counting, and other tasks requiring true recurrent state (where Transformers/Mamba fail)
- **Multi-head RNNs**: Block-diagonal recurrent matrices enable parallel processing similar to multi-head attention
- **Language modeling at scale**: 165M parameter models trained on 15B tokens with competitive perplexity

## Limitations

- **Still sequential over time**: Does not parallelize across the sequence dimension (unlike parallel scan for linear recurrences). Sequence length still processed step-by-step.
- **Head dimension bounded by register/SRAM capacity**: CUDA fused kernels limited to $d_{\text{head}} \leq 128$ (forward) and $d_{\text{head}} \leq 64$ (backward) on H100. Triton limited to $d_{\text{head}} \leq 128$ forward, 64 backward.
- **Not faster than FlashAttention-2 for same model quality**: At 165M scale, LSTM with FlashRNN is ~25% slower per step than FlashAttention-2 for equal head dimension, ~140% slower for single-head. The advantage is in state-tracking capability, not raw throughput.
- **BFloat16 only**: Optimized for bfloat16; float32 supported but slower. Some numerical deviations observed vs. cuDNN reference.
- **Requires custom CUDA/Triton kernels**: Cannot use standard PyTorch autograd; requires the FlashRNN library.

## Implementation Notes

```python
# FlashRNN Fused Kernel — Pseudocode (forward pass)
# Key idea: R stays in registers for the entire time loop

def flashrnn_fused_forward(R_gs, x_tbg, b_g, s_0bs):
    """
    R_gs: recurrent weights [N_gates, d_head, d_head] — loaded to registers ONCE
    x_tbg: input projections [T, B, N_gates, d_head] — streamed from HBM
    b_g:   biases [N_gates, d_head] — loaded to registers
    s_0bs: initial states [N_states, B, d_head]
    """
    # Load R and biases to registers/SRAM (done ONCE)
    R_reg = load_to_registers(R_gs)
    b_reg = load_to_registers(b_g)

    for batch_block in parallel_blocks(B):
        s = load_to_sram(s_0bs[:, batch_block, :])

        for t in range(T):
            # Tiled matmul: y = R @ s[0] (accumulated in SRAM)
            # Tiles of R from registers, tiles of s from SRAM
            y = tiled_matmul_register(R_reg, s[0])  # stays in SRAM

            # Fused pointwise: g = x + y + b, then s_new = P(s, g)
            g = x_tbg[t, batch_block] + y + b_reg
            s = pointwise_recurrence(s, g)  # e.g., LSTM cell update

            # Write gates to HBM (needed for backward)
            write_to_hbm(g, t)

            # Grid-level sync for inter-SM state sharing
            grid_sync()

    return s  # final states

# The critical optimization: R is loaded to GPU registers ONCE
# and reused for every time step T, eliminating O(T) HBM reads.
# States stay in SRAM between steps. Only inputs stream from HBM.
```

## References

- Pöppel, Beck & Hochreiter. "FlashRNN: I/O-Aware Optimization of Traditional RNNs on Modern Hardware." arXiv:2412.07752, 2024.
- GitHub: https://github.com/NX-AI/flashrnn
- Beck et al. "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517, 2024.
- Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
- Sharvil. "HASTE: Haste LSTM/GRU CUDA kernels." https://github.com/lmnt-com/haste, 2020.

# 182: Fused Chunkwise SSD with Atomic Inter-Chunk State Passing

**Category**: kernel
**Gain type**: efficiency
**Source**: Astra, Dao, Hoque — "Accelerating Mamba2 with Kernel Fusion" (PyTorch Blog, Feb 2026)
**Paper**: [papers/fused-chunkwise-ssd-atomic-state-passing.pdf]
**Documented**: 2026-02-16

## Description

The chunkwise-parallel formulation of linear RNNs (used in Mamba-2/SSD, GLA, TFLA, etc.) naturally decomposes into multiple sequential kernel launches: (1) compute cumulative gates, (2) compute per-chunk states in isolation, (3) propagate states across chunks (inter-chunk recurrence), (4) compute intra-chunk attention, and (5) scan each chunk's output using the propagated state. Each launch incurs CPU-GPU synchronization overhead and forces intermediate results to be written to and read from HBM between kernels.

**Fused Chunkwise SSD** merges all five SSD kernels into a **single Triton kernel launch** using **atomic-based inter-chunk synchronization** to handle the sequential state-passing dependency without requiring separate kernel launches. The key insight is that the inter-chunk state propagation (which is inherently sequential across chunks) can be overlapped with the independent intra-chunk computation by assigning each chunk to a thread block and using atomics to serialize only the state-passing step while the rest of the computation proceeds in parallel.

This is the first end-to-end Triton fusion of all five SSD kernels. On NVIDIA A100 and H100 GPUs, the fused kernel achieves **1.50×–2.51× speedup** on the SSD portion, translating to **8–13% end-to-end speedup** for Mamba-2 2.7B inference (batch=1, seq=128K).

## Mathematical Form

**Mamba-2 SSD Chunkwise Decomposition:**

The State Space Dual (SSD) model computes:

$$
y_t = C_t h_t, \quad h_t = A_t h_{t-1} + B_t x_t
$$

where $A_t = \text{diag}(e^{-\Delta_t \cdot a})$ is a diagonal decay matrix, $B_t \in \mathbb{R}^{N \times 1}$, $C_t \in \mathbb{R}^{1 \times N}$, and $h_t \in \mathbb{R}^{N \times D}$ is the hidden state.

**Five-kernel decomposition** (original unfused):

**Kernel 1 — Chunk Cumsum:** Compute per-token $\Delta_t$ and cumulative decay within each chunk:

$$
\bar{\Delta}_{k,i} = \sum_{j=1}^{i} \Delta_{kL+j}, \quad \bar{A}_{k,i} = e^{-\bar{\Delta}_{k,i} \cdot a}
$$

**Kernel 2 — Chunk State:** Compute each chunk's final state in isolation (ignoring prior chunks):

$$
\hat{h}_k = \sum_{i=1}^{L} \bar{A}_{k,L-i} \cdot B_{kL+i} \cdot x_{kL+i}^T
$$

**Kernel 3 — State Passing (sequential across chunks):**

$$
h_k = \bar{A}_{k,L} \cdot h_{k-1} + \hat{h}_k
$$

This is the recurrence that creates inter-chunk dependencies.

**Kernel 4 — Chunk BMM:** Compute intra-chunk quadratic attention:

$$
Y_{\text{intra}}^{(k)} = \left(C^{(k)} \cdot M^{(k)} \cdot {B^{(k)}}^T\right) \cdot X^{(k)}
$$

where $M^{(k)}$ is the causal decay mask within chunk $k$.

**Kernel 5 — Chunk Scan:** Combine intra-chunk result with inter-chunk state:

$$
Y^{(k)} = Y_{\text{intra}}^{(k)} + C^{(k)} \cdot h_{k-1}
$$

**Fused Kernel — Atomic Synchronization Strategy:**

In the fused kernel, thread block $k$ (handling chunk $k$) executes all five stages sequentially within a single launch:

$$
\text{TB}_k: \quad \underbrace{\text{CumSum}_k}_{\text{local}} \to \underbrace{\hat{h}_k}_{\text{local}} \to \underbrace{\text{AtomicWait}(k-1)}_{\text{serialize}} \to \underbrace{h_k = \bar{A}_{k,L} h_{k-1} + \hat{h}_k}_{\text{propagate}} \to \underbrace{\text{AtomicSignal}(k)}_{\text{release}} \to \underbrace{Y^{(k)}}_{\text{local}}
$$

**Key Definitions:**

- $T$ — total sequence length
- $L$ — chunk size
- $N_c = T / L$ — number of chunks
- $N$ — state dimension (SSM state size)
- $D$ — model dimension (per head)
- $h_k \in \mathbb{R}^{N \times D}$ — hidden state at end of chunk $k$
- $\hat{h}_k$ — isolated chunk state (computed without knowledge of $h_{k-1}$)
- $\bar{A}_{k,L}$ — cumulative decay over chunk $k$

## Complexity

| Metric | Unfused (5 launches) | Fused (1 launch) |
|--------|---------------------|------------------|
| Kernel launches | 5 | 1 |
| HBM round-trips for intermediates | 4 (between each kernel pair) | 0 (registers/L1) |
| State Passing serialization | Implicit (global sync) | Atomics (overlapped) |
| CPU-GPU sync overhead | 5× launch + scheduling | 1× launch |

**Amdahl's Law Analysis:**

If State Passing is $\frac{1}{7}$ of total compute time and the other 4 kernels share $\frac{6}{7}$:

$$
T_{\text{fused}} = T_{\text{SP}} + \max(T_{\text{other}}, T_{\text{SP\_sync}}) = \frac{1}{7} + \max\!\left(\frac{6}{7},\; \frac{1}{7} \times N_c \cdot t_{\text{atomic}}\right)
$$

When atomic sync overhead is low (which it is for contiguous chunks), the serialization penalty is ~14% over ideal parallelism — far less than the 5-launch overhead it replaces.

**Wall-clock performance:**

| Setting | Unfused | Fused | Speedup |
|---------|---------|-------|---------|
| SSD portion (A100/H100) | baseline | — | **1.50×–2.51×** |
| Mamba-2 2.7B end-to-end (batch=1, seq=128K) | baseline | — | **8–13%** |

**Memory:** Intermediate tensors ($\hat{h}_k$, cumulative gates, etc.) stay in registers/L1 instead of being written to HBM between kernel launches, reducing peak memory by the size of all intermediate buffers.

## Applicability

- **Mamba-2 / SSD (primary application):** The technique was developed for and validated on the Mamba-2 SSD module. Directly applicable to any model using the SSD chunkwise formulation.

- **TFLA / Flash Linear Attention:** TFLA's two-level algorithm has the same structure: inter-chunk state recurrence (sequential) + intra-chunk parallel matmuls. The atomic state-passing fusion can merge TFLA's recurrent kernel and parallel kernel into a single launch, avoiding HBM materialization of chunk boundary states $C_k$.

- **GLA (Gated Linear Attention):** GLA's chunkwise kernel similarly separates state computation, state propagation, and output computation into multiple kernels. The same fusion strategy applies.

- **DeltaNet / DeltaProduct:** The chunkwise-parallel DeltaNet formulation has inter-chunk state updates that could use atomic synchronization for single-launch fusion.

- **Any chunkwise-parallel linear RNN:** The pattern of "compute local chunk results → propagate states sequentially → combine" is universal to chunkwise formulations. Atomic-based fusion applies whenever the sequential dependency is between adjacent chunks.

## Limitations

- **Atomic serialization overhead:** While atomic-based state passing overlaps well with independent computation, the serialization still creates a critical path proportional to $N_c$ (number of chunks). For very long sequences with small chunks, this could become a bottleneck.

- **Triton-specific:** The implementation relies on Triton's ability to express atomics and fine-grained synchronization within a single kernel. Porting to other frameworks requires equivalent capabilities.

- **Thread block grid matching:** The fused kernel requires that thread block grids for all five stages are aligned — each chunk maps to the same thread block across all stages. This constrains the parallelization strategy (e.g., cannot independently parallelize state channels in Kernel 3).

- **Not yet applied to TFLA:** The technique has only been demonstrated for Mamba-2 SSD. Extending to TFLA (which has a more complex two-level tiling structure) requires adapting the atomic synchronization to work with TFLA's nested tile loops.

- **Debugging complexity:** Atomic synchronization within a fused kernel is harder to debug than separate kernel launches with implicit global synchronization.

- **Diminishing returns for compute-bound regimes:** When the model is already compute-bound (large state dimension, large chunks), the memory and launch overhead savings from fusion are proportionally smaller.

## Implementation Notes

```python
# Pseudocode for fused chunkwise SSD kernel
# Based on PyTorch blog description (Astra, Dao, Hoque, Feb 2026)

@triton.jit
def fused_ssd_kernel(
    X, A, B, C, dt,    # inputs
    Y,                   # output
    H_global,            # global state buffer (for inter-chunk communication)
    sync_flags,          # atomic synchronization flags
    T, L, N, D,         # dimensions
    BLOCK_L: tl.constexpr,  # chunk size
    BLOCK_N: tl.constexpr,  # state tile size
    BLOCK_D: tl.constexpr,  # model dim tile size
):
    """
    Single-launch fused SSD kernel.
    Each thread block handles one chunk k.
    Inter-chunk state passing uses atomics.
    """
    chunk_id = tl.program_id(0)  # chunk index k
    head_id = tl.program_id(1)   # head index

    # ============================================
    # Stage 1: Chunk Cumsum (local to this chunk)
    # ============================================
    # Compute cumulative dt and decay within this chunk
    # All in registers — no HBM write needed
    cum_dt = compute_cumulative_dt(dt, chunk_id, L)
    cum_decay = tl.exp(-cum_dt * A_val)

    # ============================================
    # Stage 2: Chunk State (local, isolated)
    # ============================================
    # Compute h_hat_k = sum_i decay(L-i) * B_i * x_i^T
    # This is the chunk's contribution to the state,
    # ignoring the previous chunk's state
    h_hat = compute_isolated_chunk_state(
        X, B, cum_decay, chunk_id, L, N, D
    )  # h_hat in registers: N x D

    # ============================================
    # Stage 3: State Passing (serialized via atomics)
    # ============================================
    if chunk_id > 0:
        # WAIT: spin until previous chunk signals completion
        while tl.atomic_cas(sync_flags + chunk_id - 1, 1, 1) != 1:
            pass  # busy-wait (actual impl uses more efficient sync)

        # READ: load h_{k-1} from global buffer
        # (already in L1/L2 cache since chunk k-1 just wrote it)
        h_prev = tl.load(H_global + (chunk_id - 1) * N * D)
    else:
        h_prev = tl.zeros((N, D))

    # COMPUTE: h_k = decay_total * h_{k-1} + h_hat_k
    total_decay = cum_decay[-1]  # decay across entire chunk
    h_k = total_decay * h_prev + h_hat

    # WRITE + SIGNAL: store h_k and notify chunk k+1
    tl.store(H_global + chunk_id * N * D, h_k)
    tl.atomic_xchg(sync_flags + chunk_id, 1)  # signal completion

    # ============================================
    # Stage 4 + 5: BMM + Chunk Scan (local)
    # ============================================
    # Compute intra-chunk quadratic attention
    Y_intra = compute_intra_chunk_attention(
        C, B, X, cum_decay, chunk_id, L, N, D
    )

    # Add inter-chunk contribution: C_i @ h_{k-1}
    Y_inter = compute_inter_chunk_output(
        C, h_prev, cum_decay, chunk_id, L, N, D
    )

    # Final output
    Y_chunk = Y_intra + Y_inter

    # Write to HBM only once (final result)
    tl.store(Y + chunk_id * L * D, Y_chunk)
```

**Key engineering insights:**

1. **Cache locality of state passing:** Since chunk $k-1$ writes $h_{k-1}$ to HBM just before chunk $k$ reads it, the data is likely still in L2 cache, making the "HBM" read effectively an L2 hit.

2. **Overlap opportunity:** While thread block $k$ waits for the atomic signal from $k-1$ (Stage 3), it has already computed Stages 1-2. The wait is only for the state propagation. Meanwhile, the GPU scheduler can execute other thread blocks (different heads, different batch elements) on the same SM.

3. **No intermediate HBM tensors:** The cumulative gates, isolated chunk states, and intra-chunk attention scores never leave registers/SMEM. Only the final output $Y$ and the global state buffer $H$ (one $N \times D$ matrix per chunk) touch HBM.

4. **Applicable to TFLA:** The same pattern applies to TFLA's inter-chunk state recurrence $C_k = \bar{g}_k C_{k-1} + (\bar{a}_k \odot K^{(k)})^T V^{(k)}$. The atomic approach fuses TFLA's recurrent kernel and parallel kernel into one launch.

## References

- Astra, R., Dao, T., & Hoque, A. (2026). "Accelerating Mamba2 with Kernel Fusion." PyTorch Blog. https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/
- Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). "Tiled Flash Linear Attention." NeurIPS 2025. arXiv:2503.14376.
- Yang, S., Wang, B., et al. (2024). "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. arXiv:2312.06635.

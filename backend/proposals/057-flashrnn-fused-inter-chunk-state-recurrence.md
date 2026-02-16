---
status: ongoing
priority: high
created: 2026-02-15
based_on: flashrnn-io-aware-fused-recurrence (212), fused-chunkwise-ssd-atomic-state-passing (182), tfla-two-level-tiled-chunkwise-parallelism (158), chunkwise-parallel-scan (026), io-aware-tiling (066), warp-specialized-pipelining (141), gemm-softmax-two-stage-async-pipelining (208)
experiment_number: 057
experiment_log: experiment-log-057.md
---

# FlashRNN-Style Fused Inter-Chunk State Recurrence for Chunkwise Linear RNNs

## Hypothesis

Replacing the **separate inter-chunk state propagation kernel** in chunkwise linear RNNs (GLA, Gated DeltaNet, mLSTM) with a **FlashRNN-style persistent fused recurrence** — where the per-chunk transition matrices $A_k^{(C)}$ are cached in registers/SRAM and the sequential state scan $h_k = A_k^{(C)} h_{k-1} + \hat{h}_k$ runs without any HBM round-trips for intermediate states — will achieve **$1.3$–$1.8\times$ speedup for the inter-chunk phase** and **$1.05$–$1.15\times$ end-to-end training speedup** by eliminating $O(G)$ HBM reads/writes of the $d_k \times d_v$ state matrix at each chunk boundary (where $G = T/C$ chunks).

## Background

### The inter-chunk state scan: a hidden HBM bottleneck

In chunkwise-parallel linear RNNs, the sequence is split into $G = T/C$ chunks. Within each chunk, efficient parallel computation (quadratic attention or WY transform) runs on tensor cores. Between chunks, the boundary states must be propagated sequentially:

$$
h_k = A_k^{(C)} h_{k-1} + \hat{h}_k, \quad k = 1, \ldots, G
$$

where $A_k^{(C)} \in \mathbb{R}^{d_k \times d_k}$ is the cumulative per-chunk transition (diagonal for GLA/Mamba-2, low-rank+diagonal for DeltaNet) and $\hat{h}_k \in \mathbb{R}^{d_k \times d_v}$ is the isolated per-chunk state contribution.

**Current approaches** handle this in one of three ways:

1. **Sequential loop** (fla-org, most implementations): A Python/Triton loop over $k = 1, \ldots, G$. Each iteration reads $A_k^{(C)}, \hat{h}_k$ from HBM, computes the update, writes $h_k$ to HBM. **Total HBM traffic**: $G \times (d_k^2 + 2 d_k d_v) \times 2$ bytes (read $A_k$, read $\hat{h}_k$, write $h_k$; read previous $h_{k-1}$).

2. **Atomic state-passing** (trick 182, fused SSD): Merges the inter-chunk scan into the same kernel as intra-chunk computation. Avoids separate kernel launch but still writes/reads state to global memory for synchronization between thread blocks handling different chunks.

3. **MatMulScan** (proposal 044): Reformulates the scan as batched matmuls. Good for tensor core utilization but doesn't address the HBM traffic for intermediate states.

**The problem**: For $d_k = 64, d_v = 64$, each state $h_k$ is $64 \times 64 \times 2 = 8$ KB. For $G = 64$ chunks ($T = 4096, C = 64$): total state traffic is $64 \times 3 \times 8\text{KB} = 1.5$ MB per head. With $H = 16$ heads: 24 MB per layer. At 24 layers: 576 MB of HBM traffic *just for inter-chunk state passing*. At H100's 3.35 TB/s, this is ~170 μs — comparable to or larger than the inter-chunk computation itself.

### FlashRNN's insight applied to chunk-level recurrence

FlashRNN (trick 212) demonstrated that for sequential RNN recurrences, the dominant cost is not computation but **HBM bandwidth** for reading/writing intermediate states and the recurrent weight matrix $R$ at every time step. By caching $R$ in registers and keeping states in SRAM for the entire recurrence, FlashRNN achieved 50× speedups.

**Key observation**: The inter-chunk state scan is structurally identical to a small RNN with:
- "Time steps" = $G$ chunks (typically 32–128)
- "Hidden state" = $h_k \in \mathbb{R}^{d_k \times d_v}$ (the boundary state matrix)
- "Recurrent weight" = $A_k^{(C)}$ (the chunk transition — different per step for GLA, but diagonal or low-rank structured)
- "Input" = $\hat{h}_k$ (the isolated chunk contribution)

For GLA where $A_k^{(C)} = \text{diag}(\gamma_k^{(1)}, \ldots, \gamma_k^{(d_k)})$ is diagonal, the entire recurrence decomposes into $d_k$ independent scalar recurrences applied to $d_v$-dimensional vectors:

$$
h_k[i, :] = \gamma_k^{(i)} h_{k-1}[i, :] + \hat{h}_k[i, :], \quad i = 1, \ldots, d_k
$$

Each scalar-vector recurrence reads one scalar $\gamma_k^{(i)}$ and one $d_v$-dimensional vector $\hat{h}_k[i, :]$ per step — if these are in SRAM, the cost is purely compute-bound.

### What's different from existing proposals

- **Proposal 044** (MatMulScan): Reformulates the scan's control flow as tensor-core matmuls. Complementary — our approach addresses HBM traffic, MatMulScan addresses compute efficiency. Can be combined.
- **Proposal 040** (Persistent Megakernel): Fuses the *entire* layer. Much more ambitious; our approach is a targeted optimization of just the inter-chunk scan that can be implemented independently as a drop-in kernel replacement.
- **Trick 182** (Fused SSD Atomic State-Passing): Uses atomics for inter-chunk sync within a fused kernel. The states still transit through global memory (L2/HBM). Our approach keeps states in SRAM/registers entirely.

## Related Work

- **FlashRNN** (Pöppel et al., 2024): IO-aware fused RNN kernels that cache $R$ in registers. Applied to LSTM/GRU/sLSTM. Not applied to the inter-chunk state scan of chunkwise linear RNNs.
- **Fused Chunkwise SSD** (Astra et al., Feb 2026): Fuses all 5 SSD kernels into 1 launch with atomics. The inter-chunk state passing uses global-memory atomics, not SRAM-resident recurrence.
- **TFLA** (Beck et al., NeurIPS 2025): Introduces two-level tiling for chunkwise linear RNNs. The inter-chunk recurrence is a separate kernel call.
- **Mamba-2 SSD** (Dao & Gu, 2024): Original chunkwise decomposition. Inter-chunk scan is a separate kernel.

**Gap**: No existing work applies FlashRNN's register-caching + SRAM-persistent-state strategy specifically to the inter-chunk state propagation in chunkwise linear RNNs. All existing approaches either use global memory for inter-chunk state synchronization or treat the inter-chunk scan as a black-box parallel scan.

## Mathematical Formulation

**Standard Inter-Chunk State Scan (separate kernel):**

For $G = T/C$ chunks:

$$
h_0 = \mathbf{0}, \quad h_k = A_k^{(C)} h_{k-1} + \hat{h}_k, \quad k = 1, \ldots, G
$$

where $h_k \in \mathbb{R}^{d_k \times d_v}$, $A_k^{(C)} \in \mathbb{R}^{d_k \times d_k}$ (diagonal for GLA), $\hat{h}_k \in \mathbb{R}^{d_k \times d_v}$.

**HBM traffic per head**: $G \times (d_k^2 + 2 d_k d_v) \times \text{sizeof(bf16)}$.

**Proposed FlashRNN-Style Fused Scan:**

A single persistent thread block (or warp group) handles the entire scan for one head:

$$
\text{Load } \{\gamma_k^{(i)}\}_{k=1}^{G} \text{ for all } i \text{ (prefetch to SRAM)}, \quad \text{Load } \hat{h}_k \text{ streaming from HBM}
$$

$$
\text{For } k = 1, \ldots, G: \quad h_k[i, j] = \gamma_k^{(i)} \cdot h_{k-1}[i, j] + \hat{h}_k[i, j] \quad \text{(in SRAM/registers)}
$$

$$
\text{Store } h_k \text{ to HBM (for use by intra-chunk output kernel)}
$$

**For GLA (diagonal $A_k$):**

The diagonal structure means the recurrence decomposes into $d_k$ independent rows. Each thread/warp handles a subset of rows, maintaining its portion of $h_k$ in registers:

$$
\text{Thread } i: \quad \mathbf{r}_k = \gamma_k^{(i)} \cdot \mathbf{r}_{k-1} + \hat{h}_k[i, :], \quad \mathbf{r}_k \in \mathbb{R}^{d_v} \text{ (in registers)}
$$

For $d_v = 64$ in bf16: each row is 128 bytes = 64 registers. One thread handles one row → $d_k = 64$ threads handle the full state. A warp (32 threads) handles 32 rows; 2 warps handle the full $64 \times 64$ state.

**For Gated DeltaNet (low-rank $A_k = \gamma_k (I - W_k Y_k^\top)$):**

The transition is not diagonal but is structured: diagonal decay $\gamma_k$ plus a rank-$n_h$ correction from the WY representation. The FlashRNN approach still works:

1. Cache the WY factors $W_k, Y_k \in \mathbb{R}^{d_k \times n_h}$ in shared memory (small: $d_k \times n_h \times 2 \times 2 = 64 \times 4 \times 4 = 1$ KB per chunk)
2. State $h_k$ stays in SRAM
3. Per-step update: $h_k = \gamma_k (h_{k-1} - W_k (Y_k^\top h_{k-1})) + \hat{h}_k$

The $Y_k^\top h_{k-1}$ operation is $n_h \times d_k \times d_v$ — a small matmul that fits in shared memory. The $W_k \cdot (\cdot)$ operation is $d_k \times n_h \times d_v$ — similarly small.

**Async Pipelining (trick 208 applied):**

While the recurrence for chunk $k$ computes in SRAM, TMA can **asynchronously prefetch** $\hat{h}_{k+1}$ and $\gamma_{k+1}$ from HBM:

$$
\text{Stage 1 (TC/ALU):} \quad h_k = \gamma_k \cdot h_{k-1} + \hat{h}_k \quad \| \quad \text{Stage 2 (TMA):} \quad \text{prefetch } \hat{h}_{k+1}, \gamma_{k+1}
$$

This overlaps the sequential compute with the memory loads, hiding the HBM latency for reading the next chunk's data.

**Key Variables:**

- $G = T/C$ — number of chunks
- $C$ — chunk size (64–256)
- $d_k$ — key/state dimension (64–128)
- $d_v$ — value dimension (64–128)
- $h_k \in \mathbb{R}^{d_k \times d_v}$ — chunk boundary state
- $\gamma_k^{(i)} \in (0, 1)$ — per-dimension cumulative decay over chunk $k$
- $\hat{h}_k \in \mathbb{R}^{d_k \times d_v}$ — isolated chunk state contribution

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Gated DeltaNet (unchanged architecture) |
| Kernel | FlashRNN-fused inter-chunk state scan |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d_{\text{model}} = 768$–$2048$ |
| Heads | $H = 8$–$16$ |
| Head dim | $d_k = d_v = 64$ |
| Chunk size | $C = 64, 128, 256$ |
| Sequence length | $T = 2048, 4096, 8192$ |

### Baseline

1. **fla-org sequential loop**: The current default inter-chunk scan in `flash-linear-attention`. Sequential Python/Triton loop, $G$ iterations each reading/writing state from HBM.
2. **Fused SSD atomic** (trick 182): Atomic state-passing within a fused kernel. States pass through L2/global memory.
3. **Standard parallel scan** (CUB-style): Blelloch scan with decoupled lookback. Good for large $G$ but each scan step accesses global memory.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Inter-chunk kernel time | $\leq 0.6\times$ fla-org baseline | Wall-clock μs per head on H100 |
| Inter-chunk HBM traffic | $\leq 0.3\times$ baseline | Nsight Compute L2 sector reads/writes |
| End-to-end throughput | $\geq 1.08\times$ baseline | Tokens/sec for full model training |
| Model quality | Identical (bit-exact) | Validation loss comparison |
| SRAM usage per head | $\leq 32$ KB | Shared memory allocation |

### Estimated Compute

**MVE (kernel microbenchmark)**: < 15 minutes on single H100
**Full profiling**: ~4 GPU-hours on H100
**End-to-end training (350M)**: ~80 GPU-hours on H100
**Total**: ~85 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**

- The inter-chunk scan kernel time drops by $1.3$–$1.8\times$ because intermediate states never leave SRAM. The $d_k \times d_v = 64 \times 64$ state matrix (8 KB in bf16) fits comfortably in shared memory (H100: 228 KB per SM), and the per-step compute ($d_k \times d_v$ elementwise multiply-add for GLA) is ~8K FLOPs — fully compute-bound when state is in SRAM.
- End-to-end training speedup of $1.05$–$1.15\times$, since the inter-chunk scan is ~5–15% of total layer time (varies with $C$ and $T$; larger $T/C$ = bigger fraction).
- Model outputs are **bit-exact** — this is a pure kernel optimization with no algorithmic change.
- TMA async pipelining hides 80%+ of the $\hat{h}_k$ read latency, as the per-chunk compute is sufficient to overlap with a single TMA load.

**If hypothesis is wrong:**

- **Scenario A: Compute-bound, not memory-bound**: If the inter-chunk scan is already compute-bound (unlikely for the elementwise GLA case but possible for Gated DeltaNet's rank-$n_h$ update), the SRAM caching provides no benefit. **Learn**: The bottleneck is the sequential computation itself, not HBM traffic. Fall back to MatMulScan (proposal 044) to parallelize the computation.
- **Scenario B: Too few chunks for persistent kernel overhead**: For small $G < 16$, the persistent kernel launch + SRAM setup overhead exceeds the HBM savings. **Learn**: The optimization only matters at long sequences ($T \geq 4096$) or small chunks ($C = 64$). Report the $G^*$ crossover point.
- **Scenario C: Register pressure**: The $d_v = 64$ state row requires 64 bf16 registers per thread. If this causes occupancy to drop below 1 active warp per SM, throughput may decrease. **Learn**: Need to reduce $d_v$ per thread via blocking.

## Minimum Viable Experiment

### Setup
- **Kernel**: Triton kernel implementing FlashRNN-style fused inter-chunk scan for GLA (diagonal case)
- **Configuration**: $G = 64$ chunks, $d_k = d_v = 64$, $H = 16$ heads, batch = 4
- **Comparison**: fla-org's sequential loop inter-chunk scan
- **Hardware**: Single H100 GPU
- **Compute**: < 10 minutes

### Implementation Sketch

```python
@triton.jit
def flashrnn_inter_chunk_scan(
    h_local_ptr,   # [G, H, dk, dv] — isolated chunk states (HBM)
    gamma_ptr,      # [G, H, dk] — cumulative per-chunk decays (HBM)
    h_out_ptr,      # [G, H, dk, dv] — propagated states (HBM)
    G, H, dk, dv,
    BLOCK_DV: tl.constexpr,
):
    """FlashRNN-style fused inter-chunk scan.
    One program handles one head, one row of the state matrix."""
    head_id = tl.program_id(0)
    row_id = tl.program_id(1)  # which row of dk

    # State row lives in registers for entire scan
    h_row = tl.zeros([BLOCK_DV], dtype=tl.bfloat16)  # SRAM-resident

    for k in range(G):
        # Load gamma (1 scalar) and h_local row (dv elements) from HBM
        gamma = tl.load(gamma_ptr + k * H * dk + head_id * dk + row_id)
        h_local_row = tl.load(
            h_local_ptr + (k * H * dk * dv + head_id * dk * dv
                           + row_id * dv + tl.arange(0, BLOCK_DV))
        )

        # Recurrence in registers — no HBM write until needed
        h_row = gamma * h_row + h_local_row

        # Write result to HBM (needed by intra-chunk output kernel)
        tl.store(
            h_out_ptr + (k * H * dk * dv + head_id * dk * dv
                         + row_id * dv + tl.arange(0, BLOCK_DV)),
            h_row
        )
```

### Success Criteria
- Inter-chunk scan kernel time decreases by $> 20\%$ versus fla-org sequential loop
- L2 sector reads decrease by $> 50\%$ (measured via Nsight Compute)
- Results are bit-exact with the baseline
- Consistent across $G \in \{32, 64, 128\}$

### Failure Criteria
- If kernel time does *not* decrease despite fewer L2 accesses: the scan is compute-bound at this configuration. The optimization targets the wrong bottleneck.
- If occupancy drops below 25%: register pressure from holding the $d_v$-dim state row is too high. Need to split into multiple passes.

### Why This Test Is Sufficient
- The kernel microbenchmark directly tests the core hypothesis (HBM traffic reduction → speedup). If it works for one head, it works for all heads (trivially parallelized over the head dimension).
- The GLA diagonal case is the simplest; if it shows gains here, the Gated DeltaNet (low-rank) case will show even larger gains (since the low-rank matvec adds more computation per step, making the HBM traffic overhead proportionally larger).
- The kernel is a drop-in replacement — no model-level changes needed.

## Memory Access Pattern Analysis

**Current (sequential loop):**
- Per chunk boundary: Read $\gamma_k$ ($d_k$ values), read $\hat{h}_k$ ($d_k \times d_v$), read $h_{k-1}$ ($d_k \times d_v$), write $h_k$ ($d_k \times d_v$)
- Total HBM ops per step: $d_k + 3 d_k d_v$ elements = $64 + 3 \times 4096 = 12,352$ bf16 values = 24.7 KB
- **Coalesced?** Yes (contiguous memory layout)
- **Arithmetic intensity**: $(d_k \times d_v)$ FLOPs / 24.7 KB = 4096 / 24704 ≈ 0.17 FLOPs/byte → **extremely memory-bound**

**Proposed (FlashRNN-fused):**
- Per chunk boundary: Read $\gamma_k$ ($d_k$ values), read $\hat{h}_k$ ($d_k \times d_v$) — from HBM via TMA. State $h_{k-1}$ is **already in SRAM/registers**.
- Write $h_k$ to HBM (needed by downstream kernel)
- Total HBM ops per step: $d_k + 2 d_k d_v$ = $64 + 8192 = 8256$ bf16 values = 16.5 KB
- **Improvement**: 24.7 KB → 16.5 KB = **1.5× less HBM traffic** (eliminating the $h_{k-1}$ re-read)
- With TMA prefetch: The $\hat{h}_{k+1}$ load overlaps with the step-$k$ compute, hiding latency
- **Arithmetic intensity**: 4096 / 16512 ≈ 0.25 FLOPs/byte → still memory-bound but less so

**Further optimization — deferred write:**
If the downstream intra-chunk output kernel can consume $h_k$ from SRAM (via persistent kernel fusion, as in trick 182), the $h_k$ write can also be eliminated:
- Total HBM per step: $d_k + d_k d_v$ = 4160 values = 8.3 KB
- **Improvement vs baseline**: 24.7 KB → 8.3 KB = **3× less HBM traffic**

## Parallelism Analysis

- **Warp divergence**: None — all threads execute the same loop
- **Load imbalance**: None — each head is an independent program; all heads have same $G$
- **Tensor core mapping**: The GLA diagonal case is elementwise (no matmuls) — uses vector ALU. For Gated DeltaNet, the rank-$n_h$ correction involves small matmuls ($n_h \times d_v$) that can use tensor cores if $n_h \geq 16$
- **Sequential bottleneck**: $G$ sequential steps — inherent to the recurrence. But each step is $O(d_k \times d_v)$ independent ops across the state matrix.
- **SM utilization**: $H$ heads × batch_size independent programs. For $H = 16, B = 8$: 128 programs → good SM saturation on H100 (132 SMs)

## Theoretical Analysis

| Operation | Sequential Loop (baseline) | FlashRNN-Fused (proposed) |
|-----------|--------------------------|--------------------------|
| HBM reads per step | $O(d_k^2 + 2 d_k d_v)$ | $O(d_k + d_k d_v)$ |
| HBM writes per step | $O(d_k d_v)$ | $O(d_k d_v)$ (or $0$ if fused) |
| SRAM for state | $0$ | $O(d_k \times d_v)$ per head |
| Kernel launches | $1$ (but Python loop inside) | $1$ (persistent) |
| Total HBM traffic | $G \times O(d_k^2 + 3 d_k d_v)$ | $G \times O(d_k + 2 d_k d_v)$ |
| Async overlap | None | TMA prefetch of $\hat{h}_{k+1}$ |

Crossover: The fused kernel is always better when $d_k d_v > 0$ (i.e., always). The absolute savings grow with $G$: at $G = 128, d_k = d_v = 64$, savings are ~1 MB per head = 16 MB per layer.

## Risks & Limitations

1. **SRAM capacity**: The state $h_k \in \mathbb{R}^{d_k \times d_v}$ requires $d_k \times d_v \times 2$ bytes = 8 KB at $d_k = d_v = 64$. Plus the WY factors for DeltaNet: ~1 KB. H100 shared memory is 228 KB per SM — easily sufficient for multiple heads per SM. **Not a risk at typical dimensions.**

2. **Output writes still required**: We must write $h_k$ to HBM for the downstream intra-chunk output kernel. This limits the savings to ~1.5× rather than the theoretical 3× (which requires full layer fusion as in proposal 040). **Mitigation**: Combine with trick 182's atomic state-passing to pipe $h_k$ directly to the intra-chunk kernel via L2/DSM.

3. **Backward pass complexity**: The backward pass through the inter-chunk scan requires $\partial h_k / \partial h_{k-1} = A_k^{(C)}$ and $\partial h_k / \partial \hat{h}_k = I$. The same FlashRNN strategy applies: run the backward scan (reversed) with states in SRAM. The $\delta h$ gradient states are the same size as forward states.

4. **Small $G$ overhead**: For $G < 16$ (short sequences), the persistent kernel's setup cost may exceed the HBM savings. **Mitigation**: Fall back to the sequential loop for small $G$; use a dispatch mechanism.

5. **Gated DeltaNet rank-$n_h$ update**: The per-step update $h_k = \gamma_k (h_{k-1} - W_k (Y_k^\top h_{k-1})) + \hat{h}_k$ involves matmuls of size $n_h \times d_k$ and $d_k \times n_h$. For $n_h = 4$, these are small enough for SRAM compute. For $n_h > 8$, register pressure may become an issue.

## Follow-up Experiments

1. **Combine with MatMulScan (proposal 044)**: Use FlashRNN-fused scan for the sequential recurrence *and* MatMulScan for the parallel prefix components. The FlashRNN-fused approach handles the memory side; MatMulScan handles the compute side.

2. **Integrate with trick 182 (fused SSD)**: Replace the atomic state-passing in the fused SSD kernel with the FlashRNN-style SRAM-resident scan. This would make the entire fused SSD kernel fully SRAM-resident for inter-chunk states.

3. **Backward pass fusion**: Apply the same strategy to the backward inter-chunk gradient scan. The gradient $\delta h_k$ has the same structure as the forward scan but runs in reverse.

4. **TMA double-buffering**: Use double-buffered TMA loads to prefetch $\hat{h}_{k+2}$ while computing step $k$ and loading $\hat{h}_{k+1}$. This hides more HBM latency at the cost of 2× SRAM for the input buffer.

5. **Multi-head packing**: Pack multiple heads' scans into a single persistent kernel, sharing the SM's compute resources. With $d_k = d_v = 64$, each head's state is 8 KB — up to 28 heads could fit in H100's 228 KB SRAM simultaneously.

## Human Review

(To be filled by reviewer)

## References

- Pöppel, Beck & Hochreiter (2024). FlashRNN: I/O-Aware Optimization of Traditional RNNs on Modern Hardware. arXiv:2412.07752.
- Astra, Dao & Hoque (2026). Accelerating Mamba2 with Kernel Fusion. PyTorch Blog.
- Beck et al. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Dao & Gu (2024). Transformers are SSMs. ICML 2024.
- Yang, Wang, Shen, Panda & Kim (2024). Gated Linear Attention Transformers. ICML 2024.
- Yang, Kautz & Hatamizadeh (2025). Gated Delta Networks. ICLR 2025.
- Shah et al. (2024). FlashAttention-3. NeurIPS 2024.

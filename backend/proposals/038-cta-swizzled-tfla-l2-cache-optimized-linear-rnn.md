---
status: ongoing
priority: high
created: 2026-02-15
based_on: cta-tile-swizzling, tfla-two-level-tiled-chunkwise-parallelism, chunkwise-parallel-scan, io-aware-tiling, chimera-block-reorder-compute-fusion, epilogue-visitor-tree-fusion
experiment_number: 038
experiment_log: experiment-log-038.md
---

# CTA-Swizzled TFLA: L2 Cache-Optimized Linear RNN Kernels

## Hypothesis

Applying **CTA tile swizzling** to TFLA's two-level tiled kernel will improve L2 cache hit rates by **30–60%** on the memory-bound components (boundary state loads, gate mask computation, inter-chunk state reads), yielding a **1.1–1.25× end-to-end throughput improvement** for GLA/mLSTM training at sequence lengths $T \geq 4096$. Additionally, fusing the output gating/normalization epilogue via an **EVT-style epilogue visitor** into the TFLA parallel kernel eliminates 2–3 HBM round-trips per chunk, providing a further **1.05–1.15×** speedup. Combined, these pure kernel-level optimizations deliver **1.15–1.4× faster training** with zero architectural changes.

## Background

### TFLA's tile grid creates a swizzling opportunity

TFLA (Beck et al., NeurIPS 2025) parallelizes over 5 dimensions: batch ($B$), heads ($H$), chunks ($N_c = T/L$), query sequence tiles ($N_{Lhq} = L/B_{Lhq}$), and embedding tiles ($N_{dhv} = d_{hv}/B_{dhv}$). The GPU launches $B \times H \times N_c \times N_{Lhq} \times N_{dhv}$ CTAs for the parallel kernel.

For a typical configuration ($B = 8, H = 16, L = 256, T = 8192, d_{hv} = 64, B_{Lhq} = 64, B_{dhv} = 64$):
- $N_c = 32$ chunks, $N_{Lhq} = 4$ query tiles, $N_{dhv} = 1$ embedding tile
- Total CTAs per (batch, head): $32 \times 4 \times 1 = 128$
- Total CTAs: $8 \times 16 \times 128 = 16{,}384$

These CTAs have strong **data sharing patterns**:
1. **Same-chunk tiles** share $K^{(k)}, V^{(k)}$ data (loaded from HBM)
2. **Adjacent-chunk tiles** share boundary state $C_k$ (loaded from HBM)
3. **Same query-tile across chunks** share Q projection data

The default row-major CTA scheduling scatters these related tiles across SMs, causing L2 cache thrashing for the shared data. CTA swizzling co-locates related tiles, keeping shared data resident in L2.

### TFLA's memory-bound components

While TFLA's core matmuls ($QK^\top, SV$) are compute-bound on H100, several components are memory-bound:

| Component | Access pattern | Arithmetic intensity |
|-----------|---------------|---------------------|
| Boundary state $C_k$ load | $d_q \times d_{hv}$ per chunk, shared across $N_{Lhq}$ tiles | Low — each tile reads same $C_k$ |
| Gate mask $D^{(k)}$ computation | Elementwise on $B_{Lhq} \times B_{Lkv}$ | Low — mostly exp/sigmoid |
| Output normalization | Reduction + division on $B_{Lhq} \times B_{dhv}$ | Low — memory-bound |
| Output write | $B_{Lhq} \times B_{dhv}$ per tile | Memory-bound |

For these memory-bound components, L2 cache locality from swizzling directly translates to speedup.

### EVT epilogue fusion opportunity

After TFLA's parallel kernel computes $H^{(k)}$, the mLSTM/GLA layer typically applies:
1. **Output normalization**: $\hat{H} = H / \max(|n|, 1)$ where $n$ is a normalization state
2. **Output gating**: $\hat{H} = \hat{H} \odot \sigma(x W_{\text{og}})$ (output gate)
3. **Residual addition**: $y = x + \hat{H} W_O$

Currently these are separate kernels, each reading/writing $H$ from/to HBM. Using EVT (Epilogue Visitor Tree), we can fuse steps 1–2 into TFLA's parallel kernel epilogue: when the accumulator $h_{\text{acc}}$ is ready in registers, we apply normalization and gating before writing to HBM.

### What's different from existing proposals

- **Proposal 032** (Chimera-Fused Chunkwise SSM): Optimizes intra-chunk GEMM chain ordering using Chimera's analytical framework. Our proposal is complementary — we optimize the CTA scheduling (which tiles run on which SMs) and epilogue fusion, not the GEMM execution order within a tile.
- **Proposal 033** (EVT-Fused SSM Epilogues): Proposes EVT fusion for SSM projection layers. Our proposal applies EVT specifically to the TFLA parallel kernel's output path, which has different fusion opportunities (normalization + gating within the tiled output write).
- **Proposal 034** (Stream-K BRGEMM State Accumulation): Optimizes the recurrent kernel's state accumulation. Our proposal optimizes the parallel kernel's CTA scheduling and epilogue, which is the dominant cost.

## Related Work

- **SwizzlePerf** (Tschand et al., 2025): Uses LLMs to automatically find optimal swizzle patterns. Achieves 1.03–2.06× speedup on standard kernels (GEMM, softmax). Our approach applies swizzling specifically to TFLA's multi-dimensional tile grid, which has richer sharing patterns than standard GEMM.
- **Sawtooth Wavefront Reordering** (2026): Reduces L2 misses by 50–67% via wavefront-based CTA scheduling. Similar spirit to our swizzling, but targets different kernel structures.
- **TFLA** (Beck et al., NeurIPS 2025): Achieves SOTA kernel throughput but uses default CTA scheduling. The paper notes that further kernel optimizations are possible but does not explore swizzling or epilogue fusion.
- **FlashAttention-3** (Shah et al., 2024): Uses warp specialization and TMA for softmax attention. Does not apply CTA swizzling to the tile grid.

**Gap**: No existing work applies CTA tile swizzling or EVT epilogue fusion to TFLA or any chunkwise linear attention kernel.

## Mathematical Formulation

### TFLA CTA Grid Structure

The TFLA parallel kernel launches a 3D CTA grid (within each batch×head):

$$
\text{Grid} = N_c \times N_{Lhq} \times N_{dhv}
$$

where $N_c = T/L$, $N_{Lhq} = L/B_{Lhq}$, $N_{dhv} = d_{hv}/B_{dhv}$.

**Default scheduling** assigns CTA with linear PID $p$ to grid coordinates:

$$
(c, i, j) = \left(\left\lfloor \frac{p}{N_{Lhq} \cdot N_{dhv}} \right\rfloor, \; \left\lfloor \frac{p \bmod (N_{Lhq} \cdot N_{dhv})}{N_{dhv}} \right\rfloor, \; p \bmod N_{dhv}\right)
$$

### Proposed: Swizzled CTA Scheduling

We apply swizzling over the $(c, i)$ dimensions (chunk index × query tile index), grouping into super-tiles of size $2^s \times 2^s$:

$$
c' = \left\lfloor \frac{p'}{2^s} \right\rfloor \cdot 2^s + (p' \bmod 2^s), \quad i' = \left\lfloor \frac{p' \bmod (2^s \cdot N_{Lhq})}{2^s} \right\rfloor
$$

where $p' = p / N_{dhv}$ (embedding tiles are already independent, no sharing).

**Swizzle factor selection** for TFLA:

$$
s^* = \arg\min_s \; \text{HBM}_{\text{traffic}}(s) = \arg\min_s \left[\frac{N_c \cdot N_{Lhq}}{2^{2s}} \cdot \text{cold\_start} + N_c \cdot N_{Lhq} \cdot \text{per\_tile}\right]
$$

Since tiles within the same chunk share $K^{(k)}, V^{(k)}$ data ($L \cdot d \cdot 2$ bytes per chunk), the optimal $s$ should group all $N_{Lhq}$ query tiles of each chunk together:

$$
s^* = \lceil \log_2 N_{Lhq} \rceil
$$

For $L = 256, B_{Lhq} = 64$: $N_{Lhq} = 4$, so $s^* = 2$ (4×4 super-tiles).

### L2 Cache Reuse Analysis

**Shared data per chunk $k$:**

| Data | Size (bytes, FP16) | Shared across |
|------|-------------------|---------------|
| $K^{(k)}$ | $L \cdot d_q \cdot 2 = 256 \times 64 \times 2 = 32$ KB | All $N_{Lhq}$ tiles in chunk $k$ |
| $V^{(k)}$ | $L \cdot d_{hv} \cdot 2 = 32$ KB | All $N_{Lhq}$ tiles in chunk $k$ |
| $C_{k-1}$ | $d_q \cdot d_{hv} \cdot 2 = 8$ KB | All $N_{Lhq}$ tiles in chunk $k$ |
| Gate values $f, i$ | $L \cdot 2 \cdot 2 = 1$ KB | All tiles in chunk $k$ |

**Total shared data per chunk**: $\sim 73$ KB — fits comfortably in H100's L2 cache (50 MB shared across 132 SMs, ~380 KB per SM cluster).

**Without swizzling**: If same-chunk tiles are scattered across SM clusters, each cluster loads its own copy → $N_{Lhq} \times 73$ KB = 292 KB total L2 traffic.

**With swizzling** ($s = 2$): Same-chunk tiles co-located on same SM cluster → $1 \times 73$ KB = 73 KB total L2 traffic → **4× reduction** for shared data.

### EVT Epilogue Fusion

**Standard (unfused) output path:**

```
TFLA parallel kernel → write H to HBM (BW₁)
Norm kernel: read H from HBM (BW₂) → compute H/max(|n|,1) → write Ĥ to HBM (BW₃)
Gate kernel: read Ĥ from HBM (BW₄) → compute Ĥ⊙σ(gate) → write to HBM (BW₅)
```

Total HBM bandwidth: $5 \times T \times d_{hv} \times 2$ bytes = $5 \times 8192 \times 64 \times 2 = 5.2$ MB per head.

**Fused (EVT epilogue):**

```
TFLA parallel kernel → in registers: H/max(|n|,1) ⊙ σ(gate) → write to HBM (BW₁)
```

Total HBM bandwidth: $1 \times T \times d_{hv} \times 2 = 1.0$ MB per head → **5× reduction**.

The EVT epilogue applies two operations to each element of the $B_{Lhq} \times B_{dhv}$ accumulator before the store instruction:

$$
\text{out}_{ij} = \frac{h_{\text{acc},ij}}{\max(|n_i|, 1)} \cdot \sigma(g_j)
$$

where $n_i$ (normalization denominator, 1 value per query position) and $g_j$ (gate value, 1 value per embedding dim) are loaded from HBM once per tile.

### Key Variables

- $T$ — sequence length
- $L$ — TFLA chunk size (128–1024)
- $B_{Lhq}$ — query tile size (32–128)
- $B_{Lkv}$ — KV tile size (32–128)
- $d_q, d_{hv}$ — head dimensions
- $s$ — swizzle factor ($\lceil \log_2 N_{Lhq} \rceil$)
- $N_c = T/L$ — number of chunks
- $N_{Lhq} = L/B_{Lhq}$ — query tiles per chunk
- $C_k \in \mathbb{R}^{d_q \times d_{hv}}$ — boundary state

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | mLSTMsig / GLA (unchanged architecture) |
| Kernel | TFLA + CTA swizzling + EVT epilogue |
| Layers | $L_{\text{layers}} = 12$–$24$ |
| Hidden dim | $d_{\text{model}} = 768$–$2048$ |
| Heads | $H = 8$–$16$ |
| Head dim | $d_q = d_{hv} = 64$–$128$ |
| Chunk size | $L = 128, 256, 512$ |
| Query tile | $B_{Lhq} = 64$ |
| KV tile | $B_{Lkv} = 64$ |

### Baseline

1. **TFLA (default scheduling)**: Current SOTA from Beck et al. — no swizzling, separate norm/gate kernels
2. **FlashAttention-3**: Softmax attention with warp specialization — includes built-in swizzling
3. **FLA (single-level)**: Standard Flash Linear Attention with $L = 64$

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| L2 hit rate | $> 80\%$ (vs ~50% baseline) | NVIDIA Nsight Compute profiling |
| Parallel kernel throughput | $\geq 1.1\times$ vs TFLA baseline | TFLOPs/s on H100 |
| End-to-end training throughput | $\geq 1.15\times$ vs TFLA baseline | Tokens/sec for full model |
| Memory (unchanged) | Same as TFLA baseline | Peak HBM usage |
| Model quality (unchanged) | Identical to TFLA baseline | Bit-exact outputs |

### Estimated Compute

- **MVE (kernel benchmarking)**: ~15 minutes on single H100 (~$5)
- **Profiling with Nsight**: 2 GPU-hours on H100 (~$8)
- **Full training comparison**: 8 GPU-hours on H100 (~$32)

## Expected Outcome

**If hypothesis is correct:**

1. **L2 hit rate**: Increases from ~50% to ~80%+ for TFLA's parallel kernel. The 73 KB shared data per chunk fits in L2 and stays resident across all $N_{Lhq}$ tiles when swizzled.

2. **Parallel kernel speedup**: 1.1–1.25× on memory-bound components. Since TFLA's parallel kernel is ~60% compute-bound (matmuls) and ~40% memory-bound (state loads, gate masks, output writes), improving the memory-bound portion by 2× gives overall 1.1–1.25× kernel speedup.

3. **EVT epilogue savings**: Eliminating 4 out of 5 HBM round-trips for the output path saves ~4.2 MB per head per forward pass. At $H = 16, B = 8$: $16 \times 8 \times 4.2 = 538$ MB saved — significant at large batch sizes.

4. **Combined speedup**: $1.15 \times 1.1 = 1.27\times$ end-to-end (optimistic); $1.1 \times 1.05 = 1.15\times$ (conservative).

5. **Zero quality change**: This is a pure kernel optimization — the mathematical computation is identical, so model outputs are bit-exact.

**If hypothesis is wrong:**

- **Scenario A**: L2 hit rate improves but no wall-clock speedup
  - **Learn**: TFLA's parallel kernel is already fully compute-bound at this configuration. The memory-bound components are negligible.
  - **Insight**: Focus optimization efforts on the compute-bound matmuls (e.g., better tile shapes for tensor cores, instruction scheduling).

- **Scenario B**: Swizzling helps but EVT fusion doesn't
  - **Learn**: The norm/gate kernels are already fast enough that launch overhead dominates over HBM savings. Or: the EVT visitor adds register pressure that hurts the GEMM occupancy.
  - **Next**: Benchmark EVT register pressure impact; potentially use partial fusion (gate only, skip norm).

- **Scenario C**: Swizzling causes load imbalance
  - **Learn**: The super-tile grouping creates waves where some SMs are idle. Need adaptive swizzle factor tuned to the specific grid dimensions.
  - **Next**: Implement SwizzlePerf-style auto-tuning to find the optimal $s$ per configuration.

## Minimum Viable Experiment

### Setup
- **Kernel**: TFLA mLSTMsig parallel kernel from `mlstm_kernels` (open-source Triton)
- **Modification**: Add swizzle remapping to the PID computation (5 lines of Triton code)
- **Configuration**: $B = 4, H = 8, T = 8192, d = 64, L = 256, B_{Lhq} = 64, B_{Lkv} = 64$
- **Hardware**: Single H100 GPU
- **Benchmark**: Measure kernel execution time (median of 100 runs) with and without swizzling
- **Profile**: Use Nsight Compute to measure L2 hit rates before and after
- **Compute**: Single GPU, $< 15$ minutes

### Implementation (5-line change)

```python
# In TFLA parallel kernel, replace:
chunk_id = tl.program_id(0)
query_tile_id = tl.program_id(1)

# With swizzled version:
pid = tl.program_id(0)  # linearized (chunk × query_tile)
SWIZZLE = 2  # log2(N_Lhq) = log2(4) = 2
group_id = pid // (4 * N_Lhq)  # 4 = 2^SWIZZLE
group_size = min(N_c - group_id * 4, 4)
chunk_id = group_id * 4 + (pid % group_size)
query_tile_id = (pid % (group_size * N_Lhq)) // group_size
```

### Success Criteria
- L2 hit rate increases by $> 20\%$ (absolute) as measured by Nsight Compute
- Parallel kernel execution time decreases by $> 5\%$ (statistically significant, $p < 0.05$)
- Results consistent across 3 different sequence lengths ($T = 4096, 8192, 16384$)

### Failure Criteria
- L2 hit rate increases but kernel time does not decrease — kernel is compute-bound, not memory-bound
- Swizzling causes $> 5\%$ slowdown due to load imbalance — super-tile grouping creates bubbles

### Why This Test Is Sufficient
- **Directly measures the hardware effect**: L2 hit rate is the causal mechanism. If it improves but throughput doesn't, the kernel is compute-bound and swizzling won't help at any scale. If both improve, the effect will persist (and likely grow) at larger scales where the tile grid is bigger and cache pressure is worse.
- **5-line change**: The modification is trivial to implement and revert, making the experiment low-risk.
- **Open-source baseline**: The `mlstm_kernels` repository provides an optimized Triton implementation as the baseline, ensuring we compare against real SOTA rather than a strawman.

## Theoretical Analysis

### Complexity Comparison

| Operation | TFLA (default) | TFLA + Swizzle + EVT |
|-----------|---------------|---------------------|
| Parallel kernel FLOPs | $O(N_c L^2 (d_q + d_{hv}))$ | Same (unchanged) |
| Parallel kernel HBM | $O(N_c L (d_q + d_{hv}))$ | Same (data volume unchanged) |
| L2 cache misses | $\sim N_c \cdot N_{Lhq} \cdot (L d + d^2)$ | $\sim N_c \cdot (L d + d^2)$ ($N_{Lhq}\times$ reduction) |
| Epilogue HBM traffic | $5 \times T \cdot d_{hv}$ | $1 \times T \cdot d_{hv}$ ($5\times$ reduction) |
| Extra kernel launches | 2 (norm + gate) | 0 (fused) |

### GPU Hardware Analysis

**Memory Access Pattern:**
- Swizzled CTAs access $K^{(k)}, V^{(k)}$ sequentially within each chunk — **coalesced reads** preserved
- Gate values accessed per-position — coalesced within swizzled tiles (consecutive positions land on same SM cluster)
- Boundary state $C_k$ is a small $d_q \times d_{hv}$ matrix — fits in L1 cache; L2 benefit is from avoiding re-fetch across scattered CTAs

**Parallelism:**
- Swizzling reorders CTAs but does not reduce parallelism — same number of independent work items
- Potential load imbalance only at grid boundaries (last super-tile may be partial)
- On H100 with 132 SMs: 16,384 CTAs gives $> 100$ waves — any imbalance averages out

**Tensor Core Compatibility:**
- Zero impact on tensor core utilization — swizzling changes CTA-to-SM mapping, not the matmul operations within each CTA
- EVT epilogue adds elementwise ops in registers between the last MMA and the store — no tensor core impact

**Register Pressure:**
- EVT fusion adds ~8 registers per thread (for norm denominator $n_i$, gate value $g_j$, and temporaries)
- TFLA parallel kernel uses ~128 registers per thread (for $B_{Lhq} \times B_{dhv}$ accumulator in FP32)
- Total: ~136 registers — within H100's 256-register limit, occupancy should not decrease
- If occupancy drops: fall back to partial fusion (gate only, 4 extra registers)

**Shared Memory:**
- No additional shared memory required — swizzling is a PID remapping computed in registers
- EVT loads $n_i$ (1 float per query position) and $g_j$ (1 float per embedding dim) — negligible shared memory

## Risks & Limitations

### Risk 1: TFLA Already Compute-Bound
- **Issue**: If TFLA's parallel kernel is >90% compute-bound (matmuls saturate tensor cores), improving L2 hit rate has negligible wall-clock impact.
- **Mitigation**: Profile first with Nsight Compute to measure the roofline position. If already compute-bound, skip swizzling and focus on EVT fusion only.
- **Detection**: If L2 hit rate improves 30%+ but kernel time improves <2%, the kernel is compute-bound.

### Risk 2: Swizzle-Induced Load Imbalance
- **Issue**: If $N_c$ or $N_{Lhq}$ is not divisible by $2^s$, the last super-tile has fewer CTAs, leaving SMs idle at the end of each wave.
- **Mitigation**: Use adaptive $s$ based on grid dimensions. For $N_{Lhq} = 4$, $s = 2$ gives perfect division. For odd $N_{Lhq}$, fall back to $s = 1$.

### Risk 3: EVT Register Pressure
- **Issue**: Adding 8 registers for the epilogue visitor may reduce occupancy from 2 active warps to 1 on some SMs, halving throughput.
- **Mitigation**: Profile register usage with Nsight. If occupancy drops, use partial fusion or spill the norm/gate values to shared memory.

### Risk 4: Triton Limitations
- **Issue**: Triton's JIT compiler may not support arbitrary PID remapping efficiently, or may not fuse epilogue operations into the GEMM epilogue.
- **Mitigation**: Start with Triton (modify existing `mlstm_kernels` code). If Triton is insufficient, implement in CUTLASS 3.x which has native EVT support and explicit CTA swizzle policies.

### Risk 5: Architecture-Specific Tuning
- **Issue**: Optimal swizzle factor differs between A100 (108 SMs, 40 MB L2) and H100 (132 SMs, 50 MB L2) and MI300X (8 XCDs, 256 MB L2). Results on one GPU may not transfer.
- **Mitigation**: Test on both A100 and H100. Report L2 hit rate alongside speedup so the causal mechanism is clear.

## Follow-up Experiments

### If Successful:
1. **Auto-tune swizzle factor**: Use SwizzlePerf-style search over $s \in \{0, 1, 2, 3\}$ and tile dimensions to find optimal configuration per GPU architecture
2. **Apply to recurrent kernel**: TFLA's recurrent kernel (state accumulation) also has tile-level sharing across chunks — swizzling may help there too
3. **Combine with Chimera (proposal 032)**: Apply Chimera's analytical GEMM-chain reordering within each swizzled super-tile to optimize intra-tile data reuse
4. **Extend EVT fusion**: Fuse the residual addition and next-layer LayerNorm into the same epilogue (4 ops in 1 kernel launch)
5. **Apply to DeltaNet/GLA TFLA kernels**: Test swizzling on the fla-org implementations (different tile shapes, different sharing patterns)
6. **Hilbert-curve scheduling**: Test space-filling curve CTA ordering instead of block swizzling for better locality at non-power-of-2 grid sizes

### If Unsuccessful:
1. **Profile-guided optimization**: Use Nsight traces to identify the actual bottleneck (instruction latency? memory bank conflicts? warp scheduling?)
2. **Warp specialization**: Instead of swizzling, use FlashAttention-3-style warp specialization within TFLA tiles (producer/consumer warps for K/V prefetching)
3. **TMA async loads**: Use H100's Tensor Memory Accelerator to overlap K/V loads with compute, reducing the sensitivity to L2 hit rates
4. **Different tile shapes**: Explore $B_{Lhq} \times B_{Lkv}$ configurations that better match MMA tile shapes (e.g., $128 \times 32$ instead of $64 \times 64$)

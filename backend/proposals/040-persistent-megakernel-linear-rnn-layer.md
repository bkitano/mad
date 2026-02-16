---
status: completed
priority: high
created: 2026-02-15
based_on: persistent-megakernel-fusion (091), epilogue-visitor-tree-fusion (039), kernel-fusion (068), fusionstitching-multi-level-reuse (051), horizontal-kernel-fusion (061), warp-specialized-pipelining (141), tfla-two-level-tiled-chunkwise-parallelism (158), bilinear-gating-glu (005), post-attention-sigmoid-gating (094)
experiment_number: 040
experiment_log: experiment-log-040.md
results_file: 040_results.md
---

# Persistent Megakernel Fusion for Full Linear RNN Layers

## Hypothesis

Fusing the entire linear RNN layer — input projections ($W_Q, W_K, W_V$), chunkwise recurrence (scan or matmul), output gating (SiLU/sigmoid), output projection ($W_O$), and residual-add + normalization — into a **single persistent megakernel** will reduce HBM round-trips from $\geq 6$ (current multi-kernel approach) to $1$, yielding $1.5$–$2.5\times$ wall-clock speedup for linear RNN layers during pretraining, with the largest gains at small-to-medium batch sizes where kernel launch overhead and memory bandwidth dominate.

## Background

**Current implementation:** A typical GLA or mLSTM layer in a modern pretraining framework requires **6+ separate kernel launches** per layer:

1. **Input projections**: $Q = x W_Q$, $K = x W_K$, $V = x W_V$ (3 GEMMs or 1 fused GEMM)
2. **Gate computation**: $\gamma = \sigma(x W_\gamma)$ (GEMM + elementwise)
3. **Chunkwise scan/attention**: The TFLA kernel (1 fused kernel)
4. **Output gating**: $o = \text{SiLU}(\text{gate}) \odot \text{scan\_output}$ (elementwise)
5. **Output projection**: $y = o W_O$ (GEMM)
6. **Residual + LayerNorm**: $z = \text{LN}(x + y)$ (reduction + elementwise)

Each kernel launch incurs:
- **Launch overhead**: ~5-10 μs per launch × 6 = 30-60 μs per layer
- **HBM round-trip**: Each kernel reads inputs from and writes outputs to HBM (high-bandwidth memory). Intermediate tensors ($Q, K, V, \gamma$, scan output, gated output) are materialized in HBM between kernels.
- **Total HBM traffic**: $\geq 6 \times O(B \cdot T \cdot d)$ reads + writes of intermediate activations

For a 350M-parameter model with 24 layers at batch size 8, sequence length 4096, $d = 1024$: each intermediate tensor is $8 \times 4096 \times 1024 \times 2 = 64$ MB. Six intermediates = 384 MB per layer, 9.2 GB total across layers. At H100's 3.35 TB/s HBM bandwidth, just reading/writing intermediates takes ~2.7 ms — a significant fraction of total training time.

**What Mirage MPK showed (Jia et al., 2025):** Compiling an entire Llama-1B inference pass into a single megakernel reduces latency by 1.2–6.7× by eliminating all inter-kernel HBM materialization. The key insight: **intermediates stay in SMEM/registers** across fused operations.

**Gap filled:**
- Proposal 033 (EVT-fused SSM SwiGLU epilogues) fuses only the *epilogue* (post-scan gating + output projection) into the scan kernel's epilogue. It doesn't fuse the *prologue* (input projections) or the residual/norm.
- Proposal 032 (Chimera-fused chunkwise SSM) optimizes inter-block execution order within the chunkwise computation but doesn't fuse across the full layer.
- No existing proposal attempts full layer-level megakernel fusion for linear RNNs during pretraining.

## Related Work

- **Mirage Persistent Kernel (Jia et al., Dec 2025)**: Compiler for inference megakernels. Demonstrated on Transformer inference (Llama). Doesn't handle training (backward pass) or linear RNN architectures. Our approach targets **pretraining** and specifically the linear RNN computation graph.
- **Hazy Research "No Bubbles" (May 2025)**: Hand-designed megakernel for Llama-1B inference achieving 78% memory bandwidth utilization. Inference-only, Transformer-only. Shows the potential gains from megakernel fusion.
- **FlashMoE (091)**: Persistent megakernel for MoE layers, fusing gate + dispatch + FFN + combine into one kernel. Our approach is analogous but for linear RNN layers instead of MoE.
- **ThunderMLA (Hazy Research, Mar 2025)**: Fused MLA (Multi-Latent Attention) kernel for decode. Close to our goal but targets attention decode, not linear RNN training.
- **FusionStitching (051)**: Multi-level data reuse framework (thread-local, intra-warp, intra-block). Provides principles for our fusion but hasn't been applied to linear RNN layers.

**Our approach differs**: We fuse the complete linear RNN layer (prologue projections + chunkwise recurrence + epilogue gating + output projection + residual/norm) into a single persistent kernel for **training**, including both forward and backward passes.

## Mathematical Formulation

**Standard GLA Layer (6 kernels):**

$$
Q = x W_Q, \quad K = x W_K, \quad V = x W_V, \quad \gamma = \sigma(x W_\gamma)
$$

$$
S_t = \gamma_t S_{t-1} + k_t v_t^T \quad \text{(chunkwise via TFLA)}
$$

$$
o_t = q_t^T S_t \quad \text{(readout)}
$$

$$
y = \text{SiLU}(x W_g) \odot (o W_O) \quad \text{(gated output)}
$$

$$
z = \text{LayerNorm}(x + y) \quad \text{(residual + norm)}
$$

**HBM Traffic Analysis (current):**

| Kernel | HBM Reads | HBM Writes | Total |
|--------|-----------|------------|-------|
| Input proj | $x$ ($BTd$) | $Q,K,V,\gamma$ ($4BTd$) | $5BTd$ |
| Chunkwise | $Q,K,V,\gamma$ ($4BTd$) | $o$ ($BTd$) | $5BTd$ |
| Output gate | $o, x W_g$ ($2BTd$) | $y$ ($BTd$) | $3BTd$ |
| Output proj | $y$ ($BTd$) | $y'$ ($BTd$) | $2BTd$ |
| Residual+Norm | $x, y'$ ($2BTd$) | $z$ ($BTd$) | $3BTd$ |
| **Total** | | | $\mathbf{18 BTd}$ |

**Fused Megakernel:**

| Phase | HBM Reads | HBM Writes | Notes |
|-------|-----------|------------|-------|
| Load $x$ | $BTd$ | — | Read once |
| Stream $W_Q, W_K, W_V, W_\gamma$ | $5d^2$ | — | Weight tiles streamed |
| Store $z$ | — | $BTd$ | Final output only |
| **Total** | $\mathbf{BTd + 5d^2}$ | $\mathbf{BTd}$ | $\approx \mathbf{2BTd + 5d^2}$ |

**HBM Traffic Reduction:**

$$
\text{Reduction} = \frac{18 BTd}{2 BTd + 5d^2} \approx 9\times \quad \text{(for } BT \gg d \text{)}
$$

For $B=8, T=4096, d=1024$: current = 18 × 64 MB = 1.15 GB; fused = 2 × 64 + 5 × 4 MB = 148 MB. A $7.8\times$ reduction in HBM traffic.

**Megakernel Execution Model:**

The persistent kernel uses a **tile-streaming pipeline**:

1. **Phase 1 (Prologue):** Load tile of $x$ from HBM → compute $Q, K, V, \gamma$ via tiled GEMM with weight tiles → store $Q, K, V, \gamma$ in SMEM (not HBM)
2. **Phase 2 (Recurrence):** Run chunkwise linear attention using $Q, K, V, \gamma$ from SMEM → produce scan output $o$ in SMEM
3. **Phase 3 (Epilogue):** Compute gated output $\text{SiLU}(g) \odot o$ → output projection $y = o' W_O$ (tiled GEMM) → residual add $x + y$ → fused LayerNorm → store $z$ to HBM

**Key insight:** $Q, K, V, \gamma$, and $o$ are **never written to HBM**. They exist only in SMEM/registers between phases. This is feasible because the chunkwise computation processes tiles of size $B_{Lhq} \times d$ which fit in SMEM (~16 KB per tile on H100's 228 KB SMEM budget).

**SMEM Budget Analysis:**

| Buffer | Size | Notes |
|--------|------|-------|
| $x$ tile | $B_{Lhq} \times d$ × 2 bytes | Input tile |
| $Q$ tile | $B_{Lhq} \times d_q$ × 2 bytes | Query tile |
| $K, V$ tiles | $B_{Lkv} \times (d_q + d_v)$ × 2 bytes | Key/value tiles |
| $\gamma$ tile | $B_{Lhq}$ × 4 bytes | Decay gates (FP32) |
| $W$ tile | $d \times B_d$ × 2 bytes | Weight tile for projection |
| Accumulators | $B_{Lhq} \times d_v$ × 4 bytes | FP32 output accumulator |
| Pipeline buffers (2×) | 2× above | Double-buffered |

For $B_{Lhq} = B_{Lkv} = 64$, $d_q = d_v = 128$, $d = 1024$, $B_d = 64$:
- $x$: 128 KB (too large!)

**Resolution:** Tile the hidden dimension $d$ as well. Process $d$ in blocks of $B_d = 64$–$128$. Each GEMM tile ($B_{Lhq} \times B_d \times d_q$) fits in SMEM. This requires accumulating the GEMM output across $d / B_d$ iterations.

Revised SMEM budget per tile:
- $x$ tile: $64 \times 128 \times 2 = 16$ KB
- $W_Q$ tile: $128 \times 128 \times 2 = 32$ KB
- $Q$ accumulator: $64 \times 128 \times 4 = 32$ KB
- Double-buffered: $2 \times (16 + 32) = 96$ KB + 32 KB accum = 128 KB

This fits in H100's 228 KB SMEM. For the chunkwise phase, $K, V$ tiles reuse the $W$ buffer space (sequential phases, not concurrent).

**Key Variables:**
- $B$ — batch size
- $T$ — sequence length
- $d$ — model hidden dimension
- $d_q, d_v$ — head dimensions
- $H$ — number of heads
- $B_{Lhq}, B_{Lkv}$ — sequence tile sizes
- $B_d$ — hidden dimension tile size
- $\gamma_t$ — per-token decay gate

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA (Gated Linear Attention) |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d = 768$–$1024$ |
| Heads | $H = 12$–$16$ |
| Head dim | $d_q = d_v = 64$–$128$ |
| Chunk size (outer) | $L = 512$ |
| Inner tiles | $B_{Lhq} = B_{Lkv} = 64$ |
| Hidden tile | $B_d = 128$ |
| GPU | H100 (SM90a) |

### Baseline
1. **Multi-kernel GLA** (current): Separate cuBLAS GEMMs + Triton TFLA kernel + elementwise kernels. 6+ launches per layer. Complexity: $O(BTd^2 + BTd_q d_v)$
2. **Partially-fused GLA** (Proposal 033 epilogue fusion): Fuses output gating into scan epilogue. 4 launches per layer.
3. **FlashAttention-3 (softmax)**: For throughput comparison — the ceiling for fused attention kernels on Hopper.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Layer throughput | $> 1.5\times$ multi-kernel | Tokens/sec per layer on H100 |
| HBM traffic | $< 3 BTd$ bytes | `ncu` DRAM read+write |
| Kernel launches per layer | $1$ | Count via profiler |
| End-to-end training | $> 1.2\times$ speedup | Wall-clock for 350M model pretraining |
| Quality | Identical | Match multi-kernel perplexity |

### Estimated Compute
**MVE**: < 30 minutes on single H100 (microbenchmark)
**Phase 1** (kernel development + profiling): ~20 GPU-hours on H100
**Phase 2** (end-to-end pretraining, 350M params, 10B tokens): ~100 GPU-hours on H100
**Total**: ~120 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- $1.5$–$2.5\times$ speedup per layer at batch size 8–16 (memory-bandwidth-bound regime)
- Kernel launch overhead eliminated entirely (from ~50 μs/layer to ~8 μs/layer)
- HBM traffic reduced from $18 BTd$ to $\leq 3 BTd$, verified by `ncu` DRAM counters
- End-to-end pretraining speedup of $1.2$–$1.5\times$ (other overheads: optimizer, data loading, gradient sync)
- Largest gains at small batch sizes where memory-bandwidth dominates; diminishing returns at very large batches where GEMMs become compute-bound

**If hypothesis is wrong:**
- If SMEM budget is insufficient: the input tile $x$ at full $d$ doesn't fit alongside weight tiles and scan state. This would require a 3-phase approach with HBM spills for the largest buffers, reducing the traffic savings. Still better than 6 kernels.
- If prologue GEMMs dominate: the input projections ($O(BTd^2)$) are compute-bound and already well-optimized by cuBLAS. Fusing them with the scan may reduce GEMM efficiency due to suboptimal tiling. This would motivate keeping projections as separate cuBLAS calls and only fusing scan + epilogue + residual (a "semi-megakernel").
- If backward pass fusion is infeasible: training requires backward through all fused operations. The backward of the chunkwise scan interleaved with backward of GEMMs is complex. Mitigation: fuse forward only; backward uses multi-kernel (still useful for inference-heavy workloads).

## Minimum Viable Experiment

### Setup
- **Kernel**: Forward-pass-only megakernel fusing: linear projection ($x W_V$) → scalar gated scan ($\gamma_t s_{t-1} + k_t v_t$) → output gating ($\text{SiLU}(g) \odot o$). A 3-operation fusion.
- **Model**: Single-head, $d = 256$, $d_v = 64$, $T = 2048$, $B = 4$
- **Baseline**: 3 separate Triton kernels (GEMM, scan, elementwise)
- **Compute**: Single H100, < 20 minutes

### Success Criteria
- Fused kernel achieves $> 1.3\times$ throughput over 3-kernel baseline
- `ncu` DRAM read+write reduced by $> 2\times$
- Output matches 3-kernel baseline within BF16 tolerance

### Failure Criteria
- If fused kernel is $< 1.05\times$ faster: the GEMM is so compute-bound that eliminating the SMEM→HBM→SMEM round-trip between GEMM and scan doesn't matter. The bottleneck is compute, not memory.
- If numerical accuracy is poor: intermediate values in reduced precision (BF16) accumulate errors across fused phases. Would need FP32 intermediate buffers, increasing SMEM pressure.

### Why This Test Is Sufficient
- The 3-operation fusion captures the core mechanism: keeping $V = x W_V$ in SMEM between the projection and the scan. If this works, extending to full layer fusion adds more operations but doesn't change the fundamental memory-traffic argument.
- The SMEM budget at $d = 256$ is manageable (~8 KB per tile); scaling to $d = 1024$ is a tiling challenge, not a fundamental limitation.
- Forward-only suffices to validate the throughput gain; backward adds implementation complexity but doesn't change the HBM traffic analysis.

## Memory Access Pattern Analysis

**Coalesced access:** Input $x$ and output $z$ are read/written in contiguous tiles (batch × sequence × hidden). Weight matrices are tiled and loaded via TMA (hardware-managed coalescing).

**Cache-friendly:** Persistent kernel retains all intermediate results in SMEM/registers. No L2 cache pressure from intermediates. Only weights stream through L2.

**Arithmetic intensity:** For full layer: $\sim 8 d_q d_v + 2 d^2$ FLOPs per token / $2d + d$ bytes per token read+write = $(8 \times 128^2 + 2 \times 1024^2) / (3 \times 1024 \times 2) \approx 373$ FLOPs/byte. Very compute-bound overall, but the multi-kernel approach artificially makes it memory-bound by materializing intermediates.

**HBM bandwidth:** With fusion, effective bandwidth utilization increases because the only HBM traffic is loading $x$ (once) and storing $z$ (once), plus streaming weight tiles. Weight tiles benefit from H100's large L2 cache (50 MB) when batch × sequence is small.

## Parallelism Analysis

**SM saturation:** Each CTA processes one output tile ($B_{Lhq} \times d_v$ or $B_{Lhq} \times B_d$). With $T / B_{Lhq} \times H \times d / B_d$ CTAs, there are typically $> 1000$ tiles — excellent parallelism.

**No warp divergence:** All phases execute the same instruction sequence per CTA. Phase transitions are uniform across warps.

**Tensor core mapping:** Input projections and output projections are standard GEMMs → full tensor core utilization. Chunkwise scan includes matmuls (tensor core) + elementwise decay (CUDA core). The pingpong schedule (from Proposal 039) can overlap these.

**Sequential bottleneck:** Inter-chunk state propagation is sequential but lightweight ($O(d_v^2)$ per chunk boundary). Negligible compared to intra-chunk work.

## Theoretical Analysis

Complexity comparison (per layer, forward pass):

| Operation | Multi-Kernel | Megakernel |
|-----------|-------------|------------|
| FLOPs | $O(BTd^2 + BT d_q d_v)$ | Same |
| HBM reads | $\geq 9 BTd + 5d^2$ | $BTd + 5d^2$ |
| HBM writes | $\geq 9 BTd$ | $BTd$ |
| Kernel launches | 6+ | 1 |
| Launch overhead | $\sim 50$ μs | $\sim 8$ μs |
| SMEM per CTA | $O(B_{tile} \times d)$ | $O(B_{tile} \times B_d)$ |

**Crossover:** The megakernel is advantageous when the layer is memory-bandwidth-bound, i.e., when the arithmetic intensity of *individual kernels* is low even though the overall layer is compute-bound. This happens at small batch sizes ($B \leq 16$) where individual GEMMs have small $M$ dimension.

## Risks & Limitations

1. **SMEM budget:** The full layer requires holding input tile, weight tile, Q/K/V/γ intermediates, scan state, and output accumulator simultaneously. At $d = 1024$, this may exceed H100's 228 KB SMEM. Mitigation: tile the hidden dimension $d$ in blocks of $B_d = 64$–$128$; process sequentially within the CTA.

2. **GEMM efficiency:** cuBLAS GEMMs use highly optimized tiling, split-K, and auto-tuning. A fused megakernel's GEMM tiles may be suboptimal due to additional SMEM pressure from scan state. Mitigation: benchmark at each fusion level to find the sweet spot.

3. **Backward pass complexity:** Training requires backward through all fused operations. The backward of the chunkwise scan interleaved with backward of GEMMs creates complex data dependencies. Mitigation: start with forward-only fusion (useful for inference and gradient checkpointing scenarios); extend to backward incrementally.

4. **Generality:** Different linear RNN variants (GLA, mLSTM, DeltaNet) have different scan structures. A single megakernel may need per-architecture specialization. Mitigation: use a template-based approach (like EVT's visitor tree) where the scan phase is a pluggable component.

5. **Debugging and maintenance:** Megakernels are notoriously hard to debug and maintain. Mitigation: modular design with compile-time composition (CUTLASS 3.x style); extensive testing via comparison with multi-kernel reference.

6. **Diminishing returns at large batch:** When batch × sequence is very large, the GEMMs become compute-bound and cuBLAS's tuned kernels may match or beat the fused kernel. The fusion primarily helps in the memory-bound regime (small-to-medium batch sizes).

## Follow-up Experiments

1. **Backward pass fusion:** Extend the megakernel to fuse the backward pass as well, keeping gradient intermediates in SMEM. This roughly doubles the SMEM requirement but also doubles the HBM traffic savings.

2. **Multi-layer fusion:** Fuse 2–3 consecutive linear RNN layers into a single megakernel (Mirage-style). The output $z$ of layer $l$ is the input $x$ of layer $l+1$ — keeping it in SMEM saves another $2BTd$ HBM traffic per layer boundary.

3. **MoE variant:** For models with MoE FFN layers interleaved with linear RNN layers (e.g., Jamba), extend the megakernel to include the MoE dispatch/combine (combining with FlashMoE's persistent kernel approach).

4. **FP8 quantized projections:** Use FP8 precision for the prologue/epilogue GEMMs within the megakernel to double GEMM throughput while keeping the scan in BF16/FP32 for numerical stability.

5. **Compiler-generated kernel:** Use Mirage MPK or Tawa compiler to automatically generate the megakernel from a high-level specification, reducing implementation effort and enabling exploration of different fusion strategies.

## Human Review

(To be filled by reviewer)

---
status: completed
priority: high
created: 2026-02-15
based_on: warp-specialized-pipelining (141), tfla-two-level-tiled-chunkwise-parallelism (158), io-aware-tiling (066), epilogue-visitor-tree-fusion (039), twill-joint-swp-warp-specialization (135), batch-reduce-gemm
experiment_number: 039
experiment_log: experiment-log-039.md
results_file: 039_results.md
---

# Warp-Specialized Pingpong Pipelining for Chunkwise Linear RNN Kernels

## Hypothesis

Applying FlashAttention-3's **warp-specialized producer-consumer pipelining** and **pingpong scheduling** to TFLA-style chunkwise linear RNN kernels will yield $1.5$–$2\times$ wall-clock speedup over current TFLA kernels by overlapping the inter-chunk state scan (sequential, memory-bound) with intra-chunk matmuls (compute-bound) on separate warp groups — exploiting that WGMMA and TMA run on physically separate hardware units on Hopper GPUs.

## Background

**Current bottleneck in linear RNN training kernels:** TFLA (Tiled Flash Linear Attention) achieves state-of-the-art throughput for linear RNNs (mLSTM, GLA, DeltaNet) by using two-level tiling: an outer chunk size $L$ (large) and inner tile sizes $B_{Lhq}, B_{Lkv}$ (fit SRAM). However, the current TFLA implementation uses a **single-phase pipeline**: load tiles → compute intra-chunk matmuls → store intermediate states → compute inter-chunk scan → repeat. This serializes memory operations and compute, leaving significant hardware idle.

**What FlashAttention-3 showed:** On Hopper GPUs, warp specialization splits warps into producers (TMA loads) and consumers (WGMMA compute), with a pingpong scheduler that alternates between two consumer warpgroups. This achieves 75% tensor core utilization (vs 35% for FlashAttention-2) by overlapping the softmax reduction (sequential, element-wise) with the next iteration's GEMM.

**Key observation:** The chunkwise linear RNN computation has an **analogous structure** to FlashAttention's loop:

| FlashAttention | Chunkwise Linear RNN |
|----------------|---------------------|
| $Q K^T$ matmul (GEMM) | $Q_i K_j^T$ intra-chunk matmul (GEMM) |
| Softmax + rescaling (sequential, elem-wise) | Decay masking $\odot D_{ij}$ + state accumulation (sequential, elem-wise) |
| $P V$ matmul (GEMM) | $(Q_i K_j^T \odot D_{ij}) V_j$ (GEMM) |
| Online max/sum across tiles | Inter-chunk state update $h_{i+1} = \bar{A} h_i + \ldots$ |

In both cases, a sequential element-wise operation (softmax / decay mask + state scan) is **sandwiched between two GEMMs**. FlashAttention-3's pingpong scheduler hides ~50% of the softmax cost by running it concurrently with the next tile's GEMM. The same strategy should apply to the decay mask and state accumulation in linear RNNs.

**Gap filled:** Existing Proposal 038 (CTA-swizzled TFLA) optimizes L2 cache reuse for TFLA but does not address the intra-SM pipeline efficiency. This proposal targets the complementary optimization: maximizing tensor core utilization within each SM via warp specialization. The two optimizations are orthogonal and composable.

## Related Work

- **TFLA (Beck et al., 2025)**: Introduced two-level tiled chunkwise parallelism for mLSTM/GLA kernels. Achieves 2× speedup over Mamba-2 and beats FlashAttention-3 on long sequences. Uses Triton; does not use warp specialization or TMA/WGMMA. Our approach extends TFLA with Hopper-specific optimizations.
- **FlashAttention-3 (Shah et al., 2024)**: Demonstrated warp specialization + pingpong for softmax attention on Hopper, achieving 75% utilization. Applied to standard attention only, not linear attention or SSM kernels. Our approach adapts the same pipelining strategy to the different computation graph of chunkwise linear RNNs.
- **Twill (Huang et al., 2025)**: Compiler that automatically discovers optimal SWP + warp specialization schedules. Could potentially auto-discover the schedule we propose, but hasn't been applied to chunkwise linear RNN kernels. Our approach uses Twill's formulation to find the optimal schedule for the specific linear RNN computation DAG.
- **Tawa (Bao et al., 2025)**: Automatic warp specialization compiler. Handles general tensor programs but hasn't been evaluated on linear RNN kernels.
- **Mirage MPK (Jia et al., 2025)**: Megakernel compiler for inference. Focuses on multi-kernel fusion across layers, not intra-kernel pipeline optimization within a single operator.

**Our approach differs**: No existing work applies warp-specialized pingpong pipelining specifically to chunkwise linear RNN/SSM kernels. The computation graph differs from attention (decay masks vs softmax, state scan vs online max), requiring a different pipeline schedule.

## Mathematical Formulation

**Standard Chunkwise Linear RNN (TFLA inner loop):**

For chunk $i$, inner tile $j$, the core computation is:

$$
O_i += (Q_i K_j^T \odot D_{ij}) V_j
$$

where:
- $Q_i \in \mathbb{R}^{B_{Lhq} \times d_q}$ — query tile
- $K_j \in \mathbb{R}^{B_{Lkv} \times d_q}$ — key tile
- $V_j \in \mathbb{R}^{B_{Lkv} \times d_v}$ — value tile
- $D_{ij} \in \mathbb{R}^{B_{Lhq} \times B_{Lkv}}$ — causal decay mask ($D_{ij}[a,b] = \prod_{t=a}^{b-1} \gamma_t$ for GLA)

**Inter-chunk state update:**

$$
h_{i+1} = \bar{A}_i h_i + \sum_j (K_j^T \odot \tilde{D}_j) V_j
$$

where $\bar{A}_i = \prod_{t \in \text{chunk}_i} \gamma_t$ is the chunk-level decay and $\tilde{D}_j$ encodes intra-chunk decay weights.

**Proposed Pipeline Schedule (Pingpong, 2 consumer warpgroups):**

| Stage | Producer Warp | Consumer WG-0 | Consumer WG-1 |
|-------|:------------:|:-------------:|:-------------:|
| 0 | Load $K_0, V_0$ | — | — |
| 1 | Load $K_1, V_1$ | WGMMA: $S_0 = Q K_0^T$ | — |
| 2 | Load $K_2, V_2$ | Decay: $S_0 \odot D_0$ | WGMMA: $S_1 = Q K_1^T$ |
| 3 | Load $K_3, V_3$ | WGMMA: $O += S_0 V_0$ | Decay: $S_1 \odot D_1$ |
| 4 | Load $K_4, V_4$ | Decay+scan: state update | WGMMA: $O += S_1 V_1$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |

**Key insight:** The decay mask computation ($S \odot D$) and inter-chunk state scan are element-wise operations that run on the CUDA cores, while WGMMA runs on the tensor cores. These are **physically separate hardware units** — they can execute simultaneously on the same SM.

**Overlap fraction:** Let $T_{\text{gemm}}$ be the time for one WGMMA call (e.g., $B_{Lhq} \times d_q \times B_{Lkv}$ matmul) and $T_{\text{decay}}$ be the time for the decay mask + accumulation. The speedup from pingpong is:

$$
\text{Speedup} = \frac{2 T_{\text{gemm}} + T_{\text{decay}}}{2 T_{\text{gemm}} + \max(0, T_{\text{decay}} - T_{\text{gemm}})}
$$

When $T_{\text{decay}} \leq T_{\text{gemm}}$ (decay fully hidden): speedup $= \frac{2 T_{\text{gemm}} + T_{\text{decay}}}{2 T_{\text{gemm}}} \approx 1 + \frac{T_{\text{decay}}}{2 T_{\text{gemm}}}$.

For typical shapes ($B_{Lhq} = B_{Lkv} = 64$, $d_q = 128$): $T_{\text{gemm}} \gg T_{\text{decay}}$ and the decay is fully hidden, yielding speedup from better pipeline occupancy.

**Key Variables:**
- $B_{Lhq}, B_{Lkv}$ — inner tile sizes (typically 64)
- $d_q, d_v$ — head dimensions (typically 128)
- $L$ — outer chunk size (256–1024)
- $\gamma_t \in (0, 1)$ — per-token decay (input-dependent for GLA/Mamba-2)
- $s$ — number of pipeline stages in SMEM circular buffer (typically 2–3)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Kernel type | Chunkwise linear RNN (GLA/mLSTM/DeltaNet) |
| GPU | H100 (Hopper, SM90a) |
| Inner tiles | $B_{Lhq} = B_{Lkv} = 64$ |
| Head dim | $d_q = d_v = 128$ |
| Chunk size | $L = 512$ |
| Pipeline stages | $s = 2$ (SMEM double-buffering) |
| Warp groups | 1 producer + 2 consumers (pingpong) |
| Precision | BF16 compute, FP32 accumulators |

### Baseline
1. **TFLA (Triton, current SOTA)**: $O(T L d^2 / L)$ — two-level tiled chunkwise kernel without warp specialization. From Beck et al., 2025.
2. **FLA v1 (Triton)**: Standard Flash Linear Attention with small chunks. From Yang et al., 2024.
3. **Mamba-2 kernel (CUDA)**: Custom selective scan kernel. From Dao & Gu, 2024.
4. **FlashAttention-3 (CUTLASS)**: Warp-specialized softmax attention — ceiling for tensor core utilization on Hopper.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $> 1.5\times$ TFLA tokens/sec | Training forward+backward on H100 |
| Tensor core utilization | $> 60\%$ | `ncu` profiling (SM active cycles) |
| TFLOPS/s | $> 500$ (BF16) | Measured via `ncu` |
| Memory | $\leq$ TFLA | Peak GPU memory (same model) |
| Quality | $=$ TFLA (bit-exact) | Validate numerics match |

### Estimated Compute

**MVE**: < 1 hour implementation + profiling on single H100
**Phase 1** (kernel microbenchmark): ~5 GPU-hours on H100
**Phase 2** (end-to-end pretraining, 350M params): ~50 GPU-hours on H100
**Total**: ~55 GPU-hours (small-medium scale)

## Expected Outcome

**If hypothesis is correct:**
- $1.5$–$2\times$ speedup over TFLA on H100 for GLA/mLSTM forward pass
- Tensor core utilization rises from ~40% (TFLA, Triton) to ~60%+ (warp-specialized CUTLASS)
- The decay mask and state scan are fully hidden behind WGMMA for typical head dimensions ($d \geq 64$)
- Backward pass benefits similarly (the backward chunkwise computation has the same GEMM-elementwise-GEMM sandwich structure)
- End-to-end pretraining sees $1.2$–$1.5\times$ wall-clock speedup (attention-equivalent layers dominate training time)

**If hypothesis is wrong:**
- If decay/state operations aren't fully hidden: the sequential state scan across chunks is too long to hide. This would point to needing larger matmul tiles to increase $T_{\text{gemm}}$ relative to $T_{\text{scan}}$ — i.e., the optimal tile sizes for warp-specialized linear RNNs differ from attention
- If register pressure is too high: two consumer warpgroups + producer + decay state requires too many registers, killing occupancy. This would motivate a 1-consumer variant with partial overlap
- If Triton can match: Triton's auto-tuner on Hopper already achieves good pipelining without explicit warp specialization. This would validate the compiler approach (Twill/Tawa) over manual kernel engineering

## Minimum Viable Experiment

### Setup
- **Kernel**: Single chunkwise GLA forward pass kernel in CUTLASS 3.x with warp specialization
- **Shapes**: $B = 4$, $T = 4096$, $H = 32$, $d = 128$ (typical 350M model config)
- **Baseline**: TFLA Triton kernel (same shapes)
- **Compute**: Single H100, < 30 minutes

### Success Criteria
- Warp-specialized kernel achieves $> 1.3\times$ throughput (tokens/sec) over TFLA Triton baseline
- `ncu` shows $> 55\%$ tensor core utilization (vs ~40% for TFLA)
- Numerical output matches TFLA within BF16 tolerance ($\|O_{\text{ours}} - O_{\text{tfla}}\|_\infty < 10^{-2}$)

### Failure Criteria
- If throughput gain is $< 1.1\times$: warp specialization doesn't help for this computation graph (the bottleneck is elsewhere — e.g., SMEM bank conflicts, register pressure)
- If SM occupancy drops below 25%: register pressure from two consumer warpgroups is prohibitive

### Why This Test Is Sufficient
- The forward pass kernel contains all the key operations (GEMM, decay mask, state scan, TMA loads) — if overlap works here, backward pass is structurally identical
- The shapes are representative of real pretraining workloads
- Throughput at kernel level directly translates to end-to-end training speedup (this kernel is the bottleneck layer)

## Memory Access Pattern Analysis

**Coalesced access:** TMA handles all global→SMEM transfers with hardware-managed coalescing. WGMMA reads from SMEM with predefined layouts (row-major or swizzled). No uncoalesced access.

**Cache-friendly:** Inner tiles ($64 \times 128$) fit in SMEM (64×128×2 bytes = 16 KB per tile; 3–4 tiles simultaneously = 48–64 KB, well within H100's 228 KB SMEM per SM).

**Arithmetic intensity:** For each $B \times d \times B$ GEMM tile: $B^2 d$ FLOPs / $(Bd + Bd) \times 2$ bytes = $Bd/(4)$ ≈ $64 \times 128 / 4 = 2048$ FLOPs/byte. This is highly compute-bound — ideal for tensor core saturation.

**HBM bandwidth:** The two-level tiling means each $K, V$ tile is loaded once from HBM per outer chunk. Total HBM traffic: $O(T \cdot d \cdot L / B_{Lkv})$ — same as TFLA, no regression.

## Parallelism Analysis

**SM saturation:** Each CTA processes one $(B_{Lhq}, d_v)$ output tile. With $T/L$ chunks × $H$ heads × $B/L$ sequence tiles, there are typically $>100$ CTAs — sufficient to saturate H100's 132 SMs.

**No warp divergence:** All warps within a warpgroup execute the same WGMMA instruction. Producer warps execute TMA instructions (uniform).

**Tensor core mapping:** All dominant operations are matmuls: $QK^T$ (GEMM), $SV$ (GEMM), $K^T V$ (GEMM for state). Decay masking is element-wise (CUDA cores). State scan at chunk boundaries is a small matmul ($d \times d$) — fits a single WGMMA.

**Sequential bottleneck:** The inter-chunk state propagation $h_{i+1} = \bar{A} h_i + \ldots$ is sequential across chunks. However, this is a $d \times d$ matmul per chunk boundary — negligible compared to the $O(L)$ intra-chunk work. TFLA already handles this; our optimization targets the intra-chunk pipeline.

## Theoretical Analysis

Complexity comparison (per layer, sequence length $T$):

| Operation | TFLA (current) | Warp-Specialized TFLA |
|-----------|---------------|----------------------|
| FLOPs | $O(T d^2)$ | $O(T d^2)$ — same |
| HBM reads | $O(T d)$ | $O(T d)$ — same |
| Wall-clock | $T_{\text{gemm}} + T_{\text{decay}} + T_{\text{load}}$ | $\max(T_{\text{gemm}}, T_{\text{decay}}) + T_{\text{overlap\_loss}}$ |
| TC utilization | ~40% | ~60–75% |
| Kernel launches | 1 | 1 |

The theoretical speedup is entirely from **pipeline utilization**: doing the same work with less idle time on the same hardware.

## Risks & Limitations

1. **H100-only:** Warp specialization with TMA + WGMMA requires Hopper (SM90a). Not available on A100 or consumer GPUs. Mitigation: the kernel also runs (without speedup) on A100 via fallback path.

2. **Implementation complexity:** Writing warp-specialized CUTLASS kernels requires PTX-level async instructions, barrier management, and careful SMEM layout. Mitigation: use Twill compiler to auto-derive the schedule, or start from FlashAttention-3's kernel as a template.

3. **Register pressure:** Two consumer warpgroups each maintaining accumulator state ($B_{Lhq} \times d_v$ in FP32 = 64×128×4 = 32 KB per warpgroup) may exceed register budget, reducing occupancy. Mitigation: reduce tile sizes or use 1 consumer warpgroup with simpler overlap.

4. **Decay mask variation:** Different linear RNNs have different decay structures (GLA: scalar per head; DeltaNet: outer product update; mLSTM: matrix decay). The pipeline schedule may need per-architecture tuning. Mitigation: start with GLA (simplest scalar decay).

5. **Diminishing returns at small $d$:** If $d$ is very small (e.g., 32), the GEMM tiles are small and WGMMA underutilizes tensor cores regardless of scheduling. Speedup is most significant for $d \geq 64$.

## Follow-up Experiments

1. **Twill auto-derivation:** Use the Twill compiler to automatically discover the optimal SWP + WS schedule for the chunkwise linear RNN computation graph. Compare against our hand-designed schedule to validate (or improve upon) it.

2. **FP8 variant:** Hopper supports FP8 WGMMA with 2× higher throughput. If the linear RNN computation is FP8-compatible (requires careful scaling of decay factors), this could yield an additional 2× on top of the pipeline gains.

3. **Backward pass pipelining:** Apply the same pingpong schedule to the backward chunkwise computation, which has an identical GEMM-elementwise-GEMM sandwich structure.

4. **Composing with CTA swizzling (Proposal 038):** Apply CTA tile swizzling for L2 cache optimization on top of intra-SM warp specialization. The two optimizations are orthogonal — CTA swizzling improves inter-SM data reuse while warp specialization improves intra-SM pipeline utilization.

5. **Blackwell (SM100) adaptation:** NVIDIA Blackwell introduces new async primitives. Adapt the pipeline schedule for the next GPU generation.

## Human Review

(To be filled by reviewer)

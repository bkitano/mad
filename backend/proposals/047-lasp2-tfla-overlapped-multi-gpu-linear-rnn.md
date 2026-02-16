---
status: ongoing
priority: high
created: 2026-02-15
based_on: lasp2-allgather-sequence-parallelism (176), tfla-two-level-tiled-chunkwise-parallelism (158), gla-secondary-chunking-log-space-gating (177), warp-specialized-pipelining (141), flux-communication-computation-overlap-fusion (049), io-aware-tiling (066)
experiment_number: 047
experiment_log: experiment-log-047.md
---

# LASP-2 AllGather-Overlapped TFLA for Multi-GPU Linear RNN Pretraining

## Hypothesis

Co-designing **LASP-2's AllGather-based sequence parallelism** with **TFLA's two-level tiled intra-chunk kernels** into a single communication-overlapped pipeline will achieve **$1.15$–$1.30\times$ end-to-end training throughput** over naively composing LASP-2 + TFLA sequentially on $W \geq 8$ GPUs at sequence lengths $T \geq 32\text{K}$. The key insight: LASP-2's AllGather communication volume is tiny ($O(d^2)$ per GPU, independent of sequence length) and can be **fully hidden** behind TFLA's compute-dominant intra-chunk matmuls ($\geq 87\%$ tensor core utilization) using Hopper TMA async loads and warp-specialized producer-consumer scheduling, eliminating the AllGather latency from the critical path entirely.

## Background

### The two-level parallelism gap in linear RNN pretraining

Linear RNN pretraining at scale requires parallelism at two levels:

1. **Intra-GPU (kernel level)**: TFLA (Beck et al., NeurIPS 2025) provides state-of-the-art single-GPU kernel efficiency by adding a second level of tiling within each chunk, achieving $> 87\%$ tensor core utilization and outperforming FlashAttention-3 for $T \geq 4096$.

2. **Inter-GPU (distributed level)**: LASP-2 (Sun et al., 2025) provides optimal sequence parallelism for linear attention by replacing ring P2P with a single AllGather of the $d \times d$ memory state per GPU, achieving 36.6% speedup over Ring Attention.

**Current practice**: These two levels are composed **naively** — LASP-2 handles the distributed communication, then TFLA handles the local computation, with synchronization barriers between them. This leaves performance on the table:

| Phase | Duration | Hardware | Bottleneck |
|-------|----------|----------|------------|
| TFLA intra-chunk compute | $\sim 60\%$ of wall-clock | Tensor cores (WGMMA) | Compute-bound ✓ |
| TFLA inter-chunk scan | $\sim 10\%$ of wall-clock | Scalar ALU / CUB scan | Memory-bound |
| LASP-2 AllGather | $\sim 8\%$ of wall-clock | NVLink/IB | Communication-bound |
| LASP-2 prefix sum | $\sim 2\%$ of wall-clock | Scalar ALU | Sequential |
| Synchronization barriers | $\sim 5$–$15\%$ of wall-clock | Idle | **Wasted** |

The synchronization barriers between communication and computation waste $5$–$15\%$ of wall-clock time. Furthermore, the inter-chunk scan and AllGather are both memory/communication-bound and could overlap with compute-bound operations.

### The overlap opportunity

LASP-2's workflow for each layer is:

1. Compute local intra-chunk attention: $O_j^{\text{intra}} = (Q_j K_j^\top \odot M_j) V_j$ — this is TFLA's domain
2. Compute local memory states: $M_j = K_j^\top V_j$ — small matmul ($d \times C \times d$)
3. **AllGather** memory states across GPUs: $M_{\text{total}} = \text{AllGather}(\{M_w\}_{w=1}^W)$ — volume $W \times B \times H \times d^2$ floats
4. Compute cross-chunk prefix sum: $M_{1:j} = \sum_{i \leq j} G_{j:i} M_i$ — sequential over chunks
5. Compute inter-chunk output: $O_j^{\text{inter}} = Q_j \cdot M_{1:j}$ — matmul ($C \times d \times d$)

**The key observation**: Steps 1 and 2 are independent of step 3. TFLA's intra-chunk computation (step 1) is the dominant cost and is purely compute-bound. LASP-2's AllGather (step 3) is purely communication-bound. **These can overlap perfectly on Hopper GPUs** using warp specialization:

- **Producer warp group**: Issues TMA-based AllGather of $M_w$ from other GPUs into shared memory, then signals consumers
- **Consumer warp group**: Runs TFLA's intra-chunk matmuls on tensor cores (WGMMA)
- **Overlap**: Producer loads $M_{\text{remote}}$ while consumer computes $O^{\text{intra}}$ — zero idle time

Furthermore, step 4 (prefix sum) can overlap with the **tail end** of step 1 (later chunks' intra-chunk compute), and step 5 can be fused into TFLA's epilogue (adding $O^{\text{inter}}$ to $O^{\text{intra}}$ via an epilogue visitor tree, trick 039).

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster?** Yes — the AllGather volume is tiny: for $d = 128$, $H = 8$, $B = 1$: $8 \times 128^2 \times 2 = 256$ KB per GPU in bf16. On NVLink (900 GB/s bidirectional on H100), this takes $\sim 0.3 \mu s$. TFLA's intra-chunk compute takes $> 100 \mu s$ per chunk. The communication is $< 0.3\%$ of compute time and can be fully hidden.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — take TFLA's existing Triton/CUDA kernel, add a warp group that issues `cp.async.bulk` (TMA) to AllGather $M_w$ from peer GPUs into shared memory while the compute warp group runs WGMMA. The fusion is identical to FlashAttention-3's pingpong scheduling (trick 141) but with NVLink communication replacing DRAM loads.

3. **Does it reduce HBM bandwidth?** Yes — by fusing the AllGather result ($M_{\text{total}}$) directly into the epilogue, we avoid writing $M_{\text{total}}$ to HBM and reading it back. This saves $2 \times W \times d^2$ HBM bytes per layer.

### Memory access pattern analysis

**AllGather (producer warp group):**
- TMA-based `cp.async.bulk.tensor` from peer GPU GMEM → local SMEM
- Coalesced: TMA issues contiguous 128-byte transactions
- Non-blocking: producer signals consumer via `mbarrier` when data arrives
- Volume: $d^2 \times 2 = 32$ KB per head in bf16 ($d = 128$) — fits in one SMEM bank

**TFLA intra-chunk (consumer warp group):**
- Standard WGMMA: loads Q/K/V tiles via TMA from GMEM, computes matmuls in registers
- Arithmetic intensity: $\geq 128$ FLOPs/byte (well above compute-bound threshold)
- No dependency on AllGather result until epilogue phase

**Epilogue fusion:**
- After intra-chunk output $O^{\text{intra}}$ is in registers, load $M_{1:j}$ from SMEM (placed there by producer after prefix sum)
- Compute $O^{\text{inter}}_j = Q_j \cdot M_{1:j}$ via one additional WGMMA
- Add: $O_j = O_j^{\text{intra}} + O_j^{\text{inter}}$ in registers
- Store fused result to GMEM: single coalesced write

### Parallelism analysis

**Warp specialization (H100 Hopper):**
- 4 warp groups per CTA: 1 producer (TMA + NVLink) + 3 consumers (WGMMA)
- Producer/consumer communicate via `mbarrier` in SMEM — zero-overhead synchronization
- Consumer pipeline depth: 2 (pingpong double-buffering) — consumer never stalls on producer

**SM saturation:**
- With $T_{\text{local}} = T/W$ tokens per GPU and $C = 256$: $G_{\text{local}} = T_{\text{local}}/C$ chunks per GPU
- Each chunk occupies $\sim 1$ CTA. For $T = 32\text{K}$, $W = 8$: $G_{\text{local}} = 4096 / 256 = 16$ chunks
- With $H = 32$ heads, $B = 4$ batch: $16 \times 32 \times 4 = 2048$ CTAs — saturates 132 SMs on H100

**Load balance:**
- All GPUs have exactly $T_{\text{local}} = T/W$ tokens (LASP-2's assumption)
- Within each GPU, TFLA tiles balance work across sub-chunks

## Related Work

- **LASP-2** (Sun et al., arXiv:2502.07563, Feb 2025): AllGather-based sequence parallelism for linear attention. Mentions TFLA compatibility ("intra-chunk quadratic cost can use TFLA") but **does NOT** implement the overlapped co-design. Uses sequential composition with synchronization barriers.
- **TFLA** (Beck et al., arXiv:2503.14376, NeurIPS 2025): Two-level tiled kernel for linear RNNs on single GPU. **No distributed training support** — no sequence parallelism integration.
- **FlashAttention-3** (Shah et al., 2024): Introduced warp-specialized pingpong scheduling for overlapping WGMMA with TMA loads. We adopt this pattern but replace DRAM↔SRAM pipelining with NVLink↔SRAM pipelining for inter-GPU communication.
- **Flux** (Li et al., 2024): Communication-computation overlap for GEMM + AllReduce in tensor parallelism. Overlaps AllReduce with the next layer's GEMM. We adapt this philosophy to sequence parallelism, overlapping AllGather with the **same layer's** intra-chunk compute.
- **ZeCO** (arXiv:2507.01004, Jul 2025): Zero-communication overhead for linear attention via analytical redundant computation. Different approach — eliminates communication by recomputing, which increases FLOPs. Our approach keeps FLOPs constant and hides communication latency.

**Gap**: No existing work co-designs the kernel-level tiling (TFLA) with distributed sequence parallelism (LASP-2) into a single overlapped pipeline. All existing implementations compose them sequentially with synchronization barriers.

## Mathematical Formulation

### LASP-2 + TFLA Sequential Composition (Baseline)

For GPU $w$ processing local sequence $[x_{wG_l}^{(w)}, \ldots, x_{(w+1)G_l - 1}^{(w)}]$ split into $G_l = T_{\text{local}} / C$ chunks:

**Phase 1 — Local TFLA compute (independent per GPU):**

$$
O_j^{\text{intra},(w)} = \text{TFLA}(Q_j^{(w)}, K_j^{(w)}, V_j^{(w)}, M_j^{(w)}), \quad j = 1, \ldots, G_l
$$

$$
M_j^{(w)} = (K_j^{(w)})^\top V_j^{(w)} \in \mathbb{R}^{d \times d}
$$

**Phase 2 — AllGather + prefix sum (synchronous):**

$$
\bar{M}^{(w)} = \sum_{j=1}^{G_l} \gamma_j^{(w)} M_j^{(w)} \in \mathbb{R}^{d \times d}
$$

$$
\{\bar{M}^{(1)}, \ldots, \bar{M}^{(W)}\} = \text{AllGather}(\bar{M}^{(w)})
$$

$$
\bar{M}_{1:w} = \sum_{i=1}^{w} \prod_{l=i+1}^{w} G_l \cdot \bar{M}^{(i)} \quad \text{(causal prefix sum)}
$$

**Phase 3 — Inter-chunk output (depends on Phase 2):**

$$
O_j^{\text{inter},(w)} = Q_j^{(w)} \cdot \bar{M}_{1:w,j}, \quad O_j^{(w)} = O_j^{\text{intra},(w)} + O_j^{\text{inter},(w)}
$$

**Critical path** = Phase 1 + Phase 2 + Phase 3 (sequential).

### LASP-2 + TFLA Overlapped Pipeline (Proposed)

**Fused kernel — single persistent megakernel per layer:**

```
For each CTA (handles one chunk j, one head h):
  Producer warp group (async, non-blocking):
    1. Compute local M_j = K_j^T @ V_j (small GEMM)
    2. Issue AllGather of M_bar via cp.async.bulk to peer GPUs
    3. Wait for AllGather completion (mbarrier)
    4. Compute prefix sum M_{1:w,j} in SMEM
    5. Signal consumer: M_{1:w,j} ready in SMEM

  Consumer warp groups (compute-bound, runs concurrently):
    1. Run TFLA intra-chunk: O_j^intra = TFLA(Q_j, K_j, V_j)
    2. Wait for producer signal (mbarrier)
    3. Compute O_j^inter = Q_j @ M_{1:w,j} (one WGMMA, SMEM operand)
    4. Fuse: O_j = O_j^intra + O_j^inter (register add)
    5. Store O_j to GMEM
```

**Critical path** = max(TFLA compute, AllGather + prefix sum) ≈ **TFLA compute alone** (since AllGather is hidden).

### Timing Analysis

For $d = 128$, $C = 256$, $H = 32$, $W = 8$ GPUs, $T = 32\text{K}$:

| Operation | Sequential (baseline) | Overlapped (proposed) |
|-----------|----------------------|----------------------|
| TFLA intra-chunk per chunk | $\sim 120$ µs | $\sim 120$ µs |
| AllGather ($d^2$ per head) | $\sim 0.5$ µs | **Hidden** ($0$ µs) |
| Prefix sum ($W$ steps) | $\sim 0.3$ µs | **Overlapped** |
| Inter-chunk WGMMA | $\sim 5$ µs | Fused into epilogue |
| Sync barriers | $\sim 15$ µs | **Eliminated** |
| **Total per chunk** | $\sim 141$ µs | $\sim 125$ µs |
| **Speedup** | — | **$1.13\times$** |

At larger $W$ (more GPUs), the AllGather latency grows as $O(\log W)$ in tree-reduce mode, making the overlap benefit larger.

### Key Variables

- $W$ — number of GPUs (sequence parallelism degree)
- $T$ — total sequence length
- $T_{\text{local}} = T / W$ — per-GPU sequence length
- $C$ — chunk size (256–1024 with TFLA)
- $G_l = T_{\text{local}} / C$ — chunks per GPU
- $d$ — head dimension (64–128)
- $H$ — number of heads
- $B$ — batch size
- $\bar{M}^{(w)} \in \mathbb{R}^{d \times d}$ — per-GPU aggregated memory state

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / mLSTM / Gated DeltaNet |
| Layers | $L = 24$ |
| Hidden dim | $d_{\text{model}} = 2048$ |
| Head dim | $d = 128$ |
| Heads | $H = 16$ |
| Chunk size | $C = 256$ (TFLA) |
| Sequence length | $T = 32\text{K}$–$131\text{K}$ |
| GPUs | $W = 8$–$64$ (H100 NVLink) |
| Distributed strategy | LASP-2 (sequence) + FSDP (data/weight) |

### Baseline

1. **LASP-2 + FLA (C=64)**: Standard LASP-2 with Flash Linear Attention kernel, $C = 64$
2. **LASP-2 + TFLA (sequential)**: LASP-2 with TFLA kernel, but with synchronization barriers (naive composition)
3. **Ring Attention + TFLA**: Ring-style P2P sequence parallelism with TFLA kernel
4. **LASP-2 + FlashAttention-3 (softmax baseline)**: For hybrid models, compare linear RNN layers vs. softmax layers

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $\geq 1.15\times$ LASP-2+TFLA sequential | Tokens/sec, 8×H100 |
| Communication overhead | $< 3\%$ of wall-clock | NCCL profiling |
| Scaling efficiency | $\geq 85\%$ weak scaling, 8→64 GPUs | Throughput ratio |
| Quality | $= $ LASP-2+TFLA sequential | Perplexity (must be bit-identical) |
| Peak memory | $\leq $ LASP-2+TFLA sequential | GB per GPU |

### Estimated Compute

**MVE (kernel benchmark)**: $\sim 30$ minutes on 8×H100 ($\sim \$30$) — benchmark overlapped vs. sequential
**Small-scale (370M model)**: $\sim 64$ GPU-hours on 8×H100 ($\sim \$250$) — pretraining validation
**Full-scale (1.3B model)**: $\sim 512$ GPU-hours on 64×H100 ($\sim \$2000$) — scaling study

## Expected Outcome

**If hypothesis is correct:**

- Overlapped pipeline achieves $1.15$–$1.30\times$ throughput over sequential composition at $W = 8$–$64$ GPUs
- AllGather latency is completely hidden: communication overhead $< 1\%$ of wall-clock
- Weak scaling efficiency $> 90\%$ from 8 to 64 GPUs (vs. $\sim 85\%$ for sequential LASP-2)
- The speedup increases with $W$ (more GPUs = more AllGather latency to hide)
- Quality is bit-identical to sequential composition (same computation, different scheduling)

**If hypothesis is wrong:**

- **Scenario A**: Warp specialization overhead is too high — the producer warp group steals SM resources from the consumer, reducing TFLA's tensor core utilization below the threshold where overlap is beneficial. **What we learn**: The optimal producer:consumer ratio may need $> 4$ warp groups (requiring larger CTA sizes). **Mitigation**: Use 8 warp groups (1 producer + 7 consumers).
- **Scenario B**: The AllGather is already negligible — at $W = 8$ with NVLink, the AllGather takes $< 1$ µs and the sync barrier is only $\sim 2$ µs, so the overlap saves $< 2\%$. **What we learn**: Overlap matters more for $W \geq 32$ or InfiniBand (higher latency). **Mitigation**: Test at $W = 32$–$64$ where AllGather takes $10$–$100$ µs.
- **Scenario C**: The prefix sum (step 4) is the bottleneck, not the AllGather — the $O(W)$ sequential prefix sum over memory states cannot be overlapped because each step depends on the previous GPU's result. **What we learn**: Need to apply MatMulScan (proposal 044) to the inter-GPU prefix sum to parallelize it. **Mitigation**: Replace the sequential prefix sum with a parallel tree-reduce prefix scan.

## Minimum Viable Experiment

### Setup

- **Task**: Kernel microbenchmark — measure wall-clock for overlapped vs. sequential LASP-2 + TFLA
- **Hardware**: 8×H100 NVLink (single node)
- **Model**: No model needed — synthetic Q, K, V tensors
- **Configuration**: $d = 128$, $H = 16$, $C = 256$, $B = 4$, $T = 32\text{K}$ ($T_{\text{local}} = 4\text{K}$ per GPU)
- **Compute**: $< 10$ minutes (kernel compilation + benchmarking)

### Implementation Steps

1. **Baseline** (5 min): Run LASP-2 + TFLA sequentially using existing implementations from `flash-linear-attention` library + `lasp` library. Measure wall-clock per layer.

2. **Overlapped prototype** (5 min): Implement a Triton kernel that:
   - Launches TFLA intra-chunk compute on consumer warps
   - Simultaneously issues `torch.distributed.all_gather` on a separate CUDA stream
   - Uses `torch.cuda.Event` to synchronize (not warp-specialized yet, but stream-level overlap)
   - Fuses inter-chunk output into TFLA's epilogue

   This is not the full warp-specialized pipeline but tests whether stream-level overlap already gives benefit.

### Success Criteria

- Stream-level overlapped version achieves $\geq 1.05\times$ throughput over sequential at $W = 8$
- AllGather does NOT appear on the critical path in NVIDIA Nsight Systems trace
- Total per-layer time is within $5\%$ of TFLA-only time (communication is fully hidden)

### Failure Criteria

- **Kill if**: Overlapped version is $\geq 0.95\times$ slower — the CUDA stream synchronization overhead or memory contention makes overlap counterproductive
- **Kill if**: AllGather time at $W = 8$ is $< 0.5\%$ of TFLA time — overlap isn't worth the engineering effort at this scale (revisit at $W \geq 32$)

### Why This Test Is Sufficient

- The microbenchmark directly tests the overlap hypothesis without model training
- Stream-level overlap is a lower bound on warp-specialized overlap — if stream overlap helps, warp specialization will help more
- $W = 8$ on NVLink is the easiest case for communication; if overlap helps here, it helps more on multi-node InfiniBand
- The timing breakdown identifies exactly which phase is on the critical path

## Theoretical Analysis

### Communication-Computation Overlap Ratio

Define the **overlap ratio** $\rho$:

$$
\rho = \frac{T_{\text{comm}}}{T_{\text{compute}}} = \frac{T_{\text{AllGather}} + T_{\text{prefix\_sum}}}{T_{\text{TFLA\_intra}}}
$$

For the overlap to be fully beneficial, we need $\rho < 1$ (communication fits inside compute).

**AllGather time** (bandwidth-limited):

$$
T_{\text{AllGather}} = \frac{(W-1) \cdot B \cdot H \cdot d^2 \cdot 2}{\text{BW}_{\text{NVLink}}} \quad \text{(ring AllGather, bf16)}
$$

For $W = 8$, $B = 4$, $H = 16$, $d = 128$: volume $= 7 \times 4 \times 16 \times 128^2 \times 2 = 14.7$ MB. At 900 GB/s: $T_{\text{AllGather}} = 16$ µs.

**TFLA intra-chunk time** (compute-limited):

$$
T_{\text{TFLA}} = \frac{2 \cdot C^2 \cdot d \cdot G_l \cdot H \cdot B}{\text{TFLOPS}_{\text{TC}}} = \frac{2 \times 256^2 \times 128 \times 16 \times 16 \times 4}{989 \times 10^{12}}
$$

$= \frac{5.5 \times 10^{11}}{989 \times 10^{12}} = 0.56$ ms $= 560$ µs.

**Overlap ratio**: $\rho = 16 / 560 = 0.029$ — communication is only $2.9\%$ of compute. **Full overlap is trivially achievable.**

### Scaling Analysis

| $W$ (GPUs) | $T_{\text{AllGather}}$ (µs) | $\rho$ | Overlap benefit |
|---|---|---|---|
| 8 | 16 | 0.029 | $2.9\%$ saved |
| 16 | 35 | 0.063 | $6.3\%$ saved |
| 32 | 80 | 0.14 | $14\%$ saved |
| 64 | 180 | 0.32 | $32\%$ saved |
| 128 (multi-node IB) | 500 | 0.89 | $89\%$ saved |

**The benefit scales with $W$**: at 128 GPUs on InfiniBand, the AllGather becomes a significant fraction of compute time, and overlapping it saves nearly the entire communication cost.

### Complexity Comparison

| Operation | Sequential | Overlapped |
|-----------|-----------|-----------|
| Critical path | $T_{\text{TFLA}} + T_{\text{AllGather}} + T_{\text{prefix}} + T_{\text{inter}}$ | $\max(T_{\text{TFLA}}, T_{\text{AllGather}} + T_{\text{prefix}}) + T_{\text{inter\_fused}}$ |
| HBM writes | $O(T_{\text{local}} \cdot d + W \cdot d^2)$ | $O(T_{\text{local}} \cdot d)$ (fused) |
| Kernel launches | 3 (TFLA + AllGather + inter) | 1 (fused megakernel) |
| Sync barriers | 2 per layer | 0 per layer |

## Risks & Limitations

1. **Warp specialization complexity**: The full overlapped pipeline requires Hopper-specific features (TMA, warp groups, `mbarrier`, WGMMA). Not available on A100 or older GPUs. **Mitigation**: Use CUDA stream overlap as a simpler alternative for Ampere, which gives partial overlap benefit without warp specialization.

2. **NVLink topology sensitivity**: The AllGather pattern depends on the GPU interconnect topology. On multi-node InfiniBand, latency is higher but so is the overlap benefit. On PCIe-connected GPUs, the bandwidth may be too low for the AllGather to complete within the TFLA compute window. **Mitigation**: Profile on target hardware; adjust $C$ (larger chunk = more compute time = more overlap window).

3. **SMEM pressure**: The producer warp group needs SMEM for the AllGather buffer ($W \times d^2 \times 2 = 256$ KB for $W = 8$, $d = 128$). On H100 with 256 KB SMEM, this leaves little room for TFLA's Q/K/V tiles. **Mitigation**: Double-buffer the AllGather: receive into one buffer while computing with the other. Or reduce per-CTA AllGather to single-head ($d^2 \times 2 = 32$ KB) and iterate over heads.

4. **Prefix sum serialization**: The inter-GPU prefix sum over memory states is sequential ($O(W)$ steps). At large $W$, this becomes a bottleneck that cannot be overlapped. **Mitigation**: Apply parallel tree-reduce prefix sum ($O(\log W)$ steps) or MatMulScan (proposal 044) to the inter-GPU scan.

5. **Implementation effort**: Building a fused persistent megakernel with warp specialization is significant engineering. The Triton language does not natively support warp-group-level programming. May need CUTLASS or raw CUDA. **Mitigation**: Start with stream-level overlap in PyTorch (easy), validate the benefit, then invest in the full warp-specialized kernel.

## Follow-up Experiments

1. **Warp-specialized vs. stream-level overlap**: Compare the two overlap strategies to quantify the benefit of warp specialization over simple CUDA stream overlap. Expected: warp specialization gives $1.05$–$1.10\times$ additional benefit over stream overlap due to eliminated launch overhead and finer-grained synchronization.

2. **Combine with Proposal 044 (MatMulScan)**: Replace the sequential inter-GPU prefix sum with MatMulScan to parallelize it. This addresses the $O(W)$ serialization at large $W$.

3. **Hybrid LASP-2H overlap**: Extend the overlap to hybrid linear+softmax models (LASP-2H). Softmax layers use different communication patterns (AllGather of KV pairs, larger volume). The overlap benefit may be smaller but still significant.

4. **Multi-node scaling**: Test on 8-node (64 GPU) InfiniBand clusters where AllGather latency is $> 100$ µs. The overlap benefit should be much larger here.

5. **Adaptive chunk sizing**: Use larger chunks ($C = 512$–$1024$) on fewer GPUs and smaller chunks ($C = 128$–$256$) on more GPUs to balance the TFLA compute window with the AllGather latency.

## Human Review

(To be filled by reviewer)

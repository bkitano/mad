---
status: ongoing
priority: high
created: 2026-02-15
based_on: dhelix-strand-interleaved-microbatch-overlap (187), lasp2-allgather-sequence-parallelism (176), tfla-two-level-tiled-chunkwise-parallelism (158), gla-secondary-chunking-log-space-gating (177), fused-chunkwise-ssd-atomic-state-passing (182), flux-communication-computation-overlap-fusion (049)
experiment_number: 049
experiment_log: experiment-log-049.md
---

# DHelix Strand-Interleaved Distributed Linear RNN Training

## Hypothesis

Applying **DHelix strand interleaving** to distributed linear RNN (GLA/mLSTM/Gated DeltaNet) pretraining with **LASP-2 sequence parallelism** will hide $\geq 75\%$ of inter-GPU communication cost, yielding $1.2$–$1.5\times$ wall-clock training throughput improvement on multi-node clusters. The key insight: linear RNN layers have a fundamentally different **computation/communication timing profile** than Transformer layers — the chunkwise recurrence creates an asymmetric forward/backward compute pattern where (a) the inter-chunk scan is sequential and lightweight, providing natural "bubbles" for communication, and (b) the backward pass requires a reverse-direction scan with different memory access patterns. DHelix's DP-based operator pairing can exploit these structural asymmetries to find higher-quality overlap schedules than generic Transformer-targeted overlap.

## Background

### The communication bottleneck in distributed linear RNN training

Distributed pretraining of linear RNN models at scale (1B+ parameters, 8+ GPUs) requires multiple parallelism dimensions:

1. **Tensor Parallelism (TP)**: Splits $W_Q, W_K, W_V, W_O$ projections across GPUs. Each linear RNN layer requires AllGather (before projection) + ReduceScatter (after output projection) per layer — identical to Transformers.

2. **Sequence Parallelism (SP) via LASP-2**: For linear attention models, LASP-2 splits the sequence across GPUs. Each GPU processes local chunks, then performs AllGather of the $d \times d$ inter-chunk memory states to propagate global context. This AllGather is unique to linear RNNs — Transformers use ring-attention or context parallelism instead.

3. **Pipeline Parallelism (PP)**: Standard 1F1B or interleaved schedule, with Send/Recv between pipeline stages.

**Communication cost breakdown for GLA-1.3B (8×H100, TP=4, SP=2, PP=1):**

| Communication | Volume per layer | Frequency | % of layer time |
|--------------|-----------------|-----------|-----------------|
| TP AllGather ($W$ proj) | $B \times T/\text{SP} \times d$ | 1 per layer | ~12% |
| TP ReduceScatter ($W_O$) | $B \times T/\text{SP} \times d$ | 1 per layer | ~12% |
| LASP-2 AllGather (state) | $H \times d_k \times d_v$ | 1 per layer | ~3% |
| LASP-2 ReduceScatter (grad) | $H \times d_k \times d_v$ | 1 per layer (bwd) | ~3% |
| **Total communication** | | | **~30%** |

On clusters with PCIe interconnect (e.g., A40, L40), communication can consume **40–55%** of training time due to lower bandwidth.

### Why DHelix is a natural fit for linear RNNs

DHelix's strand interleaving (SI) overlaps the forward pass of micro-batch $\beta$ with the backward pass of micro-batch $\alpha$. For Transformers, the dominant computation is FlashAttention (which has moderate overlap effectiveness with communication, OEF $\approx 0.70$–$0.85$). For **linear RNNs**, the computation graph is different:

1. **Chunkwise intra-chunk matmul** ($Q K^\top$ and $S V$): These are standard GEMMs — excellent communication overlap candidates (OEF $\approx 0.85$–$0.99$ with AllGather/ReduceScatter).

2. **Inter-chunk sequential scan**: The scan across chunks is lightweight ($O(G \cdot d_k \cdot d_v)$ where $G$ is the number of chunks) and runs on scalar ALU or small matmuls (MatMulScan). This creates natural **idle tensor core cycles** during which communication can proceed without contention.

3. **LASP-2 AllGather**: The state AllGather transfers $H \times d_k \times d_v$ elements per layer — small compared to TP AllGather ($B \times T \times d$). This means LASP-2 communication can be fully hidden behind even small computation segments.

4. **Backward pass asymmetry**: The backward chunkwise computation has a 3-pass structure (recompute forward states → backward scan → gradient accumulation). The recompute pass is compute-bound; the backward scan has the same overlap opportunities as the forward scan. DHelix can pair the forward intra-chunk matmul ($\beta$-strand) with the backward gradient accumulation ($\alpha$-strand).

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than baseline on A40/L40 clusters?** Yes — DHelix shows 31–40% gains for Transformers on A40 where communication is 30–55% of total time. Linear RNNs have similar communication fractions with LASP-2 SP. The scan's sequential phases provide even better overlap opportunities (no SM contention with communication).

2. **Can I sketch the CUDA kernel structure in 5 minutes?** Yes — the kernel structure is unchanged from LASP-2 + TFLA. DHelix only changes the **scheduling** of existing operators across micro-batches via CUDA streams. No new kernels needed — only operator reordering and stream assignment.

3. **Does it reduce HBM bandwidth or increase compute utilization?** Yes — it increases **effective utilization** by filling communication bubbles with cross-strand computation. The FLOPs and bandwidth are unchanged, but the overlap eliminates idle cycles.

## Related Work

- **DHelix (Wang et al., 2024)**: Introduced strand interleaving for Transformer/MoE distributed training. Evaluated on Llama, GPT, and Phi models with 64×A40 and 64×A800 clusters. Achieves 31–40% throughput improvement. **Not evaluated on linear RNN architectures** — all experiments use softmax attention + MLP/MoE layers. Our proposal adapts DHelix specifically for the linear RNN computation graph with LASP-2 communication.
- **LASP-2 (Sun et al., 2025)**: Sequence parallelism for linear attention via AllGather of memory states. Includes intra-batch overlap of AllGather with intra-chunk attention. **Our approach**: Adds inter-batch overlap via DHelix SI on top of LASP-2's existing intra-batch overlap — these are orthogonal and composable.
- **FLUX (Chang et al., 2024)**: Fuses AllGather/ReduceScatter into GEMM at tile granularity within a single micro-batch. **Our approach**: FLUX handles intra-operator overlap; DHelix handles inter-micro-batch overlap. They compose: the $\beta$-strand can use FLUX-fused GEMM kernels while the $\alpha$-strand's communication runs concurrently.
- **MegaScale (Jiang et al., 2024)**: Intra-batch overlap via operator decomposition. Achieves ~26% communication hiding. **Our approach**: DHelix achieves ~82% hiding by exploiting inter-batch freedom.
- **Proposal 047 (LASP-2 + TFLA overlap)**: Proposes overlapping LASP-2 AllGather with TFLA intra-chunk computation. **Our approach**: Extends proposal 047 with full DHelix scheduling — overlapping not just LASP-2 AllGather but ALL communication (TP, SP, PP) across micro-batches.

**Gap**: No existing work applies DHelix-style strand interleaving to linear RNN architectures. The linear RNN computation graph (chunkwise matmul + sequential scan + LASP-2 state AllGather) has different operator timing profiles than Transformers, requiring architecture-specific profiling and DP-based schedule optimization.

## Mathematical Formulation

### Operator Decomposition for Linear RNN Layer

A single GLA layer with LASP-2 SP decomposes into the following operators per micro-batch:

**Forward pass operators:**

$$
\mathcal{F} = \{F_1, F_2, F_3, F_4, F_5, F_6, F_7\}
$$

| Operator | Description | Type | Time (µs) |
|----------|-------------|------|-----------|
| $F_1$: AG($x$) | TP AllGather input | Communication | $T_{\text{ag}}$ |
| $F_2$: GEMM($Q,K,V,\gamma$) | Input projections | Computation | $T_{\text{proj}}$ |
| $F_3$: IntraChunk | TFLA intra-chunk attention | Computation | $T_{\text{intra}}$ |
| $F_4$: AG(state) | LASP-2 AllGather states | Communication | $T_{\text{sp}}$ |
| $F_5$: InterChunk | Inter-chunk scan | Computation | $T_{\text{scan}}$ |
| $F_6$: Gate + GEMM($W_O$) | Output gating + projection | Computation | $T_{\text{out}}$ |
| $F_7$: RS($y$) | TP ReduceScatter output | Communication | $T_{\text{rs}}$ |

**Backward pass operators:**

$$
\mathcal{B} = \{B_1, B_2, B_3, B_4, B_5, B_6, B_7\}
$$

| Operator | Description | Type | Time (µs) |
|----------|-------------|------|-----------|
| $B_1$: AG($dy$) | TP AllGather gradient | Communication | $T_{\text{ag}}$ |
| $B_2$: dGEMM($W_O$) | Output projection gradient | Computation | $T_{\text{dout}}$ |
| $B_3$: dIntraChunk | Backward intra-chunk | Computation | $T_{\text{dintra}}$ |
| $B_4$: RS(dstate) | LASP-2 ReduceScatter grad | Communication | $T_{\text{sp}}$ |
| $B_5$: dInterChunk | Backward inter-chunk scan | Computation | $T_{\text{dscan}}$ |
| $B_6$: dGEMM($W_Q,W_K,W_V$) | Input projection gradients | Computation | $T_{\text{dproj}}$ |
| $B_7$: RS($dx$) | TP ReduceScatter grad | Communication | $T_{\text{rs}}$ |

### DHelix DP Schedule for Linear RNN

The DP objective is to pair forward operators from $\beta$-strand with backward operators from $\alpha$-strand to minimize makespan:

$$
T_{\text{opt}}(i, j) = \min \begin{cases} T_{\text{opt}}(i-1, j) + P(F_i^{\beta}, \emptyset) \\ T_{\text{opt}}(i, j-1) + P(\emptyset, B_j^{\alpha}) \\ T_{\text{opt}}(i-1, j-1) + P(F_i^{\beta}, B_j^{\alpha}) \end{cases}
$$

where $P(F_i^{\beta}, B_j^{\alpha})$ is the profiled overlapped execution time.

**Expected high-quality pairings for linear RNN:**

| $\beta$-strand (forward) | $\alpha$-strand (backward) | Expected OEF |
|--------------------------|---------------------------|--------------|
| $F_1$: AG($x$) | $B_2$: dGEMM($W_O$) | $0.85$–$0.99$ |
| $F_3$: IntraChunk | $B_1$: AG($dy$) | $0.80$–$0.90$ |
| $F_4$: AG(state) | $B_3$: dIntraChunk | $0.85$–$0.95$ |
| $F_5$: InterChunk | $B_4$: RS(dstate) | $0.90$–$1.00$ |
| $F_6$: Gate + GEMM | $B_7$: RS($dx$) | $0.85$–$0.99$ |

The inter-chunk scan ($F_5$) paired with LASP-2 ReduceScatter ($B_4$) is uniquely favorable for linear RNNs: the scan runs on scalar ALU/small matmuls, leaving tensor cores and NIC completely free for the ReduceScatter.

### Throughput Model

Let $T_{\text{comp}}$ = total computation time per layer (forward + backward), $T_{\text{comm}}$ = total communication time:

**Without DHelix (sequential):**

$$
T_{\text{layer}} = T_{\text{comp}} + T_{\text{comm}}
$$

**With DHelix (SI, ideal):**

$$
T_{\text{layer}} = T_{\text{comp}} + (1 - \text{OEF}_{\text{avg}}) \cdot T_{\text{comm}}
$$

For linear RNN with $T_{\text{comm}} / T_{\text{layer}} = 0.30$ and $\text{OEF}_{\text{avg}} = 0.85$:

$$
\text{Speedup} = \frac{T_{\text{comp}} + T_{\text{comm}}}{T_{\text{comp}} + 0.15 \cdot T_{\text{comm}}} = \frac{1.0}{0.70 + 0.045} = 1.34\times
$$

### Key Variables

- $\mathcal{F}, \mathcal{B}$ — forward and backward operator sequences
- $P(i, j)$ — profiled overlapped execution time for operator pair $(i, j)$
- $\text{OEF}_{i,j}$ — overlap effectiveness factor: $(T_i + T_j - P_{i,j}) / \min(T_i, T_j)$
- $T_{\text{ag}}, T_{\text{rs}}, T_{\text{sp}}$ — communication times for AllGather, ReduceScatter, LASP-2 state sync
- $T_{\text{intra}}, T_{\text{scan}}$ — chunkwise intra-chunk and inter-chunk computation times

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA-1.3B / Gated DeltaNet-1.3B |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 16$ |
| Head dim | $d_k = d_v = 128$ |
| Chunk size | $C = 128$ |
| Sequence length | $T = 8192$ |
| Cluster | 32–64 GPUs (A40 or H100) |
| Parallelism | TP=4, SP=2 (LASP-2), PP=2 |
| Micro-batch | 2 per GPU (for SI interleaving) |

### Baseline

1. **Megatron-LM baseline**: Standard TP+PP for GLA-1.3B, no communication overlap — current practice
2. **Intra-batch overlap (FLUX-style)**: AllGather/ReduceScatter fused into GEMM tiles — best existing single-batch overlap
3. **LASP-2 intra-batch overlap**: AllGather of state overlapped with intra-chunk attention — proposal 047's approach
4. **DHelix for Transformer**: Same model size but with softmax attention — to calibrate DHelix gains for this cluster

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $\geq 1.25\times$ Megatron-LM baseline | Tokens/sec/GPU on 32×A40 |
| Communication hiding | $\geq 75\%$ of comm time hidden | Profiled overlap via nsys |
| MFU | $> 50\%$ on A40 | Model FLOPs utilization |
| Memory overhead | $< 5\%$ over single-strand | Peak GPU memory |
| Quality | Identical | Perplexity matches non-SI training |

### Estimated Compute

**Phase 1 — Profiling (MVE)**: ~2 GPU-hours on 8×A40 ($\sim \$16$)
- Profile all operator pairs for GLA layer on target cluster
- Run DP search for optimal schedule
- Microbenchmark overlapped vs. non-overlapped throughput

**Phase 2 — Integration**: ~16 GPU-hours on 32×A40 ($\sim \$128$)
- Integrate DHelix scheduling into LASP-2 training framework
- Validate correctness on short training runs (100 steps)

**Phase 3 — Full pretraining**: ~256 GPU-hours on 32×A40 ($\sim \$1024$)
- GLA-1.3B pretraining on 15B tokens with DHelix vs. baselines

## Expected Outcome

**If hypothesis is correct:**

- $1.25$–$1.5\times$ throughput on A40 cluster (PCIe bandwidth bottleneck makes communication hiding very profitable)
- $1.1$–$1.25\times$ throughput on H100 cluster (NVLink is faster, less room for improvement)
- Communication hidden: $\geq 75\%$ on A40, $\geq 50\%$ on H100
- The inter-chunk scan ↔ communication pairing achieves near-perfect OEF ($\geq 0.95$) due to no compute resource contention
- LASP-2's state AllGather is fully hidden behind intra-chunk matmul from the opposite strand

**Quantitative prediction:**

| Cluster | Baseline (TFLOPS/GPU) | DHelix (TFLOPS/GPU) | Gain |
|---------|----------------------|---------------------|------|
| 32×A40 (PCIe) | 55–65 | 75–90 | +30–40% |
| 32×H100 (NVLink) | 180–200 | 200–230 | +10–15% |

**If hypothesis is wrong:**

- **Scenario A**: Linear RNN's chunkwise computation is too fine-grained for DHelix's operator-level overlap. The individual operators ($F_3, F_5$) are shorter than communication operations, meaning no single computation operator can fully hide a communication operation. **What we learn**: Need sub-operator overlap (FLUX-style tile-level fusion) within the chunkwise kernel. **Mitigation**: Combine DHelix (inter-batch) with FLUX (intra-batch tile-level) for two-level overlap.
- **Scenario B**: The 3-pass backward structure of GLA creates memory pressure that conflicts with SI's dual micro-batch requirement. The recompute-forward pass in backward doubles activation memory, leaving insufficient memory for the second micro-batch. **What we learn**: SI requires selective recomputation strategies. **Mitigation**: Use gradient checkpointing at every 2nd layer, reducing activation memory to accommodate SI.
- **Scenario C**: W-shaped pipeline folding creates load imbalance for linear RNN layers that have variable computation time (due to data-dependent gating). **What we learn**: Need adaptive pipeline scheduling. **Mitigation**: Profile per-layer timing variance and adjust folding assignment.

## Minimum Viable Experiment

### Setup
- **Model**: GLA with 4 layers, $d = 512$, $H = 8$, ~50M params
- **Cluster**: 8×A40 (or 8×A100), TP=4, SP=2
- **Task**: Measure operator-level OEF for all GLA operator pairs (no actual training)
- **Method**:
  1. Profile each GLA operator solo (14 operators: 7 fwd + 7 bwd)
  2. Profile all productive pairs (comp ↔ comm): ~40 pairs
  3. Run DP search to find optimal schedule
  4. Execute 100 forward+backward iterations with DHelix schedule
  5. Compare throughput vs. sequential (non-overlapped) execution
- **Compute**: Single 8-GPU node, $< 30$ minutes

### Success Criteria
- Average OEF for comp ↔ comm pairs $\geq 0.80$
- DP-optimized schedule achieves $\geq 1.15\times$ throughput over sequential baseline
- The inter-chunk scan ↔ communication pair achieves OEF $\geq 0.90$
- Memory overhead from dual micro-batch $\leq 10\%$ of single micro-batch

### Failure Criteria
- **Kill if**: Average OEF $< 0.50$ — the operators interfere too much when co-executed
- **Kill if**: Memory overhead $> 30\%$ — SI is impractical for pretraining memory budgets
- **Kill if**: DP schedule is $< 1.05\times$ faster than sequential — the scheduling overhead negates the overlap benefit

### Why This Test Is Sufficient
- The OEF values are the fundamental signal: they determine whether overlapping computation with communication is productive for the linear RNN operator mix
- If OEF values are high (as expected from DHelix's Transformer results), the full-scale training speedup follows directly
- The 8-GPU microbenchmark captures the communication patterns of the full-scale experiment (same TP/SP communication operations, just smaller)
- No actual pretraining needed — the throughput gain is purely a scheduling optimization that doesn't affect model quality

## Memory Access Pattern Analysis

**Coalesced access:** All individual operators (GEMMs, TFLA kernel, elementwise ops) maintain their original coalesced access patterns. DHelix only changes operator scheduling, not operator implementation.

**Cache-friendly:** Inter-strand operators use different CUDA streams, operating on different micro-batch data. L2 cache sharing between strands is beneficial when both strands access the same weight matrices (which are shared across micro-batches).

**Arithmetic intensity:** Unchanged from single-strand training. The dominant operations are GEMMs (high arithmetic intensity) and communication (zero arithmetic intensity, pure bandwidth). Overlap pairs a high-AI operation with a zero-AI operation — ideal for resource complementarity.

**HBM bandwidth:** Two micro-batches access $2\times$ the activation data, but at staggered times. Net effect: activation data streams are interleaved rather than sequential, maintaining the same peak bandwidth requirement.

## Parallelism Analysis

**SM saturation:** During overlapped execution, the compute-bound operator (GEMM or TFLA) saturates SMs while the communication operator uses the NIC DMA engine and NCCL's dedicated SM allocation (typically 2–4 SMs). This is the ideal overlap scenario — different hardware resources.

**Warp divergence:** None introduced by DHelix. Each stream's operators execute normally on their assigned SMs.

**Tensor core mapping:** Unchanged. GEMMs and TFLA matmuls use tensor cores as before. The inter-chunk scan uses scalar ALU as before. DHelix scheduling is orthogonal to tensor core utilization.

**Sequential bottleneck:** The inter-chunk scan is still sequential, but its short duration ($< 5\%$ of layer time) makes it an ideal "communication window" rather than a bottleneck.

## Theoretical Analysis

Complexity comparison per training step:

| Operation | Baseline | DHelix-SI |
|-----------|----------|-----------|
| Computation FLOPs | $F$ | $F$ (identical) |
| Communication volume | $V$ | $V$ (identical) |
| Exposed communication time | $T_{\text{comm}}$ | $(1 - \text{OEF}_{\text{avg}}) \cdot T_{\text{comm}}$ |
| Total wall-clock | $T_{\text{comp}} + T_{\text{comm}}$ | $T_{\text{comp}} + 0.15 \cdot T_{\text{comm}}$ |
| Memory overhead | $M$ | $M + 0.03 M$ (SI memory) |

Crossover point: DHelix is beneficial whenever $T_{\text{comm}} / T_{\text{total}} > 10\%$. For multi-GPU linear RNN training, this is almost always satisfied.

## Risks & Limitations

1. **Profiling cost**: DHelix requires ~140 operator pair profiles per hardware configuration. For linear RNNs, the number of operators per layer is similar to Transformers (~14), so profiling cost is comparable (~10–30 minutes). However, different chunk sizes ($C$) and head configurations may require re-profiling.

2. **NCCL SM contention**: NCCL kernels consume 2–4 SMs for communication processing. During SI, both strands' NCCL operations compete for these SMs. **Mitigation**: Tune `NCCL_NTHREADS` and `NCCL_MAX_NCHANNELS` to limit NCCL's SM usage.

3. **Activation memory for two micro-batches**: SI processes two micro-batches simultaneously, requiring activation storage for both. For GLA with recomputation (storing only $S_{[n]}$ states), the overhead is small (~3% as shown by DHelix). Without recomputation, the overhead could be ~20%.

4. **W-shaped pipeline + LASP-2 interaction**: Model folding for PP changes which layers are co-located on each GPU. LASP-2's SP dimension must be compatible with the folded layer assignment. **Mitigation**: Keep SP within each PP stage (SP does not cross PP boundaries).

5. **Not useful for single-GPU or DP-only training**: DHelix requires model parallelism (TP/SP/PP) to generate communication to overlap. Single-GPU or pure DP training has no intra-layer communication.

6. **Lower gains on fast interconnects**: On H100 SXM with 900 GB/s NVLink, communication is already fast. DHelix's gains for linear RNNs may be only 10–15% on such clusters (vs. 30–40% on PCIe clusters).

## Follow-up Experiments

1. **Combine with proposal 040 (persistent megakernel)**: If the linear RNN layer is fused into a single persistent kernel, DHelix's operator decomposition needs to be adapted. The megakernel exposes fewer overlap points but each is longer-lived. DHelix's DP would optimize at sub-layer granularity within the megakernel.

2. **Combine with proposal 048 (segmented MatMulScan packing)**: Packed variable-length training with DHelix. The packing removes padding waste; DHelix hides communication. These are orthogonal optimizations that compose for maximum throughput.

3. **Hybrid linear-attention + softmax-attention models**: Models like Jamba or Griffin that interleave linear RNN layers with softmax attention layers would have a heterogeneous computation graph. DHelix's DP search would find different pairing schedules for each layer type.

4. **Expert Parallelism overlap**: For MoE variants of linear RNN models, the All-to-All communication from expert parallelism provides additional overlap targets. Profile and optimize the schedule for linear-RNN-MoE models.

5. **Blackwell adaptation**: NVIDIA Blackwell introduces new NVLink-C2C and NVSwitch generations. Re-profile and optimize DHelix schedules for the new hardware.

## Human Review

(To be filled by reviewer)

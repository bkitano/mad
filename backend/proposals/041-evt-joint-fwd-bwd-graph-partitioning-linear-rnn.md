---
status: ongoing
priority: high
created: 2026-02-15
based_on: epilogue-visitor-tree-fusion, chimera-block-reorder-compute-fusion, chunkwise-parallel-scan, io-aware-tiling, kernel-fusion, bilinear-gating-glu, batch-reduce-gemm
experiment_number: 041
experiment_log: experiment-log-041.md
---

# EVT Joint Forward-Backward Graph Partitioning for Linear RNN Training

## Hypothesis

Applying EVT's ILP-based graph partitioner to the **joint forward-backward computation DAG** of a chunkwise linear RNN layer (GLA/Mamba-2/DeltaNet) — finding the optimal partitioning of the combined fwd+bwd GEMM chain into fusible epilogue subgraphs — will reduce total HBM traffic by $35$–$50\%$ and achieve $1.3$–$1.7\times$ wall-clock training throughput improvement over the current approach (separate forward and backward Triton kernels with PyTorch autograd), because: (a) shared tensors ($Q$, $K$, $V$, gates) that are written in forward and re-read in backward can be kept on-chip across the fwd→bwd boundary, (b) backward epilogue operations (gradient gating, decay-weighted gradient accumulation, residual gradient addition) can be fused into GEMM epilogues just like forward operations, and (c) the ILP partitioner provably minimizes HBM materialization subject to register/SMEM constraints.

## Background

### The training bottleneck: backward pass is 2–3× slower and unoptimized

All existing kernel-level proposals for chunkwise linear RNNs (proposals 032–040) focus primarily on the **forward pass**. The backward pass is typically left to PyTorch autograd, which:

1. **Stores all forward intermediates** ($Q$, $K$, $V$, gates $\alpha$, intra-chunk scores $S$, state corrections) to HBM for the backward pass — creating massive memory pressure
2. **Launches many separate kernels** for each gradient computation — each kernel reads inputs from HBM, computes one gradient, and writes back
3. **Does not fuse** gradient gating, decay-weighted accumulation, or residual gradient addition into GEMM epilogues

The backward pass of a chunkwise linear RNN layer involves the same structural pattern as the forward:

**Forward:**
$$O_j = \underbrace{(\underbrace{Q_j K_j^\top}_{\text{GEMM1}} \odot M_j)}_{\text{masked scores}} V_j + Q_j \cdot h_{jC}$$

**Backward (gradient w.r.t. $V_j$):**
$$\nabla_{V_j} = (\underbrace{K_j Q_j^\top}_{\text{GEMM1-bwd}} \odot M_j^\top) \nabla_{O_j} + h_{jC}^\top Q_j^\top \nabla_{O_j}$$

**Backward (gradient w.r.t. $Q_j$):**
$$\nabla_{Q_j} = (\nabla_{O_j} V_j^\top \odot M_j) K_j + \nabla_{O_j} h_{jC}^\top$$

These backward GEMM chains have identical structure to the forward: GEMM → elementwise mask → GEMM, with an additive state correction. They benefit from the same fusion techniques (Chimera block reordering, EVT epilogue fusion) but **no proposal applies these to the backward pass**.

### Why joint fwd+bwd partitioning is better than separate optimization

EVT's ILP-based graph partitioner operates on the full computation DAG. For a linear RNN layer, the joint forward-backward DAG has these key properties:

1. **Shared tensors**: $Q_j$, $K_j$, $V_j$ are computed during forward projection and consumed in both forward attention and backward gradient computation. Currently, they are written to HBM after forward projection and re-read for both forward attention and backward. Joint partitioning can identify that keeping these tensors in registers/SMEM across multiple GEMMs reduces HBM traffic.

2. **Recomputation opportunities**: Instead of storing forward intermediates ($S_j = Q_j K_j^\top \odot M_j$) for backward, recomputing them during the backward pass within a fused kernel trades FLOPs for HBM bandwidth — exactly the FlashAttention insight, but applied to linear attention's backward pass.

3. **Cross-boundary fusion**: The forward output gating $o_j = \sigma(g_j) \odot O_j$ and its backward gradient $\nabla_{O_j} = \sigma(g_j) \odot \nabla_{o_j}$ share the same gate values. Joint partitioning can fuse the backward gate application into the same epilogue that computes the forward gate, if the backward gradient is available (i.e., in a layer-by-layer backward scheme).

### Why EVT is the right abstraction

EVT's composable visitor-node tree maps directly to the post-GEMM operations in both forward and backward:

| Forward Epilogue | Backward Epilogue |
|-----------------|-------------------|
| Decay mask application: $S_j \odot M_j$ | Transposed mask: $\nabla_S \odot M_j^\top$ |
| SiLU gating: $\text{SiLU}(g) \odot O$ | Gradient gating: $\sigma(g)(1 + g(1-\sigma(g))) \odot \nabla_o$ |
| Residual add: $o + x$ | Gradient passthrough: $\nabla_o + \nabla_{\text{residual}}$ |
| State correction add: $O + Q \cdot h$ | State gradient add: $\nabla_Q + \nabla_O \cdot h^\top$ |

Each of these operations can be expressed as EVT visitor nodes (Compute, AuxLoad, AuxStore). The ILP partitioner then finds the optimal grouping.

### What this proposal does NOT do

This proposal does **not** propose new architectural modifications. It is a pure **systems/kernel optimization** that takes an existing model architecture (GLA, Mamba-2, DeltaNet) and finds the optimal kernel fusion strategy for its training backward pass.

## Related Work

- **[EVT (Chen et al., ASPLOS 2024)](https://dl.acm.org/doi/10.1145/3620666.3651369)**: Introduced the ILP-based graph partitioner for joint forward-backward fusion. Applied to standard transformer MLPs, not to linear RNN/SSM layers. Our proposal extends EVT to the chunkwise linear attention backward pass.
- **[GLA (Yang et al., ICML 2024)](https://arxiv.org/abs/2312.06635)**: Derives the chunkwise parallel algorithm and provides Triton forward+backward kernels. The backward kernel is hand-written without systematic fusion optimization. Our proposal applies EVT's automated partitioner to GLA's backward DAG.
- **[TFLA (Zhong et al., 2025)](https://arxiv.org/abs/2503.14376)**: Introduces two-level tiling for forward pass. Mentions backward pass as future work. Our proposal addresses backward pass fusion.
- **[FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)**: Applies recomputation-over-storage to softmax attention backward. Our proposal applies the same principle to linear attention's backward pass with EVT-guided optimal partitioning.
- **Proposal 032 (Chimera-Fused Chunkwise SSM)**: Applies Chimera to forward intra-chunk GEMM chain. Our proposal extends to backward and uses EVT's ILP partitioner for joint optimization.
- **Proposal 033 (EVT-Fused SSM Epilogues)**: Fuses forward elementwise epilogues. Our proposal extends to backward epilogues and uses the ILP partitioner for cross-boundary optimization.

No existing work applies EVT's ILP-based joint forward-backward graph partitioning to linear RNN layers.

## Mathematical Formulation

### Joint Forward-Backward DAG for Chunkwise GLA

Consider a single chunk $j$ with inputs $Q_j, K_j, V_j \in \mathbb{R}^{C \times d}$, gates $g_j \in \mathbb{R}^{C}$, decay rates $\alpha_j \in \mathbb{R}^{C}$, and boundary state $h_{jC} \in \mathbb{R}^{n \times d_v}$.

**Forward computation graph (7 nodes):**

$$
\begin{aligned}
\text{F1:}\quad & S_j = Q_j K_j^\top & \quad & \text{GEMM: } C \times d \times C \\
\text{F2:}\quad & \tilde{S}_j = S_j \odot M_j & \quad & \text{Elementwise (EVT)} \\
\text{F3:}\quad & O_j^{\text{intra}} = \tilde{S}_j V_j & \quad & \text{GEMM: } C \times C \times d \\
\text{F4:}\quad & O_j^{\text{state}} = Q_j h_{jC} & \quad & \text{GEMM: } C \times n \times d_v \\
\text{F5:}\quad & O_j = O_j^{\text{intra}} + O_j^{\text{state}} & \quad & \text{Elementwise (EVT)} \\
\text{F6:}\quad & o_j = \text{SiLU}(g_j) \odot O_j & \quad & \text{Elementwise (EVT)} \\
\text{F7:}\quad & z_j = o_j + x_j & \quad & \text{Residual (EVT)} \\
\end{aligned}
$$

**Backward computation graph (8 nodes, given $\nabla_{z_j}$):**

$$
\begin{aligned}
\text{B1:}\quad & \nabla_{o_j} = \nabla_{z_j} & \quad & \text{(identity for residual)} \\
\text{B2:}\quad & \nabla_{O_j} = \sigma'(g_j) \odot \nabla_{o_j} & \quad & \text{Elementwise (EVT)} \\
\text{B3:}\quad & \nabla_{g_j} = (\sigma(g_j)(1 + g_j(1-\sigma(g_j)))) \cdot O_j \cdot \nabla_{o_j} & \quad & \text{Elementwise (EVT)} \\
\text{B4:}\quad & \nabla_{\tilde{S}_j} = \nabla_{O_j} V_j^\top & \quad & \text{GEMM: } C \times d \times C \\
\text{B5:}\quad & \nabla_{V_j} = \tilde{S}_j^\top \nabla_{O_j} & \quad & \text{GEMM: } C \times C \times d \\
\text{B6:}\quad & \nabla_{S_j} = \nabla_{\tilde{S}_j} \odot M_j & \quad & \text{Elementwise (EVT)} \\
\text{B7:}\quad & \nabla_{Q_j} = \nabla_{S_j} K_j + \nabla_{O_j} h_{jC}^\top & \quad & \text{GEMM + add (EVT)} \\
\text{B8:}\quad & \nabla_{K_j} = \nabla_{S_j}^\top Q_j & \quad & \text{GEMM: } C \times C \times d \\
\end{aligned}
$$

### HBM Traffic Analysis

**Current approach (separate fwd + bwd kernels):**

| Tensor | Written (fwd) | Read (bwd) | Size |
|--------|--------------|-----------|------|
| $Q_j$ | $Cd$ | $Cd$ | $Cd$ bytes (×2 for read+write) |
| $K_j$ | $Cd$ | $Cd$ | $Cd$ bytes (×2) |
| $V_j$ | $Cd$ | $Cd$ | $Cd$ bytes (×2) |
| $g_j$ | $C$ | $C$ | $C$ bytes (×2) |
| $\tilde{S}_j$ (or recomputed) | $C^2$ | $C^2$ | $C^2$ bytes (×2) |
| $O_j$ | $Cd$ | $Cd$ | $Cd$ bytes (×2) |
| $o_j$ | $Cd$ | — | $Cd$ bytes (×1) |
| $\nabla_{O_j}$ | $Cd$ | — | $Cd$ bytes |
| Gradients | — | $3Cd$ | $3Cd$ bytes |

Total HBM traffic $\approx 14Cd + 2C^2$ per chunk (in BF16 elements).

**Joint EVT-partitioned approach:**

The ILP partitioner can identify these fusion opportunities:

1. **F1+F2+F3 fused** (Chimera GEMM chain): $\tilde{S}_j$ never materialized → saves $2C^2$
2. **F5+F6+F7 fused** (EVT epilogue): state correction, gating, residual in one epilogue → saves $2Cd$
3. **B2+B3 fused** (EVT epilogue): gradient gating computed once → saves $Cd$
4. **B4+B6+B7 fused** (Chimera + EVT): $\nabla_{\tilde{S}_j}$ never materialized → saves $2C^2$
5. **B5 + recomputed $\tilde{S}_j$** (recomputation): avoid storing $\tilde{S}_j$ → saves $C^2$ storage
6. **F6 stores $O_j$ for B3** via AuxStore: already in EVT tree → no extra kernel

Optimized total HBM traffic $\approx 8Cd$ per chunk.

**Reduction:** $(14Cd + 2C^2) \to 8Cd$ = savings of $6Cd + 2C^2$. For $C = 64$, $d = 128$: from 118,784 to 65,536 elements = **45% reduction**.

### ILP Formulation

Let $G = (V, E)$ be the joint DAG with $|V| = 15$ nodes (F1–F7 + B1–B8). The ILP assigns each node $v$ to a partition $p_v \in \{1, \ldots, P\}$:

$$
\min \sum_{(u,v) \in E} x_{uv} \quad \text{s.t.} \quad x_{uv} \geq |p_u - p_v|
$$

$$
\sum_{v \in S_k} r_v \leq R_{\max} \quad \forall \text{ partition } S_k
$$

$$
\sum_{v \in S_k} s_v \leq \text{SMEM}_{\max} \quad \forall \text{ partition } S_k
$$

where:
- $x_{uv} = 1$ if edge $(u,v)$ crosses partition boundaries (requiring HBM materialization)
- $r_v$ = register cost of node $v$ (proportional to tile size of its GEMM/elementwise op)
- $s_v$ = shared memory cost (for AuxLoad/AuxStore of auxiliary tensors)
- $R_{\max} = 255$ registers per thread (H100)
- $\text{SMEM}_{\max} = 228$ KB (H100 per SM)

The ILP has 15 integer variables and ~30 constraints — trivially solvable by any ILP solver in milliseconds.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Mamba-2 SSD / Gated DeltaNet |
| Layers | $L = 12$ (small) to $L = 48$ (large) |
| Hidden dim | $d = 768$ (small) to $d = 2048$ (large) |
| State dim | $n = 16$ (GLA default) |
| Chunk size | $C = 64$ (default) to $C = 256$ (TFLA) |
| Sequence length | $T = 2048$ to $T = 8192$ |

### Baseline

1. **flash-linear-attention (fla) library**: Current SOTA Triton kernels for GLA/DeltaNet forward + backward. Backward uses separate Triton kernels with autograd. Complexity: forward $O(TC^2d)$, backward $O(TC^2d)$ (same asymptotic, ~2.5× wall-clock of forward).
2. **TFLA forward + autograd backward**: TFLA optimized forward + standard backward.
3. **Proposal 032+033 forward + standard backward**: Chimera+EVT optimized forward + unoptimized backward (shows the "forward-only optimization" ceiling).

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.3\times$ fla baseline | tokens/sec on A100/H100 |
| Peak memory | $< 0.7\times$ baseline | `torch.cuda.max_memory_allocated()` |
| HBM traffic (per layer) | $< 0.55\times$ baseline | NVIDIA Nsight Compute `dram_read + dram_write` |
| Backward wall-clock | $< 0.6\times$ baseline | `torch.cuda.Event` timing |
| Quality | Identical (exact) | Bit-exact output comparison |

### Estimated Compute

- **ILP solver**: < 1 second on CPU (one-time compile-time cost)
- **Kernel development**: ~2 weeks (CUTLASS EVT + Triton hybrid)
- **Benchmarking**: 4–8 GPU-hours on A100 (sweep over model sizes, sequence lengths)
- **Total**: Small experiment (~10 GPU-hours)

## Expected Outcome

**If hypothesis is correct:**
- $1.3$–$1.7\times$ training throughput improvement (from backward pass optimization)
- $30$–$50\%$ peak memory reduction (from eliminating stored forward intermediates)
- The backward pass closes the gap with the forward pass (currently backward is 2–3× slower)

**If hypothesis is wrong:**
- Register pressure from fusing too many operations may force spilling, negating HBM traffic savings
- The ILP solution may show that the optimal partitioning is essentially what hand-written kernels already do
- We learn the **theoretical ceiling** for backward pass fusion — even if the implementation is hard, knowing the optimal partition guides future kernel design

## Minimum Viable Experiment

### Setup
- **Model**: Tiny GLA model: 1 layer, $d = 64$, $n = 16$, $C = 32$, ~50K params
- **Task**: Synthetic autoregressive next-token prediction on random sequences ($T = 256$)
- **Implementation**: Write a single fused Triton kernel that combines forward attention GEMM chain + backward gradient GEMM chain for the intra-chunk computation (nodes F1–F3 + B4–B8), using `tl.dot` for GEMMs and register-resident intermediates
- **Comparison**: Same model with separate forward/backward Triton kernels from fla library
- **Data**: 1K random sequences of length 256
- **Compute**: Single A100 GPU, < 10 minutes

### Success Criteria
- **HBM traffic reduction** $\geq 25\%$ for the fused kernel vs. separate kernels (measured via Nsight Compute)
- **Wall-clock speedup** $\geq 1.15\times$ for the combined fwd+bwd pass
- Bit-exact numerical agreement with the reference implementation

### Failure Criteria
- If the fused kernel is **slower** than separate kernels due to register spilling or occupancy loss, the approach is not viable at this scale
- If the ILP partitioner's solution matches what manual fusion already achieves, there is no benefit to automated partitioning

### Why This Test Is Sufficient
- The MVE tests the core hypothesis: can forward and backward GEMM chains share tensors on-chip to reduce HBM traffic? If yes at small scale, scaling up adds more GEMMs to the chain (more fusion opportunities), not fewer.
- Register pressure is the main risk, and it manifests at any scale (small tiles actually have *more* register pressure per element).
- The Triton prototype validates the algorithm; a CUTLASS EVT implementation would provide additional speedup from TMA and warp specialization.

## Theoretical Analysis

Complexity comparison per chunk:

| Operation | Baseline (separate fwd+bwd) | Proposed (joint EVT partition) |
|-----------|---------------------------|-------------------------------|
| Forward FLOPs | $2C^2d + 2Cnd_v$ | $2C^2d + 2Cnd_v$ (identical) |
| Backward FLOPs | $4C^2d + 2Cnd_v$ | $4C^2d + 2Cnd_v + C^2d$ (recompute $\tilde{S}_j$) |
| Forward HBM | $7Cd + C^2$ | $4Cd$ |
| Backward HBM | $7Cd + C^2$ | $4Cd$ |
| Total HBM | $14Cd + 2C^2$ | $8Cd$ |
| Kernel launches | $4$–$6$ | $2$ (fwd partition + bwd partition) |

Crossover: The recomputation overhead ($C^2 d$ extra FLOPs) is always worthwhile because HBM bandwidth ($\sim 2$ TB/s on A100) is the bottleneck, not compute ($\sim 312$ TFLOPS). The arithmetic intensity of the fused kernel increases from $\sim 64$ FLOP/byte to $\sim 112$ FLOP/byte, moving it firmly into the compute-bound regime.

## Memory Access Pattern Analysis

1. **Coalesced access**: All GEMM operands are loaded as contiguous tiles via `tl.load` with coalesced memory patterns. EVT AuxLoad/AuxStore uses TMA for asynchronous coalesced transfers.
2. **Cache-friendly**: The joint partition keeps $Q_j$, $K_j$, $V_j$ in SMEM across fwd→bwd boundary. Temporal locality is maximized: each tensor is loaded once and consumed multiple times within the partition.
3. **Arithmetic intensity**: Baseline: $\frac{6C^2d + 4Cnd_v}{(14Cd + 2C^2) \cdot 2}$ ≈ 64 FLOP/byte. Proposed: $\frac{7C^2d + 4Cnd_v}{8Cd \cdot 2}$ ≈ 112 FLOP/byte (for $C=64$, $d=128$, $n=16$).
4. **SMEM capacity**: The fused kernel needs to hold $Q_j$ ($C \times d$), $K_j$ ($C \times d$), $V_j$ ($C \times d$) tiles simultaneously. For tile size $C_t = 32$, $d_t = 64$: $3 \times 32 \times 64 \times 2$ bytes = 12 KB. Well within H100's 228 KB.

## Parallelism Analysis

1. **SM saturation**: Each chunk is an independent work unit. With $T/C$ chunks per sequence and batch size $B$: total CTAs = $B \times T/C$. For $B=8$, $T=2048$, $C=64$: 256 CTAs, saturating A100's 108 SMs.
2. **No warp divergence**: All threads in a warp execute the same GEMM/elementwise operations (SIMT-friendly).
3. **Tensor core utilization**: All GEMMs use `tl.dot` which maps to WMMA/WGMMA instructions. The added recomputation GEMMs use the same tile shapes.
4. **No sequential bottlenecks**: The inter-chunk scan is sequential but is not part of this proposal (handled by existing scan kernels). This proposal optimizes the intra-chunk computation which is fully parallel across chunks.

## Risks & Limitations

1. **Register pressure**: Fusing fwd+bwd increases live tensor count. If the ILP partitioner must split more than expected, the benefit shrinks.
2. **CUTLASS EVT maturity**: EVT is relatively new (ASPLOS 2024) and primarily supports simple epilogue patterns. Extending to multi-GEMM backward chains may require custom visitor nodes.
3. **Triton limitations**: Triton's compiler may not achieve the register allocation quality needed for the fused kernel. A CUTLASS implementation may be necessary for full benefit.
4. **Non-causal models**: For non-causal (bidirectional) models, the backward pass structure differs and may not benefit from the same partitioning.
5. **Compiler-generated kernels**: Future compiler advances (Mirage, Tawa) may automatically discover similar fusions, making manual EVT partitioning redundant.

## Follow-up Experiments

1. **Multi-layer fusion**: Extend the joint DAG to span 2–3 consecutive layers, fusing the output of one layer's backward with the next layer's backward (similar to Proposal 040's multi-layer megakernel, but for backward).
2. **FP8 backward quantization**: Combine EVT backward fusion with FP8 gradient quantization in the epilogue (quantize gradients while still in registers).
3. **Combine with Proposal 039 (warp specialization)**: Use producer-consumer warp groups within the fused fwd+bwd kernel for overlapping TMA loads with WGMMA compute.
4. **Automatic partitioning benchmark**: Compare ILP-optimal partition against hand-tuned fusion to quantify the value of automated partitioning.
5. **DeltaNet/DeltaProduct backward**: Apply the same technique to DeltaNet's backward pass, which has a different structure (Householder reflection gradients) but similar GEMM chains.

## Human Review

(To be filled by reviewer)

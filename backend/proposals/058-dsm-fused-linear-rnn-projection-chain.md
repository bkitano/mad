---
status: completed
priority: high
created: 2026-02-15
based_on: flashfuser-dsm-inter-core-fusion (046), epilogue-visitor-tree-fusion (039), bilinear-gating-glu (005), post-attention-sigmoid-gating (094), tfla-two-level-tiled-chunkwise-parallelism (158), io-aware-tiling (066), warp-specialized-pipelining (141), input-dependent-gating (065)
experiment_number: 058
experiment_log: experiment-log-058.md
results_file: 058_results.md
---

# DSM-Fused Linear RNN Projection Chain: Eliminating HBM Round-Trips via Distributed Shared Memory

## Hypothesis

Using **Distributed Shared Memory (DSM)** — the inter-SM on-chip interconnect on H100 — to fuse the **input projection GEMMs** ($xW_Q, xW_K, xW_V, xW_g$) with their **downstream elementwise activations** (sigmoid gating, SiLU, normalization) and the **output projection GEMM** ($oW_O$) with its **upstream gating** ($o \odot \text{SiLU}(\text{gate})$) into two DSM-fused kernels will reduce per-layer HBM traffic by **$40$–$55\%$** and achieve **$1.2$–$1.5\times$ wall-clock speedup** for linear RNN layers (GLA, Gated DeltaNet, mLSTM) during pretraining, by keeping the large intermediate activation tensors ($Q, K, V, \text{gate}$) in the ~3.6 MB DSM buffer instead of writing them to HBM between kernel launches.

## Background

### The projection bottleneck: 6 HBM round-trips per layer

A modern gated linear RNN layer (GLA, Gated DeltaNet, mLSTM) consists of:

1. **Input projections** (4 GEMMs): $Q = xW_Q$, $K = xW_K$, $V = xW_V$, $g_{\text{raw}} = xW_g$
2. **Elementwise activations**: $K' = \text{normalize}(K)$, $\alpha = \sigma(xW_\alpha)$, $g = \text{SiLU}(g_{\text{raw}})$
3. **Chunkwise scan** (TFLA kernel): The core recurrence
4. **Output gating**: $o = \text{scan\_output} \odot g$
5. **Output projection** (GEMM): $y = oW_O$
6. **Residual + norm**: $z = \text{LayerNorm}(x + y)$

Each of steps 1-6 is a separate kernel launch. Between steps 1→2→3, the tensors $Q, K, V, g_{\text{raw}} \in \mathbb{R}^{B \times T \times d}$ are written to HBM then read back — even though they're consumed immediately.

**HBM traffic for a typical config** ($B = 8, T = 4096, d = 2048$):

- Each tensor: $8 \times 4096 \times 2048 \times 2 = 128$ MB (bf16)
- 4 projections: write 512 MB, read 512 MB = **1.024 GB** just for input projections
- Activations: read 512 MB, write 512 MB = **1.024 GB** for the elementwise ops
- Output gating: read 256 MB (scan output + gate), write 128 MB = **384 MB**
- Output projection + residual: read 256 MB, write 128 MB = **384 MB**

**Total per-layer HBM traffic for projection chain**: ~2.8 GB. At H100's 3.35 TB/s: ~835 μs.

The chunkwise scan itself (step 3) is a single well-optimized kernel. The surrounding projection chain is **fragmented** into 6+ kernels with massive HBM round-trips.

### Why DSM is the right tool

**EVT fusion** (proposal 033) can fuse elementwise activations into GEMM epilogues but **cannot fuse across two consecutive GEMMs** — the intermediate tensor between $xW_g$ and $oW_O$ exceeds single-SM SMEM capacity (227 KB on H100). For $B_{Lhq} = 64, d = 2048$: the gate tensor tile is $64 \times 2048 \times 2 = 256$ KB > 227 KB.

**FlashFuser's DSM** (trick 046) solves this by pooling SMEM across a 16-SM cluster: 16 × 227 KB = 3.6 MB. This allows fusing the projection GEMM + activation + next GEMM without HBM round-trips, keeping the intermediate tensor on-chip.

**The specific fusion opportunity**:

**Fusion Group A (Input Prologue):** Fuse the 4 input projection GEMMs with their activations:
```
x → [W_Q|W_K|W_V|W_g] → [Q', K', V', g_raw] → activations → [Q, K', V, gate, alpha]
```
Using a single wide GEMM $x [W_Q; W_K; W_V; W_g]$ followed by an EVT epilogue for activations, the 4+3 kernels collapse into 1.

**Fusion Group B (Output Epilogue):** Fuse output gating + output projection + residual:
```
scan_output → gate × scan_output → W_O → + x_residual → LayerNorm
```
Using DSM: the output projection GEMM reads `gate × scan_output` from DSM (gate stays on-chip from Group A if the scan is also DSM-fused) or reads `scan_output` via TMA and loads `gate` from DSM.

### What's different from existing proposals

- **Proposal 033** (EVT-Fused SSM Epilogues): Fuses *only* elementwise ops into GEMM epilogues. Cannot handle the gate tensor that spans across the scan boundary. Our approach uses DSM to bridge larger intermediates.
- **Proposal 040** (Persistent Megakernel): Fuses the *entire* layer into one kernel, including the chunkwise scan. Extremely ambitious; our approach is more surgical — we fuse the projection chains while leaving the scan kernel untouched.
- **Proposal 039** (Warp-Specialized Pingpong): Optimizes *within* the scan kernel. Our approach optimizes *around* it — the two are fully complementary.

## Related Work

- **FlashFuser** (Huang et al., Dec 2025): DSM kernel fusion compiler for compute-intensive operator chains. Applied to Transformer FFN fusion ($\text{Linear} \to \text{GELU} \to \text{Linear}$). Achieved 3.3× over cuBLAS. **Not applied to linear RNN layers** — the computation graph differs (4-way projection + scan + gating, not simple FFN).
- **Mirage MPK** (Jia et al., Dec 2025): Megakernel compiler for inference. Fuses entire model graphs. Inference-only; doesn't handle training backward passes or the chunkwise scan structure.
- **Chimera** (Yang et al., 2024): Block-reorder fusion for GEMM chains. Doesn't use DSM; limited to single-SM SMEM. Cannot fuse the projection chain when intermediates exceed 227 KB.
- **BOLT** (Zheng et al., 2024): Compiler-level fusion for element-wise + reduction patterns. No inter-SM communication; same SMEM limitation as Chimera.

**Gap**: No existing work applies DSM-based inter-SM fusion specifically to the projection chain of linear RNN layers. FlashFuser's DSM primitives are designed for FFN chains ($\text{Linear} \to \phi \to \text{Linear}$); our application extends them to the more complex 4-way projection + gating + output projection pattern of gated linear RNNs.

## Mathematical Formulation

### Standard GLA Layer (6+ kernels):

**Kernel 1 (Fused projection GEMM):**

$$
[Q_{\text{raw}}; K_{\text{raw}}; V; g_{\text{raw}}; \alpha_{\text{raw}}] = x \cdot [W_Q; W_K; W_V; W_g; W_\alpha] \quad \text{(one wide GEMM)}
$$

where $x \in \mathbb{R}^{B T \times d}$ and $[W_Q; \ldots; W_\alpha] \in \mathbb{R}^{d \times (3 d_k + d_v + n)}$.

**Kernels 2-4 (Elementwise activations — separate launches):**

$$
K = \text{normalize}(K_{\text{raw}}), \quad g = \text{SiLU}(g_{\text{raw}}), \quad \alpha = \sigma(\alpha_{\text{raw}})
$$

**Kernel 5 (Chunkwise scan — TFLA):**

$$
S_t = \alpha_t S_{t-1} + k_t v_t^\top, \quad o_t = q_t^\top S_t \quad \text{(chunkwise)}
$$

**Kernel 6 (Output gating):**

$$
\hat{o} = o \odot g
$$

**Kernel 7 (Output projection + residual):**

$$
y = \hat{o} W_O + x
$$

### Proposed: DSM-Fused Projection Chain (2 kernels + scan)

**Fused Kernel A (Input Prologue — 1 DSM-fused launch):**

Using FlashFuser's `dsm_all_exchange` for the GEMM reduction and EVT for elementwise:

$$
x \xrightarrow{\text{GEMM}_{\text{fused}}} [Q_{\text{raw}}; K_{\text{raw}}; V; g_{\text{raw}}; \alpha_{\text{raw}}] \xrightarrow[\text{EVT epilogue}]{\text{in-register}} [Q; K'; V; g; \alpha]
$$

The activation functions (normalize, SiLU, sigmoid) are applied in the EVT epilogue while the GEMM output is still in registers. The results are written to HBM once (for consumption by the scan kernel).

**HBM savings**: Eliminates the read-back of raw projections for activation kernels. Saves $\sim 2 \times (3d_k + d_v + n) \times BT \times 2$ bytes = ~1 GB for the typical config.

**Scan Kernel (unchanged — TFLA):**

$$
\text{TFLA}(Q, K', V, \alpha) \to o \quad \text{(reads Q, K', V, }\alpha\text{ from HBM; writes }o\text{ to HBM)}
$$

**Fused Kernel B (Output Epilogue — 1 DSM-fused launch):**

$$
o \xrightarrow{\text{load via TMA}} o \xrightarrow[\text{DSM: load }g]{\odot} \hat{o} \xrightarrow[\text{GEMM}]{W_O} y_{\text{raw}} \xrightarrow[\text{EVT: AuxLoad}]{+ x} y
$$

The key DSM optimization: the gate tensor $g$ computed in Fused Kernel A is written to DSM (cluster-local SMEM) rather than HBM. Fused Kernel B reads $g$ from DSM, avoiding one full HBM round-trip.

**For this to work**, Kernels A and B must be scheduled on the **same SM cluster**. FlashFuser's cluster scheduling ensures this via hardware thread block clusters (a Hopper feature).

**Alternative (if DSM for gate bridging is infeasible)**: Even without DSM cross-kernel bridging, simply fusing the projection + activation into one EVT-enriched GEMM (Kernel A) and the gating + output projection + residual into another (Kernel B) saves 4 kernel launches and their associated HBM round-trips.

### HBM Traffic Analysis

| Step | Unfused (bytes) | DSM-Fused (bytes) | Savings |
|------|----------------|-------------------|---------|
| Input projections + activations | $2 \times (4d_k + n) \times BT \times 2$ | $(4d_k + n) \times BT \times 2$ (write once) | **50%** |
| Output gating | $2 \times d_v \times BT \times 2$ | $0$ (via DSM or fused GEMM) | **100%** |
| Output projection + residual | $2 \times d_v \times BT \times 2 + d \times BT \times 2$ | $d \times BT \times 2$ (write once) | **33%** |
| **Total projection chain** | $\sim 2.8$ GB | $\sim 1.4$ GB | **50%** |

### Key Variables

- $B$ — batch size
- $T$ — sequence length
- $d$ — model dimension
- $d_k, d_v$ — head key/value dimensions
- $n$ — SSM state dimension
- $H$ — number of heads
- $\text{cls}$ — DSM cluster size (up to 16 SMs on H100)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Gated DeltaNet / mLSTM (unchanged) |
| Kernel | DSM-fused input prologue + TFLA scan + DSM-fused output epilogue |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d = 1024$–$2048$ |
| Heads | $H = 8$–$16$ |
| Head dim | $d_k = d_v = 64$–$128$ |
| Cluster size | 4–16 SMs per cluster |
| GPU | H100 (required for DSM/cluster support) |

### Baseline

1. **fla-org (multi-kernel)**: Current flash-linear-attention implementation — separate kernels for projections, activations, scan, gating, output projection. Well-optimized Triton.
2. **EVT-only fusion** (proposal 033 baseline): Fuse activations into projection GEMM epilogues but no DSM. Measures the value of EVT alone.
3. **FlashAttention-3 Transformer** (throughput reference): Well-optimized softmax attention with warp specialization.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Per-layer HBM traffic | $\leq 0.55\times$ baseline | Nsight Compute L2 sector traffic |
| Projection chain kernel time | $\leq 0.65\times$ baseline | μs for all non-scan kernels |
| End-to-end training throughput | $\geq 1.15\times$ baseline | Tokens/sec for full model |
| Model quality | Identical (bit-exact) | Validation loss comparison |
| DSM utilization | $> 60\%$ DSM bandwidth | Nsight cluster metrics |

### Estimated Compute

**MVE (kernel microbenchmark)**: ~30 minutes on single H100
**Full profiling + EVT integration**: ~8 GPU-hours on H100
**End-to-end training comparison (350M params)**: ~100 GPU-hours
**Total**: ~110 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**

- The input prologue (projection + activations) collapses from 4-7 kernel launches to 1, saving ~500 μs per layer at the typical config. For 24 layers: ~12 ms savings per forward pass.
- The output epilogue (gating + projection + residual) collapses from 3 kernels to 1, saving ~400 μs per layer. For 24 layers: ~10 ms savings.
- Combined: ~22 ms savings per forward pass. At ~80 ms baseline forward time for 350M model: **$\sim 1.28\times$ speedup** for the forward pass. Including backward (which has similar fusion opportunities): **$\sim 1.2\times$ end-to-end**.
- Model outputs are **bit-exact** — pure kernel optimization with no algorithmic change.

**If hypothesis is wrong:**

- **Scenario A: DSM bandwidth insufficient**: If DSM's inter-SM NoC becomes a bottleneck (e.g., 16 SMs contending for limited DSM bandwidth during the all-exchange), the fusion provides less benefit than expected. **Learn**: Measure the DSM bandwidth saturation point. Reduce cluster size from 16 to 4 SMs and retest.
- **Scenario B: EVT-only is sufficient**: If the simple EVT fusion (no DSM) already captures 80%+ of the benefit by fusing activations into GEMM epilogues, the DSM complexity isn't justified. **Learn**: The bottleneck is kernel launches, not intermediate tensor size. Stick with EVT-only fusion (proposal 033).
- **Scenario C: The scan dominates**: If the TFLA chunkwise scan is $> 80\%$ of per-layer time, the projection chain is a small fraction and fusion has limited end-to-end impact. **Learn**: Focus optimization efforts on the scan kernel itself (proposals 038, 039, 044).

## Minimum Viable Experiment

### Setup

- **Task**: Microbenchmark the input projection chain for a GLA layer
- **Configuration**: $B = 8, T = 4096, d = 2048, d_k = d_v = 128, H = 16$
- **Comparison A**: Unfused (4 separate GEMM kernels + 3 activation kernels)
- **Comparison B**: Single fused GEMM $x [W_Q; W_K; W_V; W_g; W_\alpha]$ with EVT epilogue (no DSM)
- **Comparison C**: DSM-fused GEMM chain with cluster scheduling (if supported by CUTLASS 3.x)
- **Hardware**: Single H100 GPU
- **Compute**: < 30 minutes

### Implementation

Step 1: Implement the fused projection GEMM using CUTLASS 3.x's EVT API:

```cpp
// EVT tree for input projection epilogue:
// AccFetch -> Scale(alpha) -> SiLU/sigmoid/normalize -> Store
using EpilogueGate = cutlass::epilogue::collective::Sm90EVT<
    cutlass::epilogue::collective::Sm90Compute<
        cutlass::epilogue::thread::SiLU,    // SiLU for gate channel
        ElementOutput, ElementCompute>,
    cutlass::epilogue::collective::Sm90AccFetch>;

using EpilogueAlpha = cutlass::epilogue::collective::Sm90EVT<
    cutlass::epilogue::collective::Sm90Compute<
        cutlass::epilogue::thread::Sigmoid,  // Sigmoid for decay
        ElementOutput, ElementCompute>,
    cutlass::epilogue::collective::Sm90AccFetch>;
```

Step 2 (if DSM available): Use FlashFuser's `dsm_comm` primitives for cross-kernel gate passing.

### Success Criteria

- Fused projection + activation kernel (EVT, no DSM) is $> 30\%$ faster than 7 separate kernels
- HBM traffic (Nsight) decreases by $> 40\%$ for the projection chain
- Results are numerically identical (bit-exact in bf16)

### Failure Criteria

- If fused kernel is $< 10\%$ faster: GEMM computation dominates, and the activation/launch overhead is negligible at this problem size. The projection chain is compute-bound, not memory-bound.
- If DSM cluster scheduling fails (Hopper bug / unsupported config): Fall back to EVT-only fusion as the primary result.

### Why This Test Is Sufficient

- The microbenchmark directly measures the dominant cost (HBM traffic for intermediate tensors). If the fused kernel eliminates these round-trips, the savings will transfer to any model size and sequence length.
- EVT fusion is a mature technology (CUTLASS 3.x ships it). The risk is in DSM integration, which is why we test EVT-only as a fallback.
- The projection chain's structure is identical across GLA, Gated DeltaNet, and mLSTM — a result for one transfers to all.

## Memory Access Pattern Analysis

**Unfused (current):**
- Input projection: Read $x$ ($BT \times d \times 2$ bytes), read $W_Q$ ($d \times d_k \times 2$ bytes), write $Q$ ($BT \times d_k \times 2$ bytes). Repeated for $K, V, g, \alpha$.
- Activation: Read raw projection, write activated tensor. All from/to HBM.
- **Coalesced**: Yes (contiguous memory layout for each tensor)
- **Arithmetic intensity**: Projection GEMM is $O(d \cdot d_k)$ FLOPs / $O(d + d_k)$ bytes per row = $O(d)$ — high arithmetic intensity for the GEMM itself, but the *inter-kernel* traffic is the bottleneck.

**Fused (proposed):**
- Single wide GEMM: Read $x$ once ($BT \times d \times 2$ bytes), read $[W_Q; W_K; W_V; W_g; W_\alpha]$ once ($d \times (3d_k + d_v + n) \times 2$ bytes), write $[Q; K'; V; g; \alpha]$ once.
- **Coalesced**: Yes — the wide GEMM output is contiguous per output channel.
- **Arithmetic intensity**: Same as unfused GEMM (compute-dominated). But total HBM traffic halved (no intermediate writes/reads).

## Parallelism Analysis

- **Warp divergence**: None — standard GEMM structure
- **Load imbalance**: Minimal — the wide GEMM evenly distributes work across tiles. The EVT epilogue adds ~5% overhead per tile.
- **Tensor core mapping**: Perfect — the fused projection is a single GEMM with shape $BT \times d \times (3d_k + d_v + n)$. Standard MMA tiling applies.
- **Sequential bottleneck**: None for the projection chain (fully parallel across tokens and heads)
- **SM utilization**: For $BT = 32768, d = 2048$: the GEMM has ~$32K \times 2K / (128 \times 128) \approx 4000$ output tiles — excellent SM saturation.

## Theoretical Analysis

| Operation | Unfused (7 kernels) | EVT-Fused (1 GEMM + EVT) | DSM-Fused (1 DSM kernel) |
|-----------|--------------------|--------------------------|-----------------------|
| Kernel launches | 7 | 1 | 1 |
| HBM writes (intermediates) | $5 \times BT \times d_k \times 2$ | $5 \times BT \times d_k \times 2$ (output only) | Same as EVT |
| HBM reads (intermediates) | $5 \times BT \times d_k \times 2$ | $0$ (in-register) | $0$ |
| Launch overhead | $7 \times 10\mu s = 70\mu s$ | $10\mu s$ | $10\mu s$ |
| HBM traffic total | $\sim 2.8$ GB | $\sim 1.5$ GB | $\sim 1.4$ GB |

**Crossover**: The fused approach is always better when kernel launches > 1. The absolute savings grow with $BT \times d$: at $BT = 32768, d = 2048$, savings are ~1.3 GB per layer = ~31 GB total across 24 layers.

## Risks & Limitations

1. **H100-only**: DSM requires Hopper's thread block cluster feature. A100 and earlier GPUs do not support DSM. **Mitigation**: The EVT-only variant (no DSM) works on A100 and provides most of the benefit. The DSM optimization is a Hopper-specific bonus.

2. **CUTLASS integration complexity**: Implementing custom EVT trees for the 5-way projection epilogue requires CUTLASS 3.x expertise. **Mitigation**: Start with the simple case (single GEMM + SiLU/sigmoid EVT) and add complexity incrementally.

3. **Wide GEMM efficiency**: The fused projection GEMM has shape $BT \times d \times (4d_k + n)$, which has a wide $N$ dimension. If $N$ is not divisible by the MMA tile size, there's padding waste. **Mitigation**: Choose $d_k$ and $n$ as multiples of 128 to align with MMA tiles.

4. **Backward pass**: The backward pass requires the intermediate activations (gate values, raw projections) for the chain rule. In the fused forward pass, these intermediates are not written to HBM. **Options**: (a) Recompute during backward (activation checkpointing — standard practice), or (b) write to HBM asynchronously during the forward epilogue (adds back some HBM traffic but overlaps with compute).

5. **Gate tensor bridging across scan**: If we want to pass the gate tensor from the input prologue to the output epilogue without HBM round-trips (via DSM), the gate must persist in the cluster's SMEM during the scan. This is only feasible if the scan kernel runs on the same cluster — which requires full megakernel fusion (proposal 040). **Mitigation**: For the MVE, write the gate to HBM after Kernel A and read it in Kernel B. The DSM gate-bridging is a follow-up optimization.

## Follow-up Experiments

1. **Combine with proposal 039** (warp-specialized TFLA): Apply warp specialization to the scan kernel and DSM fusion to the projection chain. These are fully orthogonal optimizations that should compound.

2. **Combine with proposal 057** (FlashRNN-fused inter-chunk scan): The FlashRNN-fused scan keeps inter-chunk states in SRAM; this proposal keeps projection intermediates in DSM/EVT. Together, they minimize HBM traffic for the entire layer except the scan kernel's KV tile loads.

3. **GatedFFN fusion**: Apply the same DSM fusion to the FFN sublayer's SwiGLU chain ($x \to W_1, W_3 \to \text{SiLU}(W_1 x) \odot W_3 x \to W_2$). This is exactly FlashFuser's original application; combining it with the linear RNN projection fusion gives a full-layer DSM optimization.

4. **Training backward pass fusion**: Apply DSM fusion to the backward pass GEMM chain (gradient projections + activation gradients). The backward graph has similar inter-GEMM elementwise ops that can be EVT-fused.

5. **Multi-GPU DSM clusters**: On DGX H100 systems with NVLink, explore whether DSM can be extended across GPUs for tensor-parallel projection fusion. This would combine communication overlap (trick 049, FLUX) with kernel fusion.

## Human Review

(To be filled by reviewer)

## References

- Huang et al. (2025). FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection. arXiv:2512.12949.
- Chen et al. (2024). EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree. ASPLOS 2024.
- Beck et al. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Yang, Wang, Shen, Panda & Kim (2024). Gated Linear Attention Transformers. ICML 2024.
- Yang, Kautz & Hatamizadeh (2025). Gated Delta Networks. ICLR 2025.
- Shazeer (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Qiu et al. (2025). Gated Attention for Large Language Models. NeurIPS 2025 Best Paper.
- Shah et al. (2024). FlashAttention-3. NeurIPS 2024.
- NVIDIA CUTLASS 3.x documentation — EVT and thread block cluster APIs.

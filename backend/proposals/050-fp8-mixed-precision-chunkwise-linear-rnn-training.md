---
status: ongoing
priority: high
created: 2026-02-15
based_on: tfla-two-level-tiled-chunkwise-parallelism (158), gla-secondary-chunking-log-space-gating (177), warp-specialized-pipelining (141), fused-chunkwise-ssd-atomic-state-passing (182), matmulscan-tcu-parallel-scan (167), io-aware-tiling (066)
experiment_number: 050
experiment_log: experiment-log-050.md
---

# FP8 Mixed-Precision Chunkwise Linear RNN Training Kernel

## Hypothesis

Implementing the intra-chunk matmuls ($QK^\top$ and $SV$) of chunkwise linear RNN training in **FP8 precision** (E4M3 for forward, E5M2 for backward) while keeping the inter-chunk state accumulation and log-space gating in BF16/FP32 will achieve $1.4$–$1.8\times$ wall-clock speedup for linear RNN layers on H100 GPUs, with $<0.5\%$ perplexity degradation compared to BF16 training. The key insight: the chunkwise linear RNN computation naturally decomposes into (a) **large matmuls** that dominate FLOPs and map perfectly to FP8 WGMMA instructions with $2\times$ throughput, and (b) **small sequential operations** (state scan, gating, log-space cumulative products) that require higher precision but account for $<10\%$ of total FLOPs — creating an ideal mixed-precision split with no precision-critical operations in the FP8 path.

## Background

### The FP8 opportunity on Hopper

NVIDIA H100 provides $2\times$ the tensor core throughput for FP8 operations (1978 TFLOPS bf16 → 3958 TFLOPS FP8 for dense matmul). FlashAttention-3 demonstrated that softmax attention can exploit FP8 for the $QK^\top$ and $PV$ matmuls while keeping softmax normalization in FP32, achieving measurable speedups on real workloads.

### Why chunkwise linear RNNs are ideal for FP8

The chunkwise computation of GLA/mLSTM/Gated DeltaNet has a **cleaner precision decomposition** than softmax attention:

**Softmax attention FP8 challenges:**
- The softmax $\exp(x - \max(x)) / \sum$ involves global normalization requiring high precision
- Online softmax (trick 083) iteratively rescales partial results, creating precision dependencies between FP8 matmul tiles and FP32 softmax — requiring careful rescaling at tile boundaries
- FlashAttention-3 handles this with per-tile scaling factors and FP32 accumulation, but the frequent rescaling adds overhead

**Linear RNN FP8 advantages:**
- The intra-chunk "attention" matrix $P = QK^\top \odot D$ has no softmax — it's just a gated matmul. No global normalization needed
- The gating matrix $D$ is applied as an element-wise mask *after* the matmul, not interleaved with it
- The $SV$ matmul (applying attention weights to values) is a standard matmul with no normalization
- The inter-chunk state accumulation $S_{k+1} = \gamma S_k + K^\top V$ is a small $d_k \times d_v$ update — easy to keep in BF16/FP32
- Log-space gating (GLA's secondary chunking) is already segregated into the diagonal $c \times c$ blocks — these small blocks are naturally computed in FP32 and account for $<13\%$ of intra-chunk FLOPs

This means the FP8 boundary in the kernel aligns with the existing **secondary chunking boundary** from GLA: inter-sub-chunk matmuls use FP8 WGMMA, intra-sub-chunk log-space blocks use FP32 — no new precision boundaries needed.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than BF16 TFLA on H100?** Yes — the intra-chunk matmuls ($QK^\top$ and $SV$) account for $>70\%$ of layer FLOPs. FP8 WGMMA provides $2\times$ throughput for these matmuls. Even with overhead for FP8 quantization/dequantization, the net speedup should be $1.4$–$1.7\times$ for the intra-chunk kernel.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — start from the TFLA Triton kernel. Replace `tl.dot(Q, K.T)` with FP8 dot: quantize $Q, K$ tiles to FP8 E4M3 before the dot, accumulate in FP32, dequantize. The gating mask and state scan remain in BF16/FP32 unchanged. For CUTLASS, swap the WGMMA instruction from `SM90_16x8x16_F16F16F32` to `SM90_16x8x32_F8F8F32`.

3. **Does it reduce HBM bandwidth or increase compute utilization?** Both — (a) FP8 activations ($Q, K, V$) are half the size of BF16, reducing HBM read traffic by up to $50\%$ for tile loads; (b) FP8 WGMMA achieves $2\times$ higher TFLOPS, pushing the kernel from memory-bound toward compute-bound.

## Related Work

- **FlashAttention-3 FP8 (Dao et al., 2024)**: Demonstrated FP8 softmax attention on Hopper with E4M3 forward / E5M2 backward. Key technique: block quantization with per-tensor or per-block scaling factors to maintain accuracy. **Our approach**: Adapts this to chunkwise linear attention, which has a simpler precision decomposition (no softmax normalization interleaved with FP8 matmuls).
- **FP8-LM (Peng et al., 2023)**: FP8 training framework for Transformers. Uses delayed scaling (compute scaling factors from previous iteration's statistics). **Our approach**: Applies delayed scaling specifically to the chunkwise linear RNN computation, where the gating provides an additional accuracy buffer (gates attenuate error propagation across chunks).
- **MOSS (Li et al., 2025)**: Microscaling for FP8 training with two-level scaling (global scale + local power-of-two scales). Achieves FP8 training comparable to BF16. **Our approach**: Applies microscaling within the chunkwise kernel, where the tile structure naturally provides the "local" scale granularity.
- **Transformer Engine (NVIDIA)**: Library for mixed-precision Transformer training. Supports FP8 for linear layers (GEMM). **Does not support** chunkwise linear attention — the fused TFLA kernel is outside TE's scope.
- **GLA secondary chunking (Yang et al., 2024)**: The secondary chunking separates log-space intra-sub-chunk computation from standard inter-sub-chunk matmuls. **Our approach**: Exploits this separation as the FP8/FP32 boundary — inter-sub-chunk matmuls in FP8, intra-sub-chunk in FP32.

**Gap**: No existing work applies FP8 precision to the chunkwise linear RNN training kernel. FP8 has been applied to Transformer attention (FlashAttention-3), linear layers (Transformer Engine), and end-to-end training (FP8-LM, MOSS), but not to the specific computation pattern of chunkwise linear attention with gating.

## Mathematical Formulation

### FP8 Precision Split

**GLA chunkwise computation per chunk, with secondary sub-chunks of size $c$:**

For inter-sub-chunk blocks $(i, j)$ with $i > j$ (the dominant FLOPs):

$$
P_{[i][j]} = \text{dequant}\left(\text{FP8-WGMMA}\left(\text{quant}(\tilde{Q}_{[i]}), \text{quant}(\tilde{K}_{[j]}^\top)\right)\right) \odot M_{[i][j]}
$$

where:
- $\text{quant}(X) = \text{clamp}(\text{round}(X / s_X), -448, 448)$ converts BF16 → FP8 E4M3
- $s_X$ is the per-tile scaling factor: $s_X = \max(|X|) / 448$ (delayed from previous chunk/iteration)
- $\text{dequant}$ rescales: $\hat{P} = \text{FP32\_accum} \times s_Q \times s_K$
- $M_{[i][j]}$ is the causal gating mask (BF16, applied element-wise after dequantization)
- The WGMMA instruction computes in FP8 with FP32 accumulation — **no precision loss in the accumulation**

For intra-sub-chunk diagonal blocks $(i, i)$:

$$
P_{[i][i],mn} = \sum_{k=1}^{d_k} Q_{[i],mk} K_{[i],nk} \exp(\log B_{[i],mk} - \log B_{[i],nk}), \quad m \geq n
$$

This uses **FP32** throughout — the exp and log operations require full precision. Only $C/c$ such blocks exist per chunk.

**The $SV$ product (applying attention weights to values):**

$$
O_{[i]}^{\text{intra}} = \sum_j P_{[i][j]}^{\text{FP8}} V_{[j]} = \text{dequant}\left(\text{FP8-WGMMA}\left(\text{quant}(P_{[i]}), \text{quant}(V)\right)\right)
$$

Again, the matmul itself is FP8, with FP32 accumulation and BF16 dequantized output.

**Inter-chunk state accumulation:**

$$
S_{k+1} = \gamma_{k+1} S_k + (K_{[k+1]} \odot \Gamma_{[k+1]})^\top V_{[k+1]}
$$

This uses **BF16** for the $K^\top V$ product (small: $d_k \times d_v$ per chunk) and **FP32** for the state $S$ accumulation (to prevent error drift across the sequence).

### Scaling Factor Strategy

**Per-tile delayed scaling (following FP8-LM):**

For each WGMMA tile of shape $m_{\text{tile}} \times k_{\text{tile}}$:

$$
s_Q^{(t)} = \text{amax}(Q_{\text{tile}}^{(t-1)}) / 448, \quad s_K^{(t)} = \text{amax}(K_{\text{tile}}^{(t-1)}) / 448
$$

The scaling factors are computed from the **previous chunk's** tile statistics (delayed by one chunk iteration). Within a chunk, all sub-chunk tiles of $Q$ share one scaling factor, and all tiles of $K$ share one scaling factor — this amortizes the amax computation.

**Why delayed scaling works well for linear RNNs:**

The gating factor $\gamma_t \in (0, 1)$ naturally attenuates the contribution of distant tokens. Even if FP8 quantization introduces small errors in $P_{[i][j]}$, the gating mask $M_{[i][j]}$ exponentially downweights contributions from far-away sub-chunks ($j \ll i$). This means FP8 errors in the most-attenuated (and thus most error-tolerant) sub-chunk interactions are suppressed by the gate.

### FP8 Data Format

| Tensor | Forward precision | Backward precision | Justification |
|--------|------------------|-------------------|---------------|
| $Q, K$ tiles | E4M3 | E5M2 | Higher range for gradients |
| $V$ tiles | E4M3 | E5M2 | Standard FP8 split |
| $P$ (attention weights) | FP32 accum → E4M3 | FP32 accum → E5M2 | Accumulated then requantized |
| $D$ (gating mask) | BF16 | BF16 | Element-wise, no matmul |
| $\log B$ (log-space gates) | FP32 | FP32 | exp/log require precision |
| $S$ (inter-chunk state) | FP32 | FP32 | Accumulates across sequence |
| Scaling factors $s_Q, s_K, s_V$ | FP32 | FP32 | Metadata |

### Key Variables

- $Q, K \in \mathbb{R}^{T \times d_k}$ — queries and keys
- $V \in \mathbb{R}^{T \times d_v}$ — values
- $P \in \mathbb{R}^{C \times C}$ — intra-chunk attention matrix
- $S \in \mathbb{R}^{d_k \times d_v}$ — inter-chunk hidden state
- $C$ — primary chunk size (128–512)
- $c$ — secondary sub-chunk size (16–32)
- $s_Q, s_K, s_V$ — per-tile FP8 scaling factors
- E4M3 — 4-bit exponent, 3-bit mantissa (range $[-448, 448]$, precision ~$0.0625$)
- E5M2 — 5-bit exponent, 2-bit mantissa (range $[-57344, 57344]$, precision ~$0.25$)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / mLSTM / Gated DeltaNet |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 16$ |
| Head dim | $d_k = d_v = 128$ |
| Primary chunk | $C = 128$ |
| Secondary sub-chunk | $c = 16$ |
| GPU | H100 SXM (SM90a, FP8 WGMMA) |
| FP8 format | E4M3 forward, E5M2 backward |
| Scaling | Per-tile delayed scaling |

### Baseline

1. **BF16 TFLA (current SOTA)**: GLA chunkwise kernel with secondary chunking, all matmuls in BF16 WGMMA. Throughput: ~45 Ktok/s at 1.3B on H100.
2. **BF16 FlashAttention-3**: Softmax attention baseline for throughput comparison.
3. **FP8 FlashAttention-3**: FP8 softmax attention on H100 — the ceiling for FP8 attention kernels.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Kernel throughput | $\geq 1.4\times$ BF16 TFLA | TFLOPS for chunkwise kernel on H100 |
| End-to-end throughput | $\geq 1.3\times$ BF16 training | Tokens/sec for 1.3B pretraining |
| Quality | PPL within $0.5\%$ of BF16 | Perplexity on validation set after 15B tokens |
| HBM traffic | $\leq 0.65\times$ BF16 | Bytes read/written via ncu |
| FP8 utilization | $> 60\%$ of peak FP8 TFLOPS | ncu tensor core utilization |

### Estimated Compute

**MVE (kernel microbenchmark)**: $< 30$ minutes on single H100
- Implement FP8 quantize/dequant within existing Triton TFLA kernel
- Benchmark throughput vs. BF16 baseline on synthetic data
- Verify numerical accuracy against FP32 reference

**Phase 1 (quality validation)**: ~64 GPU-hours on H100 ($\sim \$256$)
- Train GLA-370M with FP8 chunkwise kernel on 10B tokens
- Compare perplexity curve with BF16 baseline
- Ablate: FP8 $QK^\top$ only, FP8 $SV$ only, FP8 both

**Phase 2 (full scale)**: ~256 GPU-hours on H100 ($\sim \$1024$)
- Train GLA-1.3B with FP8 on 15B tokens
- Full comparison with BF16 TFLA and FP8 FlashAttention-3

## Expected Outcome

**If hypothesis is correct:**

- Intra-chunk kernel throughput: $1.5$–$1.8\times$ BF16 TFLA (FP8 WGMMA $2\times$ raw throughput, minus quantization overhead and FP32 intra-sub-chunk blocks)
- End-to-end training throughput: $1.3$–$1.5\times$ (other operations — projections, normalization, optimizer — also benefit from FP8 via Transformer Engine)
- Perplexity within $0.3$–$0.5\%$ of BF16 baseline (gating attenuates FP8 errors)
- HBM traffic reduced by $35$–$50\%$ (FP8 tensors are half the size of BF16)

**Breakdown of speedup sources:**

| Source | Contribution |
|--------|-------------|
| $2\times$ WGMMA throughput for inter-sub-chunk matmuls | $+40$–$60\%$ |
| $0.5\times$ HBM traffic for $Q, K, V$ tiles | $+15$–$25\%$ |
| Quantization/dequantization overhead | $-10$–$15\%$ |
| FP32 intra-sub-chunk blocks unchanged | $-5$–$10\%$ |
| **Net** | **$+40$–$60\%$** |

**If hypothesis is wrong:**

- **Scenario A**: FP8 quantization error in $QK^\top$ corrupts the attention pattern, causing significant perplexity degradation ($>1\%$). **What we learn**: Linear attention without softmax normalization may be more sensitive to quantization than softmax attention (which has a normalizing effect). **Mitigation**: Use block-level microscaling (MOSS-style) with finer granularity (per-row or per-$c$-block scaling factors) to reduce quantization error.
- **Scenario B**: The FP8 quantization/dequantization overhead is larger than expected, eating into the $2\times$ WGMMA speedup. **What we learn**: The current Triton FP8 support adds too many instructions for quantize/dequant. **Mitigation**: Use CUTLASS 3.x with native FP8 WGMMA and fused quantization in the prologue/epilogue, avoiding separate quantize kernels.
- **Scenario C**: The inter-chunk state $S$ drifts due to accumulated FP8 errors in $K^\top V$ across many chunks. **What we learn**: Long-range error accumulation in the recurrence is a problem. **Mitigation**: Keep $K^\top V$ in BF16 (small matmul, not the bottleneck) and only use FP8 for the large intra-chunk matmuls.

## Minimum Viable Experiment

### Setup
- **Kernel**: FP8 variant of GLA intra-chunk attention for a single chunk
- **Shapes**: $C = 128$, $c = 16$, $d_k = d_v = 128$, $H = 1$, batch = 16
- **Implementation**: Triton kernel with `tl.dot` using FP8 input types
- **Baseline**: Same kernel with BF16 inputs
- **Compute**: Single H100, $< 10$ minutes

### Success Criteria
- FP8 kernel achieves $\geq 1.3\times$ throughput over BF16 kernel for the intra-chunk forward pass
- Maximum absolute error vs. FP32 reference: $< 5 \times 10^{-2}$ (acceptable for BF16-level training)
- Relative error in output $O$: $\|O_{\text{fp8}} - O_{\text{fp32}}\|_2 / \|O_{\text{fp32}}\|_2 < 10^{-2}$
- FP8 WGMMA utilization $> 40\%$ of peak (verified via ncu)

### Failure Criteria
- **Kill if**: FP8 kernel is $< 1.1\times$ faster than BF16 — the quantization overhead negates the WGMMA speedup (would indicate Triton's FP8 codegen is too inefficient; switch to CUTLASS)
- **Kill if**: Relative error $> 0.1$ — FP8 destroys the attention pattern even for a single chunk (fundamental precision issue, not fixable by scaling)
- **Kill if**: ncu shows $< 20\%$ FP8 utilization — the kernel is memory-bound anyway and FP8 compute speedup doesn't help

### Why This Test Is Sufficient
- The intra-chunk matmul is the dominant FLOP component ($>70\%$). If FP8 speeds up this kernel, the full training benefits proportionally
- Numerical accuracy for a single chunk validates the precision decomposition. Multi-chunk error accumulation can be tested separately via the state scan (which remains in FP32)
- Triton's FP8 support on H100 is the fastest path to a working prototype. If Triton is too slow, the MVE result tells us whether to invest in a CUTLASS implementation

## Memory Access Pattern Analysis

**Coalesced access:** $Q, K, V$ tiles are loaded from HBM in contiguous blocks. FP8 tensors have half the byte width of BF16, meaning each 128-byte cache line loads $2\times$ more elements — improving effective bandwidth utilization.

**Cache-friendly:** Inner tiles ($c \times d_k$) at FP8 are $c \times d_k \times 1 = 2$ KB (vs. 4 KB at BF16). This means $2\times$ more tiles fit in the 228 KB SMEM budget on H100, potentially allowing larger tiles or deeper pipelining.

**Arithmetic intensity:** For BF16 WGMMA on $c \times d_k \times c$ tiles: $2 c^2 d_k$ FLOPs / $(2 c d_k + 2 c d_k) \times 2$ bytes = $c / 4$ FLOPs/byte. For FP8 WGMMA: same FLOPs but inputs are 1 byte each: $2 c^2 d_k$ / $(c d_k + c d_k) \times 1$ = $c / 1$ FLOPs/byte. FP8 has $4\times$ higher arithmetic intensity — pushing the kernel firmly into the compute-bound regime where the $2\times$ WGMMA speedup translates directly to wall-clock improvement.

**HBM bandwidth:** FP8 $Q, K, V$ from HBM are half the size of BF16. For $B = 8, T = 4096, d = 2048$: BF16 activations = 192 MB; FP8 activations = 96 MB per layer. This saves 2.3 GB across 24 layers — enabling larger batch sizes or longer sequences within the same memory budget.

## Parallelism Analysis

**SM saturation:** Identical to BF16 TFLA — same number of CTAs, same tile decomposition. FP8 does not change the parallelism structure.

**No warp divergence:** FP8 quantization is element-wise (uniform across warps). The secondary chunking FP8/FP32 split is at sub-chunk granularity — all warps within a sub-chunk tile execute the same precision path.

**Tensor core mapping:** FP8 WGMMA on H100 uses the same `wgmma.mma_async` instruction family with different type specifiers. The tile shapes are compatible: FP8 WGMMA supports $M=16, N=8, K=32$ (vs. $K=16$ for BF16), meaning $2\times$ more $K$-dimension reduction per instruction — this is where the $2\times$ throughput comes from.

**Sequential bottleneck:** Inter-chunk state scan remains in FP32 on scalar ALU. Unchanged from BF16. This is $<5\%$ of total compute.

## Theoretical Analysis

Complexity comparison per layer forward pass:

| Operation | BF16 TFLA | FP8 TFLA (proposed) |
|-----------|----------|---------------------|
| Intra-chunk FLOPs | $O(C^2 d_k \cdot T/C) = O(TCd_k)$ | Same FLOPs, $2\times$ throughput |
| Intra-chunk HBM | $O(T \cdot d) \times 2$ bytes | $O(T \cdot d) \times 1$ byte |
| Inter-chunk FLOPs | $O(T/C \cdot d_k \cdot d_v)$ | Same (BF16) |
| Quantization overhead | $0$ | $O(T \cdot d)$ (element-wise) |
| Scaling factor compute | $0$ | $O(T/C \cdot d)$ (per-tile amax) |

**Effective speedup model:**

$$
\text{Speedup} = \frac{T_{\text{bf16}}}{T_{\text{fp8}}} = \frac{T_{\text{inter-sub}} + T_{\text{intra-sub}} + T_{\text{scan}}}{T_{\text{inter-sub}} / 2 + T_{\text{quant}} + T_{\text{intra-sub}} + T_{\text{scan}}}
$$

With $T_{\text{inter-sub}} : T_{\text{intra-sub}} : T_{\text{scan}} \approx 0.77 : 0.13 : 0.10$ (for $C=128, c=16$):

$$
\text{Speedup} \approx \frac{0.77 + 0.13 + 0.10}{0.385 + 0.05 + 0.13 + 0.10} = \frac{1.0}{0.665} \approx 1.50\times
$$

(assuming quantization overhead of ~5% of total time)

## Risks & Limitations

1. **FP8 E4M3 dynamic range**: E4M3 has range $[-448, 448]$. For attention weights $P_{ij}$ that can be large (especially without softmax normalization), overflow is possible. **Mitigation**: Per-tile scaling factors ensure values fit within range. Additionally, the gating mask $D$ attenuates large values.

2. **Error accumulation in backward pass**: FP8 E5M2 for gradients has only 2 mantissa bits (precision ~0.25). For the backward through the intra-chunk matmul, gradient errors may accumulate. **Mitigation**: The $SV$ backward computes $dS = P^\top dO$ and $dV = S^\top dO$ — both are matmuls where FP32 accumulation preserves accuracy regardless of input precision.

3. **Triton FP8 maturity**: Triton's FP8 support on Hopper is functional but not as optimized as CUTLASS. The MVE may show lower-than-expected speedup due to Triton codegen inefficiency. **Mitigation**: If Triton FP8 speedup is $< 1.2\times$, switch to CUTLASS 3.x with native FP8 WGMMA for the production kernel.

4. **Scaling factor latency**: Per-tile scaling factors require computing `amax` over each tile. With delayed scaling (using previous iteration's amax), this adds no latency to the critical path but may cause accuracy issues if activation statistics change rapidly. **Mitigation**: Use block-level scaling (compute amax per sub-chunk block) rather than global scaling.

5. **Only benefits H100/H200/Blackwell**: FP8 WGMMA is only available on SM90a (Hopper) and newer. A100 and consumer GPUs do not benefit. **Mitigation**: The kernel falls back to BF16 on older hardware.

6. **Interaction with gradient checkpointing**: If activations are checkpointed and recomputed in the backward pass, the FP8 recomputation must use the same scaling factors as the original forward pass to maintain numerical consistency. **Mitigation**: Save scaling factors alongside checkpoint data (small overhead: 1 float per tile).

## Follow-up Experiments

1. **FP8 input projections**: Extend FP8 beyond the chunkwise kernel to include the $W_Q, W_K, W_V$ projections (via Transformer Engine). The full FP8 path: input → FP8 GEMM → FP8 chunkwise kernel → FP8 GEMM → output.

2. **FP4 on Blackwell**: NVIDIA Blackwell supports FP4 tensor cores ($4\times$ throughput). The same precision decomposition applies: FP4 for intra-chunk matmuls, FP8/BF16 for state scan. Test whether FP4 maintains training quality.

3. **Microscaling (MX format)**: Instead of per-tile scaling factors, use MXFP8 (microscaling FP8) with shared exponents per 32 elements. This reduces quantization error while maintaining FP8 throughput, and is supported natively on Blackwell.

4. **FP8 + warp specialization (proposal 039)**: Combine FP8 matmuls with warp-specialized pipelining. Producer warps handle FP8 quantization + TMA loads; consumer warps execute FP8 WGMMA. This maximizes both precision throughput and pipeline utilization.

5. **FP8 state scan (MatMulScan)**: If the inter-chunk scan uses MatMulScan (proposal 044), the scan's batched matmuls against constant $L_s$ matrices could also use FP8 — since $L_s$ is a fixed constant with known range, scaling is trivial. This would FP8-ify $>95\%$ of all FLOPs in the linear RNN layer.

6. **Quality-throughput Pareto frontier**: Sweep over precision configurations: (a) all BF16, (b) FP8 $QK^\top$ only, (c) FP8 both matmuls, (d) FP8 + FP8 state scan, (e) FP8 + FP8 projections. Plot perplexity vs. throughput to find the optimal operating point.

## Human Review

(To be filled by reviewer)

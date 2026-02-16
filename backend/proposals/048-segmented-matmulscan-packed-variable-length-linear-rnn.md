---
status: ongoing
priority: high
created: 2026-02-15
based_on: segmented-matmul-scan-scd (172), matmulscan-tcu-parallel-scan (167), chunkwise-parallel-scan (026), segmented-scan (107), gla-secondary-chunking-log-space-gating (177), tfla-two-level-tiled-chunkwise-parallelism (158), io-aware-tiling (066)
experiment_number: 048
experiment_log: experiment-log-048.md
---

# Segmented MatMulScan for Packed Variable-Length Chunkwise Linear RNN Training

## Hypothesis

Applying **Segmented MatMulScan** (the SCD/SSCR algorithms from trick 172) to the **inter-chunk state propagation** in packed variable-length chunkwise linear RNN training will achieve **$1.5$–$3.0\times$ training throughput** over the current practice of padding sequences to maximum length, while maintaining **bit-identical outputs** to unpacked single-sequence processing. The key insight: in packed training, multiple variable-length sequences are concatenated into a single long "super-sequence" and chunked together, but the inter-chunk scan must **not propagate state across sequence boundaries**. Segmented MatMulScan performs a speculative unsegmented scan via tensor-core matmuls (MMU), then corrects the mis-speculated cross-boundary accumulations with lightweight vector operations (VCU) — achieving near-unsegmented-scan throughput with correct segmented semantics.

## Background

### The variable-length sequence problem in linear RNN pretraining

Real pretraining data consists of documents of **highly variable lengths**. A typical pretraining batch might contain:

| Sequence | Length | Padded to $T_{\max}$ | Waste |
|----------|--------|-----------------------|-------|
| Doc 1 | 4096 | 8192 | 50% |
| Doc 2 | 1523 | 8192 | 81% |
| Doc 3 | 8192 | 8192 | 0% |
| Doc 4 | 347 | 8192 | 96% |
| **Average** | **3540** | **8192** | **57%** |

Padding wastes $> 50\%$ of compute on average. For softmax attention, **FlashAttention-2** supports variable-length sequences natively via the `cu_seqlens` interface, and **sequence packing** concatenates multiple sequences into a single tensor with boundary masks.

For **linear RNNs** (GLA, Mamba, DeltaNet), the situation is worse:

1. **Mamba/Mamba-2**: PackMamba (He et al., 2024) modifies the selective scan to zero out cross-sequence state propagation. But this uses scalar operations, not tensor cores.

2. **GLA/TFLA**: The `flash-linear-attention` library supports variable-length via `cu_seqlens` but uses either (a) padding to chunk boundaries or (b) sequential per-sequence processing. Neither is optimal.

3. **The inter-chunk scan is the bottleneck**: In chunkwise training, the intra-chunk computation is parallel and handles packing naturally (just mask the attention within chunks that span boundaries). But the **inter-chunk scan** must restart at sequence boundaries — this is exactly a **segmented prefix scan**.

### Current approaches and their limitations

**Approach 1: Padding to max length**
- Pad all sequences to $T_{\max}$, set decay $\alpha_t = 0$ at padding positions
- Wastes $50$–$80\%$ compute on padding tokens
- Simple but extremely wasteful

**Approach 2: Per-sequence processing (sequential)**
- Process each sequence independently in a loop
- No wasted compute but very low GPU utilization (small batch sizes per sequence)
- Throughput limited by smallest sequence in batch

**Approach 3: PackMamba-style packing**
- Pack sequences end-to-end into super-sequence
- Modify selective scan to zero state at boundaries
- Better than approaches 1-2 but still uses scalar scan (no tensor cores for the segmented part)
- Only implemented for Mamba's specific scan structure

**Approach 4: Flag-based segmented scan (trick 107)**
- Use (flag, value) pair transformation: extend scan operator to carry a boundary flag
- The flag resets accumulation at segment boundaries
- Standard approach but runs on scalar ALU — does not use tensor cores
- Each step is: `if flag: reset; else: accumulate` — warp divergent

### Why Segmented MatMulScan is the right solution

Segmented MatMulScan (Sobczyk et al., 2025) provides an elegant decomposition:

1. **Speculative unsegmented scan** via MatMulScan: Compute a regular prefix scan over the entire packed super-sequence, ignoring boundaries. This runs entirely on tensor cores (MMU) as batched matmuls against constant $L_s$ matrices.

2. **Compress**: Gather the scan values at segment endpoints. This is a gather operation (VCU).

3. **Correct** (Differentiation/Revert): Subtract the over-accumulated cross-boundary contributions. This is elementwise subtraction (VCU).

The key property: $\geq 90\%$ of the work is in step 1 (unsegmented MatMulScan on tensor cores), and steps 2-3 are lightweight corrections. This gives near-unsegmented-scan throughput with correct segmented semantics.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than padding?** Yes — padding wastes $> 50\%$ of all compute. Even with the overhead of the correction steps, segmented MatMulScan on packed data uses $< 110\%$ of the FLOPs needed for a single unsegmented scan, while padding uses $200$–$400\%$ of the FLOPs.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — the core is MatMulScan (proposal 044's kernel) with two additional passes: (a) gather segment endpoints into a buffer, (b) scatter corrections back. Both are standard gather/scatter operations.

3. **Does it reduce HBM bandwidth?** Yes — packed data is denser (no padding), so the same HBM bandwidth processes more useful tokens. The correction passes add $O(S \cdot n \cdot d_v)$ HBM traffic where $S$ is the number of segments, which is negligible ($S \ll G$).

### Memory access pattern analysis

**MatMulScan (speculative, step 1):**
- Identical to proposal 044: batched matmuls against constant $L_s$, $B_s$ matrices
- Coalesced: data is packed contiguously (no padding gaps)
- Tensor core: $L_s$ is $s \times s$, batched over all scan lanes $P = n \times d_v$
- Arithmetic intensity: $O(s)$ FLOPs/byte per level — compute-bound for $s \geq 4$

**Compress (step 2):**
- Gather: read $S$ elements from positions marked by segment endpoint flags
- Access pattern: **irregular** — segment boundaries are at arbitrary positions
- Coalescing: Poor within a warp if boundaries are irregularly spaced
- Mitigation: Pre-sort segment boundaries into contiguous buffer; or use warp-level ballot to skip warps with no boundaries (common case: most warps have zero boundaries)
- Volume: $S \times n \times d_v$ elements where $S \ll G$ (typically $S / G < 5\%$)

**Correct (step 3):**
- Scatter: subtract corrections at positions within each segment
- Can be implemented as: broadcast segment correction to all positions within segment via gather of segment ID → correction value
- Access pattern: broadcast-style (each segment ID maps to one correction value)
- Coalescing: Good if segments are aligned to warp boundaries; mediocre otherwise
- Mitigation: Pad segments to warp-aligned boundaries (< 3% overhead for average segment $> 100$ tokens)

### Parallelism analysis

**SM saturation:**
- The packed super-sequence has $T_{\text{total}} = \sum_i T_i$ tokens (no padding)
- With $G = T_{\text{total}} / C$ chunks: for $T_{\text{total}} = 32\text{K}$ and $C = 64$: $G = 512$ chunks
- With $H = 32$ heads and $B = 1$: $512 \times 32 = 16384$ scan lanes — massively parallel
- Steps 2-3 have $S$ segment boundaries to process — typically $S = 32$–$256$ (one per document in the batch)

**Warp divergence:**
- Step 1 (MatMulScan): None — uniform matmuls for all scan lanes
- Step 2 (Compress): Potential divergence at boundary detection, but boundaries are sparse ($< 1\%$ of positions have a flag) so the impact is negligible
- Step 3 (Correct): Broadcast-style — uniform within each segment

**Tensor core mapping:**
- Step 1: Batched GEMM against $L_s$ — direct tensor core use (WGMMA/MMA)
- Steps 2-3: Scalar/vector operations — no tensor core use, but these are $< 10\%$ of total work

## Related Work

- **PackMamba** (He et al., LNCS 2024): Packs variable-length sequences for Mamba training, modifying selective scan and convolution operators to prevent cross-sequence access. Achieves $3.06\times$ speedup over padding on A100. **Our approach**: Applies to GLA/TFLA (not just Mamba), uses tensor-core-friendly segmented scan (MatMulScan) instead of scalar modifications, and is architecture-agnostic (works with any chunkwise linear RNN).
- **FlashAttention-2 variable-length** (Dao, 2023): Uses `cu_seqlens` to handle variable-length sequences in softmax attention. The kernel skips computation for padding tokens. **Our approach**: Analogous variable-length support but for the linear RNN inter-chunk scan, which has a different computational pattern (prefix scan vs. block-sparse attention).
- **Segmented MatMulScan** (Sobczyk et al., arXiv:2506.23906, 2025): Proposed the SCD/SSCR algorithms for generic segmented prefix sums on accelerators with MMU+VCU. Validated on Ascend 910B for sparse matrix-vector products. **Our approach**: Applies SCD/SSCR specifically to the inter-chunk state propagation in chunkwise linear RNNs, with GLA/TFLA-specific adaptations (matrix-valued scan elements, log-space gating).
- **Proposal 044 (MatMulScan inter-chunk scan)**: Applies unsegmented MatMulScan to the inter-chunk scan. **Our approach**: Extends proposal 044 to handle packed variable-length sequences via segmented MatMulScan. Proposal 044 is a prerequisite — our proposal adds the segmented correction layer on top.
- **Mamba-2 SSD packing**: Sets $\alpha_t = 0$ at sequence boundaries to zero out state propagation. This is a special case of segmented scan where the "correction" is applied within the scan operator itself. **Our approach**: Separates the speculative scan from the correction, enabling tensor core use for the main scan.

**Gap**: No existing work applies tensor-core-friendly segmented prefix scan to the inter-chunk state propagation of chunkwise linear RNNs for packed variable-length training. PackMamba handles Mamba but not GLA/TFLA, and uses scalar modifications rather than the MMU+VCU decomposition.

## Mathematical Formulation

### Packed Variable-Length Chunkwise GLA

Given $S$ sequences of lengths $\{T_1, \ldots, T_S\}$ packed into a super-sequence of total length $T_{\text{total}} = \sum_i T_i$:

$$
\mathbf{x}_{\text{packed}} = [x_1^{(1)}, \ldots, x_{T_1}^{(1)}, x_1^{(2)}, \ldots, x_{T_2}^{(2)}, \ldots, x_1^{(S)}, \ldots, x_{T_S}^{(S)}]
$$

The packed super-sequence is divided into $G = \lceil T_{\text{total}} / C \rceil$ chunks. Most chunks belong entirely to one sequence; some chunks **span sequence boundaries**.

**Segment boundary flags:**

$$
f_t = \begin{cases} 1 & \text{if token } t \text{ is the first token of a new sequence} \\ 0 & \text{otherwise} \end{cases}
$$

**Chunk-level boundary flags:**

$$
F_j = \bigvee_{t \in \text{chunk}_j} f_t = \begin{cases} 1 & \text{if chunk } j \text{ contains a sequence boundary} \\ 0 & \text{otherwise} \end{cases}
$$

### Standard Chunkwise Inter-Chunk Scan (without packing)

$$
h_j = A_j^{(C)} h_{j-1} + h_j^{\text{local}}, \quad j = 1, \ldots, G
$$

where $A_j^{(C)} = \prod_{t \in \text{chunk}_j} \text{diag}(\alpha_t)$ and $h_j^{\text{local}} = \sum_{t \in \text{chunk}_j} \gamma_{j,t} K_t^\top V_t$.

### Segmented Inter-Chunk Scan (with packing, correct behavior)

At sequence boundaries, the state must reset:

$$
h_j = \begin{cases} A_j^{(C)} h_{j-1} + h_j^{\text{local}} & \text{if } F_j = 0 \text{ (no boundary in chunk } j\text{)} \\ h_j^{\text{local, post}} & \text{if } F_j = 1 \text{ (boundary in chunk } j\text{)} \end{cases}
$$

where $h_j^{\text{local, post}}$ is the local state accumulated from only the tokens **after** the last boundary within chunk $j$.

More precisely, for a boundary at position $b$ within chunk $j$:

$$
A_j^{(C, \text{post})} = \prod_{t=b+1}^{(j+1)C-1} \text{diag}(\alpha_t), \quad h_j^{\text{local, post}} = \sum_{t=b}^{(j+1)C-1} \gamma_{j,t}^{\text{post}} K_t^\top V_t
$$

This means the inter-chunk scan is a **segmented prefix scan** with segments defined by consecutive runs of chunks without boundaries.

### Segmented MatMulScan (Proposed)

**Step 1 — Speculative unsegmented scan (MatMulScan on MMU):**

Compute the unsegmented prefix scan over all $G$ chunks, treating the packed super-sequence as if it were a single sequence:

$$
\hat{h}_j = \hat{A}_{1:j} h_j^{\text{local}} + \hat{A}_{1:j} \sum_{i < j} \hat{A}_{i+1:j}^{-1} h_i^{\text{local}}
$$

For **diagonal** state transitions ($A_j = \text{diag}(\alpha_j)$), this decomposes into $P = n \times d_v$ independent scalar prefix scans, computed via MatMulScan:

$$
\hat{\mathbf{x}} = \text{MatMulScan}(\mathbf{x}, s) \quad \text{where } \mathbf{x}_j = h_j^{\text{local}} / A_{1:j}
$$

This uses batched GEMM against constant $L_s \in \mathbb{R}^{s \times s}$ — all on tensor cores.

**Step 2 — Compress (gather segment endpoints on VCU):**

Identify chunk-level segment boundaries $\{j : F_j = 1\}$ and gather the speculative scan values at the **end of each segment** (the chunk before each boundary):

$$
\mathbf{w}_k = \hat{h}_{j_k - 1} \quad \text{for each segment boundary } j_k
$$

This is a gather of $S$ elements from positions indexed by $\{j_1 - 1, j_2 - 1, \ldots, j_S - 1\}$.

**Step 3 — Correct/Revert (vector differentiation on VCU):**

The speculative scan over-accumulated across boundaries. For each position $j$ in segment $k$ (between boundaries $j_{k-1}$ and $j_k$), the correction is:

$$
h_j^{\text{correct}} = \hat{h}_j - A_{j_{k-1}+1:j} \cdot \mathbf{w}_{k-1}
$$

where $A_{j_{k-1}+1:j} = \prod_{l=j_{k-1}+1}^{j} \alpha_l$ is the cumulative decay from the segment start to position $j$.

Since $\hat{h}_j$ already includes $A_{1:j}$, the correction simplifies to:

$$
h_j = \hat{h}_j - \frac{A_{1:j}}{A_{1:j_{k-1}}} \cdot \hat{h}_{j_{k-1}}
$$

This is an elementwise multiply-subtract per position, broadcast from $S$ boundary corrections to $G$ chunk positions.

### Cost Analysis

| Step | Operation | Hardware | FLOPs | HBM Traffic |
|------|-----------|----------|-------|-------------|
| 1. MatMulScan | Batched GEMM × $O(\log_s G)$ levels | Tensor cores (MMU) | $O(G \cdot P \cdot s)$ | $O(G \cdot P)$ |
| 2. Compress | Gather $S$ endpoints | Scalar ALU (VCU) | $O(S \cdot P)$ | $O(S \cdot P)$ |
| 3. Correct | Broadcast + subtract | Scalar ALU (VCU) | $O(G \cdot P)$ | $O(G \cdot P)$ |
| **Total** | | | $O(G \cdot P \cdot s)$ | $O(G \cdot P)$ |

Overhead over unsegmented MatMulScan: steps 2+3 add $O(S \cdot P + G \cdot P)$ scalar ops, which is negligible compared to step 1's tensor-core compute.

### Key Variables

- $T_{\text{total}} = \sum_i T_i$ — total packed sequence length
- $G = T_{\text{total}} / C$ — number of chunks in packed super-sequence
- $S$ — number of sequences (= number of segment boundaries)
- $C$ — chunk size (64–256)
- $n$ — state dimension per head
- $d_v$ — value dimension per head
- $P = n \times d_v$ — independent scan lanes
- $s$ — MatMulScan radix (4 or 8)
- $F_j$ — chunk-level segment boundary flag
- $L_s, B_s \in \mathbb{R}^{s \times s}$ — constant MatMulScan matrices

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / mLSTM / Gated DeltaNet |
| Layers | $L = 24$ |
| Hidden dim | $d_{\text{model}} = 2048$ |
| Head dim | $d = 64$ |
| Heads | $H = 32$ |
| State dim | $n = 16$ (diagonal) |
| Value dim | $d_v = 64$ |
| Chunk size | $C = 64$ |
| Packing | Greedy bin-packing to $T_{\max} = 8192$ |
| MatMulScan radix | $s = 4$ |

### Baseline

1. **Padded GLA**: Pad all sequences to $T_{\max} = 8192$, standard chunkwise scan — current practice
2. **Per-sequence GLA**: Process each sequence independently (no packing) — low utilization baseline
3. **PackMamba-style**: Pack + modify scan to zero state at boundaries (scalar scan, no tensor cores)
4. **Flag-based segmented scan**: Standard segmented scan with (flag, value) pairs — scalar ALU, no tensor cores

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $\geq 1.5\times$ padded baseline | Useful tokens/sec (excluding padding) |
| Compute efficiency | $> 90\%$ of unsegmented scan speed | µs per segmented scan vs. unsegmented |
| Memory | $< 0.7\times$ padded baseline | Peak GPU memory |
| Quality | Bit-identical to per-sequence | Perplexity on validation set |
| Boundary correctness | $0$ cross-sequence leakage | Ablation: verify $h_j = 0$ after boundary |

### Estimated Compute

**MVE (microbenchmark)**: $< 10$ minutes on single GPU — compare segmented MatMulScan throughput vs. baselines
**Small-scale (370M model)**: $\sim 16$ GPU-hours on A100 ($\sim \$64$) — validate quality + throughput on pretraining
**Full-scale (1.3B model)**: $\sim 128$ GPU-hours on A100 ($\sim \$512$) — end-to-end pretraining comparison

## Expected Outcome

**If hypothesis is correct:**

- Segmented MatMulScan achieves $> 90\%$ of unsegmented MatMulScan throughput (steps 2-3 overhead is $< 10\%$)
- Packed training with segmented scan is $1.5$–$3.0\times$ faster than padding (depending on sequence length variance)
- For typical pretraining data (mean length $\sim 2000$, max $8192$): packing density $\sim 90\%$ vs. $\sim 50\%$ with padding
- Memory savings of $\sim 40\%$ (no padding tokens stored)
- Quality is bit-identical to per-sequence processing (the segmented scan produces the same outputs)

**Quantitative predictions:**

| Scenario | Padding waste | Segmented overhead | Net speedup |
|----------|--------------|-------------------|-------------|
| Uniform lengths (all $T_{\max}$) | 0% | 0% | $1.0\times$ (no benefit) |
| Moderate variance ($\sigma / \mu = 0.5$) | 35% | 5% | $1.46\times$ |
| High variance ($\sigma / \mu = 1.0$) | 55% | 8% | $2.04\times$ |
| Very high variance (web crawl) | 70% | 10% | $3.0\times$ |

**If hypothesis is wrong:**

- **Scenario A**: The Compress step (gather) dominates — irregular gather at segment boundaries causes severe cache thrashing. **What we learn**: Need to sort segments by boundary position or pad to cache-line boundaries. **Mitigation**: Pre-sort packed sequences so boundaries align with power-of-2 positions.
- **Scenario B**: Intra-chunk boundary handling is the bottleneck, not inter-chunk scan — chunks that span sequence boundaries need special handling (mask the attention, split the local state accumulation). This adds per-chunk overhead that overwhelms the scan savings. **What we learn**: Need to pack sequences to chunk-aligned boundaries ($T_i$ rounded up to multiple of $C$). **Mitigation**: Chunk-aligned packing with small padding (wastes $< C/2$ tokens per sequence on average).
- **Scenario C**: Quality degrades due to incorrect boundary handling — cross-sequence leakage in the speculative scan isn't fully corrected. **What we learn**: Implementation bug or numerical precision issue. **Fix**: Add assertion tests that verify $h_j = h_j^{\text{local, post}}$ at every boundary chunk.

## Minimum Viable Experiment

### Setup

- **Task**: Microbenchmark — compare segmented scan implementations for correctness and throughput on synthetic packed data
- **Input**:
  - Generate $S = 32$ sequences with lengths drawn from $\text{LogNormal}(\mu = 7, \sigma = 1)$ (mean $\sim 1100$, range $100$–$8000$)
  - Pack into super-sequence of $T_{\text{total}} \sim 35\text{K}$
  - Generate per-chunk diagonal transitions $\alpha_j \in (0, 1)^{16}$ and local states $h_j^{\text{local}} \in \mathbb{R}^{16 \times 64}$
  - Generate boundary flags $F_j$ from the packing
- **Implementations**:
  1. Sequential loop (reference, exact)
  2. Flag-based segmented scan (Triton, scalar)
  3. Segmented MatMulScan with $s = 4$ (Triton, tensor core)
  4. Unsegmented MatMulScan (upper bound on throughput)
- **Hardware**: Single A100 or H100
- **Compute**: $< 5$ minutes (kernel compilation + benchmarking)

### Success Criteria

- Segmented MatMulScan achieves $\geq 85\%$ of unsegmented MatMulScan throughput for $G = 512$ chunks
- Segmented MatMulScan achieves $\geq 1.3\times$ throughput over flag-based segmented scan
- Numerical accuracy: max absolute error $< 10^{-3}$ (bf16) vs. sequential reference
- All boundary chunks produce correct segmented output (verified against sequential reference)

### Failure Criteria

- **Kill if**: Segmented MatMulScan is slower than flag-based segmented scan — the overhead of Compress + Correct exceeds the tensor-core benefit of the speculative scan
- **Kill if**: Numerical errors $> 10^{-1}$ at boundary chunks — the correction step fails to properly subtract cross-boundary accumulations
- **Kill if**: Segmented MatMulScan achieves $< 50\%$ of unsegmented throughput — the correction overhead is too large

### Why This Test Is Sufficient

- The microbenchmark directly measures the core hypothesis: segmented MatMulScan throughput relative to unsegmented
- The synthetic data has realistic sequence length variance (LogNormal distribution matches web crawl data)
- Numerical correctness at bf16 is sufficient for pretraining
- No model training needed — the scan correctness and throughput are the fundamental questions
- If the kernel is fast and correct on synthetic data, it will be fast and correct within a full model

## Theoretical Analysis

### Throughput Model

Let $\tau_{\text{TC}}$ be tensor core throughput (FLOPs/s) and $\tau_{\text{ALU}}$ be scalar ALU throughput. On H100:

$$
\tau_{\text{TC}} \approx 989 \text{ TFLOPS (bf16)}, \quad \tau_{\text{ALU}} \approx 67 \text{ TFLOPS (fp32)}
$$

**Unsegmented MatMulScan time:**

$$
T_{\text{unseg}} = \frac{G \cdot P \cdot s \cdot 2}{\tau_{\text{TC}}} \cdot \log_s G
$$

**Segmented correction time:**

$$
T_{\text{correct}} = \frac{S \cdot P + G \cdot P}{\tau_{\text{ALU}}}
$$

**Overhead ratio:**

$$
\rho = \frac{T_{\text{correct}}}{T_{\text{unseg}}} = \frac{(S + G) \cdot P / \tau_{\text{ALU}}}{G \cdot P \cdot 2s \cdot \log_s G / \tau_{\text{TC}}} = \frac{(1 + S/G) \cdot \tau_{\text{TC}}}{2s \log_s G \cdot \tau_{\text{ALU}}}
$$

For $G = 512$, $S = 32$, $s = 4$:

$$
\rho = \frac{(1 + 32/512) \cdot 989}{2 \times 4 \times 4.5 \times 67} = \frac{1050}{2412} \approx 0.44
$$

So the correction adds $\sim 44\%$ to the scan time. But since the scan itself is only $\sim 10\%$ of total training time, the net overhead is $\sim 4\%$, while the packing savings is $\sim 50\%$.

### Complexity Comparison

| Approach | Useful FLOPs | Wasted FLOPs | TC Utilization | Net throughput |
|----------|-------------|-------------|----------------|---------------|
| Padded (baseline) | $F$ | $\sim F$ (50% waste) | High | $0.5 \times \text{peak}$ |
| Per-sequence | $F$ | $0$ | Low (small batch) | $0.2 \times \text{peak}$ |
| Flag-segmented scan | $F$ | $0$ | Low (scalar scan) | $0.7 \times \text{peak}$ |
| **Segmented MatMulScan** | $F$ | $0.05F$ (correction) | High (TC scan) | $\mathbf{0.9 \times \text{peak}}$ |

## Risks & Limitations

1. **Intra-chunk boundary handling**: Chunks that span sequence boundaries need special attention masking within the chunk. The intra-chunk attention $Q_j K_j^\top \odot M_j$ must be masked so tokens from different sequences don't attend to each other. **Mitigation**: Apply a block-diagonal mask within boundary chunks. This is already handled by FlashAttention-2's `cu_seqlens` interface; adapt the same approach for TFLA's intra-chunk computation.

2. **Chunk-aligned packing**: If sequence lengths are not multiples of $C$, the last chunk of each sequence has partial occupancy. **Mitigation**: Use "greedy packing" that minimizes partially-occupied chunks: sort sequences by length and pack short sequences together so boundaries align with chunk boundaries.

3. **Compress gather inefficiency**: The gather step (step 2) accesses irregular memory positions. On GPUs with large L2 cache (50 MB on H100), the $S \times P$ elements ($S = 32$, $P = 1024$: 32K elements = 64 KB in bf16) easily fit in L2. **Risk**: For very large $S$ (e.g., $S = 1024$ short documents), the gather becomes more expensive. **Mitigation**: Limit packing to $S \leq 128$ documents per super-sequence.

4. **Backward pass**: The backward pass through the segmented scan requires a reverse-direction segmented scan. Segmented MatMulScan applies identically in reverse — swap upsweep/downsweep and reverse the segment flags. **No additional complexity**.

5. **Non-causal attention interaction**: In hybrid models with softmax attention layers, the packed sequences also need boundary handling for the attention layers. **Mitigation**: Use FlashAttention-2's `cu_seqlens` for softmax layers and segmented MatMulScan for linear RNN layers — orthogonal solutions that compose.

6. **Implementation complexity**: Requires a custom Triton kernel for segmented MatMulScan. The unsegmented MatMulScan kernel (proposal 044) is a prerequisite. The segmented extension adds $\sim 30\%$ more kernel code for the Compress and Correct passes.

## Follow-up Experiments

1. **Combine with proposal 047 (LASP-2 + TFLA overlap)**: Use segmented MatMulScan for the local inter-chunk scan on each GPU, then LASP-2 AllGather for the inter-GPU scan. The inter-GPU scan does NOT need segmentation (each GPU processes disjoint sequences). This gives the best of both proposals.

2. **Adaptive packing strategy**: Learn the optimal packing based on sequence length distribution to minimize partially-occupied boundary chunks. Compare greedy packing, first-fit decreasing, and learned packing policies.

3. **Dense transition support**: Extend to DeltaNet/DeltaProduct with dense $n \times n$ state transitions. The segmented MatMulScan correction (step 3) requires per-segment matrix inverse or division, which is more complex for dense matrices. **Approach**: Use the Woodbury identity to compute the correction efficiently.

4. **Chunk-size adaptation**: Use different chunk sizes for different segments based on their length: large $C = 256$ for long sequences (better quality, lower scan overhead) and small $C = 64$ for short sequences (less waste). This requires a heterogeneous chunkwise kernel.

5. **Benchmark on real pretraining data**: Profile the packing density and sequence length variance on SlimPajama, The Pile, and RedPajama to estimate real-world speedup. Expected: web crawl data has the highest variance (and highest speedup).

## Human Review

(To be filled by reviewer)

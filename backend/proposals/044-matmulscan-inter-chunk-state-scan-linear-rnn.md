---
status: ongoing
priority: high
created: 2026-02-15
based_on: matmulscan-tcu-parallel-scan (167), chunkwise-parallel-scan (026), decoupled-lookback-single-pass-scan (166), recurrence-to-scan-reduction (101), batch-reduce-gemm (batch-reduce-gemm), io-aware-tiling (066), tfla-two-level-tiled-chunkwise-parallelism (158)
experiment_number: 044
experiment_log: experiment-log-044.md
---

# MatMulScan Tensor-Core Inter-Chunk State Propagation for Linear RNNs

## Hypothesis

Replacing the **scalar-scan-based** inter-chunk state propagation in chunkwise linear RNNs (GLA, DeltaNet, mLSTM) with **MatMulScan** — which reformulates the prefix scan as batched matrix multiplications against fixed constant matrices $L_s$ and $B_s$ — will achieve $1.2$–$1.8\times$ speedup for the inter-chunk aggregation phase by routing all scan operations through **tensor cores** instead of scalar ALU, with the largest gains when the number of chunks $G = T/C$ is large ($G \geq 64$) and the scan elements are small matrices (diagonal or low-rank state transitions).

## Background

### The inter-chunk scan bottleneck

In chunkwise parallel linear RNNs (TFLA, Mamba-2/SSD, GLA), the sequence of $T$ tokens is divided into $G = T/C$ chunks. Within each chunk, computation proceeds via the efficient quadratic-form attention ($O(C^2 d)$) or recurrence. But between chunks, the boundary states must be propagated via a **parallel prefix scan**:

$$
h_j = A_j^{(C)} h_{j-1} + h_j^{\text{local}}, \quad j = 1, \ldots, G
$$

where $A_j^{(C)}$ is the accumulated per-chunk state transition and $h_j^{\text{local}}$ is the per-chunk contribution. This is an associative scan over the operator $(A, b) \circ (A', b') = (A \cdot A', A \cdot b' + b)$.

**Current implementation**: The inter-chunk scan uses either:
1. **CUB DeviceScan** (decoupled look-back, trick 166): Optimized for scalar/vector scans, achieves near-memcpy throughput. But for matrix-valued scans, each "scalar" operation is actually an $n \times n$ or $d \times d$ matrix multiply — which the CUB scan framework treats as a black-box binary operator, missing opportunities for tensor core utilization.
2. **Sequential loop**: For small $G$ (e.g., $G = 32$ with $T = 2048, C = 64$), the inter-chunk scan is often just a sequential loop over chunks, since $G$ is too small to benefit from parallelism.

### Why MatMulScan is a better fit

MatMulScan (Zouzias & McColl, 2024) reformulates the prefix scan algorithm itself as a sequence of **batched matrix multiplications** against two constant matrices:

- $L_s$: Lower-triangular all-ones matrix (for local prefix sums in the upsweep)
- $B_s$: Broadcast matrix (for scalar propagation in the downsweep)

The key insight: **the scan algorithm's control flow** (not just the per-element operator) becomes a tensor-core operation. In CUB's decoupled look-back, the scan structure uses scalar additions and comparisons; MatMulScan replaces these with constant-matrix multiplications.

**Radix $s$ trade-off**: MatMulScan generalizes Brent-Kung to arbitrary radix $s$. Higher $s$ means:
- Fewer sequential levels: $O(\log_s G)$ instead of $O(\log_2 G)$
- More work per level: each matmul is $s \times s$ instead of $2 \times 2$
- Better tensor core utilization: $s = 16$ matches NVIDIA's native $16 \times 16$ MMA tile

For $G = 64$ chunks and $s = 4$: depth = $2 \log_4(64) - 1 = 5$ levels (vs. 11 for $s = 2$), with each level doing batched $4 \times 4$ matmuls.

### The matrix-valued scan specialization

For diagonal SSMs (Mamba-2), the per-chunk transition $A_j^{(C)} = \text{diag}(\alpha_1^{(j)}, \ldots, \alpha_n^{(j)})$ is diagonal. The scan operator $(A, b) \circ (A', b') = (A \cdot A', A \cdot b' + b)$ decomposes into $n$ independent scalar scans (for $A$) and $n \times d_v$ element-wise scans (for $b$). Each scalar scan can use MatMulScan with the scalar radix-$s$ trick:

$$
\text{prefix\_sum}([a_1, \ldots, a_s]) = L_s \cdot [a_1, \ldots, a_s]^\top
$$

For $n = 16$ parallel state channels and $d_v = 64$: we have $16 + 16 \times 64 = 1040$ independent scalar scans, each of length $G$. These can be **batched** into a single large MatMulScan call:

$$
Z = L_s \cdot \text{reshape}(X, [G/s, s, 1040]) \quad \text{(batched matmul)}
$$

This is a batched GEMM of shape $(G/s) \times s \times 1040$ against the constant $s \times s$ matrix $L_s$ — a perfect fit for tensor cores.

### For dense transitions (DeltaNet, DeltaProduct)

When $A_j^{(C)}$ is a dense $n \times n$ matrix, the scan operator involves matrix multiplication. The MatMulScan approach still applies, but now the "elements" being scanned are $n \times n$ matrices. The $L_s$ matrix operates on a **Kronecker-lifted** version:

$$
L_s \otimes I_n : \text{prefix product of } n \times n \text{ matrices}
$$

This maps to batched GEMMs where each "scalar multiplication" is replaced by an $n \times n$ matmul. For $n = 16$: the effective matmul shape is $16s \times 16s$, which for $s = 4$ gives $64 \times 64$ — well-suited to tensor cores.

### What's different from existing proposals

| Aspect | Proposal 026 (Cyclic Reduction) | Proposal 034 (BRGEMM) | This Proposal (044) |
|--------|-------------------------------|----------------------|-------------------|
| **Target** | Full SSM recurrence (all $T$ steps) | Intra-chunk state accumulation | Inter-chunk prefix scan ($G$ steps) |
| **Algorithm** | Block Schur complement elimination | In-register accumulation | Brent-Kung via batched matmuls |
| **Technique** | Cyclic reduction + RandMScan | BRGEMM + Stream-K | MatMulScan ($L_s, B_s$ constant matrices) |
| **When useful** | Dense $n \times n$ transitions | Any (in-register reuse) | Any (replaces scan infrastructure) |
| **Modifies** | Entire parallelization strategy | Intra-chunk kernel | Only the inter-chunk scan pass |

This proposal is **complementary** to 026 and 034: it optimizes the inter-chunk scan pass that both approaches still require for global state propagation.

## Related Work

- **MatMulScan (Zouzias & McColl, Euro-Par 2023 / arXiv:2411.17887)**: Proposed the algorithm but for scalar prefix sums, with applications to gradient boosting and sorting. Did NOT apply it to SSM/linear RNN inter-chunk state propagation or to matrix-valued scans.
- **Mamba-2 SSD (Dao & Gu, 2024)**: Identified that "matmul FLOPs are up to 16× faster" on modern accelerators and redesigned the SSM layer accordingly. Uses standard chunkwise scan for inter-chunk propagation — does not apply MatMulScan.
- **FlashFFTConv (Fu et al., 2024)**: Uses Monarch decomposition to route FFT through tensor cores. Related philosophy (make everything a matmul) but applied to convolution, not to prefix scans.
- **Centaurus (ICLR 2025 submission)**: Treats SSM as tensor contractions and optimizes contraction order. Does not address the scan parallelization.
- **CUB DeviceScan (Merrill & Garland, 2016)**: The de facto standard GPU scan. Achieves near-memcpy throughput for scalar scans but doesn't use tensor cores. MatMulScan subsumes CUB's approach in the TCU model.
- **TFLA (Beck et al., NeurIPS 2025)**: Two-level tiling for linear RNNs. Optimizes intra-chunk computation. The inter-chunk scan is left to standard implementations. Our proposal directly optimizes TFLA's inter-chunk pass.

**Gap**: No existing work applies MatMulScan's TCU-model prefix scan to the inter-chunk state propagation in chunkwise linear RNNs. All existing implementations use CUB-style scalar scans or sequential loops for this phase.

## Mathematical Formulation

**Standard Inter-Chunk Scan (current):**

Given $G$ chunks with per-chunk transitions $\{(A_j, b_j)\}_{j=1}^G$, compute the prefix scan:

$$
(A_{1:j}, b_{1:j}) = (A_j, b_j) \circ (A_{j-1}, b_{j-1}) \circ \cdots \circ (A_1, b_1)
$$

where $\circ$ is the associative operator:

$$
(A, b) \circ (A', b') = (A \cdot A', A \cdot b' + b)
$$

**Current implementation**: Blelloch scan or CUB decoupled look-back. Each $\circ$ invocation requires one matrix multiply ($A \cdot A'$: $O(n^2)$ for diagonal, $O(n^3)$ for dense) + one matrix-vector multiply ($A \cdot b'$: $O(n d_v)$) + one addition ($O(n d_v)$). Total: $O(G n^3 \log G)$ for dense, $O(G n d_v \log G)$ for diagonal.

**MatMulScan Inter-Chunk Scan (proposed):**

**For diagonal transitions** ($A_j = \text{diag}(\alpha_j)$, the common case in Mamba-2/GLA):

The multiplicative scan $\alpha_{1:j} = \prod_{i=1}^{j} \alpha_i$ and additive scan $b_{1:j} = \sum_{i=1}^{j} \alpha_{i+1:j} \cdot b_i$ decompose into independent element-wise scans. Stack all $n \times d_v$ elements into a vector of length $P = n \cdot d_v$:

**Step 1 — Multiplicative prefix product (for $\alpha$)**:

Convert to additive via log: $\log \alpha_{1:j} = \sum_{i=1}^{j} \log \alpha_i$. Apply MatMulScan:

$$
[\log \alpha_{1:1}, \ldots, \log \alpha_{1:G}] = \text{MatMulScan}([\log \alpha_1, \ldots, \log \alpha_G], s)
$$

Exponentiate to recover $\alpha_{1:j} = \exp(\log \alpha_{1:j})$.

**Step 2 — Weighted additive scan (for $b$)**:

The state $b_{1:j} = \sum_{i=1}^{j} (\prod_{l=i+1}^{j} \alpha_l) b_i$ is a weighted prefix sum. Decompose:

$$
b_{1:j} = \alpha_{1:j} \sum_{i=1}^{j} \frac{b_i}{\alpha_{1:i}}
$$

First compute $\tilde{b}_i = b_i / \alpha_{1:i}$ (element-wise division by the cumulative product from Step 1), then run a standard additive prefix sum via MatMulScan, then multiply by $\alpha_{1:j}$.

**MatMulScan detail (for radix $s = 4$):**

At each level, reshape the scan vector into groups of $s$ and multiply by the constant matrix:

$$
L_4 = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}
$$

This is a batched GEMM: $(G/(s \cdot s)) \times s \times s$ tiles, each multiplied by the constant $L_4$. For $G = 256$ and $s = 4$: $256/16 = 16$ independent $4 \times 4$ matmuls — a single batched GEMM call.

**Parallelism**: All $P = n \cdot d_v$ independent scans execute as a **single batched GEMM** of shape $(16 \times P) \times 4 \times 4$. For $n = 16, d_v = 64$: $P = 1024$, giving a batched GEMM with batch size $16 \times 1024 = 16384$ — massively parallel.

**For dense transitions** ($A_j \in \mathbb{R}^{n \times n}$, as in DeltaNet):

The scan operator $(A, b) \circ (A', b') = (AA', Ab' + b)$ is a matrix-valued scan. MatMulScan applies with the radix-$s$ local prefix product encoded as:

$$
[A_1, A_2, A_3, A_4] \mapsto [A_1, A_2 A_1, A_3 A_2 A_1, A_4 A_3 A_2 A_1]
$$

This is **not** a simple $L_s$ multiplication but rather a **sequential prefix product** within each group of $s$. However, for $s = 4$, the sequential chain is only 3 matrix multiplications — short enough to be efficient as a fused kernel. The upsweep then reduces $G$ chunks to $G/s$ in one level.

**Key Variables:**
- $G = T/C$ — number of chunks
- $s$ — MatMulScan radix (typically 4 or 8)
- $n$ — state dimension
- $d_v$ — value dimension
- $P = n \cdot d_v$ — total independent scan lanes (for diagonal case)
- $L_s, B_s \in \mathbb{R}^{s \times s}$ — constant scan matrices

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Mamba-2 / Gated DeltaNet |
| Layers | $L = 24$ |
| Hidden dim | $d = 1024$ |
| State dim | $n = 16$ (diagonal) or $n = 64$ (dense) |
| Value dim | $d_v = 64$ |
| Chunk size | $C = 64$ (giving $G = T/64$) |
| MatMulScan radix | $s = 4$ or $s = 8$ |
| Sequence lengths | $T \in \{2048, 4096, 8192, 16384\}$ |

### Baseline
1. **CUB DeviceScan** (decoupled look-back): Standard GPU scan for inter-chunk propagation. Near-memcpy throughput for scalar scans but doesn't use tensor cores for the scan structure itself.
2. **Sequential loop**: For small $G$, a simple `for j in range(G)` loop. This is what many current implementations use.
3. **Blelloch scan (Triton)**: Custom Triton implementation of work-efficient parallel scan, as used in Mamba.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Inter-chunk scan time | $\leq 0.6\times$ CUB baseline | Wall-clock μs on H100 |
| End-to-end throughput | $\geq 1.05\times$ baseline | Tokens/sec on H100 |
| Tensor core utilization | $> 50\%$ during scan | NVIDIA Nsight profiling |
| Numerical accuracy | $\|y_{\text{MS}} - y_{\text{ref}}\|_\infty < 10^{-4}$ | Max absolute error vs. sequential |
| Scaling with $G$ | Speedup increases with $G$ | Vary $T$ from 2K to 16K |

### Estimated Compute

**MVE**: < 10 minutes, single GPU (microbenchmark only)
**Kernel development + benchmarking**: ~10 GPU-hours
**Integration into GLA/Mamba-2 + training validation**: ~50 GPU-hours
**Total**: ~60 GPU-hours (medium-small scale)

## Expected Outcome

**If hypothesis is correct:**
- For $G = 64$ ($T = 4096, C = 64$), MatMulScan with $s = 4$ achieves $1.3\times$ speedup over CUB for the inter-chunk scan
- For $G = 256$ ($T = 16384, C = 64$), speedup reaches $1.5$–$1.8\times$ due to better amortization of tensor core setup
- The tensor core utilization during the scan phase increases from ~0% (scalar CUB) to $> 50\%$
- End-to-end training throughput improves by $1.05$–$1.15\times$ (the inter-chunk scan is ~10–20% of total per-layer time, depending on $G$)
- The speedup is orthogonal to intra-chunk optimizations (TFLA, proposal 038, 032) — they compose

**If hypothesis is wrong:**
- If MatMulScan is slower: The $L_s$ matrix is lower-triangular, meaning ~50% of the tensor core's compute is wasted on zeros. The gather/scatter overhead between MatMulScan levels may dominate. This would indicate that tensor cores are not beneficial for scans at these sizes ($s \times s$ is too small for efficient MMA utilization). **What we learn**: The minimum GEMM size for tensor core advantage on H100 is above $4 \times 4$; need $s \geq 16$ for benefit, which increases total work.
- If speedup is negligible at small $G$: The inter-chunk scan is not the bottleneck — the intra-chunk computation dominates. This would redirect effort toward proposal 032/034 (intra-chunk optimizations). **What we learn**: The compute/memory balance shifts toward intra-chunk at typical sequence lengths.
- If numerical issues arise: The log-domain conversion for multiplicative scans (log → sum → exp) may lose precision. The exponential amplifies errors from the prefix sum. **Mitigation**: Use the Kahan summation trick within MatMulScan, or operate directly in the multiplicative domain using matrix prefix products.

## Minimum Viable Experiment

### Setup
- **Task**: Pure microbenchmark — compare inter-chunk scan implementations for correctness and throughput
- **Input**: Randomly generated per-chunk transitions and states:
  - Diagonal case: $\alpha_j \in \mathbb{R}^{16}$, $b_j \in \mathbb{R}^{16 \times 64}$, $G = 64, 128, 256$
  - Dense case: $A_j \in \mathbb{R}^{16 \times 16}$, $b_j \in \mathbb{R}^{16 \times 64}$, $G = 64$
- **Implementation**: Triton kernel for MatMulScan ($s = 4$ and $s = 8$), vs. Triton Blelloch scan, vs. sequential Python loop
- **Hardware**: Single H100 or A100
- **Compute**: < 5 minutes (kernel compilation + benchmarking)

### Success Criteria
- MatMulScan ($s = 4$) achieves $\geq 1.2\times$ throughput over Triton Blelloch scan for $G = 128$, diagonal case
- MatMulScan achieves $\geq 1.3\times$ for $G = 256$
- Numerical accuracy: max absolute error $< 10^{-3}$ (bf16) or $< 10^{-5}$ (fp32) vs. sequential reference
- The Triton kernel compiles and runs without errors for all tested configurations

### Failure Criteria
- If MatMulScan is $\geq 0.9\times$ slower than Blelloch for all $G$: The overhead of reshaping data for MatMulScan + the $4 \times 4$ matmuls being too small for tensor cores makes this approach unviable. Kill the idea.
- If numerical errors $> 10^{-1}$: The log-exp conversion for multiplicative scans is too unstable. Need to redesign for direct multiplicative MatMulScan (without log domain).

### Why This Test Is Sufficient
- The microbenchmark directly measures the kernel throughput — the primary hypothesis. If the scan kernel isn't faster in isolation, it won't be faster end-to-end.
- The diagonal case with $G = 128$ is representative of typical pretraining workloads ($T = 8192, C = 64$).
- Numerical accuracy at bf16 is sufficient for pretraining — if it works in bf16, the quality impact is negligible.
- No model training is needed to validate the core hypothesis about scan speed.

## Memory Access Pattern Analysis

**CUB Decoupled Look-back (current):**
- Access pattern: Scan-specific. Each CTA processes a tile, writes aggregate to global memory, then looks back through predecessor descriptors.
- Arithmetic intensity: Low — each element is read once, written once, with $O(1)$ scalar operations per element. Total HBM traffic: $\sim 2G \cdot n \cdot d_v$ words.
- Tensor cores: **Not used** — all operations are scalar additions/multiplications.

**MatMulScan (proposed):**
- Access pattern: Two-phase (upsweep + downsweep). At each level, gather strided elements, batch-matmul against constant $L_s$ or $B_s$, scatter results.
- Arithmetic intensity: Higher than CUB — each level performs $s^2$ multiply-adds per group of $s$ elements. For $s = 4$: intensity = $4^2 / (2 \times 4) = 2$ FLOPs/byte.
- Tensor cores: **Yes** — the batch-matmul against $L_s$ and $B_s$ routes through WGMMA/MMA instructions.
- Shared memory: $L_s$ and $B_s$ are $s \times s$ constant matrices. For $s = 4$: 32 bytes in bf16. Fits easily in constant memory or registers.
- Cache behavior: The strided gather/scatter may cause non-contiguous memory access. **Mitigation**: Pre-transpose data into groups of $s$ contiguous elements at each level.

## Parallelism Analysis

- **SM saturation**: With $P = n \times d_v = 1024$ independent scan lanes and $G/s^2$ batched matmuls per level, the total batch count is $1024 \times G/16$ for $s = 4$. For $G = 128$: batch = $1024 \times 8 = 8192$ — easily saturates 108 SMs on A100.
- **Warp divergence**: None — all threads in a warp execute the same matmul instruction against the constant matrix.
- **Load imbalance**: None within a level (all groups have the same $s$ elements). Between levels, decreasing group counts may leave SMs idle at the top of the tree. **Mitigation**: Batch all levels' matmuls into a single batched GEMM call.
- **Tensor core mapping**: $s = 4$ gives $4 \times 4$ matmuls. These are smaller than the $16 \times 16$ native MMA tile, so each matmul uses only $1/16$ of the tile's capacity. **Consideration**: Use $s = 16$ to fully utilize the MMA tile, accepting the $\sim 8n$ work overhead (vs. $\sim 2n$ for $s = 2$).

## Theoretical Analysis

Complexity comparison for inter-chunk scan ($G$ chunks, diagonal $n$-dim state):

| Operation | Blelloch Scan | CUB Decoupled Look-back | MatMulScan ($s = 4$) |
|-----------|--------------|------------------------|---------------------|
| Depth (sequential levels) | $2 \log_2 G$ | $O(\log G)$ amortized | $2 \log_4 G - 1 \approx \log_2 G$ |
| Work (total ops) | $2G \cdot P$ | $\sim 2G \cdot P$ | $\sim 3G \cdot P$ (50% overhead) |
| Tensor core utilization | 0% | 0% | **$\geq 50\%$** |
| Effective throughput | $\sim 0.06\times$ peak | $\sim 0.10\times$ peak (near memcpy) | $\sim 0.3$–$0.5\times$ peak |

The key trade: MatMulScan does $\sim 50\%$ more work but at $\sim 5$–$10\times$ higher throughput (tensor cores vs. scalar ALU), for a net $\sim 3$–$6\times$ speedup.

Crossover point: MatMulScan is beneficial when $\text{TC\_throughput} / \text{ALU\_throughput} > \text{work\_ratio}$. On H100: TC throughput is $\sim 989$ TFLOPS (bf16) vs. $\sim 67$ TFLOPS (fp32 scalar), ratio $\sim 15\times$. Work ratio for $s = 4$ is $\sim 1.5\times$. So MatMulScan should be $\sim 10\times$ faster *per FLOP* — the question is whether the gather/scatter overhead dominates.

## Risks & Limitations

1. **Gather/scatter overhead**: MatMulScan requires strided memory access at each level. On GPUs, non-contiguous access degrades throughput. The original paper notes this as a "critical bottleneck." **Mitigation**: Pre-sort data into level-specific layouts in shared memory; use TMA async loads on H100 for efficient strided access.

2. **Small matmul inefficiency**: $4 \times 4$ matmuls use only $1/16$ of a $16 \times 16$ MMA tile. This wastes $15/16$ of tensor core capacity. **Mitigation**: Use $s = 16$ (native MMA size), accepting $\sim 8n$ work; or batch multiple scans into a single GEMM where the batch dimension fills the tile.

3. **Log-domain instability**: Converting multiplicative scans to additive via log/exp introduces numerical error, especially for values near 0 or for long products that overflow/underflow. **Mitigation**: Use log-sum-exp stabilization; or implement a direct multiplicative MatMulScan where $L_s$ encodes prefix products instead of prefix sums (this requires modifying $L_s$ to have entries $A_i A_{i-1} \cdots A_j$ instead of all-ones).

4. **Non-determinism**: Like CUB, the gather/scatter pattern in MatMulScan means the reduction order may vary between runs for floating-point, producing non-bitwise-reproducible results. This is acceptable for pretraining.

5. **Limited applicability for small $G$**: When $G < s^2 = 16$ (e.g., $T = 1024, C = 64$: $G = 16$), MatMulScan has only 1–2 levels, and the overhead of setup dominates. At very small $G$, the sequential loop is faster. **Mitigation**: Fall back to sequential scan for $G < 16$.

6. **Implementation complexity**: A custom Triton/CUDA kernel is required. The Triton programming model doesn't natively support the strided gather/scatter pattern of MatMulScan. May need to use `tl.load` with computed offsets + `tl.store` with scatter.

## Follow-up Experiments

1. **Radix optimization**: Sweep $s \in \{2, 4, 8, 16\}$ to find the optimal radix for different $G$ and hardware. Expected: $s = 4$ optimal for $G \leq 256$, $s = 8$–$16$ for $G \geq 512$.

2. **Fused intra-chunk + inter-chunk**: Fuse the MatMulScan inter-chunk pass into the TFLA intra-chunk kernel's epilogue (proposal 040). The last CTA to finish each chunk's intra-chunk computation can immediately begin the MatMulScan upsweep for inter-chunk aggregation, overlapping computation.

3. **MatMulScan for backward pass**: The backward pass through the scan requires a reverse-direction scan. MatMulScan applies identically in reverse (swap upsweep/downsweep order). Benchmark forward + backward speedup.

4. **Dense-transition MatMulScan**: For DeltaNet/DeltaProduct with $n \times n$ dense transitions, the per-group prefix product is a chain of $s$ matrix multiplications. Profile whether this is faster than the standard Blelloch scan with $n \times n$ operators, which does $O(\log G)$ such chains.

5. **Hybrid CUB + MatMulScan**: Use CUB's decoupled look-back for the global aggregation (it has near-optimal latency) but replace the local tile scan within each CTA with MatMulScan's $L_s$-based local prefix sum. This gets the best of both: CUB's efficient global communication + MatMulScan's tensor-core local computation.

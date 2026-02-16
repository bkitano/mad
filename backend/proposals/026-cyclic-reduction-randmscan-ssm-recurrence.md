---
status: completed
priority: high
created: 2026-02-15
based_on: cyclic-reduction-block-tridiagonal, randmscan-matrix-parallel-scan, chunkwise-parallel-scan, blelloch-work-efficient-scan, recurrence-to-scan-reduction, householder-product-parameterization, wy-representation, io-aware-tiling
experiment_number: 026
results_file: 026_results.md
---

# Cyclic Reduction with RandMScan for Dense SSM Recurrences

## Hypothesis

For SSM recurrences with **non-diagonal** state-transition matrices (e.g., DeltaNet/DeltaProduct with $A_t = I - \beta_t v_t k_t^\top$, or Monarch-structured SSMs), the standard parallel scan approach incurs $O(T n^3 \log T)$ total work because the associative operator requires $n \times n$ matrix multiplications. **Cyclic reduction** applied to the block-bidiagonal formulation of the same recurrence achieves the same $O(\log T)$ parallel depth but with only $O(T n^3)$ total work — eliminating the $\log T$ factor from the work complexity. Combined with **RandMScan's matrix-based local scan** (which reformulates scan primitives as tensor-core-friendly matmuls) and its **barrier-free random-jump global aggregation**, this yields a practical $1.5\times$–$2\times$ speedup over standard prefix scan for dense-transition SSMs on modern GPU hardware.

## Background

The linear recurrence $h_t = A_t h_{t-1} + B_t x_t$ with input-dependent $A_t \in \mathbb{R}^{n \times n}$ is the computational core of modern SSMs. For training, this recurrence must be parallelized across the time dimension $T$.

**The standard approach** (Blelloch prefix scan): Define the associative operator $(A_t, b_t) \circ (A_s, b_s) = (A_t A_s, A_t b_s + b_t)$ and run a parallel prefix scan. This achieves $O(\log T)$ parallel depth but $O(T n^3 \log T)$ total work — each of the $O(T \log T)$ operator applications requires an $n \times n$ matrix multiply.

**The cyclic reduction alternative**: Rewrite the recurrence as a block-bidiagonal system:

$$
\begin{pmatrix} I & & & \\ -A_2 & I & & \\ & -A_3 & I & \\ & & \ddots & \ddots \\ & & & -A_T & I \end{pmatrix} \begin{pmatrix} h_1 \\ h_2 \\ \vdots \\ h_T \end{pmatrix} = \begin{pmatrix} B_1 x_1 + A_1 h_0 \\ B_2 x_2 \\ \vdots \\ B_T x_T \end{pmatrix}
$$

Cyclic reduction solves this by recursively eliminating even-indexed unknowns via Schur complements, producing a half-sized system that retains block-bidiagonal structure. After $\lceil \log_2 T \rceil$ levels, we reach a single block that is solved directly, then back-substitute to recover all states.

**Key advantage over prefix scan**: At each recursion level $l$, there are $T/2^l$ independent block eliminations, each costing $O(n^3)$ (one matrix inverse + two matrix multiplies). The total work across all levels is:

$$
\text{Work}_{\text{CR}} = \sum_{l=0}^{\log_2 T} \frac{T}{2^l} \cdot O(n^3) = O(T n^3) \quad \text{(geometric series)}
$$

vs. $\text{Work}_{\text{scan}} = O(T n^3 \log T)$ for prefix scan. This is a $\log T$ factor savings in total work.

**Why this matters now**: For diagonal SSMs (Mamba, S4D), $n \times n$ matrix operations reduce to element-wise operations ($O(n)$ per step), so the $\log T$ overhead of prefix scan is negligible. But for **non-diagonal** architectures that achieve greater expressivity — DeltaProduct (Householder products), Monarch SSMs (Proposal 006), group-and-shuffle SSMs (Proposal 016), hyperoctahedral SSMs (Proposal 017), OH-DeltaProduct (Proposal 020) — the $n^3$ cost of each operator application makes the $\log T$ factor significant. For $T = 4096$ and $n = 64$: the savings is $\log_2(4096) = 12\times$ in total work.

**RandMScan integration**: The cyclic reduction algorithm at each level requires local computation (Schur complement block operations) that maps naturally to matrix multiplications on tensor cores. RandMScan's two-stage framework — (1) matrix-based local computation using upper/lower triangular matmuls, and (2) barrier-free random-jump global aggregation — provides the hardware-efficient execution model for cyclic reduction's recursive structure.

**Gap filled**:
- All 24 existing proposals use either standard prefix scan or chunkwise parallel scan for parallelization
- Cyclic reduction has been documented as a trick but never applied in any proposal
- RandMScan has been documented but never used in any proposal
- No proposal addresses the $O(T n^3 \log T)$ work overhead of prefix scan for dense-transition SSMs

## Mathematical Formulation

**Block-Bidiagonal System from SSM Recurrence:**

The recurrence $h_t = A_t h_{t-1} + b_t$ (where $b_t = B_t x_t$) yields:

$$
\mathcal{A} \, \mathbf{h} = \mathbf{b}, \quad \mathcal{A} = \begin{pmatrix} I \\ -A_2 & I \\ & -A_3 & I \\ & & \ddots & \ddots \\ & & & -A_T & I \end{pmatrix}
$$

This is a lower block-bidiagonal matrix with $n \times n$ blocks.

**Cyclic Reduction — Level 0 (First Elimination):**

Reorder unknowns into odd indices $\{1, 3, 5, \ldots\}$ and even indices $\{2, 4, 6, \ldots\}$. Eliminate even unknowns:

For even index $2j$:
$$
h_{2j} = A_{2j} h_{2j-1} + b_{2j}
$$

Substitute into the equation for odd index $2j+1$:
$$
h_{2j+1} = A_{2j+1} h_{2j} + b_{2j+1} = A_{2j+1} A_{2j} h_{2j-1} + A_{2j+1} b_{2j} + b_{2j+1}
$$

This gives a new recurrence on odd indices:
$$
h_{2j+1} = \tilde{A}_{j}^{(1)} h_{2j-1} + \tilde{b}_{j}^{(1)}
$$

where:
$$
\tilde{A}_{j}^{(1)} = A_{2j+1} A_{2j}, \quad \tilde{b}_{j}^{(1)} = A_{2j+1} b_{2j} + b_{2j+1}
$$

**General Level $l$:**

At level $l$, we have $T/2^l$ unknowns with transition matrices $\{\tilde{A}_j^{(l)}\}$. The elimination produces:

$$
\tilde{A}_{j}^{(l+1)} = \tilde{A}_{2j+1}^{(l)} \tilde{A}_{2j}^{(l)}, \quad \tilde{b}_{j}^{(l+1)} = \tilde{A}_{2j+1}^{(l)} \tilde{b}_{2j}^{(l)} + \tilde{b}_{2j+1}^{(l)}
$$

**Crucially**: all $T/2^{l+1}$ updates at level $l$ are **independent** and can execute in parallel.

**Back-Substitution:**

After reaching a single block at level $\lceil \log_2 T \rceil$, back-substitute:
$$
h_{2j} = A_{2j} h_{2j-1} + b_{2j} \quad \text{(level } l \text{ to level } l-1\text{)}
$$

All even-index recoveries at each level are again independent.

**Comparison of Total Work:**

$$
\text{Work}_{\text{prefix scan}} = 2T \cdot M(n) \cdot \lceil \log_2 T \rceil
$$

$$
\text{Work}_{\text{cyclic reduction}} = \sum_{l=0}^{\lceil \log_2 T \rceil - 1} \frac{T}{2^l} \cdot M(n) = 2T \cdot M(n)
$$

where $M(n) = O(n^3)$ is the cost of one matrix multiply. The cyclic reduction work is $\log_2 T$ times smaller.

**Parallel Depth (both methods):**

$$
\text{Depth}_{\text{prefix scan}} = O(\log T) \cdot O(1) = O(\log T)
$$

$$
\text{Depth}_{\text{cyclic reduction}} = 2 \lceil \log_2 T \rceil \cdot O(1) = O(\log T)
$$

Both achieve $O(\log T)$ depth; cyclic reduction has a $2\times$ constant (forward + back-substitution) but much less total work.

**RandMScan Integration — Local Stage:**

Within each level of cyclic reduction, the $T/2^l$ independent matrix multiplications ($\tilde{A}_{2j+1}^{(l)} \tilde{A}_{2j}^{(l)}$) can be batched into a single large GEMM:

$$
\tilde{\mathcal{A}}^{(l+1)} = \text{BatchedGEMM}\left(\tilde{\mathcal{A}}_{\text{odd}}^{(l)}, \tilde{\mathcal{A}}_{\text{even}}^{(l)}\right)
$$

This maps directly to tensor core operations. RandMScan's matrix-based local scan reformulates the per-level computation as structured matrix multiplies using upper/lower triangular masks, maximizing tensor core utilization.

**RandMScan Integration — Global Stage:**

For the inter-level communication (propagating cumulative products across the recursion tree), the random-jump strategy eliminates global barriers. Each computational unit processes its local cyclic reduction levels asynchronously, then uses random-jump to aggregate cross-unit results with $O(\log P)$ latency for $P$ processing units.

**Key Variables:**
- $h_t \in \mathbb{R}^n$ — state vector at time $t$
- $A_t \in \mathbb{R}^{n \times n}$ — state-transition matrix (input-dependent, non-diagonal)
- $T$ — sequence length
- $n$ — state dimension
- $C$ — chunk size for the outer chunkwise algorithm
- $M(n)$ — cost of $n \times n$ matrix multiply ($\Theta(n^3)$ or $\Theta(n^\omega)$ for Strassen-like)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | DeltaProduct / Monarch SSM with CR-RandMScan parallelization |
| Layers | $L = 6$–$12$ |
| Hidden dim | $d = 256$–$512$ |
| State dim | $n = 32$–$64$ per head |
| Heads | 8 |
| Sequence length | $T = 2048$–$8192$ |
| Parallelization | Cyclic reduction + RandMScan (replaces prefix scan) |
| Chunk size | $C = 64$–$128$ (for outer chunkwise algorithm) |
| Inner parallelization | Cyclic reduction within each chunk's recurrence |

### Baseline

1. **Prefix scan (Blelloch)**: Standard parallel scan for non-diagonal SSM recurrence. Work: $O(TC n^3 \log C)$ for intra-chunk, $O((T/C) n^3 \log(T/C))$ for inter-chunk
2. **Chunkwise parallel (Mamba-2/SSD)**: Quadratic intra-chunk + scan inter-chunk. Work: $O(TC^2 d + (T/C) n^3 \log(T/C))$
3. **Sequential recurrence**: $O(T n^3)$ total work, $O(T)$ depth. Lower bound on work for any parallelization
4. **Cyclic reduction without RandMScan**: Standard cyclic reduction with synchronous barriers. Same work as CR-RandMScan but with more communication overhead

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Wall-clock training time | $< 0.7\times$ prefix scan | Seconds per batch at $T = 4096$ |
| Total FLOP count | $< 0.5\times$ prefix scan | Measured via profiler |
| Perplexity | $=$ prefix scan ($\pm 0.1$) | WikiText-103 validation (should be identical) |
| GPU utilization | $> 80\%$ tensor core | nsight-compute profiling |
| Memory peak | $\leq$ prefix scan | Peak GPU memory |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (microbenchmark: CR vs scan): ~5 GPU-hours on A100
**Phase 2** (end-to-end DeltaProduct training): ~60 GPU-hours on A100
**Total**: ~65 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- For DeltaProduct with $n = 64$, $T = 4096$: cyclic reduction achieves $5\times$–$10\times$ reduction in total FLOP count for the scan operation (saving the $\log_2 4096 = 12\times$ factor, minus overhead)
- Wall-clock speedup of $1.5\times$–$2\times$ for the full recurrence computation (FLOP savings partially offset by less regular memory access patterns)
- RandMScan's barrier-free global stage provides an additional $10\%$–$20\%$ latency reduction over synchronous cyclic reduction, especially on multi-SM GPU execution
- The total training throughput improvement compounds with chunkwise parallelism: for intra-chunk recurrence, cyclic reduction on chunks of size $C = 64$ saves $\log_2 64 = 6\times$ in work
- Model quality (perplexity, state-tracking accuracy) is numerically identical to prefix scan, since cyclic reduction computes the **exact same result** — just with a different schedule of operations

**If hypothesis is wrong:**
- If wall-clock time doesn't improve despite FLOP reduction: the memory access pattern of cyclic reduction (doubling stride at each level) causes cache misses that negate the FLOP savings. This would indicate that modern GPUs are memory-bandwidth-bound for these operations, not compute-bound. Mitigation: use IO-aware tiling to keep the working set in shared memory
- If RandMScan adds overhead vs. synchronous CR: the random-jump global stage's non-deterministic scheduling causes thread divergence on GPUs (unlike PE arrays where it was designed). This would mean the barrier-free property doesn't help on SIMT architectures
- If the implementation is too complex to be practical: cyclic reduction's two-phase (forward elimination + back-substitution) structure is harder to implement in CUDA than the simpler prefix scan. This would be a useful negative result about the engineering-theory gap

## Minimum Viable Experiment

### Setup
- **Model**: Standalone scan kernel benchmark (no full model needed for the core claim)
- **Task**: Compute the parallel scan $h_t = A_t h_{t-1} + b_t$ for random $A_t \in \mathbb{R}^{n \times n}$, $b_t \in \mathbb{R}^n$ with $n = 32$, $T = 1024$
- **Implementation**: PyTorch batched GEMM for both prefix scan and cyclic reduction
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- **FLOP count**: Cyclic reduction uses $\leq 2T \cdot n^3$ multiply-adds vs. $\geq 2T \cdot n^3 \cdot \log_2 T$ for prefix scan. Verified by counting GEMM calls.
- **Numerical accuracy**: $\|h^{\text{CR}} - h^{\text{scan}}\|_\infty / \|h^{\text{scan}}\|_\infty < 10^{-5}$ (both compute the same mathematical result)
- **Wall-clock time** (PyTorch-level, not optimized): CR is at least $2\times$ faster than naive prefix scan for $n = 32$, $T = 1024$
- **Scaling**: The speedup ratio increases with $T$ (because the $\log T$ factor grows)

### Failure Criteria
- If wall-clock time for CR is *slower* than prefix scan at $n = 32$, $T = 1024$ in PyTorch: the overhead of managing the two-phase recursion and non-uniform parallelism at each level outweighs the FLOP savings at this scale. Would need a custom CUDA kernel to see benefits.
- If numerical results differ by more than $10^{-4}$: floating-point non-associativity in the different evaluation orders causes meaningful divergence. Would need higher precision or numerically stabilized variants.

### Why This Test Is Sufficient
- The core claim is purely computational (same math, fewer FLOPs), so a standalone benchmark directly validates it without requiring a full model training run
- If cyclic reduction is faster for the scan kernel, the end-to-end model benefit follows by substitution — the scan is a module that can be swapped independently
- The $n = 32$ setting is realistic (DeltaProduct uses $n = 16$–$64$ per head) and $T = 1024$ has $\log_2 T = 10$, enough to show meaningful work savings
- Scaling to $T = 4096$ only increases the savings (more $\log T$ factor to eliminate)

## Theoretical Analysis

Complexity comparison for sequence length $T$, state dim $n$:

| Operation | Prefix Scan | Cyclic Reduction | CR + RandMScan |
|-----------|------------|-----------------|----------------|
| Total work | $O(T n^3 \log T)$ | $O(T n^3)$ | $O(T n^3)$ |
| Parallel depth | $O(\log T)$ | $O(\log T)$ | $O(\log T)$ |
| Memory | $O(T n^2)$ | $O(T n^2)$ | $O(T n^2)$ |
| Barrier count | $O(\log T)$ global | $O(\log T)$ global | $O(1)$ global + $O(\log P)$ random-jump |
| Communication volume | $O(T n^2 \log T)$ | $O(T n^2)$ | $O(T n^2)$ |

Work-efficiency: Cyclic reduction is **work-optimal** — its total work matches the sequential algorithm $O(T n^3)$. Prefix scan is work-suboptimal by a factor of $\log T$.

**When this matters**: The $\log T$ savings is multiplicative with $n^3$. For diagonal SSMs ($n \times n$ operations are $O(n)$), the savings is $\log T \cdot n$ which is small. For dense SSMs ($O(n^3)$ per operation), the savings is $\log T \cdot n^3$, which dominates for $n \geq 16$:

| $n$ | $T$ | Prefix scan work | CR work | Speedup |
|-----|------|-----------------|---------|---------|
| 16 | 1024 | $4.2 \times 10^7$ | $4.2 \times 10^6$ | $10\times$ |
| 32 | 2048 | $7.2 \times 10^8$ | $6.7 \times 10^7$ | $10.7\times$ |
| 64 | 4096 | $6.4 \times 10^{10}$ | $5.4 \times 10^9$ | $12\times$ |

## Risks & Limitations

1. **Memory access patterns**: Cyclic reduction at level $l$ accesses blocks with stride $2^l$, creating increasingly non-local memory accesses. On GPUs, this may cause L2 cache thrashing that negates the FLOP savings. Mitigation: reorder data at each level to maintain spatial locality (at the cost of additional memcpy), or use IO-aware tiling to keep the active working set in shared memory.

2. **Two-phase overhead**: Unlike prefix scan (which has a single forward pass in some formulations), cyclic reduction requires a forward elimination phase AND a back-substitution phase. The back-substitution has the same depth ($O(\log T)$) and involves the same number of matmuls, so the total depth is $2\log T$ vs. $\log T$ for prefix scan — a $2\times$ depth penalty. However, total work is still much lower.

3. **Non-commutative operator**: For SSM recurrences, the scan operator $(A, b) \circ (A', b') = (AA', Ab' + b)$ is non-commutative. Cyclic reduction is derived from commutativity-independent elimination (it works for arbitrary block-bidiagonal systems), so this is not a mathematical issue — but implementations must be careful about multiplication order.

4. **Batch parallelism vs. time parallelism**: On GPUs, batch parallelism (across multiple sequences) often provides enough parallel work to saturate tensor cores, reducing the importance of time parallelism. If batch size is large enough that the prefix scan's extra work is hidden behind data parallelism, cyclic reduction's advantage shrinks. This proposal matters most for large-$n$, moderate-batch settings.

5. **Integration with chunkwise algorithms**: The standard Mamba-2/SSD training already uses chunkwise parallelism where intra-chunk is quadratic attention. Cyclic reduction would replace the inter-chunk prefix scan. For $T/C$ chunks, the inter-chunk savings is a factor of $\log(T/C)$, which is smaller than $\log T$. The intra-chunk application would replace the intra-chunk scan (if used) with cyclic reduction on chunk-length sequences.

6. **CUDA implementation complexity**: Cyclic reduction requires careful scheduling of batched GEMMs with varying batch sizes at each recursion level ($T/2, T/4, \ldots, 1$). This is less regular than prefix scan's uniform parallelism and may require a custom persistent kernel for good occupancy.

## Follow-up Experiments

1. **Hybrid CR-Scan**: Use cyclic reduction for the first $k$ levels (where parallelism is high) and switch to sequential recurrence for the remaining $\log T - k$ levels (where the system is small enough for direct solve). This balances the memory locality advantages of sequential computation with the parallelism of CR.

2. **CR for Householder/WY accumulation**: Apply cyclic reduction to the UT-transform accumulation in DeltaProduct/OH-DeltaProduct (Proposal 020). The WY representation accumulates Householder reflections via a recurrence $W_{j+1} = W_j - \beta v v^\top W_j$ that is also block-structured. CR could parallelize this more efficiently than the sequential UT transform.

3. **CR on custom hardware (NPU/TPU)**: RandMScan was designed for PE-array accelerators (SOFA, Ascend), not GPUs. Test CR-RandMScan on TPU v4/v5 or Ascend 910B where the matrix-based local scan maps directly to the systolic array and the random-jump global stage exploits the high-bandwidth inter-chip network.

4. **CR for multi-head SSM with shared transitions**: When multiple heads share the same $A_t$ (as in some efficient SSM designs), the cyclic reduction's matrix products are shared across heads, amortizing the cost. Compare against per-head prefix scan.

5. **Approximate cyclic reduction**: Instead of exact Schur complement elimination ($\tilde{A}_j^{(l+1)} = A_{2j+1}^{(l)} A_{2j}^{(l)}$), use Neumann series approximation (Proposal 011) for the matrix products at later levels where the matrices are more contractive. This would further reduce total work at the cost of approximation error.

## Human Review

(To be filled by reviewer)

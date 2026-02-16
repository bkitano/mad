# 121: Stream-K Work-Centric GEMM Decomposition

**Category**: kernel
**Gain type**: efficiency
**Source**: Osama, Merrill, Cecka, Garland, Owens — "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU" (PPoPP 2023)
**Paper**: [papers/stream-k-work-centric-gemm.pdf]
**Documented**: 2025-02-15

## Description

Stream-K is a work-centric parallelization strategy for GPU matrix multiplication (GEMM) that eliminates **quantization inefficiency** — the GPU underutilization that occurs when the number of output tiles does not evenly divide the number of streaming multiprocessors (SMs). Traditional GEMM implementations use a *data-parallel* decomposition that assigns one cooperative thread array (CTA) per output tile. When the number of tiles is not a multiple of the SM count, the final "wave" of CTAs leaves SMs idle, wasting up to 50% of compute capacity for certain problem shapes.

Stream-K instead partitions the aggregate MAC-loop (multiply-accumulate) iterations — not output tiles — evenly across a fixed-size grid of CTAs equal to the number of SMs. Each CTA processes a contiguous range of iterations in the linearized $m \to n \to k$ iteration space, crossing output tile boundaries as needed. This achieves near-perfect load balancing regardless of problem shape, from a single compiled kernel (no ensemble of tile-size variants needed).

The key insight is that a single MAC-loop iteration is orders of magnitude cheaper than an entire output tile, so the granularity of work distribution becomes fine enough to eliminate quantization waste entirely.

## Mathematical Form

**GEMM definition:**

$$
\mathbf{C} = \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}, \quad \mathbf{A} \in \mathbb{R}^{m \times k}, \; \mathbf{B} \in \mathbb{R}^{k \times n}, \; \mathbf{C} \in \mathbb{R}^{m \times n}
$$

**Tiled GEMM:** Block the computation with tile sizes $B_M, B_N, B_K$:

- Number of output tiles: $t = \lceil m / B_M \rceil \times \lceil n / B_N \rceil$
- MAC-loop iterations per tile: $\text{iters\_per\_tile} = \lceil k / B_K \rceil$
- Total MAC-loop iterations: $\text{total\_iters} = t \times \text{iters\_per\_tile}$

**Data-parallel decomposition** (traditional):

$$
\text{Utilization}_{\text{DP}} = \frac{t}{\lceil t / p \rceil \times p}
$$

where $p$ is the number of SMs. When $t \bmod p \neq 0$, the last wave has idle SMs.

**Stream-K decomposition:**

Partition $\text{total\_iters}$ evenly across $g$ CTAs (typically $g = p$):

$$
\text{iters\_per\_cta} = \lceil \text{total\_iters} / g \rceil
$$

Each CTA $x \in \{0, \ldots, g-1\}$ processes iterations $[x \cdot \text{iters\_per\_cta}, \; (x+1) \cdot \text{iters\_per\_cta})$, mapping each iteration index back to its output tile and $k$-axis position:

$$
\text{tile\_idx} = \lfloor \text{iter} / \text{iters\_per\_tile} \rfloor, \quad \text{local\_iter} = \text{iter} \bmod \text{iters\_per\_tile}
$$

$$
\text{Utilization}_{\text{SK}} \approx 1.0 \quad \text{(for all problem shapes)}
$$

**Fixup mechanism:** When a CTA's iteration range spans a tile boundary, it produces a partial sum for the tile it exits. The CTA that finishes the tile (the one whose $k=0$ iteration falls on that tile) accumulates all partial sums before writing the final output:

$$
\mathbf{C}_{\text{tile}} = \sum_{c=0}^{s-1} \text{partial}[c] \quad \text{where } s = \text{FixupPeers}(g)
$$

**Key Definitions:**

- $t$ — total number of output tiles in the GEMM
- $p$ — number of SM cores on the GPU
- $g$ — grid size (number of CTAs launched, typically $g = p$)
- $B_M, B_N, B_K$ — blocking factors for the M, N, K dimensions
- MAC-loop iteration — one multiply-accumulate step of volume $B_M \times B_N \times B_K$
- Quantization efficiency — $t / (\lceil t/p \rceil \cdot p)$, fraction of SMs doing useful work

**Grid size selection model:**

$$
\text{time}_{\text{CTA}}(g) = a + b \cdot \mathbb{1}[\text{FixupPeers}(g) > 1] + c \cdot \text{ItersPerCta}(g) + d \cdot (\text{FixupPeers}(g) - 1)
$$

where $a$ = fixed overhead, $b$ = partial sum store cost, $c$ = per-iteration cost, $d$ = per-collaborator fixup cost.

## Complexity

| Metric | Data-Parallel | Stream-K |
|--------|--------------|----------|
| Quantization efficiency | $t / (\lceil t/p \rceil \cdot p)$ (can be $< 0.5$) | $\approx 1.0$ |
| Splitting seams (partial sums) | $0$ | $O(p)$ |
| Kernel variants needed | Ensemble of 3-20+ | $1$ |
| Communication overhead | $0$ | $O(p)$ (independent of problem size) |

**Memory:** $O(p \cdot B_M \cdot B_N)$ temporary storage for partial sums (scales with SM count, not problem size).

**Performance:**
- Average 1.23× (FP64) and 1.63× (FP16→32) speedup vs. CUTLASS data-parallel
- Average 1.06× (FP64) and 1.13× (FP16→32) speedup vs. cuBLAS ensemble
- Peak speedup up to 14× (FP64) and 6.7× (FP16→32) for pathological shapes
- Much tighter performance spread across 32,824 problem geometries

## Applicability

- **Transformer attention**: The GEMM shapes in attention ($\mathbf{Q}\mathbf{K}^T$ and $\text{softmax} \times \mathbf{V}$) often have non-square, sequence-length-dependent shapes that quantize poorly on data-parallel decompositions. Stream-K provides consistent performance regardless of sequence length.
- **Linear layers**: MLP projections in transformers (up-project, down-project, gate) at various batch sizes and hidden dimensions.
- **Batched GEMM**: Small per-batch GEMM shapes common in multi-head attention can severely under-occupy SMs; Stream-K maintains utilization.
- **Variable-length workloads**: Inference serving with mixed sequence lengths creates diverse GEMM shapes — Stream-K handles all of them with a single kernel.
- **Convolutions**: Conv2D via im2col+GEMM benefits from the same load-balancing.

## Limitations

- **Fixup synchronization**: CTAs that share output tiles must synchronize via global memory atomics — introduces latency when $> 2$ CTAs contribute to one tile.
- **Tile-processing skew**: When $g$ is not a multiple of $t$, different CTAs start at different $k$-axis offsets, potentially reducing L2 cache reuse across CTAs. The hybrid "two-tile SK + data-parallel" schedule mitigates this.
- **Memory-bound regime**: Stream-K targets compute-bound problems (high arithmetic intensity). For small, memory-bound GEMMs, the fixup overhead may not be amortized.
- **Architecture coupling**: Optimal grid-size selection constants ($a, b, c, d$) must be calibrated per GPU architecture via microbenchmarks (done once, compiled statically).

## Implementation Notes

```python
# Stream-K pseudocode (simplified from Algorithm 5 in the paper)
def stream_k_gemm(A, B, C, m, n, k, BLK_M, BLK_N, BLK_K, g):
    """
    g = number of CTAs (typically = number of SMs)
    Each CTA gets an even share of total MAC-loop iterations.
    """
    iters_per_tile = ceil(k / BLK_K)
    total_iters = ceil(m/BLK_M) * ceil(n/BLK_N) * iters_per_tile
    iters_per_cta = ceil(total_iters / g)

    # Each CTA x processes iterations [iter_begin, iter_end)
    for x in parallel(g):  # one CTA per SM
        iter_begin = x * iters_per_cta
        iter_end = min((x + 1) * iters_per_cta, total_iters)
        iter = iter_begin

        while iter < iter_end:
            tile_idx = iter // iters_per_tile
            tile_iter = tile_idx * iters_per_tile
            tile_iter_end = tile_iter + iters_per_tile
            local_iter = iter - tile_iter
            local_end = min(tile_iter_end, iter_end) - tile_iter

            # Compute partial accumulator for this tile's k-range
            accum = MacLoop(tile_idx, local_iter, local_end)

            # Determine if this CTA started or ended this tile
            tile_started = (iter == tile_iter)
            tile_ended = (iter_end >= tile_iter_end)

            if not tile_started:
                # Store partial sum for fixup
                StorePartials(partials[x], accum)
                Signal(flags[x])
            else:
                if not tile_ended:
                    StorePartials(partials[x], accum)
                    Signal(flags[x])
                else:
                    # Accumulate partials from peer CTAs, then store
                    for cta in contributing_ctas(tile_idx):
                        Wait(flags[cta])
                        accum += LoadPartials(partials[cta])
                    StoreTile(C, tile_idx, accum)

            iter = tile_iter_end  # move to next tile
```

## References

- Osama, M., Merrill, D., Cecka, C., Garland, M., Owens, J.D. "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU." PPoPP 2023. arXiv:2301.03598
- Osama, M., et al. "Stream-K++: Adaptive GPU GEMM Kernel Scheduling and Selection using Bloom Filters." ISC 2025. arXiv:2408.11417
- NVIDIA CUTLASS 2.11+ (open-source implementation): https://github.com/NVIDIA/cutlass
- Kerr, A., Merrill, D., Demouth, J., Tran, J. "CUTLASS: Fast Linear Algebra in CUDA C++." 2017.

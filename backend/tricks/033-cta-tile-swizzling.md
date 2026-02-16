# 033: CTA Tile Swizzling for L2 Cache Locality

**Category**: kernel
**Gain type**: efficiency
**Source**: NVIDIA CUTLASS threadblock swizzle; Tschand et al. "SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization" (2025); Locality-Aware CTA Scheduling (ACM TACO 2021)
**Paper**: [papers/cta-swizzle-rasterization.pdf]
**Documented**: 2025-06-15

## Description

CTA tile swizzling is a technique that remaps the assignment of cooperative thread arrays (CTAs) to output tiles in GPU kernels, replacing the default row-major (linear) rasterization order with a spatially-aware order that maximizes L2 cache reuse. The key insight is that adjacent output tiles share input data: in GEMM, tiles in the same row of $\mathbf{C}$ share rows of $\mathbf{A}$, and tiles in the same column share columns of $\mathbf{B}$. If CTAs processing these related tiles execute concurrently (or close in time) on the same streaming multiprocessor cluster, their shared data remains resident in L2 cache rather than being evicted and re-fetched from HBM.

By default, GPU hardware schedules CTAs in row-major order across SMs (or round-robin across accelerator complex dies on multi-die GPUs like AMD MI300X). This scatters spatially-adjacent tiles across distant compute units with separate L2 caches, causing cache thrashing. Swizzling applies a cheap arithmetic remapping of CTA program IDs — typically using bit operations, modular arithmetic, or space-filling curves — so that spatially-neighboring tiles land on the same SM cluster or XCD, dramatically improving L2 hit rates.

The technique is purely a scheduling optimization: it changes **which CTA computes which tile**, not the computation itself. It adds negligible overhead (a few integer operations per CTA at launch) but can yield up to 2.06× speedup and 70% improvement in L2 hit rate for memory-bound kernels.

## Mathematical Form

**Problem setup:**

Consider a tiled GEMM computing $\mathbf{C} = \mathbf{A}\mathbf{B}$ with $\mathbf{A} \in \mathbb{R}^{M \times K}$, $\mathbf{B} \in \mathbb{R}^{K \times N}$, $\mathbf{C} \in \mathbb{R}^{M \times N}$. The output is tiled into a grid of $T_M \times T_N$ tiles where $T_M = \lceil M / B_M \rceil$ and $T_N = \lceil N / B_N \rceil$.

**Default row-major rasterization:**

CTA with linear index $\text{pid}$ maps to tile coordinates:

$$
(i, j) = \left(\left\lfloor \frac{\text{pid}}{T_N} \right\rfloor, \; \text{pid} \bmod T_N\right)
$$

This causes CTAs $0, 1, \ldots, T_N - 1$ (an entire row) to execute before any CTA in the next row. On a GPU with $p$ SMs, only $p$ CTAs execute concurrently, so tiles in the same column of $\mathbf{C}$ (which share columns of $\mathbf{B}$) may be separated by many waves.

**Swizzled rasterization (CUTLASS-style):**

For a swizzle factor $s$ (where the swizzle width is $2^s$ tiles), the remapping uses bit manipulation:

$$
i' = \text{pid}_x \gg s, \quad j' = (\text{pid}_y \ll s) + (\text{pid}_x \;\&\; (2^s - 1))
$$

This groups tiles into $2^s \times 2^s$ super-tiles, ensuring that all tiles within a super-tile are assigned to consecutively-launched CTAs. Tiles within a super-tile share rows of $\mathbf{A}$ and columns of $\mathbf{B}$, maximizing L2 reuse.

**Adaptive swizzle factor selection:**

$$
s = \begin{cases}
3 & \text{if } T_N \geq 8 \text{ and } \lceil T_N / 8 \rceil \geq 6 \\
2 & \text{if } T_N \geq 4 \text{ and } \lceil T_N / 4 \rceil \geq 3 \\
1 & \text{if } T_N \geq 2 \text{ and } \lceil T_N / 2 \rceil \geq 2 \\
0 & \text{otherwise (no swizzle)}
\end{cases}
$$

**Multi-die swizzling (XCD-aware, for AMD MI300X):**

For a GPU with $D$ accelerator complex dies (XCDs), each with its own L2 cache, the swizzle co-locates related tiles on the same XCD:

$$
b_{\text{per\_xcd}} = \left\lceil \frac{T_{\text{total}}}{D} \right\rceil
$$

$$
\text{pid}' = (\text{pid} \bmod D) \cdot b_{\text{per\_xcd}} + \left\lfloor \frac{\text{pid}}{D} \right\rfloor
$$

This ensures $b_{\text{per\_xcd}}$ contiguous tiles are assigned to each XCD, keeping shared data in the same die's L2 cache.

**L2 cache reuse analysis:**

For tile $(i, j)$ computing $\mathbf{C}_{ij} = \sum_k \mathbf{A}_{ik} \mathbf{B}_{kj}$:

- Row sharing: tiles $(i, j)$ and $(i, j')$ both load $\mathbf{A}_{ik}$ for all $k$ — shared data volume $= M_{\text{tile}} \times K$
- Column sharing: tiles $(i, j)$ and $(i', j)$ both load $\mathbf{B}_{kj}$ — shared data volume $= K \times N_{\text{tile}}$

Without swizzling, the probability that two tiles sharing data are co-resident in L2 depends on the ratio of concurrent CTAs to total tiles. With swizzling, tiles within each $2^s \times 2^s$ super-tile are guaranteed to be temporally co-located, giving:

$$
\text{L2 reuse factor} \approx \min(2^s, T_M) + \min(2^s, T_N) - 1
$$

**Key Definitions:**

- CTA — Cooperative Thread Array (a.k.a. threadblock); the unit of work scheduling on an SM
- PID — Program ID; the linear index assigned to each CTA by the GPU scheduler
- XCD — Accelerator Complex Die; a chiplet on multi-die GPUs (e.g., AMD MI300X has 8 XCDs)
- Swizzle factor $s$ — Controls the super-tile width $2^s$; larger = more cache reuse but may cause load imbalance
- Rasterization order — The mapping from linear CTA index to 2D tile coordinates

## Complexity

| Metric | Row-Major Raster | Swizzled Raster |
|--------|-----------------|-----------------|
| L2 hit rate (GEMM) | ~30-60% | ~74-100% |
| L2 hit rate (Softmax) | ~30% | ~96% |
| L2 hit rate (Transpose) | ~55% | ~96% |
| Compute overhead | $0$ | $O(1)$ per CTA (few integer ops) |
| Speedup range | baseline | 1.03× – 2.06× |

**Memory:** No additional memory required. The swizzle is a pure index remapping computed on-the-fly.

**Performance highlights (from SwizzlePerf on AMD MI300X):**

| Kernel | L2 Hit Rate Improvement | Speedup |
|--------|------------------------|---------|
| GEMM | +14% | 1.03× |
| Softmax | +66.5% | 2.06× |
| Transpose | +40.9% | 1.73× |
| Stencil 2D | +69.9% | 1.73× |
| LayerNorm | +8.0% | 1.03× |

Compute-bound kernels (GEMM) see modest speedups despite large L2 improvements because they are not memory-bottlenecked. Memory-bound kernels (softmax, transpose, stencil) see dramatic speedups.

## Applicability

- **GEMM / Linear layers**: All tiled matrix multiplications benefit from swizzled rasterization, especially when tile counts don't fully occupy the GPU (relevant to attention $\mathbf{Q}\mathbf{K}^T$ and $\text{softmax} \times \mathbf{V}$)
- **Softmax / LayerNorm**: Reduction kernels where all chunks of the same row should be co-located on the same XCD/SM cluster for L2 reuse of partial sums
- **Transpose operations**: Reading and writing the same data in different orders benefits greatly from co-locating read and write tiles
- **Stencil / convolution**: Neighboring tiles share halo regions; swizzling keeps halos in L2
- **Attention kernels**: FlashAttention-style kernels can benefit from swizzled scheduling of Q/K/V tile blocks
- **Multi-die GPUs**: Particularly impactful on AMD MI300X (8 XCDs) and future chiplet-based GPU architectures where L2 is physically distributed

## Limitations

- **Compute-bound kernels see limited speedup**: If a kernel is already compute-bound (high arithmetic intensity), improving L2 hit rate has marginal effect on wall-clock time
- **Problem-size dependent**: The optimal swizzle factor depends on the tile grid dimensions; over-swizzling can create no-op CTAs or load imbalance
- **Architecture-specific**: The optimal swizzle pattern depends on the number of SMs, L2 cache size, and die topology — different for NVIDIA A100 vs H100 vs AMD MI300X
- **Interaction with persistent kernels**: When combined with persistent kernel scheduling (e.g., Stream-K), the swizzle must be coordinated with the tile scheduler to avoid conflicts
- **Not composable across kernels**: Swizzling optimizes L2 reuse within a single kernel launch; it does not help with inter-kernel data reuse (that requires kernel fusion)

## Implementation Notes

```python
import triton
import triton.language as tl

# GEMM kernel with CTA tile swizzling (CUTLASS-style)
@triton.jit
def matmul_swizzled(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SWIZZLE_LOG: tl.constexpr,  # log2 of swizzle width (e.g., 3 for 8x8)
):
    # Standard linear PID
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks_m = tl.cdiv(M, BLOCK_M)

    # --- Swizzle remapping ---
    # Group tiles into 2^SWIZZLE_LOG × 2^SWIZZLE_LOG super-tiles
    swizzle_width = 1 << SWIZZLE_LOG
    group_id = pid // (swizzle_width * num_blocks_n)
    first_pid_m = group_id * swizzle_width
    group_size_m = min(num_blocks_m - first_pid_m, swizzle_width)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % (group_size_m * num_blocks_n)) // group_size_m

    # Now pid_m, pid_n index into the tile grid with improved locality
    # Tiles within each super-tile are launched consecutively,
    # maximizing L2 cache reuse for shared A rows and B columns

    # Standard tiled GEMM computation follows...
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + k + tl.arange(0, BLOCK_K)[None, :])
        b = tl.load(B + (k + tl.arange(0, BLOCK_K)[:, None]) * N + offs_n[None, :])
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


# XCD-aware swizzling for multi-die GPUs (AMD MI300X)
@triton.jit
def swizzle_xcd(pid, num_blocks, NUM_XCDS: tl.constexpr):
    """Remap PID so contiguous blocks land on the same XCD."""
    blocks_per_xcd = tl.cdiv(num_blocks, NUM_XCDS)
    # Interleave → contiguous: blocks 0,D,2D,... go to XCD 0, etc.
    new_pid = (pid % NUM_XCDS) * blocks_per_xcd + (pid // NUM_XCDS)
    return new_pid
```

## References

- Tschand, A., Awad, M., Swann, R., Ramakrishnan, K., Ma, J., Lowery, K., Dasika, G., Reddi, V.J. "SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization." 2025. arXiv:2508.20258
- NVIDIA CUTLASS. `threadblock_swizzle.h` — Threadblock rasterization strategies. https://github.com/NVIDIA/cutlass
- Khairy, M., et al. "Locality-Aware CTA Scheduling for Gaming Applications." ACM TACO, 2021.
- Gordić, A. "Inside NVIDIA GPUs: Anatomy of high performance matmul kernels." 2024. (Hilbert-curve CTA scheduling discussion)
- NVIDIA. "CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design." NVIDIA Developer Blog, 2024.

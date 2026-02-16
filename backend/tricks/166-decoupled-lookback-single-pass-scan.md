# 166: Decoupled Look-back Single-Pass Scan

**Category**: parallelization
**Gain type**: efficiency
**Source**: Merrill & Garland (NVIDIA Technical Report NVR-2016-002, 2016)
**Paper**: [papers/decoupled-lookback-single-pass-scan.pdf]
**Documented**: 2026-02-15

## Description

The decoupled look-back method is the foundational single-pass parallel prefix scan algorithm used in NVIDIA's CUB library and underpinning virtually all high-performance GPU scan implementations today. It achieves near-optimal $\sim 2n$ data movement (vs. $\sim 3n$--$4n$ for reduce-then-scan or scan-then-propagate), approaching the throughput of a bare `memcpy` operation.

The core insight is a **decoupled look-back** strategy: rather than waiting on a single predecessor (as in chained-scan) or requiring global synchronization barriers (as in reduce-then-scan), each thread block independently **looks back** through a window of predecessors at progressively increasing distances, accumulating partial aggregates until it encounters a predecessor that has already computed its full inclusive prefix. This performs bounded redundant work to **decouple** local computation from the latencies of global prefix propagation.

This is directly relevant to SSM recurrences and linear attention: the chunkwise parallel scan pattern (trick 026) uses exactly this kind of inter-chunk prefix propagation. The decoupled look-back mechanism is the GPU-optimal way to implement the global aggregation stage, and is the production algorithm behind CUB's `DeviceScan`, which Mamba and other SSM implementations rely on.

## Mathematical Form

**Problem:** Given input $x = [x_0, x_1, \ldots, x_{n-1}]$ and an associative binary operator $\oplus$, compute the inclusive prefix scan:

$$
y_i = x_0 \oplus x_1 \oplus \cdots \oplus x_i, \quad \forall\, i \in \{0, \ldots, n-1\}
$$

**Partition into tiles:** Divide input into $G$ tiles of size $b$, assigned to thread blocks $B_0, B_1, \ldots, B_{G-1}$.

**Per-partition descriptor:** Each partition $j$ maintains a status descriptor with three fields:

- $\text{aggregate}_j$ — the reduction of all elements within partition $j$: $\text{aggregate}_j = x_{jb} \oplus x_{jb+1} \oplus \cdots \oplus x_{(j+1)b-1}$
- $\text{inclusive\_prefix}_j$ — the inclusive prefix across all partitions $0$ through $j$
- $\text{status\_flag}_j \in \{X, A, P\}$ — invalid, aggregate available, or prefix available

**Algorithm (per thread block $B_j$):**

1. **Local reduce:** Compute $\text{aggregate}_j$ via block-wide reduction. Post to global memory, set $\text{status\_flag}_j \leftarrow A$. (First partition: copy to $\text{inclusive\_prefix}_j$, set flag $\leftarrow P$, skip to step 3.)

2. **Decoupled look-back:** Initialize $\text{exclusive\_prefix} \leftarrow \text{identity}$. For predecessor $p = j-1, j-2, \ldots$:
   - If $\text{status\_flag}_p = X$: spin-wait (poll).
   - If $\text{status\_flag}_p = A$: accumulate $\text{exclusive\_prefix} \leftarrow \text{aggregate}_p \oplus \text{exclusive\_prefix}$; continue to $p-1$.
   - If $\text{status\_flag}_p = P$: $\text{exclusive\_prefix} \leftarrow \text{inclusive\_prefix}_p \oplus \text{exclusive\_prefix}$; **terminate** look-back.

3. **Compute and post inclusive prefix:**
$$
\text{inclusive\_prefix}_j = \text{exclusive\_prefix} \oplus \text{aggregate}_j
$$
Set $\text{status\_flag}_j \leftarrow P$.

4. **Local scan with seed:** Perform block-wide scan of partition $j$'s elements, seeded with $\text{exclusive\_prefix}_j$.

**Key invariant:** Once any partition posts status $P$, all partitions before it are guaranteed to have posted at least status $A$. This ensures the look-back traverses at most $O(p)$ predecessors (where $p$ is the number of physical processors), giving constant bounded redundant work per partition.

## Complexity

| Metric | Reduce-then-Scan | Scan-then-Propagate | Chained-Scan | **Decoupled Look-back** |
|--------|-----------------|--------------------|--------------|-----------------------|
| Kernel launches | 3 | $O(\log G)$ | 1 | **1** |
| Global data movement | $\sim 3n$ | $\sim 4n$ | $\sim 2n$ | $\sim 2n$ |
| Global barriers | 2 | $O(\log G)$ | 0 (serial chain) | **0** |
| Single-pass | No | No | Yes | **Yes** |
| In-place compaction | No | No | No | **Yes** |

**Work:** $O(n)$ total — each element is read once, written once, plus $O(p)$ redundant look-back reads (constant per partition, independent of $n$).

**Depth (span):** $O(b + p)$ where $b$ = tile size, $p$ = number of active thread blocks. The look-back chain is bounded by $p$ hops, each of constant cost.

**Memory:** Only $O(G)$ = $O(n/b)$ extra words for partition descriptors (3 words per partition: aggregate, inclusive_prefix, status_flag).

**Measured throughput (CUB on NVIDIA GPUs):**

| GPU | Peak Throughput | vs. memcpy |
|-----|----------------|------------|
| Tesla M40 (Maxwell) | 31.0 B items/sec | ~100% |
| Tesla K40 (Kepler) | 26.8 B items/sec | ~100% |
| Tesla C2050 (Fermi) | 14.4 B items/sec | ~100% |

**CUB speedup over alternatives (harmonic mean across sizes):**

| vs. StreamScan | vs. MGPU | vs. Thrust |
|----------------|----------|------------|
| 1.60x | 1.19x | 2.80x |

## Applicability

- **SSM chunkwise parallelism:** The inter-chunk aggregation in Mamba/S5/S6 chunkwise parallel scan directly uses this pattern. Each chunk computes a local scan, then the global prefix propagation across chunks is exactly the decoupled look-back mechanism. CUB's `DeviceScan` (which implements this algorithm) is the default backend.
- **Linear attention cumulative sums:** The $S_t = \lambda_t S_{t-1} + v_t k_t^\top$ recurrence parallelized via scan relies on this for GPU-efficient global aggregation.
- **Top-$p$ / top-$k$ sampling:** Prefix sums for cumulative probability computation in LLM inference use CUB scan.
- **In-place compaction algorithms:** `select-if`, `partition-if`, `reduce-by-key`, `run-length-encode` — all built on this single-pass scan primitive in CUB. Throughput improvements of 4.1--7.1x over Thrust equivalents.
- **Any parallel scan on NVIDIA GPUs:** This is the de facto standard; all production scan workloads flow through this algorithm.

## Limitations

- **Requires forward-progress guarantees (FPG):** The spin-waiting in the look-back phase assumes that predecessor thread blocks will eventually make progress. NVIDIA GPUs provide this guarantee through hardware scheduling, but **non-NVIDIA hardware** (AMD, Intel, WebGPU) may not. This is the motivation for the Decoupled Fallback variant (Smith et al., SPAA 2025).
- **Non-deterministic for floating-point:** The look-back accumulates aggregates in a data-dependent order that varies between runs. For pseudo-associative operators (floating-point addition), results are not bitwise reproducible across runs. The paper explicitly notes this as a trade-off.
- **Memory fence overhead:** Each partition descriptor update requires a memory fence to ensure consistent visibility across thread blocks. On architectures where fences are expensive, this can add latency. The paper describes a fence-free optimization by packing status flag + value into a single 64-bit atomic.
- **Serial dependency chain in look-back:** While bounded by $O(p)$, the look-back is inherently serial within a single thread block. This is mitigated by **parallelized look-back**: a full warp simultaneously inspects $t$ predecessor partitions, reducing the serial chain to $O(p/t)$ rounds.
- **Tile size constraints:** Performance is optimized when the tile's working set fits in on-chip memory (registers + shared memory). Very large tiles improve I/O efficiency but may spill to HBM.

## Implementation Notes

```python
import torch
from enum import IntEnum

class StatusFlag(IntEnum):
    INVALID = 0      # X: no data available
    AGGREGATE = 1    # A: partition aggregate available
    PREFIX = 2       # P: inclusive prefix available

def decoupled_lookback_scan(x, tile_size=2048):
    """
    Decoupled look-back single-pass prefix scan (simplified simulation).
    In practice, this runs as a single CUDA kernel with thread blocks
    processing tiles in dynamically-assigned order.
    """
    n = len(x)
    num_tiles = (n + tile_size - 1) // tile_size

    # Partition descriptors (in global memory)
    status = torch.zeros(num_tiles, dtype=torch.long)        # flag
    aggregates = torch.zeros(num_tiles, dtype=x.dtype)       # aggregate
    inc_prefixes = torch.zeros(num_tiles, dtype=x.dtype)     # inclusive_prefix

    output = torch.zeros_like(x)

    for j in range(num_tiles):  # In GPU: thread blocks run in parallel
        start = j * tile_size
        end = min(start + tile_size, n)
        tile = x[start:end]

        # Step 1: Compute partition-wide aggregate (reduction)
        agg = tile.sum()
        aggregates[j] = agg
        # memory_fence()

        if j == 0:
            # First partition: aggregate IS the inclusive prefix
            inc_prefixes[j] = agg
            status[j] = StatusFlag.PREFIX
            exclusive_prefix = 0.0
        else:
            status[j] = StatusFlag.AGGREGATE

            # Step 2: Decoupled look-back
            exclusive_prefix = 0.0
            p = j - 1
            while True:
                while status[p] == StatusFlag.INVALID:
                    pass  # spin-wait (GPU: warp polls global memory)

                if status[p] == StatusFlag.AGGREGATE:
                    exclusive_prefix = aggregates[p] + exclusive_prefix
                    p -= 1  # look further back
                elif status[p] == StatusFlag.PREFIX:
                    exclusive_prefix = inc_prefixes[p] + exclusive_prefix
                    break   # found a complete prefix, done!

            # Step 3: Post inclusive prefix
            inc_prefixes[j] = exclusive_prefix + agg
            # memory_fence()
            status[j] = StatusFlag.PREFIX

        # Step 4: Local scan seeded with exclusive prefix
        output[start:end] = exclusive_prefix + torch.cumsum(tile, 0)

    return output
```

**GPU implementation key details:**
- Thread blocks obtain partition indices via `atomicAdd` on a global counter (ensures ordered scheduling)
- Status flag + value packed into single 64-bit word for fence-free atomic updates
- Look-back parallelized across warp lanes: each lane inspects a different predecessor simultaneously
- Block-wide scan uses hybrid Brent-Kung / Kogge-Stone strategy tuned per architecture

## References

- Merrill, D. and Garland, M. (2016). Single-pass Parallel Prefix Scan with Decoupled Look-back. NVIDIA Technical Report NVR-2016-002.
- Merrill, D. (2013). CUB: A library of warp-wide, block-wide, and device-wide GPU parallel primitives. NVIDIA Research.
- Smith, T. et al. (2025). Decoupled Fallback: A Portable Single-Pass GPU Scan. Proc. 37th ACM SPAA.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.
- Brent, R.P. and Kung, H.T. (1982). A Regular Layout for Parallel Adders. IEEE Trans. Computers.
- Yan, S. et al. (2013). StreamScan: Fast Scan Algorithms for GPUs without Global Barrier Synchronization. Proc. ACM PPoPP.

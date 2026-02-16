# 173: Decoupled Fallback — Portable Single-Pass GPU Scan

**Category**: parallelization
**Gain type**: flexibility
**Source**: Smith, Levien & Owens (SPAA 2025)
**Paper**: [papers/decoupled-fallback-scan.pdf]
**Documented**: 2026-02-15

## Description

The Decoupled Look-back algorithm (trick 166) is the fastest known single-pass parallel prefix scan, achieving near-memcpy throughput on NVIDIA GPUs. It is the backbone of CUB's `DeviceScan`, which underpins Mamba, linear attention, and virtually all production scan workloads on NVIDIA hardware. However, it has a critical portability limitation: it relies on **forward-progress guarantees (FPG)** — the property that all launched thread blocks will eventually make progress regardless of scheduling order. Without FPG, a thread block spinning in the look-back phase may wait forever for a predecessor that the GPU scheduler never runs, causing **permanent starvation** (deadlock).

This guarantee is provided by NVIDIA's hardware scheduler but is **not available** on AMD GPUs, Intel GPUs, Apple Metal, WebGPU, or Vulkan (which makes no forward-progress guarantees in its specification). This forces developers targeting non-NVIDIA platforms to fall back to slower multi-pass algorithms like Reduce-then-Scan ($\sim 3n$ data movement vs. $\sim 2n$ for single-pass), sacrificing 30-50% performance.

**Decoupled Fallback** solves this by augmenting the Decoupled Look-back with a **work-stealing fallback mechanism**: when a thread block detects that a predecessor is stalled (by exceeding a configurable spin count), it proactively computes that predecessor's reduction itself rather than waiting indefinitely. Multiple thread blocks may attempt the same fallback simultaneously, but an **atomic compare-and-swap** ensures exactly one broadcasts the result. This transforms the indefinite spin into bounded redundant work, guaranteeing termination without FPG while maintaining near-identical throughput to Decoupled Look-back when FPG is available.

The result is a single-pass scan algorithm that is **portable across all GPU architectures** (CUDA, Metal, Vulkan, WebGPU, D3D12) while matching the throughput of the NVIDIA-specific Decoupled Look-back.

## Mathematical Form

**Problem:** Given input $x = [x_0, x_1, \ldots, x_{n-1}]$ and an associative binary operator $\oplus$, compute:

$$
y_i = x_0 \oplus x_1 \oplus \cdots \oplus x_i, \quad \forall\, i \in \{0, \ldots, n-1\}
$$

**Partition:** Divide input into $G$ tiles of size $b$, assigned to thread blocks $B_0, B_1, \ldots, B_{G-1}$ (tile indices assigned via atomic counter for ordered scheduling).

**Per-partition descriptor (same as Decoupled Look-back):**

- $\text{aggregate}_j$ — reduction of all elements within partition $j$
- $\text{inclusive\_prefix}_j$ — prefix across partitions $0$ through $j$
- $\text{status\_flag}_j \in \{X, A, P\}$ — invalid, aggregate ready, prefix ready

**Decoupled Fallback Algorithm (per thread block $B_j$):**

1. **Local reduce:** Compute $\text{aggregate}_j = x_{jb} \oplus x_{jb+1} \oplus \cdots \oplus x_{(j+1)b-1}$. Post to global memory, set $\text{status\_flag}_j \leftarrow A$.

2. **Look-back with fallback:** Initialize $\text{exclusive\_prefix} \leftarrow \text{identity}$, set $\text{spin\_count} \leftarrow 0$. For predecessor $p = j-1, j-2, \ldots$:

$$
\textbf{while } \text{status\_flag}_p = X \textbf{ do:}
$$
$$
\quad \text{spin\_count} \leftarrow \text{spin\_count} + 1
$$
$$
\quad \textbf{if } \text{spin\_count} > \text{MAX\_SPIN} \textbf{ then:}
$$
$$
\qquad \text{FALLBACK}(p) \quad \text{// work-steal: compute predecessor's reduction}
$$

- If $\text{status\_flag}_p = A$: accumulate $\text{exclusive\_prefix} \leftarrow \text{aggregate}_p \oplus \text{exclusive\_prefix}$; continue to $p-1$
- If $\text{status\_flag}_p = P$: $\text{exclusive\_prefix} \leftarrow \text{inclusive\_prefix}_p \oplus \text{exclusive\_prefix}$; **stop**

3. **Post inclusive prefix:**
$$
\text{inclusive\_prefix}_j = \text{exclusive\_prefix} \oplus \text{aggregate}_j
$$
Set $\text{status\_flag}_j \leftarrow P$.

4. **Local scan seeded** with $\text{exclusive\_prefix}_j$.

**FALLBACK procedure for stalled tile $p$:**

```
FALLBACK(p):
    // Compute tile p's reduction ourselves (redundant work)
    local_agg = reduce(x[p*b : (p+1)*b])

    // Atomically try to be the one to post the result
    success = atomic_compare_and_swap(
        status_flag[p],
        expected=X,        // Only post if still invalid
        desired=A,
        value=local_agg
    )
    // If another thread block already posted, that's fine — use their result
    // Reset spin count and continue look-back from status_flag[p]
```

**Key Properties:**

1. **Bounded redundant work:** Each fallback computes at most one tile's reduction ($O(b)$ work). At most $O(G)$ fallbacks can occur across the entire scan, giving total redundant work $O(n)$ — same order as the useful work.

2. **Exactly-once broadcast:** The atomic CAS ensures that even if multiple thread blocks simultaneously perform fallback on the same stalled tile, only one posts its aggregate. All others observe the posted result on their next read.

3. **Graceful degradation:** On hardware with FPG (NVIDIA), the fallback path is never triggered — the spin count threshold is never exceeded, and the algorithm behaves identically to Decoupled Look-back. On hardware without FPG, the fallback adds a small constant overhead per stalled tile.

4. **No global barriers:** Like Decoupled Look-back, the algorithm requires zero global synchronization barriers, enabling fully asynchronous single-pass execution.

## Complexity

| Metric | Reduce-then-Scan | Decoupled Look-back | **Decoupled Fallback** |
|--------|-----------------|--------------------|-----------------------|
| Kernel launches | 3 | 1 | **1** |
| Global data movement | $\sim 3n$ | $\sim 2n$ | $\sim 2n$ |
| Global barriers | 2 | 0 | **0** |
| Single-pass | No | Yes | **Yes** |
| Requires FPG | No | **Yes** | **No** |
| Portable (WebGPU/Vulkan) | Yes | No | **Yes** |
| Redundant work (worst case) | 0 | $O(p)$ look-back | $O(n)$ fallback + $O(p)$ look-back |
| Redundant work (FPG hardware) | 0 | $O(p)$ | $O(p)$ (same — fallback never triggers) |

**Work:** $O(n)$ total. Each element is read once, written once. Redundant fallback work is bounded by $O(n)$ worst case, but in practice is negligible (most tiles complete before any fallback is needed).

**Depth (span):** $O(b + p)$ where $b$ = tile size, $p$ = number of active thread blocks. Identical to Decoupled Look-back.

**Memory:** $O(G) = O(n/b)$ extra words for partition descriptors (same as Decoupled Look-back: aggregate, prefix, status flag per tile).

**Performance (near-identical to Decoupled Look-back):**

On NVIDIA hardware (where FPG is available), Decoupled Fallback performs identically to Decoupled Look-back since the fallback path is never triggered. On AMD, Apple, and other GPUs without FPG, it achieves performance very close to what Decoupled Look-back achieves on NVIDIA — dramatically faster than the Reduce-then-Scan fallback previously required.

**Comparison to alternatives on non-FPG hardware:**

| Method | Throughput relative to memcpy | Portable |
|--------|------------------------------|----------|
| Reduce-then-Scan (3 passes) | ~60-70% | Yes |
| Tree reduction (multi-pass) | ~50-65% | Yes |
| Decoupled Look-back | **Deadlocks** | No |
| **Decoupled Fallback** | **~95-100%** | **Yes** |

## Applicability

- **Cross-platform SSM inference:** Deploying Mamba/S5 models on non-NVIDIA hardware (AMD MI300X, Apple M-series, Intel Arc) requires a portable scan primitive. Decoupled Fallback enables single-pass scan on all these platforms without falling back to slow multi-pass algorithms. This is critical for on-device inference (Apple Neural Engine, Qualcomm Hexagon) where CUDA is unavailable.

- **WebGPU ML inference:** Browser-based ML inference (e.g., WebLLM, ONNX Runtime Web) runs on WebGPU, which inherits Vulkan/Metal's lack of FPG. Decoupled Fallback enables efficient scan-based operations (cumulative sums for linear attention, prefix sums for sampling) in the browser.

- **Multi-vendor GPU clusters:** Training runs spanning NVIDIA + AMD GPUs (increasingly common in cloud providers) need portable scan primitives. Decoupled Fallback provides a single implementation that works optimally on both.

- **LLM sampling (top-p/top-k):** Prefix sums for cumulative probability computation in nucleus sampling use scan. On NVIDIA this goes through CUB; on other platforms, Decoupled Fallback provides the equivalent single-pass performance.

- **Parallel sort and compaction:** Radix sort, stream compaction (`select-if`), and run-length encoding all reduce to prefix scan. Decoupled Fallback makes these portable across GPU vendors.

- **Vulkan compute pipelines:** Vulkan-based renderers (e.g., Vello by the same authors) need efficient GPU prefix sums for path rendering. Decoupled Fallback is already integrated into the Vello rendering pipeline.

## Limitations

- **Redundant computation on non-FPG hardware:** When fallbacks occur, multiple thread blocks may redundantly compute the same tile's reduction before the CAS resolves. This wastes a small amount of compute, though the total redundant work is bounded.

- **Spin count threshold tuning:** The MAX_SPIN parameter trades off latency for fallback frequency. Too low: unnecessary fallbacks on hardware that is just slow to schedule. Too high: delayed fallback response to actual stalls. The paper tunes this per-architecture (typically thousands of cycles).

- **Atomic CAS overhead:** The fallback mechanism requires device-scope atomic compare-and-swap, which has varying costs across GPU architectures. On some hardware (notably older AMD GCN), device-scope atomics are expensive. Modern RDNA3 and Apple M-series have efficient atomics.

- **Still requires basic atomic semantics:** While FPG is not needed, the algorithm does require acquire/release atomic semantics at device scope and atomic CAS. Some very restricted shader environments (certain WebGPU configurations, Metal with limited buffer modes) may not provide these.

- **Does not address the non-determinism issue:** Like Decoupled Look-back, the accumulation order in the look-back phase is data-dependent and non-deterministic for floating-point operations. Results are not bitwise reproducible across runs.

- **Limited to additive-style scans:** The fallback mechanism computes a tile's reduction from scratch. For expensive associative operators (e.g., matrix products in SSM recurrences), the redundant fallback computation may be costly. Most effective for scalar prefix sums (addition, max, min).

## Implementation Notes

```python
import torch
from enum import IntEnum

class StatusFlag(IntEnum):
    INVALID = 0       # X: no data posted yet
    AGGREGATE = 1     # A: partition aggregate available
    PREFIX = 2        # P: inclusive prefix available

MAX_SPIN = 4096  # Tuned per architecture

def decoupled_fallback_scan(x, tile_size=2048):
    """
    Decoupled Fallback single-pass prefix scan.
    Portable to all GPU architectures (no FPG required).

    Key difference from Decoupled Look-back (trick 166):
    - When spin count exceeds MAX_SPIN, performs fallback reduction
    - Atomic CAS ensures exactly one thread block posts fallback result
    """
    n = len(x)
    num_tiles = (n + tile_size - 1) // tile_size

    # Partition descriptors (in global/device memory)
    status = torch.zeros(num_tiles, dtype=torch.long)
    aggregates = torch.zeros(num_tiles, dtype=x.dtype)
    inc_prefixes = torch.zeros(num_tiles, dtype=x.dtype)
    output = torch.zeros_like(x)

    for j in range(num_tiles):  # On GPU: thread blocks run in parallel
        start = j * tile_size
        end = min(start + tile_size, n)
        tile = x[start:end]

        # Step 1: Compute partition aggregate
        agg = tile.sum()
        aggregates[j] = agg
        # memory_fence()  // device-scope release

        if j == 0:
            inc_prefixes[j] = agg
            status[j] = StatusFlag.PREFIX
            exclusive_prefix = 0.0
        else:
            status[j] = StatusFlag.AGGREGATE

            # Step 2: Look-back with fallback
            exclusive_prefix = 0.0
            p = j - 1
            while True:
                spin_count = 0
                while status[p] == StatusFlag.INVALID:
                    spin_count += 1
                    if spin_count > MAX_SPIN:
                        # === FALLBACK: work-steal tile p's reduction ===
                        p_start = p * tile_size
                        p_end = min(p_start + tile_size, n)
                        fallback_agg = x[p_start:p_end].sum()

                        # Atomic CAS: only post if still INVALID
                        if status[p] == StatusFlag.INVALID:
                            aggregates[p] = fallback_agg
                            status[p] = StatusFlag.AGGREGATE
                        # Reset spin and re-read
                        spin_count = 0
                        break

                if status[p] == StatusFlag.AGGREGATE:
                    exclusive_prefix = aggregates[p] + exclusive_prefix
                    p -= 1
                elif status[p] == StatusFlag.PREFIX:
                    exclusive_prefix = inc_prefixes[p] + exclusive_prefix
                    break

            # Step 3: Post inclusive prefix
            inc_prefixes[j] = exclusive_prefix + agg
            # memory_fence()  // device-scope release
            status[j] = StatusFlag.PREFIX

        # Step 4: Local scan seeded with exclusive prefix
        output[start:end] = exclusive_prefix + torch.cumsum(tile, 0)

    return output
```

**GPU implementation details:**

- **Tile index assignment:** `atomicAdd` on a global counter ensures ordered scheduling (same as Decoupled Look-back)
- **Status packing:** Status flag + aggregate packed into a single 64-bit word for atomic CAS. On 32-bit atomic hardware (WebGPU), use separate flag and value with careful ordering
- **Parallelized look-back:** Full warp inspects multiple predecessors simultaneously, reducing serial chain to $O(p/32)$ rounds
- **Fallback parallelization:** The entire warp/workgroup collaboratively computes the fallback reduction (not just one thread), ensuring the fallback is fast
- **Spin count calibration:** Typically set to $\sim$4000-8000 cycles on Apple M-series, $\sim$2000-4000 on AMD RDNA3. Should never trigger on NVIDIA (set higher than any realistic scheduling delay)

**Integration with production frameworks:**

- The algorithm is implemented in Vello (GPU vector graphics renderer) for Metal/Vulkan/WebGPU
- Available in the GPUPrefixSums repository with CUDA, D3D12, Unity, and WGPU backends
- Can serve as a drop-in replacement for CUB's `DeviceScan` on non-NVIDIA platforms

## References

- Smith, T., Levien, R., and Owens, J.D. (2025). Decoupled Fallback: A Portable Single-Pass GPU Scan. Proc. 37th ACM Symposium on Parallelism in Algorithms and Architectures (SPAA). ACM.
- Merrill, D. and Garland, M. (2016). Single-pass Parallel Prefix Scan with Decoupled Look-back. NVIDIA Technical Report NVR-2016-002.
- Levien, R. (2021). Prefix sum on portable compute shaders. Blog post.
- GPUPrefixSums repository: https://github.com/b0nes164/GPUPrefixSums
- Vello GPU renderer: https://github.com/linebender/vello
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.

# 061: Horizontal Kernel Fusion (HFUSE)

**Category**: kernel
**Gain type**: efficiency
**Source**: Li, Zheng, Pekhimenko, Long — "Automatic Horizontal Fusion for GPU Kernels" (CGO 2022)
**Paper**: [papers/horizontal-kernel-fusion-hfuse.pdf]
**Documented**: 2026-02-15

## Description

Horizontal kernel fusion is a GPU optimization technique that merges two **independent** (non-data-dependent) kernels into a single kernel launch, where each kernel's threads run side-by-side within the same thread block. Unlike standard vertical fusion — which chains dependent kernels to eliminate intermediate HBM round-trips — horizontal fusion targets a completely different bottleneck: **instruction latency hiding via thread-level parallelism (TLP)**.

The key insight is that GPU warp schedulers can interleave instructions from different warps to hide latency. When two kernels with complementary resource profiles (e.g., one memory-intensive and one compute-intensive) run in the same thread block, the warp scheduler can issue compute instructions from one kernel while the other kernel's warps are stalled waiting for memory. This increases issue slot utilization — the percentage of cycles where at least one warp is active — reducing the total time both kernels occupy the GPU.

In the horizontally fused kernel, the thread space is **partitioned** between the two original kernels: threads $[0, d_1)$ execute kernel $K_1$'s code, and threads $[d_1, d_1 + d_2)$ execute kernel $K_2$'s code. Branch statements dispatch each thread to the correct kernel's instructions based on its thread ID. Synchronization barriers (`__syncthreads()`) are replaced with partial barriers (`bar.sync` PTX instructions) that only synchronize threads belonging to the same original kernel, preserving correctness without blocking cross-kernel threads.

HFUSE is a source-to-source CUDA compiler that automates this process, including an automatic profiling step to search for the optimal thread space partition and register bound.

## Mathematical Form

**Thread space partition:**

Given two kernels $K_1$ and $K_2$ with block dimensions $d_1$ and $d_2$, the fused kernel $F$ has block dimension $d_0 = d_1 + d_2$. For a thread with global index $\text{tid}$ in the fused kernel:

$$
\text{kernel\_assignment}(\text{tid}) = \begin{cases} K_1 & \text{if } \text{tid} < d_1 \\ K_2 & \text{if } d_1 \leq \text{tid} < d_1 + d_2 \end{cases}
$$

The original thread IDs are remapped:

$$
\text{tid}_1 = \text{tid}, \quad \text{tid}_2 = \text{tid} - d_1
$$

**Issue slot utilization model:**

For two kernels with individual issue slot utilizations $I_{k_1}, I_{k_2}$ and elapsed cycles $C_{k_1}, C_{k_2}$, the average utilization when run natively in parallel is:

$$
I_{k_1 + k_2} = \frac{I_{k_1} \cdot C_{k_1} + I_{k_2} \cdot C_{k_2}}{C_{k_1} + C_{k_2}}
$$

Horizontal fusion can exceed this because warps from both kernels coexist **within the same SM**, enabling the warp scheduler to interleave instructions at cycle granularity rather than kernel-launch granularity.

**Optimal partition search:**

The search algorithm iterates over candidate partitions $d_1 \in \{128, 256, 384, \ldots, d_0\}$ and for each generates two fused kernel variants (with and without a register bound $r_0$):

$$
r_0 = \frac{\text{SMNRegs}}{b_0 \cdot d_0}
$$

where:
- $\text{SMNRegs}$ = registers per SM (64K for Pascal/Volta)
- $b_0 = \min\!\left(\min(b_1, b_2),\; \frac{\text{SMShMem}}{\text{ShMem}(F)},\; \frac{\text{SMNThreads}}{d_0}\right)$ = target concurrent blocks
- $b_1 = \frac{\text{SMNRegs}}{d_1 \cdot \text{NRegs}(S_1)}$, $b_2 = \frac{\text{SMNRegs}}{d_2 \cdot \text{NRegs}(S_2)}$ = blocks per SM for each original kernel

The partition and register bound that minimize elapsed cycles are selected.

**Partial barrier replacement:**

Each `__syncthreads()` in kernel $K_i$ is replaced with:

$$
\texttt{bar.sync } i, \; d_i
$$

where $i$ is the barrier ID (1 for $K_1$, 2 for $K_2$) and $d_i$ is the number of threads participating. This synchronizes only threads from the corresponding original kernel.

**Key Definitions:**

- Issue slot utilization — Percentage of GPU cycles where at least one warp is active (not stalled)
- MemInst stall — Percentage of stalls caused by waiting for memory instructions
- Occupancy — Ratio of active warps to maximum warps supported per SM
- Thread space partition — The assignment of fused kernel threads to original kernel code paths
- Partial barrier — PTX `bar.sync` instruction that synchronizes a subset of threads in a block

## Complexity

| Metric | Native (parallel streams) | Horizontal Fusion |
|--------|--------------------------|-------------------|
| Kernel launches | $2$ | $1$ |
| Issue slot utilization | $\frac{I_1 C_1 + I_2 C_2}{C_1 + C_2}$ | Higher (intra-SM interleaving) |
| Register pressure | $\max(r_1, r_2)$ per SM | $r_1 + r_2$ per block (may reduce occupancy) |
| Shared memory | $\max(s_1, s_2)$ per SM | $s_1 + s_2$ per block |

**Memory:** No additional HBM traffic — horizontal fusion does not change memory access patterns, only scheduling. However, increased per-block register/shared memory usage may reduce occupancy.

**Performance (from paper, representative pairs with execution time ratio ~1:1):**

| Kernel Pair | Speedup (1080Ti) | Speedup (V100) |
|-------------|-----------------|----------------|
| Batchnorm + Hist | 33.4% | 15.8% |
| Hist + Maxpool | 32.5% | 56.0% |
| Hist + Upsample | 51.4% | 5.7% |
| Blake256 + Ethash | 47.4% | 64.7% |
| Ethash + SHA256 | 35.1% | 44.1% |
| Im2Col + Maxpool | 25.3% | -7.5% |

Average speedup across favorable pairs: 12.4%–55.1% (1080Ti), 2.5%–60.8% (V100).

## Applicability

- **Transformer MLP + attention overlap**: In architectures with parallel branches (e.g., PaLM-style parallel attention + FFN), the two independent computations can be horizontally fused to share SM resources
- **Multi-head attention**: Independent per-head kernels (especially for small head dimensions that under-occupy SMs) can be horizontally fused
- **Normalization + histogram/diagnostic kernels**: Training diagnostics (gradient histograms, activation statistics) running alongside normalization kernels benefit strongly
- **MoE expert computation**: Independent expert forward passes in Mixture-of-Experts layers are natural candidates — each expert's kernel may individually under-occupy the GPU
- **Any pair of concurrent, resource-complementary kernels**: Most beneficial when one kernel is memory-intensive (high MemInst stall) and the other is compute-intensive (high issue slot utilization), enabling the warp scheduler to interleave their instructions

## Limitations

- **Occupancy reduction**: The fused kernel requires the combined registers and shared memory of both kernels per block. If this exceeds SM limits, fewer blocks execute concurrently, potentially negating the TLP gains. Register bounding can mitigate this at the cost of register spilling.
- **Same-resource kernels**: Fusing two compute-intensive or two memory-intensive kernels provides little benefit (warp scheduler cannot hide latencies when all warps stall on the same resource). The Blake256+Blake2B and Blake256+SHA256 pairs show this clearly with negative speedups.
- **Barrier complexity**: Replacing `__syncthreads()` with partial barriers requires PTX-level manipulation. The `bar.sync` instruction supports only 16 barrier IDs per block, limiting fusion to at most 16 kernels (practically 2).
- **Thread space search cost**: The automatic profiling requires compiling and benchmarking multiple partition configurations, adding to build time.
- **Block dimension alignment**: Thread partitions must be multiples of 128 (4 warps) for efficient warp scheduling; irregular partitions cause performance degradation.
- **No support for recursive kernels**: HFUSE inlines all function calls and cannot handle recursion (rare in GPU kernels but a hard limitation).

## Implementation Notes

```python
# Pseudocode for horizontal kernel fusion (HFUSE approach)

def generate_horizontal_fused_kernel(K1, K2, d1, d2):
    """
    Fuse two independent CUDA kernels horizontally.
    K1 gets threads [0, d1), K2 gets threads [d1, d1+d2).
    """
    d0 = d1 + d2  # fused block dimension

    # Prologue: compute original thread IDs and block dims
    prologue = f"""
    int global_tid = threadIdx.x + threadIdx.y * blockDim.x * ...;
    int tid_1, tid_2, blockDim_x, blockDim_y;
    if (global_tid < {d1}) {{
        // Thread belongs to K1
        blockDim_x = {d1} / {K1.blockDim_y};
        tid_1 = global_tid;
    }} else {{
        // Thread belongs to K2
        blockDim_x = {d2} / {K2.blockDim_y};
        tid_2 = global_tid - {d1};
    }}
    """

    # Replace __syncthreads() with partial barriers
    K1_code = K1.source.replace(
        "__syncthreads()",
        f'asm("bar.sync 1, {d1}");'  # only syncs K1's threads
    )
    K2_code = K2.source.replace(
        "__syncthreads()",
        f'asm("bar.sync 2, {d2}");'  # only syncs K2's threads
    )

    # Guarded execution: branch on thread ID
    fused = f"""
    __global__ void fused_kernel(...) {{
        {prologue}
        // K1 variables and K2 variables (renamed to avoid conflicts)
        ...
        if (global_tid < {d1}) goto K2_end;
        {K2_code}
        K2_end:
        if (global_tid >= {d1}) goto K1_end;  // skip K1 if K2 thread
        {K1_code}
        K1_end:
    }}
    """
    return fused


def search_best_partition(K1, K2, d0):
    """Search for optimal thread space partition via profiling."""
    best_time = float('inf')
    for d1 in range(128, d0, 128):
        d2 = d0 - d1
        # Generate without register bound
        F = generate_horizontal_fused_kernel(K1, K2, d1, d2)
        t = profile(F)
        if t < best_time:
            best_time = t; best_F = F

        # Generate with register bound (may improve occupancy)
        b1 = SM_REGS // (d1 * K1.num_regs)
        b2 = SM_REGS // (d2 * K2.num_regs)
        b0 = min(min(b1, b2), SM_SHMEM // F.shmem, SM_THREADS // d0)
        r0 = SM_REGS // (b0 * d0)
        F_reg = compile_with_register_bound(F, r0)
        t = profile(F_reg)
        if t < best_time:
            best_time = t; best_F = F_reg

    return best_F
```

## References

- Li, A., Zheng, B., Pekhimenko, G., Long, F. "Automatic Horizontal Fusion for GPU Kernels." CGO 2022. arXiv:2007.01277
- Wang, G., Lin, Y., Yi, W. "Kernel Fusion: An Effective Method for Better Power Efficiency on Multithreaded GPU." IEEE Green Computing, 2010.
- Filipovic, M., Madzin, J., Fousek, J., Matyska, L. "Optimizing CUDA Code by Kernel Fusion: Application on BLAS." The Journal of Supercomputing, 2015.
- NVIDIA. "Parallel Thread Execution ISA" — `bar.sync` partial barrier documentation.

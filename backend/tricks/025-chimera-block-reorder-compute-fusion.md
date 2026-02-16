# 025: Chimera: Analytical Block-Reorder Compute-Intensive Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Zheng, Chen, Song, Chen, Li, Yan, Lin, Leng, Liang — "Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion" (HPCA 2023)
**Paper**: [papers/chimera-compute-intensive-fusion.pdf]
**Documented**: 2026-02-15

## Description

Chimera is a compiler framework that generates fused kernels for **chains of compute-intensive operators** (GEMM chains, convolution chains) by analytically optimizing the **inter-block execution order** to maximize data reuse across operator boundaries. While most kernel fusion work focuses on fusing memory-intensive element-wise operations into compute-intensive ones, Chimera addresses a different and increasingly important problem: fusing multiple compute-intensive operators together, such as the batch GEMM chain in self-attention ($QK^T \to \text{softmax} \to AV$) or consecutive convolution layers in CNNs.

The fundamental insight is that when compute-intensive operators are tiled into computation blocks, the **execution order** of these blocks across different operators in the chain critically affects data reuse. Each operator's tile accesses tiles of input/output data; by interleaving blocks from different operators in the right order, intermediate results can be kept in on-chip memory (L1/L2 cache, shared memory) rather than spilling to DRAM. Different block orderings lead to dramatically different data movement volumes — the wrong order can cause $10\times$ more DRAM traffic than the optimal one.

Chimera's key contribution is an **analytical model** that, given a permutation of the block execution order, precisely computes the total data movement volume as a closed-form function of the tiling parameters. This avoids expensive hardware profiling or auto-tuning: the optimization is solved via Lagrange multipliers in the continuous domain, then rounded to integer tile sizes. The analytical solution has provably bounded approximation ratio to the true optimum.

For intra-block optimization, Chimera uses **replaceable micro kernels** — a unified high-level abstraction that maps to hardware-specific low-level implementations (CPU assembly, GPU Tensor Core intrinsics, NPU pragmas). This allows Chimera to target CPU, GPU, and NPU with the same framework.

On batch GEMM chains from BERT, ViT, and MLP-Mixer, Chimera achieves up to $2.87\times$ (CPU), $2.29\times$ (GPU), and $2.39\times$ (NPU) speedup over hand-tuned libraries.

## Mathematical Form

**Block decomposition:**

Each compute-intensive operator is decomposed into computation blocks controlled by decomposition parameters $\vec{S} = (s_1, s_2, \ldots, s_I)$ where $I$ is the number of loop dimensions. For a GEMM chain $C = A \times B$, $E = C \times D$ with dimensions $(m, n, k, l)$:

$$
\vec{S} = (T_M, T_N, T_K, T_L)
$$

Each block is a small loop nest over a tile of output data, fitting in on-chip memory.

**Data movement volume model:**

For a given block execution order (permutation $\text{Perm} = (l_{p_1}, l_{p_2}, \ldots, l_{p_I})$), the data movement volume for each tensor $T$ is:

$$
\text{DM}(T) = \text{DF}(T) \times \prod_{\substack{l_{p_i} \in \text{outer loops} \\ \text{that access } T}} \left\lceil \frac{L_{p_i}}{s_{p_i}} \right\rceil
$$

where $\text{DF}(T) = \prod_{d \in \text{dims}(T)} s_d$ is the data tile footprint of $T$.

Three key observations govern data movement:
1. Loops whose iteration variables are **not** used in $T$'s access indices cause no data movement for $T$
2. Once a loop causes data movement for $T$, **all outer loops** also cause data movement (tile replacement)
3. **Private loops** (appearing only in producer operators) do not cause data movement in consumer tensors

**GEMM chain example ($C = A \times B$, $E = C \times D$):**

Under block execution order $mlkn$ (execute dimension $l$ first, then $k$, $n$, $m$ outermost):

| Tensor | Data Movement | Data Footprint |
|--------|--------------|----------------|
| $A$ | $MK \lceil \frac{L}{T_L} \rceil$ | $T_M T_K$ |
| $B$ | $KL \lceil \frac{M}{T_M} \rceil$ | $T_K T_L$ |
| $C$ | $0$ (on-chip intermediate) | $T_M T_L$ |
| $D$ | $NL \lceil \frac{M}{T_M} \rceil$ | $T_L T_N$ |
| $E$ | $MN \lceil \frac{L}{T_L} \rceil$ | $T_M T_N$ |

Total data movement for the GEMM chain:

$$
\text{DV}_{\text{GEMM chain}} = MK \left\lceil \frac{L}{T_L} \right\rceil + KL \left\lceil \frac{M}{T_M} \right\rceil + NL \left\lceil \frac{M}{T_M} \right\rceil + MN \left\lceil \frac{L}{T_L} \right\rceil
$$

Note: intermediate $C$ has zero DRAM movement because it is fully reused on-chip.

**Constrained optimization:**

$$
\min_{\vec{S}} \text{DV}, \quad \text{s.t. } \text{MU} \leq \text{MemoryCapacity}
$$

where $\text{MU} = \max(\text{GEMM1}_{\text{MU}}, \text{GEMM2}_{\text{MU}})$ is the peak memory usage across all blocks:

$$
\text{GEMM1}_{\text{MU}} = T_M T_K + T_K T_L + T_M T_L
$$

$$
\text{GEMM2}_{\text{MU}} = T_M T_L + T_L T_N + T_M T_N
$$

**Analytical solution via Lagrange multipliers:**

Using $MC$ as shorthand for MemoryCapacity and $\alpha$ as a lower bound on $T_N, T_K$:

$$
DV^* = \frac{2ML(K + N)}{T_M^*}
$$

$$
T_M^* = T_L^* = -\alpha + \sqrt{\alpha^2 + MC}, \quad T_N^* = T_K^* = \alpha
$$

The approximation ratio of the integer-rounded solution $DV_{\text{app}}$ to the true optimum $DV^*$ is bounded:

$$
\frac{DV_{\text{app}}}{DV^*} \leq \max_{X \in \{M, L\}} \left\{ 1 + \frac{\sqrt{MC}}{X} + \frac{1}{\min\{X, \sqrt{MC}\}} \right\}
$$

This approaches $1$ when $X \gg \sqrt{MC}$ (typical for large model dimensions).

**Multi-level memory hierarchy extension:**

For $D$ levels of on-chip memory with bandwidths $bw_d$, the cost at level $d$ is:

$$
\text{Cost}_d(\vec{S}_d) = \frac{DV_d(\vec{S}_d)}{bw_d}
$$

The overall optimization minimizes the slowest memory stage:

$$
\min_{\vec{S}_1, \vec{S}_2, \ldots, \vec{S}_D} \left\{ \max\{\text{Cost}_1(\vec{S}_1), \ldots, \text{Cost}_D(\vec{S}_D)\} \right\}
$$

$$
\text{s.t. } \text{MU}_d \leq MC_d \quad \forall d \in [1, D]
$$

**Replaceable micro kernel abstraction:**

For a computation block, the arithmetic intensity is:

$$
AI = \frac{\#\text{ComputeInst}}{\#\text{LoadStoreInst}}
$$

For matrix multiplication micro kernels with parameters $(MI, NI, MII, KI)$:

$$
\#\text{ComputeInst} = MI \times NI \times KI
$$

$$
\#\text{LoadStoreInst} = KI \times (MI + NI) + NI
$$

$$
\text{RegUsed} = MI \times NI + MI + MII
$$

Maximize $AI$ subject to $\text{RegUsed} \leq \#\text{Registers}$.

**Key Definitions:**

- Block execution order — the permutation of tiled loop dimensions that determines how blocks from different operators are interleaved
- Data tile footprint ($\text{DF}$) — the on-chip memory required for one block's slice of a tensor
- Data movement volume ($\text{DV}$) — total bytes transferred between on-chip and off-chip memory
- Replaceable micro kernel — hardware-agnostic computation block abstraction with pluggable backends

## Complexity

| Aspect | Unfused (separate library calls) | Chimera (fused) |
|--------|--------------------------------|-----------------|
| DRAM access | $\sum_i \text{DV}(\text{op}_i)$ (each writes/reads intermediates) | $\text{DV}^*$ (intermediates on-chip) |
| Intermediate traffic | $O(MNK)$ per GEMM boundary | $0$ (reused in cache/SRAM) |
| Optimization method | N/A (fixed library kernels) | Analytical (Lagrange, $O(I!)$ orders) |
| Hardware profiling | Not needed (library calls) | Not needed (analytical model) |

**DRAM reduction (batch GEMM chains, CPU):**

Chimera reduces DRAM access by $9.86\%$–$59.54\%$ compared to PyTorch on batch GEMM chains, with L2/L3 cache hit rates significantly improved.

**End-to-end performance (A100 GPU, full networks):**

| Model | vs. PyTorch+CuDNN | vs. Relay+CuDNN | vs. Relay+Ansor |
|-------|-------------------|-----------------|-----------------|
| Transformer-Small | $1.42\times$ | — | — |
| BERT-Base | $1.42\times$ | — | — |
| ViT-Base/16 | $1.42\times$ | — | — |
| GEOMEAN | $1.42\times$ | $1.31\times$ | $1.22\times$ |

**Subgraph speedups (batch GEMM + softmax on GPU):**

- vs. PyTorch: average $1.62\times$, up to $7.89\times$
- vs. Relay: up to $2.29\times$
- vs. Ansor: up to $2.29\times$

**Optimization time:** Chimera's analytical approach is $21.89\times$ faster than Ansor's tuning-based approach (seconds vs. hours), achieving $1.39\times$ speedup in solution quality.

## Applicability

- **Self-attention batch GEMM chains**: The $QK^T \to \text{softmax} \to AV$ chain is a batch GEMM chain where Chimera's inter-block reordering keeps the intermediate attention matrix on-chip, reducing DRAM by up to $59\%$
- **MLP-Mixer / Feed-forward layers**: Consecutive linear layers ($\text{Linear}_1 \to \text{Linear}_2$) form GEMM chains that benefit from fused block scheduling
- **Convolution chains in CNNs**: Consecutive convolution layers (e.g., $3\times3$ → $1\times1$ in ResNet bottleneck) can be fused; point-wise convolutions especially benefit when memory-bound
- **Multi-head attention**: Each head's GEMM chain can be independently fused; Chimera handles the batch dimension naturally
- **Cross-hardware deployment**: The replaceable micro kernel abstraction allows the same fusion strategy to target CPU (AVX-512), GPU (Tensor Cores), and NPU (Ascend) without changing the inter-block optimization
- **Any chain of compute-intensive operators**: Applicable whenever two or more GEMMs/convolutions are chained with lightweight intermediate operators (softmax, ReLU, etc.)

## Limitations

- **Chain topology only**: Chimera targets sequential chains of operators; it does not handle arbitrary DAG topologies (e.g., skip connections creating diamond patterns)
- **Compute-intensive operators only**: Memory-intensive operators between compute-intensive ones (softmax, ReLU) are handled but the focus is on fusing the compute-intensive boundaries; pure memory-intensive chains should use FusionStitching or XLA
- **Design space grows factorially**: For $I$ independent loop dimensions, there are $I!$ possible block execution orders; for chains with many operators this can be large (mitigated by shared loops between operators reducing the effective count)
- **Convolution limitations**: $3\times3$ convolutions with sliding windows can cause recomputations after fusion; Chimera is most effective when the second convolution is point-wise ($1\times1$) or otherwise memory-bound
- **Single-batch NPU bottleneck**: On NPU with small unified buffer, large GEMMs in the chain can bottleneck on intermediate transfer, making fusion counterproductive for some configurations
- **Integer rounding gap**: The continuous Lagrange solution is rounded to integers, introducing a bounded but nonzero approximation gap

## Implementation Notes

```python
# Pseudocode for Chimera's inter-block optimization

def compute_data_movement(ops, perm, tile_sizes):
    """
    Algorithm 1 from paper: compute total data movement volume
    for a given block execution order (permutation) and tile sizes.
    """
    DV = 0
    MU = 0

    for op in ops:
        total_DF = 0
        for tensor in op.all_tensors():
            DF = get_footprint(tensor, tile_sizes)
            total_DF += DF

            if tensor in ops.io_tensors():
                DM = DF
                keep_reuse = True
                # Traverse loops from inner to outer
                for loop in reversed(perm):
                    if loop in op.all_loops():
                        if loop accesses tensor:
                            keep_reuse = False
                    if not keep_reuse:
                        DM *= ceil(loop.trip_count / tile_sizes[loop])
                DV += DM

        # Remove private loops (only affect producer, not consumer)
        for loop in perm:
            if loop.is_private_to(op):
                perm.erase(loop)

        MU = max(MU, total_DF)

    return DV, MU


def optimize_tile_sizes(ops, perm, memory_capacity):
    """
    Solve for optimal tile sizes using Lagrange multipliers.
    Returns analytical solution for GEMM chain.
    """
    # For GEMM chain: C = A*B (dims M,K,L), E = C*D (dims M,L,N)
    M, N, K, L = ops.dimensions

    # Solve continuous relaxation
    alpha = lower_bound(N, K)  # minimum tile size
    T_M_star = -alpha + sqrt(alpha**2 + memory_capacity)
    T_L_star = T_M_star
    T_N_star = alpha
    T_K_star = alpha

    # Round to integers
    T_M = min(floor(T_M_star), M)
    T_L = min(floor(T_L_star), L)
    T_N = min(floor(T_N_star), N)
    T_K = min(floor(T_K_star), K)

    DV_opt = 2 * M * L * (K + N) / T_M

    return (T_M, T_N, T_K, T_L), DV_opt


def chimera_compile(operator_chain, target_hw):
    """
    Full Chimera compilation pipeline.
    """
    # Step 1: Block decomposition
    loops = operator_chain.all_independent_loops()

    # Step 2: Enumerate block execution orders
    best_dv = float('inf')
    for perm in permutations(loops):
        # Step 3: Analytically solve for optimal tile sizes
        tiles, dv = optimize_tile_sizes(operator_chain, perm, target_hw.memory)
        if dv < best_dv:
            best_dv = dv
            best_perm = perm
            best_tiles = tiles

    # Step 4: Select replaceable micro kernel for target hardware
    if target_hw == 'gpu':
        micro_kernel = gpu_tensor_core_kernel(best_tiles)
    elif target_hw == 'cpu':
        micro_kernel = cpu_avx512_kernel(best_tiles)
    elif target_hw == 'npu':
        micro_kernel = npu_cube_kernel(best_tiles)

    # Step 5: Generate fused kernel code
    return codegen(best_perm, best_tiles, micro_kernel)
```

## References

- Zheng, S., Chen, S., Song, P., Chen, R., Li, X., Yan, S., Lin, D., Leng, J., Liang, Y. "Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion." HPCA 2023, pp. 1113-1126.
- Zheng, S., Zhang, X., Li, S., Wei, S., Yin, S. "Atomic: Atomic Dataflow based Graph-level Workload Orchestration for Scalable DNN Accelerators." HPCA 2022.
- Li, R., Xu, A., Sukumaran-Rajam, A., Rountev, A., Sadayappan, P. "Analytical Characterization and Design Space Exploration for Optimization of CNNs." ASPLOS 2021.

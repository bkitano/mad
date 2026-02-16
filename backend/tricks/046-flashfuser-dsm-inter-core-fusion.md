# 046: FlashFuser: DSM Inter-Core Kernel Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Huang, Zhou, Liu, Luo, Diao, Guo, Zhai, Feng, Zhang, Wu, Leng — "FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection" (arXiv 2512.12949, Dec 2025)
**Paper**: [papers/flashfuser-dsm-kernel-fusion.pdf]
**Documented**: 2026-02-15

## Description

FlashFuser is the first compiler framework to exploit **Distributed Shared Memory (DSM)** — the inter-SM on-chip interconnect on NVIDIA H100 and later GPUs — for kernel fusion of compute-intensive operator chains. Existing kernel fusion techniques (epilogue fusion, FusionStitching, Chimera, BOLT) store intermediate results in registers or per-SM shared memory (SMEM). When the intermediate tensor between two fused GEMMs exceeds the SMEM capacity of a single SM (~227 KB on H100), these frameworks abandon fusion and fall back to writing intermediates to HBM. This is exactly what happens for large FFN layers in LLMs, where the intermediate activation between the two linear projections can be megabytes in size.

FlashFuser solves this by introducing DSM as an **L1.5 cache tier** in the memory hierarchy. On H100, DSM connects the SMEM of up to 16 SMs within a *cluster* via a high-bandwidth, low-latency SM-to-SM Network-on-Chip (NoC). This effectively expands the on-chip memory pool from ~227 KB (single SM SMEM) to ~3.6 MB (16 SMs × 227 KB), enabling fusion of operator chains that were previously infeasible. The key insight is that by spilling intermediate results to DSM instead of HBM, the compiler avoids the costly global memory round-trip (2–3 TB/s HBM bandwidth) while staying on-chip (higher bandwidth, lower latency than HBM).

FlashFuser contributes three components: (1) a **`dsm_comm` communication primitive** that formalizes inter-SM data exchange patterns (shuffle, reduce, all-exchange) needed for GEMM chain fusion; (2) a **dataflow analyzer** that determines optimal loop scheduling, tile selection, and hierarchical resource mapping across reg → SMEM → DSM → HBM; and (3) a **fusion search engine** with pruning rules and an analytical cost model that navigates the vastly expanded search space introduced by DSM.

On an NVIDIA H100 GPU, FlashFuser reduces global memory access by 58%, delivers kernel speedups of 3.3× over cuBLAS/CUTLASS and 4.1× over state-of-the-art compilers (Chimera, BOLT), and achieves 1.24× end-to-end inference speedup on real LLMs (LLaMA-3-70B, Qwen2.5-14B/32B).

## Mathematical Form

**Problem: Fused GEMM chain**

Consider a standard FFN: two consecutive GEMMs with an activation:

$$
C = A \cdot B, \quad E = \phi(C) \cdot D
$$

where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $D \in \mathbb{R}^{N \times L}$, and $\phi$ is an element-wise activation (GELU, SiLU, etc.).

**Unfused execution** requires writing intermediate $C \in \mathbb{R}^{M \times N}$ to HBM:

$$
\text{HBM traffic}_{\text{unfused}} = |A| + |B| + 2|C| + |D| + |E|
$$

The $2|C|$ term (write then read) dominates for large $N$.

**Single-SM fusion** keeps $C$ in registers/SMEM, but fails when:

$$
|C_{\text{tile}}| > \text{Cap}_{\text{SMEM}} \approx 227\text{ KB (H100)}
$$

This occurs for typical LLM FFN dimensions (e.g., GPT-6.7B with $N = 16384$).

**DSM-enabled fusion** spills $C_{\text{tile}}$ across the cluster:

$$
|C_{\text{tile}}| \leq \sum_{i=1}^{|\text{cluster}|} \text{Cap}_{\text{SMEM}}^{(i)} = |\text{cluster}| \times \text{Cap}_{\text{SMEM}}
$$

For a cluster of 16 SMs: effective capacity $\approx 16 \times 227\text{ KB} \approx 3.6\text{ MB}$.

**DSM communication primitives:**

Three primitives formalize inter-SM data exchange within a cluster:

1. **`dsm_all_exchange`**: All-reduce within the cluster. After $\text{GEMM}_0$, each block holds a partial sum $C_{i,j}(p)$. The primitive performs:

$$
C_{i,j} = \bigoplus_{p=0}^{\text{cls}_k - 1} C_{i,j}(p)
$$

where $\oplus$ can be addition (standard FFN) or multiplication (gated FFN). Each block receives the complete, fully-accumulated intermediate tile.

2. **`dsm_shuffle`**: Ring communication pattern. Blocks within a Shuffle Group exchange their slices of $C$ so that each block obtains the data needed for its portion of $\text{GEMM}_1$:

$$
\text{Block}_i \text{ sends } C_{\text{slice}_i} \to \text{Block}_{(i+1) \bmod \text{cls}_{\text{shuffle}}}
$$

3. **`dsm_scatter_reduce`** (+ `inter_cluster_reduce`): Hierarchical reduction for the Store phase. Intra-cluster scatter-reduce via DSM, followed by inter-cluster reduction via TMA:

$$
E_{i,j} = \text{inter\_cluster\_reduce}\!\left(\text{dsm\_scatter\_reduce}\!\left(\{E_{i,j}^{(p)}\}_{p \in \text{cluster}}\right)\right)
$$

**Cluster parameterization:**

The cluster geometry is defined by $\text{cls} = (\text{cls}_m, \text{cls}_n, \text{cls}_k, \text{cls}_l)$, specifying how many SMs are allocated along each GEMM dimension. Two derived parameters control communication:

$$
\text{cls}_{\text{shuffle}} = \frac{\text{cls}_l}{\text{cls}_k}, \quad \text{cls}_{\text{reduce}} = \frac{\text{cls}_n}{\text{cls}_{\text{shuffle}}} = \frac{\text{cls}_n \times \text{cls}_k}{\text{cls}_l}
$$

**Analytical cost model:**

The data movement cost at each memory level $l$ is:

$$
C_l(\mathcal{T}_l) = \frac{V_l(\mathcal{T}_l)}{B_l}
$$

where $V_l$ is the data volume transferred at level $l$ under tiling strategy $\mathcal{T}_l$, and $B_l$ is the bandwidth of level $l$. The optimization minimizes the bottleneck:

$$
\min_{\mathcal{T}_1, \ldots, \mathcal{T}_L} \left\{ \max_{l=1,\ldots,L} C_l(\mathcal{T}_l) \right\}
$$

subject to memory capacity constraints:

$$
U_l(\mathcal{T}_l) \leq \text{Cap}_l, \quad \forall l \in \{1, \ldots, L\}
$$

**Key Definitions:**

- $\text{DSM}$ — Distributed Shared Memory; the SM-to-SM NoC on H100+ that connects SMEM of SMs within a cluster, acting as an L1.5 cache (~1.6–2.5 TB/s bandwidth, ~160–470 cycle latency depending on cluster size)
- Cluster — A group of 1–16 SMs connected via DSM; the maximum cluster size is a hardware limit (16 on H100)
- $\text{cls}_i$ — Cluster dimension along GEMM dimension $i$; the product $\prod \text{cls}_i \leq 16$
- Shuffle Group — A subset of blocks within a cluster that exchange data via ring communication
- Loop schedule — The permutation $s$ that determines the nesting order of dimensions $\{M, N, K, L\}$ (e.g., MNKL, MLNK), determining which dimensions are spatial vs. temporal

## Complexity

| Scenario | Global Memory Access | Kernel Launches |
|----------|---------------------|-----------------|
| Unfused 2-GEMM FFN | $2(|A| + |B| + |C| + |D| + |E|)$ | $2+$ (GEMM + activation + GEMM) |
| Single-SM fusion (if fits) | $|A| + |B| + |D| + |E|$ (no $C$ to HBM) | $1$ |
| Single-SM fusion (C too large) | **Falls back to unfused** | $2+$ |
| FlashFuser DSM fusion | $|A| + |B| + |D| + |E|$ (C stays on-chip via DSM) | $1$ |

**Memory access reduction:** 58% average reduction in global memory access vs. PyTorch (unfused).

**Performance (NVIDIA H100 SXM):**

| Workload | FlashFuser Speedup |
|----------|--------------------|
| GEMM chains (avg over 10 configs) | 5.4× over BOLT, 4.6× over Chimera, 3.1× over PyTorch |
| Conv chains (avg over 8 configs) | 6.3× over BOLT, 6.4× over Chimera, 3.9× over PyTorch |
| Gated FFN / SwiGLU (avg over 8 configs) | 3.3× over cuBLAS, 4.1× over Chimera |
| End-to-end LLM inference (LLaMA-3-70B, Qwen2.5) | 1.24× average speedup |

**Search space:** The introduction of DSM expands the fusion search space from ~$10^4$ (prior work) to ~$10^6$ candidates. FlashFuser's pruning rules reduce this by >99.99%, and the search engine accelerates compilation by 12–864× vs. brute force.

## Applicability

- **Transformer FFN layers**: The primary target — fusing Linear → Activation → Linear into a single kernel where the intermediate activation exceeds single-SM SMEM capacity. This covers standard FFNs in GPT, LLaMA, OPT, BERT, and other architectures
- **Gated FFN / SwiGLU**: Fuses the branched structure where two parallel GEMMs produce gate and up projections, followed by element-wise gating and a down projection. FlashFuser handles the `dsm_all_exchange` with a `Mul` operation for the gating step
- **Convolutional chains**: im2col-based convolutions followed by activations and further convolutions (ResNet-style blocks) benefit similarly, with 3.9–6.3× speedups
- **Any GEMM chain on H100+**: Any computation of the form $E = f_2(f_1(A \cdot B) \cdot D)$ where the intermediate is too large for single-SM fusion but fits in cluster-level DSM
- **Architectures with inter-core connectivity**: While evaluated on NVIDIA H100, the `dsm_comm` abstraction is topology-agnostic and applies to Graphcore IPU (crossbar SMEM), Cerebras WSE (mesh L1), and future GPUs with expanded DSM

## Limitations

- **H100+ hardware requirement**: DSM is only available on NVIDIA Hopper (H100) and later architectures. Older GPUs (A100, V100) cannot use this technique
- **Cluster size ceiling**: Maximum 16 SMs per cluster on H100. If the intermediate tensor exceeds $16 \times 227\text{ KB} \approx 3.6\text{ MB}$, fusion still fails and falls back to HBM
- **DSM bandwidth degrades with cluster size**: As shown in Figure 4 of the paper, DSM bandwidth decreases from ~2.5 TB/s (2 SMs) to ~1.6 TB/s (16 SMs), and latency increases. Very large clusters may not outperform HBM for some access patterns
- **Static search**: The fusion plan is determined offline; runtime kernel selection uses binning/table-lookup for the M dimension only. Fully dynamic shapes require multiple pre-compiled variants
- **CUTLASS dependency**: Code generation is built on NVIDIA CUTLASS, limiting portability to AMD ROCm or other platforms
- **Single cluster evaluated**: Inter-cluster communication uses TMA (global memory path), so the technique benefits most when the entire GEMM chain maps within a single cluster. Multi-cluster fusion has limited DSM benefit

## Implementation Notes

```python
# Pseudocode for FlashFuser's DSM-based GEMM chain fusion
# Fuses: E = activation(A @ B) @ D into a single kernel

# --- dsm_comm primitive abstraction ---
class DSMComm:
    """Three DSM communication patterns for inter-SM data exchange."""

    @staticmethod
    def dsm_all_exchange(partial_tiles, op='add'):
        """All-reduce within cluster. Each block gets fully accumulated result.
        op='add' for standard FFN, op='mul' for gated FFN."""
        # Uses NVIDIA mbarrier for many-to-many synchronization
        # Ring-based implementation via TMA for data movement
        result = reduce(op, partial_tiles)  # across cls_k blocks
        broadcast(result, all_blocks_in_cluster)
        return result

    @staticmethod
    def dsm_shuffle(tiles, shuffle_group_size):
        """Ring communication: blocks exchange slices for GEMM1 input."""
        # Each block sends its slice to next block in ring
        for step in range(shuffle_group_size - 1):
            send_to_neighbor(tiles[my_slice], direction='right')
            receive_from_neighbor(tiles[neighbor_slice], direction='left')

    @staticmethod
    def dsm_scatter_reduce(partial_results, reduce_groups):
        """Hierarchical reduction: intra-cluster scatter-reduce +
        inter-cluster TMA-based reduce for final output."""
        # Step 1: Intra-cluster scatter-reduce via DSM
        local_result = scatter_reduce_dsm(partial_results)
        # Step 2: Inter-cluster reduce via TMA (through global memory)
        if reduce_groups > 1:
            final = tma_atomic_reduce(local_result)
        else:
            final = local_result
        return final


# --- Dataflow Analyzer (Algorithm 1 from paper) ---
def dataflow_analyzer(graph, device, loop_schedule, tile_sizes, resource_mapping):
    """Analyze data movement across memory hierarchy (reg -> SMEM -> DSM -> HBM).
    Returns total data volume and hierarchical spilling plan."""
    mapping_plan = {}
    total_dv = 0

    for tensor in graph.tensors():
        footprint = get_footprint(tensor, tile_sizes.block)

        if tensor in graph.io_tensors():
            # I/O tensors always go through global memory
            dm = footprint
            for s_i in reversed(loop_schedule):
                if s_i.accesses(tensor):
                    dm *= (s_i.size / tile_sizes[s_i].block)
            total_dv += dm
        else:
            # Reused tensor: greedily place across memory hierarchy
            remaining = footprint
            for level in ['reg', 'smem', 'dsm', 'global']:
                if remaining <= 0:
                    break
                alloc = min(remaining, device.capacity[level])
                mapping_plan[tensor] = (level, alloc)
                remaining -= alloc
                # Update data volume for DSM traffic
                if level == 'dsm':
                    total_dv += compute_dsm_traffic(tensor, tile_sizes.cluster)

    return total_dv, mapping_plan


# --- Fusion Search Engine (Algorithm 2 from paper) ---
def fusion_search(graph, device, top_k=11):
    """Search over loop schedules, tile sizes, and resource mappings.
    Uses pruning rules to reduce search space from ~10^13 to ~10^6."""
    all_candidates = enumerate_all_candidates(graph)
    pruned = prune_candidates(all_candidates)
    # Pruning rules:
    # 1. Divisible tile sizes (hardware-aware)
    # 2. Cluster size constraint (product of cls dims <= 16)
    # 3. Activation constraint (accumulation dim must be innermost)
    # 4. Dependency constraint (L spatial => need inter-tile comm)
    # 5. Memory capacity limit (tensor fits in lowest available cache)

    top_k_list = []
    for (schedule, tiles, resource_map) in pruned:
        dv, plan = dataflow_analyzer(graph, device, schedule, tiles, resource_map)
        cost = calculate_cost(dv)  # C_l = V_l / B_l, take max over levels
        update_top_k(top_k_list, (cost, plan), top_k)

    # Profile top-K on hardware to find actual best
    best = profile_best(top_k_list, device)
    return best
```

## References

- Huang, Z., Zhou, Y., Liu, Z., Luo, X., Diao, Y., Guo, M., Zhai, J., Feng, Y., Zhang, C., Wu, A., Leng, J. "FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection." arXiv:2512.12949, Dec 2025.
- Zheng, S., Chen, S., Song, R., Chen, X., Li, S., Yan, D., Lin, J., Leng, J., Liang, Y. "Chimera: An Analytical Optimizing Framework for Effective Compute-Intensive Operators Fusion." HPCA 2023.
- Xing, L., Wang, S., Zhang, J., Chen, A., Zhu, Y. "BOLT: Bridging the Gap between Auto-Tuners and Hardware-Native Performance." MLSys 2022.
- Zhang, D., Yang, X., Zhou, D., Cheng, D. "MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators." SC24, 2024.
- NVIDIA. "CUDA C++ Programming Guide: Distributed Shared Memory." https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory

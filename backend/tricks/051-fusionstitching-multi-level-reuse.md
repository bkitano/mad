# 051: FusionStitching: Multi-Level Data Reuse Stitching

**Category**: kernel
**Gain type**: efficiency
**Source**: Zheng, Zhao, Long, Zhu, Zhu, Zhao, Diao, Yang, Lin — "FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads" (arXiv 2021, deployed at Alibaba)
**Paper**: [papers/fusionstitching-multi-level-reuse.pdf]
**Documented**: 2026-02-15

## Description

FusionStitching is a JIT kernel fusion compiler that extends standard fusion techniques by introducing **multi-level data reuse** between fused operators. While traditional JIT fusion engines (e.g., XLA) only support **thread-local** data transfer — where each thread independently computes and passes values via registers — FusionStitching adds two additional reuse levels: **intra-warp reuse** (via register shuffle) and **intra-block reuse** (via shared memory). This enables fusing operator patterns that were previously unfusable, particularly chains involving reductions interleaved with element-wise operations.

The core insight is that memory-intensive operations (element-wise, transpose, reduce, broadcast, gather, slice) frequently require different threads to consume the same intermediate value. In standard XLA fusion, the only option is for each thread to **recompute** the shared intermediate independently, which is prohibitively expensive for operations like `reduce`, `tan`, `log`, and `exp`. FusionStitching instead allows a producing thread to **share** its result with consuming threads through register shuffle (within a warp, 32 threads) or shared memory (within a thread block), avoiding both recomputation and global memory round-trips.

FusionStitching addresses two main challenges: (1) **code generation** — how to generate efficient fused kernels for complex fusion patterns with non-homogeneous parallelism, solved by four kernel composition schemes selected via a latency-evaluator cost model; and (2) **fusion exploration** — how to find the optimal fusion plan from an exponentially large search space ($O(2^V)$ possible combinations), solved by an approximate dynamic programming approach with $O(V + E)$ complexity guided by a lightweight delta-evaluator cost model.

Deployed on Alibaba's production GPU cluster for 4+ months, FusionStitching saves ~7,000 GPU hours per month across ~30,000 tasks.

## Mathematical Form

**Four kernel composition schemes:**

Given two dependent operators $A$ (producer) and $B$ (consumer) to be stitched:

**1. Kernel Packing** — No data dependence between ops; merge into single loop:

$$
\text{For } i \in [0, N): \quad y_1[i] = f_1(x[i]); \quad y_2[i] = f_2(x[i])
$$

All element-wise ops with same parallelism dimension are packed into a single loop body, eliminating kernel launch overhead.

**2. Thread Composition** — Each thread transfers data to itself via registers (XLA-style):

$$
\text{Thread } t: \quad r = A(x[t]); \quad y[t] = B(r)
$$

Intermediate $r$ stays in thread-local registers. May introduce redundant computation when $B$ needs values produced by other threads.

**3. Warp Composition** — Intra-warp data transfer via register shuffle:

$$
\text{Warp } w, \text{ lane } l: \quad r_l = A(x[\text{idx}(w, l)]); \quad \bar{r} = \bigoplus_{l=0}^{31} r_l; \quad y[\text{idx}(w, l)] = B(\bar{r})
$$

After thread $l$ produces $r_l$, the warp-level reduction $\bar{r}$ is computed using `__shfl_down_sync()` / `__shfl_xor_sync()`, and the result is broadcast to all 32 lanes. Typical case: warp reduction followed by element-wise consumption.

**4. Block Composition** — Intra-block data transfer via shared memory:

$$
\text{Block } b, \text{ thread } t: \quad r_t = A(x[\text{idx}(b, t)]); \quad \texttt{shmem}[\text{offset}(t)] = r_t; \quad \texttt{\_\_syncthreads()}; \quad y[\text{idx}(b, t)] = B(\texttt{shmem}[\cdot])
$$

All threads in the block write intermediates to shared memory, synchronize, then read the data they need. Enables composing non-homogeneous parallelism (e.g., a row reduction followed by a per-element broadcast).

**Latency-evaluator cost model:**

$$
L = N_{\text{wave}} \times L_{\text{warp}}
$$

$$
N_{\text{wave}} = \frac{N_{\text{warp}}}{\text{Occupancy}}
$$

$$
L_{\text{warp}} = N_{\text{instruction}} \times \text{CPI}
$$

where $N_{\text{wave}}$ is the number of warp-wave passes over the GPU, $L_{\text{warp}}$ is per-warp latency in cycles, and Occupancy is bounded by register and shared memory usage.

**Shared memory optimization via dominance-tree analysis:**

Given computation graph $G$ and shared memory allocation requests, FusionStitching traverses in topological order. For op $v$ needing shared memory:

$$
\text{alloc}(v) = \begin{cases} \text{reuse}(\text{alloc}(u)) & \text{if } u \text{ dominates } v \text{ and } \text{alloc}(u) \text{ is dead} \\ \text{new\_alloc}() & \text{otherwise} \end{cases}
$$

This minimizes peak shared memory usage by reusing buffers whose values are no longer live.

**Fusion exploration — approximate dynamic programming:**

For computation graph $G = (V, E)$, define fusion pattern $P_i = (V_i, E_i) \subseteq G$ and fusion plan $S = \{P_0, \ldots, P_{k-1}\}$ (disjoint). The objective is:

$$
\max_{S} \sum_{i=1}^{k} f(P_i)
$$

where the score function combines three terms:

$$
f = T_{\text{reduced\_mem}} + T_{\text{reduced\_calls}} - T_{\text{penalty}}
$$

- $T_{\text{reduced\_mem}}$: estimated memory access latency saved by fusion
- $T_{\text{reduced\_calls}}$: kernel launch overhead saved (number of fused kernels × context switch time)
- $T_{\text{penalty}}$: performance penalty from fusion (reduced parallelism, resource contention)

Candidate patterns are generated per-vertex in post-order via **PatternReduction** with divide-and-conquer (group consumers, enumerate combinations per group, keep top-$k$), then assembled into a global plan via beam search (width 3).

**Key Definitions:**

- $V$ — vertices (operators) in the computation graph
- $E$ — edges (data dependencies) between operators
- Warp composition — fusing ops with intra-warp register-shuffle data transfer
- Block composition — fusing ops with intra-block shared-memory data transfer
- Dominance tree — tree structure where node $u$ dominates $v$ if every path from root to $v$ passes through $u$

## Complexity

| Aspect | XLA (thread-local only) | FusionStitching |
|--------|------------------------|-----------------|
| Fusion search | Rule-based, greedy | Approx. DP, $O(V + E)$ |
| Data reuse levels | 1 (registers) | 3 (registers, shuffle, shmem) |
| Fusible patterns | Element-wise chains, tail-reductions | Arbitrary memory-intensive DAGs |
| Kernel calls (DIEN-train) | 6,842 | 2,109 |
| Memory-intensive kernel time | baseline | $1.39\times$ avg, up to $1.74\times$ faster |

**Memory:** Eliminates up to 66% of global memory traffic for memory-intensive operations (CRNN benchmark: 667.6 MB → 225.8 MB). Shared memory overhead is managed via dominance-tree reuse.

**End-to-end speedup:**

| Model | vs. TensorFlow | vs. XLA |
|-------|---------------|---------|
| BERT-train | $1.39\times$ | $1.02\times$ |
| DIEN-train | $2.42\times$ | $1.82\times$ |
| Transformer | $1.34\times$ | $1.08\times$ |
| CRNN | $2.42\times$ | $1.62\times$ |
| Average | $1.66\times$ | $1.45\times$ |

## Applicability

- **Layer normalization**: FusionStitching fuses the full LayerNorm (element-wise → reduce → element-wise → reduce → element-wise) into a single kernel where XLA requires 4 separate fusions, achieving $1.23\times$ speedup over XLA's 4-kernel version
- **Transformer attention**: Softmax involves exp → reduce-sum → div, which requires cross-thread data sharing; warp/block composition enables full softmax fusion
- **RNN/LSTM cells**: Recurrent cells contain interleaved element-wise and reduction operations that benefit from multi-level reuse
- **Recommendation models (DIEN)**: Large memory-intensive op graphs with up to 10,406 kernel calls in TensorFlow; FusionStitching reduces to 2,109
- **Any model with many memory-intensive ops**: Models where memory-intensive op time exceeds 15-40% of total (common in NLP, speech, recommendation)

## Limitations

- **Does not fuse compute-intensive ops**: FusionStitching targets memory-intensive operators only; GEMM and convolution remain as separate library calls (cuBLAS/cuDNN)
- **Shared memory contention**: Block composition requires shared memory, which can reduce occupancy and hurt parallelism for large-granularity fusions
- **Dynamic shapes**: Built on XLA's service framework, which is not friendly to dynamic tensor shapes; models with variable sequence lengths may not benefit
- **JIT compilation overhead**: First-iteration compilation adds up to 30 minutes of overhead (amortized over subsequent iterations for training, or prepared once for inference)
- **Warp composition limited to warp-aligned reductions**: Register shuffle only works within a 32-thread warp; cross-warp reductions must use block composition
- **Does not handle inter-block communication**: No fusion of patterns requiring global-memory synchronization between thread blocks

## Implementation Notes

```python
# Pseudocode for FusionStitching composition scheme selection

def generate_fused_kernel(fusion_pattern):
    """
    Generate GPU kernel for a fusion pattern using composition schemes.
    """
    # Step 1: Classify ops
    ops = fusion_pattern.operators
    light_ewise = [op for op in ops if op.type == 'element_wise' and op.cost == 'light']
    heavy_ewise = [op for op in ops if op.type == 'element_wise' and op.cost == 'heavy']
    reductions = [op for op in ops if op.type == 'reduction']

    # Step 2: Identify sub-roots (reduction and heavy element-wise ops)
    # and group remaining ops by data dependency
    sub_roots = reductions + heavy_ewise
    groups = group_by_subroot(ops, sub_roots)

    # Step 3: For each group, select composition scheme
    for group in groups:
        if no_cross_thread_dependency(group):
            scheme = 'kernel_packing'  # or thread_composition
        elif all_within_warp(group):
            scheme = 'warp_composition'
            # Producer stores to register, consumers read via __shfl_sync
        else:
            scheme = 'block_composition'
            # Producer stores to shared memory, __syncthreads, consumers read

    # Step 4: Allocate shared memory with dominance-tree reuse
    shmem_map = dominance_tree_alloc(fusion_pattern.graph)

    # Step 5: Enumerate schedule combinations, evaluate with latency model
    best_config = None
    for schedule_combo in enumerate_schedules(groups):
        for launch_dim in candidate_launch_dims(schedule_combo):
            latency = estimate_latency(schedule_combo, launch_dim, shmem_map)
            if best_config is None or latency < best_config.latency:
                best_config = (schedule_combo, launch_dim, shmem_map)

    return emit_gpu_ir(best_config)


# Example: LayerNorm fusion with block composition
# XLA produces 4 separate kernels; FusionStitching produces 1
#
# Kernel structure (single fused kernel):
#   1. Load input x[i] from HBM (one read)
#   2. Element-wise ops (subtract mean, etc.) → registers
#   3. Warp-level reduce (register shuffle) → partial sums
#   4. Block-level reduce (shared memory) → mean, variance
#   5. Element-wise normalize → registers
#   6. Store output y[i] to HBM (one write)
#
# Total HBM traffic: 2 * |x| (read input + write output)
# vs. XLA: 8 * |x| (4 kernels × read + write each)
```

## References

- Zheng, Z., Zhao, P., Long, G., Zhu, F., Zhu, K., Zhao, W., Diao, L., Yang, J., Lin, W. "FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads." arXiv:2009.10924, 2021.
- Zheng, Z., Zhao, P., Long, G., Zhu, F., Zhu, K., Zhao, W., Diao, L., Yang, J., Lin, W. "FusionStitching: Deep Fusion and Code Generation for Tensorflow Computations on GPUs." arXiv:1811.05213, 2018 (earlier version).
- Cooper, K. D., Harvey, T. J., Kennedy, K. "A Simple, Fast Dominance Algorithm." 2001.

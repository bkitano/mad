# 075: MCFuser: Memory-Bound Compute-Intensive (MBCI) Operator Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhang, Yang, Zhou, Cheng — "MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators" (SC 2024)
**Paper**: [papers/mcfuser-mbci-operator-fusion.pdf]
**Documented**: 2025-06-15

## Description

MCFuser is a compiler framework that identifies and efficiently fuses a previously overlooked class of GPU operators: **memory-bound compute-intensive (MBCI) operators** — operators that are nominally compute-intensive (like GEMM) but become memory-bandwidth-bound when tensor dimensions shrink. Traditional compiler heuristics classify GEMM as compute-intensive and therefore "non-fusible," missing significant optimization opportunities when the reduction dimension $K$ is small relative to the output dimensions $M, N$.

The key insight is that the compute-to-memory-access ratio $\phi$ of a GEMM $C_{M,N} = A_{M,K} \times B_{K,N}$ is:

$$
\phi = \frac{2 T_M T_N K}{2 T_M T_N K + T_M K + T_N K} \approx \frac{2K}{2K + \frac{K}{T_N} + \frac{K}{T_M}}
$$

When $K$ decreases (e.g., from 1024 to 64), $\phi$ drops below the GPU's compute-to-bandwidth ratio $\mathcal{P}/\mathcal{W}$, making the GEMM memory-bound. In Transformer self-attention modules, this happens frequently: despite self-attention constituting only 11–19% of FLOPs (for sequence lengths 512–2048 in BERT-Large), it dominates 39–61% of execution time because the attention head dimension $H = 64$ makes the intermediate GEMMs memory-bound.

MCFuser addresses three challenges that prevented prior work from fusing MBCI operator chains:

1. **Incomplete search space**: Prior frameworks (AStitch, DNNFusion, Chimera) either refuse to fuse compute-intensive operators or limit their search to nested loop permutations. MCFuser introduces **tiling expressions** — a compact notation $\vec{l} = (l_1, l_2, \ldots, l_J)$ encoding both loop structure (nested vs. sequential) and tile sizes — that exhaustively enumerate all valid fusion strategies for MBCI chains. Both **deep tiling** (fully nested loops, $x!$ permutations) and **flat tiling** (sequential loops, reduced depth) are explored.

2. **Redundant memory access**: When fusing multiple compute-intensive operators, coupling their computation loops often introduces redundant loads. MCFuser performs **DAG-based memory access analysis** that represents the data flow as a directed acyclic graph of loop nodes, compute nodes, and load/store nodes. By analyzing scope and order dependencies, MCFuser identifies memory access statements that can be **relocated to outer loops** (reducing trip count) or **eliminated entirely** when loop extents become trivially 1.

3. **Expensive tuning**: Auto-tuners like Ansor require hours of profiling. MCFuser replaces runtime measurement with an **analytical performance model** that estimates execution time from three components — memory access overhead $t_{\text{mem}}$, computation overhead $t_{\text{comp}}$, and a parallelism slowdown factor $\alpha$ — enabling the search to converge in seconds rather than hours.

MCFuser achieves up to 5.9× speedup over Ansor, 6.6× over PyTorch, and 3.3× over FlashAttention on self-attention modules, while reducing tuning time by over 70×.

## Mathematical Form

**Compute-to-memory ratio (determines if MBCI):**

For a GEMM with tile sizes $T_M, T_N$ and reduction dimension $K$:

$$
\phi = \frac{2 T_M T_N K}{2 T_M T_N K + T_M K + T_N K}
$$

The operator is memory-bound when:

$$
\phi < \frac{\mathcal{P}}{\mathcal{W}}
$$

where $\mathcal{P}$ is peak compute throughput (TFLOPS) and $\mathcal{W}$ is memory bandwidth (TB/s). For an A100 ($\mathcal{P} = 312$ TFLOPS FP16, $\mathcal{W} = 2$ TB/s), the threshold $\mathcal{P}/\mathcal{W} \approx 156$ ops/byte.

**Tiling expressions:**

A GEMM chain $C = A \times B, \; E = C \times D$ over dimensions $m, k, n, h$ has four cross-tile loops. The tiling expression describes the loop structure:

- **Deep tiling**: all loops nested, e.g., $mhnk$ means $m \supset h \supset n \supset k$ (4! = 24 permutations)
- **Flat tiling**: some loops sequential, e.g., $mn(k,h)$ means $m \supset n$ with $k, h$ at the same level

Each tiling expression, combined with tile sizes $\vec{T} = (T_m, T_n, T_k, T_h)$ (multiples of 16 for tensor cores), defines a unique fused kernel candidate.

**Memory access optimization via DAG analysis:**

For a tiling expression $mhnk$ with Load/Store primitives, the DAG identifies:

- **Scope dependency**: $L_A$ is dominated by loops $m, h, n, k$ (must execute within their scopes)
- **Order dependency**: $C_C$ must precede $L_D$ (data flow)
- **Dead node elimination**: When tile size $T_k = K$ (loop $k$ has extent 1), node $k$ and its dependencies can be removed, allowing $L_A$ to move to the $m$ loop scope

The memory access volume for statement $S_t$ with tile size $TS_{X_i}$ and surrounding loops $LP\_set(S)$:

$$
\text{Volume}(S_t) = TS_{X_i} \times \prod_{l_j \in LP\_set(S)} l_j
$$

**Store relocation optimization:**

Store node $S_E$ in expression $mhnk$ is surrounded by loops $m, h, n, k$, but $k$ is not used to index $E$. Relocating $S_E$ from inside loop $k$ to outside it (expression $mh(n(k, L_A, L_B, C_C), L_D, C_E, S_E)$) reduces store traffic by factor $k$:

$$
\text{Volume}_{\text{before}}(S_E) = T_E \cdot m \cdot h \cdot n \cdot k, \quad \text{Volume}_{\text{after}}(S_E) = T_E \cdot m \cdot h \cdot n
$$

**Analytical performance model:**

$$
t_{\text{estm}} = (t_{\text{mem}} + t_{\text{comp}}) \times \alpha
$$

where:

$$
t_{\text{mem}} = \sum_{S_{X_i} \in \{L_{X_i}\} + \{S_{X_i}\}} \frac{TS_{X_i} \times \prod_{l_j \in LP\_set(S)} l_j}{\mathcal{W}}
$$

$$
t_{\text{comp}} = \sum_{S_{X_i} \in \{C_{X_i}\}} \frac{Fp_{X_i} \times \prod_{l_j \in LP\_set(S)} l_j}{\mathcal{P}}
$$

$$
\alpha = \frac{N_{\text{block}} + N_{\text{SM}}}{N_{\text{block}}}
$$

where $N_{\text{SM}}$ is the number of streaming multiprocessors, $N_{\text{block}}$ is the number of thread blocks in the fused kernel, and $\alpha$ captures the parallelism slowdown when there are too few blocks to fill all SMs.

**Shared memory estimation (for pruning):**

$$
Shm_{\text{estm}} = \sum_{\{X_i\}, X_i \in \mathbb{R}^{L_i \times L_j}} (T_{L_i} \times T_{L_j})
$$

Candidates with $Shm_{\text{estm}} > 1.2 \times Shm_{\max}$ are pruned (>90% accuracy in predicting feasibility).

**Key Definitions:**

- MBCI operator — A compute-intensive operator (e.g., GEMM) that becomes memory-bound due to small reduction dimensions
- Tiling expression — A compact notation encoding the loop nesting structure and tile sizes of a fused kernel candidate
- Deep tiling — All loops fully nested (maximizes reuse but limits parallelism)
- Flat tiling — Some loops at the same level (increases parallelism, may reduce reuse)
- DAG analysis — Data flow graph analysis that identifies redundant memory accesses by examining scope and order dependencies between loop, compute, and load/store nodes
- Analytical performance model — A formula estimating kernel execution time from memory bandwidth, compute throughput, and parallelism without runtime measurement

## Complexity

| Method | Avg. Speedup (A100) | Avg. Speedup (RTX 3080) | Tuning Time |
|--------|---------------------|------------------------|-------------|
| PyTorch | 1× (baseline) | 1× (baseline) | — |
| Ansor | ~1.0× | ~1.0× | 4895s (GEMM), 2897s (self-attn) |
| BOLT | ~1.5× | N/A (no sm86) | 88s (GEMM), N/A |
| FlashAttention | ~2.5× | — | — (hand-tuned) |
| MCFuser | **6.6×** (GEMM), **8.1×** (self-attn) | **3.7×** (GEMM), **5.8×** (self-attn) | **35s** (GEMM), **39s** (self-attn) |

**Tuning time comparison:**

| Component | BOLT | Ansor | MCFuser |
|-----------|------|-------|---------|
| GEMM chains | 88s | 4895s | **35s** (2.5×/139× faster) |
| Self-attention | — | 2897s | **39s** (74× faster) |
| End-to-end (Bert-Large) | 383s | 4.06h | **2.98h** (1.36× faster) |

**Search space pruning effectiveness (GEMM chain, $M=N=1024, K=H=512$):**

| Stage | Candidates |
|-------|-----------|
| Original | $\sim 10^8$ |
| + Rule 1 (deduplication) | $\sim 10^6$ (26→5 tiling expr.) |
| + Rule 2 (SMEM overflow) | $\sim 10^5$ (5→3 tiling expr.) |
| + Rule 3 (padding) | $\sim 10^4$ (99% reduction) |
| + Rule 4 (SMEM limit) | $\sim 10^4$ (40% further reduction) |

**Memory:** Fused kernels eliminate intermediate HBM materialization between GEMM operators. For a two-GEMM chain $C = A \times B, E = C \times D$, fusion eliminates the $O(M \times N)$ write/read of $C$.

## Applicability

- **Transformer self-attention modules**: The primary use case — the $QK^T$ and $\text{score} \times V$ GEMMs in self-attention have small $K = H$ (head dimension, typically 64–128), making them MBCI operators. MCFuser fuses both GEMMs with intermediate softmax into a single kernel, achieving 3.3× speedup over FlashAttention on certain configurations
- **Batch GEMM chains**: Any chain of batched GEMMs where the reduction dimension varies (e.g., attention score computation followed by value aggregation) benefits from MBCI-aware tiling
- **MLP layers with small hidden dimensions**: When the MLP intermediate dimension is small (as in distilled or pruned models), the up/down projection GEMMs become MBCI operators amenable to fusion
- **ViT (Vision Transformer) attention**: ViT models with small patch counts lead to small $M, N$ dimensions, pushing attention GEMMs into memory-bound territory. MCFuser achieves speedups on ViT-Base through ViT-Huge configurations
- **MLP-Mixer**: The token-mixing and channel-mixing MLPs involve batched GEMMs that can be MBCI depending on sequence length and hidden dimensions
- **Any model with BERT-like self-attention**: End-to-end evaluation on Bert-Small, Bert-Base, and Bert-Large shows 1.21–1.45× speedup over the best compiler baseline

## Limitations

- **MBCI identification is dimension-dependent**: Whether an operator is MBCI depends on runtime tensor dimensions, not just the operator type. A GEMM that is compute-bound at sequence length 2048 may be memory-bound at length 128 — MCFuser's analysis must be re-run per configuration
- **Two compute-intensive operators evaluated**: The paper primarily evaluates chains of two GEMMs (the attention pattern). Extension to longer chains (3+ GEMMs) is claimed to generalize but not extensively benchmarked
- **No softmax fusion**: MCFuser focuses on fusing GEMM chains and applies standard fusion for memory-intensive operators (softmax, layernorm) via existing frameworks (Relay, Ansor). The GEMM + softmax + GEMM fusion requires combining MCFuser with these complementary tools
- **Analytical model accuracy**: The performance model achieves correlation coefficients of 0.80–0.92 with actual performance — sufficient for ranking candidates but not for precise latency prediction. The top-$n$ profiling step is still needed
- **NVIDIA-specific**: Built on TVM + Triton, targeting NVIDIA GPUs (A100, RTX 3080). The analytical model parameters ($\mathcal{P}$, $\mathcal{W}$, $N_{\text{SM}}$, $Shm_{\max}$) must be re-calibrated per GPU architecture
- **Static tensor dimensions**: MCFuser generates kernels for fixed tensor dimensions. Dynamic shapes require re-compilation or pre-compiling a set of candidate kernels
- **No backward pass**: The paper evaluates inference-mode performance only; training (backward pass) fusion is not addressed

## Implementation Notes

```python
# Pseudocode for MCFuser's search and optimization pipeline

def mcfuser_optimize(gemm_chain, gpu_spec):
    """
    Optimize a chain of GEMM operators for MBCI fusion.

    gemm_chain: list of (M, N, K) tuples for each GEMM
    gpu_spec: (P_compute, W_bandwidth, N_SM, Shm_max)
    """
    P, W, N_SM, Shm_max = gpu_spec

    # Step 1: Generate tiling expressions
    num_loops = count_cross_tile_loops(gemm_chain)  # e.g., 4 for 2-GEMM

    # Deep tiling: all permutations
    deep_tilings = permutations(range(num_loops))  # num_loops! options
    # Flat tiling: sequential loop pairs
    flat_tilings = find_flat_tilings(gemm_chain)

    all_tilings = deep_tilings + flat_tilings

    # Step 2: Generate tile sizes (multiples of 16, up to dim size)
    tile_sizes = []
    for dim in gemm_chain.dimensions:
        tile_sizes.append([t for t in range(16, dim+1, 16)])

    # Step 3: Enumerate candidates
    candidates = [(tiling, tiles)
                  for tiling in all_tilings
                  for tiles in product(*tile_sizes)]

    # Step 4: Prune search space
    # Rule 1: Deduplication — merge tilings with same sub-tiling expression
    candidates = deduplicate(candidates)  # 26 → 5 expressions

    # Rule 2: Prevent SMEM overflow from partial results
    candidates = [c for c in candidates
                  if not causes_smem_overflow(c)]  # 5 → 3 expressions

    # Rule 3: Avoid excessive padding
    candidates = [c for c in candidates
                  if padding_ratio(c) < 0.05]  # 99% reduction

    # Rule 4: Shared memory limit
    candidates = [c for c in candidates
                  if estimate_smem(c) <= 1.2 * Shm_max]  # 40% reduction

    # Step 5: DAG-based memory access optimization
    for c in candidates:
        dag = build_dag(c.tiling, gemm_chain)
        # Relocate stores to outermost valid loop
        optimize_store_placement(dag)
        # Remove dead nodes when tile size = dim size (loop extent = 1)
        eliminate_dead_nodes(dag)
        c.optimized_dag = dag

    # Step 6: Analytical performance estimation
    for c in candidates:
        t_mem = sum(
            tile_size(s) * product(loop_extents(s)) / W
            for s in c.load_store_nodes
        )
        t_comp = sum(
            flops_per_tile(s) * product(loop_extents(s)) / P
            for s in c.compute_nodes
        )
        N_block = product(
            dim // tile for dim, tile in zip(c.spatial_dims, c.tiles)
        )
        alpha = (N_block + N_SM) / N_block
        c.estimated_time = (t_mem + t_comp) * alpha

    # Step 7: Heuristic search (evolutionary algorithm)
    population = random.sample(candidates, N=100)
    best_candidate = None

    while True:
        # Sort by estimated performance
        population.sort(key=lambda c: c.estimated_time)
        top_k = population[:8]  # top-n for actual measurement

        # Measure actual performance of top candidates
        measured = [(c, profile_kernel(c)) for c in top_k]
        best_measured = min(measured, key=lambda x: x[1])

        if best_candidate and abs(best_measured[1] - best_time) < epsilon:
            break  # Converged

        best_candidate, best_time = best_measured

        # Mutate: weighted sampling + tile size mutation
        weights = [1.0 / c.estimated_time for c in population]
        population = weighted_sample_and_mutate(population, weights)

    return best_candidate


# Example: GEMM chain tiling expression mhnk
# (m outer, h, n, k inner — deep tiling)
#
# for i1, i2 in grid(m, h):     # spatial loops → blockIdx
#     Load(tile A)                # A[i1*Tm:(i1+1)*Tm, :]
#     for i3 in range(n):        # reduction loop 1
#         Load(tile B)            # B[:, i3*Tn:(i3+1)*Tn]
#         Compute(tile C)         # C += A @ B
#         Load(tile D)            # D[i3*Tn:(i3+1)*Tn, :]
#         for i4 in range(k):    # reduction loop 2 (k extent may be 1!)
#             Compute(tile E)     # E += C @ D
#         Store(tile E)           # moved outside k loop by DAG optimization
#
# When K = H (tile Tk = K, k loop has extent 1):
#   DAG removes k node → L_A moves to m loop scope
#   Memory traffic for A reduced by factor h*n
```

## References

- Zhang, Z., Yang, D., Zhou, X., Cheng, D. "MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators." SC 2024. arXiv:2506.22169
- Zheng, L., et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.
- Xing, J., et al. "BOLT: Bridging the Gap between Auto-tuners and Hardware-native Performance." ICML Workshop on ML and Systems, 2022.
- Zheng, S., et al. "Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion." HPCA 2023.
- Zheng, Z., et al. "AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-intensive ML Training and Inference on Modern SIMT Architectures." ASPLOS 2022.
- Niu, W., et al. "DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion." PLDI 2021.
- Chen, T., et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
- Tillet, P., Kung, H. T., Cox, D. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MLSys 2019.

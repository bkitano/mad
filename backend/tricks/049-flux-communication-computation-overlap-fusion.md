# 049: Flux: Fine-Grained Communication-Computation Overlap via Kernel Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Chang, Bao, Hou, Jiang, Zheng, Zhong, Zhang, Song, Jiang, Yao, Lin, Jin, Liu — "FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion" (arXiv 2024)
**Paper**: [papers/flux-communication-overlap-kernel-fusion.pdf]
**Documented**: 2026-02-15

## Description

Flux is a fine-grained communication overlapping technique that fuses inter-GPU collective communication (AllGather, ReduceScatter) directly into GEMM kernel epilogues and prologues, hiding up to 96% of communication latency behind dependent computation. This addresses a critical bottleneck in distributed training and inference: tensor parallelism partitions layers across GPUs, requiring AllGather before and ReduceScatter after each GEMM, and communication can consume 20–75% of total runtime depending on GPU architecture and interconnect.

The key insight is that **tile-level decomposition** — far finer-grained than prior chunk-level approaches — naturally maps communication into GEMM's existing tiled execution model. In a tiled GEMM, each thread block computes one output tile independently. Flux fuses communication into the GEMM kernel so that:

1. **GEMM-ReduceScatter**: As each thread block finishes computing its output tile in the GEMM epilogue, it immediately writes the result to the correct remote GPU via peer-to-peer (P2P) memory access, rather than writing to local HBM first and launching a separate ReduceScatter kernel. Communication happens tile-by-tile as computation completes.

2. **AllGather-GEMM**: The host asynchronously transfers input tiles from remote GPUs and sets per-tile signals. In the GEMM prologue, each thread block waits (via spinning) on the signal for its input tile before beginning computation. Tiles whose data has already arrived proceed immediately; tiles waiting for data are context-switched by the warp scheduler, hiding the latency behind other warps' computation.

Unlike prior "medium-grained" methods that split a single GEMM into $N_{\text{TP}}$ smaller GEMMs (degrading GEMM efficiency due to reduced parallelism and SM underutilization), Flux launches **a single large GEMM kernel** with communication woven into its prologue/epilogue. This preserves full GEMM compute efficiency while achieving superior overlap.

Flux is implemented on NVIDIA CUTLASS with auto-tuning across GPU architectures (A100, H800) and interconnects (PCIe, NVLink), and achieves up to 1.24× training speedup and 1.66× prefill speedup on real LLM workloads.

## Mathematical Form

**Tensor-parallel MLP forward pass** (the canonical use case):

Given $N_{\text{TP}}$ GPUs with tensor parallelism, the MLP forward pass on each GPU $r$ computes:

$$
\mathbf{Y}_r = \text{GeLU}\!\left(\mathbf{X}_{\text{full}} \cdot \mathbf{W}_{1,r}\right) \cdot \mathbf{W}_{2,r}
$$

where $\mathbf{X}_{\text{full}} \in \mathbb{R}^{BL \times E}$ is the full activation (gathered from all GPUs), $\mathbf{W}_{1,r} \in \mathbb{R}^{E \times F/N_{\text{TP}}}$ and $\mathbf{W}_{2,r} \in \mathbb{R}^{F/N_{\text{TP}} \times E}$ are local weight shards.

**Communication pattern:**
- Before GEMM1: $\text{AllGather}(\mathbf{X}_r) \to \mathbf{X}_{\text{full}}$ (gather sharded activations)
- After GEMM2: $\text{ReduceScatter}(\mathbf{Y}_r^{\text{partial}}) \to \mathbf{Y}_r$ (sum partial results across GPUs)

**Non-overlapped execution time:**

$$
T_{\text{non-overlap}} = T_{\text{AllGather}} + T_{\text{GEMM1}} + T_{\text{GeLU}} + T_{\text{GEMM2}} + T_{\text{ReduceScatter}}
$$

**Effective Communication Time (ECT):**

$$
\text{ECT} = T_{\text{overall}} - T_{\text{GEMM}_{\text{non-split}}}
$$

where $T_{\text{GEMM}_{\text{non-split}}}$ is the time of the fastest non-split GEMM kernel. A perfect overlapping method achieves $\text{ECT} = 0$.

**Overlap Efficiency:**

$$
E_{\text{overlap}} = 1 - \frac{\text{ECT}_{\text{overlap}}}{\text{ECT}_{\text{non-overlap}}}
$$

A value of 1.0 means perfect overlap (communication fully hidden); negative values mean the method is slower than no overlap.

**Fused GEMM-ReduceScatter** (Algorithm 1 from paper):

For $\mathbf{C} = \mathbf{A} \times \mathbf{B}$ with $N_{\text{TP}}$-way tensor parallelism, each thread block:

1. Compute tile coordinates: $[m, n] \leftarrow \text{TileCoord}(\text{blockIdx}, \text{rank}, N_{\text{TP}})$
2. Execute standard GEMM mainloop: accumulate $\text{acc} = \sum_k \mathbf{A}_{mk} \mathbf{B}_{kn}$
3. In the epilogue, select the output destination:

$$
C_{\text{dest}} \leftarrow \text{GetOutput}(\{C_0, C_1, \ldots, C_{N_{\text{TP}}-1}\}, N_{\text{TP}}, m, n)
$$

4. Write directly to the destination GPU's memory via P2P:

$$
\text{Write}(C_{\text{dest}}, \text{acc}) \quad \text{// remote write, no local HBM intermediate}
$$

For ReduceScatter, the output pointer is selected based on $m$ (row index) to route each tile's result to the correct rank's partition.

**Fused AllGather-GEMM** (Algorithm 2):

Each thread block waits for its input tile's signal before computing:

1. $\text{signal} \leftarrow \text{GetSignal}(\text{signal\_list}, N_{\text{TP}}, m, n)$
2. $\text{WaitSignal}(\text{signal})$ — spin until the host sets the signal after transferring the tile
3. Execute standard GEMM: $C_{mn} = A_{\text{agg}}[m, :] \times B[:, n]$

Communication tiles are transferred asynchronously by the host, which sets signals upon completion. Warps waiting for signals are context-switched by the GPU scheduler, allowing warps with ready data to compute.

**Tile coordinate swizzling** (for ReduceScatter):

To avoid memory controller contention when multiple ranks write to the same partition simultaneously, tile coordinates are shifted by rank:

$$
\text{tile\_coord}(r) = (\text{tile\_idx} + r \cdot \text{shift}) \bmod T_{\text{total}}
$$

This ensures that at any time step, different ranks write to different memory partitions.

**Key Definitions:**

- $N_{\text{TP}}$ — Degree of tensor parallelism (number of GPUs sharing a layer)
- $E$ — Model hidden dimension (embedding size)
- $F$ — MLP intermediate dimension (typically $4E$ or $8E/3$ for SwiGLU)
- $BL$ — Batch size × sequence length (the M dimension of GEMM)
- P2P — Peer-to-peer GPU memory access (via NVLink or PCIe BAR1)
- TileCoord — Function mapping thread block index to output tile, incorporating rank-aware swizzling
- Signal — A 32-bit flag in GPU memory set by the host to indicate data arrival

## Complexity

| Method | GEMM Efficiency | Communication Overlap | Kernel Launches |
|--------|----------------|----------------------|-----------------|
| Non-overlapping (PyTorch+NCCL) | Optimal ($T_g$) | 0% | 2 (GEMM + collective) |
| Medium-grained (TransformerEngine) | Degraded ($T_m > T_g$) | Partial | $2 N_{\text{TP}}$ (split GEMMs + P2P ops) |
| Flux (fine-grained fusion) | Near-optimal ($T_f \geq T_g$) | Up to 96% | 1 (fused kernel) |

**Overlap efficiency (from paper, operation-level):**

| Configuration | Flux Overlap Eff. | TransformerEngine Overlap Eff. |
|---------------|-------------------|-------------------------------|
| A100 PCIe | 41% to 57% | -125% to 36% |
| A100 NVLink | 36% to 96% | -99% to 74% |
| H800 NVLink | 37% to 93% | -40% to 80% |

**Model-level speedups (end-to-end):**

| Workload | vs. Non-overlapping | vs. TransformerEngine |
|----------|--------------------|-----------------------|
| GPT-3 175B training (A100 PCIe) | 1.37× | 1.24× |
| GPT-3 175B prefill (A100 PCIe) | 2.06× | 1.38× |
| GPT-3 175B training (H800 NVL) | 1.14× | 1.10× |
| GPT-3 175B prefill (H800 NVL) | 1.41× | 1.26× |
| Llama-2 70B decoding (H800 NVL, bs=512) | 1.26× | 1.04× |

**Memory:** Flux requires $O(N_{\text{TP}})$ output pointers and $O(T_{\text{comm}})$ signal flags — negligible overhead. No additional activation memory beyond standard GEMM.

## Applicability

- **Tensor-parallel Transformer training**: The primary use case — all GEMM layers (attention QKV projection, output projection, MLP up/gate/down projections) in tensor-parallel training benefit from fused AllGather-GEMM and GEMM-ReduceScatter
- **LLM inference (prefill and decoding)**: Prefill with large batch sizes benefits strongly (up to 2.06× speedup). Decoding with smaller batch sizes also benefits (1.26–1.76×), especially on slower interconnects
- **MoE layers**: Expert-parallel MoE requires AlltoAll communication between experts; Flux's tile-level fusion naturally extends to this pattern
- **Sequence parallelism**: When combined with sequence parallelism (splitting along the sequence dimension), Flux handles the associated AllGather/ReduceScatter on activations
- **Multi-node training**: Flux supports 16-way tensor parallelism across 2 nodes (8 GPUs per node), overlapping both intra-node NVLink and inter-node network communication
- **Any GEMM-centric distributed computation**: Applicable wherever a GEMM is immediately preceded or followed by a collective communication operation

## Limitations

- **P2P requirement for ReduceScatter**: The fused ReduceScatter writes directly to remote GPU memory, requiring peer-to-peer support (NVLink or PCIe BAR1). All modern multi-GPU nodes support this, but it may not work across all network fabrics without NVSHMEM.
- **Small M dimension**: When the GEMM's M dimension is very small (e.g., $m = 64$ in decoding), there are few tiles to overlap communication with, and the per-tile overhead becomes significant. Flux can be 0.95× slower than TransformerEngine in this edge case.
- **CUTLASS dependency**: Flux is implemented as CUTLASS template specializations, tightly coupling it to NVIDIA's ecosystem. Porting to AMD ROCm or other backends requires significant effort.
- **Auto-tuning cost**: Flux auto-tunes across GEMM algorithms, tile sizes, communication tile sizes, pull/push modes, and communication orders — the search space is large, though this is a one-time cost per configuration.
- **No prologue fusion for ReduceScatter**: Only the AlltoAll (write) portion of ReduceScatter is fused into the epilogue; the local reduction is handled separately (with marginal additional overhead).
- **Interconnect sensitivity**: On very fast interconnects (H800 NVLink), communication is already fast relative to computation, so the overlap benefit is smaller (37–93%) compared to slow interconnects (A100 PCIe: 41–57%).

## Implementation Notes

```python
# Pseudocode for Flux fused GEMM-ReduceScatter kernel

def fused_gemm_reduce_scatter(A, B, output_ptrs, rank, N_TP):
    """
    Fused GEMM + ReduceScatter.
    output_ptrs[i] points to GPU i's output partition.
    Each thread block writes its tile directly to the correct remote GPU.
    """
    # 1. Tile coordinate with rank-aware swizzling
    m, n = tile_coord_swizzled(blockIdx, rank, N_TP)

    # 2. Standard GEMM mainloop (unchanged)
    acc = zeros(TILE_M, TILE_N)
    for k in range(0, K, TILE_K):
        a_tile = load_smem(A, m, k)
        b_tile = load_smem(B, k, n)
        acc += wgmma(a_tile, b_tile)  # or mma on Ampere

    # 3. Epilogue: write directly to destination GPU
    dest_rank = m // (M // N_TP)  # which GPU owns this row partition
    dest_ptr = output_ptrs[dest_rank]
    local_m = m % (M // N_TP)

    # P2P remote write (via st/cp.async.bulk to remote GPU memory)
    write_tile(dest_ptr, local_m, n, acc)
    # No intermediate local HBM write! Tile goes directly to remote GPU.


def fused_allgather_gemm(A_agg, B, C, signal_list, rank, N_TP):
    """
    Fused AllGather + GEMM.
    Host transfers input tiles and sets signals; kernel waits on signals.
    """
    m, n = tile_coord(blockIdx, rank, N_TP)

    # 1. Prologue: wait for input tile to arrive from remote GPU
    signal = get_signal(signal_list, N_TP, m, n)
    wait_signal(signal)  # spin until host sets this signal
    # While spinning, warp scheduler switches to other ready warps

    # 2. Standard GEMM (input A_agg is now populated by host-side AllGather)
    C[m, n] = gemm_tile(A_agg, B, m, n)


# Host-side AllGather communication (runs concurrently with kernel)
def host_allgather(A_list, A_agg_list, signal_list, rank, N_TP, tiles_comm):
    """Asynchronously gather tiles from remote GPUs, setting signals."""
    for tile in tiles_comm:
        if pull_mode:
            # Pull from remote GPU's memory to local aggregation buffer
            src = get_remote_ptr(A_list, tile)
            dst = get_local_ptr(A_agg_list, tile)
            cuda_memcpy_async(dst, src, tile.size)
        else:
            # Push local data to remote GPU's aggregation buffer
            src = get_local_ptr(A_list, tile)
            dst = get_remote_ptr(A_agg_list, tile)
            cuda_memcpy_async(dst, src, tile.size)

        # Signal the kernel that this tile is ready
        signal = get_signal_host(signal_list, tile)
        set_signal(signal)  # kernel warps waiting on this signal will proceed
```

## References

- Chang, L.-W., Bao, W., Hou, Q., Jiang, C., Zheng, N., Zhong, Y., Zhang, X., Song, Z., Jiang, Z., Yao, C., Lin, H., Jin, X., Liu, X. "FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion." 2024. arXiv:2406.06858
- Jangda, A., et al. "Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads." ASPLOS 2022.
- Wang, S., et al. "Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models." ASPLOS 2023.
- NVIDIA. "TransformerEngine." https://github.com/NVIDIA/TransformerEngine
- NVIDIA. "CUTLASS." https://github.com/NVIDIA/cutlass
- NVIDIA. "NVSHMEM." https://developer.nvidia.com/nvshmem

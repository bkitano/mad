# 091: Persistent Megakernel Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Aimuyo, Oh, Singh — "FlashMoE: Fast Distributed MoE in a Single Kernel" (NeurIPS 2025)
**Paper**: [papers/flashmoe-persistent-megakernel.pdf]
**Documented**: 2026-02-15

## Description

Persistent megakernel fusion is a GPU kernel design pattern that fuses an *entire compound operator* — including multiple GEMMs, element-wise operations, gating, and inter-GPU communication — into a single persistent GPU kernel that remains active for the entire duration of the operator. Unlike standard kernel fusion (which chains a few adjacent operations) or epilogue fusion (which appends post-GEMM ops), persistent megakernel fusion eliminates **all** kernel launch boundaries within a complex multi-phase computation, keeping the GPU fully occupied with a single launch.

The technique was demonstrated in FlashMoE, which fuses the full distributed Mixture-of-Experts (MoE) layer — gate computation, token dispatch (AlltoAll), expert FFN (two GEMMs with activation), expert-combine (AlltoAll), and output weighting — into one kernel. Prior MoE implementations launch 33–550 separate GPU kernels per layer; FlashMoE launches exactly **1**.

The key enabling abstractions are:

1. **Actor-based concurrency model**: GPU thread blocks and warps are specialized into distinct roles — **Processors** (compute GEMMs and element-wise ops), a **Scheduler** (assigns tasks to processors based on readiness), and **Subscribers** (decode incoming communication packets from remote GPUs). This replaces CPU-managed kernel orchestration with fully GPU-resident, reactive task scheduling.

2. **Tile-level task abstraction**: All computation is decomposed into uniform *tasks* operating on tiles. A task descriptor $t = (\mathcal{M}, \star, \phi)$ encodes metadata, a binary tensor operation (matmul or Hadamard product), and an element-wise activation. This unified interface allows the Scheduler to dynamically assign heterogeneous work (FFN GEMM0, GEMM1, combine) to any available Processor.

3. **Device-initiated communication**: Instead of CPU-launched bulk-synchronous collectives (AlltoAll via NCCL), FlashMoE uses NVSHMEM for one-sided, device-initiated (R)DMA transfers directly from within the persistent kernel, enabling tile-granularity overlap of communication and computation.

4. **Symmetric tensor layout**: A write-conflict-free memory layout $L \in \mathbb{R}^{P \times R \times B \times E \times C \times H}$ enables fully non-blocking one-sided memory accesses without synchronization, where temporal buffering ($R=2$ rounds, $B$ staging buffers) eliminates read-write conflicts.

The result is **93.17% SM utilization** (vs. 9.67–59.11% for baselines), up to **6x lower latency**, **9x higher GPU utilization**, and **5.7x higher throughput** on 8 H100 GPUs.

## Mathematical Form

**MoE forward pass** (the full computation fused into one kernel):

Given input tokens $A \in \mathbb{R}^{S \times H}$, expert weights $X \in \mathbb{R}^{E \times H \times D}$, and $E$ local experts per GPU across $P$ GPUs:

**Step 1 — Fused Gate:**

$$
G_\phi = \text{softmax}(A \cdot W_g) \in \mathbb{R}^{S \times E_W}
$$

where $E_W$ is the total number of experts across all GPUs. Top-$k$ routing selects experts $E_{i,k}$ and affinities $g_{i,e}$ for each token $i$.

**Step 2 — Dispatch (AlltoAll):**

Tokens are sent to their assigned experts' GPUs via device-initiated (R)DMA. The routing table:

$$
T_\phi \in (\mathbb{N} \times \mathbb{R})^{E \times C}, \quad T_\phi(e, c) = (i, w)
$$

indicates token $i$ dispatched to expert $e$ at slot $c$ with combine weight $w$.

**Step 3 — Expert FFN (two GEMMs + activation):**

$$
\text{FFN}(x) = W_2 \cdot \phi(x W_1 + b_1) + b_2
$$

Expressed as two tasks:

$$
t_1 = (\mathcal{M}, \cdot, \phi_1): \quad C_1 \leftarrow \phi_1(A B_1 + D_1)
$$
$$
t_2 = (\mathcal{M}, \cdot, \phi_2): \quad C_2 \leftarrow \phi_2(C_1 B_2 + D_2)
$$

where $\phi_1$ is the activation (GELU/SiLU) and $\phi_2$ is identity.

**Step 4 — Expert-Combine (AlltoAll + weighted sum):**

$$
C_i = \sum_{k=1}^{K} g_{i,e}, \quad \mathbf{h}_i = \sum_{k=1}^{K} \frac{g_{i,e}}{C_i} \cdot \mathbf{h}_i^k
$$

Expressed as task $t_3 = (\mathcal{M}, \odot, \phi_2)$:

$$
\mathcal{F}_{t_3}(A, S, C, C) := C \leftarrow \phi_2(A \odot S + C)
$$

where $\odot$ is the Hadamard product for token-wise weighted combination.

**Unified task execution** (Processor actor):

For any task $t = (\mathcal{M}, \star, \phi)$:

$$
\mathcal{F}_t(A, B, C, D) := C \leftarrow \phi(A \star_t B + D)
$$

where $\star_t$ instantiates as matrix multiply ($\cdot$) for FFN tasks or Hadamard product ($\odot$) for combine tasks.

**Symmetric tensor layout** for conflict-free communication:

$$
L \in \mathbb{R}^{P \times R \times B \times E \times C \times H}
$$

where $P$ = expert parallel world size, $R = 2$ (dispatch + combine rounds), $B$ = staging buffers, $E$ = local experts, $C$ = expert capacity, $H$ = hidden dimension.

$$
\text{Size}(L) \approx 4 \cdot \text{Size}(T)
$$

contributing $\leq 2\%$ of GPU memory for inference of popular models.

**Key Definitions:**

- Persistent kernel — A GPU kernel that remains resident on SMs for the entire duration of the operator, with work dynamically assigned by an in-kernel scheduler rather than by the CPU
- Actor model — Concurrent computation model where independent actors (Processor, Scheduler, Subscriber) communicate via messages through shared/global memory
- Task descriptor — Tuple $(\mathcal{M}, \star, \phi)$ encoding all information needed for a Processor to execute a tile-level computation
- NVSHMEM — NVIDIA's implementation of OpenSHMEM providing a Partitioned Global Address Space (PGAS) for device-initiated inter-GPU communication
- Expert capacity $C$ — Maximum number of tokens an expert can receive per forward pass

## Complexity

| Metric | Baseline MoE (DeepEP/Megatron) | Persistent Megakernel (FlashMoE) |
|--------|-------------------------------|----------------------------------|
| Kernel launches per layer | 33–550 | **1** |
| SM utilization | 9.67%–59.11% | **93.17%** |
| CPU involvement | Continuous (launch + sync) | Single launch only |
| Communication model | Bulk-synchronous AlltoAll | Tile-level async (R)DMA |
| Idle gaps | Frequent (kernel boundaries) | Eliminated |

**Performance (8 H100 GPUs, 32 experts, 16K tokens, top-2 routing):**

| Metric | FlashMoE vs. Best Baseline |
|--------|---------------------------|
| Forward latency | up to **6.4x** lower (vs. Megatron-TE at 8 GPUs) |
| GPU SM utilization | **9x** higher (vs. FasterMoE) |
| Throughput | **5.7x** higher (17.7 MTokens/s at 8 GPUs) |
| Overlap efficiency | **4x** better weak scaling efficiency |
| Expert scalability | **6.6x** faster at 128 experts (8 GPUs) |

**Memory:** Symmetric tensor layout adds $\sim 4\times$ token buffer size ($\leq 2\%$ GPU memory). Tile dimensions (128, 64) balance register pressure, shared memory, and occupancy.

## Applicability

- **Distributed MoE inference**: The primary use case — any MoE model with expert parallelism (DeepSeek, Mixtral, DBRX, Snowflake Arctic, Qwen3, GPT-OSS) benefits from fusing the full MoE layer into a single kernel
- **MoE training (future)**: The paper targets inference; extending to training requires fusing backward computation and gradient communication with new task descriptors
- **Any multi-phase distributed operator**: The actor-based persistent kernel pattern generalizes beyond MoE to any operator requiring interleaved computation and communication across GPUs (e.g., expert-parallel attention, pipeline-parallel micro-batches)
- **Ultra-sparse MoE configurations**: FlashMoE's in-kernel scheduling and payload-efficient communication maintain uniform latency even with 128+ experts, where baselines degrade superlinearly due to kernel launch overhead
- **Heterogeneous interconnects**: The design works across NVLink and PCIe, with even larger relative gains on slower interconnects where communication overlap is more critical

## Limitations

- **Engineering complexity**: Building a fully fused persistent kernel with actor-based concurrency, NVSHMEM integration, and custom GEMM requires deep GPU systems expertise — far beyond standard CUDA/Triton programming
- **Inference only (currently)**: The paper does not address training; backward pass fusion requires additional task descriptors for gradient computation and communication
- **FP16 inefficiency**: The current implementation's FP16 path is suboptimal due to insufficient tuning; all reported results use FP32 (despite baselines using FP16), meaning actual gains may be even larger with optimized lower-precision support
- **NVIDIA-specific**: Relies on NVSHMEM for device-initiated communication and CUTLASS for in-kernel GEMM — porting to AMD ROCm or other platforms requires reimplementation of both subsystems
- **Static expert capacity**: Expert capacity $C$ must be pre-allocated; dynamic load imbalance beyond $C$ causes token dropping (though in-place padding mitigates wasted communication)
- **Single-node evaluated**: Results shown for up to 8 GPUs on a single node with NVLink; multi-node performance with RDMA over Infiniband is projected but not yet demonstrated

## Implementation Notes

```python
# Pseudocode for FlashMoE persistent megakernel (Algorithm 1 from paper)

def flashmoe_persistent_kernel(A, O, X, N):
    """
    Single persistent GPU kernel for the entire distributed MoE layer.
    A: input token matrix (S x H)
    O: output token matrix (S x H)
    X: expert weight tensor (E x H x D)
    N: number of thread blocks on GPU
    """
    # Step 1: Fused gate computation (all blocks participate)
    T_phi, G_phi = fused_gate(A)  # routing table + affinity scores

    block_id = get_block_idx()

    if block_id + 1 < N:
        # --- PROCESSOR role (N-1 blocks do compute) ---
        dispatch(T_phi, A)  # encode & send tokens to remote GPUs
        processor_start()   # enter reactive task loop

        # Processor event loop (runs until all tasks complete):
        # while True:
        #     task = wait_for_task_from_scheduler()  # spin on SMEM signal
        #     if task.is_poison_pill(): break
        #     # Execute: C <- phi(A *_t B + D)
        #     acc = gemm_mainloop(task.A, task.B, task.tile_coords)
        #     result = task.activation(acc + task.D)
        #     store_tile(task.C_ptr, result)
        #     if task.is_remote:
        #         nvshmem_put(dest_gpu, result)  # device-initiated RDMA
        #     notify_scheduler(task.id, COMPLETE)
    else:
        # --- OS BLOCK (last block: 1 warp = Scheduler, 3 warps = Subscriber) ---
        warp_id = get_warp_id()
        if warp_id == 0:
            # SCHEDULER: assign tasks to Processors based on readiness
            scheduler_start()
            # Scheduler loop:
            # while tasks_remaining > 0:
            #     ready_tasks = check_dependency_signals()
            #     for task in ready_tasks:
            #         idle_processor = find_idle_processor()  # work-conserving
            #         assign_task(idle_processor, task)  # write to SMEM
            #     tasks_remaining -= len(completed_tasks)
        else:
            # SUBSCRIBER: decode incoming tile packets from remote GPUs
            subscriber_start(T_phi, G_phi, O, X)
            # Subscriber loop:
            # while packets_remaining > 0:
            #     packet = poll_nvshmem_signal()  # spin on arrival signal
            #     task = decode_task_descriptor(packet)
            #     notify_scheduler(task)  # signal via SMEM
            #     packets_remaining -= 1


# Actor interaction chain (tile granularity):
#
# GPU_j dispatches tile --> Subscriber_i decodes --> Scheduler_i assigns
# --> Processor_i executes GEMM0 --> Scheduler_i assigns GEMM1
# --> Processor_i executes GEMM1 --> Processor_i sends combine tile
# --> Subscriber_j decodes --> Scheduler_j assigns combine
# --> Processor_j executes combine --> Output tile written


# Tile dimensions chosen for optimal balance:
TILE_M = 128  # tile height
TILE_N = 64   # tile width
THREADS_PER_BLOCK = 128  # fixed
# Rationale: larger tiles increase register pressure and sync overhead;
# smaller tiles underutilize GPU. (128, 64) balances occupancy and
# arithmetic intensity across H100 and A100.
```

## References

- Aimuyo, O. J., Oh, B., Singh, R. "FlashMoE: Fast Distributed MoE in a Single Kernel." NeurIPS 2025. arXiv:2506.04667
- Agha, G. A. "Actors: A Model of Concurrent Computation in Distributed Systems." MIT AI Lab Technical Report, 1985.
- NVIDIA. "NVSHMEM Library." https://docs.nvidia.com/nvshmem/api/nvshmem_api.html
- NVIDIA. "CUTLASS." https://github.com/NVIDIA/cutlass
- Rajbhandari, S., et al. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." ICML 2022.
- Zhang, S., et al. "Comet: Fine-grained Computation-Communication Overlapping for Mixture-of-Experts." MLSys 2025. arXiv:2502.19811
- He, J., et al. "FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models." PPoPP 2022.

# 183: Tawa — Automatic Warp Specialization via Asynchronous References

**Category**: kernel
**Gain type**: efficiency
**Source**: Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight, Zhang, Grover — "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References" (arXiv 2510.14719, Dec 2025)
**Paper**: [papers/tawa-automatic-warp-specialization.pdf]
**Documented**: 2026-02-16

## Description

Writing high-performance GPU kernels for modern hardware (NVIDIA Hopper/Blackwell) requires **warp specialization** — splitting warps into producer (TMA data movement) and consumer (Tensor Core compute) roles that execute concurrently. This is critical because Tensor Cores contribute 90%+ of compute throughput on Hopper, and keeping them fed requires overlapping asynchronous TMA loads with WGMMA execution. However, implementing warp specialization manually requires hundreds of lines of low-level PTX, careful barrier management, circular buffer orchestration, and pipeline staging — making it labor-intensive, error-prone, and fragile.

**Tawa** is an automated compiler that transforms standard, unmodified Triton tile programs into warp-specialized kernels with full TMA/WGMMA pipelining. The key innovation is the **asynchronous reference (`aref`)** — a first-class IR abstraction that models a one-slot communication channel between producer and consumer warps, backed by hardware `mbarrier` synchronization. The `aref` has three operations with formal semantics:

1. **`put(a, v)`** — Producer writes value $v$ to the buffer, flipping state from `empty` to `full`
2. **`get(a)`** — Consumer reads from the buffer when `full`, transitioning to a borrowed state
3. **`consumed(a)`** — Consumer signals it is done, restoring `empty` for the next iteration

Tawa applies three compiler passes: (1) **task-aware partitioning** that splits the computation graph into producer/consumer warp groups based on semantic analysis, (2) **`aref` insertion** that creates communication channels between partitions, and (3) **multi-granularity pipelining** that overlaps TMA loads, address computation (CUDA cores), and matrix multiply (Tensor Cores) at both fine-grained (instruction-level) and coarse-grained (stage-level) granularities.

On H100 GPUs, Tawa achieves **1.1× speedup over cuBLAS** for GEMM (matching the highly optimized closed-source library), **1.2× over Triton** for attention (reaching 96% of hand-tuned FlashAttention-3 CUTLASS C++ performance), and up to **79% hardware utilization** — all from concise Triton-Python code with zero manual warp specialization.

## Mathematical Form

**The `aref` Abstraction:**

An `aref` $a$ maps to a state $\sigma(a) = \langle buf, F, E \rangle$ where $buf$ is the data buffer, $F$ is the `full` mbarrier flag, and $E$ is the `empty` mbarrier flag. Initially $E = 1, F = 0$.

**Formal semantics:**

$$
\textbf{put:} \quad \sigma(a).E = 1 \implies \langle \sigma, \text{put}(a, v) \rangle \to \langle \sigma[a \mapsto \langle buf = v, F = 1, E = 0 \rangle], \epsilon \rangle
$$

$$
\textbf{get:} \quad \sigma(a).F = 1 \implies \langle \sigma, \text{get}(a) \rangle \to \langle \sigma[a \mapsto \langle buf, F = 0, E = 0 \rangle], buf \rangle
$$

$$
\textbf{consumed:} \quad \langle \sigma, \text{consumed}(a) \rangle \to \langle \sigma[a \mapsto \langle buf, F = 0, E = 1 \rangle], \epsilon \rangle
$$

The `put`→`get`→`consumed` cycle induces a happens-before chain: producer writes are visible to consumers, and buffer reuse is safe after `consumed`. Multiple `aref` instances form a $D$-slot cyclic buffer for deep pipelining.

**Task-Aware Partitioning Algorithm:**

Given a computation graph $G = (V, E)$ from the Triton MLIR:

1. **Backward traversal** from side-effecting sinks (stores), tagging each node:
   - **Iteration statements** (orange): Address computation, pointer arithmetic → producer
   - **Tile statements** (blue): WGMMA, element-wise transforms → consumer

2. **Dependency-closed subgraphs**: Partition $V$ into producer $V_P$ and consumer $V_C$ such that each partition is self-contained. Nodes used by both partitions are duplicated.

3. **Loop distribution**: The original loop is cloned into two WG regions. Each carries an isomorphic loop over tiles $k = 0, \ldots, K-1$. Communication between WGs is via `aref`.

On Hopper: $V_P \to$ WG0 (1 warp, TMA loads), $V_C \to$ WG1 (4 warps, WGMMA). On Blackwell: additional partitions for load, compute, reduction, and epilogue warp groups.

**Multi-Granularity Pipelining:**

*Fine-grained pipeline (within consumer WG):*

Overlaps MMA address calculation (CUDA cores) with MMA execution (Tensor Cores) through a bounded pipeline of depth $P$:

$$
\text{Iteration } k: \quad \text{WGMMA.issue}(k) \to \text{WGMMA.wait}(\text{pendings} = P) \to \text{aref.consumed}(k - P)
$$

The $k$-th MMA is issued asynchronously; the consumer only stalls when $P$ operations are in flight, keeping the Tensor Core pipeline saturated.

*Coarse-grained pipeline (across stages):*

Each iteration $j$ decomposes into stages $T_j$ (Tensor Core), $C_j$ (CUDA Core transform), and optionally $U_j$ (downstream Tensor Core):

$$
\text{Steady state:} \quad T_j \parallel C_{j-1} \parallel U_{j-1}
$$

For attention: $T_j = QK^T$ (GEMM0), $C_j = \text{softmax}$, $U_j = PV$ (GEMM1). This is the compiler's automatic discovery of the pingpong-like schedule from FlashAttention-3.

**Key Definitions:**

- `aref` — Asynchronous reference; IR abstraction for inter-warp communication via mbarrier-guarded buffers
- $D$ — Buffer depth (number of `aref` slots in cyclic buffer); controls prefetch distance
- $P$ — MMA pipeline depth; number of outstanding WGMMA instructions before stalling
- WG — Warp Group (4 warps = 128 threads on Hopper); the unit of WGMMA execution
- TMA — Tensor Memory Accelerator; hardware unit for async global→shared memory copies
- WGMMA — Warp Group Matrix Multiply-Accumulate; async Tensor Core instruction
- mbarrier — Memory barrier; hardware synchronization primitive with arrival count semantics

## Complexity

**GEMM throughput (M = N = 8192, varied K, H100 SXM5):**

| Framework | FP16 TFLOPs/s (avg) | FP8 TFLOPs/s (avg) | vs cuBLAS |
|-----------|---------------------|---------------------|-----------|
| cuBLAS (v12.7) | baseline | baseline | 1.00× |
| Triton (0c7edf) | 0.88× | 0.82× | — |
| TileLang | 0.87× | 0.81× | — |
| ThunderKittens | 0.91× | — | — |
| **Tawa** | **1.01×** | **1.06×** | **1.01–1.06×** |

**Attention throughput (batch=4, nheads=4, d=128, H100 SXM5):**

| Framework | FP16 (L=16K) | FP8 (L=16K) | vs FA3 (CUTLASS) |
|-----------|-------------|-------------|-----------------|
| FA3 (CUTLASS C++) | ~740 TFLOPs/s | ~1000 TFLOPs/s | 1.00× |
| Triton | ~530 TFLOPs/s | ~620 TFLOPs/s | 0.72× |
| **Tawa** | **~710 TFLOPs/s** | **~890 TFLOPs/s** | **0.96×** |

**Ablation (GEMM, K=16384, FP16):**

| Configuration | TFLOPs/s | vs Baseline |
|---------------|----------|-------------|
| Triton (no WS) | 104 | 1.00× |
| + Auto WS (1 compute WG) | 393 | 3.78× |
| + Cooperative WGs (2 compute WGs) | 395 | 3.80× |
| + Large tile size (128×256×64) | 572 | 5.50× |
| + Persistent kernel | 632 | 6.08× |
| + Better aref size (D=3, P=2) | **718** | **6.90×** |

**MHA ablation (L=16384, FP16):**

| Configuration | TFLOPs/s |
|---------------|----------|
| Triton (no WS) | 209 |
| + Auto WS + Cooperative WGs | 593 |
| + Coarse-grained pipeline | 645 |
| + Better aref size | **654** |

## Applicability

- **TFLA / Flash Linear Attention kernels (high priority):** TFLA is written in Triton. Tawa can automatically transform TFLA's tile program into a warp-specialized kernel — overlapping TMA loads of $Q, K, V$ tiles with the inner-tile WGMMA computations ($QK^T$, $SV$, $QC$). This would bring TFLA closer to FlashAttention-3's hardware utilization without manual CUDA rewriting. Users simply set `enable_warp_specialization=True`.

- **Mamba-2 / SSD Triton kernels:** The chunkwise SSD kernel has the same GEMM + element-wise + GEMM structure that Tawa's coarse-grained pipeline handles automatically. The `aref` abstraction could manage the inter-chunk state buffer pipelining.

- **DeltaNet chunkwise kernels:** DeltaNet's intra-chunk computation involves UT transform (forward substitution) followed by matmuls — a perfect match for Tawa's $T_j \to C_j \to U_j$ coarse-grained pipeline.

- **Any Triton kernel with back-to-back matmuls:** Tawa's automatic partitioning works on any computation graph with TMA loads feeding Tensor Core operations. MLP layers, gated linear units, and MoE expert computation all benefit.

- **FP8 workloads:** Tawa's gains are even larger for FP8 (1.24× over ThunderKittens) because smaller FP8 tiles make Tensor Core computation so fast that memory transfer becomes the bottleneck — exactly what warp specialization addresses.

## Limitations

- **Hopper+ hardware required:** Relies on TMA and WGMMA instructions available only on H100/H200/B100+ (sm_90+). Not applicable to A100 or earlier.

- **No pingpong scheduling (yet):** Tawa currently implements producer-consumer pipelines but does not yet generate the inter-warp-group pingpong schedule from FlashAttention-3 (where two consumer WGs alternate GEMM and softmax). The authors note this as future work. This means Tawa reaches 96% of FA3 but not 100%.

- **Register pressure with deep pipelines:** Increasing the `aref` buffer depth $D$ and MMA pipeline depth $P$ improves throughput but increases register pressure. At $D = 3, P = 3$, register spilling can negate gains. The sweet spot is typically $D = 2\text{--}3, P = 1\text{--}2$.

- **Compiler overhead:** Tawa adds compilation passes to the Triton pipeline. While the overhead is small for production kernels (compiled once), it increases iteration time during development.

- **Limited to double-buffering communication patterns:** The current `aref` design targets producer-consumer pipelines. More complex patterns like multicast (one producer, multiple consumers) or reduction require extensions.

- **Triton ecosystem dependency:** Tawa is built on OpenAI Triton's MLIR infrastructure. It does not apply to hand-written CUDA kernels or other DSLs (though the `aref` concept is portable).

## Implementation Notes

```python
# Tawa usage: automatic warp specialization for a Triton GEMM kernel
# No code modifications needed — just enable the flag

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_desc, b_desc, c_ptr,
    M, N, K,
    stride_cm, stride_cn,
    Mt: tl.constexpr, Nt: tl.constexpr, Kt: tl.constexpr,
):
    """Standard Triton GEMM — NO warp specialization annotations."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, Mt)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    o_am = pid_m * Mt
    o_bn = pid_n * Nt
    o_k = 0

    acc = tl.zeros((Mt, Nt), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, Kt)):
        a = tl.tma_load(a_desc, [o_am, o_k], [Mt, Kt])
        b = tl.tma_load(b_desc, [o_bn, o_k], [Nt, Kt])
        acc += tl.dot(a, b.T, acc=acc)
        o_k += Kt

    # Store result
    offs_cm = pid_m * Mt + tl.arange(0, Mt)
    offs_cn = pid_n * Nt + tl.arange(0, Nt)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :]
    tl.store(c_ptrs, acc)


# Launch with Tawa: just add enable_warp_specialization=True
# Tawa automatically:
# 1. Partitions into producer (TMA) and consumer (WGMMA) warp groups
# 2. Inserts aref communication channels
# 3. Creates multi-stage pipeline with circular buffer
# 4. Generates optimized PTX with mbarrier synchronization
matmul_kernel[(grid,)](
    a_desc, b_desc, c_ptr, M, N, K,
    stride_cm, stride_cn,
    Mt=128, Nt=256, Kt=64,
    enable_warp_specialization=True,  # <-- This is the only change
)


# What Tawa generates internally (simplified MLIR):
#
# WG0 (producer, 1 warp):
#   for k in range(K // Kt):
#     tma_load(a[k]) -> aref.put(a_buf[k % D])
#     tma_load(b[k]) -> aref.put(b_buf[k % D])
#
# WG1 (consumer, 4 warps):
#   for k in range(K // Kt):
#     a = aref.get(a_buf[k % D])
#     b = aref.get(b_buf[k % D])
#     wgmma.issue(a, b, acc)        # async Tensor Core
#     wgmma.wait(pendings=P)         # stall only when pipeline full
#     aref.consumed(a_buf[(k-P) % D])
#     aref.consumed(b_buf[(k-P) % D])
#   # epilogue: store acc


# For attention kernels, Tawa additionally generates
# a coarse-grained pipeline that overlaps QK^T, softmax, PV:
#
# WG0 (producer):  Load K_j, V_j via TMA
# WG1 (consumer):
#   S_next = WGMMA(Q, K_j)       [Tensor Core]
#   softmax(S_cur)                 [CUDA Core, overlapped with WGMMA]
#   O += WGMMA(P_cur, V_{j-1})   [Tensor Core]
```

**Key engineering insights:**

1. **Upstreamed to Triton:** Tawa has been merged into OpenAI Triton (PR #6288), available at `triton/triton/tree/aref_auto_ws`. Users of Triton-based frameworks (FLA, TFLA, Mamba) can benefit immediately.

2. **Zero annotation required:** Unlike TileLang (requires `T.pipelined` and `T.copy` annotations) or CUTLASS (requires manual template configuration), Tawa works on unmodified Triton kernels.

3. **Cooperative warp groups:** When register pressure limits the consumer WG's tile size, Tawa can split computation across multiple consumer WGs that cooperatively process the same tile, pooling their registers to enable larger tiles. This is transparent to the user.

4. **Persistent kernels:** Tawa automatically converts to persistent kernel mode when beneficial — launching one CTA per SM and iterating over tiles, eliminating launch overhead for long-running kernels.

5. **Directly applicable to TFLA:** Since TFLA's mLSTM kernels are written in Triton (https://github.com/NX-AI/mlstm_kernels), enabling `warp_specialization=True` could yield significant speedups by overlapping TMA loads of Q, K, V, gate tiles with the inner WGMMA computations — potentially closing the gap between TFLA's Triton implementation and a hand-optimized CUTLASS version.

## References

- Chen, H., Fan, B., Collins, A., Hagedorn, B., Gaburov, E., Masuda, M., Brookhart, M., Sullivan, C., Knight, J., Zhang, Z., & Grover, V. (2025). "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References." arXiv:2510.14719.
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024. arXiv:2407.08608.
- Spector, B., Arora, A., Singhal, A., Parthasarathy, D., Fu, D., & Ré, C. (2025). "ThunderKittens: Simple, Fast, and Adorable Kernels." ICLR 2025.
- Wang, L., Cheng, Y., et al. (2025). "TileLang: A Composable Tiled Programming Model for AI Systems." arXiv:2504.17577.
- Soi, R., Yadav, R., Kjolstad, F., Aiken, A., et al. (2025). "Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs." arXiv:2512.18134.
- Tillet, P., Kung, H., & Cox, D. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." SIGPLAN 2019.

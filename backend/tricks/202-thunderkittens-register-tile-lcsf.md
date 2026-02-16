# 202: ThunderKittens — Register-Tile LCSF Kernel Framework

**Category**: kernel
**Gain type**: efficiency
**Source**: Spector, Arora, Singhal, Fu & Ré (2024) — Stanford / Hazy Research (ICLR 2025)
**Paper**: [papers/thunderkittens-register-tile-dsl.pdf]
**Documented**: 2026-02-15

## Description

ThunderKittens (TK) is a C++ embedded DSL that provides a minimal, opinionated set of abstractions mapping to the three levels of GPU parallelism — warp, thread-block, and grid — to write high-performance AI kernels. The central insight is that a **small number of tile-based abstractions** can achieve state-of-the-art performance across a broad range of AI operations (GEMM, attention, linear attention, SSMs, FFT convolutions) while being dramatically simpler than CUTLASS/CuTe templates or hand-tuned CUDA.

**Why this matters for TFLA-style kernels:** TFLA (trick 158) tiles linear attention matmuls at two sequence-dimension levels, but the innermost tile computations ($B_{Lhq} \times B_{dq}$ @ $B_{dq} \times B_{Lkv}$, etc.) rely on tensor core matrix-multiply-accumulate (MMA) instructions. TK provides the **register-level tile primitives** that implement these innermost matmuls optimally:

1. **16×16 register tiles** (`rt_bf<16, N>`) as the fundamental data structure, sized to match tensor core MMA operand requirements
2. **Automatic bank-conflict-free SMEM layouts** via compile-time layout selection (strided at 32, 64, or 128 bytes), eliminating the 8-way bank conflicts that even FlashAttention-3's CUTLASS-based kernel suffers from
3. **Load-Compute-Store-Finish (LCSF) template** for warp-specialized producer-consumer pipelining, overlapping TMA/async loads with tensor core compute
4. **Persistent grid scheduling** to amortize block launch/teardown costs and improve L2 cache reuse

On H100, TK kernels match CuBLAS on GEMM (up to 855 TFLOPS), match FlashAttention-3 on attention forward pass, outperform FA3 by 10-40% on attention backward pass, achieve **14× speedup** over FLA Triton on polynomial-based linear attention, and **3× speedup** over Mamba-2 Triton on SSM kernels.

## Mathematical Form

TK does not introduce new mathematical operations — it provides efficient primitives for existing operations. The key abstraction is the **tile**:

**Register tile (warp-level):**

$$
\texttt{rt\_bf<16, N>} \quad \leftrightarrow \quad A \in \mathbb{R}^{16 \times N}_{BF16}
$$

A register tile is distributed across 32 threads of a warp (4 "quadrants"), with each thread owning a fragment of the $16 \times N$ matrix in the layout required by tensor core MMA instructions.

**Shared tile (block-level):**

$$
\texttt{st\_bf<M, N>} \quad \leftrightarrow \quad B \in \mathbb{R}^{M \times N}_{BF16} \text{ in SMEM}
$$

Shared tiles use one of three bank-conflict-minimizing layouts (stride 32B, 64B, or 128B), selected at compile time based on tile width.

**Core operations on tiles (PyTorch-like API):**

| TK Operation | Equivalent | Hardware Unit |
|---|---|---|
| `mma_AB(dst, A, B)` | $D \mathrel{+}= A \cdot B$ | Tensor core (WGMMA) |
| `mma_ABt(dst, A, B)` | $D \mathrel{+}= A \cdot B^\top$ | Tensor core (WGMMA) |
| `mul(dst, A, B)` | $D = A \odot B$ | FMA pipeline |
| `exp(dst, src)` | $D_{ij} = e^{S_{ij}}$ | XU (special function) |
| `sub_row(dst, src, vec)` | $D_{ij} = S_{ij} - v_j$ | ALU |
| `load(reg, smem[idx])` | SMEM → registers | Load/store unit |
| `tma::load_async(smem, glob, ...)` | HBM → SMEM (async) | TMA unit |

**LCSF Template — Producer-Consumer Pipelining:**

The LCSF template partitions warps in a thread block into two roles:

$$
\text{Thread block} = \underbrace{W_0}_{\text{load/store workers}} \cup \underbrace{W_1, \ldots, W_{n-1}}_{\text{compute workers}}
$$

With $N$-stage pipelining ($N$ shared memory buffers):

$$
\text{Stage}_{t}: \quad \underbrace{\text{TMA load}(B_{t+N})}_{\text{producer } W_0} \parallel \underbrace{\text{MMA}(B_t)}_{\text{consumer } W_{1..n}}
$$

The key efficiency gain: while compute workers execute tensor core MMA on buffer $B_t$, load workers asynchronously prefetch $B_{t+N}$ via TMA, hiding the HBM latency behind compute.

**Cost Model:**

TK provides a simplified cost model for kernel execution time:

$$
C_{\text{Overall}} = \max\left(\underbrace{C_{\text{HBM}}, C_{\text{L2}}, C_{\text{L1}}, C_{\text{Shared}}}_{\text{Memory}}, \underbrace{C_{\text{Tensor}}, C_{\text{ALU}}, C_{\text{FMA}}, C_{\text{XU}}}_{\text{Compute}}\right) + \underbrace{C_{\text{Setup}} + C_{\text{Sync}}}_{\text{Overhead}}
$$

This represents the ideal case of perfect overlapping — actual performance lies between this max and the sum of all components.

## Complexity

TK does not change asymptotic complexity; it improves constant factors by maximizing hardware utilization.

**Measured throughput (H100 SXM, BF16):**

| Operation | Best Baseline | TK | Speedup |
|-----------|--------------|-----|---------|
| GEMM 16384×16384 | CuBLAS: 804 TFLOPS | 793 TFLOPS | 0.99× |
| Attn Fwd (causal, D=128, L=12288) | FA3: 598 TFLOPS | 550 TFLOPS | 0.92× |
| Attn Bwd (causal, D=128, L=12288) | FA3: 449 TFLOPS | 494 TFLOPS | **1.10×** |
| Attn Bwd (non-causal, D=128, L=12288) | FA3: 500 TFLOPS | 553 TFLOPS | **1.11×** |
| Mamba-2 SSM (D=64, L=16384) | Triton: 40 TFLOPS | 140 TFLOPS | **3.5×** |
| Based linear attn (D=64, L=16384) | FLA Triton: 25 TFLOPS | 217 TFLOPS | **8.7×** |
| Hedgehog linear attn (D=128, L=16384) | FLA Triton: 22 TFLOPS | 142 TFLOPS | **6.5×** |
| Long conv (D=1024, L=4096) | FlashFFTConv: 13 TFLOPS | 61 TFLOPS | **4.7×** |
| Fused dropout-residual-norm (D=1024) | FlashNorm Triton: 0.8 TFLOPS | 1.2 TFLOPS | **1.5×** |

**Why TK is faster (NCU profiling, attn bwd vs FA3):**

| Metric | FA3 Bwd | TK Bwd |
|--------|---------|--------|
| Tensor core utilization | 61.2% | 58.2% |
| Issue slot utilization | 25.1% | 34.8% |
| HBM throughput (GB/s) | 328 | 490 |
| HBM stall cycles | 1.83 | 1.63 |
| Shared memory stall cycles | 0.92 | **0.14** |
| Bank conflicts | **9.6-way** | **0** |

The key finding: FA3 suffers from **up to 9.6-way bank conflicts** in shared memory despite using CUTLASS/CuTe. TK's automatic layout selection eliminates these entirely (0.14 vs 0.92 stall cycles).

**Pipeline depth effect on GEMM (4096×4096):**

| Pipeline Stages | TFLOPS |
|----------------|--------|
| 1 | 260 |
| 2 | 484 |
| 3 | 683 |
| 4 | 760 |

**Memory:** TK library is <1 MB (vs CuBLAS 689 MB, CUTLASS 22 MB, Triton 12.6 MB).

## Applicability

- **TFLA intra-chunk matmuls (direct application):** TFLA's innermost operations — $Q_i K_j^\top$ ($B_{Lhq} \times B_{dq}$ @ $B_{dq} \times B_{Lkv}$), $S_{ij} V_j$ ($B_{Lhq} \times B_{Lkv}$ @ $B_{Lkv} \times B_{dhv}$), and $\bar{Q}_i C_{k-1}$ ($B_{Lhq} \times B_{dq}$ @ $B_{dq} \times B_{dhv}$) — are all matmuls that map directly to TK's `mma_AB` / `mma_ABt` primitives on 16×16 register tiles. The LCSF template can pipeline the loading of K/V tiles from SMEM while computing on the current tile.

- **Linear attention (Based, Hedgehog):** Already demonstrated: 8.7× and 6.5× speedup over FLA Triton. The chunkwise-parallel structure of these architectures maps naturally to TK's tile abstractions.

- **Mamba-2 / SSD kernels:** 3.5× speedup demonstrated. The structured state-space duality computation involves matmuls that benefit from TK's register tiling.

- **FlashAttention-style softmax attention:** Matches FA3 forward, 10-40% faster backward. The LCSF template's producer-consumer pattern naturally expresses the tiled attention algorithm.

- **Long convolutions (SSMs, Hyena):** 4.7× speedup over FlashFFTConv via FFT computed with Monarch matrix factorization on register tiles.

- **Fused elementwise + norm ops:** Dropout-residual-layernorm fusion, rotary encoding — 1.2-1.5× over Triton.

## Limitations

- **NVIDIA-only (H100/A100):** TK uses CUDA-specific intrinsics (WGMMA, TMA, cp.async). AMD port exists separately as HipKittens (arXiv:2511.08083) but requires different algorithms due to different memory hierarchy.

- **C++ only, no Python JIT:** Unlike Triton, TK requires writing C++ device code. This makes rapid prototyping harder — the TFLA authors used Triton precisely for its accessibility. TK is best suited for production kernels after the algorithm is validated.

- **Fixed 16×16 tile granularity:** The fundamental tile is 16×16. For operations with dimensions not divisible by 16, padding or special handling is needed. TFLA's tile sizes ($B_{Lhq}$, $B_{Lkv}$) would need to be multiples of 16.

- **Manual kernel structure:** While TK simplifies the low-level details (layouts, bank conflicts, sync), the developer still must choose the parallelization strategy (which dimensions to parallelize vs. loop over), occupancy, and pipeline depth. The LCSF template provides a framework but not an auto-tuner.

- **Block launch order requires manual tuning:** L2 cache reuse depends on block execution order, which can cause >50% performance degradation if wrong (e.g., row-major vs. column-major for 16384×16384 GEMM). TK provides persistent grid support but the optimal order must be determined empirically.

## Implementation Notes

```cpp
// ThunderKittens attention kernel (simplified from Figure 5 in the paper)
// Demonstrates the LCSF producer-consumer template

using namespace kittens;

// Warp-level: 16x64 register tiles for Q@K^T and att@V
rt_bf<16, 64> k_reg, v_reg;

// === COMPUTE WORKER (consumer) ===

// Step 1: Load K from shared memory to register tile
load(k_reg, k_smem[subtile]);

// Step 2: Compute Q@K^T via tensor core MMA
zero(att);
mma_ABt(att, q_reg, k_reg, att);  // att += Q * K^T

// Step 3: Softmax (non-tensor-core ops on register tiles)
sub_row(att, att, max_vec);   // subtract max for stability
exp(att, att);                // exponentiate
div_row(att, att, norm_vec);  // normalize

// Step 4: Convert to BF16 for next MMA
copy(att_bf16, att);

// Step 5: Load V from shared memory to register tile
load(v_reg, v_smem[subtile]);
auto &v_reg_col = swap_layout_inplace(v_reg);  // row → col major

// Step 6: Compute att@V via tensor core MMA
mma_AB(o_reg, att_mma, v_reg_col, o_reg);  // o += att * V

// === LOAD WORKER (producer) runs concurrently ===
// if(warpgroup::warpid() == 0) {
//     tma::expect(inputs_arrived, input);
//     tma::load_async(block.k, globals.K, {batch, head, iter, 0}, inputs_arrived);
//     tma::load_async(block.v, globals.V, {batch, head, iter, 0}, inputs_arrived);
// } else arrive(inputs_arrived);
```

**Applying TK to TFLA:**

A TFLA kernel built on TK primitives would:

1. **Outer parallelization:** Grid over (batch, head, chunk $k$, query tile $i$, value dim tile $dhv\_b$) — same as TFLA
2. **LCSF template:** Producer warps load K/V tiles from HBM→SMEM; consumer warps run the three TFLA matmuls on register tiles
3. **Inner loop:** Consumer iterates over $B_{Lkv}$ tiles (key-value sequence) and $B_{dq}$ tiles (query dimension), accumulating $S_{ij}$ in SMEM and $h\_acc$ in registers
4. **Bank-conflict-free layouts:** TK automatically selects SMEM layouts for K, V, Q tiles that avoid the bank conflicts affecting Triton-based TFLA
5. **Pipeline depth:** Use 3-4 stage pipeline buffers to fully hide HBM latency behind tensor core compute

The 14× linear attention speedup over FLA Triton suggests that a TK-based TFLA kernel could achieve significantly higher throughput than the current Triton implementation, primarily by eliminating bank conflicts and enabling deeper async pipelining.

**GPU efficiency analysis:**

1. **Memory access coalesced:** TK's TMA loads guarantee coalesced access patterns; register tile layouts are optimized for tensor core consumption
2. **Bank-conflict-free:** Compile-time layout selection eliminates SMEM bank conflicts (0 stall cycles vs 0.92 for FA3)
3. **High arithmetic intensity:** LCSF pipelining keeps tensor cores fed continuously; 4-stage pipeline achieves 760 TFLOPS on GEMM (93% of peak)
4. **Maps to tensor cores:** All matmuls use WGMMA instructions via `mma_AB` / `mma_ABt`
5. **No sequential bottlenecks:** Producer-consumer pattern is fully pipelined; the only sequential part is the inter-chunk state recurrence (inherited from TFLA)

## References

- Spector, B. F., Arora, S., Singhal, A., Fu, D. Y., & Ré, C. (2024). ThunderKittens: Simple, Fast, and Adorable AI Kernels. arXiv:2410.20399. ICLR 2025.
- Code: https://github.com/HazyResearch/ThunderKittens
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. arXiv:2503.14376 (TFLA, trick 158).
- Shah, J., et al. (2024). FlashAttention-3. arXiv:2407.08691.
- Dao, T. & Gu, A. (2024). Transformers are SSMs. ICML 2024.
- Spector, B. F., et al. (2025). ParallelKittens: Multi-GPU AI Kernels. arXiv:2511.13940.

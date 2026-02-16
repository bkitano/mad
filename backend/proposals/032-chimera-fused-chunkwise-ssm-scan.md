---
status: ongoing
priority: high
created: 2026-02-15
based_on: chimera-block-reorder-compute-fusion, chunkwise-parallel-scan, io-aware-tiling, kernel-fusion, warp-specialized-pipelining, bilinear-gating-glu, online-softmax, stream-k-work-centric-gemm
experiment_number: 032
experiment_log: experiment-log-032.md
---

# Chimera-Fused Chunkwise SSM: Analytical GEMM-Chain Fusion for Linear Attention Chunks

## Hypothesis

Applying Chimera-style analytical GEMM-chain fusion to the **intra-chunk computation** of chunkwise parallel SSMs (GLA/DeltaNet/Mamba-2 SSD) — fusing the Q·K^T → decay-mask → attn·V GEMM chain into a single kernel with analytically-optimized block execution order — will reduce HBM traffic by $40$–$60\%$ and achieve $1.3$–$1.8\times$ wall-clock speedup for training, because the intermediate attention-like matrices within each chunk are never materialized to HBM and the block-reordering analytically minimizes DRAM round-trips for the input/output tensors.

## Background

### The chunkwise parallel scan bottleneck

Modern gated SSMs (GLA, Gated DeltaNet, Mamba-2 SSD) use **chunkwise parallel computation** to balance recurrent expressivity with GPU parallelism. The key operation within each chunk of size $C$ is structurally identical to a causal attention computation:

$$
\text{Intra-chunk: } O_j = \underbrace{(\underbrace{Q_j K_j^\top}_{\text{GEMM1}} \odot \underbrace{M_j}_{\text{decay mask}})}_{\text{masked scores}} \cdot \underbrace{V_j}_{\text{GEMM2}}
$$

where $Q_j, K_j, V_j \in \mathbb{R}^{C \times d}$ are the queries, keys, values for chunk $j$, and $M_j \in \mathbb{R}^{C \times C}$ is a causal decay mask. This is exactly the attention GEMM chain that Chimera was designed to optimize.

**Current state:** Existing implementations (flash-linear-attention, Mamba-2) either:
1. **Materialize the $C \times C$ intermediate** to HBM between GEMM1 and GEMM2 (naive approach — wastes bandwidth)
2. **Use FlashAttention-style tiling** to avoid materialization (better, but doesn't analytically optimize the block execution order)

Neither approach applies Chimera's **analytical data movement model** to find the provably optimal block execution order for the specific GEMM chain structure of the chunkwise SSM.

### Why Chimera is the right tool

Chimera's core contribution is an analytical model that, given a GEMM chain $C = A \times B$, $E = C \times D$, computes the **exact data movement volume** as a closed-form function of the tile sizes and block execution order. The model predicts:

1. Which intermediate tensors can be kept entirely on-chip (zero DRAM traffic)
2. What the optimal tile sizes are (via Lagrange multiplier solution)
3. What block execution order minimizes total DRAM traffic

For the chunkwise SSM, the GEMM chain is:
- **GEMM1**: $S_j = Q_j \cdot K_j^\top$ (dimensions: $C \times d \times C$)
- **Elementwise**: $\tilde{S}_j = S_j \odot M_j$ (fused with GEMM1 epilogue)
- **GEMM2**: $O_j = \tilde{S}_j \cdot V_j$ (dimensions: $C \times C \times d$)

This is exactly the structure Chimera handles: a 2-GEMM chain with shared intermediate $\tilde{S}_j \in \mathbb{R}^{C \times C}$. Chimera's analysis guarantees that $\tilde{S}_j$ has **zero DRAM traffic** when the block execution order is chosen correctly.

### Why this hasn't been done

1. **Chimera targets general GEMM chains**, not specifically the chunkwise SSM pattern. Adapting it requires understanding the specific shapes ($C \times d$ vs. $C \times C$) and the causal mask structure.
2. **FlashAttention solved the attention case** for softmax attention, but the chunkwise SSM has **different structure**: no softmax normalization, but includes a multiplicative decay mask and potentially an additive state correction term. These differences change the optimal fusion strategy.
3. **Chunkwise SSM also has an inter-chunk component** (boundary state propagation) that must be fused with the intra-chunk GEMM chain for maximum benefit.

### Distinction from FlashAttention / FlashLinearAttention

| Aspect | FlashAttention-2 | FlashLinearAttention | This Proposal |
|--------|------------------|---------------------|---------------|
| **Target** | Softmax attention | Linear attention | Chunkwise SSM intra-chunk |
| **Normalization** | Online softmax | None (or simple) | Decay mask + state correction |
| **Block order** | Heuristic (outer-K) | Heuristic | **Analytically optimal** |
| **Tile size** | Manually tuned | Manually tuned | **Lagrange-optimized** |
| **Intermediate** | Never materialized | Partially materialized | **Zero DRAM (proven)** |
| **Inter-chunk** | N/A | Separate kernel | **Fused with intra-chunk** |

### Key innovation: fusing the state correction

The chunkwise SSM has a unique structure beyond plain attention: after computing the intra-chunk output $O_j$, there's a **state correction** that propagates the boundary state $h_{jC}$ through the chunk:

$$
\hat{O}_j = O_j + \text{diag}(\text{cumcausal}(\alpha_j)) \cdot h_{jC} \cdot C_j^\top
$$

This correction is a rank-$n$ additive term. In current implementations, this requires a separate kernel launch. By extending Chimera's fusion framework to include this correction as a third stage in the GEMM chain, we can fuse it into the same kernel, eliminating one additional HBM round-trip.

## Mathematical Formulation

### Chunkwise SSM Intra-Chunk as a GEMM Chain

For chunk $j$ with $Q_j, K_j, V_j \in \mathbb{R}^{C \times d}$, decay rates $\alpha_j \in \mathbb{R}^C$, and boundary state $h_{jC} \in \mathbb{R}^{n \times d}$:

**GEMM1 (score computation):**

$$
S_j = Q_j \cdot K_j^\top \in \mathbb{R}^{C \times C}
$$

**Decay masking (fused elementwise):**

$$
\tilde{S}_j = S_j \odot M_j, \quad M_{j,st} = \prod_{i=s+1}^{t} \alpha_{j,i} \quad \text{(lower-triangular causal decay)}
$$

**GEMM2 (output aggregation):**

$$
O_j^{\text{intra}} = \tilde{S}_j \cdot V_j \in \mathbb{R}^{C \times d}
$$

**State correction (rank-$n$ additive):**

$$
O_j = O_j^{\text{intra}} + G_j \cdot (h_{jC} \cdot C_j^\top)^\top
$$

where $G_j \in \mathbb{R}^{C \times n}$ is the cumulative gate product propagating the boundary state.

### Chimera Data Movement Model Applied

**Loop dimensions:** The GEMM chain has 3 dimensions:
- $m$: query position within chunk (range $C$), tile size $T_M$
- $l$: key/value position within chunk (range $C$), tile size $T_L$ (shared between GEMM1 and GEMM2)
- $k$: head dimension (range $d$) for GEMM1 / same for GEMM2, tile size $T_K$

**Tensor access patterns:**

| Tensor | Dims | Accessed by loops |
|--------|------|-------------------|
| $Q_j$ | $C \times d$ | $m, k$ |
| $K_j$ | $C \times d$ | $l, k$ |
| $\tilde{S}_j$ | $C \times C$ | $m, l$ (intermediate — on-chip) |
| $V_j$ | $C \times d$ | $l, k$ |
| $O_j$ | $C \times d$ | $m, k$ |

**Chimera analysis for order $mlk$ (iterate over $k$ innermost, $l$ middle, $m$ outermost):**

| Tensor | Data Movement | Data Footprint |
|--------|---------------|----------------|
| $Q_j$ | $C \cdot d$ (loaded once) | $T_M \cdot T_K$ |
| $K_j$ | $C \cdot d \cdot \lceil C / T_M \rceil$ | $T_L \cdot T_K$ |
| $\tilde{S}_j$ | **0** (fully on-chip) | $T_M \cdot T_L$ |
| $V_j$ | $C \cdot d \cdot \lceil C / T_M \rceil$ | $T_L \cdot T_K$ |
| $O_j$ | $C \cdot d$ (written once) | $T_M \cdot T_K$ |

**Total DRAM traffic:**

$$
\text{DV}_{mlk} = 2Cd + 2Cd \cdot \left\lceil \frac{C}{T_M} \right\rceil
$$

Compare with unfused (materialize $\tilde{S}_j$ to HBM):

$$
\text{DV}_{\text{unfused}} = 2Cd + 2C^2 + 2Cd = 4Cd + 2C^2
$$

For typical values ($C = 64$, $d = 128$):
- Unfused: $4 \cdot 64 \cdot 128 + 2 \cdot 64^2 = 32768 + 8192 = 40960$ elements
- Fused ($T_M = 32$): $2 \cdot 64 \cdot 128 + 2 \cdot 64 \cdot 128 \cdot 2 = 16384 + 32768 = 49152$ — wait, this is worse?

The issue is that for small $C$ (chunk size), the intermediate $\tilde{S}_j \in \mathbb{R}^{C \times C}$ is small. The real win comes when we **also fuse the state correction and output gating** into the chain:

### Extended GEMM Chain (3-stage fusion)

**Stage 1**: $\tilde{S}_j = (Q_j K_j^\top) \odot M_j$ (GEMM + elementwise)
**Stage 2**: $O_j^{\text{intra}} = \tilde{S}_j \cdot V_j$ (GEMM)
**Stage 3**: $O_j = (O_j^{\text{intra}} + G_j h_{jC}^\top) \odot g_j$ (correction + gating)

where $g_j = \text{Swish}(x_j W_{gate})$ is the SwiGLU gate. In the unfused case, stages 1, 2, and 3 are separate kernels with HBM round-trips between each:

$$
\text{DV}_{\text{unfused}} = \underbrace{2Cd + C^2}_{\text{GEMM1 I/O}} + \underbrace{C^2 + Cd + Cd}_{\text{GEMM2 I/O}} + \underbrace{Cd + Cn + Cd + Cd}_{\text{correction+gate I/O}}
$$

$$
= 7Cd + 2C^2 + Cn
$$

With Chimera 3-stage fusion (all intermediates on-chip):

$$
\text{DV}_{\text{fused}} = \underbrace{Cd}_{\text{Q load}} + \underbrace{Cd}_{\text{K load}} + \underbrace{Cd}_{\text{V load}} + \underbrace{Cn}_{\text{h load}} + \underbrace{Cd}_{\text{gate load}} + \underbrace{Cd}_{\text{O write}} = 5Cd + Cn
$$

**DRAM reduction factor:**

$$
\frac{\text{DV}_{\text{unfused}}}{\text{DV}_{\text{fused}}} = \frac{7Cd + 2C^2 + Cn}{5Cd + Cn}
$$

For $C = 64, d = 128, n = 16$:

$$
= \frac{7 \cdot 64 \cdot 128 + 2 \cdot 4096 + 64 \cdot 16}{5 \cdot 64 \cdot 128 + 64 \cdot 16} = \frac{57344 + 8192 + 1024}{40960 + 1024} = \frac{66560}{41984} \approx 1.59\times
$$

### Optimal Tile Sizes (Lagrange Multiplier Solution)

Following Chimera's analytical optimization with shared memory constraint $\text{MC}$ (e.g., 128 KB on A100, 256 KB on H100):

The on-chip memory usage is:

$$
\text{MU} = T_M \cdot T_K + T_L \cdot T_K + T_M \cdot T_L + T_L \cdot T_K + T_M \cdot T_K
$$
$$
= 2T_M T_K + 2T_L T_K + T_M T_L
$$

(tiles for $Q$, $K$, $\tilde{S}$, $V$, $O$ — with $K$ and $V$ having the same tile footprint).

The Lagrange optimization yields:

$$
T_M^* = T_L^* = \sqrt{\frac{\text{MC}}{5}} \approx \sqrt{\frac{128 \cdot 1024 / 2}{5}} \approx 114 \quad \text{(FP16 elements)}
$$

Rounded to Tensor Core-friendly size: $T_M = T_L = 64$ or $128$.

For $T_K$: constrained by $T_K = \min(d, \text{MC} / (2T_M + 2T_L))$. With $T_M = T_L = 64$: $T_K = \min(128, 128 \cdot 1024 / (2 \cdot 256)) \approx 256$ → $T_K = 128$ (full head dim).

### Memory Access Pattern Analysis

**Coalesced access:** $Q_j, K_j, V_j$ are stored in $[C, d]$ layout (contiguous along $d$). Loading $T_M \times T_K$ tiles of $Q$ reads contiguous memory along the $d$ dimension — fully coalesced.

**Cache-friendly:** The $mlk$ block order ensures $Q$ tiles are reused across all $l$-iterations (temporal locality), $K$ and $V$ tiles are loaded once per $(m, l)$ pair and used for both GEMM1 and GEMM2 (data reuse across operators).

**Arithmetic intensity:**

$$
\text{AI}_{\text{fused}} = \frac{2C^2 d + 2C^2 d}{(5Cd + Cn) \times 2} \approx \frac{4C^2 d}{10Cd} = \frac{2C}{5}
$$

For $C = 64$: AI $\approx 25.6$ — **compute-bound** on A100 (AI threshold ~100 for FP16 on A100). For $C = 256$: AI $\approx 102$ — at the compute/memory boundary. This confirms that fusion primarily helps by avoiding **extra HBM round-trips**, not by changing the compute-bound nature.

### Parallelism Analysis

**Inter-chunk parallelism:** All $T/C$ chunks are independent for the intra-chunk computation. With $T = 2048, C = 64$: 32 independent chunk tasks, each decomposed into Chimera tiles.

**Tensor Core mapping:** GEMM1 ($Q \cdot K^\top$) and GEMM2 ($\tilde{S} \cdot V$) both map to standard mma/WGMMA instructions. The decay mask $M_j$ and gate $g_j$ are elementwise operations fused into the GEMM epilogue (free on modern GPU architectures via epilogue visitor trees).

**No sequential bottleneck within chunk:** The $mlk$ loop nest is fully parallelizable across the $m$ dimension (query positions) and pipelineable across $l$ (key positions).

**Warp specialization opportunity (H100):** Producer warps load $K_j, V_j$ tiles via TMA while consumer warps execute GEMM1 on previously loaded tiles — exactly the FlashAttention-3 pattern, but applied to the chunkwise GEMM chain.

### Key Variables

- $C$ — chunk size (typically 64–256)
- $d$ — head dimension (typically 64–128)
- $n$ — state dimension per head (typically 16–64)
- $T_M, T_L, T_K$ — tile sizes for Chimera block decomposition
- $M_j$ — causal decay mask (precomputed, lower-triangular)
- $h_{jC}$ — boundary state from inter-chunk scan
- $g_j$ — SwiGLU gate values

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Gated DeltaNet with Chimera-fused chunks |
| Layers | $L = 12$ |
| Hidden dim | $d_{model} = 768$ |
| Head dim | $d = 64$ |
| Heads | $H = 12$ |
| State dim | $n = 16$ per head |
| Chunk size | $C = 64$ (and $C = 128, 256$ for ablation) |
| Kernel | Triton with Chimera-optimized block ordering |
| Tile sizes | Analytically determined per hardware target |

### Baseline

1. **Naive chunkwise (separate kernels)**: GEMM1, mask, GEMM2, correction, gate as 5 separate kernels — worst case for HBM traffic
2. **FlashLinearAttention**: Existing fused implementation (heuristic tiling) — current state-of-the-art
3. **FlashAttention-2 (softmax)**: For reference on the fusion quality ceiling
4. **Unfused matmul chain (cuBLAS)**: Pure GEMM calls with materialized intermediates

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.3\times$ FlashLinearAttention | Tokens/sec, A100, batch 16, seq 2048 |
| DRAM traffic | $< 0.6\times$ naive | NCU memory throughput profiling |
| Tensor Core utilization | $> 60\%$ | NCU SM active cycles |
| Kernel launch count | 1 per chunk (vs 3–5 naive) | nsight systems trace |
| Quality (PPL) | Numerically identical | WikiText-103 (same model, different kernel) |
| Shared memory usage | $< 128$ KB per CTA (A100) | Compile-time analysis |
| Chunk sizes supported | $C \in \{32, 64, 128, 256\}$ | Parameterized kernel |

### Estimated Compute

**MVE (kernel benchmark)**: ~30 minutes on single GPU (~$1.50) — kernel development + benchmarking
**Small-scale (integration test)**: 2 GPU-hours on A100 (~$8) — verify correctness on small model
**Full-scale**: 24 GPU-hours on A100 (~$100) — end-to-end training comparison

## Expected Outcome

**If hypothesis is correct:**

- Chimera-fused chunkwise SSM achieves $1.3\times$ throughput over FlashLinearAttention at $C = 64$, scaling to $1.5$–$1.8\times$ at $C = 256$ (larger chunks = more fusion benefit)
- DRAM traffic reduced by $40$–$59\%$ (measured via NCU), matching the analytical prediction
- Intermediate $\tilde{S}_j$ matrix has exactly 0 HBM traffic (verified via memory profiling)
- Kernel launch count drops from 3–5 to 1 per chunk, eliminating launch overhead
- No quality change (this is a pure kernel optimization — numerically equivalent to unfused)
- On H100, warp-specialized variant achieves additional $1.2\times$ over A100 version (from TMA + WGMMA pipelining)

**If hypothesis is wrong:**

- **Scenario A**: Chimera tile sizes conflict with Tensor Core requirements (mma tiles are 16×16×16, Chimera may suggest non-aligned sizes) — **Learn**: need to add Tensor Core alignment constraints to Chimera's optimization. **Fix**: constrain $T_M, T_L, T_K$ to multiples of 16.
- **Scenario B**: The chunk size $C$ is too small for fusion to help — $\tilde{S}_j \in \mathbb{R}^{64 \times 64}$ is only 8 KB (FP16), which fits trivially in L2 cache even without explicit fusion. **Learn**: Chimera's benefit is for large intermediates; for small $C$, the L2 cache already provides implicit fusion. **Follow-up**: test at $C = 256$ or $512$ where intermediates are 64–128 KB (exceed L1 cache).
- **Scenario C**: The Triton compiler's auto-tuner already finds a near-optimal schedule — Chimera's analytical solution doesn't improve over Triton's search. **Learn**: Triton's auto-tuning is sufficient for simple 2-GEMM chains. **Value**: Chimera becomes useful only for 3+ stage chains (e.g., including the inter-chunk correction).

## Minimum Viable Experiment

### Setup
- **Model**: No model training needed — this is a **kernel benchmark**
- **Task**: Benchmark the fused vs. unfused chunkwise SSM kernel on synthetic data
- **Data**: Random $Q, K, V \in \mathbb{R}^{B \times H \times C \times d}$ with $B = 16, H = 12, C = 64, d = 64$
- **Compute**: Single GPU, $< 30$ minutes (kernel development in Triton + benchmarking)

### Implementation Sketch

```python
import triton
import triton.language as tl

@triton.jit
def chimera_fused_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, Gate_ptr, H_ptr, O_ptr,
    C: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Chimera-fused intra-chunk computation.

    Fuses: S = QK^T, S_masked = S * M, O_intra = S_masked @ V,
           O = (O_intra + G @ H) * gate
    into a single kernel with zero intermediate HBM traffic.
    """
    # Block indices
    pid_m = tl.program_id(0)  # Query block
    # pid for batch/head handled by grid dims 1, 2

    # Load Q tile: [BLOCK_M, D] — loaded once, reused across all L blocks
    q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_tile = tl.load(Q_ptr + q_offsets[:, None] * D + tl.arange(0, BLOCK_K)[None, :])

    # Accumulate output in registers: [BLOCK_M, D]
    o_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Chimera order: iterate over L (key positions) in outer loop
    # For each L block: compute GEMM1 partial -> mask -> accumulate GEMM2 partial
    for l_start in range(0, C, BLOCK_L):
        # Load K tile: [BLOCK_L, D]
        k_offsets = l_start + tl.arange(0, BLOCK_L)
        k_tile = tl.load(K_ptr + k_offsets[:, None] * D + tl.arange(0, BLOCK_K)[None, :])

        # GEMM1: S_partial = Q_tile @ K_tile^T -> [BLOCK_M, BLOCK_L] (in registers!)
        s_partial = tl.dot(q_tile, tl.trans(k_tile))

        # Apply causal decay mask (precomputed or computed on-the-fly)
        m_offsets = q_offsets[:, None] - k_offsets[None, :]
        causal_mask = (m_offsets >= 0).to(tl.float32)
        s_partial = s_partial * causal_mask  # TODO: add decay rates

        # Load V tile: [BLOCK_L, D] — reuse K memory slot
        v_tile = tl.load(V_ptr + k_offsets[:, None] * D + tl.arange(0, BLOCK_K)[None, :])

        # GEMM2: O_partial += S_partial @ V_tile -> [BLOCK_M, D] (accumulated in registers)
        o_acc += tl.dot(s_partial.to(v_tile.dtype), v_tile)

    # Fuse state correction: O += G @ H^T (rank-n update, cheap)
    # ... (load G, H tiles and add rank-n correction)

    # Fuse gating: O *= sigmoid(gate)
    gate_tile = tl.load(Gate_ptr + q_offsets[:, None] * D + tl.arange(0, BLOCK_K)[None, :])
    o_final = o_acc * tl.sigmoid(gate_tile)

    # Write output: [BLOCK_M, D] — only HBM write for output
    tl.store(O_ptr + q_offsets[:, None] * D + tl.arange(0, BLOCK_K)[None, :], o_final)
```

### Benchmark Protocol

```python
import torch
import time

def benchmark_fused_vs_unfused(B=16, H=12, C=64, D=64, n_warmup=50, n_bench=200):
    Q = torch.randn(B, H, C, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, C, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, C, D, device='cuda', dtype=torch.float16)
    gate = torch.randn(B, H, C, D, device='cuda', dtype=torch.float16)

    # Unfused baseline
    def unfused():
        S = torch.matmul(Q, K.transpose(-1, -2))
        mask = torch.tril(torch.ones(C, C, device='cuda'))
        S = S * mask
        O = torch.matmul(S, V)
        return O * torch.sigmoid(gate)

    # Fused kernel (Triton)
    def fused():
        return chimera_fused_chunk(Q, K, V, gate)

    # Warmup
    for _ in range(n_warmup):
        unfused()
        fused()
    torch.cuda.synchronize()

    # Benchmark
    for name, fn in [("unfused", unfused), ("fused", fused)]:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_bench):
            fn()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"{name}: {elapsed/n_bench*1000:.2f} ms")
```

### Success Criteria

- Fused kernel is $> 1.2\times$ faster than unfused at $C = 64, d = 64$
- Fused kernel is $> 1.5\times$ faster than unfused at $C = 256, d = 128$
- NCU profiling shows 0 bytes of DRAM traffic for intermediate $\tilde{S}$ matrix
- Numerical output matches unfused to within FP16 precision ($< 10^{-3}$ relative error)
- Kernel compiles and runs correctly for all $C \in \{32, 64, 128, 256\}$

### Failure Criteria

- **Kill if**: Fused kernel is $< 1.05\times$ faster than unfused at any chunk size — L2 cache already handles the intermediate, making explicit fusion unnecessary
- **Kill if**: Triton auto-tuner finds equivalent or better schedule without Chimera's analytical guidance — the analytical model doesn't add value over search
- **Investigate if**: Speedup is significant ($> 1.2\times$) but only at $C \geq 256$ — fusion only helps for large chunks, suggesting we should use larger chunk sizes in practice

### Why This Test Is Sufficient

- This is a **kernel-level optimization** — success is measured purely by wall-clock throughput and DRAM traffic, not model quality (which is numerically identical)
- The benchmark directly measures the core mechanism: DRAM traffic reduction via intermediate elimination
- If the fused kernel is faster at $C = 64$ (the common chunk size), it immediately benefits all GLA/DeltaNet/Mamba-2 implementations
- The Triton implementation can be integrated into flash-linear-attention with minimal code changes (replace the intra-chunk kernel)

## Theoretical Analysis

### Complexity Comparison

| Operation | Unfused (3 kernels) | Fused (1 kernel) | FlashLinearAttention |
|-----------|--------------------|--------------------|---------------------|
| DRAM reads | $5Cd + C^2 + Cn$ | $3Cd + Cn$ | $\sim 3Cd + Cn$ |
| DRAM writes | $2Cd + C^2$ | $Cd$ | $\sim Cd$ |
| **Total DRAM** | $7Cd + 2C^2 + Cn$ | $4Cd + Cn$ | $\sim 4Cd + Cn$ |
| Kernel launches | 3–5 | **1** | 1 |
| Shared mem | $O(T_M \cdot T_K)$ per kernel | $O(T_M T_L + T_L T_K)$ | Similar |
| Tensor Core util. | ~40% (launch gaps) | **>60%** | ~50-60% |

### DRAM Traffic Reduction

$$
\text{Reduction} = 1 - \frac{4Cd + Cn}{7Cd + 2C^2 + Cn} = 1 - \frac{4 \cdot 64 \cdot 64 + 64 \cdot 16}{7 \cdot 64 \cdot 64 + 2 \cdot 64^2 + 64 \cdot 16} = 1 - \frac{17408}{37632} \approx 54\%
$$

### Advantage Over FlashLinearAttention

The main advantage of Chimera-guided fusion over existing FlashLinearAttention is:

1. **Analytically optimal tile sizes**: Chimera's Lagrange solution provably minimizes DRAM traffic given hardware SRAM constraints. FlashLinearAttention uses manually-tuned or auto-tuned tile sizes that may not be optimal.

2. **3-stage fusion** (state correction + gating): Existing implementations typically separate the state correction into a second kernel. Chimera's framework naturally extends to 3+ GEMM chains.

3. **Cross-hardware portability**: Chimera's tile size optimization takes SRAM capacity as input, automatically adapting to A100 (128 KB shared) vs. H100 (256 KB shared) vs. future hardware.

### Hardware-Specific Considerations

**A100:**
- 192 KB shared memory per SM → use 128 KB for data tiles, 64 KB for control
- Tile sizes: $T_M = 64, T_L = 64, T_K = 64$ → tiles use $3 \times 64 \times 64 \times 2 = 24$ KB + accumulators
- mma.sync instructions for GEMM1 and GEMM2

**H100:**
- 256 KB shared memory per SM → larger tiles possible
- TMA async load for $K, V$ tiles while computing GEMM1 on $Q, K$ from previous iteration
- WGMMA instructions (asynchronous) → overlap GEMM2 with next iteration's GEMM1
- Tile sizes: $T_M = 128, T_L = 64, T_K = 128$ → tiles use $\sim 48$ KB + accumulators

**Register pressure:**
- $\tilde{S}$ accumulator: $T_M \times T_L$ FP32 values = $64 \times 64 = 4096$ = 16 KB in registers (128 registers per thread × 32 threads per warp)
- May need to split across warps or use register spilling for large tile sizes

## Risks & Limitations

1. **Triton compiler limitations**: Triton may not support the precise loop nest ordering that Chimera prescribes. Triton's auto-tuner operates at a higher level (launch grid, block sizes) but doesn't expose inner loop ordering directly. **Mitigation**: Use Triton's `tl.dot` within explicit Python loops for the $l$-dimension iteration, which forces the $mlk$ order.

2. **Register pressure for large tiles**: The $\tilde{S}$ partial scores matrix ($T_M \times T_L$ FP32) must fit in registers. For $T_M = T_L = 128$: 16384 FP32 values = 64 KB of registers — exceeds per-thread register budget. **Mitigation**: Use $T_L = 32$–$64$ and iterate over the $l$ dimension.

3. **Causal mask overhead**: The triangular causal decay mask $M_j$ cannot be efficiently precomputed as a dense matrix — it must be generated on-the-fly within the kernel. For $C = 64$, this is a simple comparison + multiplication; for $C = 256$, the mask computation may become non-trivial with learned decay rates.

4. **Backward pass complexity**: The backward pass requires computing gradients through the fused chain, which involves transposed GEMM chains and recomputation of intermediates. Chimera's forward-pass analysis must be extended to the backward pass. **Mitigation**: Start with fused forward-only; backward remains unfused. This still helps inference and reduces forward-pass training time.

5. **Marginal gain over FlashLinearAttention**: Existing fused implementations may already achieve most of the benefit. Chimera's analytical approach guarantees optimality but the gap may be small (5–10% vs. 30–50%). **Mitigation**: The MVE directly measures the gap; if < 10%, the idea is not worth pursuing further.

6. **Chunk-size dependence**: The fusion benefit scales with chunk size $C$. For very small $C$ (e.g., $C = 16$), the intermediates are tiny and L2 cache handles everything — no fusion needed. The practical benefit window is $C \in [64, 512]$.

## Follow-up Experiments

1. **Extend to backward pass**: Apply Chimera analysis to the backward GEMM chain (grad-Q, grad-K, grad-V computations), which has similar structure but transposed.
2. **Inter-chunk fusion**: Fuse the inter-chunk associative scan (boundary state propagation) with the intra-chunk kernel, creating a fully-fused layer-level kernel.
3. **Combine with Proposal 031 (VNM sparse)**: The fused kernel can use VNM-sparse projections for $Q, K, V$ computation before the GEMM chain, creating a doubly-optimized pipeline (sparse projections + fused chunk computation).
4. **Stream-K for irregular chunk shapes**: When sequences don't divide evenly by chunk size, the last chunk is smaller, causing load imbalance. Apply Stream-K's work-centric decomposition to balance work across chunks of varying sizes.
5. **H100 warp-specialized variant**: Implement the full warp-specialization pattern (producer/consumer warps, pingpong scheduling) within the fused chunk kernel, following FlashAttention-3's design.
6. **Auto-tune vs. analytical**: Compare Chimera's analytical tile sizes against Triton's auto-tuner and a random search baseline to quantify the value of the analytical model.

## Human Review

(To be filled by reviewer)

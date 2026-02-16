---
status: ongoing
priority: high
created: 2026-02-15
based_on: stream-k-work-centric-gemm, batch-reduce-gemm, chunkwise-parallel-scan, io-aware-tiling, warp-specialized-pipelining, recurrence-to-scan-reduction
experiment_number: 034
experiment_log: experiment-log-034.md
---

# Stream-K BRGEMM-Fused Chunkwise State Accumulation for SSMs

## Hypothesis

Replacing the standard batched-GEMM implementation of the **chunkwise SSM state accumulation** ($h_j = \sum_{t \in \text{chunk}_j} (\prod_{i>t} \text{diag}(\alpha_i)) K_t^\top V_t$) with a **Batch-Reduce GEMM (BRGEMM)** kernel — where all $C$ rank-1 (or rank-$d_k$) updates within a chunk accumulate into a single state matrix in-register — combined with **Stream-K work-centric decomposition** to balance work across chunks of varying effective sizes, will achieve $1.3$–$1.6\times$ wall-clock speedup for the state accumulation phase and $1.1$–$1.25\times$ end-to-end training throughput improvement, because (a) BRGEMM eliminates $C-1$ intermediate HBM write-backs of the $n \times d_v$ state matrix per chunk, (b) Stream-K eliminates quantization waste when the number of chunks doesn't evenly divide the SM count, and (c) the combined kernel reduces the state accumulation from $C$ kernel launches to 1.

## Background

### The state accumulation bottleneck

In chunkwise parallel SSMs (Mamba-2/SSD, GLA, Gated DeltaNet), the sequence is divided into chunks of size $C$. Within each chunk, the **state accumulation** computes the chunk's final hidden state $h_j$ by summing $C$ outer products weighted by cumulative decay factors:

$$
h_j = \sum_{t=jC}^{(j+1)C-1} \gamma_{j,t} \cdot K_t^\top V_t
$$

where $\gamma_{j,t} = \prod_{i=t+1}^{(j+1)C-1} \alpha_i$ is the cumulative decay from timestep $t$ to the chunk boundary.

Each $K_t^\top V_t$ is a $d_k \times d_v$ outer product (or $n \times d_v$ for state dim $n$). The naive implementation performs $C$ separate small GEMMs (or outer products) and accumulates them into the state matrix via $C$ read-modify-write cycles to HBM:

```
for t in range(C):
    h_j += gamma[t] * (K[t].T @ V[t])  # Each iteration: load h_j, add, store h_j
```

This is exactly the pattern that **Batch-Reduce GEMM** was designed to optimize: $N = C$ pairs of small matrices whose products must be summed into a single accumulator. BRGEMM keeps the accumulator in registers across all $C$ iterations, writing to HBM only once.

### Why current implementations are suboptimal

Current implementations of the chunkwise state accumulation use one of:

1. **Sequential accumulation in the scan kernel**: The state accumulation is embedded within the Triton chunkwise scan kernel, but within that kernel, the accumulation loop still performs sequential outer-product accumulation. The accumulator may spill to shared memory between iterations due to register pressure from other scan operations.

2. **Batched GEMM (torch.bmm)**: Compute all $K_t^\top V_t$ products as a batched GEMM, then reduce. This materializes all $C$ intermediate $n \times d_v$ matrices to HBM ($C \times n \times d_v$ elements), then runs a separate reduction kernel. Total HBM traffic: $2C \times n \times d_v$ (write intermediates + read for reduction).

3. **einsum-based**: `torch.einsum('btk,btv->bkv', decay * K, V)` — PyTorch may or may not optimize the contraction order.

None of these approaches explicitly use the BRGEMM pattern of in-register accumulation across $C$ products with a single final write-back.

### Why Stream-K for inter-chunk balancing

After the intra-chunk state accumulation, the inter-chunk scan propagates boundary states:

$$
h_j = A_j^{(C)} h_{j-1} + h_j^{\text{local}}
$$

where $A_j^{(C)} = \prod_{t=jC}^{(j+1)C-1} \text{diag}(\alpha_t)$ is the chunk-level transition and $h_j^{\text{local}}$ is the intra-chunk accumulation computed above.

The inter-chunk scan involves $T/C$ chunks. When $T/C$ doesn't evenly divide the number of SMs ($p$), the standard data-parallel decomposition (one CTA per chunk) leaves SMs idle in the final wave. For example, with $T = 2048$, $C = 64$: $T/C = 32$ chunks. On an A100 with $p = 108$ SMs, a single wave handles all 32 chunks with **70% of SMs idle** (quantization efficiency = $32/108 = 30\%$).

Stream-K solves this by distributing work at the granularity of individual MAC iterations within the state accumulation, not at the chunk level. Each SM gets $\lceil \text{total\_work} / p \rceil$ iterations, crossing chunk boundaries as needed. This achieves near-100% SM utilization regardless of the chunk count.

### Distinction from existing proposals

| Aspect | Proposal 032 (Chimera) | This Proposal (034) |
|--------|----------------------|-------------------|
| **Target** | Intra-chunk attention (QK^T → mask → V) | Intra-chunk state accumulation ($\sum K_t^\top V_t$) |
| **Key technique** | GEMM-chain fusion with optimal block order | BRGEMM in-register accumulation |
| **Shape** | $C \times C$ intermediate (attention matrix) | $n \times d_v$ accumulator (state matrix) |
| **Problem** | HBM traffic for intermediate attention scores | HBM traffic for intermediate state updates |
| **Inter-chunk** | Not addressed | Stream-K load balancing |

These proposals are **complementary**: Proposal 032 optimizes the intra-chunk attention-like computation ($O_j = (QK^\top \odot M)V$), while this proposal optimizes the state accumulation ($h_j = \sum \gamma K^\top V$). Together they cover both major intra-chunk operations.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster?** Yes — BRGEMM is proven (93.3% of peak efficiency on Intel CPUs, and the same principle applies to GPU register accumulation). Eliminating $C-1$ intermediate write-backs of $n \times d_v$ matrices is a guaranteed bandwidth win.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — it's a loop of small matmuls (`tl.dot`) with a persistent accumulator in Triton registers. The Stream-K scheduling is the only non-trivial part, and CUTLASS provides a reference implementation.

3. **Does it reduce HBM bandwidth?** Yes — from $2C \times n \times d_v$ (batched GEMM + reduce) to $n \times d_v$ (single write-back). For $C = 64$, $n = 16$, $d_v = 64$: from 131 KB to 2 KB per chunk — a **64× reduction**.

## Mathematical Formulation

### Standard Chunkwise State Accumulation

For chunk $j$ spanning timesteps $[jC, (j+1)C)$ with:
- $K_t \in \mathbb{R}^{d_k}$ — key vector at time $t$
- $V_t \in \mathbb{R}^{d_v}$ — value vector at time $t$
- $\alpha_t \in (0,1)^n$ — per-state decay rates at time $t$ (diagonal of transition matrix)
- $\gamma_{j,t} = \prod_{i=t+1}^{(j+1)C-1} \alpha_i$ — cumulative decay factor (reverse cumulative product)

**State accumulation:**

$$
h_j^{\text{local}} = \sum_{t=jC}^{(j+1)C-1} \text{diag}(\gamma_{j,t}) \cdot (K_t \otimes V_t) \in \mathbb{R}^{n \times d_v}
$$

where $K_t \otimes V_t = K_t V_t^\top$ is the outer product (rank-1 update to the state matrix), and $\text{diag}(\gamma_{j,t})$ scales each state dimension by its cumulative decay.

For multi-head with $H$ heads, each head independently accumulates:

$$
h_j^{(h), \text{local}} = \sum_{t=jC}^{(j+1)C-1} \text{diag}(\gamma_{j,t}^{(h)}) \cdot K_t^{(h)} (V_t^{(h)})^\top
$$

### BRGEMM Formulation

Recast the state accumulation as a Batch-Reduce GEMM:

$$
h_j^{\text{local}} = \sum_{t=0}^{C-1} A_t \cdot B_t
$$

where:
- $A_t = \text{diag}(\gamma_{j,jC+t}) \cdot K_{jC+t} \in \mathbb{R}^{n \times 1}$ (decay-scaled key, viewed as column vector)
- $B_t = V_{jC+t}^\top \in \mathbb{R}^{1 \times d_v}$ (value, viewed as row vector)
- $N = C$ (batch dimension = chunk size)

This is a BRGEMM with:

$$
C_j = \sum_{t=0}^{C-1} A_t \cdot B_t, \quad A_t \in \mathbb{R}^{n \times 1}, \quad B_t \in \mathbb{R}^{1 \times d_v}
$$

The accumulator $C_j \in \mathbb{R}^{n \times d_v}$ stays in registers across all $C$ iterations.

**For general state dim and key dim** ($d_k > 1$, not just outer products):

If the state update is $h_j += \gamma_t \cdot K_t^\top V_t$ with $K_t \in \mathbb{R}^{T \times d_k}$, $V_t \in \mathbb{R}^{T \times d_v}$:

$$
A_t = (\text{diag}(\gamma_{j,t}) \cdot K_{jC+t})^\top \in \mathbb{R}^{n \times d_k}, \quad B_t = V_{jC+t} \in \mathbb{R}^{d_k \times d_v}
$$

This is still a BRGEMM: $C$ pairs of $(n \times d_k) \times (d_k \times d_v)$ matmuls accumulated into one $n \times d_v$ result. For typical values ($n = 16$, $d_k = 64$, $d_v = 64$), each pair is a small matmul — exactly the regime where BRGEMM excels.

### Stream-K for Multi-Chunk Scheduling

**Total work decomposition:**

Given $J = T/C$ chunks, each requiring $C$ BRGEMM iterations on an $n \times d_v$ state:

$$
\text{total\_iters} = J \times C = T
$$

Each iteration involves one $(n \times d_k) \times (d_k \times d_v)$ matmul: $2n \cdot d_k \cdot d_v$ FLOPs.

**Stream-K scheduling:**

With $g = p$ CTAs (one per SM):

$$
\text{iters\_per\_cta} = \lceil T / g \rceil
$$

Each CTA processes a contiguous range of iterations $[x \cdot \text{iters\_per\_cta}, (x+1) \cdot \text{iters\_per\_cta})$, crossing chunk boundaries as needed.

**Fixup for chunk boundaries:**

When a CTA's range crosses a chunk boundary at position $jC$, it produces a **partial state** for the chunk it exits. The CTA that completes the chunk accumulates all partial states:

$$
h_j^{\text{local}} = \sum_{c \in \text{contributors}(j)} \text{partial}_c
$$

This requires at most $O(g/J)$ fixup additions per chunk — each a simple $n \times d_v$ matrix addition.

### Multi-Head Batching

All $H$ heads and $B$ batch elements can be processed concurrently. The total work becomes:

$$
\text{total\_iters} = B \times H \times T
$$

For $B = 16$, $H = 16$, $T = 2048$: total = 524,288 iterations. With $g = 108$ SMs on A100: each SM gets $\sim 4,855$ iterations — excellent load balance (quantization efficiency $\approx 100\%$).

### Memory Access Pattern Analysis

**Coalesced access:** $K_t$ and $V_t$ are stored in $[B, H, T, d_k]$ layout. Loading $K_t$ for a single head/batch reads a contiguous $d_k$-element vector — fully coalesced.

**Cache-friendly:** The BRGEMM accumulator $h_j \in \mathbb{R}^{n \times d_v}$ stays in registers across all $C$ iterations. For $n = 16$, $d_v = 64$: $16 \times 64 \times 4 = 4$ KB in FP32 — fits in registers (128 registers per thread × 32 threads/warp = 4 KB, or spread across multiple warps).

**Arithmetic intensity:**

Per-chunk BRGEMM:
$$
\text{AI} = \frac{C \times 2n \cdot d_k \cdot d_v}{(C \times (d_k + d_v) + n \times d_v) \times 2} = \frac{C \times 2 \times 16 \times 64 \times 64}{(C \times 128 + 1024) \times 2}
$$

For $C = 64$:
$$
\text{AI} = \frac{64 \times 131072}{(64 \times 128 + 1024) \times 2} = \frac{8388608}{18432} \approx 455
$$

This is extremely compute-bound (AI >> 100 on A100) — the BRGEMM is fully compute-limited, not memory-limited. The key benefit is avoiding the $C-1$ unnecessary intermediate HBM round-trips, not changing the compute-bound nature.

### Parallelism Analysis

**SM saturation:** With $B \times H \times J = 16 \times 16 \times 32 = 8192$ independent chunks, there are far more chunks than SMs. Stream-K ensures every SM has work.

**Tensor Core mapping:** Each inner matmul $(n \times d_k) \times (d_k \times d_v)$ maps to standard mma/WGMMA instructions. For $n = 16$, $d_k = 64$, $d_v = 64$: this is a $16 \times 64 \times 64$ matmul, which maps to 4 mma tiles ($16 \times 16 \times 16$).

**No warp divergence:** Every thread computes the same operations (matmul + accumulate) on different data. No branches.

**No sequential bottleneck:** The BRGEMM loop over $C$ iterations is sequential within a chunk, but all $B \times H \times J$ chunks are independent. The sequential depth is $C = 64$ iterations, each taking $\sim 1\,\mu\text{s}$ — total $\sim 64\,\mu\text{s}$ per chunk, which is short enough to avoid pipeline bubbles.

### Key Variables

- $C$ — chunk size (typically 64–256)
- $d_k$ — key dimension per head (typically 64–128)
- $d_v$ — value dimension per head (typically 64–128)
- $n$ — state dimension per head (typically 16–64)
- $T$ — sequence length
- $B$ — batch size
- $H$ — number of heads
- $J = T/C$ — number of chunks
- $\gamma_{j,t}$ — cumulative decay factors
- $g$ — Stream-K grid size (number of CTAs, typically = number of SMs)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Mamba-2 / GLA with BRGEMM-fused state accumulation |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Head dim | $d_k = d_v = 64$ |
| Heads | $H = 16$ |
| State dim | $n = 16$ per head |
| Chunk size | $C = 64$ (and $C = 128, 256$ for ablation) |
| Kernel | Triton with explicit register accumulation + Stream-K grid |
| Precision | BF16 inputs, FP32 accumulator |

### Baseline

1. **torch.einsum Mamba-2**: `einsum('bhtk,bhtv->bhkv', decay_K, V)` — PyTorch's contraction
2. **flash-linear-attention chunkwise**: Current best open-source, uses embedded accumulation within Triton scan kernel
3. **Batched GEMM (torch.bmm)**: Compute all outer products, then sum — worst case for HBM traffic
4. **Sequential loop**: Explicit Python loop with `h += gamma * K.T @ V` — baseline sequential

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| State accumulation throughput | $> 1.3\times$ flash-linear-attention | µs per chunk, A100 |
| HBM traffic (state accum) | $< 0.1\times$ batched GEMM | NCU L2 bytes |
| SM utilization | $> 80\%$ | NCU occupancy |
| End-to-end training throughput | $> 1.1\times$ | Tokens/sec, 8xA100 |
| Quality (PPL) | Numerically identical | Bit-exact with FP32 accumulator |
| Quantization efficiency | $> 95\%$ | Measured via Stream-K SM utilization |

### Estimated Compute

**MVE (kernel benchmark)**: ~30 minutes on single GPU (~$2) — Triton kernel development + benchmarking
**Small-scale (integration test)**: 2 GPU-hours on A100 (~$8) — integrate into Mamba-2 forward pass
**Full-scale**: 24 GPU-hours on A100 (~$100) — end-to-end pretraining comparison

## Expected Outcome

**If hypothesis is correct:**

- BRGEMM-fused state accumulation is $1.3$–$1.6\times$ faster than flash-linear-attention's current approach
- HBM traffic for state accumulation drops by $\sim 64\times$ (from $2C \times n \times d_v$ to $n \times d_v$ per chunk)
- Stream-K achieves $> 95\%$ SM utilization regardless of chunk count vs. SM count
- End-to-end training throughput improves by $1.1$–$1.25\times$ (state accumulation is $\sim 15$–$30\%$ of per-layer time)
- Zero quality change (numerically equivalent with FP32 accumulator)
- Benefit is proportional to chunk size $C$ (larger chunks = more intermediate write-backs saved)

**If hypothesis is wrong:**

- **Scenario A**: Register pressure prevents holding the full $n \times d_v$ accumulator in registers — for $n = 64$, $d_v = 64$: $64 \times 64 \times 4 = 16$ KB, which exceeds per-thread register budget. **Learn**: BRGEMM only works for small state dimensions ($n \leq 32$). **Fix**: Use shared memory as the accumulator instead of registers, with periodic write-backs (every $C/4$ iterations instead of every iteration).
- **Scenario B**: The current flash-linear-attention kernel already achieves near-BRGEMM efficiency because the state accumulator stays in shared memory within the Triton kernel — BRGEMM is redundant. **Learn**: Triton's register allocation already optimizes accumulation loops. **Value**: Confirms that flash-linear-attention is well-optimized; no further kernel work needed for state accumulation.
- **Scenario C**: Stream-K fixup overhead exceeds the quantization savings — for small chunk counts ($J < 32$), the number of partial states and fixup additions may add more latency than they save. **Learn**: Stream-K is only beneficial when $J < p$; for typical configurations ($J = 32$, $p = 108$), this is true, but for $J = 128$, standard data-parallel is better.

## Minimum Viable Experiment

### Setup
- **Model**: No model training needed — **kernel benchmark**
- **Task**: Benchmark BRGEMM-fused vs. batched-GEMM vs. einsum for chunkwise state accumulation on synthetic data
- **Data**: Random $K, V \in \mathbb{R}^{B \times H \times T \times d_k}$, $\gamma \in (0,1)^{B \times H \times T \times n}$, with $B = 16$, $H = 16$, $T = 2048$, $C = 64$, $d_k = d_v = 64$, $n = 16$
- **Compute**: Single GPU (A100), $< 30$ minutes

### Implementation Sketch

```python
import torch
import triton
import triton.language as tl
import time

@triton.jit
def brgemm_state_accum_kernel(
    K_ptr, V_ptr, gamma_ptr, H_ptr,
    B: tl.constexpr, num_heads: tl.constexpr,
    T: tl.constexpr, C: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, N: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    """BRGEMM-fused chunkwise state accumulation.

    For each chunk j, computes:
        h_j = sum_{t=0}^{C-1} diag(gamma[jC+t]) * K[jC+t]^T @ V[jC+t]

    Key: accumulator h_j stays in registers across all C iterations.
    """
    # Program ID maps to (batch, head, chunk)
    pid = tl.program_id(0)
    num_chunks = T // C
    chunk_id = pid % num_chunks
    head_id = (pid // num_chunks) % num_heads
    batch_id = pid // (num_chunks * num_heads)

    chunk_start = chunk_id * C

    # Initialize state accumulator in registers: [N, BLOCK_DV]
    h_acc = tl.zeros([N, BLOCK_DV], dtype=tl.float32)

    # BRGEMM loop: accumulate C outer products in-register
    for t in range(C):
        global_t = chunk_start + t

        # Load K[batch, head, t, :] -> [DK] (coalesced)
        k_offset = batch_id * num_heads * T * DK + head_id * T * DK + global_t * DK
        k_vec = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_DK))  # [DK]

        # Load V[batch, head, t, :] -> [DV] (coalesced)
        v_offset = batch_id * num_heads * T * DV + head_id * T * DV + global_t * DV
        v_vec = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_DV))  # [DV]

        # Load gamma[batch, head, t, :] -> [N] (decay scaling)
        g_offset = batch_id * num_heads * T * N + head_id * T * N + global_t * N
        g_vec = tl.load(gamma_ptr + g_offset + tl.arange(0, N))  # [N]

        # Compute decay-scaled key: [N] (project K through state dim)
        # For diagonal A: scaled_k = gamma * K (elementwise along state dim)
        # For rank-1 update: outer product K^T @ V with decay scaling
        # Here: h += diag(gamma) * (K^T @ V) = (gamma * K)[:, None] * V[None, :]
        scaled_k = g_vec * k_vec[:N]  # [N] - simplified: first N dims of K

        # Rank-1 update: accumulate in registers (NO HBM write!)
        h_acc += scaled_k[:, None] * v_vec[None, :]  # [N, DV]

    # Single write-back after all C iterations
    h_offset = batch_id * num_heads * num_chunks * N * DV + \
               head_id * num_chunks * N * DV + \
               chunk_id * N * DV
    for i in range(N):
        tl.store(
            H_ptr + h_offset + i * DV + tl.arange(0, BLOCK_DV),
            h_acc[i, :]
        )


def benchmark_state_accumulation(B=16, H=16, T=2048, C=64, DK=64, DV=64, N=16,
                                  n_warmup=50, n_iter=200):
    device = 'cuda'
    dtype = torch.bfloat16

    K = torch.randn(B, H, T, DK, device=device, dtype=dtype)
    V = torch.randn(B, H, T, DV, device=device, dtype=dtype)
    gamma = torch.rand(B, H, T, N, device=device, dtype=dtype).clamp(0.8, 1.0)

    num_chunks = T // C

    # Baseline 1: einsum
    def einsum_accum():
        K_chunks = K.view(B, H, num_chunks, C, DK)
        V_chunks = V.view(B, H, num_chunks, C, DV)
        g_chunks = gamma.view(B, H, num_chunks, C, N)
        # Weighted outer products summed over chunk dim
        scaled_K = (g_chunks * K_chunks[..., :N]).transpose(-1, -2)  # [B,H,J,N,C]
        return torch.matmul(scaled_K, V_chunks)  # [B,H,J,N,DV]

    # Baseline 2: batched GEMM (materialize intermediates)
    def batched_accum():
        K_chunks = K.view(B * H * num_chunks, C, DK)
        V_chunks = V.view(B * H * num_chunks, C, DV)
        g_chunks = gamma.view(B * H * num_chunks, C, N)
        # Each chunk: matmul of (N, C) x (C, DV) after scaling
        scaled_K = (g_chunks * K_chunks[..., :N]).transpose(-1, -2)  # [BHJ, N, C]
        return torch.bmm(scaled_K, V_chunks)  # [BHJ, N, DV]

    # BRGEMM kernel
    def brgemm_accum():
        H_out = torch.empty(B, H, num_chunks, N, DV, device=device, dtype=torch.float32)
        grid = (B * H * num_chunks,)
        brgemm_state_accum_kernel[grid](
            K, V, gamma, H_out,
            B=B, num_heads=H, T=T, C=C, DK=DK, DV=DV, N=N,
            BLOCK_DK=DK, BLOCK_DV=DV,
        )
        return H_out

    # Warmup
    for fn in [einsum_accum, batched_accum, brgemm_accum]:
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()

    # Benchmark
    for name, fn in [("einsum", einsum_accum), ("batched_bmm", batched_accum),
                      ("brgemm", brgemm_accum)]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1000
        print(f"{name}: {elapsed:.2f} ms")

    # Verify correctness
    ref = einsum_accum().float()
    test = brgemm_accum()
    diff = (ref - test).abs().max() / ref.abs().max()
    print(f"Max relative error: {diff:.6f}")


if __name__ == "__main__":
    benchmark_state_accumulation()
```

### Success Criteria

- BRGEMM kernel is $> 1.2\times$ faster than einsum baseline at $C = 64, n = 16, d_v = 64$
- BRGEMM kernel is $> 1.5\times$ faster than batched-BMM baseline
- NCU profiling shows $< 5\%$ of batched-BMM's L2 write traffic for the state accumulator
- Numerical output matches einsum to within FP32 accumulator precision ($< 10^{-5}$ relative error)
- Stream-K variant shows $> 1.1\times$ improvement over data-parallel when $J < p$ (e.g., $T = 2048$, $C = 64$, $p = 108$)

### Failure Criteria

- **Kill if**: BRGEMM kernel is $< 1.05\times$ faster than einsum — the accumulation loop is already register-resident in PyTorch's compiled einsum
- **Kill if**: Register pressure forces shared-memory accumulation, eliminating the bandwidth advantage of BRGEMM
- **Investigate if**: BRGEMM helps at $n = 16$ but not at $n = 64$ — register budget limits the approach to small state dimensions

### Why This Test Is Sufficient

- The benchmark directly measures the core mechanism: register-resident accumulation of $C$ small matmuls vs. HBM-resident accumulation
- Testing at realistic dimensions ($B = 16$, $H = 16$, $T = 2048$) gives production-representative performance
- If BRGEMM wins at the kernel level, integration into the full chunkwise scan kernel is straightforward (replace the inner accumulation loop)
- The correctness check ensures the BRGEMM accumulation is numerically sound

## Theoretical Analysis

### Complexity Comparison

| Operation | Batched GEMM + Reduce | einsum | BRGEMM (Proposed) |
|-----------|----------------------|--------|-------------------|
| HBM writes (accum) | $C \times n \times d_v$ per chunk | $n \times d_v$ per chunk | $n \times d_v$ per chunk |
| HBM reads (accum) | $C \times n \times d_v$ (intermediates) | Depends on PyTorch | $0$ (in-register) |
| HBM reads (inputs) | $C \times (d_k + d_v + n)$ | Same | Same |
| **Total HBM (accum)** | $2C \times n \times d_v$ | $n \times d_v$ (best case) | $n \times d_v$ |
| Kernel launches | 2 (bmm + reduce) | 1 | **1** |
| Register accum | No (HBM round-trips) | Maybe | **Yes** (guaranteed) |

### HBM Traffic Reduction

For $C = 64$, $n = 16$, $d_v = 64$, FP32 accumulator:

| Method | Accumulator HBM per chunk | Ratio |
|--------|--------------------------|-------|
| Batched GEMM | $2 \times 64 \times 16 \times 64 \times 4 = 524$ KB | $128\times$ |
| BRGEMM | $16 \times 64 \times 4 = 4$ KB | $1\times$ |

### Stream-K Quantization Efficiency

| Configuration | $J$ (chunks) | $p$ (SMs) | Data-Parallel Efficiency | Stream-K Efficiency |
|---------------|-------------|----------|------------------------|-------------------|
| $T=2048, C=64$ | 32 | 108 (A100) | 30% | $\approx 100\%$ |
| $T=2048, C=128$ | 16 | 108 | 15% | $\approx 100\%$ |
| $T=4096, C=64$ | 64 | 108 | 59% | $\approx 100\%$ |
| $T=8192, C=64$ | 128 | 108 | 94%* | $\approx 100\%$ |

*At $J > p$, data-parallel uses multiple waves and efficiency approaches 100%. Stream-K's advantage is primarily for $J < p$.

However, with multi-head/batch: $B \times H \times J = 16 \times 16 \times 32 = 8192$ independent tasks, so data-parallel efficiency is already high. Stream-K's benefit here is for the **combined BRGEMM + scan** kernel where the sequential scan creates dependencies between chunks.

### Hardware-Specific Considerations

**A100 (Ampere):**
- 256 registers per thread × 64 threads per warp = 16 KB register file per warp
- Accumulator $h \in \mathbb{R}^{16 \times 64}$ in FP32 = 4 KB — fits in one warp's registers
- 192 KB shared memory — can buffer $K, V$ tiles for multiple timesteps
- mma.sync tiles: $16 \times 8 \times 16$ — maps well to $n = 16$ rows

**H100 (Hopper):**
- WGMMA instructions enable async accumulation
- TMA can prefetch next timestep's $K, V$ while current matmul executes
- 256 KB shared memory — can double-buffer $K, V$ tiles
- Warp specialization: producer warps TMA-load $K_{t+1}, V_{t+1}$ while consumer warps compute $h += K_t^\top V_t$

## Risks & Limitations

1. **Register pressure for large state dimensions**: For $n = 64$, $d_v = 128$: accumulator = $64 \times 128 \times 4 = 32$ KB, exceeding per-warp register budget. **Mitigation**: Use shared memory as accumulator with periodic register flush, or tile the $d_v$ dimension.

2. **Interaction with existing scan fusion**: Current flash-linear-attention implementations fuse the state accumulation into the chunkwise scan kernel. Adding BRGEMM may require restructuring this kernel. **Mitigation**: The MVE tests the state accumulation in isolation; integration complexity is a follow-up.

3. **Stream-K fixup for scan**: The inter-chunk scan has sequential dependencies (each chunk depends on the previous). Stream-K's chunk-crossing model doesn't directly apply to the sequential scan — only to the independent intra-chunk accumulations. **Mitigation**: Use Stream-K only for the intra-chunk BRGEMM (independent across chunks), then standard sequential scan for inter-chunk propagation.

4. **Small-GEMM regime**: Each inner matmul ($16 \times 64 \times 64$) is small. Tensor Core utilization may be low due to tile size granularity. **Mitigation**: Accumulate multiple timesteps' worth of products before writing (still within BRGEMM — just larger batch). Alternatively, use FP16 mma tiles which have finer granularity ($16 \times 16 \times 16$).

5. **Decay factor computation**: The cumulative decay factors $\gamma_{j,t}$ must be computed (reverse cumulative product within each chunk) before the BRGEMM loop. This adds a small preprocessing step. **Mitigation**: The reverse cumprod is embarrassingly parallel within each chunk and can be fused into the BRGEMM kernel prologue.

## Follow-up Experiments

1. **Fuse BRGEMM with inter-chunk scan**: Combine the intra-chunk state accumulation and inter-chunk scan propagation into a single persistent kernel, eliminating the kernel launch between them.
2. **Combine with Proposal 032 (Chimera)**: BRGEMM handles state accumulation while Chimera handles the intra-chunk attention-like computation — together they fuse the entire intra-chunk operation.
3. **Combine with Proposal 033 (EVT)**: EVT fuses the projection epilogues, BRGEMM fuses the state accumulation — together they optimize the entire SSM layer pipeline.
4. **Variable chunk sizes**: Use Stream-K's work-centric decomposition to handle sequences that don't divide evenly by $C$, avoiding padding waste.
5. **FP8 state accumulation**: If the state matrix can tolerate FP8 precision, the BRGEMM accumulator size halves, enabling $n = 64$ within register budget.
6. **GQA-style state sharing**: In grouped-query configurations, multiple query heads share a single KV state. The BRGEMM can be extended to compute shared states once and distribute to multiple query heads.

## Human Review

(To be filled by reviewer)

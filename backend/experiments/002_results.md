# Experiment 002 Results: SSD-DeltaNet Block Decomposition

**Proposal**: proposals/002-ssd-deltanet-wy-hybrid.md
**Code**: code/002/
**Experiment Log**: See experiments/experiment-log.md (section: Experiment 002)
**Date**: 2026-02-15
**Cost**: ~$0.10 (T4 GPU, ~3 minutes on Modal)
**Runtime**: ~3 minutes (GPU benchmark), ~15 minutes total (including CPU debug runs)

## Setup

Implemented and benchmarked two PyTorch implementations of DeltaNet's forward pass:

1. **Naive WY**: Standard sequential delta rule accumulation, processing one token at a time. State update: `S_t = S_{t-1} + beta_t * k_t * (v_t - S_{t-1}^T k_t)^T`

2. **Block-SSD**: SSD-style sub-block decomposition:
   - **Inter-block** (matmul): `output_inter = Q_sb @ S_init.T` and value correction `V' = V - K @ S_init`
   - **Intra-block** (sequential): Delta rule on deviation `M_t = S_t - S_init`, evolving from M_0=0 with corrected values

Settings: T=512, d=64, C=64, Q=16, 200 benchmark iterations on T4 GPU.

## Results

### Primary Metrics (T=512, d=64, Q=16)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Speedup (block vs naive) | > 1.3x | 0.84x | :x: FAIL |
| Numerical error (L-inf) | < 1e-5 | 9.54e-06 | :white_check_mark: PASS |
| Matmul fraction of FLOPs | > 60% | 4.1% | :x: FAIL |

### Sub-block Size Sweep (T4 GPU)

| Q (sub-block) | Naive (ms) | Block (ms) | Speedup |
|----------------|-----------|-----------|---------|
| 4 | 85.3 | 115.2 | 0.74x |
| 8 | 85.3 | 105.0 | 0.81x |
| 16 | 85.3 | 100.1 | 0.85x |
| 32 | 85.3 | 97.8 | 0.87x |
| 64 | 85.3 | 95.6 | 0.89x |
| 128 | 85.3 | 95.0 | 0.90x |
| 256 | 85.3 | 94.2 | 0.91x |

Best sub-block Q=256 gives 0.91x -- still slower than naive.

### Sequence Length Scaling (d=64, Q=16)

Speedup is ~0.88-0.89x across all T (128-2048). No regime where block-SSD wins.

### State Dimension Scaling (T=512, Q=16)

Speedup is ~0.88-0.89x across all d (32-256). No regime where block-SSD wins.

## Success Criteria

- :x: **Speedup > 1.3x**: 0.84x -- Block-SSD is **16% SLOWER** than naive
- :white_check_mark: **Numerical error < 1e-5**: 9.54e-06 -- Algebraic decomposition is correct
- :x: **Matmul fraction > 60%**: 4.1% -- Sequential loop dominates all compute

## Root Cause Analysis

### Why is Block-SSD slower?

1. **Python-level CUDA kernel launch overhead**: The sequential intra-block loop launches 3746 individual CUDA kernels for T=512. Each `cudaLaunchKernel` has ~8us overhead, totaling ~30ms of pure dispatch overhead.

2. **Extra FLOPs**: The block decomposition adds 67% more total FLOPs vs naive:
   - Block: 2 x Q x d^2 (inter matmuls) + Q x 3 x d^2 (intra sequential) = 5Qd^2 per sub-block
   - Naive: Q x 3 x d^2 per Q tokens

3. **Tiny matmuls**: Inter-block matmuls are (Q x d) x (d x d) = 16 x 64 x 64 = 65K FLOPs each -- far too small for tensor core utilization. The 64 `aten::mm` calls total only 1.7ms.

4. **Sequential bottleneck unbroken**: The intra-block loop still processes Q=16 tokens sequentially. The same loop runs in both naive (T=512 steps) and block (T/Q=32 sub-blocks x Q=16 steps), but block has additional overhead per sub-block.

### Why did the UT Transform matmul approach fail?

The UT transform computes WY factors for the Householder-like product Phi = prod(I - beta_i k_i k_i^T), NOT for the cumulative DeltaNet state M_t. The DeltaNet delta rule has an additive term (beta_t k_t v_t^T) that breaks the Householder factorization. Verified: `Phi_fwd = I - K^T @ W_ut` (error ~6e-8), but `M_T != W^T @ U` (error ~0.67).

## What We Learned

1. **The algebraic decomposition is correct**: The state deviation M_t = S_t - S_init evolves via the delta rule with corrected values v'_t = v_t - S_init^T k_t, yielding output o_t = S_init @ q_t + M_t @ q_t. Numerical error < 1e-5 across all settings.

2. **PyTorch-level block decomposition cannot work**: The overhead of Python loops, CUDA kernel launches, and small-matrix dispatch exceeds any tensor core benefit. This is a fundamental limitation of the PyTorch implementation strategy.

3. **The proposal's speedup claim requires fused kernels**: A Triton/CUDA kernel (~1000 lines) that processes entire sub-blocks in shared memory -- eliminating per-element kernel launches -- is necessary for the speedup to materialize.

4. **The UT transform has a specific role**: It gives WY factors for the cumulative A-product (Phi), not the cumulative state. Inside a fused kernel, this could efficiently compute final states at sub-block boundaries, but cannot replace the intra-block sequential computation.

## Decision

**ABANDON** (at PyTorch level)

The proposal's core hypothesis -- that SSD-style block decomposition accelerates DeltaNet -- is **not testable** with pure PyTorch implementations. The algebraic restructuring is mathematically correct but the implementation overhead (kernel launches, Python loops, small matmul dispatch) overwhelms any potential speedup.

The MVE failure criterion is triggered: "Block-SSD is slower than naive -> kill the idea."

However, it's important to note that the failure is at the **implementation level**, not the **mathematical level**. The decomposition is exact (error < 1e-5), and the matmul-heavy structure would benefit from tensor cores IF the operations were fused into a single kernel launch per sub-block.

## Next Steps

1. **Do NOT proceed** with further PyTorch-level optimization of SSD-DeltaNet
2. **If kernel engineering is feasible** (~1 week, ~1000 lines of Triton), a fused kernel implementing the block decomposition could potentially achieve the proposed 1.5-3x speedup by:
   - Processing entire sub-blocks in shared memory (eliminating kernel launch overhead)
   - Using tensor cores for the inter-block matmuls at larger batch sizes
   - Fusing value correction + intra-block loop into a single kernel
3. **The UT transform** (verified correct for Phi product) could be useful inside such a kernel for efficient state propagation at sub-block boundaries
4. **Alternative**: Look into existing optimized DeltaNet implementations (e.g., flash-linear-attention) that may already solve this problem with custom kernels

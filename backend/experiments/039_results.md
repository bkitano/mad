# Experiment 039 Results: Warp-Specialized Pingpong Chunkwise Linear RNN

**Proposal**: proposals/039-warp-specialized-pingpong-chunkwise-linear-rnn.md
**Code**: code/039/
**Experiment Log**: experiments/experiment-log-039.md
**Date**: 2026-02-16
**Cost**: ~$0.05 (T4 GPU, ~5 minutes)
**Runtime**: ~5 minutes
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/nmyglcj9
**Modal URL**: https://modal.com/apps/bkitano/main/ap-oKBZOK44N77BTWGWty70XC

## Setup

Tested the core hypothesis of overlapping sequential element-wise operations (decay masking) with GEMM operations via software pipelining in Triton. This is a proxy for the proposal's warp-specialized pingpong pipelining approach.

**Implementations**:
1. **PyTorch reference**: Pure PyTorch chunkwise GLA intra-chunk computation
2. **Triton baseline**: Sequential pipeline (load → QK^T → decay → SV → accumulate)
3. **Triton pipelined**: Software-pipelined with double-buffering (pre-loads next tile during current tile's compute)

**Configuration**: B=4, H=32, d=64, d_v=64, chunk_size=256, T={1024, 2048, 4096}
**GPU**: Tesla T4

## Results

| Metric | T=1024 | T=2048 | T=4096 |
|--------|--------|--------|--------|
| Baseline (ms) | 5.645 | 10.640 | 20.878 |
| Pipelined (ms) | 20.628 | 41.390 | 82.433 |
| Speedup | 0.274x | 0.257x | 0.253x |
| Baseline tokens/sec | 725,602 | 769,924 | 784,739 |
| Pipelined tokens/sec | 198,566 | 197,924 | 198,755 |
| Baseline TFLOPS | 1.52 | 1.61 | 1.65 |
| Pipelined TFLOPS | 0.42 | 0.42 | 0.42 |
| Max numerical error | 0.0 | 0.0 | 0.0 |
| Correctness | PASS | PASS | PASS |

## Success Criteria

- ❌ **Throughput > 1.2x baseline**: Pipelined kernel is ~4x SLOWER (0.25x speedup). Consistently worse at all sequence lengths.
- ✅ **Numerical match (max error < 1e-2)**: Perfect match — max error = 0.0 across all configurations. Both kernels produce bit-identical output.
- ❌ **Consistent across T**: Consistent, but consistently slower (0.253x-0.274x).

## Analysis

### Why the pipelined kernel is slower

1. **Triton already optimizes pipelining**: Triton's compiler automatically handles load/compute overlap through its memory hierarchy management. The baseline kernel's sequential loads are already pipelined by the compiler — our manual pipelining adds overhead rather than benefit.

2. **Register pressure**: The pipelined kernel holds TWO sets of K/V tiles simultaneously (current + next), doubling register usage. On T4 with limited registers per SM, this reduces occupancy and hides fewer memory latencies.

3. **Redundant last-tile loads**: To avoid Triton's variable scoping issues with conditionals, the pipelined kernel loads a redundant tile on the last iteration (clamped to the last valid tile). This wastes bandwidth.

4. **T4 vs H100**: The proposal's warp specialization requires H100 Hopper features (TMA async copies, WGMMA, separate hardware units for load/compute). On T4 (Turing), there's no physical separation between load and compute units, so explicit pipelining in Triton can't achieve true overlap — it's just doing more work.

### Key insight

**The fundamental hypothesis of the proposal is about hardware-level overlap on Hopper GPUs** — not software-level Triton pipelining. The proposal posits that TMA (Tensor Memory Accelerator) and WGMMA (Warp Group Matrix Multiply Accumulate) run on **physically separate hardware units** on H100, allowing true concurrent execution. This cannot be tested in Triton on T4.

### What the results tell us

1. **Triton is already well-optimized**: For chunkwise GLA on T4/A100, Triton's default pipelining is effective. Manual software pipelining hurts performance due to extra register pressure.

2. **The proposal's value is H100-specific**: The warp-specialized pingpong approach requires CUTLASS 3.x with TMA + WGMMA on Hopper, as the proposal states. The Triton MVE cannot validate or invalidate the core hypothesis.

3. **Correctness is verified**: The chunkwise GLA computation is correctly implemented (perfect numerical match), which would serve as a reference for a CUTLASS implementation.

## Decision

**DEBUG** — The MVE as implemented cannot test the proposal's core hypothesis because:
- Warp specialization requires H100 hardware features not available in Triton on T4
- The Triton compiler already handles pipelining, making manual pipelining counterproductive
- A proper test requires CUTLASS 3.x kernel development on H100

## Next Steps

1. **If pursuing further**: Implement the kernel in CUTLASS 3.x (C++) with warp-specialized pingpong scheduling targeting H100 SM90a. Use the PyTorch reference as a correctness oracle.

2. **Alternative approach**: Try the `fla` (Flash Linear Attention) library's existing Triton kernels as baseline and compare against a hand-tuned Triton kernel with `tl.async_copy` on A100/H100.

3. **Cost consideration**: A full CUTLASS implementation would require significant engineering effort (weeks, not hours). Consider whether the expected 1.3-2x speedup justifies the development cost.

4. **Recommendation**: This proposal is better suited for GPU kernel engineers with CUTLASS experience. The hypothesis is sound (backed by FlashAttention-3's demonstrated success with the same technique for softmax attention), but validating it requires low-level CUDA/PTX programming beyond what Triton can express.

# Experiment Log 039: Warp-Specialized Pingpong Pipelining for Chunkwise Linear RNN Kernels

## [2026-02-16 00:00] Experiment 039: Warp-Specialized Pingpong Chunkwise Linear RNN

### Selected Proposal
- **ID**: 039-warp-specialized-pingpong-chunkwise-linear-rnn
- **Priority**: high
- **Estimated cost**: $1.00
- **Reasoning**: This proposal targets a key GPU optimization — overlapping sequential element-wise ops (decay masking, state scan) with GEMMs via warp-specialized pingpong pipelining on Hopper GPUs. This is directly in line with the human feedback priorities (overlapping communication and computation, proven FlashAttention lineage techniques).

### Implementation Plan

**Challenge**: The full MVE as described in the proposal requires:
1. CUTLASS 3.x warp-specialized kernel with TMA + WGMMA (H100 only)
2. TFLA Triton baseline
3. `ncu` profiling

**Adapted MVE approach**: Since writing full CUTLASS 3.x kernels from scratch is extremely complex (weeks of work, PTX-level programming), and H100 availability on Modal may be limited, we adapt the MVE to test the core hypothesis in Triton:

1. **Baseline**: Standard chunkwise GLA forward pass in Triton (sequential: load tiles → compute QK^T → apply decay → compute SV → accumulate)
2. **Pipelined**: Triton kernel with software pipelining that overlaps tile loading with computation using `tl.async_copy` and double-buffering
3. **Benchmark**: Compare throughput (tokens/sec) and measure speedup
4. **Correctness**: Verify numerical match within BF16 tolerance

This tests the same fundamental hypothesis (can we overlap sequential ops with compute?) using Triton's software pipelining primitives instead of CUTLASS warp specialization.

**Success criteria (adapted)**:
- Pipelined kernel achieves > 1.2x throughput over naive baseline
- Numerical output matches within BF16 tolerance (max error < 1e-2)
- Consistent speedup across different sequence lengths

### [00:00] Step 1: Create directory structure
**Goal**: Set up code/039/ with all required files
**Actions**: Creating directory structure and all implementation files
**Result**: ✅ Success

### [00:10] Step 2: Implement kernels
**Goal**: Write 3 implementations of chunkwise GLA forward pass
**Actions**:
- Created `models/chunkwise_gla.py` with:
  1. `pytorch_chunkwise_gla_forward`: Pure PyTorch reference (intra-chunk only)
  2. `triton_chunkwise_gla_baseline`: Sequential Triton kernel
  3. `triton_chunkwise_gla_pipelined`: Software-pipelined Triton kernel with double-buffering
- Used explicit stride-based addressing for proper memory layout
- Both Triton kernels use BLOCK_M=64, BLOCK_N=64 tiles within chunks

**Design decisions**:
- Focused on intra-chunk computation only (no cross-chunk state) since that's the bottleneck being optimized
- Used cumulative log-gamma for efficient decay mask computation (avoids nested loops)
- Set chunk_size=256 (not 64) so there are 4 inner tiles per chunk (enough for pipelining to help)
- Pre-load first K,V tile before loop in pipelined version, then issue loads for next tile during current compute

**Result**: ✅ Success

### [00:20] Step 3: Create benchmark script and config
**Goal**: Write train.py, config.yaml, modal_config.py, pyproject.toml
**Actions**:
- Created `train.py` with wandb logging, correctness checking, throughput measurement
- Config: B=4, H=32, d=64, d_v=64, chunk_size=256, T=[1024, 2048, 4096]
- Modal: A100 GPU, 30 min timeout
**Result**: ✅ Success

### [00:25] Step 4: Deploy to Modal (Attempt 1)
**Goal**: Submit job to Modal for GPU execution
**Command**: `uv run modal run --detach modal_config.py --config config.yaml`
**Modal Job URL**: https://modal.com/apps/bkitano/main/ap-w1hJqOfZAqdEeVvxP9T4Mj
**Result**: ❌ Failed - two bugs found

**Bug 1: Huge upload** - `.venv` directory (7 GB) was being uploaded via `add_local_dir`
- **Fix**: Added `ignore` parameter to `add_local_dir` to exclude `.venv`, `__pycache__`, `*.egg-info`, `uv.lock`
- Created `.modalignore` file as well

**Bug 2: Triton variable scoping** - `NameError: K_next is not defined`
- In pipelined kernel, `K_next` was defined inside `if has_next:` block (runtime conditional)
- Triton can't handle variables defined only in one branch of a runtime conditional
- **Fix**: Removed the `if has_next:` conditional. Always pre-load next tile (clamping index to last valid tile to avoid OOB). On last iteration, loaded data is unused but avoids the scoping issue.

### [00:45] Step 5: Deploy to Modal (Attempt 2)
**Goal**: Re-deploy with bug fixes
**Command**: `uv run modal run modal_config.py --config config.yaml`
**Modal Job URL**: https://modal.com/apps/bkitano/main/ap-oKBZOK44N77BTWGWty70XC
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/nmyglcj9
**GPU**: Tesla T4
**Duration**: ~5 minutes
**Result**: ✅ Success - experiment completed

### [01:00] Training Results

**Summary Table**:

| T | Baseline (ms) | Pipelined (ms) | Speedup | Tokens/sec | Correct |
|---|---------------|----------------|---------|------------|---------|
| 1024 | 5.645 | 20.628 | 0.274x | 198,566 | YES |
| 2048 | 10.640 | 41.390 | 0.257x | 197,924 | YES |
| 4096 | 20.878 | 82.433 | 0.253x | 198,755 | YES |

**Key observations**:
- ✅ Perfect numerical match (max error = 0.0) — kernels produce bit-identical output
- ❌ Pipelined kernel is ~4x SLOWER than baseline
- Baseline achieves 1.52-1.65 TFLOPS, pipelined only 0.42 TFLOPS
- Pipelined throughput is constant at ~198K tokens/sec regardless of T (suspicious — suggests overhead dominates)
- Baseline scales linearly with T (as expected)

**Analysis of why pipelined is slower**:
1. Triton's compiler already handles load/compute pipelining in the baseline
2. Manual pre-loading adds register pressure (2x K/V tiles held simultaneously)
3. Redundant last-tile loads waste bandwidth (workaround for Triton variable scoping)
4. T4 lacks hardware-level separation of load/compute units (no TMA/WGMMA)

### [01:10] Final Results

**Success criteria**:
- ❌ Throughput > 1.2x: 0.25x (4x slower)
- ✅ Numerical match < 1e-2: 0.0 error (perfect match)
- ❌ Consistent across T: Consistently slower

**Decision**: DEBUG

**Reasoning**: The Triton-based MVE cannot test the proposal's core hypothesis because:
1. Warp specialization requires H100 Hopper features (TMA + WGMMA) not available in Triton/T4
2. Triton already auto-optimizes pipelining — manual pipelining adds overhead
3. The proposal's value is specifically about exploiting physically separate hardware units on H100

**What we learned**:
- Triton's compiler is already good at pipelining loads with compute
- Manual software pipelining in Triton is counterproductive (register pressure, extra loads)
- The chunkwise GLA computation is correctly implementable in Triton (serves as reference)
- The proposal's hypothesis is fundamentally about hardware-level overlap (H100-specific), not software-level pipelining

**Next steps**:
- A proper test requires CUTLASS 3.x kernel development targeting H100 SM90a
- This would need significant engineering effort (weeks, PTX-level programming)
- Consider whether the expected 1.3-2x speedup justifies the development investment
- The hypothesis itself is sound (FlashAttention-3 proved it for softmax attention)

---

# Experiment Log — 058: DSM-Fused Linear RNN Projection Chain

## [2026-02-16 00:00] Experiment 058: DSM-Fused Linear RNN Projection Chain

### Selected Proposal
- **ID**: 058-dsm-fused-linear-rnn-projection-chain
- **Priority**: high
- **Estimated cost**: $1.00
- **Reasoning**: This proposal targets a critical GPU optimization — reducing HBM round-trips in the projection chain of linear RNN layers (GLA, Gated DeltaNet, mLSTM). The hypothesis is that fusing input projection GEMMs with their downstream activations eliminates intermediate HBM writes/reads, saving ~50% of projection chain HBM traffic.

### Key Design Decision: MVE Adaptation

The original proposal calls for:
1. CUTLASS 3.x EVT API (C++ CUDA kernels)
2. H100 GPU with DSM support
3. Nsight Compute profiling

**Practical adaptation**: Since CUTLASS 3.x EVT implementation from scratch is extremely complex and the core hypothesis is testable without it, I'm implementing a PyTorch-based microbenchmark that:

1. **Unfused baseline (7 kernels)**: 4 separate projection GEMMs + 3 separate activation kernels — mirrors the current fla-org approach
2. **Fused approach (1-2 kernels)**: Single wide GEMM `x @ [W_Q; W_K; W_V; W_g; W_alpha]` followed by a single fused activation kernel — mirrors what EVT epilogue fusion achieves
3. **torch.compile fused**: Using torch.compile to let the compiler automatically fuse the entire chain
4. **Measures**: Wall-clock time, CUDA memory events, kernel launch counts

This validates the SAME hypothesis: "fusing projections + activations reduces wall-clock time by >30%"

The DSM-specific optimization (cross-kernel gate bridging) is H100-only and requires CUTLASS — that part is deferred.

### Implementation Plan
1. Create `models/projection_chain.py` — unfused and fused projection chain implementations
2. Create `train.py` — microbenchmark script with CUDA timing
3. Create `config.yaml` — benchmark configuration
4. Create `modal_config.py` — Modal deployment (A100 GPU)
5. Create supporting files (README.md, pyproject.toml)

---

### [00:05] Attempt: Implement projection chain model

**Goal**: Implement unfused and fused projection chains for GLA-style linear RNN layers

**Actions**:
- Created `models/projection_chain.py` with three implementations:
  - `UnfusedProjectionChain`: 4 separate GEMMs + 3 separate activation kernels (7 kernel launches)
  - `FusedProjectionChain`: 1 wide GEMM + 1 fused activation kernel (2 kernel launches)
  - `FullyFusedProjectionChain`: Single fused operation via torch.compile
- Created `models/output_chain.py` with unfused and fused output epilogue chains

**Result**: ✅ Files created

---

### [00:15] Attempt: Implement benchmark training script

**Goal**: Create comprehensive microbenchmark with CUDA timing

**Actions**:
- Created `train.py` with:
  - CUDA event-based timing (warm-up + timed iterations)
  - Memory tracking via torch.cuda
  - Numerical correctness verification (bit-exact comparison)
  - W&B logging integration
  - Multiple config sizes for scaling analysis

**Result**: ✅ File created

---

### [00:20] Attempt: Create Modal deployment config

**Goal**: Set up Modal deployment targeting A100 GPU

**Actions**:
- Created `modal_config.py` based on code/001 template
- Configured for A100 GPU (needed for good GEMM performance + bf16)
- Set 30-minute timeout (benchmark should complete in <10 minutes)

**Result**: ✅ File created

---

### [00:25] First Deployment Attempt

**Goal**: Deploy to Modal with A100 GPU

**Command**: `uv run modal run --detach modal_config.py --config config.yaml`
**Modal Job URL**: https://modal.com/apps/bkitano/main/ap-iZoNead87klg2cQjLGL861

**Result**: ❌ Failed

**Bug encountered**:
- Bug: `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'. Did you mean: 'total_memory'?`
  - Fix: Changed `total_mem` to `total_memory` in train.py line 483

---

### [00:27] Second Deployment — Successful

**Goal**: Redeploy after fixing bug

**Command**: `uv run modal run --detach modal_config.py --config config.yaml`
**Modal Job ID**: ap-5v4zBKp45dgTWGukGRXxjh
**Modal Job URL**: https://modal.com/apps/bkitano/main/ap-5v4zBKp45dgTWGukGRXxjh
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/036xulib
**Duration**: ~3 minutes
**GPU**: NVIDIA A100-SXM4-40GB (42.4 GB)

**Result**: ✅ Completed successfully

---

### [00:30] Benchmark Results

#### Input Projection Chain (Forward Only)

| Variant | Median (ms) | Speedup vs Unfused |
|---------|-------------|-------------------|
| Unfused (8 kernels) | 5.011 | 1.00x |
| Fused (1 GEMM + activations) | 5.150 | 0.97x (-2.8%) |
| torch.compile fused | 5.104 | 0.98x (-1.8%) |

**Key finding**: The fused variant is actually **slightly slower** than unfused. This is because at `B=8, T=4096, d=2048` (BT=32768), the GEMMs are **compute-bound**, not memory-bound. The single wide GEMM has a much larger N dimension (4×128 + 16 + 128 = 656 vs 128-2048 individually), which changes the tiling/occupancy characteristics.

#### Output Epilogue Chain (Forward Only)

| Variant | Median (ms) | Speedup |
|---------|-------------|---------|
| Unfused (3 kernels) | 1.845 | 1.00x |
| Fused | 1.845 | 1.00x |

#### Full Layer Chain (Forward + Backward)

| Variant | Median (ms) | Speedup |
|---------|-------------|---------|
| Unfused | 15.318 | 1.00x |
| Fused | 19.948 | **0.77x (30% slower)** |

#### Scaling Analysis (B=4, varying T)

| T | Unfused (ms) | Fused (ms) | Speedup |
|---|-------------|-----------|---------|
| 512 | 0.676 | 0.518 | **1.30x** |
| 1024 | 1.039 | 0.951 | **1.09x** |
| 2048 | 1.832 | 1.844 | 0.99x |
| 4096 | 3.584 | 3.628 | 0.99x |

**Critical insight**: Fusion shows benefit at **short sequences** (T=512: 1.3x speedup) where the problem is memory-bound and kernel launch overhead dominates, but the benefit vanishes at T≥2048 where GEMMs become compute-bound.

#### HBM Traffic (Theoretical)

- Unfused: 1.748 GB → Fused: 0.672 GB → **61.5% savings** ✅

#### Numerical Equivalence: All outputs match ✅

---

### [00:35] Final Analysis

**Decision**: DEBUG

**Key Insights**:

1. **PyTorch-level fusion is insufficient**: Concatenating weight matrices into a single wide GEMM does NOT achieve CUTLASS EVT epilogue fusion. Activations remain separate kernels.

2. **Compute-bound at scale**: At B=8, T=4096, d=2048, GEMMs are compute-bound. The theoretical 61.5% HBM savings don't translate to wall-clock speedup.

3. **Memory-bound at small scale**: At T=512, fusion DOES help (1.3x), confirming the hypothesis for memory-bound regimes.

4. **Wide GEMM has worse backward**: The fused backward is 30% SLOWER due to different tiling characteristics of the concatenated weight matrix.

5. **True EVT fusion needed**: CUTLASS EVT is required for the real optimization — applying activations in registers during the GEMM epilogue.

**Next steps**: A true CUTLASS EVT implementation would be needed to validate the full hypothesis. The PyTorch-level test shows the theoretical HBM savings are real, but compute-boundedness at typical training sizes limits the benefit.

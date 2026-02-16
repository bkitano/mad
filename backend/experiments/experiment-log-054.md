# Experiment Log — 054: SageAttention2-Style INT4 Smoothing for Chunkwise Linear RNN

## [2026-02-16 00:00] Experiment 054: INT4 Smoothing for GLA

### Selected Proposal
- **ID**: 054-sage-int4-smoothing-chunkwise-linear-rnn
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Tests whether INT4 quantization with Q+K smoothing can accelerate GLA chunkwise intra-chunk computation while maintaining accuracy. High priority because it targets wall-clock GPU speedup for pretraining — directly aligned with human feedback priorities.

### Key Design Decisions

**GPU Requirement**: The proposal specifies Ada (RTX4090) or Hopper (H100) for INT4 `mma.m16n8k64`. However, Modal's T4 GPUs (Turing) do NOT support INT4 tensor cores. We need at minimum an A100 (which supports INT8 but not INT4) or ideally H100.

**Practical MVE Strategy**: Since INT4 tensor cores require specific hardware and Triton's INT4 `tl.dot` support is still evolving (per the proposal's own risk #5), our MVE will:

1. **Simulate INT4 quantization in PyTorch** — quantize to INT4 range ([-7, 7]) and measure cosine similarity to validate the smoothing technique works for GLA attention patterns
2. **Benchmark FP8 matmul** as a proxy for mixed-precision speedup (FP8 is well-supported on H100 via Triton)
3. **Train a small GLA on copying task** with simulated INT4+FP8 quantization to verify training quality holds
4. If cosine similarity and quality check out, the kernel-level throughput gain is a hardware property (INT4 = 4x throughput) that doesn't need to be re-proven in an MVE

This approach validates the core hypothesis (smoothing enables accurate INT4 quantization of GLA attention) while avoiding the high-risk Triton INT4 kernel development.

### Implementation Plan
1. Implement GLA chunkwise forward pass (BF16 baseline) in PyTorch
2. Implement INT4 quantization with Q+K smoothing (simulated)
3. Implement FP8 quantization for SV matmul (simulated)
4. Write microbenchmark comparing cosine similarity: BF16 vs INT4 (no smooth) vs INT4 (smooth)
5. Write copying task training script
6. Deploy on Modal with A100 GPU (FP8 support)
7. Report results

### [00:05] Creating directory structure and initial files
**Goal**: Set up code/054/ with all necessary files
**Actions**: Creating models/, train.py, config.yaml, etc.


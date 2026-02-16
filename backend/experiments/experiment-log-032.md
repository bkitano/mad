# Experiment Log 032: Chimera-Fused Chunkwise SSM Scan

## [2026-02-16 00:00] Experiment 032: Chimera-Fused Chunkwise SSM Kernel Benchmark

### Selected Proposal
- **ID**: 032-chimera-fused-chunkwise-ssm-scan
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: This is a kernel-level optimization benchmark. The proposal aims to fuse the intra-chunk GEMM chain (QK^T -> decay mask -> attn*V) of chunkwise parallel SSMs into a single Triton kernel using Chimera-style analytical block ordering. This eliminates HBM traffic for the intermediate S matrix. This directly aligns with the human feedback prioritizing kernel fusion and HBM bandwidth reduction.

### Implementation Plan
1. Implement the fused Triton kernel (`chimera_fused_chunk_kernel`) that keeps S in registers
2. Implement the unfused baseline using PyTorch matmul operations
3. Write benchmark script that tests multiple chunk sizes (C=32, 64, 128, 256)
4. Add correctness verification (numerical comparison fused vs unfused)
5. Log all results to wandb
6. Deploy on Modal with T4 GPU

### Key Design Decisions
- **This is a kernel benchmark, not model training** - no model, optimizer, or dataset needed
- **Synthetic random data** - Q, K, V are random tensors
- **Success criteria from proposal**:
  - Fused kernel > 1.2x faster than unfused at C=64, d=64
  - Fused kernel > 1.5x faster than unfused at C=256, d=128
  - Numerical output matches unfused to within FP16 precision (< 1e-3 relative error)
  - Kernel compiles and runs correctly for all C in {32, 64, 128, 256}
- **Failure criteria**:
  - Kill if fused < 1.05x faster at any chunk size
  - Kill if Triton auto-tuner finds equivalent schedule without Chimera guidance

---

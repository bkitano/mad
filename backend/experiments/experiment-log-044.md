# Experiment Log 044: MatMulScan Inter-Chunk State Scan for Linear RNNs

## [2026-02-16 00:00] Experiment 044: MatMulScan Tensor-Core Inter-Chunk State Propagation

### Selected Proposal
- **ID**: 044-matmulscan-inter-chunk-state-scan-linear-rnn
- **Priority**: high
- **Estimated cost**: $0.17
- **Reasoning**: This proposal targets a key bottleneck in chunkwise linear RNNs - the inter-chunk state propagation. By reformulating the prefix scan as batched matrix multiplications against constant matrices, it can leverage tensor cores for speedup. This is a pure microbenchmark experiment, so it's fast and cheap to validate.

### Implementation Plan
1. Implement sequential scan reference (Python/PyTorch) - ground truth
2. Implement Blelloch parallel prefix scan (Triton kernel) - baseline
3. Implement MatMulScan with radix s=4 and s=8 (Triton kernel) - proposed method
4. Create benchmarking harness that:
   - Tests diagonal case: alpha in R^16, b in R^(16x64), G in {64, 128, 256}
   - Tests dense case: A in R^(16x16), b in R^(16x64), G=64
   - Measures throughput (us per scan)
   - Checks numerical accuracy vs sequential reference
5. Deploy to Modal on A100 (need tensor cores)
6. Log results to wandb

### Success Criteria (from proposal)
- MatMulScan (s=4) achieves >= 1.2x throughput over Triton Blelloch scan for G=128, diagonal case
- MatMulScan achieves >= 1.3x for G=256
- Numerical accuracy: max absolute error < 1e-3 (bf16) or < 1e-5 (fp32) vs sequential reference
- The Triton kernel compiles and runs without errors for all tested configurations

### Failure Criteria
- If MatMulScan is >= 0.9x slower than Blelloch for all G: approach unviable
- If numerical errors > 1e-1: log-exp conversion too unstable

---

## [2026-02-16 00:05] Starting Implementation

### Design Decisions

**Approach**: This is a microbenchmark, not a training experiment. We need to:
1. Generate random scan elements (transitions and states)
2. Implement three scan variants and compare throughput + accuracy

**Key insight about MatMulScan**: The core idea is that a prefix sum can be computed by reshaping into groups of s elements and multiplying by the lower-triangular all-ones matrix L_s. For a radix-4 scan:
- Level 1: Reshape [G] -> [G/4, 4], multiply each group by L_4 (local prefix sums)
- Level 2: Take every 4th element (the group sums), reshape [G/4] -> [G/16, 4], multiply by L_4
- Downsweep: Broadcast partial sums back down

**Diagonal case simplification**: For diagonal SSMs, the multiplicative scan (product of diag matrices) decomposes into n independent scalar scans. The additive weighted scan also decomposes element-wise. So we have P = n * d_v independent scalar scans that can be batched into a single GEMM.

**Implementation choice**: Using Triton for all GPU kernels. The MatMulScan will use tl.dot for the batched matrix multiplication against L_s.

---

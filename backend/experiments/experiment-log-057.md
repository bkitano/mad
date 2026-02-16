# Experiment Log 057: FlashRNN-Style Fused Inter-Chunk State Recurrence

## [2026-02-16 00:00] Experiment 057: FlashRNN-Style Fused Inter-Chunk State Recurrence

### Selected Proposal
- **ID**: 057-flashrnn-fused-inter-chunk-state-recurrence
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: High priority proposal targeting a real HBM bottleneck in chunkwise linear RNNs. The inter-chunk state scan reads/writes state matrices from HBM at each chunk boundary. By keeping states in registers/SRAM (FlashRNN-style), we can reduce HBM traffic by ~1.5x. This is a kernel microbenchmark - pure systems optimization, no model changes.

### Implementation Plan
1. Implement baseline sequential inter-chunk scan (reads h_{k-1} from HBM each step)
2. Implement FlashRNN-style fused scan (keeps state in registers across all G steps)
3. Write benchmarking script that compares both approaches
4. Verify numerical correctness (bit-exact results)
5. Measure timing across G in {32, 64, 128}
6. Deploy to Modal on GPU (needs GPU for Triton kernels)
7. Report results vs success criteria

### Key Design Decisions
- **GLA diagonal case only**: Simplest case where A_k = diag(gamma_k). Each row of the state matrix is an independent scalar-vector recurrence.
- **Triton kernels**: Both baseline and proposed use Triton for fair comparison
- **Microbenchmark focus**: No full model needed - just the kernel comparison
- **Configuration**: G=64 chunks, dk=dv=64, H=16 heads, batch=4

---

## [2026-02-16 00:01] Attempt: Implementing kernels and benchmark

**Goal**: Write complete MVE code

**Actions**:
- Creating models/flashrnn_scan.py with both kernels
- Creating train.py as benchmark script
- Creating modal_config.py for GPU deployment
- Creating config.yaml and pyproject.toml


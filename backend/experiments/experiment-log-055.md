# Experiment Log 055: ZeCO All-Scan Sequence Parallelism for Gated DeltaNet

## [2026-02-16 00:00] Experiment 055: ZeCO All-Scan SP with WY State Factorization

### Selected Proposal
- **ID**: 055-zeco-allscan-gated-deltanet-sequence-parallelism
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Core novelty is extending ZeCO's All-Scan collective to handle Gated DeltaNet's non-diagonal transitions via WY factorization. The MVE is a standalone microbenchmark comparing communication latency of WY-All-Scan vs LASP-2 AllGather vs ZeCO-GLA All-Scan.

### MVE Summary
- **What**: Standalone All-Scan collective microbenchmark (no full model training)
- **Where**: 8x GPU node (H100 preferred, A100 acceptable)
- **Shapes**: d_k = d_v = 128, H = 16, P in {2, 4, 8}
- **Comparison**: WY-All-Scan vs LASP-2 AllGather vs ZeCO-GLA All-Scan
- **Success criteria**:
  1. WY-All-Scan latency at P=8 < 2x ZeCO-GLA latency
  2. WY-All-Scan latency at P=8 < 0.5x LASP-2 AllGather latency
  3. Numerical output matches single-device prefix scan to BF16 precision (< 1e-3 relative error)
  4. WY correction matmul overhead < 0.5 ms on H100 per pipeline stage

### Implementation Plan
1. Implement WY-factored All-Scan collective using PyTorch distributed (NCCL)
2. Implement LASP-2 AllGather baseline for comparison
3. Implement ZeCO-GLA All-Scan (diagonal-only) as reference
4. Implement single-device sequential prefix scan for numerical verification
5. Create benchmark harness with proper warmup and timing
6. Deploy on Modal with multi-GPU (8x A100 or H100)
7. Collect latency metrics and numerical accuracy

---

## [2026-02-16 00:05] Implementation: Core Architecture

### Design Decisions
1. **Using PyTorch distributed (NCCL backend)**: Standard for multi-GPU communication. P2P send/recv maps to NCCL's point-to-point API.
2. **Simulating All-Scan as sequential P2P pipeline**: Device 0 sends to 1, which processes and sends to 2, etc. This is the core ZeCO pattern.
3. **BF16 computation with FP32 WY accumulation**: Per proposal recommendation for numerical stability.
4. **Using A100 on Modal**: H100s may not be available; A100s have NVLink and NCCL P2P support.

### Files to Create
- `models/__init__.py` - Empty init
- `models/wy_allscan.py` - WY-factored All-Scan collective
- `models/lasp2_allgather.py` - LASP-2 AllGather baseline
- `models/zeco_gla_allscan.py` - ZeCO-GLA (diagonal) All-Scan reference
- `models/sequential_scan.py` - Single-device sequential scan for verification
- `benchmark.py` - Main benchmark harness
- `modal_config.py` - Modal deployment
- `config.yaml` - Configuration
- `pyproject.toml` - Dependencies
- `README.md` - Instructions

---

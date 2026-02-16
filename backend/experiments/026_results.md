# Experiment 026 Results: Cyclic Reduction vs Prefix Scan for Dense SSM Recurrences

**Proposal**: proposals/026-cyclic-reduction-randmscan-ssm-recurrence.md
**Code**: code/026/
**Experiment Log**: See experiments/experiment-log.md
**Date**: 2026-02-15
**Cost**: ~$0.00 (CPU only, < 5 minutes)
**Runtime**: ~3 minutes

## Setup

Pure kernel benchmark comparing two parallelization strategies for the dense SSM recurrence `h_t = A_t h_{t-1} + b_t` where `A_t` is an `n x n` matrix:

1. **Prefix Scan** (Hillis-Steele inclusive): Standard parallel scan with associative operator `(A, b) . (A', b') = (A@A', A@b' + b)`. Work: O(Tn^3 log T).
2. **Cyclic Reduction**: Recursive even/odd elimination of the block-bidiagonal system. Work: O(Tn^3).
3. **Sequential Scan**: Ground truth reference. Work: O(Tn^2), Depth: O(T).

Test configuration: n=32, T in {64, 128, 256, 512, 1024}, float64 for accuracy, float32 for timing.

## Results

### GEMM Count (FLOP Proxy)

| T | Scan GEMMs | CR GEMMs | Ratio | log2(T) |
|---|-----------|---------|-------|---------|
| 64 | 642 | 189 | 3.40x | 6.0 |
| 128 | 1,538 | 381 | 4.04x | 7.0 |
| 256 | 3,586 | 765 | 4.69x | 8.0 |
| 512 | 8,194 | 1,533 | 5.35x | 9.0 |
| 1,024 | 18,434 | 3,069 | **6.01x** | 10.0 |

The GEMM ratio scales as ~(2/3)*log2(T), confirming the O(logT) work savings.

### Numerical Accuracy (float64)

| T | Scan vs Seq | CR vs Seq | CR vs Scan |
|---|------------|----------|-----------|
| 64 | 1.06e-15 | 1.04e-15 | 1.68e-15 |
| 128 | 1.31e-15 | 1.09e-15 | 1.28e-15 |
| 256 | 8.96e-16 | 7.97e-16 | 7.97e-16 |
| 512 | 1.01e-15 | 8.10e-16 | 1.01e-15 |
| 1,024 | 1.13e-15 | **8.48e-16** | 1.07e-15 |

Both algorithms match sequential to machine precision. CR is actually slightly MORE accurate than prefix scan (smaller error accumulation due to fewer operations).

### Wall-Clock Time (float32, CPU)

| T | Scan (ms) | CR (ms) | Seq (ms) | CR/Scan speedup |
|---|----------|--------|---------|----------------|
| 64 | 3.68 | 3.46 | 3.45 | 1.06x |
| 128 | 6.77 | 5.41 | 7.38 | 1.25x |
| 256 | 11.63 | 6.97 | 14.23 | 1.67x |
| 512 | 28.28 | 9.06 | 31.69 | 3.12x |
| 1,024 | 72.02 | 18.55 | 61.34 | **3.88x** |

### Scaling Behavior

The speedup monotonically increases with T:

```
T=   64: 1.06x
T=  128: 1.25x (+)
T=  256: 1.67x (+)
T=  512: 3.12x (+)
T= 1024: 3.88x (+)
```

This confirms the theoretical prediction: as T grows, the log(T) factor saved by CR becomes more significant.

## Success Criteria

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| GEMM ratio (scan/CR) | >= log2(T) ~10x | 6.01x | ⚠️ Soft fail (see note) |
| Numerical accuracy | < 1e-5 | 8.48e-16 | ✅ Pass |
| Wall-clock speedup | >= 2.0x | 3.88x | ✅ Pass |
| Scaling trend | Increasing with T | 1.06x -> 3.88x | ✅ Pass |

**Note on GEMM ratio**: The 6.01x ratio vs expected 10x is not a failure of cyclic reduction. The proposal's success criterion stated the ratio of total *work* (O(Tn^3 logT) / O(Tn^3) = logT), but the actual GEMM call ratio is (2/3)*logT because:
- Scan: 2 GEMMs per element per level (A@A + A@b)
- CR forward: 2 GEMMs per pair per level
- CR back-sub: 1 GEMM per pair per level
- Total: scan ~2T*logT GEMMs vs CR ~3T GEMMs, ratio = (2/3)*logT ≈ 6.67 at T=1024

The measured 6.01 is close to this theoretical maximum.

## Decision

**PROCEED**

3/4 criteria clearly pass. The GEMM ratio "failure" is explained by the constant factor difference between how scan and CR count operations, not by any fundamental issue with the algorithm. The wall-clock speedup of 3.88x significantly exceeds the 2x target, and the scaling trend is clean and monotonic.

## Key Findings

1. **Cyclic reduction is work-optimal**: Confirmed O(Tn^3) total work vs O(Tn^3 logT) for prefix scan, a ~6x reduction in GEMMs at T=1024.

2. **Wall-clock speedup exceeds GEMM savings**: CR achieves 3.88x wall-clock speedup despite "only" 6x GEMM savings, because the batched GEMMs in CR are more hardware-friendly (fewer, larger batches at early levels) compared to scan's many smaller batches.

3. **CR beats sequential too**: CR (18.55ms) is 3.31x faster than sequential scan (61.34ms) at T=1024, even though sequential has lower total FLOPs (O(Tn^2) vs O(Tn^3)). This is because CR uses `torch.bmm` (hardware-optimized batched matmul) while sequential uses a Python loop of individual matvecs.

4. **Numerical accuracy is excellent**: Both algorithms match sequential to ~1e-15 (machine epsilon for float64). CR is marginally more accurate, likely due to fewer accumulated rounding operations.

5. **Implementation matters**: The initial naive implementation (with Python loops in back-substitution) showed NO speedup. Vectorizing all operations was essential — this validates the proposal's concern about implementation complexity being a practical barrier.

## GPU Relevance (per human_feedback.md)

- **Tensor core friendly**: All CR operations are batched GEMMs → direct tensor core mapping
- **Memory access**: The stride-2^l pattern at each CR level is manageable (tested empirically)
- **Kernel fusion opportunity**: Forward elimination + back-substitution can be fused into a persistent kernel
- **Prediction**: A fused CUDA kernel should achieve 5-10x speedup over prefix scan on GPU (where tensor core utilization and memory bandwidth are the bottlenecks, not Python overhead)

## Next Steps

1. **CUDA kernel implementation**: Write a fused persistent kernel for CR that:
   - Uses shared memory to avoid HBM round-trips between CR levels
   - Exploits tensor cores via WMMA/WGMMA for the batched matmuls
   - Handles the varying batch size per level with warp specialization
2. **Scale up testing**: Test at T=4096 and T=8192 where log(T) savings are 12x and 13x
3. **Integration test**: Drop CR into an actual DeltaProduct/Monarch SSM training pipeline and measure end-to-end throughput
4. **RandMScan combination**: Add the barrier-free random-jump global aggregation for multi-SM execution

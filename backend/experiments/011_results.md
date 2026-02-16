# Experiment 011 Results: Neumann-Approximate Resolvent for DPLR SSM Kernel

**Proposal**: proposals/011-neumann-resolvent-chunkwise-ssm.md
**Code**: code/011/
**Experiment Log**: See experiments/experiment-log.md
**Date**: 2026-02-15
**Cost**: $0.00 (CPU-only, no GPU needed)
**Runtime**: ~3 minutes

## Setup

Tested whether the truncated Neumann series approximation of the DPLR SSM resolvent $(zI - A)^{-1}$ can match the exact Woodbury computation. Four tests:
1. **Kernel accuracy sweep**: Relative error vs exact at k={2,4,6,8,12,16}
2. **Near-resonance robustness**: BF16 behavior when z ≈ lambda_i
3. **Speed comparison**: Wall-clock time at N={32,64,128,256}
4. **Spectral radius distribution**: Convergence guarantee check

Parameters: N=64 state dim, r=1 low-rank, d=8 hidden dim, L=1024 frequencies, HiPPO-LegS initialization.

### Implementation Bugs Fixed
- **Woodbury sign error**: Corrected from $(I + F)^{-1}$ to $(I - F)^{-1}$ for $(M - PQ^*)^{-1}$
- **Neumann factorization order**: Corrected from $D_z (I-E)^{-1}$ to $(I-E)^{-1} D_z$
- **Efficient kernel base term**: Fixed $C D_z^2 B$ → $C D_z B$ in batched computation
- **Complex dtype mismatch**: Ensured consistent float64/complex128 throughout

## Results

### Test 1: Kernel Accuracy Sweep

| k | Mean Error | Max Error | Status |
|---|-----------|----------|--------|
| 2 | 2.67e-03 | 6.87e-03 | ❌ Fail |
| **4** | **1.94e-05** | **6.65e-05** | **✅ Pass** |
| 6 | 1.71e-07 | 6.86e-07 | ✅ Pass |
| 8 | 1.65e-09 | 7.27e-09 | ✅ Pass |
| 12 | 1.78e-13 | 8.49e-13 | ✅ Pass |
| 16 | 6.79e-16 | 6.92e-16 | ✅ Pass |

**Key finding**: Each +2 in k reduces error by ~100x, consistent with spectral radius ρ ≈ 0.01. k=4 is sufficient for < 1e-3 accuracy by a wide margin (6.65e-5 worst case). k=2 fails marginally at 6.87e-3.

### Test 2: Near-Resonance Robustness

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Neumann finite at ε=1e-3 | Finite | Finite | ✅ Pass |
| Woodbury overflow at ε=1e-3 | Overflow | Finite | ⚠️ Neutral |

**Both methods produce finite results** at all tested epsilon values (1e-1 to 1e-4). The near-resonance concern is a non-issue for HiPPO-LegS initialization because eigenvalues have real part -0.5, ensuring |z - λ| ≥ 0.5 for z on the unit circle. D_z entries bounded by 2.

### Test 3: Speed Comparison

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Speedup at N=64 | > 1x | 1.07x | ✅ Pass |
| Speedup at N=128 | > 1x | 3.77x | ✅ Pass |
| Speedup at N=256 | > 1x | 8.92x | ✅ Pass |

**Neumann efficient formula scales as O(LNr)** vs Woodbury's O(LN²). For r=1, this gives dramatic speedups at larger N. At N=256, Neumann is 8.9x faster.

### Test 4: Spectral Radius (Convergence Guarantee)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Frac(ρ > 1) | < 10% | 0.0% | ✅ Pass |
| Max spectral radius | < 1.0 | 0.107 | ✅ Pass |

Spectral radius is tiny (mean 0.013-0.054 across trials). Convergence is guaranteed for all frequencies.

## Success Criteria

- ✅ **Relative kernel error < 1e-3 for k ≤ 8**: Achieved 6.65e-5 at k=4, 7.27e-9 at k=8
- ⚠️ **Near-resonance robustness**: Both methods finite — near-resonance is not a practical issue with HiPPO init
- ✅ **Speed N ≥ 64**: Neumann 1.07x-8.92x faster (increasing with N)
- ✅ **Convergence guarantee**: 0% of frequencies have ρ > 1

## Decision

**PROCEED** (with caveats)

### Why Proceed
1. **Accuracy is excellent**: k=4 Neumann matches exact Woodbury to < 1e-4 relative error, far exceeding the 1e-3 target
2. **Speed scaling is real**: O(LNr) vs O(LN²) gives 3.8-8.9x speedup at N=128-256
3. **Convergence is guaranteed**: Zero divergent frequencies with HiPPO initialization
4. **Minimal overhead**: Only 4 scalar multiplications per frequency (for r=1, k=4)

### Caveats and Limitations
1. **Near-resonance motivation is weak**: The proposal's primary selling point (BF16 stability near resonance) doesn't materialize with standard initializations. The eigenvalues are well-separated from evaluation points.
2. **Speed comes from efficient formula, not GEMM**: The speedup is from avoiding the N×N resolvent matrix, not from GEMM-vs-division. The Cauchy kernel trick (S4's standard method) achieves similar scaling without Neumann approximation.
3. **r=1 is trivially efficient**: With r=1, the "core GEMM" is just scalar multiplication. The proposal's GEMM advantage would only manifest at r ≥ 2.
4. **Untested with learned parameters**: HiPPO initialization gives small P, Q norms (spectral radius ~0.01). After training, P and Q may grow, potentially breaking convergence.

## Next Steps

1. **Test with r=2 initialization**: Verify that GEMM advantage materializes for rank-2 corrections
2. **Test with learned (non-HiPPO) parameters**: Sample P, Q with larger norms to stress convergence
3. **Compare against Cauchy kernel trick**: The standard S4 evaluation also avoids O(N²) — is Neumann actually faster?
4. **GPU benchmark**: CPU results may not transfer to GPU where GEMM/elementwise-op ratios differ
5. **If proceeding to full experiment**: Focus on speed at large N rather than numerical stability as the value proposition

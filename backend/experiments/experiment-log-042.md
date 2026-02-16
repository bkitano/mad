# Experiment Log 042: Contraction-Ordered Multi-Operand Chunkwise GLA Fusion

## [2026-02-16 00:00] Experiment 042: Contraction-Ordered Chunkwise GLA Multi-Operand Fusion

### Selected Proposal
- **ID**: 042-contraction-ordered-chunkwise-gla-multi-operand-fusion
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Pure kernel microbenchmark comparing two contraction orderings for GLA intra-chunk computation. No model training needed - just timing random tensor operations on GPU. A100 for ~10 min.

### MVE Summary
- **Task**: Time the intra-chunk computation for random tensors
- **Implementations**:
  - Path 1 (standard): Q*K^T -> mask -> *V + Q*h
  - Path 2 (right-associated with rank-r mask correction): Q*(cumsum(K^T*V)) - correction + Q*h
- **Sweep**: C in {32,64,128}, d in {64,128,256}, r in {1,4,8,16,C}
- **Success criteria**: Path 2 >= 10% faster for at least one realistic config (C=64, d=128, r<=8)
- **Additional**: Numerical agreement within epsilon < 1e-3 relative error (BF16)

### Implementation Plan
1. Implement both paths as PyTorch operations (not Triton kernels - keep MVE minimal)
2. Create benchmark script to sweep configurations
3. Measure wall-clock time and numerical correctness
4. Deploy to Modal on A100
5. Report results

### [00:00] Design Decision: PyTorch vs Triton
The proposal says "Triton kernels" but for MVE we'll use PyTorch operations first. This gives us:
- FLOP count validation (do the math paths actually differ in practice?)
- Numerical correctness verification
- Speedup signal from contraction ordering (even without fusion)
If PyTorch shows speedup, it validates the contraction ordering hypothesis. Triton fusion is a follow-up.

### [00:01] Attempt: Creating experiment directory and files
**Goal**: Set up code/042 with all necessary files
**Actions**: Creating models, benchmark script, modal config, etc.

# Experiment 005 Results: Segmented-HSS Linear Attention

**Proposal**: proposals/005-segmented-hss-linear-attention.md
**Code**: code/005/
**Experiment Log**: See experiments/experiment-log.md
**Date**: 2026-02-15
**Cost**: ~$0.15 (T4 GPU, ~10 minutes total)
**Runtime**: 562s total (Dense: 76.6s, HSS: 485.5s)
**Modal App**: ap-ojbC0Ula7kmEQPMzWBCWYJ
**W&B**: https://wandb.ai/bkitano/mve-005-hss-linear-attention

## Setup

Implemented a single HSS linear attention layer (d=64, r=8, ~21K params) and compared against a dense linear attention baseline on a hierarchical copying task:
- **Task**: Copy 8 tokens with hierarchy-dependent delays (level-1: +0, level-2: +4, level-3: +8 steps)
- **Data**: 2000 synthetic sequences, 1400 train / 300 val / 300 test
- **HSS Implementation**: Flat tensor representation (not recursive Python tree) for GPU compatibility
- **Feature map**: phi(x) = elu(x) + 1

## Results

| Metric | Target | Dense-LinAttn | HSS-LinAttn | Status |
|--------|--------|---------------|-------------|--------|
| Test Accuracy | ≥ 90% | 25.46% | 25.58% | ❌ FAIL (both) |
| Level 1 Accuracy | High | 22.83% | 24.33% | ❌ FAIL |
| Level 2 Accuracy | High | 28.17% | 27.00% | ❌ FAIL |
| Level 3 Accuracy | High | 22.67% | 24.00% | ❌ FAIL |
| State Low-Rank | Yes | N/A | Yes (100% energy in top-8 SVs) | ✅ PASS |
| Memory Ratio | < 0.2 | 1.0 (baseline) | 0.656 | ❌ FAIL |
| Training Speed | Comparable | 1.0x | 0.16x (6.3x slower) | ❌ FAIL |

## Success Criteria

- ❌ **Criterion 1: Accuracy ≥ 90%**: Both models plateau at ~25% (barely above random for position-dependent guessing). The hierarchical copying task appears too hard for single-layer recurrent linear attention. Neither model can effectively recall tokens at delayed positions.

- ✅ **Criterion 2: Hierarchical state structure**: The HSS state's off-diagonal blocks ARE low-rank — 100% of energy captured by top-8 singular values. This means the HSS structure constraint is naturally compatible with linear attention state evolution.

- ❌ **Criterion 3: Memory ratio < 0.2**: At d=64, r=8, the HSS ratio is 0.656. The proposal itself predicted this: "HSS advantage grows with d; may need d ≥ 1024 to see significant memory savings." At d=64, the tree overhead (leaf blocks + basis matrices at each level) is substantial relative to d².

## Additional Findings

### HSS is 6.3x slower than Dense on GPU
- Dense: 1.1s/epoch, HSS: 7.0s/epoch
- Root cause: sequential Python loop through timesteps with per-timestep HSS operations (einsum on small tensors)
- This confirms the human feedback concern: "Sequential tree traversals: HSS hierarchies, recursive decompositions that can't parallelize"
- Even the flat tensor implementation can't overcome the inherent sequential bottleneck

### Task Design Issue
- 25% accuracy suggests the model is learning position-based heuristics rather than actual copying
- Single-layer linear attention lacks the capacity for this multi-scale retrieval task
- A transformer with multi-head softmax attention would likely solve this easily
- The task tests attention capacity, not just HSS structure — but the linear attention baseline also fails

### HSS approximation quality
- Random basis initialization gives ~97% relative error on rank-1 updates
- But the trained network learns to project into compatible subspaces
- After training, off-diagonal blocks are perfectly low-rank (r=8 captures all energy)
- The approximation works — it just doesn't help because both models lack capacity

## Decision

**ABANDON**

### Reasoning
1. **Both models fail** — HSS doesn't underperform dense, but both achieve only ~25%. This means the experiment doesn't discriminate between HSS and dense.
2. **Memory advantage absent at MVE scale** — HSS only helps at d ≥ 1024, but running experiments at that scale requires much more compute and a real language modeling task.
3. **Severe speed penalty** — 6.3x slower training makes HSS impractical even if accuracy were acceptable. This directly conflicts with the human feedback requirement for "wall-clock GPU speedup."
4. **GPU-unfriendly architecture** — The sequential tree traversals in HSS operations cannot efficiently utilize GPU parallelism. This was predicted by the human feedback flagging "sequential tree traversals" and "exotic math structures."

### What We Learned
1. **HSS structure is naturally compatible with linear attention states** — off-diagonal blocks are low-rank after training
2. **But HSS doesn't help in practice** — no accuracy improvement, significant speed cost, no memory benefit at small d
3. **Human feedback was correct** — HSS hierarchies are fundamentally GPU-unfriendly due to sequential tree traversals
4. **The crossover point for memory (d ≥ 1024) is too large** for efficient MVE validation

## Next Steps

**Do NOT proceed to full WikiText-103 training.** The fundamental issues (speed, memory at practical scales) make this approach unviable for GPU pretraining efficiency.

If revisiting this direction:
1. Consider diagonal + low-rank (DPLR) instead of full HSS — simpler, more GPU-friendly
2. Or block-diagonal + FFT (Monarch matrices) for structured state compression
3. The linear attention + hierarchical structure idea has merit, but needs a GPU-friendly implementation (not tree-based)

# Experiment 029 Results: Circulant FAVOR+ Linear Attention

**Proposal**: proposals/029-circulant-favor-plus-linear-attention.md
**Code**: code/029/
**Experiment Log**: See experiments/experiment-log.md
**Date**: 2026-02-15
**Cost**: ~$0.10 (4 runs on T4, total ~20 minutes GPU time)
**Runtime**: ~20 minutes total across 4 attempts

## Setup

Implemented the MVE from proposal 029: testing whether circulant random projections can replace dense projections in FAVOR+ while preserving softmax kernel approximation quality.

**Task**: Associative recall — given 8 key-value pairs and a query key, output the corresponding value.
- 5K train / 1K test sequences, seq_len=64, vocabulary size 16

**Models compared** (all ~119K params, d=64, 4 heads, 2 layers):
1. **Dense FAVOR+**: Standard FAVOR+ with dense orthogonal random features
2. **C-FAVOR+**: Circulant random features via FFT (learnable r, fixed s)
3. **ReLU Linear Attention**: Simple ReLU(Q) @ ReLU(K)^T (baseline)
4. **Softmax Attention**: Standard softmax (quality ceiling)

## Results

### Final Run (Attempt 4: m=64, L2 normalization, no max-subtraction)

| Metric | Dense FAVOR+ | C-FAVOR+ | ReLU Linear | Softmax |
|--------|-------------|----------|-------------|---------|
| Test Accuracy | 23.1% | 23.8% | **98.5%** | **99.8%** |
| Best Val Accuracy | 23.1% | 23.8% | 98.5% | 99.8% |
| Train Accuracy | 93.5% | 90.3% | 100.0% | 100.0% |
| Parameters | 118,674 | 119,186 | 118,674 | 118,674 |
| Feature Map (ms) | 0.413 | 0.745 | 0.018 | N/A |
| NaN count | 0 | 0 | 0 | 0 |

### All Attempts Summary

| Attempt | Change | Dense FAVOR+ | C-FAVOR+ | ReLU | Softmax |
|---------|--------|-------------|----------|------|---------|
| 1 | d=32, 1 layer, 1 head | 23.0% | 20.3% | 26.7% | 26.0% |
| 2 | d=64, 2 layers, 4 heads, m=16 | 19.8% | 22.0% | 97.6% | 99.9% |
| 3 | + L2 norm + max-subtraction | 23.1% | 24.8% | 99.4% | 99.9% |
| 4 | + m=64, remove max-sub | 23.1% | 23.8% | 98.5% | 99.8% |

## Success Criteria

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| C-FAVOR+ > 90% accuracy | > 90% | 23.8% | ❌ FAIL |
| C-FAVOR+ FM faster than dense | > 1.0x | 0.56x | ❌ FAIL |
| FAVOR+ > ReLU by 20% | > 20% gap | -75.4% gap | ❌ FAIL |
| C-FAVOR+ within 10% of dense | < 10% gap | 0.7% gap | ✅ PASS |

## Key Findings

### 1. Circulant DOES preserve FAVOR+ quality
C-FAVOR+ ≈ Dense FAVOR+ in all 4 runs (within 1-3% accuracy). The circulant projection from CBE theory successfully preserves the random feature quality. **The proposal's core mathematical claim is validated.**

### 2. FAVOR+ fundamentally fails on associative recall
Both FAVOR+ variants achieve ~23% test accuracy (vs 6.25% random chance) while memorizing training data (>90% train acc). The massive overfitting gap (93% train vs 23% test) indicates that FAVOR+ random features don't capture the discriminative signal needed for precise key-value retrieval.

### 3. ReLU linear attention dominates FAVOR+
Simple ReLU features (98.5%) dramatically outperform FAVOR+ positive random features (23%). This contradicts the proposal's expectation that "Both FAVOR+ variants significantly outperform ReLU linear attention (> 20% accuracy gap)."

### 4. Feature map speed at d_k=16
The circulant FFT is 0.56x SLOWER than dense matmul at d_k=16. This confirms the proposal's Risk #2: "FFT constant factor: At typical head dimensions (d = 64–128), the cuFFT overhead may dominate." The crossover requires larger d_k.

## Bugs & Fixes Discovered

1. **Missing L2 normalization** (Attempt 1-2): Without normalizing Q,K to unit sphere, FAVOR+ features span 37 million x range, making KV state accumulation degenerate.
2. **Per-token max-subtraction** (Attempt 3): Subtracting max per-token breaks the kernel approximation by making features token-local. Cross-token comparison semantics are destroyed.
3. **Insufficient random features** (Attempt 2-3): m=d_k=16 gives only 5% top-1 match rate for kernel approximation. However, increasing to m=64 didn't help (Attempt 4), suggesting the issue is deeper than approximation quality.

## Decision

**ABANDON**

## Reasoning

While C-FAVOR+ successfully matches dense FAVOR+ quality (validating the circulant optimization), the FAVOR+ foundation itself fails catastrophically on associative recall. The 75% accuracy gap between FAVOR+ and simple ReLU linear attention makes the circulant speedup irrelevant — optimizing a broken approach doesn't make it useful.

The proposal inadvertently chose a task that exposes FAVOR+'s fundamental weakness: random feature maps don't provide the precise key-value binding needed for associative recall, regardless of whether the projection is dense or circulant. FAVOR+ may work better for soft attention patterns in language modeling, but the MVE convincingly shows the approach is not viable for the canonical attention quality test.

## Next Steps

- **Do NOT** proceed to full experiment (WikiText-103, LRA)
- If revisiting FAVOR+ at all:
  - Test on language modeling where soft attention may be sufficient
  - Use d_k ≥ 128 where circulant FFT speedup is real
  - Add the proposal's cosine reweighting + sigmoid gating (orthogonal improvements)
- **Consider pivoting**: The ReLU linear attention baseline (98.5%) suggests that simpler feature maps may be better starting points for efficient attention
- The circulant optimization itself is sound — could apply it to other random feature methods that work better than FAVOR+

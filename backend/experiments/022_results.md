# Experiment 022 Results: Displacement-Rank SSM (DR-SSM)

**Proposal**: proposals/022-displacement-rank-ssm-state-transitions.md
**Code**: code/022/
**Experiment Log**: See experiments/experiment-log.md
**Date**: 2026-02-15
**Cost**: ~$0.00 (CPU-only, no GPU)
**Runtime**: ~15 minutes total (primary experiment + diagnostics)

## Setup

Implemented and tested Displacement-Rank SSM (DR-SSM) with Cauchy-like state transitions parameterized by displacement rank α. The model uses:
- **Architecture**: 2-layer DR-SSM with pre-norm residual connections, MLP classification head
- **State transition**: A_{ij} = d_i δ_{ij} + Σ_k G_{ik} H_{jk} / (s_i - s_j) where s are Chebyshev nodes
- **Task**: S5 permutation composition (120-class classification)
- **Configs tested**: α ∈ {0, 1, 2, 4, 16(dense)} at d_model=64, n=16, 2 layers

## Results

### Primary Experiment (seq_len=12, 5K train samples, 80 epochs)

| Model | α | Test Acc | Best Val Acc | Epochs | Params | Forward (ms) | NaN |
|-------|---|----------|-------------|--------|--------|-------------|-----|
| Diagonal | 0 | 33.6% | 37.0% | 80 | 30,648 | 11.77 | 0 |
| DR-SSM | 1 | **95.8%** | 94.6% | 59 | 34,808 | 28.94 | 0 |
| DR-SSM | 2 | 87.6% | 90.0% | 80 | 38,968 | 36.20 | 0 |
| DR-SSM | 4 | **95.8%** | 95.8% | 27 | 47,288 | 44.07 | 0 |
| Dense | 16 | **97.4%** | 97.2% | 12 | 63,928 | 34.68 | 0 |

### Supplementary: seq_len=20 (harder, 2^20 > training set)

| Model | α | Best Val Acc (40 epochs) |
|-------|---|------------------------|
| Diagonal | 0 | 7.6% |
| DR-SSM | 1 | 3.4% |
| DR-SSM | 4 | 1.0% |
| Dense | 16 | **97.2%** (17 epochs) |

### Speed & Throughput

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| α=4 / Dense speed | > 0.6× | 0.79× | ✅ Pass |
| Cauchy / Dense matvec throughput | > 0.3× | 0.20× | ❌ Fail |
| Cauchy matvec time | - | 0.589 ms | - |
| Dense matvec time | - | 0.121 ms | - |

### Truncation Error

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Relative error after 20 compositions | < 10% | 0.00% | ✅ Pass |

## Success Criteria

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| α=4 accuracy on S5 | > 85% | 95.8% (seq12) / 1.0% (seq20) | ⚠️ Pass on easy / ❌ Fail on hard |
| α=1 accuracy on S5 | < 70% | 95.8% (seq12) / 3.4% (seq20) | ❌ Fail (too good on easy, too bad on hard) |
| α=0 accuracy on S5 | < 50% | 33.6% (seq12) / 7.6% (seq20) | ✅ Pass |
| α=4 speed vs dense | > 0.6× | 0.79× | ✅ Pass |
| α=4 accuracy gap vs dense | < 5% | 1.6% (seq12) | ✅ Pass |
| Cauchy matvec throughput | > 0.3× dense | 0.20× | ❌ Fail |
| Truncation error | < 10% | 0.00% | ✅ Pass |
| **α=4 beats α=1** (kill criterion) | α=4 > α=1 | 95.8% = 95.8% | **❌ KILL** |

## Failure Criteria Triggered

1. **❌ KILL: α=4 does NOT outperform α=1** on the S5 task at either sequence length. At seq_len=12 they tie (95.8%); at seq_len=20 both fail completely (1.0% and 3.4% respectively). The additional Cauchy mixing capacity from higher displacement rank provides no benefit.

2. **❌ Cauchy matvec overhead**: At n=16, the Cauchy matvec is 4.9× slower than dense matvec (0.20× throughput vs 0.3× target). The proposal noted this risk: "the O(n log n) Cauchy matvec has high constant factors from FFT. For n=64, a dense O(n²)=4096 matvec may be faster."

## Key Finding: Optimization Barrier

The most important discovery is that the Cauchy structure creates **fundamental optimization difficulties**:

- **At seq_len=12** (easy task, 2^12 < training data): All models can memorize, so Cauchy structure doesn't help but doesn't hurt either.
- **At seq_len=20** (hard task, genuine generalization): **Only the Dense model learns** (97.2%). All Cauchy-structured models (α=1, 2, 4) fail completely (<4% accuracy).

**Root cause**: The 1/(s_i - s_j) terms in the Cauchy kernel create ill-conditioned gradients:
- Without generator normalization: constant NaN/Inf (1196 NaN events in 60 epochs)
- With Frobenius normalization: generators constrained too tightly, can't learn effective mixing
- With learned scale factor: model learns to keep generators near zero (~0.0001), becoming essentially diagonal

The Cauchy structure's theoretical expressivity is irrelevant if the model can't be trained.

## Decision

**ABANDON**

The displacement-rank SSM hypothesis is not supported by experimental evidence:

1. **No rank-scaling signal**: Increasing α from 1 to 4 provides zero benefit on the canonical S5 benchmark
2. **Optimization failure**: Cauchy structure prevents learning on the harder (genuinely useful) version of the task
3. **Slower than dense**: The Cauchy matvec is actually slower than dense at practical n=16
4. **Dense works**: A simple dense SSM with tanh scaling solves the task trivially

The proposal's elegant mathematical framework (displacement rank as capacity knob, closure under composition, Cauchy kernel compatibility) does not translate to practical benefit.

## What We Learned

1. **Theoretical expressivity ≠ practical learnability**: The Cauchy-like matrix class is rich enough in theory, but the 1/(s_i - s_j) kernel creates optimization barriers that prevent learning
2. **n=16 is too small for Cauchy benefits**: At this scale, dense O(n²) is faster and more trainable than O(αn log n) Cauchy
3. **α=1 already saturates easy tasks**: On simple S5 tasks, even α=1 (S4-equivalent DPLR) is enough
4. **The gap is in optimization, not expressivity**: Dense SSM has identical mathematical structure to Cauchy-like with α=n, but different parameterization that works much better

## Next Steps

- **Do NOT proceed** to full experiment
- The displacement rank framework should NOT be pursued in its current form
- **Possible salvage paths** (low priority):
  - Test at larger n (>256) where Chebyshev nodes are better separated
  - Try Stein-type displacement (A - MAN = GH^T) which avoids Sylvester singularities
  - Investigate learnable displacement operators instead of fixed Chebyshev nodes
- **Recommended instead**: Focus on DPLR (S4) and Monarch-based SSMs which have proven optimization properties

# MVE 017: Hyperoctahedral Signed-Permutation SSM

## Overview

Tests whether signed permutation matrices (hyperoctahedral group B_n = Z_2^n ⋊ S_n) provide a useful inductive bias for state-space model state transitions.

**Proposal**: `proposals/017-hyperoctahedral-signed-permutation-ssm.md`

## Task

**B3 Composition**: Given a sequence of generators of B_3 (3 adjacent transpositions + 1 sign flip), predict the composed signed permutation at each position.

- |B_3| = 2^3 * 3! = 48 elements
- 3 generators: σ_1 (swap 0,1), σ_2 (swap 1,2), τ (flip sign 0)
- Sequences of length 8-16

## Models

1. **HyperSSM**: Signed permutation state transitions via Gumbel-Sinkhorn + sigmoid signs
2. **DiagonalSSM**: Standard diagonal SSM baseline (abelian, can't represent B_3)
3. **PermOnlySSM**: Permutation-only (no signs) — tests value of Z_2^n component

## Success Criteria

1. HyperSSM > 90% accuracy on B3 composition
2. DiagonalSSM < 60% accuracy (can't represent non-abelian group)
3. PermOnlySSM < 75% accuracy (missing sign dynamics)
4. No NaN/Inf during training

## Setup & Run

```bash
# Install dependencies
pip install torch numpy pyyaml tqdm

# Run experiment
cd code/017/
python train.py --config config.yaml
```

## Expected Runtime

- Single GPU: < 8 minutes
- CPU: 10-20 minutes
- Cost: ~$0.27

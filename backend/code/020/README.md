# Experiment 007: OH-DeltaProduct MVE

**Proposal**: 020-oscillatory-householder-deltaproduct

## Overview

Tests the Oscillatory Householder DeltaProduct (OH-DeltaProduct) architecture,
which decomposes state transitions into:
- **Oscillatory component** (R_t): LinOSS-style rotation-contraction for stability
- **Householder component** (H_t): Product of reflections for state-tracking expressivity

## Task: S3 Permutation Composition

Compose sequences of S3 (symmetric group on 3 elements) permutations.
S3 is the simplest non-abelian group (6 elements), requiring non-commutative state tracking.

## Models Compared

1. **OH-DeltaProduct** (full, β∈(0,2)) — the proposed model
2. **LinOSS-only** (no Householder) — should fail on non-abelian S3
3. **DeltaProduct-only** (no oscillatory, β∈(0,2)) — should work but with NaN risk
4. **OH-DeltaProduct** (β∈(0,1) ablation) — tests need for negative eigenvalues
5. **DeltaProduct-only** (β∈(0,1)) — ablation control

## Setup

```bash
pip install -e .
python train.py --config config.yaml
```

## Success Criteria

1. OH-DeltaProduct > 95% accuracy on S3
2. LinOSS-only < 40% accuracy
3. DeltaProduct-only > 90% but NaN events
4. OH-DeltaProduct 0% NaN rate
5. β∈(0,1) ablation < 60% accuracy

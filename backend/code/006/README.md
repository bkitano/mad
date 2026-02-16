# MVE 006: Monarch-Gated State Transition SSM

## Hypothesis

Replacing diagonal state transitions in SSMs with input-dependent Monarch-factored
transitions achieves expressivity comparable to full dense transitions while maintaining
near-diagonal computational cost.

## Task

**S5 Permutation Group Composition**: Given a sequence of transpositions from S5 (the
symmetric group on 5 elements), predict the resulting permutation.

This is the canonical test for coordinate mixing ability:
- Diagonal SSMs operate per-coordinate and provably cannot solve this
- Monarch's built-in permutation enables coordinate routing needed for group composition

## Setup

- **Model**: 2 layers, d=64, n=64 (8x8 blocks), ~120K params
- **Data**: 10K synthetic sequences of length 10-50
- **Compute**: Single T4 GPU, < 10 minutes

## Success Criteria

1. Monarch-Gated SSM > 85% accuracy on S5 composition (seq len 20)
2. Diagonal SSM baseline < 50% accuracy on same task
3. Forward pass < 3x slower than diagonal SSM
4. No NaN/Inf during training

## Running

### Via Modal (recommended):
```bash
cd code/006
uv run modal run --detach -m train.modal_config --config config.yaml
```

### Locally (for debugging only):
```bash
cd code/006
uv run python -m train.run_config --config config.yaml
```

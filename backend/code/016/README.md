# MVE 016: GS-Monomial SSM on S5 Permutation Composition

## Proposal
016-gs-monomial-ssm-state-transition.md

## Architecture
**GS-Monomial SSM**: State transition A_t = L_t @ P_shuffle @ R_t
- R_t, L_t: block-diagonal monomial matrices (permutation x diagonal per block)
- P_shuffle: fixed stride permutation for cross-block mixing
- Block size b=4, r=4 blocks, state dim n=16

## Task
**S5 Permutation Composition**: Input sequence of S5 generators, predict prefix composition.
- S5 = symmetric group on 5 elements (120 elements, non-solvable)
- Canonical benchmark for SSM expressivity (Merrill et al., ICML 2024)
- Diagonal SSMs provably cannot represent S5

## Models
1. **GS-Monomial SSM** (with shuffle): Full model with cross-block mixing
2. **Block-Diagonal Only** (no shuffle): Ablation without P_shuffle
3. **Diagonal SSM**: Baseline (expected to fail)

## Success Criteria
1. GS-Monomial > 90% accuracy, Diagonal < 30%
2. GS-Monomial > 85% (PD-SSM comparable)
3. Cross-block mixing benefit > 10 percentage points

## Run

### Local (CPU)
```bash
python train.py --config config.yaml
```

### Modal (GPU)
```bash
modal run --detach modal_config.py --config config.yaml
```

# Experiment 003: DPLR Column-Sparse SSM

**Proposal**: `proposals/003-dplr-column-sparse-cauchy-kernel.md`

## Overview

Tests whether a fixed permutation P applied to a DPLR (Diagonal Plus Low-Rank)
state matrix provides measurable benefit on a parity task that requires
inter-dimension coupling.

**State transition**: `A = P (Lambda + p q^T) P^T`

## Setup

```bash
cd code/003
pip install -e .
# or: pip install torch numpy tqdm pyyaml
```

## Run

```bash
# Run all 4 variants (identity, cyclic, bit-reversal, learned):
python train.py

# Run single variant:
python train.py --p_type identity
python train.py --p_type cyclic
python train.py --p_type bit_reversal
python train.py --p_type learned

# Custom config:
python train.py --config config.yaml --n_seeds 3
```

## Models Compared

1. **DPLR (P = I)**: Standard diagonal + rank-1, no permutation (baseline)
2. **DPLR-CS (cyclic P)**: Cyclic shift permutation
3. **DPLR-CS (bit-reversal P)**: Bit-reversal permutation (maximizes mixing)
4. **DPLR-CS (learned P)**: Doubly-stochastic relaxation with Sinkhorn

## Success Criteria

| Model | Expected Accuracy |
|-------|-------------------|
| DPLR (P = I) | ~50-70% (struggles without coupling) |
| DPLR-CS (any P) | >90% |

## Expected Runtime

~5 minutes total on single GPU (or CPU for this scale).

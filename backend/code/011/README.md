# MVE 011: Neumann Resolvent Kernel Accuracy Test

## Overview

Tests whether the Neumann series approximation of the DPLR SSM resolvent
(zI - A)^{-1} can match the exact Woodbury computation in accuracy, while
offering better numerical stability in BF16 and competitive speed.

**Proposal**: proposals/011-neumann-resolvent-chunkwise-ssm.md

## What This Tests

1. **Kernel accuracy**: Relative error of Neumann vs exact Woodbury across truncation orders k={2,4,6,8,12,16}
2. **Near-resonance robustness**: Behavior when z ≈ lambda_i (eigenvalue)
3. **Speed comparison**: Wall-clock time across state dimensions N={32,64,128,256}
4. **Convergence guarantee**: Spectral radius distribution check

## Success Criteria

- Relative kernel error < 1e-3 for k <= 8
- Neumann in BF16 produces finite results near resonance
- Neumann faster than Woodbury for N >= 64
- < 10% of frequencies have spectral radius > 1

## Setup

```bash
# Install dependencies
pip install torch numpy pyyaml modal

# Run on Modal (recommended)
modal run --detach modal_config.py --config config.yaml

# Run locally (for debugging)
python run_experiment.py --config config.yaml
```

## Files

- `models/resolvent.py` - Core resolvent implementations (Woodbury + Neumann)
- `run_experiment.py` - Experiment script with all 4 tests
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment configuration

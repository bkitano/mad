# Experiment 013: Circulant SSM with Fourier-Domain Parallel Scan

## Overview

This MVE tests whether circulant SSMs, diagonalized via FFT into the Fourier domain,
can learn cyclic group composition (Z_8) better than standard diagonal SSMs.

**Key insight**: Circulant matrices are diagonalized by the DFT, so the recurrence
`h_t = A(x_t) h_{t-1} + B_t x_t` becomes element-wise in frequency space:
`h_hat_t = diag(a_hat(x_t)) h_hat_{t-1} + B_hat_t x_hat_t`

This enables O(log T)-depth parallel scans while maintaining full coordinate mixing.

## Setup

```bash
cd code/013
pip install -e .
```

## Run

```bash
# Run both models (circulant + diagonal baseline)
python train.py

# Run individual models
python train.py --model circulant
python train.py --model diagonal

# Specify device
python train.py --device cuda
python train.py --device cpu
```

## Task

**Z_8 Cyclic Group Composition**: Given a sequence of elements from {0,1,...,7},
predict the cumulative sum modulo 8 at each position.

Example: Input [3, 5, 2] -> Output [3, 0, 2] (cumsum mod 8)

## Success Criteria

1. Circ-SSM achieves >90% accuracy on Z_8 composition at seq_len=32
2. Diagonal SSM baseline achieves <60% accuracy on same task
3. Forward pass throughput of Circ-SSM is >0.5x diagonal SSM
4. Numerical error: ||h_spatial - IFFT(h_scan)||_inf < 1e-4 in FP32

## Architecture

- **Model size**: 2 layers, d_model=64, state_dim=64, ~100K params
- **Circulant SSM**: Input-dependent Fourier eigenvalues with magnitude+phase parameterization
- **Diagonal SSM**: Standard input-dependent diagonal decay (no coordinate mixing)

## File Structure

```
code/013/
├── README.md           # This file
├── config.yaml         # Experiment configuration
├── pyproject.toml      # Dependencies
├── train.py            # Training script
├── models/
│   ├── __init__.py
│   ├── circulant_ssm.py   # Circulant SSM with Fourier-domain scan
│   └── diagonal_ssm.py   # Diagonal SSM baseline
└── tasks/
    ├── __init__.py
    └── z8_dataset.py      # Z_8 cyclic group dataset
```

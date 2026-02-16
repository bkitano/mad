# MVE 014: Log-Semiring SSM — Selective Copying

**Proposal**: `proposals/014-log-semiring-ssm-scan.md`

## Overview

Tests whether replacing the standard (R, +, ×) semiring in SSM parallel scans
with the logarithmic semiring (R, logsumexp, +) enables exact softmax-like
attention. The key recurrence:

    ℓ_t = logsumexp(a_t + ℓ_{t-1}, b_t)

computes the log-partition function of a softmax distribution over input history.

## Task: Selective Copying

Given `[tok_0, _, tok_1, _, ..., tok_7, QUERY:j]`, predict `tok_j`.
This requires **sharp attention** (selecting exactly one position), which is the
core capability of softmax that linear attention lacks.

## Models

| Model | Description | Expected Accuracy |
|-------|-------------|-------------------|
| LogSSM | Log-semiring scan (proposed) | > 90% |
| LinearAttention | RetNet-style (ELU+1 features) | < 60% |
| DiagonalSSM | Mamba-style diagonal gating | < 70% |

## Setup

```bash
pip install -e .
```

## Run

```bash
python train.py
```

Results are saved to `results.yaml`.

## Architecture

- 2 layers, D=64, d_head=16, H=4 heads, ~80K params
- Sequence length: 32
- 8 tokens to remember, 1 query per sequence
- 5000 training sequences
- Target runtime: < 8 minutes on single GPU

# MVE 008: Cosine-Reweighted Log-Linear Attention

## Overview

Tests whether composing cosFormer's cosine reweighting with log-linear attention's hierarchical Fenwick tree states produces compounding improvements on associative recall.

## Task

**Multi-Query Associative Recall (MQAR)**: Given `k1 v1 k2 v2 ... k8 v8 [SEP] q1 q2 ... q8`, recall the associated values. Uses 8 KV pairs at sequence length T=128 with head dimension d=16 to stress-test state capacity.

## Models Compared

| Model | Kernel | States | Expected Accuracy |
|-------|--------|--------|-------------------|
| vanilla_linear | ELU+1 | 1 (d×d) | Low baseline |
| cosformer | ReLU+cos | 2 (d×d) | < 50% (capacity-limited) |
| log_linear | ELU+1 | O(log T) | 60-70% |
| cos_log_linear | ReLU+cos | 2×O(log T) | > 80% |

## Setup

```bash
cd code/008
uv sync
```

## Run (Modal - recommended)

```bash
modal run --detach modal_config.py --config config.yaml
```

## Run (local - for debugging only)

```bash
uv run python train.py --config config.yaml
```

## Success Criteria

1. cos-LogLinear > 80% accuracy on MQAR
2. cosFormer < 50% accuracy (capacity-limited at d=16 with 8 KV pairs)
3. Vanilla log-linear achieves 60-70% accuracy
4. No NaN/Inf during training

## Proposal

See `proposals/008-cosine-log-linear-attention.md`

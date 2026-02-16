# MVE 031: V:N:M Sparse SSM Projections with S-STE Training

## Overview

Tests whether V:N:M hierarchical structured sparsity (60-75%) applied to projection matrices
in a gated SSM (Mamba-2 style) can maintain quality while enabling real hardware speedup.

**Key Insight**: Projection matrices (W_Q, W_K, W_V, W_gate, W_O) constitute ~75% of layer FLOPs.
VNM sparsity at 75% achieves 1.5-1.7x measured speedup on Sparse Tensor Cores, unlike 2:4's
marginal 1.1-1.3x. S-STE enables training sparse from scratch without mask oscillation.

## Setup

```bash
cd code/031
pip install -e .
```

## Run

```bash
# Run all configurations (Dense, 2:4, V:2:6, V:2:8, Iso-param)
python train.py --config config.yaml

# Run single config
python train.py --config config.yaml --single dense
python train.py --config config.yaml --single v2_8

# With GPU
python train.py --config config.yaml --device cuda
```

## Task

**MQAR (Multi-Query Associative Recall)**: 4 KV pairs, seq_len=64, vocab=16.
Tests whether VNM-sparse projections can accurately project queries and keys for retrieval.

## Configurations Tested

| Config | VNM M | Sparsity | Description |
|--------|-------|----------|-------------|
| Dense | 0 | 0% | Quality upper bound |
| 2:4 | 4 | 50% | Standard structured sparsity |
| V:2:6 | 6 | 67% | Intermediate VNM |
| V:2:8 | 8 | 75% | Target VNM sparsity |
| Iso-param | 0 | 0% | Smaller dense model matching VNM params |

## Success Criteria

- Dense > 95% accuracy
- 2:4 > 90% accuracy
- VNM 75% > 80% accuracy
- Iso-param < 70% accuracy (sparse > small-dense)
- S-STE mask flip rate converges within training

## Files

- `models/vnm_sparse_linear.py` - VNM sparse linear layer with S-STE
- `models/gated_ssm.py` - Gated SSM model (Mamba-2 style)
- `data/generate.py` - MQAR synthetic data generator
- `train.py` - Training script with all baselines
- `config.yaml` - Experiment configuration

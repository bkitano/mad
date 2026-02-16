# MVE 005: Segmented-HSS Linear Attention

## Overview

Tests whether HSS (Hierarchically Semi-Separable) matrix structure can efficiently
represent the state matrix in linear attention while maintaining accuracy on
hierarchical tasks.

**Proposal**: `proposals/005-segmented-hss-linear-attention.md`

## Architecture

- **HSS Linear Attention**: Stores state S_t in HSS form (O(r*d*log(d)) vs O(d^2) for dense)
- **Dense Linear Attention**: Baseline with full d x d state matrix
- **Task**: Hierarchical copying - recall tokens at different delays based on hierarchy level

## Key Parameters (MVE)

- d_model = 64, d_head = 64, r = 8
- ~10K parameters per model
- 5K synthetic sequences

## Success Criteria

1. **Accuracy ≥ 90%** on hierarchical copying (test set)
2. **Hierarchical state structure**: Off-diagonal blocks are low-rank
3. **Memory ratio < 0.2**: HSS uses less than 20% of dense memory

## Setup & Run

### On Modal (recommended)
```bash
modal run --detach modal_config.py --config config.yaml
```

### Locally (for debugging only)
```bash
pip install -e .
python train.py --config config.yaml
```

## Cost Estimate

- T4 GPU: ~5 minutes = ~$0.04
- Total estimated cost: < $0.10

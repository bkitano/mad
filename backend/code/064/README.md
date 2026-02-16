# MVE 064: Residual KDA with Channel-Wise Auxiliary Decay

## Overview

Tests whether augmenting KDA (Kimi Delta Attention) with a residual error-correcting
auxiliary state using **channel-wise decay** (RKDA) improves associative recall over:
1. Standard KDA (no residual)
2. KDA + scalar-decay residual (RDN-style)

## Key Idea

KDA already has per-channel decay for its primary state. The auxiliary residual state
accumulates prediction errors `r_t = v_t - S_{t-1}^T k_t` and provides a second-order
correction to the output. Channel-wise decay in the residual state allows **per-feature
forgetting** of stale residuals, which is impossible with scalar decay.

## Architecture

- **Model**: 2-layer KDA, d=128, d_k=d_v=64, n=2 heads (~80K-120K params)
- **Task**: MQAR with 16 key-value pairs at length 256
- **Variants**: KDA, KDA+scalar-residual, RKDA (channel-wise residual)

## Success Criteria

1. RKDA achieves >= 5% absolute accuracy improvement over KDA on MQAR
2. RKDA outperforms KDA + scalar-residual by >= 2%
3. Training convergence is stable (no NaN, no divergence)

## Running

### On Modal (recommended)
```bash
uv run modal run --detach modal_config.py --config config.yaml
```

### Locally (for debugging only)
```bash
python train.py --config config.yaml --no-wandb
```

## Files

- `models/rkda.py` - KDA/RKDA model implementation (all three variants)
- `data/generate.py` - MQAR dataset generator
- `train.py` - Training and evaluation script
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment configuration

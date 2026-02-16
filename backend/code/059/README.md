# MVE 059: Second-Order KDA (SO-KDA)

## Overview

Tests whether augmenting KDA's delta rule removal with HLA's second-order key metric
improves associative recall accuracy.

**Key innovation**: Replace KDA's rank-1 removal `beta_t * k_t * k_t^T` with an adapted
removal using the running key covariance: `beta_t * k_tilde_t * k_t^T` where
`k_tilde = M_t @ k_t / ||M_t @ k_t||` and `M_t = gamma_M * M_{t-1} + k_t @ k_t^T`.

## Variants

| Variant | Description | Expected MQAR Acc |
|---------|-------------|-------------------|
| GLA | Gated Linear Attention (no delta rule) | ~70% |
| KDA | Kimi Delta Attention (standard delta rule) | ~85% |
| SO-KDA | Second-Order KDA (adapted delta rule) | >95% |

## Setup & Run

### Local (for testing)
```bash
cd code/059
pip install -e .
python train.py --config config.yaml --device cpu --no-wandb
```

### Modal (for real run)
```bash
cd code/059
uv run modal run --detach modal_config.py --config config.yaml
```

## Files

- `models/gla.py` - GLA baseline (diagonal gating, no delta rule)
- `models/kda.py` - KDA baseline (delta rule with k_t removal)
- `models/so_kda.py` - SO-KDA (delta rule with M_t-adapted removal)
- `data/generate.py` - MQAR dataset generator
- `train.py` - Training script with wandb logging
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment config

## Success Criteria

- SO-KDA > 95% MQAR accuracy at T=512 with 16 pairs
- SO-KDA > KDA > GLA ordering maintained
- SO-KDA time overhead < 2x KDA
- Length generalization test at T=1024

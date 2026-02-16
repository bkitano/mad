# MVE 054: SageAttention2-Style INT4 Smoothing for Chunkwise Linear RNN

## Overview

Tests whether SageAttention2's per-thread INT4 quantization with Q+K smoothing
can accurately quantize the QK^T matmul in GLA chunkwise linear attention.

**Proposal**: `proposals/054-sage-int4-smoothing-chunkwise-linear-rnn.md`

## What This MVE Tests

1. **Cosine similarity**: Does INT4+smoothing preserve QK^T accuracy (> 99%)?
2. **Smoothing impact**: Does smoothing significantly improve INT4 accuracy vs raw INT4?
3. **Training quality**: Does INT4+FP8 quantization during training preserve model quality?

## Architecture

- 2-layer GLA, d=256, dk=128, dv=256, 2 heads (~1M params)
- Chunkwise forward with C=128, c=16
- Copying task (seq len 66, vocab 19)

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| INT4 cosine similarity with smoothing | > 99% | TBD |
| INT4 cosine similarity without smoothing | < 85% | TBD |
| Copying task loss (INT4+smooth) within 5% of BF16 | < 1.05x | TBD |
| Smoothing provides significant improvement | > 5% cosine gain | TBD |

## Running

### On Modal (recommended)
```bash
uv run modal run --detach modal_config.py --config config.yaml
```

### Locally (for debugging only)
```bash
uv run python train.py --config config.yaml
```

## Files

- `models/gla.py` — GLA model with chunkwise forward pass
- `models/quantization.py` — INT4/FP8 quantization with smoothing
- `benchmark.py` — Cosine similarity microbenchmark
- `train.py` — Main training script (benchmark + copying task)
- `config.yaml` — Experiment configuration
- `modal_config.py` — Modal deployment

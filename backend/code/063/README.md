# MVE 063: MFA-Style Shared Latent Projections for Linear RNN Training

## Overview

Tests whether MFA-style shared latent projections can replace independent per-head Q/K/V
projections in GLA (Gated Linear Attention) without degrading associative recall quality.

## Architecture

**Standard GLA** (baseline):
- Independent W_q, W_k, W_v per head
- n=2 heads, d_k=d_v=32
- State S in R^{32 x 32} per head

**MFA-GLA** (experiment):
- Shared down-projections S_q, S_k, S_v in R^{128 x 64}
- Per-head Q_c in R^{64 x 64} (query rotation in latent space)
- Per-head V_c in R^{64 x 32} (value projection)
- Keys shared across heads (latent MQA)
- m=4 heads (2x more than standard), C=64 latent dim
- State S in R^{64 x 32} per head

## Task

Multi-Query Associative Recall (MQAR): 8 KV pairs, seq_len=128, vocab=64

## Success Criteria

1. MFA-GLA >= 90% MQAR accuracy
2. Forward pass time <= 1.2x baseline
3. Convergence within 1.5x baseline epochs

## Running

```bash
# Via Modal (recommended)
uv run modal run --detach modal_config.py --config config.yaml

# Locally (for testing)
python train.py --config config.yaml --no-wandb
```

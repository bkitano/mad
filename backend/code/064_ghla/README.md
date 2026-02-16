# MVE 064: Gated Second-Order Linear Attention (GHLA)

## Proposal
Tests whether adding data-dependent diagonal gating to Higher-Order Linear Attention's (HLA) second-moment key metric improves quality over first-order GLA and ungated HLA on associative recall.

## Architecture
- **GHLA**: S_t^K = Diag(alpha_t^K) S_{t-1}^K Diag(alpha_t^K) + k_t k_t^T (gated second-order)
- **GLA baseline**: S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T (first-order)
- **HLA baseline**: S_t^K = S_{t-1}^K + k_t k_t^T (ungated second-order)
- **HLA-decay baseline**: S_t^K = gamma * S_{t-1}^K + k_t k_t^T (fixed decay)

## Task
Multi-Query Associative Recall (MQAR): Store 8 key-value pairs, then retrieve values given queries.

## Setup
```bash
# Install dependencies
pip install -e .

# Run locally (for testing)
python train.py --config config.yaml --no-wandb

# Run on Modal (production)
uv run modal run --detach modal_config.py --config config.yaml
```

## Success Criteria
1. GHLA > 90% accuracy at 8 associations where GLA < 70%
2. GHLA > HLA by > 5% accuracy (demonstrating gating value)
3. GHLA >= HLA-decay (data-dependent gating at least as good as fixed)

## Model Specs (from proposal)
- 2 layers, d_model=64, d_k=16, d_v=32, 2 heads
- ~100K params per variant
- 10K synthetic sequences, length 128

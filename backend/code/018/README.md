# MVE 018: Hutchinson Trace-Guided Adaptive Rank for DPLR SSMs

## Overview

Tests whether using cheap Hutchinson trace estimates to dynamically allocate rank across DPLR SSM layers can improve parameter efficiency.

**Hypothesis**: Different layers need different ranks. By measuring each layer's low-rank importance via power-series log-det and reallocating a fixed rank budget proportionally, we can match high-rank quality with fewer parameters.

## Architecture

- **Model**: 4-layer DPLR SSM (A = Λ + PQ*)
- **Task**: Sequential CIFAR-10 (sCIFAR-10), length-1024 sequences
- **State dim**: n=32, **Hidden dim**: d=64, **Max rank**: r=8
- **~80K parameters**

## Procedure

1. **Phase 1 (Warmup)**: Train all layers at r=8 for 2K steps
2. **Phase 2 (Measurement)**: Compute importance scores via power-series log-det
3. **Phase 3 (Pruning)**: Allocate ranks proportional to importance (budget=16), SVD truncate
4. **Phase 4 (Fine-tuning)**: Train with adapted ranks for 2K more steps

## Baselines

- **Fixed r=4** (budget 16): Same total low-rank parameters as adaptive
- **Fixed r=8** (budget 32): Upper bound on quality

## Success Criteria

1. Importance scores are non-uniform: max/min > 2.0
2. Adaptive (budget 16) ≥ 95% of fixed r=8 (budget 32) accuracy
3. Adaptive (budget 16) outperforms fixed r=4 (budget 16) by > 1%

## Setup & Run

```bash
# Install dependencies
cd code/018
uv sync

# Run on Modal (recommended)
uv run modal run --detach modal_config.py --config config.yaml

# Run locally (not recommended, slow on CPU)
uv run python train.py --config config.yaml
```

## Files

- `models/dplr_ssm.py` — DPLR SSM model with importance scoring and rank truncation
- `train.py` — Training script with 4-phase procedure and baselines
- `config.yaml` — Experiment configuration
- `modal_config.py` — Modal GPU deployment
- `pyproject.toml` — Dependencies

## Key Equations

**DPLR state matrix**: A = Λ + PQ* (diagonal + low-rank)

**Importance score** (power-series log-det):
```
I(ℓ) = E_ω[|Σ_{k=1}^4 (-1)^{k+1}/k · tr((Q*(iωI - Λ)^{-1}P)^k)|]
```

**Rank allocation**: r_ℓ = round(I(ℓ) / Σ_j I(j) · R_total)

**SVD truncation**: PQ* = UΣV* → P_new = U_{:r}Σ_{:r}^{1/2}, Q_new = V_{:r}Σ_{:r}^{1/2}

# Experiment 042: Contraction-Ordered Multi-Operand Chunkwise GLA Fusion

## Overview

This experiment benchmarks two contraction orderings for the GLA (Gated Linear Attention) intra-chunk computation:

- **Path 1 (Standard)**: `(Q @ K^T * M) @ V + Q @ h` — left-to-right evaluation
- **Path 2 (Right-associated)**: `Q @ cumsum(K^T V) - rank_r_correction + Q @ h` — right-associated with low-rank mask approximation

## Setup

```bash
cd code/042
uv pip install -e .
```

## Run on Modal (Required)

```bash
uv run modal run --detach modal_config.py --config config.yaml
```

## Run locally (for debugging only)

```bash
uv run python train.py --config config.yaml
```

## Configuration Sweep

- Chunk sizes (C): 32, 64, 128
- Head dimensions (d): 64, 128, 256
- Approximation ranks (r): 1, 4, 8, 16, C

## Success Criteria

1. Path 2 is >= 10% faster than Path 1 for at least one realistic configuration (e.g., C=64, d=128, r<=8)
2. Numerical agreement within epsilon < 1e-3 relative error (BF16)

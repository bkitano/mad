# MVE 043: Newton-Schulz Orthogonal State Projection for DeltaNet

## Hypothesis

Replacing the sequential UT transform forward substitution in DeltaNet's chunkwise training
with Newton-Schulz polar orthogonalization achieves equivalent model quality while using
only pure tensor-core GEMMs (bfloat16-safe, doubly-exponential convergence).

## Setup

- **Model**: 2-layer DeltaNet, d=32, 2 heads, chunk_size=32, ~80K params
- **Task**: S3 permutation composition (6-element symmetric group)
- **Data**: 5K sequences of length up to 20
- **Variants**: UT transform (baseline) vs Newton-Schulz q=2 (proposed)

## Success Criteria

1. NS achieves >90% S3 accuracy (matching UT within 3%)
2. Orthogonality error ||I - X_2 X_2^T||_F < 1e-3
3. Per-chunk kernel time: NS <= 0.8x UT
4. Stable training (no NaN/Inf) for all epochs

## Run

### Local (CPU, for testing)
```bash
cd code/043
uv run python -m train.run_config --config config.yaml
```

### Modal (GPU)
```bash
cd code/043
uv run modal run --detach modal_config.py --config config.yaml
```

## Files

```
code/043/
├── README.md              # This file
├── pyproject.toml         # Dependencies
├── config.yaml            # Experiment config
├── modal_config.py        # Modal GPU deployment
├── models/
│   ├── __init__.py
│   └── deltanet.py        # UT + NS DeltaNet implementations
├── tasks/
│   └── s3/
│       ├── __init__.py
│       ├── tokens.py      # S3 symmetric group token system
│       └── dataset.py     # S3 curriculum dataset
└── train/
    ├── __init__.py
    ├── train.py           # Training/evaluation functions
    └── run_config.py      # Config-based training entry point
```

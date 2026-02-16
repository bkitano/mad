# MVE 001: Column-Sparse Negative-Eigenvalue DeltaNet

Minimum Viable Experiment for testing the CS-NEG-DeltaNet hypothesis.

## Hypothesis

Combining column-sparse transition matrices (PD-SSM) with the negative eigenvalue extension
in a DeltaNet-style architecture will achieve strictly greater expressivity than either
technique alone, enabling simulation of automata over non-solvable groups with sign-flipping
dynamics.

## Task: D4 Dihedral Group State Tracking

The D4 dihedral group (symmetries of a square) has 8 elements and requires **both**:
1. **Permutation routing** (rotations permute corners)
2. **Sign-flipping dynamics** (reflections change orientation)

This makes it the minimal test case for CS-NEG-DeltaNet.

### Group Structure

- **Elements**: {e, r, r², r³, s, sr, sr², sr³}
- **Generators**: r (90° rotation), s (reflection)
- **Relations**: r⁴ = e, s² = e, srs = r⁻¹

### Task Format

- **Input**: [BOS, g₁, g₂, ..., gₖ, EOS, PAD, ...]
- **Output (scan)**: [IGNORE, p₁, p₂, ..., pₖ, IGNORE, ...] where pᵢ = g₁ · g₂ · ... · gᵢ

## Models

| Model | β Range | Permutation | Expected Accuracy |
|-------|---------|-------------|-------------------|
| Standard DeltaNet | (0, 1) | No | ~50% |
| NEG-DeltaNet | (0, 2) | No | ~60-70% |
| CS-DeltaNet | (0, 1) | Yes | ~60-70% |
| **CS-NEG-DeltaNet** | (0, 2) | Yes | **>90%** |

## Setup

```bash
cd code/001
uv sync
```

## Running Experiments

### Local (CPU/MPS)

```bash
# Run standard DeltaNet baseline
uv run python -m train.launch --config configs/standard_deltanet.yaml --local

# Run CS-NEG-DeltaNet (full model)
uv run python -m train.launch --config configs/cs_neg_deltanet.yaml --local
```

### Modal (GPU)

```bash
# Run all 4 variants in parallel
uv run python -m train.launch --config configs/standard_deltanet.yaml
uv run python -m train.launch --config configs/neg_deltanet.yaml
uv run python -m train.launch --config configs/cs_deltanet.yaml
uv run python -m train.launch --config configs/cs_neg_deltanet.yaml
```

## Success Criteria

The idea works if:
- **CS-NEG-DeltaNet** achieves **>90% accuracy** on D4 state tracking
- Both ablations (**NEG-only** and **CS-only**) achieve **<75% accuracy**

## Failure Criteria

- **Kill the idea if**: CS-NEG-DeltaNet performs no better than the best single-trick baseline
- **Kill the idea if**: Training is unstable and doesn't converge within 1000 steps
- **Pause and investigate if**: CS-NEG works but so does one of the ablations

## Files

```
001/
├── configs/               # YAML experiment configs
│   ├── standard_deltanet.yaml
│   ├── neg_deltanet.yaml
│   ├── cs_deltanet.yaml
│   └── cs_neg_deltanet.yaml
├── tasks/
│   └── d4/               # D4 dihedral group task
│       ├── tokens.py     # Token system (8 elements)
│       └── dataset.py    # Curriculum dataset
├── models/
│   ├── deltanet.py       # Standard/NEG DeltaNet
│   └── cs_deltanet.py    # Column-Sparse DeltaNet
├── train/
│   ├── train.py          # Core training functions
│   ├── run_config.py     # Config-based training
│   ├── modal_config.py   # Modal GPU deployment
│   └── launch.py         # CLI launcher
└── docs/
    └── README.md         # This file
```

## Related

- **Proposal**: `proposals/001-column-sparse-negative-eigenvalue-deltanet.md`
- **Based on**: column-sparse-transition-matrices, negative-eigenvalue-extension

# MVE 002: SSD-DeltaNet Block Decomposition Benchmark

## Overview

Tests whether SSD-style block decomposition provides measurable speedup for DeltaNet's WY representation by restructuring computation into matmul-heavy operations.

**Proposal**: `proposals/002-ssd-deltanet-wy-hybrid.md`

## What This Tests

- **Naive WY**: Standard sequential delta rule accumulation — O(T*d²) sequential operations
- **Block-SSD**: SSD-style block decomposition — splits into sub-blocks of size Q, computes Q×Q output matrices via matmuls, propagates state between sub-blocks

The core claim: converting scalar operations to matmuls gives speedup even in pure PyTorch, before custom CUDA kernels.

## Success Criteria

| Metric | Target |
|--------|--------|
| Speedup | > 1.3× (block vs naive) |
| Numerical error | ‖y_naive - y_block‖_∞ < 10⁻⁵ |
| Matmul fraction | > 60% of FLOPs in matmul ops |

## Setup

```bash
pip install torch numpy pyyaml
```

## Run

```bash
# Local (CPU)
python train.py --config config.yaml

# On Modal (GPU)
modal run --detach modal_config.py --config config.yaml
```

## Configuration

See `config.yaml`:
- T=512, d=64, C=64, Q=16 (from proposal MVE section)
- Benchmarks across T={128, 256, 512, 1024} and d={32, 64, 128}

## Architecture

```
code/002/
├── README.md
├── config.yaml
├── pyproject.toml
├── modal_config.py
├── train.py              # Benchmark script
└── models/
    ├── __init__.py
    ├── naive_wy.py       # Naive sequential WY accumulation
    └── block_ssd.py      # SSD-style block decomposition
```

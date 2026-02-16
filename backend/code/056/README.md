# MVE 056: FlashMask Tile-Skip for Chunkwise Linear RNN

## Overview

This experiment benchmarks the FlashMask tile-skip optimization for GLA (Gated Linear Attention) intra-chunk computation. The key idea is to skip fully-masked sub-chunk tiles when processing document-packed sequences, avoiding wasted HBM loads and compute.

## Setup

```bash
# Install dependencies
pip install -e .

# Or run via Modal (recommended)
modal run --detach modal_config.py --config config.yaml
```

## Structure

```
code/056/
├── README.md              # This file
├── pyproject.toml         # Dependencies
├── config.yaml            # Experiment config
├── modal_config.py        # Modal deployment
├── train.py               # Main benchmark script
└── models/
    ├── __init__.py
    ├── gla_intra_chunk.py # GLA kernel implementations (baseline + tile-skip)
    └── data_generator.py  # Synthetic document-packed data generator
```

## What It Tests

1. **Numerical correctness**: Triton kernels match PyTorch reference (< 1e-3 relative error)
2. **Tile skip analysis**: Counts skippable tiles for varying document lengths
3. **Kernel throughput**: Measures speedup of tile-skip vs baseline
4. **Tile classification overhead**: Ensures skip-check adds < 2% overhead
5. **Mask memory**: Verifies O(T) memory for column-sparse representation

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Speedup at avg_doc_len=64 | > 1.4x |
| Speedup at avg_doc_len=16 | > 1.8x |
| Tile classification overhead | < 2% |
| Numerical correctness | < 1e-3 relative error |
| Mask memory | O(T) |

## References

- Proposal: proposals/056-flashmask-tile-skip-chunkwise-linear-rnn.md
- FlashMask (Wang et al., ICLR 2025): Column-sparse mask for FlashAttention
- GLA (Yang et al., ICML 2024): Chunkwise linear attention with secondary chunking

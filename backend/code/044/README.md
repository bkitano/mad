# Experiment 044: MatMulScan Inter-Chunk State Scan Benchmark

## Overview

Microbenchmark comparing inter-chunk prefix scan implementations for chunkwise linear RNNs (GLA, Mamba-2, DeltaNet).

**Hypothesis**: Reformulating the inter-chunk prefix scan as batched matrix multiplications against constant lower-triangular matrices (MatMulScan) can achieve 1.2-1.8x speedup over standard Blelloch scans by routing computation through tensor cores.

## Methods Compared

1. **Sequential scan** - Ground truth reference (sequential over G chunks, parallel over state dims)
2. **Blelloch scan** (Triton) - Work-efficient parallel prefix scan baseline
3. **MatMulScan** (Triton) - Proposed: prefix scan via batched matmul with L_s constant matrix
4. **torch.cumsum** - PyTorch native cumulative sum baseline

## Setup

```bash
# Deploy to Modal
cd code/044
uv run modal run --detach modal_config.py --config config.yaml
```

## Configuration

See `config.yaml` for benchmark parameters:
- `G_values`: [64, 128, 256] (number of chunks)
- `radix_values`: [4, 8] (MatMulScan radix)
- `n`: 16 (state dimension)
- `d_v`: 64 (value dimension)

## Success Criteria

1. MatMulScan (s=4) achieves >= 1.2x throughput over Blelloch for G=128
2. MatMulScan achieves >= 1.3x for G=256
3. Numerical accuracy: max abs error < 1e-3 (bf16) or < 1e-5 (fp32)
4. All Triton kernels compile and run without errors

## Files

- `models/scans.py` - All scan implementations (sequential, Blelloch, MatMulScan)
- `train.py` - Benchmarking script with wandb logging
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment configuration

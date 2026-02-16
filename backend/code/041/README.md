# Experiment 041: EVT Joint Forward-Backward Graph Partitioning for Linear RNN Training

## Overview

This MVE tests whether fusing forward and backward GEMM chains in chunkwise linear RNN (GLA) training can reduce HBM traffic and improve wall-clock throughput.

**Core idea**: In standard training, the forward pass computes `S_tilde = (Q @ K^T) * M` and stores it to HBM for the backward pass. The backward pass re-reads Q, K, V, M, S_tilde from HBM. The fused kernel keeps all these tensors in registers across the forward→backward boundary, eliminating redundant HBM reads/writes.

## Model

- **Architecture**: Tiny GLA (Gated Linear Attention), 1 layer
- **Dimensions**: d=64, n=16, C=32, T=256, vocab=256
- **Parameters**: ~50K
- **Task**: Autoregressive next-token prediction on random sequences

## Success Criteria

1. **HBM traffic reduction** >= 25% (theoretical + empirical)
2. **Wall-clock speedup** >= 1.15x for combined fwd+bwd pass
3. **Numerical agreement** with reference implementation (max relative diff < 1e-3)

## Running

### On Modal (recommended)
```bash
cd code/041
uv run modal run --detach modal_config.py --config config.yaml
```

### Locally (testing only)
```bash
cd code/041
uv run python train.py --config config.yaml
```

## Files

- `models/gla_baseline.py` - Baseline GLA with SEPARATE forward/backward Triton kernels
- `models/gla_fused.py` - Fused GLA with JOINT forward+backward Triton kernel
- `train.py` - Benchmarking + training script
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment configuration

## Key Measurements

The script measures:
1. **Numerical correctness**: Compares Triton kernels against PyTorch reference
2. **HBM traffic**: Theoretical estimation + empirical memory measurement
3. **Wall-clock timing**: CUDA events for precise kernel timing
4. **Scaling**: Speedup across different batch sizes
5. **End-to-end training**: Validates model trains correctly

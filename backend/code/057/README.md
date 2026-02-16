# MVE 057: FlashRNN-Style Fused Inter-Chunk State Recurrence

## Overview

Kernel microbenchmark testing whether keeping inter-chunk state in Triton registers
(FlashRNN-style) provides speedup over the standard approach of reading/writing
state from HBM at each chunk boundary.

**Proposal**: `proposals/057-flashrnn-fused-inter-chunk-state-recurrence.md`

## What's Being Tested

The inter-chunk state scan in chunkwise linear RNNs (GLA):

```
h_0 = 0
h_k = gamma_k * h_{k-1} + h_hat_k,  k = 1, ..., G
```

- **Baseline**: Each step reads h_{k-1} from HBM, computes, writes h_k to HBM
- **Proposed**: State stays in Triton registers; only gamma_k and h_hat_k are read from HBM

## Setup & Run

### Deploy to Modal (recommended)
```bash
cd code/057
uv run modal run --detach modal_config.py --config config.yaml
```

### Configuration
Edit `config.yaml` to change benchmark parameters:
- `batch_size`, `num_heads`, `dk`, `dv`: Problem dimensions
- `G_values`: List of chunk counts to test
- `warmup`, `repeats`: Benchmark timing parameters

## Success Criteria

1. Inter-chunk scan kernel time decreases by > 20% vs baseline
2. Results are bit-exact with the baseline
3. Consistent across G in {32, 64, 128}

## Files

- `models/flashrnn_scan.py` - Triton kernels (baseline + proposed) and PyTorch reference
- `train.py` - Benchmark script with timing and correctness checks
- `config.yaml` - Benchmark configuration
- `modal_config.py` - Modal GPU deployment

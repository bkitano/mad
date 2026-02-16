# Experiment 055: WY-All-Scan Sequence Parallelism for Gated DeltaNet

## Overview

Microbenchmark comparing three multi-GPU communication primitives for sequence parallelism:

1. **WY-All-Scan** (ours): P2P pipeline sending WY-factored state transitions
2. **ZeCO-GLA All-Scan**: P2P pipeline for diagonal-only transitions (baseline ceiling)
3. **LASP-2 AllGather**: AllGather of full state + factors from all P devices

## Key Idea

Gated DeltaNet's state transition involves Householder reflections (non-diagonal),
which can be represented in WY form. Instead of transmitting the full d_k x d_k
transition matrix, we transmit the WY factors (W, K), achieving P-independent
communication volume.

## Setup

```bash
# Install dependencies
uv pip install -e .

# Deploy on Modal (8x A100)
modal run --detach modal_config.py --config config.yaml
```

## Success Criteria

1. WY-All-Scan latency at P=8 < 2x ZeCO-GLA latency
2. WY-All-Scan latency at P=8 < 0.5x LASP-2 latency
3. Numerical error < 1e-3 (BF16 precision)
4. WY correction matmul < 0.5ms per pipeline stage

## Files

- `benchmark.py` - Main benchmark harness (run via torchrun)
- `models/wy_allscan.py` - WY-factored All-Scan implementation
- `models/lasp2_allgather.py` - LASP-2 AllGather baseline
- `models/zeco_gla_allscan.py` - ZeCO-GLA All-Scan reference
- `models/sequential_scan.py` - Single-device sequential scan (ground truth)
- `modal_config.py` - Modal deployment configuration
- `config.yaml` - Experiment configuration

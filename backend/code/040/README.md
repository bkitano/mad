# Experiment 040: Persistent Megakernel Fusion MVE

## Overview

Tests whether fusing 3 operations (linear projection + gated scan + SiLU output gating) into a single Triton kernel provides meaningful throughput gains by eliminating intermediate HBM round-trips.

From proposal: `proposals/040-persistent-megakernel-linear-rnn-layer.md`

## Architecture

**Baseline (3 kernels):**
1. `V = x @ W_V` (GEMM: [B,T,d] @ [d,d_v] -> [B,T,d_v])
2. `s_t = gamma_t * s_{t-1} + v_t` (gated scan over T)
3. `out = SiLU(gate) * scan_output` (elementwise)

**Fused (1 kernel):**
- Same 3 operations in a single kernel
- V and scan state stay in registers (never written to HBM)
- Only x, W_V, gamma, gate read from HBM; only out written to HBM

## Parameters

- B=4, T=2048, d=256, d_v=64 (single head)
- Forward-pass only
- FP16 precision

## Success Criteria

1. Fused kernel >1.3x throughput vs 3-kernel baseline
2. Numerical correctness within BF16 tolerance
3. (Optional) DRAM traffic reduced >2x (requires ncu profiling)

## Setup & Run

### Local (requires GPU)
```bash
cd code/040
pip install -e .
python train.py --config config.yaml
```

### Modal (recommended)
```bash
cd code/040
uv run modal run --detach modal_config.py --config config.yaml
```

## Files

- `models/baseline_kernels.py` - 3 separate Triton kernels (baseline)
- `models/fused_megakernel.py` - Single fused Triton kernel
- `train.py` - Benchmark script with wandb logging
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal GPU deployment

# MVE 063: DyT-Fused Normalization-Free Chunkwise Linear RNN

## Overview

Tests whether Dynamic Tanh (DyT) can replace RMSNorm in GLA (Gated Linear Attention)
without quality degradation. This is the prerequisite for the full kernel fusion
experiment that could yield 5-15% throughput improvements.

## Variants Tested

| # | Variant | Norm Type | Placement |
|---|---------|-----------|-----------|
| 1 | preln_rmsnorm | RMSNorm | Pre-LN (baseline) |
| 2 | preln_dyt | DyT | Pre-LN |
| 3 | periln_rmsnorm | RMSNorm | Peri-LN |
| 4 | periln_dyt | DyT | Peri-LN |

## Architecture

- 2-layer GLA, d=64, d_k=32, d_v=64, 2 heads
- ~156K parameters per variant
- Synthetic LM task (256-token vocab, 128-length sequences)

## Success Criteria

1. DyT variants match RMSNorm within 0.5 perplexity points
2. Peri-LN DyT has best stability (lowest gradient norm variance)
3. No NaN/Inf in any variant
4. DyT activation distributions approximate RMSNorm outputs

## Running

```bash
cd code/063_dyt
uv run modal run --detach modal_config.py --config config.yaml
```

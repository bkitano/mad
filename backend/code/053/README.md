# Experiment 053: MLA-Inspired Latent State Compression for Linear RNN Inference

## Overview

Tests whether GLA hidden states (d_k x d_v matrices) can be compressed into a low-rank
latent vector c_t (d_c << d_k * d_v) using SVD, and whether readout can be computed via
weight absorption in the compressed space without decompression.

Analogous to DeepSeek-V2's MLA for Transformer KV caches, but applied to linear RNN
recurrent states.

## Setup

```bash
# Install dependencies
uv pip install -e .

# Run locally (for testing)
uv run python train.py --config config.yaml

# Run on Modal (recommended)
modal run --detach modal_config.py --config config.yaml
```

## Architecture

- **Model**: 2-layer GLA, d=128, d_k=64, d_v=128, 2 heads (~200K params)
- **Task**: Language modeling on synthetic bigram data (256 vocab, 128 seq len)
- **Latent dims tested**: d_c in {8, 16, 32, 64}

## Protocol

1. Train small GLA model to convergence on synthetic LM task
2. Collect hidden states S_t on validation data
3. Compute SVD of empirical state covariance to check effective rank
4. Initialize W_down, W_up from top-d_c SVD components
5. Measure readout error: ||q_t^T S_t - q_tilde^T c_t||^2 / ||q_t^T S_t||^2
6. Run compressed inference and measure perplexity

## Success Criteria

- Effective rank of states << min(d_k, d_v) (top-16 SVs capture >90% energy)
- d_c=32 achieves <5% relative readout error
- Compressed inference perplexity within 2 points of full inference
- Per-step latency measurably decreases

## Files

- `models/gla.py` - GLA model implementation with recurrent inference mode
- `models/latent_state.py` - SVD analysis, compression matrices, compressed inference
- `train.py` - Two-phase training + analysis script
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal deployment configuration

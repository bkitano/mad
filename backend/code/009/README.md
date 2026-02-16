# MVE 009: Post-Sigmoid Gating for Linear Attention

**Proposal**: `proposals/009-post-sigmoid-gating-linear-attention.md`

## Hypothesis

Applying post-readout sigmoid gating to cosFormer (linear attention) will break the
low-rank bottleneck in the linear readout path, improving MQAR accuracy by >20
percentage points compared to ungated cosFormer.

## Architecture

```
cosFormer readout (standard):
  o_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)    # linear attention output
  output = o_t @ W_O                           # output projection

cosFormer readout (gated):
  o_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)    # linear attention output
  gate = sigmoid(x_t @ W_g)                    # input-dependent gate
  output = (o_t * gate) @ W_O                  # gated output projection
```

The gate `W_g` is zero-initialized so it starts at `sigmoid(0) = 0.5` (benign scaling).

## Task: Multi-Query Associative Recall (MQAR)

- Store 4 key-value pairs in a sequence
- Query 2 of the keys and predict their values
- Sequence length T=11 (8 KV tokens + 1 SEP + 2 query pairs)
- Vocabulary size: 16 content tokens + 3 special tokens

## Setup

```bash
# Local
pip install torch pyyaml tqdm
python train.py --config config.yaml

# Modal (recommended)
modal run --detach modal_config.py --config config.yaml
```

## Success Criteria

1. Gated cosFormer > 75% accuracy on MQAR with 4 KV pairs at d_k=16
2. Ungated cosFormer < 55% accuracy on the same task
3. Improvement persists across 3 random seeds
4. Training is stable (no NaN/Inf) and wall-clock overhead < 5%

## Model Size

~80K parameters (d_model=64, n_heads=4, d_k=16, 2 layers)

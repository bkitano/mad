# Experiment 003: Oscillatory-Gated Selective SSM (OscGate-SSM)

**Proposal**: `proposals/007-oscillatory-gated-selective-ssm.md`
**Priority**: High
**Estimated Cost**: ~$0.40

## Overview

Tests whether making oscillatory parameters (frequency ω and damping ζ) **input-dependent** while preserving stability-by-construction produces an SSM that achieves selectivity on the selective copying task.

## Key Hypothesis

OscGate-SSM (input-dependent ω(x_t), ζ(x_t)) will solve selective copying (>90% accuracy) while LinOSS (fixed ω, ζ — LTI) will fail (<40% accuracy), proving that input-dependent oscillatory parameters enable content-dependent gating.

## Models

| Model | Type | Selectivity | Stability |
|-------|------|-------------|-----------|
| OscGate-SSM | LTV oscillatory | ✅ Input-dependent ω, ζ | ✅ By construction |
| LinOSS | LTI oscillatory | ❌ Fixed ω, ζ | ✅ By construction |
| DiagonalSSM | LTV diagonal | ✅ Input-dependent α | ⚠️ Sigmoid-bounded |

## Task: Selective Copying

Given input `[c0, c1, c2, ..., SEP, ..., idx, ...]`, predict the content token `c_idx`.

- Sequence length: 32
- Content tokens: 8
- Vocabulary: 16 content tokens + special tokens
- 10K total samples (8K train, 1K val, 1K test)

## Setup

```bash
cd code/003
pip install -e .
```

## Run

```bash
# Train all models
python train.py

# Train specific model
python train.py --model oscgate
python train.py --model linoss
python train.py --model diagonal
```

## Success Criteria

1. OscGate-SSM > 90% test accuracy
2. LinOSS < 40% test accuracy
3. 0 NaN/Inf events during OscGate-SSM training
4. OscGate-SSM forward pass < 3× slower than DiagonalSSM

## Results

See `results.yaml` after training.

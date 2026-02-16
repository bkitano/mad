# Experiment 006: Tropical-Gated SSM (TG-SSM)

**Proposal**: 015-tropical-gated-ssm-scan

## Overview

Tests whether replacing the standard (R, +, x) semiring in SSM state recurrences
with the tropical semiring (R u {-inf}, max, +) produces a model with hard
winner-take-all state dynamics that achieves precise long-range retrieval.

## Key Innovation

- **Tropical recurrence**: l_t = max(a_t + l_{t-1}, b_t) instead of standard SSM
- **Log-semiring annealing**: Train with smooth logsumexp, anneal to hard max
- **Parallel scan compatible**: The tropical semiring satisfies associativity

## Task: Multi-Query Associative Recall (MQAR)

Input: [k1, v1, k2, v2, ..., k8, v8, SEP, q1, q2, ..., q8]
Target: [-, -, ..., -, val(q1), val(q2), ..., val(q8)]

## Setup

```bash
pip install -e .
python train.py --config config.yaml
```

## Success Criteria

1. TG-SSM > 95% MQAR accuracy (linear attention < 80%)
2. Length generalization: < 5% accuracy drop from T=25 to T=73

## Model (~150K params)

- 2-layer TG-SSM, d=64, H=4, d_k=16
- Tropical scan with mu annealing (1 -> 100)
- SwiGLU FFN, RMSNorm, residual connections

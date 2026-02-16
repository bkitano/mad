# MVE 004: Displacement-Rank SSM (DR-SSM)

**Proposal**: 022-displacement-rank-ssm-state-transitions
**Task**: S5 Permutation Composition
**Estimated cost**: ~$0.40 (< 10 min on single GPU)

## Overview

Tests whether Cauchy-like matrices with displacement rank α can serve as
SSM state transitions, providing a tunable capacity knob between diagonal
(α=0, Mamba) and dense (α=n, full mixing) SSMs.

The key scientific question: does displacement rank control expressivity
on the canonical S5 non-abelian state tracking benchmark?

## Setup

```bash
pip install torch numpy tqdm pyyaml
```

## Run

```bash
cd code/004
python train.py --config config.yaml
```

## What It Does

1. Generates 10K S5 permutation composition sequences (length 20)
2. Trains DR-SSM at α ∈ {0, 1, 2, 4, 16} (16 = n = dense baseline)
3. Benchmarks Cauchy matvec vs dense matvec throughput
4. Tests generator truncation error accumulation
5. Reports all success/failure criteria

## Success Criteria

1. **Rank-scaling signal**: α=4 > 85% accuracy, α=1 < 70%, α=0 < 50%
2. **Efficiency**: α=4 trains at > 0.6× dense speed, matches accuracy within 5%
3. **Cauchy throughput**: > 0.3× dense matvec throughput at n=16

## Model Architecture

- 2-layer DR-SSM with pre-norm residual connections
- d_model=64, n=16 (state dim)
- Cauchy-like transition: A_{ij} = d_i δ_{ij} + Σ_k G_{ik} H_{jk} / (s_i - s_j)
- Chebyshev displacement nodes
- MLP classification head (takes last hidden state)

## Files

- `train.py` — Main training + benchmarking script
- `models/dr_ssm.py` — DR-SSM model (Cauchy-like transitions) + Dense SSM baseline
- `data/generate.py` — S5 permutation composition task generator
- `config.yaml` — Experiment configuration

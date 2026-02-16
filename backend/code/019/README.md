# Experiment 019: Capacitance-Coupled Multi-Scale SSM (MC-SSM)

## Hypothesis

A multi-scale SSM where independent blocks at different timescales are coupled through a small k×k capacitance matrix will outperform a monolithic SSM on tasks requiring simultaneous multi-scale pattern detection.

## MVE Task: Nested Periodic Detection

Classify which frequencies are present in a noisy multi-frequency signal:
- f1 = 1/8 (fast, period=8 steps)
- f2 = 1/32 (medium, period=32 steps)
- f3 = 1/128 (slow, period=128 steps)

8 classes (all 2^3 combinations of present/absent frequencies).

## Models

1. **MC-SSM** (proposed): k=4 scale blocks with capacitance coupling
2. **Monolithic SSM** (baseline): Same total state dim, single scale
3. **Uncoupled MS-SSM** (ablation): Multi-scale blocks, no coupling (C=0)

## Setup

```bash
pip install torch numpy tqdm pyyaml
cd code/019
python train.py           # Run all models
python train.py --model mc_ssm     # Run only MC-SSM
python train.py --model monolithic # Run only monolithic
python train.py --model uncoupled  # Run only uncoupled
```

## Success Criteria

1. MC-SSM > 90% accuracy on 3-frequency classification
2. Monolithic baseline < 75% (struggles with slowest frequency)
3. Uncoupled baseline 70-80% (detects scales but can't integrate)
4. Timescale separation > 10x between fastest and slowest

## Architecture Details

- **Model**: 2-layer MC-SSM, d=64, n=32 total state
- **Scales**: k=4, n_per_scale=8
- **Timescales**: Geometric spacing, dt ∈ {0.001, ..., 1.0}
- **Capacitance**: Input-dependent k×k with diagonal dominance constraint
- **Data**: 10K synthetic sequences, length 512
- **Target runtime**: < 10 minutes on single GPU

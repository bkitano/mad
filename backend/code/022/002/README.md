# MVE 002: Oscillatory-DPLR SSM

**Proposal**: [004-oscillatory-dplr-ssm.md](../../proposals/004-oscillatory-dplr-ssm.md)

## Overview

This Minimum Viable Experiment implements and validates the core mechanism of the Oscillatory-DPLR State Space Model:

- **Oscillatory eigenvalues**: λ = -ζω + iω√(1-ζ²) from damped harmonic oscillator
- **DPLR structure**: A = Λ + PQ^T (diagonal + low-rank)
- **Bilinear discretization**: Guarantees stability (|λ_d| ≤ 1)
- **Test task**: Damped oscillation extrapolation (train on T=128, test on T=512)

## Task: Damped Oscillation Extrapolation

- **Input**: Unit impulse at t=0
- **Target**: Damped sinusoid y(t) = A·e^(-ζωt)·sin(ω√(1-ζ²)t + φ)
- **Parameters**: ω ~ U(0.01, 0.1), ζ ~ U(0.2, 0.8), random A and φ
- **Training**: 8K sequences of length 128
- **Testing**: 1K sequences of length 512 (4× extrapolation)

## Success Criteria

From proposal section "Minimum Viable Experiment":

1. ✅ **Training fit**: MSE < 1e-3 on training sequences
2. ✅ **Extrapolation**: MSE < 1e-2 on 4× longer test sequences
3. ✅ **Interpretability**: Learned ω_i cluster near ground-truth range [0.01, 0.1]

## Failure Criteria

- ❌ MSE > 1e-1 on training (cannot fit basic oscillations)
- ❌ Extrapolation MSE > 10× training MSE (complete failure to generalize)
- ❌ Learned ω_i collapse to single value or diverge outside [0.001, 1]

## Model Architecture

```
Oscillatory-DPLR SSM (Tiny)
├── n = 16 (state dimension)
├── r = 2 (low-rank component)
├── d_input = 1
├── d_output = 1
└── ~5K parameters

Oscillatory parameters:
- ω_i (n,): Natural frequencies, initialized log-uniform in [0.01, 0.1]
- ζ_i (n,): Damping ratios, initialized uniform via sigmoid in [0, 1]

DPLR components:
- Λ (n, n): Diagonal with oscillatory eigenvalues
- P (n, r): Low-rank factor 1
- Q (n, r): Low-rank factor 2
- A = Λ + PQ^T

Input/output:
- B (n, d_input): Input projection
- C (d_output, n): Output projection
```

## Setup

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, CPU is fine for MVE)

### Installation

```bash
cd code/002
pip install -e .
```

Or with uv (faster):

```bash
cd code/002
uv pip install -e .
```

## Running the Experiment

### Quick Start

```bash
python train.py
```

This will:
1. Generate 10K synthetic damped oscillation sequences
2. Train Oscillatory-DPLR SSM for 50 epochs (~2-3 minutes on CPU)
3. Evaluate on 4× extrapolation test set
4. Check success/failure criteria
5. Save results to `results.yaml` and model to `best_model.pt`

### Custom Configuration

Edit `config.yaml` to modify:
- Model size (n, r)
- Data parameters (sequence lengths, frequency ranges)
- Training hyperparameters (lr, epochs, batch size)

### Using GPU

```bash
python train.py --device cuda
```

## Expected Results

If the oscillatory-DPLR mechanism works correctly:

- **Training MSE**: ~1e-4 to 1e-3 (fits damped oscillations)
- **Test MSE (extrapolation)**: ~1e-3 to 1e-2 (generalizes to 4× length)
- **Learned ω**: Mean ~0.05, range [0.02, 0.08] (matches ground truth)
- **Learned ζ**: Mean ~0.5, range [0.3, 0.7] (matches ground truth)

## Files

```
code/002/
├── README.md              # This file
├── pyproject.toml         # Dependencies
├── config.yaml            # Experiment configuration
├── train.py               # Main training script
├── models/
│   ├── __init__.py
│   └── osc_dplr_ssm.py    # Oscillatory-DPLR SSM implementation
└── data/
    ├── __init__.py
    └── generate.py        # Synthetic data generation
```

## Results Analysis

After running, check `results.yaml`:

```yaml
success_criteria:
  train_fit: true/false
  extrapolation: true/false
  interpretability: true/false
verdict: PROCEED/DEBUG/INVESTIGATE
train_losses: [...]
val_losses: [...]
test_loss: 0.00xxx
analysis:
  omega_learned_mean: 0.0xxx
  omega_learned_std: 0.0xxx
  ...
```

## Decision Rule

- ✅ **All success criteria met** → Proceed to full LRA experiments
- ❌ **Any failure criterion** → Debug parameterization (check gradients, discretization) before scaling

## Estimated Cost

- **Runtime**: ~5 minutes on CPU, <1 minute on GPU
- **Cost**: $0 (CPU) or ~$0.01 (GPU)
- **Well below**: Budget limit of $10

## Next Steps

If MVE succeeds:
1. Implement full-scale model (n=256, r=16, 6 layers)
2. Run on LRA benchmark tasks
3. Compare to S4D and S5 baselines
4. Analyze initialization robustness and extrapolation

If MVE fails:
1. Check gradient flow through ω, ζ parameters
2. Verify bilinear discretization numerics
3. Inspect learned eigenvalue distribution
4. Ablate low-rank component (test pure diagonal)

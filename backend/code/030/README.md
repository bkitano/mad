# MVE 030: Group-Matrix Displacement Rank SSM (GM-DR-SSM)

## Proposal

From `proposals/030-group-matrix-displacement-rank-ssm.md`

**Hypothesis**: Parameterizing SSM state transitions as low displacement rank group matrices for B_4 enables non-abelian state tracking with tunable equivariance deviation via displacement rank r.

## Setup

```bash
cd code/030
pip install -e .
```

## Run

### Local
```bash
python train.py --config config.yaml
```

### Modal (recommended)
```bash
modal run --detach modal_config.py --config config.yaml
```

## Architecture

- **Model**: 1-layer GM-DR-SSM, n=4 (B_4 group), d_model=32, ~5K params
- **Task**: S3 (6-element) and D4 (8-element) state tracking
- **Data**: 5K sequences of length 32, random group generator products

### GM-DR-SSM State Update

```
h_t = A(x_t) h_{t-1} + B x_t

A(x_t) = Σ_g α_g(x_t) B_g  +  Σ_i diag(a_i(x_t)) B_{g_i}
         [group conv kernel]    [displacement perturbation]
```

Where:
- B_g are signed permutation matrices from B_4 = Z_2^4 ⋊ S_4
- α_g(x_t) = softmax(W_alpha @ x_t) are input-dependent kernel weights
- a_i(x_t) = tanh(W_a @ x_t) * ε are perturbation vectors (displacement rank r)
- r controls deviation from exact group equivariance

## Success Criteria

1. GM-DR-SSM (r=2) > 95% accuracy on S3 state tracking
2. GM-DR-SSM (r=2) > 90% accuracy on D4 state tracking
3. Diagonal SSM < 30% on both (near random chance)
4. r=0 underperforms r=2 by >10% (displacement rank adds value)

## Ablation

| Model | r | Expected |
|-------|---|----------|
| GM-DR-SSM | 0 | Exact group conv, limited expressivity |
| GM-DR-SSM | 1 | Minimal perturbation |
| GM-DR-SSM | 2 | Sweet spot (proposal hypothesis) |
| GM-DR-SSM | 4 | Higher rank, may overfit at small scale |
| DiagonalSSM | - | Abelian only, ~random on non-abelian tasks |

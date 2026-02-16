# Experiment 028: Neumann-Cayley Orthogonal SSM (NC-SSM)

## Proposal

028-neumann-cayley-input-dependent-orthogonal-ssm

## Hypothesis

Replacing the exact O(n^3) matrix inversion in the Cayley transform with a
k-term Neumann series approximation enables input-dependent near-orthogonal
state transitions at O(kn^2) per token, achieving near-exact orthogonality
while being parallelizable via chunkwise scan.

## MVE Setup

- **Model**: 2-layer NC-SSM, d=64, n=8, k=4, rho_max=0.3, ~60K params
- **Task**: S5 permutation composition (non-abelian state tracking)
- **Data**: 10K random sequences of S5 generators, seq_len=20
- **Baseline**: Diagonal SSM (should fail at S5 because it's abelian)

## Success Criteria

1. NC-SSM > 80% accuracy on S5 where DiagonalSSM < 30%
2. Orthogonality deviation ||W^T W - I||_F < 0.1 maintained throughout
3. Training loss converges (no NaN/Inf from approximate orthogonality)

## Run

```bash
cd code/028
pip install -e .
python train.py --config config.yaml
```

## Architecture Details

The Cayley transform maps skew-symmetric A to orthogonal W:
  W = (I + A)^{-1} (I - A)

Neumann approximation (k=4, radix-2 binary splitting):
  A2 = A @ A                    [1 GEMM]
  S4 = (I - A) @ (I + A^2)     [1 GEMM]
  W  = S4 @ (I - A)            [1 GEMM]

For rho_max=0.3 and k=4: orthogonality deviation < 2%.

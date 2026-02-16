# MVE 027: Cayley-Circulant Orthogonal SSM (CC-SSM)

## Proposal
See `proposals/027-cayley-circulant-orthogonal-ssm.md`

## Hypothesis
A state transition matrix parameterized as a Cayley transform of a skew-circulant matrix achieves:
1. Exact orthogonality (|lambda| = 1) by construction
2. O(n log n) per-step cost via FFT
3. Superior long-range memory retention vs diagonal SSMs

## Task: Delayed Copy
- Input: k tokens, then T padding steps, then SEP
- Target: reproduce the k tokens after SEP
- Tests long-range memory retention (must carry information across T steps)

## Setup
```bash
pip install -e .
python train.py --config config.yaml
```

## Models
- **CC-SSM**: Cayley-circulant orthogonal SSM (|lambda| = 1)
- **DiagonalSSM**: Standard diagonal SSM baseline (|lambda| < 1)

## Configuration
- d_model=64, state_dim=32, 2 layers, ~80K params
- vocab_size=8, k=5 tokens, delays T={50, 100, 200, 500}
- 5K train, 1K test

## Success Criteria
1. CC-SSM > 99% accuracy at T=500, DiagonalSSM < 80%
2. CC-SSM > 90% at T=200
3. Speed ratio < 10x
4. No NaN/Inf
5. |lambda| = 1 preserved after training

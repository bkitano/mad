# MVE 025: Nystrom Landmark Compression for Chunkwise SSM

## Proposal

[025-nystrom-landmark-chunkwise-ssm](../../proposals/025-nystrom-landmark-chunkwise-ssm.md)

## Hypothesis

In chunkwise-parallel SSM training, inter-chunk state transfer via dense n x n
transition matrices is a memory/compute bottleneck. Nystrom landmark compression
can reduce this from O(n^2) to O(nm) while preserving essential low-rank structure.

## Task

**Delayed copy**: Copy 8 content tokens after a gap of 64 positions (spanning 2
chunk boundaries). This directly exercises inter-chunk state transfer since the
model must carry token information across chunk boundaries via the state recurrence.

## Models

1. **Full Chunkwise SSM**: Standard O(n^2) inter-chunk state transfer (baseline)
2. **Nystrom-compressed SSM**: O(mn) inter-chunk via learned Nystrom projection (m=8, n=32, 4x compression)

## Architecture

- 2-layer chunkwise SSM blocks with SwiGLU FFN
- d_model=64, state_dim=32, chunk_size=32
- Input-dependent transition matrices A_t (near-identity + small perturbation)
- Learned projection P in R^{m x n} for Nystrom compression

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Nystrom copy accuracy | > 90% |
| Full copy accuracy | > 95% |
| Accuracy gap | < 5% |
| Memory (inter-chunk) | O(mn) < O(n^2) |
| Approx error | ||T - T_hat||_F / ||T||_F < 0.1 |

## Setup & Run

### Local
```bash
pip install -e .
python train.py --config config.yaml
```

### Modal (recommended)
```bash
modal run --detach modal_config.py --config config.yaml
```

## Estimated Cost

- GPU: T4
- Runtime: < 10 minutes
- Cost: ~$0.08

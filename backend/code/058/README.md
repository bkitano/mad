# MVE 058: DSM-Fused Linear RNN Projection Chain

## Summary

This experiment benchmarks the input projection chain of a GLA-style linear RNN layer, comparing unfused (7-8 separate kernel launches) vs fused (1 wide GEMM + fused activations) approaches.

**Proposal**: [058-dsm-fused-linear-rnn-projection-chain](../../proposals/058-dsm-fused-linear-rnn-projection-chain.md)

## Hypothesis

Fusing the 5 input projection GEMMs (Q, K, V, gate, alpha) into a single wide GEMM and applying activations (normalize, SiLU, sigmoid) in-kernel eliminates intermediate HBM round-trips, achieving:
- >30% wall-clock speedup for the projection chain
- >40% HBM traffic reduction
- Numerically identical results

## Setup

```
B=8, T=4096, d=2048, d_k=d_v=128, H=16 (bf16)
```

### Variants Benchmarked

1. **Unfused (baseline)**: 5 separate GEMMs + 3 separate activation kernels (8 launches)
2. **Fused**: 1 wide GEMM + split + 3 activation kernels (~4 launches)
3. **Compiled**: torch.compile fused (automatic kernel fusion)

## Running

### On Modal (required)

```bash
cd code/058
uv run modal run --detach modal_config.py --config config.yaml
```

### Local (for debugging only — CPU, no meaningful benchmarks)

```bash
cd code/058
uv run python train.py --config config.yaml
```

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Fused speedup | >30% faster | CUDA event timing |
| HBM traffic reduction | >40% | Theoretical calculation |
| Numerical equivalence | Bit-exact (bf16) | torch.allclose |

## Notes

- The PyTorch-level fusion (single wide GEMM) does NOT achieve full EVT epilogue fusion — activations are still separate kernel launches after the GEMM
- True CUTLASS EVT fusion would apply activations in registers during the GEMM epilogue
- DSM (Distributed Shared Memory) features require H100 and CUTLASS 3.x, not tested here
- This MVE validates the hypothesis that the projection chain benefits from fusion

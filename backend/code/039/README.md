# MVE 039: Warp-Specialized Pingpong Pipelining for Chunkwise Linear RNN Kernels

## Hypothesis

Overlapping sequential element-wise operations (decay masking, state scan) with GEMM operations via software pipelining improves throughput for chunkwise linear RNN kernels.

This MVE tests the core insight from FlashAttention-3's warp-specialized pingpong scheduling, adapted for the chunkwise GLA (Gated Linear Attention) computation.

## Architecture

The chunkwise GLA forward pass has a GEMM-elementwise-GEMM sandwich structure:
1. **GEMM 1**: `S = Q @ K^T` (tensor cores)
2. **Element-wise**: `S = S * D` (decay masking, CUDA cores)
3. **GEMM 2**: `O += S @ V` (tensor cores)

The pipelined version overlaps next tile loading with current tile's decay computation.

## Implementations

| Implementation | Description |
|----------------|-------------|
| PyTorch reference | Pure PyTorch for correctness verification |
| Triton baseline | Sequential: load → QK^T → decay → SV → accumulate |
| Triton pipelined | Double-buffered: pre-loads next tile during current GEMM+decay |

## Success Criteria

1. **Throughput**: Pipelined kernel achieves > 1.2x speedup over baseline
2. **Correctness**: Numerical match within tolerance (max error < 1e-2)
3. **Consistency**: Speedup holds across different sequence lengths

## Setup

```bash
cd code/039
uv sync
```

## Run on Modal (GPU)

```bash
uv run modal run --detach modal_config.py --config config.yaml
```

## Run locally (CPU, for testing only)

```bash
uv run python train.py --config config.yaml
```

## Config

See `config.yaml` for benchmark parameters:
- `B=4, H=32, d=64, d_v=64` (typical 350M model dimensions)
- `chunk_size=64` (inner tile size)
- `T_values=[256, 512, 1024, 2048]` (sequence lengths to test)

## Notes

This is an MVE (Minimum Viable Experiment) testing the concept in Triton.
The full experiment would implement CUTLASS 3.x warp-specialized kernels
with TMA + WGMMA on H100 (Hopper SM90a) for maximum overlap.

If Triton's compiler already optimizes the pipelining automatically,
the baseline and pipelined versions may show similar performance —
this itself is a valid result indicating the compiler handles this optimization.

# Experiment 026 Scaled: Cyclic Reduction for Dense SSM Training

**Proposal**: 026-cyclic-reduction-randmscan-ssm-recurrence
**MVE Results**: 3.88x CPU speedup at T=1024, n=32; 6.01x GEMM reduction
**Status**: Ready for GPU validation

## Hypothesis

Cyclic reduction (CR) achieves O(Tn³) work vs. O(Tn³ log T) for prefix scan when parallelizing dense SSM recurrences (h_t = A_t h_{t-1} + b_t with non-diagonal A_t). On GPUs, this translates to 1.5-2x wall-clock speedup for training DeltaNet/DeltaProduct models due to:
1. Fewer total GEMMs (log T factor savings)
2. Better batched GEMM sizes (more work per kernel launch)
3. Tensor core friendly operations (all bmm-based)

## Experiment Design

### Phase 1: GPU Kernel Benchmark (2 GPU-hours)

Validate CR speedup on A100 with realistic SSM parameters:
- State dimensions: n ∈ {32, 64, 128}
- Sequence lengths: T ∈ {1024, 2048, 4096, 8192}
- Precision: FP16 with tensor cores
- Metrics: Wall-clock time, TFLOP/s, memory bandwidth utilization

**Success criteria:**
- CR achieves ≥1.5x speedup vs. prefix scan at T=4096, n=64
- GPU utilization >80% for both methods (proves we're not bottlenecked by Python overhead)
- Memory usage ≤ prefix scan

### Phase 2: DeltaNet Integration (10 GPU-hours)

Train small DeltaNet models on WikiText-103 with CR-based recurrence:
- Model: 6 layers, 256 hidden dim, 8 heads, 32 state dim per head
- Sequence length: T=2048
- Dataset: WikiText-103 (10K train steps, ~3M tokens)
- Baseline: Same model with prefix scan parallelization

**Success criteria:**
- Training throughput: ≥1.3x tokens/sec vs. baseline
- Final perplexity: Within 2% of baseline (proves correctness)
- Peak memory: ≤ baseline

**Model size:**
- Parameters: ~10M (6×256 hidden, 8 heads, 32 state/head)
- Memory: ~4GB model + 8GB activations at batch=32, T=2048
- Fits comfortably in 1×A100 (80GB)

## Hardware Requirements

- **Phase 1**: 1×A100 (40GB sufficient), 2 hours → 2 A100-hours
- **Phase 2**: 1×A100 (80GB for T=2048 at batch=32), 10 hours → 10 A100-hours
- **Total**: 12 A100-hours ≈ $18-24 at $1.50-2.00/hr

## Implementation Plan

### Kernel Optimization Strategy

The MVE implementation uses PyTorch's `torch.bmm`, which is suboptimal for CR's varying batch sizes per level. For the scaled experiment:

1. **Phase 1a**: Profile existing PyTorch implementation on GPU
   - Identify if speedup transfers from CPU to GPU
   - Measure tensor core utilization with `torch.cuda.profiler`

2. **Phase 1b** (if needed): Optimize critical paths
   - Use `torch.compile()` to JIT-fuse operations
   - Tune CUDA kernel launch parameters
   - Add memory pooling to reduce allocation overhead

3. **Phase 1c** (if needed): Custom CUDA kernel (stretch goal)
   - Persistent kernel with shared memory for CR levels
   - Only implement if PyTorch version is <1.3x speedup

### Integration Strategy

Replace the sequential recurrence in DeltaNet with CR:
```python
# Baseline: Sequential scan (O(T) depth)
for t in range(T):
    h[t] = A[t] @ h[t-1] + b[t]

# Proposed: Cyclic reduction (O(log T) depth, O(T n^3) work)
h = cyclic_reduction(A, b)
```

### Baselines

1. **Sequential scan** (O(Tn²) work, O(T) depth) - lower bound on work
2. **Prefix scan** (O(Tn³ log T) work, O(log T) depth) - current state-of-art
3. **Chunkwise parallel** (quadratic intra-chunk + scan inter-chunk) - Mamba-2 style

## Success Criteria Summary

| Metric | Phase 1 Target | Phase 2 Target |
|--------|---------------|---------------|
| Wall-clock speedup | ≥1.5x @ T=4096 | ≥1.3x training throughput |
| GPU utilization | >80% tensor core | >70% overall |
| Memory usage | ≤100% of baseline | ≤100% of baseline |
| Perplexity | N/A | Within 2% of baseline |
| Total cost | <$5 | <$25 |

## Expected Outcomes

**If successful:**
- CR is GPU-efficient for dense SSM training at T≥2048
- Validates the O(log T) work savings translate to wall-clock speedup
- Opens path for scaling to T=8192+ and n=128+ (larger state spaces)
- Technique applies to: DeltaNet, DeltaProduct, Monarch SSM, OH-DeltaProduct

**If unsuccessful:**
- Memory access pattern (stride 2^l at each level) dominates over FLOP savings
- GPU kernel launch overhead negates batched GEMM efficiency
- Would require custom fused kernel or algorithmic changes (e.g., hierarchical blocking)

## Risk Mitigation

### Risk 1: PyTorch bmm not efficient enough for CR's access pattern
**Mitigation**: Early stopping after 1 GPU-hour if no speedup. Fall back to profiling report explaining bottleneck.

### Risk 2: Training integration introduces bugs
**Mitigation**: Extensive unit tests. Validate h_CR = h_sequential to machine precision before training.

### Risk 3: Cost overrun
**Mitigation**: Use gradient accumulation (batch=8, accum=4 → effective batch=32). Stop after 5K steps if results are clear.

## Files

```
code/026_scaled/
├── README.md              # This file
├── config.yaml            # Experiment hyperparameters
├── train.py               # Main training script (Phase 2)
├── benchmark_kernels.py   # GPU kernel benchmark (Phase 1)
├── models/
│   ├── cyclic_reduction_gpu.py    # Optimized CR for GPU
│   ├── prefix_scan_gpu.py         # Optimized prefix scan baseline
│   ├── deltanet_cr.py             # DeltaNet with CR recurrence
│   └── deltanet_baseline.py       # DeltaNet with prefix scan
├── data/
│   └── wikitext.py        # WikiText-103 data loader
└── scripts/
    ├── phase1_benchmark.sh   # Run Phase 1 GPU benchmarks
    ├── phase2_train.sh       # Run Phase 2 training
    └── analyze_results.py    # Generate comparison plots
```

## Usage

### Phase 1: GPU Kernel Benchmark
```bash
# Single GPU benchmark
python benchmark_kernels.py --device cuda --T 4096 --n 64 --trials 100

# Full sweep
bash scripts/phase1_benchmark.sh
```

### Phase 2: DeltaNet Training
```bash
# Train with cyclic reduction
python train.py --method cr --batch_size 32 --seq_len 2048 --max_steps 10000

# Train baseline
python train.py --method scan --batch_size 32 --seq_len 2048 --max_steps 10000

# Compare results
python scripts/analyze_results.py --exp1 runs/cr --exp2 runs/scan
```

## References

- Proposal 026: Cyclic Reduction with RandMScan for Dense SSM Recurrences
- MVE 008: Cyclic Reduction vs Prefix Scan (CPU validation)
- DeltaNet paper: https://arxiv.org/abs/2102.11174
- Cyclic Reduction for Tridiagonal Systems (Hockney 1965)

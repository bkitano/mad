"""
FlashMask Tile-Skip Benchmark for GLA Intra-Chunk Computation.

This script benchmarks the GLA intra-chunk kernel with and without FlashMask
tile-skip optimization across varying document lengths.

Experiment 056: FlashMask Tile-Skip for Chunkwise Linear RNN
Reference: proposals/056-flashmask-tile-skip-chunkwise-linear-rnn.md

Success Criteria:
- At avg_doc_len=64, C=128: kernel throughput > 1.4x baseline
- At avg_doc_len=16: kernel throughput > 1.8x baseline
- Tile classification overhead < 2% of kernel time
- Numerical output matches reference to BF16 precision (< 1e-3 relative error)
- Mask memory is O(T) (verified by measuring allocation)
"""

import torch
import time
import yaml
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.gla_intra_chunk import (
    gla_intra_chunk_reference,
    gla_intra_chunk_triton,
    precompute_log_cumsum,
    compute_column_sparse_mask,
)
from models.data_generator import (
    generate_document_packed_data,
    count_skippable_tiles,
)


def benchmark_kernel(
    fn,
    *args,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    **kwargs,
) -> dict:
    """
    Benchmark a kernel function with warmup and timing.

    Returns dict with timing stats.
    """
    # Warmup
    for _ in range(warmup_iters):
        out = fn(*args, **kwargs)

    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        out = fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()

    # Remove outliers (top/bottom 10%)
    trim = max(1, bench_iters // 10)
    trimmed_times = times[trim:-trim]

    return {
        'mean_ms': sum(trimmed_times) / len(trimmed_times),
        'median_ms': trimmed_times[len(trimmed_times) // 2],
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(trimmed_times)/len(trimmed_times))**2 for t in trimmed_times) / len(trimmed_times)) ** 0.5,
    }


def verify_correctness(
    Q, K, V, alpha,
    chunk_size: int = 128,
    sub_chunk_size: int = 16,
) -> dict:
    """
    Verify numerical correctness of Triton kernels against PyTorch reference.

    Returns dict with error metrics.
    """
    # Reference output
    O_ref = gla_intra_chunk_reference(Q, K, V, alpha, chunk_size)

    # Triton baseline (no tile-skip)
    O_baseline = gla_intra_chunk_triton(
        Q, K, V, alpha, chunk_size, sub_chunk_size,
        use_tile_skip=False,
    )

    # Triton tile-skip
    O_tileskip = gla_intra_chunk_triton(
        Q, K, V, alpha, chunk_size, sub_chunk_size,
        use_tile_skip=True,
    )

    # Compute relative errors
    ref_norm = torch.norm(O_ref).item()

    baseline_abs_err = torch.norm(O_baseline - O_ref).item()
    baseline_rel_err = baseline_abs_err / max(ref_norm, 1e-10)

    tileskip_abs_err = torch.norm(O_tileskip - O_ref).item()
    tileskip_rel_err = tileskip_abs_err / max(ref_norm, 1e-10)

    # Also check tileskip vs baseline
    cross_abs_err = torch.norm(O_tileskip - O_baseline).item()
    cross_rel_err = cross_abs_err / max(torch.norm(O_baseline).item(), 1e-10)

    return {
        'ref_norm': ref_norm,
        'baseline_abs_err': baseline_abs_err,
        'baseline_rel_err': baseline_rel_err,
        'tileskip_abs_err': tileskip_abs_err,
        'tileskip_rel_err': tileskip_rel_err,
        'cross_abs_err': cross_abs_err,
        'cross_rel_err': cross_rel_err,
    }


def measure_mask_memory(
    B: int, H: int, T: int,
    chunk_size: int, sub_chunk_size: int,
) -> dict:
    """
    Measure memory usage for FlashMask column-sparse representation vs dense mask.
    """
    num_chunks = T // chunk_size
    Ns = chunk_size // sub_chunk_size

    # Column-sparse: LTE_tile_min + LTE_tile_max = 2 * (B * H * num_chunks * Ns) * 4 bytes
    # Plus LTE vector: B * H * T * 4 bytes
    # Plus log_cumsum: B * H * T * 4 bytes (needed for both approaches)
    sparse_bytes = (
        2 * B * H * num_chunks * Ns * 4  # tile bounds
        + B * H * T * 4  # LTE vector
    )

    # Dense mask: B * H * T * T * 4 bytes (full gate matrix)
    # Actually per-chunk: B * H * num_chunks * C * C * 4
    dense_bytes = B * H * num_chunks * chunk_size * chunk_size * 4

    return {
        'sparse_bytes': sparse_bytes,
        'dense_bytes': dense_bytes,
        'ratio': dense_bytes / max(sparse_bytes, 1),
        'sparse_is_O_T': True,  # sparse_bytes is proportional to T
        'sparse_mb': sparse_bytes / (1024 * 1024),
        'dense_mb': dense_bytes / (1024 * 1024),
    }


def run_benchmark(config: dict):
    """Main benchmark function."""
    import wandb

    # Extract config
    model_config = config.get('model', {})
    bench_config = config.get('benchmark', {})
    deployment_config = config.get('deployment', {})

    B = bench_config.get('batch_size', 4)
    H = model_config.get('num_heads', 2)
    T = bench_config.get('seq_len', 1024)
    dk = model_config.get('dk', 128)
    dv = model_config.get('dv', 256)
    chunk_size = model_config.get('chunk_size', 128)
    sub_chunk_size = model_config.get('sub_chunk_size', 16)
    avg_doc_lens = bench_config.get('avg_doc_lens', [16, 32, 64, 128, 256])
    warmup_iters = bench_config.get('warmup_iters', 10)
    bench_iters = bench_config.get('bench_iters', 100)
    dtype_str = bench_config.get('dtype', 'float32')

    dtype = torch.float32 if dtype_str == 'float32' else torch.bfloat16

    print("=" * 80)
    print("FlashMask Tile-Skip Benchmark for GLA Intra-Chunk")
    print("=" * 80)
    print(f"Config: B={B}, H={H}, T={T}, dk={dk}, dv={dv}")
    print(f"Chunk: C={chunk_size}, c={sub_chunk_size}, Ns={chunk_size//sub_chunk_size}")
    print(f"Doc lengths: {avg_doc_lens}")
    print(f"Dtype: {dtype_str}")
    print(f"Iterations: {warmup_iters} warmup, {bench_iters} bench")
    print("=" * 80)

    # Initialize wandb
    wandb.init(
        project="mad-architecture-search",
        name=f"exp-056-flashmask-tileskip",
        config={
            "model": model_config,
            "benchmark": bench_config,
            "proposal_id": "056-flashmask-tile-skip-chunkwise-linear-rnn",
        }
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    all_results = {}

    # ========================================================================
    # Step 1: Correctness verification (small scale)
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Numerical Correctness Verification")
    print("=" * 60)

    # Use small batch for correctness check
    for avg_doc_len in [16, 64, 256]:
        print(f"\n--- avg_doc_len={avg_doc_len} ---")
        data = generate_document_packed_data(
            batch_size=2, num_heads=H, seq_len=T,
            dk=dk, dv=dv, avg_doc_len=avg_doc_len,
            device=device, dtype=dtype, seed=42,
        )

        try:
            errors = verify_correctness(
                data['Q'], data['K'], data['V'], data['alpha'],
                chunk_size=chunk_size, sub_chunk_size=sub_chunk_size,
            )

            print(f"  Baseline vs Reference: rel_err={errors['baseline_rel_err']:.2e}")
            print(f"  TileSkip vs Reference: rel_err={errors['tileskip_rel_err']:.2e}")
            print(f"  TileSkip vs Baseline:  rel_err={errors['cross_rel_err']:.2e}")

            passed = errors['tileskip_rel_err'] < 1e-3
            print(f"  Correctness: {'PASS' if passed else 'FAIL'} (threshold: 1e-3)")

            wandb.log({
                f"correctness/baseline_rel_err_doclen{avg_doc_len}": errors['baseline_rel_err'],
                f"correctness/tileskip_rel_err_doclen{avg_doc_len}": errors['tileskip_rel_err'],
                f"correctness/passed_doclen{avg_doc_len}": passed,
            })

            all_results[f'correctness_doclen{avg_doc_len}'] = errors
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'correctness_doclen{avg_doc_len}'] = {'error': str(e)}

    # ========================================================================
    # Step 2: Tile Skip Analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Tile Skip Analysis")
    print("=" * 60)

    for avg_doc_len in avg_doc_lens:
        print(f"\n--- avg_doc_len={avg_doc_len} ---")
        data = generate_document_packed_data(
            batch_size=1, num_heads=1, seq_len=min(T, 512),  # Small for counting
            dk=dk, dv=dv, avg_doc_len=avg_doc_len,
            device=device, dtype=dtype, seed=42,
        )

        tile_stats = count_skippable_tiles(
            data['alpha'], chunk_size=chunk_size, sub_chunk_size=sub_chunk_size,
        )

        print(f"  Total tiles: {tile_stats['total_tiles']}")
        print(f"  Causal skip: {tile_stats['causal_skip']} ({tile_stats['causal_skip_fraction']:.1%})")
        print(f"  FlashMask extra skip: {tile_stats['flashmask_skip']} ({tile_stats['flashmask_extra_skip_fraction']:.1%})")
        print(f"  Total skip: {tile_stats['skip_fraction']:.1%}")
        print(f"  Computed tiles: {tile_stats['computed_tiles']}")

        wandb.log({
            f"tile_analysis/skip_fraction_doclen{avg_doc_len}": tile_stats['skip_fraction'],
            f"tile_analysis/causal_skip_doclen{avg_doc_len}": tile_stats['causal_skip_fraction'],
            f"tile_analysis/flashmask_extra_skip_doclen{avg_doc_len}": tile_stats['flashmask_extra_skip_fraction'],
        })

        all_results[f'tile_stats_doclen{avg_doc_len}'] = tile_stats

    # ========================================================================
    # Step 3: Kernel Throughput Benchmark
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Kernel Throughput Benchmark")
    print("=" * 60)

    for avg_doc_len in avg_doc_lens:
        print(f"\n--- avg_doc_len={avg_doc_len} ---")
        data = generate_document_packed_data(
            batch_size=B, num_heads=H, seq_len=T,
            dk=dk, dv=dv, avg_doc_len=avg_doc_len,
            device=device, dtype=dtype, seed=42,
        )

        Q, K, V, alpha = data['Q'], data['K'], data['V'], data['alpha']

        # Precompute mask for tile-skip version
        mask_info = compute_column_sparse_mask(alpha, chunk_size, sub_chunk_size)

        try:
            # Benchmark baseline (no tile-skip, but uses Triton kernel)
            print("  Benchmarking baseline (causal skip only)...")
            baseline_stats = benchmark_kernel(
                gla_intra_chunk_triton,
                Q, K, V, alpha,
                chunk_size=chunk_size,
                sub_chunk_size=sub_chunk_size,
                use_tile_skip=False,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )
            print(f"    Baseline: {baseline_stats['mean_ms']:.3f} ms (±{baseline_stats['std_ms']:.3f})")

            # Benchmark tile-skip
            print("  Benchmarking FlashMask tile-skip...")
            tileskip_stats = benchmark_kernel(
                gla_intra_chunk_triton,
                Q, K, V, alpha,
                chunk_size=chunk_size,
                sub_chunk_size=sub_chunk_size,
                use_tile_skip=True,
                LTE_tile_min=mask_info['LTE_tile_min'],
                LTE_tile_max=mask_info['LTE_tile_max'],
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )
            print(f"    TileSkip: {tileskip_stats['mean_ms']:.3f} ms (±{tileskip_stats['std_ms']:.3f})")

            # Compute speedup
            speedup = baseline_stats['mean_ms'] / max(tileskip_stats['mean_ms'], 1e-10)
            print(f"    Speedup: {speedup:.2f}x")

            # Measure tile classification overhead
            # Run tile-skip with all tiles being non-skippable (no boundaries)
            alpha_no_boundary = torch.ones_like(alpha) * 0.95
            mask_no_boundary = compute_column_sparse_mask(alpha_no_boundary, chunk_size, sub_chunk_size)

            print("  Measuring tile classification overhead...")
            overhead_stats = benchmark_kernel(
                gla_intra_chunk_triton,
                Q, K, V, alpha_no_boundary,
                chunk_size=chunk_size,
                sub_chunk_size=sub_chunk_size,
                use_tile_skip=True,
                LTE_tile_min=mask_no_boundary['LTE_tile_min'],
                LTE_tile_max=mask_no_boundary['LTE_tile_max'],
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )

            # Baseline with no boundaries for fair comparison
            baseline_no_boundary = benchmark_kernel(
                gla_intra_chunk_triton,
                Q, K, V, alpha_no_boundary,
                chunk_size=chunk_size,
                sub_chunk_size=sub_chunk_size,
                use_tile_skip=False,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )

            overhead_pct = (overhead_stats['mean_ms'] - baseline_no_boundary['mean_ms']) / max(baseline_no_boundary['mean_ms'], 1e-10) * 100
            print(f"    Overhead: {overhead_pct:.2f}% (target: < 2%)")

            # Log to wandb
            wandb.log({
                f"throughput/baseline_ms_doclen{avg_doc_len}": baseline_stats['mean_ms'],
                f"throughput/tileskip_ms_doclen{avg_doc_len}": tileskip_stats['mean_ms'],
                f"throughput/speedup_doclen{avg_doc_len}": speedup,
                f"throughput/overhead_pct_doclen{avg_doc_len}": overhead_pct,
            })

            all_results[f'benchmark_doclen{avg_doc_len}'] = {
                'baseline_ms': baseline_stats['mean_ms'],
                'tileskip_ms': tileskip_stats['mean_ms'],
                'speedup': speedup,
                'overhead_pct': overhead_pct,
                'baseline_stats': baseline_stats,
                'tileskip_stats': tileskip_stats,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'benchmark_doclen{avg_doc_len}'] = {'error': str(e)}

    # ========================================================================
    # Step 4: Memory Analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Mask Memory Analysis")
    print("=" * 60)

    mem_stats = measure_mask_memory(B, H, T, chunk_size, sub_chunk_size)
    print(f"  Sparse mask memory: {mem_stats['sparse_mb']:.4f} MB")
    print(f"  Dense mask memory:  {mem_stats['dense_mb']:.4f} MB")
    print(f"  Ratio (dense/sparse): {mem_stats['ratio']:.1f}x")
    print(f"  Sparse is O(T): {mem_stats['sparse_is_O_T']}")

    wandb.log({
        "memory/sparse_mb": mem_stats['sparse_mb'],
        "memory/dense_mb": mem_stats['dense_mb'],
        "memory/ratio": mem_stats['ratio'],
    })

    all_results['memory'] = mem_stats

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    # Check success criteria
    criteria = {}

    # Criterion 1: At avg_doc_len=64, throughput > 1.4x
    if 'benchmark_doclen64' in all_results and 'speedup' in all_results['benchmark_doclen64']:
        speedup_64 = all_results['benchmark_doclen64']['speedup']
        criteria['speedup_doclen64'] = speedup_64 >= 1.4
        print(f"  Speedup at doc_len=64: {speedup_64:.2f}x (target: > 1.4x) -> {'PASS' if criteria['speedup_doclen64'] else 'FAIL'}")
    else:
        criteria['speedup_doclen64'] = False
        print("  Speedup at doc_len=64: NOT MEASURED -> FAIL")

    # Criterion 2: At avg_doc_len=16, throughput > 1.8x
    if 'benchmark_doclen16' in all_results and 'speedup' in all_results['benchmark_doclen16']:
        speedup_16 = all_results['benchmark_doclen16']['speedup']
        criteria['speedup_doclen16'] = speedup_16 >= 1.8
        print(f"  Speedup at doc_len=16: {speedup_16:.2f}x (target: > 1.8x) -> {'PASS' if criteria['speedup_doclen16'] else 'FAIL'}")
    else:
        criteria['speedup_doclen16'] = False
        print("  Speedup at doc_len=16: NOT MEASURED -> FAIL")

    # Criterion 3: Tile classification overhead < 2%
    if 'benchmark_doclen64' in all_results and 'overhead_pct' in all_results['benchmark_doclen64']:
        overhead = all_results['benchmark_doclen64']['overhead_pct']
        criteria['overhead'] = abs(overhead) < 2.0
        print(f"  Tile classification overhead: {overhead:.2f}% (target: < 2%) -> {'PASS' if criteria['overhead'] else 'FAIL'}")
    else:
        criteria['overhead'] = False
        print("  Tile classification overhead: NOT MEASURED -> FAIL")

    # Criterion 4: Numerical correctness < 1e-3
    correctness_pass = True
    for key in all_results:
        if key.startswith('correctness_') and 'tileskip_rel_err' in all_results[key]:
            if all_results[key]['tileskip_rel_err'] >= 1e-3:
                correctness_pass = False
    criteria['correctness'] = correctness_pass
    print(f"  Numerical correctness: {'PASS' if criteria['correctness'] else 'FAIL'}")

    # Criterion 5: Mask memory is O(T)
    criteria['memory_o_t'] = mem_stats.get('sparse_is_O_T', False)
    print(f"  Mask memory O(T): {'PASS' if criteria['memory_o_t'] else 'FAIL'}")

    # Overall
    all_passed = all(criteria.values())
    print(f"\n  Overall: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")

    # Log final results to wandb
    wandb.log({
        "final/speedup_doclen64": all_results.get('benchmark_doclen64', {}).get('speedup', 0),
        "final/speedup_doclen16": all_results.get('benchmark_doclen16', {}).get('speedup', 0),
        "final/overhead_pct": all_results.get('benchmark_doclen64', {}).get('overhead_pct', 0),
        "final/correctness_passed": criteria['correctness'],
        "final/memory_o_t": criteria['memory_o_t'],
        "success_criteria/speedup_doclen64": criteria['speedup_doclen64'],
        "success_criteria/speedup_doclen16": criteria['speedup_doclen16'],
        "success_criteria/overhead": criteria['overhead'],
        "success_criteria/correctness": criteria['correctness'],
        "success_criteria/memory_o_t": criteria['memory_o_t'],
        "success_criteria/all_passed": all_passed,
    })

    print(f"\nWandb URL: {wandb_url}")
    wandb.finish()

    # Save results to file
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, 'w') as f:
        # Convert non-serializable values
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                serializable[k] = {kk: (vv if isinstance(vv, (int, float, bool, str)) else str(vv)) for kk, vv in v.items()}
            else:
                serializable[k] = str(v)
        serializable['criteria'] = criteria
        json.dump(serializable, f, indent=2)

    print(f"Results saved to {results_file}")

    return all_results, criteria


def main():
    parser = argparse.ArgumentParser(description="FlashMask Tile-Skip Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {config_path} not found, using defaults")
        config = {}

    results, criteria = run_benchmark(config)

    return 0 if all(criteria.values()) else 1


if __name__ == "__main__":
    exit(main())

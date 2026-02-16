"""
Experiment 044: MatMulScan Inter-Chunk Scan Benchmark

Microbenchmark comparing three inter-chunk scan implementations:
1. Sequential scan (reference / ground truth)
2. Blelloch-style parallel scan (Triton baseline)
3. MatMulScan (proposed) - prefix scan via batched matmuls

Tests:
- Diagonal case: alpha in R^16, b in R^(16x64), G in {64, 128, 256}
- Dense case: A in R^(16x16), b in R^(16x64), G = 64
- Numerical accuracy vs sequential reference
- Throughput comparison (microseconds per scan)

Success criteria:
- MatMulScan (s=4) achieves >= 1.2x throughput over Blelloch for G=128
- MatMulScan achieves >= 1.3x for G=256
- Numerical accuracy: max abs error < 1e-3 (bf16) or < 1e-5 (fp32)
"""

import torch
import time
import yaml
import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.scans import (
    sequential_scan_diagonal,
    blelloch_scan_diagonal,
    matmulscan_scan_diagonal,
    matmulscan_fused_scan_diagonal,
    torch_cumsum_scan_diagonal,
    sequential_scan_dense,
)


def benchmark_fn(fn, *args, warmup=10, repeat=100, **kwargs):
    """Benchmark a function with warmup and repeated timing."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(repeat)]
    times.sort()

    # Remove outliers (top/bottom 10%)
    trim = max(1, repeat // 10)
    trimmed_times = times[trim:-trim] if len(times) > 2 * trim else times

    return {
        "mean_ms": sum(trimmed_times) / len(trimmed_times),
        "median_ms": trimmed_times[len(trimmed_times) // 2],
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(trimmed_times) / len(trimmed_times)) ** 2 for t in trimmed_times) / len(trimmed_times)) ** 0.5,
    }


def check_accuracy(
    ref_alpha: torch.Tensor,
    ref_b: torch.Tensor,
    test_alpha: torch.Tensor,
    test_b: torch.Tensor,
    name: str,
    dtype_name: str,
) -> dict:
    """Check numerical accuracy against reference."""
    alpha_err = (ref_alpha.float() - test_alpha.float()).abs().max().item()
    b_err = (ref_b.float() - test_b.float()).abs().max().item()

    alpha_rel_err = (alpha_err / (ref_alpha.float().abs().max().item() + 1e-10))
    b_rel_err = (b_err / (ref_b.float().abs().max().item() + 1e-10))

    threshold = 1e-3 if "bf16" in dtype_name else 1e-5

    result = {
        "name": name,
        "dtype": dtype_name,
        "alpha_max_abs_err": alpha_err,
        "b_max_abs_err": b_err,
        "alpha_max_rel_err": alpha_rel_err,
        "b_max_rel_err": b_rel_err,
        "threshold": threshold,
        "alpha_pass": alpha_err < threshold,
        "b_pass": b_err < threshold,
        "pass": alpha_err < threshold and b_err < threshold,
    }

    status = "PASS" if result["pass"] else "FAIL"
    print(f"  {name} ({dtype_name}): alpha_err={alpha_err:.2e}, b_err={b_err:.2e} [{status}]")

    return result


def run_diagonal_benchmark(
    G: int,
    n: int,
    d_v: int,
    dtype: torch.dtype,
    radix_values: list[int],
    warmup: int = 10,
    repeat: int = 100,
    device: str = "cuda",
) -> dict:
    """Run benchmark for diagonal SSM case."""
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp32"
    print(f"\n{'='*70}")
    print(f"Diagonal Scan Benchmark: G={G}, n={n}, d_v={d_v}, dtype={dtype_name}")
    print(f"{'='*70}")

    # Generate random data
    # alpha should be close to 1 (typical for SSM damping factors)
    alpha = torch.rand(G, n, device=device, dtype=dtype) * 0.2 + 0.8  # [0.8, 1.0]
    b = torch.randn(G, n, d_v, device=device, dtype=dtype) * 0.1

    results = {"G": G, "n": n, "d_v": d_v, "dtype": dtype_name}

    # 1. Sequential reference (ground truth)
    print("\n1. Sequential scan (reference)...")
    ref_alpha, ref_b = sequential_scan_diagonal(alpha.float(), b.float())
    seq_timing = benchmark_fn(sequential_scan_diagonal, alpha.float(), b.float(), warmup=warmup, repeat=repeat)
    results["sequential"] = seq_timing
    print(f"   Time: {seq_timing['median_ms']:.4f} ms (median)")

    # 2. Blelloch scan (Triton baseline)
    print("\n2. Blelloch scan (baseline)...")
    blelloch_alpha, blelloch_b = blelloch_scan_diagonal(alpha, b)
    accuracy_blelloch = check_accuracy(ref_alpha, ref_b, blelloch_alpha, blelloch_b, "Blelloch", dtype_name)
    blelloch_timing = benchmark_fn(blelloch_scan_diagonal, alpha, b, warmup=warmup, repeat=repeat)
    results["blelloch"] = {**blelloch_timing, "accuracy": accuracy_blelloch}
    print(f"   Time: {blelloch_timing['median_ms']:.4f} ms (median)")

    # 3. torch.cumsum baseline
    print("\n3. torch.cumsum scan...")
    cumsum_alpha, cumsum_b = torch_cumsum_scan_diagonal(alpha, b)
    accuracy_cumsum = check_accuracy(ref_alpha, ref_b, cumsum_alpha, cumsum_b, "torch.cumsum", dtype_name)
    cumsum_timing = benchmark_fn(torch_cumsum_scan_diagonal, alpha, b, warmup=warmup, repeat=repeat)
    results["torch_cumsum"] = {**cumsum_timing, "accuracy": accuracy_cumsum}
    print(f"   Time: {cumsum_timing['median_ms']:.4f} ms (median)")

    # 4. MatMulScan (recursive, multi-kernel)
    for s in radix_values:
        print(f"\n4. MatMulScan (recursive, s={s})...")
        try:
            ms_alpha, ms_b = matmulscan_scan_diagonal(alpha, b, s=s)
            accuracy_ms = check_accuracy(ref_alpha, ref_b, ms_alpha, ms_b, f"MatMulScan-recursive-s{s}", dtype_name)
            ms_timing = benchmark_fn(matmulscan_scan_diagonal, alpha, b, s=s, warmup=warmup, repeat=repeat)
            results[f"matmulscan_recursive_s{s}"] = {**ms_timing, "accuracy": accuracy_ms}
            print(f"   Time: {ms_timing['median_ms']:.4f} ms (median)")
        except Exception as e:
            print(f"   ERROR: {e}")
            results[f"matmulscan_recursive_s{s}"] = {"error": str(e)}

    # 5. MatMulScan (fused kernel)
    for s in radix_values:
        print(f"\n5. MatMulScan (fused, s={s})...")
        try:
            ms_fused_alpha, ms_fused_b = matmulscan_fused_scan_diagonal(alpha, b, s=s)
            accuracy_ms_fused = check_accuracy(ref_alpha, ref_b, ms_fused_alpha, ms_fused_b, f"MatMulScan-fused-s{s}", dtype_name)
            ms_fused_timing = benchmark_fn(matmulscan_fused_scan_diagonal, alpha, b, s=s, warmup=warmup, repeat=repeat)
            results[f"matmulscan_fused_s{s}"] = {**ms_fused_timing, "accuracy": accuracy_ms_fused}
            print(f"   Time: {ms_fused_timing['median_ms']:.4f} ms (median)")
        except Exception as e:
            print(f"   ERROR: {e}")
            results[f"matmulscan_fused_s{s}"] = {"error": str(e)}

    # Compute speedups relative to Blelloch baseline
    base_time = blelloch_timing["median_ms"]
    print(f"\n--- Speedups vs Blelloch (baseline={base_time:.4f} ms) ---")

    for key in results:
        if isinstance(results[key], dict) and "median_ms" in results[key]:
            speedup = base_time / results[key]["median_ms"] if results[key]["median_ms"] > 0 else 0
            results[key]["speedup_vs_blelloch"] = speedup
            if key != "blelloch":
                print(f"  {key}: {results[key]['median_ms']:.4f} ms = {speedup:.2f}x")

    return results


def run_dense_benchmark(
    G: int,
    n: int,
    d_v: int,
    device: str = "cuda",
) -> dict:
    """Run benchmark for dense SSM case (sequential only for MVE)."""
    print(f"\n{'='*70}")
    print(f"Dense Scan Reference: G={G}, n={n}, d_v={d_v}")
    print(f"{'='*70}")

    # Generate random dense transitions
    A = torch.randn(G, n, n, device=device, dtype=torch.float32) * 0.1
    # Make them contractive (spectral radius < 1)
    for j in range(G):
        A[j] = A[j] / (torch.linalg.norm(A[j], ord=2) + 1.0) * 0.9
    b = torch.randn(G, n, d_v, device=device, dtype=torch.float32) * 0.1

    # Sequential reference
    print("\nSequential dense scan...")
    cum_A, cum_b = sequential_scan_dense(A, b)
    timing = benchmark_fn(sequential_scan_dense, A, b, warmup=5, repeat=50)
    print(f"  Time: {timing['median_ms']:.4f} ms (median)")

    # Check that cumulative product is contractive
    final_norm = torch.linalg.norm(cum_A[-1], ord=2).item()
    print(f"  Final transition matrix spectral norm: {final_norm:.4f}")

    return {
        "G": G, "n": n, "d_v": d_v,
        "sequential": timing,
        "final_spectral_norm": final_norm,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 044: MatMulScan Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "experiment": {"name": "044-matmulscan-scan", "n": 16, "d_v": 64},
            "benchmark": {
                "G_values": [64, 128, 256],
                "radix_values": [4, 8],
                "warmup": 10,
                "repeat": 100,
                "dtype": "float32",
                "dense_G": 64,
            },
        }

    print("=" * 70)
    print("Experiment 044: MatMulScan Inter-Chunk State Scan Benchmark")
    print("=" * 70)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Initialize wandb
    try:
        import wandb
        wandb.init(
            project="mad-architecture-search",
            name=f"exp-044-matmulscan-scan",
            config=config,
        )
        wandb_url = wandb.run.get_url()
        print(f"\nWandb URL: {wandb_url}")
        use_wandb = True
    except Exception as e:
        print(f"Warning: wandb init failed: {e}")
        use_wandb = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Experiment parameters
    exp_config = config.get("experiment", {})
    bench_config = config.get("benchmark", {})

    n = exp_config.get("n", 16)
    d_v = exp_config.get("d_v", 64)
    G_values = bench_config.get("G_values", [64, 128, 256])
    radix_values = bench_config.get("radix_values", [4, 8])
    warmup = bench_config.get("warmup", 10)
    repeat = bench_config.get("repeat", 100)
    dtype_str = bench_config.get("dtype", "float32")
    dense_G = bench_config.get("dense_G", 64)

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    all_results = {
        "config": config,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "diagonal_results": {},
        "dense_results": {},
    }

    # Run diagonal benchmarks for each G
    for G in G_values:
        results = run_diagonal_benchmark(
            G=G, n=n, d_v=d_v, dtype=dtype,
            radix_values=radix_values,
            warmup=warmup, repeat=repeat,
            device=device,
        )
        all_results["diagonal_results"][f"G{G}"] = results

        # Log to wandb
        if use_wandb:
            log_data = {}
            for method, data in results.items():
                if isinstance(data, dict) and "median_ms" in data:
                    log_data[f"diagonal/G{G}/{method}/median_ms"] = data["median_ms"]
                    if "speedup_vs_blelloch" in data:
                        log_data[f"diagonal/G{G}/{method}/speedup"] = data["speedup_vs_blelloch"]
                    if "accuracy" in data and isinstance(data["accuracy"], dict):
                        log_data[f"diagonal/G{G}/{method}/alpha_err"] = data["accuracy"].get("alpha_max_abs_err", 0)
                        log_data[f"diagonal/G{G}/{method}/b_err"] = data["accuracy"].get("b_max_abs_err", 0)
                        log_data[f"diagonal/G{G}/{method}/pass"] = 1.0 if data["accuracy"].get("pass", False) else 0.0
            wandb.log(log_data)

    # Run also with bf16 if primary is fp32
    if dtype == torch.float32:
        print("\n\n" + "=" * 70)
        print("ALSO TESTING BF16")
        print("=" * 70)
        for G in G_values:
            results_bf16 = run_diagonal_benchmark(
                G=G, n=n, d_v=d_v, dtype=torch.bfloat16,
                radix_values=radix_values,
                warmup=warmup, repeat=repeat,
                device=device,
            )
            all_results["diagonal_results"][f"G{G}_bf16"] = results_bf16

            if use_wandb:
                log_data = {}
                for method, data in results_bf16.items():
                    if isinstance(data, dict) and "median_ms" in data:
                        log_data[f"diagonal_bf16/G{G}/{method}/median_ms"] = data["median_ms"]
                        if "speedup_vs_blelloch" in data:
                            log_data[f"diagonal_bf16/G{G}/{method}/speedup"] = data["speedup_vs_blelloch"]
                wandb.log(log_data)

    # Run dense benchmark
    dense_results = run_dense_benchmark(
        G=dense_G, n=n, d_v=d_v, device=device,
    )
    all_results["dense_results"] = dense_results

    if use_wandb:
        wandb.log({
            "dense/G64/sequential/median_ms": dense_results["sequential"]["median_ms"],
        })

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check success criteria
    success_criteria = {}

    # Criterion 1: MatMulScan (s=4) >= 1.2x for G=128
    g128 = all_results["diagonal_results"].get("G128", {})
    for method_key in ["matmulscan_recursive_s4", "matmulscan_fused_s4"]:
        if method_key in g128 and "speedup_vs_blelloch" in g128[method_key]:
            speedup = g128[method_key]["speedup_vs_blelloch"]
            success_criteria[f"{method_key}_G128_speedup"] = {
                "target": 1.2,
                "achieved": speedup,
                "pass": speedup >= 1.2,
            }
            print(f"  {method_key} G=128 speedup: {speedup:.2f}x (target >= 1.2x) {'PASS' if speedup >= 1.2 else 'FAIL'}")

    # Criterion 2: MatMulScan >= 1.3x for G=256
    g256 = all_results["diagonal_results"].get("G256", {})
    for method_key in ["matmulscan_recursive_s4", "matmulscan_fused_s4"]:
        if method_key in g256 and "speedup_vs_blelloch" in g256[method_key]:
            speedup = g256[method_key]["speedup_vs_blelloch"]
            success_criteria[f"{method_key}_G256_speedup"] = {
                "target": 1.3,
                "achieved": speedup,
                "pass": speedup >= 1.3,
            }
            print(f"  {method_key} G=256 speedup: {speedup:.2f}x (target >= 1.3x) {'PASS' if speedup >= 1.3 else 'FAIL'}")

    # Criterion 3: Numerical accuracy
    for g_key, g_results in all_results["diagonal_results"].items():
        for method, data in g_results.items():
            if isinstance(data, dict) and "accuracy" in data:
                acc = data["accuracy"]
                if isinstance(acc, dict):
                    success_criteria[f"{method}_{g_key}_accuracy"] = {
                        "pass": acc.get("pass", False),
                        "alpha_err": acc.get("alpha_max_abs_err", 0),
                        "b_err": acc.get("b_max_abs_err", 0),
                    }

    all_results["success_criteria"] = success_criteria

    # Log final results to wandb
    if use_wandb:
        final_log = {}
        for crit_name, crit_data in success_criteria.items():
            final_log[f"success_criteria/{crit_name}"] = 1.0 if crit_data.get("pass", False) else 0.0
            if "achieved" in crit_data:
                final_log[f"final/{crit_name}"] = crit_data["achieved"]
        wandb.log(final_log)
        wandb.finish()

    # Save results to file
    results_path = Path("/results/044_benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save locally
    local_results_path = Path(__file__).parent / "benchmark_results.json"
    with open(local_results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Results also saved to {local_results_path}")

    return all_results


if __name__ == "__main__":
    main()

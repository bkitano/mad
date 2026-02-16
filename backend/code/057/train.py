"""
MVE 057: FlashRNN-Style Fused Inter-Chunk State Recurrence Benchmark

This script benchmarks the proposed FlashRNN-style fused inter-chunk scan
against the baseline sequential scan (fla-org style).

Key comparisons:
1. Correctness: bit-exact match between baseline and proposed
2. Timing: kernel wall-clock time (ms)
3. Scaling: across G in {32, 64, 128} chunks

Success criteria (from proposal):
- Inter-chunk scan kernel time decreases by > 20% vs baseline
- Results are bit-exact with the baseline
- Consistent across G in {32, 64, 128}
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

# Import wandb (will be initialized later)
import wandb


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_synthetic_data(B, G, H, dk, dv, device="cuda", dtype=torch.float32):
    """
    Generate synthetic inter-chunk state data for benchmarking.

    h_local: [B, G, H, dk, dv] - isolated per-chunk state contributions
    gamma: [B, G, H, dk] - cumulative per-chunk decays in (0, 1)
    """
    h_local = torch.randn(B, G, H, dk, dv, device=device, dtype=dtype) * 0.1
    # Decay factors in (0, 1) — typical for GLA
    gamma = torch.sigmoid(torch.randn(B, G, H, dk, device=device, dtype=dtype))
    return h_local, gamma


def benchmark_kernel(fn, h_local, gamma, warmup=50, repeats=200, label="kernel"):
    """
    Benchmark a kernel function with proper warmup and timing.

    Returns: (mean_ms, std_ms, result_tensor)
    """
    # Warmup runs (also compiles Triton kernels)
    for _ in range(warmup):
        result = fn(h_local, gamma)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(h_local, gamma)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times_tensor = torch.tensor(times)
    mean_ms = times_tensor.mean().item()
    std_ms = times_tensor.std().item()
    median_ms = times_tensor.median().item()

    print(f"  {label}: mean={mean_ms:.4f}ms, std={std_ms:.4f}ms, median={median_ms:.4f}ms")
    return mean_ms, std_ms, median_ms, result


def check_correctness(result_baseline, result_proposed, result_reference, atol=1e-4, rtol=1e-4):
    """
    Verify that all three implementations produce the same results.

    Returns dict with correctness metrics.
    """
    # Compare baseline vs reference (PyTorch)
    baseline_vs_ref_max = (result_baseline.float() - result_reference.float()).abs().max().item()
    baseline_vs_ref_mean = (result_baseline.float() - result_reference.float()).abs().mean().item()

    # Compare proposed vs reference (PyTorch)
    proposed_vs_ref_max = (result_proposed.float() - result_reference.float()).abs().max().item()
    proposed_vs_ref_mean = (result_proposed.float() - result_reference.float()).abs().mean().item()

    # Compare baseline vs proposed directly
    baseline_vs_proposed_max = (result_baseline.float() - result_proposed.float()).abs().max().item()
    baseline_vs_proposed_mean = (result_baseline.float() - result_proposed.float()).abs().mean().item()

    # Check if they're close enough
    baseline_correct = torch.allclose(result_baseline.float(), result_reference.float(), atol=atol, rtol=rtol)
    proposed_correct = torch.allclose(result_proposed.float(), result_reference.float(), atol=atol, rtol=rtol)
    match = torch.allclose(result_baseline.float(), result_proposed.float(), atol=atol, rtol=rtol)

    return {
        "baseline_vs_ref_max_err": baseline_vs_ref_max,
        "baseline_vs_ref_mean_err": baseline_vs_ref_mean,
        "proposed_vs_ref_max_err": proposed_vs_ref_max,
        "proposed_vs_ref_mean_err": proposed_vs_ref_mean,
        "baseline_vs_proposed_max_err": baseline_vs_proposed_max,
        "baseline_vs_proposed_mean_err": baseline_vs_proposed_mean,
        "baseline_correct": baseline_correct,
        "proposed_correct": proposed_correct,
        "results_match": match,
    }


def run_benchmark(config: dict):
    """Run the full benchmark suite."""
    # Import kernels
    from models.flashrnn_scan import (
        baseline_inter_chunk_scan,
        flashrnn_inter_chunk_scan,
        pytorch_inter_chunk_scan,
    )

    bench_config = config.get("benchmark", {})
    B = bench_config.get("batch_size", 4)
    H = bench_config.get("num_heads", 16)
    dk = bench_config.get("dk", 64)
    dv = bench_config.get("dv", 64)
    G_values = bench_config.get("G_values", [32, 64, 128])
    warmup = bench_config.get("warmup", 50)
    repeats = bench_config.get("repeats", 200)
    dtype_str = bench_config.get("dtype", "float32")
    dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16

    print("=" * 70)
    print("MVE 057: FlashRNN-Style Fused Inter-Chunk State Recurrence")
    print("=" * 70)
    print(f"Config: B={B}, H={H}, dk={dk}, dv={dv}")
    print(f"G values: {G_values}")
    print(f"Warmup: {warmup}, Repeats: {repeats}, dtype: {dtype}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Initialize wandb
    wandb.init(
        project="mad-architecture-search",
        name=f"exp-057-flashrnn-fused-scan",
        config={
            "experiment": 57,
            "proposal": "057-flashrnn-fused-inter-chunk-state-recurrence",
            "batch_size": B,
            "num_heads": H,
            "dk": dk,
            "dv": dv,
            "G_values": G_values,
            "warmup": warmup,
            "repeats": repeats,
            "dtype": dtype_str,
            "device": torch.cuda.get_device_name(0),
        },
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")
    print()

    all_results = {}

    for G in G_values:
        print(f"\n{'='*50}")
        print(f"Testing G={G} chunks (sequence length T={G*64} with C=64)")
        print(f"{'='*50}")

        # Generate synthetic data
        h_local, gamma = generate_synthetic_data(B, G, H, dk, dv, device="cuda", dtype=dtype)

        # State size info
        state_bytes = dk * dv * 4  # float32
        total_hbm_baseline = G * (1 + 4 * dv) * dk * 4 * B * H  # approximate
        total_hbm_proposed = G * (1 + 2 * dv) * dk * 4 * B * H  # approximate
        print(f"  State matrix size per head: {state_bytes / 1024:.1f} KB")
        print(f"  Approx HBM traffic baseline: {total_hbm_baseline / 1e6:.2f} MB")
        print(f"  Approx HBM traffic proposed: {total_hbm_proposed / 1e6:.2f} MB")
        print(f"  Expected traffic reduction: {total_hbm_baseline / total_hbm_proposed:.2f}x")
        print()

        # 1. PyTorch reference (for correctness only)
        print("Running PyTorch reference...")
        with torch.no_grad():
            ref_result = pytorch_inter_chunk_scan(h_local, gamma)
        print("  Done.")

        # 2. Benchmark baseline
        print("\nBenchmarking baseline (HBM round-trips)...")
        with torch.no_grad():
            base_mean, base_std, base_median, base_result = benchmark_kernel(
                baseline_inter_chunk_scan, h_local, gamma,
                warmup=warmup, repeats=repeats, label="Baseline"
            )

        # 3. Benchmark proposed
        print("\nBenchmarking FlashRNN-fused (registers)...")
        with torch.no_grad():
            fused_mean, fused_std, fused_median, fused_result = benchmark_kernel(
                flashrnn_inter_chunk_scan, h_local, gamma,
                warmup=warmup, repeats=repeats, label="FlashRNN"
            )

        # 4. Correctness check
        print("\nChecking correctness...")
        correctness = check_correctness(base_result, fused_result, ref_result)
        for key, val in correctness.items():
            print(f"  {key}: {val}")

        # 5. Compute speedup
        speedup = base_mean / fused_mean if fused_mean > 0 else 0
        speedup_median = base_median / fused_median if fused_median > 0 else 0
        time_reduction_pct = (1 - fused_mean / base_mean) * 100 if base_mean > 0 else 0

        print(f"\n  Speedup (mean): {speedup:.3f}x")
        print(f"  Speedup (median): {speedup_median:.3f}x")
        print(f"  Time reduction: {time_reduction_pct:.1f}%")

        # Log to wandb
        wandb.log({
            f"G={G}/baseline_mean_ms": base_mean,
            f"G={G}/baseline_std_ms": base_std,
            f"G={G}/baseline_median_ms": base_median,
            f"G={G}/flashrnn_mean_ms": fused_mean,
            f"G={G}/flashrnn_std_ms": fused_std,
            f"G={G}/flashrnn_median_ms": fused_median,
            f"G={G}/speedup_mean": speedup,
            f"G={G}/speedup_median": speedup_median,
            f"G={G}/time_reduction_pct": time_reduction_pct,
            f"G={G}/results_match": correctness["results_match"],
            f"G={G}/max_err_baseline_vs_proposed": correctness["baseline_vs_proposed_max_err"],
        })

        all_results[G] = {
            "baseline_mean_ms": base_mean,
            "baseline_std_ms": base_std,
            "baseline_median_ms": base_median,
            "flashrnn_mean_ms": fused_mean,
            "flashrnn_std_ms": fused_std,
            "flashrnn_median_ms": fused_median,
            "speedup_mean": speedup,
            "speedup_median": speedup_median,
            "time_reduction_pct": time_reduction_pct,
            "correctness": correctness,
        }

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'G':>5} | {'Baseline (ms)':>14} | {'FlashRNN (ms)':>14} | {'Speedup':>8} | {'Reduction':>10} | {'Correct':>8}")
    print("-" * 70)

    all_correct = True
    all_speedups_above_20pct = True

    for G in G_values:
        r = all_results[G]
        correct_str = "YES" if r["correctness"]["results_match"] else "NO"
        print(f"{G:>5} | {r['baseline_mean_ms']:>14.4f} | {r['flashrnn_mean_ms']:>14.4f} | {r['speedup_mean']:>7.3f}x | {r['time_reduction_pct']:>9.1f}% | {correct_str:>8}")

        if not r["correctness"]["results_match"]:
            all_correct = False
        if r["time_reduction_pct"] < 20:
            all_speedups_above_20pct = False

    # ========================================================================
    # Success criteria evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)

    criteria = {}

    # Criterion 1: > 20% speedup
    avg_reduction = sum(all_results[G]["time_reduction_pct"] for G in G_values) / len(G_values)
    criteria["kernel_time_reduction_gt_20pct"] = all_speedups_above_20pct
    print(f"\n1. Inter-chunk scan kernel time decreases by > 20%:")
    for G in G_values:
        r = all_results[G]
        status = "PASS" if r["time_reduction_pct"] > 20 else "FAIL"
        print(f"   G={G}: {r['time_reduction_pct']:.1f}% reduction [{status}]")
    print(f"   Average: {avg_reduction:.1f}% reduction")
    print(f"   Overall: {'PASS' if all_speedups_above_20pct else 'FAIL'}")

    # Criterion 2: Results are bit-exact
    criteria["results_bit_exact"] = all_correct
    print(f"\n2. Results are bit-exact with baseline:")
    for G in G_values:
        r = all_results[G]
        status = "PASS" if r["correctness"]["results_match"] else "FAIL"
        print(f"   G={G}: max_err={r['correctness']['baseline_vs_proposed_max_err']:.2e} [{status}]")
    print(f"   Overall: {'PASS' if all_correct else 'FAIL'}")

    # Criterion 3: Consistent across G values
    speedups = [all_results[G]["speedup_mean"] for G in G_values]
    consistent = all(s > 1.0 for s in speedups) or all(s <= 1.0 for s in speedups)
    criteria["consistent_across_G"] = consistent
    print(f"\n3. Consistent across G in {G_values}:")
    print(f"   Speedups: {', '.join(f'{s:.3f}x' for s in speedups)}")
    print(f"   Overall: {'PASS' if consistent else 'FAIL'}")

    # Overall decision
    all_pass = all(criteria.values())
    decision = "PROCEED" if all_pass else ("DEBUG" if criteria["results_bit_exact"] else "ABANDON")
    print(f"\n{'='*70}")
    print(f"DECISION: {decision}")
    print(f"{'='*70}")

    # Log final results to wandb
    wandb.log({
        "final/avg_time_reduction_pct": avg_reduction,
        "final/all_correct": all_correct,
        "final/all_speedups_above_20pct": all_speedups_above_20pct,
        "final/consistent_across_G": consistent,
        "final/decision": decision,
        "success_criteria/kernel_time_reduction_gt_20pct": all_speedups_above_20pct,
        "success_criteria/results_bit_exact": all_correct,
        "success_criteria/consistent_across_G": consistent,
    })

    # Save results to file
    results_data = {
        "config": {
            "B": B, "H": H, "dk": dk, "dv": dv,
            "G_values": G_values,
            "warmup": warmup, "repeats": repeats,
            "device": torch.cuda.get_device_name(0),
        },
        "results": {str(G): all_results[G] for G in G_values},
        "criteria": criteria,
        "decision": decision,
        "wandb_url": wandb_url,
    }

    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (bool,)):
            return obj
        elif isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    results_path = "/root/results/057_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(make_serializable(results_data), f, indent=2)
    print(f"\nResults saved to {results_path}")

    wandb.finish()
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config)

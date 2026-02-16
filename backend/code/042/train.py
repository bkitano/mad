"""
Experiment 042: Contraction-Ordered Multi-Operand Chunkwise GLA Fusion

Benchmark comparing two contraction orderings for GLA intra-chunk computation:
  - Path 1 (Standard): Q@K^T -> mask -> @V + Q@h
  - Path 2 (Right-associated): Q@cumsum(K^T V) - rank-r correction + Q@h

Sweeps over:
  - C (chunk size): {32, 64, 128}
  - d (head dimension): {64, 128, 256}
  - r (mask approximation rank): {1, 4, 8, 16, C}

Success criteria:
  - Path 2 >= 10% faster than Path 1 for at least one realistic config
  - Numerical agreement within epsilon < 1e-3 relative error (BF16)

Usage:
    python train.py --config config.yaml
"""

import argparse
import time
import yaml
import json
import torch
import numpy as np
from pathlib import Path

# Attempt wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not available, logging to stdout only")

from models.gla_paths import (
    build_causal_decay_mask,
    path1_standard,
    path2_right_associated,
    decompose_mask_low_rank,
    compute_flops,
    measure_effective_rank,
)


def benchmark_single_config(
    C: int,
    d_k: int,
    d_v: int,
    rank: int,
    batch_size: int = 16,
    n_warmup: int = 50,
    n_trials: int = 200,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Benchmark Path 1 vs Path 2 for a single configuration.

    We batch the computation: run batch_size independent chunks per call,
    timing the full batch to get stable measurements.

    Returns dict with timing and correctness metrics.
    """
    results = {}

    # Generate random tensors (BF16 as specified in proposal)
    Q = torch.randn(batch_size, C, d_k, device=device, dtype=dtype)
    K = torch.randn(batch_size, C, d_k, device=device, dtype=dtype)
    V = torch.randn(batch_size, C, d_v, device=device, dtype=dtype)
    h = torch.randn(batch_size, d_k, d_v, device=device, dtype=dtype)

    # Decay rates: alpha in (0.9, 1.0) -- typical for GLA (slow decay)
    alpha = 0.9 + 0.1 * torch.rand(batch_size, C, device=device, dtype=dtype)

    # Build masks for all batches
    masks = torch.stack([build_causal_decay_mask(alpha[b]) for b in range(batch_size)])  # (B, C, C)

    # Precompute mask decomposition for Path 2 (done once, not timed)
    decomps = []
    effective_ranks = []
    for b in range(batch_size):
        U, S_diag, W = decompose_mask_low_rank(masks[b], rank)
        decomps.append((U, S_diag, W))
        eff_rank = measure_effective_rank(masks[b])
        effective_ranks.append(eff_rank)

    avg_eff_rank = np.mean(effective_ranks)
    results["effective_rank_mean"] = float(avg_eff_rank)
    results["effective_rank_std"] = float(np.std(effective_ranks))

    # Compute theoretical FLOPs
    flop_info = compute_flops(C, d_k, d_v, rank)
    results["theoretical_flops"] = flop_info

    # --- Benchmark Path 1 ---
    torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup):
        for b in range(batch_size):
            _ = path1_standard(Q[b], K[b], V[b], masks[b], h[b])
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_trials):
        for b in range(batch_size):
            out1 = path1_standard(Q[b], K[b], V[b], masks[b], h[b])
    end_event.record()
    torch.cuda.synchronize()

    path1_time_ms = start_event.elapsed_time(end_event)
    path1_per_call_us = (path1_time_ms * 1000) / (n_trials * batch_size)
    results["path1_total_ms"] = path1_time_ms
    results["path1_per_call_us"] = path1_per_call_us

    # --- Benchmark Path 2 ---
    torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup):
        for b in range(batch_size):
            U, S_diag, W = decomps[b]
            _ = path2_right_associated(Q[b], K[b], V[b], masks[b], h[b], rank, U, S_diag, W)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_trials):
        for b in range(batch_size):
            U, S_diag, W = decomps[b]
            out2 = path2_right_associated(Q[b], K[b], V[b], masks[b], h[b], rank, U, S_diag, W)
    end_event.record()
    torch.cuda.synchronize()

    path2_time_ms = start_event.elapsed_time(end_event)
    path2_per_call_us = (path2_time_ms * 1000) / (n_trials * batch_size)
    results["path2_total_ms"] = path2_time_ms
    results["path2_per_call_us"] = path2_per_call_us

    # --- Speedup ---
    speedup = path1_per_call_us / path2_per_call_us if path2_per_call_us > 0 else 0
    results["speedup"] = speedup
    results["path2_faster_pct"] = (speedup - 1.0) * 100

    # --- Numerical correctness ---
    # Compare Path 1 (exact) vs Path 2 (approximate with rank r)
    rel_errors = []
    for b in range(batch_size):
        ref = path1_standard(Q[b], K[b], V[b], masks[b], h[b])
        U, S_diag, W = decomps[b]
        approx = path2_right_associated(Q[b], K[b], V[b], masks[b], h[b], rank, U, S_diag, W)

        # Relative error
        ref_f32 = ref.float()
        approx_f32 = approx.float()
        denom = ref_f32.norm() + 1e-8
        rel_error = (ref_f32 - approx_f32).norm() / denom
        rel_errors.append(rel_error.item())

    results["rel_error_mean"] = float(np.mean(rel_errors))
    results["rel_error_max"] = float(np.max(rel_errors))
    results["numerical_pass"] = results["rel_error_max"] < 1e-3

    return results


def benchmark_batched_config(
    C: int,
    d_k: int,
    d_v: int,
    rank: int,
    alpha_min: float = 0.9,
    batch_size: int = 64,
    n_warmup: int = 50,
    n_trials: int = 200,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Batched benchmark: process all chunks in parallel using batched matmuls.
    This is more realistic for GPU utilization.
    """
    results = {}

    # Generate random tensors
    Q = torch.randn(batch_size, C, d_k, device=device, dtype=dtype)
    K = torch.randn(batch_size, C, d_k, device=device, dtype=dtype)
    V = torch.randn(batch_size, C, d_v, device=device, dtype=dtype)
    h = torch.randn(batch_size, d_k, d_v, device=device, dtype=dtype)

    # Decay rates: alpha in (alpha_min, 1.0)
    alpha_range = 1.0 - alpha_min
    alpha = alpha_min + alpha_range * torch.rand(batch_size, C, device=device, dtype=dtype)
    masks = torch.stack([build_causal_decay_mask(alpha[b]) for b in range(batch_size)])

    # Measure effective rank
    effective_ranks = [measure_effective_rank(masks[b]) for b in range(batch_size)]
    results["effective_rank_mean"] = float(np.mean(effective_ranks))

    # Precompute decomposition
    L = torch.tril(torch.ones(C, C, device=device, dtype=dtype)).unsqueeze(0).expand(batch_size, -1, -1)
    Delta = L - masks  # (B, C, C)

    # Batch SVD for decomposition (use float32)
    Delta_f32 = Delta.float()
    U_full, S_full, Vh_full = torch.linalg.svd(Delta_f32, full_matrices=False)
    r = min(rank, C)
    U_batch = U_full[:, :, :r].to(dtype)      # (B, C, r)
    S_batch = S_full[:, :r].to(dtype)          # (B, r)
    W_batch = Vh_full[:, :r, :].transpose(1, 2).to(dtype)  # (B, C, r)

    flop_info = compute_flops(C, d_k, d_v, rank)
    results["theoretical_flops"] = flop_info

    # --- Batched Path 1 ---
    def run_path1_batched():
        S = torch.bmm(Q, K.transpose(1, 2))  # (B, C, C)
        S_tilde = S * masks                    # (B, C, C)
        O_intra = torch.bmm(S_tilde, V)       # (B, C, d_v)
        O_state = torch.bmm(Q, h)             # (B, C, d_v)
        return O_intra + O_state

    # Warmup
    for _ in range(n_warmup):
        _ = run_path1_batched()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_trials):
        out1 = run_path1_batched()
    end_event.record()
    torch.cuda.synchronize()

    path1_time_ms = start_event.elapsed_time(end_event)
    results["path1_total_ms"] = path1_time_ms
    results["path1_per_call_us"] = (path1_time_ms * 1000) / n_trials

    # --- Batched Path 2 ---
    def run_path2_batched():
        # Step 1: cumulative K^T V outer products
        KV_outer = K.unsqueeze(3) * V.unsqueeze(2)  # (B, C, d_k, d_v)
        KV_cumsum = torch.cumsum(KV_outer, dim=1)    # (B, C, d_k, d_v)

        # Step 2: Q @ cumsum
        O_right = torch.einsum('bcq,bcqv->bcv', Q, KV_cumsum)  # (B, C, d_v)

        # Step 3-5: Rank-r correction
        US = U_batch * S_batch.unsqueeze(1)  # (B, C, r)
        causal = torch.tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))

        correction = torch.zeros(batch_size, C, d_v, device=Q.device, dtype=Q.dtype)
        for i in range(r):
            Q_mod = Q * US[:, :, i:i+1]     # (B, C, d_k)
            K_mod = K * W_batch[:, :, i:i+1] # (B, C, d_k)
            attn_r = torch.bmm(Q_mod, K_mod.transpose(1, 2))  # (B, C, C)
            attn_r = attn_r * causal.unsqueeze(0)
            correction = correction + torch.bmm(attn_r, V)

        # Step 6: State correction
        O_state = torch.bmm(Q, h)  # (B, C, d_v)

        return O_right - correction + O_state

    # Warmup
    for _ in range(n_warmup):
        _ = run_path2_batched()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_trials):
        out2 = run_path2_batched()
    end_event.record()
    torch.cuda.synchronize()

    path2_time_ms = start_event.elapsed_time(end_event)
    results["path2_total_ms"] = path2_time_ms
    results["path2_per_call_us"] = (path2_time_ms * 1000) / n_trials

    # Speedup
    speedup = results["path1_per_call_us"] / results["path2_per_call_us"] if results["path2_per_call_us"] > 0 else 0
    results["speedup"] = speedup
    results["path2_faster_pct"] = (speedup - 1.0) * 100

    # Numerical correctness
    ref = run_path1_batched()
    approx = run_path2_batched()
    ref_f32 = ref.float()
    approx_f32 = approx.float()
    per_batch_errors = []
    for b in range(batch_size):
        denom = ref_f32[b].norm() + 1e-8
        err = (ref_f32[b] - approx_f32[b]).norm() / denom
        per_batch_errors.append(err.item())

    results["rel_error_mean"] = float(np.mean(per_batch_errors))
    results["rel_error_max"] = float(np.max(per_batch_errors))
    results["numerical_pass"] = results["rel_error_max"] < 1e-3

    return results


def run_full_sweep(config: dict) -> dict:
    """Run the full configuration sweep as specified in the proposal."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU -- timings will not be meaningful for GPU comparison")

    sweep_config = config.get("sweep", {})
    C_values = sweep_config.get("chunk_sizes", [32, 64, 128])
    d_values = sweep_config.get("head_dims", [64, 128, 256])
    r_values_base = sweep_config.get("ranks", [1, 4, 8, 16])
    alpha_mins = sweep_config.get("alpha_mins", [0.9, 0.99, 0.999])
    batch_size = sweep_config.get("batch_size", 64)
    n_warmup = sweep_config.get("n_warmup", 50)
    n_trials = sweep_config.get("n_trials", 200)
    use_batched = sweep_config.get("batched", True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    all_results = []
    best_speedup = 0.0
    best_config = None
    any_path2_wins = False

    total_configs = 0
    for C in C_values:
        r_values = r_values_base + [C]  # Add r=C (full rank)
        for d in d_values:
            for r in r_values:
                if r > C:
                    continue
                for _ in alpha_mins:
                    total_configs += 1

    print(f"\n{'='*80}")
    print(f"Experiment 042: GLA Contraction Ordering Benchmark")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Configurations: {total_configs}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup: {n_warmup}, Trials: {n_trials}")
    print(f"Alpha ranges: {alpha_mins}")
    print(f"{'='*80}\n")

    config_idx = 0
    for C in C_values:
        r_values = r_values_base + [C]
        for d in d_values:
            d_k = d_v = d  # symmetric head dimension
            for r in r_values:
                if r > C:
                    continue
                for alpha_min in alpha_mins:
                    config_idx += 1

                    print(f"\n[{config_idx}/{total_configs}] C={C}, d={d}, r={r}, alpha_min={alpha_min}")
                    print("-" * 60)

                    try:
                        if use_batched:
                            result = benchmark_batched_config(
                                C=C, d_k=d_k, d_v=d_v, rank=r,
                                alpha_min=alpha_min,
                                batch_size=batch_size,
                                n_warmup=n_warmup, n_trials=n_trials,
                                device=device, dtype=dtype,
                            )
                        else:
                            result = benchmark_single_config(
                                C=C, d_k=d_k, d_v=d_v, rank=r,
                                batch_size=min(batch_size, 16),
                                n_warmup=n_warmup, n_trials=n_trials,
                                device=device, dtype=dtype,
                            )

                        result["C"] = C
                        result["d_k"] = d_k
                        result["d_v"] = d_v
                        result["rank"] = r
                        result["alpha_min"] = alpha_min
                        result["rho"] = d / C  # dimension ratio

                        all_results.append(result)

                        # Print summary
                        print(f"  Path 1: {result['path1_per_call_us']:.1f} us/call")
                        print(f"  Path 2: {result['path2_per_call_us']:.1f} us/call")
                        print(f"  Speedup: {result['speedup']:.3f}x ({result['path2_faster_pct']:+.1f}%)")
                        print(f"  Rel error: mean={result['rel_error_mean']:.2e}, max={result['rel_error_max']:.2e}")
                        print(f"  Numerical pass: {'YES' if result['numerical_pass'] else 'NO'}")
                        print(f"  Effective mask rank: {result['effective_rank_mean']:.1f}")
                        flops = result["theoretical_flops"]
                        print(f"  Theoretical FLOP ratio (P2/P1): {flops['ratio']:.3f}")

                        # Track best
                        if result["speedup"] > best_speedup:
                            best_speedup = result["speedup"]
                            best_config = {"C": C, "d": d, "r": r, "alpha_min": alpha_min}

                        if result["speedup"] > 1.10:
                            any_path2_wins = True
                            print(f"  *** PATH 2 WINS (>10% faster) ***")

                        # Log to wandb
                        if HAS_WANDB and wandb.run is not None:
                            wandb.log({
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/path1_us": result["path1_per_call_us"],
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/path2_us": result["path2_per_call_us"],
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/speedup": result["speedup"],
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/rel_error_max": result["rel_error_max"],
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/effective_rank": result["effective_rank_mean"],
                                f"benchmark/C{C}_d{d}_r{r}_a{alpha_min}/flop_ratio": flops["ratio"],
                                "config_idx": config_idx,
                            })

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "C": C, "d_k": d_k, "d_v": d_v, "rank": r,
                            "alpha_min": alpha_min, "error": str(e),
                        })

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nBest speedup: {best_speedup:.3f}x at config {best_config}")
    print(f"Path 2 wins (>10%): {'YES' if any_path2_wins else 'NO'}")

    # Print table of all results
    print(f"\n{'C':>4} {'d':>4} {'r':>4} {'alpha':>6} {'rho':>5} {'P1(us)':>8} {'P2(us)':>8} {'Speed':>7} {'ErrMax':>8} {'EffRank':>8} {'FLOPr':>6}")
    print("-" * 85)
    for r in all_results:
        if "error" in r:
            print(f"{r['C']:>4} {r['d_k']:>4} {r['rank']:>4}   ERROR: {r['error'][:40]}")
            continue
        print(
            f"{r['C']:>4} {r['d_k']:>4} {r['rank']:>4} "
            f"{r.get('alpha_min', 0.9):>6.3f} "
            f"{r.get('rho', 0):>5.1f} "
            f"{r['path1_per_call_us']:>8.1f} "
            f"{r['path2_per_call_us']:>8.1f} "
            f"{r['speedup']:>7.3f} "
            f"{r['rel_error_max']:>8.2e} "
            f"{r['effective_rank_mean']:>8.1f} "
            f"{r['theoretical_flops']['ratio']:>6.2f}"
        )

    # Success criteria evaluation
    success_speed = any_path2_wins
    success_numerical = all(
        r.get("numerical_pass", False) for r in all_results if "error" not in r and r.get("rank", 0) <= 8
    )

    print(f"\n{'='*80}")
    print(f"SUCCESS CRITERIA")
    print(f"{'='*80}")
    print(f"  Path 2 >= 10% faster for any realistic config: {'PASS' if success_speed else 'FAIL'}")
    print(f"  Numerical agreement < 1e-3 for r<=8: {'PASS' if success_numerical else 'FAIL'}")
    print(f"  Overall: {'PASS' if (success_speed and success_numerical) else 'FAIL'}")

    # Log final results to wandb
    if HAS_WANDB and wandb.run is not None:
        wandb.log({
            "final/best_speedup": best_speedup,
            "final/best_config_C": best_config["C"] if best_config else -1,
            "final/best_config_d": best_config["d"] if best_config else -1,
            "final/best_config_r": best_config["r"] if best_config else -1,
            "success_criteria/path2_10pct_faster": success_speed,
            "success_criteria/numerical_agreement": success_numerical,
            "success_criteria/overall": success_speed and success_numerical,
        })

    return {
        "all_results": all_results,
        "best_speedup": best_speedup,
        "best_config": best_config,
        "success_speed": success_speed,
        "success_numerical": success_numerical,
        "overall_pass": success_speed and success_numerical,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 042: GLA Contraction Ordering")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {config_path} not found, using defaults")
        config = {}

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="mad-architecture-search",
            name="exp-042-gla-contraction-ordering",
            config=config,
            tags=["exp-042", "kernel-benchmark", "contraction-ordering", "gla"],
        )
        wandb_url = wandb.run.get_url()
        print(f"Wandb URL: {wandb_url}")
    else:
        wandb_url = "N/A"

    # Run the sweep
    results = run_full_sweep(config)

    # Save results to file
    output_path = Path("results.json")
    # Convert non-serializable items
    serializable_results = {
        "best_speedup": results["best_speedup"],
        "best_config": results["best_config"],
        "success_speed": results["success_speed"],
        "success_numerical": results["success_numerical"],
        "overall_pass": results["overall_pass"],
        "wandb_url": wandb_url,
        "configs": [],
    }
    for r in results["all_results"]:
        entry = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, dict, list, type(None))):
                entry[k] = v
        serializable_results["configs"].append(entry)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if HAS_WANDB and wandb.run is not None:
        wandb.save(str(output_path))
        wandb.finish()

    return results


if __name__ == "__main__":
    main()

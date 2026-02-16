"""
Benchmark script for Experiment 040: Persistent Megakernel Fusion MVE

Compares 4 approaches:
  1. PyTorch baseline (3 separate CUDA kernels via native ops)
  2. Triton baseline (3 separate Triton kernels: matmul + scan + silu_gate)
  3. Semi-fused: cuBLAS GEMM + fused scan+gate Triton kernel (2 launches)
  4. Full-fused: all 3 ops in 1 Triton kernel (1 launch, no tensor cores)

The key insight being tested: fusing scan+gate eliminates one HBM intermediate
(scan_output), which is the most practical fusion target. Full fusion loses
tensor core utilization on the GEMM and is expected to be slower.

From proposal 040 MVE:
  - Single-head, d=256, d_v=64, T=2048, B=4
  - Forward-pass only
  - Success: >1.3x throughput for fused vs baseline, numerical correctness
"""

import argparse
import yaml
import torch
import wandb

from models.baseline_kernels import pytorch_baseline_forward, triton_baseline_forward
from models.fused_megakernel import triton_fused_forward, triton_fused_forward_v2


def generate_inputs(B, T, d, d_v, dtype=torch.float16, device="cuda"):
    """Generate synthetic inputs for the benchmark."""
    x = torch.randn(B, T, d, device=device, dtype=dtype)
    W_V = torch.randn(d, d_v, device=device, dtype=dtype) * (d ** -0.5)
    gamma = torch.sigmoid(torch.randn(B, T, d_v, device=device, dtype=dtype))
    gate = torch.randn(B, T, d_v, device=device, dtype=dtype)
    return x, W_V, gamma, gate


def warmup(fn, args, n_warmup=10):
    """Warmup GPU caches and JIT compilation."""
    for _ in range(n_warmup):
        _ = fn(*args)
    torch.cuda.synchronize()


def benchmark_fn(fn, args, n_iters=100, label=""):
    """Benchmark a function with CUDA events for accurate timing."""
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    for i in range(n_iters):
        start_events[i].record()
        _ = fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    std_ms = (sum((t - avg_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

    print(f"  {label}: avg={avg_ms:.4f}ms, min={min_ms:.4f}ms, max={max_ms:.4f}ms, std={std_ms:.4f}ms")
    return {"avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms, "std_ms": std_ms}


def check_correctness(ref_out, test_out, label, dtype):
    """Check numerical correctness against reference."""
    max_err = (ref_out.float() - test_out.float()).abs().max().item()
    mean_err = (ref_out.float() - test_out.float()).abs().mean().item()
    ref_max = ref_out.float().abs().max().item() + 1e-8
    rel_err = max_err / ref_max
    tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    passed = max_err < tol * ref_max + 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  {label}: max_err={max_err:.6f}, mean_err={mean_err:.6f}, rel_err={rel_err:.6f} [{status}]")
    return {"max_err": max_err, "mean_err": mean_err, "rel_err": rel_err, "passed": passed}


def run_benchmark(config: dict):
    """Run the full benchmark suite."""
    B = config.get("B", 4)
    T = config.get("T", 2048)
    d = config.get("d", 256)
    d_v = config.get("d_v", 64)
    dtype_str = config.get("dtype", "float16")
    n_warmup = config.get("n_warmup", 20)
    n_iters = config.get("n_iters", 200)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA required!")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*70}")
    print(f"Experiment 040: Persistent Megakernel Fusion MVE (v2)")
    print(f"{'='*70}")
    print(f"GPU: {gpu_name}")
    print(f"Config: B={B}, T={T}, d={d}, d_v={d_v}, dtype={dtype_str}")
    print(f"Iterations: {n_warmup} warmup + {n_iters} benchmark")
    print(f"{'='*70}\n")

    wandb.init(
        project="mad-architecture-search",
        name=f"exp-040-v2-megakernel-{gpu_name.replace(' ', '-')}",
        config={
            "experiment": 40,
            "version": 2,
            "B": B, "T": T, "d": d, "d_v": d_v,
            "dtype": dtype_str,
            "gpu": gpu_name,
            "proposal_id": "040-persistent-megakernel-linear-rnn-layer",
        }
    )
    print(f"Wandb URL: {wandb.run.url}\n")

    x, W_V, gamma, gate = generate_inputs(B, T, d, d_v, dtype=dtype, device=device)
    print(f"Inputs: x={x.shape}, W_V={W_V.shape}, gamma={gamma.shape}, gate={gate.shape}\n")

    # ============================================================
    # 1. PyTorch baseline (reference for correctness)
    # ============================================================
    print("1. PyTorch baseline (correctness reference):")
    ref_out = pytorch_baseline_forward(x, W_V, gamma, gate)
    warmup(pytorch_baseline_forward, (x, W_V, gamma, gate), n_warmup=3)
    pytorch_times = benchmark_fn(pytorch_baseline_forward, (x, W_V, gamma, gate),
                                 n_iters=min(n_iters, 10), label="PyTorch")
    print()

    # ============================================================
    # 2. Triton baseline (3 separate kernels)
    # ============================================================
    print("2. Triton baseline (3 separate kernels):")
    triton_out = triton_baseline_forward(x, W_V, gamma, gate)
    check_correctness(ref_out, triton_out, "vs PyTorch", dtype)
    warmup(triton_baseline_forward, (x, W_V, gamma, gate), n_warmup=n_warmup)
    triton_times = benchmark_fn(triton_baseline_forward, (x, W_V, gamma, gate),
                                n_iters=n_iters, label="Triton 3-kernel")
    print()

    # ============================================================
    # 3. Semi-fused: cuBLAS GEMM + fused scan+gate (2 kernels)
    # This is the most practical fusion approach.
    # ============================================================
    print("3. Semi-fused: cuBLAS GEMM + fused scan+gate (2 kernels):")
    semi_out = triton_fused_forward_v2(x, W_V, gamma, gate)
    corr_semi = check_correctness(ref_out, semi_out, "vs PyTorch", dtype)
    warmup(triton_fused_forward_v2, (x, W_V, gamma, gate), n_warmup=n_warmup)
    semi_times = benchmark_fn(triton_fused_forward_v2, (x, W_V, gamma, gate),
                              n_iters=n_iters, label="Semi-fused")
    print()

    # ============================================================
    # 4. Full-fused: all 3 ops in 1 kernel (no tensor cores on GEMM)
    # ============================================================
    print("4. Full-fused megakernel (1 kernel, per-timestep GEMM):")
    full_out = triton_fused_forward(x, W_V, gamma, gate)
    corr_full = check_correctness(ref_out, full_out, "vs PyTorch", dtype)
    warmup(triton_fused_forward, (x, W_V, gamma, gate), n_warmup=n_warmup)
    full_times = benchmark_fn(triton_fused_forward, (x, W_V, gamma, gate),
                              n_iters=n_iters, label="Full-fused")
    print()

    # ============================================================
    # Results summary
    # ============================================================
    tokens = B * T
    results = {
        "pytorch": {"ms": pytorch_times["avg_ms"], "tps": tokens / (pytorch_times["avg_ms"] / 1000)},
        "triton_3k": {"ms": triton_times["avg_ms"], "tps": tokens / (triton_times["avg_ms"] / 1000)},
        "semi_fused": {"ms": semi_times["avg_ms"], "tps": tokens / (semi_times["avg_ms"] / 1000)},
        "full_fused": {"ms": full_times["avg_ms"], "tps": tokens / (full_times["avg_ms"] / 1000)},
    }

    speedup_semi = triton_times["avg_ms"] / semi_times["avg_ms"]
    speedup_full = triton_times["avg_ms"] / full_times["avg_ms"]

    print(f"{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<45} {'Avg (ms)':<12} {'Tokens/s':<15} {'vs 3-kern':<10}")
    print(f"{'-'*80}")
    print(f"{'1. PyTorch (reference, 3 sep kernels)':<45} {pytorch_times['avg_ms']:<12.4f} {results['pytorch']['tps']:<15.0f} {'N/A':<10}")
    print(f"{'2. Triton (3 separate kernels)':<45} {triton_times['avg_ms']:<12.4f} {results['triton_3k']['tps']:<15.0f} {'1.00x':<10}")
    print(f"{'3. Semi-fused (cuBLAS + scan+gate, 2 kern)':<45} {semi_times['avg_ms']:<12.4f} {results['semi_fused']['tps']:<15.0f} {f'{speedup_semi:.2f}x':<10}")
    print(f"{'4. Full-fused (1 kernel, no tensor cores)':<45} {full_times['avg_ms']:<12.4f} {results['full_fused']['tps']:<15.0f} {f'{speedup_full:.2f}x':<10}")
    print(f"{'='*80}")
    print()

    # Success criteria
    # Use best fusion approach (semi-fused is the practical one)
    best_fused_speedup = max(speedup_semi, speedup_full)
    best_fused_name = "semi-fused" if speedup_semi >= speedup_full else "full-fused"
    criterion_1 = best_fused_speedup > 1.3
    criterion_2 = corr_semi["passed"] and corr_full["passed"]

    print("SUCCESS CRITERIA (from proposal 040 MVE):")
    print(f"  1. Fused >1.3x throughput vs 3-kernel: best={best_fused_speedup:.2f}x ({best_fused_name}) {'PASS' if criterion_1 else 'FAIL'}")
    print(f"  2. Numerical correctness: {'PASS' if criterion_2 else 'FAIL'}")
    print()

    # Log to wandb
    wandb.log({
        "benchmark/pytorch_avg_ms": pytorch_times["avg_ms"],
        "benchmark/triton_3kernel_avg_ms": triton_times["avg_ms"],
        "benchmark/triton_3kernel_min_ms": triton_times["min_ms"],
        "benchmark/semi_fused_avg_ms": semi_times["avg_ms"],
        "benchmark/semi_fused_min_ms": semi_times["min_ms"],
        "benchmark/full_fused_avg_ms": full_times["avg_ms"],
        "benchmark/full_fused_min_ms": full_times["min_ms"],
        "benchmark/speedup_semi_vs_triton": speedup_semi,
        "benchmark/speedup_full_vs_triton": speedup_full,
        "correctness/semi_passed": corr_semi["passed"],
        "correctness/semi_max_err": corr_semi["max_err"],
        "correctness/full_passed": corr_full["passed"],
        "correctness/full_max_err": corr_full["max_err"],
        "success_criteria/throughput_1_3x": criterion_1,
        "success_criteria/numerical_correctness": criterion_2,
    })

    # ============================================================
    # Size sweep: semi-fused vs triton 3-kernel
    # ============================================================
    print("\nSIZE SWEEP: Semi-fused vs Triton 3-kernel:")
    print(f"{'B':<6} {'T':<8} {'Triton 3k (ms)':<18} {'Semi-fused (ms)':<18} {'Speedup':<10}")
    print(f"{'-'*60}")

    sweep_configs = [
        (1, 512), (1, 1024), (1, 2048), (1, 4096),
        (4, 512), (4, 1024), (4, 2048), (4, 4096),
        (8, 512), (8, 1024), (8, 2048),
        (16, 512), (16, 1024), (16, 2048),
    ]

    for sweep_B, sweep_T in sweep_configs:
        try:
            sx, sW, sg, sgate = generate_inputs(sweep_B, sweep_T, d, d_v, dtype=dtype, device=device)
            warmup(triton_baseline_forward, (sx, sW, sg, sgate), n_warmup=5)
            warmup(triton_fused_forward_v2, (sx, sW, sg, sgate), n_warmup=5)

            st = benchmark_fn(triton_baseline_forward, (sx, sW, sg, sgate),
                              n_iters=50, label=f"B={sweep_B},T={sweep_T} triton")
            ft = benchmark_fn(triton_fused_forward_v2, (sx, sW, sg, sgate),
                              n_iters=50, label=f"B={sweep_B},T={sweep_T} semi-fused")

            sp = st["avg_ms"] / ft["avg_ms"]
            print(f"{sweep_B:<6} {sweep_T:<8} {st['avg_ms']:<18.4f} {ft['avg_ms']:<18.4f} {sp:<10.2f}x")

            wandb.log({
                f"sweep/B{sweep_B}_T{sweep_T}_triton_ms": st["avg_ms"],
                f"sweep/B{sweep_B}_T{sweep_T}_semi_fused_ms": ft["avg_ms"],
                f"sweep/B{sweep_B}_T{sweep_T}_speedup": sp,
            })
        except Exception as e:
            print(f"{sweep_B:<6} {sweep_T:<8} ERROR: {e}")

    # Final decision
    overall_pass = criterion_1 and criterion_2
    decision = "PROCEED" if overall_pass else ("DEBUG" if criterion_2 else "ABANDON")

    wandb.log({
        "final/overall_pass": overall_pass,
        "final/decision": decision,
        "final/best_speedup": best_fused_speedup,
        "final/best_approach": best_fused_name,
    })

    print(f"\n{'='*70}")
    print(f"DECISION: {decision}")
    print(f"Best fusion approach: {best_fused_name} ({best_fused_speedup:.2f}x)")
    if overall_pass:
        print("Success criteria met. Proceed to full layer fusion.")
    elif criterion_2:
        print(f"Correctness OK but speedup ({best_fused_speedup:.2f}x) < 1.3x target.")
        print("Analysis: The scan operation is inherently sequential over T, so the")
        print("fusion benefit depends on whether scan+gate HBM savings outweigh")
        print("the overhead. At small d_v=64, the intermediate is tiny (B*T*d_v*2 bytes).")
        print("Fusion may show more benefit at larger d_v or with more intermediates.")
    else:
        print("Numerical correctness failed.")
    print(f"{'='*70}")

    wandb.finish()
    return {"speedup_semi": speedup_semi, "speedup_full": speedup_full,
            "best_speedup": best_fused_speedup, "correctness": criterion_2,
            "overall_pass": overall_pass, "decision": decision}


def main():
    parser = argparse.ArgumentParser(description="Experiment 040: Megakernel Fusion MVE")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results = run_benchmark(config.get("benchmark", {}))
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()

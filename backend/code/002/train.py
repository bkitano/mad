"""
MVE 002: SSD-DeltaNet Block Decomposition Benchmark

Pure forward pass throughput benchmark comparing:
1. Naive WY: Sequential delta rule (T sequential matvecs)
2. Block-SSD: Sub-block decomposition (T/Q sub-blocks × Q sequential steps + matmuls)

The Block-SSD approach replaces T separate d×d matvecs with T/Q batched
(Q,d)×(d,d) matmuls + T sequential steps within small sub-blocks.

Success Criteria (from proposal):
1. Speedup > 1.3× (block vs naive)
2. Numerical error ||y_naive - y_block||_inf < 1e-5
3. Matmul fraction > 60% of compute time in matmul ops

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import os
import time
import yaml
import torch
import torch.profiler


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_fn(fn, *args, warmup: int = 10, n_iter: int = 100, **kwargs):
    """Benchmark a function with warmup and timing."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_iter


def count_matmul_fraction(fn, *args, **kwargs):
    """Profile matmul fraction using PyTorch profiler."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn(*args, **kwargs)

    matmul_time = 0
    total_time = 0
    matmul_keywords = ['mm', 'matmul', 'bmm', 'gemm', 'addmm', 'linear', 'dot']

    events = prof.key_averages()
    for event in events:
        name = event.key.lower()
        # Try CUDA time first, fall back to CPU time
        work = getattr(event, 'self_cuda_time_total', None)
        if work is None or work == 0:
            work = event.self_cpu_time_total

        is_matmul = any(kw in name for kw in matmul_keywords)
        if is_matmul:
            matmul_time += work
        total_time += work

    fraction = matmul_time / total_time if total_time > 0 else 0.0
    return fraction, events


def run_accuracy_test(naive_fn, block_fn, T, d, chunk_size, sub_block,
                      n_trials=10, device='cpu'):
    """Test numerical equivalence."""
    max_errors = []

    for trial in range(n_trials):
        torch.manual_seed(42 + trial)
        K = torch.randn(T, d, device=device)
        V = torch.randn(T, d, device=device)
        beta = torch.sigmoid(torch.randn(T, device=device))
        Q = torch.randn(T, d, device=device)
        K = K / K.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            y_naive = naive_fn(K, V, beta, Q)
            y_block = block_fn(K, V, beta, Q, chunk_size=chunk_size, sub_block=sub_block)

        max_errors.append((y_naive - y_block).abs().max().item())

    return {
        "max_error": max(max_errors),
        "mean_error": sum(max_errors) / len(max_errors),
    }


def run_speedup_benchmark(naive_fn, block_fn, T, d, chunk_size, sub_block,
                           warmup=10, n_iter=100, device='cpu'):
    """Benchmark speedup."""
    torch.manual_seed(42)
    K = torch.randn(T, d, device=device)
    V = torch.randn(T, d, device=device)
    beta = torch.sigmoid(torch.randn(T, device=device))
    Q = torch.randn(T, d, device=device)
    K = K / K.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        naive_time = benchmark_fn(naive_fn, K, V, beta, Q,
                                   warmup=warmup, n_iter=n_iter)
        block_time = benchmark_fn(block_fn, K, V, beta, Q,
                                   chunk_size=chunk_size, sub_block=sub_block,
                                   warmup=warmup, n_iter=n_iter)

    return {
        "naive_time_ms": naive_time * 1000,
        "block_time_ms": block_time * 1000,
        "speedup": naive_time / block_time,
    }


def run_scaling_test(naive_fn, block_fn, d, chunk_size, sub_block,
                     T_values, warmup=5, n_iter=50, device='cpu'):
    """Test speedup scaling with sequence length."""
    results = []
    for T in T_values:
        torch.manual_seed(42)
        K = torch.randn(T, d, device=device)
        V = torch.randn(T, d, device=device)
        beta = torch.sigmoid(torch.randn(T, device=device))
        Q = torch.randn(T, d, device=device)
        K = K / K.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            nt = benchmark_fn(naive_fn, K, V, beta, Q,
                               warmup=warmup, n_iter=n_iter)
            bt = benchmark_fn(block_fn, K, V, beta, Q,
                               chunk_size=chunk_size, sub_block=sub_block,
                               warmup=warmup, n_iter=n_iter)

        results.append({"T": T, "naive_ms": nt * 1000, "block_ms": bt * 1000,
                        "speedup": nt / bt})
    return results


def run_subblock_sweep(naive_fn, block_fn, T, d, chunk_size,
                        sb_values, warmup=5, n_iter=50, device='cpu'):
    """Sweep sub-block sizes."""
    torch.manual_seed(42)
    K = torch.randn(T, d, device=device)
    V = torch.randn(T, d, device=device)
    beta = torch.sigmoid(torch.randn(T, device=device))
    Q = torch.randn(T, d, device=device)
    K = K / K.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        nt = benchmark_fn(naive_fn, K, V, beta, Q, warmup=warmup, n_iter=n_iter)

    results = []
    for sb in sb_values:
        with torch.no_grad():
            bt = benchmark_fn(block_fn, K, V, beta, Q,
                               chunk_size=chunk_size, sub_block=sb,
                               warmup=warmup, n_iter=n_iter)
            y_n = naive_fn(K, V, beta, Q)
            y_b = block_fn(K, V, beta, Q, chunk_size=chunk_size, sub_block=sb)
            err = (y_n - y_b).abs().max().item()

        results.append({"sub_block": sb, "naive_ms": nt * 1000, "block_ms": bt * 1000,
                        "speedup": nt / bt, "error": err})
    return results


def main():
    parser = argparse.ArgumentParser(description="MVE 002: SSD-DeltaNet Benchmark")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    bench = config.get("benchmark", {})

    T = bench.get("seq_len", 512)
    d = bench.get("state_dim", 64)
    chunk_size = bench.get("chunk_size", 64)
    sub_block = bench.get("sub_block", 16)
    warmup = bench.get("warmup", 10)
    n_iter = bench.get("n_iter", 100)
    n_acc_trials = bench.get("n_accuracy_trials", 10)

    device_name = bench.get("device", "auto")
    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    print("=" * 60)
    print("MVE 002: SSD-DeltaNet Block Decomposition Benchmark")
    print("=" * 60)
    print(f"  T={T}, d={d}, C={chunk_size}, Q={sub_block}")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    from models.naive_wy import naive_wy_forward
    from models.block_ssd import block_ssd_forward

    results = {"config": config, "device": device}

    # ═══════════════════════════════════════════
    # Test 1: Numerical Accuracy
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Test 1: Numerical Accuracy")
    print("=" * 60)

    acc = run_accuracy_test(naive_wy_forward, block_ssd_forward,
                            T, d, chunk_size, sub_block, n_acc_trials, device)
    passes_accuracy = acc["max_error"] < 1e-5
    print(f"  Max error: {acc['max_error']:.2e}")
    print(f"  Mean error: {acc['mean_error']:.2e}")
    print(f"  Target: < 1e-5")
    print(f"  {'✅ PASS' if passes_accuracy else '❌ FAIL'}")
    print()

    results["accuracy"] = {**acc, "passes": passes_accuracy}

    # ═══════════════════════════════════════════
    # Test 2: Speedup (primary metric)
    # ═══════════════════════════════════════════
    print("=" * 60)
    print(f"Test 2: Speedup (T={T}, d={d}, Q={sub_block})")
    print("=" * 60)

    sp = run_speedup_benchmark(naive_wy_forward, block_ssd_forward,
                                T, d, chunk_size, sub_block, warmup, n_iter, device)
    passes_speedup = sp["speedup"] > 1.3
    print(f"  Naive: {sp['naive_time_ms']:.3f} ms")
    print(f"  Block: {sp['block_time_ms']:.3f} ms")
    print(f"  Speedup: {sp['speedup']:.2f}x")
    print(f"  Target: > 1.3x")
    print(f"  {'✅ PASS' if passes_speedup else '❌ FAIL'}")
    print()

    results["speedup"] = {**sp, "passes": passes_speedup}

    # ═══════════════════════════════════════════
    # Test 3: Matmul Fraction
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Test 3: Matmul Fraction in Block-SSD")
    print("=" * 60)

    torch.manual_seed(42)
    K_prof = torch.randn(T, d, device=device)
    V_prof = torch.randn(T, d, device=device)
    beta_prof = torch.sigmoid(torch.randn(T, device=device))
    Q_prof = torch.randn(T, d, device=device)
    K_prof = K_prof / K_prof.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        mf, events = count_matmul_fraction(
            block_ssd_forward, K_prof, V_prof, beta_prof, Q_prof,
            chunk_size=chunk_size, sub_block=sub_block
        )

    passes_matmul = mf > 0.6
    print(f"  Matmul fraction: {mf:.1%}")
    print(f"  Target: > 60%")
    print(f"  {'✅ PASS' if passes_matmul else '❌ FAIL'}")

    with torch.no_grad():
        mf_naive, _ = count_matmul_fraction(
            naive_wy_forward, K_prof, V_prof, beta_prof, Q_prof
        )
    print(f"  Naive matmul fraction: {mf_naive:.1%}")
    print(f"  Block matmul fraction: {mf:.1%}")
    print()

    print("  Block-SSD top profiler events:")
    for evt in events[:15]:
        time_val = getattr(evt, 'self_cuda_time_total', None)
        if time_val is None or time_val == 0:
            time_val = evt.self_cpu_time_total
        print(f"    {evt.key[:50]:50s} count={evt.count:>5d} time={int(time_val):>8d}us")
    print()

    results["matmul_fraction"] = {"block": mf, "naive": mf_naive, "passes": passes_matmul}

    # ═══════════════════════════════════════════
    # Test 4: Sub-block Size Sweep
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Test 4: Sub-block Size Sweep")
    print("=" * 60)

    sb_values = bench.get("sb_values", [4, 8, 16, 32, 64, 128])
    sb_values = [sb for sb in sb_values if sb <= T]
    sb_results = run_subblock_sweep(naive_wy_forward, block_ssd_forward,
                                     T, d, chunk_size, sb_values,
                                     warmup=warmup // 2, n_iter=n_iter // 2, device=device)

    print(f"  {'Q':>5} | {'Naive (ms)':>12} | {'Block (ms)':>12} | {'Speedup':>8} | {'Error':>10}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")
    best_sb = None
    best_speedup = 0
    for r in sb_results:
        print(f"  {r['sub_block']:>5} | {r['naive_ms']:>12.3f} | {r['block_ms']:>12.3f} | {r['speedup']:>7.2f}x | {r['error']:>10.2e}")
        if r["speedup"] > best_speedup:
            best_speedup = r["speedup"]
            best_sb = r["sub_block"]
    print(f"\n  Best sub-block: Q={best_sb} (speedup={best_speedup:.2f}x)")
    print()

    results["sb_sweep"] = sb_results

    # ═══════════════════════════════════════════
    # Test 5: Sequence Length Scaling
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Test 5: Sequence Length Scaling")
    print("=" * 60)

    T_values = bench.get("T_values", [128, 256, 512, 1024])
    optimal_sb = best_sb if best_sb else sub_block
    sc_results = run_scaling_test(naive_wy_forward, block_ssd_forward,
                                   d, chunk_size, optimal_sb, T_values,
                                   warmup=warmup // 2, n_iter=n_iter // 2, device=device)

    print(f"  (Using Q={optimal_sb})")
    print(f"  {'T':>6} | {'Naive (ms)':>12} | {'Block (ms)':>12} | {'Speedup':>8}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for r in sc_results:
        print(f"  {r['T']:>6} | {r['naive_ms']:>12.3f} | {r['block_ms']:>12.3f} | {r['speedup']:>7.2f}x")
    print()

    results["scaling"] = sc_results

    # ═══════════════════════════════════════════
    # Test 6: Dimension Scaling
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Test 6: Dimension Scaling")
    print("=" * 60)

    d_values = bench.get("d_values", [32, 64, 128, 256])
    dim_results = []
    for d_test in d_values:
        torch.manual_seed(42)
        K_t = torch.randn(T, d_test, device=device)
        V_t = torch.randn(T, d_test, device=device)
        beta_t = torch.sigmoid(torch.randn(T, device=device))
        Q_t = torch.randn(T, d_test, device=device)
        K_t = K_t / K_t.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            nt = benchmark_fn(naive_wy_forward, K_t, V_t, beta_t, Q_t,
                               warmup=warmup // 2, n_iter=max(n_iter // 4, 5))
            bt = benchmark_fn(block_ssd_forward, K_t, V_t, beta_t, Q_t,
                               chunk_size=chunk_size, sub_block=optimal_sb,
                               warmup=warmup // 2, n_iter=max(n_iter // 4, 5))

        dim_results.append({"d": d_test, "naive_ms": nt * 1000, "block_ms": bt * 1000,
                           "speedup": nt / bt})

    print(f"  (Using Q={optimal_sb})")
    print(f"  {'d':>6} | {'Naive (ms)':>12} | {'Block (ms)':>12} | {'Speedup':>8}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for r in dim_results:
        print(f"  {r['d']:>6} | {r['naive_ms']:>12.3f} | {r['block_ms']:>12.3f} | {r['speedup']:>7.2f}x")
    print()

    results["dim_scaling"] = dim_results

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    criteria = [
        ("Speedup > 1.3x", passes_speedup, f"{sp['speedup']:.2f}x"),
        ("Numerical error < 1e-5", passes_accuracy, f"{acc['max_error']:.2e}"),
        ("Matmul fraction > 60%", passes_matmul, f"{mf:.1%}"),
    ]

    n_pass = sum(1 for _, p, _ in criteria if p)
    for name, passed, value in criteria:
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} — {name} (achieved: {value})")

    print(f"\n  Best sub-block: Q={best_sb} ({best_speedup:.2f}x speedup)")
    print()

    if n_pass == 3:
        decision = "PROCEED"
        reason = "All success criteria met"
    elif passes_accuracy and passes_speedup:
        decision = "PROCEED"
        reason = f"{n_pass}/3 criteria met, core claims validated"
    elif passes_accuracy:
        if best_speedup > 1.1:
            decision = "INVESTIGATE"
            reason = f"Decomposition correct but speedup only {best_speedup:.2f}x (may need GPU/custom kernels)"
        else:
            decision = "ABANDON"
            reason = "Block-SSD slower than naive (overhead > benefit)"
    else:
        if acc['max_error'] > 1e-3:
            decision = "ABANDON"
            reason = f"Numerical error {acc['max_error']:.2e} too large"
        else:
            decision = "INVESTIGATE"
            reason = f"Moderate numerical error {acc['max_error']:.2e}"

    print(f"  DECISION: {decision}")
    print(f"  Reason: {reason}")
    print()

    results["summary"] = {
        "n_pass": n_pass,
        "decision": decision,
        "reason": reason,
        "best_sub_block": best_sb,
        "best_speedup": best_speedup,
    }

    # Save results
    output_dir = os.path.dirname(os.path.abspath(args.config))
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

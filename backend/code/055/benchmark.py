"""
Benchmark harness for WY-All-Scan vs LASP-2 vs ZeCO-GLA.

This is the main executable for experiment 055.
It runs on multiple GPUs using torch.distributed (NCCL backend)
and compares the three communication primitives.

Usage:
    torchrun --nproc_per_node=N benchmark.py --config config.yaml

Where N is the number of GPUs to use (2, 4, or 8).
"""

import argparse
import os
import sys
import time
import json
import yaml
import torch
import torch.distributed as dist

# Add parent to path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.wy_allscan import wy_allscan, wy_allscan_pipelined
from models.lasp2_allgather import lasp2_allgather_scan
from models.zeco_gla_allscan import zeco_gla_allscan
from models.sequential_scan import sequential_prefix_scan, sequential_prefix_scan_diagonal


def setup_distributed():
    """Initialize torch.distributed with NCCL backend."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def generate_test_data(H, d_v, d_k, device, dtype=torch.bfloat16, seed=42):
    """
    Generate random test data for a single device.

    Returns state matrix, WY factors, and gating vector.
    Uses seed + rank for reproducibility while varying across devices.
    """
    rank = dist.get_rank()
    gen = torch.Generator(device="cpu").manual_seed(seed + rank)

    # Local final state: random (d_v, d_k)
    S_local = torch.randn(H, d_v, d_k, generator=gen).to(device=device, dtype=dtype)

    # Cumulative gating: values in (0, 1) representing decay
    gamma_dev = torch.sigmoid(torch.randn(H, d_k, generator=gen)).to(device=device, dtype=dtype)

    # WY factors: random orthogonal-ish matrices
    # Make W, K small to keep WY product well-conditioned
    W_dev = (torch.randn(H, d_k, d_k, generator=gen) * 0.1).to(device=device, dtype=dtype)
    K_dev = (torch.randn(H, d_k, d_k, generator=gen) * 0.1).to(device=device, dtype=dtype)

    return S_local, gamma_dev, W_dev, K_dev


def gather_all_data(S_local, gamma_dev, W_dev, K_dev):
    """Gather all device data to rank 0 for sequential verification."""
    world_size = dist.get_world_size()

    S_all = [torch.empty_like(S_local) for _ in range(world_size)]
    gamma_all = [torch.empty_like(gamma_dev) for _ in range(world_size)]
    W_all = [torch.empty_like(W_dev) for _ in range(world_size)]
    K_all = [torch.empty_like(K_dev) for _ in range(world_size)]

    dist.all_gather(S_all, S_local)
    dist.all_gather(gamma_all, gamma_dev)
    dist.all_gather(W_all, W_dev)
    dist.all_gather(K_all, K_dev)

    return S_all, gamma_all, W_all, K_all


def compute_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative error between two tensors."""
    a_f = a.float()
    b_f = b.float()
    denom = torch.norm(b_f)
    if denom < 1e-10:
        return torch.norm(a_f - b_f).item()
    return (torch.norm(a_f - b_f) / denom).item()


def benchmark_method(method_fn, args, num_warmup=5, num_trials=20, method_name=""):
    """Run a method with warmup and timing."""
    rank = dist.get_rank()

    # Warmup
    for _ in range(num_warmup):
        result, _ = method_fn(*args)
        dist.barrier()

    # Timed runs
    torch.cuda.synchronize()
    dist.barrier()

    latencies = []
    all_timings = []
    for trial in range(num_trials):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        result, timing = method_fn(*args)

        torch.cuda.synchronize()
        dist.barrier()
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)
        all_timings.append(timing)

    # Compute stats
    latencies_t = torch.tensor(latencies)
    stats = {
        "mean_ms": latencies_t.mean().item(),
        "std_ms": latencies_t.std().item(),
        "min_ms": latencies_t.min().item(),
        "max_ms": latencies_t.max().item(),
        "median_ms": latencies_t.median().item(),
        "p90_ms": latencies_t.quantile(0.9).item(),
    }

    # Average internal timings
    if all_timings and all_timings[0]:
        for key in all_timings[0]:
            vals = [t[key] for t in all_timings if key in t]
            stats[f"internal_{key}"] = sum(vals) / len(vals)

    return result, stats


def run_wy_correction_microbench(d_v, d_k, H, device, dtype=torch.bfloat16, num_trials=1000):
    """
    Microbenchmark the WY correction matmul alone.

    This measures the compute overhead per pipeline stage:
        S_corrected = S_recv - (S_recv @ W.T) @ K

    Success criterion: < 0.5ms on H100
    """
    S = torch.randn(d_v, d_k, device=device, dtype=dtype)
    W = torch.randn(d_k, d_k, device=device, dtype=dtype) * 0.1
    K = torch.randn(d_k, d_k, device=device, dtype=dtype) * 0.1

    # Warmup
    for _ in range(50):
        tmp = S @ W.T
        result = S - tmp @ K
    torch.cuda.synchronize()

    # Time single-head correction
    latencies = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tmp = S @ W.T
        result = S - tmp @ K
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    # Time H-head batch correction
    S_batch = torch.randn(H, d_v, d_k, device=device, dtype=dtype)
    W_batch = torch.randn(H, d_k, d_k, device=device, dtype=dtype) * 0.1
    K_batch = torch.randn(H, d_k, d_k, device=device, dtype=dtype) * 0.1

    for _ in range(50):
        tmp = torch.bmm(S_batch, W_batch.transpose(1, 2))
        result_batch = S_batch - torch.bmm(tmp, K_batch)
    torch.cuda.synchronize()

    batch_latencies = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tmp = torch.bmm(S_batch, W_batch.transpose(1, 2))
        result_batch = S_batch - torch.bmm(tmp, K_batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000)

    lat_t = torch.tensor(latencies)
    blat_t = torch.tensor(batch_latencies)

    return {
        "single_head_mean_ms": lat_t.mean().item(),
        "single_head_median_ms": lat_t.median().item(),
        "single_head_min_ms": lat_t.min().item(),
        "batch_heads_mean_ms": blat_t.mean().item(),
        "batch_heads_median_ms": blat_t.median().item(),
        "batch_heads_min_ms": blat_t.min().item(),
        "H": H,
        "d_v": d_v,
        "d_k": d_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark WY-All-Scan vs LASP-2 vs ZeCO-GLA")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup distributed
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    # Extract config
    d_k = config.get("d_k", 128)
    d_v = config.get("d_v", 128)
    H = config.get("H", 16)
    num_warmup = config.get("num_warmup", 5)
    num_trials = config.get("num_trials", 20)
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Experiment 055: WY-All-Scan Microbenchmark")
        print(f"{'='*70}")
        print(f"World size (P): {world_size}")
        print(f"d_k={d_k}, d_v={d_v}, H={H}, dtype={dtype_str}")
        print(f"Warmup: {num_warmup}, Trials: {num_trials}")

        # Compute communication volumes
        wy_vol = H * (d_v * d_k + 2 * d_k * d_k + d_k) * 2  # bytes (bf16)
        zeco_vol = H * d_v * d_k * 2
        lasp2_vol = world_size * H * (d_v * d_k + 2 * d_k * d_k + d_k) * 2

        print(f"\nCommunication volumes (bytes per device):")
        print(f"  WY-All-Scan:  {wy_vol:,} ({wy_vol/1024:.1f} KB)")
        print(f"  ZeCO-GLA:     {zeco_vol:,} ({zeco_vol/1024:.1f} KB)")
        print(f"  LASP-2:       {lasp2_vol:,} ({lasp2_vol/1024:.1f} KB)")
        print(f"  WY/ZeCO ratio: {wy_vol/zeco_vol:.2f}x")
        print(f"  LASP2/WY ratio: {lasp2_vol/wy_vol:.2f}x")

    # Initialize wandb on rank 0
    wandb_url = None
    try:
        import wandb
        if rank == 0:
            wandb.init(
                project="mad-architecture-search",
                name=f"exp-055-wy-allscan-P{world_size}",
                config={
                    "experiment": 55,
                    "proposal": "055-zeco-allscan-gated-deltanet-sp",
                    "d_k": d_k,
                    "d_v": d_v,
                    "H": H,
                    "world_size": world_size,
                    "dtype": dtype_str,
                    "num_warmup": num_warmup,
                    "num_trials": num_trials,
                },
            )
            wandb_url = wandb.run.get_url()
            print(f"\nWandb URL: {wandb_url}")
    except ImportError:
        if rank == 0:
            print("WARNING: wandb not available, skipping logging")
        wandb = None

    # --- Generate test data ---
    S_local, gamma_dev, W_dev, K_dev = generate_test_data(H, d_v, d_k, device, dtype)

    if rank == 0:
        print(f"\nTest data shapes:")
        print(f"  S_local: {S_local.shape}")
        print(f"  gamma_dev: {gamma_dev.shape}")
        print(f"  W_dev: {W_dev.shape}")
        print(f"  K_dev: {K_dev.shape}")

    # =================================================================
    # 1. WY Correction Matmul Microbenchmark
    # =================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("1. WY Correction Matmul Microbenchmark (single device)")
        print(f"{'='*70}")

    wy_micro = run_wy_correction_microbench(d_v, d_k, H, device, dtype)
    if rank == 0:
        print(f"  Single head: {wy_micro['single_head_mean_ms']:.4f} ms (mean), "
              f"{wy_micro['single_head_median_ms']:.4f} ms (median)")
        print(f"  {H}-head batch: {wy_micro['batch_heads_mean_ms']:.4f} ms (mean), "
              f"{wy_micro['batch_heads_median_ms']:.4f} ms (median)")
        print(f"  Success criterion (< 0.5 ms per stage): "
              f"{'PASS' if wy_micro['batch_heads_median_ms'] < 0.5 else 'FAIL'}")

    # =================================================================
    # 2. Run WY-All-Scan
    # =================================================================
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*70}")
        print("2. WY-All-Scan (Gated DeltaNet)")
        print(f"{'='*70}")

    wy_result, wy_stats = benchmark_method(
        wy_allscan,
        (S_local, gamma_dev, W_dev, K_dev),
        num_warmup=num_warmup,
        num_trials=num_trials,
        method_name="wy_allscan",
    )

    if rank == 0:
        print(f"  Mean latency:   {wy_stats['mean_ms']:.3f} ms")
        print(f"  Median latency: {wy_stats['median_ms']:.3f} ms")
        print(f"  Min latency:    {wy_stats['min_ms']:.3f} ms")
        print(f"  P90 latency:    {wy_stats['p90_ms']:.3f} ms")
        if 'internal_comm_ms' in wy_stats:
            print(f"  Internal comm:  {wy_stats['internal_comm_ms']:.3f} ms")
            print(f"  Internal compute: {wy_stats['internal_compute_ms']:.3f} ms")

    # =================================================================
    # 3. Run WY-All-Scan (Pipelined variant)
    # =================================================================
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*70}")
        print("3. WY-All-Scan Pipelined (Gated DeltaNet)")
        print(f"{'='*70}")

    wy_pipe_result, wy_pipe_stats = benchmark_method(
        wy_allscan_pipelined,
        (S_local, gamma_dev, W_dev, K_dev),
        num_warmup=num_warmup,
        num_trials=num_trials,
        method_name="wy_allscan_pipelined",
    )

    if rank == 0:
        print(f"  Mean latency:   {wy_pipe_stats['mean_ms']:.3f} ms")
        print(f"  Median latency: {wy_pipe_stats['median_ms']:.3f} ms")
        print(f"  Min latency:    {wy_pipe_stats['min_ms']:.3f} ms")
        print(f"  P90 latency:    {wy_pipe_stats['p90_ms']:.3f} ms")

    # =================================================================
    # 4. Run ZeCO-GLA All-Scan (diagonal baseline)
    # =================================================================
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*70}")
        print("4. ZeCO-GLA All-Scan (Diagonal baseline)")
        print(f"{'='*70}")

    zeco_result, zeco_stats = benchmark_method(
        zeco_gla_allscan,
        (S_local, gamma_dev),
        num_warmup=num_warmup,
        num_trials=num_trials,
        method_name="zeco_gla",
    )

    if rank == 0:
        print(f"  Mean latency:   {zeco_stats['mean_ms']:.3f} ms")
        print(f"  Median latency: {zeco_stats['median_ms']:.3f} ms")
        print(f"  Min latency:    {zeco_stats['min_ms']:.3f} ms")
        print(f"  P90 latency:    {zeco_stats['p90_ms']:.3f} ms")

    # =================================================================
    # 5. Run LASP-2 AllGather
    # =================================================================
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*70}")
        print("5. LASP-2 AllGather (baseline)")
        print(f"{'='*70}")

    lasp2_result, lasp2_stats = benchmark_method(
        lasp2_allgather_scan,
        (S_local, gamma_dev, W_dev, K_dev),
        num_warmup=num_warmup,
        num_trials=num_trials,
        method_name="lasp2",
    )

    if rank == 0:
        print(f"  Mean latency:   {lasp2_stats['mean_ms']:.3f} ms")
        print(f"  Median latency: {lasp2_stats['median_ms']:.3f} ms")
        print(f"  Min latency:    {lasp2_stats['min_ms']:.3f} ms")
        print(f"  P90 latency:    {lasp2_stats['p90_ms']:.3f} ms")
        if 'internal_comm_ms' in lasp2_stats:
            print(f"  Internal comm:  {lasp2_stats['internal_comm_ms']:.3f} ms")
            print(f"  Internal compute: {lasp2_stats['internal_compute_ms']:.3f} ms")

    # =================================================================
    # 6. Numerical Verification
    # =================================================================
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*70}")
        print("6. Numerical Verification")
        print(f"{'='*70}")

    # Gather all data to rank 0 for sequential reference computation
    S_all, gamma_all, W_all, K_all = gather_all_data(S_local, gamma_dev, W_dev, K_dev)

    if rank == 0:
        # Compute sequential reference (ground truth) in FP32
        S_list = [s.float() for s in S_all]
        gamma_list = [g.float() for g in gamma_all]
        W_list = [w.float() for w in W_all]
        K_list = [k.float() for k in K_all]

        # WY reference
        ref_states_wy = []
        for h in range(H):
            s_per_head = [S_list[p][h] for p in range(world_size)]
            g_per_head = [gamma_list[p][h] for p in range(world_size)]
            w_per_head = [W_list[p][h] for p in range(world_size)]
            k_per_head = [K_list[p][h] for p in range(world_size)]
            ref_h = sequential_prefix_scan(s_per_head, g_per_head, w_per_head, k_per_head)
            ref_states_wy.append(ref_h)

        # Diagonal reference
        ref_states_diag = []
        for h in range(H):
            s_per_head = [S_list[p][h] for p in range(world_size)]
            g_per_head = [gamma_list[p][h] for p in range(world_size)]
            ref_h = sequential_prefix_scan_diagonal(s_per_head, g_per_head)
            ref_states_diag.append(ref_h)

    # Gather distributed results to rank 0
    wy_results_all = [torch.empty_like(wy_result) for _ in range(world_size)]
    wy_pipe_results_all = [torch.empty_like(wy_pipe_result) for _ in range(world_size)]
    zeco_results_all = [torch.empty_like(zeco_result) for _ in range(world_size)]
    lasp2_results_all = [torch.empty_like(lasp2_result) for _ in range(world_size)]

    dist.all_gather(wy_results_all, wy_result)
    dist.all_gather(wy_pipe_results_all, wy_pipe_result)
    dist.all_gather(zeco_results_all, zeco_result)
    dist.all_gather(lasp2_results_all, lasp2_result)

    numerical_results = {}
    if rank == 0:
        # Verify WY-All-Scan against sequential reference
        wy_errors = []
        for p in range(world_size):
            for h in range(H):
                ref = ref_states_wy[h][p]  # FP32 reference
                got = wy_results_all[p][h].float()
                err = compute_relative_error(got, ref)
                wy_errors.append(err)

        wy_max_err = max(wy_errors)
        wy_mean_err = sum(wy_errors) / len(wy_errors)

        # Verify WY-All-Scan pipelined
        wy_pipe_errors = []
        for p in range(world_size):
            for h in range(H):
                ref = ref_states_wy[h][p]
                got = wy_pipe_results_all[p][h].float()
                err = compute_relative_error(got, ref)
                wy_pipe_errors.append(err)

        wy_pipe_max_err = max(wy_pipe_errors)
        wy_pipe_mean_err = sum(wy_pipe_errors) / len(wy_pipe_errors)

        # Verify LASP-2
        lasp2_errors = []
        for p in range(world_size):
            for h in range(H):
                ref = ref_states_wy[h][p]
                got = lasp2_results_all[p][h].float()
                err = compute_relative_error(got, ref)
                lasp2_errors.append(err)

        lasp2_max_err = max(lasp2_errors)
        lasp2_mean_err = sum(lasp2_errors) / len(lasp2_errors)

        # Verify ZeCO-GLA
        zeco_errors = []
        for p in range(world_size):
            for h in range(H):
                ref = ref_states_diag[h][p]
                got = zeco_results_all[p][h].float()
                err = compute_relative_error(got, ref)
                zeco_errors.append(err)

        zeco_max_err = max(zeco_errors)
        zeco_mean_err = sum(zeco_errors) / len(zeco_errors)

        print(f"  WY-All-Scan:     max_err={wy_max_err:.6e}, mean_err={wy_mean_err:.6e}")
        print(f"  WY-Pipelined:    max_err={wy_pipe_max_err:.6e}, mean_err={wy_pipe_mean_err:.6e}")
        print(f"  LASP-2:          max_err={lasp2_max_err:.6e}, mean_err={lasp2_mean_err:.6e}")
        print(f"  ZeCO-GLA:        max_err={zeco_max_err:.6e}, mean_err={zeco_mean_err:.6e}")

        numerical_results = {
            "wy_allscan_max_err": wy_max_err,
            "wy_allscan_mean_err": wy_mean_err,
            "wy_pipelined_max_err": wy_pipe_max_err,
            "wy_pipelined_mean_err": wy_pipe_mean_err,
            "lasp2_max_err": lasp2_max_err,
            "lasp2_mean_err": lasp2_mean_err,
            "zeco_gla_max_err": zeco_max_err,
            "zeco_gla_mean_err": zeco_mean_err,
        }

        wy_numerical_pass = wy_max_err < 1e-3
        print(f"\n  Numerical accuracy criterion (< 1e-3): "
              f"{'PASS' if wy_numerical_pass else 'FAIL'} (max_err={wy_max_err:.6e})")

    # =================================================================
    # 7. Summary & Success Criteria
    # =================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("7. Summary & Success Criteria")
        print(f"{'='*70}")

        # Latency comparisons
        wy_median = wy_stats['median_ms']
        zeco_median = zeco_stats['median_ms']
        lasp2_median = lasp2_stats['median_ms']

        wy_vs_zeco = wy_median / zeco_median if zeco_median > 0 else float('inf')
        wy_vs_lasp2 = wy_median / lasp2_median if lasp2_median > 0 else float('inf')

        print(f"\n  Latency comparison (median):")
        print(f"    WY-All-Scan:    {wy_median:.3f} ms")
        print(f"    ZeCO-GLA:       {zeco_median:.3f} ms")
        print(f"    LASP-2:         {lasp2_median:.3f} ms")
        print(f"    WY/ZeCO ratio:  {wy_vs_zeco:.2f}x")
        print(f"    WY/LASP2 ratio: {wy_vs_lasp2:.2f}x")

        # Success criteria evaluation
        criterion_1 = wy_vs_zeco < 2.0
        criterion_2 = wy_vs_lasp2 < 0.5
        criterion_3 = wy_max_err < 1e-3
        criterion_4 = wy_micro['batch_heads_median_ms'] < 0.5

        print(f"\n  Success Criteria:")
        print(f"    1. WY-AllScan < 2x ZeCO-GLA:    {'PASS' if criterion_1 else 'FAIL'} "
              f"(ratio={wy_vs_zeco:.2f}x, target <2.0x)")
        print(f"    2. WY-AllScan < 0.5x LASP-2:    {'PASS' if criterion_2 else 'FAIL'} "
              f"(ratio={wy_vs_lasp2:.2f}x, target <0.5x)")
        print(f"    3. Numerical error < 1e-3:       {'PASS' if criterion_3 else 'FAIL'} "
              f"(max_err={wy_max_err:.6e})")
        print(f"    4. WY correction < 0.5ms:        {'PASS' if criterion_4 else 'FAIL'} "
              f"(median={wy_micro['batch_heads_median_ms']:.4f}ms)")

        all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4
        print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

        # Log to wandb
        if wandb is not None:
            try:
                wandb.log({
                    "latency/wy_allscan_median_ms": wy_median,
                    "latency/wy_allscan_mean_ms": wy_stats['mean_ms'],
                    "latency/wy_pipelined_median_ms": wy_pipe_stats['median_ms'],
                    "latency/zeco_gla_median_ms": zeco_median,
                    "latency/zeco_gla_mean_ms": zeco_stats['mean_ms'],
                    "latency/lasp2_median_ms": lasp2_median,
                    "latency/lasp2_mean_ms": lasp2_stats['mean_ms'],
                    "ratios/wy_vs_zeco": wy_vs_zeco,
                    "ratios/wy_vs_lasp2": wy_vs_lasp2,
                    "numerical/wy_max_err": wy_max_err,
                    "numerical/wy_mean_err": wy_mean_err,
                    "numerical/lasp2_max_err": lasp2_max_err,
                    "numerical/zeco_max_err": zeco_max_err,
                    "microbench/wy_correction_single_ms": wy_micro['single_head_mean_ms'],
                    "microbench/wy_correction_batch_ms": wy_micro['batch_heads_mean_ms'],
                    "success_criteria/wy_lt_2x_zeco": criterion_1,
                    "success_criteria/wy_lt_0.5x_lasp2": criterion_2,
                    "success_criteria/numerical_lt_1e3": criterion_3,
                    "success_criteria/wy_correction_lt_0.5ms": criterion_4,
                    "success_criteria/all_pass": all_pass,
                    "world_size": world_size,
                })
            except Exception as e:
                print(f"  WARNING: wandb logging failed: {e}")

        # Save results to file
        results = {
            "world_size": world_size,
            "d_k": d_k,
            "d_v": d_v,
            "H": H,
            "dtype": dtype_str,
            "latency": {
                "wy_allscan": wy_stats,
                "wy_pipelined": wy_pipe_stats,
                "zeco_gla": zeco_stats,
                "lasp2": lasp2_stats,
            },
            "numerical": numerical_results,
            "microbench": wy_micro,
            "ratios": {
                "wy_vs_zeco": wy_vs_zeco,
                "wy_vs_lasp2": wy_vs_lasp2,
            },
            "success_criteria": {
                "wy_lt_2x_zeco": criterion_1,
                "wy_lt_0.5x_lasp2": criterion_2,
                "numerical_lt_1e3": criterion_3,
                "wy_correction_lt_0.5ms": criterion_4,
                "all_pass": all_pass,
            },
            "wandb_url": wandb_url,
        }

        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_P{world_size}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")

        # Also save to Modal volume if available
        vol_path = "/results"
        if os.path.isdir(vol_path):
            vol_results_path = os.path.join(vol_path, f"055_results_P{world_size}.json")
            with open(vol_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Results also saved to Modal volume: {vol_results_path}")

    # Cleanup
    if rank == 0 and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

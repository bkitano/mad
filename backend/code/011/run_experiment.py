"""
MVE 011: Neumann Resolvent Kernel Accuracy Test

This script validates whether the Neumann series approximation of the
DPLR SSM resolvent (zI - A)^{-1} can match the exact Woodbury computation.

Three tests:
1. Kernel accuracy sweep: Vary truncation order k in {2,4,6,8,12,16}
   - Measure relative error ||K_k - K_exact||_2 / ||K_exact||_2
2. Near-resonance robustness: Set z_j ≈ lambda_i + epsilon (epsilon=1e-3)
   - Compare BF16 Neumann vs BF16 Woodbury for overflow behavior
3. Speed comparison: Wall-clock time for Neumann vs Woodbury
   - Vary N in {32, 64, 128, 256}

Success criteria (from proposal):
  - Relative kernel error < 1e-3 for k <= 8
  - Near-resonance: Neumann in BF16 produces finite results, Woodbury overflows
  - GEMM speed: Neumann faster than Woodbury in BF16 for N >= 64

Failure criteria:
  - Kill if: k > 16 needed for 1e-3 accuracy
  - Kill if: spectral radius > 1 for > 10% of frequencies with HiPPO init
"""

import torch
import time
import json
import argparse
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.resolvent import (
    hippo_legs_init,
    random_B_C,
    compute_ssm_kernel_exact,
    compute_ssm_kernel_neumann,
    compute_spectral_radius,
    woodbury_resolvent,
    neumann_resolvent,
)


def generate_frequencies(L: int, dtype=torch.complex128) -> torch.Tensor:
    """Generate L frequency points on the unit circle (roots of unity)."""
    # z_j = exp(2 pi i j / L) for j = 0, ..., L-1
    angles = 2 * torch.pi * torch.arange(L, dtype=torch.float64) / L
    z = torch.complex(torch.cos(angles), torch.sin(angles))
    return z.to(dtype)


def test_kernel_accuracy(config: dict, device: torch.device) -> dict:
    """
    Test 1: Kernel accuracy sweep.

    Vary truncation order k and measure relative error vs exact Woodbury.
    """
    N = config.get("state_dim", 64)
    r = config.get("rank", 1)
    d = config.get("hidden_dim", 8)
    L = config.get("seq_len", 1024)
    k_values = config.get("k_values", [2, 4, 6, 8, 12, 16])
    n_trials = config.get("n_trials", 5)

    print(f"\n{'='*60}")
    print(f"TEST 1: Kernel Accuracy Sweep")
    print(f"N={N}, r={r}, d={d}, L={L}, trials={n_trials}")
    print(f"k values: {k_values}")
    print(f"{'='*60}")

    results = {"k_values": k_values, "errors": {}, "spectral_radii": {}}

    for trial in range(n_trials):
        torch.manual_seed(42 + trial)

        # Initialize DPLR parameters
        Lambda, P, Q = hippo_legs_init(N, r)
        B, C = random_B_C(N, d)

        # Move to device (use float64 for accuracy baseline)
        Lambda = Lambda.to(device)
        P = P.to(device)
        Q = Q.to(device)
        B = B.to(device)
        C = C.to(device)

        # Generate frequencies
        z = generate_frequencies(L).to(device)

        # Compute exact kernel (FP64 for ground truth)
        K_exact = compute_ssm_kernel_exact(z, Lambda, P, Q, B, C)
        K_exact_norm = torch.norm(K_exact).item()

        # Compute spectral radius
        rho = compute_spectral_radius(z, Lambda, P, Q)
        rho_stats = {
            "mean": rho.mean().item(),
            "max": rho.max().item(),
            "min": rho.min().item(),
            "frac_above_1": (rho > 1.0).float().mean().item(),
            "median": rho.median().item(),
        }

        if trial == 0:
            results["spectral_radii"] = rho_stats
            print(f"\nSpectral radius stats (trial 0):")
            print(f"  Mean: {rho_stats['mean']:.4f}")
            print(f"  Median: {rho_stats['median']:.4f}")
            print(f"  Max: {rho_stats['max']:.4f}")
            print(f"  Min: {rho_stats['min']:.4f}")
            print(f"  Fraction > 1: {rho_stats['frac_above_1']:.4f}")

        for k in k_values:
            # Compute Neumann kernel
            K_neumann = compute_ssm_kernel_neumann(z, Lambda, P, Q, B, C, k=k)

            # Relative error
            rel_error = torch.norm(K_neumann - K_exact).item() / (K_exact_norm + 1e-30)

            key = str(k)
            if key not in results["errors"]:
                results["errors"][key] = []
            results["errors"][key].append(rel_error)

    # Compute statistics
    print(f"\nKernel accuracy results:")
    print(f"{'k':>4} | {'Mean Error':>12} | {'Std Error':>12} | {'Max Error':>12} | {'Pass?':>6}")
    print(f"{'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")

    summary = {}
    for k in k_values:
        key = str(k)
        errors = results["errors"][key]
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        std_err = (sum((e - mean_err)**2 for e in errors) / len(errors)) ** 0.5
        passed = max_err < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"{k:>4} | {mean_err:>12.2e} | {std_err:>12.2e} | {max_err:>12.2e} | {status:>6}")
        summary[key] = {
            "mean_error": mean_err,
            "std_error": std_err,
            "max_error": max_err,
            "passed": passed,
        }

    results["summary"] = summary
    return results


def test_near_resonance(config: dict, device: torch.device) -> dict:
    """
    Test 2: Near-resonance robustness.

    Set z_j close to eigenvalue lambda_i and test BF16 behavior.
    Expect: Woodbury in BF16 overflows, Neumann in BF16 stays finite.
    """
    N = config.get("state_dim", 64)
    r = config.get("rank", 1)
    d = config.get("hidden_dim", 8)
    epsilon_values = config.get("epsilon_values", [1e-1, 1e-2, 1e-3, 1e-4])
    k_test = config.get("k_resonance", 8)

    print(f"\n{'='*60}")
    print(f"TEST 2: Near-Resonance Robustness")
    print(f"N={N}, r={r}, d={d}, k={k_test}")
    print(f"Epsilon values: {epsilon_values}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    Lambda, P, Q = hippo_legs_init(N, r)
    B, C = random_B_C(N, d)

    Lambda = Lambda.to(device)
    P = P.to(device)
    Q = Q.to(device)
    B = B.to(device)
    C = C.to(device)

    results = {"epsilon_values": epsilon_values, "tests": []}

    for eps in epsilon_values:
        # Create near-resonance frequencies: z_j = lambda_j + eps
        # Take first 16 eigenvalues and perturb them slightly
        n_test = min(16, N)
        z_near = Lambda[:n_test] + eps

        # Test in FP64 (ground truth)
        K_exact_fp64 = compute_ssm_kernel_exact(z_near, Lambda, P, Q, B, C)
        K_neumann_fp64 = compute_ssm_kernel_neumann(z_near, Lambda, P, Q, B, C, k=k_test)

        exact_finite_fp64 = torch.isfinite(K_exact_fp64).all().item()
        neumann_finite_fp64 = torch.isfinite(K_neumann_fp64).all().item()

        # Test in FP32
        z_f32 = z_near.to(torch.complex64)
        Lambda_f32 = Lambda.to(torch.complex64)
        P_f32 = P.to(torch.complex64)
        Q_f32 = Q.to(torch.complex64)
        B_f32 = B.to(torch.complex64)
        C_f32 = C.to(torch.complex64)

        K_exact_fp32 = compute_ssm_kernel_exact(z_f32, Lambda_f32, P_f32, Q_f32, B_f32, C_f32)
        K_neumann_fp32 = compute_ssm_kernel_neumann(z_f32, Lambda_f32, P_f32, Q_f32, B_f32, C_f32, k=k_test)

        exact_finite_fp32 = torch.isfinite(K_exact_fp32).all().item()
        neumann_finite_fp32 = torch.isfinite(K_neumann_fp32).all().item()

        # Test BF16 behavior (via casting)
        # Note: complex BF16 is not directly supported in PyTorch, so we test
        # the real/imag components separately
        try:
            # We'll simulate BF16 precision by casting to bfloat16 and back
            z_bf16 = z_near.to(torch.complex64)  # BF16 doesn't have complex variant
            Lambda_bf16 = Lambda.to(torch.complex64)

            # Simulate BF16 precision on D_z computation (the critical path)
            D_z_full_prec = 1.0 / (z_bf16.unsqueeze(1) - Lambda_bf16.unsqueeze(0))
            # Cast intermediate to BF16 precision
            D_z_real_bf16 = D_z_full_prec.real.to(torch.bfloat16).to(torch.float32)
            D_z_imag_bf16 = D_z_full_prec.imag.to(torch.bfloat16).to(torch.float32)
            D_z_bf16 = torch.complex(D_z_real_bf16, D_z_imag_bf16)

            woodbury_d_z_finite = torch.isfinite(D_z_bf16).all().item()
            woodbury_d_z_max = D_z_full_prec.abs().max().item()

            # For Neumann, the critical path is F = Q^* D_z P (small r x r matrix)
            D_z_P_bf16 = D_z_bf16.unsqueeze(2) * P_f32.unsqueeze(0)
            F_bf16 = torch.einsum('rn,lnk->lrk', Q_f32.conj().T, D_z_P_bf16)
            neumann_f_finite = torch.isfinite(F_bf16).all().item()

            bf16_test = {
                "woodbury_dz_finite": woodbury_d_z_finite,
                "woodbury_dz_max": woodbury_d_z_max,
                "neumann_f_finite": neumann_f_finite,
            }
        except Exception as e:
            bf16_test = {"error": str(e)}

        test_result = {
            "epsilon": eps,
            "fp64": {"exact_finite": exact_finite_fp64, "neumann_finite": neumann_finite_fp64},
            "fp32": {"exact_finite": exact_finite_fp32, "neumann_finite": neumann_finite_fp32},
            "bf16": bf16_test,
        }
        results["tests"].append(test_result)

        print(f"\nEpsilon = {eps:.0e}:")
        print(f"  FP64: Exact finite={exact_finite_fp64}, Neumann finite={neumann_finite_fp64}")
        print(f"  FP32: Exact finite={exact_finite_fp32}, Neumann finite={neumann_finite_fp32}")
        if "error" not in bf16_test:
            print(f"  BF16: Woodbury D_z finite={bf16_test['woodbury_dz_finite']} (max={bf16_test['woodbury_dz_max']:.2e})")
            print(f"  BF16: Neumann F finite={bf16_test['neumann_f_finite']}")
        else:
            print(f"  BF16: Error - {bf16_test['error']}")

    return results


def test_speed_comparison(config: dict, device: torch.device) -> dict:
    """
    Test 3: Speed comparison between Woodbury and Neumann.

    Measure wall-clock time for kernel computation at various N.
    """
    N_values = config.get("N_values", [32, 64, 128, 256])
    r = config.get("rank", 1)
    d = config.get("hidden_dim", 8)
    L = config.get("seq_len", 1024)
    k_test = config.get("k_speed", 8)
    n_warmup = config.get("n_warmup", 3)
    n_runs = config.get("n_runs", 10)

    print(f"\n{'='*60}")
    print(f"TEST 3: Speed Comparison")
    print(f"r={r}, d={d}, L={L}, k={k_test}")
    print(f"N values: {N_values}")
    print(f"{'='*60}")

    results = {"N_values": N_values, "timings": {}}

    for N in N_values:
        torch.manual_seed(42)
        Lambda, P, Q = hippo_legs_init(N, r)
        B, C = random_B_C(N, d)
        z = generate_frequencies(L)

        # Move to device
        Lambda = Lambda.to(device)
        P = P.to(device)
        Q = Q.to(device)
        B = B.to(device)
        C = C.to(device)
        z = z.to(device)

        # Warmup
        for _ in range(n_warmup):
            _ = compute_ssm_kernel_exact(z, Lambda, P, Q, B, C)
            _ = compute_ssm_kernel_neumann(z, Lambda, P, Q, B, C, k=k_test)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Time exact Woodbury
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            K_exact = compute_ssm_kernel_exact(z, Lambda, P, Q, B, C)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_exact = (time.perf_counter() - t0) / n_runs

        # Time Neumann
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            K_neumann = compute_ssm_kernel_neumann(z, Lambda, P, Q, B, C, k=k_test)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_neumann = (time.perf_counter() - t0) / n_runs

        # Also time the "efficient" Neumann (compute_ssm_kernel_neumann avoids forming N x N matrix)
        # vs Woodbury which forms N x N resolvent

        speedup = t_exact / t_neumann if t_neumann > 0 else float('inf')

        results["timings"][str(N)] = {
            "exact_ms": t_exact * 1000,
            "neumann_ms": t_neumann * 1000,
            "speedup": speedup,
        }

        print(f"\nN={N}:")
        print(f"  Exact Woodbury: {t_exact*1000:.2f} ms")
        print(f"  Neumann (k={k_test}): {t_neumann*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def test_spectral_radius_distribution(config: dict, device: torch.device) -> dict:
    """
    Test 4: Check spectral radius distribution to verify convergence guarantee.

    Kill criterion: > 10% of frequencies have spectral radius > 1
    """
    N = config.get("state_dim", 64)
    r = config.get("rank", 1)
    L = config.get("seq_len", 1024)
    n_trials = config.get("n_trials", 5)

    print(f"\n{'='*60}")
    print(f"TEST 4: Spectral Radius Distribution")
    print(f"N={N}, r={r}, L={L}, trials={n_trials}")
    print(f"{'='*60}")

    all_frac_above_1 = []

    for trial in range(n_trials):
        torch.manual_seed(42 + trial)
        Lambda, P, Q = hippo_legs_init(N, r)
        Lambda = Lambda.to(device)
        P = P.to(device)
        Q = Q.to(device)

        z = generate_frequencies(L).to(device)
        rho = compute_spectral_radius(z, Lambda, P, Q)

        frac_above_1 = (rho > 1.0).float().mean().item()
        all_frac_above_1.append(frac_above_1)

        print(f"  Trial {trial}: frac(rho > 1) = {frac_above_1:.4f}, "
              f"max(rho) = {rho.max().item():.4f}, "
              f"mean(rho) = {rho.mean().item():.4f}")

    mean_frac = sum(all_frac_above_1) / len(all_frac_above_1)
    max_frac = max(all_frac_above_1)

    passed = max_frac < 0.10  # Kill if > 10% diverge
    status = "PASS" if passed else "FAIL"

    print(f"\nMean frac(rho > 1): {mean_frac:.4f}")
    print(f"Max frac(rho > 1): {max_frac:.4f}")
    print(f"Convergence check: {status} (threshold: 10%)")

    return {
        "frac_above_1_per_trial": all_frac_above_1,
        "mean_frac_above_1": mean_frac,
        "max_frac_above_1": max_frac,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="MVE 011: Neumann Resolvent Kernel Accuracy Test")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {config_path} not found, using defaults")
        config = {}

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    experiment_config = config.get("experiment", {})

    print(f"\n{'#'*60}")
    print(f"# MVE 011: Neumann Resolvent Kernel Accuracy Test")
    print(f"# Device: {device}")
    print(f"{'#'*60}")

    all_results = {"config": experiment_config, "device": str(device)}

    # Test 1: Kernel accuracy sweep
    accuracy_results = test_kernel_accuracy(experiment_config, device)
    all_results["kernel_accuracy"] = accuracy_results

    # Test 2: Near-resonance robustness
    resonance_results = test_near_resonance(experiment_config, device)
    all_results["near_resonance"] = resonance_results

    # Test 3: Speed comparison
    speed_results = test_speed_comparison(experiment_config, device)
    all_results["speed_comparison"] = speed_results

    # Test 4: Spectral radius distribution (convergence check)
    spectral_results = test_spectral_radius_distribution(experiment_config, device)
    all_results["spectral_radius"] = spectral_results

    # Summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")

    # Check success criteria
    accuracy_pass = all(
        accuracy_results["summary"].get(str(k), {}).get("passed", False)
        for k in [2, 4, 6, 8]
    )
    print(f"\n1. Kernel accuracy (k<=8, error < 1e-3): {'PASS' if accuracy_pass else 'FAIL'}")

    convergence_pass = spectral_results["passed"]
    print(f"2. Convergence guarantee (< 10% divergent): {'PASS' if convergence_pass else 'FAIL'}")

    # Check if any k > 16 is needed
    min_k_for_accuracy = None
    for k in [2, 4, 6, 8, 12, 16]:
        key = str(k)
        if key in accuracy_results["summary"] and accuracy_results["summary"][key]["passed"]:
            min_k_for_accuracy = k
            break
    if min_k_for_accuracy:
        print(f"3. Minimum k for < 1e-3 accuracy: k={min_k_for_accuracy}")
    else:
        print(f"3. Minimum k for < 1e-3 accuracy: NOT ACHIEVED (k > 16 needed)")

    # Overall decision
    print(f"\n{'='*60}")
    if accuracy_pass and convergence_pass:
        print("DECISION: PROCEED - Neumann resolvent is viable")
    elif not convergence_pass:
        print("DECISION: KILL - Spectral radius > 1 for too many frequencies")
    elif not accuracy_pass:
        if min_k_for_accuracy and min_k_for_accuracy <= 16:
            print(f"DECISION: DEBUG - Accuracy achieved at k={min_k_for_accuracy} but not k<=8")
        else:
            print("DECISION: KILL - Cannot achieve 1e-3 accuracy even at k=16")
    print(f"{'='*60}")

    # Save results
    results_path = Path(__file__).parent / "results.json"
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        if isinstance(obj, (float,)) and (obj != obj or obj == float('inf') or obj == float('-inf')):
            return str(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(v) for v in d]
        return make_serializable(d)

    all_results = recursive_convert(all_results)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

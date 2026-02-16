"""
Microbenchmark for INT4 QK^T quantization accuracy.

Measures:
1. Cosine similarity of QK^T: BF16 vs INT4 (no smooth) vs INT4 (smooth)
2. Element-wise MSE between reference and quantized outputs
3. Distribution statistics of quantization error

This is the first part of the MVE protocol from proposal 054:
"Compare accuracy: Cosine similarity of QK^T output vs BF16 reference across 100 random inputs"

Success criteria:
- INT4 QK^T cosine similarity > 99% with smoothing
- INT4 QK^T cosine similarity < 85% without smoothing
"""

import torch
import torch.nn.functional as F
import time
import json
from pathlib import Path
from typing import Dict, List

from models.quantization import (
    int4_matmul_with_smoothing,
    fp8_matmul,
    compute_cosine_similarity,
    smooth_qk,
    quantize_int4_per_thread,
)


def benchmark_cosine_similarity(
    n_trials: int = 100,
    batch_size: int = 4,
    n_heads: int = 2,
    sub_chunk_size: int = 16,
    dk: int = 128,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run cosine similarity benchmark comparing BF16 vs INT4 QK^T.

    Generates random Q, K tensors (simulating GLA's gated Q/K after
    gate absorption) and measures how well INT4 quantization preserves
    the QK^T output.
    """
    cos_sims_no_smooth = []
    cos_sims_smooth = []
    mse_no_smooth = []
    mse_smooth = []

    for trial in range(n_trials):
        # Generate random Q, K with realistic statistics
        # GLA's Q, K are L2-normalized and then multiplied by gate factors
        Q = torch.randn(batch_size, n_heads, sub_chunk_size, dk, device=device)
        K = torch.randn(batch_size, n_heads, sub_chunk_size, dk, device=device)

        # L2 normalize (as in GLA)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Apply random gate scaling (simulating Lambda, Gamma gate absorption)
        gate_scale_q = torch.rand(batch_size, n_heads, sub_chunk_size, 1, device=device) * 2
        gate_scale_k = torch.rand(batch_size, n_heads, sub_chunk_size, 1, device=device) * 2
        Q = Q * gate_scale_q
        K = K * gate_scale_k

        # BF16 reference
        P_ref = torch.matmul(Q, K.transpose(-2, -1))

        # INT4 without smoothing
        P_no_smooth = int4_matmul_with_smoothing(Q, K, smooth=False)

        # INT4 with smoothing
        P_smooth = int4_matmul_with_smoothing(Q, K, smooth=True)

        # Cosine similarity
        cos_no = compute_cosine_similarity(P_ref, P_no_smooth)
        cos_sm = compute_cosine_similarity(P_ref, P_smooth)

        cos_sims_no_smooth.append(cos_no)
        cos_sims_smooth.append(cos_sm)

        # MSE
        mse_no = F.mse_loss(P_no_smooth, P_ref).item()
        mse_sm = F.mse_loss(P_smooth, P_ref).item()

        mse_no_smooth.append(mse_no)
        mse_smooth.append(mse_sm)

    results = {
        "cosine_sim_no_smooth_mean": sum(cos_sims_no_smooth) / len(cos_sims_no_smooth),
        "cosine_sim_no_smooth_min": min(cos_sims_no_smooth),
        "cosine_sim_no_smooth_max": max(cos_sims_no_smooth),
        "cosine_sim_smooth_mean": sum(cos_sims_smooth) / len(cos_sims_smooth),
        "cosine_sim_smooth_min": min(cos_sims_smooth),
        "cosine_sim_smooth_max": max(cos_sims_smooth),
        "mse_no_smooth_mean": sum(mse_no_smooth) / len(mse_no_smooth),
        "mse_smooth_mean": sum(mse_smooth) / len(mse_smooth),
    }

    return results


def benchmark_fp8_sv(
    n_trials: int = 100,
    batch_size: int = 4,
    n_heads: int = 2,
    c: int = 16,
    dv: int = 256,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark FP8 quantization accuracy for the SV matmul.

    Measures cosine similarity and MSE of FP8 P@V vs BF16 P@V.
    """
    cos_sims = []
    mses = []

    for trial in range(n_trials):
        # Random attention scores (output of QK^T with gate mask applied)
        P = torch.randn(batch_size, n_heads, c, c, device=device) * 0.5
        # Apply causal mask
        mask = torch.tril(torch.ones(c, c, device=device))
        P = P * mask

        # Random values
        V = torch.randn(batch_size, n_heads, c, dv, device=device)

        # BF16 reference
        O_ref = torch.matmul(P, V)

        # FP8 quantized
        O_fp8 = fp8_matmul(P, V)

        cos_sim = compute_cosine_similarity(O_ref, O_fp8)
        mse = F.mse_loss(O_fp8, O_ref).item()

        cos_sims.append(cos_sim)
        mses.append(mse)

    return {
        "fp8_sv_cosine_sim_mean": sum(cos_sims) / len(cos_sims),
        "fp8_sv_cosine_sim_min": min(cos_sims),
        "fp8_sv_mse_mean": sum(mses) / len(mses),
    }


def benchmark_varying_dk(
    dk_values: List[int] = [64, 128, 256],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark INT4 accuracy across different key dimensions.

    Tests whether smoothing effectiveness varies with dk.
    """
    results = {}
    for dk in dk_values:
        res = benchmark_cosine_similarity(
            n_trials=50, dk=dk, device=device,
        )
        results[f"dk_{dk}"] = res
    return results


def benchmark_varying_subchunk(
    c_values: List[int] = [8, 16, 32],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark INT4 accuracy across different sub-chunk sizes.

    Tests whether quantization quality varies with sub-chunk size c.
    """
    results = {}
    for c in c_values:
        res = benchmark_cosine_similarity(
            n_trials=50, sub_chunk_size=c, device=device,
        )
        results[f"c_{c}"] = res
    return results


def run_all_benchmarks(device: str = "cuda") -> Dict:
    """Run all microbenchmarks and return combined results."""
    print("=" * 60)
    print("INT4 QK^T Quantization Accuracy Benchmark")
    print("=" * 60)

    # Main cosine similarity benchmark
    print("\n[1/4] Running main cosine similarity benchmark (100 trials)...")
    main_results = benchmark_cosine_similarity(n_trials=100, device=device)
    print(f"  INT4 no smoothing — cosine sim: {main_results['cosine_sim_no_smooth_mean']:.4f}")
    print(f"  INT4 with smoothing — cosine sim: {main_results['cosine_sim_smooth_mean']:.4f}")
    print(f"  INT4 no smoothing — MSE: {main_results['mse_no_smooth_mean']:.6f}")
    print(f"  INT4 with smoothing — MSE: {main_results['mse_smooth_mean']:.6f}")

    # FP8 SV benchmark
    print("\n[2/4] Running FP8 SV matmul benchmark (100 trials)...")
    fp8_results = benchmark_fp8_sv(n_trials=100, device=device)
    print(f"  FP8 SV cosine sim: {fp8_results['fp8_sv_cosine_sim_mean']:.4f}")
    print(f"  FP8 SV MSE: {fp8_results['fp8_sv_mse_mean']:.6f}")

    # Varying dk
    print("\n[3/4] Running dk sweep benchmark...")
    dk_results = benchmark_varying_dk(device=device)
    for dk_key, dk_res in dk_results.items():
        print(f"  {dk_key}: smooth cos_sim={dk_res['cosine_sim_smooth_mean']:.4f}, "
              f"no_smooth cos_sim={dk_res['cosine_sim_no_smooth_mean']:.4f}")

    # Varying sub-chunk size
    print("\n[4/4] Running sub-chunk size sweep benchmark...")
    c_results = benchmark_varying_subchunk(device=device)
    for c_key, c_res in c_results.items():
        print(f"  {c_key}: smooth cos_sim={c_res['cosine_sim_smooth_mean']:.4f}, "
              f"no_smooth cos_sim={c_res['cosine_sim_no_smooth_mean']:.4f}")

    # Compile all results
    all_results = {
        "main": main_results,
        "fp8_sv": fp8_results,
        "dk_sweep": dk_results,
        "subchunk_sweep": c_results,
    }

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    smooth_cos = main_results["cosine_sim_smooth_mean"]
    no_smooth_cos = main_results["cosine_sim_no_smooth_mean"]

    crit1 = smooth_cos > 0.99
    crit2 = no_smooth_cos < 0.85
    crit3 = main_results["mse_smooth_mean"] < main_results["mse_no_smooth_mean"]

    print(f"  [{'PASS' if crit1 else 'FAIL'}] INT4 cosine sim > 99% with smoothing: {smooth_cos:.4f}")
    print(f"  [{'PASS' if crit2 else 'INFO'}] INT4 cosine sim < 85% without smoothing: {no_smooth_cos:.4f}")
    print(f"  [{'PASS' if crit3 else 'FAIL'}] Smoothing reduces MSE: "
          f"{main_results['mse_smooth_mean']:.6f} < {main_results['mse_no_smooth_mean']:.6f}")

    all_results["success_criteria"] = {
        "cosine_sim_smooth_gt_99pct": crit1,
        "cosine_sim_no_smooth_lt_85pct": crit2,
        "smoothing_reduces_mse": crit3,
    }

    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_all_benchmarks(device=device)

    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

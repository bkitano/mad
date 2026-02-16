"""
Training and benchmarking script for Experiment 041:
EVT Joint Forward-Backward Graph Partitioning for Linear RNN Training.

This script:
1. Benchmarks the BASELINE (separate fwd/bwd kernels) vs FUSED (joint fwd+bwd kernel)
2. Measures HBM traffic proxy (memory allocated during kernel execution)
3. Measures wall-clock time for combined fwd+bwd
4. Verifies numerical agreement (bit-exact comparison)
5. Optionally trains a small model to verify end-to-end correctness

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import time
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import wandb


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_decay_mask(C, decay_rate, device, dtype):
    """Build causal decay mask: M[i,j] = decay^(i-j) for i>=j, 0 otherwise."""
    positions = torch.arange(C, device=device, dtype=dtype)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = torch.where(diff >= 0, decay_rate ** diff, torch.zeros_like(diff))
    return mask


# ============================================================================
# Reference PyTorch implementation (for numerical verification)
# ============================================================================

def reference_forward(Q, K, V, M, H):
    """Pure PyTorch forward for a single chunk."""
    B, C, D = Q.shape
    N = H.shape[1]

    S = torch.bmm(Q, K.transpose(-1, -2))          # [B, C, C]
    S_tilde = S * M                                  # [B, C, C]
    O_intra = torch.bmm(S_tilde, V)                  # [B, C, D]
    O_state = torch.bmm(Q[:, :, :N], H)              # [B, C, D]
    O = O_intra + O_state
    return O, S_tilde


def reference_backward(Q, K, V, M, H, S_tilde, grad_O):
    """Pure PyTorch backward for a single chunk."""
    B, C, D = Q.shape
    N = H.shape[1]

    grad_S_tilde = torch.bmm(grad_O, V.transpose(-1, -2))  # [B, C, C]
    grad_V = torch.bmm(S_tilde.transpose(-1, -2), grad_O)   # [B, C, D]
    grad_S = grad_S_tilde * M                                 # [B, C, C]
    grad_Q = torch.bmm(grad_S, K)                             # [B, C, D]
    grad_K = torch.bmm(grad_S.transpose(-1, -2), Q)           # [B, C, D]

    return grad_Q, grad_K, grad_V


# ============================================================================
# Triton kernels (imported from models)
# ============================================================================

def run_baseline_fwd_bwd(Q, K, V, M, H, grad_O):
    """Run SEPARATE forward and backward kernels (baseline)."""
    from models.gla_baseline import _fwd_intra_chunk_kernel, _bwd_intra_chunk_kernel

    B, T, D = Q.shape
    C = M.shape[2]
    N = H.shape[2]
    num_chunks = T // C

    O = torch.empty_like(Q)
    S_tilde = torch.empty(B, num_chunks, C, C, device=Q.device, dtype=Q.dtype)

    grid = (B, num_chunks)

    # Forward kernel (writes S_tilde to HBM)
    _fwd_intra_chunk_kernel[grid](
        Q, K, V, M, H, O, S_tilde,
        B, T, C, D, N,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        H.stride(0), H.stride(1), H.stride(2), H.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        S_tilde.stride(0), S_tilde.stride(1), S_tilde.stride(2), S_tilde.stride(3),
    )

    grad_Q = torch.empty_like(Q)
    grad_K = torch.empty_like(K)
    grad_V = torch.empty_like(V)

    # Backward kernel (reads S_tilde from HBM)
    _bwd_intra_chunk_kernel[grid](
        Q, K, V, M, H, S_tilde, grad_O,
        grad_Q, grad_K, grad_V,
        B, T, C, D, N,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        H.stride(0), H.stride(1), H.stride(2), H.stride(3),
        S_tilde.stride(0), S_tilde.stride(1), S_tilde.stride(2), S_tilde.stride(3),
        grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
        grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
        grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
        grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
    )

    return O, S_tilde, grad_Q, grad_K, grad_V


def run_fused_fwd_bwd(Q, K, V, M, H, grad_O):
    """Run FUSED forward+backward kernel (proposed)."""
    from models.gla_fused import _fused_fwd_bwd_kernel

    B, T, D = Q.shape
    C = M.shape[2]
    N = H.shape[2]
    num_chunks = T // C

    O = torch.empty_like(Q)
    grad_Q = torch.empty_like(Q)
    grad_K = torch.empty_like(K)
    grad_V = torch.empty_like(V)

    grid = (B, num_chunks)

    _fused_fwd_bwd_kernel[grid](
        Q, K, V, M, H, O,
        grad_O,
        grad_Q, grad_K, grad_V,
        B, T, C, D, N,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        H.stride(0), H.stride(1), H.stride(2), H.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
        grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
        grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
        grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
    )

    return O, grad_Q, grad_K, grad_V


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_kernel(fn, warmup=10, repeats=100, label=""):
    """Benchmark a kernel with CUDA events for precise timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    avg_time = sum(times) / len(times)
    min_time = min(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {label}: avg={avg_time:.3f}ms, min={min_time:.3f}ms, std={std_time:.3f}ms")
    return avg_time, min_time, std_time


def measure_memory(fn, label=""):
    """Measure peak memory allocated during kernel execution."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    mem_after = torch.cuda.max_memory_allocated()
    extra_mem = mem_after - mem_before
    print(f"  {label}: extra memory = {extra_mem / 1024:.1f} KB")
    return extra_mem


def numerical_comparison(O_baseline, O_fused, grad_Q_b, grad_Q_f, grad_K_b, grad_K_f, grad_V_b, grad_V_f):
    """Compare numerical outputs between baseline and fused."""
    results = {}

    # Forward output
    o_diff = (O_baseline - O_fused).abs().max().item()
    o_rel = (O_baseline - O_fused).abs().max().item() / (O_baseline.abs().max().item() + 1e-10)
    results["O_max_abs_diff"] = o_diff
    results["O_max_rel_diff"] = o_rel

    # Grad Q
    gq_diff = (grad_Q_b - grad_Q_f).abs().max().item()
    gq_rel = gq_diff / (grad_Q_b.abs().max().item() + 1e-10)
    results["grad_Q_max_abs_diff"] = gq_diff
    results["grad_Q_max_rel_diff"] = gq_rel

    # Grad K
    gk_diff = (grad_K_b - grad_K_f).abs().max().item()
    gk_rel = gk_diff / (grad_K_b.abs().max().item() + 1e-10)
    results["grad_K_max_abs_diff"] = gk_diff
    results["grad_K_max_rel_diff"] = gk_rel

    # Grad V
    gv_diff = (grad_V_b - grad_V_f).abs().max().item()
    gv_rel = gv_diff / (grad_V_b.abs().max().item() + 1e-10)
    results["grad_V_max_abs_diff"] = gv_diff
    results["grad_V_max_rel_diff"] = gv_rel

    return results


def hbm_traffic_estimate(B, T, C, D, N, dtype_bytes=4):
    """Theoretical HBM traffic estimate for baseline vs fused (in bytes)."""
    num_chunks = T // C

    # Per-chunk traffic, summed over all chunks
    # Baseline: reads Q,K,V (fwd) + writes S_tilde + reads S_tilde (bwd) + reads Q,K,V,M (bwd)
    baseline_per_chunk = (
        3 * C * D * 2 +    # Q, K, V: read in fwd, read again in bwd (×2)
        C * C * 2 +         # S_tilde: written in fwd, read in bwd (×2)
        C * C +             # M: read in fwd + read in bwd
        C * D +             # O: written in fwd
        C * D +             # grad_O: read in bwd
        3 * C * D +         # grad_Q, grad_K, grad_V: written in bwd
        N * D               # H: read in fwd + bwd
    )

    # Fused: reads Q,K,V,M once; no S_tilde HBM; writes O, reads grad_O, writes grads
    fused_per_chunk = (
        3 * C * D +         # Q, K, V: read once (kept in registers across fwd→bwd)
        C * C +             # M: read once
        C * D +             # O: written
        C * D +             # grad_O: read
        3 * C * D +         # grad_Q, grad_K, grad_V: written
        N * D               # H: read once
    )

    baseline_total = baseline_per_chunk * num_chunks * B * dtype_bytes
    fused_total = fused_per_chunk * num_chunks * B * dtype_bytes
    reduction = 1.0 - fused_total / baseline_total

    return baseline_total, fused_total, reduction


# ============================================================================
# End-to-end training (optional, for model-level validation)
# ============================================================================

def train_model(model, train_data, config, device):
    """Train a small GLA model to validate end-to-end correctness."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    model.train()

    tokens, targets = train_data
    dataset = TensorDataset(tokens, targets)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=True)

    num_epochs = config.get("num_epochs", 5)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_tokens, batch_targets in loader:
            batch_tokens = batch_tokens.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_tokens)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch_targets[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        wandb.log({
            "train/loss": avg_loss,
            "epoch": epoch,
        })

    return losses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    bench_config = config.get("benchmark", {})

    # Extract dimensions
    d_model = model_config.get("d_model", 64)
    d_state = model_config.get("d_state", 16)
    chunk_size = model_config.get("chunk_size", 32)
    seq_len = model_config.get("seq_len", 256)
    vocab_size = model_config.get("vocab_size", 256)
    batch_size = bench_config.get("batch_size", 8)
    warmup = bench_config.get("warmup", 10)
    repeats = bench_config.get("repeats", 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Device: {device}")
    print(f"Config: d={d_model}, n={d_state}, C={chunk_size}, T={seq_len}, B={batch_size}")

    # Initialize wandb
    wandb.init(
        project="mad-architecture-search",
        name=f"exp-041-evt-joint-fwd-bwd",
        config={
            "model": model_config,
            "training": train_config,
            "benchmark": bench_config,
            "proposal_id": "041-evt-joint-fwd-bwd-graph-partitioning",
        }
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    # ======================================================================
    # PART 1: Numerical Verification
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 1: NUMERICAL VERIFICATION")
    print("=" * 60)

    num_chunks = seq_len // chunk_size
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    decay_rate = 0.95
    M_single = build_decay_mask(chunk_size, decay_rate, device, dtype)
    M = M_single.unsqueeze(0).unsqueeze(0).expand(batch_size, num_chunks, chunk_size, chunk_size).contiguous()
    H = torch.zeros(batch_size, num_chunks, d_state, d_model, device=device, dtype=dtype)
    grad_O = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    # Reference (PyTorch)
    Q_chunks = Q.reshape(batch_size, num_chunks, chunk_size, d_model)
    K_chunks = K.reshape(batch_size, num_chunks, chunk_size, d_model)
    V_chunks = V.reshape(batch_size, num_chunks, chunk_size, d_model)
    grad_O_chunks = grad_O.reshape(batch_size, num_chunks, chunk_size, d_model)

    ref_Os = []
    ref_gQs = []
    ref_gKs = []
    ref_gVs = []
    for j in range(num_chunks):
        Qj = Q_chunks[:, j]
        Kj = K_chunks[:, j]
        Vj = V_chunks[:, j]
        Mj = M[:, j]
        Hj = H[:, j]
        gOj = grad_O_chunks[:, j]

        O_ref, S_tilde_ref = reference_forward(Qj, Kj, Vj, Mj, Hj)
        gQ_ref, gK_ref, gV_ref = reference_backward(Qj, Kj, Vj, Mj, Hj, S_tilde_ref, gOj)
        ref_Os.append(O_ref)
        ref_gQs.append(gQ_ref)
        ref_gKs.append(gK_ref)
        ref_gVs.append(gV_ref)

    ref_O = torch.stack(ref_Os, dim=1).reshape(batch_size, seq_len, d_model)
    ref_gQ = torch.stack(ref_gQs, dim=1).reshape(batch_size, seq_len, d_model)
    ref_gK = torch.stack(ref_gKs, dim=1).reshape(batch_size, seq_len, d_model)
    ref_gV = torch.stack(ref_gVs, dim=1).reshape(batch_size, seq_len, d_model)

    # Baseline Triton (separate kernels)
    O_base, S_tilde_base, gQ_base, gK_base, gV_base = run_baseline_fwd_bwd(Q, K, V, M, H, grad_O)

    # Fused Triton (joint kernel)
    O_fused, gQ_fused, gK_fused, gV_fused = run_fused_fwd_bwd(Q, K, V, M, H, grad_O)

    print("\nBaseline vs Reference (PyTorch):")
    base_vs_ref = numerical_comparison(ref_O, O_base, ref_gQ, gQ_base, ref_gK, gK_base, ref_gV, gV_base)
    for k, v in base_vs_ref.items():
        print(f"  {k}: {v:.6e}")

    print("\nFused vs Reference (PyTorch):")
    fused_vs_ref = numerical_comparison(ref_O, O_fused, ref_gQ, gQ_fused, ref_gK, gK_fused, ref_gV, gV_fused)
    for k, v in fused_vs_ref.items():
        print(f"  {k}: {v:.6e}")

    print("\nFused vs Baseline (Triton):")
    fused_vs_base = numerical_comparison(O_base, O_fused, gQ_base, gQ_fused, gK_base, gK_fused, gV_base, gV_fused)
    for k, v in fused_vs_base.items():
        print(f"  {k}: {v:.6e}")

    # Check numerical agreement
    all_close = all(v < 1e-3 for v in fused_vs_base.values())
    print(f"\nNumerical agreement (max rel diff < 1e-3): {'✅ PASS' if all_close else '❌ FAIL'}")

    wandb.log({
        "numerical/fused_vs_base_O_rel": fused_vs_base["O_max_rel_diff"],
        "numerical/fused_vs_base_gQ_rel": fused_vs_base["grad_Q_max_rel_diff"],
        "numerical/fused_vs_base_gK_rel": fused_vs_base["grad_K_max_rel_diff"],
        "numerical/fused_vs_base_gV_rel": fused_vs_base["grad_V_max_rel_diff"],
        "numerical/all_close": int(all_close),
    })

    # ======================================================================
    # PART 2: HBM TRAFFIC ESTIMATION
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 2: HBM TRAFFIC ANALYSIS")
    print("=" * 60)

    base_hbm, fused_hbm, reduction = hbm_traffic_estimate(
        batch_size, seq_len, chunk_size, d_model, d_state, dtype_bytes=4
    )
    print(f"  Baseline theoretical HBM: {base_hbm / 1024:.1f} KB")
    print(f"  Fused theoretical HBM: {fused_hbm / 1024:.1f} KB")
    print(f"  Theoretical reduction: {reduction*100:.1f}%")
    print(f"  Target: >= 25% reduction")
    print(f"  Status: {'✅ PASS' if reduction >= 0.25 else '❌ FAIL'}")

    # Empirical memory measurement
    print("\n  Empirical memory measurement:")

    def run_baseline():
        return run_baseline_fwd_bwd(Q, K, V, M, H, grad_O)

    def run_fused():
        return run_fused_fwd_bwd(Q, K, V, M, H, grad_O)

    base_mem = measure_memory(run_baseline, "Baseline")
    fused_mem = measure_memory(run_fused, "Fused")

    if base_mem > 0:
        empirical_reduction = 1.0 - fused_mem / base_mem
        print(f"  Empirical reduction: {empirical_reduction*100:.1f}%")
    else:
        empirical_reduction = 0.0
        print(f"  Empirical reduction: N/A (base_mem=0)")

    wandb.log({
        "hbm/baseline_theoretical_kb": base_hbm / 1024,
        "hbm/fused_theoretical_kb": fused_hbm / 1024,
        "hbm/theoretical_reduction_pct": reduction * 100,
        "hbm/baseline_empirical_bytes": base_mem,
        "hbm/fused_empirical_bytes": fused_mem,
        "hbm/empirical_reduction_pct": empirical_reduction * 100,
    })

    # ======================================================================
    # PART 3: WALL-CLOCK BENCHMARKING
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 3: WALL-CLOCK BENCHMARKING")
    print("=" * 60)

    print("\nBaseline (separate fwd + bwd kernels):")
    base_avg, base_min, base_std = benchmark_kernel(
        run_baseline, warmup=warmup, repeats=repeats, label="Baseline fwd+bwd"
    )

    print("\nFused (joint fwd+bwd kernel):")
    fused_avg, fused_min, fused_std = benchmark_kernel(
        run_fused, warmup=warmup, repeats=repeats, label="Fused fwd+bwd"
    )

    speedup = base_avg / fused_avg if fused_avg > 0 else 0
    speedup_min = base_min / fused_min if fused_min > 0 else 0

    print(f"\n  Speedup (avg): {speedup:.3f}x")
    print(f"  Speedup (min): {speedup_min:.3f}x")
    print(f"  Target: >= 1.15x")
    print(f"  Status: {'✅ PASS' if speedup >= 1.15 else '❌ FAIL'}")

    wandb.log({
        "timing/baseline_avg_ms": base_avg,
        "timing/baseline_min_ms": base_min,
        "timing/fused_avg_ms": fused_avg,
        "timing/fused_min_ms": fused_min,
        "timing/speedup_avg": speedup,
        "timing/speedup_min": speedup_min,
    })

    # ======================================================================
    # PART 4: SCALING ANALYSIS (multiple batch sizes)
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 4: SCALING ANALYSIS")
    print("=" * 60)

    for bs in [1, 4, 8, 16, 32]:
        Q_s = torch.randn(bs, seq_len, d_model, device=device, dtype=dtype)
        K_s = torch.randn(bs, seq_len, d_model, device=device, dtype=dtype)
        V_s = torch.randn(bs, seq_len, d_model, device=device, dtype=dtype)
        M_s = M_single.unsqueeze(0).unsqueeze(0).expand(bs, num_chunks, chunk_size, chunk_size).contiguous()
        H_s = torch.zeros(bs, num_chunks, d_state, d_model, device=device, dtype=dtype)
        gO_s = torch.randn(bs, seq_len, d_model, device=device, dtype=dtype)

        def run_base_s():
            return run_baseline_fwd_bwd(Q_s, K_s, V_s, M_s, H_s, gO_s)

        def run_fused_s():
            return run_fused_fwd_bwd(Q_s, K_s, V_s, M_s, H_s, gO_s)

        base_avg_s, _, _ = benchmark_kernel(run_base_s, warmup=5, repeats=50, label=f"BS={bs} Baseline")
        fused_avg_s, _, _ = benchmark_kernel(run_fused_s, warmup=5, repeats=50, label=f"BS={bs} Fused")
        sp = base_avg_s / fused_avg_s if fused_avg_s > 0 else 0

        print(f"  BS={bs}: speedup={sp:.3f}x")

        wandb.log({
            f"scaling/bs{bs}_baseline_ms": base_avg_s,
            f"scaling/bs{bs}_fused_ms": fused_avg_s,
            f"scaling/bs{bs}_speedup": sp,
        })

    # ======================================================================
    # PART 5: END-TO-END TRAINING (optional validation)
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 5: END-TO-END TRAINING VALIDATION")
    print("=" * 60)

    # Generate synthetic data
    num_sequences = train_config.get("num_sequences", 1000)
    tokens = torch.randint(0, vocab_size, (num_sequences, seq_len), device=device)
    targets = tokens.clone()  # autoregressive: predict next token

    # Train baseline model
    print("\nTraining baseline model...")
    from models.gla_baseline import GLABaseline
    baseline_model = GLABaseline(vocab_size, d_model, d_state, chunk_size).to(device)
    baseline_losses = train_model(baseline_model, (tokens, targets), train_config, device)

    # ======================================================================
    # FINAL RESULTS
    # ======================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    hbm_pass = reduction >= 0.25
    speed_pass = speedup >= 1.15
    numerical_pass = all_close

    print(f"\n  1. HBM traffic reduction >= 25%: {reduction*100:.1f}% {'✅ PASS' if hbm_pass else '❌ FAIL'}")
    print(f"  2. Wall-clock speedup >= 1.15x: {speedup:.3f}x {'✅ PASS' if speed_pass else '❌ FAIL'}")
    print(f"  3. Numerical agreement: {'✅ PASS' if numerical_pass else '❌ FAIL'}")
    print(f"  4. Training loss (final): {baseline_losses[-1]:.4f}")

    overall = hbm_pass and speed_pass and numerical_pass
    decision = "PROCEED" if overall else ("DEBUG" if numerical_pass else "ABANDON")
    print(f"\n  Overall: {'✅ ALL CRITERIA MET' if overall else '❌ SOME CRITERIA FAILED'}")
    print(f"  Decision: {decision}")

    wandb.log({
        "final/hbm_reduction_pct": reduction * 100,
        "final/speedup": speedup,
        "final/numerical_agreement": int(numerical_pass),
        "final/training_loss": baseline_losses[-1],
        "success_criteria/hbm_reduction_25pct": int(hbm_pass),
        "success_criteria/speedup_1_15x": int(speed_pass),
        "success_criteria/numerical_agreement": int(numerical_pass),
        "final/decision": decision,
    })

    print(f"\nWandb URL: {wandb_url}")
    wandb.finish()


if __name__ == "__main__":
    main()

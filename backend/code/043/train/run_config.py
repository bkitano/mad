"""
Config-based training script for MVE 043: Newton-Schulz Orthogonal DeltaNet

Trains both UT baseline and NS variant on S3 permutation composition,
comparing accuracy and orthogonality error.

Usage:
    python -m train.run_config --config config.yaml
"""

import argparse
import time
import yaml
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deltanet import ChunkedDeltaNet
from tasks.s3.tokens import S3TokenSystem
from tasks.s3.dataset import S3CurriculumWrapper
from train.train import train_epoch, evaluate


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(model_config: dict, token_system: S3TokenSystem, use_ns: bool) -> ChunkedDeltaNet:
    """Create a ChunkedDeltaNet model."""
    return ChunkedDeltaNet(
        num_tokens=token_system.num_tokens,
        num_classes=token_system.num_group_elements,
        eos_idx=token_system.EOS_IDX,
        max_seq_len=model_config.get("max_seq_len", 64),
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        chunk_size=model_config.get("chunk_size", 32),
        dropout=model_config.get("dropout", 0.1),
        allow_neg_eigval=model_config.get("allow_neg_eigval", True),
        use_newton_schulz=use_ns,
        ns_iters=model_config.get("ns_iters", 2),
    )


def run_microbenchmark(model_config: dict, token_system: S3TokenSystem, device: str, n_warmup: int = 10, n_runs: int = 50):
    """
    Microbenchmark comparing per-chunk computation time between UT and NS variants.

    This directly measures the per-chunk kernel time - the primary speedup metric.
    """
    from models.deltanet import compute_ut_transform, compute_ns_transform

    d = model_config["d_model"] // model_config["nhead"]
    C = model_config.get("chunk_size", 32)
    nhead = model_config["nhead"]
    batch_size = 16
    ns_iters = model_config.get("ns_iters", 2)

    # Generate random key vectors and betas
    K = torch.randn(batch_size * nhead, C, d, device=device)
    K = K / (K.norm(dim=-1, keepdim=True) + 1e-6)  # L2 normalize
    beta = torch.sigmoid(torch.randn(batch_size * nhead, C, device=device)) * 2.0  # (0, 2)

    # Warmup
    for _ in range(n_warmup):
        _ = compute_ut_transform(K, beta, C)
        _ = compute_ns_transform(K, beta, C, ns_iters=ns_iters)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark UT
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = compute_ut_transform(K, beta, C)
    if device == "cuda":
        torch.cuda.synchronize()
    ut_time = (time.perf_counter() - t0) / n_runs

    # Benchmark NS
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = compute_ns_transform(K, beta, C, ns_iters=ns_iters)
    if device == "cuda":
        torch.cuda.synchronize()
    ns_time = (time.perf_counter() - t0) / n_runs

    return {
        "ut_time_ms": ut_time * 1000,
        "ns_time_ms": ns_time * 1000,
        "speedup": ut_time / ns_time if ns_time > 0 else 0,
        "chunk_size": C,
        "head_dim": d,
        "batch_heads": batch_size * nhead,
    }


def train_variant(
    variant_name: str,
    use_ns: bool,
    config: dict,
    token_system: S3TokenSystem,
    device: str,
    wandb_group: str,
):
    """Train a single variant (UT or NS) and return results."""
    model_config = config["model"]
    train_config = config["training"]
    dataset_config = config["dataset"]

    # Add max_seq_len to model config
    model_config["max_seq_len"] = dataset_config["max_seq_len"]

    # Create model
    model = create_model(model_config, token_system, use_ns=use_ns)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"Training {variant_name} (use_ns={use_ns})")
    print(f"Parameters: {param_count:,}")
    print(f"{'='*60}")

    # Init wandb run
    run_name = f"exp-043-{variant_name}"
    wandb_run = wandb.init(
        project="mad-architecture-search",
        name=run_name,
        group=wandb_group,
        config={
            "variant": variant_name,
            "use_newton_schulz": use_ns,
            "model": model_config,
            "training": train_config,
            "dataset": dataset_config,
            "proposal_id": "043-newton-schulz-orthogonal-deltanet-transition",
        },
        reinit=True,
    )
    wandb_url = wandb_run.get_url()
    print(f"Wandb URL: {wandb_url}")

    # Create curriculum
    curriculum = S3CurriculumWrapper(
        token_system=token_system,
        max_k=dataset_config["max_k"],
        samples_per_k=dataset_config["samples_per_k"],
        max_seq_len=dataset_config["max_seq_len"],
        test_size=dataset_config.get("test_size", 0.2),
        use_generators_only=dataset_config.get("use_generators_only", True),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.999))),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    best_val_acc = 0.0
    nan_count = 0

    # Curriculum training
    for stage in range(1, curriculum.num_stages() + 1):
        print(f"\nStage {stage}: k=1..{stage}")

        train_ds, test_ds = curriculum.get_stage(stage)
        train_loader = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=train_config["batch_size"], shuffle=False)

        max_epochs = train_config.get("max_epochs_per_stage", 100)
        max_val_acc_target = train_config.get("max_val_acc", 0.99)

        for epoch in range(max_epochs):
            loss, acc, orth_err, raw_orth_err, nan_det = train_epoch(
                model, train_loader, optimizer, criterion,
                device=device,
                max_grad_norm=train_config.get("gradient_clip", 1.0),
            )

            val_loss, val_acc, val_k_acc, val_orth_err, val_raw_orth_err = evaluate(
                model, test_loader, criterion, device=device,
            )

            global_step += 1

            if nan_det:
                nan_count += 1

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            log_dict = {
                "global_step": global_step,
                "stage": stage,
                "epoch": epoch + 1,
                "train/loss": loss,
                "train/accuracy": acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "orthogonality/ns_error": orth_err,
                "orthogonality/raw_error": raw_orth_err,
                "orthogonality/val_ns_error": val_orth_err,
                "orthogonality/val_raw_error": val_raw_orth_err,
                "stability/nan_count": nan_count,
            }
            for k, kacc in val_k_acc.items():
                log_dict[f"val/accuracy_k{k}"] = kacc
            wandb.log(log_dict)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: loss={loss:.4f} acc={acc:.4f} "
                      f"val_acc={val_acc:.4f} orth={orth_err:.6f} raw_orth={raw_orth_err:.6f}")

            if val_acc >= max_val_acc_target:
                print(f"  Reached target accuracy {max_val_acc_target:.2%}")
                break

    # Final evaluation
    _, final_test = curriculum.get_stage(curriculum.num_stages())
    final_loader = DataLoader(final_test, batch_size=train_config["batch_size"], shuffle=False)
    final_loss, final_acc, final_k_acc, final_orth, final_raw_orth = evaluate(
        model, final_loader, criterion, device=device,
    )

    print(f"\nFinal {variant_name}: acc={final_acc:.4f} orth_err={final_orth:.6f}")
    print(f"Per-k accuracy: {final_k_acc}")

    # Log final results
    wandb.log({
        "final/test_accuracy": final_acc,
        "final/test_loss": final_loss,
        "final/best_val_accuracy": best_val_acc,
        "final/orthogonality_error": final_orth,
        "final/raw_orthogonality_error": final_raw_orth,
        "final/nan_count": nan_count,
        "final/stable_training": nan_count == 0,
    })

    wandb.finish()

    return {
        "variant": variant_name,
        "final_accuracy": final_acc,
        "best_val_accuracy": best_val_acc,
        "final_orth_error": final_orth,
        "final_raw_orth_error": final_raw_orth,
        "nan_count": nan_count,
        "param_count": param_count,
        "per_k_accuracy": final_k_acc,
        "wandb_url": wandb_url,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Token system
    token_system = S3TokenSystem()
    print(f"S3 token system: {token_system.num_group_elements} elements, {token_system.num_tokens} tokens")

    # Wandb group for this experiment run
    wandb_group = f"exp-043-ns-deltanet-{time.strftime('%Y%m%d-%H%M%S')}"

    # ==========================================
    # 1. Run microbenchmark
    # ==========================================
    print("\n" + "="*60)
    print("MICROBENCHMARK: UT vs NS chunk computation")
    print("="*60)
    bench = run_microbenchmark(config["model"], token_system, device)
    print(f"  UT time: {bench['ut_time_ms']:.3f} ms")
    print(f"  NS time: {bench['ns_time_ms']:.3f} ms")
    print(f"  Speedup: {bench['speedup']:.3f}x")
    print(f"  Chunk size: {bench['chunk_size']}, Head dim: {bench['head_dim']}")

    # ==========================================
    # 2. Train UT baseline
    # ==========================================
    ut_results = train_variant(
        variant_name="ut-baseline",
        use_ns=False,
        config=config,
        token_system=token_system,
        device=device,
        wandb_group=wandb_group,
    )

    # ==========================================
    # 3. Train NS variant
    # ==========================================
    ns_results = train_variant(
        variant_name="ns-proposed",
        use_ns=True,
        config=config,
        token_system=token_system,
        device=device,
        wandb_group=wandb_group,
    )

    # ==========================================
    # 4. Summary comparison
    # ==========================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nUT Baseline:")
    print(f"  Final accuracy: {ut_results['final_accuracy']:.4f}")
    print(f"  Best val accuracy: {ut_results['best_val_accuracy']:.4f}")
    print(f"  NaN count: {ut_results['nan_count']}")
    print(f"  Wandb: {ut_results['wandb_url']}")

    print(f"\nNS Proposed:")
    print(f"  Final accuracy: {ns_results['final_accuracy']:.4f}")
    print(f"  Best val accuracy: {ns_results['best_val_accuracy']:.4f}")
    print(f"  Orth error (NS): {ns_results['final_orth_error']:.6f}")
    print(f"  Orth error (raw): {ns_results['final_raw_orth_error']:.6f}")
    print(f"  NaN count: {ns_results['nan_count']}")
    print(f"  Wandb: {ns_results['wandb_url']}")

    print(f"\nMicrobenchmark:")
    print(f"  UT: {bench['ut_time_ms']:.3f} ms")
    print(f"  NS: {bench['ns_time_ms']:.3f} ms")
    print(f"  Speedup: {bench['speedup']:.3f}x")

    # ==========================================
    # 5. Success criteria evaluation
    # ==========================================
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)

    # Criterion 1: NS achieves >90% S3 accuracy (matching UT within 3%)
    ns_acc = ns_results['final_accuracy']
    ut_acc = ut_results['final_accuracy']
    acc_gap = abs(ut_acc - ns_acc)
    criterion_1 = ns_acc > 0.90 and acc_gap < 0.03
    print(f"1. NS >90% accuracy, within 3% of UT: {'PASS' if criterion_1 else 'FAIL'}")
    print(f"   NS={ns_acc:.4f}, UT={ut_acc:.4f}, gap={acc_gap:.4f}")

    # Criterion 2: Orthogonality error < 1e-3
    orth_err = ns_results['final_orth_error']
    criterion_2 = orth_err < 1e-3
    print(f"2. Orth error < 1e-3: {'PASS' if criterion_2 else 'FAIL'}")
    print(f"   Error={orth_err:.6e}")

    # Criterion 3: Kernel speedup (NS <= 0.8x UT time)
    speedup = bench['speedup']
    criterion_3 = speedup >= 1.25  # 1/0.8 = 1.25x speedup
    print(f"3. NS kernel <= 0.8x UT time (>= 1.25x speedup): {'PASS' if criterion_3 else 'FAIL'}")
    print(f"   Speedup={speedup:.3f}x")

    # Criterion 4: Stable training (no NaN/Inf)
    criterion_4 = ns_results['nan_count'] == 0
    print(f"4. Stable training (no NaN): {'PASS' if criterion_4 else 'FAIL'}")
    print(f"   NaN count={ns_results['nan_count']}")

    all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Log summary to a final wandb run
    summary_run = wandb.init(
        project="mad-architecture-search",
        name=f"exp-043-summary",
        group=wandb_group,
        config=config,
        reinit=True,
    )
    wandb.log({
        "summary/ut_accuracy": ut_acc,
        "summary/ns_accuracy": ns_acc,
        "summary/accuracy_gap": acc_gap,
        "summary/orth_error": orth_err,
        "summary/kernel_speedup": speedup,
        "summary/ns_stable": criterion_4,
        "success_criteria/accuracy_match": criterion_1,
        "success_criteria/orth_error": criterion_2,
        "success_criteria/kernel_speedup": criterion_3,
        "success_criteria/stable_training": criterion_4,
        "success_criteria/all_pass": all_pass,
        "benchmark/ut_time_ms": bench["ut_time_ms"],
        "benchmark/ns_time_ms": bench["ns_time_ms"],
    })
    wandb.finish()


if __name__ == "__main__":
    main()

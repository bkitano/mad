"""
Training script for MVE 059: Second-Order KDA (SO-KDA)

Runs three variants on MQAR:
1. GLA (Gated Linear Attention) - no delta rule baseline
2. KDA (Kimi Delta Attention) - standard delta rule
3. SO-KDA (Second-Order KDA) - adapted delta rule with key metric

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --variant so_kda
"""

import argparse
import os
import sys
import time
import math
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.gla import GLAModel
from models.kda import KDAModel
from models.so_kda import SOKDAModel
from data.generate import generate_mqar_data


def build_model(variant: str, config: dict, device: torch.device) -> nn.Module:
    """Build model for the specified variant."""
    model_cfg = config["model"]
    data_cfg = config["data"]

    common_kwargs = dict(
        vocab_size=data_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_head=model_cfg["d_head"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_seq_len"],
    )

    if variant == "gla":
        model = GLAModel(**common_kwargs)
    elif variant == "kda":
        model = KDAModel(**common_kwargs)
    elif variant == "so_kda":
        model = SOKDAModel(
            gamma_m_init=model_cfg.get("gamma_m_init", 0.99),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model.to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, query_mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        query_mask = query_mask.to(device)

        optimizer.zero_grad()

        logits = model(inputs)  # (B, T, V)

        # Only compute loss at query positions
        mask = query_mask.bool()
        if mask.sum() == 0:
            continue

        logits_masked = logits[mask]  # (num_queries, V)
        targets_masked = targets[mask]  # (num_queries,)

        loss = F.cross_entropy(logits_masked, targets_masked)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        preds = logits_masked.argmax(dim=-1)
        correct += (preds == targets_masked).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, query_mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        query_mask = query_mask.to(device)

        logits = model(inputs)

        mask = query_mask.bool()
        if mask.sum() == 0:
            continue

        logits_masked = logits[mask]
        targets_masked = targets[mask]

        loss = F.cross_entropy(logits_masked, targets_masked)

        total_loss += loss.item() * mask.sum().item()
        preds = logits_masked.argmax(dim=-1)
        correct += (preds == targets_masked).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_single_variant(
    variant: str,
    config: dict,
    device: torch.device,
    train_ds,
    val_ds,
    test_ds,
    gen_ds=None,
    use_wandb: bool = True,
) -> dict:
    """Train and evaluate a single variant."""
    train_cfg = config["training"]
    seed = config.get("seed", 42)

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"])
    gen_loader = DataLoader(gen_ds, batch_size=train_cfg["batch_size"]) if gen_ds else None

    # Build model
    model = build_model(variant, config, device)
    param_info = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Variant: {variant.upper()}")
    print(f"Parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")
    print(f"{'='*60}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg["warmup_epochs"], train_cfg["epochs"]
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    epoch_times = []

    for epoch in range(train_cfg["epochs"]):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, train_cfg["grad_clip"]
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Log to wandb
        if use_wandb and HAS_WANDB and wandb.run is not None:
            log_dict = {
                f"{variant}/train_loss": train_loss,
                f"{variant}/train_acc": train_acc,
                f"{variant}/val_loss": val_loss,
                f"{variant}/val_acc": val_acc,
                f"{variant}/epoch_time": epoch_time,
                f"{variant}/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            # Log gamma_M values for SO-KDA
            if variant == "so_kda" and hasattr(model, "get_gamma_m_values"):
                gamma_values = model.get_gamma_m_values()
                for layer_name, values in gamma_values.items():
                    for h_idx, val in enumerate(values):
                        log_dict[f"{variant}/gamma_m/{layer_name}/head_{h_idx}"] = val

            wandb.log(log_dict)

        # Print progress
        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(
                f"  Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
                f"| Val Loss {val_loss:.4f} Acc {val_acc:.4f} | Time {epoch_time:.1f}s"
            )

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if val_acc >= 0.99:
            print(f"  >> Early stop: val accuracy reached {val_acc:.4f} at epoch {epoch}")
            break

        if patience_counter >= train_cfg["patience"]:
            print(f"  >> Early stop: no improvement for {train_cfg['patience']} epochs")
            break

    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n  Test: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")

    # Length generalization
    gen_acc = None
    if gen_loader is not None:
        gen_loss, gen_acc = evaluate(model, gen_loader, device)
        print(f"  Gen (T={config['data']['gen_seq_len']}): Loss {gen_loss:.4f}, Accuracy {gen_acc:.4f}")

    results = {
        "variant": variant,
        "params": param_info,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "gen_acc": gen_acc,
        "total_time_s": total_time,
        "avg_epoch_time_s": avg_epoch_time,
        "epochs_trained": epoch + 1,
    }

    # Add gamma_M values for SO-KDA
    if variant == "so_kda" and hasattr(model, "get_gamma_m_values"):
        results["gamma_m_values"] = model.get_gamma_m_values()

    return results


def main():
    parser = argparse.ArgumentParser(description="MVE 059: Second-Order KDA")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--variant", type=str, default=None, help="Run only this variant")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="mad-architecture-search",
            name="exp-059-so-kda",
            config=config,
            tags=["mve", "059", "so-kda", "mqar"],
        )
        print(f"Wandb URL: {wandb.run.get_url()}")

    # Generate data
    print("\nGenerating MQAR data...")
    train_ds, val_ds, test_ds, gen_ds = generate_mqar_data(config, seed=config["seed"])
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    if gen_ds:
        print(f"  Gen (T={config['data']['gen_seq_len']}): {len(gen_ds)} samples")

    # Determine which variants to run
    variants = [args.variant] if args.variant else config.get("variants", ["gla", "kda", "so_kda"])

    # Run each variant
    all_results = {}
    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"# Running variant: {variant.upper()}")
        print(f"{'#'*60}")

        results = run_single_variant(
            variant=variant,
            config=config,
            device=device,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            gen_ds=gen_ds,
            use_wandb=use_wandb,
        )
        all_results[variant] = results

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Variant':<12} {'Params':>10} {'Test Acc':>10} {'Gen Acc':>10} {'Time':>10} {'Epochs':>8}")
    print(f"{'-'*70}")

    for variant, r in all_results.items():
        gen_str = f"{r['gen_acc']:.4f}" if r['gen_acc'] is not None else "N/A"
        print(
            f"{variant:<12} {r['params']['total']:>10,} {r['test_acc']:>10.4f} "
            f"{gen_str:>10} {r['total_time_s']:>9.1f}s {r['epochs_trained']:>8}"
        )

    # Check success/failure criteria
    print(f"\n{'='*70}")
    print(f"{'SUCCESS CRITERIA':^70}")
    print(f"{'='*70}")

    criteria = config.get("success_criteria", {})
    criteria_results = {}

    # Check SO-KDA accuracy
    if "so_kda" in all_results:
        so_kda_target = criteria.get("so_kda_accuracy", 0.95)
        so_kda_actual = all_results["so_kda"]["test_acc"]
        passed = so_kda_actual >= so_kda_target
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] SO-KDA accuracy: {so_kda_actual:.4f} (target: >= {so_kda_target})")
        criteria_results["so_kda_accuracy"] = {
            "target": so_kda_target,
            "actual": so_kda_actual,
            "passed": passed,
        }

    # Check relative performance: SO-KDA > KDA > GLA
    if "so_kda" in all_results and "kda" in all_results:
        so_kda_better = all_results["so_kda"]["test_acc"] > all_results["kda"]["test_acc"]
        status = "PASS" if so_kda_better else "FAIL"
        print(f"  [{status}] SO-KDA > KDA: {all_results['so_kda']['test_acc']:.4f} vs {all_results['kda']['test_acc']:.4f}")
        criteria_results["so_kda_better_than_kda"] = {
            "so_kda": all_results["so_kda"]["test_acc"],
            "kda": all_results["kda"]["test_acc"],
            "passed": so_kda_better,
        }

    if "kda" in all_results and "gla" in all_results:
        kda_better = all_results["kda"]["test_acc"] > all_results["gla"]["test_acc"]
        status = "PASS" if kda_better else "FAIL"
        print(f"  [{status}] KDA > GLA: {all_results['kda']['test_acc']:.4f} vs {all_results['gla']['test_acc']:.4f}")
        criteria_results["kda_better_than_gla"] = {
            "kda": all_results["kda"]["test_acc"],
            "gla": all_results["gla"]["test_acc"],
            "passed": kda_better,
        }

    # Check timing overhead: SO-KDA step time < 2x KDA
    if "so_kda" in all_results and "kda" in all_results:
        ratio = all_results["so_kda"]["avg_epoch_time_s"] / max(all_results["kda"]["avg_epoch_time_s"], 1e-6)
        overhead_ok = ratio < 2.0
        status = "PASS" if overhead_ok else "FAIL"
        print(f"  [{status}] SO-KDA/KDA time ratio: {ratio:.2f}x (target: < 2.0x)")
        criteria_results["timing_overhead"] = {
            "ratio": ratio,
            "target_max": 2.0,
            "passed": overhead_ok,
        }

    # Check length generalization
    if "so_kda" in all_results and all_results["so_kda"]["gen_acc"] is not None:
        gen_acc = all_results["so_kda"]["gen_acc"]
        gen_ok = gen_acc > 0.5  # At least above chance
        status = "PASS" if gen_ok else "FAIL"
        print(f"  [{status}] SO-KDA length gen (T=1024): {gen_acc:.4f} (target: > 0.50)")
        criteria_results["length_generalization"] = {
            "actual": gen_acc,
            "target": 0.50,
            "passed": gen_ok,
        }

    # Overall verdict
    all_passed = all(v.get("passed", False) for v in criteria_results.values())
    so_kda_beats_kda = criteria_results.get("so_kda_better_than_kda", {}).get("passed", False)

    if all_passed:
        verdict = "PROCEED"
    elif so_kda_beats_kda:
        verdict = "DEBUG"
    else:
        verdict = "ABANDON"

    print(f"\n  VERDICT: {verdict}")

    # Log to wandb
    if use_wandb and HAS_WANDB and wandb.run is not None:
        final_log = {"verdict": verdict}
        for variant, r in all_results.items():
            final_log[f"final/{variant}/test_acc"] = r["test_acc"]
            final_log[f"final/{variant}/best_val_acc"] = r["best_val_acc"]
            if r["gen_acc"] is not None:
                final_log[f"final/{variant}/gen_acc"] = r["gen_acc"]
            final_log[f"final/{variant}/total_time_s"] = r["total_time_s"]

        for name, cr in criteria_results.items():
            final_log[f"success_criteria/{name}"] = cr.get("passed", False)

        wandb.log(final_log)
        wandb.finish()

    # Save results to file
    results_path = Path(args.config).parent / "results.json"
    save_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": config,
        "results": {},
        "criteria": criteria_results,
        "verdict": verdict,
    }
    for variant, r in all_results.items():
        save_results["results"][variant] = {
            k: v for k, v in r.items()
            if not isinstance(v, (torch.Tensor,))
        }

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

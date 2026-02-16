"""
Training script for MVE 064 (GHLA): Gated Second-Order Linear Attention

Trains and evaluates 4 model variants on MQAR:
1. GLA (first-order, gated) — baseline
2. HLA (second-order, ungated) — baseline
3. HLA-decay (second-order, fixed gamma=0.99) — baseline
4. GHLA (second-order, data-dependent gating) — proposed

Success criteria (from proposal):
- GHLA > 90% accuracy at 8 associations where GLA < 70%
- GHLA > HLA by > 5% accuracy (demonstrating gating value)
- GHLA >= HLA-decay (data-dependent gating at least as good)

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --variant ghla
"""

import argparse
import os
import sys
import time
import math
import yaml
import json
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.ghla import GHLAModel
from models.gla import GLAModel
from models.hla import HLAModel, HLADecayModel
from data.mqar import generate_mqar_data, MQARDataset


MODEL_REGISTRY = {
    "ghla": GHLAModel,
    "gla": GLAModel,
    "hla": HLAModel,
    "hla_decay": HLADecayModel,
}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant: str, config: dict, device: torch.device) -> nn.Module:
    """Build model for the specified variant."""
    model_cfg = config["model"]
    data_cfg = config["data"]

    common_kwargs = dict(
        vocab_size=data_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        d_k=model_cfg["d_k"],
        d_v=model_cfg["d_v"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_seq_len=data_cfg["seq_len"],
    )

    if variant == "hla_decay":
        common_kwargs["gamma"] = model_cfg.get("gamma", 0.99)

    model_cls = MODEL_REGISTRY[variant]
    model = model_cls(**common_kwargs)
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

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (B, T, V)

        # Only compute loss at answer positions (target != -100)
        mask = targets != -100
        if mask.sum() == 0:
            continue

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        preds = logits_flat.argmax(dim=-1)
        valid = targets_flat != -100
        correct += (preds[valid] == targets_flat[valid]).sum().item()
        total += valid.sum().item()

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

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        mask = targets_flat != -100
        total_loss += loss.item() * mask.sum().item()
        preds = logits_flat.argmax(dim=-1)
        correct += (preds[mask] == targets_flat[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), 0.01)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.01)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_single_variant(
    variant: str,
    config: dict,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    use_wandb: bool = True,
) -> dict:
    """Train and evaluate a single variant."""
    train_cfg = config["training"]
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_model(variant, config, device)
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Variant: {variant.upper()}")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*60}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg.get("warmup_epochs", 5), train_cfg["epochs"]
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(train_cfg["epochs"]):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, train_cfg.get("grad_clip", 1.0)
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log to wandb
        if use_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({
                f"{variant}/train_loss": train_loss,
                f"{variant}/train_acc": train_acc,
                f"{variant}/val_loss": val_loss,
                f"{variant}/val_acc": val_acc,
                f"{variant}/epoch_time": epoch_time,
                f"{variant}/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

        # Print progress
        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(
                f"  Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
                f"| Val Loss {val_loss:.4f} Acc {val_acc:.4f} | {epoch_time:.1f}s"
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
            print(f"  >> Early stop: val accuracy reached {val_acc:.4f}")
            break

        if patience_counter >= train_cfg.get("patience", 30):
            print(f"  >> Early stop: no improvement for {train_cfg.get('patience', 30)} epochs")
            break

        if math.isnan(train_loss):
            print(f"  >> ABORT: NaN detected at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n  Test: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")

    results = {
        "variant": variant,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_time_s": total_time,
        "epochs_trained": epoch + 1,
        "converged": not math.isnan(train_loss),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="MVE 064: Gated Second-Order Linear Attention (GHLA)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variant", type=str, default=None, help="Run only this variant")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
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
            name="exp-064-ghla-mqar",
            config=config,
            tags=["mve", "064", "ghla", "gla", "hla", "second-order", "mqar"],
        )
        print(f"Wandb URL: {wandb.run.get_url()}")

    # Generate MQAR data
    data_cfg = config["data"]
    print(f"\nGenerating MQAR data...")
    print(f"  {data_cfg['num_kv_pairs']} KV pairs, vocab_size={data_cfg['vocab_size']}, seq_len={data_cfg['seq_len']}")

    # Generate train, val, test
    total_samples = data_cfg["num_train"] + data_cfg["num_val"] + data_cfg["num_test"]
    inputs, targets = generate_mqar_data(
        num_samples=total_samples,
        num_kv_pairs=data_cfg["num_kv_pairs"],
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
        seed=config.get("seed", 42),
    )

    # Split into train/val/test
    full_dataset = MQARDataset(inputs, targets)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [data_cfg["num_train"], data_cfg["num_val"], data_cfg["num_test"]],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Determine variants to run
    variants = [args.variant] if args.variant else config.get("variants", ["gla", "hla", "hla_decay", "ghla"])

    # Run each variant
    all_results = {}
    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"# Running variant: {variant.upper()}")
        print(f"{'#'*60}")

        try:
            results = run_single_variant(
                variant=variant,
                config=config,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                use_wandb=use_wandb,
            )
            all_results[variant] = results
        except Exception as e:
            print(f"  ERROR: {variant} failed with: {e}")
            traceback.print_exc()
            all_results[variant] = {
                "variant": variant,
                "error": str(e),
                "test_acc": 0.0,
                "best_val_acc": 0.0,
                "converged": False,
            }

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Variant':<15} {'Params':>10} {'Test Acc':>10} {'Best Val':>10} {'Time':>10}")
    print(f"{'-'*70}")

    for variant, r in all_results.items():
        params = r.get('n_params', 'N/A')
        time_s = r.get('total_time_s', 0)
        print(
            f"{variant:<15} {str(params):>10} {r['test_acc']:>10.4f} "
            f"{r['best_val_acc']:>10.4f} {time_s:>9.1f}s"
        )

    # Check success criteria
    print(f"\n{'='*70}")
    print(f"{'SUCCESS CRITERIA':^70}")
    print(f"{'='*70}")

    criteria_results = {}

    # Criterion 1: GHLA > 90% where GLA < 70%
    if "ghla" in all_results and "gla" in all_results:
        ghla_acc = all_results["ghla"]["test_acc"]
        gla_acc = all_results["gla"]["test_acc"]
        # The criterion is GHLA > 90% AND GLA < 70%
        # If GLA already > 70%, we need to check relative improvement
        ghla_high = ghla_acc > 0.90
        gla_low = gla_acc < 0.70
        passed = ghla_high and gla_low
        # Also check if GHLA beats GLA significantly even if GLA is > 70%
        ghla_beats_gla = ghla_acc > gla_acc + 0.05

        status = "PASS" if passed else "CONDITIONAL" if ghla_beats_gla else "FAIL"
        print(f"  [{status}] GHLA > 90% & GLA < 70%: GHLA={ghla_acc:.4f}, GLA={gla_acc:.4f}")
        if not passed and ghla_beats_gla:
            print(f"         (GLA > 70% but GHLA still beats GLA by {ghla_acc - gla_acc:.4f})")
        criteria_results["ghla_90_gla_70"] = {
            "ghla_acc": ghla_acc,
            "gla_acc": gla_acc,
            "ghla_beats_gla_by": ghla_acc - gla_acc,
            "passed": passed,
            "ghla_beats_gla": ghla_beats_gla,
        }

    # Criterion 2: GHLA > HLA by > 5% (gating value)
    if "ghla" in all_results and "hla" in all_results:
        ghla_acc = all_results["ghla"]["test_acc"]
        hla_acc = all_results["hla"]["test_acc"]
        improvement = ghla_acc - hla_acc
        passed = improvement > 0.05
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] GHLA > HLA by > 5%: improvement = {improvement:.4f} ({ghla_acc:.4f} vs {hla_acc:.4f})")
        criteria_results["ghla_over_hla_5pct"] = {
            "improvement": improvement,
            "ghla_acc": ghla_acc,
            "hla_acc": hla_acc,
            "passed": passed,
        }

    # Criterion 3: GHLA >= HLA-decay
    if "ghla" in all_results and "hla_decay" in all_results:
        ghla_acc = all_results["ghla"]["test_acc"]
        hla_decay_acc = all_results["hla_decay"]["test_acc"]
        passed = ghla_acc >= hla_decay_acc
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] GHLA >= HLA-decay: {ghla_acc:.4f} vs {hla_decay_acc:.4f}")
        criteria_results["ghla_geq_hla_decay"] = {
            "ghla_acc": ghla_acc,
            "hla_decay_acc": hla_decay_acc,
            "passed": passed,
        }

    # Convergence check
    all_converged = all(r.get("converged", False) for r in all_results.values())
    status = "PASS" if all_converged else "FAIL"
    print(f"  [{status}] All variants converged: {all_converged}")
    criteria_results["convergence"] = {"passed": all_converged}

    # Overall verdict
    primary_passed = all(
        criteria_results.get(c, {}).get("passed", False)
        for c in ["ghla_90_gla_70", "ghla_over_hla_5pct", "ghla_geq_hla_decay"]
    )
    any_ghla_advantage = criteria_results.get("ghla_90_gla_70", {}).get("ghla_beats_gla", False)

    if primary_passed:
        verdict = "PROCEED"
    elif any_ghla_advantage and criteria_results.get("convergence", {}).get("passed", False):
        verdict = "PROCEED (partial - GHLA shows advantage but thresholds not all met)"
    elif criteria_results.get("convergence", {}).get("passed", False):
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
            if "n_params" in r:
                final_log[f"final/{variant}/params"] = r["n_params"]

        for name, cr in criteria_results.items():
            final_log[f"success_criteria/{name}"] = cr.get("passed", False)

        wandb.log(final_log)
        wandb.finish()

    # Save results to JSON
    results_path = Path(args.config).parent / "results.json"
    save_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": config,
        "results": all_results,
        "criteria": criteria_results,
        "verdict": verdict,
    }

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

"""
Training script for MVE 030: Group-Matrix Displacement Rank SSM

Runs ablation over displacement rank r in {0, 1, 2, 4} plus diagonal baseline
on S3 and D4 state tracking tasks.

Success criteria (from proposal):
    1. GM-DR-SSM (r=2) > 95% accuracy on S3 (6 states)
    2. GM-DR-SSM (r=2) > 90% accuracy on D4 (8 states)
    3. Diagonal SSM < 30% on both (near random)
    4. r=0 underperforms r=2 by >10% (displacement rank adds value)

Failure criteria:
    - GM-DR-SSM (r=2) < 50% on S3 -> kill the idea
    - r=0 matches r=2 -> displacement rank adds no value
    - Model doesn't converge within 1000 steps -> optimization pathological

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.generate import create_datasets
from models.gmdr_ssm import GMDRSSMClassifier
from models.diagonal_ssm import DiagonalSSMClassifier


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for tokens, targets in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tokens)  # (batch, seq_len, num_classes)

        # Reshape for cross-entropy
        batch, seq_len, num_classes = logits.shape
        loss = criterion(logits.reshape(-1, num_classes), targets.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += batch * seq_len

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for tokens, targets in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        logits = model(tokens)
        batch, seq_len, num_classes = logits.shape
        loss = criterion(logits.reshape(-1, num_classes), targets.reshape(-1))

        total_loss += loss.item() * batch
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += batch * seq_len

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: str,
    model_name: str,
) -> dict:
    """Train a model and return results."""
    training_cfg = config["training"]
    lr = training_cfg["lr"]
    max_epochs = training_cfg["max_epochs"]
    patience = training_cfg["patience"]
    max_grad_norm = training_cfg.get("gradient_clip", 1.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({n_params:,} params)")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    nan_count = 0
    history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )

        # Check for NaN
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            nan_count += 1
            print(f"  Epoch {epoch}: NaN/Inf detected! (count={nan_count})")
            if nan_count >= 5:
                print("  ABORTING: Too many NaN/Inf events")
                break
            continue

        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()

        # Log progress
        if epoch % 10 == 0 or epoch <= 5 or val_acc > 0.9:
            print(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} best={best_val_acc:.3f}"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Early stopping
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

        # Stop if converged
        if val_acc >= 0.99:
            print(f"  Converged at epoch {epoch} (val_acc={val_acc:.3f})")
            break

    elapsed = time.time() - start_time

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    results = {
        "model_name": model_name,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "nan_count": nan_count,
        "total_epochs": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
    }

    print(f"\n  FINAL: test_acc={test_acc:.3f} best_val_acc={best_val_acc:.3f} "
          f"epochs={len(history)} time={elapsed:.1f}s nan_count={nan_count}")

    return results


def run_experiment(config: dict) -> dict:
    """Run the full MVE experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    d_model = model_cfg["d_model"]
    n = model_cfg["n"]
    num_layers = model_cfg["num_layers"]
    kernel_size = model_cfg["kernel_size"]
    perturbation_scale = model_cfg["perturbation_scale"]
    max_seq_len = dataset_cfg["seq_len"] + 1

    disp_ranks = model_cfg["disp_ranks"]  # [0, 1, 2, 4]
    groups = dataset_cfg["groups"]  # ["s3", "d4"]
    batch_size = training_cfg["batch_size"]

    all_results = {}

    for group_name in groups:
        print(f"\n{'#'*70}")
        print(f"# Group: {group_name.upper()}")
        print(f"{'#'*70}")

        # Create datasets
        train_ds, test_ds = create_datasets(
            group_name=group_name,
            num_train=dataset_cfg["num_train"],
            num_test=dataset_cfg["num_test"],
            seq_len=dataset_cfg["seq_len"],
            seed=dataset_cfg.get("seed", 42),
        )
        vocab_size = train_ds.vocab_size
        num_classes = train_ds.num_classes

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

        print(f"Data: {len(train_ds)} train, {len(test_ds)} test, "
              f"vocab={vocab_size}, classes={num_classes}, seq_len={dataset_cfg['seq_len']}")

        group_results = {}

        # --- GM-DR-SSM with different displacement ranks ---
        for r in disp_ranks:
            model_name = f"GM-DR-SSM(r={r})"
            model = GMDRSSMClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                d_model=d_model,
                n=n,
                num_layers=num_layers,
                kernel_size=kernel_size,
                disp_rank=r,
                perturbation_scale=perturbation_scale,
                max_seq_len=max_seq_len,
            )
            results = train_model(
                model, train_loader, test_loader, config, device, model_name
            )
            group_results[model_name] = results

        # --- Diagonal SSM baseline ---
        model_name = "DiagonalSSM"
        model = DiagonalSSMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=d_model,
            n=n,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        results = train_model(
            model, train_loader, test_loader, config, device, model_name
        )
        group_results[model_name] = results

        all_results[group_name] = group_results

    return all_results


def evaluate_criteria(results: dict) -> dict:
    """Evaluate success/failure criteria from the proposal."""
    criteria = {}

    # S3 criteria
    if "s3" in results:
        s3 = results["s3"]
        r2_acc = s3.get("GM-DR-SSM(r=2)", {}).get("test_acc", 0)
        r0_acc = s3.get("GM-DR-SSM(r=0)", {}).get("test_acc", 0)
        diag_acc = s3.get("DiagonalSSM", {}).get("test_acc", 0)

        criteria["s3_r2_gt_95"] = {
            "description": "GM-DR-SSM(r=2) > 95% on S3",
            "target": 0.95,
            "achieved": r2_acc,
            "pass": r2_acc > 0.95,
        }
        criteria["s3_diag_lt_30"] = {
            "description": "DiagonalSSM < 30% on S3",
            "target": 0.30,
            "achieved": diag_acc,
            "pass": diag_acc < 0.30,
        }
        criteria["s3_r0_vs_r2"] = {
            "description": "r=0 underperforms r=2 by >10% on S3",
            "target": 0.10,
            "achieved": r2_acc - r0_acc,
            "pass": (r2_acc - r0_acc) > 0.10,
        }
        # Failure criterion
        criteria["s3_kill_check"] = {
            "description": "KILL if GM-DR-SSM(r=2) < 50% on S3",
            "target": 0.50,
            "achieved": r2_acc,
            "pass": r2_acc >= 0.50,
        }

    # D4 criteria
    if "d4" in results:
        d4 = results["d4"]
        r2_acc = d4.get("GM-DR-SSM(r=2)", {}).get("test_acc", 0)
        r0_acc = d4.get("GM-DR-SSM(r=0)", {}).get("test_acc", 0)
        diag_acc = d4.get("DiagonalSSM", {}).get("test_acc", 0)

        criteria["d4_r2_gt_90"] = {
            "description": "GM-DR-SSM(r=2) > 90% on D4",
            "target": 0.90,
            "achieved": r2_acc,
            "pass": r2_acc > 0.90,
        }
        criteria["d4_diag_lt_30"] = {
            "description": "DiagonalSSM < 30% on D4",
            "target": 0.30,
            "achieved": diag_acc,
            "pass": diag_acc < 0.30,
        }
        criteria["d4_r0_vs_r2"] = {
            "description": "r=0 underperforms r=2 by >10% on D4",
            "target": 0.10,
            "achieved": r2_acc - r0_acc,
            "pass": (r2_acc - r0_acc) > 0.10,
        }

    return criteria


def print_summary(results: dict, criteria: dict):
    """Print a formatted summary of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY: GM-DR-SSM MVE 030")
    print("=" * 70)

    for group_name, group_results in results.items():
        print(f"\n--- {group_name.upper()} State Tracking ---")
        print(f"{'Model':<25} {'Params':>8} {'Test Acc':>10} {'Val Acc':>10} {'Epochs':>8} {'NaN':>5}")
        print("-" * 70)
        for model_name, r in sorted(group_results.items()):
            print(
                f"{model_name:<25} {r['n_params']:>8,} {r['test_acc']:>10.3f} "
                f"{r['best_val_acc']:>10.3f} {r['total_epochs']:>8} {r['nan_count']:>5}"
            )

    print(f"\n--- Success Criteria ---")
    n_pass = 0
    n_total = 0
    for key, c in criteria.items():
        status = "PASS" if c["pass"] else "FAIL"
        icon = "\u2705" if c["pass"] else "\u274c"
        print(f"  {icon} {c['description']}: {c['achieved']:.3f} (target: {c['target']}) [{status}]")
        n_total += 1
        if c["pass"]:
            n_pass += 1

    print(f"\n  Result: {n_pass}/{n_total} criteria passed")

    # Decision
    kill = any(
        not c["pass"]
        for key, c in criteria.items()
        if "kill" in key
    )
    if kill:
        print("\n  DECISION: ABANDON (kill criterion triggered)")
    elif n_pass >= n_total * 0.75:
        print("\n  DECISION: PROCEED")
    elif n_pass >= n_total * 0.5:
        print("\n  DECISION: DEBUG (partial success)")
    else:
        print("\n  DECISION: ABANDON")


def main():
    parser = argparse.ArgumentParser(description="MVE 030: GM-DR-SSM Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("MVE 030: Group-Matrix Displacement Rank SSM")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Model: d_model={config['model']['d_model']}, n={config['model']['n']}, "
          f"layers={config['model']['num_layers']}")
    print(f"Displacement ranks: {config['model']['disp_ranks']}")
    print(f"Groups: {config['dataset']['groups']}")
    print(f"Seq len: {config['dataset']['seq_len']}")

    # Run experiment
    results = run_experiment(config)

    # Evaluate criteria
    criteria = evaluate_criteria(results)

    # Print summary
    print_summary(results, criteria)

    # Save results
    output_dir = Path(__file__).parent
    results_file = output_dir / "results.json"

    # Strip history for JSON serialization (too large)
    results_compact = {}
    for group_name, group_results in results.items():
        results_compact[group_name] = {}
        for model_name, r in group_results.items():
            r_compact = {k: v for k, v in r.items() if k != "history"}
            results_compact[group_name][model_name] = r_compact

    output = {
        "results": results_compact,
        "criteria": criteria,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

"""
Training script for MVE 014: Log-Semiring SSM on Selective Copying.

Trains three models and compares accuracy:
1. LogSSM (proposed) — expected > 90% accuracy
2. Linear Attention (baseline) — expected < 60% accuracy
3. Diagonal SSM (baseline) — expected < 70% accuracy

Success criteria (from proposal):
- LogSSM achieves > 90% accuracy on selective copying at length 32
- Standard linear attention achieves < 60% on the same task
- Diagonal SSM achieves < 70% on the same task
- Training converges within 500 steps (no instability)
"""

import os
import sys
import time
import yaml
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.log_ssm import LogSSMClassifier
from models.baselines import LinearAttentionClassifier, DiagonalSSMClassifier
from data.selective_copy import generate_selective_copy_data


IGNORE_INDEX = -100


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns (loss, accuracy, nan_count)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0
    n_batches = 0

    for tokens, targets, query_pos in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tokens)  # (batch, seq_len, num_classes)

        # Flatten for loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Accuracy on query positions only
        preds = logits.argmax(dim=-1)  # (batch, seq_len)
        for i in range(tokens.size(0)):
            qp = query_pos[i].item()
            if targets[i, qp] != IGNORE_INDEX:
                correct += (preds[i, qp] == targets[i, qp]).item()
                total += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, nan_count


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for tokens, targets, query_pos in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            logits = model(tokens)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            loss = criterion(logits_flat, targets_flat)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1

            preds = logits.argmax(dim=-1)
            for i in range(tokens.size(0)):
                qp = query_pos[i].item()
                if targets[i, qp] != IGNORE_INDEX:
                    correct += (preds[i, qp] == targets[i, qp]).item()
                    total += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: str,
) -> dict:
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"]["lr"] * 0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_val_acc = 0.0
    best_epoch = 0
    patience = config["training"].get("patience", 50)
    no_improve = 0
    total_nan = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        total_nan += nan_count

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                f"Best: {best_val_acc:.3f} (ep {best_epoch+1})"
                + (f" | NaN: {nan_count}" if nan_count > 0 else "")
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time

    # Compute total training steps
    steps_per_epoch = len(train_loader)
    total_steps = (epoch + 1) * steps_per_epoch

    # Restore best model
    model.load_state_dict(best_state)
    model = model.to(device)

    print(f"  Training time: {elapsed:.1f}s | Best val acc: {best_val_acc:.3f} at epoch {best_epoch+1}")
    print(f"  Total steps: {total_steps} | Total NaN batches: {total_nan}")

    return {
        "model_name": model_name,
        "n_params": count_parameters(model),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
        "total_steps": total_steps,
        "total_nan": total_nan,
        "elapsed_sec": elapsed,
        "history": history,
        "model": model,
    }


def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating selective copying data...")
    train_loader, val_loader, test_loader, dataset_info = generate_selective_copy_data(
        num_train=config["data"]["num_train"],
        num_val=config["data"]["num_val"],
        num_test=config["data"]["num_test"],
        seq_len=config["data"]["seq_len"],
        n_memory_tokens=config["data"]["n_memory_tokens"],
        n_content_vocab=config["data"]["n_content_vocab"],
        batch_size=config["training"]["batch_size"],
    )
    print(f"Dataset info: {dataset_info}")

    vocab_size = dataset_info["vocab_size"]
    num_classes = dataset_info["num_classes"]
    seq_len = dataset_info["seq_len"]

    # =========================================================================
    # Model 1: LogSSM (proposed)
    # =========================================================================
    set_seed(config.get("seed", 42))
    log_ssm = LogSSMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=config["model"]["d_model"],
        d_head=config["model"]["d_head"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["training"]["dropout"],
        max_seq_len=seq_len,
    )

    log_ssm_results = train_model("LogSSM", log_ssm, train_loader, val_loader, config, device)

    # =========================================================================
    # Model 2: Linear Attention (baseline)
    # =========================================================================
    set_seed(config.get("seed", 42))
    lin_attn = LinearAttentionClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["training"]["dropout"],
        max_seq_len=seq_len,
    )

    lin_attn_results = train_model("LinearAttention", lin_attn, train_loader, val_loader, config, device)

    # =========================================================================
    # Model 3: Diagonal SSM (baseline)
    # =========================================================================
    set_seed(config.get("seed", 42))
    diag_ssm = DiagonalSSMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=config["model"]["d_model"],
        state_dim=config["model"]["d_model"],  # state_dim = d_model
        n_layers=config["model"]["n_layers"],
        dropout=config["training"]["dropout"],
        max_seq_len=seq_len,
    )

    diag_ssm_results = train_model("DiagonalSSM", diag_ssm, train_loader, val_loader, config, device)

    # =========================================================================
    # Test evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    all_results = {}
    for name, res in [
        ("LogSSM", log_ssm_results),
        ("LinearAttention", lin_attn_results),
        ("DiagonalSSM", diag_ssm_results),
    ]:
        model = res["model"]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        res["test_acc"] = test_acc
        res["test_loss"] = test_loss
        all_results[name] = res
        print(f"  {name:20s} | Test Acc: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    # =========================================================================
    # Success criteria evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    criteria = {}

    # Criterion 1: LogSSM > 90%
    c1 = all_results["LogSSM"]["test_acc"] > 0.90
    criteria["logssm_gt_90"] = {
        "target": "> 90%",
        "achieved": f"{all_results['LogSSM']['test_acc']:.1%}",
        "pass": c1,
    }
    print(f"  LogSSM > 90%: {'✅ PASS' if c1 else '❌ FAIL'} ({all_results['LogSSM']['test_acc']:.1%})")

    # Criterion 2: Linear Attention < 60%
    c2 = all_results["LinearAttention"]["test_acc"] < 0.60
    criteria["linattn_lt_60"] = {
        "target": "< 60%",
        "achieved": f"{all_results['LinearAttention']['test_acc']:.1%}",
        "pass": c2,
    }
    print(f"  LinearAttention < 60%: {'✅ PASS' if c2 else '❌ FAIL'} ({all_results['LinearAttention']['test_acc']:.1%})")

    # Criterion 3: Diagonal SSM < 70%
    c3 = all_results["DiagonalSSM"]["test_acc"] < 0.70
    criteria["diagssm_lt_70"] = {
        "target": "< 70%",
        "achieved": f"{all_results['DiagonalSSM']['test_acc']:.1%}",
        "pass": c3,
    }
    print(f"  DiagonalSSM < 70%: {'✅ PASS' if c3 else '❌ FAIL'} ({all_results['DiagonalSSM']['test_acc']:.1%})")

    # Criterion 4: Convergence within 500 steps
    c4 = all_results["LogSSM"]["total_nan"] == 0 and all_results["LogSSM"]["total_steps"] <= 500
    criteria["convergence_500_steps"] = {
        "target": "converge within 500 steps, no NaN",
        "achieved": f"{all_results['LogSSM']['total_steps']} steps, {all_results['LogSSM']['total_nan']} NaN",
        "pass": c4,
    }
    print(f"  Convergence within 500 steps: {'✅ PASS' if c4 else '❌ FAIL'} ({all_results['LogSSM']['total_steps']} steps)")

    # Overall verdict
    all_pass = all(c["pass"] for c in criteria.values())
    # Also check partial pass: LogSSM works even if baselines are higher than expected
    logssm_works = criteria["logssm_gt_90"]["pass"]

    if all_pass:
        verdict = "PROCEED"
    elif logssm_works:
        verdict = "PROCEED (LogSSM works; baseline criteria may need adjustment)"
    elif all_results["LogSSM"]["total_nan"] > 0:
        verdict = "DEBUG (numerical instability)"
    else:
        verdict = "DEBUG (LogSSM accuracy too low)"

    print(f"\n  VERDICT: {verdict}")

    # =========================================================================
    # Save results
    # =========================================================================
    results_path = Path(__file__).parent / "results.yaml"
    results_dict = {
        "models": {},
        "criteria": criteria,
        "verdict": verdict,
    }
    for name, res in all_results.items():
        results_dict["models"][name] = {
            "n_params": res["n_params"],
            "best_val_acc": float(res["best_val_acc"]),
            "test_acc": float(res["test_acc"]),
            "test_loss": float(res["test_loss"]),
            "best_epoch": res["best_epoch"],
            "total_epochs": res["total_epochs"],
            "total_steps": res["total_steps"],
            "total_nan": res["total_nan"],
            "elapsed_sec": round(res["elapsed_sec"], 1),
        }

    with open(results_path, "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False, sort_keys=False)

    print(f"\nResults saved to {results_path}")

    return all_results, criteria, verdict


if __name__ == "__main__":
    main()

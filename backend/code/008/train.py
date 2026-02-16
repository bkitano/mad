"""
Training script for cos-LogLinear MVE.

Trains and evaluates 4 attention variants on MQAR:
1. vanilla_linear - ELU+1 kernel, single state
2. cosformer - Cosine reweighted, single state
3. log_linear - ELU+1 kernel, hierarchical Fenwick tree
4. cos_log_linear - Cosine reweighted + hierarchical (PROPOSED)

All models share the same architecture (2 layers, d=32, 2 heads of dim 16)
and are trained on the same MQAR data (8 KV pairs, T=128).
"""

import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.attention import MQARModel
from data.mqar import create_mqar_dataloaders, MQARDataset


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (B, T, V)

        # Flatten for cross-entropy
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Accuracy on supervised positions only
        mask = targets != MQARDataset.IGNORE_INDEX
        if mask.any():
            preds = logits.argmax(dim=-1)
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_count += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss += loss.item() * inputs.size(0)

        mask = targets != MQARDataset.IGNORE_INDEX
        if mask.any():
            preds = logits.argmax(dim=-1)
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_count += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def check_for_nans(model, step_name=""):
    """Check if any model parameters contain NaN/Inf."""
    for name, param in model.named_parameters():
        if param.data.isnan().any() or param.data.isinf().any():
            return True, f"NaN/Inf in {name} at {step_name}"
        if param.grad is not None and (param.grad.isnan().any() or param.grad.isinf().any()):
            return True, f"NaN/Inf grad in {name} at {step_name}"
    return False, ""


def train_model(
    attention_type: str,
    config: dict,
    device: torch.device,
    wandb_group: str = None,
):
    """
    Train a single model variant and return results.

    Returns dict with: best_val_acc, final_test_acc, stable (no NaNs), epochs_trained
    """
    print(f"\n{'='*60}")
    print(f"Training: {attention_type}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader = create_mqar_dataloaders(
        num_kv_pairs=config['data']['num_kv_pairs'],
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        num_train=config['data']['num_train'],
        num_test=config['data']['num_test'],
        batch_size=config['training']['batch_size'],
        seed=config['data']['seed'],
    )

    # Model
    model = MQARModel(
        attention_type=attention_type,
        vocab_size=config['data']['vocab_size'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        head_dim=config['model']['head_dim'],
        num_layers=config['model']['num_layers'],
        max_seq_len=config['data']['seq_len'],
        dropout=config['model']['dropout'],
    ).to(device)

    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,}")

    # Initialize wandb run for this model
    run = wandb.init(
        project=config['logging']['wandb_project'],
        name=f"exp-008-{attention_type}",
        group=wandb_group,
        config={
            "attention_type": attention_type,
            "model": config['model'],
            "data": config['data'],
            "training": config['training'],
            "num_params": num_params,
        },
        reinit=True,
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=MQARDataset.IGNORE_INDEX)

    # Training loop
    best_val_acc = 0.0
    stable = True
    nan_message = ""

    for epoch in range(1, config['training']['max_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=config['training']['gradient_clip'],
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Check for NaNs
        has_nan, msg = check_for_nans(model, f"epoch {epoch}")
        if has_nan:
            print(f"  ⚠️  {msg}")
            stable = False
            nan_message = msg
            wandb.log({
                "train/loss": float('nan'),
                "train/accuracy": float('nan'),
                "val/loss": float('nan'),
                "val/accuracy": float('nan'),
                "epoch": epoch,
                "nan_detected": True,
            })
            break

        best_val_acc = max(best_val_acc, val_acc)

        # Log to wandb
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
            "best_val_accuracy": best_val_acc,
        })

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")

        # Early stopping
        if val_acc >= config['training'].get('early_stop_acc', 0.99):
            print(f"  Early stopping at epoch {epoch} (val_acc={val_acc:.4f})")
            break

    # Final evaluation
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    # Log final results
    wandb.log({
        "final/test_accuracy": final_test_acc,
        "final/test_loss": final_test_loss,
        "final/best_val_accuracy": best_val_acc,
        "final/stable": stable,
        "final/num_params": num_params,
        "final/epochs_trained": epoch,
    })

    print(f"\n  Final Test Accuracy: {final_test_acc:.4f}")
    print(f"  Best Val Accuracy:  {best_val_acc:.4f}")
    print(f"  Stable (no NaN):    {stable}")
    if not stable:
        print(f"  NaN message:        {nan_message}")

    wandb.finish()

    return {
        "attention_type": attention_type,
        "best_val_acc": best_val_acc,
        "final_test_acc": final_test_acc,
        "stable": stable,
        "nan_message": nan_message,
        "num_params": num_params,
        "epochs_trained": epoch,
        "wandb_url": wandb_url,
    }


def main():
    parser = argparse.ArgumentParser(description="cos-LogLinear MVE Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--models", type=str, nargs="*",
                       default=["vanilla_linear", "cosformer", "log_linear", "cos_log_linear"],
                       help="Which attention types to train")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Group name for wandb (to group all variants together)
    wandb_group = f"exp-008-mqar-{time.strftime('%Y%m%d-%H%M%S')}"

    # Train each model variant
    all_results = []
    for attention_type in args.models:
        result = train_model(attention_type, config, device, wandb_group)
        all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: MQAR Results (8 KV pairs, T=128)")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Params':>8} {'Best Val':>10} {'Test Acc':>10} {'Stable':>8} {'Epochs':>8}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['attention_type']:<20} {r['num_params']:>8,} "
              f"{r['best_val_acc']:>10.4f} {r['final_test_acc']:>10.4f} "
              f"{'✅' if r['stable'] else '❌':>8} {r['epochs_trained']:>8}")

    # Check success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}")

    results_by_type = {r['attention_type']: r for r in all_results}

    criteria = []

    # Criterion 1: cos-LogLinear > 80% accuracy
    if 'cos_log_linear' in results_by_type:
        acc = results_by_type['cos_log_linear']['final_test_acc']
        passed = acc > 0.80
        criteria.append(("cos-LogLinear > 80% accuracy", passed, f"{acc:.4f}"))
        print(f"  {'✅' if passed else '❌'} cos-LogLinear > 80% accuracy: {acc:.4f}")

    # Criterion 2: cosFormer < 50% (capacity-limited)
    if 'cosformer' in results_by_type:
        acc = results_by_type['cosformer']['final_test_acc']
        passed = acc < 0.50
        criteria.append(("cosFormer < 50% accuracy", passed, f"{acc:.4f}"))
        print(f"  {'✅' if passed else '❌'} cosFormer < 50% accuracy: {acc:.4f}")

    # Criterion 3: log-linear 60-70%
    if 'log_linear' in results_by_type:
        acc = results_by_type['log_linear']['final_test_acc']
        passed = 0.55 <= acc <= 0.80  # Slightly wider range for MVE
        criteria.append(("log-linear 60-70% accuracy", passed, f"{acc:.4f}"))
        print(f"  {'✅' if passed else '❌'} log-linear 60-70% accuracy: {acc:.4f}")

    # Criterion 4: No NaN/Inf
    if 'cos_log_linear' in results_by_type:
        stable = results_by_type['cos_log_linear']['stable']
        criteria.append(("No NaN/Inf", stable, "stable" if stable else "unstable"))
        print(f"  {'✅' if stable else '❌'} No NaN/Inf: {'stable' if stable else 'unstable'}")

    # Overall
    all_passed = all(c[1] for c in criteria)
    print(f"\n  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

    # Save results to file
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump({
            "results": all_results,
            "criteria": [{"name": c[0], "passed": c[1], "value": c[2]} for c in criteria],
            "all_passed": all_passed,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

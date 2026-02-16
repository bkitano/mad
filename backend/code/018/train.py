"""
Training script for Experiment 018: Hutchinson Adaptive Rank DPLR SSM.

Implements the full MVE procedure:
1. Phase 1 (Warmup): Train all layers at r_max for warmup_steps
2. Phase 2 (Measurement): Compute importance scores via power-series log-det
3. Phase 3 (Pruning): Allocate ranks proportional to importance, SVD truncate
4. Phase 4 (Fine-tuning): Train with adapted ranks for finetune_steps

Also trains baselines:
- Baseline A: Fixed r=4, trained for total steps
- Baseline B: Fixed r=8, trained for total steps

Usage:
    python train.py --config config.yaml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
import wandb

# Add parent to path for model imports
sys.path.insert(0, str(Path(__file__).parent))

from models.dplr_ssm import DPLRSSMModel


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Data: Sequential CIFAR-10 ──────────────────────────────────────────────

def get_scifar_data(data_dir: str = "/tmp/cifar10", num_train: int = 40000, num_val: int = 5000, num_test: int = 5000):
    """
    Load CIFAR-10 and flatten to sequential format (length 1024, dim 3).

    Each image (32x32x3) becomes a sequence of 1024 RGB pixels.
    Normalize pixel values to [0, 1].
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()

        trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

        # Flatten: (3, 32, 32) -> (1024, 3) by treating each pixel as a 3-dim feature
        def flatten_dataset(dataset, n_samples):
            images = []
            labels = []
            for i in range(min(n_samples, len(dataset))):
                img, label = dataset[i]
                # img shape: (3, 32, 32)
                # Reshape to (1024, 3): flatten spatial dims, keep channels as features
                img_flat = img.permute(1, 2, 0).reshape(1024, 3)  # (1024, 3)
                images.append(img_flat)
                labels.append(label)
            return torch.stack(images), torch.tensor(labels)

        train_x, train_y = flatten_dataset(trainset, num_train)
        # Split trainset into train and val
        val_x, val_y = train_x[num_train-num_val:num_train], train_y[num_train-num_val:num_train]
        train_x, train_y = train_x[:num_train-num_val], train_y[:num_train-num_val]
        test_x, test_y = flatten_dataset(testset, num_test)

        print(f"sCIFAR-10 data loaded: train={len(train_x)}, val={len(val_x)}, test={len(test_x)}")
        print(f"Sequence shape: {train_x.shape[1:]}")  # (1024, 3)

        return train_x, train_y, val_x, val_y, test_x, test_y

    except ImportError:
        print("torchvision not available, generating synthetic sCIFAR-like data")
        return generate_synthetic_scifar(num_train, num_val, num_test)


def generate_synthetic_scifar(num_train: int = 40000, num_val: int = 5000, num_test: int = 5000):
    """
    Generate synthetic sequential classification data as fallback.
    Creates sequences where the class depends on long-range patterns.
    """
    seq_len = 1024
    input_dim = 3
    num_classes = 10

    def make_data(n):
        # Create sequences with class-dependent frequency patterns
        x = torch.randn(n, seq_len, input_dim) * 0.1
        labels = torch.randint(0, num_classes, (n,))

        for i in range(n):
            c = labels[i].item()
            # Add class-specific frequency component
            freq = (c + 1) * 2 * math.pi / seq_len
            t = torch.arange(seq_len, dtype=torch.float32)
            for d in range(input_dim):
                phase = c * 0.5 + d * 0.3
                x[i, :, d] += 0.5 * torch.sin(freq * t + phase)

        return x, labels

    train_x, train_y = make_data(num_train)
    val_x, val_y = make_data(num_val)
    test_x, test_y = make_data(num_test)

    print(f"Synthetic sCIFAR data: train={num_train}, val={num_val}, test={num_test}")
    return train_x, train_y, val_x, val_y, test_x, test_y


# ─── Training Loop ──────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    return total_loss / total, correct / total


def train_steps(model, train_loader, optimizer, criterion, device, num_steps,
                val_loader=None, max_grad_norm=1.0, log_every=100,
                wandb_prefix="", scheduler=None):
    """
    Train for a fixed number of gradient steps (not epochs).
    Returns final (train_loss, train_acc, val_loss, val_acc).
    """
    model.train()
    step = 0
    total_loss = 0.0
    correct = 0
    total = 0

    train_iter = iter(train_loader)
    best_val_acc = 0.0

    while step < num_steps:
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

        step += 1

        if step % log_every == 0:
            avg_loss = total_loss / total
            avg_acc = correct / total

            log_dict = {
                f"{wandb_prefix}train/loss": avg_loss,
                f"{wandb_prefix}train/accuracy": avg_acc,
                f"{wandb_prefix}step": step,
            }

            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, criterion, batch_x.device)
                log_dict[f"{wandb_prefix}val/loss"] = val_loss
                log_dict[f"{wandb_prefix}val/accuracy"] = val_acc
                best_val_acc = max(best_val_acc, val_acc)
                log_dict[f"{wandb_prefix}val/best_accuracy"] = best_val_acc
                print(f"  Step {step}/{num_steps}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                model.train()  # back to train mode
            else:
                print(f"  Step {step}/{num_steps}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}")

            wandb.log(log_dict)

            # Reset running averages
            total_loss = 0.0
            correct = 0
            total = 0

    # Final eval
    final_val_loss, final_val_acc = 0.0, 0.0
    if val_loader is not None:
        final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    return total_loss / max(total, 1), correct / max(total, 1), final_val_loss, final_val_acc


# ─── Importance Scoring & Rank Allocation ────────────────────────────────────

def compute_all_importance_scores(model, method="logdet", num_freqs=16, k_max=4):
    """
    Compute importance scores for all SSM layers.

    Args:
        model: DPLRSSMModel
        method: "logdet" (power-series) or "hutchinson"
        num_freqs: Number of frequency samples
        k_max: Power series truncation order

    Returns:
        List of importance scores, one per layer
    """
    ssm_layers = model.get_ssm_layers()
    scores = []
    for i, layer in enumerate(ssm_layers):
        if method == "logdet":
            score = layer.compute_importance_logdet(num_freqs=num_freqs, k_max=k_max)
        else:
            score = layer.compute_importance_hutchinson(num_freqs=num_freqs)
        scores.append(score)
        print(f"  Layer {i}: importance = {score:.6f} (rank={layer.r})")
    return scores


def allocate_ranks(scores, total_budget, r_min=1, r_max=8):
    """
    Allocate ranks proportional to importance scores.

    (Proposal eq): r_l = round(I_l / sum(I_j) * R_total)
    with r_l >= r_min.

    Args:
        scores: List of importance scores per layer
        total_budget: Total rank budget R_total
        r_min: Minimum rank per layer
        r_max: Maximum rank per layer

    Returns:
        List of allocated ranks
    """
    num_layers = len(scores)
    scores = np.array(scores)

    # Normalize
    total = scores.sum()
    if total < 1e-12:
        # All scores near zero — uniform allocation
        return [max(r_min, total_budget // num_layers)] * num_layers

    fractions = scores / total

    # Allocate proportionally, ensuring minimum
    ranks = np.maximum(np.round(fractions * total_budget).astype(int), r_min)
    ranks = np.minimum(ranks, r_max)

    # Adjust to meet budget exactly
    while ranks.sum() > total_budget:
        # Reduce the layer with the lowest importance that's above r_min
        eligible = np.where(ranks > r_min)[0]
        if len(eligible) == 0:
            break
        # Pick the one with lowest importance among eligible
        idx = eligible[np.argmin(scores[eligible])]
        ranks[idx] -= 1

    while ranks.sum() < total_budget:
        # Increase the layer with the highest importance that's below r_max
        eligible = np.where(ranks < r_max)[0]
        if len(eligible) == 0:
            break
        idx = eligible[np.argmax(scores[eligible])]
        ranks[idx] += 1

    return ranks.tolist()


def truncate_model_ranks(model, new_ranks, device):
    """
    Truncate each SSM layer to its assigned rank via SVD.

    Args:
        model: DPLRSSMModel
        new_ranks: List of new ranks per layer
        device: torch device
    """
    ssm_layers = model.get_ssm_layers()
    for i, (layer, new_r) in enumerate(zip(ssm_layers, new_ranks)):
        old_r = layer.r
        if new_r < old_r:
            print(f"  Layer {i}: truncating rank {old_r} -> {new_r}")
            layer.truncate_rank(new_r)
        else:
            print(f"  Layer {i}: keeping rank {old_r} (target={new_r})")

    # Move model to device after truncation (parameters may have been recreated)
    model.to(device)


# ─── Main Experiment ─────────────────────────────────────────────────────────

def run_baseline(name, model_config, train_loader, val_loader, test_loader,
                 training_config, device, criterion):
    """Run a baseline with fixed rank for the full training duration."""
    print(f"\n{'='*60}")
    print(f"Running Baseline: {name}")
    print(f"{'='*60}")

    model = DPLRSSMModel(**model_config).to(device)
    total_params = model.count_params()
    lr_params = model.count_lr_params()
    print(f"  Total params: {total_params}, LR params: {lr_params}")
    print(f"  Ranks: {model.get_ranks()}")

    wandb_prefix = f"baseline_{name}/"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    total_steps = training_config["warmup_steps"] + training_config["finetune_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"  Training for {total_steps} steps...")
    _, _, val_loss, val_acc = train_steps(
        model, train_loader, optimizer, criterion, device,
        num_steps=total_steps, val_loader=val_loader,
        max_grad_norm=training_config.get("gradient_clip", 1.0),
        log_every=training_config.get("log_every", 100),
        wandb_prefix=wandb_prefix,
        scheduler=scheduler,
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Final: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    wandb.log({
        f"final/{name}_val_accuracy": val_acc,
        f"final/{name}_test_accuracy": test_acc,
        f"final/{name}_total_params": total_params,
        f"final/{name}_lr_params": lr_params,
    })

    return test_acc, val_acc, total_params, lr_params


def run_adaptive(model_config, train_loader, val_loader, test_loader,
                 training_config, device, criterion):
    """
    Run the adaptive rank experiment (full 4-phase procedure).

    Phase 1: Train at r_max for warmup_steps
    Phase 2: Compute importance scores
    Phase 3: Allocate ranks and SVD-truncate
    Phase 4: Fine-tune with adapted ranks
    """
    print(f"\n{'='*60}")
    print(f"Running Adaptive Rank Experiment")
    print(f"{'='*60}")

    # Start at r_max
    model = DPLRSSMModel(**model_config).to(device)
    total_params_initial = model.count_params()
    lr_params_initial = model.count_lr_params()
    print(f"  Initial params: {total_params_initial}, LR params: {lr_params_initial}")
    print(f"  Initial ranks: {model.get_ranks()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    total_steps = training_config["warmup_steps"] + training_config["finetune_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Phase 1: Warmup ──
    print(f"\n--- Phase 1: Warmup ({training_config['warmup_steps']} steps at r_max) ---")
    _, _, val_loss, val_acc = train_steps(
        model, train_loader, optimizer, criterion, device,
        num_steps=training_config["warmup_steps"],
        val_loader=val_loader,
        max_grad_norm=training_config.get("gradient_clip", 1.0),
        log_every=training_config.get("log_every", 100),
        wandb_prefix="adaptive/phase1_",
        scheduler=scheduler,
    )
    print(f"  After warmup: val_acc={val_acc:.4f}")
    wandb.log({"adaptive/post_warmup_val_accuracy": val_acc})

    # ── Phase 2: Measure importance ──
    print(f"\n--- Phase 2: Compute Importance Scores ---")
    method = training_config.get("importance_method", "logdet")
    num_freqs = training_config.get("num_freqs", 16)
    k_max = training_config.get("k_max", 4)

    scores = compute_all_importance_scores(model, method=method, num_freqs=num_freqs, k_max=k_max)

    # Log importance scores
    for i, s in enumerate(scores):
        wandb.log({f"importance/layer_{i}": s})

    # Compute importance ratio (success criterion)
    max_score = max(scores)
    min_score = min(scores)
    importance_ratio = max_score / (min_score + 1e-12)
    coeff_var = np.std(scores) / (np.mean(scores) + 1e-12)
    print(f"\n  Importance ratio (max/min): {importance_ratio:.4f}")
    print(f"  Coefficient of variation: {coeff_var:.4f}")
    wandb.log({
        "importance/max_min_ratio": importance_ratio,
        "importance/coefficient_of_variation": coeff_var,
    })

    # ── Phase 3: Rank allocation & truncation ──
    print(f"\n--- Phase 3: Rank Allocation (budget={training_config['rank_budget']}) ---")
    new_ranks = allocate_ranks(
        scores,
        total_budget=training_config["rank_budget"],
        r_min=training_config.get("r_min", 1),
        r_max=model_config["r"],
    )
    print(f"  Allocated ranks: {new_ranks} (sum={sum(new_ranks)})")
    wandb.log({"adaptive/allocated_ranks": new_ranks})

    # Perform SVD truncation
    truncate_model_ranks(model, new_ranks, device)
    print(f"  Post-truncation ranks: {model.get_ranks()}")

    total_params_final = model.count_params()
    lr_params_final = model.count_lr_params()
    print(f"  Post-truncation params: {total_params_final}, LR params: {lr_params_final}")

    # Check accuracy after truncation (before fine-tuning)
    post_trunc_loss, post_trunc_acc = evaluate(model, val_loader, criterion, device)
    print(f"  Post-truncation val_acc: {post_trunc_acc:.4f}")
    wandb.log({
        "adaptive/post_truncation_val_accuracy": post_trunc_acc,
        "adaptive/post_truncation_params": total_params_final,
        "adaptive/post_truncation_lr_params": lr_params_final,
    })

    # ── Phase 4: Fine-tuning ──
    print(f"\n--- Phase 4: Fine-tuning ({training_config['finetune_steps']} steps) ---")

    # Re-create optimizer for truncated model (parameters changed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"] * training_config.get("finetune_lr_ratio", 0.5),
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config["finetune_steps"]
    )

    _, _, val_loss, val_acc = train_steps(
        model, train_loader, optimizer, criterion, device,
        num_steps=training_config["finetune_steps"],
        val_loader=val_loader,
        max_grad_norm=training_config.get("gradient_clip", 1.0),
        log_every=training_config.get("log_every", 100),
        wandb_prefix="adaptive/phase4_",
        scheduler=finetune_scheduler,
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Final: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    wandb.log({
        "final/adaptive_val_accuracy": val_acc,
        "final/adaptive_test_accuracy": test_acc,
        "final/adaptive_total_params": total_params_final,
        "final/adaptive_lr_params": lr_params_final,
        "final/adaptive_ranks": model.get_ranks(),
    })

    return test_acc, val_acc, total_params_final, lr_params_final, scores, new_ranks, importance_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # Extract configs
    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config.get("data", {})
    logging_cfg = config.get("logging", {})

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Initialize W&B ──
    wandb.init(
        project=logging_cfg.get("wandb_project", "mad-architecture-search"),
        name=f"exp-018-adaptive-rank-dplr",
        config=config,
    )
    print(f"Wandb URL: {wandb.run.get_url()}")

    # ── Load Data ──
    print("\n--- Loading Data ---")
    train_x, train_y, val_x, val_y, test_x, test_y = get_scifar_data(
        num_train=data_cfg.get("num_train", 40000),
        num_val=data_cfg.get("num_val", 5000),
        num_test=data_cfg.get("num_test", 5000),
    )

    batch_size = training_cfg.get("batch_size", 64)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    input_dim = train_x.shape[-1]  # 3 for CIFAR (RGB), 1 for grayscale
    criterion = nn.CrossEntropyLoss()

    # ── Run Baselines ──

    # Baseline A: Fixed r=4 (same budget as adaptive)
    baseline_a_cfg = {
        "input_dim": input_dim,
        "d": model_cfg["d"],
        "n": model_cfg["n"],
        "r": training_cfg["rank_budget"] // model_cfg["num_layers"],  # 16/4 = 4
        "num_layers": model_cfg["num_layers"],
        "num_classes": data_cfg.get("num_classes", 10),
        "dropout": model_cfg.get("dropout", 0.1),
    }
    r4_test_acc, r4_val_acc, r4_params, r4_lr_params = run_baseline(
        "fixed_r4", baseline_a_cfg, train_loader, val_loader, test_loader,
        training_cfg, device, criterion
    )

    # Baseline B: Fixed r=8 (max rank, upper bound)
    baseline_b_cfg = {
        "input_dim": input_dim,
        "d": model_cfg["d"],
        "n": model_cfg["n"],
        "r": model_cfg["r"],  # r_max = 8
        "num_layers": model_cfg["num_layers"],
        "num_classes": data_cfg.get("num_classes", 10),
        "dropout": model_cfg.get("dropout", 0.1),
    }
    r8_test_acc, r8_val_acc, r8_params, r8_lr_params = run_baseline(
        "fixed_r8", baseline_b_cfg, train_loader, val_loader, test_loader,
        training_cfg, device, criterion
    )

    # ── Run Adaptive ──
    adaptive_cfg = {
        "input_dim": input_dim,
        "d": model_cfg["d"],
        "n": model_cfg["n"],
        "r": model_cfg["r"],  # Start at r_max
        "num_layers": model_cfg["num_layers"],
        "num_classes": data_cfg.get("num_classes", 10),
        "dropout": model_cfg.get("dropout", 0.1),
    }
    (adaptive_test_acc, adaptive_val_acc, adaptive_params, adaptive_lr_params,
     importance_scores, allocated_ranks, importance_ratio) = run_adaptive(
        adaptive_cfg, train_loader, val_loader, test_loader,
        training_cfg, device, criterion
    )

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 018 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline A (fixed r=4, budget=16):")
    print(f"  Test accuracy: {r4_test_acc:.4f}")
    print(f"  Total params: {r4_params}, LR params: {r4_lr_params}")
    print(f"\nBaseline B (fixed r=8, budget=32):")
    print(f"  Test accuracy: {r8_test_acc:.4f}")
    print(f"  Total params: {r8_params}, LR params: {r8_lr_params}")
    print(f"\nAdaptive (budget=16):")
    print(f"  Test accuracy: {adaptive_test_acc:.4f}")
    print(f"  Total params: {adaptive_params}, LR params: {adaptive_lr_params}")
    print(f"  Allocated ranks: {allocated_ranks}")
    print(f"  Importance scores: {[f'{s:.6f}' for s in importance_scores]}")
    print(f"  Importance ratio (max/min): {importance_ratio:.4f}")

    # ── Check Success Criteria ──
    print(f"\n{'='*60}")
    print(f"SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")

    # Criterion 1: Importance scores are non-uniform (max/min > 2)
    crit1 = importance_ratio > 2.0
    print(f"\n1. Importance ratio > 2.0: {'✅ PASS' if crit1 else '❌ FAIL'}")
    print(f"   Ratio: {importance_ratio:.4f} (need > 2.0)")

    # Criterion 2: Adaptive >= 95% of fixed r=8 accuracy
    ratio_to_r8 = adaptive_test_acc / (r8_test_acc + 1e-12)
    crit2 = ratio_to_r8 >= 0.95
    print(f"\n2. Adaptive >= 95% of r=8 accuracy: {'✅ PASS' if crit2 else '❌ FAIL'}")
    print(f"   Adaptive: {adaptive_test_acc:.4f}, r=8: {r8_test_acc:.4f}, ratio: {ratio_to_r8:.4f}")

    # Criterion 3: Adaptive outperforms fixed r=4 by > 1%
    improvement = adaptive_test_acc - r4_test_acc
    crit3 = improvement > 0.01
    print(f"\n3. Adaptive > r=4 by > 1%: {'✅ PASS' if crit3 else '❌ FAIL'}")
    print(f"   Adaptive: {adaptive_test_acc:.4f}, r=4: {r4_test_acc:.4f}, diff: {improvement:.4f}")

    # Failure criteria
    fail1 = importance_ratio < 1.3
    fail2 = (adaptive_test_acc - r4_test_acc) < -0.05  # >5% drop after truncation
    print(f"\nFailure check - Uniform importance (ratio < 1.3): {'⚠️ YES' if fail1 else '✅ NO'}")
    print(f"Failure check - Significant accuracy drop (>5%): {'⚠️ YES' if fail2 else '✅ NO'}")

    # Log final summary
    wandb.log({
        "success_criteria/importance_ratio_gt_2": crit1,
        "success_criteria/adaptive_ge_95pct_r8": crit2,
        "success_criteria/adaptive_gt_r4_by_1pct": crit3,
        "success_criteria/importance_ratio": importance_ratio,
        "success_criteria/ratio_to_r8": ratio_to_r8,
        "success_criteria/improvement_over_r4": improvement,
        "failure_criteria/uniform_importance": fail1,
        "failure_criteria/accuracy_degradation": fail2,
    })

    wandb.finish()

    # Return results dict for programmatic use
    return {
        "r4_test_acc": r4_test_acc,
        "r8_test_acc": r8_test_acc,
        "adaptive_test_acc": adaptive_test_acc,
        "importance_ratio": importance_ratio,
        "allocated_ranks": allocated_ranks,
        "importance_scores": importance_scores,
        "crit1_pass": crit1,
        "crit2_pass": crit2,
        "crit3_pass": crit3,
    }


if __name__ == "__main__":
    main()

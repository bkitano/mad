#!/usr/bin/env python3
"""
Training script for MVE 017: Hyperoctahedral Signed-Permutation SSM.

Trains 3 models on B3 composition task:
1. HyperSSM (signed permutation): Expected >90% accuracy
2. DiagonalSSM (baseline): Expected <60% accuracy
3. PermOnlySSM (ablation): Expected <75% accuracy

The gap between HyperSSM and PermOnlySSM demonstrates the value of the Z_2^n
sign component in B_n = Z_2^n ⋊ S_n.

Usage:
    python train.py [--config config.yaml]
"""

import argparse
import time
import yaml
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.hyper_ssm import HyperSSM
from models.diagonal_ssm import DiagonalSSM
from models.perm_only_ssm import PermOnlySSM
from tasks.b3_group import B3Group, B3CompositionDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_model(model_type: str, config: dict, device: torch.device) -> nn.Module:
    """Factory for creating models."""
    common = {
        'vocab_size': B3CompositionDataset.NUM_TOKENS,
        'num_classes': B3CompositionDataset.NUM_CLASSES,
        'd_model': config['model']['d_model'],
        'state_dim': config['model']['state_dim'],
        'num_heads': config['model']['num_heads'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'max_seq_len': config['dataset']['max_seq_len'],
    }

    if model_type == 'HyperSSM':
        model = HyperSSM(
            **common,
            sinkhorn_iters=config['model']['sinkhorn_iters'],
            tau=config['model']['tau'],
            use_hard_perm=config['model']['use_hard_perm'],
        )
    elif model_type == 'DiagonalSSM':
        model = DiagonalSSM(**common)
    elif model_type == 'PermOnlySSM':
        model = PermOnlySSM(
            **common,
            sinkhorn_iters=config['model']['sinkhorn_iters'],
            tau=config['model']['tau'],
            use_hard_perm=config['model']['use_hard_perm'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    nan_count = 0

    for inputs, targets, ks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, seq_len, num_classes)

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Accuracy on non-ignored positions
        mask = targets_flat != -100
        if mask.any():
            preds = logits_flat[mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[mask]).sum().item()
            total_count += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_count if total_count > 0 else 0
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model, return (loss, accuracy, per_k_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    nan_count = 0

    # Per-k tracking
    k_correct = {}
    k_count = {}

    for inputs, targets, ks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)

        mask = targets_flat != -100
        if mask.any():
            preds = logits_flat[mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[mask]).sum().item()
            total_count += mask.sum().item()

        # Per-k accuracy
        for i, k in enumerate(ks):
            k_val = k.item()
            sample_targets = targets[i]
            sample_logits = logits[i]
            sample_mask = sample_targets != -100
            if sample_mask.any():
                sample_preds = sample_logits[sample_mask].argmax(dim=-1)
                correct = (sample_preds == sample_targets[sample_mask]).sum().item()
                count = sample_mask.sum().item()
                k_correct[k_val] = k_correct.get(k_val, 0) + correct
                k_count[k_val] = k_count.get(k_val, 0) + count

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_count if total_count > 0 else 0

    per_k_acc = {}
    for k in sorted(k_correct.keys()):
        per_k_acc[k] = k_correct[k] / k_count[k] if k_count[k] > 0 else 0

    return avg_loss, accuracy, per_k_acc, nan_count


def train_model(model_type: str, config: dict, train_loader, test_loader, device):
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_type}")
    print(f"{'='*60}")

    model = create_model(model_type, config, device)
    num_params = model.count_params()
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        eps=config['training']['op_eps'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
        eta_min=config['training']['lr'] * 0.01,
    )

    best_val_acc = 0.0
    best_epoch = 0
    total_nan = 0
    train_start = time.time()

    for epoch in range(config['training']['max_epochs']):
        # Train
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config['training']['gradient_clip'],
        )
        total_nan += nan_count

        # Evaluate
        val_loss, val_acc, per_k_acc, val_nan = evaluate(model, test_loader, device)
        total_nan += val_nan

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f} (ep {best_epoch+1})")

        # Early stopping if we hit target
        if val_acc >= config['training']['target_acc']:
            print(f"  Target accuracy {config['training']['target_acc']} reached at epoch {epoch+1}!")
            break

        # Patience-based early stopping
        if epoch - best_epoch >= config['training']['patience']:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {config['training']['patience']} epochs)")
            break

    train_time = time.time() - train_start

    # Final evaluation
    final_loss, final_acc, final_k_acc, _ = evaluate(model, test_loader, device)

    # Speed benchmark
    speed_ms = benchmark_speed(model, device, config)

    results = {
        'model_type': model_type,
        'params': num_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch + 1,
        'final_test_acc': final_acc,
        'final_test_loss': final_loss,
        'per_k_acc': final_k_acc,
        'total_nan': total_nan,
        'train_time_s': train_time,
        'speed_ms': speed_ms,
        'epochs_trained': epoch + 1,
    }

    print(f"\n  Final Results for {model_type}:")
    print(f"    Test Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"    Best Val Acc:  {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"    NaN events:    {total_nan}")
    print(f"    Train time:    {train_time:.1f}s")
    print(f"    Speed:         {speed_ms:.2f} ms/batch")

    return results


def benchmark_speed(model, device, config, num_iters=50):
    """Benchmark forward pass speed (ms per batch)."""
    model.eval()
    batch_size = config['training']['batch_size']
    seq_len = config['dataset']['max_seq_len']

    # Warmup
    dummy = torch.randint(0, 3, (batch_size, seq_len), device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.time() - start) * 1000 / num_iters  # ms per batch
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config(config_path)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(config['training']['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(config['training']['seed'])

    # Create B3 group and dataset
    print("\nCreating B3 composition dataset...")
    group = B3Group()
    print(f"  B3 group order: {len(group.elements)} elements")
    print(f"  Generators: σ_1 (swap 0,1), σ_2 (swap 1,2), τ (flip sign 0)")

    dataset = B3CompositionDataset(
        group=group,
        num_samples=config['dataset']['num_samples'],
        min_k=config['dataset']['min_k'],
        max_k=config['dataset']['max_k'],
        max_seq_len=config['dataset']['max_seq_len'],
        seed=config['training']['seed'],
    )

    # Train/test split
    test_size = int(len(dataset) * config['dataset']['test_ratio'])
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config['training']['seed']),
    )

    print(f"  Total: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    print(f"  Sequence lengths: {config['dataset']['min_k']}-{config['dataset']['max_k']}")
    print(f"  Max padded length: {config['dataset']['max_seq_len']}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Train all 3 models
    model_types = ['HyperSSM', 'DiagonalSSM', 'PermOnlySSM']
    all_results = {}

    for model_type in model_types:
        # Reset seed for each model for fair comparison
        torch.manual_seed(config['training']['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed(config['training']['seed'])

        results = train_model(model_type, config, train_loader, test_loader, device)
        all_results[model_type] = results

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: B3 Composition Task Results")
    print(f"{'='*80}")
    print(f"\n{'Model':<20} {'Params':>10} {'Test Acc':>10} {'Val Acc':>10} {'NaN':>6} {'Time(s)':>8} {'ms/batch':>10}")
    print(f"{'-'*84}")

    for name, r in all_results.items():
        print(f"{name:<20} {r['params']:>10,} {r['final_test_acc']*100:>9.1f}% "
              f"{r['best_val_acc']*100:>9.1f}% {r['total_nan']:>6} {r['train_time_s']:>8.1f} {r['speed_ms']:>10.2f}")

    # Success criteria evaluation
    print(f"\n{'='*80}")
    print(f"SUCCESS CRITERIA EVALUATION")
    print(f"{'='*80}")

    hyper_acc = all_results['HyperSSM']['final_test_acc']
    diag_acc = all_results['DiagonalSSM']['final_test_acc']
    perm_acc = all_results['PermOnlySSM']['final_test_acc']
    total_nan = sum(r['total_nan'] for r in all_results.values())

    criterion_1 = hyper_acc > 0.90
    criterion_2 = diag_acc < 0.60
    criterion_3 = perm_acc < 0.75
    criterion_4 = total_nan == 0

    print(f"\n1. HyperSSM > 90% on B3 composition:")
    print(f"   {'PASS' if criterion_1 else 'FAIL'}: {hyper_acc*100:.1f}% {'>' if criterion_1 else '<='} 90%")

    print(f"\n2. DiagonalSSM < 60% on same task:")
    print(f"   {'PASS' if criterion_2 else 'FAIL'}: {diag_acc*100:.1f}% {'<' if criterion_2 else '>='} 60%")

    print(f"\n3. PermOnlySSM < 75% (value of Z_2^n signs):")
    print(f"   {'PASS' if criterion_3 else 'FAIL'}: {perm_acc*100:.1f}% {'<' if criterion_3 else '>='} 75%")

    print(f"\n4. No NaN/Inf events (stability):")
    print(f"   {'PASS' if criterion_4 else 'FAIL'}: {total_nan} NaN/Inf events")

    # Speed comparison
    hyper_speed = all_results['HyperSSM']['speed_ms']
    diag_speed = all_results['DiagonalSSM']['speed_ms']
    perm_speed = all_results['PermOnlySSM']['speed_ms']
    speed_ratio = hyper_speed / diag_speed if diag_speed > 0 else float('inf')

    print(f"\n5. Speed overhead (informational):")
    print(f"   HyperSSM: {hyper_speed:.2f} ms/batch")
    print(f"   DiagonalSSM: {diag_speed:.2f} ms/batch")
    print(f"   PermOnlySSM: {perm_speed:.2f} ms/batch")
    print(f"   HyperSSM / DiagonalSSM ratio: {speed_ratio:.2f}x")

    # Per-k accuracy breakdown
    print(f"\n{'='*80}")
    print(f"PER-K ACCURACY BREAKDOWN")
    print(f"{'='*80}")

    all_ks = sorted(set().union(
        *[r['per_k_acc'].keys() for r in all_results.values()]
    ))

    if all_ks:
        header = f"{'k':>4}"
        for name in model_types:
            header += f" {name:>15}"
        print(header)

        for k in all_ks:
            row = f"{k:>4}"
            for name in model_types:
                acc = all_results[name]['per_k_acc'].get(k, 0)
                row += f" {acc*100:>14.1f}%"
            print(row)

    # Overall decision
    passed = sum([criterion_1, criterion_2, criterion_3, criterion_4])
    print(f"\n{'='*80}")
    print(f"DECISION: {passed}/4 criteria passed")

    if passed >= 3:
        print("PROCEED: Sufficient evidence that HyperSSM works")
    elif passed >= 2:
        print("DEBUG: Partial success, investigate failures")
    else:
        print("ABANDON: Insufficient evidence")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

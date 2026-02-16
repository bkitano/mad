"""
Training script for Experiment 019: Capacitance-Coupled Multi-Scale SSM.

Runs three models on nested periodic detection task:
1. MC-SSM (proposed): Multi-scale with capacitance coupling
2. Monolithic SSM (baseline): Single-scale, same total state dim
3. Uncoupled MS-SSM (ablation): Multi-scale without coupling (C=0)

Usage:
    python train.py                    # Run all models
    python train.py --model mc_ssm     # Run only MC-SSM
    python train.py --model monolithic # Run only monolithic
    python train.py --model uncoupled  # Run only uncoupled
"""

import argparse
import time
import math
import yaml
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mc_ssm import MultiScaleCapacitanceSSM, MonolithicSSM, UncoupledMultiScaleSSM
from data.generate import create_dataloaders, N_CLASSES


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * signals.shape[0]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += signals.shape[0]

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Per-class accuracy tracking
    per_class_correct = torch.zeros(N_CLASSES)
    per_class_total = torch.zeros(N_CLASSES)

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        total_loss += loss.item() * signals.shape[0]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += signals.shape[0]

        # Per-class
        for c in range(N_CLASSES):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    per_class_acc = {}
    for c in range(N_CLASSES):
        if per_class_total[c] > 0:
            per_class_acc[c] = (per_class_correct[c] / per_class_total[c]).item()
        else:
            per_class_acc[c] = 0.0

    return avg_loss, avg_acc, per_class_acc


def create_model(model_type: str, config: dict, device: torch.device):
    """Create model from config."""
    if model_type == "mc_ssm":
        model = MultiScaleCapacitanceSSM(
            d_input=config['d_input'],
            d_model=config['d_model'],
            n_state=config['n_state'],
            k_scales=config['k_scales'],
            n_layers=config['n_layers'],
            n_classes=config['n_classes'],
            dt_min=config['dt_min'],
            dt_max=config['dt_max'],
            dropout=config['dropout'],
            use_coupling=True
        )
    elif model_type == "monolithic":
        model = MonolithicSSM(
            d_input=config['d_input'],
            d_model=config['d_model'],
            n_state=config['n_state'],
            n_layers=config['n_layers'],
            n_classes=config['n_classes'],
            dropout=config['dropout']
        )
    elif model_type == "uncoupled":
        model = UncoupledMultiScaleSSM(
            d_input=config['d_input'],
            d_model=config['d_model'],
            n_state=config['n_state'],
            k_scales=config['k_scales'],
            n_layers=config['n_layers'],
            n_classes=config['n_classes'],
            dt_min=config['dt_min'],
            dt_max=config['dt_max'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_model(model_type: str, config: dict, device: torch.device,
                train_loader, val_loader, test_loader):
    """Train and evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_type.upper()}")
    print(f"{'='*60}")

    model = create_model(model_type, config, device)
    n_params = count_parameters(model)
    print(f"Model type: {model.model_type}")
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_epochs'], eta_min=config['lr'] * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(1, config['max_epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config['grad_clip']
        )

        # Evaluate
        val_loss, val_acc, val_per_class = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model state
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['max_epochs']}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if val_acc >= config.get('target_acc', 0.99):
            print(f"  -> Target accuracy {config.get('target_acc', 0.99):.2f} reached at epoch {epoch}!")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_loss, test_acc, test_per_class = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Per-class test accuracy:")
    for c in range(N_CLASSES):
        freq_mask = [(c >> i) & 1 for i in range(3)]
        print(f"  Class {c} (f1={freq_mask[0]}, f2={freq_mask[1]}, f3={freq_mask[2]}): {test_per_class[c]:.4f}")

    # Get capacitance stats if applicable
    cap_stats = {}
    if hasattr(model, 'get_capacitance_stats'):
        cap_stats = model.get_capacitance_stats()
        if cap_stats:
            print(f"\nCapacitance Statistics:")
            for layer_name, stats in cap_stats.items():
                print(f"  {layer_name}:")
                print(f"    dt_scales: {stats['dt_scales']}")
                print(f"    A_diag mean per scale: {stats['A_diag_mean_per_scale']}")

    # Check timescale separation
    dt_ratio = None
    if hasattr(model, 'layers') and hasattr(model.layers[0], 'dt_scales'):
        dt_vals = model.layers[0].dt_scales.cpu().tolist()
        dt_ratio = max(dt_vals) / min(dt_vals) if min(dt_vals) > 0 else float('inf')
        print(f"\nTimescale ratio (max/min dt): {dt_ratio:.1f}x")

    results = {
        'model_type': model_type,
        'model_name': model.model_type,
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_per_class_acc': {int(k): float(v) for k, v in test_per_class.items()},
        'training_time_s': elapsed,
        'history': history,
        'capacitance_stats': cap_stats,
        'dt_ratio': dt_ratio,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Train MC-SSM experiment 019')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'mc_ssm', 'monolithic', 'uncoupled'],
                        help='Which model to train')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])

    # Create dataloaders
    print("Generating synthetic data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        n_train=config['data']['n_train'],
        n_val=config['data']['n_val'],
        n_test=config['data']['n_test'],
        seq_len=config['data']['seq_len'],
        noise_std=config['data']['noise_std'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 0),
    )
    print(f"Data: {config['data']['n_train']} train, "
          f"{config['data']['n_val']} val, {config['data']['n_test']} test")
    print(f"Sequence length: {config['data']['seq_len']}")
    print(f"Classes: {N_CLASSES}")

    # Model config (shared across variants)
    model_config = {
        'd_input': 1,  # 1D signal
        'd_model': config['model']['d_model'],
        'n_state': config['model']['n_state'],
        'k_scales': config['model']['k_scales'],
        'n_layers': config['model']['n_layers'],
        'n_classes': N_CLASSES,
        'dt_min': config['model']['dt_min'],
        'dt_max': config['model']['dt_max'],
        'dropout': config['training']['dropout'],
        'lr': config['training']['lr'],
        'weight_decay': config['training']['weight_decay'],
        'max_epochs': config['training']['max_epochs'],
        'grad_clip': config['training']['grad_clip'],
        'target_acc': config['training'].get('target_acc', 0.99),
    }

    # Train models
    all_results = {}
    models_to_train = ['mc_ssm', 'monolithic', 'uncoupled'] if args.model == 'all' else [args.model]

    for model_type in models_to_train:
        results = train_model(
            model_type, model_config, device,
            train_loader, val_loader, test_loader
        )
        all_results[model_type] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Time':>10}")
    print(f"{'-'*65}")
    for model_type, results in all_results.items():
        print(f"{results['model_name']:<25} {results['n_params']:>10,} "
              f"{results['best_val_acc']:>10.4f} {results['test_acc']:>10.4f} "
              f"{results['training_time_s']:>8.1f}s")

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(results_dir, 'results.yaml')

    # Convert history to simpler format for YAML
    save_results = {}
    for model_type, results in all_results.items():
        save_results[model_type] = {
            'model_name': results['model_name'],
            'n_params': results['n_params'],
            'best_val_acc': float(results['best_val_acc']),
            'best_epoch': results['best_epoch'],
            'test_acc': float(results['test_acc']),
            'test_loss': float(results['test_loss']),
            'test_per_class_acc': results['test_per_class_acc'],
            'training_time_s': float(results['training_time_s']),
            'capacitance_stats': results.get('capacitance_stats', {}),
            'dt_ratio': float(results['dt_ratio']) if results.get('dt_ratio') else None,
            'final_train_acc': float(results['history'][-1]['train_acc']),
            'final_train_loss': float(results['history'][-1]['train_loss']),
        }

    with open(results_path, 'w') as f:
        yaml.dump(save_results, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {results_path}")

    # Check success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")

    if 'mc_ssm' in all_results:
        mc_acc = all_results['mc_ssm']['test_acc']
        print(f"1. MC-SSM accuracy > 90%: {mc_acc:.1%} -> {'PASS' if mc_acc > 0.9 else 'FAIL'}")

    if 'monolithic' in all_results:
        mono_acc = all_results['monolithic']['test_acc']
        print(f"2. Monolithic accuracy < 75%: {mono_acc:.1%} -> {'PASS' if mono_acc < 0.75 else 'FAIL'}")

    if 'uncoupled' in all_results:
        uncoupled_acc = all_results['uncoupled']['test_acc']
        print(f"3. Uncoupled accuracy 70-80%: {uncoupled_acc:.1%} -> "
              f"{'PASS' if 0.7 <= uncoupled_acc <= 0.8 else 'FAIL'}")

    if 'mc_ssm' in all_results and all_results['mc_ssm'].get('dt_ratio'):
        dt_ratio = all_results['mc_ssm']['dt_ratio']
        print(f"4. Timescale separation > 10x: {dt_ratio:.1f}x -> {'PASS' if dt_ratio > 10 else 'FAIL'}")


if __name__ == "__main__":
    main()

"""
Training script for Experiment 013: Circulant SSM vs Diagonal SSM on Z8 composition.

Runs both models on the same Z8 cyclic group composition task and compares:
1. Final accuracy (Circ-SSM should be >90%, Diagonal should be <60%)
2. Forward pass throughput (Circ-SSM should be >0.5x diagonal)
3. Numerical error of Fourier-domain computation (<1e-4 in FP32)

Usage:
    python train.py                    # Run full experiment (both models)
    python train.py --model circulant  # Run only circulant model
    python train.py --model diagonal   # Run only diagonal model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import time
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models.circulant_ssm import CirculantSSMModel, CirculantSSMLayer
from models.diagonal_ssm import DiagonalSSMModel
from tasks.z8_dataset import (
    create_z8_dataloaders,
    VOCAB_SIZE,
    NUM_CLASSES,
    IGNORE_INDEX,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)  # (B, T, num_classes)

        # Reshape for cross-entropy
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accuracy
        mask = labels != IGNORE_INDEX
        preds = logits.argmax(dim=-1)
        correct = (preds == labels) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        mask = labels != IGNORE_INDEX
        preds = logits.argmax(dim=-1)
        correct = (preds == labels) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return {"loss": avg_loss, "accuracy": accuracy}


def measure_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    seq_len: int = 64,
    num_warmup: int = 10,
    num_trials: int = 50,
) -> float:
    """
    Measure forward pass throughput in tokens/second.

    Returns average tokens/sec over num_trials forward passes.
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(0, 8, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed trials
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_tokens = batch_size * seq_len * num_trials
    tokens_per_sec = total_tokens / elapsed

    return tokens_per_sec


def check_numerical_error(model: CirculantSSMModel, device: torch.device) -> dict:
    """
    Check numerical error between spatial and Fourier-domain computation.

    Success criterion: ||h_spatial - IFFT(h_scan)||_inf < 1e-4 in FP32
    """
    model.eval()

    # Use first SSM layer for the check
    ssm_layer = model.layers[0]['ssm']

    # Create random input
    x = torch.randn(4, 32, model.d_model, device=device)

    with torch.no_grad():
        error, imag_max = ssm_layer.forward_with_spatial_check(x)

    return {"scan_vs_spatial_error": error, "imag_max": imag_max}


def train_model(
    model_type: str,
    config: dict,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_32_loader: torch.utils.data.DataLoader,
) -> dict:
    """
    Train a model and return results.

    Args:
        model_type: 'circulant' or 'diagonal'
        config: experiment config
        device: torch device
        train_loader, val_loader, test_32_loader: data loaders

    Returns:
        dict with all metrics
    """
    model_cfg = config['model']
    train_cfg = config['training']

    # Create model
    if model_type == 'circulant':
        model = CirculantSSMModel(
            vocab_size=VOCAB_SIZE,
            d_model=model_cfg['d_model'],
            state_dim=model_cfg['state_dim'],
            num_layers=model_cfg['num_layers'],
            num_classes=NUM_CLASSES,
            max_seq_len=config['dataset']['max_seq_len'],
            dropout=model_cfg.get('dropout', 0.1),
        )
    elif model_type == 'diagonal':
        model = DiagonalSSMModel(
            vocab_size=VOCAB_SIZE,
            d_model=model_cfg['d_model'],
            state_dim=model_cfg['state_dim'],
            num_layers=model_cfg['num_layers'],
            num_classes=NUM_CLASSES,
            max_seq_len=config['dataset']['max_seq_len'],
            dropout=model_cfg.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    num_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} SSM")
    print(f"Parameters: {num_params:,}")
    print(f"{'='*60}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 0.01),
        betas=(train_cfg.get('beta1', 0.9), train_cfg.get('beta2', 0.999)),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['max_epochs'], eta_min=train_cfg['lr'] * 0.1
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(1, train_cfg['max_epochs'] + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=train_cfg.get('gradient_clip', 1.0),
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })

        if epoch % 5 == 0 or epoch <= 3:
            print(
                f"  Epoch {epoch:3d}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        # Early stopping if we hit target accuracy
        if val_metrics['accuracy'] >= train_cfg.get('target_accuracy', 0.95):
            print(f"  -> Target accuracy reached at epoch {epoch}!")
            break

    train_time = time.time() - start_time

    # Evaluate on fixed-length test set (seq_len=32)
    test_32_metrics = evaluate(model, test_32_loader, device)
    print(f"\n  Test (seq_len=32): acc={test_32_metrics['accuracy']:.4f}")

    # Throughput measurement
    throughput = measure_throughput(model, device, batch_size=16, seq_len=64)
    print(f"  Throughput: {throughput:,.0f} tokens/sec")

    # Numerical error check (only for circulant)
    numerical_error = None
    if model_type == 'circulant':
        numerical_error = check_numerical_error(model, device)
        print(f"  Numerical error (spatial vs Fourier): {numerical_error['scan_vs_spatial_error']:.2e}")
        print(f"  Max imaginary part: {numerical_error['imag_max']:.2e}")

    results = {
        'model_type': model_type,
        'num_params': num_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_32_acc': test_32_metrics['accuracy'],
        'test_32_loss': test_32_metrics['loss'],
        'throughput_tokens_sec': throughput,
        'numerical_error': numerical_error,
        'train_time_sec': train_time,
        'history': history,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 013: Circulant SSM MVE")
    parser.add_argument(
        '--model', type=str, default='both',
        choices=['circulant', 'diagonal', 'both'],
        help='Which model to train'
    )
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).parent / 'config.yaml'),
        help='Path to config file'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu/mps). Auto-detects if not specified.'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Create data loaders
    ds_cfg = config['dataset']
    train_loader, val_loader, test_32_loader = create_z8_dataloaders(
        num_samples=ds_cfg['num_samples'],
        min_len=ds_cfg['min_len'],
        max_len=ds_cfg['max_len'],
        max_seq_len=ds_cfg['max_seq_len'],
        batch_size=config['training']['batch_size'],
        test_fraction=ds_cfg.get('test_fraction', 0.2),
        seed=ds_cfg.get('seed', 42),
    )

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test (len=32): {len(test_32_loader.dataset)} samples")

    # Run experiments
    all_results = {}

    models_to_run = ['circulant', 'diagonal'] if args.model == 'both' else [args.model]

    for model_type in models_to_run:
        results = train_model(
            model_type=model_type,
            config=config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_32_loader=test_32_loader,
        )
        all_results[model_type] = results

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 013 SUMMARY")
    print("=" * 60)

    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()} SSM:")
        print(f"  Parameters: {results['num_params']:,}")
        print(f"  Best val accuracy: {results['best_val_acc']:.4f}")
        print(f"  Test accuracy (len=32): {results['test_32_acc']:.4f}")
        print(f"  Throughput: {results['throughput_tokens_sec']:,.0f} tokens/sec")
        print(f"  Training time: {results['train_time_sec']:.1f}s")
        if results['numerical_error'] is not None:
            print(f"  Numerical error: {results['numerical_error']:.2e}")

    # Check success criteria
    if len(all_results) == 2:
        circ = all_results['circulant']
        diag = all_results['diagonal']

        print("\n" + "-" * 60)
        print("SUCCESS CRITERIA EVALUATION")
        print("-" * 60)

        # Criterion 1: Circ-SSM >90% on Z8 at seq_len=32
        c1 = circ['test_32_acc'] > 0.90
        print(f"  [{'PASS' if c1 else 'FAIL'}] Circ-SSM >90% on Z8 (len=32): {circ['test_32_acc']:.4f}")

        # Criterion 2: Diagonal <60% on same task
        c2 = diag['test_32_acc'] < 0.60
        print(f"  [{'PASS' if c2 else 'FAIL'}] Diagonal <60% on Z8 (len=32): {diag['test_32_acc']:.4f}")

        # Criterion 3: Throughput ratio >0.5x
        ratio = circ['throughput_tokens_sec'] / max(diag['throughput_tokens_sec'], 1)
        c3 = ratio > 0.5
        print(f"  [{'PASS' if c3 else 'FAIL'}] Throughput ratio >0.5x: {ratio:.3f}x")

        # Criterion 4: Numerical error <1e-4
        num_err = circ['numerical_error']['scan_vs_spatial_error'] if circ['numerical_error'] else None
        c4 = num_err is not None and num_err < 1e-4
        print(f"  [{'PASS' if c4 else 'FAIL'}] Numerical error <1e-4: {num_err:.2e}" if num_err else "  [FAIL] Numerical error: N/A")

        overall = c1 and c2 and c3 and c4
        print(f"\n  Overall: {'ALL PASS' if overall else 'SOME FAILURES'}")

    # Save results to JSON
    results_path = Path(__file__).parent / 'results.json'
    # Convert history for JSON serialization
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {key: val for key, val in v.items()}
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

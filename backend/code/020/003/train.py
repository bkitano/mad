"""
Training script for Experiment 003: OscGate-SSM MVE.

Trains and evaluates:
1. OscGate-SSM (input-dependent oscillatory SSM) — the proposed model
2. LinOSS (fixed oscillatory SSM, LTI) — baseline that should FAIL
3. DiagonalSSM (input-dependent diagonal) — for speed comparison

All on the selective copying task.

Success criteria (from proposal 007):
1. OscGate-SSM > 90% accuracy at length 32
2. LinOSS < 40% accuracy (proving selectivity matters)
3. No NaN/Inf during training
4. Forward pass < 3× slower than DiagonalSSM
"""

import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.oscgate_ssm import OscGateSSMClassifier
from models.linoss import LinOSSClassifier
from models.diagonal_ssm import DiagonalSSMClassifier
from data.generate import generate_selective_copying_data


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        (avg_loss, accuracy, nan_count)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, seq_len, num_classes)

        # Only compute loss at positions with valid targets (not -1)
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        # Check gradients for NaN/Inf
        grad_nan = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_nan = True
                break
        if grad_nan:
            nan_count += 1
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Compute accuracy only at valid positions (target != -1)
        valid_mask = targets_flat != -1
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model.

    Returns:
        (avg_loss, accuracy, nan_count)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)

        valid_mask = targets_flat != -1
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, nan_count


def benchmark_forward_pass(
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> float:
    """
    Benchmark forward pass wall-clock time.

    Returns:
        Average forward pass time in milliseconds.
    """
    model.eval()
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times)


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Full training loop for a single model.

    Returns dict with training history and final results.
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*60}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['beta1'], config['training']['beta2']),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr'] * 0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = 0.0
    best_state = None
    total_nan_count = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'nan_counts': [],
    }

    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 30)
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_nan = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=config['training']['gradient_clip'],
        )
        val_loss, val_acc, val_nan = evaluate(model, val_loader, criterion, device)

        total_nan_count += train_nan + val_nan
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['nan_counts'].append(train_nan + val_nan)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"NaN: {train_nan + val_nan} | "
                f"Best Val Acc: {best_val_acc:.4f}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

        # If already at > 95%, can stop early
        if val_acc > 0.95 and train_acc > 0.95:
            print(f"  Target accuracy reached at epoch {epoch}")
            break

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_nan = evaluate(model, test_loader, criterion, device)
    total_nan_count += test_nan

    print(f"\n  Final Test Accuracy: {test_acc:.4f}")
    print(f"  Total NaN/Inf events: {total_nan_count}")

    return {
        'model_name': model_name,
        'num_params': count_parameters(model),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_nan_count': total_nan_count,
        'epochs_trained': len(history['train_loss']),
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='OscGate-SSM MVE Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'oscgate', 'linoss', 'diagonal'],
                        help='Which model(s) to train')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Generate data
    print("\nGenerating selective copying data...")
    train_loader, val_loader, test_loader, dataset_info = generate_selective_copying_data(
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        content_len=config['data']['content_len'],
        num_queries=config['data'].get('num_queries', 4),
        vocab_size=config['data']['vocab_size'],
        batch_size=config['training']['batch_size'],
    )
    print(f"  Total vocab size: {dataset_info['total_vocab_size']}")
    print(f"  Sequence length: {dataset_info['seq_len']}")
    print(f"  Content length: {dataset_info['content_len']}")
    print(f"  Num queries: {dataset_info['num_queries']}")

    total_vocab = dataset_info['total_vocab_size']
    num_classes = total_vocab  # Predict any token
    d_model = config['model']['d_model']
    m = config['model']['m']
    n = 2 * m  # State dim for all models
    num_layers = config['model'].get('num_layers', 2)
    seq_len = dataset_info['seq_len']

    results = {}

    # ============================================================
    # 1. OscGate-SSM (proposed model)
    # ============================================================
    if args.model in ['all', 'oscgate']:
        oscgate_model = OscGateSSMClassifier(
            vocab_size=total_vocab,
            d_model=d_model,
            m=m,
            num_classes=num_classes,
            num_layers=num_layers,
            dt=config['model']['dt'],
            omega_max=config['model']['omega_max'],
            dropout=config['training']['dropout'],
        )
        results['oscgate'] = train_model(
            'OscGate-SSM', oscgate_model,
            train_loader, val_loader, test_loader,
            config, device,
        )

    # ============================================================
    # 2. LinOSS baseline (LTI — should FAIL)
    # ============================================================
    if args.model in ['all', 'linoss']:
        set_seed(config.get('seed', 42))
        linoss_model = LinOSSClassifier(
            vocab_size=total_vocab,
            d_model=d_model,
            m=m,
            num_classes=num_classes,
            num_layers=num_layers,
            dt=config['model']['dt'],
            dropout=config['training']['dropout'],
        )
        results['linoss'] = train_model(
            'LinOSS (LTI baseline)', linoss_model,
            train_loader, val_loader, test_loader,
            config, device,
        )

    # ============================================================
    # 3. DiagonalSSM (speed comparison baseline)
    # ============================================================
    if args.model in ['all', 'diagonal']:
        set_seed(config.get('seed', 42))
        diag_model = DiagonalSSMClassifier(
            vocab_size=total_vocab,
            d_model=d_model,
            n=n,  # Same state dim as OscGate-SSM
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=config['training']['dropout'],
        )
        results['diagonal'] = train_model(
            'DiagonalSSM (speed baseline)', diag_model,
            train_loader, val_loader, test_loader,
            config, device,
        )

    # ============================================================
    # Speed benchmark
    # ============================================================
    print(f"\n{'='*60}")
    print("Speed Benchmark (forward pass)")
    print(f"{'='*60}")

    speed_results = {}
    if args.model in ['all', 'oscgate']:
        oscgate_bench = OscGateSSMClassifier(
            vocab_size=total_vocab, d_model=d_model, m=m,
            num_classes=num_classes, num_layers=num_layers,
            dt=config['model']['dt'],
            omega_max=config['model']['omega_max'],
        ).to(device)
        t_oscgate = benchmark_forward_pass(
            oscgate_bench, device,
            seq_len=seq_len,
            batch_size=config['training']['batch_size'],
            vocab_size=total_vocab,
        )
        speed_results['oscgate'] = t_oscgate
        print(f"  OscGate-SSM: {t_oscgate:.2f} ms")

    if args.model in ['all', 'diagonal']:
        diag_bench = DiagonalSSMClassifier(
            vocab_size=total_vocab, d_model=d_model, n=n,
            num_classes=num_classes, num_layers=num_layers,
        ).to(device)
        t_diag = benchmark_forward_pass(
            diag_bench, device,
            seq_len=seq_len,
            batch_size=config['training']['batch_size'],
            vocab_size=total_vocab,
        )
        speed_results['diagonal'] = t_diag
        print(f"  DiagonalSSM: {t_diag:.2f} ms")

    if 'oscgate' in speed_results and 'diagonal' in speed_results:
        ratio = speed_results['oscgate'] / speed_results['diagonal']
        speed_results['ratio'] = ratio
        print(f"  Speed ratio (OscGate/Diagonal): {ratio:.2f}x")

    # ============================================================
    # Summary & Success Criteria
    # ============================================================
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    criteria = {}

    # Criterion 1: OscGate-SSM > 90% accuracy
    if 'oscgate' in results:
        c1 = results['oscgate']['test_acc'] > 0.90
        criteria['oscgate_accuracy'] = {
            'target': '> 90%',
            'achieved': f"{results['oscgate']['test_acc']*100:.1f}%",
            'passed': c1,
        }
        print(f"  1. OscGate-SSM accuracy > 90%: "
              f"{'✅ PASS' if c1 else '❌ FAIL'} "
              f"({results['oscgate']['test_acc']*100:.1f}%)")

    # Criterion 2: LinOSS < 40% accuracy
    if 'linoss' in results:
        c2 = results['linoss']['test_acc'] < 0.40
        criteria['linoss_accuracy'] = {
            'target': '< 40%',
            'achieved': f"{results['linoss']['test_acc']*100:.1f}%",
            'passed': c2,
        }
        print(f"  2. LinOSS accuracy < 40%: "
              f"{'✅ PASS' if c2 else '❌ FAIL'} "
              f"({results['linoss']['test_acc']*100:.1f}%)")

    # Criterion 3: No NaN/Inf
    if 'oscgate' in results:
        c3 = results['oscgate']['total_nan_count'] == 0
        criteria['stability'] = {
            'target': '0 NaN/Inf',
            'achieved': str(results['oscgate']['total_nan_count']),
            'passed': c3,
        }
        print(f"  3. No NaN/Inf events: "
              f"{'✅ PASS' if c3 else '❌ FAIL'} "
              f"(count: {results['oscgate']['total_nan_count']})")

    # Criterion 4: Speed < 3× diagonal
    if 'ratio' in speed_results:
        c4 = speed_results['ratio'] < 3.0
        criteria['speed'] = {
            'target': '< 3.0×',
            'achieved': f"{speed_results['ratio']:.2f}×",
            'passed': c4,
        }
        print(f"  4. Speed ratio < 3×: "
              f"{'✅ PASS' if c4 else '❌ FAIL'} "
              f"({speed_results['ratio']:.2f}×)")

    # Overall verdict
    all_passed = all(c.get('passed', False) for c in criteria.values())
    any_passed = any(c.get('passed', False) for c in criteria.values())

    if all_passed:
        verdict = 'PROCEED'
    elif any_passed:
        verdict = 'DEBUG'
    else:
        verdict = 'ABANDON'

    print(f"\n  VERDICT: {verdict}")

    # ============================================================
    # Save results
    # ============================================================
    save_results = {
        'config': config,
        'models': {},
        'speed_benchmark': speed_results,
        'success_criteria': criteria,
        'verdict': verdict,
    }
    for name, res in results.items():
        save_results['models'][name] = {
            'num_params': res['num_params'],
            'best_val_acc': float(res['best_val_acc']),
            'test_acc': float(res['test_acc']),
            'test_loss': float(res['test_loss']),
            'total_nan_count': res['total_nan_count'],
            'epochs_trained': res['epochs_trained'],
        }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(save_results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

    return save_results


if __name__ == '__main__':
    main()

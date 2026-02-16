"""
Training script for Experiment 006: Monarch-Gated State Transition SSM MVE.

Trains and evaluates Monarch-Gated SSM vs Diagonal SSM (and optionally Dense SSM)
on the S5 permutation composition task.

From proposal 006-monarch-gated-state-transition:
  "S5 permutation group composition -- given a sequence of generators of S5,
   predict the resulting permutation."

Success criteria (from proposal):
1. Monarch-Gated SSM achieves >85% accuracy on S5 composition (seq_len=20)
2. Diagonal SSM baseline achieves <50% accuracy on the same task
3. Forward pass of Monarch-Gated SSM is <3x slower than diagonal SSM

Failure criteria:
1. Monarch-Gated SSM cannot beat diagonal SSM at any seq_len
2. Forward pass is >5x slower than diagonal
"""

import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.monarch_ssm import SSMClassifier
from data.generate import generate_s5_data, NUM_CLASSES, NUM_GENERATORS


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch. Returns (avg_loss, accuracy, nan_count)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue

        loss.backward()

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
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy, nan_count)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy, nan_count


def benchmark_forward_pass(model, device, seq_len, batch_size, vocab_size,
                           n_warmup=10, n_runs=50):
    """Benchmark forward pass. Returns avg time in milliseconds."""
    model.eval()
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

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
            times.append((end - start) * 1000)

    return np.mean(times)


def train_model(model_name, model, train_loader, val_loader, test_loader,
                config, device):
    """Full training loop for a single model. Returns results dict."""
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

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    total_nan_count = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'nan_counts': [],
    }

    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 40)
    patience_counter = 0
    train_start = time.perf_counter()

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
                f"Best Val: {best_val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

        if val_acc > 0.95 and train_acc > 0.95:
            print(f"  Target accuracy reached at epoch {epoch}")
            break

    train_time = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_nan = evaluate(model, test_loader, criterion, device)
    total_nan_count += test_nan

    print(f"\n  Final Test Accuracy: {test_acc:.4f}")
    print(f"  Total NaN/Inf events: {total_nan_count}")
    print(f"  Training time: {train_time:.1f}s")

    return {
        'model_name': model_name,
        'num_params': count_parameters(model),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_nan_count': total_nan_count,
        'epochs_trained': len(history['train_loss']),
        'train_time_s': train_time,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='Monarch-Gated SSM MVE Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Generate data
    print("\nGenerating S5 permutation composition data...")
    train_loader, val_loader, test_loader, dataset_info = generate_s5_data(
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        seq_len=config['data']['seq_len'],
        batch_size=config['training']['batch_size'],
    )
    print(f"  Vocab size: {dataset_info['vocab_size']} (generators)")
    print(f"  Num classes: {dataset_info['num_classes']} (|S5| = 120)")
    print(f"  Sequence length: {dataset_info['seq_len']}")

    d_model = config['model']['d_model']
    n = config['model']['n']
    num_layers = config['model']['num_layers']
    seq_len = dataset_info['seq_len']
    ssm_types = config.get('ssm_types', ['monarch', 'diagonal', 'dense'])

    # ============================================================
    # Train each model type
    # ============================================================
    all_results = {}

    for ssm_type in ssm_types:
        set_seed(config.get('seed', 42))

        model = SSMClassifier(
            vocab_size=dataset_info['vocab_size'],
            d_model=d_model,
            n=n,
            num_classes=dataset_info['num_classes'],
            num_layers=num_layers,
            ssm_type=ssm_type,
            dropout=config['training']['dropout'],
        )

        result = train_model(
            f"{ssm_type.capitalize()}-SSM", model,
            train_loader, val_loader, test_loader, config, device,
        )
        all_results[ssm_type] = result

    # ============================================================
    # Speed Benchmark
    # ============================================================
    print(f"\n{'='*60}")
    print("SPEED BENCHMARK (forward pass)")
    print(f"{'='*60}")

    speed_results = {}
    for ssm_type in ssm_types:
        set_seed(42)
        model = SSMClassifier(
            vocab_size=dataset_info['vocab_size'],
            d_model=d_model, n=n,
            num_classes=dataset_info['num_classes'],
            num_layers=num_layers, ssm_type=ssm_type,
        ).to(device)

        t = benchmark_forward_pass(
            model, device, seq_len=seq_len,
            batch_size=config['training']['batch_size'],
            vocab_size=dataset_info['vocab_size'],
        )
        speed_results[ssm_type] = t
        print(f"  {ssm_type.capitalize()}-SSM: {t:.2f} ms")

    speed_ratio = None
    if 'monarch' in speed_results and 'diagonal' in speed_results:
        speed_ratio = speed_results['monarch'] / speed_results['diagonal']
        print(f"\n  Speed ratio (Monarch/Diagonal): {speed_ratio:.2f}x")
        print(f"  Target: < 3.0x  |  {'PASS' if speed_ratio < 3.0 else 'FAIL'}")

    # ============================================================
    # Success Criteria Evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    criteria = {}

    # Criterion 1: Monarch-Gated SSM > 85% on S5
    if 'monarch' in all_results:
        monarch_acc = all_results['monarch']['test_acc']
        c1 = monarch_acc > 0.85
        criteria['monarch_accuracy'] = {
            'target': '>85%',
            'achieved': f'{monarch_acc*100:.1f}%',
            'passed': c1,
        }
        print(f"  1. Monarch-SSM accuracy: {monarch_acc*100:.1f}% "
              f"{'PASS' if c1 else 'FAIL'} (target >85%)")

    # Criterion 2: Diagonal SSM < 50%
    if 'diagonal' in all_results:
        diag_acc = all_results['diagonal']['test_acc']
        c2 = diag_acc < 0.50
        criteria['diagonal_accuracy'] = {
            'target': '<50%',
            'achieved': f'{diag_acc*100:.1f}%',
            'passed': c2,
        }
        print(f"  2. Diagonal-SSM accuracy: {diag_acc*100:.1f}% "
              f"{'PASS' if c2 else 'FAIL'} (target <50%)")

    # Criterion 3: Speed ratio < 3x
    if speed_ratio is not None:
        c3 = speed_ratio < 3.0
        criteria['speed_ratio'] = {
            'target': '<3.0x',
            'achieved': f'{speed_ratio:.2f}x',
            'passed': c3,
        }
        print(f"  3. Speed ratio (Monarch/Diagonal): {speed_ratio:.2f}x "
              f"{'PASS' if c3 else 'FAIL'} (target <3.0x)")

    # Failure criterion: Monarch must beat diagonal
    if 'monarch' in all_results and 'diagonal' in all_results:
        monarch_beats_diag = all_results['monarch']['test_acc'] > all_results['diagonal']['test_acc']
        criteria['monarch_beats_diagonal'] = {
            'target': 'Monarch > Diagonal',
            'achieved': (f"Monarch={all_results['monarch']['test_acc']*100:.1f}% "
                        f"vs Diagonal={all_results['diagonal']['test_acc']*100:.1f}%"),
            'passed': monarch_beats_diag,
        }
        print(f"  4. Monarch beats Diagonal: "
              f"{'PASS' if monarch_beats_diag else 'KILL -- coordinate mixing insufficient'}")

    # Failure criterion: Speed not > 5x
    if speed_ratio is not None:
        speed_not_terrible = speed_ratio < 5.0
        criteria['speed_not_terrible'] = {
            'target': '<5.0x',
            'achieved': f'{speed_ratio:.2f}x',
            'passed': speed_not_terrible,
        }
        print(f"  5. Speed not >5x slower: "
              f"{'PASS' if speed_not_terrible else 'KILL -- BMM overhead too high'}")

    # NaN check
    if 'monarch' in all_results:
        no_nans = all_results['monarch']['total_nan_count'] == 0
        criteria['stability'] = {
            'target': '0 NaN/Inf events',
            'achieved': f"{all_results['monarch']['total_nan_count']} events",
            'passed': no_nans,
        }
        print(f"  6. Stability (no NaN/Inf): "
              f"{'PASS' if no_nans else 'FAIL'} "
              f"({all_results['monarch']['total_nan_count']} events)")

    # Overall verdict
    key_criteria = ['monarch_accuracy', 'diagonal_accuracy', 'speed_ratio']
    key_passed = all(criteria.get(k, {}).get('passed', False)
                     for k in key_criteria if k in criteria)
    all_success = all(c.get('passed', False) for c in criteria.values())

    if all_success:
        verdict = 'PROCEED'
    elif key_passed:
        verdict = 'PROCEED'
    elif not criteria.get('monarch_beats_diagonal', {}).get('passed', True):
        verdict = 'ABANDON'
    else:
        verdict = 'DEBUG'

    print(f"\n  VERDICT: {verdict}")

    # ============================================================
    # Save results
    # ============================================================
    save_results = {
        'config': config,
        'models': {},
        'speed_benchmark': {k: float(v) for k, v in speed_results.items()},
        'speed_ratio_monarch_over_diagonal': float(speed_ratio) if speed_ratio else None,
        'success_criteria': {k: v for k, v in criteria.items()},
        'verdict': verdict,
    }

    for ssm_type, res in all_results.items():
        save_results['models'][ssm_type] = {
            'model_name': res['model_name'],
            'num_params': res['num_params'],
            'best_val_acc': float(res['best_val_acc']),
            'test_acc': float(res['test_acc']),
            'test_loss': float(res['test_loss']),
            'total_nan_count': res['total_nan_count'],
            'epochs_trained': res['epochs_trained'],
            'train_time_s': float(res['train_time_s']),
        }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(save_results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params':>8} {'Test Acc':>10} {'Time (ms)':>10} {'NaN':>5}")
    print(f"{'-'*55}")
    for ssm_type in ssm_types:
        if ssm_type in all_results:
            r = all_results[ssm_type]
            t = speed_results.get(ssm_type, 0)
            print(f"{r['model_name']:<20} {r['num_params']:>8,} "
                  f"{r['test_acc']*100:>9.1f}% {t:>9.2f} {r['total_nan_count']:>5}")
    print(f"{'='*60}")

    return save_results


if __name__ == '__main__':
    main()

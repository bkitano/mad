"""
Training script for Experiment 007: OH-DeltaProduct MVE.

Trains and evaluates 5 model variants on S3 permutation composition:
  1. OH-DeltaProduct (full model, β ∈ (0,2)) — proposed model
  2. LinOSS-only (n_h=0, oscillatory only) — should FAIL on S3
  3. DeltaProduct-only (no oscillatory, β ∈ (0,2)) — should work but NaN risk
  4. OH-DeltaProduct β-restricted (β ∈ (0,1)) — ablation: should underperform
  5. DeltaProduct-only β-restricted (β ∈ (0,1)) — ablation baseline

Success criteria (from proposal 020):
  1. OH-DeltaProduct > 95% accuracy on S3
  2. LinOSS-only < 40% accuracy
  3. DeltaProduct-only > 90% BUT > 5% NaN rate with β ∈ (0,2)
  4. OH-DeltaProduct has 0% NaN rate
  5. β ∈ (0,1) ablation < 60% accuracy
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

from models.oh_deltaproduct import OHDeltaProductClassifier
from models.linoss_only import LinOSSOnlyClassifier
from models.deltaproduct_only import DeltaProductOnlyClassifier
from data.generate import generate_s3_data, S3_SIZE, TOTAL_VOCAB


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_nan_inf(tensor: torch.Tensor) -> bool:
    """Return True if tensor contains NaN or Inf."""
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Train for one epoch.
    Returns: (avg_loss, accuracy, nan_count)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, seq_len, num_classes)

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN/Inf in loss
        if check_nan_inf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        # Check gradients for NaN/Inf
        grad_nan = False
        for p in model.parameters():
            if p.grad is not None and check_nan_inf(p.grad):
                grad_nan = True
                break
        if grad_nan:
            nan_count += 1
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        num_batches += 1

        # Accuracy at valid positions (target != -100)
        valid_mask = targets_flat != -100
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model.
    Returns: (avg_loss, accuracy, nan_count)
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
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        if check_nan_inf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)

        valid_mask = targets_flat != -100
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy, nan_count


def train_model(model_name, model, train_loader, val_loader, test_loader, config, device):
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

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

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

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_nan = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=config['training']['gradient_clip'],
        )
        val_loss, val_acc, val_nan = evaluate(model, val_loader, criterion, device)

        epoch_nan = train_nan + val_nan
        total_nan_count += epoch_nan
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['nan_counts'].append(epoch_nan)

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
                f"NaN: {epoch_nan} | Best: {best_val_acc:.4f}"
            )

        # Early stopping on patience
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

        # Early stopping on target reached
        if val_acc > 0.97 and train_acc > 0.97:
            print(f"  Target accuracy reached at epoch {epoch}")
            break

    # Load best model for final eval
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
    parser = argparse.ArgumentParser(description='OH-DeltaProduct MVE Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'oh_dp', 'linoss', 'dp', 'oh_dp_half', 'dp_half'],
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

    # Generate S3 composition data
    print("\nGenerating S3 permutation composition data...")
    train_loader, val_loader, test_loader, dataset_info = generate_s3_data(
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        seq_len=config['data']['seq_len'],
        use_generators_only=True,
        batch_size=config['training']['batch_size'],
    )
    print(f"  Vocab size: {dataset_info['total_vocab_size']}")
    print(f"  Num classes: {dataset_info['num_classes']} (S3 has 6 elements)")
    print(f"  Padded seq len: {dataset_info['seq_len']}")
    print(f"  Group size: {dataset_info['group_size']}")

    # Common model args
    d_model = config['model']['d_model']
    m = config['model']['m']
    n_h = config['model']['n_h']
    num_layers = config['model']['num_layers']
    dt = config['model']['dt']
    omega_max = config['model']['omega_max']
    dropout = config['training']['dropout']
    vocab_size = dataset_info['total_vocab_size']
    num_classes = dataset_info['num_classes']

    results = {}
    start_time = time.time()

    # ============================================================
    # 1. OH-DeltaProduct (full model, β ∈ (0,2))
    # ============================================================
    if args.model in ['all', 'oh_dp']:
        set_seed(config.get('seed', 42))
        oh_dp_model = OHDeltaProductClassifier(
            vocab_size=vocab_size, d_model=d_model, m=m,
            num_classes=num_classes, n_h=n_h, num_layers=num_layers,
            dt=dt, omega_max=omega_max, use_oscillatory=True,
            beta_range='full', dropout=dropout,
        )
        results['oh_dp'] = train_model(
            'OH-DeltaProduct (full, β∈(0,2))', oh_dp_model,
            train_loader, val_loader, test_loader, config, device,
        )

    # ============================================================
    # 2. LinOSS-only (n_h=0, should FAIL on S3)
    # ============================================================
    if args.model in ['all', 'linoss']:
        set_seed(config.get('seed', 42))
        linoss_model = LinOSSOnlyClassifier(
            vocab_size=vocab_size, d_model=d_model, m=m,
            num_classes=num_classes, num_layers=num_layers,
            dt=dt, omega_max=omega_max, dropout=dropout,
        )
        results['linoss'] = train_model(
            'LinOSS-only (no Householder)', linoss_model,
            train_loader, val_loader, test_loader, config, device,
        )

    # ============================================================
    # 3. DeltaProduct-only (no oscillatory, β ∈ (0,2)) — NaN risk
    # ============================================================
    if args.model in ['all', 'dp']:
        set_seed(config.get('seed', 42))
        dp_model = DeltaProductOnlyClassifier(
            vocab_size=vocab_size, d_model=d_model, m=m,
            num_classes=num_classes, n_h=n_h, num_layers=num_layers,
            beta_range='full', dropout=dropout,
        )
        results['dp'] = train_model(
            'DeltaProduct-only (β∈(0,2), no osc)', dp_model,
            train_loader, val_loader, test_loader, config, device,
        )

    # ============================================================
    # 4. OH-DeltaProduct β-restricted (β ∈ (0,1)) — ablation
    # ============================================================
    if args.model in ['all', 'oh_dp_half']:
        set_seed(config.get('seed', 42))
        oh_dp_half_model = OHDeltaProductClassifier(
            vocab_size=vocab_size, d_model=d_model, m=m,
            num_classes=num_classes, n_h=n_h, num_layers=num_layers,
            dt=dt, omega_max=omega_max, use_oscillatory=True,
            beta_range='half', dropout=dropout,
        )
        results['oh_dp_half'] = train_model(
            'OH-DeltaProduct (β∈(0,1) ablation)', oh_dp_half_model,
            train_loader, val_loader, test_loader, config, device,
        )

    # ============================================================
    # 5. DeltaProduct-only β-restricted (β ∈ (0,1)) — control
    # ============================================================
    if args.model in ['all', 'dp_half']:
        set_seed(config.get('seed', 42))
        dp_half_model = DeltaProductOnlyClassifier(
            vocab_size=vocab_size, d_model=d_model, m=m,
            num_classes=num_classes, n_h=n_h, num_layers=num_layers,
            beta_range='half', dropout=dropout,
        )
        results['dp_half'] = train_model(
            'DeltaProduct-only (β∈(0,1) ablation)', dp_half_model,
            train_loader, val_loader, test_loader, config, device,
        )

    total_time = time.time() - start_time

    # ============================================================
    # Summary & Success Criteria
    # ============================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n{'Model':<45} {'Params':>8} {'Test Acc':>10} {'NaN Count':>10}")
    print("-" * 73)
    for key, res in results.items():
        print(f"  {res['model_name']:<43} {res['num_params']:>8,} "
              f"{res['test_acc']*100:>9.1f}% {res['total_nan_count']:>10}")

    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*70}")

    criteria = {}

    # Criterion 1: OH-DeltaProduct > 95% accuracy
    if 'oh_dp' in results:
        c1 = results['oh_dp']['test_acc'] > 0.95
        criteria['oh_dp_accuracy'] = {
            'target': '> 95%',
            'achieved': f"{results['oh_dp']['test_acc']*100:.1f}%",
            'passed': c1,
        }
        print(f"  1. OH-DeltaProduct > 95%: "
              f"{'✅ PASS' if c1 else '❌ FAIL'} "
              f"({results['oh_dp']['test_acc']*100:.1f}%)")

    # Criterion 2: LinOSS-only < 40% accuracy
    if 'linoss' in results:
        c2 = results['linoss']['test_acc'] < 0.40
        criteria['linoss_accuracy'] = {
            'target': '< 40%',
            'achieved': f"{results['linoss']['test_acc']*100:.1f}%",
            'passed': c2,
        }
        print(f"  2. LinOSS-only < 40%: "
              f"{'✅ PASS' if c2 else '❌ FAIL'} "
              f"({results['linoss']['test_acc']*100:.1f}%)")

    # Criterion 3: DeltaProduct-only > 90% BUT NaN events
    if 'dp' in results:
        dp_acc = results['dp']['test_acc'] > 0.90
        dp_epochs = results['dp']['epochs_trained']
        dp_nan_rate = results['dp']['total_nan_count'] / max(dp_epochs, 1)
        dp_nan_high = dp_nan_rate > 0.05 or results['dp']['total_nan_count'] > 5
        criteria['dp_accuracy'] = {
            'target': '> 90%',
            'achieved': f"{results['dp']['test_acc']*100:.1f}%",
            'passed': dp_acc,
        }
        criteria['dp_nan_rate'] = {
            'target': '> 5% NaN rate',
            'achieved': f"{results['dp']['total_nan_count']} NaN events ({dp_nan_rate*100:.1f}% rate)",
            'passed': dp_nan_high,
        }
        print(f"  3a. DeltaProduct-only > 90%: "
              f"{'✅ PASS' if dp_acc else '❌ FAIL'} "
              f"({results['dp']['test_acc']*100:.1f}%)")
        print(f"  3b. DeltaProduct-only NaN rate > 5%: "
              f"{'✅ PASS' if dp_nan_high else '❌ FAIL'} "
              f"({results['dp']['total_nan_count']} events, {dp_nan_rate*100:.1f}% rate)")

    # Criterion 4: OH-DeltaProduct has 0% NaN rate
    if 'oh_dp' in results:
        c4 = results['oh_dp']['total_nan_count'] == 0
        criteria['oh_dp_stability'] = {
            'target': '0 NaN/Inf',
            'achieved': str(results['oh_dp']['total_nan_count']),
            'passed': c4,
        }
        print(f"  4. OH-DeltaProduct 0% NaN: "
              f"{'✅ PASS' if c4 else '❌ FAIL'} "
              f"(count: {results['oh_dp']['total_nan_count']})")

    # Criterion 5: β ∈ (0,1) ablation < 60% accuracy
    if 'oh_dp_half' in results:
        c5 = results['oh_dp_half']['test_acc'] < 0.60
        criteria['beta_half_accuracy'] = {
            'target': '< 60%',
            'achieved': f"{results['oh_dp_half']['test_acc']*100:.1f}%",
            'passed': c5,
        }
        print(f"  5. β∈(0,1) ablation < 60%: "
              f"{'✅ PASS' if c5 else '❌ FAIL'} "
              f"({results['oh_dp_half']['test_acc']*100:.1f}%)")

    # Overall verdict
    num_passed = sum(1 for c in criteria.values() if c.get('passed', False))
    num_total = len(criteria)

    if num_passed == num_total:
        verdict = 'PROCEED'
    elif num_passed >= num_total * 0.5:
        verdict = 'DEBUG'
    else:
        verdict = 'ABANDON'

    print(f"\n  Criteria passed: {num_passed}/{num_total}")
    print(f"  VERDICT: {verdict}")

    # ============================================================
    # Save results
    # ============================================================
    save_results = {
        'config': config,
        'models': {},
        'success_criteria': criteria,
        'verdict': verdict,
        'total_time_seconds': total_time,
    }
    for name, res in results.items():
        save_results['models'][name] = {
            'model_name': res['model_name'],
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

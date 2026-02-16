"""
Training script for Experiment 028: Neumann-Cayley Orthogonal SSM MVE.

Trains and evaluates:
1. NC-SSM (Neumann-Cayley orthogonal SSM) -- the proposed model
2. DiagonalSSM (input-dependent diagonal) -- baseline that should FAIL on S5

Both on the S5 permutation composition task.

Success criteria (from proposal 028):
1. NC-SSM > 80% accuracy on S5 composition where DiagonalSSM < 30%
2. Orthogonality deviation ||W^T W - I||_F < 0.1 maintained throughout training
3. Training loss converges (no divergence from approximate orthogonality)

Failure criteria:
1. If model diverges (NaN/Inf) within 1000 steps, Neumann approx too loose
2. If S5 accuracy < 40% after 5000 steps, combination fails
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

from models.nc_ssm import NCSSMClassifier
from models.diagonal_ssm import DiagonalSSMClassifier
from data.generate import generate_s5_data


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_ncssm(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """
    Train NC-SSM for one epoch. Returns metrics including orthogonality deviation.

    For S5 task, we only care about the LAST position's prediction
    (the composed permutation).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0
    ortho_devs = []

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # NC-SSM returns (logits, ortho_dev)
        logits, ortho_dev = model(inputs)

        # For S5 task: predict at LAST position
        last_logits = logits[:, -1, :]  # (batch, num_classes)
        loss = criterion(last_logits, targets)

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
        preds = last_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)
        ortho_devs.append(ortho_dev.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_ortho_dev = np.mean(ortho_devs) if ortho_devs else float('inf')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'nan_count': nan_count,
        'ortho_dev': avg_ortho_dev,
    }


@torch.no_grad()
def evaluate_ncssm(model: nn.Module, loader, device: torch.device) -> dict:
    """Evaluate NC-SSM."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0
    ortho_devs = []

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, ortho_dev = model(inputs)
        last_logits = logits[:, -1, :]
        loss = criterion(last_logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)
        preds = last_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)
        ortho_devs.append(ortho_dev.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_ortho_dev = np.mean(ortho_devs) if ortho_devs else float('inf')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'nan_count': nan_count,
        'ortho_dev': avg_ortho_dev,
    }


def train_epoch_diagonal(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train DiagonalSSM for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        last_logits = logits[:, -1, :]
        loss = criterion(last_logits, targets)

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
        preds = last_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'nan_count': nan_count,
    }


@torch.no_grad()
def evaluate_diagonal(model: nn.Module, loader, device: torch.device) -> dict:
    """Evaluate DiagonalSSM."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        last_logits = logits[:, -1, :]
        loss = criterion(last_logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        total_loss += loss.item() * inputs.size(0)
        preds = last_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'nan_count': nan_count,
    }


def train_model(
    model_name: str,
    model: nn.Module,
    train_fn,
    eval_fn,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    device: torch.device,
) -> dict:
    """Full training loop for a single model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*60}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr'] * 0.01,
    )

    best_val_acc = 0.0
    best_state = None
    total_nan_count = 0
    total_steps = 0
    max_ortho_dev = 0.0  # Track max orthogonality deviation

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'ortho_devs': [],
    }

    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 30)
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_fn(
            model, train_loader, optimizer, device,
            max_grad_norm=config['training']['gradient_clip'],
        )
        val_metrics = eval_fn(model, val_loader, device)

        total_nan_count += train_metrics['nan_count'] + val_metrics.get('nan_count', 0)
        total_steps += len(train_loader)
        scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Track orthogonality deviation for NC-SSM
        ortho_dev = train_metrics.get('ortho_dev', None)
        if ortho_dev is not None:
            history['ortho_devs'].append(ortho_dev)
            max_ortho_dev = max(max_ortho_dev, ortho_dev)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or epoch == epochs:
            ortho_str = f" OrtDev: {ortho_dev:.4f}" if ortho_dev is not None else ""
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train L: {train_metrics['loss']:.4f} A: {train_metrics['accuracy']:.4f} | "
                f"Val L: {val_metrics['loss']:.4f} A: {val_metrics['accuracy']:.4f} | "
                f"NaN: {train_metrics['nan_count']}{ortho_str} | "
                f"Best: {best_val_acc:.4f}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

        # Early success
        if val_metrics['accuracy'] > 0.95 and train_metrics['accuracy'] > 0.95:
            print(f"  Target accuracy reached at epoch {epoch}")
            break

        # Failure check: divergence within early steps
        if total_nan_count > 50 and epoch < 20:
            print(f"  DIVERGENCE DETECTED: {total_nan_count} NaN/Inf events in {epoch} epochs")
            break

    elapsed = time.time() - start_time

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_fn(model, test_loader, device)
    total_nan_count += test_metrics.get('nan_count', 0)

    print(f"\n  Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Total NaN/Inf events: {total_nan_count}")
    print(f"  Training time: {elapsed:.1f}s")
    if history['ortho_devs']:
        print(f"  Max ortho deviation: {max_ortho_dev:.4f}")
        print(f"  Final ortho deviation: {history['ortho_devs'][-1]:.4f}")

    return {
        'model_name': model_name,
        'num_params': count_parameters(model),
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'total_nan_count': total_nan_count,
        'epochs_trained': len(history['train_loss']),
        'training_time_s': elapsed,
        'max_ortho_dev': max_ortho_dev,
        'final_ortho_dev': history['ortho_devs'][-1] if history['ortho_devs'] else None,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='NC-SSM MVE Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'ncssm', 'diagonal'],
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
    print("\nGenerating S5 permutation composition data...")
    train_loader, val_loader, test_loader, dataset_info = generate_s5_data(
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        seq_len=config['data']['seq_len'],
        batch_size=config['training']['batch_size'],
    )
    print(f"  Vocab size (generators): {dataset_info['vocab_size']}")
    print(f"  Num classes (|S5|): {dataset_info['num_classes']}")
    print(f"  Sequence length: {dataset_info['seq_len']}")

    num_classes = dataset_info['num_classes']
    vocab_size = dataset_info['vocab_size']
    d_model = config['model']['d_model']
    n = config['model']['n']
    k = config['model']['k']
    rho_max = config['model']['rho_max']
    num_layers = config['model']['num_layers']

    results = {}

    # ============================================================
    # 1. NC-SSM (proposed model)
    # ============================================================
    if args.model in ['all', 'ncssm']:
        set_seed(config.get('seed', 42))
        ncssm_model = NCSSMClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            n=n,
            num_classes=num_classes,
            num_layers=num_layers,
            k=k,
            rho_max=rho_max,
            dropout=config['training']['dropout'],
        )
        results['ncssm'] = train_model(
            'NC-SSM (Neumann-Cayley)', ncssm_model,
            train_epoch_ncssm, evaluate_ncssm,
            train_loader, val_loader, test_loader,
            config, device,
        )

    # ============================================================
    # 2. DiagonalSSM baseline
    # ============================================================
    if args.model in ['all', 'diagonal']:
        set_seed(config.get('seed', 42))
        diag_model = DiagonalSSMClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            n=n,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=config['training']['dropout'],
        )
        results['diagonal'] = train_model(
            'DiagonalSSM (baseline)', diag_model,
            train_epoch_diagonal, evaluate_diagonal,
            train_loader, val_loader, test_loader,
            config, device,
        )

    # ============================================================
    # Summary & Success Criteria
    # ============================================================
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    criteria = {}

    # Criterion 1: NC-SSM > 80% accuracy
    if 'ncssm' in results:
        c1 = results['ncssm']['test_acc'] > 0.80
        criteria['ncssm_accuracy'] = {
            'target': '> 80%',
            'achieved': f"{results['ncssm']['test_acc']*100:.1f}%",
            'passed': c1,
        }
        print(f"  1. NC-SSM accuracy > 80%: "
              f"{'PASS' if c1 else 'FAIL'} "
              f"({results['ncssm']['test_acc']*100:.1f}%)")

    # Criterion 1b: DiagonalSSM < 30%
    if 'diagonal' in results:
        c1b = results['diagonal']['test_acc'] < 0.30
        criteria['diagonal_accuracy'] = {
            'target': '< 30%',
            'achieved': f"{results['diagonal']['test_acc']*100:.1f}%",
            'passed': c1b,
        }
        print(f"  1b. DiagonalSSM accuracy < 30%: "
              f"{'PASS' if c1b else 'FAIL'} "
              f"({results['diagonal']['test_acc']*100:.1f}%)")

    # Criterion 2: Orthogonality deviation < 0.1
    if 'ncssm' in results and results['ncssm']['max_ortho_dev'] is not None:
        c2 = results['ncssm']['max_ortho_dev'] < 0.1
        criteria['orthogonality'] = {
            'target': '< 0.1',
            'achieved': f"{results['ncssm']['max_ortho_dev']:.4f}",
            'passed': c2,
        }
        print(f"  2. Ortho deviation < 0.1: "
              f"{'PASS' if c2 else 'FAIL'} "
              f"(max: {results['ncssm']['max_ortho_dev']:.4f})")

    # Criterion 3: No divergence (NaN/Inf)
    if 'ncssm' in results:
        c3 = results['ncssm']['total_nan_count'] == 0
        criteria['stability'] = {
            'target': '0 NaN/Inf',
            'achieved': str(results['ncssm']['total_nan_count']),
            'passed': c3,
        }
        print(f"  3. No NaN/Inf (stability): "
              f"{'PASS' if c3 else 'FAIL'} "
              f"(count: {results['ncssm']['total_nan_count']})")

    # Criterion 4: Training loss converges
    if 'ncssm' in results:
        hist = results['ncssm']['history']
        if len(hist['train_loss']) >= 2:
            converged = hist['train_loss'][-1] < hist['train_loss'][0]
            c4 = converged
        else:
            c4 = False
        criteria['convergence'] = {
            'target': 'Loss decreasing',
            'achieved': f"{hist['train_loss'][0]:.3f} -> {hist['train_loss'][-1]:.3f}" if len(hist['train_loss']) >= 2 else 'N/A',
            'passed': c4,
        }
        print(f"  4. Loss convergence: "
              f"{'PASS' if c4 else 'FAIL'} "
              f"({criteria['convergence']['achieved']})")

    # Check failure criteria
    print(f"\n  FAILURE CRITERIA CHECK:")
    if 'ncssm' in results:
        # Failure 1: Divergence within 1000 steps (~13 epochs at 78 batches/epoch)
        if results['ncssm']['total_nan_count'] > 10 and results['ncssm']['epochs_trained'] < 15:
            print(f"  FAILURE: Neumann approximation too loose for stability")
        else:
            print(f"  OK: No early divergence")

        # Failure 2: < 40% accuracy after sufficient training
        steps = results['ncssm']['epochs_trained'] * len(train_loader)
        if steps >= 5000 and results['ncssm']['test_acc'] < 0.40:
            print(f"  FAILURE: S5 accuracy too low after {steps} steps ({results['ncssm']['test_acc']*100:.1f}%)")
        elif results['ncssm']['test_acc'] < 0.40:
            print(f"  WARNING: Accuracy below 40% ({results['ncssm']['test_acc']*100:.1f}%) but only {steps} steps trained")
        else:
            print(f"  OK: S5 accuracy above 40% ({results['ncssm']['test_acc']*100:.1f}%)")

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
            'training_time_s': float(res['training_time_s']),
            'max_ortho_dev': float(res['max_ortho_dev']) if res.get('max_ortho_dev') else None,
            'final_ortho_dev': float(res['final_ortho_dev']) if res.get('final_ortho_dev') else None,
        }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(save_results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

    return save_results


if __name__ == '__main__':
    main()

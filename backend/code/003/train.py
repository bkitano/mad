"""
Training script for DPLR-CS SSM Parity Experiment.

MVE for proposal 003: Tests whether fixed permutation P applied to DPLR
provides measurable benefit on parity task requiring inter-dimension coupling.

Compares 4 models:
1. DPLR (P = I): Standard diagonal + rank-1, no permutation
2. DPLR-CS (cyclic P): Cyclic shift permutation
3. DPLR-CS (bit-reversal P): Bit-reversal permutation (maximizes mixing)
4. DPLR-CS (learned P): Doubly-stochastic relaxation

Success criteria:
- DPLR (P=I): ~50-70% accuracy (struggles without coupling)
- Any DPLR-CS (P != I): >90% accuracy

Usage:
    python train.py                    # Run all variants
    python train.py --p_type identity  # Run single variant
    python train.py --config config.yaml  # Use config file
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import yaml
from tqdm import tqdm

from models.dplr_cs_ssm import DPLRCSModel


# ============================================================
# Data Generation
# ============================================================

def generate_parity_data(
    T: int = 32,
    n_samples: int = 10000,
    seed: int = 42,
) -> tuple:
    """Generate parity task data.

    Input: Binary sequence x_1, ..., x_T in {0, 1}
    Output: XOR of all bits = sum(x) mod 2

    This task requires information from ALL positions to propagate
    to a single output. A diagonal SSM can only track each dimension
    independently — it cannot aggregate across dimensions without
    B and C doing all the work.

    Args:
        T: Sequence length
        n_samples: Number of samples
        seed: Random seed

    Returns:
        X: Tensor of shape (n_samples, T) with binary values
        Y: Tensor of shape (n_samples,) with parity labels (0 or 1)
    """
    torch.manual_seed(seed)
    X = torch.randint(0, 2, (n_samples, T)).float()
    Y = (X.sum(dim=1) % 2).long()  # Parity: 0 or 1
    return X, Y


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple:
    """Train for one epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)  # (batch, 2)
        loss = criterion(logits, Y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == Y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on dataset.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == Y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


# ============================================================
# Single Run
# ============================================================

def run_single(
    P_type: str,
    config: dict,
    device: torch.device,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run training for a single permutation type.

    Args:
        P_type: Permutation type ('identity', 'cyclic', 'bit_reversal', 'learned')
        config: Configuration dictionary
        device: Torch device
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with training metrics and results
    """
    torch.manual_seed(seed)

    # Data
    data_cfg = config.get('data', {})
    T = data_cfg.get('seq_len', 32)
    n_samples = data_cfg.get('n_samples', 10000)
    test_fraction = data_cfg.get('test_fraction', 0.2)

    X, Y = generate_parity_data(T=T, n_samples=n_samples, seed=seed)
    dataset = TensorDataset(X, Y)

    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(seed))

    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model_cfg = config.get('model', {})
    model = DPLRCSModel(
        d_input=1,
        d_model=model_cfg.get('d_model', 32),
        state_dim=model_cfg.get('state_dim', 16),
        P_type=P_type,
        dt=model_cfg.get('dt', 1.0),
    ).to(device)

    n_params = model.count_parameters()
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Model: DPLR-CS SSM (P={P_type})")
        print(f"  Parameters: {n_params:,}")
        print(f"  State dim: {model_cfg.get('state_dim', 16)}")
        print(f"  d_model: {model_cfg.get('d_model', 32)}")
        print(f"  Sequence length: {T}")
        print(f"  Device: {device}")
        print(f"{'='*60}")

    # Optimizer & Loss
    lr = float(train_cfg.get('lr', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler
    max_epochs = train_cfg.get('max_epochs', 200)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    best_test_acc = 0.0
    best_epoch = 0
    train_history = []
    patience = train_cfg.get('patience', 30)
    patience_counter = 0
    early_stop_acc = train_cfg.get('early_stop_acc', 0.99)

    start_time = time.time()

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=train_cfg.get('gradient_clip', 1.0),
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"Test: loss={test_loss:.4f} acc={test_acc:.3f} | "
                  f"Best={best_test_acc:.3f}")

        # Early stopping
        if best_test_acc >= early_stop_acc:
            if verbose:
                print(f"  Early stop: test acc {best_test_acc:.3f} >= {early_stop_acc}")
            break
        if patience_counter >= patience:
            if verbose:
                print(f"  Patience exhausted at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time

    results = {
        'P_type': P_type,
        'n_params': n_params,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_history[-1]['train_acc'],
        'final_test_acc': train_history[-1]['test_acc'],
        'final_train_loss': train_history[-1]['train_loss'],
        'final_test_loss': train_history[-1]['test_loss'],
        'total_epochs': len(train_history),
        'elapsed_seconds': elapsed,
        'history': train_history,
    }

    if verbose:
        print(f"\n  Results for P={P_type}:")
        print(f"    Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})")
        print(f"    Final train accuracy: {train_history[-1]['train_acc']:.4f}")
        print(f"    Training time: {elapsed:.1f}s")

    return results


# ============================================================
# Main: Run all variants
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DPLR-CS SSM Parity Experiment")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--p_type', type=str, default=None,
                        choices=['identity', 'cyclic', 'bit_reversal', 'learned'],
                        help='Run single permutation type (default: run all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of random seeds to average over')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")
        config = {}

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Determine which permutation types to run
    if args.p_type:
        p_types = [args.p_type]
    else:
        p_types = ['identity', 'cyclic', 'bit_reversal', 'learned']

    # Run experiments with multiple seeds
    seeds = [args.seed + i * 100 for i in range(args.n_seeds)]
    all_results = {}

    for p_type in p_types:
        print(f"\n{'#'*60}")
        print(f"  Running P_type = {p_type} ({args.n_seeds} seeds)")
        print(f"{'#'*60}")

        seed_results = []
        for seed in seeds:
            result = run_single(p_type, config, device, seed=seed, verbose=True)
            seed_results.append(result)

        # Aggregate over seeds
        best_accs = [r['best_test_acc'] for r in seed_results]
        mean_acc = sum(best_accs) / len(best_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in best_accs) / len(best_accs)) ** 0.5
        mean_time = sum(r['elapsed_seconds'] for r in seed_results) / len(seed_results)

        all_results[p_type] = {
            'mean_best_acc': mean_acc,
            'std_best_acc': std_acc,
            'best_accs': best_accs,
            'mean_time': mean_time,
            'n_params': seed_results[0]['n_params'],
            'seed_results': seed_results,
        }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 003 RESULTS: DPLR-CS SSM Parity Task")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Accuracy':>12} {'Std':>8} {'Time (s)':>10} {'Params':>8}")
    print("-" * 70)

    for p_type, res in all_results.items():
        print(f"DPLR-CS (P={p_type}){'':<{25 - len(f'DPLR-CS (P={p_type})')}} "
              f"{res['mean_best_acc']:>11.4f} "
              f"{res['std_best_acc']:>7.4f} "
              f"{res['mean_time']:>9.1f} "
              f"{res['n_params']:>8d}")

    # ============================================================
    # Success/Failure Criteria Check
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUCCESS / FAILURE CRITERIA")
    print("=" * 70)

    identity_acc = all_results.get('identity', {}).get('mean_best_acc', 0)
    non_identity_accs = {k: v['mean_best_acc'] for k, v in all_results.items() if k != 'identity'}

    # Check success: P != I should get >90%, P = I should get <75%
    if identity_acc < 0.75 and any(a > 0.90 for a in non_identity_accs.values()):
        print("\n  ✅ SUCCESS: Permutation provides inter-dimension coupling!")
        print(f"     Identity accuracy: {identity_acc:.4f} (< 75% ✓)")
        for p, acc in non_identity_accs.items():
            status = "✅" if acc > 0.90 else "❌"
            print(f"     {p} accuracy: {acc:.4f} {'> 90% ✓' if acc > 0.90 else '< 90% ✗'} {status}")
        decision = "PROCEED"
    elif all(a > 0.90 for a in list(non_identity_accs.values()) + [identity_acc]):
        print("\n  ⚠️  ALL models achieve >90% — parity may be too easy for this config")
        print(f"     Identity accuracy: {identity_acc:.4f}")
        for p, acc in non_identity_accs.items():
            print(f"     {p} accuracy: {acc:.4f}")
        decision = "INVESTIGATE"
        print("  → Need harder task or smaller model to separate the variants")
    elif all(abs(a - identity_acc) < 0.05 for a in non_identity_accs.values()):
        print("\n  ❌ FAILURE: All permutations perform same as identity")
        print(f"     Identity accuracy: {identity_acc:.4f}")
        for p, acc in non_identity_accs.items():
            print(f"     {p} accuracy: {acc:.4f} (diff: {acc - identity_acc:+.4f})")
        decision = "KILL"
        print("  → Permutation is absorbed into B/C reparameterization")
    else:
        print("\n  🔍 MIXED RESULTS — needs investigation")
        print(f"     Identity accuracy: {identity_acc:.4f}")
        for p, acc in non_identity_accs.items():
            print(f"     {p} accuracy: {acc:.4f}")
        decision = "DEBUG"

    # Check learned vs fixed
    if 'learned' in non_identity_accs and len(non_identity_accs) > 1:
        fixed_accs = [a for k, a in non_identity_accs.items() if k != 'learned']
        learned_acc = non_identity_accs['learned']
        best_fixed = max(fixed_accs)
        if learned_acc > best_fixed + 0.02:
            print(f"\n  ✅ Learned P outperforms best fixed P: {learned_acc:.4f} vs {best_fixed:.4f}")
        elif learned_acc < best_fixed - 0.02:
            print(f"\n  ⚠️  Learned P underperforms best fixed P: {learned_acc:.4f} vs {best_fixed:.4f}")
            print("     Learning permutation may not add value")
        else:
            print(f"\n  ≈ Learned P ≈ best fixed P: {learned_acc:.4f} vs {best_fixed:.4f}")

    print(f"\n  DECISION: {decision}")
    print("=" * 70)

    # Save results to JSON
    results_dir = Path(__file__).parent
    results_path = results_dir / 'results.json'
    save_results = {}
    for p_type, res in all_results.items():
        save_results[p_type] = {
            'mean_best_acc': res['mean_best_acc'],
            'std_best_acc': res['std_best_acc'],
            'best_accs': res['best_accs'],
            'mean_time': res['mean_time'],
            'n_params': res['n_params'],
        }
    save_results['decision'] = decision
    save_results['timestamp'] = datetime.now().isoformat()

    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results, decision


if __name__ == '__main__':
    main()

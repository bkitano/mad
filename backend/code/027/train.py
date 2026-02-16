"""
Training script for CC-SSM MVE (Proposal 027).

Runs delayed copy task comparing:
  1. CC-SSM: Cayley-circulant orthogonal SSM (|lambda| = 1 by construction)
  2. DiagonalSSM: Standard diagonal SSM with sigmoid decay (|lambda| < 1)

Tests across delays T = {50, 100, 200, 500}.

Success criteria (from proposal):
  1. CC-SSM > 99% copy accuracy at T=500 where diagonal SSM < 80%
  2. CC-SSM > 90% at T=200 (else: implementation bug)
  3. Speed < 10x diagonal SSM
  4. No NaN/Inf during training
  5. |lambda| = 1 preserved after training
"""

import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cc_ssm import CCSSM
from models.diagonal_ssm import DiagonalSSM
from data.generate import create_dataloaders, IGNORE_INDEX


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch. Returns (loss, accuracy, nan_count)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, seq_len, vocab_size)

        # Flatten for loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()

        # Check for NaN in gradients
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                nan_count += 1
                break

        if has_nan_grad:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Accuracy on valid (non-IGNORE) positions only
        valid_mask = targets_flat != IGNORE_INDEX
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_valid += valid_mask.sum().item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item() * inputs.size(0)

        valid_mask = targets_flat != IGNORE_INDEX
        if valid_mask.any():
            preds = logits_flat[valid_mask].argmax(dim=-1)
            total_correct += (preds == targets_flat[valid_mask]).sum().item()
            total_valid += valid_mask.sum().item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return avg_loss, accuracy


def benchmark_speed(model, seq_len, vocab_size, device, batch_size=32, n_iters=50):
    """Benchmark forward pass speed. Returns avg ms per batch."""
    model.eval()
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # Time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000  # ms per batch

    return elapsed


def run_experiment(config):
    """Run the full MVE experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    d_model = config['model']['d_model']
    state_dim = config['model']['state_dim']
    num_layers = config['model']['num_layers']
    dropout = config['model'].get('dropout', 0.1)
    vocab_size = config['data']['vocab_size']
    k = config['data']['k']
    delays = config['data']['delays']
    num_train = config['data']['num_train']
    num_test = config['data']['num_test']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    max_epochs = config['training']['max_epochs']
    patience = config['training']['patience']
    grad_clip = config['training']['gradient_clip']

    results = {}
    total_nan_count = 0

    for delay in delays:
        print(f"\n{'='*70}")
        print(f"DELAY T = {delay}")
        print(f"{'='*70}")

        # Max seq len for this delay
        max_seq_len = 2 * k + delay + 1 + 10  # +10 buffer for position embeddings

        # Create data
        train_loader, test_loader, total_vocab, seq_len = create_dataloaders(
            num_train=num_train,
            num_test=num_test,
            vocab_size=vocab_size,
            k=k,
            delay=delay,
            batch_size=batch_size,
        )
        print(f"Sequence length: {seq_len}, Total vocab: {total_vocab}")

        delay_results = {}

        for model_name, ModelClass in [('CC-SSM', CCSSM), ('DiagonalSSM', DiagonalSSM)]:
            print(f"\n--- {model_name} (delay={delay}) ---")

            # Create model
            model = ModelClass(
                vocab_size=total_vocab,
                d_model=d_model,
                state_dim=state_dim,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len,
            ).to(device)

            n_params = count_parameters(model)
            print(f"Parameters: {n_params:,}")

            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config['training'].get('weight_decay', 0.01),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

            # Training loop
            best_val_acc = 0.0
            epochs_no_improve = 0
            model_nan_count = 0
            train_start = time.perf_counter()

            for epoch in range(max_epochs):
                train_loss, train_acc, nan_count = train_epoch(
                    model, train_loader, optimizer, criterion, device, grad_clip
                )
                model_nan_count += nan_count
                scheduler.step()

                val_loss, val_acc = evaluate(model, test_loader, criterion, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                          f"val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}")

                # Early stopping
                if best_val_acc >= 0.995:
                    print(f"  Early stop: reached {best_val_acc:.4f} >= 99.5%")
                    break
                if epochs_no_improve >= patience:
                    print(f"  Early stop: no improvement for {patience} epochs")
                    break

            train_time = time.perf_counter() - train_start

            # Final evaluation
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # Check eigenvalue magnitudes
            eig_mags = model.get_eigenvalue_magnitudes()
            eig_mag_info = []
            for i, mags in enumerate(eig_mags):
                min_mag = mags.min().item()
                max_mag = mags.max().item()
                mean_mag = mags.mean().item()
                eig_mag_info.append({
                    'layer': i,
                    'min': min_mag,
                    'max': max_mag,
                    'mean': mean_mag,
                })
                print(f"  Layer {i} |lambda|: min={min_mag:.6f} max={max_mag:.6f} mean={mean_mag:.6f}")

            # Speed benchmark
            speed_ms = benchmark_speed(model, seq_len, total_vocab, device,
                                       batch_size=min(32, batch_size), n_iters=20)

            total_nan_count += model_nan_count

            delay_results[model_name] = {
                'test_acc': test_acc,
                'best_val_acc': best_val_acc,
                'test_loss': test_loss,
                'train_time_s': train_time,
                'speed_ms': speed_ms,
                'params': n_params,
                'epochs': epoch + 1,
                'nan_count': model_nan_count,
                'eig_magnitudes': eig_mag_info,
            }

            print(f"  Final: test_acc={test_acc:.4f} train_time={train_time:.1f}s "
                  f"speed={speed_ms:.2f}ms/batch NaN_count={model_nan_count}")

        results[delay] = delay_results

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Delay':>6} | {'CC-SSM Acc':>10} | {'DiagSSM Acc':>11} | {'CC Speed (ms)':>13} | {'Diag Speed (ms)':>15} | {'Speed Ratio':>11}")
    print("-" * 80)

    for delay in delays:
        cc = results[delay]['CC-SSM']
        diag = results[delay]['DiagonalSSM']
        speed_ratio = cc['speed_ms'] / diag['speed_ms'] if diag['speed_ms'] > 0 else float('inf')
        print(f"{delay:>6} | {cc['test_acc']:>10.4f} | {diag['test_acc']:>11.4f} | "
              f"{cc['speed_ms']:>13.2f} | {diag['speed_ms']:>15.2f} | {speed_ratio:>11.2f}x")

    # Evaluate success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*70}")

    criteria = {}

    # 1. CC-SSM > 99% at T=500 where diagonal < 80%
    if 500 in results:
        cc_500 = results[500]['CC-SSM']['test_acc']
        diag_500 = results[500]['DiagonalSSM']['test_acc']
        c1_pass = cc_500 > 0.99 and diag_500 < 0.80
        criteria['criterion_1'] = {
            'description': 'CC-SSM > 99% at T=500, DiagSSM < 80%',
            'cc_acc': cc_500,
            'diag_acc': diag_500,
            'pass': c1_pass,
        }
        status = "PASS" if c1_pass else "FAIL"
        print(f"1. {status}: CC-SSM={cc_500:.4f} (target >0.99), DiagSSM={diag_500:.4f} (target <0.80)")
    else:
        print("1. SKIP: T=500 not in delays")

    # 2. CC-SSM > 90% at T=200
    if 200 in results:
        cc_200 = results[200]['CC-SSM']['test_acc']
        c2_pass = cc_200 > 0.90
        criteria['criterion_2'] = {
            'description': 'CC-SSM > 90% at T=200 (else: bug)',
            'cc_acc': cc_200,
            'pass': c2_pass,
        }
        status = "PASS" if c2_pass else "FAIL"
        print(f"2. {status}: CC-SSM at T=200 = {cc_200:.4f} (target >0.90)")

    # 3. Speed < 10x diagonal
    speed_ratios = []
    for delay in delays:
        cc_speed = results[delay]['CC-SSM']['speed_ms']
        diag_speed = results[delay]['DiagonalSSM']['speed_ms']
        ratio = cc_speed / diag_speed if diag_speed > 0 else float('inf')
        speed_ratios.append(ratio)
    max_speed_ratio = max(speed_ratios)
    c3_pass = max_speed_ratio < 10.0
    criteria['criterion_3'] = {
        'description': 'Speed < 10x diagonal SSM',
        'max_speed_ratio': max_speed_ratio,
        'pass': c3_pass,
    }
    status = "PASS" if c3_pass else "FAIL"
    print(f"3. {status}: Max speed ratio = {max_speed_ratio:.2f}x (target <10x)")

    # 4. No NaN/Inf
    c4_pass = total_nan_count == 0
    criteria['criterion_4'] = {
        'description': 'No NaN/Inf during training',
        'nan_count': total_nan_count,
        'pass': c4_pass,
    }
    status = "PASS" if c4_pass else "FAIL"
    print(f"4. {status}: Total NaN/Inf events = {total_nan_count}")

    # 5. |lambda| = 1 preserved for CC-SSM
    all_eig_ok = True
    for delay in delays:
        for eig in results[delay]['CC-SSM']['eig_magnitudes']:
            if abs(eig['min'] - 1.0) > 0.01 or abs(eig['max'] - 1.0) > 0.01:
                all_eig_ok = False
    criteria['criterion_5'] = {
        'description': '|lambda| = 1 preserved after training',
        'pass': all_eig_ok,
    }
    status = "PASS" if all_eig_ok else "FAIL"
    print(f"5. {status}: |lambda| = 1 preserved = {all_eig_ok}")

    # Overall decision
    n_pass = sum(1 for c in criteria.values() if c['pass'])
    n_total = len(criteria)
    print(f"\nOverall: {n_pass}/{n_total} criteria passed")

    if n_pass >= 4:
        print("Decision: PROCEED")
    elif n_pass >= 3:
        print("Decision: PROCEED with caveats")
    elif n_pass >= 2:
        print("Decision: DEBUG")
    else:
        print("Decision: ABANDON")

    return results, criteria


def main():
    parser = argparse.ArgumentParser(description='CC-SSM MVE Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results, criteria = run_experiment(config)

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(results_dir, 'run_results.yaml')
    # Convert results to serializable format
    serializable_results = {}
    for delay, delay_results in results.items():
        serializable_results[delay] = {}
        for model_name, model_results in delay_results.items():
            serializable_results[delay][model_name] = {
                k: v for k, v in model_results.items()
                if k != 'eig_magnitudes'
            }
            # Flatten eig magnitudes
            serializable_results[delay][model_name]['eig_magnitudes'] = [
                {k2: round(v2, 6) for k2, v2 in eig.items()}
                for eig in model_results['eig_magnitudes']
            ]

    with open(results_file, 'w') as f:
        yaml.dump({
            'results': serializable_results,
            'criteria': criteria,
        }, f, default_flow_style=False)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

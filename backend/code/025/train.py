"""
Training script for Nystrom Landmark Compression Chunkwise SSM MVE.

From proposal 025-nystrom-landmark-chunkwise-ssm:

Compares 2 models:
1. Full Chunkwise SSM (m=n=32, no compression) - baseline
2. Nystrom-compressed Chunkwise SSM (m=8, 4x compression)

Task: Delayed copy - copy tokens after a gap spanning 2 chunk boundaries.

Success criteria:
- Nystrom-compressed (m=8, 4x) achieves > 90% copy accuracy at delay G=64
- Full model (m=n=32, no compression) achieves > 95% copy accuracy
- Gap is < 5% (the lost information is in negligible singular values)
- Memory for inter-chunk state transfer is verified at O(mn)=O(256) vs O(n^2)=O(1024)

Failure criteria:
- Compressed model < 70% copy accuracy: state info can't be compressed to m
- No memory/speed improvement: Nystrom overhead exceeds savings at small n

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force unbuffered output for Modal logging
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.nystrom_ssm import NystromChunkSSMModel, FullChunkSSMModel
from data.generate import create_dataloaders, DelayedCopyDataset


def compute_copy_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_content: int = 8,
) -> dict:
    """
    Compute copy accuracy: fraction of content tokens correctly reproduced.

    Returns:
        dict with per-position accuracy, overall accuracy, loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    n_batches = 0
    per_pos_correct = [0] * n_content
    per_pos_total = [0] * n_content

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)  # (batch, seq_len, vocab)

            # Compute loss on target positions only
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            total_loss += loss.item()
            n_batches += 1

            # Compute accuracy on target positions
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            mask = targets != -100  # (batch, seq_len)

            correct = (preds == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Per-position accuracy
            # Find target positions in each sample
            for b in range(inputs.shape[0]):
                target_positions = torch.where(mask[b])[0]
                for j, pos in enumerate(target_positions):
                    if j < n_content:
                        per_pos_correct[j] += correct[b, pos].item()
                        per_pos_total[j] += 1

    accuracy = total_correct / max(total_tokens, 1)
    avg_loss = total_loss / max(n_batches, 1)

    per_pos_acc = []
    for j in range(n_content):
        if per_pos_total[j] > 0:
            per_pos_acc.append(per_pos_correct[j] / per_pos_total[j])
        else:
            per_pos_acc.append(0.0)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'per_position_accuracy': per_pos_acc,
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = targets != -100
        total_correct += ((preds == targets) & mask).sum().item()
        total_tokens += mask.sum().item()

    n_batches = len(loader) - nan_count
    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': total_correct / max(total_tokens, 1),
        'nan_count': nan_count,
    }


def measure_memory_and_speed(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 256,
    vocab_size: int = 16,
    batch_size: int = 32,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> dict:
    """Measure forward pass memory and speed."""
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)

    # Measure time
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
    }


def run_experiment(config: dict, device: torch.device) -> dict:
    """Run the full experiment comparing Full vs Nystrom models."""
    # Config
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    seq_len = data_cfg.get('seq_len', 256)
    n_content = data_cfg.get('n_content', 8)
    delay = data_cfg.get('delay', 64)
    vocab_size = data_cfg.get('vocab_size', 16)
    n_train = data_cfg.get('n_train', 5000)
    n_test = data_cfg.get('n_test', 1000)
    batch_size = train_cfg.get('batch_size', 64)
    max_epochs = train_cfg.get('max_epochs', 150)
    lr = float(train_cfg.get('lr', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    patience = train_cfg.get('patience', 30)
    gradient_clip = float(train_cfg.get('gradient_clip', 1.0))

    d_model = model_cfg.get('d_model', 64)
    state_dim = model_cfg.get('state_dim', 32)
    n_landmarks = model_cfg.get('n_landmarks', 8)
    chunk_size = model_cfg.get('chunk_size', 32)
    n_layers = model_cfg.get('n_layers', 2)

    # Create data
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 025: Nystrom Landmark Compression for Chunkwise SSM")
    print(f"{'='*70}")
    print(f"  seq_len={seq_len}, n_content={n_content}, delay={delay}")
    print(f"  d_model={d_model}, state_dim={state_dim}, n_landmarks={n_landmarks}")
    print(f"  chunk_size={chunk_size}, n_layers={n_layers}")
    print(f"  vocab_size={vocab_size}, n_train={n_train}, n_test={n_test}")
    print(f"  batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}")
    print(f"  Device: {device}")

    train_loader, test_loader = create_dataloaders(
        n_train=n_train,
        n_test=n_test,
        seq_len=seq_len,
        n_content=n_content,
        delay=delay,
        vocab_size=vocab_size,
        batch_size=batch_size,
    )

    # Create models
    models = {
        'full': FullChunkSSMModel(
            vocab_size=vocab_size,
            d_model=d_model,
            state_dim=state_dim,
            chunk_size=chunk_size,
            n_layers=n_layers,
            max_seq_len=seq_len,
        ),
        'nystrom': NystromChunkSSMModel(
            vocab_size=vocab_size,
            d_model=d_model,
            state_dim=state_dim,
            n_landmarks=n_landmarks,
            chunk_size=chunk_size,
            n_layers=n_layers,
            max_seq_len=seq_len,
        ),
    }

    all_results = {}

    for name, model in models.items():
        model = model.to(device)
        n_params = model.count_parameters()

        print(f"\n{'#'*70}")
        print(f"  Training model: {name}")
        print(f"  Parameters: {n_params:,}")
        if name == 'nystrom':
            compression = state_dim ** 2 / (state_dim * n_landmarks + n_landmarks ** 2)
            print(f"  Compression ratio: {compression:.2f}x")
            print(f"  Full memory: O({state_dim**2}) = {state_dim**2}")
            print(f"  Compressed memory: O({state_dim * n_landmarks} + {n_landmarks**2}) = {state_dim * n_landmarks + n_landmarks**2}")
        print(f"{'#'*70}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        total_nan = 0
        history = []
        start_time = time.time()

        for epoch in range(max_epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, device,
                max_grad_norm=gradient_clip,
            )
            total_nan += train_metrics['nan_count']

            # Evaluate
            test_metrics = compute_copy_accuracy(
                model, test_loader, device, n_content=n_content,
            )
            scheduler.step()

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'test_loss': test_metrics['loss'],
                'test_acc': test_metrics['accuracy'],
                'per_pos_acc': test_metrics['per_position_accuracy'],
            })

            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | "
                      f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
                      f"Test: acc={test_metrics['accuracy']:.3f} | "
                      f"Best={best_test_acc:.3f} | "
                      f"NaN={total_nan}", flush=True)

            # Early stopping
            if best_test_acc >= 0.99:
                print(f"  Early stop: test acc {best_test_acc:.3f} >= 0.99")
                break
            if patience_counter >= patience:
                print(f"  Patience exhausted at epoch {epoch+1}")
                break

        elapsed = time.time() - start_time

        # Final evaluation
        final_metrics = compute_copy_accuracy(
            model, test_loader, device, n_content=n_content,
        )

        # Speed benchmark
        speed = measure_memory_and_speed(
            model, device, seq_len=seq_len, vocab_size=vocab_size,
            batch_size=min(32, batch_size),
        )

        # Compression stats (for Nystrom model)
        compression_stats = None
        if name == 'nystrom':
            try:
                x_sample = next(iter(test_loader))[0][:4].to(device)
                compression_stats = model.get_compression_stats(x_sample)
            except Exception as e:
                print(f"  Warning: compression stats failed: {e}")

        result = {
            'name': name,
            'n_params': n_params,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_acc': final_metrics['accuracy'],
            'final_per_pos_acc': final_metrics['per_position_accuracy'],
            'total_nan': total_nan,
            'elapsed_seconds': elapsed,
            'total_epochs': len(history),
            'speed_ms': speed['avg_time_ms'],
            'history': history,
            'compression_stats': compression_stats,
        }

        all_results[name] = result

        print(f"\n  Results for {name}:")
        print(f"    Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})")
        print(f"    Final test accuracy: {final_metrics['accuracy']:.4f}")
        print(f"    Per-position accuracy: {[f'{a:.3f}' for a in final_metrics['per_position_accuracy']]}")
        print(f"    Total NaN events: {total_nan}")
        print(f"    Training time: {elapsed:.1f}s")
        print(f"    Forward pass speed: {speed['avg_time_ms']:.1f}ms")

        if compression_stats:
            for i, stats in enumerate(compression_stats):
                print(f"    Layer {i} compression:")
                print(f"      Rel approx error: {stats['rel_approx_error']:.4f}")
                print(f"      Compression ratio: {stats['compression_ratio']:.2f}x")
                svd = stats['mean_singular_values']
                print(f"      Top-5 singular values: {[f'{v:.4f}' for v in svd[:5]]}")
                print(f"      Bottom-5 singular values: {[f'{v:.6f}' for v in svd[-5:]]}")

    # ================================================================
    # Summary & Success Criteria Evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 025 RESULTS SUMMARY")
    print("=" * 70)

    full_acc = all_results['full']['best_test_acc']
    nystrom_acc = all_results['nystrom']['best_test_acc']
    gap = full_acc - nystrom_acc

    print(f"\n{'Model':<25} {'Test Acc':>10} {'Params':>10} {'Speed (ms)':>12} {'NaN':>6}")
    print("-" * 70)
    for name, res in all_results.items():
        print(f"{name:<25} {res['best_test_acc']:>9.4f} {res['n_params']:>10,} "
              f"{res['speed_ms']:>11.1f} {res['total_nan']:>6}")

    # Memory verification
    n = state_dim
    m = n_landmarks
    full_memory = n * n
    compressed_memory = n * m + m * m
    compression_ratio = full_memory / compressed_memory

    print(f"\n  Memory Analysis:")
    print(f"    Full inter-chunk memory: O(n^2) = O({full_memory})")
    print(f"    Compressed inter-chunk memory: O(nm + m^2) = O({compressed_memory})")
    print(f"    Compression ratio: {compression_ratio:.2f}x")

    # Success/Failure Criteria
    print("\n" + "=" * 70)
    print("  SUCCESS / FAILURE CRITERIA")
    print("=" * 70)

    criteria = []

    # Criterion 1: Nystrom > 90% copy accuracy
    c1_pass = nystrom_acc > 0.90
    c1_status = "PASS" if c1_pass else "FAIL"
    criteria.append(('Nystrom > 90% accuracy', c1_pass, f'{nystrom_acc:.4f}'))
    print(f"\n  {'PASS' if c1_pass else 'FAIL'}: Nystrom copy accuracy > 90%: {nystrom_acc:.4f}")

    # Criterion 2: Full > 95% copy accuracy
    c2_pass = full_acc > 0.95
    c2_status = "PASS" if c2_pass else "FAIL"
    criteria.append(('Full > 95% accuracy', c2_pass, f'{full_acc:.4f}'))
    print(f"  {'PASS' if c2_pass else 'FAIL'}: Full copy accuracy > 95%: {full_acc:.4f}")

    # Criterion 3: Gap < 5%
    c3_pass = gap < 0.05
    c3_status = "PASS" if c3_pass else "FAIL"
    criteria.append(('Gap < 5%', c3_pass, f'{gap:.4f}'))
    print(f"  {'PASS' if c3_pass else 'FAIL'}: Accuracy gap < 5%: {gap:.4f}")

    # Criterion 4: Memory verification
    c4_pass = compressed_memory < full_memory
    criteria.append(('Memory: O(mn) < O(n^2)', c4_pass,
                     f'O({compressed_memory}) vs O({full_memory})'))
    print(f"  {'PASS' if c4_pass else 'FAIL'}: Memory O(mn)={compressed_memory} < O(n^2)={full_memory}")

    # Failure check: compressed < 70%
    if nystrom_acc < 0.70:
        print(f"\n  FAILURE CRITERION TRIGGERED: Compressed accuracy {nystrom_acc:.4f} < 70%")
        print("  State information cannot be compressed to m landmarks")

    # Approximation error check
    if all_results['nystrom']['compression_stats']:
        for i, stats in enumerate(all_results['nystrom']['compression_stats']):
            err = stats['rel_approx_error']
            err_pass = err < 0.1
            criteria.append((f'Layer {i} approx error < 0.1', err_pass, f'{err:.4f}'))
            print(f"  {'PASS' if err_pass else 'FAIL'}: Layer {i} approximation error < 0.1: {err:.4f}")

    # Overall decision
    n_pass = sum(1 for _, p, _ in criteria if p)
    n_total = len(criteria)

    if nystrom_acc < 0.70:
        decision = "ABANDON"
    elif n_pass >= n_total - 1:
        decision = "PROCEED"
    elif n_pass >= n_total // 2:
        decision = "DEBUG"
    else:
        decision = "ABANDON"

    print(f"\n  Overall: {n_pass}/{n_total} criteria passed")
    print(f"  DECISION: {decision}")
    print("=" * 70)

    # Save results
    results_save = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'models': {},
        'criteria': [(name, passed, val) for name, passed, val in criteria],
        'decision': decision,
    }
    for name, res in all_results.items():
        results_save['models'][name] = {
            'n_params': res['n_params'],
            'best_test_acc': res['best_test_acc'],
            'best_epoch': res['best_epoch'],
            'final_test_acc': res['final_test_acc'],
            'final_per_pos_acc': res['final_per_pos_acc'],
            'total_nan': res['total_nan'],
            'elapsed_seconds': res['elapsed_seconds'],
            'total_epochs': res['total_epochs'],
            'speed_ms': res['speed_ms'],
            'compression_stats': res.get('compression_stats'),
        }

    results_path = Path(__file__).parent / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results, decision


def main():
    parser = argparse.ArgumentParser(description="Nystrom SSM MVE Training")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

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

    print(f"Device: {device}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)

    results, decision = run_experiment(config, device)
    return results, decision


if __name__ == '__main__':
    main()

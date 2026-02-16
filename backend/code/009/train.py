"""
Training script for Post-Sigmoid Gating Linear Attention MVE (Proposal 009).

Compares cosFormer with and without post-readout sigmoid gating on MQAR task.

Success criteria:
  1. Gated cosFormer > 75% accuracy on MQAR with 4 KV pairs at d_k=16
  2. Ungated cosFormer < 55% accuracy on the same task
  3. Improvement persists across 3 random seeds (not lucky initialization)
  4. Training is stable (no NaN/Inf) and wall-clock time increases < 5%

Failure criteria:
  - Gated and ungated within 3% of each other -> gate doesn't help
  - Gate causes training instability (NaN/Inf or loss divergence)
  - Gate adds > 10% wall-clock overhead

Usage:
    python train.py                        # Run with defaults
    python train.py --config config.yaml   # Use config file
    python train.py --seed 42 --n_seeds 1  # Quick single-seed test
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml

from models.cosformer import CosFormerModel
from data.generate import create_mqar_datasets


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
) -> tuple[float, float, int]:
    """Train for one epoch.

    Returns:
        avg_loss, accuracy (on answer positions only), nan_count
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (B, T, vocab_size)

        # Reshape for cross-entropy: (B*T, vocab) vs (B*T,)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), targets.view(B * T))

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break

        if has_nan_grad:
            nan_count += 1
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * B

        # Accuracy on answer positions only (where target != -100)
        mask = targets != -100
        if mask.any():
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, nan_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on dataset.

    Returns:
        avg_loss, accuracy (on answer positions only)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), targets.view(B * T))

        total_loss += loss.item() * B

        mask = targets != -100
        if mask.any():
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ============================================================
# Gate Statistics
# ============================================================

@torch.no_grad()
def compute_gate_stats(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Compute gate activation statistics.

    Monitors gate health: healthy gates should have bimodal distribution
    (near 0 and near 1). Uniform ~0.5 means no selectivity.
    """
    model.eval()
    if not model.use_gate:
        return {}

    all_gate_values = []

    # Hook to capture gate activations
    gate_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Compute gate values from the input x
            x = input[0]  # (B, T, d_model)
            B, T, D = x.shape
            gate_vals = torch.sigmoid(
                module.W_gate(x).view(B, T, module.n_heads, module.d_k)
            )
            gate_activations[layer_idx] = gate_vals.cpu()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.blocks):
        if block.attn.use_gate:
            h = block.attn.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Run one batch through
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    _ = model(inputs)

    for h in hooks:
        h.remove()

    stats = {}
    for layer_idx, gate_vals in gate_activations.items():
        vals = gate_vals.flatten()
        stats[f'layer_{layer_idx}_gate_mean'] = vals.mean().item()
        stats[f'layer_{layer_idx}_gate_std'] = vals.std().item()
        stats[f'layer_{layer_idx}_gate_min'] = vals.min().item()
        stats[f'layer_{layer_idx}_gate_max'] = vals.max().item()
        stats[f'layer_{layer_idx}_gate_below_0.1'] = (vals < 0.1).float().mean().item()
        stats[f'layer_{layer_idx}_gate_above_0.9'] = (vals > 0.9).float().mean().item()

    return stats


# ============================================================
# Wall-Clock Timing
# ============================================================

def time_forward_pass(model: nn.Module, inputs: torch.Tensor, device: torch.device,
                      n_warmup: int = 5, n_runs: int = 20) -> float:
    """Time the forward pass for overhead measurement.

    Returns:
        Average forward pass time in milliseconds.
    """
    model.eval()
    inputs = inputs.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inputs)

    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)


# ============================================================
# Single Run
# ============================================================

def run_single(
    use_gate: bool,
    config: dict,
    device: torch.device,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run training for a single model configuration.

    Args:
        use_gate: Whether to use post-readout sigmoid gating
        config: Configuration dictionary
        device: Torch device
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with training metrics and results
    """
    torch.manual_seed(seed)

    # Data config
    data_cfg = config.get('data', {})
    n_train = data_cfg.get('n_train', 10000)
    n_test = data_cfg.get('n_test', 2000)
    n_kv_pairs = data_cfg.get('n_kv_pairs', 4)
    n_queries = data_cfg.get('n_queries', 2)
    vocab_size = data_cfg.get('vocab_size', 16)

    train_ds, test_ds, total_vocab = create_mqar_datasets(
        n_train=n_train,
        n_test=n_test,
        n_kv_pairs=n_kv_pairs,
        n_queries=n_queries,
        vocab_size=vocab_size,
        seed=seed,
    )

    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == 'cuda'))

    # Model config
    model_cfg = config.get('model', {})
    seq_len = 2 * n_kv_pairs + 1 + 2 * n_queries  # KV pairs + SEP + queries

    model = CosFormerModel(
        vocab_size=total_vocab,
        d_model=model_cfg.get('d_model', 64),
        n_heads=model_cfg.get('n_heads', 4),
        d_k=model_cfg.get('d_k', 16),
        n_layers=model_cfg.get('n_layers', 2),
        max_seq_len=seq_len + 8,  # small buffer
        use_gate=use_gate,
        dropout=model_cfg.get('dropout', 0.0),
        ffn_mult=model_cfg.get('ffn_mult', 4),
    ).to(device)

    n_params = model.count_parameters()
    gate_label = "gated" if use_gate else "ungated"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Model: cosFormer ({gate_label})")
        print(f"  Parameters: {n_params:,}")
        print(f"  d_model={model_cfg.get('d_model', 64)}, "
              f"n_heads={model_cfg.get('n_heads', 4)}, "
              f"d_k={model_cfg.get('d_k', 16)}, "
              f"n_layers={model_cfg.get('n_layers', 2)}")
        print(f"  Seq length: {seq_len}, KV pairs: {n_kv_pairs}, Queries: {n_queries}")
        print(f"  Seed: {seed}, Device: {device}")
        print(f"{'='*60}")

    # Optimizer & Loss
    lr = float(train_cfg.get('lr', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    max_epochs = train_cfg.get('max_epochs', 200)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    best_test_acc = 0.0
    best_epoch = 0
    train_history = []
    patience = train_cfg.get('patience', 30)
    patience_counter = 0
    early_stop_acc = train_cfg.get('early_stop_acc', 0.99)
    total_nan_count = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=train_cfg.get('gradient_clip', 1.0),
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        total_nan_count += nan_count

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'nan_count': nan_count,
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
                  f"Best={best_test_acc:.3f}"
                  + (f" | NaN={nan_count}" if nan_count > 0 else ""))

        # Early stopping
        if best_test_acc >= early_stop_acc:
            if verbose:
                print(f"  Early stop: test acc {best_test_acc:.3f} >= {early_stop_acc}")
            break
        if patience_counter >= patience:
            if verbose:
                print(f"  Patience exhausted at epoch {epoch+1}")
            break

        # Abort on excessive NaN
        if total_nan_count > 50:
            if verbose:
                print(f"  ABORT: Too many NaN events ({total_nan_count})")
            break

    elapsed = time.time() - start_time

    # Gate statistics (for gated model)
    gate_stats = {}
    if use_gate:
        gate_stats = compute_gate_stats(model, test_loader, device)

    # Wall-clock timing
    sample_input = next(iter(test_loader))[0][:16]  # Small batch for timing
    fwd_time_ms = time_forward_pass(model, sample_input, device)

    results = {
        'use_gate': use_gate,
        'gate_label': gate_label,
        'n_params': n_params,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_history[-1]['train_acc'],
        'final_test_acc': train_history[-1]['test_acc'],
        'final_train_loss': train_history[-1]['train_loss'],
        'final_test_loss': train_history[-1]['test_loss'],
        'total_epochs': len(train_history),
        'elapsed_seconds': elapsed,
        'total_nan_count': total_nan_count,
        'fwd_time_ms': fwd_time_ms,
        'gate_stats': gate_stats,
        'history': train_history,
    }

    if verbose:
        print(f"\n  Results ({gate_label}):")
        print(f"    Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})")
        print(f"    Final train accuracy: {train_history[-1]['train_acc']:.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        print(f"    Forward pass: {fwd_time_ms:.2f} ms")
        print(f"    NaN events: {total_nan_count}")
        if gate_stats:
            for k, v in gate_stats.items():
                print(f"    {k}: {v:.4f}")

    return results


# ============================================================
# Main: Run both gated and ungated
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Post-Sigmoid Gating Linear Attention MVE")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of random seeds to average over')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
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

    # Multiple seeds for statistical rigor
    seeds = [args.seed + i * 100 for i in range(args.n_seeds)]

    all_results = {}

    for use_gate in [False, True]:
        gate_label = "gated" if use_gate else "ungated"
        print(f"\n{'#'*60}")
        print(f"  Running cosFormer ({gate_label}) — {args.n_seeds} seeds")
        print(f"{'#'*60}")

        seed_results = []
        for seed in seeds:
            result = run_single(use_gate, config, device, seed=seed, verbose=True)
            seed_results.append(result)

        # Aggregate over seeds
        best_accs = [r['best_test_acc'] for r in seed_results]
        mean_acc = sum(best_accs) / len(best_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in best_accs) / len(best_accs)) ** 0.5
        mean_time = sum(r['elapsed_seconds'] for r in seed_results) / len(seed_results)
        mean_fwd_ms = sum(r['fwd_time_ms'] for r in seed_results) / len(seed_results)
        total_nans = sum(r['total_nan_count'] for r in seed_results)

        all_results[gate_label] = {
            'mean_best_acc': mean_acc,
            'std_best_acc': std_acc,
            'best_accs': best_accs,
            'mean_time': mean_time,
            'mean_fwd_ms': mean_fwd_ms,
            'total_nans': total_nans,
            'n_params': seed_results[0]['n_params'],
            'seed_results': seed_results,
        }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 009 RESULTS: Post-Sigmoid Gating for Linear Attention")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Accuracy':>12} {'Std':>8} {'Fwd (ms)':>10} "
          f"{'Time (s)':>10} {'Params':>8} {'NaN':>6}")
    print("-" * 80)

    for label, res in all_results.items():
        print(f"cosFormer ({label}){'':<{25 - len(f'cosFormer ({label})')}} "
              f"{res['mean_best_acc']:>11.4f} "
              f"{res['std_best_acc']:>7.4f} "
              f"{res['mean_fwd_ms']:>9.2f} "
              f"{res['mean_time']:>9.1f} "
              f"{res['n_params']:>8d} "
              f"{res['total_nans']:>5d}")

    # ============================================================
    # Success/Failure Criteria
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUCCESS / FAILURE CRITERIA")
    print("=" * 70)

    gated = all_results.get('gated', {})
    ungated = all_results.get('ungated', {})

    gated_acc = gated.get('mean_best_acc', 0)
    ungated_acc = ungated.get('mean_best_acc', 0)
    gated_std = gated.get('std_best_acc', 0)
    ungated_std = ungated.get('std_best_acc', 0)
    gated_fwd = gated.get('mean_fwd_ms', 0)
    ungated_fwd = ungated.get('mean_fwd_ms', 0)
    gated_nans = gated.get('total_nans', 0)

    print(f"\n  Gated accuracy:   {gated_acc:.4f} +/- {gated_std:.4f}")
    print(f"  Ungated accuracy: {ungated_acc:.4f} +/- {ungated_std:.4f}")
    print(f"  Improvement:      {(gated_acc - ungated_acc)*100:.1f} percentage points")

    # Criterion 1: Gated > 75%
    c1 = gated_acc > 0.75
    print(f"\n  {'✅' if c1 else '❌'} Criterion 1: Gated > 75% accuracy"
          f" — {gated_acc:.4f} {'> 0.75' if c1 else '< 0.75'}")

    # Criterion 2: Ungated < 55%
    c2 = ungated_acc < 0.55
    print(f"  {'✅' if c2 else '❌'} Criterion 2: Ungated < 55% accuracy"
          f" — {ungated_acc:.4f} {'< 0.55' if c2 else '>= 0.55'}")

    # Criterion 3: Persists across seeds
    gated_accs = gated.get('best_accs', [])
    if len(gated_accs) >= 3:
        all_above_threshold = all(a > ungated_acc + 0.05 for a in gated_accs)
        c3 = all_above_threshold
        print(f"  {'✅' if c3 else '❌'} Criterion 3: Persists across seeds"
              f" — accs: {[f'{a:.3f}' for a in gated_accs]}")
    else:
        c3 = False
        print(f"  ⚠️  Criterion 3: Not enough seeds to evaluate")

    # Criterion 4: Training stable (no NaN) and < 5% wall-clock overhead
    c4_stable = gated_nans == 0
    if ungated_fwd > 0:
        overhead = (gated_fwd - ungated_fwd) / ungated_fwd * 100
    else:
        overhead = 0
    c4_fast = overhead < 5.0
    c4 = c4_stable and c4_fast
    print(f"  {'✅' if c4 else '❌'} Criterion 4: Stable + < 5% overhead"
          f" — NaN={gated_nans}, overhead={overhead:.1f}%")

    # Failure criteria
    print(f"\n  Failure checks:")
    f1 = abs(gated_acc - ungated_acc) < 0.03
    print(f"  {'❌ FAIL' if f1 else '✅ OK'} Within 3%: "
          f"|{gated_acc:.4f} - {ungated_acc:.4f}| = {abs(gated_acc - ungated_acc):.4f}")

    f2 = gated_nans > 0
    print(f"  {'❌ FAIL' if f2 else '✅ OK'} Training instability: {gated_nans} NaN events")

    f3 = overhead > 10.0
    print(f"  {'❌ FAIL' if f3 else '✅ OK'} Overhead > 10%: {overhead:.1f}%")

    # Overall decision
    n_success = sum([c1, c2, c3, c4])
    any_failure = f1 or f2

    if n_success >= 3 and not any_failure:
        decision = "PROCEED"
    elif any_failure:
        decision = "KILL" if f1 else "DEBUG"
    elif n_success >= 2:
        decision = "PROCEED (soft)"
    else:
        decision = "DEBUG"

    print(f"\n  DECISION: {decision} ({n_success}/4 criteria met)")
    print("=" * 70)

    # Gate statistics summary
    if gated.get('seed_results'):
        print("\n  Gate Statistics (last seed):")
        last_gate_stats = gated['seed_results'][-1].get('gate_stats', {})
        for k, v in last_gate_stats.items():
            print(f"    {k}: {v:.4f}")

    # Save results
    results_path = Path(__file__).parent / 'results.json'
    save_results = {
        'gated': {
            'mean_best_acc': gated_acc,
            'std_best_acc': gated_std,
            'best_accs': gated.get('best_accs', []),
            'mean_fwd_ms': gated_fwd,
            'mean_time': gated.get('mean_time', 0),
            'n_params': gated.get('n_params', 0),
            'total_nans': gated_nans,
        },
        'ungated': {
            'mean_best_acc': ungated_acc,
            'std_best_acc': ungated_std,
            'best_accs': ungated.get('best_accs', []),
            'mean_fwd_ms': ungated_fwd,
            'mean_time': ungated.get('mean_time', 0),
            'n_params': ungated.get('n_params', 0),
            'total_nans': ungated.get('total_nans', 0),
        },
        'criteria': {
            'c1_gated_above_75': c1,
            'c2_ungated_below_55': c2,
            'c3_persists_across_seeds': c3,
            'c4_stable_and_fast': c4,
        },
        'decision': decision,
        'overhead_pct': overhead,
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results, decision


if __name__ == '__main__':
    main()

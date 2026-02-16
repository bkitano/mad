"""
Training script for MVE 031: V:N:M Sparse SSM Projections with S-STE Training

Runs MQAR (Multi-Query Associative Recall) task with multiple sparsity configurations:
  1. Dense (baseline) — quality upper bound
  2. 2:4 Sparse (50%) — shows marginal improvement of 2:4 alone
  3. V:2:6 (67%) — intermediate VNM sparsity
  4. V:2:8 (75%) — target VNM sparsity
  5. Iso-parameter Dense — smaller d_model matching VNM param count

Success criteria (from proposal):
  - Dense Mamba-2 > 95% accuracy on MQAR at 4 KV pairs
  - 2:4 sparse > 90% accuracy
  - VNM 75% > 80% accuracy
  - Iso-parameter dense < 70% accuracy (sparse > small-dense)
  - S-STE mask flip rate converges within 500 training steps

Failure criteria:
  - Kill if VNM 75% < 50% accuracy
  - Kill if VNM performs worse than iso-parameter dense
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import time
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models import GatedSSMModel
from data import generate_mqar_dataset


def train_epoch(model, loader, optimizer, criterion, device, track_masks=False):
    """Train for one epoch. Returns (loss, accuracy, mask_flip_rates)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    mask_flip_rates = {}

    for tokens, targets, query_pos in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tokens)  # (B, T, vocab_size)

        # Flatten for loss computation
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        # Only compute loss at query positions (where target != -1)
        valid_mask = targets_flat != -1
        if valid_mask.sum() == 0:
            continue

        loss = criterion(logits_flat[valid_mask], targets_flat[valid_mask])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Accuracy at query positions
        preds = logits_flat.argmax(dim=-1)
        correct += (preds[valid_mask] == targets_flat[valid_mask]).sum().item()
        total += valid_mask.sum().item()

    # Track mask flip rates periodically
    if track_masks:
        mask_flip_rates = model.get_all_mask_flip_rates()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, mask_flip_rates


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for tokens, targets, query_pos in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            logits = model(tokens)
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = targets.view(-1)

            valid_mask = targets_flat != -1
            if valid_mask.sum() == 0:
                continue

            loss = criterion(logits_flat[valid_mask], targets_flat[valid_mask])
            total_loss += loss.item()
            num_batches += 1

            preds = logits_flat.argmax(dim=-1)
            correct += (preds[valid_mask] == targets_flat[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def compute_iso_param_d_model(target_params: int, config: dict) -> int:
    """
    Compute d_model for iso-parameter dense model.

    The projection params scale as ~5 * d_model * d_inner + embedding + head.
    We solve for d_model that gives approximately target_params total.
    """
    n_layers = config['model']['n_layers']
    n_heads = config['model']['n_heads']
    d_head = config['model']['d_head']
    vocab_size = config['data']['vocab_size']
    state_dim = config['model']['state_dim']

    # Rough formula: params ≈ vocab*d + n_layers*(5*d*d_inner + d*state_dim) + d*vocab
    # where d_inner = n_heads * d_head
    # For iso-param, we want to find d_model such that total params ≈ target_params
    # with d_inner = n_heads * d_head (keeping d_head and n_heads fixed makes it too constrained)
    # Instead: for iso-param, just scale d_model down, keeping d_inner = d_model
    # This is a simplification

    # Approximate: params ≈ 2*vocab*d + n_layers * 5 * d^2
    # Solve: target = 2*v*d + 5*n*d^2
    # 5*n*d^2 + 2*v*d - target = 0
    a = 5 * n_layers
    b = 2 * vocab_size
    c = -target_params

    discriminant = b**2 - 4*a*c
    d_model = int((-b + math.sqrt(discriminant)) / (2 * a))

    # Round to nearest multiple of n_heads
    d_model = max(n_heads, (d_model // n_heads) * n_heads)

    return d_model


def run_single_config(config: dict, sparsity_name: str, vnm_M: int, d_model_override: int = None,
                      device: str = 'cpu', seed: int = 42):
    """
    Run a single training configuration.

    Returns:
        dict with metrics and results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_loader, val_loader, test_loader = generate_mqar_dataset(
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        num_kv_pairs=config['data']['num_kv_pairs'],
        seq_len=config['data']['seq_len'],
        vocab_size=config['data']['vocab_size'],
        batch_size=config['training']['batch_size'],
        seed=seed,
    )

    # Model
    d_model = d_model_override or config['model']['d_model']
    d_head = config['model']['d_head']
    n_heads = config['model']['n_heads']

    # Adjust d_head for iso-parameter model if d_model is smaller
    if d_model_override and d_model_override < n_heads * d_head:
        d_head = d_model_override // n_heads

    model = GatedSSMModel(
        vocab_size=config['data']['vocab_size'],
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_layers=config['model']['n_layers'],
        state_dim=config['model']['state_dim'],
        vnm_M=vnm_M,
        use_gate=config['model'].get('use_gate', True),
        dropout=config['model'].get('dropout', 0.1),
    ).to(device)

    param_info = model.count_parameters()
    print(f"\n{'='*70}")
    print(f"Configuration: {sparsity_name}")
    print(f"  d_model={d_model}, d_head={d_head}, n_heads={n_heads}")
    print(f"  VNM M={vnm_M}")
    print(f"  Total params: {param_info['total']:,}")
    print(f"  Effective params: {param_info['effective']:,}")
    print(f"  Sparsity: {param_info['sparsity']:.2%}")
    print(f"{'='*70}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    criterion = nn.CrossEntropyLoss()

    # Training
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    mask_flip_history = []

    start_time = time.time()

    for epoch in range(config['training']['epochs']):
        # Track masks every 50 steps in first 500 steps, then every 200 steps
        track_masks = (epoch < 10 or epoch % 4 == 0) and vnm_M > 0

        train_loss, train_acc, mask_flips = train_epoch(
            model, train_loader, optimizer, criterion, device, track_masks=track_masks
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if mask_flips:
            mask_flip_history.append({
                'epoch': epoch,
                'rates': mask_flips,
                'mean_rate': np.mean(list(mask_flips.values())) if mask_flips else 0,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            mask_info = ""
            if mask_flips:
                mean_rate = np.mean(list(mask_flips.values()))
                mask_info = f", mask_flip={mean_rate:.4f}"
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}{mask_info}")

        # Early stopping if we reach high accuracy
        if val_acc >= 0.99:
            print(f"  Early stop at epoch {epoch+1} with val_acc={val_acc:.4f}")
            break

    elapsed = time.time() - start_time

    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Get final sparsity stats
    sparsity_stats = model.get_all_sparsities()

    print(f"\n  Final test accuracy: {test_acc:.4f}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    if sparsity_stats:
        mean_sparsity = np.mean(list(sparsity_stats.values()))
        print(f"  Mean sparsity: {mean_sparsity:.2%}")

    # Check mask convergence
    mask_converged = True
    if mask_flip_history and len(mask_flip_history) >= 2:
        # Check if mask flip rate decreased over training
        early_rates = [h['mean_rate'] for h in mask_flip_history[:3]]
        late_rates = [h['mean_rate'] for h in mask_flip_history[-3:]]
        early_mean = np.mean(early_rates) if early_rates else 0
        late_mean = np.mean(late_rates) if late_rates else 0
        mask_converged = late_mean < early_mean * 0.5 or late_mean < 0.01

    return {
        'name': sparsity_name,
        'vnm_M': vnm_M,
        'd_model': d_model,
        'd_head': d_head,
        'total_params': param_info['total'],
        'effective_params': param_info['effective'],
        'overall_sparsity': param_info['sparsity'],
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'train_time': elapsed,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'sparsity_stats': sparsity_stats,
        'mask_flip_history': mask_flip_history,
        'mask_converged': mask_converged,
        'epochs_run': len(train_losses),
    }


def main():
    parser = argparse.ArgumentParser(description="MVE 031: VNM Sparse SSM Projections")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single", type=str, default=None,
                        help="Run only a single config: dense, 2_4, v2_6, v2_8, iso_param")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("MVE 031: V:N:M Sparse SSM Projections with S-STE Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"MQAR: {config['data']['num_kv_pairs']} KV pairs, "
          f"seq_len={config['data']['seq_len']}, vocab={config['data']['vocab_size']}")
    print()

    # Define sparsity configurations to test
    configs = {
        'dense':     {'name': 'Dense (baseline)',     'vnm_M': 0,  'd_model': None},
        '2_4':       {'name': '2:4 Sparse (50%)',     'vnm_M': 4,  'd_model': None},
        'v2_6':      {'name': 'V:2:6 Sparse (67%)',   'vnm_M': 6,  'd_model': None},
        'v2_8':      {'name': 'V:2:8 Sparse (75%)',   'vnm_M': 8,  'd_model': None},
    }

    if args.single:
        if args.single not in configs and args.single != 'iso_param':
            print(f"Unknown config: {args.single}. Available: {list(configs.keys()) + ['iso_param']}")
            return
        if args.single == 'iso_param':
            configs = {'iso_param': None}  # Will be set after VNM run
        else:
            configs = {args.single: configs[args.single]}

    # Run all configurations
    all_results = {}
    seed = config.get('seed', 42)

    for config_key, cfg in configs.items():
        if config_key == 'iso_param':
            continue  # Handle after VNM runs
        result = run_single_config(
            config, cfg['name'], cfg['vnm_M'],
            d_model_override=cfg['d_model'],
            device=args.device, seed=seed,
        )
        all_results[config_key] = result

    # Iso-parameter dense baseline:
    # Find effective params of VNM 75% and create a dense model with same param count
    if 'v2_8' in all_results and ('iso_param' not in configs or args.single is None):
        vnm_effective = all_results['v2_8']['effective_params']
        iso_d_model = compute_iso_param_d_model(vnm_effective, config)

        print(f"\nIso-parameter dense: d_model={iso_d_model} "
              f"(matching VNM 75% effective params: {vnm_effective:,})")

        result = run_single_config(
            config, f'Iso-param Dense (d={iso_d_model})', 0,
            d_model_override=iso_d_model,
            device=args.device, seed=seed,
        )
        all_results['iso_param'] = result

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: MVE 031 Results")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Params':>10} {'Eff.Params':>12} {'Sparsity':>10} "
          f"{'Test Acc':>10} {'Val Acc':>10} {'Time':>8}")
    print("-" * 100)

    for key, result in all_results.items():
        print(f"{result['name']:<30} {result['total_params']:>10,} {result['effective_params']:>12,} "
              f"{result['overall_sparsity']:>9.1%} {result['test_acc']:>10.4f} "
              f"{result['best_val_acc']:>10.4f} {result['train_time']:>7.1f}s")

    # Check success/failure criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    criteria = {}

    if 'dense' in all_results:
        dense_acc = all_results['dense']['test_acc']
        c1_pass = dense_acc > 0.95
        criteria['dense_>95%'] = c1_pass
        print(f"1. Dense > 95% accuracy: {dense_acc:.4f} - {'✅ PASS' if c1_pass else '❌ FAIL'}")

    if '2_4' in all_results:
        two_four_acc = all_results['2_4']['test_acc']
        c2_pass = two_four_acc > 0.90
        criteria['2:4_>90%'] = c2_pass
        print(f"2. 2:4 Sparse > 90% accuracy: {two_four_acc:.4f} - {'✅ PASS' if c2_pass else '❌ FAIL'}")

    if 'v2_8' in all_results:
        vnm_acc = all_results['v2_8']['test_acc']
        c3_pass = vnm_acc > 0.80
        criteria['VNM_75%_>80%'] = c3_pass
        print(f"3. VNM 75% > 80% accuracy: {vnm_acc:.4f} - {'✅ PASS' if c3_pass else '❌ FAIL'}")

    if 'iso_param' in all_results:
        iso_acc = all_results['iso_param']['test_acc']
        c4_pass = iso_acc < 0.70
        criteria['iso_param_<70%'] = c4_pass
        print(f"4. Iso-param dense < 70% accuracy: {iso_acc:.4f} - {'✅ PASS' if c4_pass else '❌ FAIL'}")

    if 'v2_8' in all_results:
        mc = all_results['v2_8']['mask_converged']
        criteria['mask_converge'] = mc
        print(f"5. S-STE mask flip rate converges: {'✅ PASS' if mc else '❌ FAIL'}")

    # Failure criteria
    print("\nFAILURE CRITERIA CHECK:")
    kill = False

    if 'v2_8' in all_results:
        vnm_acc = all_results['v2_8']['test_acc']
        if vnm_acc < 0.50:
            print(f"❌ KILL: VNM 75% accuracy {vnm_acc:.4f} < 50%")
            kill = True
        else:
            print(f"✅ OK: VNM 75% accuracy {vnm_acc:.4f} >= 50%")

    if 'v2_8' in all_results and 'iso_param' in all_results:
        vnm_acc = all_results['v2_8']['test_acc']
        iso_acc = all_results['iso_param']['test_acc']
        if vnm_acc < iso_acc:
            print(f"❌ KILL: VNM ({vnm_acc:.4f}) worse than iso-param ({iso_acc:.4f})")
            kill = True
        else:
            print(f"✅ OK: VNM ({vnm_acc:.4f}) >= iso-param ({iso_acc:.4f})")

    if '2_4' in all_results and 'v2_8' in all_results:
        two_four_acc = all_results['2_4']['test_acc']
        vnm_acc = all_results['v2_8']['test_acc']
        if abs(two_four_acc - vnm_acc) < 0.02:
            print(f"⚠️  INVESTIGATE: 2:4 ({two_four_acc:.4f}) ≈ VNM ({vnm_acc:.4f}) — "
                  f"column pruning adds no benefit")

    # Overall verdict
    print("\n" + "=" * 80)
    all_pass = all(criteria.values()) if criteria else False
    any_kill = kill

    if any_kill:
        verdict = "ABANDON"
        print("🚫 OVERALL: ABANDON — Kill criteria triggered")
    elif all_pass:
        verdict = "PROCEED"
        print("🎉 OVERALL: PROCEED — All success criteria met")
    else:
        verdict = "DEBUG"
        print("⚙️  OVERALL: DEBUG — Some criteria not met, investigate further")

    print("=" * 80)

    # Save results
    results = {
        'config': config,
        'device': args.device,
        'results': {k: {
            'name': v['name'],
            'vnm_M': v['vnm_M'],
            'd_model': v['d_model'],
            'd_head': v['d_head'],
            'total_params': v['total_params'],
            'effective_params': v['effective_params'],
            'overall_sparsity': float(v['overall_sparsity']),
            'test_loss': float(v['test_loss']),
            'test_acc': float(v['test_acc']),
            'best_val_acc': float(v['best_val_acc']),
            'train_time': float(v['train_time']),
            'mask_converged': bool(v['mask_converged']),
            'epochs_run': v['epochs_run'],
        } for k, v in all_results.items()},
        'criteria': {k: bool(v) for k, v in criteria.items()},
        'verdict': verdict,
    }

    with open("results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print("\nResults saved to results.yaml")


if __name__ == "__main__":
    main()

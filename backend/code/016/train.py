"""
Training script for MVE 016: GS-Monomial SSM on S5 Permutation Composition

Trains 3 models:
1. GS-Monomial SSM (with shuffle) — full model
2. GS-Monomial SSM (no shuffle) — ablation: block-diagonal only
3. Diagonal SSM — baseline: provably unable to represent S5

Success criteria (from proposal):
1. GS-Monomial > 90% accuracy on S5 where Diagonal < 30%
2. GS-Monomial matches or exceeds column-sparse PD-SSM baseline (> 85%)
3. Cross-block mixing benefit: GS-Monomial > Block-Diagonal-Only by > 10%

Failure criteria:
1. If accuracy < 50% → learning dynamics broken
2. If removing P_shuffle makes < 5% difference → cross-block mixing useless
"""

import argparse
import os
import sys
import time
import json

# Force unbuffered output for Modal logging
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate import create_dataloaders, IGNORE_INDEX
from models.gs_monomial_ssm import GSMonomialSSM
from models.diagonal_ssm import DiagonalSSM


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    for tokens, targets, masks in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(tokens, masks)  # (batch, seq_len, num_classes)

        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN
        if torch.isnan(loss):
            nan_count += 1
            continue

        loss.backward()

        # Check for NaN in gradients
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                nan_count += 1
                break

        if has_nan_grad:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size

        # Accuracy on non-ignored positions
        preds = logits_flat.argmax(dim=-1)
        valid = targets_flat != IGNORE_INDEX
        correct += ((preds == targets_flat) & valid).sum().item()
        total += valid.sum().item()

    avg_loss = total_loss / max(total / seq_len, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, nan_count


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, targets, masks in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            logits = model(tokens, masks)
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            targets_flat = targets.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size

            preds = logits_flat.argmax(dim=-1)
            valid = targets_flat != IGNORE_INDEX
            correct += ((preds == targets_flat) & valid).sum().item()
            total += valid.sum().item()

    avg_loss = total_loss / max(total / seq_len, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def speed_benchmark(model, device, seq_len=22, d_model=32, num_iters=50):
    """Benchmark forward pass speed."""
    model.eval()
    batch = 16
    tokens = torch.randint(0, 5, (batch, seq_len), device=device)
    mask = torch.ones(batch, seq_len, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(tokens, mask)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            model(tokens, mask)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_iters * 1000  # ms

    return elapsed


def train_model(model, model_name, train_loader, val_loader, config, device):
    """
    Train a single model with early stopping.

    Returns dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {count_params(model):,}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['max_epochs'],
        eta_min=config['lr'] * 0.01,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience = config.get('patience', 30)
    patience_counter = 0
    total_nan = 0

    for epoch in range(1, config['max_epochs'] + 1):
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=config.get('max_grad_norm', 1.0),
        )
        total_nan += nan_count

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or val_acc > 0.8:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f} best_val={best_val_acc:.4f} nan={nan_count}", flush=True)

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
            break

        # Failure: too many NaN
        if total_nan > 100:
            print(f"  ABORT: {total_nan} NaN events — training unstable")
            break

    return {
        'model_name': model_name,
        'params': count_params(model),
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_loss': train_loss,
        'final_train_acc': train_acc,
        'total_nan': total_nan,
        'model': model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    print(f"Config: {json.dumps(config, indent=2)}", flush=True)

    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    if device == 'cuda':
        torch.cuda.manual_seed(config.get('seed', 42))

    # ---- Create data ----
    print("Generating data...", flush=True)
    t0 = time.time()
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['training']

    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        num_train=data_cfg['num_train'],
        num_val=data_cfg.get('num_val', 1000),
        num_test=data_cfg.get('num_test', 1000),
        seq_len=data_cfg['seq_len'],
        batch_size=train_cfg['batch_size'],
    )

    print(f"Data generated in {time.time()-t0:.1f}s", flush=True)
    print(f"\nDataset info: {dataset_info}", flush=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}", flush=True)

    num_tokens = dataset_info['num_tokens']
    num_classes = dataset_info['num_classes']
    max_seq_len = dataset_info['max_seq_len']

    # ---- Build models ----
    print("Building models...", flush=True)
    models = {}

    # 1. GS-Monomial SSM (with shuffle) — full model
    models['gs_monomial'] = GSMonomialSSM(
        num_tokens=num_tokens,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        d_model=model_cfg['d_model'],
        state_dim=model_cfg['state_dim'],
        block_size=model_cfg['block_size'],
        num_layers=model_cfg['num_layers'],
        sinkhorn_iters=model_cfg.get('sinkhorn_iters', 5),
        tau=model_cfg.get('tau', 0.5),
        dropout=model_cfg.get('dropout', 0.1),
        use_shuffle=True,
    )

    # 2. Block-Diagonal Only (no shuffle) — ablation
    models['block_diag_only'] = GSMonomialSSM(
        num_tokens=num_tokens,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        d_model=model_cfg['d_model'],
        state_dim=model_cfg['state_dim'],
        block_size=model_cfg['block_size'],
        num_layers=model_cfg['num_layers'],
        sinkhorn_iters=model_cfg.get('sinkhorn_iters', 5),
        tau=model_cfg.get('tau', 0.5),
        dropout=model_cfg.get('dropout', 0.1),
        use_shuffle=False,
    )

    # 3. Diagonal SSM — baseline
    models['diagonal'] = DiagonalSSM(
        num_tokens=num_tokens,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        d_model=model_cfg['d_model'],
        state_dim=model_cfg['state_dim'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg.get('dropout', 0.1),
    )

    # ---- Train all models ----
    results = {}
    start_time = time.time()

    for name, model in models.items():
        result = train_model(
            model=model,
            model_name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            device=device,
        )
        results[name] = result

    train_time = time.time() - start_time

    # ---- Evaluate on test set ----
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    test_results = {}

    for name, result in results.items():
        model = result['model']
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_results[name] = test_acc
        print(f"  {name}: test_acc={test_acc:.4f} ({test_acc*100:.1f}%)")

    # ---- Speed benchmark ----
    print(f"\n{'='*60}")
    print("SPEED BENCHMARK")
    print(f"{'='*60}")

    speed_results = {}
    for name, result in results.items():
        model = result['model']
        ms = speed_benchmark(model, device, max_seq_len)
        speed_results[name] = ms
        print(f"  {name}: {ms:.2f} ms/batch")

    # ---- Evaluate success criteria ----
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    gs_acc = test_results['gs_monomial']
    bd_acc = test_results['block_diag_only']
    diag_acc = test_results['diagonal']
    shuffle_benefit = (gs_acc - bd_acc) * 100  # percentage points

    # Criterion 1: GS-Monomial > 90% where Diagonal < 30%
    c1_gs = gs_acc > 0.90
    c1_diag = diag_acc < 0.30
    c1 = c1_gs and c1_diag
    print(f"\n1. GS-Monomial > 90% AND Diagonal < 30%:")
    print(f"   GS-Monomial: {gs_acc*100:.1f}% {'PASS' if c1_gs else 'FAIL'}")
    print(f"   Diagonal:    {diag_acc*100:.1f}% {'PASS' if c1_diag else 'FAIL'}")
    print(f"   Overall: {'PASS' if c1 else 'FAIL'}")

    # Criterion 2: GS-Monomial > 85% (matching PD-SSM level)
    c2 = gs_acc > 0.85
    print(f"\n2. GS-Monomial > 85% (PD-SSM comparable):")
    print(f"   GS-Monomial: {gs_acc*100:.1f}% {'PASS' if c2 else 'FAIL'}")

    # Criterion 3: Cross-block mixing benefit > 10 percentage points
    c3 = shuffle_benefit > 10.0
    print(f"\n3. Cross-block mixing benefit > 10pp:")
    print(f"   GS-Monomial (shuffle):    {gs_acc*100:.1f}%")
    print(f"   Block-Diagonal (no shuf): {bd_acc*100:.1f}%")
    print(f"   Benefit: {shuffle_benefit:.1f}pp {'PASS' if c3 else 'FAIL'}")

    # Failure criteria
    f1 = gs_acc < 0.50
    f2 = abs(shuffle_benefit) < 5.0

    if f1:
        print(f"\n!!! FAILURE CRITERION 1 TRIGGERED: GS-Monomial < 50% ({gs_acc*100:.1f}%) — learning dynamics broken")
    if f2:
        print(f"\n!!! FAILURE CRITERION 2 TRIGGERED: Shuffle benefit < 5pp ({shuffle_benefit:.1f}pp) — cross-block mixing useless")

    # ---- Stability check ----
    total_nan = sum(r['total_nan'] for r in results.values())
    stable = total_nan == 0
    print(f"\nStability: {total_nan} total NaN events {'(STABLE)' if stable else '(UNSTABLE)'}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"| Model              | Params | Val Acc | Test Acc | Speed (ms) |")
    print(f"|{'-'*20}|{'-'*8}|{'-'*9}|{'-'*10}|{'-'*12}|")
    for name in ['gs_monomial', 'block_diag_only', 'diagonal']:
        r = results[name]
        print(f"| {name:18s} | {r['params']:6,} | {r['best_val_acc']*100:6.1f}% | {test_results[name]*100:7.1f}% | {speed_results[name]:10.2f} |")

    print(f"\nTotal training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # ---- Decision ----
    all_pass = c1 and c2 and c3
    any_fail = f1 or f2

    if any_fail:
        decision = "ABANDON"
        reason = "Failure criteria triggered"
    elif all_pass:
        decision = "PROCEED"
        reason = "All success criteria met"
    elif c2:
        decision = "PROCEED"
        reason = f"Core hypothesis validated ({gs_acc*100:.1f}% > 85%); some criteria need investigation"
    else:
        decision = "DEBUG"
        reason = "Core accuracy too low"

    print(f"\nDECISION: {decision}")
    print(f"REASON: {reason}")

    # Save results to file
    output = {
        'test_accuracy': {name: float(acc) for name, acc in test_results.items()},
        'val_accuracy': {name: float(results[name]['best_val_acc']) for name in results},
        'params': {name: results[name]['params'] for name in results},
        'speed_ms': {name: float(speed_results[name]) for name in speed_results},
        'nan_events': {name: results[name]['total_nan'] for name in results},
        'shuffle_benefit_pp': float(shuffle_benefit),
        'criteria': {
            'c1_gs_above_90': bool(c1_gs),
            'c1_diag_below_30': bool(c1_diag),
            'c2_gs_above_85': bool(c2),
            'c3_shuffle_benefit_above_10pp': bool(c3),
            'f1_gs_below_50': bool(f1),
            'f2_shuffle_benefit_below_5pp': bool(f2),
        },
        'decision': decision,
        'reason': reason,
        'total_train_time_s': float(train_time),
        'config': config,
    }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

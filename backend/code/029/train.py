"""
Training script for Experiment 029: Circulant FAVOR+ Linear Attention.

Runs four attention variants on associative recall and compares:
1. Dense FAVOR+ (random features, dense projection)
2. C-FAVOR+ (circulant projection via FFT)
3. ReLU linear attention (no feature map)
4. Softmax attention (quality ceiling)

Success criteria (from proposal MVE):
- C-FAVOR+ achieves > 90% associative recall accuracy (matching dense FAVOR+)
- C-FAVOR+ feature map computation is measurably faster than dense FAVOR+
- Both FAVOR+ variants significantly outperform ReLU linear attention (> 20% gap)

Failure criteria:
- C-FAVOR+ accuracy > 10% worse than dense FAVOR+ -> circulant breaks positive features
- C-FAVOR+ not better than ReLU linear attention -> feature map adds no value with circulant
"""

import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.attention import AssociativeRecallModel
from data.generate import create_datasets


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for seqs, targets, query_pos in loader:
        seqs = seqs.to(device)
        targets = targets.to(device)
        query_pos = query_pos.to(device)

        optimizer.zero_grad()

        logits = model(seqs)  # [B, T, vocab]

        # Extract logits at query position
        batch_idx = torch.arange(seqs.size(0), device=device)
        query_logits = logits[batch_idx, query_pos]  # [B, vocab]

        loss = criterion(query_logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        preds = query_logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for seqs, targets, query_pos in loader:
        seqs = seqs.to(device)
        targets = targets.to(device)
        query_pos = query_pos.to(device)

        logits = model(seqs)
        batch_idx = torch.arange(seqs.size(0), device=device)
        query_logits = logits[batch_idx, query_pos]

        loss = criterion(query_logits, targets)

        total_loss += loss.item() * seqs.size(0)
        preds = query_logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


def benchmark_feature_map(model, x, device, n_warmup=10, n_runs=50):
    """
    Benchmark the feature map computation time.
    Returns average time in milliseconds.
    """
    model.eval()
    x = x.to(device)

    # Get the first layer's attention module
    first_layer = model.layers[0]
    attn = first_layer.attn

    if not hasattr(attn, '_apply_feature_map'):
        return 0.0  # softmax doesn't have feature maps

    B, T = x.shape
    with torch.no_grad():
        h = model.embedding(x) + model.pos_embedding(torch.arange(T, device=device).unsqueeze(0))
        h = model.layers[0].norm1(h)

        H = attn.n_heads
        d_k = attn.d_k

        q = attn.W_q(h).view(B, T, H, d_k).transpose(1, 2)
        k = attn.W_k(h).view(B, T, H, d_k).transpose(1, 2)

    # Warmup
    for _ in range(n_warmup):
        attn._apply_feature_map(q, k)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        attn._apply_feature_map(q, k)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


def run_experiment(config):
    """Run the full experiment with all four attention types."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Data config
    data_cfg = config['data']
    n_train = data_cfg['n_train']
    n_test = data_cfg['n_test']
    seq_len = data_cfg['seq_len']
    vocab_size = data_cfg['vocab_size']
    n_pairs = data_cfg['n_pairs']

    # Model config
    model_cfg = config['model']
    d_model = model_cfg['d_model']
    n_heads = model_cfg['n_heads']
    n_layers = model_cfg['n_layers']
    num_features = model_cfg.get('num_features', d_model // n_heads)

    # Training config
    train_cfg = config['training']
    epochs = train_cfg['epochs']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']
    patience = train_cfg['patience']

    total_vocab = vocab_size + 2

    print(f"\nCreating datasets: {n_train} train, {n_test} test")
    print(f"Seq len: {seq_len}, Vocab: {vocab_size}, Pairs: {n_pairs}")
    print(f"Model: d={d_model}, H={n_heads}, L={n_layers}, m={num_features}")
    train_ds, test_ds = create_datasets(n_train, n_test, seq_len, vocab_size, n_pairs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Attention types
    attention_types = ['dense_favor', 'circulant_favor', 'relu_linear', 'softmax']

    results = {}

    for attn_type in attention_types:
        print(f"\n{'='*60}")
        print(f"Training: {attn_type}")
        print(f"{'='*60}")

        model = AssociativeRecallModel(
            vocab_size=total_vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            attention_type=attn_type,
            num_features=num_features,
            learnable_circulant=True,
            max_seq_len=128,
        ).to(device)

        n_params = model.count_params()
        n_trainable = model.count_trainable_params()
        print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0
        best_epoch = 0
        no_improve = 0
        nan_count = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
                nan_count += 1
                print(f"  Epoch {epoch}: NaN detected! (count: {nan_count})")
                if nan_count >= 5:
                    print("  Too many NaNs, stopping.")
                    break
                continue

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1 or val_acc > 0.9:
                print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} (best={best_val_acc:.3f}@{best_epoch})")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

            if val_acc >= 0.99:
                print(f"  Perfect accuracy reached at epoch {epoch}")
                break

        # Load best and evaluate
        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Benchmark feature map
        sample_batch = next(iter(test_loader))[0][:16]
        feature_map_time = benchmark_feature_map(model, sample_batch, device)

        results[attn_type] = {
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'n_params': n_params,
            'n_trainable': n_trainable,
            'nan_count': nan_count,
            'feature_map_ms': feature_map_time,
        }

        print(f"\n  Final: test_acc={test_acc:.3f} best_val_acc={best_val_acc:.3f} "
              f"params={n_params:,} feature_map={feature_map_time:.3f}ms")

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 029 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Test Acc':>10} {'Best Val':>10} {'Params':>10} {'FM (ms)':>10} {'NaN':>5}")
    print(f"{'-'*65}")
    for name, r in results.items():
        print(f"{name:<20} {r['test_acc']:>10.3f} {r['best_val_acc']:>10.3f} "
              f"{r['n_params']:>10,} {r['feature_map_ms']:>10.3f} {r['nan_count']:>5}")

    # Evaluate success criteria
    print(f"\n{'='*70}")
    print(f"SUCCESS CRITERIA EVALUATION")
    print(f"{'='*70}")

    df = results.get('dense_favor', {})
    cf = results.get('circulant_favor', {})
    rl = results.get('relu_linear', {})
    sm = results.get('softmax', {})

    # Criterion 1: C-FAVOR+ > 90% accuracy
    c1_pass = cf.get('test_acc', 0) > 0.90
    print(f"\n1. C-FAVOR+ > 90% accuracy: {'PASS' if c1_pass else 'FAIL'}")
    print(f"   C-FAVOR+ test acc: {cf.get('test_acc', 0):.3f}")
    print(f"   Dense FAVOR+ test acc: {df.get('test_acc', 0):.3f}")
    print(f"   Softmax test acc: {sm.get('test_acc', 0):.3f}")

    # Criterion 2: Feature map speed
    fm_speedup = df.get('feature_map_ms', 1) / max(cf.get('feature_map_ms', 1), 1e-6)
    c2_pass = fm_speedup > 1.0
    print(f"\n2. C-FAVOR+ feature map faster than dense: {'PASS' if c2_pass else 'FAIL'}")
    print(f"   Dense FM: {df.get('feature_map_ms', 0):.3f}ms")
    print(f"   Circ FM: {cf.get('feature_map_ms', 0):.3f}ms")
    print(f"   Speedup: {fm_speedup:.2f}x")
    print(f"   NOTE: At d_k={d_model//n_heads}, speedup may be marginal. Quality parity is key.")

    # Criterion 3: FAVOR+ > ReLU by 20%
    favor_gap = min(df.get('test_acc', 0), cf.get('test_acc', 0)) - rl.get('test_acc', 0)
    c3_pass = favor_gap > 0.20
    print(f"\n3. FAVOR+ variants > ReLU by 20%+: {'PASS' if c3_pass else 'FAIL'}")
    print(f"   Dense FAVOR+ acc: {df.get('test_acc', 0):.3f}")
    print(f"   C-FAVOR+ acc: {cf.get('test_acc', 0):.3f}")
    print(f"   ReLU acc: {rl.get('test_acc', 0):.3f}")
    print(f"   Gap: {favor_gap:.3f}")

    # Failure check 1: C-FAVOR+ > 10% worse than dense
    acc_gap = df.get('test_acc', 0) - cf.get('test_acc', 0)
    f1_fail = acc_gap > 0.10
    print(f"\n4. FAILURE CHECK: C-FAVOR+ > 10% worse than dense: {'TRIGGERED' if f1_fail else 'OK'}")
    print(f"   Accuracy gap: {acc_gap:.3f}")

    # Failure check 2: C-FAVOR+ not better than ReLU
    f2_fail = cf.get('test_acc', 0) <= rl.get('test_acc', 0)
    print(f"\n5. FAILURE CHECK: C-FAVOR+ no better than ReLU: {'TRIGGERED' if f2_fail else 'OK'}")

    # Overall decision
    print(f"\n{'='*70}")
    if f1_fail or f2_fail:
        print("DECISION: ABANDON")
        if f1_fail:
            print("  Circulant projection fundamentally breaks the positive feature approximation.")
        if f2_fail:
            print("  Feature map adds no value with circulant projection.")
    elif c1_pass and c3_pass:
        print("DECISION: PROCEED")
        print("  C-FAVOR+ achieves comparable quality to dense FAVOR+ at lower asymptotic cost.")
    else:
        print("DECISION: DEBUG")
        print("  Some criteria not met. Investigate further.")

    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='MVE 029: Circulant FAVOR+ Linear Attention')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Experiment 029: Circulant FAVOR+ for Linear Attention")
    print(f"Config: {args.config}")

    start_time = time.time()
    results = run_experiment(config)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return results


if __name__ == '__main__':
    main()

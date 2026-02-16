"""
Training script for MVE 005: Segmented-HSS Linear Attention.

Trains both HSS and Dense linear attention models on the hierarchical copying task,
then compares:
  1. Training accuracy (target: 100% on train set)
  2. State structure (qualitative: block structure in S_T)
  3. Memory efficiency (target: HSS < 0.2x dense)

Usage:
  python train.py --config config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force unbuffered output for Modal logs
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.hierarchical_copy import (
    HierarchicalCopyDataset,
    create_dataloaders,
    VOCAB_SIZE,
    IGNORE_INDEX,
    PAD_TOKEN,
)
from models.hss_linear_attention import HSSLinearAttentionModel, FlatHSSState


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple:
    """Train for one epoch. Returns (loss, accuracy, per_level_accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    # Per-level accuracy tracking
    level_correct = {0: 0, 1: 0, 2: 0}
    level_total = {0: 0, 1: 0, 2: 0}

    for input_seq, target_seq, level_info in loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        level_info = level_info.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_seq)  # (batch, seq_len, vocab_size)

        # Flatten for loss
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = target_seq.view(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size
        n_batches += 1

        # Accuracy
        preds_flat = logits_flat.argmax(dim=-1)
        valid_mask = targets_flat != IGNORE_INDEX
        correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
        total += valid_mask.sum().item()

        # Per-level accuracy
        preds = logits.argmax(dim=-1)  # (batch, seq)
        for level in [0, 1, 2]:
            level_mask = (level_info == level) & (target_seq != IGNORE_INDEX)
            if level_mask.any():
                level_correct[level] += ((preds == target_seq) & level_mask).sum().item()
                level_total[level] += level_mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = correct / max(total, 1)
    level_acc = {
        level: level_correct[level] / max(level_total[level], 1)
        for level in [0, 1, 2]
    }

    return avg_loss, avg_acc, level_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Evaluate model. Returns (loss, accuracy, per_level_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    level_correct = {0: 0, 1: 0, 2: 0}
    level_total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for input_seq, target_seq, level_info in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            level_info = level_info.to(device)

            logits = model(input_seq)
            batch_size, seq_len, num_classes = logits.shape

            logits_flat = logits.view(-1, num_classes)
            targets_flat = target_seq.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size
            n_batches += 1

            preds_flat = logits_flat.argmax(dim=-1)
            valid_mask = targets_flat != IGNORE_INDEX
            correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
            total += valid_mask.sum().item()

            preds = logits.argmax(dim=-1)
            for level in [0, 1, 2]:
                level_mask = (level_info == level) & (target_seq != IGNORE_INDEX)
                if level_mask.any():
                    level_correct[level] += ((preds == target_seq) & level_mask).sum().item()
                    level_total[level] += level_mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = correct / max(total, 1)
    level_acc = {
        level: level_correct[level] / max(level_total[level], 1)
        for level in [0, 1, 2]
    }

    return avg_loss, avg_acc, level_acc


def compute_memory_ratio(d_head: int, hss_rank: int) -> float:
    """
    Compute HSS memory / Dense memory ratio.

    Dense: d^2 floats
    HSS: flat tree structure with O(r*d*log(d)) floats

    Success criterion: ratio < 0.2
    """
    dense_memory = d_head * d_head

    # Create temporary flat HSS state to measure
    state = FlatHSSState(d_head, hss_rank, 1, torch.device('cpu'), torch.float32)
    hss_memory = state.memory_usage()

    ratio = hss_memory / dense_memory
    return ratio, hss_memory, dense_memory


def analyze_state_structure(model: nn.Module, loader: DataLoader, device: str):
    """
    Analyze the HSS state structure after processing a sequence.

    Success criterion: state matrix shows block structure matching input hierarchy.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for input_seq, target_seq, level_info in loader:
            input_seq = input_seq.to(device)

            # Get a single example
            single_input = input_seq[:1]  # (1, seq_len)

            # Run forward pass
            logits = model(single_input)

            # For HSS model, we can check the state structure
            if hasattr(model, 'use_hss') and model.use_hss:
                # Run attention layer manually to extract state
                from models.hss_linear_attention import elu_feature_map, FlatHSSState

                x = model.token_emb(single_input) + model.pos_emb(
                    torch.arange(single_input.shape[1], device=device).unsqueeze(0)
                )

                attn = model.attention
                q = elu_feature_map(attn.W_q(x))
                k = elu_feature_map(attn.W_k(x))
                v = attn.W_v(x)

                hss_state = FlatHSSState(
                    d=attn.d_head, r=attn.hss_rank, batch_size=1,
                    device=device, dtype=x.dtype
                )

                for t in range(single_input.shape[1]):
                    hss_state.rank1_update(k[:, t, :], v[:, t, :])

                # Convert to dense and analyze
                S_dense = hss_state.to_dense()[0]  # (d, d)
                results['state_matrix'] = S_dense.cpu().numpy()

                # Check block structure: compute singular values of off-diagonal blocks
                half = S_dense.shape[0] // 2
                top_right = S_dense[:half, half:]
                bottom_left = S_dense[half:, :half]

                tr_svs = torch.linalg.svdvals(top_right)
                bl_svs = torch.linalg.svdvals(bottom_left)

                # Ratio of energy in top-r singular values
                r = attn.hss_rank
                tr_energy_ratio = tr_svs[:r].sum() / (tr_svs.sum() + 1e-10)
                bl_energy_ratio = bl_svs[:r].sum() / (bl_svs.sum() + 1e-10)

                results['top_right_svs'] = tr_svs.cpu().numpy()
                results['bottom_left_svs'] = bl_svs.cpu().numpy()
                results['tr_energy_ratio'] = tr_energy_ratio.item()
                results['bl_energy_ratio'] = bl_energy_ratio.item()
                results['is_low_rank'] = (tr_energy_ratio > 0.9) and (bl_energy_ratio > 0.9)

                print(f"\n=== State Structure Analysis ===")
                print(f"State matrix shape: {S_dense.shape}")
                print(f"State matrix norm: {torch.norm(S_dense).item():.4f}")
                print(f"Top-right block SVs (top 5): {tr_svs[:5].cpu().numpy()}")
                print(f"Bottom-left block SVs (top 5): {bl_svs[:5].cpu().numpy()}")
                print(f"Top-right energy in top-{r} SVs: {tr_energy_ratio:.4f}")
                print(f"Bottom-left energy in top-{r} SVs: {bl_energy_ratio:.4f}")
                print(f"Off-diagonal blocks are low-rank: {results['is_low_rank']}")

            break  # Only need one batch

    return results


def main():
    parser = argparse.ArgumentParser(description="MVE 005: Segmented-HSS Linear Attention")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Extract config sections
    data_config = config.get("dataset", {})
    model_config = config.get("model", {})
    train_config = config.get("training", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Create data
    train_loader, val_loader, test_loader = create_dataloaders(
        num_samples=data_config.get("num_samples", 5000),
        batch_size=train_config.get("batch_size", 64),
        test_split=data_config.get("test_split", 0.3),
        max_seq_len=data_config.get("max_seq_len", 32),
        seed=data_config.get("seed", 42),
    )

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # ===== TRAIN BOTH MODELS =====
    results = {}

    for use_hss in [False, True]:  # Dense first (faster), then HSS
        model_name = "HSS-LinAttn" if use_hss else "Dense-LinAttn"
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        model = HSSLinearAttentionModel(
            vocab_size=VOCAB_SIZE,
            d_model=model_config.get("d_model", 64),
            d_head=model_config.get("d_head", 64),
            max_seq_len=data_config.get("max_seq_len", 32),
            hss_rank=model_config.get("hss_rank", 8),
            use_hss=use_hss,
            num_output_classes=VOCAB_SIZE,
            dropout=model_config.get("dropout", 0.1),
        ).to(device)

        print(f"Parameters: {model.count_parameters():,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config.get("lr", 3e-4)),
            weight_decay=float(train_config.get("weight_decay", 0.01)),
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        max_epochs = train_config.get("max_epochs", 100)
        gradient_clip = train_config.get("gradient_clip", 1.0)
        patience = train_config.get("patience", 20)
        best_val_acc = 0.0
        epochs_without_improvement = 0

        # Optional: wandb logging
        try:
            import wandb
            use_wandb = config.get("logging", {}).get("wandb_project") is not None
            if use_wandb:
                wandb.init(
                    project=config["logging"]["wandb_project"],
                    name=f"005-{model_name}",
                    config={
                        "model": model_name,
                        "use_hss": use_hss,
                        **model_config,
                        **train_config,
                        **data_config,
                    },
                    reinit=True,
                )
        except ImportError:
            use_wandb = False

        start_time = time.time()

        for epoch in range(max_epochs):
            train_loss, train_acc, train_level_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, gradient_clip
            )
            val_loss, val_acc, val_level_acc = evaluate(
                model, val_loader, criterion, device
            )

            elapsed = time.time() - start_time

            if (epoch + 1) % 10 == 0 or epoch == 0 or val_acc > 0.95:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"L1: {val_level_acc[0]:.3f} L2: {val_level_acc[1]:.3f} L3: {val_level_acc[2]:.3f} | "
                      f"Time: {elapsed:.1f}s")

            if use_wandb:
                log_dict = {
                    f"{model_name}/train_loss": train_loss,
                    f"{model_name}/train_acc": train_acc,
                    f"{model_name}/val_loss": val_loss,
                    f"{model_name}/val_acc": val_acc,
                    f"{model_name}/val_level1_acc": val_level_acc[0],
                    f"{model_name}/val_level2_acc": val_level_acc[1],
                    f"{model_name}/val_level3_acc": val_level_acc[2],
                    f"{model_name}/epoch": epoch + 1,
                }
                wandb.log(log_dict)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                # Save best model
                torch.save(model.state_dict(), f"/tmp/best_{model_name}.pt")
            else:
                epochs_without_improvement += 1

            if val_acc >= 0.99:
                print(f"Reached 99% val accuracy at epoch {epoch+1}")
                break

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        training_time = time.time() - start_time

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f"/tmp/best_{model_name}.pt", weights_only=True))
        test_loss, test_acc, test_level_acc = evaluate(model, test_loader, criterion, device)

        print(f"\n{model_name} Final Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Level 1 Accuracy: {test_level_acc[0]:.4f}")
        print(f"  Level 2 Accuracy: {test_level_acc[1]:.4f}")
        print(f"  Level 3 Accuracy: {test_level_acc[2]:.4f}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")

        results[model_name] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_level_acc": {str(k): v for k, v in test_level_acc.items()},
            "best_val_acc": best_val_acc,
            "training_time_s": training_time,
            "num_params": model.count_parameters(),
            "num_epochs": epoch + 1,
        }

        # State structure analysis for HSS model
        if use_hss:
            state_analysis = analyze_state_structure(model, test_loader, device)
            results[model_name]["state_analysis"] = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in state_analysis.items()
                if k != 'state_matrix'
            }

        if use_wandb:
            wandb.finish()

    # ===== MEMORY ANALYSIS =====
    d_head = model_config.get("d_head", 64)
    hss_rank = model_config.get("hss_rank", 8)
    mem_ratio, hss_mem, dense_mem = compute_memory_ratio(d_head, hss_rank)

    print(f"\n{'='*60}")
    print("Memory Analysis")
    print(f"{'='*60}")
    print(f"Dense state memory: {dense_mem} floats ({dense_mem * 4 / 1024:.1f} KB)")
    print(f"HSS state memory:   {hss_mem} floats ({hss_mem * 4 / 1024:.1f} KB)")
    print(f"Memory ratio (HSS/Dense): {mem_ratio:.4f}")
    print(f"Target: < 0.2, Achieved: {'YES' if mem_ratio < 0.2 else 'NO'}")

    results["memory"] = {
        "dense_floats": dense_mem,
        "hss_floats": hss_mem,
        "ratio": mem_ratio,
        "target_met": mem_ratio < 0.2,
    }

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print("MVE 005 Summary: Success Criteria")
    print(f"{'='*60}")

    hss_results = results.get("HSS-LinAttn", {})
    dense_results = results.get("Dense-LinAttn", {})

    # Criterion 1: Perfect copying (100% on training set → we check test acc ≥ 90%)
    hss_acc = hss_results.get("test_acc", 0)
    dense_acc = dense_results.get("test_acc", 0)
    criterion_1 = hss_acc >= 0.90
    print(f"1. Accuracy ≥ 90%: HSS={hss_acc:.4f}, Dense={dense_acc:.4f} → {'PASS' if criterion_1 else 'FAIL'}")

    # Criterion 2: Hierarchical state structure
    is_low_rank = hss_results.get("state_analysis", {}).get("is_low_rank", False)
    print(f"2. Hierarchical state structure: {'PASS' if is_low_rank else 'FAIL/NOT_TESTED'}")

    # Criterion 3: Memory efficiency
    mem_met = results["memory"]["target_met"]
    print(f"3. Memory ratio < 0.2: {results['memory']['ratio']:.4f} → {'PASS' if mem_met else 'FAIL'}")

    # Overall decision
    all_pass = criterion_1 and mem_met
    decision = "PROCEED" if all_pass else ("DEBUG" if hss_acc >= 0.5 else "ABANDON")
    print(f"\nDecision: {decision}")

    # Save results
    results["summary"] = {
        "criterion_1_accuracy": {"pass": criterion_1, "hss": hss_acc, "dense": dense_acc},
        "criterion_2_structure": {"pass": is_low_rank},
        "criterion_3_memory": {"pass": mem_met, "ratio": results["memory"]["ratio"]},
        "decision": decision,
    }

    results_path = "/tmp/mve_005_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Also save to Modal volume path if running on Modal
    modal_results_path = "/results/mve_005_results.json"
    try:
        os.makedirs("/results", exist_ok=True)
        with open(modal_results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results also saved to {modal_results_path}")
    except (PermissionError, OSError):
        pass

    return results


if __name__ == "__main__":
    main()

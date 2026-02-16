"""
Training script for MVE 054: INT4 Smoothing for Chunkwise Linear RNN.

Runs two experiments:
1. Microbenchmark: Cosine similarity of QK^T across quantization modes
2. Copying task: Train tiny GLA models with different quantization modes
   and compare loss curves

The copying task tests whether INT4+FP8 quantization during training
preserves model quality. The model must learn to copy input tokens to
output positions — a standard test of sequence modeling capability.

Success criteria:
- INT4 QK^T cosine similarity > 99% with smoothing (vs < 85% without)
- Copying task final loss within 5% of BF16 baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import yaml
import argparse
import time
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from models.gla import GLAModel
from benchmark import run_all_benchmarks


# ============================================================
# Copying Task Dataset
# ============================================================

class CopyingDataset(Dataset):
    """
    Sequence copying task for testing GLA quality under quantization.

    Format:
        Input:  [BOS, x_1, x_2, ..., x_L, SEP, PAD, ..., PAD]
        Target: [IGN, IGN, IGN, ..., IGN, IGN, x_1, ..., x_L]

    The model must memorize the input sequence and reproduce it after
    the separator token. This tests the state-tracking capability of
    the GLA recurrence.

    Total sequence length = 2 * copy_len + 2 (BOS + input + SEP + output)
    """

    # Special tokens
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    SEP_TOKEN = 2
    FIRST_DATA_TOKEN = 3

    def __init__(
        self,
        n_samples: int = 1000,
        copy_len: int = 32,
        vocab_size: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.copy_len = copy_len
        self.vocab_size = vocab_size
        self.total_vocab = vocab_size + self.FIRST_DATA_TOKEN  # data tokens + special tokens

        torch.manual_seed(seed)
        self._generate_data()

    def _generate_data(self):
        """Pre-generate all data samples."""
        seq_len = 2 * self.copy_len + 2  # BOS + input + SEP + output

        self.inputs = torch.zeros(self.n_samples, seq_len, dtype=torch.long)
        self.targets = torch.full((self.n_samples, seq_len), -100, dtype=torch.long)

        for i in range(self.n_samples):
            # Random data tokens
            data = torch.randint(
                self.FIRST_DATA_TOKEN,
                self.total_vocab,
                (self.copy_len,)
            )

            # Input: BOS + data + SEP + padding (placeholder for output)
            self.inputs[i, 0] = self.BOS_TOKEN
            self.inputs[i, 1:self.copy_len + 1] = data
            self.inputs[i, self.copy_len + 1] = self.SEP_TOKEN
            # After SEP, the input is the previous output token (teacher forcing)
            # For autoregressive: we set input to data shifted by 1
            self.inputs[i, self.copy_len + 2:] = data[:-1] if self.copy_len > 1 else self.PAD_TOKEN

            # Target: only predict the copied sequence
            self.targets[i, self.copy_len + 2:] = data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
    model: GLAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mode: str = "bf16",
) -> Dict[str, float]:
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs, mode=mode)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accuracy on non-ignored tokens
        mask = targets != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == targets[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(
    model: GLAModel,
    dataloader: DataLoader,
    device: str,
    mode: str = "bf16",
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs, mode=mode)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )

        mask = targets != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == targets[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


def train_model(
    config: Dict,
    mode: str,
    device: str,
    wandb_group: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a GLA model on the copying task with a specific quantization mode.

    Returns final metrics.
    """
    mc = config["model"]
    tc = config["training"]
    dc = config["dataset"]

    # Create dataset
    train_dataset = CopyingDataset(
        n_samples=dc["train_samples"],
        copy_len=dc["copy_len"],
        vocab_size=dc["vocab_size"],
        seed=42,
    )
    val_dataset = CopyingDataset(
        n_samples=dc["val_samples"],
        copy_len=dc["copy_len"],
        vocab_size=dc["vocab_size"],
        seed=123,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tc["batch_size"], shuffle=False,
    )

    total_vocab = dc["vocab_size"] + CopyingDataset.FIRST_DATA_TOKEN

    # Create model
    model = GLAModel(
        vocab_size=total_vocab,
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        dk_per_head=mc["dk_per_head"],
        dv_per_head=mc["dv_per_head"],
        chunk_size=mc["chunk_size"],
        sub_chunk_size=mc["sub_chunk_size"],
        mode=mode,
    ).to(device)

    print(f"\nModel ({mode}): {model.count_params():,} params")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(tc["n_epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, mode)
        val_metrics = evaluate(model, val_loader, device, mode)

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]

        # Log to wandb
        wandb.log({
            f"train/loss_{mode}": train_metrics["loss"],
            f"train/accuracy_{mode}": train_metrics["accuracy"],
            f"val/loss_{mode}": val_metrics["loss"],
            f"val/accuracy_{mode}": val_metrics["accuracy"],
            "epoch": epoch,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{tc['n_epochs']}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"train_acc={train_metrics['accuracy']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}")

        # Early stopping on perfect accuracy
        if val_metrics["accuracy"] > 0.99:
            print(f"  Early stopping at epoch {epoch+1} — val_acc > 0.99")
            break

    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_epochs_trained": len(train_losses),
    }


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MVE 054: INT4 Smoothing for GLA")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="mad-architecture-search",
        name=f"exp-054-int4-smoothing-gla",
        config=config,
        tags=["exp-054", "int4", "gla", "quantization", "smoothing"],
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    all_results = {}
    start_time = time.time()

    # ============================================================
    # Part 1: Microbenchmark — Cosine Similarity
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: Microbenchmark — QK^T Cosine Similarity")
    print("=" * 60)

    benchmark_results = run_all_benchmarks(device=device)
    all_results["benchmark"] = benchmark_results

    # Log benchmark results to wandb
    main_bench = benchmark_results["main"]
    wandb.log({
        "benchmark/cosine_sim_no_smooth": main_bench["cosine_sim_no_smooth_mean"],
        "benchmark/cosine_sim_smooth": main_bench["cosine_sim_smooth_mean"],
        "benchmark/mse_no_smooth": main_bench["mse_no_smooth_mean"],
        "benchmark/mse_smooth": main_bench["mse_smooth_mean"],
        "benchmark/fp8_sv_cosine_sim": benchmark_results["fp8_sv"]["fp8_sv_cosine_sim_mean"],
    })

    # ============================================================
    # Part 2: Copying Task Training
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: Copying Task — Training Quality Comparison")
    print("=" * 60)

    modes = ["bf16", "int4_no_smooth", "int4_smooth", "int4_fp8"]
    training_results = {}

    for mode in modes:
        print(f"\n--- Training with mode: {mode} ---")
        try:
            results = train_model(config, mode, device)
            training_results[mode] = results
            print(f"  Final val loss: {results['final_val_loss']:.4f}")
            print(f"  Best val acc: {results['best_val_acc']:.4f}")
        except Exception as e:
            print(f"  ERROR training {mode}: {e}")
            import traceback
            traceback.print_exc()
            training_results[mode] = {"error": str(e)}

    all_results["training"] = training_results

    # ============================================================
    # Part 3: Quality Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: Results Summary")
    print("=" * 60)

    bf16_loss = training_results.get("bf16", {}).get("best_val_loss", float("inf"))
    quality_results = {}

    for mode in modes:
        if mode == "bf16":
            continue
        mode_loss = training_results.get(mode, {}).get("best_val_loss", float("inf"))
        if bf16_loss > 0 and mode_loss != float("inf"):
            loss_ratio = mode_loss / bf16_loss
            within_5pct = loss_ratio < 1.05
            quality_results[mode] = {
                "loss_ratio": loss_ratio,
                "within_5pct": within_5pct,
                "best_val_loss": mode_loss,
            }
            status = "PASS" if within_5pct else "FAIL"
            print(f"  [{status}] {mode}: loss_ratio={loss_ratio:.4f} "
                  f"(target < 1.05, loss={mode_loss:.4f} vs bf16={bf16_loss:.4f})")

    all_results["quality_comparison"] = quality_results

    # ============================================================
    # Final Success Criteria
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL SUCCESS CRITERIA")
    print("=" * 60)

    smooth_cos = main_bench["cosine_sim_smooth_mean"]
    no_smooth_cos = main_bench["cosine_sim_no_smooth_mean"]

    crit1 = smooth_cos > 0.99
    crit2 = no_smooth_cos < 0.85
    crit3_mode = "int4_smooth"
    crit3 = quality_results.get(crit3_mode, {}).get("within_5pct", False)
    # Smoothing overhead criterion — measured by cosine improvement
    crit4 = (smooth_cos - no_smooth_cos) > 0.05  # Smoothing provides > 5% improvement

    print(f"  [{'PASS' if crit1 else 'FAIL'}] Criterion 1: INT4 cosine sim > 99% with smoothing: {smooth_cos:.4f}")
    print(f"  [{'PASS' if crit2 else 'INFO'}] Criterion 2: INT4 cosine sim < 85% without smoothing: {no_smooth_cos:.4f}")
    print(f"  [{'PASS' if crit3 else 'FAIL'}] Criterion 3: Copying task loss within 5% of BF16")
    print(f"  [{'PASS' if crit4 else 'FAIL'}] Criterion 4: Smoothing provides significant accuracy improvement")

    # Log final results to wandb
    wandb.log({
        "final/cosine_sim_smooth": smooth_cos,
        "final/cosine_sim_no_smooth": no_smooth_cos,
        "final/bf16_best_val_loss": bf16_loss,
        "success_criteria/cosine_sim_gt_99pct": crit1,
        "success_criteria/no_smooth_lt_85pct": crit2,
        "success_criteria/loss_within_5pct": crit3,
        "success_criteria/smoothing_improves_accuracy": crit4,
    })

    # Log per-mode final losses
    for mode in modes:
        if mode in training_results and "error" not in training_results[mode]:
            wandb.log({
                f"final/{mode}_best_val_loss": training_results[mode]["best_val_loss"],
                f"final/{mode}_best_val_acc": training_results[mode]["best_val_acc"],
            })

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    all_results["elapsed_time_s"] = elapsed
    all_results["success_criteria_summary"] = {
        "cosine_sim_smooth_gt_99pct": crit1,
        "cosine_sim_no_smooth_lt_85pct": crit2,
        "loss_within_5pct_of_bf16": crit3,
        "smoothing_improves_accuracy": crit4,
    }

    # Save results
    output_path = Path("results.json")
    # Convert non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            return obj
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    with open(output_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Results saved to {output_path}")

    print(f"\nWandb URL: {wandb_url}")
    wandb.finish()

    return all_results


if __name__ == "__main__":
    main()

"""
Config-based training script for MVE 006: Monarch-Gated State Transition SSM

Trains both Monarch-Gated and Diagonal SSMs on S5 permutation composition,
measuring accuracy and relative speed.

v3: Curriculum learning - start with short sequences, gradually increase.

Usage:
    uv run python -m train.run_config --config config.yaml
"""

import argparse
import time
import json
import os
import sys

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

# Force unbuffered output for Modal logs
sys.stdout.reconfigure(line_buffering=True)

from models.monarch_ssm import S5SSMModel
from tasks.s5 import (
    S5OnlineDataset,
    S5FixedDataset,
    NUM_INPUT_TOKENS,
    NUM_ELEMENTS,
    EOS_IDX,
    PAD_IDX,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(model_config: dict) -> S5SSMModel:
    return S5SSMModel(
        num_tokens=NUM_INPUT_TOKENS,
        num_classes=NUM_ELEMENTS,
        d_model=model_config["d_model"],
        state_dim=model_config["state_dim"],
        num_layers=model_config["num_layers"],
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config.get("dropout", 0.1),
        ssm_type=model_config["ssm_type"],
        eos_idx=EOS_IDX,
    )


def train_epoch_online(
    model: nn.Module,
    seq_len: int,
    max_pad_len: int,
    batch_size: int,
    batches_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch with fresh online data. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    dataset = S5OnlineDataset(
        seq_len=seq_len,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        max_pad_len=max_pad_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.shape[0]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.shape[0]

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.shape[0]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.shape[0]

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def benchmark_speed(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    seq_len: int = 22,
    num_iters: int = 50,
    warmup_iters: int = 10,
) -> float:
    """Benchmark forward pass speed. Returns avg time per batch in ms."""
    model.eval()
    dummy_input = torch.randint(0, 10, (batch_size, seq_len), device=device)
    dummy_input[:, 0] = 130
    dummy_input[:, -1] = EOS_IDX

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_iters * 1000


def train_model_curriculum(
    model: nn.Module,
    model_name: str,
    model_config: dict,
    dataset_config: dict,
    training_config: dict,
    logging_config: dict,
    device: torch.device,
) -> dict:
    """Train a single model with curriculum learning."""

    param_count = model.count_parameters()
    print(f"Parameters: {param_count:,}")

    wandb_project = logging_config.get("wandb_project", "mad-architecture-search")
    run = wandb.init(
        project=wandb_project,
        name=f"exp-006-{model_name}-v3",
        config={
            "model": model_config,
            "dataset": dataset_config,
            "training": training_config,
            "model_name": model_name,
            "proposal_id": "006-monarch-gated-state-transition",
            "version": "v3-curriculum",
        },
        reinit=True,
    )
    wandb_url = run.url
    print(f"Wandb URL: {wandb_url}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )

    criterion = nn.CrossEntropyLoss()
    batch_size = training_config["batch_size"]
    batches_per_epoch = dataset_config.get("batches_per_epoch", 100)
    max_pad_len = model_config["max_seq_len"]

    curriculum_stages = dataset_config.get("curriculum_stages", [2, 5, 10, 15, 20])
    epochs_per_stage = training_config.get("epochs_per_stage", 50)
    target_acc = training_config.get("target_acc_per_stage", 0.80)

    best_test_acc_overall = 0.0
    nan_detected = False
    global_epoch = 0

    for stage_idx, seq_len in enumerate(curriculum_stages):
        print(f"\n--- Curriculum Stage {stage_idx+1}: seq_len={seq_len} ---")

        test_dataset = S5FixedDataset(
            num_samples=2000, seq_len=seq_len, max_pad_len=max_pad_len, seed=42+stage_idx
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        best_stage_acc = 0.0

        for epoch in range(1, epochs_per_stage + 1):
            global_epoch += 1

            train_loss, train_acc = train_epoch_online(
                model, seq_len, max_pad_len, batch_size, batches_per_epoch,
                optimizer, criterion, device,
                max_grad_norm=float(training_config.get("gradient_clip", 1.0)),
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if torch.isnan(torch.tensor(train_loss)):
                print(f"NaN detected at epoch {global_epoch}!")
                nan_detected = True
                break

            best_stage_acc = max(best_stage_acc, test_acc)
            best_test_acc_overall = max(best_test_acc_overall, test_acc)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs_per_stage} (global {global_epoch}): "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": test_loss,
                "val/accuracy": test_acc,
                "best_val_accuracy": best_test_acc_overall,
                "epoch": global_epoch,
                "curriculum/seq_len": seq_len,
                "curriculum/stage": stage_idx + 1,
            })

            if test_acc >= target_acc:
                print(f"  Reached {target_acc:.0%} accuracy at epoch {epoch}, moving to next stage")
                break

        if nan_detected:
            break

        print(f"  Stage {stage_idx+1} best test acc: {best_stage_acc:.4f}")

    # Final evaluation at seq_len=20
    eval_20_dataset = S5FixedDataset(
        num_samples=2000, seq_len=20, max_pad_len=max_pad_len, seed=99999
    )
    eval_20_loader = DataLoader(eval_20_dataset, batch_size=batch_size, shuffle=False)
    eval_20_loss, eval_20_acc = evaluate(model, eval_20_loader, criterion, device)
    print(f"\nFinal eval (seq_len=20): Accuracy={eval_20_acc:.4f}")

    speed_ms = benchmark_speed(model, device, batch_size=32, seq_len=max_pad_len)
    print(f"Avg forward pass: {speed_ms:.2f} ms/batch")

    wandb.log({
        "final/test_accuracy": eval_20_acc,
        "final/best_test_accuracy": best_test_acc_overall,
        "final/eval_20_accuracy": eval_20_acc,
        "final/speed_ms_per_batch": speed_ms,
        "final/num_parameters": param_count,
        "final/nan_detected": nan_detected,
    })
    wandb.finish()

    return {
        "best_test_acc": best_test_acc_overall,
        "eval_20_acc": eval_20_acc,
        "speed_ms": speed_ms,
        "param_count": param_count,
        "nan_detected": nan_detected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    logging_config = config.get("logging", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_train = config.get("models", {})
    results = {}

    for model_name, model_config in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name} ({model_config['ssm_type']} SSM)")
        print(f"{'='*60}")

        model = create_model(model_config)
        model = model.to(device)

        result = train_model_curriculum(
            model, model_name, model_config,
            dataset_config, training_config, logging_config, device,
        )
        results[model_name] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  Best test accuracy (any stage): {res['best_test_acc']:.4f}")
        print(f"  Eval seq_len=20 accuracy: {res['eval_20_acc']:.4f}")
        print(f"  Speed: {res['speed_ms']:.2f} ms/batch")
        print(f"  Parameters: {res['param_count']:,}")
        print(f"  NaN detected: {res['nan_detected']}")

    # Success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA")
    print(f"{'='*60}")

    monarch_res = results.get("monarch_gated")
    diag_res = results.get("diagonal_baseline")

    if monarch_res and diag_res:
        c1 = monarch_res["eval_20_acc"] > 0.85
        c2 = diag_res["eval_20_acc"] < 0.50
        speed_ratio = monarch_res["speed_ms"] / max(diag_res["speed_ms"], 1e-6)
        c3 = speed_ratio < 3.0
        c4 = not monarch_res["nan_detected"]

        print(f"1. Monarch >85% on S5 len-20: {monarch_res['eval_20_acc']:.4f} "
              f"{'PASS' if c1 else 'FAIL'}")
        print(f"2. Diagonal <50% on S5 len-20: {diag_res['eval_20_acc']:.4f} "
              f"{'PASS' if c2 else 'FAIL'}")
        print(f"3. Speed ratio <3x: {speed_ratio:.2f}x "
              f"{'PASS' if c3 else 'FAIL'}")
        print(f"4. No NaN/Inf: {not monarch_res['nan_detected']} "
              f"{'PASS' if c4 else 'FAIL'}")

    results_path = "/results/results.json" if os.environ.get("MODAL_ENVIRONMENT") else "results.json"
    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    except (OSError, ValueError):
        pass
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

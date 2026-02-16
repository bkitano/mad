"""
Config-based training script for MVE 001: D4 State Tracking

Usage:
    uv run accelerate launch -m train.run_config --config configs/cs_neg_deltanet.yaml
"""

import argparse
import json
import os
import tempfile
from typing import Protocol

import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from huggingface_hub import HfApi, whoami
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deltanet import GroupDeltaNet
from models.cs_deltanet import CSDeltaNet
from tasks.d4.tokens import D4TokenSystem
from tasks.d4.dataset import D4CurriculumWrapper
from train.train import evaluate, train_epoch


class TokenSystemProtocol(Protocol):
    """Protocol for token systems."""
    num_tokens: int
    num_group_elements: int
    EOS_IDX: int
    PAD_IDX: int
    BOS_IDX: int


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_wandb_name(task: str, dataset_cfg: dict, model_cfg: dict, train_cfg: dict) -> str:
    """Build a descriptive wandb run name from config."""
    task_bits = [task.upper()]
    model_bits = [
        model_cfg["type"],
        f"L{model_cfg['num_layers']}",
        f"H{model_cfg['nhead']}",
        f"D{model_cfg['d_model']}",
    ]
    if model_cfg.get("allow_neg_eigval"):
        model_bits.append("NEG")
    data_bits = [
        f"k{dataset_cfg['max_k']}",
        f"s{dataset_cfg['samples_per_k']}",
    ]
    train_bits = [
        f"bs{train_cfg['batch_size']}",
        f"lr{train_cfg['lr']}",
    ]
    return "-".join(task_bits + model_bits + data_bits + train_bits)


def create_token_system(task: str, dataset_config: dict) -> TokenSystemProtocol:
    """Create token system based on task type."""
    if task == "d4":
        return D4TokenSystem()
    else:
        raise ValueError(f"Unknown task: {task}. Supported: d4")


def create_curriculum(task: str, token_system: TokenSystemProtocol, dataset_config: dict):
    """Create curriculum dataset based on task type."""
    if task == "d4":
        return D4CurriculumWrapper(
            token_system=token_system,
            max_k=dataset_config["max_k"],
            samples_per_k=dataset_config["samples_per_k"],
            max_seq_len=dataset_config["max_seq_len"],
            test_size=dataset_config.get("test_size", 0.2),
            use_generators_only=dataset_config.get("use_generators_only", True),
            fixed_k=dataset_config.get("fixed_k", None),
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def create_model(model_config: dict, token_system: TokenSystemProtocol) -> nn.Module:
    """Create model based on config."""
    model_type = model_config["type"]

    common_args = {
        "num_tokens": token_system.num_tokens,
        "num_classes": token_system.num_group_elements,
        "eos_idx": token_system.EOS_IDX,
        "max_seq_len": model_config.get("max_seq_len", 32),
        "d_model": model_config["d_model"],
        "nhead": model_config["nhead"],
        "num_layers": model_config["num_layers"],
        "dropout": model_config.get("dropout", 0.1),
        "allow_neg_eigval": model_config.get("allow_neg_eigval", False),
    }

    if model_type == "GroupDeltaNet":
        return GroupDeltaNet(**common_args)
    elif model_type == "CSDeltaNet":
        return CSDeltaNet(
            **common_args,
            gumbel_tau=model_config.get("gumbel_tau", 1.0),
            gumbel_hard=model_config.get("gumbel_hard", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_config = config["dataset"]
    task = dataset_config.get("task", "d4")
    model_config = config["model"]
    train_config = config["training"]
    logging_config = config.get("logging", {})

    print(f"Task: {task}")
    print(f"Config: {args.config}")

    # Create token system and curriculum
    token_system = create_token_system(task, dataset_config)
    curriculum = create_curriculum(task, token_system, dataset_config)

    print(f"Token system: {token_system.__class__.__name__}")
    print(f"  num_tokens: {token_system.num_tokens}")
    print(f"  num_group_elements: {token_system.num_group_elements}")

    # Add derived values to model config
    model_config["max_seq_len"] = dataset_config["max_seq_len"]

    # Create model
    model = create_model(model_config, token_system)
    print(f"Model: {model_config['type']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  allow_neg_eigval: {model_config.get('allow_neg_eigval', False)}")

    # Optionally compile
    if model_config.get("use_compile", False) and torch.cuda.is_available():
        model = torch.compile(model)

    # Setup accelerator
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

    wandb_run_id = None
    wandb_run_name = None
    wandb_run_url = None

    # Initialize wandb
    if accelerator.is_main_process:
        wandb_project = logging_config.get("wandb_project", "mve-001-d4-state-tracking")
        wandb_run = wandb.init(
            project=wandb_project,
            name=build_wandb_name(task, dataset_config, model_config, train_config),
            config={
                "task": task,
                "dataset": dataset_config,
                "model": model_config,
                "train": train_config,
            },
        )
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name
        wandb_run_url = wandb_run.url

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.999))),
        eps=float(train_config.get("op_eps", 1e-8)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    use_curriculum = dataset_config.get("curriculum", True)
    fixed_k = dataset_config.get("fixed_k", None)

    if not use_curriculum:
        # Non-curriculum mode
        train_k = fixed_k if fixed_k else curriculum.num_stages()
        print(f"\nNon-Curriculum Mode: Training at fixed k={train_k}")

        train_dataset, test_dataset = curriculum.get_fixed_k(train_k)
        print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)
        train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

        max_epochs = train_config.get("max_epochs_per_stage", 100)
        max_val_acc = train_config.get("max_val_acc", 0.99)
        gradient_clip = train_config.get("gradient_clip", 1.0)

        for epoch in tqdm(range(max_epochs), desc=f"k={train_k}"):
            loss, accuracy = train_epoch(
                model, train_loader, optimizer, criterion,
                accelerator=accelerator, device=accelerator.device, max_grad_norm=gradient_clip,
            )
            val_loss, val_accuracy, val_k_accuracy = evaluate(
                model, test_loader, criterion, device=accelerator.device, accelerator=accelerator
            )
            global_step += 1

            print(f"k={train_k} Epoch {epoch+1}: Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if accelerator.is_main_process:
                log_dict = {
                    "global_step": global_step,
                    "k": train_k,
                    "epoch": epoch + 1,
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                }
                for k, acc in val_k_accuracy.items():
                    log_dict[f"val/accuracy_k{k}"] = acc
                wandb.log(log_dict)

            if max_val_acc is not None and val_accuracy >= max_val_acc:
                print(f"Reached target accuracy {max_val_acc:.2%} at epoch {epoch+1}")
                break

    else:
        # Curriculum training
        for stage in range(1, curriculum.num_stages() + 1):
            print(f"\n{'='*50}")
            print(f"Curriculum Stage {stage}: k=1 to k={stage}")
            print(f"{'='*50}")

            train_dataset, test_dataset = curriculum.get_stage(stage)
            print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

            train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)
            train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

            max_epochs = train_config.get("max_epochs_per_stage", 100)
            max_val_acc = train_config.get("max_val_acc", 0.99)
            gradient_clip = train_config.get("gradient_clip", 1.0)

            for epoch in tqdm(range(max_epochs), desc=f"Stage {stage}"):
                loss, accuracy = train_epoch(
                    model, train_loader, optimizer, criterion,
                    accelerator=accelerator, device=accelerator.device, max_grad_norm=gradient_clip,
                )
                val_loss, val_accuracy, val_k_accuracy = evaluate(
                    model, test_loader, criterion, device=accelerator.device, accelerator=accelerator
                )
                global_step += 1

                print(f"Stage {stage} Epoch {epoch+1}: Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

                if accelerator.is_main_process:
                    log_dict = {
                        "global_step": global_step,
                        "stage": stage,
                        "epoch": epoch + 1,
                        "train/loss": loss,
                        "train/accuracy": accuracy,
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy,
                    }
                    for k, acc in val_k_accuracy.items():
                        log_dict[f"val/accuracy_k{k}"] = acc
                    wandb.log(log_dict)

                if max_val_acc is not None and val_accuracy >= max_val_acc:
                    print(f"Reached target accuracy {max_val_acc:.2%} at stage {stage} epoch {epoch+1}")
                    break

    # Final evaluation
    if use_curriculum:
        _, final_test = curriculum.get_stage(curriculum.num_stages())
    else:
        train_k = fixed_k if fixed_k else curriculum.num_stages()
        _, final_test = curriculum.get_fixed_k(train_k)

    final_test_loader = DataLoader(final_test, batch_size=train_config["batch_size"], shuffle=False)
    final_test_loader = accelerator.prepare(final_test_loader)

    final_loss, final_accuracy, final_k_accuracy = evaluate(
        model, final_test_loader, criterion, device=accelerator.device, accelerator=accelerator
    )

    print(f"\nFinal Test: Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    print(f"Per-k accuracy: {final_k_accuracy}")

    if accelerator.is_main_process:
        log_dict = {
            "test/final_loss": final_loss,
            "test/final_accuracy": final_accuracy,
        }
        for k, acc in final_k_accuracy.items():
            log_dict[f"test/accuracy_k{k}"] = acc
        wandb.log(log_dict)
        wandb.finish()

    # Push to HuggingFace Hub
    if accelerator.is_main_process and logging_config.get("hf_repo_id"):
        repo_id = logging_config["hf_repo_id"]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = wandb_run_id or "no-wandb"
            run_dir = os.path.join(tmpdir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)

            metadata = {
                "wandb": {"run_id": wandb_run_id, "name": wandb_run_name, "url": wandb_run_url},
                "task": task,
                "dataset": dataset_config,
                "model": model_config,
                "train": train_config,
                "final_accuracy": final_accuracy,
            }
            metadata_path = os.path.join(run_dir, "config.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            api = HfApi()
            api.create_repo(repo_id, exist_ok=True)
            api.upload_file(path_or_fileobj=model_path, path_in_repo=f"runs/{run_id}/model.pt", repo_id=repo_id)
            api.upload_file(path_or_fileobj=metadata_path, path_in_repo=f"runs/{run_id}/config.json", repo_id=repo_id)
            print(f"Model pushed to https://huggingface.co/{repo_id}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

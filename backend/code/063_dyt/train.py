"""
Training script for MVE 063: DyT vs RMSNorm in GLA.

Trains 4 variants of a 2-layer GLA model:
1. Pre-LN RMSNorm (baseline)
2. Pre-LN DyT
3. Peri-LN RMSNorm
4. Peri-LN DyT

On a synthetic autoregressive language modeling task.
Compares perplexity, training stability, and activation statistics.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

import wandb

from models.gla import GLAModel
from models.normalization import DyT, RMSNorm


# ==============================================================================
# Synthetic Data Generation
# ==============================================================================

class SyntheticLMDataset(Dataset):
    """
    Synthetic language modeling dataset with Zipf + bigram patterns.
    Tests normalization quality on realistic token distributions.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        rng = np.random.RandomState(seed)

        # Zipf-distributed unigram probabilities
        ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
        probs = 1.0 / (ranks ** 1.0)
        probs = probs / probs.sum()

        # Bigram transition matrix with local structure
        bigram = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for i in range(vocab_size):
            base = probs.copy()
            for offset in range(-3, 4):
                j = (i + offset) % vocab_size
                base[j] *= 3.0
            bigram[i] = base / base.sum()

        # Generate all sequences
        self.sequences = np.zeros((n_samples, seq_len), dtype=np.int64)
        for i in range(n_samples):
            self.sequences[i, 0] = rng.choice(vocab_size, p=probs)
            for t in range(1, seq_len):
                prev = self.sequences[i, t - 1]
                self.sequences[i, t] = rng.choice(vocab_size, p=bigram[prev])

        self.sequences = torch.from_numpy(self.sequences)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


# ==============================================================================
# Training Utilities
# ==============================================================================

def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


def compute_gradient_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def collect_activation_stats(model, dataloader, device, n_batches=10):
    """Collect activation stats from normalization layer outputs."""
    model.eval()
    hook_outputs = defaultdict(list)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (DyT, RMSNorm)):
            def make_hook(n):
                def hook(mod, inp, out):
                    hook_outputs[n].append(out.detach().cpu())
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    for h in hooks:
        h.remove()

    stats = {}
    for name, outputs in hook_outputs.items():
        all_out = torch.cat(outputs, dim=0)
        stats[name] = {
            "mean": all_out.mean().item(),
            "std": all_out.std().item(),
            "min": all_out.min().item(),
            "max": all_out.max().item(),
            "abs_mean": all_out.abs().mean().item(),
        }

    model.train()
    return stats


# ==============================================================================
# Training Loop for Single Variant
# ==============================================================================

def train_variant(variant_name, config, device, seed=42):
    """Train a single GLA variant and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    parts = variant_name.split("_")
    peri_ln = parts[0] == "periln"
    norm_type = parts[1]

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    model = GLAModel(
        vocab_size=data_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        d_k=model_cfg["d_k"],
        d_v=model_cfg["d_v"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_seq_len=data_cfg["seq_len"],
        norm_type=norm_type,
        peri_ln=peri_ln,
        d_ff=model_cfg.get("d_ff"),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"Parameters: {n_params:,}")
    print(f"Norm: {norm_type}, Peri-LN: {peri_ln}")
    print(f"{'='*60}")

    train_dataset = SyntheticLMDataset(
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
        n_samples=data_cfg["train_samples"],
        seed=seed,
    )
    val_dataset = SyntheticLMDataset(
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
        n_samples=data_cfg["val_samples"],
        seed=seed + 1000,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.999)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    n_epochs = train_cfg["epochs"]
    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "variant": variant_name,
        "n_params": n_params,
        "train_losses": [],
        "val_losses": [],
        "train_perplexities": [],
        "val_perplexities": [],
        "grad_norms": [],
        "grad_norm_variances": [],
        "had_nan": False,
        "had_inf": False,
        "alpha_values": [],
    }

    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        grad_norms_in_epoch = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                metrics["had_nan"] = metrics["had_nan"] or torch.isnan(loss).item()
                metrics["had_inf"] = metrics["had_inf"] or torch.isinf(loss).item()
                print(f"  WARNING: NaN/Inf loss at epoch {epoch}")
                break

            optimizer.zero_grad()
            loss.backward()

            if train_cfg.get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])

            grad_norm = compute_gradient_norm(model)
            grad_norms_in_epoch.append(grad_norm)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(f"  Epoch {epoch}: All batches had NaN/Inf, stopping")
            break

        train_loss = epoch_loss / n_batches
        train_ppl = compute_perplexity(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1

        val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
        val_ppl = compute_perplexity(val_loss)

        mean_grad_norm = np.mean(grad_norms_in_epoch) if grad_norms_in_epoch else 0
        var_grad_norm = np.var(grad_norms_in_epoch) if len(grad_norms_in_epoch) > 1 else 0

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_loss)
        metrics["train_perplexities"].append(train_ppl)
        metrics["val_perplexities"].append(val_ppl)
        metrics["grad_norms"].append(float(mean_grad_norm))
        metrics["grad_norm_variances"].append(float(var_grad_norm))

        alpha_stats = model.get_activation_stats()
        metrics["alpha_values"].append(alpha_stats)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Log to wandb
        log_dict = {
            f"{variant_name}/train_loss": train_loss,
            f"{variant_name}/val_loss": val_loss,
            f"{variant_name}/train_ppl": train_ppl,
            f"{variant_name}/val_ppl": val_ppl,
            f"{variant_name}/grad_norm_mean": float(mean_grad_norm),
            f"{variant_name}/grad_norm_var": float(var_grad_norm),
            f"{variant_name}/lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        }
        for k, v in alpha_stats.items():
            log_dict[f"{variant_name}/alpha/{k}"] = v
        wandb.log(log_dict)

        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            alpha_str = ""
            if alpha_stats:
                alpha_vals = [f"{v:.4f}" for v in alpha_stats.values()]
                alpha_str = f" | alphas=[{','.join(alpha_vals)}]"
            print(
                f"  Epoch {epoch:3d}/{n_epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f} "
                f"grad_norm={mean_grad_norm:.4f}{alpha_str}"
            )

    # Collect activation statistics
    act_stats = collect_activation_stats(model, val_loader, device)
    metrics["activation_stats"] = act_stats

    metrics["best_val_loss"] = best_val_loss
    metrics["best_val_ppl"] = compute_perplexity(best_val_loss)
    metrics["final_train_loss"] = metrics["train_losses"][-1] if metrics["train_losses"] else float("inf")
    metrics["final_val_loss"] = metrics["val_losses"][-1] if metrics["val_losses"] else float("inf")
    metrics["final_train_ppl"] = metrics["train_perplexities"][-1] if metrics["train_perplexities"] else float("inf")
    metrics["final_val_ppl"] = metrics["val_perplexities"][-1] if metrics["val_perplexities"] else float("inf")
    metrics["mean_grad_norm_variance"] = float(np.mean(metrics["grad_norm_variances"])) if metrics["grad_norm_variances"] else 0

    return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wandb.init(
        project="mad-architecture-search",
        name="exp-063-dyt-vs-rmsnorm-gla",
        config=config,
        tags=["exp-063", "dyt", "rmsnorm", "gla", "normalization"],
    )
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    variants = ["preln_rmsnorm", "preln_dyt", "periln_rmsnorm", "periln_dyt"]
    all_metrics = {}
    start_time = time.time()

    for variant in variants:
        variant_start = time.time()
        metrics = train_variant(variant, config, device, seed=config.get("seed", 42))
        variant_time = time.time() - variant_start
        metrics["training_time_seconds"] = variant_time
        all_metrics[variant] = metrics
        print(f"\n  Variant {variant} completed in {variant_time:.1f}s")

    total_time = time.time() - start_time

    # ==============================================================================
    # Analysis & Success Criteria
    # ==============================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    preln_rmsnorm_ppl = all_metrics["preln_rmsnorm"]["best_val_ppl"]
    preln_dyt_ppl = all_metrics["preln_dyt"]["best_val_ppl"]
    periln_rmsnorm_ppl = all_metrics["periln_rmsnorm"]["best_val_ppl"]
    periln_dyt_ppl = all_metrics["periln_dyt"]["best_val_ppl"]

    ppl_diff_preln = abs(preln_dyt_ppl - preln_rmsnorm_ppl)
    ppl_diff_periln = abs(periln_dyt_ppl - periln_rmsnorm_ppl)

    criterion1_preln = ppl_diff_preln <= 0.5
    criterion1_periln = ppl_diff_periln <= 0.5
    criterion1 = criterion1_preln and criterion1_periln

    print(f"\nCriterion 1: DyT matches RMSNorm within 0.5 perplexity")
    print(f"  Pre-LN: RMSNorm={preln_rmsnorm_ppl:.2f}, DyT={preln_dyt_ppl:.2f}, diff={ppl_diff_preln:.2f} {'PASS' if criterion1_preln else 'FAIL'}")
    print(f"  Peri-LN: RMSNorm={periln_rmsnorm_ppl:.2f}, DyT={periln_dyt_ppl:.2f}, diff={ppl_diff_periln:.2f} {'PASS' if criterion1_periln else 'FAIL'}")

    grad_norm_vars = {v: all_metrics[v]["mean_grad_norm_variance"] for v in variants}
    best_stability = min(grad_norm_vars, key=grad_norm_vars.get)
    criterion2 = best_stability == "periln_dyt"

    print(f"\nCriterion 2: Peri-LN DyT has best stability (lowest grad norm variance)")
    for v in variants:
        marker = " <-- best" if v == best_stability else ""
        print(f"  {v}: grad_norm_var={grad_norm_vars[v]:.6f}{marker}")
    print(f"  {'PASS' if criterion2 else 'FAIL'} (best: {best_stability})")

    nan_inf_results = {}
    for v in variants:
        nan_inf_results[v] = {"nan": all_metrics[v]["had_nan"], "inf": all_metrics[v]["had_inf"]}
    criterion3 = all(not r["nan"] and not r["inf"] for r in nan_inf_results.values())

    print(f"\nCriterion 3: No NaN/Inf in any variant")
    for v in variants:
        status = "CLEAN" if not nan_inf_results[v]["nan"] and not nan_inf_results[v]["inf"] else "ISSUES"
        print(f"  {v}: NaN={nan_inf_results[v]['nan']}, Inf={nan_inf_results[v]['inf']} ({status})")
    print(f"  {'PASS' if criterion3 else 'FAIL'}")

    criterion4 = True
    print(f"\nCriterion 4: Activation distribution similarity")
    for v in ["preln_dyt", "periln_dyt"]:
        for name, stats in all_metrics[v].get("activation_stats", {}).items():
            if abs(stats["mean"]) > 10 or stats["std"] > 10:
                criterion4 = False
                print(f"  WARNING: {v}/{name} extreme activations: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    for v in variants:
        print(f"  {v} activations:")
        for name, stats in all_metrics[v].get("activation_stats", {}).items():
            print(f"    {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, abs_mean={stats['abs_mean']:.4f}")
    print(f"  {'PASS' if criterion4 else 'FAIL'}")

    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"\n{'Variant':<20} {'Params':<10} {'Best PPL':<10} {'Final PPL':<10} {'Grad Var':<12} {'NaN/Inf':<10} {'Time (s)':<10}")
    print("-" * 82)
    for v in variants:
        m = all_metrics[v]
        nan_inf = f"{'Y' if m['had_nan'] else 'N'}/{'Y' if m['had_inf'] else 'N'}"
        print(f"{v:<20} {m['n_params']:<10,} {m['best_val_ppl']:<10.2f} {m['final_val_ppl']:<10.2f} {m['mean_grad_norm_variance']:<12.6f} {nan_inf:<10} {m['training_time_seconds']:<10.1f}")

    n_criteria_pass = sum([criterion1, criterion2, criterion3, criterion4])
    if criterion1 and criterion3:
        decision = "PROCEED"
    elif criterion3 and ppl_diff_preln <= 2.0 and ppl_diff_periln <= 2.0:
        decision = "DEBUG"
    else:
        decision = "ABANDON"

    print(f"\nSuccess Criteria: {n_criteria_pass}/4 passed")
    print(f"Decision: {decision}")
    print(f"Total training time: {total_time:.1f}s")

    wandb.log({
        "final/preln_rmsnorm_ppl": preln_rmsnorm_ppl,
        "final/preln_dyt_ppl": preln_dyt_ppl,
        "final/periln_rmsnorm_ppl": periln_rmsnorm_ppl,
        "final/periln_dyt_ppl": periln_dyt_ppl,
        "final/ppl_diff_preln": ppl_diff_preln,
        "final/ppl_diff_periln": ppl_diff_periln,
        "success_criteria/ppl_match": criterion1,
        "success_criteria/periln_dyt_best_stability": criterion2,
        "success_criteria/no_nan_inf": criterion3,
        "success_criteria/activation_similarity": criterion4,
        "success_criteria/n_passed": n_criteria_pass,
        "decision": decision,
        "total_time_seconds": total_time,
    })

    # Save results
    results = {
        "all_metrics": {
            v: {k: val for k, val in m.items() if k not in ["activation_stats", "alpha_values"]}
            for v, m in all_metrics.items()
        },
        "criteria": {
            "criterion1_ppl_match": criterion1,
            "criterion1_preln": criterion1_preln,
            "criterion1_periln": criterion1_periln,
            "criterion2_best_stability": criterion2,
            "criterion3_no_nan_inf": criterion3,
            "criterion4_activation_similarity": criterion4,
        },
        "ppl_diffs": {"preln": ppl_diff_preln, "periln": ppl_diff_periln},
        "decision": decision,
        "total_time_seconds": total_time,
        "wandb_url": wandb_url,
    }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

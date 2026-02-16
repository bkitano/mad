"""Core training and evaluation functions for MVE 043."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm


IGNORE_INDEX = -100


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple:
    """
    Train for one epoch.

    Returns: (loss, accuracy, mean_orth_error, mean_raw_orth_error)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_orth_errors = []
    all_raw_orth_errors = []
    nan_detected = False

    for tokens, targets, masks, _ in tqdm(loader, desc="Training", leave=False):
        tokens = tokens.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(tokens, masks)

        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            nan_detected = True
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size

        preds_flat = logits_flat.argmax(dim=-1)
        valid_mask = targets_flat != IGNORE_INDEX
        correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
        total += valid_mask.sum().item()

        # Collect orthogonality errors
        if hasattr(model, 'get_orth_errors'):
            orth_errs = model.get_orth_errors()
            if orth_errs:
                all_orth_errors.extend(orth_errs)
        if hasattr(model, 'get_raw_orth_errors'):
            raw_errs = model.get_raw_orth_errors()
            if raw_errs:
                all_raw_orth_errors.extend(raw_errs)

    avg_loss = total_loss / (total / seq_len) if total > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0.0
    mean_orth = sum(all_orth_errors) / len(all_orth_errors) if all_orth_errors else 0.0
    mean_raw_orth = sum(all_raw_orth_errors) / len(all_raw_orth_errors) if all_raw_orth_errors else 0.0

    return avg_loss, avg_acc, mean_orth, mean_raw_orth, nan_detected


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """
    Evaluate model.

    Returns: (loss, accuracy, per_k_accuracy, mean_orth_error, mean_raw_orth_error)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    k_correct = {}
    k_total = {}
    all_orth_errors = []
    all_raw_orth_errors = []

    with torch.no_grad():
        for tokens, targets, masks, ks in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            logits = model(tokens, masks)
            batch_size, seq_len, num_classes = logits.shape

            logits_flat = logits.view(-1, num_classes)
            targets_flat = targets.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size

            preds_flat = logits_flat.argmax(dim=-1)
            valid_mask = targets_flat != IGNORE_INDEX
            correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
            total += valid_mask.sum().item()

            # Per-k accuracy
            preds = logits.argmax(dim=-1)
            for i, k in enumerate(ks):
                k_val = k.item() if isinstance(k, torch.Tensor) else k
                if k_val not in k_correct:
                    k_correct[k_val] = 0
                    k_total[k_val] = 0
                final_pos = k_val
                if targets[i, final_pos] != IGNORE_INDEX:
                    is_correct = (preds[i, final_pos] == targets[i, final_pos]).item()
                    k_correct[k_val] += is_correct
                    k_total[k_val] += 1

            # Collect orthogonality errors
            if hasattr(model, 'get_orth_errors'):
                orth_errs = model.get_orth_errors()
                if orth_errs:
                    all_orth_errors.extend(orth_errs)
            if hasattr(model, 'get_raw_orth_errors'):
                raw_errs = model.get_raw_orth_errors()
                if raw_errs:
                    all_raw_orth_errors.extend(raw_errs)

    k_acc = {k: k_correct[k] / k_total[k] if k_total[k] > 0 else 0.0 for k in sorted(k_correct.keys())}
    avg_loss = total_loss / (total / seq_len) if total > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0.0
    mean_orth = sum(all_orth_errors) / len(all_orth_errors) if all_orth_errors else 0.0
    mean_raw_orth = sum(all_raw_orth_errors) / len(all_raw_orth_errors) if all_raw_orth_errors else 0.0

    return avg_loss, avg_acc, k_acc, mean_orth, mean_raw_orth

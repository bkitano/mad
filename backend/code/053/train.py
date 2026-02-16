"""
Training script for Experiment 053: MLA-Inspired Latent State Compression.

Two phases:
1. Train a small GLA model on language modeling (synthetic data)
2. Analyze hidden state effective rank via SVD, measure readout error,
   and evaluate compressed inference perplexity

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.gla import GLAModel
from models.latent_state import LatentStateCompressor, CompressedGLAHead


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticLMDataset(Dataset):
    """
    Generates synthetic token sequences for language modeling.

    Creates sequences with local structure (repeated patterns, bigram dependencies)
    to give the GLA model something learnable. This avoids needing to download
    WikiText-2 while still testing the core hypothesis about state compressibility.

    The sequences have a mixture of:
    - Bigram patterns: P(next_token | current_token) is non-uniform
    - Trigram patterns: some 3-token motifs
    - Random noise: to prevent trivial memorization
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        rng = np.random.RandomState(seed)

        # Create a bigram transition matrix (sparse-ish)
        # Each token has ~5 likely successors
        self.transition = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        for i in range(vocab_size):
            # Pick 5 random successors with high probability
            successors = rng.choice(vocab_size, size=min(5, vocab_size), replace=False)
            probs = rng.dirichlet(np.ones(len(successors)) * 2.0)
            for s, p in zip(successors, probs):
                self.transition[i, s] = p * 0.8  # 80% from bigram
            # Distribute remaining 20% uniformly
            remaining = 0.2 / vocab_size
            self.transition[i] += remaining
            # Renormalize
            self.transition[i] /= self.transition[i].sum()

        # Generate all sequences
        self.sequences = []
        for _ in range(num_samples):
            seq = [rng.randint(0, vocab_size)]
            for t in range(seq_len - 1):
                next_token = rng.choice(vocab_size, p=self.transition[seq[-1]])
                seq.append(next_token)
            self.sequences.append(np.array(seq, dtype=np.int64))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(seq[1:], dtype=torch.long)
        return {'input_ids': input_ids, 'targets': targets}


# ============================================================================
# Phase 1: Train GLA Model
# ============================================================================

def train_gla(
    model: GLAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Train GLA model on language modeling task."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
    )

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_val_ppl = float('inf')

    print(f"\n{'='*60}")
    print(f"Phase 1: Training GLA Model")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"{'='*60}\n")

    for epoch in range(config['training']['max_epochs']):
        # Train
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            logits, _ = model(input_ids, return_states=False)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        scheduler.step()
        train_loss = total_loss / total_tokens
        train_ppl = math.exp(min(train_loss, 20))

        # Validate
        val_loss, val_ppl = evaluate_perplexity(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl

        # Log
        log_dict = {
            'phase1/train_loss': train_loss,
            'phase1/train_ppl': train_ppl,
            'phase1/val_loss': val_loss,
            'phase1/val_ppl': val_ppl,
            'phase1/best_val_ppl': best_val_ppl,
            'phase1/lr': scheduler.get_last_lr()[0],
            'epoch': epoch,
        }

        if HAS_WANDB and wandb.run is not None:
            wandb.log(log_dict)

        if epoch % 5 == 0 or epoch == config['training']['max_epochs'] - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} PPL: {train_ppl:.2f} | "
                  f"Val Loss: {val_loss:.4f} PPL: {val_ppl:.2f}")

        # Early stopping if we're doing well enough
        if val_ppl < config['training'].get('target_ppl', 10.0):
            print(f"Reached target PPL {val_ppl:.2f} at epoch {epoch}")
            break

    return {
        'final_train_loss': train_loss,
        'final_train_ppl': train_ppl,
        'best_val_loss': best_val_loss,
        'best_val_ppl': best_val_ppl,
    }


def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            if hasattr(model, 'forward'):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# ============================================================================
# Phase 2: SVD Analysis + Compressed Inference
# ============================================================================

def phase2_analysis(
    model: GLAModel,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict:
    """
    Phase 2: Analyze hidden states and evaluate compressed inference.

    Steps:
    1. Collect hidden states S_t from all layers/heads
    2. Compute SVD of empirical state covariance
    3. Report effective rank and energy distribution
    4. For each d_c, compute readout error and compressed inference perplexity
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: State Compression Analysis")
    print(f"{'='*60}\n")

    d_k = config['model']['d_k']
    d_v = config['model']['d_v']
    latent_dims = config['compression']['latent_dims']

    compressor = LatentStateCompressor(d_k, d_v, device)

    # Step 1: Collect states
    print("Step 1: Collecting hidden states...")
    t0 = time.time()
    collected_states = compressor.collect_states(
        model, val_loader,
        max_batches=config['compression'].get('max_collection_batches', 50),
    )
    print(f"  Collected states from {len(collected_states)} (layer, head) pairs in {time.time()-t0:.1f}s")

    results = {
        'svd_analysis': {},
        'readout_errors': {},
        'compressed_ppl': {},
    }

    # Step 2: SVD analysis per (layer, head)
    print("\nStep 2: SVD analysis of hidden state covariance...")
    for key, states in collected_states.items():
        layer_idx, head_idx = key
        print(f"\n  Layer {layer_idx}, Head {head_idx}:")

        svd_result = compressor.analyze_effective_rank(
            states,
            top_k_values=latent_dims + [d_k],  # also compute for full d_k
        )

        print(f"    Effective rank: {svd_result['effective_rank']:.1f} / {min(d_k, d_v)}")
        print(f"    Energy by top-k SVs:")
        for k, energy in svd_result['energy_by_k'].items():
            print(f"      top-{k}: {energy:.4f} ({energy*100:.1f}%)")

        results['svd_analysis'][f"L{layer_idx}_H{head_idx}"] = {
            'effective_rank': svd_result['effective_rank'],
            'energy_by_k': {str(k): v for k, v in svd_result['energy_by_k'].items()},
            'num_samples': svd_result['num_samples'],
        }

        # Log to wandb
        if HAS_WANDB and wandb.run is not None:
            for k, energy in svd_result['energy_by_k'].items():
                wandb.log({
                    f'svd/L{layer_idx}_H{head_idx}/energy_top_{k}': energy,
                })
            wandb.log({
                f'svd/L{layer_idx}_H{head_idx}/effective_rank': svd_result['effective_rank'],
            })

        # Step 3: Readout error for each d_c
        print(f"\n    Readout errors by d_c:")
        for d_c in latent_dims:
            W_down, W_up = compressor.create_compression_matrices(svd_result, d_c)

            error_result = compressor.compute_readout_error(
                model, val_loader, W_down, W_up,
                max_batches=config['compression'].get('max_error_batches', 10),
            )

            rel_error = error_result['relative_readout_error']
            print(f"      d_c={d_c:3d}: relative readout error = {rel_error:.6f} ({rel_error*100:.3f}%)")

            results['readout_errors'][f"L{layer_idx}_H{head_idx}_dc{d_c}"] = {
                'relative_readout_error': rel_error,
                'num_readouts': error_result['num_readouts'],
                'd_c': d_c,
            }

            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    f'readout_error/L{layer_idx}_H{head_idx}/dc_{d_c}': rel_error,
                })

            # Check per-timestep error trend (does it grow over time?)
            per_t = error_result['per_timestep_error']
            if len(per_t) > 1:
                timesteps = sorted(per_t.keys())
                first_half = np.mean([per_t[t] for t in timesteps[:len(timesteps)//2]])
                second_half = np.mean([per_t[t] for t in timesteps[len(timesteps)//2:]])
                growth_ratio = second_half / max(first_half, 1e-10)
                print(f"              error growth (2nd half / 1st half): {growth_ratio:.2f}x")

                results['readout_errors'][f"L{layer_idx}_H{head_idx}_dc{d_c}"]['error_growth_ratio'] = growth_ratio

    # Step 4: Compressed inference perplexity
    print(f"\n\nStep 3: Compressed inference perplexity...")
    full_val_loss, full_val_ppl = evaluate_perplexity(model, val_loader, device)
    print(f"  Full model perplexity: {full_val_ppl:.2f}")

    results['full_val_ppl'] = full_val_ppl
    results['full_val_loss'] = full_val_loss

    # For each d_c, create a compressed model and evaluate perplexity
    for d_c in latent_dims:
        print(f"\n  d_c = {d_c}:")

        compressed_ppl = evaluate_compressed_perplexity(
            model, val_loader, collected_states, compressor, d_c, device
        )

        ppl_diff = compressed_ppl - full_val_ppl
        print(f"    Compressed PPL: {compressed_ppl:.2f} (delta: {ppl_diff:+.2f})")

        results['compressed_ppl'][f"dc_{d_c}"] = {
            'ppl': compressed_ppl,
            'ppl_delta': ppl_diff,
        }

        if HAS_WANDB and wandb.run is not None:
            wandb.log({
                f'compressed_ppl/dc_{d_c}': compressed_ppl,
                f'compressed_ppl_delta/dc_{d_c}': ppl_diff,
            })

    # Step 5: Latency comparison (simplified - not kernel-optimized)
    print(f"\n\nStep 4: Latency comparison (single-step)...")
    latency_results = measure_latency(model, d_k, d_v, latent_dims, device, config)
    results['latency'] = latency_results

    return results


def evaluate_compressed_perplexity(
    model: GLAModel,
    val_loader: DataLoader,
    collected_states: Dict,
    compressor: LatentStateCompressor,
    d_c: int,
    device: torch.device,
) -> float:
    """
    Evaluate perplexity using compressed inference.

    For each layer/head, compute SVD-based compression matrices,
    then run inference using compressed state recurrence.
    """
    model.eval()

    # Get SVD results and compression matrices for all heads
    compression_info = {}
    for key, states in collected_states.items():
        svd_result = compressor.analyze_effective_rank(states, top_k_values=[d_c])
        W_down, W_up = compressor.create_compression_matrices(svd_result, d_c)
        compression_info[key] = (W_down, W_up, svd_result)

    # Run compressed inference
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            batch_size, seq_len = input_ids.shape

            # Run the model with compression applied
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = model.embedding(input_ids) + model.pos_embedding(positions)

            for layer_idx, layer in enumerate(model.layers):
                residual = x
                x_normed = layer.norm1(x)

                head_outputs = []
                for head_idx, head in enumerate(layer.heads):
                    key = (layer_idx, head_idx)
                    if key in compression_info:
                        W_down, W_up, _ = compression_info[key]
                        # Run compressed recurrence
                        head_out = run_compressed_head(
                            head, x_normed, W_down, W_up, d_c, device
                        )
                    else:
                        head_out, _ = head(x_normed, return_states=False)
                    head_outputs.append(head_out)

                concat = torch.cat(head_outputs, dim=-1)
                attn_out = layer.W_o(concat)
                x = residual + layer.dropout(attn_out)

                residual = x
                x = layer.norm2(x)
                x = residual + layer.ffn(x)

            x = model.norm(x)
            logits = model.lm_head(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return ppl


def run_compressed_head(
    head: nn.Module,
    x: torch.Tensor,
    W_down: torch.Tensor,
    W_up: torch.Tensor,
    d_c: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a single GLA head with compressed state recurrence.

    c_t = A_t^c @ c_{t-1} + W_down @ vec(k_t v_t^T)
    where A_t^c = sum_j alpha_{t,j} * M_j

    Readout: o_t = q_t^T @ reshape(W_up @ c_t)
    """
    batch, seq_len, _ = x.shape
    d_k = head.d_k
    d_v = head.d_v

    q = head.W_q(x)
    k = head.W_k(x)
    v = head.W_v(x)
    alpha = torch.sigmoid(head.W_alpha(x))

    # Precompute M_j matrices: M_j = W_down[:, j*d_v:(j+1)*d_v] @ W_up[j*d_v:(j+1)*d_v, :]
    M = torch.zeros(d_k, d_c, d_c, device=device, dtype=x.dtype)
    for j in range(d_k):
        start = j * d_v
        end = (j + 1) * d_v
        M[j] = W_down[:, start:end] @ W_up[start:end, :]

    # Initialize compressed state
    c = torch.zeros(batch, d_c, device=device, dtype=x.dtype)

    outputs = []

    for t in range(seq_len):
        q_t = q[:, t, :]
        k_t = k[:, t, :]
        v_t = v[:, t, :]
        alpha_t = alpha[:, t, :]

        # Compressed transition: A_t^c = sum_j alpha_{t,j} * M_j
        A_t = torch.einsum('bj,jcd->bcd', alpha_t, M)  # (batch, d_c, d_c)

        # Input term: W_down @ vec(k_t v_t^T) = W_down @ (k_t kron v_t)
        kv = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(-2))  # (batch, d_k, d_v)
        kv_flat = kv.reshape(batch, -1)  # (batch, d_k * d_v)
        b_t = (W_down @ kv_flat.T).T  # (batch, d_c)

        # State update
        c = torch.bmm(A_t, c.unsqueeze(-1)).squeeze(-1) + b_t

        # Readout: decompress then read
        S_approx_flat = (W_up @ c.T).T  # (batch, d_k * d_v)
        S_approx = S_approx_flat.reshape(batch, d_k, d_v)
        o_t = torch.bmm(q_t.unsqueeze(-2), S_approx).squeeze(-2)

        outputs.append(o_t)

    return torch.stack(outputs, dim=1)


def measure_latency(
    model: GLAModel,
    d_k: int,
    d_v: int,
    latent_dims: List[int],
    device: torch.device,
    config: Dict,
) -> Dict:
    """
    Measure per-step latency for full vs compressed inference.

    Note: This is a rough comparison without kernel optimization.
    The key metric is whether the compressed path is faster even
    in pure PyTorch (which would underestimate the improvement from
    optimized kernels).
    """
    results = {}
    batch_size = 1
    d_model = config['model']['d_model']
    n_warmup = 20
    n_measure = 100

    model.eval()

    # Full inference latency (single recurrent step)
    with torch.no_grad():
        # Prepare input
        x_t = torch.randn(batch_size, d_model, device=device)
        S = torch.randn(batch_size, d_k, d_v, device=device)
        head = model.layers[0].heads[0]

        # Warmup
        for _ in range(n_warmup):
            head.recurrent_step(x_t, S)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_measure):
            _, S_new, _ = head.recurrent_step(x_t, S)
            S = S_new
        if device.type == 'cuda':
            torch.cuda.synchronize()
        full_latency_us = (time.time() - t0) / n_measure * 1e6

    results['full_step_latency_us'] = full_latency_us
    print(f"  Full step latency: {full_latency_us:.1f} us")

    # Compressed inference latency for each d_c
    for d_c in latent_dims:
        with torch.no_grad():
            # Create dummy compression matrices
            W_down = torch.randn(d_c, d_k * d_v, device=device) / math.sqrt(d_k * d_v)
            W_up = torch.randn(d_k * d_v, d_c, device=device) / math.sqrt(d_c)

            # Precompute M_j
            M = torch.zeros(d_k, d_c, d_c, device=device)
            for j in range(d_k):
                start = j * d_v
                end = (j + 1) * d_v
                M[j] = W_down[:, start:end] @ W_up[start:end, :]

            c = torch.randn(batch_size, d_c, device=device)
            alpha_t = torch.rand(batch_size, d_k, device=device)
            k_t = torch.randn(batch_size, d_k, device=device)
            v_t = torch.randn(batch_size, d_v, device=device)
            q_t = torch.randn(batch_size, d_k, device=device)

            # Warmup
            for _ in range(n_warmup):
                A_t = torch.einsum('bj,jcd->bcd', alpha_t, M)
                kv = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
                kv_flat = kv.reshape(batch_size, -1)
                b_t = (W_down @ kv_flat.T).T
                c_new = torch.bmm(A_t, c.unsqueeze(-1)).squeeze(-1) + b_t
                S_flat = (W_up @ c_new.T).T
                S_approx = S_flat.reshape(batch_size, d_k, d_v)
                o_t = torch.bmm(q_t.unsqueeze(-2), S_approx).squeeze(-2)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(n_measure):
                A_t = torch.einsum('bj,jcd->bcd', alpha_t, M)
                kv = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
                kv_flat = kv.reshape(batch_size, -1)
                b_t = (W_down @ kv_flat.T).T
                c = torch.bmm(A_t, c.unsqueeze(-1)).squeeze(-1) + b_t
                S_flat = (W_up @ c.T).T
                S_approx = S_flat.reshape(batch_size, d_k, d_v)
                o_t = torch.bmm(q_t.unsqueeze(-2), S_approx).squeeze(-2)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            compressed_latency_us = (time.time() - t0) / n_measure * 1e6

        speedup = full_latency_us / max(compressed_latency_us, 1e-3)
        results[f'dc_{d_c}_latency_us'] = compressed_latency_us
        results[f'dc_{d_c}_speedup'] = speedup
        print(f"  d_c={d_c:3d} step latency: {compressed_latency_us:.1f} us (speedup: {speedup:.2f}x)")

        if HAS_WANDB and wandb.run is not None:
            wandb.log({
                f'latency/dc_{d_c}_us': compressed_latency_us,
                f'latency/dc_{d_c}_speedup': speedup,
            })

    if HAS_WANDB and wandb.run is not None:
        wandb.log({'latency/full_step_us': full_latency_us})

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 053: MLA Latent State Compression")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="mad-architecture-search",
            name="exp-053-mla-latent-state-compression",
            config=config,
        )
        wandb_url = wandb.run.get_url()
        print(f"Wandb URL: {wandb_url}")
    else:
        print("WARNING: wandb not available, logging to stdout only")

    # Create datasets
    print("\nCreating synthetic datasets...")
    train_dataset = SyntheticLMDataset(
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        num_samples=config['data']['train_samples'],
        seed=42,
    )
    val_dataset = SyntheticLMDataset(
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        num_samples=config['data']['val_samples'],
        seed=123,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Vocab: {config['data']['vocab_size']}, Seq len: {config['data']['seq_len']}")

    # Create model
    model = GLAModel(
        vocab_size=config['data']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_k=config['model']['d_k'],
        d_v=config['model']['d_v'],
        dropout=config['model']['dropout'],
    ).to(device)

    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"State size per head: {config['model']['d_k']} x {config['model']['d_v']} = "
          f"{config['model']['d_k'] * config['model']['d_v']:,} elements")

    # Phase 1: Train GLA model
    phase1_start = time.time()
    phase1_results = train_gla(model, train_loader, val_loader, config, device)
    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 completed in {phase1_time:.1f}s")
    print(f"Best validation PPL: {phase1_results['best_val_ppl']:.2f}")

    # Phase 2: State compression analysis
    phase2_start = time.time()
    phase2_results = phase2_analysis(model, val_loader, config, device)
    phase2_time = time.time() - phase2_start
    print(f"\nPhase 2 completed in {phase2_time:.1f}s")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 053 SUMMARY")
    print(f"{'='*60}")

    print(f"\nPhase 1 (Training):")
    print(f"  Best val PPL: {phase1_results['best_val_ppl']:.2f}")
    print(f"  Time: {phase1_time:.1f}s")

    print(f"\nPhase 2 (Compression Analysis):")

    # SVD Analysis
    print(f"\n  SVD Effective Rank Analysis:")
    for key, info in phase2_results['svd_analysis'].items():
        print(f"    {key}: effective rank = {info['effective_rank']:.1f}")
        for k, energy in info['energy_by_k'].items():
            print(f"      top-{k}: {float(energy)*100:.1f}% energy")

    # Readout Errors
    print(f"\n  Readout Errors (relative):")
    for key, info in phase2_results['readout_errors'].items():
        err = info['relative_readout_error']
        growth = info.get('error_growth_ratio', 'N/A')
        print(f"    {key}: {err*100:.3f}% (growth: {growth})")

    # Compressed PPL
    print(f"\n  Compressed Inference Perplexity (full PPL: {phase2_results['full_val_ppl']:.2f}):")
    for key, info in phase2_results['compressed_ppl'].items():
        print(f"    {key}: PPL = {info['ppl']:.2f} (delta: {info['ppl_delta']:+.2f})")

    # Latency
    if 'latency' in phase2_results:
        print(f"\n  Latency:")
        lat = phase2_results['latency']
        print(f"    Full step: {lat.get('full_step_latency_us', 0):.1f} us")
        for d_c in config['compression']['latent_dims']:
            dc_lat = lat.get(f'dc_{d_c}_latency_us', 0)
            dc_speedup = lat.get(f'dc_{d_c}_speedup', 0)
            print(f"    d_c={d_c}: {dc_lat:.1f} us ({dc_speedup:.2f}x speedup)")

    # ============================================================
    # Success Criteria Evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    # Criterion 1: Effective rank << min(d_k, d_v)
    min_dk_dv = min(config['model']['d_k'], config['model']['d_v'])
    effective_ranks = [info['effective_rank'] for info in phase2_results['svd_analysis'].values()]
    avg_effective_rank = np.mean(effective_ranks)
    crit1_pass = avg_effective_rank < 0.5 * min_dk_dv

    # Check if top-16 SVs capture >90% energy
    energy_16_values = []
    for info in phase2_results['svd_analysis'].values():
        if '16' in info['energy_by_k']:
            energy_16_values.append(float(info['energy_by_k']['16']))
    avg_energy_16 = np.mean(energy_16_values) if energy_16_values else 0.0
    crit1b_pass = avg_energy_16 > 0.90

    # Criterion 2: d_c=32 readout error < 5%
    dc32_errors = [
        info['relative_readout_error']
        for key, info in phase2_results['readout_errors'].items()
        if '_dc32' in key
    ]
    avg_dc32_error = np.mean(dc32_errors) if dc32_errors else 1.0
    crit2_pass = avg_dc32_error < 0.05

    # Criterion 3: Compressed PPL within 2 points
    dc32_ppl_info = phase2_results['compressed_ppl'].get('dc_32', {})
    dc32_ppl_delta = dc32_ppl_info.get('ppl_delta', float('inf'))
    crit3_pass = abs(dc32_ppl_delta) < 2.0

    # Criterion 4: Latency decrease
    full_lat = phase2_results.get('latency', {}).get('full_step_latency_us', 1.0)
    dc32_lat = phase2_results.get('latency', {}).get('dc_32_latency_us', full_lat)
    crit4_pass = dc32_lat < full_lat

    status = lambda p: "PASS" if p else "FAIL"

    print(f"\n  1. Effective rank << min(d_k, d_v)={min_dk_dv}:")
    print(f"     Avg effective rank: {avg_effective_rank:.1f} [{status(crit1_pass)}]")
    print(f"     Top-16 SVs capture >90% energy: {avg_energy_16*100:.1f}% [{status(crit1b_pass)}]")
    print(f"\n  2. d_c=32 readout error < 5%:")
    print(f"     Avg readout error: {avg_dc32_error*100:.3f}% [{status(crit2_pass)}]")
    print(f"\n  3. Compressed PPL within 2 points (d_c=32):")
    print(f"     PPL delta: {dc32_ppl_delta:+.2f} [{status(crit3_pass)}]")
    print(f"\n  4. Latency decrease (d_c=32):")
    print(f"     Full: {full_lat:.1f} us, Compressed: {dc32_lat:.1f} us [{status(crit4_pass)}]")

    overall_pass = crit1_pass and crit2_pass and crit3_pass
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    # Log final results to wandb
    if HAS_WANDB and wandb.run is not None:
        wandb.log({
            'final/phase1_best_val_ppl': phase1_results['best_val_ppl'],
            'final/avg_effective_rank': avg_effective_rank,
            'final/avg_energy_top16': avg_energy_16,
            'final/avg_dc32_readout_error': avg_dc32_error,
            'final/dc32_ppl_delta': dc32_ppl_delta,
            'final/full_latency_us': full_lat,
            'final/dc32_latency_us': dc32_lat,
            'success_criteria/effective_rank_low': crit1_pass,
            'success_criteria/top16_energy_90pct': crit1b_pass,
            'success_criteria/dc32_readout_error_lt_5pct': crit2_pass,
            'success_criteria/compressed_ppl_within_2': crit3_pass,
            'success_criteria/latency_decrease': crit4_pass,
            'success_criteria/overall': overall_pass,
        })
        wandb.finish()

    # Save results to file
    results_summary = {
        'phase1': phase1_results,
        'phase2': {
            'svd_analysis': phase2_results['svd_analysis'],
            'readout_errors': {
                k: {sk: sv for sk, sv in v.items() if sk != 'per_timestep_error'}
                for k, v in phase2_results['readout_errors'].items()
            },
            'compressed_ppl': phase2_results['compressed_ppl'],
            'full_val_ppl': phase2_results['full_val_ppl'],
            'latency': phase2_results.get('latency', {}),
        },
        'success_criteria': {
            'effective_rank_low': crit1_pass,
            'top16_energy_90pct': crit1b_pass,
            'dc32_readout_error_lt_5pct': crit2_pass,
            'compressed_ppl_within_2': crit3_pass,
            'latency_decrease': crit4_pass,
            'overall': overall_pass,
        },
        'timing': {
            'phase1_seconds': phase1_time,
            'phase2_seconds': phase2_time,
            'total_seconds': phase1_time + phase2_time,
        },
    }

    results_path = Path(__file__).parent / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results_summary


if __name__ == '__main__':
    main()

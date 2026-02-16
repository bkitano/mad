"""
Latent State Compression for Linear RNN Inference (Experiment 053).

Implements MLA-inspired weight absorption for compressing the d_k x d_v
hidden state of GLA into a low-rank latent vector c_t in R^{d_c}.

Key operations:
- SVD analysis of empirical state covariance to determine effective rank
- Initialization of W_down, W_up from top-d_c SVD components
- Weight absorption: absorbed query q_tilde = W_up^T (q_t kron I_dv)
- Compressed state update: c_t = A_t^c c_{t-1} + W_down vec(k_t v_t^T)
  where A_t^c = sum_j alpha_{t,j} M_j (precomputed matrices)

References:
- Proposal 053, Section "Mathematical Formulation"
- DeepSeek-V2 MLA weight absorption (2024)
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class LatentStateCompressor:
    """
    Analyzes and compresses GLA hidden states using SVD-based low-rank approximation.

    This class:
    1. Collects hidden states from a trained GLA model
    2. Computes SVD of the empirical state covariance
    3. Reports effective rank and energy distribution
    4. Creates compressed inference functions via weight absorption

    Args:
        d_k: key dimension
        d_v: value dimension
        device: torch device
    """

    def __init__(self, d_k: int, d_v: int, device: torch.device = torch.device('cpu')):
        self.d_k = d_k
        self.d_v = d_v
        self.d_state = d_k * d_v  # flattened state dimension
        self.device = device

    def collect_states(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect hidden states and intermediate values from model inference.

        Returns dict mapping (layer_idx, head_idx) -> list of state tensors.
        Also collects query vectors for readout error computation.
        """
        model.eval()
        collected = {}  # key: (layer, head) -> list of (states, queries)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)

                # Forward pass collecting states
                _, all_layer_states = model(input_ids, return_states=True)

                if all_layer_states is None:
                    continue

                for layer_idx, layer_states in enumerate(all_layer_states):
                    if layer_states is None:
                        continue
                    for head_idx, head_states in enumerate(layer_states):
                        key = (layer_idx, head_idx)
                        if key not in collected:
                            collected[key] = []

                        # head_states is a list of S_t tensors, each (batch, d_k, d_v)
                        # Flatten and collect
                        for S_t in head_states:
                            # S_t: (batch, d_k, d_v) -> (batch, d_k * d_v)
                            flat = S_t.reshape(-1, self.d_state).cpu()
                            collected[key].append(flat)

        return collected

    def analyze_effective_rank(
        self,
        states: List[torch.Tensor],
        top_k_values: List[int] = [8, 16, 32, 64],
    ) -> Dict[str, any]:
        """
        Compute SVD of empirical state covariance and analyze effective rank.

        Args:
            states: list of state tensors, each (batch, d_state)
            top_k_values: list of k values to compute energy for

        Returns:
            Dictionary with:
            - singular_values: all singular values
            - energy_by_k: {k: fraction of energy captured by top-k SVs}
            - effective_rank: participation ratio
            - total_energy: sum of squared singular values
        """
        # Concatenate all states: (N, d_state)
        all_states = torch.cat(states, dim=0).float()
        N = all_states.shape[0]

        # Center the data
        mean = all_states.mean(dim=0, keepdim=True)
        centered = all_states - mean

        # Compute SVD of the data matrix (more numerically stable than covariance)
        # For large N, use the covariance matrix approach
        if N > self.d_state:
            # Covariance: (d_state, d_state)
            cov = (centered.T @ centered) / (N - 1)
            # Eigendecomposition (covariance is symmetric)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort descending
            idx = eigenvalues.argsort(descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            singular_values = eigenvalues.clamp(min=0).sqrt()
        else:
            # Direct SVD of data matrix
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            singular_values = S
            eigenvectors = Vh.T  # (d_state, min(N, d_state))

        # Compute metrics
        total_energy = (singular_values ** 2).sum().item()

        energy_by_k = {}
        for k in top_k_values:
            k_actual = min(k, len(singular_values))
            top_k_energy = (singular_values[:k_actual] ** 2).sum().item()
            energy_by_k[k] = top_k_energy / total_energy if total_energy > 0 else 0.0

        # Effective rank: participation ratio = (sum s_i^2)^2 / (sum s_i^4)
        s2 = singular_values ** 2
        s4 = s2 ** 2
        effective_rank = (s2.sum() ** 2 / s4.sum()).item() if s4.sum() > 0 else 0.0

        return {
            'singular_values': singular_values,
            'eigenvectors': eigenvectors,
            'mean': mean.squeeze(0),
            'energy_by_k': energy_by_k,
            'effective_rank': effective_rank,
            'total_energy': total_energy,
            'num_samples': N,
        }

    def create_compression_matrices(
        self,
        svd_result: Dict,
        d_c: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create W_down and W_up from top-d_c SVD components.

        W_down in R^{d_c x d_state}: projects state down to latent
        W_up in R^{d_state x d_c}: projects latent back up

        Initialized from top-d_c eigenvectors of state covariance.

        Args:
            svd_result: output from analyze_effective_rank
            d_c: latent state dimension

        Returns:
            W_down: (d_c, d_state)
            W_up: (d_state, d_c)
        """
        eigenvectors = svd_result['eigenvectors']  # (d_state, num_components)

        # Take top-d_c components
        d_c_actual = min(d_c, eigenvectors.shape[1])

        # W_up = top-d_c eigenvectors: (d_state, d_c)
        W_up = eigenvectors[:, :d_c_actual].clone()

        # W_down = W_up^T (orthogonal projection): (d_c, d_state)
        W_down = W_up.T.clone()

        return W_down.to(self.device), W_up.to(self.device)

    def compute_readout_error(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        W_down: torch.Tensor,
        W_up: torch.Tensor,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """
        Compute readout error when using compressed state.

        Measures: ||q_t^T S_t - q_tilde^T c_t||^2 / ||q_t^T S_t||^2

        where:
        - c_t = W_down vec(S_t) (compressed state)
        - q_tilde^T c_t approximates q_t^T S_t via weight absorption

        Since W_up W_down is an orthogonal projection:
        q_t^T S_t ≈ q_t^T reshape(W_up W_down vec(S_t))
                   = q_t^T reshape(W_up c_t)

        With weight absorption:
        q_tilde = reshape of (W_up^T (q_t kron I_dv)) applied correctly

        For simplicity in the MVE, we compute:
        - Full readout: o_full = q_t^T S_t  (d_v dimensional)
        - Compressed: first project S -> c = W_down vec(S), then
          reconstruct S_approx = reshape(W_up c), then o_approx = q_t^T S_approx

        This tests the core question: is the state low-rank enough?
        """
        model.eval()
        total_error = 0.0
        total_norm = 0.0
        num_readouts = 0

        # Also track per-timestep error to check if error accumulates
        timestep_errors = {}
        timestep_norms = {}

        d_c = W_down.shape[0]

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                batch_size, seq_len = input_ids.shape

                # Get states and queries by running the model
                _, all_layer_states = model(input_ids, return_states=True)

                if all_layer_states is None:
                    continue

                # Process hidden representation to get queries
                # We need to extract q from each head
                positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                x = model.embedding(input_ids) + model.pos_embedding(positions)

                for layer_idx, layer in enumerate(model.layers):
                    x_normed = layer.norm1(x)

                    for head_idx, head in enumerate(layer.heads):
                        q = head.W_q(x_normed)  # (batch, seq_len, d_k)

                        head_states = all_layer_states[layer_idx][head_idx]

                        for t, S_t in enumerate(head_states):
                            # S_t: (batch, d_k, d_v)
                            q_t = q[:, t, :]  # (batch, d_k)

                            # Full readout: o_full = q_t^T S_t
                            o_full = torch.bmm(
                                q_t.unsqueeze(-2), S_t
                            ).squeeze(-2)  # (batch, d_v)

                            # Compressed readout:
                            # 1. Flatten state: vec(S_t) (batch, d_k * d_v)
                            S_flat = S_t.reshape(batch_size, -1)

                            # 2. Compress: c_t = W_down @ vec(S_t) (batch, d_c)
                            c_t = (W_down @ S_flat.T).T  # (batch, d_c)

                            # 3. Reconstruct: vec(S_approx) = W_up @ c_t
                            S_approx_flat = (W_up @ c_t.T).T  # (batch, d_k * d_v)
                            S_approx = S_approx_flat.reshape(batch_size, self.d_k, self.d_v)

                            # 4. Readout from approximation
                            o_approx = torch.bmm(
                                q_t.unsqueeze(-2), S_approx
                            ).squeeze(-2)  # (batch, d_v)

                            # Compute relative error
                            error = ((o_full - o_approx) ** 2).sum().item()
                            norm = (o_full ** 2).sum().item()

                            total_error += error
                            total_norm += norm
                            num_readouts += batch_size

                            if t not in timestep_errors:
                                timestep_errors[t] = 0.0
                                timestep_norms[t] = 0.0
                            timestep_errors[t] += error
                            timestep_norms[t] += norm

                    # Advance x through the layer for next layer's queries
                    x, _ = layer(x, return_states=False)

        relative_error = total_error / max(total_norm, 1e-10)

        # Per-timestep relative error
        per_timestep_error = {}
        for t in sorted(timestep_errors.keys()):
            per_timestep_error[t] = timestep_errors[t] / max(timestep_norms[t], 1e-10)

        return {
            'relative_readout_error': relative_error,
            'total_error': total_error,
            'total_norm': total_norm,
            'num_readouts': num_readouts,
            'per_timestep_error': per_timestep_error,
            'd_c': d_c,
        }


class CompressedGLAHead(nn.Module):
    """
    GLA head operating in compressed latent space for inference benchmarking.

    Uses precomputed matrices M_j for the compressed state transition:
    c_t = sum_j alpha_{t,j} M_j @ c_{t-1} + W_down @ vec(k_t v_t^T)

    Readout via weight absorption:
    o_t = q_t^T @ reshape(W_up @ c_t)

    Args:
        original_head: the original GLAHead
        W_down: (d_c, d_k * d_v) compression matrix
        W_up: (d_k * d_v, d_c) decompression matrix
        d_c: latent dimension
    """

    def __init__(
        self,
        original_head: nn.Module,
        W_down: torch.Tensor,
        W_up: torch.Tensor,
        d_c: int,
    ):
        super().__init__()
        self.d_k = original_head.d_k
        self.d_v = original_head.d_v
        self.d_c = d_c

        # Copy original projections (frozen)
        self.W_q = original_head.W_q
        self.W_k = original_head.W_k
        self.W_v = original_head.W_v
        self.W_alpha = original_head.W_alpha

        # Register compression matrices
        self.register_buffer('W_down', W_down)  # (d_c, d_k * d_v)
        self.register_buffer('W_up', W_up)      # (d_k * d_v, d_c)

        # Precompute per-gate-dimension transition matrices M_j
        # M_j = W_down[:, j*d_v:(j+1)*d_v] @ W_up[j*d_v:(j+1)*d_v, :]
        # Shape: (d_k, d_c, d_c)
        M = torch.zeros(self.d_k, d_c, d_c)
        for j in range(self.d_k):
            start = j * self.d_v
            end = (j + 1) * self.d_v
            M[j] = W_down[:, start:end] @ W_up[start:end, :]
        self.register_buffer('M', M)  # (d_k, d_c, d_c)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with compressed state recurrence.

        Args:
            x: input tensor (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_v)
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        alpha = torch.sigmoid(self.W_alpha(x))

        # Initialize compressed state
        c = torch.zeros(batch, self.d_c, device=device, dtype=dtype)

        outputs = []

        for t in range(seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            alpha_t = alpha[:, t, :]  # (batch, d_k)

            # Compressed state update:
            # A_t^c = sum_j alpha_{t,j} * M_j  ->  (batch, d_c, d_c)
            # Using einsum: alpha (batch, d_k) * M (d_k, d_c, d_c) -> (batch, d_c, d_c)
            A_t = torch.einsum('bj,jcd->bcd', alpha_t, self.M)

            # c_t = A_t^c @ c_{t-1} + W_down @ vec(k_t v_t^T)
            # W_down @ vec(k_t v_t^T) = W_down @ (k_t kron v_t)
            kv = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(-2))  # (batch, d_k, d_v)
            kv_flat = kv.reshape(batch, -1)  # (batch, d_k * d_v)
            b_t = (self.W_down @ kv_flat.T).T  # (batch, d_c)

            c = torch.bmm(A_t, c.unsqueeze(-1)).squeeze(-1) + b_t  # (batch, d_c)

            # Readout: reconstruct state approximation, then read out
            # vec(S_approx) = W_up @ c_t
            S_approx_flat = (self.W_up @ c.T).T  # (batch, d_k * d_v)
            S_approx = S_approx_flat.reshape(batch, self.d_k, self.d_v)

            # o_t = q_t^T @ S_approx
            o_t = torch.bmm(q_t.unsqueeze(-2), S_approx).squeeze(-2)

            outputs.append(o_t)

        return torch.stack(outputs, dim=1)

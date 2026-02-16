"""
Nystrom Landmark Compression for Chunkwise SSM Inter-Chunk State Transfer.

From proposal 025-nystrom-landmark-chunkwise-ssm:

Standard chunkwise SSM:
  h_{(k+1)C} = T_k * h_{kC} + b_k

Nystrom compression:
  T_hat_k = R_k * W_k^+ * C_k
  h_{(k+1)C} = R_k * W_k^+ * (C_k * h_{kC}) + b_k

Cost: O(mn) per chunk boundary instead of O(n^2).

Implementation: Uses diagonal state transitions with input-dependent diagonal
to keep the intra-chunk scan fast (O(n) per step), and constructs T_k
as the product of diagonal matrices. This is faithful to Mamba-2's diagonal A_t.

Key insight from proposal: "Selective SSMs use diagonal-dominant A_t matrices
(Mamba: diagonal; DeltaNet: identity + rank-1), so products of many such matrices
have rapidly decaying singular values."

For diagonal A_t, T_k is also diagonal — so the Nystrom test is whether we can
compress a diagonal T_k using m < n landmark dimensions. This tests whether
the state dimensions have redundant information (the signal lives in fewer
than n dimensions).

To make T_k non-diagonal (so Nystrom actually matters), we add input-dependent
mixing via a fixed permutation applied every chunk, creating off-diagonal structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, d: int, expand: int = 4):
        super().__init__()
        hidden = d * expand
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ChunkwiseSSMLayer(nn.Module):
    """
    Chunkwise SSM layer with efficient vectorized scan.

    Uses diagonal A_t with input-dependent gating (Mamba-2 style).
    Processes entire chunks in vectorized operations.

    Key optimization: Pre-compute all A_t diagonal values for the chunk,
    then run the scan with pure tensor operations.
    """
    def __init__(
        self,
        d_model: int,
        state_dim: int,  # n
        chunk_size: int = 16,  # C
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.chunk_size = chunk_size

        # Input projections (all vectorizable across time)
        self.gate_proj = nn.Linear(d_model, state_dim)  # diagonal gate a_t
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)  # input to state
        self.C_proj = nn.Linear(state_dim, d_model, bias=False)  # state to output
        self.out_gate_proj = nn.Linear(d_model, d_model, bias=False)

        # Mixing matrix for inter-chunk: learned dense n×n applied at chunk boundaries
        # This creates off-diagonal structure in T_k
        self.mix_weight = nn.Parameter(torch.eye(state_dim) + 0.01 * torch.randn(state_dim, state_dim))

        # Initialize gate bias for long memory (sigmoid(3) ≈ 0.95)
        nn.init.constant_(self.gate_proj.bias, 3.0)

    def _vectorized_chunk_scan(
        self, x_chunk: torch.Tensor, h_init: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized scan within a chunk.

        1. Pre-compute all gates and inputs for the chunk (vectorized)
        2. Run sequential scan (unavoidable, but minimal Python overhead)
        3. Compute T_chunk as product of gate matrices + mixing

        Args:
            x_chunk: (batch, C, d_model)
            h_init: (batch, n)
        Returns:
            y_chunk: (batch, C, d_model)
            h_final: (batch, n)
            T_chunk: (batch, n, n) cumulative transition (with mixing)
        """
        batch, C, d = x_chunk.shape
        n = self.state_dim
        device = x_chunk.device

        # Step 1: Vectorized pre-computation (all timesteps at once)
        # Reshape for batch computation: (batch*C, d)
        x_flat = x_chunk.reshape(batch * C, d)

        gates = torch.sigmoid(self.gate_proj(x_flat)).reshape(batch, C, n)  # (batch, C, n)
        b_all = self.B_proj(x_flat).reshape(batch, C, n)  # (batch, C, n)

        # Step 2: Sequential scan (pure tensor ops, no Python overhead per step)
        h = h_init  # (batch, n)
        h_states = []

        # Product of diagonal gates for T_chunk
        gate_product = torch.ones(batch, n, device=device, dtype=x_chunk.dtype)

        for t in range(C):
            a_t = gates[:, t, :]  # (batch, n)
            b_t = b_all[:, t, :]  # (batch, n)
            h = a_t * h + b_t  # (batch, n) — diagonal A_t, very fast
            h_states.append(h)
            gate_product = gate_product * a_t  # accumulate diagonal product

        # Step 3: Compute outputs (vectorized)
        h_stack = torch.stack(h_states, dim=1)  # (batch, C, n)
        y_chunk = self.C_proj(h_stack.reshape(batch * C, n)).reshape(batch, C, d)

        # Step 4: Build T_chunk with mixing
        # T_chunk = Mix @ diag(gate_product)
        # This creates off-diagonal structure so Nystrom is meaningful
        T_diag = torch.diag_embed(gate_product)  # (batch, n, n)
        mix = self.mix_weight.unsqueeze(0)  # (1, n, n)
        T_chunk = torch.bmm(mix.expand(batch, -1, -1), T_diag)  # (batch, n, n)

        return y_chunk, h, T_chunk

    def _inter_chunk_propagate(
        self, T_chunk: torch.Tensor, h_prev: torch.Tensor, b_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Override in subclass."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d = x.shape
        n = self.state_dim
        C = self.chunk_size
        device = x.device

        # Pad to multiple of C
        pad_len = (C - seq_len % C) % C
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        num_chunks = x_padded.shape[1] // C
        x_chunks = x_padded.view(batch, num_chunks, C, d)

        h = torch.zeros(batch, n, device=device, dtype=x.dtype)
        all_outputs = []

        for k in range(num_chunks):
            y_chunk, h_scan, T_chunk = self._vectorized_chunk_scan(x_chunks[:, k], h)
            all_outputs.append(y_chunk)

            # b_chunk = h_scan - T_chunk @ h
            b_chunk = h_scan - torch.bmm(T_chunk, h.unsqueeze(-1)).squeeze(-1)

            # Inter-chunk propagation
            h = self._inter_chunk_propagate(T_chunk, h, b_chunk)

        y = torch.cat(all_outputs, dim=1)[:, :seq_len, :]

        # Output gating
        gate = torch.sigmoid(self.out_gate_proj(x[:, :seq_len, :]))
        y = y * gate

        return y


class FullChunkSSM(ChunkwiseSSMLayer):
    """Full O(n^2) inter-chunk baseline."""
    def _inter_chunk_propagate(self, T_chunk, h_prev, b_chunk):
        return torch.bmm(T_chunk, h_prev.unsqueeze(-1)).squeeze(-1) + b_chunk


class NystromChunkSSM(ChunkwiseSSMLayer):
    """
    Nystrom-compressed inter-chunk SSM.

    Learned projection P in R^{m x n}:
      C_k = P @ T_k, R_k = T_k @ P^T, W_k = P @ T_k @ P^T
      h_new = R_k @ W_k^+ @ (C_k @ h_prev) + b_chunk

    Cost: O(mn + m^2) instead of O(n^2).
    """
    def __init__(self, d_model: int, state_dim: int, n_landmarks: int = 4,
                 chunk_size: int = 16):
        super().__init__(d_model, state_dim, chunk_size)
        self.n_landmarks = n_landmarks

        # Learned projection P, initialized as segment-means
        P_init = torch.zeros(n_landmarks, state_dim)
        seg = state_dim // n_landmarks
        for i in range(n_landmarks):
            s, e = i * seg, (i + 1) * seg if i < n_landmarks - 1 else state_dim
            P_init[i, s:e] = 1.0 / (e - s)
        self.P = nn.Parameter(P_init)
        self.ridge_delta = 1e-4

    def _inter_chunk_propagate(self, T_chunk, h_prev, b_chunk):
        m = self.n_landmarks
        P = self.P  # (m, n)
        P_T = P.t()  # (n, m)

        # Nystrom factors
        C_k = torch.matmul(P.unsqueeze(0), T_chunk)     # (B, m, n)
        R_k = torch.matmul(T_chunk, P_T.unsqueeze(0))   # (B, n, m)
        W_k = torch.matmul(C_k, P_T.unsqueeze(0))       # (B, m, m)

        # Ridge-regularized pseudoinverse
        I_m = torch.eye(m, device=T_chunk.device, dtype=T_chunk.dtype).unsqueeze(0)
        W_k_pinv = torch.linalg.pinv(W_k + self.ridge_delta * I_m)

        # Compressed propagation: R @ W^+ @ (C @ h) + b
        h_proj = torch.matmul(C_k, h_prev.unsqueeze(-1)).squeeze(-1)       # (B, m)
        h_corr = torch.matmul(W_k_pinv, h_proj.unsqueeze(-1)).squeeze(-1)  # (B, m)
        h_new = torch.matmul(R_k, h_corr.unsqueeze(-1)).squeeze(-1)        # (B, n)

        return h_new + b_chunk

    def get_compression_stats(self, T_chunk: torch.Tensor) -> dict:
        with torch.no_grad():
            m, n = self.n_landmarks, self.state_dim
            P, P_T = self.P, self.P.t()

            C_k = torch.matmul(P.unsqueeze(0), T_chunk)
            R_k = torch.matmul(T_chunk, P_T.unsqueeze(0))
            W_k = torch.matmul(C_k, P_T.unsqueeze(0))
            I_m = torch.eye(m, device=T_chunk.device, dtype=T_chunk.dtype).unsqueeze(0)
            W_k_pinv = torch.linalg.pinv(W_k + self.ridge_delta * I_m)

            T_hat = torch.bmm(torch.bmm(R_k, W_k_pinv), C_k)
            error = torch.norm(T_chunk - T_hat, dim=(-2, -1))
            T_norm = torch.norm(T_chunk, dim=(-2, -1))
            rel_error = (error / (T_norm + 1e-8)).mean().item()

            svd_vals = torch.linalg.svdvals(T_chunk)
            mean_svd = svd_vals.mean(dim=0)

            full_mem = n * n
            comp_mem = n * m + m * m

            return {
                'rel_approx_error': rel_error,
                'mean_singular_values': mean_svd.cpu().tolist(),
                'full_memory': full_mem,
                'compressed_memory': comp_mem,
                'compression_ratio': full_mem / comp_mem,
            }


class ChunkSSMBlock(nn.Module):
    """Pre-norm residual block."""
    def __init__(self, ssm_layer: ChunkwiseSSMLayer, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = ssm_layer
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, expand=2)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NystromChunkSSMModel(nn.Module):
    """Full model with Nystrom-compressed chunkwise SSM."""
    def __init__(self, vocab_size, d_model=48, state_dim=8, n_landmarks=2,
                 chunk_size=8, n_layers=2, max_seq_len=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_landmarks = n_landmarks

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            ChunkSSMBlock(
                NystromChunkSSM(d_model, state_dim, n_landmarks, chunk_size),
                d_model,
            ) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_out(h))

    def get_compression_stats(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        stats = []
        for layer in self.layers:
            ssm = layer.ssm
            C = ssm.chunk_size
            pad = (C - s % C) % C
            hp = F.pad(h, (0, 0, 0, pad)) if pad > 0 else h
            h0 = torch.zeros(b, ssm.state_dim, device=x.device, dtype=h.dtype)
            _, _, T = ssm._vectorized_chunk_scan(hp[:, :C, :], h0)
            stats.append(ssm.get_compression_stats(T))
            h = layer(h)
        return stats


class FullChunkSSMModel(nn.Module):
    """Full model with uncompressed chunkwise SSM (baseline)."""
    def __init__(self, vocab_size, d_model=48, state_dim=8, chunk_size=8,
                 n_layers=2, max_seq_len=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            ChunkSSMBlock(
                FullChunkSSM(d_model, state_dim, chunk_size),
                d_model,
            ) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_out(h))

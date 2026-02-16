"""
GLA (Gated Linear Attention) Baseline

GLA uses diagonal gating with NO delta rule:
    S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
    o_t = S_t^T @ q_t

This is the simplest recurrent baseline - just gated accumulation without
any state correction. Expected to achieve ~70% on MQAR with 16 pairs
because it cannot overwrite stale associations.

Reference: Yang et al. (2024) "Gated Linear Attention Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GLALayer(nn.Module):
    """Single GLA layer with multi-head gated linear attention."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head

        # Input projections
        self.W_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_v = nn.Linear(d_model, self.d_inner, bias=False)

        # Per-channel decay gate: alpha_t = sigmoid(W_alpha @ x_t) in [0,1]^{d_k}
        self.W_alpha = nn.Linear(d_model, self.d_inner, bias=True)

        # Output projection
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        # Initialize decay gate bias to produce alpha ~ 0.95
        nn.init.constant_(self.W_alpha.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        B, T, _ = x.shape

        # Project to multi-head Q, K, V
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Per-channel decay gate
        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, H, T, d)

        # Recurrent scan: S_t = diag(alpha_t) * S_{t-1} + k_t @ v_t^T
        # o_t = S_t^T @ q_t
        output = self._recurrent_scan(q, k, v, alpha)  # (B, H, T, d)

        # Merge heads and project
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_inner)
        output = self.W_o(output)
        output = self.dropout(output)

        return output + residual

    def _recurrent_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recurrent scan for GLA.

        S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T    (Proposal eq: no delta rule)
        o_t = S_t^T @ q_t

        Args:
            q, k, v: (B, H, T, d)
            alpha: (B, H, T, d) per-channel decay
        Returns:
            output: (B, H, T, d)
        """
        B, H, T, d = q.shape

        # State: S in R^{d_k x d_v}  (here d_k = d_v = d)
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        for t in range(T):
            # Decay: diag(alpha_t) @ S_{t-1}
            # alpha[:,:,t,:] is (B, H, d), multiply each row of S
            S = alpha[:, :, t, :].unsqueeze(-1) * S  # (B, H, d, d)

            # Accumulate: + k_norm @ v_t^T (normalize key for fair comparison)
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]  # (B, H, d)
            k_t_norm = F.normalize(k_t, dim=-1)
            S = S + k_t_norm.unsqueeze(-1) * v_t.unsqueeze(-2)  # outer product

            # Output: o_t = S^T @ q_t
            q_t = q[:, :, t, :]  # (B, H, d)
            o_t = torch.einsum("bhji,bhj->bhi", S, q_t)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2)  # (B, H, T, d)


class GLAModel(nn.Module):
    """Full GLA model with embedding, GLA layers, and classification head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            GLALayer(d_model, n_heads, d_head, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.embedding(x) + self.pos_encoding(positions)

        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        logits = self.head(h)
        return logits

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

"""
KDA (Kimi Delta Attention) Baseline

KDA uses a constrained DPLR state transition with the delta rule:
    S_t = (I - beta_t * k_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T
    o_t = S_t^T @ q_t

The delta rule removes the old value associated with k_t before writing the new one.
This enables key-value overwriting but the removal direction is always the current key k_t.

Expected: ~85% on MQAR with 16 pairs (better than GLA but limited by myopic removal).

Reference: Kimi Team (2025) "Kimi Linear"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KDALayer(nn.Module):
    """Single KDA layer with multi-head delta attention."""

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

        # Scalar beta gate: beta_t = sigmoid(W_beta @ x_t) in [0,1]
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)

        # Output projection
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        # Initialize decay gate bias for alpha ~ 0.95
        nn.init.constant_(self.W_alpha.bias, 3.0)
        # Initialize beta gate bias for beta ~ 0.5
        nn.init.constant_(self.W_beta.bias, 0.0)

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
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Per-channel decay
        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, H, T, d)

        # Scalar beta per head
        beta = torch.sigmoid(self.W_beta(x))  # (B, T, H)
        beta = beta.transpose(1, 2)  # (B, H, T)

        # Recurrent scan with delta rule
        output = self._recurrent_scan(q, k, v, alpha, beta)

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
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recurrent scan for KDA with delta rule.

        S_t = (I - beta_t * k_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T
        o_t = S_t^T @ q_t

        The removal term (I - beta_t * k_t @ k_t^T) erases the component of S
        along the current key direction before writing the new association.

        Args:
            q, k, v: (B, H, T, d)
            alpha: (B, H, T, d) per-channel decay
            beta: (B, H, T) scalar learning rate gate
        Returns:
            output: (B, H, T, d)
        """
        B, H, T, d = q.shape

        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            alpha_t = alpha[:, :, t, :]  # (B, H, d)
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Normalize key for stable delta rule: (I - k@k^T) is projector only when ||k||=1
            k_t_norm = F.normalize(k_t, dim=-1)  # (B, H, d), unit norm

            # Step 1: Decay - diag(alpha_t) @ S_{t-1}
            S = alpha_t.unsqueeze(-1) * S  # (B, H, d, d)

            # Step 2: Delta rule removal - (I - beta_t * k_norm @ k_norm^T) @ S
            # Equivalent to: S = S - beta_t * k_norm @ (k_norm^T @ S)
            k_t_proj = torch.einsum("bhd,bhde->bhe", k_t_norm, S)  # k_norm^T @ S: (B, H, d)
            S = S - beta_t * k_t_norm.unsqueeze(-1) * k_t_proj.unsqueeze(-2)

            # Step 3: Write new association - + beta_t * k_norm @ v_t^T
            S = S + beta_t * k_t_norm.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Step 4: Output - o_t = S^T @ q_t
            o_t = torch.einsum("bhji,bhj->bhi", S, q_t)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2)


class KDAModel(nn.Module):
    """Full KDA model with embedding, KDA layers, and classification head."""

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
            KDALayer(d_model, n_heads, d_head, dropout)
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

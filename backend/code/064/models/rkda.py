"""
Residual KDA: Auxiliary Error-Correcting State with Channel-Wise Decay

Three variants for MVE comparison:
1. KDA: Standard Kimi Delta Attention with channel-wise decay (baseline)
2. KDA + Scalar Residual (RDN-style): KDA base + auxiliary residual state with scalar decay
3. RKDA (proposed): KDA base + auxiliary residual state with channel-wise decay

Mathematical formulation (from proposal 064):

Base KDA recurrence:
    S_t = (I - beta_t * k_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T
    o_t^base = S_t^T @ q_t

Residual computation:
    r_t = Clip[-c, c](v_t - S_{t-1}^T @ k_t)     (prediction error)

Auxiliary residual state (RKDA - channel-wise decay):
    R_t = (I - gamma_t * k_t @ k_t^T) * diag(alpha_t^R) * R_{t-1} + gamma_t * r_t @ k_t^T
    o_t^residual = R_t^T @ q_t

Combined output:
    o_t = o_t^base + o_t^residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KDALayer(nn.Module):
    """
    KDA layer with optional residual fitting.

    Supports three modes:
    - 'kda': Standard KDA (no residual)
    - 'scalar_residual': KDA + scalar-decay residual (RDN-style)
    - 'channel_residual': KDA + channel-wise-decay residual (RKDA, proposed)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        residual_mode: str = "none",  # "none", "scalar", "channel"
        clip_threshold: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.residual_mode = residual_mode
        self.clip_threshold = clip_threshold

        # Input projections
        self.W_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_v = nn.Linear(d_model, self.d_inner, bias=False)

        # Per-channel decay gate for primary state:
        # alpha_t = sigmoid(W_alpha @ x_t) in [0,1]^{d_k}
        self.W_alpha = nn.Linear(d_model, self.d_inner, bias=True)

        # Scalar beta gate: beta_t = sigmoid(W_beta @ x_t) in [0,1]
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)

        # Residual-specific parameters
        if residual_mode == "scalar":
            # Scalar decay for residual state: gamma_R_t = sigmoid(w_gamma_R @ x_t) in [0,1]
            self.W_gamma = nn.Linear(d_model, n_heads, bias=True)
            # Single scalar decay for residual state per head
            self.W_alpha_R_scalar = nn.Linear(d_model, n_heads, bias=True)
        elif residual_mode == "channel":
            # Scalar correction strength: gamma_t = sigmoid(w_gamma @ x_t) in [0,1]
            self.W_gamma = nn.Linear(d_model, n_heads, bias=True)
            # Per-channel decay for residual state (separate from primary):
            # alpha_t^R = sigmoid(W_alpha_R @ x_t) in [0,1]^{d_k}
            self.W_alpha_R = nn.Linear(d_model, self.d_inner, bias=True)

        # Output projection
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        # Initialize primary decay gate bias for alpha ~ 0.95
        nn.init.constant_(self.W_alpha.bias, 3.0)
        # Initialize beta gate bias for beta ~ 0.5
        nn.init.constant_(self.W_beta.bias, 0.0)

        if self.residual_mode == "scalar":
            nn.init.constant_(self.W_gamma.bias, 0.0)
            nn.init.constant_(self.W_alpha_R_scalar.bias, 3.0)
        elif self.residual_mode == "channel":
            nn.init.constant_(self.W_gamma.bias, 0.0)
            nn.init.constant_(self.W_alpha_R.bias, 3.0)

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

        # Per-channel decay for primary state
        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, H, T, d)

        # Scalar beta per head
        beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2)  # (B, H, T)

        # Compute residual gates if needed
        gamma = None
        alpha_R = None
        if self.residual_mode == "scalar":
            gamma = torch.sigmoid(self.W_gamma(x)).transpose(1, 2)  # (B, H, T) scalar
            # Scalar decay expanded to all channels (uniform)
            alpha_R_scalar = torch.sigmoid(self.W_alpha_R_scalar(x)).transpose(1, 2)  # (B, H, T)
            alpha_R = alpha_R_scalar.unsqueeze(-1).expand_as(alpha)  # (B, H, T, d) - uniform
        elif self.residual_mode == "channel":
            gamma = torch.sigmoid(self.W_gamma(x)).transpose(1, 2)  # (B, H, T)
            alpha_R = torch.sigmoid(
                self.W_alpha_R(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            )  # (B, H, T, d) - per-channel

        # Recurrent scan
        output = self._recurrent_scan(q, k, v, alpha, beta, gamma, alpha_R)

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
        gamma: torch.Tensor | None,
        alpha_R: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Recurrent scan implementing KDA with optional residual fitting.

        Base KDA (proposal eq.):
            S_t = (I - beta_t * k_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T
            o_t^base = S_t^T @ q_t

        Residual (if enabled):
            r_t = Clip(v_t - S_{t-1}^T @ k_t)
            R_t = (I - gamma_t * k_t @ k_t^T) * diag(alpha_t^R) * R_{t-1} + gamma_t * r_t @ k_t^T
            o_t = o_t^base + R_t^T @ q_t

        Args:
            q, k, v: (B, H, T, d)
            alpha: (B, H, T, d) per-channel decay for primary state
            beta: (B, H, T) scalar learning rate gate
            gamma: (B, H, T) residual correction strength (None if no residual)
            alpha_R: (B, H, T, d) decay for residual state (None if no residual)
        Returns:
            output: (B, H, T, d)
        """
        B, H, T, d = q.shape
        use_residual = self.residual_mode != "none"

        # Primary state S: (B, H, d_k, d_v) = (B, H, d, d)
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        # Auxiliary residual state R: (B, H, d_k, d_v)
        R = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype) if use_residual else None

        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            alpha_t = alpha[:, :, t, :]  # (B, H, d)
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Normalize key for stable delta rule
            k_t_norm = F.normalize(k_t, dim=-1)  # (B, H, d)

            # === PRIMARY STATE UPDATE ===

            # Compute residual r_t BEFORE updating S (using S_{t-1})
            if use_residual:
                # r_t = Clip(v_t - S_{t-1}^T @ k_t)  (proposal eq.)
                # S^T @ k = (d_v x d_k)^T @ (d_k,) = (d_v,)
                predicted = torch.einsum("bhji,bhj->bhi", S, k_t_norm)  # (B, H, d)
                r_t = v_t - predicted
                r_t = torch.clamp(r_t, -self.clip_threshold, self.clip_threshold)

            # Step 1: Decay - diag(alpha_t) @ S_{t-1}
            S = alpha_t.unsqueeze(-1) * S  # (B, H, d, d)

            # Step 2: Delta rule removal - (I - beta_t * k_norm @ k_norm^T) @ S
            k_t_proj = torch.einsum("bhd,bhde->bhe", k_t_norm, S)  # (B, H, d)
            S = S - beta_t * k_t_norm.unsqueeze(-1) * k_t_proj.unsqueeze(-2)

            # Step 3: Write new association - + beta_t * k_norm @ v_t^T
            S = S + beta_t * k_t_norm.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Step 4: Base output - o_t^base = S^T @ q_t
            o_t = torch.einsum("bhji,bhj->bhi", S, q_t)  # (B, H, d)

            # === RESIDUAL STATE UPDATE (if enabled) ===
            if use_residual:
                gamma_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
                alpha_R_t = alpha_R[:, :, t, :]  # (B, H, d)

                # Step R1: Decay residual state - diag(alpha_R_t) @ R_{t-1}
                R = alpha_R_t.unsqueeze(-1) * R

                # Step R2: Delta rule removal for residual state
                k_t_proj_R = torch.einsum("bhd,bhde->bhe", k_t_norm, R)
                R = R - gamma_t * k_t_norm.unsqueeze(-1) * k_t_proj_R.unsqueeze(-2)

                # Step R3: Write residual - + gamma_t * k_norm @ r_t^T
                R = R + gamma_t * k_t_norm.unsqueeze(-1) * r_t.unsqueeze(-2)

                # Step R4: Residual output - o_t^residual = R^T @ q_t
                o_t_residual = torch.einsum("bhji,bhj->bhi", R, q_t)

                # Combined output (proposal eq.)
                o_t = o_t + o_t_residual

            outputs.append(o_t)

        return torch.stack(outputs, dim=2)


class RKDAModel(nn.Module):
    """
    Full RKDA model with embedding, KDA/RKDA layers, and classification head.

    Supports three variants via residual_mode:
    - "none": Standard KDA (baseline)
    - "scalar": KDA + scalar-decay residual (RDN-style)
    - "channel": KDA + channel-wise-decay residual (RKDA, proposed)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_layers: int,
        residual_mode: str = "none",
        clip_threshold: float = 1.0,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            KDALayer(d_model, n_heads, d_head, residual_mode, clip_threshold, dropout)
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

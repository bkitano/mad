"""
MFA-GLA: GLA with MFA-Style Shared Latent Projections

Replaces independent per-head Q, K, V projections with:

1. Shared down-projections (Proposal "Mathematical Formulation"):
   q_latent = X @ S_q    (S_q in R^{d x C})
   k_latent = X @ S_k    (S_k in R^{d x C})
   v_latent = X @ S_v    (S_v in R^{d x C})

2. Per-head differentiation:
   Q^(h) = q_latent @ Q_c^(h)    (Q_c^(h) in R^{C x C}, rotation in latent space)
   K^(h) = k_latent               (shared across all heads - latent MQA)
   V^(h) = v_latent @ V_c^(h)    (V_c^(h) in R^{C x d_v})

State update (same as GLA, but with MFA-projected Q/K/V):
   S_t^(h) = diag(alpha_t) * S_{t-1}^(h) + k_t * v_t^(h)^T
   o_t^(h) = S_t^(h)^T @ q_t^(h)

Key dimensions:
   - State S^(h) in R^{C x d_v}: key dim is C (latent), value dim is d_v
   - q^(h) in R^C: must match key dim for state readout
   - k in R^C: shared across heads, used directly from latent space
   - v^(h) in R^{d_v}: per-head differentiated values

FLOP analysis (per token, per layer):
   Standard (n=2, d_k=32): d*n*(2*d_k+d_v) = 128*2*96 = 24,576
   MFA (m=4, C=64, d_v=32): 3*d*C + m*(C*C + C*d_v) = 24,576 + 4*(4096+2048) = 49,152
   But MFA has 4 heads vs 2 → 2x TER, testing quality at higher head count

Reference: MFA (ACL 2025, arXiv:2412.19255) - validated at 7B on softmax attention.
           First application to linear RNNs (this experiment).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFAGLALayer(nn.Module):
    """Single GLA layer with MFA-style shared latent projections.

    Architecture (Proposal "Mathematical Formulation"):
    - Shared down-projections: S_q, S_k, S_v in R^{d x C}
    - Per-head query rotations: Q_c^(h) in R^{C x C} (rotate in latent space)
    - Per-head value projections: V_c^(h) in R^{C x d_v}
    - Keys shared across heads (latent MQA) in R^C
    - Decay gate in R^C per head (key dimension)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,     # m in the proposal (MFA heads, typically > standard heads)
        d_head: int,      # d_v per head (value dimension)
        latent_dim: int,  # C in the proposal (shared latent / key dimension)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads      # m MFA heads
        self.d_head = d_head        # d_v
        self.latent_dim = latent_dim  # C (also serves as d_k for state)
        self.d_inner = n_heads * d_head  # output concat dim

        # ---- Shared down-projections: S_q, S_k, S_v in R^{d x C} ----
        self.S_q = nn.Linear(d_model, latent_dim, bias=False)
        self.S_k = nn.Linear(d_model, latent_dim, bias=False)
        self.S_v = nn.Linear(d_model, latent_dim, bias=False)

        # ---- Per-head rotation/projection matrices ----
        # Q_c^(h) in R^{C x C}: rotates q in latent space (q and k must match dim C)
        self.Q_c = nn.Parameter(torch.empty(n_heads, latent_dim, latent_dim))
        # V_c^(h) in R^{C x d_v}: projects v from latent to per-head value dim
        self.V_c = nn.Parameter(torch.empty(n_heads, latent_dim, d_head))

        # ---- Per-channel decay gate ----
        # Gate in key dimension (C) per head for state decay
        self.W_alpha = nn.Linear(d_model, n_heads * latent_dim, bias=True)

        # ---- Output projection ----
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.S_q, self.S_k, self.S_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        # Initialize per-head matrices
        for h in range(self.n_heads):
            nn.init.xavier_uniform_(self.Q_c.data[h])
            nn.init.xavier_uniform_(self.V_c.data[h])
        # Initialize decay gate bias for alpha ~ 0.95
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

        # Step 1: Shared down-projections (single GEMM each)
        # q_latent, k_latent, v_latent in R^{B x T x C}
        q_latent = self.S_q(x)  # (B, T, C)
        k_latent = self.S_k(x)  # (B, T, C)
        v_latent = self.S_v(x)  # (B, T, C)

        # Step 2: Per-head query rotation: Q^(h) = q_latent @ Q_c^(h), Q_c in R^{C x C}
        # Result: q in R^{B x m x T x C} (stays in latent space for state readout)
        q = torch.einsum("btc,hce->bhte", q_latent, self.Q_c)  # (B, H, T, C)

        # Step 3: Keys shared across all heads (latent MQA)
        # k in R^{B x T x C} -> broadcast to (B, H, T, C)
        k = k_latent.unsqueeze(1).expand(B, self.n_heads, T, self.latent_dim)

        # Step 4: Per-head value projection: V^(h) = v_latent @ V_c^(h), V_c in R^{C x d_v}
        v = torch.einsum("btc,hcd->bhtd", v_latent, self.V_c)  # (B, H, T, d_v)

        # Step 5: Per-channel decay gate in key dimension (C)
        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.latent_dim).transpose(1, 2)
        )  # (B, H, T, C)

        # Step 6: Recurrent scan with state S in R^{C x d_v}
        output = self._recurrent_scan(q, k, v, alpha)  # (B, H, T, d_v)

        # Merge heads and project
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_inner)
        output = self.W_o(output)
        output = self.dropout(output)

        return output + residual

    def _recurrent_scan(self, q, k, v, alpha):
        """
        MFA-GLA recurrent scan.

        State S^(h) in R^{C x d_v}:
            S_t^(h) = diag(alpha_t)_C * S_{t-1}^(h) + k_t * v_t^(h)^T
            o_t^(h) = S_t^(h)^T @ q_t^(h)

        where:
            q_t^(h) in R^C (rotated query in latent space)
            k_t in R^C (shared key in latent space)
            v_t^(h) in R^{d_v} (per-head value)
            alpha_t in R^C (per-channel key-dim decay)

        Args:
            q: (B, H, T, C) - per-head rotated queries
            k: (B, H, T, C) - shared keys (same across heads)
            v: (B, H, T, d_v) - per-head values
            alpha: (B, H, T, C) - per-channel decay in key dim
        Returns:
            output: (B, H, T, d_v)
        """
        B, H, T, C = q.shape
        d_v = v.shape[-1]

        S = torch.zeros(B, H, C, d_v, device=q.device, dtype=q.dtype)
        outputs = []

        for t in range(T):
            # Decay in key dimension: diag(alpha_t) @ S_{t-1}
            S = alpha[:, :, t, :].unsqueeze(-1) * S  # (B, H, C, d_v)

            # Accumulate: + k_t_norm @ v_t^T
            k_t = F.normalize(k[:, :, t, :], dim=-1)  # (B, H, C)
            v_t = v[:, :, t, :]                         # (B, H, d_v)
            S = S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # (B, H, C, d_v)

            # Output: o_t = S^T @ q_t  →  (d_v, C) @ (C,) = (d_v,)
            q_t = q[:, :, t, :]  # (B, H, C)
            o_t = torch.einsum("bhcv,bhc->bhv", S, q_t)  # (B, H, d_v)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2)  # (B, H, T, d_v)


class MFAGLAModel(nn.Module):
    """Full GLA model with MFA-style shared latent projections.

    Key differences from standard GLA:
    - Shared S_q, S_k, S_v down-projections (3 matrices instead of n*3)
    - Per-head Q_c (C x C) and V_c (C x d_v) for differentiation
    - Keys shared across heads (latent MQA)
    - State S in R^{C x d_v} (key dim is latent C, not d_k)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,     # m MFA heads (more than standard)
        d_head: int,      # d_v per head
        latent_dim: int,  # C (shared latent dimension / key dim)
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            MFAGLALayer(d_model, n_heads, d_head, latent_dim, dropout)
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
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.head(h)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

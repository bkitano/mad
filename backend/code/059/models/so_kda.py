"""
SO-KDA (Second-Order KDA): Augmenting Delta Attention with HLA's Key Metric

Proposal 059: Second-Order KDA

The key innovation: replace KDA's rank-1 removal (beta_t * k_t @ k_t^T) with
an adapted removal using the running key covariance M_t:

    M_t = gamma_M * M_{t-1} + k_t @ k_t^T    (Key metric - running covariance)
    k_tilde_t = M_t @ k_t / ||M_t @ k_t||     (Adapted removal direction)
    S_t = (I - beta_t * k_tilde_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T

The adapted removal direction k_tilde weights erasure by historical key importance:
if many past keys pointed in a similar direction, M_t amplifies removal along that
subspace. This is like using Mahalanobis distance for the delta rule correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SOKDALayer(nn.Module):
    """Single SO-KDA layer with second-order key metric adapted delta rule."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        gamma_m_init: float = 0.99,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.eps = eps

        # Input projections
        self.W_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_v = nn.Linear(d_model, self.d_inner, bias=False)

        # Per-channel decay gate: alpha_t = sigmoid(W_alpha @ x_t) in [0,1]^{d_k}
        self.W_alpha = nn.Linear(d_model, self.d_inner, bias=True)

        # Scalar beta gate: beta_t = sigmoid(W_beta @ x_t) in [0,1]
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)

        # Key metric decay: learnable per-head scalar gamma_M
        # Use inverse sigmoid parameterization so gamma_M = sigmoid(raw_gamma_m)
        raw_init = math.log(gamma_m_init / (1.0 - gamma_m_init))
        self.raw_gamma_m = nn.Parameter(torch.full((n_heads,), raw_init))

        # Output projection
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm + residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(self.W_alpha.bias, 3.0)
        nn.init.constant_(self.W_beta.bias, 0.0)

    @property
    def gamma_m(self):
        """Learnable per-head key metric decay in (0, 1)."""
        return torch.sigmoid(self.raw_gamma_m)

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

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        alpha = torch.sigmoid(
            self.W_alpha(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )

        beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2)  # (B, H, T)

        output = self._recurrent_scan(q, k, v, alpha, beta)

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
        Recurrent scan for SO-KDA with second-order key metric.

        Key metric update (Proposal eq):
            M_t = gamma_M * M_{t-1} + k_t @ k_t^T

        Adapted removal direction (Proposal eq):
            k_tilde_t = M_t @ k_t / (||M_t @ k_t|| + eps)

        State update (Proposal eq):
            S_t = (I - beta_t * k_tilde_t @ k_t^T) * diag(alpha_t) * S_{t-1} + beta_t * k_t @ v_t^T

        Output:
            o_t = S_t^T @ q_t

        Args:
            q, k, v: (B, H, T, d)
            alpha: (B, H, T, d)
            beta: (B, H, T)
        Returns:
            output: (B, H, T, d)
        """
        B, H, T, d = q.shape

        # Initialize states
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        # Key metric M initialized to eps * I for stability (Proposal: "Mitigation")
        M = self.eps * torch.eye(d, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).clone()

        gamma_m = self.gamma_m  # (H,)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            alpha_t = alpha[:, :, t, :]
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Normalize key for stable delta rule
            k_t_norm = F.normalize(k_t, dim=-1)  # (B, H, d), unit norm

            # --- Key Metric Update ---
            # M_t = gamma_M * M_{t-1} + k_norm @ k_norm^T
            # Using normalized keys so M_t is a proper correlation matrix
            gamma_m_expanded = gamma_m.view(1, H, 1, 1)
            M = gamma_m_expanded * M + k_t_norm.unsqueeze(-1) * k_t_norm.unsqueeze(-2)  # (B, H, d, d)

            # --- Adapted Removal Direction ---
            # k_tilde = M @ k_norm / (||M @ k_norm|| + eps)
            Mk = torch.einsum("bhij,bhj->bhi", M, k_t_norm)  # (B, H, d)
            Mk_norm = Mk.norm(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, H, 1)
            k_tilde = Mk / Mk_norm  # (B, H, d)

            # --- State Update with Adapted Delta Rule ---
            # Step 1: Decay
            S = alpha_t.unsqueeze(-1) * S

            # Step 2: Adapted removal - S = S - beta_t * k_tilde @ (k_norm^T @ S)
            k_t_proj = torch.einsum("bhd,bhde->bhe", k_t_norm, S)  # k_norm^T @ S: (B, H, d)
            S = S - beta_t * k_tilde.unsqueeze(-1) * k_t_proj.unsqueeze(-2)

            # Step 3: Write (use normalized key for consistent scale)
            S = S + beta_t * k_t_norm.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Step 4: Output
            o_t = torch.einsum("bhji,bhj->bhi", S, q_t)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2)


class SOKDAModel(nn.Module):
    """Full SO-KDA model with embedding, SO-KDA layers, and classification head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_layers: int,
        gamma_m_init: float = 0.99,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            SOKDALayer(d_model, n_heads, d_head, gamma_m_init, dropout)
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

    def get_gamma_m_values(self) -> dict:
        """Return current gamma_M values for each layer/head."""
        result = {}
        for i, layer in enumerate(self.layers):
            gamma = layer.gamma_m.detach().cpu().tolist()
            result[f"layer_{i}"] = gamma
        return result

"""
Baseline models for comparison with Log-Semiring SSM.

1. LinearAttentionClassifier — RetNet-style linear attention with (R, +, ×) semiring.
   Uses φ(x) = elu(x) + 1 feature map. No softmax → cannot do sharp retrieval.

2. DiagonalSSMClassifier — Mamba-style diagonal SSM with input-dependent gating.
   Standard semiring: h_t = a_t * h_{t-1} + b_t * x_t
   Can gate but lacks attention-like content matching.

Both should fail on selective copying (proposal success criteria):
- Linear attention: < 60% accuracy (diffuse attention)
- Diagonal SSM: < 70% accuracy (no content-based retrieval)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Linear Attention (RetNet-style)
# =============================================================================

class LinearAttentionLayer(nn.Module):
    """Causal linear attention with ELU+1 feature map.

    Recurrence (standard semiring):
        S_t = λ_t * S_{t-1} + φ(k_t) v_t^T
        y_t = φ(q_t) S_t

    where φ(x) = elu(x) + 1 (ensures positivity).

    This is O(d²) per step — efficient but approximate.
    The feature map cannot reproduce exact softmax patterns.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Input-dependent decay (RetNet-style)
        self.W_gate = nn.Linear(d_model, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map: φ(x) = elu(x) + 1, ensures positivity."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head)

        # Apply feature map
        q = self._feature_map(q * self.scale)
        k = self._feature_map(k)

        # Input-dependent decay λ ∈ (0, 1)
        gate = torch.sigmoid(self.W_gate(x))  # (batch, seq_len, n_heads)

        # Sequential recurrence: S_t = λ_t S_{t-1} + φ(k_t) v_t^T
        # S is (batch, n_heads, d_head, d_head) — the KV state
        S = torch.zeros(batch, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            k_t = k[:, t, :, :]  # (batch, n_heads, d_head)
            v_t = v[:, t, :, :]  # (batch, n_heads, d_head)
            q_t = q[:, t, :, :]  # (batch, n_heads, d_head)
            g_t = gate[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)

            # Update state: S_t = λ S_{t-1} + k_t v_t^T
            kv = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # (batch, n_heads, d_head, d_head)
            S = g_t * S + kv

            # Query: y_t = q_t^T S_t → (batch, n_heads, d_head)
            y_t = torch.einsum('bhi,bhij->bhj', q_t, S)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, n_heads, d_head)
        y = y.reshape(batch, seq_len, self.d_model)
        y = self.W_o(y)
        y = self.dropout(y)
        return y


class LinearAttentionBlock(nn.Module):
    """Linear attention + LayerNorm + Residual + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttentionLayer(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionClassifier(nn.Module):
    """Linear Attention baseline for selective copying.

    Expected: < 60% accuracy (cannot do sharp retrieval).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            LinearAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# 2. Diagonal SSM (Mamba-style)
# =============================================================================

class DiagonalSSMLayer(nn.Module):
    """Diagonal SSM with input-dependent gating.

    Recurrence (standard semiring):
        h_t = diag(a_t) · h_{t-1} + B_t · x_t
        y_t = C · h_t + D · x_t

    where a_t = sigmoid(W_a x_t) ∈ (0, 1) is the decay gate.

    This is O(n) per step. Good for smooth dynamics but lacks
    content-based attention (no q·k matching).
    """

    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input-dependent decay: a_t = sigmoid(W_a x_t + b_a) ∈ (0, 1)
        self.W_a = nn.Linear(d_model, state_dim)

        # Input projection: B_t maps input to state
        self.B = nn.Linear(d_model, state_dim, bias=False)

        # Output projection: C maps state to output
        self.C = nn.Linear(state_dim, d_model, bias=False)

        # Direct feedthrough
        self.D = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize decay bias high so sigmoid ≈ 0.9 (remember by default)
        nn.init.constant_(self.W_a.bias, 2.2)
        nn.init.normal_(self.W_a.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Compute input-dependent decay
        a = torch.sigmoid(self.W_a(x))  # (batch, seq_len, state_dim)

        # Input projection
        b = self.B(x)  # (batch, seq_len, state_dim)

        # Sequential scan
        h = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = a[:, t, :] * h + b[:, t, :]
            y_t = self.C(h)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        y = y + self.D(x)
        y = self.dropout(y)
        return y


class DiagonalSSMBlock(nn.Module):
    """Diagonal SSM + LayerNorm + Residual + FFN."""

    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalSSMLayer(d_model, state_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DiagonalSSMClassifier(nn.Module):
    """Diagonal SSM baseline for selective copying.

    Expected: < 70% accuracy (no content-based matching).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        state_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            DiagonalSSMBlock(d_model, state_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

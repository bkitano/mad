"""
cosFormer with optional post-readout sigmoid gating.

Implements the architecture from proposal 009: Post-Sigmoid Gating for Linear Attention.

Key equations:
  Standard cosFormer readout:
    o_t = phi(q_t)^T S_t  (linear attention with cosine reweighting)

  Gated readout (proposal eq. main):
    o_hat_t = (o_t * sigma(x_t W_g)) W_O

  where:
    - phi(q) = ReLU(q), phi(k) = ReLU(k)  (feature map)
    - Cosine reweighting: Q_cos = Q * cos(pos), Q_sin = Q * sin(pos), etc.
    - S_t = cumsum(K^T V) is the causal linear attention state
    - W_g is zero-initialized so gate starts at sigma(0) = 0.5

The gate introduces a data-dependent nonlinearity between the linear
attention output and the output projection, breaking the low-rank
bottleneck of the purely linear readout path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFormerAttention(nn.Module):
    """cosFormer attention with optional post-readout sigmoid gating.

    cosFormer uses cosine-reweighted linear attention:
      o_t = Q_cos^T @ cumsum(K_cos @ V) + Q_sin^T @ cumsum(K_sin @ V)

    With gating (proposal 009):
      o_gated = o_t * sigmoid(x_t @ W_g)
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, use_gate: bool = True,
                 gate_bias_init: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.use_gate = use_gate

        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)

        # Post-readout sigmoid gate (proposal eq.: o_hat = (o * sigma(x W_g)) W_O)
        if use_gate:
            self.W_gate = nn.Linear(d_model, n_heads * d_k, bias=True)
            # Zero-init weight, bias init to gate_bias_init
            # With bias=1.0: gate starts at sigma(1.0) ≈ 0.73 (closer to identity than 0.5)
            nn.init.zeros_(self.W_gate.weight)
            nn.init.constant_(self.W_gate.bias, gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) input hidden states

        Returns:
            (B, T, d_model) output after linear attention + optional gating
        """
        B, T, D = x.shape

        # Project to Q, K, V and reshape to (B, T, H, d_k)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k)

        # Feature map: ReLU (standard for cosFormer)
        Q = F.relu(Q)
        K = F.relu(K)

        # Cosine position reweighting
        # pos_i = i * pi / (2T) for i in 0..T-1
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        angle = positions * (math.pi / (2 * T))  # (T,)

        cos_pos = torch.cos(angle)  # (T,)
        sin_pos = torch.sin(angle)  # (T,)

        # Reweight Q and K by cos/sin position encodings
        # Shape: (B, T, H, d_k) * (T, 1, 1) -> (B, T, H, d_k)
        Q_cos = Q * cos_pos[:, None, None]
        Q_sin = Q * sin_pos[:, None, None]
        K_cos = K * cos_pos[:, None, None]
        K_sin = K * sin_pos[:, None, None]

        # Causal linear attention via cumulative state
        # S_t = cumsum_{j<=t} (K_j^T @ V_j), shape (B, T, H, d_k, d_k)
        # o_t = Q_t^T @ S_t, shape (B, T, H, d_k)

        # Compute outer products K^T V and cumsum for causal masking
        # K: (B, T, H, d_k), V: (B, T, H, d_k)
        # KV = einsum('bthd,bthe->bthde', K, V) -> (B, T, H, d_k, d_k)
        KV_cos = torch.einsum('bthd,bthe->bthde', K_cos, V)
        KV_sin = torch.einsum('bthd,bthe->bthde', K_sin, V)

        # Cumulative sum for causal attention
        S_cos = torch.cumsum(KV_cos, dim=1)  # (B, T, H, d_k, d_k)
        S_sin = torch.cumsum(KV_sin, dim=1)  # (B, T, H, d_k, d_k)

        # Readout: o_t = Q_cos^T S_cos + Q_sin^T S_sin
        o = (torch.einsum('bthd,bthde->bthe', Q_cos, S_cos) +
             torch.einsum('bthd,bthde->bthe', Q_sin, S_sin))
        # o shape: (B, T, H, d_k)

        # Normalization: compute denominator z_t = cumsum(K)
        z_cos = torch.cumsum(K_cos, dim=1)  # (B, T, H, d_k)
        z_sin = torch.cumsum(K_sin, dim=1)  # (B, T, H, d_k)
        z = (torch.einsum('bthd,bthd->bth', Q_cos, z_cos) +
             torch.einsum('bthd,bthd->bth', Q_sin, z_sin))
        # z shape: (B, T, H)

        # Normalize (add eps to avoid division by zero)
        z = z.unsqueeze(-1).clamp(min=1e-6)  # (B, T, H, 1)
        o = o / z

        # === THE KEY INNOVATION: Post-readout sigmoid gate ===
        # Proposal eq.: o_hat = (o * sigma(x_t W_g)) W_O
        if self.use_gate:
            gate = torch.sigmoid(
                self.W_gate(x).view(B, T, self.n_heads, self.d_k)
            )
            o = o * gate

        # Reshape and project output
        o = o.reshape(B, T, self.n_heads * self.d_k)
        return self.W_O(o)


class CosFormerBlock(nn.Module):
    """Single cosFormer block with pre-norm residual connection and FFN.

    Architecture:
      x -> LayerNorm -> CosFormerAttention -> + -> LayerNorm -> FFN -> +
           |                                  ^    |                   ^
           +----------------------------------+    +-------------------+
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int,
                 use_gate: bool = True, gate_bias_init: float = 1.0,
                 dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CosFormerAttention(d_model, n_heads, d_k, use_gate=use_gate,
                                       gate_bias_init=gate_bias_init)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN with SwiGLU-style gating
        self.ffn_up = nn.Linear(d_model, ffn_mult * d_model, bias=False)
        self.ffn_gate = nn.Linear(d_model, ffn_mult * d_model, bias=False)
        self.ffn_down = nn.Linear(ffn_mult * d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h))

        # Pre-norm FFN with residual (SwiGLU)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h)))

        return x


class CosFormerModel(nn.Module):
    """Full cosFormer model for MQAR task.

    Architecture:
      Token embedding -> Positional embedding -> L x CosFormerBlock -> LayerNorm -> Head

    Args:
        vocab_size: Number of tokens in vocabulary
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_k: Per-head key/query dimension
        n_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        use_gate: Whether to use post-readout sigmoid gating (the key variable)
        dropout: Dropout rate
        ffn_mult: FFN hidden dim multiplier
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_k: int = 16,
        n_layers: int = 2,
        max_seq_len: int = 128,
        use_gate: bool = True,
        gate_bias_init: float = 1.0,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.d_model = d_model
        self.gate_bias_init = gate_bias_init

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            CosFormerBlock(d_model, n_heads, d_k, use_gate=use_gate,
                          gate_bias_init=gate_bias_init,
                          dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Standard transformer initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Re-apply gate initialization (overrides the normal init above)
        # Weight: zero-init so gate depends only on bias at start
        # Bias: set to gate_bias_init (default 1.0 → sigmoid(1.0) ≈ 0.73)
        for block in self.blocks:
            if block.attn.use_gate:
                nn.init.zeros_(block.attn.W_gate.weight)
                nn.init.constant_(block.attn.W_gate.bias, self.gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) integer token indices

        Returns:
            (B, T, vocab_size) logits
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)

        h = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            h = block(h)

        h = self.norm_f(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gate_stats(self) -> dict:
        """Get statistics about gate activations for monitoring.

        Returns dict with gate value statistics per layer.
        Only meaningful after a forward pass with actual data.
        """
        stats = {}
        for i, block in enumerate(self.blocks):
            if block.attn.use_gate and hasattr(block.attn.W_gate, 'weight'):
                w = block.attn.W_gate.weight.data
                stats[f'layer_{i}_gate_weight_norm'] = w.norm().item()
                stats[f'layer_{i}_gate_weight_max'] = w.abs().max().item()
        return stats

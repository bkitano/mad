"""
Gated Linear Attention (GLA) model with configurable normalization.

Implements a minimal GLA for the MVE of proposal 063:
- 2-layer GLA, d=64, d_k=32, d_v=64, 2 heads, ~100K params
- Configurable normalization: RMSNorm or DyT
- Configurable placement: Pre-LN or Peri-LN (pre+post norm)

GLA state update (Yang et al., ICML 2024):
  q_t, k_t, v_t, alpha_t = proj(x_t)
  S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T   (gated outer product update)
  o_t = q_t^T * S_t                                (readout)

For MVE, we use a simple recurrent formulation (not chunkwise)
since we're testing quality, not throughput.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .normalization import DyT, RMSNorm


def get_norm(norm_type: str, d: int) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "rmsnorm":
        return RMSNorm(d)
    elif norm_type == "dyt":
        return DyT(d)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 7) // 8) * 8

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GLALayer(nn.Module):
    """
    Single GLA (Gated Linear Attention) layer.

    Implements the recurrent form:
      q, k, v = proj(x)
      alpha = sigmoid(gate(x))   # per-dim gating
      S_t = diag(alpha) * S_{t-1} + k_t outer v_t
      o_t = S_t @ q_t
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_v)
        alpha = torch.sigmoid(self.gate_proj(x)).view(B, T, self.n_heads, self.d_k)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Recurrent state update
        S = torch.zeros(B, self.n_heads, self.d_k, self.d_v, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            alpha_t = alpha[:, t]

            S = alpha_t.unsqueeze(-1) * S + torch.einsum('bhk,bhv->bhkv', k_t, v_t)
            o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, T, self.n_heads * self.d_v)

        return self.o_proj(output)


class GLABlock(nn.Module):
    """
    GLA block with configurable normalization and placement.

    Pre-LN:  x -> Norm -> GLA -> +res -> Norm -> FFN -> +res
    Peri-LN: x -> Norm -> GLA -> Norm -> +res -> Norm -> FFN -> Norm -> +res
    """

    def __init__(self, d_model, d_k, d_v, n_heads, norm_type="rmsnorm", peri_ln=False, d_ff=None):
        super().__init__()
        self.peri_ln = peri_ln

        self.gla = GLALayer(d_model, d_k, d_v, n_heads)
        self.ffn = SwiGLU(d_model, d_ff)

        self.pre_attn_norm = get_norm(norm_type, d_model)
        self.pre_ffn_norm = get_norm(norm_type, d_model)

        if peri_ln:
            self.post_attn_norm = get_norm(norm_type, d_model)
            self.post_ffn_norm = get_norm(norm_type, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.pre_attn_norm(x)
        h = self.gla(h)
        if self.peri_ln:
            h = self.post_attn_norm(h)
        x = residual + h

        residual = x
        h = self.pre_ffn_norm(x)
        h = self.ffn(h)
        if self.peri_ln:
            h = self.post_ffn_norm(h)
        x = residual + h

        return x


class GLAModel(nn.Module):
    """
    Autoregressive GLA language model for MVE.

    MVE config: 2 layers, d=64, d_k=32, d_v=64, 2 heads, ~156K params
    """

    def __init__(self, vocab_size, d_model=64, d_k=32, d_v=64, n_heads=2,
                 n_layers=2, max_seq_len=512, norm_type="rmsnorm", peri_ln=False,
                 d_ff=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GLABlock(d_model, d_k, d_v, n_heads, norm_type, peri_ln, d_ff)
            for _ in range(n_layers)
        ])

        self.final_norm = get_norm(norm_type, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_activation_stats(self) -> dict:
        stats = {}
        for i, block in enumerate(self.blocks):
            for name, module in block.named_modules():
                if isinstance(module, DyT):
                    stats[f"block_{i}/{name}/alpha"] = module.alpha.item()
        if isinstance(self.final_norm, DyT):
            stats["final_norm/alpha"] = self.final_norm.alpha.item()
        return stats

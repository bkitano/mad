"""
Attention mechanisms for Experiment 029: Circulant FAVOR+ Linear Attention.

Implements four attention variants for comparison:
1. Dense FAVOR+: Standard positive random features with dense projection (O(md) per token)
2. C-FAVOR+: Circulant random features via FFT (O(d log d) per token)
3. ReLU Linear Attention: No feature map, just ReLU(Q) @ ReLU(K)^T (baseline)
4. Softmax Attention: Standard softmax attention (quality ceiling)

Multi-head, multi-layer architecture with pre-norm residual connections and SwiGLU FFN.

References:
- FAVOR+ (Choromanski et al., 2021): Positive random features for softmax approximation
- CBE (Yu et al., 2014): Circulant binary embeddings preserve angular distances
- Proposal 029: Combining circulant projection with FAVOR+ feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================================================================
# Feature Map Functions (the core being tested)
# ===========================================================================

def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize along the last dimension. Critical for FAVOR+ stability."""
    return F.normalize(x, p=2, dim=-1)


def dense_favor_feature_map(x: torch.Tensor, omega: torch.Tensor, num_features: int) -> torch.Tensor:
    """
    Dense FAVOR+ positive feature map (proposal eq. 1).

    phi+(x) = exp(-||x||^2/2) / sqrt(m) * exp(Omega @ x)

    IMPORTANT: x must be L2-normalized before calling this function.
    After L2 normalization, ||x||^2 = 1, so exp(-||x||^2/2) = exp(-1/2) is constant.
    This prevents the exponential from creating extreme value ranges.

    x: [B, H, T, d_k]  (L2-normalized)
    omega: [H, m, d_k]  (per-head random projection)
    returns: [B, H, T, m]
    """
    # After L2 norm: ||x||^2 = 1 for all tokens, so exp(-||x||^2/2) = exp(-0.5) is constant
    # phi+(x) = exp(-1/2) / sqrt(m) * exp(w^T x) = exp(w^T x - 1/2) / sqrt(m)
    # NOTE: Do NOT subtract max per-token — this breaks the kernel approximation
    # by making features token-local instead of globally comparable.
    # L2 normalization already bounds the feature values.
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, H, T, 1]
    projection = torch.einsum('bhtd,hmd->bhtm', x, omega)  # [B, H, T, m]
    features = torch.exp(projection - x_norm_sq / 2) / math.sqrt(num_features)
    return features


def circulant_favor_feature_map(x: torch.Tensor, r: torch.Tensor, s: torch.Tensor,
                                 num_features: int) -> torch.Tensor:
    """
    C-FAVOR+ positive feature map via circulant projection (proposal eq. 2-5).

    For m > d_k, stacks multiple independent circulant blocks (proposal eq. stacked):
        Omega = [circ(r_1)*diag(s_1); ...; circ(r_ceil(m/d))*diag(s_ceil(m/d))]

    x: [B, H, T, d_k]  (L2-normalized)
    r: [H, num_blocks, d_k]   (learnable circulant vectors)
    s: [H, num_blocks, d_k]   (fixed sign vectors)
    returns: [B, H, T, m]
    """
    B, H, T, d_k = x.shape
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, H, T, 1]

    num_blocks = r.shape[1]

    # Process all blocks: expand x for broadcasting with blocks
    # x: [B, H, T, d_k] -> [B, H, 1, T, d_k]
    x_exp = x.unsqueeze(2)

    # Sign-flip: s .* x for each block
    # s: [H, num_blocks, d_k] -> [1, H, num_blocks, 1, d_k]
    sx = s.unsqueeze(0).unsqueeze(3) * x_exp  # [B, H, num_blocks, T, d_k]

    # FFT-based circulant projection for all blocks at once
    fft_sx = torch.fft.rfft(sx, dim=-1)         # [B, H, num_blocks, T, d_k//2+1]
    fft_r = torch.fft.rfft(r, dim=-1)           # [H, num_blocks, d_k//2+1]
    fft_result = fft_r.unsqueeze(0).unsqueeze(3) * fft_sx  # [B, H, num_blocks, T, d_k//2+1]
    projection = torch.fft.irfft(fft_result, n=d_k, dim=-1)  # [B, H, num_blocks, T, d_k]

    # Concatenate blocks: [B, H, num_blocks, T, d_k] -> [B, H, T, num_blocks*d_k]
    projection = projection.permute(0, 1, 3, 2, 4).reshape(B, H, T, num_blocks * d_k)

    # Truncate to num_features
    projection = projection[..., :num_features]

    # Same FAVOR+ feature map — L2 norm ensures stability
    features = torch.exp(projection - x_norm_sq / 2) / math.sqrt(num_features)
    return features


# ===========================================================================
# Multi-Head Attention Layers
# ===========================================================================

class MultiHeadLinearAttention(nn.Module):
    """
    Multi-head causal linear attention with pluggable feature maps.

    Supports: dense_favor, circulant_favor, relu_linear
    """

    def __init__(self, d_model: int, n_heads: int, num_features: int,
                 feature_type: str = 'dense_favor', learnable_circulant: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.num_features = num_features
        self.feature_type = feature_type

        assert d_model % n_heads == 0

        # Q, K, V projections (all heads combined)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Feature map parameters
        if feature_type == 'dense_favor':
            # Per-head ORF projection: [H, m, d_k]
            self.register_buffer('omega', self._sample_orf(n_heads, num_features, self.d_k))
        elif feature_type == 'circulant_favor':
            # For m > d_k, stack multiple circulant blocks (proposal eq. stacked blocks)
            self.num_circ_blocks = (num_features + self.d_k - 1) // self.d_k
            # Per-head, per-block circulant vectors r: [H, num_blocks, d_k]
            if learnable_circulant:
                self.r = nn.Parameter(torch.randn(n_heads, self.num_circ_blocks, self.d_k) / math.sqrt(self.d_k))
            else:
                self.register_buffer('r', torch.randn(n_heads, self.num_circ_blocks, self.d_k) / math.sqrt(self.d_k))
            # Per-head, per-block fixed sign vectors s: [H, num_blocks, d_k]
            self.register_buffer('s', (torch.randint(0, 2, (n_heads, self.num_circ_blocks, self.d_k)).float() * 2 - 1))

    def _sample_orf(self, n_heads: int, m: int, d: int) -> torch.Tensor:
        """Sample orthogonal random features for each head."""
        omegas = []
        for _ in range(n_heads):
            blocks = []
            remaining = m
            while remaining > 0:
                block_size = min(remaining, d)
                G = torch.randn(d, d)
                Q, _ = torch.linalg.qr(G)
                norms = torch.randn(d, d).norm(dim=1)
                Q = Q * norms.unsqueeze(1)
                blocks.append(Q[:block_size])
                remaining -= block_size
            omegas.append(torch.cat(blocks, dim=0))
        return torch.stack(omegas)  # [H, m, d]

    def _apply_feature_map(self, q: torch.Tensor, k: torch.Tensor):
        """
        Apply feature map to Q and K.

        q, k: [B, H, T, d_k]
        returns: q_prime, k_prime: [B, H, T, m]
        """
        if self.feature_type == 'dense_favor':
            # L2 normalize Q, K before feature map (critical for FAVOR+ stability)
            q_norm = _l2_normalize(q)
            k_norm = _l2_normalize(k)
            q_prime = dense_favor_feature_map(q_norm, self.omega, self.num_features)
            k_prime = dense_favor_feature_map(k_norm, self.omega, self.num_features)
        elif self.feature_type == 'circulant_favor':
            q_norm = _l2_normalize(q)
            k_norm = _l2_normalize(k)
            q_prime = circulant_favor_feature_map(q_norm, self.r, self.s, self.num_features)
            k_prime = circulant_favor_feature_map(k_norm, self.r, self.s, self.num_features)
        elif self.feature_type == 'relu_linear':
            q_prime = F.relu(q)
            k_prime = F.relu(k)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        return q_prime, k_prime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        # Project and reshape to multi-head
        q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)

        # Apply feature map
        q_prime, k_prime = self._apply_feature_map(q, k)  # [B, H, T, m]
        m = q_prime.shape[-1]

        # Causal linear attention via cumulative sum
        # More efficient: vectorize over heads using cumsum
        # kv_state: [B, H, m, d_k], k_sum_state: [B, H, m]
        # We accumulate k'_t v_t^T and k'_t over time

        # Use cumulative sum approach for efficiency
        # KV cumsum: for each t, sum_{s<=t} k'_s v_s^T
        kv = torch.einsum('bhti,bhtj->bhtij', k_prime, v)  # [B, H, T, m, d_k]
        kv_cumsum = torch.cumsum(kv, dim=2)  # [B, H, T, m, d_k]

        # K cumsum: for each t, sum_{s<=t} k'_s
        k_cumsum = torch.cumsum(k_prime, dim=2)  # [B, H, T, m]

        # Output: y_t = (q'_t^T @ KV_cumsum_t) / (q'_t^T @ K_cumsum_t)
        num = torch.einsum('bhti,bhtij->bhtj', q_prime, kv_cumsum)  # [B, H, T, d_k]
        den = torch.einsum('bhti,bhti->bht', q_prime, k_cumsum).unsqueeze(-1)  # [B, H, T, 1]
        den = den.clamp(min=1e-6)

        out = num / den  # [B, H, T, d_k]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, d_model]
        return self.W_o(out)


class MultiHeadSoftmaxAttention(nn.Module):
    """Multi-head standard softmax attention (quality ceiling)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, T, d_k]

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ===========================================================================
# Transformer Block with Pre-Norm
# ===========================================================================

class SwiGLU(nn.Module):
    """SwiGLU FFN: matches proposal architecture."""
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)  # SwiGLU-adjusted expansion
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual"""

    def __init__(self, d_model: int, n_heads: int, num_features: int,
                 attention_type: str, learnable_circulant: bool = True):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if attention_type == 'softmax':
            self.attn = MultiHeadSoftmaxAttention(d_model, n_heads)
        else:
            self.attn = MultiHeadLinearAttention(
                d_model, n_heads, num_features,
                feature_type=attention_type, learnable_circulant=learnable_circulant
            )

        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ===========================================================================
# Full Model
# ===========================================================================

class AssociativeRecallModel(nn.Module):
    """
    Multi-layer transformer for associative recall task.

    Architecture:
        Embedding + PosEmbed -> L x TransformerBlock -> LayerNorm -> MLP Head -> Output

    Supports 4 attention types: dense_favor, circulant_favor, relu_linear, softmax
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 attention_type: str, num_features: int = None,
                 learnable_circulant: bool = True, max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Determine num_features
        d_k = d_model // n_heads
        if num_features is None:
            num_features = d_k  # Default: m = d_k

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, num_features, attention_type, learnable_circulant)
            for _ in range(n_layers)
        ])

        # Final norm + MLP head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len] token indices
        returns: [batch, seq_len, vocab_size] logits
        """
        B, T = x.shape

        # Embed
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Head
        h = self.final_norm(h)
        logits = self.head(h)

        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

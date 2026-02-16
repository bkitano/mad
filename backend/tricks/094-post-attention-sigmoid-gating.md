# 094: Post-Attention Sigmoid Gating

**Category**: algebraic
**Gain type**: expressivity
**Source**: Qiu et al. (2025), "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
**Paper**: [papers/post-attention-sigmoid-gating.pdf]
**Documented**: 2026-02-15

## Description

Apply a head-specific, input-dependent sigmoid gate to the output of Scaled Dot-Product Attention (SDPA), *before* the output projection. This elementwise multiplicative gate introduces two critical properties into the attention mechanism: (1) **non-linearity** that breaks the low-rank bottleneck formed by the consecutive value and output linear projections $W_V W_O$, and (2) **input-dependent sparsity** that allows the model to selectively zero-out irrelevant attention outputs. The gate naturally saturates near zero for uninformative features, creating a sparse mask that eliminates the "attention sink" phenomenon (where initial tokens absorb disproportionate attention mass). This was the NeurIPS 2025 Best Paper and has been adopted in the Qwen3-Next architecture.

## Mathematical Form

**Standard Multi-Head Attention (baseline):**

$$
O = \text{MultiHead}(Q, K, V) W_O
$$

where $\text{head}_k = \text{softmax}\!\left(\frac{Q W_Q^k (K W_K^k)^T}{\sqrt{d_k}}\right) V W_V^k$.

**Gated Attention (the trick):**

$$
Y' = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(X W_\theta)
$$

where $Y \in \mathbb{R}^{n \times d_k}$ is the SDPA output, $X \in \mathbb{R}^{n \times d_{\text{model}}}$ is the pre-norm hidden state (query-dependent), $W_\theta \in \mathbb{R}^{d_{\text{model}} \times d_k}$ is a learnable projection, $\sigma$ is the sigmoid function, and $\odot$ is elementwise multiplication.

**Key Definitions:**

- $X \in \mathbb{R}^{n \times d_{\text{model}}}$ — input hidden states (after pre-norm)
- $Y \in \mathbb{R}^{n \times d_k}$ — per-head SDPA output
- $W_\theta \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — head-specific gate projection (the only new parameter)
- $\sigma(z) = \frac{1}{1+e^{-z}}$ — sigmoid activation

**Why It Works — The Low-Rank Bottleneck:**

In standard attention with GQA (grouped query attention), the per-head output is:

$$
o_i^k = \left(\sum_{j=0}^{i} S_{ij}^k \cdot X_j W_V^k\right) W_O^k
$$

Since $W_V$ is shared across query heads in the same group and $d_k < d_{\text{model}}$, the composition $W_V^k W_O^k$ forms a single low-rank linear mapping applied to all $X_j$. The gate breaks this low-rank constraint by inserting a non-linearity:

$$
o_i^k = \text{Non-Linearity-Map}\left(\sum_{j=0}^{i} S_{ij}^k \cdot X_j W_V^k\right) W_O^k
$$

**Gating positions explored (from best to worst):**

| Position | Name | Gate Score Shape | Added Params (15B MoE) |
|----------|------|-----------------|----------------------|
| After SDPA output | $G_1$ | $n \times q \times d_k$ | 201M |
| After value layer | $G_2$ | $n \times k \times d_k$ | 25M |
| After key layer | $G_3$ | $n \times k \times d_k$ | 25M |
| After query layer | $G_4$ | $n \times q \times d_k$ | 201M |
| After dense output | $G_5$ | $n \times d_{\text{model}}$ | 100M |

$G_1$ (SDPA output) is the most effective position, and head-specific, elementwise, multiplicative gating with sigmoid activation is the best configuration.

## Complexity

| Operation | Standard Attention | With Post-SDPA Gate |
|-----------|--------------------|---------------------|
| Parameters | $\text{Attn params}$ | $\text{Attn params} + h \cdot d_{\text{model}} \cdot d_k$ |
| FLOPs | $O(n^2 d_k)$ per head | $O(n^2 d_k + n \cdot d_{\text{model}} \cdot d_k)$ per head |
| Wall-time overhead | — | $< 2\%$ latency increase |

**Memory:** Additional storage for $W_\theta$ parameters is negligible ($<2$M for headwise gating in a 15B model). Activation memory adds one $n \times d_k$ tensor per head for the gate scores.

**Key insight:** The gating FLOPs are $O(n \cdot d_{\text{model}} \cdot d_k)$ per head, which is dominated by the $O(n^2 \cdot d_k)$ attention cost. This makes the gate essentially free for long sequences.

## Applicability

- **Transformer attention layers:** Direct augmentation of any multi-head softmax attention. Drop-in addition with $<2\%$ overhead
- **MoE transformers:** Tested at 15B-parameter MoE scale with 128 experts; consistently improves PPL and downstream benchmarks
- **Dense transformers:** Also validated at 1.7B dense scale; enables stable training with larger learning rates and batch sizes
- **Long-context models:** Eliminates attention sinks, enabling better length generalization (+10 points on RULER at 64k-128k context)
- **Quantization-friendly:** Reduces massive activations and outliers, beneficial for INT8/FP8 quantization

## Limitations

- The gate is **query-dependent** (computed from $X_i$) not key-value-dependent, so it cannot selectively gate based on the attended content — only based on the current query position
- Head-specific gates are critical; sharing gate scores across heads reduces benefits significantly (sparsity patterns differ per head)
- Multiplicative gating with sigmoid consistently outperforms additive gating with SiLU, but the theoretical reason is not fully understood
- Does not change the $O(n^2)$ complexity of attention itself — this trick improves expressivity and stability, not asymptotic efficiency
- Training stability improvements are observed empirically but lack rigorous theoretical explanation

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    """Post-SDPA sigmoid gating (G1 position).

    Applies head-specific, elementwise, multiplicative sigmoid gate
    after the scaled dot-product attention output, before output projection.
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k

        # Standard attention projections
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)

        # Gate projection: head-specific, elementwise
        # Each head gets its own d_model -> d_k projection
        self.W_gate = nn.Linear(d_model, n_heads * d_k, bias=False)
        # Zero-initialize for residual-friendly start
        nn.init.zeros_(self.W_gate.weight)

    def forward(self, x):
        B, N, D = x.shape

        # Standard QKV projections
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        Y = attn @ V  # (B, H, N, d_k)

        # === THE TRICK: Post-SDPA sigmoid gate ===
        gate = torch.sigmoid(
            self.W_gate(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        )  # (B, H, N, d_k) — head-specific, elementwise
        Y = Y * gate  # Multiplicative gating — introduces sparsity + non-linearity

        # Output projection
        Y = Y.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_O(Y)
```

**Key implementation details:**
- Zero-initialize gate weights so gating starts as identity ($\sigma(0) = 0.5$, scaling all values by half initially)
- Gate scores concentrate near 0 in practice (mean ~0.116), creating strong sparsity
- The gate uses the pre-norm hidden state $X$ (same input used for Q/K/V), making it query-dependent
- Headwise gating (single scalar per head) adds only ~1.6M params but still provides most of the benefit

## References

- Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., & Lin, J. (2025). Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. arXiv:2505.06708. **NeurIPS 2025 Best Paper.**
- Bondarenko, Y., Nagel, M., & Blankevoort, T. (2023). Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. NeurIPS 2023.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453.
- Sun, M., Chen, X., Kolter, J.Z., & Liu, Z. (2024). Massive Activations in Large Language Models. arXiv:2402.17762.

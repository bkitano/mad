# 229: Multi-Matrix Factorization Attention (MFA)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Hu et al., "Multi-matrix Factorization Attention", ACL 2025 Findings (arXiv:2412.19255)
**Paper**: [papers/multi-matrix-factorization-attention.pdf]
**Documented**: 2026-02-15

## Description

Multi-matrix Factorization Attention (MFA) is an attention architecture that uses low-rank matrix factorization in the Query-Key (QK) circuit to efficiently scale up both the number and dimension of attention heads while keeping KV cache size constant. The key insight is that attention capacity depends on two factors — **Total Effective Rank (TER)** (product of number of heads and per-head factorization rank) and **Shared Latent Subspace Dimension (SLSD)** (dimension of key/value shared across heads) — and MFA maximizes TER while minimizing SLSD. At 7B scale with 1T training tokens, MFA achieves 49.9% average accuracy vs MHA's 49.0% while using only 12.5% of MHA's KV cache (24.6KB vs 196.6KB per token). Its variant MFA-Key-Reuse (MFA-KR) cuts KV cache to 6.25% of MHA with competitive performance.

## Mathematical Form

**Background — Fully Parameterized Bilinear Attention (FPBA):**

The theoretical upper bound of attention capacity is:

$$
O_i = \sum_{c=1}^{H} \Big( \sum_{j=1}^{i} \phi\Big(\frac{\mathbf{x}_i W_c \mathbf{x}_j}{\sqrt{H}}\Big) \mathbf{x}_j U_c \Big)
$$

where $W_c \in \mathbb{R}^{H \times H}$ is the QK circuit and $U_c \in \mathbb{R}^{H \times H}$ is the VO circuit for channel $c$, with $\phi$ being softmax. All methods are low-rank approximations of FPBA.

**MFA Factorization:**

MFA decomposes $W_c$ as $S_q Q_c S_k^\top$ and $U_c$ as $S_v V_c O_c^\top$:

$$
O_i = \sum_{c=1}^{n} \Big( \sum_{j=1}^{i} \phi\Big(\frac{\mathbf{x}_i (S_q Q_c S_k^\top) \mathbf{x}_j^\top}{\sqrt{d}}\Big) \mathbf{x}_j S_v V_c O_c^\top \Big)
$$

Equivalently:

$$
O_i = \sum_{c=1}^{n} \Big( \sum_{j=1}^{i} \phi\Big(\frac{\mathbf{x}_i S_q Q_c (S_k^\top \mathbf{x}_j^\top)}{\sqrt{d}}\Big) (\mathbf{x}_j S_v) V_c O_c^\top \Big)
$$

**Key Definitions:**

- $S_q, S_k, S_v \in \mathbb{R}^{H \times C}$ — shared projections across all heads (down-projection to latent space)
- $Q_c, O_c \in \mathbb{R}^{C \times C}$ — head-specific projections (in the latent space)
- $V_c \in \mathbb{R}^{C \times d}$ — head-specific value projection
- $H$ — model embedding dimension
- $C$ — low-rank factorization dimension (shared latent subspace dimension, SLSD)
- $n = m$ — number of heads (can be much larger than in MHA)
- $d = H/n$ — per-head dimension

**Inference Formulation (what gets cached):**

For each token $\mathbf{x}_j$, the key and value are computed as:

$$
\mathbf{k}_j = \mathbf{x}_j S_k \in \mathbb{R}^C, \quad \mathbf{v}_j = \mathbf{x}_j S_v \in \mathbb{R}^C
$$

These are shared across all heads. KV cache per token = $2C$ (independent of number of heads $n$).

**MFA-Key-Reuse (MFA-KR):**

Re-parameterizes the value projection to reuse the key cache:

$$
S_v = S_k + \alpha \odot N \cdot S_k = (I + \text{diag}(\alpha) N) W_K
$$

where $N \in \mathbb{R}^{C \times C}$, $\alpha \in \mathbb{R}^C$ (initialized to zero for training stability), and $\odot$ is element-wise multiplication. This halves KV cache to just $C$ per token.

**Capacity Analysis:**

| Method | KV Cache | Heads | Factor. Rank/Head | SLSD | Total Effective Rank |
|--------|----------|-------|-------------------|------|---------------------|
| FPBA | $2H^2$ | $H$ | $H$ | $H$ | $H^2$ |
| MHA | $2H$ | $n$ | $d$ | $H$ | $nd$ |
| MQA | $2d$ | $n$ | $d$ | $d$ | $nd$ |
| GQA | $2gd$ | $n$ | $d$ | $gd$ | $nd$ |
| MLA | $2C$ | $m$ | $d$ | $C$ | $md$ |
| **MFA** | $2C$ | $m$ | $C$ | $C$ | $mC$ |

Since typically $H > C > d$ and $m > n$, MFA achieves higher TER ($mC$) than MHA ($nd$) or MLA ($md$) while maintaining small KV cache ($2C$).

## Complexity

| Operation | MHA | MFA | Reduction |
|-----------|-----|-----|----------|
| KV cache per token | $2 h d_h$ | $2C$ | $\frac{C}{h d_h}$ (typically 8-16x) |
| Parameters | $4 d_{\text{model}} h d_h$ | $H(3C + mC) + mC^2$ | Comparable |
| QK compute per head | $O(T^2 d_h)$ | $O(T^2 C)$ | $C/d_h$ (often $C > d$) |

**At 7B MoE scale (1.2B activated params):**

| Method | KV Cache/Token | Avg. Accuracy |
|--------|---------------|---------------|
| MHA | 196.6KB | 49.0% |
| MFA | 24.6KB (**87.5% reduction**) | **49.9%** |
| MFA-KR | 12.3KB (**93.7% reduction**) | 48.0% |
| MLA | — | 48.3% |

**Memory:** $O(C)$ per token vs $O(hd_h)$ for MHA, where $C \ll hd_h$.

## Applicability

- **Large-scale LLM pretraining**: Validated at 7B-scale MoE with 1T tokens; matches MHA scaling laws
- **Inference-constrained deployment**: Primary benefit is dramatic KV cache reduction for long-context serving
- **Drop-in replacement for MHA**: Compatible with RoPE (unlike MLA which requires decoupled RoPE), SwiGLU FFN, and standard LLM training recipes
- **Dense and MoE architectures**: Tested with both dense (1B ablation) and MoE (7B) setups
- **Hybrid combinations**: Can be combined with SSM/linear attention layers for even more aggressive cache reduction in long-context regimes

## GPU Efficiency Analysis

**Memory Access Pattern:**
- Shared projections $S_q, S_k, S_v$ are standard linear layers with coalesced access
- Per-head projections $Q_c, O_c$ are small $C \times C$ matrices — fit easily in shared memory
- KV cache reads during decoding load only $2C$ elements per token (vs $2hd_h$ for MHA)
- Significantly reduces HBM bandwidth pressure during autoregressive generation

**Parallelism:**
- Shared down-projection is a single large GEMM — saturates GPU SMs
- Per-head attention is independent across heads (same parallelism as MHA)
- No sequential bottlenecks; all operations map to matmul primitives
- Number of heads $m$ can be larger than MHA's $n$, providing more parallel work units

**Tensor Core Utilization:**
- All projections are linear layers → natural WGMMA/MMA targets
- Per-head QK compute is standard attention → FlashAttention-compatible
- Small per-head matrices $Q_c \in \mathbb{R}^{C \times C}$ may need careful tiling for tensor core efficiency

**Arithmetic Intensity:**
- During decoding, reduces bytes loaded from KV cache by $\frac{C}{hd_h}$
- For $C=128$, $h=32$, $d_h=64$: loads $256$ elements vs $4096$ — **16x less HBM bandwidth**
- Extra compute for per-head projection in latent space is small and compute-bound

## Limitations

- **Per-head projections add parameters**: Each head needs $Q_c, O_c \in \mathbb{R}^{C \times C}$ and $V_c \in \mathbb{R}^{C \times d}$, scaling as $O(mC^2)$ total
- **Prefill not accelerated**: During training/prefill, full Q, K, V are still computed — the benefit is primarily at inference decoding
- **MLA initialization sensitivity**: The paper notes MLA is very sensitive to weight initialization; MFA is more robust but still benefits from careful initialization ($W_2$ of GLU divided by $\sqrt{2 \cdot \text{layer\_idx}}$)
- **MFA-KR accuracy gap**: Key-reuse variant loses ~1-2% accuracy vs MFA at 7B scale; the gating mechanism is crucial and adds complexity
- **Not yet validated at >7B scale**: Largest experiments are 7B MoE; behavior at 70B+ is extrapolated from scaling curves
- **System-level integration**: End-to-end inference serving performance (with MFA-aware batching, paging) not yet evaluated

## Implementation Notes

```python
import torch
import torch.nn as nn

class MFAAttention(nn.Module):
    """Multi-matrix Factorization Attention.

    Key idea: shared down-projections S_q, S_k, S_v map to latent space of dim C,
    then per-head projections Q_c operate in the small C-dim space.
    KV cache stores only the C-dim latent vectors (shared across all heads).
    """
    def __init__(self, d_model, n_heads, C):
        super().__init__()
        self.d = d_model // n_heads  # per-head dim
        self.n_heads = n_heads
        self.C = C

        # Shared projections (across all heads)
        self.S_q = nn.Linear(d_model, C, bias=False)
        self.S_k = nn.Linear(d_model, C, bias=False)
        self.S_v = nn.Linear(d_model, C, bias=False)

        # Per-head projections in latent space
        # Q_c: [n_heads, C, C] — head-specific query rotation
        self.Q_c = nn.Parameter(torch.randn(n_heads, C, C) * 0.02)
        self.V_c = nn.Parameter(torch.randn(n_heads, C, self.d) * 0.02)
        self.O_c = nn.Parameter(torch.randn(n_heads, C, self.d) * 0.02)

        # Output projection
        self.W_O = nn.Linear(n_heads * self.d, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        # Shared latent projections
        q_latent = self.S_q(x)  # [B, T, C]
        k_latent = self.S_k(x)  # [B, T, C] — this is what gets cached
        v_latent = self.S_v(x)  # [B, T, C] — this is what gets cached

        # Per-head query: q_c = q_latent @ Q_c[head]
        # [B, T, C] @ [n_heads, C, C] -> [B, n_heads, T, C]
        q = torch.einsum('btc,hcd->bhtd', q_latent, self.Q_c)

        # Attention scores: q @ k^T (in latent C-space)
        # [B, n_heads, T, C] @ [B, 1, C, T] -> [B, n_heads, T, T]
        scores = torch.einsum('bhtc,bsc->bhts', q, k_latent) / (self.C ** 0.5)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Per-head value: v_c = v_latent @ V_c[head]
        # Attention output: attn @ v_c, then project by O_c
        v = torch.einsum('btc,hcd->bhtd', v_latent, self.V_c)
        out = torch.einsum('bhts,bhsd->bhtd', attn, v)

        # Per-head output projection by O_c and concatenate
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_O(out)

# KV cache comparison:
# MHA at h=32, d_h=64: 2 * 32 * 64 = 4096 elements/token
# MFA at C=128:        2 * 128 = 256 elements/token  (16x smaller)
# MFA-KR at C=128:     1 * 128 = 128 elements/token  (32x smaller)
```

## References

- Hu, J., Li, H., Zhang, Y., Wang, Z., Zhou, S., Zhang, X., Shum, H.-Y., & Jiang, D. (2024). Multi-matrix Factorization Attention. ACL 2025 Findings. arXiv:2412.19255.
- Elhage, N. et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread. (FPBA capacity analysis)
- DeepSeek-AI (2024). DeepSeek-V2. arXiv:2405.04434. (MLA baseline)
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. (MQA)
- Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. (GQA)
- Bhojanapalli, S. et al. (2020). Low-Rank Bottleneck in Multi-head Attention Models. arXiv:2002.07028. (Capacity analysis)

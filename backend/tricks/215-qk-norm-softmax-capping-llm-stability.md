# 215: QK-Norm + Softmax Capping — Combined Logit Control for LLM Training Stability

**Category**: stability
**Gain type**: efficiency
**Source**: Rybakov, Chrzanowski, Dykas, Xue & Lanir (2024) — "Methods of Improving LLM Training Stability", NVIDIA. arXiv:2410.16682
**Paper**: [papers/methods-improving-llm-training-stability.pdf]
**Documented**: 2026-02-15

## Description

One of the primary causes of LLM training divergence is **uncontrolled growth of attention logit magnitudes**. As training progresses with aggressive learning rates, the L2 norms of the QKV projection outputs grow unboundedly, pushing attention logits toward extreme values. This causes the softmax to saturate into near-one-hot distributions, producing vanishing gradients and ultimately loss spikes/divergence.

Prior work (Dehghani et al., 2023; Henry et al., 2020) proposed applying **LayerNorm to Q and K** after the QKV projection and before the dot product (QK_norm). This constrains query/key magnitudes and prevents logit explosion. Separately, **softmax capping** (Bello et al., 2017; Gemini Team, 2024) applies $\tanh$-based squashing to bound the effective logit range before softmax.

The key finding of this paper is that QK-norm and softmax capping are **complementary** — they control logit growth at different stages of the computation pipeline. Combining them (`QK_norm_cap`) allows training with **1.5× higher learning rates** without divergence compared to QK_norm alone, while also improving perplexity from 11.19 (baseline) to 11.00.

An alternative approach, `QKV_norm`, places LayerNorm **after the QKV projection** (normalizing Q, K, and V jointly) while removing the pre-normalization LayerNorm before QKV. This achieves the same 1.5× learning rate tolerance and better perplexity (10.85) with a simpler topology change.

**GPU efficiency:** Both techniques add only cheap elementwise/reduction operations (LayerNorm, tanh) that fuse trivially into existing attention kernels. No new matmuls, no irregular memory access, no additional kernel launches. The stability gain enables higher learning rates, directly translating to faster convergence in wall-clock time.

## Mathematical Form

**Standard attention logit computation (baseline):**

$$
\text{logit} = \frac{1}{\sqrt{d}} (X W^Q)(X W^K)^T
$$

$$
\text{attn} = \text{softmax}(\text{logit})
$$

where $X \in \mathbb{R}^{T \times d}$ is the input, $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ are query/key projections, and $d_k$ is the head dimension.

**Failure mode:** During divergence, $\|X W^{QKV}\|_2$ grows $> 2\times$ compared to a converging model (Table 1 of paper). Large $\|\text{logit}\|$ drives softmax toward one-hot outputs, collapsing entropy and killing gradient signal.

---

**Method 1: QK Layer Normalization (`QK_norm`):**

$$
\text{logit} = \frac{1}{\sqrt{d}} \text{LN}(X W^Q) \cdot \text{LN}(X W^K)^T
$$

where $\text{LN}(\cdot)$ is layer normalization applied independently to Q and K after projection. This decouples the magnitude and direction of Q, K — the dot product operates only on unit-variance vectors, bounding $|\text{logit}_{ij}| \leq \sqrt{d_k}$ in expectation.

---

**Method 2: Softmax Capping (`soft_cap`):**

$$
\text{capped\_softmax}(\text{logit}, c) = \text{softmax}\!\Big[\tanh\!\Big(\frac{\text{logit}}{c}\Big) \cdot c\Big]
$$

where $c$ is the capping coefficient (paper uses $c = 50$). The $\tanh$ nonlinearity squashes logits to $[-c, c]$, acting as an adaptive temperature control. For $|\text{logit}| \ll c$, the function is approximately identity; for $|\text{logit}| \gg c$, it saturates to $\pm c$.

---

**Method 3: Combined QK-Norm + Softmax Capping (`QK_norm_cap`):**

$$
\text{logit} = \frac{1}{\sqrt{d}} \text{LN}(X W^Q) \cdot \text{LN}(X W^K)^T
$$

$$
\text{attn} = \text{softmax}\!\Big[\tanh\!\Big(\frac{\text{logit}}{c}\Big) \cdot c\Big]
$$

QK-norm controls the **input magnitude** to the dot product. Softmax capping controls the **output magnitude** before softmax. Together they provide two layers of defense against logit explosion.

---

**Method 4: Post-QKV Layer Normalization (`QKV_norm`):**

Remove pre-normalization LN before QKV projection; instead apply LN to Q, K, V **after** the QKV projection:

$$
Q = \text{LN}(X W^{QKV}_{:Q}), \quad K = \text{LN}(X W^{QKV}_{:K}), \quad V = \text{LN}(X W^{QKV}_{:V})
$$

This normalizes all three projections, not just Q and K, addressing magnitude explosion in V as well. The pre-LN is removed because QKV outputs are already normalized, making double normalization redundant.

**Key Definitions:**

- $d$ — model dimension (hidden size, 1024 in paper)
- $d_k$ — query/key head dimension ($d / n_{\text{heads}} = 64$ in paper)
- $\text{LN}(\cdot)$ — layer normalization: $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$ with learnable $\gamma, \beta$
- $c$ — softmax capping coefficient ($c = 50$ in paper)
- $n_{\text{heads}} = 16$ — number of attention heads

## Complexity

| Operation | Baseline | With QK_norm_cap |
|-----------|----------|-----------------|
| Attention logit compute | $O(T^2 d_k)$ | $O(T^2 d_k)$ (unchanged) |
| LN on Q, K | — | $O(T \cdot d_k)$ per head (negligible) |
| tanh capping | — | $O(T^2)$ per head (elementwise, negligible) |
| Extra parameters | — | $4 d_k$ per head ($\gamma, \beta$ for Q and K LN) |

**Memory:** Negligible overhead — LN statistics ($\mu, \sigma$) are scalar per position/head. Capping is in-place elementwise.

**Wall-clock cost:** The paper reports no measurable throughput degradation. LN and tanh are cheap elementwise ops that fuse into the existing attention kernel (e.g., within FlashAttention's tiled computation).

**Training speedup via higher learning rate:**

| Method | Max stable LR | Relative to baseline | Perplexity (at LR=3e-4) |
|--------|--------------|---------------------|------------------------|
| bf16 baseline | 6e-3 | 1.0× | 11.19 |
| soft_cap | 40e-3 | 6.7× | 11.24 (no improvement) |
| QK_norm | 40e-3 | 6.7× | 10.84 |
| QK_norm_cap | **60e-3** | **10×** | **11.00** |
| QKV_norm | **60e-3** | **10×** | **10.85** |

The combined methods allow 1.5× higher learning rate vs. QK_norm alone (60e-3 vs. 40e-3), and 10× vs. baseline.

## Applicability

- **All transformer-based LLMs:** The techniques are architecture-agnostic within the standard transformer block. Used in production models:
  - **Gemma-2** (Google, 2024): Uses post-QKV normalization (similar to QKV_norm)
  - **Gemini** (Google, 2024): Uses softmax capping with $c = 50$
  - **PaLM** (Chowdhery et al., 2024): Uses z-loss for logit stability
  - **ViT-22B** (Dehghani et al., 2023): Uses QK layer normalization

- **Mixed-precision training (BF16/FP16):** Particularly important when training in reduced precision, where the limited mantissa makes logit explosion more likely and catastrophic.

- **Large learning rate schedules:** Critical for aggressive cosine/linear warmup schedules used in modern LLM pretraining, where transient learning rate spikes can trigger divergence.

- **Vision transformers:** Dehghani et al. (2023) showed QK-norm is essential for scaling ViTs to 22B parameters.

- **Linear attention / SSM hybrids:** Models using gated attention variants (Qiu et al., 2025; Qwen-Team, 2025) report using QK-norm and capping for stable low-precision training.

## Limitations

- **Softmax capping introduces a hyperparameter:** The capping coefficient $c$ must be tuned. Too small ($c \ll$ typical logit range) constrains model expressivity; too large ($c \gg$ logit range) provides no benefit. The paper uses $c = 50$ without extensive ablation.

- **QK-norm slightly constrains the logit distribution:** By forcing Q, K to unit variance, QK-norm removes the model's ability to use Q/K magnitude as a signal. In practice this doesn't hurt perplexity (Table 4 shows improvement), but it may limit certain attention patterns.

- **tanh gradient saturation:** For logits near the cap boundary, $\tanh'(x/c) \approx 0$, which can slow gradient flow through very large logits. This is intentional (it prevents outlier logits from dominating), but it means the model can't learn to produce extreme attention patterns.

- **Validated only at 830M scale:** The paper uses a small GPT-2-like model (830M parameters, 24 layers). The authors note future work is needed to validate at larger scales, though the constituent techniques (QK-norm, softmax capping) are used individually in production models up to hundreds of billions of parameters.

- **Doesn't address all instability sources:** Training instability can also arise from optimizer states (AdamW momentum), gradient accumulation issues, data ordering, and output logit divergence (z-loss addresses this). QK_norm_cap specifically targets the attention mechanism.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

class StableMultiHeadAttention(torch.nn.Module):
    """
    Attention with QK-norm + softmax capping (QK_norm_cap).

    Key changes from standard attention:
    1. LayerNorm on Q, K after projection (QK_norm)
    2. tanh-based capping before softmax (soft_cap)

    Both are cheap elementwise ops that fuse into FlashAttention.
    """
    def __init__(self, d_model, n_heads, capping=50.0):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.capping = capping

        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

        # QK LayerNorm (per-head, applied after QKV projection)
        self.q_norm = torch.nn.LayerNorm(self.d_k)
        self.k_norm = torch.nn.LayerNorm(self.d_k)

    def forward(self, x):
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head: (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # === QK LAYER NORMALIZATION ===
        # Constrains Q, K magnitudes → bounds logit range
        q = self.q_norm(q)  # LayerNorm over d_k dimension
        k = self.k_norm(k)  # LayerNorm over d_k dimension

        # Standard scaled dot-product
        scale = self.d_k ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # === SOFTMAX CAPPING ===
        # tanh squashes extreme logits to [-capping, capping]
        # For |logit| << capping: tanh(logit/c)*c ≈ logit (identity)
        # For |logit| >> capping: tanh(logit/c)*c ≈ ±c (saturates)
        logits = torch.tanh(logits / self.capping) * self.capping

        # Causal mask (if needed)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(causal_mask, float('-inf'))

        # Standard softmax + attention
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

# Alternative: QKV_norm (post-QKV normalization, remove pre-LN)
class QKVNormAttention(torch.nn.Module):
    """
    Variant: Apply LayerNorm AFTER the QKV projection to all three.
    Remove the pre-normalization LN before QKV.
    Achieves best perplexity (10.85 vs 11.19 baseline).
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # NOTE: No pre-LN before this projection
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

        # Post-projection normalization for Q, K, AND V
        self.q_norm = torch.nn.LayerNorm(self.d_k)
        self.k_norm = torch.nn.LayerNorm(self.d_k)
        self.v_norm = torch.nn.LayerNorm(self.d_k)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Normalize ALL three projections
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Standard attention (no capping needed — norms are controlled)
        scale = self.d_k ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

# GPU efficiency notes:
# - LayerNorm on Q, K: reduction over d_k (64 elements) → trivially fast
# - tanh capping: elementwise op, fuses into attention kernel
# - No extra matmuls, no extra kernel launches
# - Memory: +4*d_k params per head for LN (negligible)
# - Can be fused into FlashAttention by modifying the score computation
#   within each tile (add LN before score, tanh after score)
```

## References

- Rybakov, O., Chrzanowski, M., Dykas, P., Xue, J., & Lanir, B. (2024). Methods of Improving LLM Training Stability. arXiv:2410.16682.
- Dehghani, M. et al. (2023). Scaling Vision Transformers to 22 Billion Parameters. ICML 2023.
- Henry, A., Dachapally, P.R., Pawar, S., & Chen, Y. (2020). Query-Key Normalization for Transformers. arXiv:2010.04245.
- Bello, I. et al. (2017). Neural Combinatorial Optimization with Reinforcement Learning. ICLR 2017.
- Wortsman, M. et al. (2024). Small-Scale Proxies for Large-Scale Transformer Training Instabilities. ICLR 2024.
- Zhai, S. et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML 2023.
- Chowdhery, A. et al. (2024). PaLM: Scaling Language Modeling with Pathways. JMLR 2024.
- Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Touvron, H. et al. (2021). Going Deeper with Image Transformers. ICCV 2021.

# 228: Tensor Product Attention (TPA) — Factorized KV Cache

**Category**: decomposition
**Gain type**: efficiency
**Source**: Zhang et al., "Tensor Product Attention Is All You Need", NeurIPS 2025 (arXiv:2501.06425)
**Paper**: [papers/tensor-product-attention.pdf]
**Documented**: 2026-02-15

## Description

Tensor Product Attention (TPA) replaces standard multi-head attention projections with *contextual* low-rank tensor product decompositions of queries, keys, and values. Instead of projecting each token into full $h \times d_h$ matrices for Q, K, V, TPA factorizes each into a sum of rank-1 tensor products whose factors depend on the input token. At inference time, only the compact factors are cached (not full K, V), yielding 10x+ KV cache reduction while matching or exceeding MHA quality. The factorization naturally maps to matmul/einsum operations (tensor-core friendly), reduces HBM bandwidth during autoregressive decoding, and is natively compatible with RoPE.

## Mathematical Form

**Core Operation:**

For each token $t$, TPA factorizes the query, key, and value slices as:

$$
\mathbf{Q}_t = \frac{1}{R_Q} \sum_{r=1}^{R_Q} \mathbf{a}_r^Q(\mathbf{x}_t) \otimes \mathbf{b}_r^Q(\mathbf{x}_t), \quad \mathbf{K}_t = \frac{1}{R_K} \sum_{r=1}^{R_K} \mathbf{a}_r^K(\mathbf{x}_t) \otimes \mathbf{b}_r^K(\mathbf{x}_t), \quad \mathbf{V}_t = \frac{1}{R_V} \sum_{r=1}^{R_V} \mathbf{a}_r^V(\mathbf{x}_t) \otimes \mathbf{b}_r^V(\mathbf{x}_t)
$$

where $\mathbf{a}_r^Q(\mathbf{x}_t) \in \mathbb{R}^h$, $\mathbf{b}_r^Q(\mathbf{x}_t) \in \mathbb{R}^{d_h}$ are *contextual* (input-dependent) factors, and $\otimes$ denotes the outer product. Each $\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{h \times d_h}$.

**Latent Factor Maps (linear projections):**

$$
\mathbf{a}^Q(\mathbf{x}_t) = \mathbf{W}^{a^Q} \mathbf{x}_t \in \mathbb{R}^{R_Q \cdot h}, \quad \mathbf{b}^Q(\mathbf{x}_t) = \mathbf{W}^{b^Q} \mathbf{x}_t \in \mathbb{R}^{R_Q \cdot d_h}
$$

Reshaped into $\mathbf{A}_Q(\mathbf{x}_t) \in \mathbb{R}^{R_Q \times h}$ and $\mathbf{B}_Q(\mathbf{x}_t) \in \mathbb{R}^{R_Q \times d_h}$, the query tensor is:

$$
\mathbf{Q}_t = \frac{1}{R_Q} \mathbf{A}_Q(\mathbf{x}_t)^\top \mathbf{B}_Q(\mathbf{x}_t) \in \mathbb{R}^{h \times d_h}
$$

**RoPE Compatibility (pre-rotation):**

RoPE is applied to the $\mathbf{b}^K$ factor before caching:

$$
\widetilde{\mathbf{B}}_K(\mathbf{x}_t) := \text{RoPE}_t\big(\mathbf{B}_K(\mathbf{x}_t)\big)
$$

This preserves the relative-position identity: $\widetilde{\mathbf{Q}}_t \widetilde{\mathbf{K}}_s^\top = \mathbf{Q}_t \mathbf{T}_{t-s} \mathbf{K}_s^\top$ where $\mathbf{T}_{t-s}$ is the RoPE rotation matrix.

**KV Caching:** Instead of caching $\mathbf{K}_t \in \mathbb{R}^{h \times d_h}$ and $\mathbf{V}_t \in \mathbb{R}^{h \times d_h}$, TPA caches:

$$
\mathbf{A}_K(\mathbf{x}_t) \in \mathbb{R}^{R_K \times h}, \quad \widetilde{\mathbf{B}}_K(\mathbf{x}_t) \in \mathbb{R}^{R_K \times d_h}, \quad \mathbf{A}_V(\mathbf{x}_t) \in \mathbb{R}^{R_V \times h}, \quad \mathbf{B}_V(\mathbf{x}_t) \in \mathbb{R}^{R_V \times d_h}
$$

**Key Definitions:**

- $h$ — number of attention heads
- $d_h$ — dimension per head (typically 64 or 128)
- $R_Q, R_K, R_V$ — tensor product ranks for Q, K, V respectively
- $d_{\text{model}}$ — model embedding dimension
- $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ — input token embedding

**Unification with existing mechanisms:**

MHA, MQA, and GQA are all special cases of non-contextual TPA:
- **MHA**: $R_Q = R_K = R_V = h$, head-dimension factors are fixed basis vectors $\mathbf{e}_i$
- **MQA**: $R_K = R_V = 1$, head-dimension factor is $\mathbf{1}_h$ (all-ones vector)
- **GQA** with $G$ groups: $R_K = R_V = G$, head-dimension factors are group masks

## Complexity

**KV Cache Memory per Token:**

| Method | KV Cache per Token |
|--------|-------------------|
| MHA | $2 h d_h$ |
| MQA | $2 d_h$ |
| GQA ($G$ groups) | $2 G d_h$ |
| MLA | $d_c + d_h^R$ |
| **TPA** | $(R_K + R_V)(h + d_h)$ |

For typical settings ($h=32$, $d_h=64$, $R_K = R_V = 1$):
- MHA: $2 \times 32 \times 64 = 4096$ elements per token
- TPA: $(1+1)(32+64) = 192$ elements per token → **21x reduction**

**Parameter Count:**

| Method | Parameters |
|--------|-----------|
| MHA | $4 d_{\text{model}} h d_h$ |
| TPA | $d_{\text{model}}(R_Q + R_K + R_V)(h + d_h) + d_{\text{model}} h d_h$ |

**FlashTPA Decoding:** Operates via a sequence of Einstein summations on factorized components, avoiding materialization of full Q, K, V tensors. Competitive with or faster than FlashMHA/FlashGQA/FlashMLA at long sequences.

## Applicability

- **Autoregressive LLM inference**: Primary benefit — dramatically reduces KV cache memory, enabling longer contexts on fixed hardware
- **Pretraining**: Drop-in replacement for MHA in LLaMA/Qwen/Gemma-style architectures; trains as fast or faster than MHA with lower validation loss
- **Long-context serving**: At sequence length $2^{19}$ (~500K tokens), FlashTPA decoding outperforms all baselines
- **Scales with model size**: Validated at 124M, 353M, 773M, and 1.5B parameters on FineWeb-Edu-100B

## GPU Efficiency Analysis

**Memory Access Pattern:**
- Factor projections are standard linear layers (coalesced memory access)
- KV cache reads during decoding load $R_K(h + d_h)$ elements per token instead of $2 h d_h$ — significant HBM bandwidth reduction
- FlashTPA Decoding uses einsum operations that map naturally to tensor cores

**Parallelism:**
- Factor computation is embarrassingly parallel across tokens
- Attention computation proceeds head-by-head as in standard MHA — no new sequential bottlenecks
- Triton implementation available; CUDA kernel under development

**Arithmetic Intensity:**
- During decoding, the bottleneck shifts from memory-bound (loading large KV cache) to more compute-bound with smaller factors
- Reduces total HBM reads per decoding step proportional to $\frac{(R_K + R_V)(h + d_h)}{2 h d_h}$

**Tensor Core Utilization:**
- All operations are matmuls or batched outer products — natural WGMMA/MMA targets
- Pre-rotation of RoPE factors avoids runtime rotation overhead during decoding

## Limitations

- **Training overhead**: Factor projection adds parameters $d_{\text{model}}(R_Q + R_K + R_V)(h + d_h)$; at very low ranks this is small, but at higher ranks approaches MHA parameter count
- **Rank selection**: Optimal $R_Q, R_K, R_V$ must be tuned per architecture; paper uses $(16, 1, 1)$ as default
- **Prefill performance**: During prefill (training forward pass), TPA computes full Q, K, V anyway, so the speedup is primarily at inference decoding
- **Current Triton implementation**: FlashTPA is in Triton (not hand-optimized CUDA), so further speedups expected from CUDA implementation
- **Expressiveness at very low rank**: Rank-1 K,V factors may lose head-specific capacity in some settings (mitigated by TPA-KVonly variant)

## Implementation Notes

```python
# Core TPA forward pass (simplified)
import torch

def tpa_forward(x, W_aQ, W_bQ, W_aK, W_bK, W_aV, W_bV, W_O, R_Q, R_K, R_V, h, d_h):
    """
    x: [B, T, d_model]
    W_aQ: [d_model, R_Q * h], W_bQ: [d_model, R_Q * d_h]
    Returns: [B, T, d_model]
    """
    B, T, d = x.shape

    # Compute contextual factors
    A_Q = (x @ W_aQ).view(B, T, R_Q, h)      # [B, T, R_Q, h]
    B_Q = (x @ W_bQ).view(B, T, R_Q, d_h)    # [B, T, R_Q, d_h]
    A_K = (x @ W_aK).view(B, T, R_K, h)
    B_K = (x @ W_bK).view(B, T, R_K, d_h)    # Apply RoPE here
    A_V = (x @ W_aV).view(B, T, R_V, h)
    B_V = (x @ W_bV).view(B, T, R_V, d_h)

    # Form Q, K, V via tensor product: sum of outer products over rank dim
    # Q_t = (1/R_Q) * A_Q^T @ B_Q  →  [B, T, h, d_h]
    Q = torch.einsum('btrh,btrd->bthd', A_Q, B_Q) / R_Q
    K = torch.einsum('btrh,btrd->bthd', A_K, B_K) / R_K
    V = torch.einsum('btrh,btrd->bthd', A_V, B_V) / R_V

    # Standard scaled dot-product attention
    scores = torch.einsum('bihd,bjhd->bhij', Q, K) / (d_h ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bhij,bjhd->bihd', attn, V)

    # Output projection
    out = out.reshape(B, T, h * d_h) @ W_O
    return out

# KV cache at inference: store only factors, not full K, V
# Cache: A_K[t], B_K_rotated[t], A_V[t], B_V[t]
# Memory: (R_K + R_V)(h + d_h) per token vs 2*h*d_h for MHA
```

## References

- Zhang, Y., Liu, Y., Yuan, H., Qin, Z., Yuan, Y., Gu, Q., & Yao, A.C. (2025). Tensor Product Attention Is All You Need. NeurIPS 2025. arXiv:2501.06425.
- Project page: https://github.com/tensorgi/TPA
- DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. (MLA baseline comparison)
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. (MQA)
- Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. (GQA)

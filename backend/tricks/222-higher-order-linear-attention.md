# 222: Higher-Order Linear Attention (HLA)

**Category**: approximation
**Gain type**: expressivity
**Source**: Zhang, Qin & Gu (2025) — "Higher-order Linear Attention" (arXiv:2510.27258)
**Paper**: [papers/higher-order-linear-attention.pdf]
**Documented**: 2026-02-15

## Description

Standard linear attention replaces the softmax kernel with a feature map $\phi$ and maintains a first-order running sum $\sum \phi(\mathbf{k}_j) \mathbf{v}_j^\top$, yielding $O(n)$ streaming but limited expressivity (equivalent to a rank-$r$ kernel). **Higher-order Linear Attention (HLA)** generalizes this by maintaining second-moment (and higher) prefix summaries of keys, enabling a data-adaptive polynomial kernel $K_t(\mathbf{q}, \mathbf{q}') = \mathbf{q}^\top \mathbf{S}_t^K \mathbf{q}'$ that depends on all past keys. The key insight: the second-order attention matrix $\mathbf{T}_2 = (\mathbf{Q}\mathbf{K}^\top)(\mathbf{Q}\mathbf{K}^\top)^\top$ factors through the second moment $\mathbf{K}^\top\mathbf{K} \in \mathbb{R}^{d \times d}$, which can be accumulated as a compact prefix summary.

The challenge is enforcing **strict autoregressive causality** at second order without materializing $n \times n$ matrices. HLA solves this by introducing two additional cross-summaries ($\mathbf{G}_t$, $\mathbf{h}_t$) that correct for the causal mask, proven exact via a masked streaming identity (Theorem 3.1). The resulting algorithm is:

1. **Streaming**: $O(1)$ memory per token, $O(d^2 + d \, d_v)$ compute per token
2. **Parallel trainable**: via an associative (semidirect-product) scan operator
3. **Strictly causal**: produces activations identical to a serial recurrence
4. **Drop-in replacement**: for standard attention in transformer blocks

HLA extends to third-order and beyond, with the $p$-th order maintaining $O(d^p)$ state and computing degree-$p$ polynomial kernels. An asymmetric variant (AHLA) computes $\mathbf{A}\mathbf{A}^\top\mathbf{V}$ instead of $\mathbf{A}^\top\mathbf{A}\mathbf{V}$, providing complementary inductive biases at the same cost.

## Mathematical Form

**Core Operation (Second-order, unnormalized):**

The unnormalized output at time $t$ is:

$$
\mathbf{o}_t = \mathbf{q}_t^\top \mathbf{S}_t^K \mathbf{C}_t^{QV}
$$

where the prefix summaries are:

$$
\mathbf{S}_t^K := \sum_{i \leq t} \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d \times d}, \quad \mathbf{C}_t^{QV} := \sum_{i \leq t} \mathbf{q}_i \mathbf{v}_i^\top \in \mathbb{R}^{d \times d_v}, \quad \mathbf{m}_t^Q := \sum_{i \leq t} \mathbf{q}_i \in \mathbb{R}^d
$$

**Normalized variant:**

$$
\mathbf{o}_t = \frac{\mathbf{q}_t^\top \mathbf{S}_t^K \mathbf{C}_t^{QV}}{\mathbf{q}_t^\top \mathbf{S}_t^K \mathbf{m}_t^Q + \varepsilon}
$$

**Masked streaming identity (Theorem 3.1):** For strict causal masking, define cross-summaries:

$$
\mathbf{G}_t := \sum_{i \leq t} \left(\mathbf{k}_i \mathbf{k}_i^\top\right) \mathbf{C}_{i-1}^{QV} \in \mathbb{R}^{d \times d_v}, \quad \mathbf{h}_t := \sum_{i \leq t} \left(\mathbf{k}_i \mathbf{k}_i^\top\right) \mathbf{m}_{i-1}^Q \in \mathbb{R}^d
$$

Then the strictly causal output is:

$$
\mathbf{o}_t^{\text{mask}} = \mathbf{q}_t^\top \left(\mathbf{S}_t^K \mathbf{C}_t^{QV} - \mathbf{G}_t\right)
$$

**Online updates** (per-token, all $O(1)$ amortized):

$$
\mathbf{S}_t^K = \mathbf{S}_{t-1}^K + \mathbf{k}_t \mathbf{k}_t^\top, \quad \mathbf{C}_t^{QV} = \mathbf{C}_{t-1}^{QV} + \mathbf{q}_t \mathbf{v}_t^\top, \quad \mathbf{m}_t^Q = \mathbf{m}_{t-1}^Q + \mathbf{q}_t
$$

$$
\mathbf{G}_t = \mathbf{G}_{t-1} + \mathbf{k}_t (\mathbf{k}_t^\top \mathbf{C}_{t-1}^{QV}), \quad \mathbf{h}_t = \mathbf{h}_{t-1} + \mathbf{k}_t (\mathbf{k}_t^\top \mathbf{m}_{t-1}^Q)
$$

**With exponential decay** $\gamma \in (0, 1)$:

$$
\mathbf{S}_t^K = \gamma \mathbf{S}_{t-1}^K + \mathbf{k}_t \mathbf{k}_t^\top, \quad \mathbf{G}_t = \gamma \mathbf{G}_{t-1} + \mathbf{k}_t (\mathbf{k}_t^\top \mathbf{C}_{t-1}^{QV})
$$

**Key Definitions:**

- $\mathbf{q}_t, \mathbf{k}_t \in \mathbb{R}^d$ — query and key vectors at time $t$
- $\mathbf{v}_t \in \mathbb{R}^{d_v}$ — value vector at time $t$
- $\mathbf{S}_t^K \in \mathbb{R}^{d \times d}$ — second-moment key metric (learned, data-dependent kernel on query space)
- $\mathbf{C}_t^{QV} \in \mathbb{R}^{d \times d_v}$ — query-value accumulator
- $\mathbf{G}_t \in \mathbb{R}^{d \times d_v}$ — cross-summary for causal correction (numerator)
- $\mathbf{h}_t \in \mathbb{R}^d$ — cross-summary for causal correction (denominator)

**Associative scan operator (masked semidirect product):**

$$
(\mathbf{S}_A, \mathbf{C}_A, \mathbf{m}_A, \mathbf{G}_A, \mathbf{h}_A) \oplus (\mathbf{S}_B, \mathbf{C}_B, \mathbf{m}_B, \mathbf{G}_B, \mathbf{h}_B) = \left(\mathbf{S}_A + \mathbf{S}_B,\; \mathbf{C}_A + \mathbf{C}_B,\; \mathbf{m}_A + \mathbf{m}_B,\; \mathbf{G}_A + \mathbf{G}_B + \mathbf{S}_B \mathbf{C}_A,\; \mathbf{h}_A + \mathbf{h}_B + \mathbf{S}_B \mathbf{m}_A\right)
$$

This is associative (semidirect product structure), enabling Blelloch parallel prefix scans for chunk-parallel training.

## Complexity

| Operation | First-order Linear Attn | Second-order HLA |
|-----------|------------------------|------------------|
| Per-token compute | $O(r \, d_v)$ | $O(d^2 + d \, d_v)$ |
| Per-token state | $O(r \, d_v)$ | $O(d^2 + d \, d_v)$ |
| Kernel expressivity | Rank-$r$ fixed | Degree-2 polynomial, data-adaptive |
| Sequence dependence | $O(n)$ total | $O(n)$ total |
| Training (chunkwise) | $O(n \, w)$ | $O(n \, w)$ with scan |

**Memory (per head, second-order masked):** $O(d^2 + d \, d_v)$ — five matrices: $\mathbf{S}^K$ ($d \times d$, symmetric so $\frac{d(d+1)}{2}$), $\mathbf{C}^{QV}$ ($d \times d_v$), $\mathbf{m}^Q$ ($d$), $\mathbf{G}$ ($d \times d_v$), $\mathbf{h}$ ($d$).

**Multi-query optimization:** With $\mathbf{K}, \mathbf{V}$ shared across $h$ heads, $\mathbf{S}_t^K$ is shared ($O(d^2)$ once), reducing total per-layer memory from $O(h \, d^2 + h \, d \, d_v)$ to $O(d^2 + h \, d \, d_v)$.

## Applicability

- **Drop-in attention replacement** in decoder-only transformers for language modeling. HLA replaces only the attention sublayer; FFN, normalization, and positional encodings are unchanged.
- **Long-context models** where $O(n^2)$ softmax attention is prohibitive — HLA provides $O(n)$ streaming with richer-than-linear-attention expressivity.
- **Inference-bound applications** (chatbots, code assistants): constant $O(1)$ per-token state updates vs. growing KV cache.
- **Hybrid architectures**: combine HLA layers with standard attention layers to get the best of both worlds.
- **Third-order extension** provides even richer polynomial kernels at $O(d^3)$ state cost, suitable for smaller head dimensions.

## Limitations

- **State size scales as $O(d^2)$** per head for second-order, $O(d^3)$ for third-order. With head dimension $d = 128$, the $d \times d$ key metric is 16K entries — manageable in SRAM for moderate $d$ but may pressure registers/shared memory for large $d$.
- **Not a matmul** at the token level: the per-token output involves a matvec $\mathbf{q}_t^\top \mathbf{S}_t^K$ followed by a row-matrix multiply, not a single large GEMM. Tensor core utilization depends on chunk-parallel training where the intra-chunk computation is a proper matmul.
- **Cross-summary updates** ($\mathbf{G}_t$, $\mathbf{h}_t$) involve products like $\mathbf{k}_t (\mathbf{k}_t^\top \mathbf{C}_{t-1}^{QV})$ — outer-product-like operations that may have lower arithmetic intensity than pure GEMMs.
- **Empirical validation so far on moderate scale** (up to 1.3B-scale referenced in paper, with experiments on 360M). Real wall-clock comparisons against FlashAttention-2 at scale are needed.
- **Scan operator state is 5-tuple** of matrices, which increases the communication cost of the parallel prefix scan compared to first-order linear attention (3-tuple: $\mathbf{S}, \mathbf{C}, \mathbf{m}$).

## Implementation Notes

```python
# Second-order HLA with causal mask — streaming kernel (per head)
# State: S_K (d x d), C_QV (d x d_v), m_Q (d,), G (d x d_v), h (d,)
# decay gamma in (0, 1]

def hla_step(q_t, k_t, v_t, S_K, C_QV, m_Q, G, h, gamma=1.0):
    """One streaming step of masked second-order HLA."""
    # Save previous for cross-summary updates
    C_QV_prev, m_Q_prev = C_QV, m_Q

    # First-order summary updates (with optional decay)
    S_K  = gamma * S_K  + k_t[:, None] * k_t[None, :]   # d x d outer product
    C_QV = gamma * C_QV + q_t[:, None] * v_t[None, :]   # d x d_v
    m_Q  = gamma * m_Q  + q_t                            # d

    # Cross-summary updates for causal correction
    kk_dot_C = k_t * (k_t @ C_QV_prev)  # k_t * scalar -> d x d_v via broadcast
    # Actually: k_t (k_t^T C_prev) is (d,) x (1, d_v) = (d, d_v)
    r = k_t @ C_QV_prev          # (d_v,) — k_t^T C_prev
    G = gamma * G + k_t[:, None] * r[None, :]  # (d, d_v)

    s = k_t @ m_Q_prev           # scalar — k_t^T m_prev
    h = gamma * h + k_t * s      # (d,)

    # Output: masked unnormalized
    u = q_t @ S_K               # (d,) — q_t^T S_K, cost O(d^2)
    num = u @ C_QV - q_t @ G    # (d_v,) — matvec + correction
    # Optional normalization:
    # den = u @ m_Q - q_t @ h + eps
    # o_t = num / den
    return num, S_K, C_QV, m_Q, G, h

# For training: use chunkwise parallel scan with the associative operator
# (S_A, C_A, m_A, G_A, h_A) ⊕ (S_B, C_B, m_B, G_B, h_B) =
#   (S_A+S_B, C_A+C_B, m_A+m_B, G_A+G_B+S_B@C_A, h_A+h_B+S_B@m_A)
# The cross-terms S_B @ C_A are matmuls → tensor-core friendly in chunks.
```

**GPU efficiency notes:**
- **Training (chunkwise):** Intra-chunk scan is $O(\log w)$ span; inter-chunk scan is $O(B_c)$ chunks. Cross-term $\mathbf{S}_B \mathbf{C}_A$ in the scan operator is a $d \times d$ by $d \times d_v$ matmul — maps to tensor cores.
- **Inference (streaming):** Per-token cost dominated by $\mathbf{q}_t^\top \mathbf{S}_t^K$ (matvec, $O(d^2)$) and $\mathbf{u}_t \mathbf{C}_t^{QV}$ (matvec, $O(d \, d_v)$). These are memory-bandwidth-bound for typical $d = 64$–$128$.
- **Symmetric packing:** Store $\mathbf{S}_t^K$ as upper triangle ($\frac{d(d+1)}{2}$ entries) to halve bandwidth.
- **Multi-query sharing:** $\mathbf{S}_t^K$ shared across heads when $\mathbf{K}$ is shared, amortizing the $O(d^2)$ state.

## References

- Zhang, Y., Qin, Z., & Gu, Q. (2025). Higher-order Linear Attention. arXiv:2510.27258. [https://arxiv.org/abs/2510.27258](https://arxiv.org/abs/2510.27258)
- Project page: [https://github.com/yifanzhang-pro/HLA](https://github.com/yifanzhang-pro/HLA)
- Wang et al. (2020). Linformer / Linear Transformers (first-order linear attention baseline)
- Yang et al. (2024b). Gated Linear Attention (GLA) — first-order with gating
- Dao & Gu (2024). Mamba-2 / SSD — structured state space duality with chunkwise scans
- Blelloch (1990). Prefix sums and their applications (parallel scan foundation)

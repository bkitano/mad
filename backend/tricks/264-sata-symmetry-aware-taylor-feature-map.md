# 264: SATA — Symmetry-Aware Taylor Minimal Polynomial Feature Map

**Category**: approximation
**Gain type**: efficiency
**Source**: Heinsen & Kozachkov (2026) — "Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation" (arXiv:2602.00294)
**Paper**: [papers/sata-symmetry-aware-taylor.pdf]
**Documented**: 2026-02-16

## Description

Standard linear attention with polynomial feature maps approximates the softmax kernel $\exp(\mathbf{q}^\top \mathbf{k} / c)$ via a Taylor expansion of order $P$. The degree-$p$ term involves $(q^\top k)^p = \langle \mathbf{q}^{\otimes p}, \mathbf{k}^{\otimes p} \rangle$, which naively requires a $d_K^p$-dimensional feature map — exponentially large and infeasible for $p > 2$ at typical head dimensions ($d_K = 64$–$128$). This is why prior Taylor-based linear attention (BASED, trick 210) stopped at $p = 2$.

**SATA (Symmetry-Aware Taylor Approximation)** observes that $\mathbf{q}^{\otimes p}$ and $\mathbf{k}^{\otimes p}$ are **symmetric tensors**: their entries are invariant under any permutation of the $p$ indices. The $d_K^p$ entries of $\mathbf{q}^{\otimes p}$ contain only $m_p = \binom{d_K + p - 1}{p}$ **unique monomials** (the "upper hyper-triangular" region indexed by $i_1 \leq i_2 \leq \cdots \leq i_p$). Each unique monomial appears with a **combinatorial multiplicity** (the number of distinct index permutations that produce it).

SATA defines a **minimal feature map** $\Phi_p : \mathbb{R}^{d_K} \to \mathbb{R}^{m_p}$ that extracts only the $m_p$ unique monomials, tightly packed in a vector, and a **diagonal weight matrix** $C_p$ that accounts for their multiplicities. This yields:

$$
(q^\top k)^p = \langle \Phi_p(q), \Phi_p(k) \rangle_{C_p}
$$

The resulting linear attention has **constant cost per token** (independent of context length $n$), with hidden state size $(d_V + 1) \binom{d_K + P - 1}{P - 1}$ and constant FLOPs per token. The key gains:

1. **Massive dimension reduction**: At degree $p = 4$ with $d_K = 64$: naive $64^4 = 16.8$M features vs SATA $\binom{67}{4} = 766{,}480$ — a **22$\times$** reduction. At $p = 2$: $64^2 = 4096$ vs $\binom{65}{2} = 2080$ — a **2$\times$** reduction (exactly symmetric packing of the $d \times d$ matrix).
2. **Exact (no approximation)**: Unlike TensorSketch (trick 263) or FAVOR+ (trick 45), the symmetry reduction introduces zero approximation error — the only error comes from Taylor truncation.
3. **Embarrassingly parallel**: The feature map $\Phi_p(\mathbf{x})$ is a pointwise gather-and-multiply operation: index into $\mathbf{x}$ using precomputed index matrix $M_p$, then take the product along the last dimension.
4. **Independent scans per degree**: Each Taylor term $p = 0, 1, \ldots, P-1$ has an independent pair of accumulated states $(Z_{p,T}, S_{p,T})$, enabling $P$ parallel prefix scans.

## Mathematical Form

**Taylor expansion of the softmax kernel:**

$$
\exp\!\left(\frac{\mathbf{q}^\top \mathbf{k}}{c}\right) = \sum_{p=0}^{\infty} a_p \, (\mathbf{q}^\top \mathbf{k})^p, \qquad a_p := \frac{1}{p!\, c^p}
$$

where $c = \sqrt{d_K}$ is the conventional scaling constant.

**Decomposition into symmetric tensor products:**

$$
(\mathbf{q}^\top \mathbf{k})^p = \sum_{i_1=1}^{d_K} \cdots \sum_{i_p=1}^{d_K} (q_{i_1} \cdots q_{i_p})(k_{i_1} \cdots k_{i_p}) = \sum (\mathbf{q}^{\otimes p}) \odot (\mathbf{k}^{\otimes p})
$$

where $\mathbf{q}^{\otimes p} = \underbrace{\mathbf{q} \otimes \cdots \otimes \mathbf{q}}_{p}$ is an order-$p$ symmetric tensor with $d_K^p$ entries.

**Minimal basis identification:**

The symmetric tensor $\mathbf{q}^{\otimes p}$ has only $m_p$ unique entries, indexed by the upper hyper-triangular region $i_1 \leq i_2 \leq \cdots \leq i_p$:

$$
m_p = \binom{d_K + p - 1}{p}
$$

Each unique monomial $q_{i_1} q_{i_2} \cdots q_{i_p}$ appears with multiplicity equal to the multinomial coefficient (number of distinct permutations of the index tuple).

**Minimal feature map:**

$$
\Phi_p(\mathbf{x}) := \left[\prod_{j=1}^{p} x_{M_{p_{ij}}}\right]_{i=1,2,\ldots,m_p} \in \mathbb{R}^{m_p}
$$

where $M_p \in \{1, \ldots, d_K\}^{m_p \times p}$ is a constant precomputed index matrix whose $i$-th row contains the sorted index tuple $(i_1, i_2, \ldots, i_p)$ with $i_1 \leq i_2 \leq \cdots \leq i_p$.

In pseudocode: `Phi(x): return x[..., M].prod(dim=-1)`

**Weighted inner product:**

$$
(\mathbf{q}^\top \mathbf{k})^p = \langle \Phi_p(\mathbf{q}), \Phi_p(\mathbf{k}) \rangle_{C_p}
$$

where $C_p \in \mathbb{R}^{m_p \times m_p}$ is a constant diagonal matrix whose entries are the multiplicities of each monomial.

**Truncated Taylor approximation (order $P$):**

$$
\exp\!\left(\frac{\mathbf{q}^\top \mathbf{k}}{c}\right) \approx \sum_{p=0}^{P-1} a_p \langle \Phi_p(\mathbf{q}), \Phi_p(\mathbf{k}) \rangle_{C_p}
$$

**Causal linear attention with accumulated states:**

For each degree $p$, define accumulated states:

$$
Z_{p,T} := a_p \sum_{t=1}^{T} \Phi_p(\mathbf{k}_t), \qquad S_{p,T} := a_p \sum_{t=1}^{T} \Phi_p(\mathbf{k}_t) \mathbf{v}_t^\top
$$

These satisfy the linear recurrences:

$$
Z_{p,T} = Z_{p,T-1} + a_p \Phi_p(\mathbf{k}_T), \qquad S_{p,T} = S_{p,T-1} + a_p \Phi_p(\mathbf{k}_T) \mathbf{v}_T^\top
$$

**Final output (aggregating over all degrees):**

$$
\mathbf{y}_T \approx \frac{S_T}{Z_T}, \qquad S_T := \sum_{p=0}^{P-1} S_{p,T}^\top \Phi_p(\mathbf{q}_T), \qquad Z_T := \sum_{p=0}^{P-1} \langle \Phi_p(\mathbf{q}_T), Z_{p,T} \rangle_{C_p}
$$

**Key Definitions:**

- $d_K$ — key/query head dimension
- $d_V$ — value head dimension
- $P$ — number of Taylor terms (controls precision; $P = 4$ suffices for float16 accuracy)
- $m_p = \binom{d_K + p - 1}{p}$ — number of unique degree-$p$ monomials
- $M_p \in \{1, \ldots, d_K\}^{m_p \times p}$ — constant index matrix (precomputed once)
- $C_p \in \mathbb{R}^{m_p \times m_p}$ — constant diagonal multiplicity matrix (precomputed once)
- $Z_{p,T} \in \mathbb{R}^{m_p}$ — accumulated key features for degree $p$ (denominator state)
- $S_{p,T} \in \mathbb{R}^{m_p \times d_V}$ — accumulated key-value features for degree $p$ (numerator state)

## Complexity

**Hidden state size per head (all $P$ terms):**

$$
(d_V + 1) \binom{d_K + P - 1}{P - 1}
$$

**FLOPs per token (all $P$ terms):**

$$
\left(4d_V + \frac{2(P \, d_K + 1)}{d_K + 1} + 2\right) \binom{d_K + P - 1}{P - 1}
$$

| $d_K$ | $P$ | Naive dim $\sum d_K^p$ | SATA dim $\binom{d_K+P-1}{P-1}$ | Reduction |
|-------|-----|----------------------|-------------------------------|-----------|
| 16 | 4 | 69,905 | 3,876 | 18$\times$ |
| 32 | 4 | 1,082,401 | 40,920 | 26$\times$ |
| 64 | 4 | 16,843,009 | 766,480 | 22$\times$ |
| 16 | 2 | 273 | 153 | 1.8$\times$ |
| 64 | 2 | 4,161 | 2,145 | 1.9$\times$ |

**Comparison to conventional attention (KV cache):**

| Metric | Conventional Attention | SATA ($P = 4$) |
|--------|----------------------|----------------|
| State size | $n(d_K + d_V)$ — grows with $n$ | $(d_V + 1)\binom{d_K + P - 1}{P - 1}$ — **constant** |
| FLOPs/token | $n(2d_K + 2d_V + 3)$ — grows with $n$ | **constant** (see formula above) |
| Scaling | $O(n)$ per token | $O(1)$ per token |

**Comparison to HLA (trick 222) at second order:**

HLA maintains $\mathbf{S}_t^K \in \mathbb{R}^{d \times d}$ (symmetric, $d(d+1)/2$ unique entries) plus cross-summaries. SATA at $p = 2$ stores exactly $m_2 = \binom{d+1}{2} = d(d+1)/2$ features — the same count, confirming that SATA's minimal basis at degree 2 is precisely the symmetric packing of the $d \times d$ matrix. SATA generalizes this to arbitrary degree $p$, while HLA provides exact causal masking via cross-summaries (which SATA handles via parallel scan).

## Applicability

- **Drop-in replacement for softmax attention** in decoder-only language models, achieving constant per-token cost for unbounded generation.
- **Direct generalization of Higher-Order Linear Attention (HLA, trick 222)**: At $p = 2$, SATA's feature map $\Phi_2(\mathbf{k}) \in \mathbb{R}^{d(d+1)/2}$ is exactly the symmetric packing of HLA's $\mathbf{k}\mathbf{k}^\top$. SATA extends this to $p = 3, 4, \ldots$ with the tightest possible (minimal basis) state size.
- **Enabling higher-order polynomial kernels**: For $p = 4$, SATA makes higher-order polynomial attention practical by reducing the feature dimension from $d^4$ to $\binom{d+3}{4}$, bringing state sizes into manageable range (e.g., 766K at $d = 64$).
- **Many-head architectures**: Since state size decreases inversely with $d_K$, SATA enables many smaller heads (each with cheap constant-cost attention) rather than fewer large heads.
- **Complementary to IO-aware kernels**: The BASED IO-aware kernel (trick 210) fuses the degree-2 Taylor feature map into registers. SATA's symmetry-aware packing reduces the feature dimension by ~2$\times$ at degree 2 (and much more at higher degrees), directly reducing the register/SRAM footprint of such fused kernels.

## Limitations

- **Proof of concept implementation**: The current PyTorch implementation uses high-level indexing (`x[..., M].prod(dim=-1)`) which creates temporary copies rather than computing in-place. A fused CUDA kernel that directly gathers and multiplies without materializing $m_p \times p$ index arrays would significantly improve throughput.
- **Feature map is a gather-multiply, not a matmul**: $\Phi_p(\mathbf{x})$ involves gathering elements of $\mathbf{x}$ by precomputed indices and multiplying them — this is a fancy elementwise operation, not a GEMM. It doesn't map to tensor cores. However, the downstream linear attention operations (state accumulation, output computation) ARE matmul-shaped.
- **State size still large at high degree**: At $d_K = 64, P = 4$: the state has $\binom{67}{3} = 47{,}905$ features per degree, totaling ~766K across all degrees. This is manageable but large compared to GLA's $O(d^2)$ state. The benefit is richer expressivity (degree-4 polynomial kernel).
- **No training validation**: The paper validates correctness on synthetic attention recovery but does not train end-to-end language models. Downstream task performance with SATA attention is unvalidated.
- **Taylor truncation error**: With $P = 4$ terms, reconstruction error is at float16 resolution ($\sim 10^{-3}$). Higher precision may need $P = 5$--$6$, increasing state size.
- **Sequential evaluation of Taylor terms**: The current implementation evaluates $P$ scans sequentially on one GPU stream. These are independent and could run in parallel on separate streams, but this optimization is not implemented.
- **No causal cross-summary correction**: Unlike HLA (trick 222), which introduces explicit cross-summaries $\mathbf{G}_t$ for exact causal masking at second order, SATA handles causality via standard linear attention prefix sums (one per degree). This means each degree-$p$ scan is independent and simpler than HLA's semidirect product scan.

## Implementation Notes

```python
# SATA — Symmetry-Aware Taylor Feature Map
# Core algorithm from Heinsen & Kozachkov (2026)

import torch
from math import comb, factorial

def build_index_matrix(d_K, p):
    """Build the constant index matrix M_p for degree p.

    M_p has shape (m_p, p) where m_p = C(d_K + p - 1, p).
    Row i contains sorted indices (i_1 <= i_2 <= ... <= i_p).
    """
    from itertools import combinations_with_replacement
    indices = list(combinations_with_replacement(range(d_K), p))
    return torch.tensor(indices, dtype=torch.long)  # (m_p, p)

def build_weight_matrix(M_p, p):
    """Build diagonal weight matrix C_p.

    Each weight is the multinomial coefficient: the number of
    distinct permutations of the index tuple.
    """
    from collections import Counter
    weights = []
    for row in M_p:
        counts = Counter(row.tolist())
        # Multinomial: p! / (n1! * n2! * ...)
        denom = 1
        for c in counts.values():
            denom *= factorial(c)
        weights.append(factorial(p) / denom)
    return torch.tensor(weights, dtype=torch.float32)  # (m_p,)

def phi(x, M_p):
    """Compute minimal feature map Phi_p(x).

    x: (..., d_K) — input vectors
    M_p: (m_p, p) — precomputed index matrix

    Returns: (..., m_p) — tightly packed unique monomials
    """
    # Gather: x[..., M_p] has shape (..., m_p, p)
    # Multiply along last dim: product of p elements
    return x[..., M_p].prod(dim=-1)  # (..., m_p)

def sata_streaming_step(q_t, k_t, v_t, Z_states, S_states,
                        M_list, C_list, a_list, P):
    """One streaming step of SATA attention.

    Args:
        q_t: (d_K,) query at time t
        k_t: (d_K,) key at time t
        v_t: (d_V,) value at time t
        Z_states: list of P tensors, Z_p of shape (m_p,)
        S_states: list of P tensors, S_p of shape (m_p, d_V)
        M_list: precomputed index matrices [M_0, M_1, ..., M_{P-1}]
        C_list: precomputed weight vectors [C_0, C_1, ..., C_{P-1}]
        a_list: Taylor coefficients [a_0, a_1, ..., a_{P-1}]
        P: number of Taylor terms

    Returns:
        y_t: (d_V,) output
        updated Z_states, S_states
    """
    numerator = torch.zeros_like(v_t)
    denominator = 0.0

    for p in range(P):
        phi_k = phi(k_t, M_list[p])  # (m_p,)
        phi_q = phi(q_t, M_list[p])  # (m_p,)

        # Update accumulated states
        Z_states[p] = Z_states[p] + a_list[p] * phi_k
        S_states[p] = S_states[p] + a_list[p] * phi_k.unsqueeze(-1) * v_t.unsqueeze(0)

        # Compute output contributions
        # Weighted inner product with C_p diagonal
        numerator += S_states[p].T @ (C_list[p] * phi_q)  # (d_V,)
        denominator += (C_list[p] * phi_q) @ Z_states[p]   # scalar

    y_t = numerator / (denominator + 1e-6)
    return y_t, Z_states, S_states

# Relationship to HLA (trick 222):
# At p=2, phi_2(k) extracts the d*(d+1)/2 unique entries of k*k^T
# This is exactly the upper-triangular packing of HLA's S_t^K
# SATA generalizes to p=3,4,... with minimal basis size

# GPU efficiency notes:
# 1. phi() is a gather-multiply — NOT a matmul, but O(m_p * p) elementwise
# 2. State updates are rank-1 outer products + addition — SAME structure as
#    standard linear attention, just with m_p-dimensional features
# 3. The P scans are INDEPENDENT — can run on P parallel GPU streams
# 4. For training: use chunkwise parallel scan (Blelloch) per degree p
# 5. The S_p accumulation (m_p x d_V) @ (m_p,) is a matvec — memory-bound
# 6. Fused kernel opportunity: compute phi, update state, and output in one
#    kernel without materializing the m_p-dim feature vectors in HBM
```

**Concrete state sizes (per head, $P = 4$ terms):**

| $d_K$ | $d_V$ | $m_0$ | $m_1$ | $m_2$ | $m_3$ | Total $\sum m_p$ | State $(d_V+1) \cdot \sum m_p$ |
|-------|-------|-------|-------|-------|-------|-----------------|-------------------------------|
| 8 | 8 | 1 | 8 | 36 | 120 | 165 | 1,485 |
| 16 | 16 | 1 | 16 | 136 | 816 | 969 | 16,473 |
| 32 | 32 | 1 | 32 | 528 | 5,984 | 6,545 | 215,985 |
| 64 | 64 | 1 | 64 | 2,080 | 45,760 | 47,905 | 3,113,825 |

For small head sizes ($d_K = 8$--$16$), the state is very compact (< 17K entries), fitting entirely in GPU registers/SRAM. For $d_K = 64$, the degree-3 term alone has 45K features — manageable but substantial.

**Comparison to other tricks for polynomial attention:**

| Trick | Type | Dimension at $p=2, d=64$ | Dimension at $p=4, d=64$ | Exact? |
|-------|------|-------------------------|-------------------------|--------|
| Naive tensor product | exact | $64^2 = 4096$ | $64^4 = 16.8$M | Yes |
| **SATA** (this trick) | **minimal basis** | **2080** | **766,480** | **Yes** |
| TensorSketch (trick 263) | random | $D$ (tunable) | $D$ (tunable) | No |
| PolySketchFormer (trick 262) | random | $r^2$ (tunable) | $r^2$ (tunable) | No |
| BASED (trick 210) | IO-fused | 273 ($d'=16$) | N/A (not impl.) | Yes* |

*BASED uses a projected $d' = 16$ instead of $d_K = 64$, so its state is much smaller but less expressive.

## References

- Heinsen, F. A. & Kozachkov, L. (2026). Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation. arXiv:2602.00294. [https://arxiv.org/abs/2602.00294](https://arxiv.org/abs/2602.00294)
- Code: [https://github.com/glassroom/sata_attention](https://github.com/glassroom/sata_attention)
- Schatz, M. D., Low, T. M., van de Geijn, R. A., & Kolda, T. G. (2014). Exploiting Symmetry in Tensors for High Performance: Multiplication with Symmetric Tensors. SIAM J. Sci. Comput., 36(5):C453--C479. (Symmetric tensor storage and computation foundations)
- Zhang, Y., Qin, Z., & Gu, Q. (2025). Higher-order Linear Attention. arXiv:2510.27258. (HLA — uses second-moment prefix summaries, equivalent to SATA at $p = 2$)
- Arora, S., et al. (2024). Simple linear attention language models balance the recall-throughput tradeoff (BASED). ICML 2024. (IO-aware Taylor linear attention, limited to $p = 2$)
- Kacham, P., Mirrokni, V., & Zhong, P. (2024). PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels. ICLR 2024. (Random sketch approach — complementary to SATA's deterministic approach)
- Kostrikin, A., Manin, I., & Manin, Y. (1989). Linear Algebra and Geometry. (Theory of symmetric tensor minimal bases)

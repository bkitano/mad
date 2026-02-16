# 073: Log-Linear Attention

**Category**: parallelization
**Gain type**: expressivity
**Source**: Guo, Yang, Goel, Xing, Dao, and Kim (2025). Log-Linear Attention.
**Paper**: papers/log-linear-attention.pdf
**Documented**: 2026-02-11

## Description

Log-linear attention replaces the fixed-size hidden state of linear attention and SSMs with a *logarithmically growing set* of hidden states, organized via a Fenwick tree (binary indexed tree) partitioning of the sequence. This addresses a fundamental limitation: linear attention and SSMs compress all history into a constant-size state, which caps their recall ability. By maintaining $O(\log T)$ hidden states at multiple temporal resolutions, log-linear attention achieves the expressiveness closer to full softmax attention while keeping compute $O(T \log T)$ and memory $O(\log T)$ during decoding.

The key structural insight: the masking matrix $\mathbf{M}$ in the unified formulation $\mathbf{O} = (\mathbf{A} \odot \mathbf{M})\mathbf{V}$ is replaced by a *hierarchical* matrix $\mathbf{M}^{\mathcal{H}}$ — an instance of the HODLR (Hierarchically Off-Diagonal Low-Rank) class. This structured matrix admits an $O(T \log T)$ parallel training algorithm via a *chunkwise hierarchical scan*: intra-chunk attention is standard quadratic (small chunks), while inter-chunk dependencies are handled by $O(\log \frac{T}{C})$ independent scan passes, each using existing linear attention primitives.

This is a general framework: any linear attention or SSM with a chunkwise-parallel primitive can be extended to a log-linear variant by composing its attention mask with the hierarchical counterpart.

## Mathematical Form

**Core Operation (Decoding):**

$$
\mathbf{o}_t = \sum_{\ell=0}^{L-1} \lambda_t^{(\ell)} \mathbf{q}_t^\top \mathbf{S}_t^{(\ell)}
$$

where $\mathbf{S}_t^{(\ell)} \in \mathbb{R}^{d \times d}$ is the hidden state summarizing all information in bucket $\mathcal{B}_t^{(\ell)}$ at level $\ell$, and $\lambda_t^{(\ell)} \geq 0$ are learned data-dependent weights controlling attention to each temporal scale.

**Key Definitions:**

- $T$ — sequence length
- $L = \lceil \log_2(T+1) \rceil + 1$ — number of levels
- $\mathcal{B}_t^{(\ell)}$ — the bucket of tokens assigned to level $\ell$ for query at position $t$ (Fenwick tree partition)
- $\lambda_t^{(\ell)}$ — per-head scalar weights parameterized as functions of $\mathbf{x}_t$ via a linear projection
- $\mathbf{S}_t^{(\ell)}$ — recurrent memory at level $\ell$, a $d \times d$ matrix

**Fenwick Tree Partitioning:**

The prefix $[0, t)$ is partitioned into $L$ disjoint buckets using the least significant set bit function $\text{lssb}(t) = \max\{\ell \in \mathbb{N} \mid 2^\ell \text{ divides } t\}$:

$$
b_t^{(i)} = \begin{cases} t & \text{if } i = 0 \\ b_t^{(i-1)} - 2^{\text{lssb}(b_t^{(i-1)})} & \text{otherwise} \end{cases}
$$

Each bucket $\mathcal{B}_t^{(\ell)}$ has at most $2^{\ell-1}$ tokens (power-of-two sized), with the finest resolution for recent tokens.

**Parallel Form (Training):**

$$
\mathbf{O} = \left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}^{\mathcal{H}}\right)\mathbf{V}
$$

where $\mathbf{M}^{\mathcal{H}}$ is a hierarchical lower-triangular matrix:

$$
\mathbf{M}^{\mathcal{H}}_{ts} = \begin{cases} \lambda_t^{\ell(t,s)} & \text{if } s \leq t \\ 0 & \text{otherwise} \end{cases}
$$

**Hierarchical Decomposition:**

$$
\mathbf{M}^{\mathcal{H}} = \mathbf{D} + \sum_{\ell=1}^{L-1} \mathbf{M}^{(\ell)}
$$

where $\mathbf{D}$ is block-diagonal (intra-chunk) and each $\mathbf{M}^{(\ell)}$ captures inter-chunk dependencies at level $\ell$.

**Log-Linear Mamba-2:**

$$
\mathbf{O} = \left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}^S \odot \mathbf{M}^{\mathcal{H}}\right)\mathbf{V}
$$

**Log-Linear Gated DeltaNet:**

$$
\mathbf{O} = \left(\left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{L}\right)\left(\mathbf{I} + \mathbf{K}\mathbf{K}^\top \odot (\mathbf{L} - \mathbf{I})\right)^{-1} \odot \mathbf{M}^S \odot \mathbf{M}^{\mathcal{H}}\right)\mathbf{V}
$$

## Complexity

| Operation | Softmax Attention | Linear Attention | Log-Linear Attention |
|-----------|------------------|------------------|---------------------|
| Training compute | $O(T^2)$ | $O(T)$ | $O(T \log T)$ |
| Decoding time/step | $O(T)$ | $O(1)$ | $O(\log T)$ |
| Decoding memory | $O(T)$ | $O(1)$ | $O(\log T)$ |

**Memory during training:** $O(T)$ (same as linear attention, since chunk size $C$ is constant)

**Parameters added:** $< 3\%$ overhead (just the $\lambda$ projection layer per head)

The chunkwise parallel scan performs $O(\log \frac{T}{C})$ independent scans, each costing $O(T)$, for total $O(T \log T)$.

## Applicability

- **Linear attention models** (RetNet, GLA, GateLoop): Direct extension via mask composition
- **State space models** (Mamba-2): Composing SSS mask with hierarchical mask
- **Delta-rule models** (Gated DeltaNet): Full support via Woodbury-resolvent identity integration
- **Any model with chunkwise-parallel training**: The framework only requires modifying the masking matrix $\mathbf{M}$

Validated at scale: 800M parameter models trained on 50B tokens, outperforming linear counterparts on WikiText perplexity, MQAR recall, Needle-In-A-Haystack, and LongBench.

## Limitations

- Engineering complexity is higher than vanilla linear attention: intra-chunk computations require bespoke kernels
- Throughput advantage over FlashAttention-2 only emerges at sequence lengths $> 8$K
- The Fenwick tree partitioning introduces an inductive bias: recent tokens get fine-grained attention, distant tokens are compressed — may not suit all tasks
- $O(\log T)$ decoding memory/time is still worse than the $O(1)$ of pure linear attention
- Backward pass is more complex due to additional $\lambda$ gradient terms

## Implementation Notes

```python
# Pseudocode for Fenwick tree bucket assignment
def lssb(t):
    """Least significant set bit"""
    return (t & -t).bit_length() - 1 if t > 0 else 0

def fenwick_partition(t, L):
    """Partition prefix [0, t) into Fenwick tree buckets"""
    buckets = {0: [t]}  # Level 0: sentinel bucket
    b = t
    for i in range(1, L):
        if b == 0:
            break
        level = lssb(b)
        start = b - 2**level
        buckets[level] = list(range(start, b))
        b = start
    return buckets

# Key: composing masks for log-linear Mamba-2
# M = M^S ⊙ M^H (elementwise product of SSS mask and hierarchical mask)
# Training: chunkwise scan with O(log(T/C)) inter-chunk passes
# Decoding: maintain O(log T) hidden states, merge on lssb boundaries
```

Code: https://github.com/HanGuo97/log-linear-attention

## References

- Guo, Yang, Goel, Xing, Dao, and Kim (2025). Log-Linear Attention. arXiv:2506.04761.
- Dao and Gu (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State-Space Duality (Mamba-2).
- Yang et al. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Gated DeltaNet).
- Fenwick (1994). A New Data Structure for Cumulative Frequency Tables.

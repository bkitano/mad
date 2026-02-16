# 083: Online Softmax (Streaming Softmax)

**Category**: stability
**Gain type**: efficiency
**Source**: Milakov & Gimelshein "Online normalizer calculation for softmax" (2018); core enabler for FlashAttention
**Paper**: [papers/flash-attention-io-aware-tiling.pdf]
**Documented**: 2025-02-11

## Description

Online softmax (also called streaming softmax) computes the numerically-stable softmax function incrementally over blocks of data, maintaining running statistics that allow exact results without requiring a full pass over the input. This is the mathematical trick that enables tiled attention computation (FlashAttention) — without it, softmax would couple all elements of a row, forcing the entire $N \times N$ attention matrix to be materialized.

The standard numerically-stable softmax requires **three passes** over the data: (1) find the max, (2) compute $\exp(x_i - \max)$ and their sum, (3) normalize. Online softmax fuses these into a **single pass** by maintaining a running maximum $m$ and a running unnormalized sum $\ell$, rescaling previous partial results when a new maximum is encountered.

This is an instance of **algebraic aggregation** — the softmax can be decomposed over concatenated blocks because the normalization statistics $(m, \ell)$ form a composable summary that supports incremental updates.

## Mathematical Form

**Standard 3-pass softmax** of $x \in \mathbb{R}^B$:

$$
m(x) = \max_i x_i, \quad f(x) = \begin{bmatrix} e^{x_1 - m(x)} & \cdots & e^{x_B - m(x)} \end{bmatrix}, \quad \ell(x) = \sum_i f(x)_i, \quad \text{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

**Online decomposition** over concatenated blocks $x = [x^{(1)}, x^{(2)}] \in \mathbb{R}^{2B}$:

$$
m(x) = \max(m(x^{(1)}), m(x^{(2)}))
$$

$$
f(x) = \begin{bmatrix} e^{m(x^{(1)}) - m(x)} f(x^{(1)}) & e^{m(x^{(2)}) - m(x)} f(x^{(2)}) \end{bmatrix}
$$

$$
\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})
$$

**Incremental update rule** (processing block $j$):

Given running statistics $(m_{\text{old}}, \ell_{\text{old}})$ and new block scores $s_j$:

$$
m_{\text{new}} = \max(m_{\text{old}}, \max(s_j))
$$

$$
\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum_i e^{s_{j,i} - m_{\text{new}}}
$$

**Rescaling accumulated output** (for attention with values $V_j$):

$$
O_{\text{new}} = \frac{1}{\ell_{\text{new}}} \left( e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} \cdot O_{\text{old}} + e^{m(s_j) - m_{\text{new}}} \cdot \tilde{P}_j V_j \right)
$$

where $\tilde{P}_j = \exp(s_j - m(s_j))$ is the unnormalized softmax of the current block.

**Key Definitions:**

- $m$ — running maximum across all blocks processed so far
- $\ell$ — running sum of exponentials (unnormalized normalizer)
- $s_j = Q_i K_j^\top$ — attention scores for current block pair
- $\tilde{P}_j$ — local unnormalized attention weights

**Algebraic structure:** The triple $(m, \ell, O \cdot \ell)$ forms a **monoid** under the binary operation that combines two partial softmax results. This associativity enables both sequential streaming and parallel (tree-reduction) computation.

## Complexity

| Operation | Standard 3-pass | Online (streaming) |
|-----------|----------------|-------------------|
| Passes over data | $3$ | $1$ |
| Memory for scores | $O(N)$ | $O(B)$ (one block) |
| Extra storage | $O(1)$ | $O(1)$ per row ($m, \ell$) |

**Memory:** $O(1)$ additional state per row, enabling softmax over arbitrarily long sequences.

## Applicability

- **FlashAttention**: Core enabler — allows attention to be computed tile-by-tile without materializing the $N \times N$ score matrix
- **Ring Attention**: Enables distributed attention across multiple devices, each processing blocks of K, V
- **Streaming/online inference**: Process tokens one block at a time during autoregressive generation
- **Any operation requiring normalization over long sequences**: Batch norm, layer norm variants, softmax-based routing in MoE

## Limitations

- Requires careful numerical implementation (rescaling by $e^{m_{\text{old}} - m_{\text{new}}}$ can underflow if maxima differ greatly)
- Adds overhead of tracking and updating $(m, \ell)$ statistics per row — negligible for attention but nonzero
- The online approach adds slight complexity to the backward pass (need to store $m, \ell$ for recomputation)
- Only applicable when the aggregation has algebraic structure (softmax does, but arbitrary nonlinear normalizations may not)

## Implementation Notes

```python
# Online softmax: single-pass computation
def online_softmax(x_blocks):
    """Compute softmax over concatenated blocks incrementally."""
    m = float('-inf')  # running max
    ell = 0.0          # running sum of exp

    for block in x_blocks:
        m_block = block.max()
        m_new = max(m, m_block)

        # Rescale old sum and add new contributions
        ell = ell * exp(m - m_new) + sum(exp(block - m_new))
        m = m_new

    # Final softmax values (second pass only if needed)
    return [exp(block - m) / ell for block in x_blocks]

# Online softmax + attention (FlashAttention core)
def online_softmax_attention(Q_block, K_blocks, V_blocks):
    """Compute attention output for one Q block against all K,V blocks."""
    m = float('-inf')
    ell = 0.0
    O = zeros_like(Q_block @ V_blocks[0].T)  # wrong shape, illustrative

    for Kj, Vj in zip(K_blocks, V_blocks):
        scores = Q_block @ Kj.T
        m_new = max(m, scores.max(axis=1))
        P_tilde = exp(scores - m_new[:, None])

        # Rescale and accumulate
        scale = exp(m - m_new)
        ell_new = scale * ell + P_tilde.sum(axis=1)
        O = (scale[:, None] * ell[:, None] * O + P_tilde @ Vj) / ell_new[:, None]

        m = m_new
        ell = ell_new

    return O
```

## References

- Milakov, M. & Gimelshein, N. "Online normalizer calculation for softmax" (2018). arXiv:1805.02867
- Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022). arXiv:2205.14135
- Rabe, M.N. & Staats, C. "Self-attention Does Not Need O(n²) Memory" (2021). arXiv:2112.05682
- Ye, Z. "From Online Softmax to FlashAttention" (UW CSE 599M lecture notes, 2023)

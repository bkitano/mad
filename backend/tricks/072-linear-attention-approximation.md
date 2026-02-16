# 072: Linear Attention Approximation

**Category**: approximation
**Gain type**: efficiency
**Source**: Katharopoulos et al. (2020)
**Documented**: 2026-02-10

## Description

Replace the softmax attention kernel with a feature map $\phi(q)\phi(k)^\top$, which allows rewriting attention as a linear recurrence. This reduces the quadratic $O(T^2 d)$ cost of attention to $O(Td^2)$ by avoiding explicit computation of the $T \times T$ attention matrix.

## Mathematical Form

**Core Operation:**

**Standard Attention:**
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V \quad \text{— } O(T^2 d)
$$

**Linear Attention:**
$$
\text{Attn}(Q, K, V) = \phi(Q) \left(\phi(K)^\top V\right) \quad \text{— } O(Td^2)
$$

**Key Definitions:**

- $Q, K, V \in \mathbb{R}^{T \times d}$ — query, key, value matrices
- $\phi: \mathbb{R}^d \to \mathbb{R}^{d'}$ — feature map
- $T$ — sequence length
- $d$ — head dimension

**The "Right-Multiply-First" Trick:**

The key insight is to compute $(\phi(K)^\top V) \in \mathbb{R}^{d \times d}$ first as a $d \times d$ matrix, then multiply by $\phi(Q)$:

$$
\underbrace{\phi(Q)}_{T \times d} \cdot \underbrace{(\phi(K)^\top V)}_{d \times d} \quad \text{vs} \quad \underbrace{(\phi(Q) \phi(K)^\top)}_{T \times T} \cdot \underbrace{V}_{T \times d}
$$

**Common Feature Maps $\phi$:**

- $\phi(x) = \text{elu}(x) + 1$ (element-wise, simple)
- Random Fourier features: $\phi(x) = \exp(Wx) / \sqrt{m}$
- Learned feature maps
- Positive random features (for non-negative attention)

**Recurrent Form:**

Linear attention can be computed as a recurrence:
$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \quad y_t = \phi(q_t)^\top S_t
$$

where $S_t \in \mathbb{R}^{d \times d}$ is the cumulative key-value state.

## Complexity

| Operation | Standard Attention | Linear Attention |
|-----------|-------------------|------------------|
| Time | $O(T^2 d)$ | $O(T d^2)$ |
| Memory | $O(T^2)$ attention matrix | $O(d^2)$ state matrix |
| Crossover | — | Better when $T > d$ |

**Memory:** $O(d^2)$ for state matrix vs $O(T^2)$ for attention matrix

## Applicability

Any attention-based architecture. Forms the basis for DeltaNet, RetNet, Mamba-2, and most modern linear recurrence architectures.

## Limitations

- Approximation quality depends on the feature map
- No single feature map matches softmax quality on all tasks
- For $d \gg T$ (rare in practice), this is actually slower
- Cannot exactly represent the softmax kernel (only approximates it)

## Implementation Notes

```python
# Linear attention (causal, recurrent form)
def linear_attention(Q, K, V, phi=lambda x: F.elu(x) + 1):
    T, d = Q.shape
    S = torch.zeros(d, d)  # State matrix
    outputs = []

    for t in range(T):
        S = S + torch.outer(phi(K[t]), V[t])  # Update state
        y_t = phi(Q[t]) @ S                    # Query state
        outputs.append(y_t)

    return torch.stack(outputs)
```

## References

- Katharopoulos et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.
- Schlag et al. (2021). Linear Transformers Are Secretly Fast Weight Programmers.

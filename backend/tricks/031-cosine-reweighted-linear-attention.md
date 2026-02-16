# 031: Cosine-Reweighted Linear Attention (cosFormer)

**Category**: approximation
**Gain type**: efficiency
**Source**: Qin et al. (2022), "cosFormer: Rethinking Softmax in Attention" (ICLR 2022)
**Paper**: [papers/cosformer-cosine-reweighting.pdf]
**Documented**: 2026-02-15

## Description

cosFormer replaces softmax attention with a two-component linear substitute: (1) a ReLU-based feature map to enforce non-negativity of attention weights, and (2) a cosine-based distance re-weighting scheme that introduces locality bias and stabilizes training. The crucial insight is that softmax's empirical success stems from two decomposable properties — non-negative attention weights and concentrated weight distributions — neither of which requires the non-decomposable $\exp$ function. By using $\text{ReLU}$ for non-negativity and $\cos(\frac{\pi}{2} \cdot \frac{i-j}{M})$ for locality-biased re-weighting, the attention can be exactly decomposed (via Ptolemy's theorem) into two independent linear attention computations, achieving $O(Nd^2)$ time while matching or exceeding softmax quality.

## Mathematical Form

**Core Operation:**

**Step 1: ReLU for non-negativity.** Replace softmax's feature map with ReLU:

$$
\phi_{\text{linear}}(x) = \text{ReLU}(x)
$$

yielding non-negative queries and keys $Q' = \text{ReLU}(Q)$, $K' = \text{ReLU}(K)$.

**Step 2: Cosine re-weighting.** Introduce position-dependent re-weighting:

$$
s(Q'_i, K'_j) = Q'^{\top}_i K'_j \cos\left(\frac{\pi}{2} \times \frac{i - j}{M}\right)
$$

where $i, j$ are position indices and $M \geq N$ is a scaling factor.

**Step 3: Ptolemy decomposition.** By the product-to-sum identity $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$:

$$
Q'^{\top}_i K'_j \cos\left(\frac{\pi}{2} \cdot \frac{i-j}{M}\right) = \left(Q'_i \cos\frac{\pi i}{2M}\right)^\top \left(K'_j \cos\frac{\pi j}{2M}\right) + \left(Q'_i \sin\frac{\pi i}{2M}\right)^\top \left(K'_j \sin\frac{\pi j}{2M}\right)
$$

**Key Definitions:**

- $Q, K, V \in \mathbb{R}^{N \times d}$ — query, key, value matrices
- $N$ — sequence length
- $d$ — head dimension
- $M \geq N$ — re-weighting scale parameter
- $Q'_i = \text{ReLU}(Q_i) \in \mathbb{R}^d$ — non-negative query at position $i$
- $K'_j = \text{ReLU}(K_j) \in \mathbb{R}^d$ — non-negative key at position $j$

**Decomposed Queries and Keys:**

Define position-modulated variants:

$$
Q^{\cos}_i = Q'_i \cos\left(\frac{\pi i}{2M}\right), \quad Q^{\sin}_i = Q'_i \sin\left(\frac{\pi i}{2M}\right)
$$

$$
K^{\cos}_j = K'_j \cos\left(\frac{\pi j}{2M}\right), \quad K^{\sin}_j = K'_j \sin\left(\frac{\pi j}{2M}\right)
$$

**Full Output (with normalization):**

$$
\mathcal{O}_i = \frac{\sum_{j=1}^{N} Q^{\cos}_i \left((K^{\cos}_j)^\top V_j\right) + \sum_{j=1}^{N} Q^{\sin}_i \left((K^{\sin}_j)^\top V_j\right)}{\sum_{j=1}^{N} Q^{\cos}_i (K^{\cos}_j)^\top + \sum_{j=1}^{N} Q^{\sin}_i (K^{\sin}_j)^\top}
$$

**Linear-time form (right-multiply-first):**

$$
\mathcal{O} = S(Q, K)V = (Q^{\cos} K^{\cos} + Q^{\sin} K^{\sin})V = Q^{\cos}(K^{\cos\top} V) + Q^{\sin}(K^{\sin\top} V)
$$

Each term is a standard linear attention computation: the $d \times d$ matrices $K^{\cos\top} V$ and $K^{\sin\top} V$ are computed first, then multiplied by $Q^{\cos}$ and $Q^{\sin}$ respectively.

**Why Cosine Re-weighting Works:**

The $\cos(\frac{\pi}{2} \cdot \frac{i-j}{M})$ term:
1. Equals 1 when $i = j$ (self-attention preserved at full strength)
2. Decays smoothly toward 0 as $|i - j|$ grows (locality bias)
3. Is decomposable into a sum of products (enabling linear complexity)
4. Concentrates the attention distribution similarly to softmax normalization

## Complexity

| Operation | Softmax Attention | cosFormer |
|-----------|------------------|-----------|
| Time | $O(N^2 d)$ | $O(N d^2)$ |
| Space | $O(N^2 + Nd)$ | $O(Nd + d^2)$ |

**Memory:** $O(d^2)$ for the two $d \times d$ accumulated matrices vs $O(N^2)$ for the full attention matrix.

**Constant factor:** cosFormer requires $2\times$ the computation of a single linear attention (two terms from the Ptolemy decomposition), but this is a small constant factor.

**Note:** Since $d \ll N$ in typical NLP settings ($d = 64$ per head, $N = 512$–$4096$+), the complexity is effectively $O(N)$.

## Applicability

- **Causal (autoregressive) language modeling:** Achieves better perplexity than vanilla Transformer on WikiText-103 (23.1 vs 26.2 test PPL)
- **Bidirectional models:** Matches or exceeds softmax attention on GLUE, IMDB, AMAZON benchmarks
- **Long-range tasks:** Achieves best overall score on Long-Range Arena benchmark (55.23 average), outperforming all other efficient Transformer variants
- **Drop-in replacement** for attention in both encoder and decoder architectures
- Acts as a form of **relative positional encoding** — the cosine re-weighting introduces relative position information without explicit positional embeddings

## Limitations

- The locality bias from cosine re-weighting may hurt tasks requiring very long-range dependencies (e.g., Pathfinder task on LRA shows slight degradation vs. softmax)
- The scale parameter $M$ introduces a hyperparameter (though $M \geq N$ is typically sufficient)
- Like all linear attention methods, the $d \times d$ state matrix limits effective capacity when $d$ is large relative to the sequence
- The $2\times$ constant factor from the Ptolemy decomposition adds overhead compared to single-feature-map linear attention
- Non-negativity via ReLU zeros out negative features, potentially discarding useful signal in some domains

## Implementation Notes

```python
import torch
import torch.nn.functional as F
import math

def cosformer_attention(Q, K, V, causal=False):
    """cosFormer: ReLU + cosine re-weighting linear attention.

    Args:
        Q, K: (N, d) query and key matrices
        V: (N, d) value matrix
        causal: whether to use causal (autoregressive) masking
    """
    N, d = Q.shape
    M = N  # re-weighting scale

    # Step 1: ReLU for non-negativity
    Q_prime = F.relu(Q)  # (N, d)
    K_prime = F.relu(K)  # (N, d)

    # Step 2: Position-dependent cosine/sine modulation
    positions = torch.arange(N, device=Q.device, dtype=Q.dtype)
    cos_pos = torch.cos(math.pi / (2 * M) * positions).unsqueeze(-1)  # (N, 1)
    sin_pos = torch.sin(math.pi / (2 * M) * positions).unsqueeze(-1)  # (N, 1)

    Q_cos = Q_prime * cos_pos  # (N, d)
    Q_sin = Q_prime * sin_pos  # (N, d)
    K_cos = K_prime * cos_pos  # (N, d)
    K_sin = K_prime * sin_pos  # (N, d)

    if not causal:
        # Bidirectional: right-multiply-first trick
        # (K_cos^T V) and (K_sin^T V) are each (d, d)
        KV_cos = K_cos.T @ V    # (d, d)
        KV_sin = K_sin.T @ V    # (d, d)
        num = Q_cos @ KV_cos + Q_sin @ KV_sin  # (N, d)

        # Normalization
        K_cos_sum = K_cos.sum(dim=0)  # (d,)
        K_sin_sum = K_sin.sum(dim=0)  # (d,)
        denom = Q_cos @ K_cos_sum + Q_sin @ K_sin_sum  # (N,)
    else:
        # Causal: use cumulative sum (prefix sum)
        # S_t^cos = sum_{j<=t} k_j^cos v_j^T, similarly for sin
        S_cos = torch.zeros(d, d, device=Q.device)
        S_sin = torch.zeros(d, d, device=Q.device)
        z_cos = torch.zeros(d, device=Q.device)
        z_sin = torch.zeros(d, device=Q.device)
        outputs = []
        for t in range(N):
            S_cos += torch.outer(K_cos[t], V[t])
            S_sin += torch.outer(K_sin[t], V[t])
            z_cos += K_cos[t]
            z_sin += K_sin[t]
            num_t = Q_cos[t] @ S_cos + Q_sin[t] @ S_sin
            den_t = Q_cos[t] @ z_cos + Q_sin[t] @ z_sin
            outputs.append(num_t / (den_t + 1e-6))
        return torch.stack(outputs)

    return num / (denom.unsqueeze(-1) + 1e-6)
```

## References

- Qin, Z., Sun, W., Deng, H., Li, D., Wei, Y., Lv, B., Yan, J., Kong, L., & Zhong, Y. (2022). cosFormer: Rethinking Softmax in Attention. ICLR 2022.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Choromanski, K. et al. (2021). Rethinking Attention with Performers. ICLR 2021.

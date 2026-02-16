# 045: FAVOR+ (Positive Orthogonal Random Features)

**Category**: approximation
**Gain type**: efficiency
**Source**: Choromanski et al. (2021), "Rethinking Attention with Performers" (ICLR 2021)
**Paper**: [papers/favor-plus-positive-random-features.pdf]
**Documented**: 2026-02-15

## Description

FAVOR+ (Fast Attention Via positive Orthogonal Random features) is a mechanism for approximating the softmax attention kernel using random feature maps, enabling linear-time attention. The key insight is twofold: (1) use **positive** random features instead of trigonometric (sin/cos) ones to avoid numerical instability when kernel values are small, and (2) make the random projection vectors **orthogonal** via Gram-Schmidt to reduce estimator variance. Together, these yield an unbiased, low-variance, numerically stable approximation of the full softmax attention matrix that can be computed in $O(Lrd)$ time instead of $O(L^2d)$.

## Mathematical Form

**Core Operation:**

The softmax kernel $\text{SM}(\mathbf{x}, \mathbf{y}) \stackrel{\text{def}}{=} \exp(\mathbf{x}^\top \mathbf{y})$ is decomposed via a random feature map $\phi: \mathbb{R}^d \to \mathbb{R}^r_+$ such that:

$$
\text{K}(\mathbf{x}, \mathbf{y}) = \mathbb{E}[\phi(\mathbf{x})^\top \phi(\mathbf{y})]
$$

The approximate attention is then:

$$
\widehat{\text{Att}}_{\leftrightarrow}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \hat{\mathbf{D}}^{-1}(\mathbf{Q}'((\mathbf{K}')^\top \mathbf{V})), \quad \hat{\mathbf{D}} = \text{diag}(\mathbf{Q}'((\mathbf{K}')^\top \mathbf{1}_L))
$$

where $\mathbf{Q}', \mathbf{K}' \in \mathbb{R}^{L \times r}$ have rows $\phi(\mathbf{q}_i^\top)^\top$ and $\phi(\mathbf{k}_j^\top)^\top$ respectively.

**Key Definitions:**

- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$ — query, key, value matrices
- $L$ — sequence length
- $d$ — head dimension
- $r$ — number of random features ($r \ll L$, typically $r \leq d$)
- $\omega_1, \ldots, \omega_m \sim \mathcal{D}$ — random projection vectors

**General Random Feature Map:**

$$
\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{m}}(f_1(\omega_1^\top \mathbf{x}), \ldots, f_1(\omega_m^\top \mathbf{x}), \ldots, f_l(\omega_1^\top \mathbf{x}), \ldots, f_l(\omega_m^\top \mathbf{x}))
$$

for functions $f_1, \ldots, f_l : \mathbb{R} \to \mathbb{R}$, scalar $h(\mathbf{x})$, and random vectors $\omega_i$.

**Positive Random Features (PRFs) for Softmax (Lemma 1):**

$$
\text{SM}(\mathbf{x}, \mathbf{y}) = \Lambda \mathbb{E}_{\omega \sim \mathcal{N}(0, \mathbf{I}_d)} \left[\cosh(\omega^\top \mathbf{z})\right], \quad \mathbf{z} = \mathbf{x} + \mathbf{y}
$$

where $\Lambda = \exp\left(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2}\right)$.

This gives the **positive** feature map with $h(\mathbf{x}) = \exp(-\frac{\|\mathbf{x}\|^2}{2})$, $l = 1$, $f_1 = \exp$:

$$
\phi^+(\mathbf{x}) = \frac{\exp(-\|\mathbf{x}\|^2/2)}{\sqrt{m}}\left(\exp(\omega_1^\top \mathbf{x}), \ldots, \exp(\omega_m^\top \mathbf{x})\right)
$$

Or equivalently using the hyperbolic variant with $f_1(u) = \exp(u)$, $f_2(u) = \exp(-u)$:

$$
\phi^{\text{hyp}+}(\mathbf{x}) = \frac{\exp(-\|\mathbf{x}\|^2/2)}{\sqrt{2m}}\left(\exp(\omega_1^\top \mathbf{x}), \ldots, \exp(\omega_m^\top \mathbf{x}), \exp(-\omega_1^\top \mathbf{x}), \ldots, \exp(-\omega_m^\top \mathbf{x})\right)
$$

**Why Positive Features Matter:**

The standard trigonometric estimator $\widehat{\text{SM}}_m^{\text{trig}}$ uses $f_1 = \sin, f_2 = \cos$, producing features that can be negative. When kernel values $\text{SM}(\mathbf{x}, \mathbf{y}) \to 0$ (low-relevance token pairs), the trigonometric MSE diverges:

$$
\text{MSE}(\widehat{\text{SM}}_m^{\text{trig}}(\mathbf{x}, \mathbf{y})) = \frac{1}{2m}\exp(\|\mathbf{x}+\mathbf{y}\|^2)\text{SM}^{-2}(\mathbf{x},\mathbf{y})(1 - \exp(-\|\mathbf{x}-\mathbf{y}\|^2))^2
$$

while the positive estimator's MSE vanishes:

$$
\text{MSE}(\widehat{\text{SM}}_m^{+}(\mathbf{x}, \mathbf{y})) = \frac{1}{m}\exp(\|\mathbf{x}+\mathbf{y}\|^2)\text{SM}^{2}(\mathbf{x},\mathbf{y})(1 - \exp(-\|\mathbf{x}-\mathbf{y}\|^2))
$$

**Orthogonal Random Features (ORFs):**

Instead of sampling $\omega_1, \ldots, \omega_m$ independently from $\mathcal{N}(0, \mathbf{I}_d)$, enforce exact orthogonality via Gram-Schmidt while preserving unbiasedness (valid for any isotropic distribution $\mathcal{D}$). This provably reduces MSE for any $d > 0$:

$$
\text{MSE}(\widehat{\text{SM}}_m^{\text{ort}+}(\mathbf{x}, \mathbf{y})) \leq \text{MSE}(\widehat{\text{SM}}_m^{+}(\mathbf{x}, \mathbf{y})) - \frac{2(m-1)}{m(d+2)}\left(\text{SM}(\mathbf{x},\mathbf{y}) - \exp\left(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2}\right)\right)^2
$$

## Complexity

| Operation | Standard Attention | FAVOR+ |
|-----------|-------------------|--------|
| Time | $O(L^2 d)$ | $O(Lrd)$ |
| Space | $O(L^2 + Ld)$ | $O(Lr + Ld + rd)$ |

**Memory:** $O(Lr + rd)$ vs $O(L^2)$ — dominated by the $r \times d$ intermediate matrix instead of the $L \times L$ attention matrix.

**Key crossover:** When $r \ll L$ (typically $r = O(d \log d)$ suffices for $\epsilon$-approximation), the method is strictly faster. The number of random features $r$ depends on dimensionality $d$ and desired precision $\epsilon$, but **not** on sequence length $L$.

## Applicability

- **Drop-in replacement** for softmax attention in any Transformer architecture (encoder, decoder, encoder-decoder)
- Backward-compatible with pretrained Transformers via small finetuning
- Works for both bidirectional and causal (unidirectional) attention via prefix-sum formulation
- Applicable beyond Transformers: graph attention networks, hierarchical attention, reinforcement learning
- Particularly beneficial for **long sequences** ($L > 4096$) where $O(L^2)$ attention is prohibitive (e.g., protein sequences, genomics, high-resolution images)

## Limitations

- Approximation quality degrades for very sharp attention distributions (large $\|\mathbf{q}\|, \|\mathbf{k}\|$ norms)
- Requires periodic **re-drawing** of random features during training for best results
- The Lipschitz constant of the Transformer can amplify small per-layer approximation errors across depth
- For small $L$ (short sequences), the overhead of computing the feature map may negate savings
- Positive features with SMREG (regularized softmax kernel) needed for best convergence on large datasets
- ORF mechanism requires $m \leq d$; for $r > d$, must use multiple orthogonal blocks

## Implementation Notes

```python
import torch
import math

def favor_plus_attention(Q, K, V, m=None, redraw=True):
    """FAVOR+ linear attention with positive orthogonal random features.

    Args:
        Q, K: (L, d) query and key matrices
        V: (L, d) value matrix
        m: number of random features (default: d)
        redraw: whether to redraw random projections
    """
    L, d = Q.shape
    if m is None:
        m = d

    # Step 1: Generate orthogonal random features
    # Sample Gaussian, then orthogonalize via QR
    W = torch.randn(m, d)
    Q_orth, _ = torch.linalg.qr(W.T)  # (d, m)
    W_orth = Q_orth.T  # (m, d) — orthonormal rows
    # Restore correct norms (chi-distributed)
    norms = torch.randn(m, d).norm(dim=1)
    W_orth = W_orth * norms.unsqueeze(1)

    # Step 2: Positive random feature map
    # phi(x) = exp(-||x||^2/2) / sqrt(m) * exp(W @ x)
    def phi_positive(X):
        # X: (L, d), W_orth: (m, d)
        projection = X @ W_orth.T  # (L, m)
        norm_sq = (X ** 2).sum(dim=-1, keepdim=True)  # (L, 1)
        return torch.exp(projection - norm_sq / 2) / math.sqrt(m)

    # Step 3: Compute linear attention
    Q_prime = phi_positive(Q)  # (L, m)
    K_prime = phi_positive(K)  # (L, m)

    # Right-multiply-first: (K'^T V) is (m, d), then Q' @ (K'^T V) is (L, d)
    KV = K_prime.T @ V          # (m, d)
    K_sum = K_prime.sum(dim=0)  # (m,) — for normalization

    numerator = Q_prime @ KV         # (L, d)
    denominator = Q_prime @ K_sum    # (L,)

    return numerator / denominator.unsqueeze(-1)
```

## References

- Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., & Weller, A. (2021). Rethinking Attention with Performers. ICLR 2021.
- Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Belanger, D., Colwell, L., & Weller, A. (2023). FAVOR#: Sharp Attention Kernel Approximations via New Classes of Positive Random Features. ICML 2023.
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007.

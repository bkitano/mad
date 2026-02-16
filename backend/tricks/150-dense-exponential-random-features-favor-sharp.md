# 150: Dense-Exponential Random Features (FAVOR#)

**Category**: approximation
**Gain type**: efficiency
**Source**: Likhosherstov, Choromanski et al. (2023), "FAVOR#: Sharp Attention Kernel Approximations via New Classes of Positive Random Features"
**Paper**: [papers/favor-sharp-sderf.pdf]
**Documented**: 2026-02-15

## Description

Dense-Exponential Random Features (DERFs) are an extension of the Generalized Exponential RF (GERF) framework where scalar parameters $A, B, C$ are replaced by **dense matrices** $\mathbf{A}, \mathbf{B}^{(k)}, \mathbf{C}^{(k)} \in \mathbb{R}^{d \times d}$, enabling the feature map to adapt to the full covariance structure of the data rather than just scalar norms. Three practical instantiations are proposed: **Asymmetric DERFs (ADERFs)**, **Symmetric DERFs (SDERFs)**, and **Simplified ADERFs (SADERFs)**. The SDERF variant achieves up to $e^{10}\times$ further variance reduction over the previous best method (GERFs/OPRFs), while the resulting self-attention approximation method is called **FAVOR#**.

The key insight is that the shifted log-variance objective — previously treated as a heuristic (the "homogeneity heuristic" from FAVOR++) — is actually the **exact global minimum** of a certain log-variance loss, and DERFs provide a richer parameterization that allows finding a tighter minimum via matrix decompositions. Despite using $d \times d$ matrix parameters, the optimal parameters can be computed in closed form via eigendecomposition of data covariance matrices, keeping the overall complexity subquadratic in $L$.

## Mathematical Form

**Core Operation — DERF Feature Map:**

For $\omega \sim \mathcal{N}(\mathbf{0}_d, \mathbf{I}_d)$, $k \in \{1, 2\}$:

$$
f_{\text{DE}}^{(k)}(\omega, \mathbf{x}) = D\exp\left(\omega^\top\mathbf{Q}\tilde{\mathbf{A}}\mathbf{Q}^\top\omega + \omega^\top\mathbf{B}^{(k)}\mathbf{x} + \mathbf{x}^\top\mathbf{C}^{(k)}\mathbf{x}\right)
$$

where $\mathbf{A} \in \mathbb{S}_d$ (set of $d \times d$ real symmetric diagonal matrices), $\tilde{\mathbf{A}} = \mathbf{Q}\mathbf{A}\mathbf{Q}^\top$ for orthogonal $\mathbf{Q}$, and $\mathbf{B}^{(k)}, \mathbf{C}^{(k)} \in \mathbb{R}^{d \times d}$.

**Key Definitions:**

- $K^{(0)}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top\mathbf{y})$ — softmax kernel
- $K^{(\alpha)}(\mathbf{x}, \mathbf{y}) = \exp(\alpha\|\mathbf{x}\|^2 + \mathbf{x}^\top\mathbf{y} + \alpha\|\mathbf{y}\|^2)$ — scaled softmax kernel
- $\mathbf{M}^{(1)} = \frac{1}{L}\sum_{i=1}^{L}\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^\top$ — query second moment matrix
- $\mathbf{M}^{(2)} = \frac{1}{L}\sum_{j=1}^{L}\mathbf{y}^{(j)}(\mathbf{y}^{(j)})^\top$ — key second moment matrix
- $\boldsymbol{\mu}^{(4)} = \frac{1}{L}\sum_{i=1}^{L}\mathbf{x}^{(i)}$, $\boldsymbol{\mu}^{(5)} = \frac{1}{L}\sum_{j=1}^{L}\mathbf{y}^{(j)}$ — query/key means
- $M$ — number of random features

**Shifted Log-Variance Objective:**

The parameters are optimized to minimize the mean log-variance across all query-key pairs:

$$
\overline{\mathcal{L}}(\boldsymbol{\theta}; \mathcal{X}, \mathcal{Y}, \mathcal{T}) = L^{-2}\sum_{1 \leq i,j \leq L}\log\left(\text{Var}_\nu[f^{(1)}(\omega, \mathbf{x}^{(i)})f^{(2)}(\omega, \mathbf{y}^{(j)})] + K^{(0)}(\mathbf{x}^{(i)}, \mathbf{y}^{(j)})^2\right)
$$

**Validity Conditions (Theorem 4.1):** DERFs are valid RFs for $K^{(0)}$ when:

$$
8\mathbf{A} \prec \mathbf{I}_d, \quad (\mathbf{B}^{(1)})^\top(\mathbf{I}_d - 4\mathbf{A})^{-1}\mathbf{B}^{(2)} = \mathbf{I}_d, \quad \mathbf{C}^{(k)} = -\frac{1}{2}(\mathbf{B}^{(k)})^\top(\mathbf{I}_d - 4\mathbf{A})^{-1}\mathbf{B}^{(k)}
$$

$$
D = \det(\mathbf{I}_d - 4\mathbf{A})^{1/4}
$$

**DERF Variance (Theorem 4.1):**

$$
\text{Var}_{\nu_{\text{DE}}}f_{\text{DE}}^{(1)}f_{\text{DE}}^{(2)} = D^4\det(\mathbf{I}_d - 8\mathbf{A})^{-1/2}\exp\left(2\mathbf{x}^\top\left(\mathbf{C}^{(1)} + (\mathbf{B}^{(1)})^\top(\mathbf{I}_d - 8\mathbf{A})^{-1}\mathbf{B}^{(1)}\right)\mathbf{x}\right.
$$

$$
\left.+ 2\mathbf{y}^\top\left(\mathbf{C}^{(2)} + (\mathbf{B}^{(2)})^\top(\mathbf{I}_d - 8\mathbf{A})^{-1}\mathbf{B}^{(2)}\right)\mathbf{y} + 4\mathbf{x}^\top(\mathbf{B}^{(1)})^\top(\mathbf{I}_d - 8\mathbf{A})^{-1}\mathbf{B}^{(2)}\mathbf{y}\right) - K^{(0)}(\mathbf{x},\mathbf{y})^2
$$

**Asymmetric DERFs (ADERFs) — Closed-Form Solution (Theorem 4.2):**

Given data second moment matrices $\mathbf{M}^{(1)}, \mathbf{M}^{(2)}$ (both nonsingular), let $\mathbf{M}^{(k)} = \mathbf{Q}^{(k)}\boldsymbol{\Lambda}^{(k)}(\mathbf{Q}^{(k)})^\top$ be eigendecompositions. Define $\phi = 2d^{-1}\sum_{l=1}^{d}\Sigma_{l,l} + 2\mu^{(3)}$ where $\Sigma_{l,l}$ are eigenvalues. Then:

$$
A = \frac{1}{16}\left(1 - 2\phi - \sqrt{(2\phi+1)^2 + 8\phi}\right)
$$

$$
\mathbf{B}^{(k)} = \sqrt{1-4A}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top(\boldsymbol{\Lambda}^{(k)})^{-1/2}(\mathbf{Q}^{(k)})^\top
$$

$$
D = (1-4A)^{d/4}
$$

**Symmetric DERFs (SDERFs) — Closed-Form Solution (Theorem 4.3):**

When $\mathbf{B}^{(1)} = \mathbf{B}^{(2)} = \mathbf{B}$, the optimal per-dimension diagonal entries of $\mathbf{A}$ are:

$$
\mathbf{A}_{l,l} = \frac{1}{16}\left(1 - 2\Lambda_{l,l}^{(3)} - \sqrt{(2\Lambda_{l,l}^{(3)} + 1)^2 + 8\Lambda_{l,l}^{(3)}}\right)
$$

where $\boldsymbol{\Lambda}^{(3)}$ are eigenvalues of a symmetric positive semidefinite matrix derived from $\mathbf{M}^{(1)}, \mathbf{M}^{(2)}$.

**Simplified ADERFs (SADERFs):**

For practical GPU/TPU deployment, SADERFs avoid eigendecomposition by using a **diagonal rescaling** $\boldsymbol{\Psi} \in \mathbb{D}_d$:

$$
f_{\text{SADE}}^{(1)}(\omega, \mathbf{x}) = f_{\text{GE}}^{(1)}(\omega, \boldsymbol{\Psi}\mathbf{x}), \quad f_{\text{SADE}}^{(2)}(\omega, \mathbf{y}) = f_{\text{GE}}^{(2)}(\omega, \boldsymbol{\Psi}^{-1}\mathbf{y})
$$

The optimal diagonal entries have a closed-form solution:

$$
\forall\, 1 \leq l \leq d: \quad \Psi_{l,l}^* = \left(\sum_j (\mathbf{y}_l^{(j)})^2 \Big/ \sum_i (\mathbf{x}_l^{(i)})^2\right)^{1/4}
$$

computable in $O(dL)$ time. SADERFs reduce to GERFs when $\boldsymbol{\Psi} = \mathbf{I}_d$.

## Complexity

| Operation | Standard Attention | FAVOR# (SDERF) | FAVOR# (ADERF) |
|-----------|-------------------|-----------------|-----------------|
| Feature precompute | N/A | $O(LMd)$ | $O(LMd + Ld^2 + Md^2)$ |
| Attention | $O(L^2d)$ | $O(LMd)$ | $O(LMd)$ |
| Parameter tuning | N/A | $O(dL)$ | $O(Ld^2 + d^3)$ |

**Total for ADERFs:** $O(L(Md + d^2) + Md^2 + d^3)$ — subquadratic in $L$, with the $d^3$ eigendecomposition cost amortized over $L$.

**Total for SADERFs:** $O(LMd)$ — identical to FAVOR+/FAVOR++, with $O(dL)$ one-time setup.

**Memory:** $O(LM + Md)$ — same as FAVOR+.

**Variance improvement:** SDERF achieves $\approx e^{10}\times$ variance reduction over GERFs and $\approx e^5\times$ over GERFs in heterogeneous data settings.

## Applicability

- **FAVOR#** = DERF-based self-attention approximation: drop-in replacement for FAVOR+/FAVOR++ in Performers
- **Speech modeling**: FAVOR# (SDERF) consistently outperforms FAVOR++ on LibriSpeech for Conformer-Transducer, especially at small $M$ (8 or 32 random features)
- **NLP**: FAVOR# achieves new SOTA on GLUE for low-rank attention Transformers (82.69 MNLI, 92.53 SST-2)
- **SADERFs recommended for GPU**: Only require elementwise operations (diagonal scaling), no SVD/eigendecomposition, so the feature map is as GPU-friendly as FAVOR+
- Applicable to any setting requiring kernel approximation with linear complexity

## Limitations

- **ADERF/SDERF eigendecomposition**: Requires computing eigendecomposition of $\mathbf{M}^{(1)}, \mathbf{M}^{(2)}$, which is $O(d^3)$ — not GPU-friendly as a per-layer per-step operation (though amortizable)
- **SADERFs as practical compromise**: Avoid eigendecomposition but sacrifice some variance reduction compared to full SDERFs
- **Same fundamental limitation as all Performers**: Linear attention cannot replicate the sharp, sparse attention patterns of softmax; the approximation is better but still approximate
- **Dense matrix parameters**: For ADERFs/SDERFs, the $d \times d$ matrices $\mathbf{B}^{(k)}$ mean the feature map costs $O(d^2)$ per token per feature (vs $O(d)$ for FAVOR+/OPRF), though the total remains subquadratic in $L$
- **Homogeneity heuristic still used**: While proven optimal for the log-variance objective, the objective itself is an average over all query-key pairs — individual high-importance pairs may still have suboptimal approximation
- **GPU efficiency**: The core benefit (variance reduction) allows using **fewer random features $M$** for the same quality, which reduces FLOPs — but whether this translates to wall-clock speedup depends on whether $M$ reduction changes the arithmetic intensity regime

## Implementation Notes

```python
import torch
import math

def compute_saderf_params(Q, K):
    """Compute SADERF diagonal rescaling parameters.

    SADERFs are the GPU-friendly variant: no eigendecomposition needed,
    just a per-dimension rescaling of queries and keys.

    Args:
        Q: (L, d) query matrix
        K: (L, d) key matrix
    Returns:
        psi: (d,) optimal diagonal rescaling
        A: optimal scalar parameter (from GERF formula)
    """
    L, d = Q.shape

    # Per-dimension second moments (Eq. 20)
    q_sq = (Q ** 2).sum(dim=0)  # (d,)  sum_i x_l^(i)^2
    k_sq = (K ** 2).sum(dim=0)  # (d,)  sum_j y_l^(j)^2

    # Optimal diagonal: Psi*_l = (sum_j y_l^2 / sum_i x_l^2)^{1/4}
    psi = (k_sq / (q_sq + 1e-8)).pow(0.25)  # (d,)

    # Compute average ||Psi*x + Psi^{-1}*y||^2 for optimal A
    Qp = Q * psi.unsqueeze(0)         # (L, d) rescaled queries
    Kp = K / psi.unsqueeze(0)         # (L, d) rescaled keys
    q_norm_sq = (Qp ** 2).sum() / L
    k_norm_sq = (Kp ** 2).sum() / L
    cross = (Qp.sum(0) @ Kp.sum(0)) / (L * L)
    phi = (q_norm_sq + 2 * cross + k_norm_sq).item()

    # Optimal A (same GERF formula, Theorem 3.3 from CRT paper)
    A = (1 - 2*phi - math.sqrt((2*phi + 1)**2 + 8*phi)) / 8
    return psi, A

def saderf_feature_map(X, W, psi, A, is_query=True):
    """SADERF feature map: GERF applied to diagonally-rescaled input.

    Args:
        X: (L, d) input (queries or keys)
        W: (M, d) random projection matrix
        psi: (d,) diagonal rescaling from compute_saderf_params
        A: optimal scalar parameter
        is_query: if True, scale by psi; if False, scale by 1/psi
    Returns:
        phi: (L, M) positive random features
    """
    M, d = W.shape
    B_scalar = math.sqrt(1 - 4*A)
    C_scalar = -1.0
    D = (1 - 4*A) ** (d/4)

    # Diagonal rescaling
    if is_query:
        X_scaled = X * psi.unsqueeze(0)
    else:
        X_scaled = X / psi.unsqueeze(0)

    # Same GERF computation on rescaled input
    omega_norms_sq = (W ** 2).sum(dim=1)
    projection = X_scaled @ W.T
    x_norms_sq = (X_scaled ** 2).sum(dim=1, keepdim=True)

    log_features = (A * omega_norms_sq.unsqueeze(0)
                    + B_scalar * projection
                    + C_scalar * x_norms_sq)

    return (D / math.sqrt(M)) * torch.exp(log_features)

def favor_sharp(Q, K, V, M=None):
    """FAVOR# attention using SADERFs (GPU-friendly variant).

    Same linear complexity as FAVOR+, but with significantly
    lower variance in the kernel approximation.
    """
    L, d = Q.shape
    if M is None:
        M = d

    # Scale inputs
    x = Q * (d ** -0.25)
    y = K * (d ** -0.25)

    # Compute SADERF parameters
    psi, A = compute_saderf_params(x, y)

    # Generate orthogonal random projections
    n_blocks = (M + d - 1) // d
    W_blocks = []
    for _ in range(n_blocks):
        G = torch.randn(d, d)
        Q_orth, _ = torch.linalg.qr(G)
        norms = torch.randn(d, d).norm(dim=1)
        W_blocks.append(Q_orth * norms.unsqueeze(1))
    W = torch.cat(W_blocks, dim=0)[:M]

    # Compute features with asymmetric rescaling
    Q_prime = saderf_feature_map(x, W, psi, A, is_query=True)
    K_prime = saderf_feature_map(y, W, psi, A, is_query=False)

    # Linear attention
    KV = K_prime.T @ V
    K_sum = K_prime.sum(dim=0)
    num = Q_prime @ KV
    den = Q_prime @ K_sum
    return num / den.unsqueeze(-1)
```

## References

- Likhosherstov, V., Choromanski, K., Dubey, A., Liu, F., Sarlos, T., & Weller, A. (2023). FAVOR#: Sharp Attention Kernel Approximations via New Classes of Positive Random Features. arXiv:2302.00787.
- Likhosherstov, V., Choromanski, K., Dubey, A., Liu, F., Sarlos, T., & Weller, A. (2022). Chefs' Random Tables: Non-Trigonometric Random Features. NeurIPS 2022.
- Choromanski, K., Likhosherstov, V., et al. (2021). Rethinking Attention with Performers. ICLR 2021.

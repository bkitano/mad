# 149: Chefs' Random Tables (Optimal Positive Random Features)

**Category**: approximation
**Gain type**: efficiency
**Source**: Likhosherstov, Choromanski et al. (2022), "Chefs' Random Tables: Non-Trigonometric Random Features" (NeurIPS 2022)
**Paper**: [papers/chefs-random-tables-oprf.pdf]
**Documented**: 2026-02-15

## Description

Chefs' Random Tables (CRTs) are a new class of **non-trigonometric** random features (RFs) for approximating Gaussian and softmax kernels, replacing the classical Random Kitchen Sinks (RKS) approach that relies on sin/cos maps. The key instantiation, **Optimal Positive Random Features (OPRFs)**, is the first RF method providing **unbiased softmax kernel estimation with positive and bounded features**, yielding exponentially small tail probabilities and up to $e^{60}\times$ variance reduction over standard positive random features (PosRFs). OPRFs achieve this by introducing a free parameter $A$ into a generalized exponential feature map and optimizing it in closed form to minimize variance. Combined with block-orthogonal projections, this yields **FAVOR++**, a drop-in improvement to the Performer's FAVOR+ mechanism.

The main innovation is recognizing that the exponential feature map $f(\omega, \mathbf{x}) = D\exp(A\|\omega\|^2 + B\omega^\top\mathbf{x} + C\|\mathbf{x}\|^2)$ has free parameters that can be **analytically optimized** using simple data statistics (norms of queries/keys), unlike previous approaches that fixed these parameters heuristically.

## Mathematical Form

**Core Operation — Generalized Exponential RFs (GERFs):**

For $p_{\text{GE}}(\omega) \sim \mathcal{N}(\mathbf{0}_d, \mathbf{I}_d)$, define:

$$
f_{\text{GE}}^{(1)}(\omega, \mathbf{x}) = D\exp(A\|\omega\|^2 + B\omega^\top\mathbf{x} + C\|\mathbf{x}\|^2)
$$

$$
f_{\text{GE}}^{(2)}(\omega, \mathbf{y}) = D\exp(A\|\omega\|^2 + sB\omega^\top\mathbf{y} + C\|\mathbf{y}\|^2)
$$

where $A, B, C, D \in \mathbb{C}$, $s \in \{-1, +1\}$.

**Validity Conditions (Theorem 3.1):** These are valid RFs for the Gaussian kernel when:

$$
\text{Re}(1 - 4A) > 0, \quad B = \sqrt{s(1-4A)}, \quad C = -(s+1)/2, \quad D = (\sqrt[4]{1-4A})^d
$$

**Key Definitions:**

- $K(\mathbf{x}, \mathbf{y}) = \exp\left(\frac{1}{2}\|\mathbf{x} - \mathbf{y}\|^2\right)$ — Gaussian kernel
- $K_{\text{sfm}}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top\mathbf{y})$ — softmax kernel (related via $K_{\text{sfm}} = \exp(\|\mathbf{x}\|^2/2) K(\mathbf{x},\mathbf{y}) \exp(\|\mathbf{y}\|^2/2)$)
- $A \in \mathbb{R}$ — free tunable parameter controlling the variance-bias tradeoff
- $M$ — number of random features ($M \ll L$)

**Variance of GERFs (Theorem 3.2):**

$$
\text{Var}\left(f_{\text{GE}}^{(1)}f_{\text{GE}}^{(2)}\right) = \frac{1}{2}\exp\left(-(s+1)(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2)\right) \times \left[\text{Re}\left(\alpha_1\exp(\alpha_2\|\mathbf{x}+s\mathbf{y}\|^2)\right) + \alpha_3\exp(\alpha_4\|\mathbf{x}+s\mathbf{y}\|^2)\right] - K(\mathbf{x},\mathbf{y})^2
$$

where $\alpha_1 = \left(\sqrt{1 + \frac{16A^2}{1-8A}}\right)^d$, $\alpha_2 = \left(s + \frac{1}{1-8A}\right)$, $\alpha_3 = \left(1 + \frac{16|A|^2}{1-8\text{Re}(A)}\right)^{d/2}$, $\alpha_4 = \left(\frac{s}{2} + \frac{s+2|1-4A|}{2(1-8\text{Re}(A))}\right)$.

**Optimal Positive Random Features (OPRFs):**

Setting $s = +1$ (positive features) and $A$ real with $\|\mathbf{x}+\mathbf{y}\|^2 > 0$, the variance is minimized when:

$$
A^* = \frac{1}{8}\left(1 - 2\phi - \sqrt{(2\phi+1)^2 + 8\phi}\right)
$$

where $\phi = \|\mathbf{x}+\mathbf{y}\|^2$. In practice, $\phi$ is replaced by the **data-averaged statistic**:

$$
\frac{1}{L^2}\sum_{i=1}^{L}\sum_{j=1}^{L}\|\mathbf{x}_i + \mathbf{y}_j\|^2 = \frac{1}{L}\sum_i\|\mathbf{x}_i\|^2 + \frac{2}{L^2}\left(\sum_i\mathbf{x}_i\right)^\top\left(\sum_j\mathbf{y}_j\right) + \frac{1}{L}\sum_j\|\mathbf{y}_j\|^2
$$

computable in $O(Ld)$ time via precomputed norms and sums.

**Discretely-Induced Random Features (DIRFs):**

An alternative CRT family using discrete distributions. The **Poisson RF** variant:

$$
f_{\text{pois}}(\omega, \mathbf{x}) = e^{\lambda d/2 - \|\mathbf{x}\|^2/2}\prod_{l=1}^{d}\mathbf{x}_l^{\omega_l}\lambda^{-\omega_l/2}
$$

where $\omega_l \sim \text{Poisson}(\lambda)$ independently. The optimal rate parameter is:

$$
\lambda^* = d^{-1/2}\left(\sum_{l=1}^{d}\mathbf{x}_l^2\mathbf{y}_l^2\right)^{1/2}
$$

**Block-Orthogonal Projections (FAVOR++ = OPRF + orthogonality):**

Orthogonal RFs provably reduce variance for OPRFs for **any** $d > 0$ (Theorem 4.1):

$$
\text{Var}(\widehat{K}_M^{\text{ort}}(\mathbf{x},\mathbf{y})) \leq \text{Var}(\widehat{K}_M^{\text{iid}}(\mathbf{x},\mathbf{y})) - \left(1 - \frac{1}{M}\right)\frac{2}{d+2}C(\|\mathbf{x}+\mathbf{y}\|)
$$

for some $C(\|\mathbf{x}+\mathbf{y}\|) \geq 0$. This holds for any $d$, unlike TrigRFs which only benefit asymptotically.

**Uniform Convergence (Theorem 4.3):**

For $\mathbf{Q}, \mathbf{K}$ rows in an $L_2$-ball of radius $R$, with $M = \Omega(\Gamma\frac{d}{\sigma}\log(\frac{\gamma\rho}{\sigma}))$ features:

$$
\|\mathcal{K}_{\text{sfm}} - \widehat{\mathcal{K}}_{\text{sfm}}\|_\infty \leq \epsilon
$$

with constant probability, where $\Gamma = \exp(-\frac{3R^2}{\sqrt{dA}})$, $\rho = \sqrt{2}Rd^{-1/4}$.

## Complexity

| Operation | Standard Attention | FAVOR++ (OPRF) |
|-----------|-------------------|----------------|
| Time | $O(L^2d)$ | $O(LMd)$ |
| Space | $O(L^2)$ | $O(LM + Md)$ |
| OPRF parameter tuning | N/A | $O(Ld)$ (one-time) |

**Memory:** $O(LM + Md)$ vs $O(L^2)$ — identical asymptotic complexity to FAVOR+, with the only overhead being an $O(Ld)$ precomputation of data statistics.

**Variance reduction:** Up to $e^{60}\times$ over PosRFs and $e^{75}\times$ over TrigRFs in tested configurations ($d = 64$, standard Transformer regime), with no additional computational cost per random feature evaluation.

## Applicability

- **Drop-in replacement** for FAVOR+ in Performer architectures — only changes the feature map parameters, same $O(LMd)$ complexity
- Particularly beneficial when **few random features** $M$ are used (variance reduction most impactful at small $M$)
- Applicable to any kernel approximation task: kernel SVM, kernel regression, Gaussian processes
- Validated on NLP (GLUE benchmark), speech (LibriSpeech), and vision (ImageNet) tasks
- FAVOR++ outperforms FAVOR+ on **all GLUE tasks**, with uptraining from softmax Transformer checkpoints
- Speech models: 2.49-3.05% WER improvement over FAVOR+ on LibriSpeech

## Limitations

- **Homogeneity assumption**: The closed-form $A^*$ assumes query/key norms are concentrated around their mean — breaks down if norm variance is very high
- **Still an approximation**: While variance is drastically reduced, the linear attention paradigm still cannot perfectly replicate sharp softmax distributions
- **GPU efficiency concern**: The feature map computation $\exp(A\|\omega\|^2 + B\omega^\top\mathbf{x} + C\|\mathbf{x}\|^2)$ involves the same operations as FAVOR+ (one matrix multiply + elementwise exp), so the GPU speedup story is identical to Performers — limited by whether linear attention itself is faster than FlashAttention for given $L, d, M$
- **Eigendecomposition for ADERF/SDERF**: The optimal dense variants (not OPRFs themselves) require SVD/eigendecomposition which lacks mature GPU/TPU fused kernel support
- **ORF blocks**: Orthogonal projections require $M \leq d$ per block; for $M > d$, must use multiple independent orthogonal blocks, adding some implementation complexity

## Implementation Notes

```python
import torch
import math

def compute_oprf_parameter(Q, K):
    """Compute optimal A parameter for OPRFs from data statistics.

    Args:
        Q: (L, d) query matrix (already scaled by d^{-1/4})
        K: (L, d) key matrix (already scaled by d^{-1/4})
    Returns:
        A: optimal scalar parameter
    """
    L, d = Q.shape
    # Compute average ||x_i + y_j||^2 via decomposition (Eq. 8)
    q_norm_sq = (Q ** 2).sum() / L          # avg ||q||^2
    k_norm_sq = (K ** 2).sum() / L          # avg ||k||^2
    cross = (Q.sum(0) @ K.sum(0)) / (L * L) # avg q^T k
    phi = q_norm_sq + 2 * cross + k_norm_sq  # avg ||q+k||^2

    # Closed-form optimal A (Theorem 3.3, Eq. 7)
    phi = phi.item()
    A = (1 - 2*phi - math.sqrt((2*phi + 1)**2 + 8*phi)) / 8
    return A

def oprf_feature_map(X, W, A):
    """OPRF feature map: generalized exponential with optimal A.

    Args:
        X: (L, d) input (queries or keys, scaled)
        W: (M, d) random projection matrix (orthogonal rows)
        A: optimal scalar parameter from compute_oprf_parameter
    Returns:
        phi: (L, M) positive random features
    """
    M, d = W.shape
    B = math.sqrt(1 - 4*A)  # s=+1 for positive features
    C = -1.0                 # s=+1 => C = -(s+1)/2 = -1
    D = (1 - 4*A) ** (d/4)  # D = (1-4A)^{d/4}

    # f(omega, x) = D * exp(A||omega||^2 + B*omega^T*x + C*||x||^2)
    omega_norms_sq = (W ** 2).sum(dim=1)      # (M,)
    projection = X @ W.T                       # (L, M)
    x_norms_sq = (X ** 2).sum(dim=1, keepdim=True)  # (L, 1)

    log_features = (A * omega_norms_sq.unsqueeze(0)  # (1, M)
                    + B * projection                   # (L, M)
                    + C * x_norms_sq)                  # (L, 1)

    return (D / math.sqrt(M)) * torch.exp(log_features)

def favor_plus_plus(Q, K, V, M=None):
    """FAVOR++ attention using OPRFs with block-orthogonal projections."""
    L, d = Q.shape
    if M is None:
        M = d

    # Generate block-orthogonal random projections
    n_blocks = (M + d - 1) // d
    W_blocks = []
    for _ in range(n_blocks):
        G = torch.randn(d, d)
        Q_orth, _ = torch.linalg.qr(G)
        norms = torch.randn(d, d).norm(dim=1)
        W_blocks.append(Q_orth * norms.unsqueeze(1))
    W = torch.cat(W_blocks, dim=0)[:M]  # (M, d)

    # Compute optimal parameter
    x = Q * (d ** -0.25)
    y = K * (d ** -0.25)
    A = compute_oprf_parameter(x, y)

    # Compute features
    Q_prime = oprf_feature_map(x, W, A)  # (L, M)
    K_prime = oprf_feature_map(y, W, A)  # (L, M)

    # Linear attention: Q'(K'^T V) / Q'(K'^T 1)
    KV = K_prime.T @ V            # (M, d_v)
    K_sum = K_prime.sum(dim=0)    # (M,)

    num = Q_prime @ KV            # (L, d_v)
    den = Q_prime @ K_sum         # (L,)

    return num / den.unsqueeze(-1)
```

## References

- Likhosherstov, V., Choromanski, K., Dubey, A., Liu, F., Sarlos, T., & Weller, A. (2022). Chefs' Random Tables: Non-Trigonometric Random Features. NeurIPS 2022.
- Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007.

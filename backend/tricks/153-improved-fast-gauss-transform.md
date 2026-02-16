# 153: Improved Fast Gauss Transform (IFGT)

**Category**: approximation
**Gain type**: efficiency
**Source**: Yang, Duraiswami, Gumerov & Davis (ICCV 2003)
**Paper**: [papers/improved-fast-gauss-transform.pdf]
**Documented**: 2026-02-15

## Description

The Improved Fast Gauss Transform (IFGT) accelerates the evaluation of weighted sums of Gaussians from $O(MN)$ to $O((M+N) \cdot r_{p-1,d})$ where $r_{p-1,d} = \binom{p-1+d}{d}$ is the number of multivariate Taylor expansion terms. It fixes two critical defects of the original Fast Gauss Transform (FGT) of Greengard & Strain (1991):

1. **Exponential dimension scaling**: The original FGT uses Hermite expansions factored per-dimension, giving $p^d$ terms (exponential in $d$). IFGT instead treats the dot product $\Delta y_j \cdot \Delta x_i$ as a scalar and expands it via multivariate Taylor polynomials in graded lexicographic order, reducing the number of terms to $\binom{p-1+d}{d} \sim O(d^p)$ — polynomial in $d$ for fixed $p$.

2. **Uniform grid subdivision**: The original FGT uses uniform box decomposition, which wastes computation on empty boxes in high dimensions (the ratio of hypercube to inscribed hypersphere volume grows exponentially). IFGT uses **farthest-point clustering** (a 2-approximation to the $k$-center problem) to adaptively partition sources into clusters with bounded radius $\rho_x < h\rho_x$, avoiding empty-box overhead.

The key factorization exploits the Gaussian's infinite differentiability: $e^{-\|y_j - x_i\|^2/h^2} = e^{-\|\Delta y_j\|^2/h^2} \cdot e^{-\|\Delta x_i\|^2/h^2} \cdot e^{2\Delta y_j \cdot \Delta x_i / h^2}$, where the last entangled term is expanded into separable multivariate Taylor series around each cluster center.

## Mathematical Form

**Core Problem:**

Compute the weighted sum of Gaussians:

$$
G(y_j) = \sum_{i=1}^{N} q_i \, e^{-\|y_j - x_i\|^2 / h^2}, \quad j = 1, \ldots, M
$$

where $\{x_i\}_{i=1}^N$ are source points, $\{y_j\}_{j=1}^M$ are target points, $q_i$ are weights, and $h$ is the bandwidth.

**IFGT Factorization:**

Let $x_*$ be a cluster center. Define $\Delta y_j = y_j - x_*$ and $\Delta x_i = x_i - x_*$. Then:

$$
e^{-\|y_j - x_i\|^2/h^2} = e^{-\|\Delta y_j\|^2/h^2} \cdot e^{-\|\Delta x_i\|^2/h^2} \cdot e^{2\Delta y_j \cdot \Delta x_i / h^2}
$$

**Multivariate Taylor Expansion of the Entangled Term:**

Using the multinomial expansion of the dot product:

$$
e^{2\Delta y_j \cdot \Delta x_i / h^2} = \sum_{\alpha \geq 0} \frac{2^{|\alpha|}}{\alpha!} \left(\frac{\Delta y_j}{h}\right)^\alpha \left(\frac{\Delta x_i}{h}\right)^\alpha
$$

where $\alpha = (\alpha_1, \ldots, \alpha_d)$ is a multi-index with $|\alpha| = \alpha_1 + \cdots + \alpha_d$ and $\alpha! = \alpha_1! \cdots \alpha_d!$.

**Truncated Expansion (order $p-1$):**

$$
G(y_j) \approx \sum_{\|y_j - c_k\| \leq h\rho_y} \sum_{|\alpha| < p} C_\alpha^k \, e^{-\|y_j - c_k\|^2/h^2} \left(\frac{y_j - c_k}{h}\right)^\alpha
$$

where $c_k$ is the center of cluster $k$, and the expansion coefficients are:

$$
C_\alpha^k = \frac{2^{|\alpha|}}{\alpha!} \sum_{x_i \in S_k} q_i \, e^{-\|x_i - c_k\|^2/h^2} \left(\frac{x_i - c_k}{h}\right)^\alpha
$$

**Number of Expansion Terms:**

$$
r_{p-1,d} = \binom{p - 1 + d}{d}
$$

For comparison, the original FGT requires $p^d$ terms:

| $p \backslash d$ | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Original FGT ($p^d$) | 4096 (p=4,d=6) | — | — | — | — |
| IFGT ($\binom{p-1+d}{d}$) | 84 (p=4,d=6) | 462 (p=6,d=6) | 120 (p=8,d=4) | 3003 (p=10,d=6) | 455 (p=12,d=3) |

**Efficient Recursive Computation of Multivariate Monomials:**

Terms of order $k$ are computed from order $k-1$ by multiplying each variable's component between the variable's leading term and the end, using graded lexicographic ordering. This requires $r_{p-1,d}$ multiplications and $r_{p-1,d}$ storage.

**Error Bound:**

$$
|E(y)| \leq Q \left(\frac{2^p}{p!} \rho_x^p \rho_y^p + e^{-\rho_y^2}\right)
$$

where $Q = \sum |q_i|$, $\rho_x$ is the maximum source cluster radius (in units of $h$), and $\rho_y = h\rho_y$ controls the far-field cutoff.

**Key Definitions:**

- $x_i \in \mathbb{R}^d$ — source points ($N$ total)
- $y_j \in \mathbb{R}^d$ — target points ($M$ total)
- $q_i$ — source weights
- $h$ — Gaussian bandwidth
- $c_k$ — cluster centers from farthest-point clustering
- $\rho_x$ — max cluster radius / $h$ (controls expansion accuracy)
- $\rho_y$ — neighbor search radius / $h$ (controls far-field cutoff)
- $\alpha$ — multi-index in graded lexicographic order
- $r_{p-1,d} = \binom{p-1+d}{d}$ — number of multivariate Taylor terms

## Complexity

| Operation | Naive | Original FGT | IFGT |
|-----------|-------|-------------|------|
| Kernel summation | $O(MN)$ | $O((M+N) p^d)$ | $O((M+N) r_{p-1,d})$ |
| Expansion terms | — | $p^d$ | $\binom{p-1+d}{d}$ |
| Space subdivision | — | uniform grid ($O(10^d)$ boxes) | farthest-point clustering ($K$ clusters) |
| Storage | $O(MN)$ | $O(K p^d)$ | $O(K r_{p-1,d})$ |

**Memory:** $O(K \cdot r_{p-1,d})$ for storing cluster coefficients, where $K$ is the number of clusters ($K \leq N$). Since $n \leq K$, the algorithm achieves linear runtime.

**Practical scaling**: For $d=10$, $p=10$, $N=M=10000$: IFGT takes 1.57s vs. FGT's 103.9s vs. direct evaluation's 271.5s (Table 2 in paper). Linear scaling in $N$ is confirmed empirically for dimensions 4–10.

## Applicability

- **RBF/Gaussian kernel attention**: Any attention mechanism using Gaussian kernels $K(q_i, k_j) = \exp(-\|q_i - k_j\|^2 / 2\sigma^2)$ can use IFGT for $O(N)$ approximate MVMs, provided the head dimension $d$ is moderate ($d \leq 10$). This is relevant for RBF attention variants and Performer-style architectures where features live in low-dimensional spaces.
- **Kernel density estimation in t-SNE/UMAP**: The gradient computation of t-SNE requires sums of Gaussians in 2–3D embedding spaces — exactly IFGT's sweet spot.
- **Mean shift clustering acceleration**: The paper demonstrates IFGT reducing mean shift from $O(N^2)$ to $O(N)$ per iteration, applicable to feature clustering in neural networks.
- **Gaussian process inference**: GP posterior mean computation with RBF kernels is a direct application.
- **Score-based diffusion models**: Score matching loss involves sums of Gaussian kernel evaluations; IFGT can accelerate training of score-based generative models when operating in moderate-dimensional feature spaces.

## Limitations

- **Dimension ceiling around $d \approx 12$**: While much better than FGT ($p^d$), the term count $\binom{p-1+d}{d}$ still grows rapidly. For $d=12, p=10$: 293,930 terms. For $d > 12$, the expansion cost exceeds direct evaluation for practical $N$.
- **Not directly applicable to standard softmax attention**: The softmax kernel $\exp(q^T k / \sqrt{d})$ is not a radial kernel; IFGT applies to Gaussian/RBF kernels $\exp(-\|q - k\|^2)$.
- **Farthest-point clustering overhead**: The $O(NK)$ clustering step is sequential and not easily parallelizable on GPU. For large $N$, use Feder & Greene's $O(N \log K)$ algorithm.
- **GPU parallelism concerns**: The cluster-based near-neighbor search involves irregular memory access patterns. While the per-cluster Taylor evaluation is embarrassingly parallel, the spatial data structure traversal introduces warp divergence. Needs careful kernel design for GPU efficiency.
- **Bandwidth-dependent accuracy**: The error bound depends on $\rho_x \cdot \rho_y$; very large bandwidths $h$ require more clusters (smaller $\rho_x$) or higher truncation order $p$.

## Implementation Notes

```python
# Improved Fast Gauss Transform (IFGT) pseudocode
import numpy as np
from scipy.special import comb

def ifgt_gauss_sum(
    sources: np.ndarray,    # (N, d) source points
    targets: np.ndarray,    # (M, d) target points
    weights: np.ndarray,    # (N,) weights q_i
    bandwidth: float,       # h
    p: int = 8,            # truncation order
    epsilon: float = 1e-4,  # desired accuracy
) -> np.ndarray:
    """
    Compute G(y_j) = sum_i q_i * exp(-||y_j - x_i||^2 / h^2)
    in O((M+N) * r_{p-1,d}) time via IFGT.
    """
    N, d = sources.shape
    M = targets.shape[0]
    h = bandwidth

    # Number of multivariate Taylor terms
    r_pd = int(comb(p - 1 + d, d))

    # Step 1: Farthest-point clustering of sources
    # Cluster radius < h * rho_x where rho_x chosen so error < epsilon
    centers, assignments, K = farthest_point_clustering(
        sources, h, p, epsilon
    )

    # Step 2: Compute expansion coefficients for each cluster
    # C_alpha^k = (2^|alpha| / alpha!) * sum_{x_i in S_k}
    #             q_i * exp(-||x_i - c_k||^2/h^2) * ((x_i - c_k)/h)^alpha
    coeffs = np.zeros((K, r_pd))
    for k in range(K):
        mask = (assignments == k)
        dx = (sources[mask] - centers[k]) / h  # (n_k, d)
        w = weights[mask] * np.exp(-np.sum(dx**2, axis=1))  # (n_k,)

        # Compute multivariate monomials in graded lex order
        monoms = multivariate_monomials(dx, p - 1)  # (n_k, r_pd)

        # Accumulate: multiply by 2^|alpha|/alpha! factors
        for idx, alpha in enumerate(graded_lex_indices(d, p - 1)):
            factor = 2**sum(alpha) / np.prod(
                [np.math.factorial(a) for a in alpha]
            )
            coeffs[k, idx] = factor * np.sum(w * monoms[:, idx])

    # Step 3: Evaluate at targets
    result = np.zeros(M)
    for j in range(M):
        # Find neighbor clusters within range h * rho_y
        neighbor_clusters = find_neighbors(
            targets[j], centers, h, rho_y
        )
        for k in neighbor_clusters:
            dy = (targets[j] - centers[k]) / h
            exp_factor = np.exp(-np.sum(dy**2))
            # Evaluate Taylor polynomial
            target_monoms = multivariate_monomials(
                dy.reshape(1, -1), p - 1
            )  # (1, r_pd)
            result[j] += exp_factor * np.dot(
                coeffs[k], target_monoms[0]
            )

    return result

# GPU optimization notes:
# - Step 2 (source-to-multipole): parallelize over clusters
#   Each cluster's coefficient computation is a weighted reduction
#   Maps to batched GEMM: (n_k, 1)^T @ (n_k, r_pd)
# - Step 3 (multipole-to-target): parallelize over targets
#   Each target evaluates polynomial at O(n_neighbors) clusters
#   The polynomial evaluation is a dot product (r_pd elements)
# - Main GPU concern: irregular cluster sizes cause load imbalance
#   Mitigate by sorting clusters by size, padding to warp boundaries
```

## References

- Yang, C., Duraiswami, R., Gumerov, N. A., & Davis, L. (2003). Improved Fast Gauss Transform and Efficient Kernel Density Estimation. ICCV 2003.
- Greengard, L. & Strain, J. (1991). The Fast Gauss Transform. SIAM J. Sci. Statist. Comput., 12(1):79-94.
- Greengard, L. & Rokhlin, V. (1987). A fast algorithm for particle simulations. J. Comput. Phys., 73(2):325-348.
- Raykar, V. C., Yang, C., Duraiswami, R., & Gumerov, N. A. (2006). Fast computation of sums of Gaussians in high dimensions. Technical Report CS-TR-4767, University of Maryland.
- Gonzalez, T. (1985). Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38:293-306.

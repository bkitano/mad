# 044: Fast Kernel Transform (FKT)

**Category**: approximation
**Gain type**: efficiency
**Source**: Ryan, Ament, Gomes & Damle (AISTATS 2022)
**Paper**: [papers/fast-kernel-transform.pdf]
**Documented**: 2026-02-15

## Description

The Fast Kernel Transform (FKT) is a general algorithm for computing approximate matrix-vector multiplications (MVMs) with kernel matrices in $O(N \log N)$ time, applicable to any isotropic kernel — including the Cauchy kernel $K(r) = 1/(1 + r^2/\sigma^2)$ that appears in t-SNE, SSM generating functions, and Gaussian process regression. Unlike the classical Fast Multipole Method (FMM), which requires kernel-specific analytical expansions to be manually derived, FKT uses a **generalized multipole expansion** based on automatic differentiation that works for any analytic kernel function. The expansion separates kernel evaluations into radial and angular components using hyperspherical harmonics, enabling low-rank compression of far-field interactions via a Barnes-Hut tree decomposition.

The key innovation is that FKT derives its expansion coefficients automatically from the kernel's Taylor series using Faa di Bruno's formula and Gegenbauer polynomials, eliminating the need for per-kernel mathematical derivation. This makes it a "drop-in" accelerator for any kernel MVM appearing in neural network training or inference.

## Mathematical Form

**Core Problem:**

Compute the kernel matrix-vector product:

$$
z_i = \sum_{j=0}^{N} K(\|\mathbf{r}_i - \mathbf{r}_j\|) y_j
$$

where $K$ is an isotropic kernel and $\mathbf{r}_i \in \mathbb{R}^d$.

**Generalized Multipole Expansion:**

For any kernel $K$ that is analytic (except possibly at the origin), define $\varepsilon := \frac{\|\mathbf{r}'\|}{\|\mathbf{r}\|}\left(\frac{\|\mathbf{r}'\|}{\|\mathbf{r}\|} - 2\cos\gamma\right)$ where $\gamma$ is the angle between $\mathbf{r}'$ and $\mathbf{r}$. Then:

$$
K(\|\mathbf{r}' - \mathbf{r}\|) = \sum_{k=0}^{\infty} \sum_{h \in \mathcal{H}_k} Y_k^h(\mathbf{r}) Y_k^h(\mathbf{r}')^* \mathcal{K}^{(k)}(\|\mathbf{r}'\|, \|\mathbf{r}\|)
$$

**Key Definitions:**

- $Y_k^h(\mathbf{r})$ — hyperspherical harmonics (angular basis functions)
- $\mathcal{H}_k := \{(\mu_1, \ldots, \mu_{d-2}) : k \geq \mu_1 \geq \cdots \geq |\mu_{d-2}| \geq 0\}$ — multi-index set
- $\mathcal{K}^{(k)}(\|\mathbf{r}'\|, \|\mathbf{r}\|) := \sum_{j=k}^{\infty} \|\mathbf{r}'\|^j \sum_{m=1}^{j} K^{(m)}(\|\mathbf{r}\|) \|\mathbf{r}\|^{m-j} \mathcal{T}_{jkm}^{(\alpha)}$ — radial kernel coefficients
- $\mathcal{T}_{jkm}^{(\alpha)}$ — constants depending only on dimension $d$, not the kernel

**Truncated Expansion (order $p$):**

$$
K(\|\mathbf{r}' - \mathbf{r}\|) \approx \sum_{k=0}^{p} \sum_{h \in \mathcal{H}_k} Y_k^h(\mathbf{r}) Y_k^h(\mathbf{r}')^* \mathcal{K}_p^{(k)}(\|\mathbf{r}'\|, \|\mathbf{r}\|)
$$

This separates into a **low-rank factorization** for well-separated point pairs:

$$
K(\|\mathbf{r}_i - \mathbf{r}_j\|) \approx \sum_{k=0}^{\mathcal{P}} U_k(\mathbf{r}_i) V_k(\mathbf{r}_j)
$$

where $\mathcal{P} = \binom{p+d}{d} \sim d^p$ is the number of expansion terms, and:

$$
U_{k,h,j}(\mathbf{r}') := Y_k^h(\mathbf{r}')^* \|\mathbf{r}'\|^j, \quad V_{k,h,j}(\mathbf{r}) := Y_k^h(\mathbf{r}) \sum_{m=1}^{j} K^{(m)}(\|\mathbf{r}\|) \|\mathbf{r}\|^{m-j} \mathcal{T}_{jkm}^{(\alpha)}
$$

**Barnes-Hut Decomposition:**

The algorithm uses a $k$-d tree to partition space. For each node, interactions are split into:

$$
z = Ky \approx \sum_{l \in \text{leaves}} K_{N_l, l} \cdot y_l + \sum_{b \in \text{nodes}} \hat{K}_{F_b, b} \cdot y_b
$$

where $K_{N_l, l}$ are dense near-field blocks and $\hat{K}_{F_b, b}$ are low-rank far-field approximations via the multipole expansion.

## Complexity

| Operation | Naive | With FKT |
|-----------|-------|----------|
| Kernel MVM | $O(N^2)$ | $O(N \log(N/m) \cdot d^p)$ |
| Storage | $O(N^2)$ | $O(N)$ |

**Detailed cost breakdown:**

$$
\text{FKT}_{\text{cost}} = O\left(N\left(mc_n^d + (1 + c_f^d) \log(N/m) d^p\right)\right)
$$

where:
- $m$ — maximum leaf capacity in the tree
- $c_n, c_f$ — geometry-dependent constants (typically 2–5)
- $d$ — dimension of the data
- $p$ — truncation order (controls accuracy)

For $d < 6$ (typical for neural network feature spaces):

$$
\text{FKT}_{\text{cost}} = O\left(N \log(N/d^p) \times c_f^d \times d^p\right)
$$

**Truncation error** decays exponentially with $p$:

$$
|\mathcal{E}_P| \leq \sum_{k=0}^{\infty} \binom{k+d-3}{k} \left| \sum_{j=\max(p+1,k)}^{\infty} \sum_{m=1}^{j} K^{(m)}(\|\mathbf{r}\|) \|\mathbf{r}\|^m \left(\frac{\|\mathbf{r}'\|}{\|\mathbf{r}\|}\right)^j \mathcal{T}_{jkm}^{(\alpha)} \right|
$$

In practice, $p = 4$ yields residuals below $10^{-4}$ for the Cauchy kernel.

## Applicability

- **t-SNE / UMAP acceleration**: The gradient of t-SNE involves matrix-vector products with the Cauchy kernel $(1 + \|\mathbf{r}_i - \mathbf{r}_j\|^2)^{-1}$ on 2D embeddings. FKT provides exact-to-tolerance acceleration, outperforming the Barnes-Hut approximation commonly used.
- **Gaussian process inference**: Computing the posterior mean requires kernel MVMs. FKT with Matérn-3/2 kernels achieved identical accuracy to exact methods on 145K data points in ~12 minutes.
- **SSM kernel evaluation**: When S4-style models evaluate Cauchy kernels $\sum_i v_i / (\omega_j - \lambda_i)$ at many evaluation points, FKT can accelerate the computation from $O(NL)$ to $O((N+L) \log(N+L))$, providing a practical alternative to the theoretically optimal but hard-to-implement classical FMM.
- **Attention kernel approximation**: Any attention mechanism using distance-based kernels (e.g., RBF attention) can leverage FKT for quasi-linear MVMs.

## Limitations

- Does **not scale well to dimensions $d > 6$**: in high dimensions, nearby points dominate and the near-field computation becomes $O(N^2)$. The expansion size $\mathcal{P} = \binom{p+d}{d}$ grows exponentially with $d$.
- Requires the kernel to be **analytic** (except possibly at the origin); non-smooth kernels need special treatment.
- The tree construction adds overhead that makes FKT slower than dense methods for small $N$ (crossover at $N \approx 1000$–$5000$ depending on $d$).
- Currently scales **quasi-linearly** ($O(N \log N)$) rather than truly linearly like the classical FMM, because it uses tree-code style compression rather than FMM's full translation operators.
- Error bounds are descriptive rather than tight — practical accuracy must be verified empirically for each kernel.

## Implementation Notes

```python
# Pseudocode for FKT (Barnes-Hut with generalized multipole expansion)
import numpy as np
from scipy.special import sph_harm  # spherical harmonics

def fkt_matvec(points, y, kernel, kernel_derivs, p=4, theta=0.75, m=512):
    """
    Fast Kernel Transform: approximate K @ y in O(N log N).

    points: (N, d) array of data points
    y: (N,) vector to multiply
    kernel: callable K(r) -> scalar
    kernel_derivs: callable returning [K'(r), K''(r), ..., K^(p)(r)]
    p: truncation order (higher = more accurate)
    theta: distance parameter (near/far threshold)
    m: max leaf capacity
    """
    N, d = points.shape

    # Step 1: Build k-d tree
    tree = build_kd_tree(points, max_leaf=m)

    # Step 2: For each node, determine near and far sets
    z = np.zeros(N)

    for leaf in tree.leaves:
        # Dense near-field computation
        near_pts = get_near_points(leaf, theta)
        K_near = kernel_matrix(points[leaf.indices], points[near_pts])
        z[leaf.indices] += K_near @ y[near_pts]

    for node in tree.nodes:
        far_pts = get_far_points(node, theta)
        if len(far_pts) == 0:
            continue

        # Source-to-multipole: compute V_k coefficients
        # s2m[k] = sum_j V_k(r_j) * y_j for j in far_pts
        s2m = compute_s2m(points[far_pts], y[far_pts],
                          kernel_derivs, p, d)

        # Multipole-to-target: compute U_k coefficients
        # z[i] += sum_k U_k(r_i) * s2m[k] for i in node.indices
        m2t = compute_m2t(points[node.indices], s2m, p, d)
        z[node.indices] += m2t

    return z

def compute_s2m(points, y, kernel_derivs, p, d):
    """Source-to-multipole: project onto expansion basis."""
    # Uses hyperspherical harmonics Y_k^h
    # and kernel derivatives K^(m)(r) via autodiff
    P = comb(p + d, d)  # number of expansion terms
    coeffs = np.zeros(P)
    for j, (r, yj) in enumerate(zip(points, y)):
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        for k in range(p + 1):
            for h in H_k(k, d):
                idx = flatten_index(k, h)
                Ykh = hyperspherical_harmonic(k, h, r_hat)
                coeffs[idx] += np.conj(Ykh) * r_norm**k * yj
    return coeffs
```

## References

- Ryan, J. P., Ament, S., Gomes, C. P., & Damle, A. (2022). The Fast Kernel Transform. Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR 151.
- Greengard, L. & Rokhlin, V. (1987). A fast algorithm for particle simulations. Journal of Computational Physics, 73(2), 325-348.
- Barnes, J. & Hut, P. (1986). A hierarchical O(N log N) force-calculation algorithm. Nature, 324(6096), 446-449.
- Pan, V. Y. (2017). Fast Approximate Computations with Cauchy Matrices and Polynomials. arXiv:1506.02285.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022.
- Van Der Maaten, L. (2014). Accelerating t-SNE using Tree-Based Algorithms. JMLR, 15, 3221-3245.

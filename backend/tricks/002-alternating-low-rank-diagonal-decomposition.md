# 002: Alternating Low-Rank then Diagonal (Alt) Decomposition

**Category**: decomposition
**Gain type**: efficiency
**Source**: Yeon & Anitescu (2025)
**Paper**: [papers/spectral-lrpd-decomposition.pdf]
**Documented**: 2026-02-15

## Description

An iterative spectral algorithm for decomposing symmetric positive semidefinite matrices into a low-rank plus diagonal (LRPD) form: $\Sigma = D + UU^T$ where $D$ is diagonal and $UU^T$ is rank-$k$. Unlike naive approaches that directly subtract the diagonal, this method alternates between low-rank factorization and diagonal adjustment steps, provably reducing approximation error at each iteration while avoiding rotational degeneracy issues inherent in gradient-based factorization methods.

This decomposition is critical for covariance matrices, state-space models (SSMs like S4), and kernel learning where global correlations (low-rank) and local variances (diagonal) must be efficiently separated.

## Mathematical Form

**Problem Setup:**

Given symmetric PSD matrix $\Sigma \in \mathbb{R}^{n \times n}$ with eigendecomposition $\Sigma = V\Lambda V^T$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with $\lambda_1 \geq \cdots \geq \lambda_n \geq 0$, find:

$$
\min_{D, U} \|A - D - UU^T\|_F^2
$$

subject to $D$ diagonal and $\text{rank}(UU^T) \leq k$.

**Core Algorithm (Alt):**

For iteration $t$:

1. **Low-rank step:** Compute eigendecomposition of residual $R_t = A - D_{t-1}$:
   $$R_t = V\Lambda V^T$$

   Extract top-$k$ eigenvectors scaled by $\sqrt{\max(\lambda_i, 0)}$:
   $$U_t \leftarrow V_{[:, 1:k]} \sqrt{\text{diag}(\max(\lambda_1, \ldots, \lambda_k, 0))}$$

2. **Diagonal step:** Update diagonal to minimize Frobenius error:
   $$D_t \leftarrow \text{diag}(\text{diag}(A) - \text{diag}(U_t U_t^T))$$

**Key Definitions:**

- $S_k = \Sigma - U_k U_k^T = V_{\geq k} \Lambda_{\geq k} V_{\geq k}^T$ — Residual after low-rank removal
- $D_k = \text{diag}(\Sigma_{ii} - (U_k U_k^T)_{ii})$ — Diagonal correction
- $R_k = \Sigma - (D_k + U_k U_k^T) = S_k - D_k$ — Corrected residual

**Convergence Guarantee (Theorem 2.2):**

If $\delta = \lambda_k(L^*) > 0$ and $\|D^*\|_2 < \frac{\delta}{2}$, then each alternating step contracts:

$$
E(D_{t-1}, U_{t-1}) \geq E(D_{t-1}, U_t) \geq E(D_t, U_t)
$$

where $E(D, U) = \|A - D - UU^T\|_F^2$, with the diagonal errors satisfying $\|\Delta_t\|_2 \leq \|\Delta_{t-1}\|_2$.

## Complexity

| Operation | Naive | Alt (per iteration) |
|-----------|-------|---------------------|
| Eigendecomposition | $O(n^3)$ | $O(n^3)$ (full) or $O(nk^2)$ (partial) |
| Low-rank update | $O(n^2k)$ | $O(n^2k)$ |
| Diagonal update | $O(n)$ | $O(n)$ |
| Total per iteration | — | $O(n^3)$ or $O(n^2k)$ with partial eigen-solve |

**Memory:** $O(nk)$ for low-rank factors + $O(n)$ for diagonal vs $O(n^2)$ for full matrix

**Convergence:** Typically 5-20 iterations to machine precision (empirically validated on $n=150$, $k=5$)

## Applicability

- **State-space models (S4, Mamba):** DPLR parameterization of state matrices enables $O(n)$ recurrence via Woodbury identity
- **Covariance estimation:** Separating global market factors (low-rank) from asset-specific variance (diagonal)
- **Kernel approximation:** Random features with diagonal corrections for better accuracy
- **Riccati equations:** Low-rank plus diagonal solutions for high-dimensional control/filtering problems
- **Factor analysis:** Shared latent factors (low-rank) plus measurement noise (diagonal)

## Limitations

- Requires eigenvalue gap $\delta = \lambda_k(L^*) > 0$ for provable contraction (Theorem 2.2 condition)
- Full eigendecomposition is $O(n^3)$ per iteration; randomized methods needed for large-scale problems
- Assumes target matrix is well-approximated by LRPD structure; pure low-rank matrices ($D \approx 0$) reduce to standard truncated SVD
- For non-PSD matrices, need to extend with signed eigenvalues or work in PPCA parameterization

## Implementation Notes

```python
def alternating_lrd_decomposition(A, k, max_iters=20, tol=1e-6):
    """
    Alternating Low-Rank then Diagonal decomposition

    Args:
        A: n x n symmetric PSD matrix
        k: target rank
        max_iters: maximum iterations
        tol: convergence tolerance

    Returns:
        D: n x n diagonal matrix
        U: n x k matrix such that A ≈ D + U @ U.T
    """
    n = A.shape[0]
    D = np.zeros((n, n))  # Initialize diagonal to zero

    for t in range(max_iters):
        # Low-rank step: eigendecomposition of residual
        R = A - D
        eigvals, eigvecs = np.linalg.eigh(R)

        # Sort eigenvalues/vectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top k eigenvectors, scaled by sqrt(max(λ, 0))
        U = eigvecs[:, :k] @ np.diag(np.sqrt(np.maximum(eigvals[:k], 0)))

        # Diagonal step: minimize ||A - D - UU^T||_F^2
        UUT_diag = np.sum(U * U, axis=1)  # Diagonal of U @ U.T
        D_new = np.diag(np.diag(A) - UUT_diag)

        # Check convergence
        if np.linalg.norm(D_new - D, 'fro') < tol:
            break
        D = D_new

    return D, U

# Key insight: The low-rank component is uniquely determined
# up to the ordering of eigenvectors (no rotational degeneracy)
# because we use spectral decomposition directly.
```

**Woodbury Inversion for SSMs:**

When using LRPD in state-space models, invert $(\bar{A} + \bar{B}\bar{C})$ efficiently:

$$
(\bar{A} + \bar{B}\bar{C})^{-1} = (D + UU^T + \bar{B}\bar{C})^{-1}
$$

Apply Woodbury twice or combine into single correction with rank-$(k + m)$ update.

## References

- Yeon, K., & Anitescu, M. (2025). Beyond Low Rank: Fast Low-Rank + Diagonal Decomposition with a Spectral Approach. *arXiv:2512.17120*.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*.

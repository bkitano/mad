# 074: Matrix Determinant Lemma

**Category**: decomposition
**Gain type**: efficiency
**Source**: Sylvester (1851); applied to normalizing flows by Rezende & Mohamed (2015)
**Paper**: [papers/matrix-determinant-lemma.pdf]
**Documented**: 2026-02-15

## Description

The matrix determinant lemma is the determinant counterpart to the Woodbury matrix identity. While Woodbury provides an efficient formula for the *inverse* of a low-rank perturbation, the matrix determinant lemma provides an efficient formula for the *determinant*. For an invertible matrix $A$ and a rank-$k$ update $UV^T$, the lemma expresses $\det(A + UV^T)$ in terms of $\det(A)$ and a small $k \times k$ determinant, avoiding a full $O(n^3)$ determinant computation. This is critical in normalizing flows and other generative models where the log-determinant of the Jacobian must be computed at every transformation step. The rank-1 special case reduces the Jacobian log-determinant to a single scalar computation, enabling $O(D)$ planar flows that would otherwise require $O(D^3)$ per step.

## Mathematical Form

**Core Operation (rank-1):**

$$
\det(A + uv^T) = (1 + v^T A^{-1} u) \det(A)
$$

**Core Operation (rank-k generalization):**

$$
\det(A + UV^T) = \det(I_k + V^T A^{-1} U) \det(A)
$$

where $U \in \mathbb{R}^{n \times k}$, $V \in \mathbb{R}^{n \times k}$, and $k \ll n$.

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — invertible base matrix
- $u, v \in \mathbb{R}^n$ — rank-1 perturbation vectors
- $U, V \in \mathbb{R}^{n \times k}$ — low-rank perturbation factors
- $I_k$ — $k \times k$ identity matrix

**Sylvester's Determinant Identity (the deeper generalization):**

For $A \in \mathbb{R}^{D \times M}$ and $B \in \mathbb{R}^{M \times D}$ with $M \leq D$:

$$
\det(I_D + AB) = \det(I_M + BA)
$$

This reduces computing the determinant of a $D \times D$ matrix to a $M \times M$ matrix.

**Application to Planar Flows:**

For the planar flow transformation $f(\mathbf{z}) = \mathbf{z} + \mathbf{u} h(\mathbf{w}^T \mathbf{z} + b)$:

$$
\det \frac{\partial f}{\partial \mathbf{z}} = \det(I + \mathbf{u} h'(\mathbf{w}^T \mathbf{z} + b) \mathbf{w}^T) = 1 + \mathbf{u}^T h'(\mathbf{w}^T \mathbf{z} + b) \mathbf{w}
$$

This is a scalar — computed in $O(D)$ time via dot product.

**Application to Sylvester Normalizing Flows:**

For $\mathbf{z}' = \mathbf{z} + \mathbf{QR}h(\tilde{\mathbf{R}}\mathbf{Q}^T\mathbf{z} + \mathbf{b})$ with orthogonal $\mathbf{Q}$ and upper-triangular $\mathbf{R}, \tilde{\mathbf{R}}$:

$$
\det \mathbf{J} = \det\left(I_M + \text{diag}\left(h'\left(\tilde{\mathbf{R}}\mathbf{Q}^T\mathbf{z} + \mathbf{b}\right)\right) \tilde{\mathbf{R}}\mathbf{R}\right)
$$

Since $\tilde{\mathbf{R}}\mathbf{R}$ is upper triangular, the determinant is the product of diagonal entries — computable in $O(M)$.

**Log-Determinant Form (for normalizing flows):**

$$
\ln q_K(\mathbf{z}_K) = \ln q_0(\mathbf{z}_0) - \sum_{k=1}^{K} \ln |1 + \mathbf{u}_k^T \psi_k(\mathbf{z}_{k-1})|
$$

where $\psi_k(\mathbf{z}) = h'(\mathbf{w}_k^T \mathbf{z} + b_k) \mathbf{w}_k$.

## Complexity

| Operation | Naive | With Lemma |
|-----------|-------|------------|
| Jacobian determinant ($D \times D$) | $O(D^3)$ | — |
| Rank-1 Jacobian determinant | $O(D^3)$ | $O(D)$ |
| Rank-$M$ Jacobian determinant | $O(D^3)$ | $O(M^2 D)$ or $O(M)$ if triangular |
| Log-det per flow step | $O(D^3)$ | $O(D)$ |
| $K$-step normalizing flow | $O(KD^3)$ | $O(KD)$ |

**Memory:** $O(D)$ per flow step vs $O(D^2)$ for full Jacobian storage

**Combined with Woodbury:** When both the inverse and determinant of $A + UV^T$ are needed (e.g., in Gaussian process updates), the matrix determinant lemma and Woodbury identity share the intermediate computation $V^T A^{-1} U$, so both can be obtained together with minimal overhead.

## Applicability

- **Planar normalizing flows** (Rezende & Mohamed, 2015): Rank-1 Jacobian determinants in $O(D)$ — the enabling trick that makes planar flows practical
- **Sylvester normalizing flows** (van den Berg et al., 2018): Rank-$M$ generalization using Sylvester's identity, removing the single-unit bottleneck
- **Variational autoencoders**: Computing the ELBO requires log-det Jacobian terms at each flow step
- **Gaussian process regression**: Incremental updates to the marginal likelihood when adding data points
- **State space models**: Determinant computation for DPLR matrices — $\det(zI - \Lambda + PQ^*) = \det(zI - \Lambda) \cdot \det(I + Q^* D_z P)$, complementing the Woodbury resolvent
- **Low-rank fine-tuning (LoRA)**: Monitoring effective rank and conditioning of adapted weight matrices

## Limitations

- Requires $A^{-1}$ to be available or cheaply computable — if $A^{-1}$ is expensive, the lemma alone doesn't help
- Numerical instability when $1 + v^T A^{-1} u \approx 0$ (near-singular perturbation), causing log-det to diverge
- Only applies to additive low-rank perturbations — multiplicative or nonlinear perturbations need different treatment
- For normalizing flows: invertibility of the planar flow requires the constraint $\mathbf{w}^T \mathbf{u} \geq -1$ (when $h = \tanh$), which limits expressivity per step
- The rank-$k$ generalization still requires $O(k^3)$ for the small determinant, so $k$ must be small

## Implementation Notes

```python
import torch

def log_det_rank1(u, w, z, b, h_prime):
    """Log-determinant for planar flow using matrix determinant lemma.

    f(z) = z + u * h(w^T z + b)
    det(df/dz) = |1 + u^T * h'(w^T z + b) * w|
    """
    # O(D) computation
    psi = h_prime(w @ z + b) * w   # (D,)
    log_det = torch.log(torch.abs(1 + u @ psi))  # scalar
    return log_det

def log_det_sylvester(Q, R, R_tilde, z, b, h_prime):
    """Log-determinant for Sylvester flow (rank-M).

    f(z) = z + Q @ R @ h(R_tilde @ Q^T @ z + b)
    det(J) = det(I_M + diag(h'(...)) @ R_tilde @ R)
    """
    # Q: (D, M), R, R_tilde: (M, M) upper triangular
    preact = R_tilde @ (Q.T @ z) + b      # (M,)
    h_diag = h_prime(preact)               # (M,)
    # R_tilde @ R is upper triangular, so det = product of diag
    RR = R_tilde @ R                        # (M, M)
    diag_vals = 1 + h_diag * torch.diag(RR) # (M,)
    log_det = torch.sum(torch.log(torch.abs(diag_vals)))  # O(M)
    return log_det

def log_det_lowrank(A_inv, U, V):
    """General rank-k matrix determinant lemma.

    det(A + U V^T) = det(I_k + V^T A^{-1} U) * det(A)
    Returns: log|det(I_k + V^T A^{-1} U)|
    (Assumes log|det(A)| is known/precomputed)
    """
    # U, V: (n, k), A_inv: (n, n) or a function
    k = U.shape[1]
    inner = torch.eye(k) + V.T @ A_inv @ U  # (k, k)
    log_det = torch.logdet(inner)
    return log_det
```

## References

- Sylvester, J. J. (1851). On the relation between the minor determinants of linearly equivalent quadratic functions. *Philosophical Magazine*.
- Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*. arXiv:1505.05770.
- van den Berg, R., Hasenclever, L., Tomczak, J. M., & Welling, M. (2018). Sylvester Normalizing Flows for Variational Inference. *UAI*. arXiv:1803.05649.
- Hager, W. W. (1989). Updating the Inverse of a Matrix. *SIAM Review*, 31(2), 221–239.
- Kobyzev, I., Prince, S. J., & Brubaker, M. A. (2020). Normalizing Flows: An Introduction and Review of Current Methods. *IEEE TPAMI*. arXiv:1908.09257.

# 125: Tangent Space Projection for Low-Rank Plus Diagonal Matrices

**Category**: algebraic
**Gain type**: efficiency
**Source**: Bonnabel, Lambert, & Bach (2024)
**Paper**: [papers/riccati-low-rank-diagonal.pdf]
**Documented**: 2026-02-15

## Description

A differential geometry approach for computing optimal low-rank plus diagonal approximations to matrix differential equations (MDEs) on the manifold of positive semidefinite matrices. Instead of directly factorizing $\Sigma \approx D + URU^T$ at each timestep, this method derives closed-form formulas for projecting the derivative $\dot{\Sigma}$ onto the tangent space of the LRPD manifold, enabling efficient integration with $O(d \cdot p)$ cost per timestep rather than $O(d^3)$.

Critical for Kalman filtering, Riccati equations, and neural ODEs where covariance matrices evolve continuously in high dimensions.

## Mathematical Form

**Manifold Structure:**

The set $\mathcal{S}_{\text{diag}}^+(p, d)$ of LRPD matrices forms a submanifold of $\mathcal{S}^+(p, d)$ (positive semidefinite matrices):

$$
Y = URU^T + \psi
$$

where:
- $U \in \mathbb{R}^{d \times p}$ has orthonormal columns ($U^T U = I_p$)
- $R \in \mathbb{R}^{p \times p}$ is symmetric
- $\psi \in \mathbb{R}^{d \times d}$ is diagonal with strictly positive entries

**Tangent Vector:**

An infinitesimal variation $\delta Y$ at $Y \in \mathcal{S}_{\text{diag}}^+(p, d)$ lies in the tangent space if:

$$
\delta Y = \delta U R U^T + U \delta R U^T + U R \delta U^T + \delta \psi
$$

subject to:
- $\delta U^T U = 0$ (orthonormality constraint)
- $\delta \psi$ diagonal

This simplifies to:

$$
\delta Y = \delta \tilde{U} R U^T + U \delta R U^T + U R \tilde{\delta U}^T + \delta \psi
$$

where $\delta \tilde{U}$ lies in the orthogonal complement of $\operatorname{span}(U)$.

**Optimal Projection (Factor Analysis Form):**

Given arbitrary symmetric matrix $H$ and current LRPD approximation $Y = URU^T + \psi$, the orthogonal projection $P_{Y}(H)$ onto the tangent space minimizes:

$$
\min_{\delta U, \delta R, \delta \psi} \|H - \delta Y\|_F^2
$$

subject to tangent space constraints (2.1). The solution is:

$$
P_{U,R,\psi}(H) = HUU^T + UU^T H - UU^T HUU^T + \delta\psi
$$

where the matrices are:

$$
\delta U = (I - UU^T) \tilde{H} U R^{-1}
$$

$$
\delta R = U^T \tilde{H} U
$$

$$
\delta\psi = \overline{\delta\psi} = ((I - UU^T)^{\odot 2})^+ \operatorname{diag}((I - UU^T)H(I - UU^T))
$$

where $\tilde{H} = H - \delta\psi$, $\odot$ denotes Hadamard (elementwise) product, and $^+$ denotes Moore-Penrose pseudoinverse.

**Key Insight:**

The diagonal component $\delta\psi$ is computed by solving a least-squares problem for the diagonal entries:

$$
\overline{\delta\psi} = ((I - UU^T)^{\odot 2})^+ \operatorname{diag}((I - UU^T)H(I - UU^T))
$$

This decouples the diagonal optimization from the low-rank component.

## Complexity

| Operation | Naive (full matrix) | Tangent projection |
|-----------|---------------------|-------------------|
| Matrix differential equation step | $O(d^3)$ | $O(d \cdot p^2)$ |
| Projection computation | — | $O(d \cdot p^2)$ |
| Storage | $O(d^2)$ | $O(d \cdot p)$ |
| Inversion (Woodbury) | $O(d^3)$ | $O(d \cdot p^2 + p^3)$ |

**Memory:** Linear in $d$ (for diagonal) + $O(d \cdot p)$ for low-rank factors vs $O(d^2)$ for full matrix

**Per-timestep cost:** $O(d \cdot p^2)$ vs $O(d^3)$ for full covariance propagation in Kalman filter

## Applicability

- **Kalman filtering:** High-dimensional state estimation where covariance $P(t)$ evolves via Riccati equation
  $$\dot{P} = FP + PF^T + Q - PC^TR^{-1}CP$$
  Project $\dot{P}$ onto LRPD tangent space at each integration step

- **Neural ODEs with uncertainty:** Propagating low-rank plus diagonal covariance through continuous dynamics

- **Gaussian variational inference:** Approximating posterior covariances in high dimensions (e.g., for Wasserstein gradient flows)

- **Riccati equations:** Continuous-time optimal control with LRPD approximation of cost-to-go matrix

- **Covariance estimation:** Streaming updates to covariance matrices with LRPD constraint

## Limitations

- Requires the target matrix to be well-approximated by LRPD structure throughout evolution
- Rank $p$ must be chosen a priori; adaptive rank selection requires additional logic
- Projection is local; long-term integration may drift from optimal LRPD manifold without periodic reinitialization
- For rank-deficient or nearly rank-deficient $R$, numerical stability requires careful handling of Moore-Penrose inverse
- Does not apply directly to indefinite matrices (requires extension to signed LRPD)

## Implementation Notes

```python
def tangent_projection_FA(H, U, R, psi):
    """
    Orthogonal projection of symmetric matrix H onto tangent space
    of LRPD manifold at Y = U @ R @ U.T + diag(psi)

    Factor Analysis (FA) parameterization

    Args:
        H: d x d symmetric matrix (e.g., derivative of covariance)
        U: d x p matrix with orthonormal columns
        R: p x p symmetric matrix
        psi: d-vector of diagonal entries (all positive)

    Returns:
        delta_U, delta_R, delta_psi defining tangent vector
    """
    d, p = U.shape
    I_minus_UUT = np.eye(d) - U @ U.T

    # Solve for delta_psi via least-squares (equation 3.15)
    # delta_psi encodes diagonal corrections
    H_perp = I_minus_UUT @ H @ I_minus_UUT
    diag_H_perp = np.diag(H_perp)

    # (I - UU^T)^{⊙2} is elementwise square of projection matrix
    # For diagonal entries: ((I - UU^T)_{ii})^2
    diag_I_minus_UUT_sq = np.diag(I_minus_UUT)**2
    delta_psi_bar = diag_H_perp / diag_I_minus_UUT_sq  # Element-wise division

    # Correct H by removing diagonal contribution
    H_tilde = H - np.diag(delta_psi_bar)

    # Low-rank updates (equations 3.11-3.12)
    delta_U = I_minus_UUT @ H_tilde @ U @ np.linalg.inv(R)
    delta_R = U.T @ H_tilde @ U

    return delta_U, delta_R, delta_psi_bar

def integrate_lrpd_ode(F, Q, U0, R0, psi0, T, dt):
    """
    Integrate matrix ODE: dY/dt = F(Y) using LRPD approximation

    Example: Lyapunov equation dY/dt = A @ Y + Y @ A.T + Q

    Args:
        F: function mapping (U, R, psi, t) -> symmetric d x d matrix (derivative)
        Q: d x d process noise covariance
        U0, R0, psi0: Initial LRPD approximation
        T: final time
        dt: timestep

    Returns:
        U, R, psi at time T
    """
    U, R, psi = U0, R0, psi0
    t = 0.0

    while t < T:
        # Compute derivative at current Y
        Y = U @ R @ U.T + np.diag(psi)
        dY = F(U, R, psi, t)

        # Project derivative onto tangent space
        delta_U, delta_R, delta_psi = tangent_projection_FA(dY, U, R, psi)

        # Euler step (use RK4 for better accuracy)
        U_new = U + dt * delta_U
        R_new = R + dt * delta_R
        psi_new = psi + dt * delta_psi

        # Re-orthonormalize U to maintain constraint
        U_new, _ = np.linalg.qr(U_new)

        U, R, psi = U_new, R_new, psi_new
        t += dt

    return U, R, psi

# Example: Continuous-time Kalman filter covariance propagation
def kalman_lrpd_step(U, R, psi, A, Q, dt):
    """
    One step of Kalman filter covariance propagation with LRPD approximation

    dP/dt = A @ P + P @ A.T + Q
    """
    def F(U, R, psi, t):
        P = U @ R @ U.T + np.diag(psi)
        return A @ P + P @ A.T + Q

    return integrate_lrpd_ode(F, Q, U, R, psi, dt, dt/10)
```

**Woodbury Inversion:**

For LRPD matrices $Y = URU^T + \psi$:

$$
Y^{-1} = \psi^{-1} - \psi^{-1}U(R^{-1} + U^T\psi^{-1}U)^{-1}U^T\psi^{-1}
$$

Cost: $O(d \cdot p^2 + p^3)$ vs $O(d^3)$ for direct inversion.

## References

- Bonnabel, S., Lambert, M., & Bach, F. (2024). Low-Rank Plus Diagonal Approximations for Riccati-Like Matrix Differential Equations. *SIAM Journal on Matrix Analysis and Applications, 45*(3), 1669-1688.
- Archambeau, C., Cornford, D., Opper, M., & Shawe-Taylor, J. (2007). Gaussian process approximations of stochastic differential equations. *JMLR*.
- Bucy, R. S., & Joseph, P. D. (1968). *Filtering for Stochastic Processes with Applications to Guidance*. Wiley.

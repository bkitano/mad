# 185: Differentiable Compact WY Backpropagation

**Category**: decomposition
**Gain type**: flexibility
**Source**: Papanicolopulos (2024), "Derivatives of the full QR factorisation and of the compact WY representation"
**Paper**: [papers/derivatives-compact-wy.pdf]
**Documented**: 2026-02-15

## Description

Current automatic differentiation frameworks (PyTorch, JAX) support gradients for the **thin** QR factorization ($A = QR$ where $Q$ is tall-skinny) but **cannot** differentiate through the **full** QR factorization (where $Q$ is square). This paper derives closed-form expressions for the derivative of the compact WY representation $(Y, T)$ directly, enabling backpropagation through models that parameterize layers via Householder products stored in compact WY form.

The key insight is that when $Q$ is stored as $Q = I - YTY^\top$ (compact WY), one can compute $\partial Y$ and $\partial T$ directly from $\partial Q$ without ever materializing the full $Q$ matrix. This is more efficient than differentiating through the full $Q$ and critical for models like DeltaNet, DeltaProduct, and orthogonal RNNs that use WY-form state transitions during training.

The derivative formulas decompose into structured triangular operations (Hadamard products with upper/lower triangular masks, triangular solves) that map to efficient BLAS routines.

## Mathematical Form

**Setup:**

Given a tall matrix $A_{mn} \in \mathbb{R}^{m \times n}$ with $m \geq n$, the full QR factorization is:

$$
A_{mn} = Q_{mm} R_{mn}
$$

where $Q_{mm}$ is orthogonal and $R_{mn}$ is upper triangular. The compact WY representation stores $Q_{mm}$ as:

$$
Q_{mm} = I_{mm} - Y_{mn} T_{nn} Y_{mn}^\top
$$

where $Y_{mn} \in \mathbb{R}^{m \times n}$ is unit lower triangular and $T_{nn} \in \mathbb{R}^{n \times n}$ is upper triangular. Define $S_{nn} = T_{nn} Y_{nn}^\top$ (the product appearing in the CWY form).

**Derivative of the Thin QR (known result, for context):**

Given $\partial A_{mn}$ (the perturbation of $A$), define:

$$
B_{mn} = \partial A_{mn} R_{nn}^{-1}
$$

$$
E_{nn} = Q_{mn}^\top B_{mn} \quad \text{(project perturbation)}
$$

The skew-symmetric and upper-triangular decomposition:

$$
\Omega_{nn} = (\hat{L} \circ E_{nn}) - (\hat{L} \circ E_{nn})^\top
$$

$$
\Psi_{nn} = (U \circ E_{nn}) + (\hat{L} \circ E_{nn})^\top
$$

where $\hat{L}$ is the strictly lower triangular mask (ones below diagonal, zero elsewhere) and $U$ is the upper triangular mask including diagonal. Then:

$$
\partial R_{nn} = \Psi_{nn} R_{nn}
$$

$$
\partial Q_{mn} = Q_{mm} \Omega_{mn} \quad \text{where } \Omega_{pn} = E_{pn}
$$

**Derivative of the Compact WY Representation (NEW — main contribution):**

Starting from $Q_{nn} = I_{nn} - Y_{nn} S_{nn}$ where $S_{nn} = T_{nn} Y_{nn}^\top$, define the auxiliary matrix:

$$
C_{nn} = Y_{nn}^{-1} \partial Q_{nn} S_{nn}^{-1}
$$

which decomposes into:

$$
C_{nn} = \underbrace{Y_{nn}^{-1}(B_{nn} - \Psi_{nn}) S_{nn}^{-1}}_{C_{nn}^*} + S_{nn} \Psi_{nn} S_{nn}^{-1}
$$

**The derivative of $Y_{nn}$:**

$$
\partial Y_{nn} = -Y_{nn} (\hat{L} \circ C_{nn})
$$

This extracts only the strictly lower triangular part of $C_{nn}$ (since $Y_{nn}$ is unit lower triangular, its diagonal is fixed at 1 and not differentiable).

**The derivative of $T_{nn}$:**

$$
\partial T_{nn} = T_{nn} (\hat{L} \circ C_{nn})^\top - (U \circ C_{nn}) T_{nn} - S_{nn} \Psi_{nn} Y_{nn}^{-\top}
$$

**The derivative of the $Q_{mn}$ block (extending to full factorization):**

For the bottom $(m-n) \times n$ block $Y_{pn}$ (the non-trivial part of the tall $Y$ matrix):

$$
\partial Y_{pn} = -(\partial Q_{mn}) S_{nn}^{-1} + Y_{pn} (U \circ C_{nn}^*)
$$

$$
\partial Y_{mn} = -(\partial Q_{mn}) S_{nn}^{-1} - B_{pn} S_{nn}^{-1} + Y_{pn} (U \circ C_{nn}^*)
$$

**The derivative of the full $Q_{mm}$ factor** (combining thin QR + CWY derivatives):

$$
\partial Q_{mp} = (\partial Q_{mn})(Y_{pn} Y_{nn}^{-1})^\top + Y_{mn} T_{nn} S_{nn}^{-\top} (\partial Q_{pn} - Y_{pn} Y_{nn}^{-1} \partial Q_{nn})^\top
$$

**Key Definitions:**

- $A_{mn} \in \mathbb{R}^{m \times n}$ — input matrix ($m \geq n$)
- $Y_{mn} \in \mathbb{R}^{m \times n}$ — unit lower triangular Householder vector matrix
- $T_{nn} \in \mathbb{R}^{n \times n}$ — upper triangular T-factor
- $S_{nn} = T_{nn} Y_{nn}^\top \in \mathbb{R}^{n \times n}$ — combined factor
- $\hat{L}$ — strictly lower triangular mask (ones below diagonal)
- $U$ — upper triangular mask including diagonal
- $\circ$ — Hadamard (elementwise) product

## Complexity

| Operation | Full Q Materialization | Compact WY Derivative |
|-----------|----------------------|----------------------|
| Compute $\partial Q$ | $O(m^2 n)$ (materialize $Q_{mm}$) | $O(mn^2 + n^3)$ (work with $Y, T$ directly) |
| Memory for $Q$ | $O(m^2)$ | $O(mn + n^2)$ |
| Backward pass | $O(m^2 n)$ | $O(mn^2 + n^3)$ |

**Key savings:** When $m \gg n$ (tall matrices), the compact WY derivative avoids the $O(m^2)$ memory and $O(m^2 n)$ compute of materializing the full $Q$ matrix. The derivative operates directly on the $(Y, T)$ representation.

**Memory:** $O(mn)$ for $Y$ + $O(n^2)$ for $T, S, C$ auxiliary matrices. No $O(m^2)$ allocation needed.

## Applicability

- **DeltaNet / DeltaProduct training**: These models store state transitions as compact WY products. Backpropagating through the WY representation directly (rather than materializing $Q$) saves memory proportional to $d^2$ per layer per chunk, where $d$ is the head dimension.
- **Orthogonal RNNs with CWY parameterization**: Models using the CWY form (trick 152) can now compute exact gradients through the $(Y, T)$ parameterization without materializing the full orthogonal matrix.
- **Differentiable QR in optimization**: Variable projection methods for separable nonlinear least squares require $\partial Q / \partial \alpha$ — the compact WY derivative provides this efficiently.
- **PyTorch/JAX custom autograd**: The formulas can be implemented as custom backward passes for `torch.linalg.qr` with `mode='complete'`, extending AD frameworks to support full QR.
- **Neural architecture search**: Architectures that search over orthogonal structures (e.g., learnable Householder layers in normalizing flows) benefit from efficient backpropagation through the QR parameterization.

## Limitations

- **Assumes Householder-based QR**: The derivative formulas are specific to QR factorizations computed via Householder reflections (which produce the compact WY form). QR via Givens rotations produces a different $Q$ factor that cannot be represented in compact WY form, and these formulas do not apply.
- **Requires $T_{nn}$ to be invertible**: The diagonal entries of $T_{nn}$ are the Householder $\tau$ coefficients. The paper assumes $\tau \neq 0$ for all reflections. If any $\tau_i = 0$ (degenerate reflection), $T$ becomes singular and the derivatives are undefined.
- **$O(n^3)$ triangular operations**: While cheaper than $O(m^2 n)$ materialization, the $O(n^3)$ triangular solves ($S_{nn}^{-1}$, $Y_{nn}^{-1}$) are sequential and may not map well to tensor cores for small $n$.
- **Not yet implemented in major frameworks**: As of 2024, PyTorch and JAX do not include these formulas. A custom autograd function is required.
- **Numerical stability**: The inversion $Y_{nn}^{-1}$ (unit lower triangular) and $S_{nn}^{-1}$ can amplify errors in low precision (BF16/FP16). Mixed-precision strategies may be needed for the backward pass.
- **Performance not benchmarked on GPU**: The paper provides mathematical derivations but does not include wall-clock GPU benchmarks. Actual speedup depends on the ratio $m/n$ and kernel implementation quality.

## Implementation Notes

```python
import torch

def compact_wy_backward(dQ_mn, Y_mn, T_nn, R_nn, Q_mn, A_mn):
    """
    Backward pass through the compact WY representation.

    Given upstream gradient dQ_mn (gradient of loss w.r.t. Q_mn),
    compute gradients w.r.t. A_mn (the input matrix).

    Args:
        dQ_mn: (m, n) gradient w.r.t. Q factor (thin part)
        Y_mn: (m, n) unit lower triangular Householder vectors
        T_nn: (n, n) upper triangular T-factor
        R_nn: (n, n) upper triangular R-factor
        Q_mn: (m, n) thin Q factor
        A_mn: (m, n) original input matrix

    Returns:
        dA_mn: (m, n) gradient w.r.t. input matrix
    """
    m, n = Y_mn.shape

    # Extract blocks
    Y_nn = Y_mn[:n, :n]   # (n, n) unit lower triangular
    Y_pn = Y_mn[n:, :n]   # (p, n) where p = m - n

    # S = T @ Y_nn^T
    S_nn = T_nn @ Y_nn.T   # (n, n)

    # Step 1: Compute B = dA @ R^{-1}
    B_mn = torch.linalg.solve_triangular(R_nn, dQ_mn.T, upper=True).T

    # Step 2: E = Q^T @ B (project into Q's column space)
    E_nn = Q_mn.T @ B_mn   # (n, n)

    # Step 3: Skew-symmetric / upper-triangular decomposition
    L_mask = torch.tril(torch.ones(n, n), diagonal=-1)  # strict lower
    U_mask = torch.triu(torch.ones(n, n))                # upper + diag

    Omega_nn = (L_mask * E_nn) - (L_mask * E_nn).T
    Psi_nn = (U_mask * E_nn) + (L_mask * E_nn).T

    # Step 4: Derivative of compact WY
    # C = Y^{-1} @ dQ @ S^{-1}
    S_inv = torch.linalg.solve_triangular(S_nn, torch.eye(n), upper=False)
    Y_inv = torch.linalg.solve_triangular(Y_nn, torch.eye(n), upper=False)

    C_star = Y_inv @ (B_mn[:n] - Psi_nn) @ S_inv
    C_nn = C_star + S_nn @ Psi_nn @ S_inv

    # Step 5: Derivatives of Y and T
    dY_nn = -Y_nn @ (L_mask * C_nn)
    dT_nn = (T_nn @ (L_mask * C_nn).T
             - (U_mask * C_nn) @ T_nn
             - S_nn @ Psi_nn @ torch.linalg.inv(Y_nn.T))

    # Step 6: dR and dQ -> dA
    dR_nn = Psi_nn @ R_nn
    # dA = dQ @ R + Q @ dR (product rule)
    # ... (chain rule back to A)

    return dA_mn


# Key insight for neural network training:
# When the model stores state as (Y, T) in compact WY form,
# the backward pass computes gradients through (Y, T) directly.
# This avoids materializing the m x m orthogonal matrix Q,
# saving O(m^2) memory per layer.
#
# For DeltaNet with head_dim=d and chunk_size=C:
#   m = C (chunk size, ~64-256)
#   n = d (head dimension, ~64-256)
#   Savings: O(C^2) memory -> O(Cd + d^2) memory per chunk
```

**Connection to existing tricks:**

- **Trick 145 (WY Representation)**: Provides the forward pass — this trick provides the backward pass.
- **Trick 139 (UT Transform)**: Computes the T-factor efficiently for the forward pass. The derivative formulas here operate on the (Y, T) representation that the UT transform produces.
- **Trick 152 (CWY Parameterization)**: Uses $S = \frac{1}{2}I + \text{striu}(U^\top U)$ for orthogonal RNNs. The derivatives here generalize to the $\beta \neq 2$ (non-reflection) case used in DeltaNet.
- **Trick 157 (Neumann CWY Inverse)**: Approximates $S^{-1}$ in the forward pass. The backward pass formulas here could be combined with Neumann-approximated $S^{-1}$ for a fully approximate (but fast) gradient computation.

## References

- Papanicolopulos, S.-A. (2024). Derivatives of the full QR factorisation and of the factored-form and compact WY representations. arXiv:2409.13374.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. SIAM J. Sci. Stat. Comput., 10(1), 53–57.
- Roberts, D. A. O., Roberts, L. R. (2020). QR and LQ decomposition matrix backpropagation algorithms. arXiv:2009.10071.
- Walter, S. F., Lehmann, L., Lamour, R. (2012). On evaluating higher-order derivatives of the QR decomposition of tall matrices. Optimization Methods and Software, 27, 391–403.
- Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations. 4th ed., Johns Hopkins University Press.
- Elmroth, E. & Gustavson, F. G. (2000). Applying recursion to serial and parallel QR factorization leads to better performance. IBM J. Res. Dev., 44(4), 605–624.

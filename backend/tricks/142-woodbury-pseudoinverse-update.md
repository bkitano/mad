# 142: Woodbury Pseudoinverse Update

**Category**: decomposition
**Gain type**: efficiency
**Source**: Güttel, Nakatsukasa, Webb, Riley (2024); extends Sherman-Morrison-Woodbury to rectangular pseudoinverses
**Paper**: [papers/sherman-morrison-least-squares-update.pdf] (local path to downloaded PDF)
**Documented**: 2026-02-15

## Description

The classical Woodbury identity efficiently updates the inverse of a square matrix under low-rank perturbations, but does not directly apply to rectangular matrices or least-squares problems. This trick extends the Woodbury formula to the Moore-Penrose pseudoinverse: given a full-rank rectangular matrix $A \in \mathbb{R}^{m \times n}$ ($m \geq n$) and a rank-$r$ update $UV^T$, it provides an explicit formula for $(A + UV^T)^\dagger$ in terms of $A^\dagger$ and a small $2r \times 2r$ correction. This enables solving updated least-squares problems $\min_x \|b - (A + UV^T)x\|_2$ in $O((r + k)mn)$ time instead of $O(mn^2)$ from scratch --- a speedup of $O(n/r)$. This is directly relevant to neural network training where low-rank weight updates (LoRA, delta rule updates, fast weight modifications) require efficient re-solving of least-squares subproblems.

## Mathematical Form

**Core Operation:**

For $A \in \mathbb{R}^{m \times n}$, $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$ with $\text{rank}(A) = \text{rank}(A + UV^T) = n$:

$$
(A + UV^T)^\dagger = A^\dagger - MA^\dagger + (I - M)(A^TA)^{-1}VU^T
$$

where the correction matrix $M$ is:

$$
M = (A^TA)^{-1}X(I + Y^T(A^TA)^{-1}X)^{-1}Y^T
$$

**Key Definitions:**

- $A \in \mathbb{R}^{m \times n}$ --- original full-rank tall matrix ($m \geq n$)
- $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$ --- low-rank update factors ($r \ll n$)
- $A^\dagger = (A^TA)^{-1}A^T$ --- Moore-Penrose pseudoinverse of $A$
- $X = [V, A^TU] \in \mathbb{R}^{n \times 2r}$ --- augmented factor
- $Y = [(A + UV^T)^TU, V] \in \mathbb{R}^{n \times 2r}$ --- augmented factor
- $(I + Y^T(A^TA)^{-1}X)^{-1}$ --- a $2r \times 2r$ capacitance matrix inverse

**Solving Updated Least Squares:**

To compute $\hat{A}^\dagger b = (A + UV^T)^\dagger b$ directly:

$$
\hat{A}^\dagger b = (I - M)\left(A^\dagger b - (A^TA)^{-1}VU^Tb\right)
$$

**Derivation:**

Starting from $\hat{A} = A + UV^T$:

$$
\hat{A}^T\hat{A} = A^TA + VU^TA + A^TUV^T + VU^TUVV^T = A^TA + XY^T
$$

Apply the standard Woodbury identity to $(\hat{A}^T\hat{A})^{-1}$:

$$
(\hat{A}^T\hat{A})^{-1} = (A^TA)^{-1} - (A^TA)^{-1}X(I + Y^T(A^TA)^{-1}X)^{-1}Y^T(A^TA)^{-1} = (I - M)(A^TA)^{-1}
$$

Then $\hat{A}^\dagger = (\hat{A}^T\hat{A})^{-1}\hat{A}^T = (I - M)(A^TA)^{-1}(A + UV^T)^T = (I - M)A^\dagger + (I - M)(A^TA)^{-1}VU^T$.

## Algorithm

**WoodburyLS** (Algorithm 3.1 from the paper):

Given: $A$, $b$, $U$, $V$, precomputed $x_0 = A^\dagger b$ and solver for $(A^TA)^{-1}c$.

1. Compute $x_0 = A^\dagger b$ (if not already available)
2. Set $X = [V, A^TU]$ and $Y = [(A + UV^T)^TU, V]$
3. Compute $Z = (A^TA)^{-1}X$ (via $2r$ triangular solves if QR of $A$ is available)
4. Set $w = x_0 + Z_1 U^T b$ (where $Z_1$ is the first $r$ columns of $Z$)
5. Compute $\hat{w} = Z(I_{2r} + Y^TZ)^{-1}Y^T w$ (a $2r \times 2r$ system)
6. $x = w - \hat{w}$ is the solution

## Complexity

| Operation | From Scratch (QR) | With WoodburyLS |
|-----------|-------------------|-----------------|
| Solve $\min \|b - \hat{A}x\|_2$ | $O(mn^2)$ | $O((r+k)mn)$ |
| Per additional RHS | $O(mn)$ | $O(mn)$ |
| Core bottleneck | QR of $\hat{A}$ | $2r$ solves with $A^TA$ |

**Speedup factor:** $O(n/r)$ when $A$'s QR factorization is precomputed

**Measured speedups:** 20x--130x in experiments with $m = 10^5$, $n = 100$--$1000$, $r = 10$--$30$

**Memory:** Reuses precomputed $A^\dagger$ or QR factors; additional $O(nr)$ for $X, Y, Z$

## Applicability

- **Low-rank weight updates (LoRA)**: When fine-tuning a pretrained weight matrix $W$ with a low-rank adapter $\Delta W = UV^T$, solving downstream least-squares problems with $W + UV^T$ can reuse precomputed factorizations of $W$
- **DeltaNet / fast weight programmers**: The delta rule applies rank-1 updates $S_t = S_{t-1} + \beta_t(v_t - S_{t-1}k_t)k_t^T$ at each step; WoodburyLS enables efficient re-solving of associated regression subproblems without refactoring from scratch
- **Recursive least squares (RLS)**: Online learning with streaming rank-1 updates to the design matrix; the pseudoinverse update formula generalizes the classical RLS covariance update to non-square settings
- **Gauss-Newton deflation**: Finding multiple local minima of nonlinear least squares problems, where each deflation step adds a rank-1 modification to the Jacobian
- **Structured SSM training**: When SSM state matrices undergo low-rank corrections during training (e.g., DPLR adjustments), the resolvent-based transfer function can be re-evaluated using cached factorizations

## Limitations

- Requires $A$ and $A + UV^T$ to both be full rank; rank-deficient cases need separate treatment
- The update to $A^\dagger$ is generically rank $2r$ (not rank $r$ as in the square Woodbury case), since the pseudoinverse involves $(A^TA)^{-1}$ which sees a rank-$2r$ perturbation
- For very large $r$ (approaching $n$), the $2r \times 2r$ capacitance matrix solve and the $2r$ back-solves become expensive, negating the advantage
- Numerical stability depends on the condition number of $A^TA$; ill-conditioned $A$ may require iterative refinement
- Does not directly handle the case where $A + UV^T$ becomes rank-deficient (the formula assumes full rank is preserved)

## Implementation Notes

```python
import numpy as np
from scipy.linalg import solve_triangular, qr

def woodbury_ls(A, b, U, V, x0=None, R=None):
    """
    Solve min ||b - (A + U @ V.T) @ x||_2
    using precomputed solution x0 = A^+ b and
    optionally the R factor from A = Q @ R.

    A: (m, n), U: (m, r), V: (n, r)
    Returns: x, solution to the updated LS problem
    """
    m, n = A.shape
    r = U.shape[1]

    # Step 1: base solution if not provided
    if x0 is None:
        Q, R = qr(A, mode='economic')
        x0 = solve_triangular(R, Q.T @ b)

    # Step 2: form X and Y
    X = np.column_stack([V, A.T @ U])                    # (n, 2r)
    Ahat_T_U = (A + U @ V.T).T @ U                       # (n, r)
    Y = np.column_stack([Ahat_T_U, V])                    # (n, 2r)

    # Step 3: solve (A^T A)^{-1} X via triangular solves
    # Z = R^{-1} R^{-T} X
    Z = solve_triangular(R, solve_triangular(R, X, trans='T'), trans='N')

    # Step 4: form w
    Z1 = Z[:, :r]  # first r columns
    w = x0 + Z1 @ (U.T @ b)

    # Step 5: small 2r x 2r system
    cap = np.eye(2*r) + Y.T @ Z      # (2r, 2r)
    w_hat = Z @ np.linalg.solve(cap, Y.T @ w)

    # Step 6: solution
    return w - w_hat
```

## References

- Güttel, S., Nakatsukasa, Y., Webb, M., Riley, A. B. (2024). A Sherman-Morrison-Woodbury Approach to Solving Least Squares Problems with Low-Rank Updates. arXiv:2406.15120.
- Meyer, C. D. (1973). Generalized Inversion of Modified Matrices. SIAM J. Appl. Math., 24, 315-323.
- Sherman, J. and Morrison, W. J. (1950). Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix. Ann. Math. Stat., 21, 124-127.
- Woodbury, M. A. (1950). Inverting Modified Matrices. Stat. Res. Group Memo Report, Princeton.
- Hager, W. W. (1989). Updating the Inverse of a Matrix. SIAM Review, 31, 221-239.
- Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.

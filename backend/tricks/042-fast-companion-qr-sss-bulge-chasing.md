# 042: Fast Companion QR via SSS Structured Bulge Chasing

**Category**: decomposition
**Gain type**: efficiency
**Source**: Chandrasekaran, Gu, Xia, Zhu — "A Fast QR Algorithm for Companion Matrices" (2007)
**Paper**: [papers/fast-companion-qr-hss.pdf]
**Documented**: 2026-02-15

## Description

This trick exploits the fact that Hessenberg iterates of a companion matrix under QR iteration preserve low-rank off-diagonal structure (rank $\leq 3$) to design an $O(n^2)$ eigenvalue algorithm using only $O(n)$ storage. The key insight is that instead of working on the Hessenberg iterate $H$ directly, the algorithm operates on the compact SSS (Sequentially Semi-Separable) representations of its QR factors $Q$ and $R$. The orthogonal factor $Q = Q_1 Q_2 \cdots Q_{n-1}$ is a product of Givens rotations (each parameterized by a single angle), and $R$ has off-diagonal rank $\leq 2$, admitting a compact SSS representation with scalar or $O(1)$-sized generators. Structured bulge chasing is performed on $R$'s SSS form using Givens rotation swaps, achieving $O(n)$ work per bulge chasing pass instead of $O(n^2)$.

## Mathematical Form

**Companion Matrix:**

$$
C = \begin{pmatrix} a_1 & a_2 & \cdots & a_{n-1} & a_n \\ 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{pmatrix} \in \mathbb{R}^{n \times n}
$$

The eigenvalues of $C$ are the roots of $p(x) = x^n - a_1 x^{n-1} - \cdots - a_{n-1} x - a_n$.

**Orthogonal-Plus-Rank-One Structure:**

$$
C = Z^{(0)} + e_1 y^T
$$

where $Z^{(0)}$ is orthogonal (a permutation-like matrix with $\det(Z^{(0)}) = 1$) and $e_1 = (1, 0, \ldots, 0)^T$.

**Key Structural Invariant (Theorem 3.1):** For all Hessenberg iterates $H^{(k)}$ under QR iteration:

$$
\max_{1 \leq j < n} \operatorname{rank}(H^{(k)}(1:j, j+1:n)) \leq 3
$$

This follows from the decomposition $H^{(k)} = Z^{(k)} + x^{(k)} y^{(k)T}$ where $Z^{(k)}$ is orthogonal (rank-symmetric, so off-diagonal blocks have equal rank) and $\operatorname{rank}(L) = 1$ contributes at most $+2$ to each off-diagonal block rank.

**SSS Representation of QR Factors:**

The QR factorization $C = Q^{(0)} \cdot R^{(0)}$ is computed via Givens rotations:

$$
Q^{(0)} = Q_1 Q_2 \cdots Q_{n-1}, \quad Q_k = \operatorname{diag}\left(I_{k-1}, \begin{pmatrix} c_k & s_k \\ -s_k & c_k \end{pmatrix}, I_{n-k-1}\right)
$$

with $c_k^2 + s_k^2 = 1$. The SSS generators of $Q$ are scalars:

| Generator | $\mathcal{D}_i(Q)$ | $\mathcal{U}_i(Q)$ | $\mathcal{V}_i(Q)$ | $\mathcal{W}_i(Q)$ | $\mathcal{P}_i(Q)$ | $\mathcal{Q}_i(Q)$ | $\mathcal{R}_i(Q)$ |
|-----------|-----|-----|-----|-----|-----|-----|-----|
| Value | $c_{i-1}c_i$ | $c_{i-1}s_i$ | $c_i$ | $s_i$ | $1$ | $-s_i$ | $0$ |

For $R^{(0)}$, with partitioning $\{m_i = 1\}_{i=1}^n$:

$$
R_{ij}^{(0)} = \begin{cases} c_i s_{i-1} \cdots s_1 a_i - s_i & \text{if } i = j \\ c_i s_{i-1} \cdots s_1 a_j & \text{if } i < j \\ 0 & \text{if } i > j \end{cases}
$$

The SSS generators of $R$ are also $O(1)$-sized, with initial common column dimension $p = 1$.

**Structured Single Shift QR Iteration:**

Instead of forming $\hat{H} = Q^T H Q$ explicitly, the algorithm works on the QR factors:

$$
H - \sigma I = QR, \quad \hat{H} = RQ + \sigma I = Q^T H Q
$$

The bulge chasing operates on $R$'s subdiagonal. For the $i$-th step, a bulge $b_i$ is created at position $(i+1, i)$ by right-multiplying by a Givens rotation $\bar{G}_j$:

$$
\begin{pmatrix} \hat{d}_i & \hat{h}_i \\ b_i & \hat{d}_{i+1} \end{pmatrix} = \begin{pmatrix} d_i & h_i \\ 0 & d_{i+1} \end{pmatrix} \begin{pmatrix} c_i & -s_i \\ s_i & c_i \end{pmatrix}
$$

Then a new Givens rotation $\tilde{G}_j$ eliminates the bulge:

$$
\begin{pmatrix} \tilde{d}_i & \tilde{h}_i \\ 0 & \tilde{d}_{i+1} \end{pmatrix} = \begin{pmatrix} \tilde{c}_i & -\tilde{s}_i \\ \tilde{s}_i & \tilde{c}_i \end{pmatrix} \begin{pmatrix} \hat{d}_i & \hat{h}_i \\ b_i & \hat{d}_{i+1} \end{pmatrix}
$$

**Givens Swap (Key Subroutine):** When adjacent Givens rotations $Q_i Q_{i+1} G_i$ need reordering, the algorithm computes $\hat{G}_2, \hat{Q}_1, \hat{Q}_2$ such that:

$$
Q_1 Q_2 G_1 = \hat{G}_2 \hat{Q}_1 \hat{Q}_2
$$

This "swap" costs $O(1)$ operations and enables the bulge to be chased along the subdiagonal of $R$ without forming $R$ explicitly.

**SSS Generator Updates During Chasing:**

After one complete bulge chasing pass, the updated $R$ factor is:

$$
R_4 = \tilde{G}^T \cdot R_0 \cdot \bar{G}
$$

where $\bar{G} = \bar{G}_1 \bar{G}_2 \cdots \bar{G}_{n-1}$ and $\tilde{G} = \tilde{G}_1 \tilde{G}_2 \cdots \tilde{G}_{n-1}$. The SSS generators $\{d_i, u_i, v_i, w_i\}$ of $R$ are updated locally during each chasing step in $O(1)$ operations. At the end of the pass, a structure recovery step compresses the SSS generators back to their compact form via SSS matrix-matrix multiplication formulas.

**Double Shift Version:**

For real companion matrices with complex conjugate eigenvalue pairs, double shift QR iteration uses shifts $\sigma, \bar{\sigma}$ simultaneously:

$$
M = (H^2 - sH + tI), \quad s = 2\operatorname{Re}(\sigma), \; t = |\sigma|^2
$$

A $2 \times 2$ bulge is chased along $R$'s subdiagonal, requiring two Givens rotations per elimination step. The SSS generator column dimensions grow by $4$ per pass (vs. $2$ for single shift), then are compressed back.

**Key Definitions:**

- $C \in \mathbb{R}^{n \times n}$ — companion matrix whose eigenvalues are polynomial roots
- $Q = Q_1 \cdots Q_{n-1}$ — orthogonal factor as product of Givens rotations
- $R$ — upper triangular factor with off-diagonal rank $\leq 2$
- $p$ — SSS common column dimension ($p = 1$ initially, grows to $\leq 3$ during iterations)
- $\{c_i, s_i\}$ — Givens rotation parameters ($c_i^2 + s_i^2 = 1$)
- $\{\mathcal{D}_i, \mathcal{U}_i, \mathcal{V}_i, \mathcal{W}_i, \mathcal{P}_i, \mathcal{Q}_i, \mathcal{R}_i\}$ — SSS generators

## Complexity

| Operation | Standard QR | Fast Companion QR |
|-----------|-------------|-------------------|
| One QR iteration | $O(n^2)$ | $O(n)$ |
| Full eigenvalue computation | $O(n^3)$ | $O(n^2)$ |
| Storage | $O(n^2)$ | $O(n)$ |
| Givens swap | $O(n)$ | $O(1)$ |

**Memory:** $O(n)$ — only SSS generators (scalars and $O(1)$-sized matrices) are stored, not the full $n \times n$ matrices.

**Per-iteration cost:** $O(n)$ flops for a single bulge chasing pass with $O(n)$ Givens swaps each costing $O(1)$.

**Total cost:** $O(n^2)$ assuming $O(n)$ QR iterations for convergence to Schur form (empirically observed with Wilkinson shifts).

## Applicability

- **Polynomial rootfinding in neural networks**: Companion matrix eigenvalues give polynomial roots; useful for computing roots of characteristic polynomials in recurrent architectures, stability analysis of linear RNNs, and parameterizing transfer functions in state-space models
- **State-space model (SSM) initialization**: S4/S5 models require computing eigenvalues of structured matrices for HiPPO initialization; fast companion QR enables efficient spectral analysis of large polynomial systems
- **Structured recurrence parameterization**: Linear recurrences with polynomial characteristic equations can be analyzed/initialized by factoring through companion matrix eigendecomposition at $O(n^2)$ cost instead of $O(n^3)$
- **Fast spectral methods**: Any layer parameterized by roots of a polynomial (e.g., rational transfer functions, filter design) benefits from $O(n^2)$ rootfinding
- **Numerical stability analysis**: Checking stability of discretized linear dynamical systems by computing eigenvalue radii of companion matrices

## Limitations

- The off-diagonal rank $p$ grows by $2$ (single shift) or $4$ (double shift) per QR iteration before structure recovery; the compression step adds overhead
- The algorithm is inherently sequential (bulge chasing is a serial process), limiting GPU parallelization
- Numerical stability depends on the structured balancing strategy; highly ill-conditioned polynomials may require additional care
- The structure recovery step (compressing SSS generators back to minimal form) requires SSS matrix-matrix multiplication with cost $O(p^3 n)$; when $p$ is not small, this dominates
- Only applicable to companion matrices (or matrices with similarly low off-diagonal rank under QR iteration); general dense matrices do not preserve this structure

## Implementation Notes

```python
# Pseudocode for fast companion QR via SSS structured bulge chasing

import numpy as np

def fast_companion_qr(coeffs, max_iter=100, tol=1e-14):
    """
    Find all roots of p(x) = x^n - a_1*x^{n-1} - ... - a_n
    via fast QR iteration on the companion matrix.

    Args:
        coeffs: [a_1, a_2, ..., a_n] polynomial coefficients
        max_iter: maximum QR iterations
        tol: convergence tolerance

    Returns:
        eigenvalues (= polynomial roots)
    """
    n = len(coeffs)
    a = coeffs

    # Step 1: Initial QR factorization C = Q * R
    # Q represented as Givens rotation parameters {(c_i, s_i)}
    # R represented as SSS generators {d_i, u_i, v_i, w_i}
    c, s = np.zeros(n), np.zeros(n)
    c[n-1], s[n-1] = 1.0, 0.0  # convention

    # Compute initial Givens rotations to triangularize C
    # (zeroing out subdiagonal 1's from top to bottom)
    for k in range(n-1):
        # Apply Q_k^T to zero out C(k+1, k) = 1
        c[k], s[k] = compute_givens(...)  # from subdiagonal entries

    # SSS generators for R^{(0)}:
    d = np.array([c[i] * s[i-1] * ... * s[0] * a[i] - s[i]
                   for i in range(n)])  # diagonal
    u = np.array([c[i] * s[i-1] * ... * s[0]
                   for i in range(n-1)])  # upper SSS generator
    v = np.array([a[i] for i in range(1, n+1)])  # upper SSS generator
    w = np.ones(n-1)  # transition matrices (all 1 initially)

    # Step 2: Implicit QR iterations with structured bulge chasing
    eigenvalues = []
    m = n  # active matrix size

    for iteration in range(max_iter):
        # Wilkinson shift from trailing 2x2 of H
        sigma = compute_wilkinson_shift(d, u, v, c, s, m)

        # Structured bulge chasing on R
        for i in range(m - 1):
            # Create bulge at R(i+1, i) via right Givens G_bar
            d_hat_i, h_hat_i, b_i, d_hat_ip1 = create_bulge(
                d[i], d[i+1], u[i], v[i+1], c[i], s[i], sigma)

            # Eliminate bulge via left Givens G_tilde
            c_tilde, s_tilde = compute_givens(d_hat_i, b_i)
            d[i], d[i+1] = eliminate_bulge(
                c_tilde, s_tilde, d_hat_i, h_hat_i, b_i, d_hat_ip1)

            # Givens swap: push G_tilde forward through Q factors
            # This is O(1) per swap!
            c_hat, s_hat = givens_swap(c[i], s[i], c_tilde, s_tilde)

            # Update SSS generators u_i locally
            u[i] = update_u(c_tilde, s_tilde, u[i])

        # Structure recovery: compress SSS generators back to rank p
        compress_sss_generators(d, u, v, w, m)

        # Check for deflation (convergence of eigenvalues)
        if abs(subdiagonal_entry(m-1)) < tol:
            eigenvalues.append(d[m-1])
            m -= 1
            if m <= 1:
                eigenvalues.append(d[0])
                break

    return eigenvalues

def givens_swap(c1, s1, c2, s2):
    """
    Given Q_i and G_i (adjacent Givens rotations),
    compute Q_hat, G_hat such that Q_i * G_i = G_hat * Q_hat.
    Cost: O(1) operations.
    """
    c_hat = c1 * c2 - s1 * s2
    s_hat = c1 * s2 + s1 * c2
    return c_hat, s_hat
```

## References

- Chandrasekaran, Gu, Xia, Zhu, "A Fast QR Algorithm for Companion Matrices," Operator Theory: Advances and Applications, Vol. 179, pp. 111–143, Birkhäuser, 2007
- Bini, Boito, Eidelman, Gemignani, Gohberg, "A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices," Linear Algebra Appl. 432(8), pp. 2006–2031, 2010
- Chandrasekaran, Gu, "A divide-and-conquer algorithm for the eigendecomposition of symmetric block-diagonal plus semiseparable matrices," Numer. Math. 96(4), pp. 723–731, 2004
- Chandrasekaran, Dewilde, Gu, Pals, Sun, van der Veen, White, "Some fast algorithms for sequentially semiseparable representations," SIAM J. Matrix Anal. Appl. 27(2), pp. 341–364, 2005
- Eidelman, Gemignani, Gohberg, "On the fast reduction of a quasiseparable matrix to Hessenberg and tridiagonal forms," Linear Algebra Appl. 420, pp. 86–101, 2007

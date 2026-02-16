# 146: Zolotarev-ADI Displacement-Based HSS Compression

**Category**: decomposition
**Gain type**: efficiency
**Source**: Beckermann, Kressner & Wilber — "Compression Properties for Large Toeplitz-like Matrices" (2025)
**Paper**: [papers/sublinear-hss-cauchy-construction.pdf]
**Documented**: 2026-02-15

## Description

This trick provides a deterministic, adaptive method for compressing Cauchy-like matrices (derived from Toeplitz or Toeplitz-like matrices) into HODLR and HSS formats using the factored Alternating Direction Implicit (fADI) iteration with shift parameters derived from Zolotarev rational functions. The key contributions are:

1. **Explicit rank bounds** via Zolotarev numbers that quantify how compressible each off-diagonal block is, enabling adaptive rank selection for a target tolerance $\epsilon$.
2. **Deterministic compression** via fADI that replaces randomized sampling with a cheap iterative method exploiting the displacement structure of the Cauchy-like matrix.
3. **Similarity exploitation** between off-diagonal blocks at the same tree level, reducing the HODLR construction cost from $O(\rho n \log^2 n)$ to $O(\rho n \log n)$ by computing only one fADI approximation per level.
4. **Near-sublinear HSS construction**: when the Cauchy-like base matrix $\mathscr{C}$ is fully exploited, the cost of finding an approximate HSS factorization is $O(\rho^3 \log^3 n \log^3 1/\epsilon)$ — poly-logarithmic (sublinear) in $n$.

This is directly applicable to any pipeline that solves Toeplitz-like systems via the displacement structure approach (Toeplitz → Cauchy-like → hierarchical low-rank → fast solve), as used in SSMs, convolution layers, and signal processing.

## Mathematical Form

**Stage 1 — Displacement Structure of Cauchy-like Matrices:**

A Toeplitz-like matrix $T \in \mathbb{C}^{n \times n}$ with displacement rank $\rho$ satisfies:

$$
ZT - TZ = GH^*, \quad Z = \begin{pmatrix} 0 & 1 \\ I_{n-1} & 0 \end{pmatrix}, \quad \text{rank}(GH^*) = \rho \ll n
$$

Applying the DFT $F = \left(\frac{\omega^{2jk}}{\sqrt{n}}\right)$, where $\omega = e^{i\pi/n}$, transforms $T$ into a Cauchy-like matrix:

$$
C = FTF^*, \quad DC - CD = \widetilde{G}\widetilde{H}^*, \quad D = \text{diag}(\omega^0, \omega^2, \ldots, \omega^{2n-2})
$$

where $\widetilde{G} = FG$, $\widetilde{H} = H^* \text{diag}(f)^{-1} F^H$ with $f = [1, \omega, \ldots, \omega^{n-1}]^T$.

**Stage 2 — Zolotarev-Based Rank Bounds:**

For any $(m, \text{sep})$ submatrix $C_{JK}$ of $C$ (with row indices $J$, column indices $K$ separated by at least $\text{sep}$ positions from the diagonal), the $\epsilon$-rank satisfies:

$$
\text{rank}_\epsilon(C_{JK}) \leq \rho \left\lceil \frac{2}{\pi^2} \log\left(4 \frac{m + \text{sep} - 1}{\text{sep}}\right) \log\left(\frac{4}{\epsilon}\right) \right\rceil
$$

This bound is derived via Zolotarev numbers $Z_k(\mathcal{A}_J, \mathcal{A}_K)$, which measure the best rational approximation to $1/z$ on two disjoint arcs of the unit circle containing the spectra of the diagonal matrices $D_J$ and $D_K$:

$$
Z_k(\mathcal{A}_J, \mathcal{A}_K) \leq 4\xi^{-k}, \quad \xi = \exp\left(\frac{\pi^2}{2\log\left(4\frac{m+\text{sep}-1}{\text{sep}}\right)}\right) > 1
$$

**Key insight — Strongly admissible blocks** ($\text{sep} \geq m + 1$) have ranks bounded by $O(\rho \log 1/\epsilon)$, independent of $n$. This means the far-field blocks have constant rank as $n \to \infty$.

**Stage 3 — Factored ADI (fADI) for Low-Rank Approximation:**

Each off-diagonal HODLR block $X = C_v$ satisfies a Sylvester equation with diagonal coefficients:

$$
D_J X - X D_K = \widetilde{G}_J \widetilde{H}_K^*, \quad J = J_v, \quad K = J_{\tilde{v}}
$$

The fADI iteration produces a factored low-rank approximation $X \approx ZW^*$:

$$
Z_1 = (\nu_1 - \tau_1)(D_J - \nu_1 I)^{-1} \widetilde{G}_J
$$
$$
W_1 = (D_K^* - \overline{\nu}_1 I)^{-1} \widetilde{H}_K
$$
$$
Z_{j+1} = (\nu_{j+1} - \tau_{j+1})(D_J - \tau_j I)(D_J - \nu_{j+1} I)^{-1} Z_j
$$
$$
W_{j+1} = (D_K^* - \overline{\nu}_j I)(D_K^* - \overline{\tau}_{j+1} I)^{-1} W_j
$$

After $k$ iterations, $X^{(k)} = ZW^*$ has rank at most $\rho k$ and error:

$$
\|X - X^{(k)}\|_2 \leq 4\xi^{-k} \|X\|_2
$$

**Optimal shift parameters** $\{\tau_j, \nu_j\}_{j=1}^k$ are chosen as zeros and poles of the Zolotarev rational function $r_k$, computed via Jacobi elliptic functions:

$$
\tau_j = T_1\left(-\delta \, \text{dn}\left[\frac{2j-1}{2k} K(\Xi), \Xi\right]\right), \quad \nu_j = T_1\left(\delta \, \text{dn}\left[\frac{2j-1}{2k} K(\Xi), \Xi\right]\right)
$$

where $\Xi = \sqrt{1 - 1/\gamma^2}$, $K(\cdot)$ is the complete elliptic integral of the first kind, $\text{dn}[\cdot, \cdot]$ is the Jacobi elliptic function, and $T_1$ is a Möbius transformation mapping to the relevant arcs.

**Stage 4 — Level-Sharing for Faster HODLR/HSS Construction:**

Define the "base" Cauchy matrix $\mathscr{C}$ with entries:

$$
\mathscr{C}_{jk} = \frac{\omega^j \omega^k}{\omega^{2j} - \omega^{2k}} = \frac{1}{2i\sin(\pi(j-k)/n)}, \quad j \neq k
$$

All $2^{\ell+1}$ off-diagonal HODLR blocks at level $\ell$ are nearly identical (up to diagonal scaling):

$$
X^{(k)} = \mathscr{X}^{(k)} \circ \widehat{G}(J,:) \widehat{H}(K,:)^* = \sum_{j=0}^{\rho-1} \text{diag}(\widehat{G}(J,j)) \, \mathscr{X}^{(k)} \, \text{diag}(\widehat{H}(K,j))^*
$$

So only **one fADI computation per tree level** suffices, reducing HODLR construction to:

$$
O\left(\rho n \log^2 n \cdot (\log\log n + |\log \epsilon|)\right)
$$

For HSS, fully exploiting this structure yields a poly-logarithmic cost:

$$
O\left(\rho^3 \log^3 n \log^3 \frac{1}{\epsilon}\right)
$$

**Key Definitions:**

- $T \in \mathbb{C}^{n \times n}$ — Toeplitz-like matrix with displacement rank $\rho$
- $C = FTF^* \in \mathbb{C}^{n \times n}$ — Cauchy-like matrix
- $\epsilon$ — target approximation tolerance
- $\xi > 1$ — exponential convergence rate of fADI (depends on block separation)
- $p = O(\rho \log n \cdot (\log\log n + |\log\epsilon|))$ — HODLR rank
- $\text{rank}_\epsilon(X)$ — smallest $k$ such that $\sigma_{k+1}(X) \leq \epsilon \|X\|_2$

## Complexity

| Operation | Randomized HSS | Zolotarev-ADI HODLR | Zolotarev-ADI HSS |
|-----------|---------------|---------------------|-------------------|
| Construction | $O(\rho n \log^2 n)$ | $O(\rho n \log n \cdot (\log\log n + |\log\epsilon|))$ | $O(\rho^3 \log^3 n \log^3 1/\epsilon)$ |
| Overall solve $Tx = b$ | $O(n(\rho\log n \log 1/\epsilon)^2)$ | $O(n(\rho\log n \log 1/\epsilon)^2)$ | $O(n p^2)$ |
| Rank per block | data-dependent | $O(\rho \log m \log 1/\epsilon)$ | $O(\rho \log 1/\epsilon)$ (far field) |

**Memory:** $O(np)$ for HODLR, $O(np / \log_2 n)$ for HSS

**Advantage over randomized:** Deterministic error control — the fADI error bound $4\xi^{-k}\|X\|_2$ gives explicit guarantees, enabling adaptive rank selection to match a prescribed tolerance. No random matrix-vector products needed.

## Applicability

- **Toeplitz system solvers for SSMs**: Any state-space model whose recurrence kernel is Toeplitz can be solved via this pipeline (Toeplitz → Cauchy → HSS → ULV solve), with the Zolotarev-ADI step providing deterministic, adaptive compression with explicit error bounds.
- **Structured weight matrices**: Learning Toeplitz-like weight matrices (displacement rank $\rho$) in neural network layers; the compression bounds quantify achievable model compression.
- **Signal processing convolutions**: 1D and 2D convolutions expressed as Toeplitz matrix operations benefit from faster HSS construction.
- **Kernel matrices**: Cauchy-like kernel matrices arising in radial basis function networks or Gaussian process layers.
- **Adaptive precision training**: The explicit $\epsilon$-rank bounds enable training with varying precision — coarse approximations ($\epsilon = 10^{-3}$) for early epochs, refined ($\epsilon = 10^{-12}$) for fine-tuning.

## Limitations

- The fADI iteration requires evaluating Jacobi elliptic functions for optimal shift parameters; standard implementations (MATLAB's `ellipk`, `ellipj`) may lose accuracy when the spectral gap is small. The authors recommend the Schwarz-Christoffel Toolbox.
- The sublinear HSS construction $O(\rho^3 \log^3 n \log^3 1/\epsilon)$ uses an implicit factored form (equation 3.9) that supports matrix-vector products but **not** direct ULV solvers — assembling explicit low-rank factors destroys the complexity advantage.
- The Zolotarev bounds are tight for weakly admissible blocks ($\text{sep} = 1$) but the actual singular values may decay faster for specific Toeplitz matrices with special structure (e.g., banded).
- Currently validated in MATLAB using the hm-toolbox; no GPU-optimized implementation exists.
- The fADI convergence rate $\xi$ depends on the ratio $(m + \text{sep} - 1)/\text{sep}$; for blocks near the diagonal ($\text{sep} = 1$), convergence is slowest and the rank bound is $O(\rho \log m \log 1/\epsilon)$.

## Implementation Notes

```python
# Pseudocode for Zolotarev-ADI HSS compression of Cauchy-like matrix

import numpy as np
from scipy.special import ellipk, ellipj

def zolotarev_shift_parameters(alpha, beta, k):
    """
    Compute optimal fADI shift parameters from Zolotarev rational function.

    alpha, beta: arc angles defining the spectral sets A_J, A_K
    k: number of ADI iterations (determines approximation rank)

    Returns: shifts (tau_j), poles (nu_j) for j=1,...,k
    """
    kappa = np.tan(alpha/2) / np.tan(beta/2)  # cross-ratio parameter
    gamma = ((1 + kappa) / (1 - kappa))**2
    Xi = np.sqrt(1 - 1/gamma**2)
    K_val = ellipk(Xi**2)  # complete elliptic integral

    taus, nus = [], []
    for j in range(1, k+1):
        u = (2*j - 1) / (2*k) * K_val
        sn, cn, dn, _ = ellipj(u, Xi**2)
        # Map through Möbius transformation T_1
        taus.append(T1(-delta * dn))
        nus.append(T1(delta * dn))

    return np.array(taus), np.array(nus)

def fadi_compress(D_J, D_K, G_tilde_J, H_tilde_K, taus, nus):
    """
    Factored ADI iteration for low-rank approximation of
    off-diagonal block X satisfying D_J X - X D_K = G_J H_K*.

    D_J, D_K: diagonal matrices (as vectors)
    G_tilde_J, H_tilde_K: generator submatrices (m x rho)
    taus, nus: Zolotarev shift parameters

    Returns: Z, W such that X ≈ Z @ W.conj().T
    """
    k = len(taus)
    rho = G_tilde_J.shape[1]
    m = len(D_J)

    # First iteration
    Z_cols = [(nus[0] - taus[0]) / (D_J - nus[0]) * G_tilde_J[:, col]
              for col in range(rho)]
    W_cols = [H_tilde_K[:, col] / (D_K.conj() - nus[0].conj())
              for col in range(rho)]

    # Subsequent iterations (each adds rho columns)
    for j in range(1, k):
        scale_Z = (nus[j] - taus[j]) * (D_J - taus[j-1]) / (D_J - nus[j])
        scale_W = (D_K.conj() - nus[j-1].conj()) / (D_K.conj() - taus[j].conj())
        Z_cols = [scale_Z * z for z in Z_cols]  # update in-place
        W_cols = [scale_W * w for w in W_cols]
        # ... accumulate [Z_1, ..., Z_k] and [W_1, ..., W_k]

    Z = np.column_stack(Z_cols)  # m x (rho * k)
    W = np.column_stack(W_cols)  # m x (rho * k)
    return Z, W

def adaptive_hodlr_construction(C_generators, n, rho, epsilon):
    """
    Build HODLR approximation of Cauchy-like matrix C
    with adaptive rank per block using Zolotarev bounds.

    Total cost: O(rho * n * log(n) * (log(log(n)) + |log(epsilon)|))
    """
    tree = build_binary_tree(n)

    for level in range(tree.depth):
        m = n // (2**(level + 1))  # block size at this level
        sep = 1  # weakly admissible (HODLR)

        # Compute number of fADI iterations needed for epsilon accuracy
        xi = np.exp(np.pi**2 / (2 * np.log(4 * (m + sep - 1) / sep)))
        k_v = int(np.ceil(np.log(4 * epsilon**(-1)) / np.log(xi)))

        # Compute shifts (same for all blocks at this level)
        alpha = np.pi * (m - 1) / n
        beta = np.pi * (m - 1 + 2 * sep) / n
        taus, nus = zolotarev_shift_parameters(alpha, beta, k_v)

        # KEY TRICK: only one fADI computation for base matrix C_0
        # Then scale by generators for actual matrix C
        Z_base, W_base = fadi_compress(D_J_base, D_K_base,
                                        f_J, f_K, taus, nus)

        # All blocks at this level use same Z_base, W_base
        for block in tree.blocks_at_level(level):
            block.U = Z_base  # (scaled by diagonal of generators)
            block.V = W_base

    return tree
```

## References

- Beckermann, B., Kressner, D., & Wilber, H. (2025). Compression Properties for Large Toeplitz-like Matrices. arXiv:2502.09823. Submitted to SIAM Review.
- Beckermann, B. & Townsend, A. (2019). Bounds on the singular values of matrices with displacement structure. SIAM Review, 61(2), 319-344.
- Lepilov, M. & Xia, J. (2024). Rank-structured approximation of some Cauchy matrices with sublinear complexity. Numerical Linear Algebra with Applications, e2526.
- Benner, P., Li, R.-C., & Truhar, N. (2009). On the ADI method for Sylvester equations. Journal of Computational and Applied Mathematics, 233(4), 1035-1045.
- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2007). A superfast algorithm for Toeplitz systems of linear equations. SIAM J. Matrix Anal. Appl., 29(4), 1247-1266.

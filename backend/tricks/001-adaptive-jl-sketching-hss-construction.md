# 001: Adaptive Johnson-Lindenstrauss Sketching for HSS Construction

**Category**: decomposition
**Gain type**: efficiency
**Source**: Yaniv, Ghysels, Malik, Boateng, & Li (2025), Comm. Appl. Math. Comput. Sci.
**Paper**: [papers/adaptive-jl-sketching-hss.pdf]
**Documented**: 2026-02-15

## Description

This trick accelerates the construction (compression) of Hierarchically Semi-Separable (HSS) matrices by replacing the standard Gaussian random sketching operator with faster Johnson-Lindenstrauss (JL) sketching operators — specifically the Sparse Johnson-Lindenstrauss Transform (SJLT) and the Subsampled Randomized Hadamard Transform (SRHT). The sketching step $S = AR$ is the computational bottleneck in randomized HSS compression, costing $O(n^2 d)$ for Gaussian operators where $d = r + p$ is the sketch dimension. By using SJLT (which has only $\alpha$ nonzeros per row, typically $\alpha = 2$–$4$), the sketching cost drops to $O(n \alpha d)$ — a factor of $n/\alpha$ improvement. For SRHT, the cost is $O(n^2 \log d)$ via fast Hadamard transforms.

The key theoretical contribution is extending the Frobenius norm concentration bounds and range-finder bounds from Gaussian sketching to all JL sketching operators, proving that the adaptive stopping criteria used in the HSS construction algorithm remain valid. This unified framework provides theoretical lower bounds on the sketch dimension $d$ for each operator class, allowing users to trade off between sketching speed and accuracy guarantees.

Experimentally, SJLT achieves up to 2.5$\times$ speedup in serial HSS construction and up to 35$\times$ speedup in distributed-memory parallel construction (where the sparse SJLT can be duplicated on each MPI process, eliminating communication overhead for the sketching step).

## Mathematical Form

**HSS Construction via Sketching:**

Given a matrix $A \in \mathbb{C}^{n \times n}$, the HSS compression algorithm computes a sketch:

$$
S = AR, \quad S' = A^* R'
$$

where $R, R' \in \mathbb{R}^{n \times d}$ are random sketching operators with $d = r + p$ columns ($r$ = target HSS rank, $p$ = oversampling parameter).

**Johnson-Lindenstrauss Sketching Operator:**

A distribution $\mathcal{D}$ over matrices of size $d \times n$ is an $(n, d, \delta, \varepsilon)$-JL sketching operator if for any vector $x \in \mathbb{R}^n$:

$$
\Pr_{R \sim \mathcal{D}} \left[ \left| \|Rx\|^2 - \|x\|^2 \right| > \varepsilon \|x\|^2 \right] < \delta
$$

**Frobenius Norm Concentration (Theorem 1 — New General Bound):**

For any $A \in \mathbb{C}^{m \times n}$ and $(n, d, \delta/m, \varepsilon)$-JL matrix $R$, with probability at least $1 - \delta$:

$$
(1 - \varepsilon)\|A\|_F^2 \leq \|AR\|_F^2 \leq (1 + \varepsilon)\|A\|_F^2
$$

This enables the adaptive stopping criterion: the sketch norm $\|AR\|_F$ approximates $\|A\|_F$, allowing the algorithm to determine when enough columns of $R$ have been sampled.

**Operator-Specific Bounds on Sketch Dimension $d$:**

| Sketching Operator | Required $d$ for Frobenius Norm Bound |
|---|---|
| General JL | $(n, d, \delta/(2m), \varepsilon)$-JL matrix |
| Gaussian | $d \geq 20\varepsilon^{-2}\log(2/\delta)$ |
| SJLT | $d \geq C\varepsilon^{-2}\log(1/\delta)$ |
| SRHT | $d \geq 2\varepsilon^{-2}\log^2(4n^2/\delta)\log(4/\delta)$ |

**Sparse JL Transform (SJLT):**

An SJLT matrix $R \in \mathbb{R}^{n \times d}$ has exactly $\alpha$ nonzeros per row, with entries drawn from $\{1/\sqrt{\alpha}, -1/\sqrt{\alpha}\}$:

$$
R = \frac{1}{\sqrt{\alpha}}(B_+ - B_-)
$$

where $B_+, B_-$ are binary matrices with $\alpha$ nonzeros per row. The key data structure stores $B_+$ and $B_-$ separately in compressed row/column storage (CRS/CCS) *without value arrays* (since all nonzeros are 1), enabling multiplication via indexing and addition only — no floating-point multiplications.

**SJLT Sketching via Outer Products:**

For column-major storage of $A$, the sketch is computed as:

$$
AR = \sum_{i=1}^{n} A_{:i} R_{i:}
$$

Each row $R_{i:}$ has only $\alpha$ nonzero entries, so each outer product $A_{:i} R_{i:}$ adds or subtracts column $A_{:i}$ to/from $\alpha$ columns of the result. Total cost: $O(n \cdot m \cdot \alpha)$ additions (no multiplications), scaled by $1/\sqrt{\alpha}$ at the end.

**Subsampled Randomized Hadamard Transform (SRHT):**

$$
R = DHP
$$

where $D = \text{diag}(d_1, \ldots, d_n)$ with $d_i \in \{-1, +1\}$ Rademacher, $H = H_\nu$ is the normalized Hadamard matrix of size $\nu = 2^{\lceil \log_2 n \rceil}$, and $P \in \mathbb{R}^{n \times d}$ is a column subsampling matrix. When $n$ is not a power of 2, the Hadamard transform of dimension $2k$ is split into two transforms of dimensions $k$ and $r = n - k$:

$$
AH = \left[A_{m,k} H_k + \tilde{A} H_k \quad A_{m,k} H_k - \tilde{A} H_k\right]
$$

**Range-Finder Bound for JL (Theorem 5):**

For any $(n, d, \frac{\delta}{2\max(5^r, n)}, \frac{\varepsilon}{12})$-JL sketching operator with $d = r + p$:

$$
\|(I - P_Y)A\| \leq \left(\sqrt{1 + \frac{n(1+\varepsilon)}{(1-\varepsilon)}}\right) \sigma_{r+1}(A)
$$

where $Y = AR = Q\Omega$, $P_Y = QQ^\dagger$, confirming that JL sketches preserve the approximate range of the original matrix — a necessary property for HSS compression.

**Adaptive Stopping Criteria:**

The algorithm adaptively adds $\Delta d$ columns to $R$ until the new sketch residual satisfies:

$$
\frac{\|\hat{S}\|_F}{\|\hat{S}\|_F} < \varepsilon_{\text{rel}}, \quad \|\hat{S}\|_F < \varepsilon_{\text{abs}}
$$

where $\hat{S} = (I - Q_\tau Q_\tau^*)^2 \tilde{S}$ is the doubly-projected residual sketch at node $\tau$.

**Key Definitions:**

- $A \in \mathbb{C}^{n \times n}$ — Matrix to be compressed into HSS form
- $r$ — Numerical HSS rank (maximum off-diagonal rank)
- $d = r + p$ — Total sketch dimension ($p$ = oversampling, typically 10)
- $\alpha$ — Number of nonzeros per row in SJLT (typically 2–4)
- $\varepsilon_{\text{rel}}, \varepsilon_{\text{abs}}$ — Relative and absolute compression tolerances
- $d_0$ — Initial sketch size (default 128 in STRUMPACK)
- $\Delta d$ — Incremental sketch size for adaptation (default 64)

## Complexity

| Operation | Gaussian Sketch | SJLT Sketch | SRHT Sketch |
|-----------|----------------|-------------|-------------|
| Sketching $AR$ | $O(n^2 d)$ | $O(n \alpha d)$ | $O(n^2 \log d)$ |
| Sketching $A^* R$ | $O(n^2 d)$ | $O(n \alpha d)$ | $O(n^2 \log d)$ |
| HSS compression (remaining) | $O(nr^2)$ | $O(nr^2)$ | $O(nr^2)$ |
| Total HSS construction | $O(n^2 d)$ | $O(n \alpha d + nr^2)$ | $O(n^2 \log d + nr^2)$ |
| Random matrix generation | $O(nd)$ | $O(n\alpha)$ | $O(n + d)$ |

**Memory:**

- Gaussian $R$: $O(nd)$ — dense
- SJLT $R$: $O(n\alpha)$ — sparse (only index arrays, no value arrays needed)
- SRHT $R$: $O(n + d)$ — implicit (stored as $D$, $P$ vectors)

**Parallel Distributed Speedup:**

| Setting | Gaussian | SJLT ($\alpha = 4$) | Speedup |
|---------|---------|---------------------|---------|
| Serial sketching | $O(n^2 d)$ | $O(n \alpha d)$ | $\sim$2–2.5$\times$ |
| Distributed sketching (32 ranks) | $O(n^2 d / P + \text{comm})$ | $O(n \alpha d / P)$ (no comm) | 8–40$\times$ |
| Total distributed HSS | varies | varies | 1.3–35$\times$ |

The massive parallel speedup for SJLT comes from eliminating communication: the sparse SJLT operator can be duplicated on each MPI rank (low memory), enabling local sketching without inter-process communication, versus the Gaussian case which requires 2D block cyclic distribution and costly all-to-all communication.

## Applicability

1. **Large-Scale HSS Compression**: When the matrix $A$ is available explicitly or via matrix-vector products, and the sketching step dominates construction cost — SJLT reduces sketching from $O(n^2 d)$ to $O(n\alpha d)$

2. **Distributed-Memory HSS Construction**: In multi-node settings (e.g., training large neural networks on clusters), SJLT eliminates inter-node communication during the sketching phase, enabling near-linear strong scaling

3. **Attention Kernel Compression**: Compressing attention matrices $K = \text{softmax}(QK^T/\sqrt{d_k})$ into HSS form using fast sketching — the sparse structure of SJLT is particularly GPU-friendly for batched sparse matrix operations

4. **Sparse Multifrontal Solvers (STRUMPACK)**: The implementation is integrated into the STRUMPACK solver library, accelerating HSS-compressed multifrontal factorization for sparse PDE systems

5. **Online/Adaptive Settings**: The adaptive stopping criterion works with any JL operator, enabling efficient rank discovery when the HSS rank is not known a priori

6. **Neural Network Weight Compression**: Fast sketching enables efficient compression of weight matrices with hierarchical low-rank structure during model compression or quantization

## Limitations

1. **SJLT with $\alpha = 1$ is insufficient**: Experiments show that $\alpha = 1$ (one nonzero per row) produces poor approximations — $\alpha \geq 2$ is needed, with $\alpha = 4$ recommended as the default

2. **SRHT Non-Power-of-Two Issue**: When $n$ is not a power of 2, the SRHT requires padding or splitting strategies that complicate the adaptive scheme; SRHT cannot use the incremental adaptation strategy

3. **Accuracy-Speed Tradeoff**: While SJLT and SRHT are faster, they may produce slightly higher compression errors than Gaussian sketching at the strictest tolerances ($\varepsilon_{\text{rel}} = 10^{-6}$)

4. **Theoretical Bounds are Conservative**: The theoretical minimum sketch dimension $d$ for JL guarantees is much larger than what is needed in practice — users should rely on empirical guidelines ($d_0 = 128$, $\Delta d = 64$)

5. **Matrix Entry Access Still Required**: The algorithm is "partially matrix-free" — it needs both matrix-vector products and access to $O(nr)$ matrix entries (diagonal blocks), unlike the fully black-box approach

6. **Communication Bound for Gaussian**: The parallel speedup advantage of SJLT is specifically because Gaussian sketching requires expensive 2D block-cyclic communication — on shared-memory systems, the advantage is smaller (2–2.5$\times$)

## Implementation Notes

```python
# Adaptive HSS compression with JL sketching operators
# (Generalization of Gorman et al. [SIAM J. Sci. Comput., 2019])

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def construct_sjlt(n, d, alpha=4):
    """
    Construct a Sparse JL Transform matrix.

    Args:
        n: number of rows
        d: number of columns (sketch dimension)
        alpha: nonzeros per row (default 4)

    Returns:
        R: SJLT as (B_plus, B_minus, scale) where B_plus, B_minus
           are binary sparse matrices stored WITHOUT value arrays

    Storage: O(n * alpha) indices only
    """
    scale = 1.0 / np.sqrt(alpha)

    # Block construction: divide each row into d/alpha chunks
    # Select one nonzero position per chunk
    rows_plus = []
    cols_plus = []
    rows_minus = []
    cols_minus = []

    for i in range(n):
        # Select alpha random column positions
        positions = np.random.choice(d, size=alpha, replace=False)
        # Random signs
        signs = np.random.choice([-1, 1], size=alpha)

        for j, (pos, sign) in enumerate(zip(positions, signs)):
            if sign > 0:
                rows_plus.append(i)
                cols_plus.append(pos)
            else:
                rows_minus.append(i)
                cols_minus.append(pos)

    B_plus = csr_matrix(
        (np.ones(len(rows_plus)), (rows_plus, cols_plus)), shape=(n, d)
    )
    B_minus = csr_matrix(
        (np.ones(len(rows_minus)), (rows_minus, cols_minus)), shape=(n, d)
    )

    return B_plus, B_minus, scale


def sjlt_sketch(A, B_plus, B_minus, scale):
    """
    Compute S = A @ R where R = scale * (B_plus - B_minus).

    Key insight: No floating-point multiplications needed!
    Only indexing and addition/subtraction of columns of A.

    Uses outer product formulation for column-major A:
        AR = sum_i A[:,i] * R[i,:]
    Each R[i,:] has only alpha nonzeros, so we just
    add/subtract columns of A to alpha positions.

    Complexity: O(m * n * alpha) additions (no multiplications)
    """
    m, n = A.shape
    d = B_plus.shape[1]
    S = np.zeros((m, d))

    # Convert to CSR for efficient row access
    Bp_csr = B_plus.tocsr()
    Bm_csr = B_minus.tocsr()

    for i in range(n):
        # Add A[:,i] to columns where B_plus[i,:] is nonzero
        plus_cols = Bp_csr[i].indices
        for j in plus_cols:
            S[:, j] += A[:, i]

        # Subtract A[:,i] from columns where B_minus[i,:] is nonzero
        minus_cols = Bm_csr[i].indices
        for j in minus_cols:
            S[:, j] -= A[:, i]

    S *= scale
    return S


def adaptive_hss_compress_jl(A, tree, sketching='sjlt', alpha=4,
                              d0=128, delta_d=64,
                              eps_rel=1e-4, eps_abs=1e-8):
    """
    Adaptive HSS compression using JL sketching operators.

    Args:
        A: n x n matrix (dense or implicit via matvec)
        tree: binary cluster tree for HSS structure
        sketching: 'gaussian', 'sjlt', or 'srht'
        alpha: nonzeros per row for SJLT
        d0: initial sketch dimension
        delta_d: increment for adaptive expansion
        eps_rel, eps_abs: stopping tolerances

    Returns:
        HSS representation {U_tau, V_tau, D_tau, B_tau} for all nodes

    Complexity:
        Gaussian: O(n^2 * d)
        SJLT:     O(n * alpha * d + n * r^2)
        SRHT:     O(n^2 * log(d) + n * r^2)
    """
    n = A.shape[0]
    d = d0

    # Generate initial sketching operator
    if sketching == 'sjlt':
        Bp, Bm, scale = construct_sjlt(n, d, alpha)
        S = sjlt_sketch(A, Bp, Bm, scale)
    elif sketching == 'gaussian':
        R = np.random.randn(n, d) / np.sqrt(d)
        S = A @ R
    elif sketching == 'srht':
        S = srht_sketch(A, d)  # via fast Hadamard transform

    # Bottom-up compression through the tree
    converged = False
    while not converged:
        converged = True

        for tau in tree.postorder():
            I_tau = tau.indices

            if tau.is_leaf:
                # Extract diagonal block
                tau.D = A[np.ix_(I_tau, I_tau)]

                # Compress off-diagonal using sketch
                S_tau = S[I_tau, :]

                # Subtract diagonal contribution from sketch
                if sketching == 'sjlt':
                    R_tau_Bp = Bp[I_tau, :].toarray()
                    R_tau_Bm = Bm[I_tau, :].toarray()
                    R_tau = scale * (R_tau_Bp - R_tau_Bm)
                else:
                    R_tau = R[I_tau, :]

                S_off = S_tau - tau.D @ R_tau  # off-diagonal sketch

                # Compute interpolative decomposition
                tau.U, J = interpolative_decomposition(S_off, eps_rel)
                tau.V, J_col = interpolative_decomposition(
                    S_off.T, eps_rel  # for column basis
                )
            else:
                # Internal node: merge children's bases
                alpha_child, beta_child = tau.children

                # Combine local sketches using nested basis property
                S_tau = combine_child_sketches(
                    alpha_child, beta_child, S
                )

                # Check stopping criterion
                Q_tau = tau.get_basis()
                S_residual = (np.eye(len(I_tau)) - Q_tau @ Q_tau.T) @ S_tau
                # Apply twice for numerical stability
                S_hat = (np.eye(len(I_tau)) - Q_tau @ Q_tau.T) @ S_residual

                if np.linalg.norm(S_hat, 'fro') > eps_abs and \
                   np.linalg.norm(S_hat, 'fro') / np.linalg.norm(S_tau, 'fro') > eps_rel:
                    converged = False

        if not converged:
            # Expand sketch by delta_d columns
            d_new = d + delta_d
            if sketching == 'sjlt':
                # Efficiently extend: just append new binary columns
                Bp_new, Bm_new, scale_new = construct_sjlt(n, delta_d, alpha)
                S_new = sjlt_sketch(A, Bp_new, Bm_new, scale_new)
                S = np.hstack([S, S_new])
                # Update sketching operator (concatenate)
            elif sketching == 'gaussian':
                R_new = np.random.randn(n, delta_d) / np.sqrt(d_new)
                S_new = A @ R_new
                S = np.hstack([S, S_new])
                R = np.hstack([R, R_new])
            d = d_new

    return tree  # contains {U, V, D, B} for all nodes
```

**Key Implementation Insights:**

1. **SJLT eliminates multiplications**: The binary decomposition $R = \frac{1}{\sqrt{\alpha}}(B_+ - B_-)$ means the sketch $AR$ is computed entirely via column additions/subtractions of $A$, with a single scalar multiplication at the end — this is highly cache-friendly for column-major storage

2. **Parallel communication elimination**: In distributed settings, the SJLT matrix requires only $O(n\alpha)$ storage (indices only), allowing complete duplication on each MPI rank. Each rank computes its local portion of $AR$ independently with *zero communication*, versus Gaussian which requires expensive all-to-all redistribution

3. **Adaptive SJLT extension**: When the sketch is insufficient, new columns are appended to the SJLT by generating $\Delta d$ new binary columns with $\alpha$ nonzeros each — the scaling factor is adjusted from $1/\sqrt{\alpha}$ to accommodate the new total width

4. **CRS/CCS dual storage**: Storing the binary matrices in both compressed row and compressed column formats enables efficient access patterns: CRS for computing $AR$ (outer product formulation) and CCS for computing $A^*R$ (inner product formulation)

5. **Recommended defaults**: STRUMPACK uses $d_0 = 128$, $\Delta d = 64$, $\alpha = 4$ for SJLT, which balances speed and accuracy across diverse problem types

## References

- Yaniv, Y., Ghysels, P., Malik, O. A., Boateng, H. A., & Li, X. S. (2025). Construction of Hierarchically Semi-Separable matrix Representation using Adaptive Johnson-Lindenstrauss Sketching. *Communications in Applied Mathematics and Computational Science*, 20(1). arXiv:2302.01977.
- Gorman, C., Chavez, G., Ghysels, P., Mary, T., Rouet, F.-H., & Li, X. S. (2019). Robust and Accurate Stopping Criteria for Adaptive Randomized Sampling in Matrix-Free Hierarchically Semiseparable Construction. *SIAM J. Sci. Comput.*, 41(5), S61-S85.
- Kane, D. M. & Nelson, J. (2014). Sparser Johnson-Lindenstrauss Transforms. *Journal of the ACM*, 61(1).
- Ailon, N. & Chazelle, B. (2006). Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform. *Proc. 38th ACM STOC*, 557-563.
- STRUMPACK software: https://github.com/pghysels/STRUMPACK/

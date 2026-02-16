# 097: Quasi-Optimal Greedy HSS Approximation

**Category**: approximation
**Gain type**: efficiency
**Source**: Amsel, Chen, Duman Keles, Halikias, Musco, Musco, Persson (2025) — arXiv:2505.16937
**Paper**: [papers/quasi-optimal-hss-approximation.pdf]
**Documented**: 2026-02-15

## Description

This trick provides the first polynomial-time algorithm for producing a *quasi-optimal* Hierarchically Semi-Separable (HSS) approximation to an arbitrary $N \times N$ matrix $A$, using only black-box matrix-vector products with $A$ and $A^\top$. The algorithm is a modification of the empirically effective Levitt-Martinsson method, with the critical change of using *fresh random sketches* at each hierarchical level rather than reusing a single sketch matrix across levels.

The main theoretical guarantee is that the algorithm produces an HSS matrix $B$ with rank-$k$ blocks whose expected Frobenius norm error is at most $O(\log(N/k))$ times worse than the best possible HSS rank-$k$ approximation. This is achieved using only $O(k \log(N/k))$ matrix-vector products and $O(Nk^2 \log(N/k))$ additional runtime.

The key innovation is the analysis framework: a greedy level-by-level approach where at each level of the HSS hierarchy, a near-optimal Sequentially Semi-Separable (SSS) approximation is computed. The analysis shows that the error at each level is bounded by the optimal error, and these errors accumulate at most multiplicatively across the $L = \log_2(N/k)$ levels, yielding the overall $O(L)$ quasi-optimality factor.

## Mathematical Form

**HSS Telescoping Factorization:**

A matrix $B \in \text{HSS}(L, k)$ is defined by:

$$
B^{(\ell+1)} = U^{(\ell)} B^{(\ell)} (V^{(\ell)})^\top + D^{(\ell)}, \quad \ell = 1, \ldots, L
$$

with $B^{(1)} = D^{(0)}$, $B = B^{(L+1)}$, and:
- $U^{(\ell)} = \text{blockdiag}(U_1^{(\ell)}, \ldots, U_{2^\ell}^{(\ell)})$, where $U_i^{(\ell)} \in \mathbb{R}^{2k \times k}$ has orthonormal columns
- $V^{(\ell)} = \text{blockdiag}(V_1^{(\ell)}, \ldots, V_{2^\ell}^{(\ell)})$, where $V_i^{(\ell)} \in \mathbb{R}^{2k \times k}$ has orthonormal columns
- $D^{(\ell)} = \text{blockdiag}(D_1^{(\ell)}, \ldots, D_{2^\ell}^{(\ell)})$, where $D_i^{(\ell)} \in \mathbb{R}^{2k \times 2k}$

**Main Approximation Guarantee (Theorem 3.1):**

Let $B \in \text{HSS}(L, k)$ be the output of the greedy algorithm. If at each level $\ell$, the subspace approximations $U_i^{(\ell)}$ and $V_i^{(\ell)}$ satisfy near-optimality with factors $\Gamma_r$ and $\Gamma_c$, and diagonal blocks $D_i^{(\ell)}$ satisfy factor $\Gamma_d$, then:

$$
\mathbb{E}\left[\|A - B\|_F^2\right] \leq (\Gamma_r + \Gamma_c)(1 + \Gamma_d) L \cdot \min_{C \in \text{HSS}(L,k)} \|A - C\|_F^2
$$

where $L = \log_2(N/k) - 1$ is the number of levels.

**Greedy Algorithm (Algorithm 3.1):**

1. Set $A^{(L+1)} = A$
2. For $\ell = L, \ldots, 1$:
   - Partition $A^{(\ell+1)}$ into $2^\ell \times 2^\ell$ blocks of size $2k \times 2k$
   - For each block row $i$: compute $U_i^{(\ell)}$ to approximate the off-diagonal row $r_i(A^{(\ell+1)})$
   - For each block column $i$: compute $V_i^{(\ell)}$ to approximate the off-diagonal column $c_i(A^{(\ell+1)})$
   - Compute diagonal correction $D_i^{(\ell)}$
   - Set $A^{(\ell)} = (U^{(\ell)})^\top (A^{(\ell+1)} - D^{(\ell)}) V^{(\ell)}$
3. Set $D^{(0)} = A^{(1)}$

**Matvec-Based Implementation (Algorithm 4.1):**

For the implicit (black-box) access model, using fresh sketches $\Omega^{(\ell)}, \Psi^{(\ell)} \sim \text{Gaussian}(2^{\ell+1}k, s)$ at each level $\ell$:

$$
Y^{(\ell)} = A^{(\ell+1)} \Omega^{(\ell)}, \quad Z^{(\ell)} = (A^{(\ell+1)})^\top \Psi^{(\ell)}
$$

where $A^{(\ell+1)}$ is accessed recursively via:

$$
A^{(\ell+1)} \Omega^{(\ell)} = (U^{(\ell+1)})^\top (A^{(\ell+2)} \hat{\Omega}^{(\ell)} - D^{(\ell+1)} \hat{\Omega}^{(\ell)})
$$

with $\hat{\Omega}^{(\ell)} := V^{(\ell+1)} \Omega^{(\ell)}$.

**Block Nullification:**

For each block $i$, construct nullifier $P_i^{(\ell)}$ as orthonormal basis for $\text{null}(\Omega_i^{(\ell)})$, then the implicitly sketched off-diagonal block row is:

$$
Y_i^{(\ell)} P_i^{(\ell)} = r_i(A) G_i^{(\ell)}
$$

where $G_i^{(\ell)} \sim \text{Gaussian}(2^{\ell+1}k - 2k, s - 2k)$, enabling near-optimal low-rank approximation of each block row.

**Projection-Cost-Preserving Sketch (PCPS) Guarantee (Theorem 4.2):**

$$
\mathbb{E}\left[\|B - UU^\top B\|_F^2\right] \leq \left(1 + \frac{2eq}{\sqrt{(q-k)^2 - 1}}\right)^2 \|B - \llbracket B \rrbracket_k\|_F^2
$$

where $\Omega \sim \text{Gaussian}(p, q)$ with $q \geq k + 2$, and $U$ is the top-$k$ left singular vectors of $B\Omega$.

**Combined Matvec Complexity (Theorem 4.1):**

With $s \geq 3k + 2$, using $O(sL)$ matvecs and $O(NksL + Ns^2)$ additional runtime:

$$
\mathbb{E}\|A - B\|_F^2 \leq \Gamma \cdot \min_{C \in \text{HSS}(L,k)} \|A - C\|_F^2, \quad \text{where } \Gamma = O(L)
$$

Setting $s = 5k$: total $O(kL) = O(k \log(N/k))$ matvecs, $O(Nk^2 \log(N/k))$ additional runtime.

## Complexity

| Operation | Prior Best (Levitt-Martinsson) | This Algorithm |
|-----------|-------------------------------|----------------|
| Matvec queries | $O(k)$ | $O(k \log(N/k))$ |
| Additional runtime | $O(Nk^2 \log(N/k))$ | $O(Nk^2 \log(N/k))$ |
| Approximation factor $\Gamma$ | No proven guarantee | $O(\log(N/k))$ |

| Access Model | Time | Approximation Factor |
|--------------|------|---------------------|
| Explicit (dense) | $O(N^2 k)$ | $2\log_2(N/k)$ |
| Matvec (black-box) | $O(Nk^2 \log(N/k))$ + matvec cost | $O(\log(N/k))$ |

**Memory:** $O(Nk)$ for storing the HSS output (same as standard HSS)

**Lower Bound (Theorem 3.7):** There exists a matrix $A$ for which the greedy algorithm (with explicit access) satisfies:

$$
\|A - B\|_F^2 \geq (2 - \varepsilon) \min_{C \in \text{HSS}(L,k)} \|A - C\|_F^2
$$

for any $\varepsilon > 0$. Whether $O(1)$ approximation is achievable remains an open question.

## Applicability

1. **Kernel matrix compression**: Covariance and kernel matrices from Gaussian processes, attention mechanisms, and integral operators can be approximated as HSS — this algorithm provides the first provably near-optimal such compression from matvec access
2. **Operator learning**: Scientific ML applications where operators are available only through input-output pairs (simulation queries) — the matvec model is natural
3. **Hessian approximation**: Second-order optimization methods need to approximate large Hessian matrices; HSS approximation provides structured compression with guaranteed quality
4. **Structured attention layers**: Approximating dense attention matrices with HSS structure for sub-quadratic attention; this algorithm ensures the approximation is near-optimal among all HSS matrices of a given rank
5. **Preconditioning**: Constructing HSS preconditioners for iterative methods (CG, GMRES) applied to kernel systems or PDE discretizations
6. **Transfer learning / adapter layers**: Compressing weight matrices of pretrained models into HSS form for efficient fine-tuning

## Limitations

1. **Logarithmic overhead in matvec queries**: Uses $O(k \log(N/k))$ matvecs vs $O(k)$ for Levitt-Martinsson — the extra factor comes from using fresh sketches at each level; closing this gap is an open problem
2. **Approximation factor gap**: Upper bound is $O(\log(N/k))$ but lower bound is only $2 - \varepsilon$; it's unknown whether $O(1)$ approximation is achievable
3. **Adaptive matvecs**: The matvec queries at each level depend on results from previous levels (via the recursion in eq. 4.2), so they cannot all be computed in parallel — limits GPU utilization
4. **Perfect binary tree assumption**: Analysis assumes $N = 2^{L+1}k$ for clean notation, though extensions to arbitrary $N$ are possible
5. **Frobenius norm only**: Guarantees are in Frobenius norm; spectral norm bounds are weaker ($O(\sqrt{N/k})$ factor)
6. **No learned structure**: The algorithm finds the best rank-$k$ HSS approximation given a fixed binary partition tree; it doesn't optimize over tree structures

## Implementation Notes

```python
import numpy as np

def greedy_hss_approx_explicit(A, k, L=None):
    """
    Algorithm 3.1: Greedy HSS approximation with explicit access.
    Produces HSS(L, k) approximation B to N x N matrix A.
    Approximation factor: 2*L in Frobenius norm squared.
    """
    N = A.shape[0]
    if L is None:
        L = int(np.log2(N / k)) - 1

    block_size = k  # block size at finest level
    Us, Vs, Ds = [], [], []

    A_current = A.copy()

    for ell in range(L, 0, -1):
        n_blocks = 2 ** ell
        bs = A_current.shape[0] // n_blocks  # should be 2k

        U_blocks, V_blocks, D_blocks = [], [], []

        for i in range(n_blocks):
            row_slice = slice(i * bs, (i + 1) * bs)

            # Extract off-diagonal block row r_i(A) and column c_i(A)
            r_i = np.hstack([A_current[row_slice, j*bs:(j+1)*bs]
                             for j in range(n_blocks) if j != i])
            c_i = np.vstack([A_current[j*bs:(j+1)*bs, row_slice]
                             for j in range(n_blocks) if j != i])

            # Best rank-k approximation via truncated SVD
            Ur, Sr, Vr = np.linalg.svd(r_i, full_matrices=False)
            U_i = Ur[:, :k]

            Uc, Sc, Vc = np.linalg.svd(c_i.T, full_matrices=False)
            V_i = Uc[:, :k]

            # Diagonal correction
            D_i = A_current[row_slice, row_slice] - (
                U_i @ (U_i.T @ A_current[row_slice, row_slice] @ V_i) @ V_i.T
            )

            U_blocks.append(U_i)
            V_blocks.append(V_i)
            D_blocks.append(D_i)

        U_ell = block_diag(*U_blocks)
        V_ell = block_diag(*V_blocks)
        D_ell = block_diag(*D_blocks)

        Us.append(U_ell)
        Vs.append(V_ell)
        Ds.append(D_ell)

        # Recurse: A^(ell) = U^T (A^(ell+1) - D^(ell)) V
        A_current = U_ell.T @ (A_current - D_ell) @ V_ell

    D_0 = A_current  # base case
    return Us, Vs, Ds, D_0


def greedy_hss_approx_matvec(matvec_A, matvec_AT, N, k, s=None):
    """
    Algorithm 4.1: Greedy HSS approximation from matvec access.
    Uses fresh sketches at each level for quasi-optimality guarantee.

    Key difference from Levitt-Martinsson: fresh Gaussian sketches
    Omega^(ell) at each level, rather than reusing a single sketch.

    Total matvec queries: O(s * L) where s >= 3k + 2
    Approximation factor: O(L) = O(log(N/k))
    """
    if s is None:
        s = 5 * k  # recommended setting

    L = int(np.log2(N / k)) - 1

    # Sample ALL sketch matrices upfront (but they're level-specific)
    sketches = {}
    for ell in range(L, 0, -1):
        dim = 2 ** (ell + 1) * k
        sketches[ell] = {
            'Omega': np.random.randn(dim, s),
            'Psi': np.random.randn(dim, s),
            'Omega_tilde': np.random.randn(dim, s),
            'Psi_tilde': np.random.randn(dim, s),
        }

    # Compute sketches Y = A * Omega, Z = A^T * Psi at each level
    # via recursion (each requires one matvec with A)
    # ... (recursive computation as per eq. 4.2)

    # At each level, use block nullification + PCPS for
    # near-optimal subspace recovery
    for ell in range(L, 0, -1):
        for i in range(2 ** ell):
            # Block nullification: find P_i = null(Omega_i)
            # Implicit sketch of block row: Y_i * P_i = r_i(A) * G_i
            # Use top-k SVD of sketched block row for U_i
            pass  # implementation follows Algorithm 4.1

    # Reconstruct telescoping factorization
    # B = U^(L) (...(U^(1) D^(0) V^(1)^T + D^(1))...) V^(L)^T + D^(L)
    pass


def block_diag(*blocks):
    """Construct block diagonal matrix."""
    sizes = [b.shape[0] for b in blocks]
    n = sum(sizes)
    result = np.zeros((n, n))
    offset = 0
    for b in blocks:
        s = b.shape[0]
        result[offset:offset+s, offset:offset+s] = b
        offset += s
    return result
```

**Key Implementation Insights:**

1. **Fresh sketches are critical for theory**: Reusing sketches (as in Levitt-Martinsson) works well empirically but introduces correlated errors across levels that prevent theoretical guarantees
2. **Block nullification enables implicit block-row sketching**: By nullifying the diagonal block contribution from a full matvec, one can implicitly sketch individual off-diagonal block rows
3. **PCPS avoids transpose queries of block rows**: Standard randomized SVD requires access to $r_i(A)^\top$; PCPS shows that the sketch $r_i(A) G_i$ itself provides a near-optimal column space
4. **Reference implementation**: https://github.com/NoahAmsel/HSS-approximation (Python, reproduces all experiments)

## References

- Amsel, N., Chen, T., Duman Keles, F., Halikias, D., Musco, C., Musco, C., & Persson, D. (2025). Quasi-optimal hierarchically semi-separable matrix approximation. *arXiv:2505.16937*.
- Levitt, J. & Martinsson, P.-G. (2024). Linear-complexity black-box randomized compression of hierarchically block separable matrices. *SIAM Journal on Scientific Computing*, 46(4), A2531-A2554.
- Cohen, M. B., Elder, S., Musco, C., Musco, C., & Persu, M. (2015). Dimensionality reduction for k-means clustering and low rank approximation. *STOC '15*.
- Martinsson, P.-G. (2011). A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix. *SIAM Journal on Matrix Analysis and Applications*.

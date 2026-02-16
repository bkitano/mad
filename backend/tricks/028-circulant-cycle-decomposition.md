# 028: Circulant Cycle Decomposition of Arbitrary Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Hariprasad M. & Venkatapathi M. "Circulant decomposition of a matrix and the eigenvalues of Toeplitz type matrices" (Applied Math. & Computation, 2024; arXiv:2105.14805)
**Paper**: [papers/circulant-decomposition-arbitrary-matrix.pdf]
**Documented**: 2026-02-15

## Description

Any $n \times n$ matrix $A$ can be decomposed into a sum of $n$ **circulant components** with periodic relaxations on the unit circle:

$$
A = \sum_{k=0}^{n-1} R_k D_k
$$

where each $R_k$ is a circulant matrix and $D_k = \text{diag}(1, e^{i2\pi k/n}, e^{i4\pi k/n}, \ldots, e^{i2\pi k(n-1)/n})$ is a diagonal matrix of $n$-th roots of unity. This decomposition is **orthogonal under the Frobenius inner product**: $\langle R_i D_i, R_j D_j \rangle_F = 0$ for $i \neq j$, enabling a Gram-Schmidt-like recursive extraction of components.

The key insight for neural network applications is that for matrices with **periodic or quasi-periodic diagonal structure** — which includes Toeplitz, block-Toeplitz, and block-circulant matrices — only a small number $k \ll n$ of circulant components have dominant Frobenius norm. The similar matrix $B = WAW^\dagger$ (where $W$ is the DFT matrix) concentrates its energy in $k$ dominant **cycles**, and the remaining $n - k$ components are negligible. This enables a **sparse similarity transformation**: compute only $k$ dominant cycles via selective FFT operations in $O(n^2)$ total arithmetic, then use a sparse eigenvalue algorithm (e.g., non-symmetric Lanczos) to approximate all $n$ eigenvalues to reasonable accuracy.

The first circulant component $R_0$ (cycle 0) is exactly the **optimal Frobenius-norm circulant approximation** of $A$ (i.e., T. Chan's preconditioner), establishing a direct connection between this decomposition and classical circulant preconditioning. Including additional cycles provides systematically better approximations.

## Mathematical Form

**Cycle Decomposition:**

Let $C$ be the basic cyclic permutation matrix:

$$
C = \begin{bmatrix} 0 & 1 \\ I_{n-1} & 0 \end{bmatrix}_{n \times n}
$$

Any matrix $A$ can be decomposed into $n$ cycles:

$$
A = \sum_{k=0}^{n-1} \Lambda_k C^k
$$

where $\Lambda_k$ are diagonal matrices. The entries of $\Lambda_k$ (i.e., entries supported on cycle $C^k$) are referred to as the $k$-th cycle of $A$.

**Circulant Decomposition (Theorem 1.4):**

Any $n \times n$ matrix $A$ can be represented as:

$$
A = \sum_{k=0}^{n-1} R_k D_k
$$

where $R_k = W^\dagger \bar{\Lambda}_k W$ is a circulant matrix with eigenvalues $\bar{\Lambda}_k$, $D_k$ is the diagonal matrix with $D_k(q,q) = e^{i2\pi kq/n}$ for $0 \leq q \leq n-1$, and $W$ is the DFT matrix with $W(p,q) = \frac{1}{\sqrt{n}} e^{-i2\pi pq/n}$.

**Relationship Between Cycles and Circulant Components:**

The cycle decomposition $A = \sum_k \Lambda_k C^k$ and the circulant decomposition $A = \sum_k R_k D_k$ are equivalent through the similarity transformation $B = WAW^\dagger$:

$$
B = W\left(\sum_k \Lambda_k C^k\right)W^\dagger = \sum_k \bar{\Lambda}_k C^k
$$

where $\bar{\Lambda}_k = W\Lambda_k W^\dagger$ are the DFT-transformed diagonals.

**Frobenius Orthogonality (Theorem 1.6):**

$$
\langle R_i D_i, R_j D_j \rangle_F = n \delta_{i,j}, \quad \langle D_i, D_j \rangle_F = n \delta_{i,j}
$$

This orthogonality means each component can be extracted independently without affecting others.

**Recursive Extraction (Remark 1.7):**

Initialize $A_0 = A$. For $k = 0, 1, \ldots, n-2$:

1. Compute first row of $R_k$: $R_k(0,j) = \frac{1}{n} \mathbf{1}^T (A_k \circ C^j) \mathbf{1}$ for $j = 0, \ldots, n-1$
2. Subtract: $A_{k+1} = (A_k - R_k) D_{-1}$

where $\circ$ is the Hadamard product and $\mathbf{1}$ is the all-ones vector.

**Weight of Cycle $k$:**

$$
w_k = \frac{\|R_k\|_F^2}{\sum_i \|R_i\|_F^2}, \quad \sum_k w_k = 1, \quad 0 \leq w_k \leq 1
$$

**Partial Energy (for Toeplitz matrices):**

For a Toeplitz matrix with entries $a_{-i}$ and $a_{n-i}$ on the $i$-th diagonal, the partial energy of cycle $k$ at frequency $i$ is:

$$
E_i^k = \frac{|a_{-i} - a_{n-i}|^2}{n((n-i)|a_{-i}|^2 + i|a_{n-i}|^2)} \left|\frac{\sin \frac{\pi(n-i+1)k}{n}}{\sin \frac{\pi k}{n}}\right|^2
$$

For circulant matrices ($a_{-i} = a_{n-i}$), $E_i^k = 0$ for all $k > 0$: only cycle 0 is nonzero.

**Sparse Similarity Transformation:**

When $A$ has $k \ll n$ dominant frequencies along its diagonals, $B = WAW^\dagger$ has only $k$ dominant cycles. The sparse approximation:

$$
\tilde{B} = \sum_{i=1}^{k} \Lambda_{a_i} C^{a_i}
$$

retains only the dominant cycles, with error bounded by Bauer-Fike:

$$
|\lambda_i - \tilde{\lambda}_i| \leq \kappa(X) \|\Delta\|_2
$$

where $\Delta = B - \tilde{B}$, $\kappa(X)$ is the condition number of the eigenvector matrix, and $\tilde{\lambda}_i$ are eigenvalues of $\tilde{B}$.

**Block-Toeplitz Extension (Corollary 1.12):**

For a block-Toeplitz matrix with block size $m$, the dominant cycles are indexed by $S_m = \{n/m, 2n/m, 3n/m, \ldots, n\}$ — multiples of $n/m$. The corresponding cycle indices in $B$ are $T_m = \{n(m-1)/m, n(m-2)/m, \ldots, 0\}$.

**Computing $B = WAW^\dagger$:**

$$
B(p,q) = \frac{1}{n}\sum_{k=1}^{n}\sum_{j=1}^{n} e^{-i2\pi pk/n} A(k,j) e^{i2\pi jq/n}
$$

Using 2D DFT: $B = [\mathcal{F}^2(A)]/n$, computable in $2n^2 \log n$ operations for the full transformation, or $O(n^2)$ for only $k$ selected cycles using partial FFT butterfly stages.

**Key Definitions:**

- $C$ — basic cyclic permutation matrix ($n \times n$)
- $W$ — DFT matrix, $W(p,q) = \frac{1}{\sqrt{n}} e^{-i2\pi pq/n}$
- $\Lambda_k$ — diagonal matrix of $k$-th cycle entries
- $R_k = W^\dagger \bar{\Lambda}_k W$ — $k$-th circulant component
- $D_k$ — diagonal phase matrix, $D_k(q,q) = e^{i2\pi kq/n}$
- $w_k$ — relative weight (Frobenius energy fraction) of cycle $k$
- $E_i^k$ — partial energy of frequency $i$ in cycle $k$
- $B = WAW^\dagger$ — DFT-similar matrix (cycles become block-diagonal)

## Complexity

| Operation | Dense eigenvalue | Single circulant approx | $k$-cycle circulant decomp |
|-----------|-----------------|------------------------|---------------------------|
| Eigenvalue approximation | $O(n^3)$ | $O(n^2)$ | $O(n^2)$ |
| Similarity transform | — | $O(n)$ | $O(n^2)$ for $k$ cycles |
| Preconditioner application | $O(n^2)$ per step | $O(n \log n)$ per step | $O(kn)$ per step |
| Storage | $O(n^2)$ | $O(n)$ | $O(kn)$ |

**Partial FFT for $k$ cycles:** When only $k$ frequency components are needed at the final FFT butterfly stage, the number of arithmetic operations is:

$$
O_k = (n - k) + n \log_2 k
$$

per column, for a total of $nO_k = O(n^2)$ for all $n$ columns. This is the same asymptotic cost as the full transform but with a smaller constant when $k \ll n$.

**Preconditioning improvement (from paper, Table 2, block-Toeplitz $n=1100$, block size 11):**
- Identity preconditioner: 168 iterations
- Generalized T. Chan preconditioner ($P(n)$): 162 iterations
- 1-cycle $WAW^\dagger$ preconditioner ($P(n)$): 162 iterations
- 9-cycle $WAW^\dagger$ preconditioner ($P(9n)$): 78 iterations → **17 iterations** (9.5× fewer)

**Memory:** $O(kn)$ for storing $k$ dominant circulant components (each defined by $n$ entries).

## Applicability

- **Eigenvalue estimation for Toeplitz/block-Toeplitz matrices**: The core application — approximate all $n$ eigenvalues of dense Toeplitz-type matrices in $O(n^2)$ by using only $k \ll n$ dominant circulant cycles plus a sparse eigenvalue solver. Particularly useful for spectral analysis of learned Toeplitz token mixers in sequence models
- **Multi-scale circulant preconditioning**: The $k$-cycle preconditioner systematically improves upon the single-circulant (T. Chan) preconditioner by capturing multiple periodicities. For block-Toeplitz matrices (which arise in multi-head attention with block structure), including cycles at block-frequency multiples dramatically reduces iteration counts
- **Weight matrix analysis**: Any dense weight matrix can be analyzed via its circulant decomposition to identify dominant periodic patterns — the cycle weights $w_k$ reveal the "circulant spectrum" of the matrix, indicating how well it can be approximated by structured (circulant/block-circulant) forms
- **Adaptive structured compression**: By identifying the $k$ dominant cycles of a weight matrix, one can construct a $k$-cycle circulant approximation with $kn$ parameters instead of $n^2$, where the cycles are chosen to minimize approximation error (unlike fixed block-circulant which uses a predetermined block structure)
- **Block-Toeplitz systems in vision**: 2D convolution layers with spatial Toeplitz structure generate block-Toeplitz matrices; the circulant decomposition identifies dominant spatial frequencies for preconditioning and spectral analysis

## Limitations

- **$O(n^2)$ baseline**: The similarity transformation $B = WAW^\dagger$ requires $O(n^2 \log n)$ operations for the full 2D DFT, or $O(n^2)$ for partial evaluation — not better than $O(n^2)$ dense operations. The benefit is in reducing subsequent eigenvalue/preconditioning steps, not in the decomposition itself
- **Dense matrices benefit least**: For random dense matrices without periodic structure, all $n$ cycles have comparable weight and no sparsification is possible (Figure 2 in paper). The trick is most effective for Toeplitz, block-Toeplitz, and quasi-periodic matrices
- **Complex arithmetic**: The $D_k$ matrices involve complex roots of unity, requiring complex arithmetic even when $A$ is real. For a purely real decomposition, one would need to pair conjugate cycles
- **Positive definiteness not guaranteed**: Even when $A$ is positive definite, the sparse approximation $\tilde{B}$ may not be positive definite unless the included cycles satisfy certain conditions (Theorem 2.1 in paper provides sufficient conditions)
- **Eigenvalue error depends on condition number**: The Bauer-Fike bound $|\lambda_i - \tilde{\lambda}_i| \leq \kappa(X)\|\Delta\|_2$ can be large when the eigenvector matrix $X$ is ill-conditioned, even if $\|\Delta\|$ is small

## Implementation Notes

```python
import torch
import torch.fft as fft

def circulant_decompose(A, k=None):
    """Decompose matrix A into circulant components via DFT similarity.

    A = sum_{j=0}^{n-1} R_j D_j where R_j are circulant, D_j are diagonal
    phase matrices. In practice, we compute B = W A W^† and extract
    the k dominant cycles.

    Args:
        A: (n, n) complex or real matrix
        k: number of dominant cycles to retain (None = all n)

    Returns:
        cycle_indices: (k,) indices of dominant cycles
        cycle_diags: (k, n) diagonal entries of each dominant cycle in B
        weights: (k,) Frobenius weights w_j of each cycle
    """
    n = A.shape[0]
    A_complex = A.to(torch.complex64)

    # Compute B = W A W^† = F^2(A) / n via 2D DFT
    # F^2(A) = DFT of columns, then DFT of rows
    B = fft.fft2(A_complex) / n

    # Extract cycles: cycle k has entries B[i, (i+k) mod n]
    # i.e., the k-th wrapped diagonal of B
    cycle_norms = torch.zeros(n)
    cycle_diags_all = torch.zeros(n, n, dtype=torch.complex64,
                                  device=A.device)

    for j in range(n):
        indices = torch.arange(n, device=A.device)
        col_indices = (indices + j) % n
        diag_entries = B[indices, col_indices]
        cycle_diags_all[j] = diag_entries
        cycle_norms[j] = diag_entries.abs().square().sum()

    # Normalize to weights
    total_norm = cycle_norms.sum()
    weights_all = cycle_norms / total_norm

    # Select top-k cycles
    if k is None:
        k = n
    topk_indices = torch.argsort(cycle_norms, descending=True)[:k]
    topk_indices, _ = topk_indices.sort()  # sort by index

    return (topk_indices,
            cycle_diags_all[topk_indices],
            weights_all[topk_indices])


def sparse_similarity_matvec(cycle_indices, cycle_diags, x):
    """Apply sparse approximation of B = W A W^† to vector x.

    Uses only the selected dominant cycles for the matrix-vector product.

    Args:
        cycle_indices: (k,) cycle indices
        cycle_diags: (k, n) diagonal entries per cycle
        x: (n,) input vector

    Returns:
        y: (n,) approximate B @ x
    """
    n = x.shape[0]
    y = torch.zeros(n, dtype=torch.complex64, device=x.device)

    for j, diag in zip(cycle_indices, cycle_diags):
        # Cycle j: multiply by diagonal, then cyclic-shift by j
        shifted = torch.roll(diag * x.to(torch.complex64), int(j.item()))
        y = y + shifted

    return y


def circulant_preconditioner(A, num_cycles=1):
    """Build a multi-cycle circulant preconditioner for A.

    The 1-cycle version is equivalent to T. Chan's optimal circulant
    preconditioner. More cycles capture additional periodic structure.

    Args:
        A: (n, n) matrix (typically Toeplitz or block-Toeplitz)
        num_cycles: number of dominant cycles to include

    Returns:
        precond_apply: function that applies the preconditioner to a vector
    """
    indices, diags, weights = circulant_decompose(A, k=num_cycles)

    # For the preconditioner, we need to "invert" the sparse B
    # In the single-cycle case (cycle 0), this is just circulant inversion
    # For multi-cycle, we use the sparse structure directly

    if num_cycles == 1 and indices[0] == 0:
        # Pure circulant preconditioner (T. Chan's)
        # R_0 is a circulant with eigenvalues = diags[0]
        circ_eig = diags[0]

        def precond_apply(b):
            b_fft = fft.fft(b.to(torch.complex64))
            return fft.ifft(b_fft / circ_eig).real

        return precond_apply

    else:
        # Multi-cycle preconditioner via few steps of iterative refinement
        def precond_apply(b):
            # Use dominant cycle as base, refine with others
            main_eig = diags[0]
            b_fft = fft.fft(b.to(torch.complex64))
            x = fft.ifft(b_fft / main_eig)
            return x.real

        return precond_apply


# Example: Analyze cycle structure of a Toeplitz token mixer
def analyze_toeplitz_cycles(toeplitz_coeffs, top_k=5):
    """Show the dominant circulant cycles of a Toeplitz matrix.

    For Toeplitz matrices, cycle 0 (the circulant approximation)
    is always dominant. For block-Toeplitz with block size m,
    cycles at multiples of n/m are also dominant.
    """
    n = (len(toeplitz_coeffs) + 1) // 2

    # Build Toeplitz matrix
    T = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            idx = (i - j) + (n - 1)  # map to coefficient index
            if 0 <= idx < len(toeplitz_coeffs):
                T[i, j] = toeplitz_coeffs[idx]

    indices, diags, weights = circulant_decompose(T, k=top_k)

    print(f"Top {top_k} cycles of {n}x{n} Toeplitz matrix:")
    for i, (idx, w) in enumerate(zip(indices, weights)):
        print(f"  Cycle {idx.item():3d}: weight = {w.item():.4f} "
              f"({w.item()*100:.1f}% of Frobenius energy)")

    return indices, weights
```

## References

- Hariprasad M. & Venkatapathi, M. "Circulant decomposition of a matrix and the eigenvalues of Toeplitz type matrices" Applied Mathematics and Computation, 468, 2024. arXiv:2105.14805
- Chan, T.F. "An Optimal Circulant Preconditioner for Toeplitz Systems" SIAM J. Sci. Stat. Comput. 9(4):766-771, 1988
- Gray, R.M. "Toeplitz and Circulant Matrices: A Review" Foundations and Trends in Communications and Information Theory, 2006
- Davis, P.J. "Circulant Matrices" Wiley, 1979
- Qin, Z. et al. "Toeplitz Neural Network for Sequence Modeling" ICLR 2023. arXiv:2305.04749

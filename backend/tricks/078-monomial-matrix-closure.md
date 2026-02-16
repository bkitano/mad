# 078: Monomial Matrix Closure (Generalized Permutation Matrices)

**Category**: algebraic
**Gain type**: efficiency
**Source**: Classical linear algebra; applied to SSMs by IBM Research (NeurIPS 2025)
**Paper**: [papers/ot4p-orthogonal-permutation-relaxation.pdf] (related), [papers/gyro-permutation-hierarchical-nm.pdf] (related)
**Documented**: 2026-02-15

## Description

A **monomial matrix** (also called a generalized permutation matrix or scaled permutation matrix) is a square matrix with exactly one nonzero entry per row and one nonzero entry per column. Equivalently, it is the product of an invertible diagonal matrix $D$ and a permutation matrix $P$:

$$
M = P \cdot D
$$

The key computational trick is that monomial matrices form a **closed group under multiplication**: the product of two monomial matrices is again monomial, and can be computed in $O(n)$ time (rather than $O(n^2)$ or $O(n^3)$ for general matrices). This closure property is what makes column-sparse transition matrices in PD-SSM efficient: chains of $T$ state transitions $A_T \cdot A_{T-1} \cdots A_1$ remain monomial at every prefix, enabling $O(n)$-per-step parallel scans over the entire sequence.

The algebraic structure is the **wreath product** $F^\times \wr S_n \cong (F^\times)^n \rtimes S_n$, a semidirect product of the diagonal matrices by the symmetric group acting via coordinate permutation. This group is a proper subgroup of $\text{GL}(n, F)$ that is vastly more expressive than the diagonal subgroup $(F^\times)^n$ (which can only represent abelian/solvable dynamics) while maintaining the same $O(n)$ per-operation cost.

## Mathematical Form

**Core Operation — $O(n)$ monomial-monomial product:**

Given $M_1 = P_1 D_1$ and $M_2 = P_2 D_2$ where $P_i$ are permutation matrices and $D_i$ are diagonal matrices, their product is:

$$
M_2 \cdot M_1 = (P_2 D_2)(P_1 D_1) = P_2 P_1 \cdot (P_1^\top D_2 P_1) D_1
$$

This is again monomial with:
- **Permutation part:** $P_{2 \cdot 1} = P_2 P_1$ — compose permutations in $O(n)$
- **Diagonal part:** $D_{2 \cdot 1} = (P_1^\top D_2 P_1) D_1$ — permute the diagonal of $D_2$ by $P_1^{-1}$, then multiply element-wise with $D_1$, all in $O(n)$

**In index notation:** If $P_k$ maps index $i \mapsto \sigma_k(i)$ and $D_k = \text{diag}(d_k^{(1)}, \ldots, d_k^{(n)})$, then:

$$
(M_2 M_1)_{ij} = d_2^{(\sigma_2^{-1}(\sigma_1(j)))} \cdot d_1^{(j)} \quad \text{at position } (\sigma_2(\sigma_1(j)), j)
$$

More compactly, the product has:
- Permutation: $\sigma_{2 \cdot 1} = \sigma_2 \circ \sigma_1$
- Diagonal entry $j$: $d_{2 \cdot 1}^{(j)} = d_2^{(\sigma_1(j))} \cdot d_1^{(j)}$

**Monomial-vector product ($O(n)$):**

$$
(P D) \mathbf{x} = P(D\mathbf{x})
$$

Scale each entry of $\mathbf{x}$ by the corresponding diagonal, then permute — two $O(n)$ operations.

**Cumulative product for parallel scan:**

For a sequence of monomial matrices $M_1, M_2, \ldots, M_T$, the prefix products $M_{t:1} = M_t \cdots M_1$ can be computed via a parallel associative scan where the binary operator is monomial multiplication. Each application of the operator costs $O(n)$, so:

$$
\text{Total work} = O(Tn), \quad \text{Parallel depth} = O(\log T)
$$

compared to $O(Tn^2)$ work for general matrix prefix products, or $O(Tn^3)$ using naive matrix multiplication.

**Key Definitions:**

- $P \in \{0, 1\}^{n \times n}$ — permutation matrix, i.e., $P^\top P = I$ and $P$ has exactly one 1 per row and column
- $D \in \mathbb{C}^{n \times n}$ — diagonal matrix, $D = \text{diag}(d^{(1)}, \ldots, d^{(n)})$
- $S_n$ — symmetric group on $n$ elements (all permutations)
- $(F^\times)^n$ — direct product of $n$ copies of the multiplicative group of the field
- $\wr$ — wreath product: $(F^\times)^n \rtimes S_n$ where $S_n$ acts by permuting coordinates

**Group axioms verification:**

| Property | Monomial matrices |
|----------|-------------------|
| Closure | $M_1 M_2$ is monomial (shown above) |
| Associativity | Inherited from matrix multiplication |
| Identity | $I = I_{\text{perm}} \cdot I_{\text{diag}}$ |
| Inverse | $(PD)^{-1} = D^{-1} P^\top = P^\top (P D^{-1} P^\top)$ — monomial |

## Complexity

| Operation | Dense | Diagonal | Monomial ($P \cdot D$) |
|-----------|-------|----------|----------------------|
| Matrix-vector $Mx$ | $O(n^2)$ | $O(n)$ | $O(n)$ |
| Matrix-matrix $M_2 M_1$ | $O(n^3)$ | $O(n)$ | $O(n)$ |
| Parallel scan (length $T$) | $O(Tn^2 \log T)$ | $O(Tn \log T)$ | $O(Tn \log T)$ |
| Storage per matrix | $O(n^2)$ | $O(n)$ | $O(2n)$ |
| Group expressivity | $\text{GL}(n)$ | Abelian $(F^\times)^n$ | Wreath $F^\times \wr S_n$ |

**Memory:** $O(2n)$ per monomial matrix (one permutation array of $n$ integers + $n$ diagonal entries), compared to $O(n^2)$ for dense.

## Applicability

- **Column-sparse transition matrices (PD-SSM):** The monomial closure property is the mathematical foundation enabling $O(n)$-per-step parallel scans with non-diagonal transition matrices. Each timestep produces a monomial $A_t = P_t D_t$, and the associative scan composes them without ever materializing dense matrices.
- **State tracking / finite automaton emulation:** Monomial matrices can represent the full symmetric group $S_n$ (via the permutation component), enabling emulation of non-solvable group automata that diagonal SSMs provably cannot express. This is strictly more expressive than diagonal transitions.
- **Stable recurrences:** With $|d^{(i)}| < 1$ for all diagonal entries, the spectral radius satisfies $\rho(PD) = \max_i |d^{(i)}| < 1$, ensuring BIBO stability regardless of the permutation.
- **Parallel associative scans:** The $O(n)$ binary operator is compatible with Blelloch-style work-efficient scans and chunkwise parallel scan algorithms.
- **Structured weight matrices:** Monomial matrices can serve as cheap, expressive mixing layers between blocks in block-diagonal architectures (e.g., replacing fixed shuffles in group-and-shuffle designs with learned monomial transforms).

## Limitations

- **No continuous mixing:** Each state dimension maps to exactly one other dimension — there is no weighted combination or "soft" routing between states. This limits the expressivity for tasks requiring smooth interpolation.
- **Discrete permutation component:** Learning the permutation $P$ requires discrete optimization (Gumbel-Softmax, Sinkhorn, or OT4P relaxation), adding training complexity.
- **Expressivity gap with dense:** While monomial matrices can represent all permutations ($n!$ elements), they cannot represent arbitrary rotations or projections. The wreath product $F^\times \wr S_n$ has $|\text{params}| = O(n)$ compared to $O(n^2)$ for the full general linear group.
- **Complex eigenvalue structure:** The eigenvalues of a monomial matrix are determined by its cycle structure and the diagonal entries within each cycle. For a $k$-cycle with diagonal entries $d_1, \ldots, d_k$, the eigenvalues are the $k$-th roots of $d_1 d_2 \cdots d_k$. This can create complex oscillatory dynamics that may be hard to control during training.

## Implementation Notes

```python
import torch

def monomial_multiply(perm1, diag1, perm2, diag2):
    """
    Compute the product M2 * M1 = (P2 D2)(P1 D1) in O(n).

    Args:
        perm1: (n,) int — permutation indices for P1
        diag1: (n,) complex — diagonal entries for D1
        perm2: (n,) int — permutation indices for P2
        diag2: (n,) complex — diagonal entries for D2

    Returns:
        perm_out: (n,) int — composed permutation
        diag_out: (n,) complex — composed diagonal
    """
    # Compose permutations: sigma_out = sigma2 o sigma1
    perm_out = perm2[perm1]

    # Compose diagonals: d_out[j] = d2[sigma1[j]] * d1[j]
    diag_out = diag2[perm1] * diag1

    return perm_out, diag_out


def monomial_matvec(perm, diag, x):
    """
    Compute (P D) x in O(n).

    Args:
        perm: (n,) int — permutation indices
        diag: (n,) complex — diagonal entries
        x: (n,) — input vector

    Returns:
        y: (n,) — output vector P @ (D @ x)
    """
    # Scale by diagonal, then permute
    return (diag * x)[perm]  # Equivalent to P @ diag(d) @ x


def monomial_scan(perms, diags):
    """
    Compute prefix products of monomial matrices via sequential scan.
    For parallel version, use with associative scan primitive.

    Args:
        perms: (T, n) int — permutation per timestep
        diags: (T, n) complex — diagonal per timestep

    Returns:
        cum_perms: (T, n) int — cumulative permutations
        cum_diags: (T, n) complex — cumulative diagonals
    """
    T, n = perms.shape
    cum_perms = torch.zeros_like(perms)
    cum_diags = torch.zeros_like(diags)

    # Initialize
    cum_perms[0] = perms[0]
    cum_diags[0] = diags[0]

    for t in range(1, T):
        cum_perms[t], cum_diags[t] = monomial_multiply(
            cum_perms[t-1], cum_diags[t-1],
            perms[t], diags[t]
        )

    return cum_perms, cum_diags
```

## References

- Classical: Generalized permutation matrices and wreath products are standard constructions in group theory and linear algebra.
- Wikipedia: Generalized permutation matrix. https://en.wikipedia.org/wiki/Generalized_permutation_matrix
- IBM Research (2025). Efficient Transition Matrices to Enable State Tracking in State-Space Models. NeurIPS 2025. arXiv:2509.22284.
- Tran, V.H. et al. (2024). Monomial Matrix Group Equivariant Neural Functional Networks. NeurIPS 2024. arXiv:2409.11697.
- Merrill et al. (2024). The Illusion of State in State-Space Models. ICML 2024. (Diagonal SSM expressivity limitations)

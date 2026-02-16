# 143: Woodbury Resolvent Identity

**Category**: decomposition
**Gain type**: efficiency
**Source**: Woodbury (1950); applied to SSMs in Gu et al. (S4, 2022)
**Documented**: 2026-02-10

## Description

The Woodbury matrix identity provides a formula for the inverse of a matrix that is a low-rank perturbation of a diagonal (or easily invertible) matrix. In the context of SSMs, this is the key algebraic trick that makes DPLR matrices computationally tractable: instead of inverting a dense $N \times N$ matrix at each frequency, the Woodbury identity decomposes the resolvent $(zI - A)^{-1}$ into a diagonal inverse plus a rank-$r$ correction. For S4's HiPPO matrix (rank 1 or 2), this reduces each resolvent evaluation to Cauchy dot products — elementwise divisions and a small matrix solve — eliminating the $O(N^3)$ inversion bottleneck entirely.

## Mathematical Form

**Core Operation:**

For $A = \Lambda - PQ^*$ (DPLR with diagonal $\Lambda$, low-rank $P, Q \in \mathbb{C}^{N \times r}$):

$$
(zI - A)^{-1} = (zI - \Lambda + PQ^*)^{-1}
$$

Applying the Woodbury identity:

$$
(zI - \Lambda + PQ^*)^{-1} = D_z - D_z P (I + Q^* D_z P)^{-1} Q^* D_z
$$

where $D_z = (zI - \Lambda)^{-1}$ is trivially diagonal.

**Key Definitions:**

- $\Lambda \in \mathbb{C}^{N \times N}$ — diagonal matrix
- $P, Q \in \mathbb{C}^{N \times r}$ — low-rank factors (typically $r = 1$ or $2$)
- $D_z = \text{diag}\left(\frac{1}{z - \lambda_i}\right)$ — diagonal resolvent

**Term-by-Term Decomposition:**

Each term decomposes into Cauchy-like operations:

- $D_z = \text{diag}(1/(z - \lambda_i))$ — elementwise division
- $D_z P$ = column scaling by $1/(z - \lambda_i)$ — Cauchy kernel
- $Q^* D_z P \in \mathbb{C}^{r \times r}$ — tiny matrix ($r=1$ or $2$ for HiPPO)
- The $r \times r$ inverse is $O(r^3) = O(1)$ for fixed $r$

**Generating Function:**

For the generating function $\hat{K}(z) = \bar{C}(zI - \bar{A})^{-1}\bar{B}$:

$$
\hat{K}(z) = (\tilde{C}^\top D_z \tilde{B}) - (\tilde{C}^\top D_z \tilde{P})(1 + \tilde{Q}^\top D_z \tilde{P})^{-1}(\tilde{Q}^\top D_z \tilde{B})
$$

= four Cauchy dot products + a scalar division (for rank 1).

## Complexity

| Operation | Naive | With Woodbury |
|-----------|-------|---------------|
| Matrix inverse | $O(N^3)$ | — |
| LU factorization | $O(N^2)$ | — |
| Diagonal + rank-$r$ | — | $O(N + r^2)$ |
| **Per frequency** | $O(N^2)$ | $O(N)$ (since $r \ll N$) |

**Total for $L$ frequencies:** $O(NL + L \log L)$ including FFT

**Memory:** $O(N + r)$ vs $O(N^2)$

## Applicability

- Core algebraic reduction in S4 that makes DPLR training tractable
- Applies whenever the state matrix has low-rank structure: DPLR, NPLR, or any $\Lambda + UV^\top$ form
- Also used in classical control theory for transfer function evaluation
- Extends to any resolvent computation $(zI - A)^{-1}b$ where $A$ is structured
- The same identity underlies the Sherman-Morrison formula (rank-1 special case) used throughout scientific computing

## Limitations

- Requires explicit low-rank factorization $A = \Lambda + PQ^*$ — not applicable to general dense $A$
- Numerical stability depends on $(zI - \Lambda)$ being well-conditioned; when $z \approx \lambda_i$, the diagonal inverse has large entries
- The rank $r$ must be small for the trick to be efficient; high-rank corrections lose the advantage
- Only helps with resolvent/inverse computations — does not directly accelerate matrix exponentials or other functions of $A$

## Implementation Notes

```python
# Woodbury resolvent (rank-1 case)
def woodbury_resolvent(z, Lambda, p, q, b):
    # (zI - Lambda + pq*)^{-1} b
    D_z = 1.0 / (z - Lambda)  # (N,)
    D_z_b = D_z * b           # (N,)
    D_z_p = D_z * p           # (N,)

    # Scalar correction for rank-1
    correction = (q.conj() @ D_z_p)  # scalar
    scale = (q.conj() @ D_z_b) / (1 + correction)

    return D_z_b - scale * D_z_p
```

## References

- Woodbury (1950). Inverting Modified Matrices. Statistical Research Group Memo Report.
- Gu, Goel, Ré (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR.
- Hager (1989). Updating the Inverse of a Matrix. SIAM Review. (Survey of Woodbury-type identities)
- Gu, Gupta, et al. (2022). The Annotated S4. ICLR Blog Track.

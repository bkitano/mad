# 157: Neumann-Series CWY Inverse Approximation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Moreno Arcas, Sanchis, Civera & Juan (2025) — HOFT; Schreiber & Van Loan (1989)
**Paper**: [papers/hoft-householder-finetuning.pdf]
**Documented**: 2026-02-15

## Description

When using the Compact WY (CWY) representation to accumulate a product of $r$ Householder reflections as $Q = I - U S^{-1} U^\top$, the standard approach requires computing $S^{-1}$ via back-substitution on an $r \times r$ upper-triangular matrix — a sequential $O(r^2)$ operation that cannot use tensor cores. The **Neumann-series approximation** replaces this exact triangular inverse with a **truncated power series** that uses only diagonal inversions and matrix multiplications, converting the entire CWY computation into tensor-core-friendly operations.

The key insight is that $S = D + A$ where $D = \frac{1}{2}I$ is a diagonal matrix and $A = \text{striu}(U^\top U)$ is strictly upper-triangular. Since $D^{-1}A$ is strictly upper-triangular, its spectral radius is zero (all eigenvalues are zero), and the Neumann series $\sum_{i=0}^{\infty}(-D^{-1}A)^i$ converges in **exactly $r-1$ terms** (it is a finite polynomial). However, truncating to just the first 2 terms provides a remarkably accurate approximation:

$$
S^{-1} \approx D^{-1} - D^{-1}AD^{-1} = 2I - 4\,\text{striu}(U^\top U)
$$

This replaces back-substitution with a single matmul ($U^\top U$) and elementwise scaling — both perfectly parallelizable on GPUs. The approximation error grows with $r$ but remains small for practical values ($r \leq 128$), with the HOFT paper reporting orthogonality errors below $10^{-3}$ even at rank 128 on 4096-dimensional matrices.

## Mathematical Form

**Setup: CWY Representation**

Given $r$ Householder reflections with normalized vectors $u_1, \ldots, u_r \in \mathbb{R}^m$ ($\|u_i\| = 1$), define:

$$
U = [u_1 \mid \cdots \mid u_r] \in \mathbb{R}^{m \times r}
$$

The product of reflections satisfies (Joffrain et al., 2006):

$$
Q = \prod_{i=1}^{r} \left(I - \frac{u_i u_i^\top}{\tau_i}\right) = I - U S^{-1} U^\top
$$

where $S \in \mathbb{R}^{r \times r}$ is upper-triangular, computed as:

$$
S = \frac{1}{2}I + \text{striu}(U^\top U)
$$

**Neumann-Series Expansion:**

Decompose $S = D + A$ where:
- $D = \frac{1}{2}I$ (diagonal part)
- $A = \text{striu}(U^\top U)$ (strictly upper-triangular part)

Then:

$$
S^{-1} = (D + A)^{-1} = (I + D^{-1}A)^{-1} D^{-1} = \left(\sum_{i=0}^{\infty} (-D^{-1}A)^i\right) D^{-1}
$$

Since $D^{-1} = 2I$ and $D^{-1}A = 2A$ is strictly upper-triangular (nilpotent of degree $r$), the series is **exact** at $r$ terms:

$$
S^{-1} = \sum_{i=0}^{r-1} (-2A)^i \cdot 2I = 2I - 4A + 8A^2 - 16A^3 + \cdots
$$

**Truncated Approximation (2 terms):**

$$
S^{-1} \approx D^{-1} - D^{-1}AD^{-1} = 2I - 4\,\text{striu}(U^\top U)
$$

Substituting into the CWY form:

$$
Q \approx I - U\left(2I - 4\,\text{striu}(U^\top U)\right)U^\top = I + U\left(D^{-1}AD^{-1} - D^{-1}\right)U^\top
$$

**Expanded final form (Eq. 4 in HOFT):**

$$
Q \approx I + U\left(D^{-1}AD^{-1} - D^{-1}\right)U^\top
$$

**Key Definitions:**

- $U \in \mathbb{R}^{m \times r}$ — matrix of $r$ unit-normalized Householder vectors
- $S \in \mathbb{R}^{r \times r}$ — upper-triangular CWY factor
- $D = \frac{1}{2}I_r$ — diagonal part of $S$
- $A = \text{striu}(U^\top U) \in \mathbb{R}^{r \times r}$ — strictly upper-triangular part
- $r$ — number of Householder reflections (rank)
- $m$ — dimension of the vectors

**Connection to UT Transform (DeltaNet):**

In DeltaNet, the UT transform uses the **lower-triangular** analog: $T^{-1} = (I + L)^{-1}B$ where $L$ is strictly lower-triangular. The same Neumann series applies:

$$
(I + L)^{-1} = I - L + L^2 - L^3 + \cdots
$$

Truncating to 2 terms: $(I + L)^{-1} \approx I - L$, which replaces forward substitution with a single elementwise negation of $L$. This approximation is most accurate when the entries of $L$ (which equal $-\beta_i k_i^\top k_j$ for $i > j$) are small — i.e., when the key vectors are approximately orthogonal or the learning rates $\beta$ are small.

## Complexity

| Operation | Exact CWY | Neumann (2-term) | Neumann ($p$-term) |
|-----------|-----------|------------------|-------------------|
| Compute $U^\top U$ | $O(mr^2)$ matmul ✓ | $O(mr^2)$ matmul ✓ | $O(mr^2)$ matmul ✓ |
| Compute $S^{-1}$ | $O(r^2)$ sequential | $O(r^2)$ parallel ✓ | $O(pr^2)$ matmul ✓ |
| Compute $S^{-1}U^\top$ | $O(mr)$ matmul ✓ | $O(mr)$ matmul ✓ | $O(mr)$ matmul ✓ |
| **Total** | $O(mr^2) + O(r^2)$ seq | $O(mr^2)$ **all parallel** | $O(mr^2 + pr^2)$ **all parallel** |

✓ = maps to tensor cores / parallel

**The key gain:** The $O(r^2)$ sequential forward/back-substitution is eliminated entirely. All operations become matrix multiplications or elementwise operations that map to tensor cores.

**Memory:** Same as exact CWY: $O(mr)$ for $U$ + $O(r^2)$ for intermediate.

**Approximation error (from HOFT Figure 1):**

| Rank $r$ | $m = 256$ | $m = 512$ | $m = 1024$ | $m = 2048$ | $m = 4096$ |
|----------|-----------|-----------|------------|------------|------------|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 4 | ~$10^{-5}$ | ~$10^{-5}$ | ~$10^{-5}$ | ~$10^{-5}$ | ~$10^{-5}$ |
| 16 | ~$10^{-3}$ | ~$10^{-3}$ | ~$10^{-3}$ | ~$10^{-3}$ | ~$10^{-3}$ |
| 64 | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ |
| 128 | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ | ~$10^{-2}$ |

Error metric: $\|I - Q_U Q_U^\top\|_F / \sqrt{n}$ (deviation from orthogonality).

## Applicability

- **DeltaNet / DeltaProduct chunkwise training:** The UT transform's forward substitution step $(I + L)^{-1}$ can be replaced by a truncated Neumann series, converting the only sequential bottleneck in the per-chunk computation to pure matmuls. For chunk size $C = 64$ and $d = 256$, the forward substitution is already negligible, but for larger chunk sizes ($C = 256$ or $C = 512$, as enabled by TFLA), the Neumann approximation becomes significant.

- **PaTH Attention blockwise algorithm:** The per-block UT transform computation $T^{-1} = (I + L)^{-1}D$ can use truncated Neumann series, enabling fully-parallel per-block preprocessing.

- **Orthogonal fine-tuning (HOFT/SHOFT):** The primary application in the paper — parameterize orthogonal adapter matrices via CWY Householder products with fast approximate inverse, achieving 72.1% speedup over OFT and 732.5% speedup over HRA.

- **Orthogonal RNNs:** Any model using CWY-parameterized orthogonal transition matrices can replace the triangular solve with this approximation for faster per-step computation.

## Limitations

- **Approximation error grows with rank $r$:** The 2-term truncation introduces error proportional to the magnitude of $A^2 = (\text{striu}(U^\top U))^2$. When Householder vectors have large mutual inner products (non-orthogonal), the error increases. For $r \ll m$ (typical in PEFT), the error is small.

- **Not exact for full-rank ($r = m$):** When $r = m$, the full Neumann series requires $m-1$ terms (each a matrix power), which costs $O(m^3)$ — worse than back-substitution. The advantage is only for truncated (few-term) approximations.

- **Breaks strict orthogonality:** The resulting matrix $Q$ is only approximately orthogonal. For applications requiring exact orthogonality (e.g., volume-preserving normalizing flows), this approximation is unsuitable without post-hoc correction.

- **DeltaNet context:** In DeltaNet, the UT transform operates on generalized Householder reflections with $\beta \in [0, 2]$ (not necessarily $\beta = 2$). The lower-triangular matrix $L_{ij} = -\beta_i(k_i^\top k_j)$ may have larger entries than the standard CWY case, potentially increasing approximation error.

- **Higher-order terms are cheap but sequential:** Computing $A^2, A^3, \ldots$ requires sequential matrix multiplications (each depends on the previous). However, these are tensor-core-friendly matmuls of size $r \times r$, so for small $r$ they are fast.

## Implementation Notes

```python
import torch

def cwy_neumann_approximate(U, num_terms=2):
    """
    Compute approximate CWY representation using Neumann series.

    Q = I - U S^{-1} U^T  where  S^{-1} ≈ truncated Neumann series

    Args:
        U: (m, r) - normalized Householder vectors (unit columns)
        num_terms: number of Neumann series terms (2 = fast, r = exact)

    Returns:
        S_inv_approx: (r, r) - approximate inverse of S
    """
    r = U.shape[1]

    # Step 1: Compute G = U^T U  (TENSOR CORE matmul: r×m @ m×r → r×r)
    G = U.T @ U  # (r, r)

    # Step 2: Extract strictly upper-triangular part A
    A = torch.triu(G, diagonal=1)  # (r, r), strict upper triangle

    # Step 3: D^{-1} = 2I (since D = 0.5 * I)
    D_inv = 2.0

    if num_terms == 2:
        # Fast 2-term approximation:
        # S^{-1} ≈ D^{-1} - D^{-1} A D^{-1} = 2I - 4A
        S_inv = D_inv * torch.eye(r, device=U.device) - D_inv**2 * A
    else:
        # General p-term Neumann series:
        # S^{-1} = sum_{i=0}^{p-1} (-D^{-1} A)^i * D^{-1}
        neg_D_inv_A = -D_inv * A  # (-2A), strictly upper-triangular
        power = torch.eye(r, device=U.device)  # (-D^{-1}A)^0 = I
        S_inv = D_inv * power.clone()  # First term: D^{-1}

        for i in range(1, min(num_terms, r)):
            power = power @ neg_D_inv_A  # (-D^{-1}A)^i
            S_inv = S_inv + D_inv * power  # Add D^{-1} * (-D^{-1}A)^i

    return S_inv


def cwy_matvec_approximate(U, S_inv, x):
    """
    Compute Q @ x = (I - U S^{-1} U^T) x using precomputed approximate S^{-1}.
    All operations are parallel / tensor-core friendly.
    """
    # u = U^T x  (gemv, parallelizable)
    u = U.T @ x
    # s = S^{-1} u  (gemv with r×r matrix, parallelizable)
    s = S_inv @ u
    # result = x - U s  (gemv, parallelizable)
    return x - U @ s


# Application to DeltaNet UT transform:
def ut_transform_neumann(K, beta, num_terms=2):
    """
    Approximate UT transform for DeltaNet using Neumann series.
    Replaces forward substitution with matmuls.

    Args:
        K: (C, d) - key vectors within a chunk
        beta: (C,) - learning rates

    Returns:
        W: (C, d) - WY decay factors
    """
    C, d = K.shape

    # Scale keys by beta
    K_beta = K * beta.unsqueeze(-1)  # (C, d)

    # Build L = tril(diag(beta) @ K @ K^T, -1)
    # TENSOR CORE matmul: (C, d) @ (d, C) -> (C, C)
    L = (K_beta @ K.T).tril(-1)  # Strictly lower-triangular

    if num_terms == 2:
        # (I + L)^{-1} ≈ I - L
        # T ≈ (I - L) @ diag(beta)
        T = torch.diag(beta) - L @ torch.diag(beta)
    else:
        # General: (I + L)^{-1} = I - L + L^2 - L^3 + ...
        T_inv = torch.eye(C, device=K.device)
        power = torch.eye(C, device=K.device)
        for i in range(1, min(num_terms, C)):
            power = power @ (-L)
            T_inv = T_inv + power
        T = T_inv @ torch.diag(beta)

    # W = T @ K  (TENSOR CORE matmul)
    W = T @ K
    return W
```

**GPU efficiency analysis:**

- **Eliminates the only sequential bottleneck:** The forward substitution loop in the UT transform (lines 17-18 in the DeltaNet kernel) is replaced by a single matmul $L \times \text{diag}(\beta)$ for the 2-term case, or a chain of $p$ matmuls for the $p$-term case.
- **All operations are matmuls:** $U^\top U$, $L \cdot \text{diag}(\beta)$, $T \cdot K$ — all map to tensor cores.
- **Trade-off is clear:** For DeltaNet with chunk size $C = 64$ and $d = 256$, the forward substitution is ~4K FLOPs vs ~2M FLOPs for the matmuls — negligible. The Neumann trick matters when $C \gg d$ (large chunks) or when combined with TFLA's two-level tiling where inner tiles may have their own forward substitution.

**Timing results (from HOFT, Appendix D):**

| Method | Avg. Time/Epoch | Speedup vs OFT | Speedup vs HRA |
|--------|----------------|----------------|----------------|
| OFT | baseline | 1.0× | — |
| BOFT | — | ~1.0× | — |
| HRA | — | — | 1.0× |
| **HOFT** | — | **1.72×** | **8.33×** |
| **SHOFT** | — | **1.55×** | **7.50×** |

## References

- Moreno Arcas, A., Sanchis, A., Civera, J., & Juan, A. (2025). HOFT: Householder Orthogonal Fine-tuning. arXiv:2505.16531.
- Joffrain, T., Low, T. M., Quintana-Ortí, E. S., van de Geijn, R. A., & Van Zee, F. G. (2006). Accumulating Householder Transformations, Revisited. ACM Trans. Math. Softw., 32(2), 169–179.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. SIAM J. Sci. Stat. Comput., 10(1), 53–57.
- Likhosherstov, V., Davis, J., Choromanski, K., & Weller, A. (2021). CWY Parametrization: a Solution for Parallelized Optimization of Orthogonal and Stiefel Matrices. AISTATS 2021.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024.

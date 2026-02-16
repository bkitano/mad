# 184: Structured WY QR Row Update

**Category**: decomposition
**Gain type**: efficiency
**Source**: Kressner (2009), "A note on using compact WY representations for updating QR decompositions"
**Paper**: [papers/compact-wy-qr-update.pdf]
**Documented**: 2026-02-15

## Description

When updating a QR decomposition by appending new rows — computing the QR factorization of $\begin{bmatrix} A \\ B \end{bmatrix}$ where $A$ is already upper triangular — the Householder vectors in the compact WY representation inherit a **specific sparsity pattern**: $V = \begin{bmatrix} I_k \\ 0 \\ V_3 \end{bmatrix}$ where $V_3 \in \mathbb{R}^{m \times k}$ is dense and the middle $(n-k) \times k$ block is zero. Exploiting this structure reduces the cost of applying the compact WY transform from $O(3k^2p + 4mp)$ (LAPACK 3.2 with zero-scanning) down to $O(2k^2p + 4mp)$ by eliminating redundant multiplications with identity and zero blocks.

This is directly relevant to **streaming DeltaNet-like models** where the state matrix is incrementally updated by "appending" new token observations (rows) to an existing factorization. The structured update avoids recomputing from scratch and reduces the dominant GEMM cost.

## Mathematical Form

**Core Problem:**

Given $A \in \mathbb{R}^{n \times n}$ upper triangular and $B \in \mathbb{R}^{m \times n}$, compute:

$$
\begin{bmatrix} A \\ B \end{bmatrix} = Q \begin{bmatrix} R \\ 0 \end{bmatrix}
$$

**Structured Householder Vectors:**

The block QR factorization of $\begin{bmatrix} A \\ B \end{bmatrix}$ generates Householder reflectors whose vector matrix $V$ in the compact WY form $Q = I - VTV^\top$ has the structure:

$$
V = \begin{bmatrix} I_k \\ 0_{(n-k) \times k} \\ V_3 \end{bmatrix}, \quad V_3 \in \mathbb{R}^{m \times k}
$$

where $k$ is the block size and the middle $(n-k)$ rows are zero.

**Efficient Application:**

To apply $(I - VTV^\top)$ to a matrix $C = \begin{bmatrix} C_1 \\ C_2 \\ C_3 \end{bmatrix}$ with $C_1 \in \mathbb{R}^{k \times p}$, $C_2 \in \mathbb{R}^{(n-k) \times p}$, $C_3 \in \mathbb{R}^{m \times p}$:

$$
W = (C_1^\top + C_3^\top V_3) T^\top
$$

$$
C_1 \leftarrow C_1 - W^\top, \quad C_3 \leftarrow C_3 - V_3 W^\top
$$

The block $C_2$ is **untouched** — it does not participate in the computation at all.

**Standard WY Application (for comparison):**

$$
W = V^\top C = C_1 + V_3^\top C_3 \quad \text{(skipping zero block)}
$$
$$
W \leftarrow TW
$$
$$
C \leftarrow C - VW
$$

The standard approach requires multiplying with the full $V$ matrix including the $I_k$ block. The structured approach avoids this by directly computing $W = (C_1^\top + C_3^\top V_3)T^\top$, saving $k^2 p$ flops per application.

**T-factor Structure:**

The upper-triangular $T \in \mathbb{R}^{k \times k}$ in the compact WY representation is computed by the standard bordering recurrence:

$$
T_{1:j-1,j} = -\beta_j \cdot T_{1:j-1,1:j-1} \cdot (V_{:,1:j-1}^\top V_{:,j}), \quad T_{jj} = \beta_j
$$

But due to the $[I_k; 0; V_3]$ structure, the inner product $V_{:,i}^\top V_{:,j}$ simplifies to $\delta_{ij} + V_{3,i}^\top V_{3,j}$, which can be batch-computed as $I + V_3^\top V_3$.

## Complexity

| Operation | Standard LAPACK | LAPACK 3.2 (zero-scan) | Structured WY |
|-----------|----------------|----------------------|---------------|
| $W \leftarrow V^\top C$ | $k^2p + 2(m+n-k)p$ | $k^2p + 2mp$ | $2mp$ |
| $W \leftarrow TW$ | $k^2p$ | $k^2p$ | $k^2p$ |
| $C \leftarrow C - VW$ | $k^2p + 2(m+n-k)p$ | $k^2p + 2mp$ | $k^2p + 2mp$ |
| **Total** | $3k^2p + 4(m+n-k)p$ | $3k^2p + 4mp$ | $2k^2p + 4mp$ |

**Savings:** $k^2 p$ flops per block application, which is significant when $k$ is large relative to $m$.

**Memory:** $O(mk)$ for $V_3$ + $O(k^2)$ for $T$. No need to store the zero block or identity block of $V$.

**Empirical speedup (from paper, $n = 4000$):**
- $m \leq 100$: **30–40% wall-clock time reduction** vs LAPACK 3.2
- $m = 1$ (single row update): **up to 45% reduction**
- $m \gg n$: difference becomes negligible (dominated by the $4mp$ terms in both)

## Applicability

- **Streaming DeltaNet / linear attention**: When processing tokens in batches (chunks), the state update can be viewed as appending $m$ new rows (key-value pairs) to an existing triangular factor. The structured WY update avoids recomputing the full QR factorization.
- **Online / incremental learning**: Any system that maintains a QR factorization and periodically receives new data rows benefits from this structured update.
- **Batched QR on GPU**: When performing many small QR updates (e.g., per-head, per-layer), the $k^2p$ savings per update accumulate across the batch dimension.
- **Two-stage eigenvalue/SVD algorithms**: Band reduction and bulge-chasing algorithms that use QR updates to progressively reduce matrix structure.

## Limitations

- **Requires upper-triangular structure**: The sparsity pattern $[I_k; 0; V_3]$ only arises when the top block is already triangular. General QR factorization does not have this structure.
- **Savings are $O(k^2 p)$**, a lower-order term: When $m \gg k$, the $4mp$ terms dominate in both methods, making the savings negligible. The trick is most beneficial when $m \leq k$ (few new rows per update).
- **Custom kernel required**: The structured application cannot be implemented using standard LAPACK `xLARFB` calls — it requires a tailored routine that understands the $[I_k; 0; V_3]$ structure.
- **Not directly a GPU tensor-core optimization**: The savings are in FLOP count, not in memory access pattern. On GPUs where bandwidth is the bottleneck, the actual speedup may be smaller than the FLOP reduction suggests.
- **Block size constraint**: The block size $k$ must match the panel width used in the blocked QR algorithm (typically 32–64 for LAPACK). This is fixed by the factorization, not tunable.

## Implementation Notes

```python
import torch

def structured_wy_update(A, B, block_size=64):
    """
    QR update: compute QR of [A; B] where A is upper triangular.
    Exploits structured V = [I_k; 0; V3] pattern.

    Args:
        A: (n, n) upper triangular
        B: (m, n) new rows to append
        block_size: k, panel width

    Returns:
        R: (n, n) updated upper triangular factor
        V3: (m, k) dense part of Householder vectors
        T: (k, k) upper-triangular T factor
    """
    n = A.shape[0]
    m = B.shape[0]
    k = block_size

    # Stack [A; B] and factorize panel
    # Only the first k columns and the B block participate
    # Standard Householder on the (k+m) x k panel
    panel = torch.cat([A[:k, :k], B[:, :k]], dim=0)  # (k+m, k)

    # ... (standard blocked Householder on panel) ...
    # This produces V3 (m x k) and T (k x k)

    return R, V3, T


def apply_structured_wy(C1, C2, C3, V3, T):
    """
    Apply structured WY transform: C <- (I - V T V^T) C
    where V = [I_k; 0; V3].

    C1: (k, p)    — top block (modified)
    C2: (n-k, p)  — middle block (UNTOUCHED)
    C3: (m, p)    — bottom block (modified)
    V3: (m, k)    — dense part of V
    T:  (k, k)    — upper triangular T factor

    Cost: 2k^2 p + 4mp  (saves k^2 p vs standard)
    """
    # W = (C1^T + C3^T @ V3) @ T^T  — two GEMMs
    W = (C1.T + C3.T @ V3) @ T.T   # (p, k)

    # Update only C1 and C3; C2 is untouched
    C1 -= W.T          # (k, p)
    C3 -= V3 @ W.T     # (m, p) — one GEMM

    return C1, C2, C3


# Connection to DeltaNet streaming:
# In a streaming setting, A represents the state from previous chunks
# (already factored as upper triangular R), and B represents new
# key-value observations. The structured WY update incrementally
# incorporates new tokens without recomputing the full factorization.
```

## References

- Kressner, D. (2009). A note on using compact WY representations for updating QR decompositions. Technical report, ETH Zurich.
- Golub, G. H. & Van Loan, C. F. (1996). Matrix Computations. 3rd ed., Johns Hopkins University Press.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. SIAM J. Sci. Stat. Comput., 10(1), 53–57.
- Anderson, E. et al. (1999). LAPACK Users' Guide. 3rd ed., SIAM.
- Yang, S. et al. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024.

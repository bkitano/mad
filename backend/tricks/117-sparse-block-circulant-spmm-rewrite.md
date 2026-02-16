# 117: Sparse Block-Circulant SpMV-to-SpMM Rewrite

**Category**: kernel
**Gain type**: efficiency
**Source**: Romero Alcalde et al., "A Fast Sparse Block Circulant Matrix Vector Product" (Euro-Par 2014)
**Paper**: [papers/sparse-block-circulant-spmm.pdf]
**Documented**: 2026-02-15

## Description

When a large sparse matrix has **block-circulant structure** — meaning each block row is a cyclic shift of its predecessor — the standard sparse matrix-vector product (SpMV) can be rewritten as a **sparse matrix-dense matrix product (SpMM)**, yielding up to 10× speedup on CPU and 6× on GPU. The key insight is that block-circulant symmetry means only the first block row of the sparse matrix needs to be stored, and the circulant shifting property can be "transferred" from the sparse matrix to the dense input vector, reorganizing the input into a dense anti-circulant matrix.

This is a hardware-level optimization trick: SpMV is typically memory-bandwidth-bound (low arithmetic intensity, ~3–40% peak GFLOPS), while SpMM has higher arithmetic intensity and better cache/coalescing behavior. By recognizing block-circulant structure, we convert an inherently bandwidth-limited operation into a compute-friendly one.

While originally developed for computed tomography (CT) reconstruction, this pattern applies to any setting where block-circulant sparse matrices arise — including neural network layers with circulant weight structure and shared-weight architectures with cyclic symmetry.

## Mathematical Form

**Block Circulant Matrix-Vector Product:**

Let $C$ be a block circulant matrix of size $m_C \times n_C$ with $k \times k$ blocks of size $m_B \times n_B$ (so $m_C = k \cdot m_B$, $n_C = k \cdot n_B$). The first block row fully specifies $C$:

$$
C = \begin{pmatrix} A_0 & A_1 & \cdots & A_{k-2} & A_{k-1} \\ A_{k-1} & A_0 & \cdots & A_{k-3} & A_{k-2} \\ \vdots & & \ddots & & \vdots \\ A_1 & A_2 & \cdots & A_{k-1} & A_0 \end{pmatrix}
$$

The naive block-wise SpMV computes each output block $\mathbf{y}_i$ as:

$$
\mathbf{y}_i = \sum_{j=0}^{k-1} A_{(j-i) \bmod k} \, \mathbf{x}_j, \quad \text{for } i = 0, \ldots, k-1
$$

**SpMM Rewrite:**

The product $\mathbf{y} \leftarrow C\mathbf{x}$ can be reformulated as a sparse-dense matrix product $Y \leftarrow A \hat{X}$:

$$
\begin{pmatrix} \mathbf{y}_0 & \mathbf{y}_1 & \cdots & \mathbf{y}_{k-1} \end{pmatrix} \leftarrow \begin{pmatrix} A_0 & A_1 & \cdots & A_{k-1} \end{pmatrix} \begin{pmatrix} \mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_{k-2} & \mathbf{x}_{k-1} \\ \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_{k-1} & \mathbf{x}_0 \\ \vdots & & & & \vdots \\ \mathbf{x}_{k-1} & \mathbf{x}_0 & \cdots & \mathbf{x}_{k-3} & \mathbf{x}_{k-2} \end{pmatrix}
$$

The circulant property is transferred to the dense input, forming an **anti-circulant** matrix $\hat{X}$ of size $n_C \times k$. The sparse factor $A = (A_0 \; A_1 \; \cdots \; A_{k-1})$ is just the first block row of $C$ — stored once.

**Key Definitions:**

- $C \in \mathbb{R}^{m_C \times n_C}$ — full block circulant sparse matrix (never explicitly stored)
- $A = (A_0, A_1, \ldots, A_{k-1})$ — first block row, sparse, size $m_B \times n_C$
- $\hat{X} \in \mathbb{R}^{n_C \times k}$ — anti-circulant reshape of input vector $\mathbf{x}$
- $k$ — number of blocks (circulant shift count)
- $m_B, n_B$ — block dimensions
- $\text{nnz}$ — number of nonzeros in first block row $A$

**Implicit Anti-Circulant Access (no extra memory):**

Rather than explicitly constructing $\hat{X}$, element access is remapped:

$$
\hat{X}_{i,j} = X_{i', j'}, \quad \text{where } i' = i \bmod n_B, \; j' = (\lfloor i/n_B \rfloor + j) \bmod k
$$

Alternatively, store $\hat{\mathbf{x}} = (\mathbf{x}, \mathbf{x})$ (two concatenated copies) and access:

$$
\hat{X}_{i,j} = \hat{x}_{j'}, \quad \text{where } j' = i + j \cdot n_B
$$

**MAXPY Optimization (CPU):**

Merge $s$ consecutive AXPY operations from the same output row into a single MAXPY:

$$
Y[i, p] \leftarrow Y[i, p] + \sum_{l=j}^{j+s-1} A_v[l] \cdot X[A_j[l] \bmod n_B, \; (\lfloor A_j[l] / n_B \rfloor + p) \bmod k]
$$

With $s = 8$, MAXPY vectorizes well and achieves ~25% additional speedup over basic SpMM.

## Complexity

| Operation | Naive SpMV (full $C$) | SpMV (block-wise) | SpMM Rewrite |
|-----------|-----------|-----------|-------------|
| Compute | $O(k \cdot \text{nnz})$ | $O(k \cdot \text{nnz})$ | $O(k \cdot \text{nnz})$ |
| Storage | $O(k \cdot \text{nnz})$ | $O(\text{nnz})$ | $O(\text{nnz})$ |
| GFLOPS (CPU) | 3–5 | 3–5 | 20–55 |
| GFLOPS (GPU) | 5–10 | 5–10 | ~60 |

**FLOP count is identical** — the rewrite does the same arithmetic. The speedup is entirely from **better hardware utilization**: SpMM has higher arithmetic intensity, better cache locality, and enables SIMD vectorization and GPU memory coalescing.

**Memory:** $O(\text{nnz} + 2 \cdot n_C)$ — stores first block row in CSR format plus two copies of the input vector for implicit anti-circulant access. This is $k \times$ less storage than the full circulant matrix.

**Measured speedups (from paper, CT matrices with $k = 150$ blocks):**
- CPU (Intel i7): MM-2-8 kernel achieves up to 10× over basic MV
- GPU (NVIDIA GTX680): Custom SpMM achieves 6× over CUSPARSE SpMV, ~60 GFLOPS sustained

## Applicability

- **Block-circulant neural network layers**: CirCNN and similar architectures use block-circulant weight matrices for FC layers; the SpMM rewrite accelerates inference when the weight matrix is also sparse (e.g., pruned circulant weights)
- **Computed tomography / iterative reconstruction**: The original application — polar-coordinate CT system matrices are naturally block-circulant with ~150 blocks, sparse blocks with density 1–4%
- **Sparse attention with cyclic structure**: Any attention pattern that exhibits block-circulant sparsity (e.g., rotational equivariant architectures) can leverage this rewrite
- **Convolution-as-matrix-multiply**: Circular convolution layers with sparse kernels can be expressed as sparse block-circulant matrices and accelerated via this SpMM trick
- **Equivariant networks with cyclic groups**: Networks equivariant to $C_k$ (cyclic group of order $k$) produce block-circulant weight matrices; if the blocks are sparse, this rewrite applies
- **Batched inference**: The SpMM formulation naturally extends to batched inputs — replace $\hat{X}$ columns with multiple input vectors to achieve even higher arithmetic intensity

## Limitations

- **Requires block-circulant structure**: The matrix must have exact block-circulant form (each block row is a cyclic shift); approximate or perturbed circulant structures don't benefit
- **Sparse blocks required**: For dense block-circulant matrices, the standard FFT diagonalization approach ($O(k \cdot n_B \log n_B)$) is more appropriate; this trick targets the case where blocks $A_i$ are sparse
- **No asymptotic FLOP reduction**: The FLOP count is the same as naive SpMV — all improvement comes from better hardware utilization (cache locality, vectorization, coalescing)
- **Memory overhead**: Requires storing two copies of the input vector ($2 \times n_C$) or modifying sparse matrix indices; this is negligible for large problems but non-trivial for small ones
- **Implementation complexity**: Custom SpMM kernels are needed (CUSPARSE SpMM had API limitations preventing direct use); requires per-platform kernel development
- **Best for large $k$**: The benefit scales with the number of circulant blocks $k$ (storage saved = $k \times$); for small $k$, the overhead may not be worthwhile

## Implementation Notes

```python
import numpy as np
from scipy.sparse import csr_matrix

def sparse_block_circulant_spmm(A_csr, x, k, n_B):
    """Fast block-circulant SpMV via SpMM rewrite.

    Instead of k separate SpMV calls (one per block shift),
    we do a single SpMM: Y = A @ X_hat, where X_hat is the
    anti-circulant reshape of x.

    Args:
        A_csr: sparse matrix (m_B x n_C) — first block row of C
        x: input vector of length n_C = k * n_B
        k: number of circulant blocks
        n_B: columns per block

    Returns:
        y: output vector of length m_C = k * m_B
    """
    m_B = A_csr.shape[0]

    # Reshape x into matrix form: (n_B, k)
    X = x.reshape(k, n_B).T  # (n_B, k)

    # Build anti-circulant dense matrix X_hat: (n_C, k)
    # X_hat[i, j] = X[i mod n_B, (floor(i/n_B) + j) mod k]
    X_hat = np.zeros((k * n_B, k))
    for j in range(k):
        for b in range(k):
            block_col = (b + j) % k
            X_hat[b * n_B:(b + 1) * n_B, j] = X[:, block_col]

    # Single SpMM: Y = A @ X_hat, shape (m_B, k)
    Y = A_csr @ X_hat  # (m_B, k)

    # Flatten to output vector: interleave block outputs
    y = Y.T.reshape(-1)  # (k * m_B,)
    return y


# GPU pseudocode (CUDA kernel from paper):
#
# for i in range(m_B):           // parallel over thread blocks
#   for p in range(k):           // parallel over threads in block
#     w = 0
#     for j in range(Ai[i], Ai[i+1]):
#       c = Aj[j] / n_B + p
#       if c > n_B: c -= n_B
#       l = Aj[j] % n_B
#       w += Av[j] * X[l, c]
#     Y[i, p] = w
```

## References

- Romero Alcalde, E., Tomás Domínguez, A.E., Soriano Asensi, A. & Blanquer Espert, I. "A Fast Sparse Block Circulant Matrix Vector Product" Euro-Par 2014, Springer LNCS 8632, pp. 548–559. doi:10.1007/978-3-319-09873-9_46
- Ding, C. et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" MICRO 2017
- Im, E.J., Yelick, K. & Vuduc, R. "Sparsity: Optimization framework for sparse matrix kernels" Int. J. High Performance Computing Applications 18(1), 2004
- NVIDIA CUSPARSE Library, https://developer.nvidia.com/cusparse

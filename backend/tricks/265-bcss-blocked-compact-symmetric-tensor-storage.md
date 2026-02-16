# 265: BCSS — Blocked Compact Symmetric Storage for Tensor Operations

**Category**: kernel
**Gain type**: efficiency
**Source**: Schatz, Low, van de Geijn & Kolda (2014) — "Exploiting Symmetry in Tensors for High Performance" (SIAM J. Sci. Comput.)
**Paper**: [papers/symmetric-tensor-multiplication.pdf]
**Documented**: 2026-02-16

## Description

Higher-Order Linear Attention (HLA, trick 222) and its symmetry-aware extension (SATA, trick 264) both maintain symmetric tensor states: HLA's key metric $\mathbf{S}_t^K = \sum_{i \leq t} \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d \times d}$ is a symmetric matrix (order-2 symmetric tensor), and third-order HLA would maintain an order-3 symmetric tensor in $\mathbb{R}^{d \times d \times d}$. The fundamental computational operation is the **symmetric tensor times same matrix** (**sttsm**): $\mathcal{C} = [\mathcal{A}; \mathbf{X}, \ldots, \mathbf{X}] = \mathcal{A} \times_0 \mathbf{X} \times_1 \mathbf{X} \cdots \times_{m-1} \mathbf{X}$, where $\mathcal{A}$ is an order-$m$ symmetric tensor and $\mathbf{X}$ is a change-of-basis matrix.

**Blocked Compact Symmetric Storage (BCSS)** is a data structure and algorithm framework that:

1. **Stores only the unique blocks** of a symmetric tensor: an order-$m$ tensor with $\bar{n}^m$ blocks only stores $\binom{\bar{n} + m - 1}{m}$ of them (the "upper hyper-triangular" blocks where block indices $\bar{i}_0 \leq \bar{i}_1 \leq \cdots \leq \bar{i}_{m-1}$). Storage savings: $O(m!)$ relative to dense.
2. **Avoids redundant computation** by computing temporary tensors via a cascade of matrix-tensor multiplications in each mode, reusing intermediates.
3. **Exploits partial symmetry** in temporaries to further reduce both computation and storage.
4. **Uses blocked algorithms** that maintain regular memory access patterns (cache-friendly, coalesced) — the blocked format lets each block be treated as a dense sub-tensor, enabling calls to optimized BLAS/tensor-core routines.

The key insight: the conflict between "exploit symmetry to reduce storage" and "maintain regular access patterns for GPU performance" is resolved at the **block level** — blocks are dense and regular, but only the unique blocks are stored and computed.

## Mathematical Form

**The sttsm operation (order $m$, size $n$):**

$$
\mathcal{C} := [\mathcal{A}; \mathbf{X}, \ldots, \mathbf{X}] = \mathcal{A} \times_0 \mathbf{X} \times_1 \cdots \times_{m-1} \mathbf{X}
$$

where $\mathcal{A} \in \mathbb{R}^{[m,n]}$ is a symmetric order-$m$ tensor (all modes have dimension $n$), $\mathbf{X} \in \mathbb{R}^{p \times n}$, and $\mathcal{C} \in \mathbb{R}^{[m,p]}$ is the resulting symmetric tensor.

Elementwise:

$$
\gamma_{j_0 j_1 \cdots j_{m-1}} = \sum_{i_0=0}^{n-1} \cdots \sum_{i_{m-1}=0}^{n-1} \alpha_{i_0 i_1 \cdots i_{m-1}} \, \chi_{j_0 i_0} \, \chi_{j_1 i_1} \cdots \chi_{j_{m-1} i_{m-1}}
$$

**For order $m = 2$ (the matrix case):**

$$
\mathbf{C} = \mathbf{X} \mathbf{A} \mathbf{X}^\top
$$

This is exactly the operation in HLA: computing $\mathbf{q}^\top \mathbf{S}_t^K = \mathbf{q}^\top (\sum \mathbf{k}_i \mathbf{k}_i^\top)$ is a symmetric matrix-vector product, and the scan operator's cross-term $\mathbf{S}_B \mathbf{C}_A$ is symmetric-matrix times general-matrix.

**Temporary-based algorithm (order $m = 2$):**

$$
\mathbf{C} = \mathbf{X} \mathbf{A} \mathbf{X}^\top
$$

Split as: first compute $\hat{\mathbf{T}} = \mathbf{A} \hat{\mathbf{X}}^\top$ (symmetric matrix times tall matrix), then $\mathbf{C}_{j_0,:} = \hat{\mathbf{x}}_{j_0}^\top \hat{\mathbf{T}}$ (row of $\mathbf{X}$ times temporary).

Cost: $2pn^2 + p^2 n$ flops (vs $3p^2 n^2 / 2$ for naive) using temporary $\hat{\mathbf{T}} \in \mathbb{R}^{n \times p}$.

**Temporary-based algorithm (order $m = 3$):**

$$
\mathcal{C} = \mathcal{A} \times_0 \mathbf{X} \times_1 \mathbf{X} \times_2 \mathbf{X}
$$

Split into cascade:

$$
\mathbf{T}_{i_2}^{(2)} = \mathcal{A} \times_2 \hat{\mathbf{x}}_{i_2}^\top, \quad \mathbf{t}_{i_1 i_2}^{(1)} = \mathbf{T}_{i_2}^{(2)} \times_1 \hat{\mathbf{x}}_{i_1}^\top, \quad \gamma_{j_0 j_1 j_2} = \hat{\mathbf{t}}^{(1)\top}_{j_1 j_2} \times_0 \hat{\mathbf{x}}_{j_0}^\top
$$

Cost: $p(2n^3 + 2pn^2 + p^2n)$ flops (saved by reusing intermediates).

**Blocked Compact Symmetric Storage:**

Partition $\mathcal{A}$ into blocks of size $b_\mathcal{A}$ per mode. For order $m$ with $\bar{n} = n / b_\mathcal{A}$ blocks per mode:

- **Dense storage**: $\bar{n}^m$ blocks, each $b_\mathcal{A}^m$ entries $\Rightarrow$ total $n^m$
- **BCSS**: only $\binom{\bar{n} + m - 1}{m}$ unique blocks stored (upper hyper-triangular region)
- **Storage ratio**: $\frac{\binom{\bar{n}+m-1}{m}}{\bar{n}^m} \approx \frac{1}{m!}$ for large $\bar{n}$

| Order $m$ | Dense blocks | BCSS blocks | Savings factor |
|-----------|-------------|-------------|----------------|
| 2 | $\bar{n}^2$ | $\bar{n}(\bar{n}+1)/2$ | $\approx 2\times$ |
| 3 | $\bar{n}^3$ | $\bar{n}(\bar{n}+1)(\bar{n}+2)/6$ | $\approx 6\times$ |
| 4 | $\bar{n}^4$ | $\binom{\bar{n}+3}{4}$ | $\approx 24\times$ |

**Partial symmetry in temporaries:**

The intermediate tensor $\mathcal{T}^{(m-1)} = \mathcal{A} \times_{m-1} \hat{\mathbf{X}}^\top$ inherits partial symmetry from $\mathcal{A}$: it is symmetric in modes $\{0, \ldots, m-2\}$ (all modes except the one contracted). This partial symmetry can be exploited to reduce storage of temporaries:

- Temporary storage for $\mathcal{T}^{(d)}$: $\binom{\bar{n}+m-1-d}{m-d}$ unique blocks (partially symmetric in $m - d$ modes)

**Computational savings (general order $m$):**

Total flops with BCSS and temporaries:

$$
\sum_{d=0}^{m-1} 2 b_\mathcal{C}^{d+1} n^{m-d} \binom{\bar{p} + d}{d+1}
$$

The computational savings factor relative to naive is $O\!\left(\frac{(m+1)!}{2^m}\right)$, which for:
- $m = 2$: savings $\approx 1.5\times$
- $m = 3$: savings $\approx 4\times$
- $m = 4$: savings $\approx 15\times$

## Complexity

| Quantity | Dense | BCSS |
|----------|-------|------|
| Storage (order-2, size $n$) | $n^2$ | $n(n+1)/2 \approx n^2/2$ |
| Storage (order-3, size $n$) | $n^3$ | $n(n+1)(n+2)/6 \approx n^3/6$ |
| Compute ($m=2$, sttsm) | $3p^2 n^2 / 2$ | $2pn^2 + p^2 n$ |
| Compute ($m=3$, sttsm) | $p^3 n^3$ | $\approx p^3 n^3 / 4$ |

**For HLA with $d = 128$, order 2:**

| Resource | Dense | BCSS |
|----------|-------|------|
| State $\mathbf{S}^K$ storage | $128^2 = 16{,}384$ | $128 \times 129 / 2 = 8{,}256$ |
| Memory bandwidth for matvec $\mathbf{q}^\top \mathbf{S}^K$ | 16K reads | 8.3K reads |
| Scan cross-term $\mathbf{S}_B \mathbf{C}_A$ | full $128 \times 128$ GEMM | SYMM: half the reads |

**For hypothetical third-order HLA with $d = 64$:**

| Resource | Dense | BCSS |
|----------|-------|------|
| State tensor storage | $64^3 = 262{,}144$ | $\binom{66}{3} = 45{,}760$ |
| Storage savings | baseline | **5.7$\times$** |

## Applicability

- **HLA key metric $\mathbf{S}_t^K$**: The $d \times d$ running sum $\mathbf{S}_t^K = \sum \mathbf{k}_i \mathbf{k}_i^\top$ is symmetric. BCSS stores it in $d(d+1)/2$ entries, halving memory bandwidth for the $\mathbf{q}^\top \mathbf{S}_t^K$ matvec and the rank-1 update $\mathbf{S}_t^K \leftarrow \mathbf{S}_t^K + \mathbf{k}_t \mathbf{k}_t^\top$.

- **HLA associative scan cross-terms**: The scan operator computes $\mathbf{S}_B \mathbf{C}_A$ where $\mathbf{S}_B$ is symmetric. This can use BLAS SYMM (symmetric matrix-matrix multiply) instead of GEMM, reducing reads by $\sim 2\times$ and potentially using specialized tensor core paths.

- **Third-order and higher HLA extensions**: For order-3 prefix summaries, BCSS reduces storage from $d^3$ to $\binom{d+2}{3}$ — a $6\times$ savings — and computation by $\sim 4\times$ via temporary reuse.

- **SATA feature map states**: SATA's accumulated state $S_{p,T} \in \mathbb{R}^{m_p \times d_V}$ has $m_p = \binom{d_K+p-1}{p}$ rows, which is exactly the BCSS count for an order-$p$ symmetric tensor. The BCSS algorithm provides the optimal blocked computation strategy for updating and querying these states.

- **Any neural network layer with symmetric matrix state**: Applies to mLSTM's matrix memory $\mathbf{C}_t$, DeltaNet's state updates, and any linear RNN maintaining second-moment statistics.

## Limitations

- **Block size selection is critical**: BCSS storage approaches minimal ($n(n+1)/2$ for matrices) only when $\bar{n} = n/b$ is large (many blocks). With too few blocks, the overhead of storing full $b \times b$ diagonal blocks wastes space. For HLA with $d = 128$ and block size $b = 16$: $\bar{n} = 8$ blocks, BCSS stores $8 \times 9 / 2 = 36$ blocks vs 64 dense — a $1.78\times$ savings.

- **Indexing overhead on GPU**: Mapping from symmetric block indices to linear memory requires index computation that may cause warp divergence. For the matrix case ($m = 2$), this is minor (simple triangular index formula). For $m = 3$, the indexing is more complex.

- **BLAS SYMM is underutilized on GPUs**: While CUBLAS provides SYMM (symmetric matrix multiply), it's often slower than GEMM for small matrices because GPU tensor cores are optimized for dense rectangular tiles. At HLA's typical $d = 64$--$128$, the matrix is small enough that GEMM overhead is negligible, reducing BCSS's compute advantage.

- **Memory savings matter most for bandwidth-bound operations**: The $2\times$ memory savings for storing $\mathbf{S}_t^K$ directly translates to $2\times$ less HBM bandwidth for the streaming matvec $\mathbf{q}^\top \mathbf{S}_t^K$ during generation (which IS bandwidth-bound). During chunkwise training, the intra-chunk computation is compute-bound and benefits less.

- **Not applicable to non-symmetric states**: HLA's cross-summaries $\mathbf{G}_t \in \mathbb{R}^{d \times d_v}$ and first-order accumulator $\mathbf{C}_t^{QV} \in \mathbb{R}^{d \times d_v}$ are NOT symmetric — BCSS doesn't help for these.

## Implementation Notes

```python
# Blocked Compact Symmetric Storage for HLA's S_K state
# Specialized for order-2 (symmetric matrix) — the HLA use case

import torch

class SymmetricMatrixBCSS:
    """Store and operate on symmetric matrix using upper triangular packing.

    For HLA's S_K in R^{d x d}, stores d*(d+1)/2 entries instead of d^2.
    Provides rank-1 update and matrix-vector product.
    """
    def __init__(self, d, dtype=torch.float32, device='cuda'):
        self.d = d
        self.n_entries = d * (d + 1) // 2
        # Packed upper triangular storage
        self.data = torch.zeros(self.n_entries, dtype=dtype, device=device)
        # Precompute index mapping for fast access
        # tri_idx[i, j] gives the linear index into packed storage (i <= j)
        idx = torch.zeros(d, d, dtype=torch.long, device=device)
        k = 0
        for i in range(d):
            for j in range(i, d):
                idx[i, j] = k
                idx[j, i] = k  # symmetric
                k += 1
        self.tri_idx = idx

    def rank1_update(self, k_vec, gamma=1.0):
        """Update S_K += k * k^T (or gamma * S_K + k * k^T).

        Only updates the d*(d+1)/2 unique entries.
        Cost: d*(d+1)/2 multiply-adds vs d^2 for dense.
        """
        if gamma != 1.0:
            self.data.mul_(gamma)
        # Compute unique entries of k * k^T
        # Upper triangular: (k_i * k_j) for i <= j
        for i in range(self.d):
            start = self.tri_idx[i, i].item()
            # Entries (i, i), (i, i+1), ..., (i, d-1)
            n_entries_row = self.d - i
            self.data[start:start + n_entries_row] += (
                k_vec[i] * k_vec[i:]
            )

    def matvec(self, q_vec):
        """Compute q^T @ S_K using packed storage.

        Uses symmetry: only reads d*(d+1)/2 entries instead of d^2.
        Critical for inference where this is bandwidth-bound.
        """
        result = torch.zeros(self.d, dtype=q_vec.dtype, device=q_vec.device)
        for i in range(self.d):
            start = self.tri_idx[i, i].item()
            n = self.d - i
            # Diagonal entry contributes to result[i]
            result[i] += q_vec[i] * self.data[start]
            # Off-diagonal entries contribute to both result[i] and result[j]
            if n > 1:
                off_diag = self.data[start + 1:start + n]
                result[i] += q_vec[i + 1:] @ off_diag
                result[i + 1:] += q_vec[i] * off_diag
        return result

# More efficient: use torch.triangular_solve or CUBLAS SYMM
def symm_matvec_cublas(S_packed, q, d):
    """Symmetric matrix-vector product using packed format.

    In practice, unpack to full symmetric and use cuBLAS SYMV,
    or use the packed triangular format with SPMV.
    """
    # Unpack to full (for correctness; a real impl stays packed)
    S_full = torch.zeros(d, d, dtype=S_packed.dtype, device=S_packed.device)
    idx = torch.triu_indices(d, d, device=S_packed.device)
    S_full[idx[0], idx[1]] = S_packed
    S_full = S_full + S_full.T - torch.diag(S_full.diag())
    return S_full @ q

# For HLA's associative scan, the cross-term S_B @ C_A can use SYMM:
def scan_cross_term_symm(S_B_packed, C_A, d, d_v):
    """Compute S_B @ C_A where S_B is symmetric (packed).

    Uses cuBLAS SYMM: reads d*(d+1)/2 entries of S_B instead of d^2.
    C_A in R^{d x d_v} is general (not symmetric).

    For d=128, d_v=64: reads 8256 instead of 16384 entries of S_B.
    """
    # In practice: torch.linalg or custom CUDA kernel
    S_full = unpack_symmetric(S_B_packed, d)  # for prototype
    return S_full @ C_A  # (d, d_v)

# GPU efficiency analysis for HLA inference (streaming, per token):
#
# Operation               | Dense S_K     | BCSS S_K
# ----------------------- | ------------- | --------
# S_K storage             | d^2 = 16,384  | d(d+1)/2 = 8,256
# rank-1 update reads     | d^2           | d(d+1)/2
# rank-1 update writes    | d^2           | d(d+1)/2
# q^T S_K matvec reads    | d^2           | d(d+1)/2
# HBM bandwidth savings   | baseline      | ~2x
#
# For d=128 in BF16: 32KB (dense) vs 16.5KB (BCSS) per head
# At 32 heads: 1MB vs 528KB — fits more heads in SRAM
#
# The bandwidth savings directly improve inference throughput
# since the q^T S_K matvec is HBM-bandwidth-bound for typical d.
```

## References

- Schatz, M. D., Low, T. M., van de Geijn, R. A., & Kolda, T. G. (2014). Exploiting Symmetry in Tensors for High Performance: Multiplication with Symmetric Tensors. SIAM J. Sci. Comput., 36(5):C453--C479. [https://arxiv.org/abs/1301.7744](https://arxiv.org/abs/1301.7744)
- Zhang, Y., Qin, Z., & Gu, Q. (2025). Higher-order Linear Attention. arXiv:2510.27258. (HLA — primary application: symmetric $\mathbf{S}_t^K$ state)
- Heinsen, F. A. & Kozachkov, L. (2026). Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation. arXiv:2602.00294. (SATA — cites this work for symmetric tensor computation)
- Van Loan, C. F. & Ragnarsson, S. Tensor Computation Seminar. (Blocked tensor algorithms)
- NVIDIA cuBLAS Documentation — SYMM, SPMV routines for symmetric matrix operations.

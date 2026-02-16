# 071: Learned Sparse Matrix Row Permutation for SpMM

**Category**: kernel
**Gain type**: efficiency
**Source**: Mehrabi, Lee, Chatterjee, Sorin, Lee & O'Connor (Duke / UPenn / NVIDIA, ISPASS 2021)
**Paper**: [papers/learned-spmm-row-permutation.pdf]
**Documented**: 2026-02-15

## Description

Sparse Matrix-dense Matrix multiplication (SpMM) is a core operation in neural network inference with sparse weights, graph neural networks, and scientific computing. On GPUs, SpMM with the standard Compressed Sparse Row (CSR) format suffers from two problems caused by the distribution of nonzeros across rows:

1. **Load imbalance:** Rows with different numbers of nonzeros are assigned to warps within a CTA (Cooperative Thread Array), creating straggler warps that block the entire CTA from accepting new work.
2. **Poor cache locality:** The column indices of nonzeros within a row determine which cache lines of the dense matrix $B$ must be fetched. Rows with scattered nonzeros cause many cache misses, while rows with clustered nonzeros can reuse cache lines across adjacent warps.

The key insight is that **row ordering in CSR directly controls both load balance and cache access patterns**, yet standard libraries (cuSPARSE) use the natural row order. This paper proposes a family of row permutation strategies — load-balancing sorts, cache-aware sorts, and hybrid combinations — that reorder CSR rows to improve GPU utilization and memory locality. Crucially, the permutation preserves the CSR format itself (no format conversion needed) and is applied once offline for matrices reused across multiple SpMM invocations (e.g., fixed sparse weight matrices during inference).

Since no single permutation dominates across all sparsity patterns, the paper trains a **learned predictor** that selects the best permutation for a given matrix based on lightweight structural features. The predictor achieves 96% of oracle (exhaustive-search) performance, providing a practical automatic selection policy.

## Mathematical Form

**SpMM Operation:**

Given sparse matrix $A \in \mathbb{R}^{M \times N}$ (CSR format) and dense matrix $B \in \mathbb{R}^{N \times K}$, compute:

$$
C = A \cdot B, \quad C \in \mathbb{R}^{M \times K}
$$

A row permutation $\pi: [M] \to [M]$ reorders the rows of $A$ without changing the computation result (since each row of $C$ depends only on the corresponding row of $A$):

$$
C[\pi(i), :] = A[\pi(i), :] \cdot B \quad \forall i \in [M]
$$

The permuted CSR stores rows in order $\pi(0), \pi(1), \ldots, \pi(M-1)$.

**Warp Load:**

For row $R$ with $\text{nnz}_R$ nonzeros processed by a warp of $N_t = 32$ threads:

$$
\text{warp\_load}(R) = \left\lceil \frac{\text{nnz}_R}{N_t} \right\rceil
$$

Rows are striped across $N_w$ warps in a CTA: row $i$ maps to warp $j = i \bmod N_w$.

**Cache-Line Bit Mask:**

For row $R$, construct a bit mask $b \in \{0,1\}^{\lceil N/32 \rceil}$ where $b[k] = 1$ if the row has at least one nonzero in column block $[32k, 32(k+1))$. This indicates which cache lines of dense matrix $B$ must be loaded. The Hamming distance between two rows' bit masks measures their cache access dissimilarity:

$$
d_{\text{cache}}(R_i, R_j) = \| b_i \oplus b_j \|_1
$$

Rows with small Hamming distance reuse the same cache lines when co-located in the same warp group.

**Load-Balancing Permutations:**

- **Plain Sort**: Sort rows by warp load (nonzero count). Groups rows of similar density into the same CTA.
- **Flipped Sort**: Plain Sort with alternating assignment direction every $N_w$ rows, preventing all heavy rows from landing on the same warp.
- **LPT (Longest Processing Time) Sort**: Sort rows in decreasing warp load, then greedily assign each row to the warp with the smallest current load. Minimizes max warp load.

**Cache-Aware Permutations:**

- **Warp-Aware Sort**: Incrementally assign rows to minimize Hamming distance to the row 32 positions earlier (co-located in the same warp). Adjacent rows in different warps share $B$ cache lines.
- **CTA-Aware Sort**: Incrementally assign rows to create clusters of $N_w = 32$ rows with mutually similar bit masks, maximizing inter-warp cache reuse within a CTA.

**Hybrid Permutations:**

- **Hybrid-1 Sort**: LPT primary, cache-locality as tie-breaker.
- **Hybrid-2.1**: CTA-Aware primary, warp load as tie-breaker.
- **Hybrid-2.2**: CTA-Aware primary, intra-warp bit-mask distance as tie-breaker.
- **Hybrid-2.3**: Warp-Aware primary, warp load as tie-breaker.

**Learned Permutation Selector:**

Given structural features $\phi(A) \in \mathbb{R}^d$ extracted from the sparsity pattern of $A$, a classifier $f_\theta$ predicts the best permutation:

$$
\hat{\pi}^* = f_\theta(\phi(A)) = \arg\max_{\pi \in \Pi} P(\pi | \phi(A); \theta)
$$

where $\Pi = \{\text{Plain, Flipped, LPT, Warp-Aware, CTA-Aware, Hybrid-1, Hybrid-2.1, Hybrid-2.2, Hybrid-2.3}\}$.

**Features $\phi(A)$** include:
- Summary statistics: matrix dimensions, density, nnz, row-nnz mean/std/min/max
- Load balance features: warp load distribution statistics
- Cache features: bit-mask density, inter-row Hamming distance statistics
- Hardware-interaction features: estimated cache miss rates, warp utilization ratios

## Complexity

| Operation | No Permutation | With Permutation | Overhead |
|-----------|---------------|-----------------|----------|
| Permutation computation | $0$ | $O(M \log M)$ to $O(M^2)$ | One-time offline |
| SpMM execution | $O(\text{nnz} \cdot K)$ | $O(\text{nnz} \cdot K)$ | $0$ (same kernel) |
| Feature extraction | $0$ | $O(M \cdot N/32)$ | One-time offline |
| Predictor inference | $0$ | $O(d)$ (small NN) | Negligible |

**Speedup results (over 1,688 matrices from SuiteSparse, C-stationary SpMM kernel):**

| Baseline | Avg Speedup | Max Speedup |
|----------|-------------|-------------|
| Plain CSR (no perm) | **1.4x** | up to 20x |
| NVIDIA cuSPARSE | **2.6x** | — |

**Predictor accuracy:** 96% of oracle (exhaustive best-permutation) performance.

**Memory:** $O(M)$ for the permutation index array. No change to CSR storage.

## Applicability

- **Sparse neural network inference:** Any model with sparse weight matrices (structured or unstructured) stored in CSR that performs SpMM. Row permutation is applied once to the weight matrix at model load time, then every inference SpMM benefits.
- **Graph Neural Networks (GNNs):** Adjacency-matrix SpMM in message passing. The sparse adjacency is static, so one-time permutation amortizes well.
- **Structured sparsity + permutation synergy:** When combined with PA-DST or Gyro-Permutation (which permute columns/channels for weight expressivity), row permutation further optimizes the *execution efficiency* of the resulting sparse matrix on GPUs. The two are orthogonal: one permutes for accuracy, the other for hardware utilization.
- **SSM transition matrices:** Sparse transition matrices (column-sparse, block-diagonal) used in SSMs could benefit from row reordering if computed via SpMM rather than custom kernels.
- **Any reusable sparse matrix:** The trick is most valuable when the same sparse matrix is multiplied many times (amortizing the one-time permutation cost).

## Limitations

- Permutation is computed offline — not suitable for dynamically changing sparsity patterns (e.g., sparse training with mask updates every iteration).
- Load-balancing permutations can *degrade* cache locality and vice versa — neither family dominates alone, necessitating the hybrid/learned selection.
- LPT Sort harms performance for 17% of matrices (up to 0.8x slowdown) when load balancing destroys cache patterns that the original order happened to have.
- The predictor is trained on a specific GPU (GV100/2080 Ti) and kernel implementation; transferring to other GPUs or kernels may require retraining.
- Row permutation does not help when the bottleneck is not load imbalance or cache misses (e.g., very sparse matrices where bandwidth is saturated regardless of order).
- Only evaluated with CSR format; other formats (ELL, BSR, etc.) may not benefit from row permutation or may need different strategies.
- The paper evaluates on general sparse matrices, not specifically on neural network weight matrices (which may have more regular patterns than SuiteSparse).

## Implementation Notes

```python
import torch
import numpy as np

def compute_bit_mask(row_indices, N, cache_line_size=32):
    """
    Compute cache-line access bit mask for a sparse row.

    Args:
        row_indices: column indices of nonzeros in this row
        N: total number of columns
        cache_line_size: columns per cache line (128B / 4B = 32 for float32)

    Returns:
        bit_mask: (N // cache_line_size,) binary array
    """
    n_lines = (N + cache_line_size - 1) // cache_line_size
    mask = np.zeros(n_lines, dtype=np.uint8)
    for idx in row_indices:
        mask[idx // cache_line_size] = 1
    return mask

def lpt_sort(nnz_per_row, n_warps=32):
    """
    Longest Processing Time sort: minimize max warp load.

    Args:
        nnz_per_row: (M,) number of nonzeros per row
        n_warps: number of warps per CTA

    Returns:
        perm: (M,) row permutation indices
    """
    M = len(nnz_per_row)
    warp_load = np.array([nnz_per_row[i] / 32 for i in range(M)])

    # Sort rows by decreasing warp load
    sorted_rows = np.argsort(-warp_load)

    # Greedy assignment: each row to the least-loaded warp
    warp_totals = np.zeros(n_warps)
    perm = np.zeros(M, dtype=np.int64)
    warp_queues = [[] for _ in range(n_warps)]

    for row in sorted_rows:
        target_warp = np.argmin(warp_totals)
        warp_queues[target_warp].append(row)
        warp_totals[target_warp] += warp_load[row]

    # Interleave: row i goes to warp i % n_warps
    idx = 0
    max_len = max(len(q) for q in warp_queues)
    for slot in range(max_len):
        for w in range(n_warps):
            if slot < len(warp_queues[w]):
                perm[idx] = warp_queues[w][slot]
                idx += 1

    return perm

def warp_aware_sort(col_indices, row_ptr, N, n_warps=32):
    """
    Cache-aware sort: minimize Hamming distance between
    rows assigned to the same warp (32 rows apart in CSR).

    Args:
        col_indices: CSR column index array
        row_ptr: CSR row pointer array
        N: number of columns
        n_warps: warp count (rows 32 apart share a warp)

    Returns:
        perm: (M,) row permutation indices
    """
    M = len(row_ptr) - 1

    # Compute bit masks for all rows
    masks = []
    for i in range(M):
        cols = col_indices[row_ptr[i]:row_ptr[i+1]]
        masks.append(compute_bit_mask(cols, N))
    masks = np.array(masks)

    # Greedy incremental assignment
    used = np.zeros(M, dtype=bool)
    perm = np.zeros(M, dtype=np.int64)

    # Start with the row having lowest warp load
    nnz = np.array([row_ptr[i+1] - row_ptr[i] for i in range(M)])
    first = np.argmin(nnz)
    perm[0] = first
    used[first] = True

    for pos in range(1, M):
        # Find row most similar to the one n_warps positions back
        ref_pos = max(0, pos - n_warps)
        ref_mask = masks[perm[ref_pos]]

        best_row = -1
        best_dist = float('inf')
        for r in range(M):
            if not used[r]:
                dist = np.sum(ref_mask != masks[r])  # Hamming
                if dist < best_dist:
                    best_dist = dist
                    best_row = r

        perm[pos] = best_row
        used[best_row] = True

    return perm

def apply_csr_permutation(row_ptr, col_indices, values, perm):
    """
    Apply row permutation to CSR matrix (in-place reordering).
    The result is still valid CSR with rows in permuted order.
    """
    M = len(perm)
    new_row_ptr = np.zeros(M + 1, dtype=row_ptr.dtype)
    new_col_indices = []
    new_values = []

    for i, orig_row in enumerate(perm):
        start = row_ptr[orig_row]
        end = row_ptr[orig_row + 1]
        new_row_ptr[i + 1] = new_row_ptr[i] + (end - start)
        new_col_indices.append(col_indices[start:end])
        new_values.append(values[start:end])

    return (new_row_ptr,
            np.concatenate(new_col_indices),
            np.concatenate(new_values))

# Learned selector (sketch):
# 1. Extract features: density, nnz stats, warp load distribution,
#    bit-mask density, Hamming distance stats
# 2. Run through small NN/decision tree classifier
# 3. Output: best permutation strategy from {Plain, Flipped, LPT,
#    Warp-Aware, CTA-Aware, Hybrid-1, Hybrid-2.x}
# 4. Apply chosen permutation to CSR, then run standard SpMM kernel
```

## References

- Mehrabi, A., Lee, D., Chatterjee, N., Sorin, D.J., Lee, B.C. & O'Connor, M. (2021). Learning Sparse Matrix Row Permutations for Efficient SpMM on GPU Architectures. ISPASS 2021.
- Davis, T.A. & Hu, Y. (2011). The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software.
- Pichel, J.C., et al. (2005). Optimization of Sparse Matrix-Vector Multiplication Using Reordering Techniques on GPUs. Journal of Computational Science.
- Jiang, P., et al. (2020). A Novel Locality-Aware Method for Parallel Sparse Matrix Computation. IEEE TPDS.
- NVIDIA cuSPARSE Library. https://developer.nvidia.com/cusparse

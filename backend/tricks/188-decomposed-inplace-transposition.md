# 188: Decomposed In-Place Matrix Transposition (C2R/R2C)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Catanzaro, Keller & Garland, "A Decomposition for In-place Matrix Transposition" (PPoPP 2014)
**Paper**: [papers/cycle-leader-inplace-transposition.pdf]
**Documented**: 2026-02-15

## Description

In-place matrix transposition is traditionally implemented via **cycle-following** on the permutation $\sigma(l) = (l \cdot n) \bmod (mn - 1)$ induced by reinterpreting row-major storage as column-major. Cycle-following is inherently sequential (cycles have irregular lengths, leaders are hard to identify in parallel) and requires $O(mn \log mn)$ work with sub-$O(mn)$ auxiliary space. This paper shows that transposition can be **decomposed into independent row-wise and column-wise permutations**, each of which is trivially parallelizable with perfect load balancing.

The key insight is that instead of viewing transposition as a single monolithic permutation on the linearized array, one can retain the 2D view and decompose the operation into three steps:

1. **Column rotation**: rotate each column by a data-dependent amount (only needed when $\gcd(m, n) > 1$)
2. **Row shuffle**: scatter each row's elements to their destination columns via a closed-form index function
3. **Column shuffle**: gather each column's elements from their source rows via a closed-form index function

Each step operates on rows or columns **independently**, enabling embarrassingly parallel GPU execution. The column rotation resolves "collisions" (multiple elements in one row mapping to the same destination column) by exploiting the periodicity of the destination function $d_i(j) = (i + jm) \bmod n$.

This decomposition is directly applicable to neural network implementations where data layout transformations (AoS↔SoA, channel-first↔channel-last, head reshaping in multi-head attention) are performance bottlenecks.

## Mathematical Form

**Linearization:**

For an $m \times n$ matrix stored in row-major order:

$$
l_{\text{rm}}(i, j) = j + in, \quad i_{\text{rm}}(l) = \lfloor l/n \rfloor, \quad j_{\text{rm}}(l) = l \bmod n
$$

**Traditional transposition permutation** (linearized):

$$
A^T_{\text{rm}}[l] = A_{\text{rm}}[l_{\text{rm}}(j^T_{\text{rm}}(l), i^T_{\text{rm}}(l))]
$$

where $i^T_{\text{rm}}(l) = j_{\text{cm}}(l) = \lfloor l/m \rfloor$ and $j^T_{\text{rm}}(l) = i_{\text{cm}}(l) = l \bmod m$.

**C2R Decomposition (Columns to Rows):**

Let $c = \gcd(m, n)$, $a = m/c$, $b = n/c$.

The destination column of element $j$ in row $i$:

$$
d_i(j) = (i + jm) \bmod n
$$

This function is **periodic with period $b$** (Lemma 1), causing collisions when $c > 1$.

**Step 1 — Column rotation** (resolve collisions): rotate column $j$ by $\lfloor j/b \rfloor$ elements:

$$
r_j(i) = \left(i + \lfloor j/b \rfloor\right) \bmod m
$$

**Step 2 — Row shuffle**: after rotation, the conflict-free destination column is:

$$
d'_i(j) = \left(\left(i + \lfloor j/b \rfloor\right) \bmod m + jm\right) \bmod n
$$

Theorem 3 proves $d'_i$ is a bijection on $[0, n)$ for each fixed $i$.

**Step 3 — Column shuffle**: the source row of element $i$ in column $j$:

$$
s'_j(i) = \left(j + in - \lfloor i/a \rfloor\right) \bmod m
$$

This decomposes further into a column rotation $p_j(i) = (i + j) \bmod m$ followed by a row permutation $q(i) = \left(i \cdot n - \lfloor i/a \rfloor\right) \bmod m$.

**Theorem 4 (Decomposability):** In-place transposition can be decomposed into independent row-wise and column-wise permutations.

**Key Definitions:**

- $m, n$ — number of rows and columns of the original matrix
- $c = \gcd(m, n)$, $a = m/c$, $b = n/c$ — used in periodicity analysis
- $d_i(j)$ — destination column of element at position $(i, j)$
- $r_j(i)$ — column rotation index (pre-rotation to resolve collisions)
- $d'_i(j)$ — conflict-free row shuffle index (bijective)
- $s'_j(i)$ — column shuffle source index

## Complexity

| Operation | Cycle-following | Decomposed (C2R) |
|-----------|----------------|-------------------|
| Work | $O(mn \log mn)$ | $O(mn)$ |
| Auxiliary space | $< O(mn)$ | $O(\max(m, n))$ |
| Parallel depth | $O(\text{max cycle length})$ | $O(1)$ per step |
| Load balance | Poor (irregular cycles) | Perfect (uniform) |
| GPU kernel launches | Multiple (cycle detection) | 3 (rotate + row + col) |

**Memory:** $O(\max(m, n))$ auxiliary buffer (one row or one column).

**GPU throughput (Tesla K20c, 64-bit elements):**
- Decomposed C2R: **19.5 GB/s** median (up to 26 GB/s)
- Tiled algorithm (Sung): 5.33 GB/s median (32-bit only)
- For AoS↔SoA skinny matrices: **34.3 GB/s** median, **51 GB/s** peak
- AoS↔SoA via register transpose + decomposed algorithm: **180 GB/s** (45× faster than compiler-generated)

**Arithmetic intensity:** Low — this is a bandwidth-bound operation. The index computations use integer multiply/divide/mod but are optimized via strength reduction (fixed-point reciprocals replace divisions).

## Applicability

- **Data layout transformations in neural networks**: Converting between channel-first (NCHW) and channel-last (NHWC) layouts for convolution layers, which is a common bottleneck when switching between cuDNN operations that prefer different formats. The decomposed transpose enables in-place conversion without allocating a duplicate tensor
- **Multi-head attention reshaping**: The reshape from $(B, L, H, D)$ to $(B, H, L, D)$ in multi-head attention is a transposition of the middle two dimensions. When done in-place, this avoids the memory overhead of `torch.permute().contiguous()` which allocates a full copy
- **Array of Structures ↔ Structure of Arrays**: Converting between interleaved and planar formats for SIMD/vectorized processing. The decomposed algorithm enables efficient in-register transposition for small structures (e.g., complex numbers, quaternions, RGB pixels)
- **Gradient checkpointing**: When memory is extremely constrained (e.g., during activation recomputation), in-place transpose avoids the 2× peak memory that out-of-place methods require
- **Sequence model state reshaping**: SSMs and linear attention models frequently reshape state tensors between chunk-parallel and sequential forms; in-place transposition reduces memory pressure during these transitions

## Limitations

- **Not faster than out-of-place for square matrices**: When auxiliary memory is available, out-of-place transpose with tiling and shared memory achieves near-peak bandwidth. The decomposed method's advantage is specifically in the **in-place** setting or when memory is constrained
- **Three-step overhead**: The decomposition requires 3 passes over the data (6× memory traffic vs. the 2× theoretical minimum), though each pass has good spatial locality
- **Integer arithmetic cost**: The index functions involve modular arithmetic (integer division and modulus), which is relatively expensive on GPUs. Strength reduction helps but adds code complexity
- **Column rotation only needed when $\gcd(m, n) > 1$**: For coprime dimensions, the decomposition simplifies to 2 steps. For power-of-2 dimensions (common in ML), $\gcd$ is large, making all 3 steps necessary
- **Bandwidth-bound**: The operation is purely data movement — no arithmetic intensity to amortize memory access costs. On modern GPUs this means it will always be limited by HBM bandwidth

## Implementation Notes

```python
import torch

def decomposed_transpose_c2r(A):
    """In-place C2R transposition of m×n matrix A.

    Decomposes the transposition into:
    1. Column rotations (if gcd(m,n) > 1)
    2. Independent row shuffles
    3. Independent column shuffles

    Uses O(max(m,n)) auxiliary space.
    """
    m, n = A.shape
    import math
    c = math.gcd(m, n)
    a, b = m // c, n // c

    # Step 1: Column rotation (only if c > 1)
    if c > 1:
        tmp = torch.empty(m, dtype=A.dtype, device=A.device)
        for j in range(n):
            rot = j // b  # rotation amount for column j
            if rot > 0:
                col = A[:, j].clone()
                for i in range(m):
                    A[(i + rot) % m, j] = col[i]

    # Step 2: Row shuffle — each row independently
    # d'_i(j) gives the destination column for element (i, j)
    tmp = torch.empty(n, dtype=A.dtype, device=A.device)
    for i in range(m):
        for j in range(n):
            # Conflict-free destination column
            dest = ((i + (j // b)) % m + j * m) % n
            tmp[dest] = A[i, j]
        A[i, :] = tmp

    # Step 3: Column shuffle — each column independently
    tmp = torch.empty(m, dtype=A.dtype, device=A.device)
    for j in range(n):
        for i in range(m):
            # Source row for position i in column j
            src = (j + i * n - (i // a)) % m
            tmp[i] = A[src, j]
        A[:, j] = tmp

    # Now reinterpret as n×m (the dimensions are swapped)
    return A.reshape(n, m)  # logical reshape, no data movement


def gpu_transpose_kernel_sketch():
    """Sketch of how the GPU kernel would be structured.

    Key GPU optimization insights from the paper:
    1. All row operations are independent → one thread block per row
    2. All column operations are independent → one thread block per column
    3. Use shared memory for the temporary buffer (fits one row/col)
    4. Strength reduction: replace integer div/mod with multiply-shift
    5. Cache-aware column rotation: group columns into cache-line-sized
       blocks, do coarse rotation via cycle-following (analytic cycles),
       then fine rotation in shared memory
    """
    pass


# For the common case of AoS↔SoA with small structures:
def register_transpose_aos_soa(data, struct_size):
    """When struct_size is small (2-8 elements), the transposition
    can be done entirely in registers using the decomposition.

    Each SIMD lane loads one struct element, the decomposition
    determines which lane should receive which element, and
    shuffle instructions perform the exchange without shared memory.

    On K20c: 180 GB/s for AoS→SoA (45x faster than naive).
    """
    # The decomposition means we can express the transpose as:
    # 1. Within-register column rotations (warp shuffles)
    # 2. Cross-register row permutations (warp shuffles)
    # 3. Cross-register column permutations (warp shuffles)
    # Each step uses only __shfl_sync() instructions
    pass
```

## References

- Catanzaro, B., Keller, A. & Garland, M. "A Decomposition for In-place Matrix Transposition" PPoPP '14, ACM, 2014
- Wei, X. et al. "ITTPD: In-place Tensor Transposition with Permutation Decomposition on GPUs" HPC Asia 2025
- Sung, I.-J. et al. "Data layout transformation exploiting memory-level parallelism in structured grid many-core applications" PACT 2010
- Gustavson, F.G. "Recursion leads to automatic variable blocking for dense linear-algebra algorithms" IBM J. Res. Dev. 41(6), 1997

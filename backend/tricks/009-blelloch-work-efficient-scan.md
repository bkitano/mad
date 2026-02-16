# 009: Blelloch Work-Efficient Scan

**Category**: parallelization
**Gain type**: efficiency
**Source**: Blelloch (1990). Prefix Sums and Their Applications; Harris (2007). Parallel Prefix Sum (Scan) with CUDA.
**Paper**: papers/blelloch-prefix-sums-1993.pdf
**Documented**: 2026-02-12

## Description

The Blelloch scan is a two-phase parallel algorithm for computing all-prefix-sums (scan) that achieves *work-efficiency* — it performs $O(n)$ total operations, matching the sequential algorithm, while running in $O(\log n)$ parallel depth. This contrasts with the naive Hillis-Steele scan which has $O(\log n)$ depth but $O(n \log n)$ work. The algorithm builds a conceptual balanced binary tree over the data, performing an **up-sweep** (reduce) phase from leaves to root, then a **down-sweep** phase from root to leaves to distribute partial sums.

This is the foundational algorithm underlying parallel scans in SSMs (Mamba, S4), linear attention, and any model with linear recurrences. The chunkwise parallel scan (documented separately) builds *on top of* this algorithm by adding chunking for hardware efficiency.

## Mathematical Form

**Core Operation (All-Prefix-Sums):**

Given a binary associative operator $\oplus$ with identity $I$, and an array $[a_0, a_1, \ldots, a_{n-1}]$, the **inclusive scan** returns:

$$
[a_0, \; (a_0 \oplus a_1), \; \ldots, \; (a_0 \oplus a_1 \oplus \cdots \oplus a_{n-1})]
$$

The **exclusive scan** (prescan) returns:

$$
[I, \; a_0, \; (a_0 \oplus a_1), \; \ldots, \; (a_0 \oplus a_1 \oplus \cdots \oplus a_{n-2})]
$$

**Phase 1 — Up-Sweep (Reduce):**

Traverse from leaves to root, computing partial sums at internal tree nodes:

$$
\text{for } d = 0 \text{ to } \lceil\log_2 n\rceil - 1: \quad \text{for all } i \text{ from } 0 \text{ to } n-1 \text{ by } 2^{d+1} \text{ in parallel:}
$$
$$
a[i + 2^{d+1} - 1] \leftarrow a[i + 2^d - 1] \oplus a[i + 2^{d+1} - 1]
$$

After up-sweep, $a[n-1]$ holds the total reduction $a_0 \oplus a_1 \oplus \cdots \oplus a_{n-1}$.

**Phase 2 — Down-Sweep (Distribute):**

Set the root to identity, then traverse from root to leaves, distributing prefix sums:

$$
a[n-1] \leftarrow I
$$
$$
\text{for } d = \lceil\log_2 n\rceil - 1 \text{ downto } 0: \quad \text{for all } i \text{ from } 0 \text{ to } n-1 \text{ by } 2^{d+1} \text{ in parallel:}
$$
$$
t \leftarrow a[i + 2^d - 1] \quad \text{(save left child)}
$$
$$
a[i + 2^d - 1] \leftarrow a[i + 2^{d+1} - 1] \quad \text{(left child} \leftarrow \text{parent)}
$$
$$
a[i + 2^{d+1} - 1] \leftarrow t \oplus a[i + 2^{d+1} - 1] \quad \text{(right child} \leftarrow \text{saved} \oplus \text{parent)}
$$

**Key Theorem (Blelloch):** After a complete down-sweep, each vertex of the tree contains the sum of all the leaf values that precede it. The proof is inductive: if a parent has the correct prefix sum, both children must have the correct sum.

**Correctness of down-sweep rules:**

- Left child receives parent's value (sum of all leaves preceding the left subtree = sum preceding the parent)
- Right child receives $\text{sum}(L[v]) \oplus \text{prescan}[v]$, i.e., the left subtree's up-sweep total combined with the parent's prefix = sum of all leaves preceding the right subtree

## Complexity

| Metric | Naive (Hillis-Steele) | Blelloch | Sequential |
|--------|----------------------|----------|------------|
| Depth (parallel steps) | $O(\log n)$ | $O(2\log n)$ | $O(n)$ |
| Work (total operations) | $O(n \log n)$ | $O(n)$ | $O(n)$ |
| Operations per step | $O(n)$ | $O(n/2^d)$ (decreasing) | $O(1)$ |

**Memory:** $O(n)$ — in-place on the input array

**With $p$ processors and $n > p$:**

$$
T_S(n, p) = 2(\lceil n/p \rceil + \lceil \log p \rceil) = O(n/p + \log p)
$$

This is an optimal speedup over the sequential $O(n)$ when $n/p \geq \log p$.

**When to use which:**

- **Hillis-Steele**: When $n \leq p$ (more processors than data). Fewer steps ($\log n$ vs $2\log n$), but more total work. On a GPU warp of 32 threads, this is used for intra-warp scan since all threads execute in lockstep anyway.
- **Blelloch**: When $n > p$ (more data than processors). Work-efficiency dominates — the $\log n$ factor in Hillis-Steele becomes a 20x slowdown at $n = 10^6$.

## Applicability

- **State space models (Mamba, S4)**: The parallel scan over state transitions uses Blelloch's algorithm with matrix-valued elements and the associative operator being the composition of affine maps $(A, b) \bullet (C, d) = (CA, Cb + d)$
- **Linear attention**: Cumulative key-value aggregation is a scan operation
- **CTC loss computation**: Forward-backward algorithm is parallel via scan
- **GPU scan primitives**: NVIDIA's CUB and Thrust libraries implement Blelloch scan as their core primitive
- **Hierarchical GPU implementation**: Sengupta et al. (2008) compose the algorithm at three levels — intra-warp (Hillis-Steele, $w=32$), intra-block (Blelloch over warp results), and global (Blelloch over block results)

## Limitations

- **Double the parallel depth** compared to Hillis-Steele ($2\log n$ vs $\log n$ steps) — for very small arrays where work-efficiency doesn't matter, Hillis-Steele is faster
- **Shared memory bank conflicts**: The power-of-two stride pattern causes conflicts on GPUs; requires padding or address remapping to resolve
- **Synchronization barriers**: Each phase requires $\log n$ barrier synchronizations between steps; on GPUs this means $2\log n$ `__syncthreads()` calls
- **Not directly hardware-friendly for matrices**: When $\oplus$ is matrix multiplication (as in SSMs), the work per element is $O(d^3)$ for $d \times d$ matrices, and this doesn't map to tensor cores — motivating the chunked approach
- **Requires power-of-two arrays** (or padding to next power of two)

## Implementation Notes

```python
# Blelloch work-efficient scan (conceptual)
def blelloch_scan(a, op, identity):
    """
    In-place exclusive scan using Blelloch's algorithm.
    op: binary associative operator
    identity: identity element for op
    """
    n = len(a)

    # Phase 1: Up-sweep (reduce)
    for d in range(int(np.log2(n))):
        stride = 2 ** (d + 1)
        for i in range(0, n, stride):  # parallel
            a[i + stride - 1] = op(a[i + stride//2 - 1], a[i + stride - 1])

    # Phase 2: Down-sweep
    a[n - 1] = identity
    for d in range(int(np.log2(n)) - 1, -1, -1):
        stride = 2 ** (d + 1)
        for i in range(0, n, stride):  # parallel
            t = a[i + stride//2 - 1]
            a[i + stride//2 - 1] = a[i + stride - 1]        # left <- parent
            a[i + stride - 1] = op(t, a[i + stride - 1])    # right <- saved + parent

    return a

# GPU hierarchical scan (Sengupta et al. 2008 pattern):
# 1. Each thread serially scans k elements (register blocking)
# 2. Intra-warp: Hillis-Steele scan over 32 threads (no sync needed)
# 3. Intra-block: scan over per-warp partial results
# 4. Inter-block: scan over per-block partial results (separate kernel)
# 5. Each thread adds its block's offset to its local results
```

## References

- Blelloch, G.E. (1990). Prefix Sums and Their Applications. CMU-CS-90-190.
- Harris, M. (2007). Parallel Prefix Sum (Scan) with CUDA. NVIDIA GPU Gems 3.
- Sengupta, S., Harris, M., Garland, M. (2008). Efficient Parallel Scan Algorithms for GPUs. NVIDIA Technical Report NVR-2008-003.
- Hillis, W.D. and Steele, G.L. (1986). Data Parallel Algorithms. Communications of the ACM.
- Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. (Uses Blelloch scan for SSM state transitions.)

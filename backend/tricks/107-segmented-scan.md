# 107: Segmented Scan

**Category**: parallelization
**Gain type**: flexibility
**Source**: Blelloch (1990). Prefix Sums and Their Applications, Section 1.5; Sengupta, Harris, Garland (2008). Efficient Parallel Scan Algorithms for GPUs, Section 4.
**Paper**: papers/efficient-parallel-scan-gpus-2008.pdf
**Documented**: 2026-02-12

## Description

A segmented scan simultaneously performs independent parallel scans over arbitrary contiguous partitions ("segments") of an input sequence, restarting the accumulation at each segment boundary. The key insight is that segmented scan can be implemented by *transforming the operator* — given any associative operator $\oplus$, we construct a new operator $\oplus^s$ that operates on flag-value pairs and automatically respects segment boundaries. This means any existing unsegmented scan implementation instantly becomes a segmented scan with zero algorithmic changes.

For neural networks, segmented scans enable **batched variable-length sequence processing**: instead of padding sequences to the same length and wasting computation, pack multiple sequences into a single array with segment flags at boundaries, then run one segmented scan over the entire batch. This is critical for efficient training of SSMs and linear attention models on variable-length inputs.

## Mathematical Form

**Core Operation:**

Given operator $\oplus$ with identity $I_\oplus$, a data vector $a$, and a flag vector $f$ (where $f_i = 1$ marks the start of a new segment):

$$
\text{segscan}(a, f, \oplus)_i = \begin{cases}
a_0 & i = 0 \\
a_i & f_i = 1 \\
x_{i-1} \oplus a_i & f_i = 0
\end{cases}
$$

**Example:**

$$
a = [\; [3, 1] \;\;|\;\; [7, 0, 4] \;\;|\;\; [1, 6] \;\;|\;\; [3] \;]
$$
$$
f = [\; 1, 0, 1, 0, 0, 1, 0, 1 \;]
$$
$$
\text{segscan}(a, f, +) = [\; [3, 4] \;\;|\;\; [7, 7, 11] \;\;|\;\; [1, 7] \;\;|\;\; [3] \;]
$$

**Operator Transformation (Schwartz):**

Given operator $\oplus$, define operator $\oplus^s$ on flag-value pairs $(f_x, x)$:

$$
(f_x, x) \oplus^s (f_y, y) = (f_x \mid f_y, \;\; \text{if } f_y \text{ then } y \text{ else } x \oplus y)
$$

Equivalently:

$$
(f_x, x) \oplus^s (f_y, y) = \begin{cases}
(1, \; y) & \text{if } f_y = 1 \\
(f_x, \; x \oplus y) & \text{if } f_y = 0
\end{cases}
$$

**Theorem:** $\oplus^s$ is associative if $\oplus$ is associative. Therefore `scan<segmented<OP>>` applied to an array of flag-value pairs computes the segmented scan.

**Reduction to Standard First-Order Form:**

Using the "select" operator $x \times_s f$:

$$
x \times_s f = \begin{cases} I_\oplus & f = 1 \\ x & f = 0 \end{cases}
$$

The segmented scan satisfies:

$$
x_i = (x_{i-1} \times_s f_i) \oplus a_i
$$

This is in first-order recurrence form (Section 1.4 of Blelloch), so it can be solved via the tuple-augmented scan. The operator $\times_s$ is semiassociative with companion operator being logical OR.

**Direct Implementation (Conditional Indexing):**

For GPU warps, instead of operator transformation, track the minimum segment index (`mindex`) per thread:

$$
\text{mindex}[i] = \max\text{-scan}(\text{hd}[i] \cdot i)
$$

Then at each scan step with offset $2^d$, thread $i$ only accumulates from thread $i - 2^d$ if $\text{lane} \geq \text{mindex} + 2^d$ (i.e., they are in the same segment). This avoids the 2x overhead of operating on pairs.

## Complexity

| Operation | Unsegmented Scan | Segmented (Operator Transform) | Segmented (Direct) |
|-----------|-----------------|-------------------------------|-------------------|
| Work per element | $T_\oplus$ | $2T_\oplus + T_\text{flag}$ | $T_\oplus + T_\text{compare}$ |
| Parallel depth | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| Total work | $O(n)$ | $O(n)$ | $O(n)$ |
| Memory overhead | $0$ | $O(n)$ flags | $O(n)$ mindex |

**Memory:** Flags can be packed as bits (1 bit per element), but on GPUs they are typically stored as 32-bit integers for convenience. The `mindex` approach reuses the flag storage as scratch space.

**Compared to padding:** For $B$ sequences of varying lengths $L_1, \ldots, L_B$:
- Padding: processes $B \cdot L_\max$ elements, wasting $B \cdot L_\max - \sum L_i$ work
- Segmented scan: processes exactly $\sum L_i$ elements, zero waste

## Applicability

- **Batched SSM training**: Pack variable-length sequences into one array, run a single segmented scan instead of padding to max length. Critical for Mamba/S4 with heterogeneous batch lengths
- **Batched linear attention**: Cumulative KV states for multiple sequences via one segmented scan
- **Parallel quicksort**: Recursively split into segments, scan independently within each segment (Blelloch's original application)
- **Sparse matrix-vector multiply**: Each row is a segment; segmented scan computes per-row reductions
- **Beam search / tree search**: Multiple independent search paths processed as segments
- **Multi-head attention**: Each head's scan can be packed as a segment in a single scan call
- **Document-level training**: Multiple documents packed into one training sequence with segment boundaries between documents

## Limitations

- **~2x overhead with operator transformation**: Operating on pairs roughly doubles the computation per element compared to unsegmented scan
- **Memory overhead for flags**: Requires an additional flag array, though this is small (1 bit per element, typically stored as int32 for alignment)
- **Irregular segments hurt GPU utilization**: If segments have very different lengths, some warps finish early while others are still working
- **Not composable with all optimizations**: Some scan-specific GPU optimizations (e.g., warp shuffle instructions) need modification to handle segments correctly
- **Segment boundaries must be known**: Cannot handle data-dependent segmentation within the scan itself

## Implementation Notes

```python
import torch

# Method 1: Operator transformation (simple, general)
def segmented_scan_transform(values, flags, op, identity):
    """
    Segmented scan via operator transformation.
    values: (N,) data
    flags: (N,) binary segment boundaries (1 = new segment start)
    """
    # Transform operator: work on (flag, value) pairs
    def seg_op(pair_a, pair_b):
        fa, va = pair_a
        fb, vb = pair_b
        flag = fa | fb
        val = vb if fb else op(va, vb)
        return (flag, val)

    pairs = list(zip(flags, values))
    result = parallel_scan(pairs, seg_op)
    return [r[1] for r in result]

# Method 2: Direct (GPU-optimized, from Sengupta et al. 2008)
# CUDA pseudocode for intra-warp segmented scan:
"""
template<class OP, ScanKind Kind, class T>
__device__ T segscan_warp(volatile T *ptr, volatile flag_type *hd,
                          const unsigned int idx = threadIdx.x)
{
    const unsigned int lane = idx & 31;

    // Step 1: Convert head flags to minimum-index form
    if (hd[idx]) hd[idx] = lane;
    flag_type mindex = scan_warp<op_max, inclusive>(hd);

    // Step 2: Perform segmented scan — only combine within segment
    if (lane >= mindex + 1)  ptr[idx] = OP::apply(ptr[idx - 1], ptr[idx]);
    if (lane >= mindex + 2)  ptr[idx] = OP::apply(ptr[idx - 2], ptr[idx]);
    if (lane >= mindex + 4)  ptr[idx] = OP::apply(ptr[idx - 4], ptr[idx]);
    if (lane >= mindex + 8)  ptr[idx] = OP::apply(ptr[idx - 8], ptr[idx]);
    if (lane >= mindex + 16) ptr[idx] = OP::apply(ptr[idx - 16], ptr[idx]);

    return ptr[idx];
}
"""

# Method 3: For SSM batching — pack sequences and create flags
def pack_sequences_for_segmented_scan(sequences, A_list, b_list):
    """Pack variable-length sequences for batched segmented scan."""
    packed_A = torch.cat(A_list, dim=0)  # (sum(L_i), d)
    packed_b = torch.cat(b_list, dim=0)  # (sum(L_i), d)

    # Create segment flags
    flags = torch.zeros(packed_A.shape[0], dtype=torch.int32)
    offset = 0
    for seq in sequences:
        flags[offset] = 1  # Mark start of each sequence
        offset += len(seq)

    return packed_A, packed_b, flags
```

## References

- Blelloch, G.E. (1990). Prefix Sums and Their Applications. Section 1.5: Segmented Scans.
- Schwartz, J.T. (1980). Ultracomputers. ACM Transactions on Programming Languages and Systems.
- Sengupta, S., Harris, M., Garland, M. (2008). Efficient Parallel Scan Algorithms for GPUs. NVIDIA Technical Report NVR-2008-003.
- Sengupta, S., Harris, M., Zhang, Y., Owens, J.D. (2007). Scan Primitives for GPU Computing. Graphics Hardware.
- Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality (Mamba-2). (Uses segmented scans for batched SSM processing.)

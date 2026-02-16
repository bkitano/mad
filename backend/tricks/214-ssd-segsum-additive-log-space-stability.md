# 214: SSD Segsum — Additive-Only Log-Space Stability for Semiseparable Matrices

**Category**: stability
**Gain type**: efficiency
**Source**: Dao & Gu (2024) — "Transformers are SSMs" (Mamba-2), ICML 2024. arXiv:2405.21060
**Paper**: [papers/mamba2-transformers-are-ssms.pdf]
**Documented**: 2026-02-15

## Description

The State Space Dual (SSD) algorithm in Mamba-2 computes outputs through 1-semiseparable matrix multiplication, where the mask matrix entries are exponentials of cumulative sums of log-decay values. The naive approach to computing the segment sums $\log a_i + \log a_{i+1} + \cdots + \log a_{j-1}$ for all pairs $(i, j)$ would compute a single cumulative sum and then take differences: $\text{cumsum}[j] - \text{cumsum}[i]$. This suffers from **catastrophic cancellation** — when the cumulative sums grow large (which they do for long sequences), subtracting two nearly-equal large numbers destroys precision.

Without this fix, the SSD algorithm produces **NaNs immediately during training**, even in FP32.

The `segsum` trick replaces the subtraction-based approach with an **additive-only** construction: the input log-decay vector is replicated into a 2D matrix, masked to keep only the appropriate triangular region, and then a single cumulative sum along one axis produces all pairwise segment sums directly — **no subtraction required**. The result is exponentiated to produce the 1-semiseparable mask matrix $L = \exp(\text{segsum}(A))$.

This is a critical enabler for Mamba-2's chunkwise-parallel SSD algorithm, which uses this mask matrix in both the intra-chunk diagonal blocks (Step 1) and the inter-chunk state propagation (Step 3).

## Mathematical Form

**The 1-Semiseparable Matrix:**

For scalar decay parameters $a_0, a_1, \ldots, a_{T-1}$, the 1-semiseparable matrix $L = \text{1SS}(a)$ has entries:

$$
L_{ij} = \begin{cases} \prod_{k=i}^{j-1} a_k = a_{i:j}^{\times} & \text{if } j \geq i \\ 0 & \text{if } j < i \end{cases}
$$

In log space, the product becomes a sum:

$$
\log L_{ij} = \sum_{k=i}^{j-1} \log a_k \quad \text{for } j \geq i
$$

**Naive approach (unstable):**

Compute a single cumulative sum $s_k = \sum_{m=0}^{k-1} \log a_m$, then:

$$
\log L_{ij} = s_j - s_i \quad \text{(UNSTABLE — catastrophic cancellation)}
$$

When $s_i$ and $s_j$ are both large (e.g., $\sim 10^3$) but their difference is small (e.g., $\sim 1$), the subtraction loses significant digits in limited precision.

**Segsum approach (stable):**

Given log-decay vector $x \in \mathbb{R}^T$ (where $x_k = \log a_k$):

1. **Replicate:** Form a matrix by repeating $x$ along a new axis: $X_{ij} = x_i$ for all $j$
2. **Mask lower triangle:** Zero out entries where $i \geq j$ (keep strictly below diagonal): $X_{ij} \leftarrow x_i \cdot \mathbb{1}[i < j]$
3. **Cumulative sum along rows:** $S = \text{cumsum}(X, \text{dim}=-2)$
4. **Mask upper triangle:** Set entries above the diagonal to $-\infty$: $S_{ij} \leftarrow -\infty$ for $i < j$

The result $S_{ij}$ gives the segment sum $\sum_{k=i}^{j-1} x_k$ for $j \geq i$, and $-\infty$ for $j < i$.

**Key insight:** Step 3 only uses **addition** (via cumsum). There is no subtraction of large quantities. Each entry $S_{ij}$ is built up additively from zero, accumulating only the terms $x_i, x_{i+1}, \ldots, x_{j-1}$.

**Final mask matrix:**

$$
L = \exp(S) = \exp(\text{segsum}(\log a))
$$

where $\exp(-\infty) = 0$ naturally handles the causal masking.

**Key Definitions:**

- $a \in \mathbb{R}^T$ — per-timestep scalar decay values (in $[0, 1]$ for stable SSMs)
- $x = \log a \in \mathbb{R}^T$ — log-decay values (negative for decaying systems)
- $L \in \mathbb{R}^{T \times T}$ — the resulting 1-semiseparable mask matrix (lower triangular)
- $T$ — sequence length (or chunk length $Q$ in the block-decomposed SSD)

## Complexity

| Operation | Naive (subtraction) | Segsum (additive) |
|-----------|--------------------|--------------------|
| Compute segment sums | $O(T)$ cumsum + $O(T^2)$ subtraction | $O(T^2)$ cumsum |
| Memory | $O(T)$ for cumsum + $O(T^2)$ for result | $O(T^2)$ for replicated matrix |
| Numerical precision | Catastrophic cancellation at large $T$ | Stable — additions only |
| GPU operations | Subtraction → branch-free but imprecise | Cumsum → standard parallel prefix sum |

**Memory:** $O(T^2)$ — same as the output mask matrix. In the SSD algorithm, $T = Q$ (chunk size, typically 64–256), so this is $O(Q^2)$ per chunk, fitting easily in SRAM.

**FLOPs:** $O(T^2)$ for the cumulative sum over the replicated matrix. This is negligible compared to the $O(T^2 N)$ cost of the intra-chunk matmuls in SSD (where $N$ is the state/head dimension).

## Applicability

- **Mamba-2 SSD (primary):** Used in two places in the SSD block decomposition:
  - **Step 1 (intra-chunk):** `L = exp(segsum(A))` computes the $Q \times Q$ decay mask for the attention-like intra-chunk computation
  - **Step 3 (inter-chunk):** `decay_chunk = exp(segsum(pad(A_cumsum[:,:,:,-1])))` computes the decay between chunk boundaries for inter-chunk state propagation

- **Any chunkwise-parallel linear recurrence** with multiplicative gating: The segsum trick applies whenever you need to compute pairwise products of decay values over segments. This includes GLA (trick 177), GateLoop, RetNet, and DeltaNet chunkwise forms.

- **Log-space scan operations:** Any parallel scan that operates in log space to avoid overflow/underflow in products can benefit from this additive-only construction.

- **General 1-semiseparable matrix construction:** Whenever one needs to build a 1-SS matrix from its generator sequence and the generator values span a wide dynamic range.

## Limitations

- **Quadratic memory in chunk size:** The segsum materializes a $T \times T$ matrix (or $Q \times Q$ per chunk). For the SSD algorithm this is fine since $Q$ is small (64–256), but it cannot be applied to full-length sequences without chunking.

- **The replicate-and-cumsum pattern is not a standard matmul:** While the cumsum is parallelizable (prefix sum), it doesn't map to tensor cores. However, in SSD this cost is negligible compared to the actual matmuls (BMM terms).

- **Still requires exponentiation after segsum:** The `exp()` call can itself overflow for very large positive segment sums. In practice, the decay values $a_t \in (0, 1]$ ensure $\log a_t \leq 0$, so segment sums are non-positive and `exp` produces values in $(0, 1]$. For the inter-chunk case, padding with zeros handles boundary conditions.

- **Not needed for scalar-only recurrences:** If the recurrence can be computed purely as a parallel associative scan (e.g., Mamba-1's selective scan), this matrix construction is unnecessary. It's specifically needed for the chunkwise-parallel algorithm that materializes the quadratic attention-like form.

## Implementation Notes

```python
def segsum(x):
    """
    Stable segment sum: computes pairwise segment sums of x
    without subtraction-based catastrophic cancellation.

    Input:  x of shape (..., T) — log-decay values
    Output: S of shape (..., T, T) — where S[..., i, j] = sum(x[i:j]) for j >= i
                                      and S[..., i, j] = -inf for j < i

    The key trick: replicate x into a 2D matrix, mask, and cumsum.
    NO SUBTRACTION anywhere — only addition via cumsum.
    """
    T = x.size(-1)
    # Replicate: x[..., i] -> X[..., i, j] = x[..., i] for all j
    x = repeat(x, "... d -> ... d e", e=T)
    # Mask: keep only strictly-lower-triangular entries (i < j)
    # These are the terms x[i] that contribute to segment sum ending at j
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # Cumsum along the row axis: S[..., i, j] = x[0] + x[1] + ... + x[i]
    # But only for the masked entries, so S[i,j] = sum of x[i..j-1]
    x_segsum = torch.cumsum(x, dim=-2)
    # Mask upper triangle to -inf (causal: j < i means no contribution)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

# Usage in SSD algorithm (Listing 1 from Mamba-2 paper):
# Step 1: Intra-chunk diagonal blocks
A_cumsum = torch.cumsum(A, dim=-1)  # cumulative log-decay within chunk
L = torch.exp(segsum(A))            # Q x Q mask matrix, STABLE
Y_diag = einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, X)

# Step 3: Inter-chunk state propagation
decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:,:,:,-1], (1, 0))))
new_states = einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)

# WHY THIS WORKS:
# Naive:  segsum[i,j] = cumsum[j] - cumsum[i]  (SUBTRACTION of large numbers)
# Stable: segsum[i,j] = cumsum over masked 2D array (ADDITION only)
#
# Example with x = [-0.5, -0.3, -0.7, -0.2]:
#
# After replicate + lower-tri mask:
# [  0     0     0     0  ]
# [-0.5    0     0     0  ]
# [-0.5  -0.3    0     0  ]
# [-0.5  -0.3  -0.7    0  ]
#
# After cumsum along dim=-2:
# [  0     0     0     0  ]
# [-0.5    0     0     0  ]
# [-1.0  -0.3    0     0  ]
# [-1.5  -0.6  -0.7    0  ]
#
# After upper-tri mask with -inf:
# [  0    -inf  -inf  -inf]
# [-0.5    0   -inf  -inf]
# [-1.0  -0.3    0   -inf]
# [-1.5  -0.6  -0.7    0 ]
#
# segsum[i,j] = log(a_i * a_{i+1} * ... * a_{j-1})
# e.g., segsum[0,2] = x[0] + x[1] = -0.5 + -0.3 = -0.8 ✓
```

**GPU Efficiency Analysis:**

- **Memory access:** The replicate + mask + cumsum pattern has coalesced memory access. The $T \times T$ matrix fits in shared memory for typical chunk sizes ($T = 64$: 16KB in FP32, $T = 256$: 256KB).
- **Parallelism:** The cumsum along dim=-2 is a standard parallel prefix sum, fully parallelizable across the $T$ columns. Each column's prefix sum is independent.
- **Arithmetic intensity:** Low — this is a memory-bound operation. But since $T = Q$ (chunk size) is small, the entire computation fits in SRAM and the cost is negligible.
- **No tensor core usage:** The segsum itself doesn't use tensor cores, but it produces the mask matrix that is immediately consumed by the tensor-core matmuls in the SSD einsum operations.
- **Fused in practice:** In the Triton implementation (trick 182), the segsum is fused into the same kernel as the subsequent matmuls, avoiding any HBM round-trip for the $T \times T$ mask matrix.

## References

- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060.
- Dao, T. (2024). State Space Duality (Mamba-2) Blog Series, Part III — The Algorithm. https://tridao.me/blog/2024/mamba2-part3-algorithm/
- Mamba GitHub: https://github.com/state-spaces/mamba (ssd_minimal.py reference implementation)
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. (Related log-space computation in GLA)

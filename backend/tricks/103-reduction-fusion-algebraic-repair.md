# 103: Reduction Fusion with Algebraic Repair (Rolling Update & Split-K)

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhao, Johnson, Chatarasi, Adve, Misailovic — "Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs" (arXiv 2025)
**Paper**: [papers/neptune-operator-fusion-locality.pdf]
**Documented**: 2026-02-15

## Description

Reduction fusion with algebraic repair is a compiler transformation technique that enables fusing sequences of *reduction operators* with complex loop-carried data dependencies — cases where standard loop fusion produces incorrect results. The key insight is that fusion can be made correct by **intentionally breaking data dependencies** and then **constructing algebraic correction expressions** ("repair terms") that restore correctness with minimal extra computation.

This addresses a fundamental limitation of existing tensor compilers (TVM, Triton, Halide, XLA): they cannot fuse reduction loops that have inter-loop data dependencies. For example, in attention computation, the softmax normalization (row-max, exp, row-sum) and the subsequent matmul form a sequence of reductions where `s_exp` depends on `s_max` through a loop-carried dependency. Naively fusing them produces incorrect results because `s_exp` reads intermediate values of `s_max` that haven't converged yet. Today's compilers either reject the fusion or require hand-written kernels like FlashAttention.

Neptune introduces two instantiations of this paradigm:

1. **Rolling Update Fusion**: Fuses dependent reduction loop nests into a single loop by inlining predecessors and applying a *repair function* $h$ that corrects the running result at each iteration. This produces kernels similar to FlashAttention's online softmax — but automatically derived by the compiler. Well-suited for **prefill** (long sequences, large reduce dimensions).

2. **Split-K Fusion**: Splits the reduction into local and global phases via *privatization*, where the local phase runs in parallel across tiles and the global phase combines results using the repair function. This trades sequentiality for parallelism at the cost of a global synchronization. Well-suited for **decoding** (short sequences, need maximum parallelism), producing kernels similar to FlashDecoding.

On 10 attention-based operators across 4 GPU architectures (NVIDIA A100, RTX A5000, RTX 6000 Ada; AMD MI300), Neptune achieves **1.35x geomean speedup** over the best compiler baseline (Triton, TVM, FlexAttention), outperforming in 284 out of 320 configurations.

## Mathematical Form

**The reduction fusion problem:**

Consider a program with three reduction loop nests over dimension $j \in [0, n)$:

$$
\texttt{s\_max}: \quad m_i^{(j)} = \max(m_i^{(j-1)}, x_{ij})
$$
$$
\texttt{s\_exp}: \quad p_{ij} = \exp(x_{ij} - m_i^{(j)})
$$
$$
\texttt{s\_sum}: \quad s_i^{(j)} = s_i^{(j-1)} + p_{ij}
$$

Here $m_i^{(j)}$ is the running row-max and $s_i^{(j)}$ is the running row-sum. The dependency $p_{ij} = \exp(x_{ij} - m_i^{(j)})$ requires the *final* value $m_i^{(n-1)}$, but naive fusion would read the *intermediate* $m_i^{(j)}$ at iteration $j < n-1$, producing incorrect results.

**Rolling Update — the repair function:**

The correctly fused program adds a *repair term* that multiplies the running sum by a correction factor at each iteration:

$$
s_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) \cdot s_i^{(j-1)} + \exp(x_{ij} - m_i^{(j)})
$$

The repair term $\exp(m_i^{(j-1)} - m_i^{(j)})$ compensates for the fact that all previous $p_{ij'}$ values were computed with stale max values. This is exactly the online softmax trick, but Neptune derives it automatically.

**General formalization:**

For a target reduction $L_t$ with output $X_t$ and a reduce predecessor $L_r$ with output $X_r$, the fused loop nest has the general form:

$$
X_t[\phi_t(\mathbf{i})] = X_t[\phi_t(\mathbf{i})] \bigodot h\!\left(X_t[\phi_t(\mathbf{i})]^{(\text{prev}(j))}, X_r[\phi_r(\mathbf{i})]^{(j)}\right) \bigodot g\!\left(X_r[\phi_r(\mathbf{i})]^{(j)}, C[\phi_c(\mathbf{i}, j)]\right)
$$

where:
- $\bigodot$ denotes the reducer's associative binary operation (e.g., $+$ for sum, $\max$ for max)
- $h$ is the **repair function** (tag-updating function)
- $g$ is the original computation body
- $\phi_t, \phi_r, \phi_c$ are affine access functions
- Superscript $(j)$ denotes the value at iteration $j$ (a "tag")

**Definition (Tag-Updating):** A function $h$ is *tag-updating* if it satisfies:

$$
\forall j \in \mathcal{D}^s: \quad h\!\left(\mathcal{R}(f, 0 \leq j' \leq j, g(X_r[\phi_r(\mathbf{i})]^{(j')}, C[\phi_c(\mathbf{i}, j)])), X_r[\phi_r(\mathbf{i})]^{(\text{next}(j))}\right)
$$
$$
= \mathcal{R}\!\left(f, 0 \leq j' \leq j, g(X_r[\phi_r(\mathbf{i})]^{(\text{next}(j))}, C[\phi_c(\mathbf{i}, j)])\right)
$$

where $\mathcal{R}(f, 0 \leq j \leq j_0, g(j))$ applies reducer $f$ to fold over $g(j)$.

**Sufficient conditions for finding $h$ (Theorem 4.2):**

If $h$ satisfies:
1. $h(g(r,c), r, r') = g(r', c) \quad \forall r, r', c \in \mathcal{X}$ (h replaces the $r$ argument of $g$)
2. $h(x \bigodot y, r, r') = h(x, r, r') \bigodot h(y, r, r') \quad \forall x, y, r, r' \in \mathcal{X}$ (h commutes with reducer)

then $h$ is tag-updating.

**Constructive solution (Theorem 4.3):**

If $y = g(x, c)$ is invertible in the second argument (i.e., $c = g_c^{-1}(x, y)$), then:

$$
h(t, r, r') = g(r', g_c^{-1}(r, t))
$$

For the softmax example: $g(r, c) = \exp(c - r)$ where $r = m_i$ (row-max), $c = x_{ij}$ (input). Then $g_c^{-1}(r, t) = \ln(t) + r$, giving:

$$
h(t, r, r') = g(r', \ln(t) + r) = \exp(\ln(t) + r - r') = t \cdot \exp(r - r')
$$

This is exactly the $\exp(m_{\text{old}} - m_{\text{new}})$ rescaling factor from FlashAttention's online softmax.

**Split-K Fusion:**

Split-K partitions the reduce loop $l$ of size $n$ into tiles of size $k$:

**Local phase** (parallelizable across tiles $j_0 \in [0, n/k)$):

$$
s_{\text{local}}[i, j_0] = \sum_{j_1=0}^{k-1} \exp(x_{i, j_0 k + j_1} - m_{\text{local}}[i, j_0])
$$

$$
m_{\text{local}}[i, j_0] = \max_{j_1 \in [0, k)} x_{i, j_0 k + j_1}
$$

**Global phase** (sequential reduction over $n/k$ partial results):

$$
m_{\text{global}}[i] = \max_{j_0} m_{\text{local}}[i, j_0]
$$

$$
s_{\text{global}}[i] = \sum_{j_0} \exp(m_{\text{local}}[i, j_0] - m_{\text{global}}[i]) \cdot s_{\text{local}}[i, j_0]
$$

The repair function $h$ is only applied in the global phase, correcting local partial sums to use the global max. This enables full parallelism in the local phase at the cost of one global synchronization.

**Key Definitions:**

- Reduce loop nest — A loop nest containing a reduction (e.g., sum, max) with a loop-carried accumulator
- Repair function $h$ — An algebraically derived correction term that makes naive fusion correct by tag-updating the running accumulator
- Tag — A marker on tensor accesses indicating which iteration's value is referenced (e.g., $m_i^{(j)}$ vs. $m_i^{(j-1)}$)
- Privatization — Splitting a reduce loop into local (per-tile) and global (cross-tile) reductions
- Reducer $\mathcal{R}$ — An associative binary function $f$ applied via fold: $\mathcal{R}(f, j, g(j))$

## Complexity

| Approach | HBM Traffic | Parallelism | Synchronization |
|----------|------------|-------------|-----------------|
| Unfused (separate kernels) | $\sim 2k \cdot N$ per reduction | Full per-op | $k$ kernel launches |
| Rolling Update (fused) | $\sim 2N$ (single pass) | Sequential over reduce dim | None (single kernel) |
| Split-K (fused) | $\sim 2N + 2N/k$ (local + global) | Parallel local, sequential global | 1 global barrier |

**Extra computation from repair terms:**

- Rolling Update: $O(1)$ extra FLOPs per element per iteration (repair term is a short expression, e.g., one exp + one multiply)
- Split-K: $O(N/k)$ extra FLOPs in global phase (repair applied to $N/k$ partial results)

**Neptune vs. compiler baselines (geomean speedup, 10 attention operators, 4 GPUs):**

| GPU | vs. Best Compiler | vs. Triton | vs. TVM | vs. FlexAttn |
|-----|-------------------|------------|---------|--------------|
| RTX 6000 Ada | **1.15x** | 1.15x | 3.2x | 1.08x |
| RTX A5000 | **1.14x** | 1.14x | 4.1x | 1.07x |
| A100 | **1.38x** | 1.38x | 3.8x | 1.21x |
| AMD MI300 | **1.85x** | 1.85x | — | — |

**Neptune vs. manually optimized libraries (average across operators):**

| GPU | vs. cuDNN | vs. CUTLASS | vs. FlashInfer |
|-----|----------|-------------|----------------|
| A100 | 0.95x | 1.04x | — |
| Overall average | **1.07x** the best library | | |

**Resource overhead:** Neptune uses 0.68x the registers and 47% the SMEM of the baseline that uses the most of each resource.

## Applicability

- **Attention operators (prefill)**: Rolling Update Fusion produces FlashAttention-like kernels automatically from a simple attention specification — covering global, causal, GQA, ALiBi, SoftCap, and windowed attention in 38 lines of tensor expression + 28 lines of schedule
- **Attention operators (decoding)**: Split-K Fusion produces FlashDecoding-like kernels with full parallelism across the KV sequence dimension, critical for low-latency single-token generation
- **Custom attention variants**: Any attention variant (new masking pattern, score modification, normalization scheme) can be expressed as a tensor expression and automatically fused — no hand-written CUDA required
- **Softmax-containing operators**: Any operator with the max → exp → sum reduction pattern (cross-entropy loss, Sinkhorn normalization, log-sum-exp) benefits from automatic repair function derivation
- **SSM/linear attention**: Operators with recurrences that involve element-wise dependencies between reductions (e.g., gated state updates depending on running statistics)
- **Multi-GPU extensions**: The repair function paradigm is orthogonal to communication overlap — it could be combined with persistent megakernel fusion or Flux-style tile-level communication

## Limitations

- **Invertibility requirement**: The constructive solution for the repair function $h$ (Theorem 4.3) requires $g(x, c)$ to be invertible in $c$. Functions where this doesn't hold require alternative derivation or manual specification
- **Floating-point associativity**: Rolling update assumes the reducer $f$ is associative, which is true for real numbers but only approximately true for floating-point. The paper empirically validates numerical accuracy but does not provide formal error bounds
- **Repair term overhead**: Each repair term adds computation per iteration. For operators where the reduce body is already compute-heavy, this overhead may not be negligible
- **Sequential reduce dimension**: Rolling Update Fusion inherits sequentiality over the reduce dimension — it cannot parallelize the inner loop. Split-K addresses this but adds a global synchronization
- **Template-guided**: Neptune requires the user to provide a transformation template (schedule) specifying which optimizations to apply. While concise (28 lines for attention), it is not fully automatic — the user must choose between Rolling Update and Split-K based on the workload
- **Window attention limitation**: Neptune's TVM-based loop analysis could not identify the proper condition for loop partitioning in Window attention, causing it to underperform on this operator
- **No training support**: The paper evaluates inference-mode operators only; backward pass fusion with repair terms for gradient computation is not addressed

## Implementation Notes

```python
# Neptune's attention specification (38 lines of tensor expression)
# This replaces 650+ lines of hand-written Triton kernels

def create_general_attention(B, N, QS, KVS, H, mask_cond, score_mod):
    q = placeholder((B, N, QS, H), "float16", name="q")
    k = placeholder((B, N, KVS, H), "float16", name="k")
    v = placeholder((B, N, KVS, H), "float16", name="v")

    # QK^T matmul
    p = batch_matmul(q, k, trans_b=True, out_dtype="float32")

    # Score modification (masking, ALiBi bias, etc.)
    score = compute(p.shape, lambda *ax: if_then_else(
        mask_cond(*ax), score_mod(p(*ax), *ax), float("-inf")),
        name="score_mod")

    # Online softmax reductions
    j = reduce_axis((0, KVS), name="j")
    s_max = compute((B, N, QS),
        lambda b, n, i: max(score(b, n, i, j), axis=j),
        name="softmax_maxlen")
    s_exp = compute((B, N, QS, KVS),
        lambda b, n, i, j: exp(score(b, n, i, j) - s_max(b, n, i)),
        name="softmax_exp")
    s_expsum = compute((B, N, QS),
        lambda b, n, i: sum(s_exp(b, n, i, j), axis=j),
        name="softmax_expsum")

    # Normalize and matmul with V
    sv = batch_matmul(s_exp, v, trans_b=False, out_dtype="float32")
    return compute(sv.shape,
        lambda b, n, i, j: sv(b, n, i, j) / s_expsum(b, n, i),
        name="softmax_norm")


# Neptune's schedule for rolling update (28 lines)
# This tells the compiler HOW to fuse — Neptune derives repair terms automatically

def schedule_attn_with_rolling_update(sch):
    b0 = sch.get_block("batch_matmul_1")       # QK^T
    b1 = sch.get_block("T_score_mod")           # masking
    b2 = sch.get_block("T_softmax_maxelen")     # row-max reduction
    b3 = sch.get_block("T_softmax_exp")         # exp(score - max)
    b4 = sch.get_block("T_softmax_exp_cast")    # cast to fp16
    b5 = sch.get_block("T_batch_matmul_NN")     # P @ V matmul
    b6 = sch.get_block("T_softmax_expsum")      # row-sum reduction
    b7 = sch.get_block("T_softmax_norm")        # normalize
    b8 = sch.get_block("T_cast")                # output cast

    # Tile the computation
    *axes, i, j, k = sch.get_loops(b0)
    i0, j0 = sch.tile([i, j], [128, 32])
    sch.compute_at(sch.cache_read(b0, 0, "shared"), i0)
    sch.bind_block_idx(
        [*axes, i0],
        ["blockIdx.x", "blockIdx.y", "blockIdx.z"])

    # Apply rolling update fusion
    # Neptune AUTOMATICALLY derives the repair function h for each block
    sch.reverse_compute_at(b1, j0)
    b2rf = sch.rolling_update(b2, j0, factor_axis=0)  # fuse s_max
    b6rf = sch.rolling_update(b6, j0, factor_axis=0)  # fuse s_sum
    b5rf = sch.rolling_update(b5, j0, factor_axis=0)  # fuse P@V
    sch.reverse_compute_at(b7, i0)
    sch.reverse_compute_at(b8, i0)

    # Low-level tile optimizations (handled by tile compiler)
    for blk in [b0, b1, b2, b2rf, b3, b4, b5, b5rf, b6, b6rf, b7]:
        sch.set_scope(blk, 0, "shared")
    sch.split_scan_buffer(b2, j0, 0)
    sch.decompose_reduction(b5, j0)
    sch.decompose_reduction(b6, j0)


# What Neptune does internally for rolling update of s_sum:
#
# 1. Pattern match: s_sum depends on s_max through s_exp
# 2. Identify g(r, c) = exp(c - r) where r = s_max, c = score
# 3. Compute g_c^{-1}(r, t) = ln(t) + r
# 4. Derive h(t, r, r') = exp(ln(t) + r - r') = t * exp(r - r')
# 5. Rewrite fused loop to apply h at each iteration:
#    s_sum[i] = exp(max_old - max_new) * s_sum[i] + exp(score - max_new)
#
# This is exactly FlashAttention's online softmax — derived automatically!
```

## References

- Zhao, Y., Johnson, E., Chatarasi, P., Adve, V., Misailovic, S. "Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs." arXiv:2510.08726, October 2025.
- Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
- Dao, T., Sizov, G., Massa, F., Haziza, D. "FlashDecoding for Long-Context Inference." 2023.
- Chen, T., et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
- Ragan-Kelley, J., et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." PLDI 2013.
- Tillet, P., et al. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MLSys 2019.
- Dong, J., et al. "FlexAttention: A Programming Model for Generating Optimized Attention Kernels." 2024.
- Neptune source code: https://github.com/uiuc-arc/neptune

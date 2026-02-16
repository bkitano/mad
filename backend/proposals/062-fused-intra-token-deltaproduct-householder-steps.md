---
status: ongoing
priority: high
created: 2026-02-15
based_on: deltaproduct-multi-step-householder-wy (178), flashrnn-io-aware-fused-recurrence (212), gated-deltanet-chunkwise-wy-gating (203), fused-chunkwise-ssd-atomic-state-passing (182), io-aware-tiling (066), warp-specialized-pipelining (141)
experiment_number: 062
experiment_log: experiment-log-062.md
---

# Fused Intra-Token DeltaProduct: Register-Persistent Householder Steps Without Sequence Inflation

## Hypothesis

Fusing DeltaProduct's $n_h$ intra-token Householder steps into a **register-persistent loop within the chunkwise kernel** — rather than flattening them into an extended sequence of length $n_h T$ — will achieve **$1.4$–$1.8\times$ training throughput improvement** for DeltaProduct$_2$ and **$2.0$–$2.5\times$ for DeltaProduct$_3$**, by eliminating:

1. The $n_h \times$ inflation of the UT transform cost: from $O((n_h C)^2 d)$ back to $O(C^2 d)$ plus a fused $O(n_h C d)$ inner loop
2. The $n_h \times$ inflation of HBM traffic for keys, values, and gates
3. The waste of processing $n_h - 1$ "dummy" outputs per token that are immediately discarded

This would make DeltaProduct$_2$ (16 heads, $d_h = 128$) achieve **~45–50K tokens/sec** — matching or exceeding DeltaNet (8 heads, $d_h = 256$) at 40K tokens/sec — while retaining DeltaProduct's superior expressivity, length extrapolation, and state-tracking capabilities.

## Background

### DeltaProduct's throughput bottleneck

DeltaProduct takes $n_h$ gradient descent steps per token, each applying a Householder reflection $(\boldsymbol{I} - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top)$ to the state. The current implementation handles this by **flattening** the multi-step structure into the sequence dimension:

$$
[\boldsymbol{k}_{1,1}, \boldsymbol{k}_{1,2}, \ldots, \boldsymbol{k}_{1,n_h}, \boldsymbol{k}_{2,1}, \boldsymbol{k}_{2,2}, \ldots] \quad \text{(length } n_h T \text{)}
$$

This reuses DeltaNet's existing chunkwise kernel unchanged but inflates all costs by $n_h \times$:

| Cost | DeltaNet | DeltaProduct (flattened) | DeltaProduct (fused, proposed) |
|------|----------|--------------------------|-------------------------------|
| Sequence length | $T$ | $n_h T$ | $T$ |
| UT transform | $O(C^2 d)$ | $O((n_h C)^2 d)$ | $O(C^2 d + n_h C d)$ |
| HBM traffic (KVβ) | $O(T d)$ | $O(n_h T d)$ | $O(T n_h d)$ (but via SRAM) |
| Outputs computed | $T$ | $n_h T$ (discard $n_h - 1$ per token) | $T$ |

The UT transform inflation is particularly wasteful: the $(n_h C) \times (n_h C)$ lower-triangular inverse involves $n_h^2$ times more work than the $C \times C$ version, even though only $C$ of the $n_h C$ outputs are needed.

### The key insight: intra-token steps are local

Within a single token $i$, the $n_h$ Householder steps are:

$$
\boldsymbol{H}_{i,j} = (\boldsymbol{I} - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top) \boldsymbol{H}_{i,j-1} + \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{v}_{i,j}^\top, \quad j = 1, \ldots, n_h
$$

These steps are **purely sequential within token $i$** and depend only on $\boldsymbol{H}_{i,0} = \boldsymbol{H}_{i-1}$ (the state from the previous token). The inputs $(\boldsymbol{k}_{i,j}, \boldsymbol{v}_{i,j}, \beta_{i,j})$ are all precomputed from $\boldsymbol{x}_i$.

This means we can compute the per-token product transition and accumulated input **inside** the chunkwise kernel, in registers, without materializing the intermediate states to HBM or inflating the sequence:

**Per-token fused computation:**

$$
\boldsymbol{A}_i = \prod_{j=1}^{n_h} (\boldsymbol{I} - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top) \in \mathbb{R}^{d_k \times d_k}
$$

$$
\boldsymbol{B}_i = \sum_{j=1}^{n_h} \left(\prod_{l=j+1}^{n_h} (\boldsymbol{I} - \beta_{i,l} \boldsymbol{k}_{i,l} \boldsymbol{k}_{i,l}^\top)\right) \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{v}_{i,j}^\top \in \mathbb{R}^{d_k \times d_v}
$$

Then the recurrence at the token level becomes:

$$
\boldsymbol{H}_i = \boldsymbol{A}_i \boldsymbol{H}_{i-1} + \boldsymbol{B}_i
$$

This is a standard linear recurrence over $T$ tokens with dense (but structured) $\boldsymbol{A}_i$. The UT transform in the chunkwise algorithm now operates on the $C \times C$ token-level Gram matrix, not the $(n_h C) \times (n_h C)$ flattened one.

### Connection to FlashRNN

FlashRNN showed that fusing sequential steps with register-persistent state gives massive speedups by eliminating HBM round-trips. Our approach applies the same principle, but at the **intra-token level** rather than the intra-sequence level:

| | FlashRNN | Fused DeltaProduct (proposed) |
|---|---------|-------------------------------|
| What's fused | Time steps $t = 1, \ldots, T$ | Householder steps $j = 1, \ldots, n_h$ per token |
| What's register-resident | Weight matrix $R$ | Keys $\boldsymbol{k}_{i,j}$, betas $\beta_{i,j}$ |
| HBM traffic eliminated | $T$ reads of $R$ | $n_h$ reads of intermediate states per token |
| Sequential dependency | Cross-token (fundamental) | Intra-token (can be fused) |

### What's different from proposal 057

Proposal 057 addresses **inter-chunk** state propagation (the sequential scan across chunks $k = 1, \ldots, G$). Our proposal addresses **intra-token** step fusion within DeltaProduct (the $n_h$ Householder steps per token). These are orthogonal optimizations at different levels of the computation hierarchy.

## Related Work

- **DeltaProduct** (Siems et al., NeurIPS 2025): Introduced the multi-step Householder recurrence. Explicitly notes "A custom kernel that fuses the $n_h$ steps within a single chunk could potentially be faster" as future work (Section 4.3 / Limitations). Our proposal implements exactly this.
- **FlashRNN** (Pöppel et al., 2024): Demonstrated that fusing sequential recurrence steps with register-resident weights achieves 50× speedups. We apply the same I/O-aware principle to a different sequential dependency (intra-token Householder steps).
- **Gated DeltaNet** (Yang et al., ICLR 2025): Uses a single Householder step per token ($n_h = 1$) with scalar decay gate. The chunkwise WY kernel is optimized for this case. Our work generalizes the kernel to handle $n_h > 1$.
- **KDA / Kimi Linear** (Moonshot AI, 2025): Uses per-channel decay + single Householder (constrained DPLR). The transition matrix is diagonal + rank-1. DeltaProduct generalizes to diagonal + rank-$n_h$ via products.
- **Compact WY Autodiff** (trick 185): Efficient backpropagation through products of Householder reflections via the compact WY representation. Our fused kernel needs to support backward pass through the $n_h$-step product — the compact WY representation provides the gradient formulas.

No existing work fuses DeltaProduct's intra-token Householder steps into a register-persistent kernel. The DeltaProduct paper itself identifies this as future work.

## Mathematical Formulation

**Flattened approach (current DeltaProduct):**

The $n_h$ steps per token are interleaved into the sequence:

$$
\tilde{T} = n_h T, \quad \tilde{C} = n_h C
$$

UT transform operates on $\tilde{C} \times \tilde{C}$ Gram matrix:

$$
\tilde{\boldsymbol{M}} = (\boldsymbol{I} + \text{StrictTril}(\tilde{\boldsymbol{K}} \tilde{\boldsymbol{K}}^\top \odot \tilde{\boldsymbol{\beta}}))^{-1} \in \mathbb{R}^{n_h C \times n_h C}
$$

Cost: $O((n_h C)^2 d_k + (n_h C)^2 d_v)$ per chunk.

**Proposed fused approach:**

**Step 1: Precompute per-token transition and input (in registers/SRAM).**

For each token $i$ in a chunk, given $(\boldsymbol{k}_{i,j}, \boldsymbol{v}_{i,j}, \beta_{i,j})_{j=1}^{n_h}$:

$$
\boldsymbol{A}_i = \prod_{j=n_h}^{1} (\boldsymbol{I} - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top)
$$

Note: this product is accumulated **right-to-left** because the recurrence applies step 1 first. The product can be computed incrementally:

$$
\boldsymbol{A}_i^{(0)} = \boldsymbol{I}, \quad \boldsymbol{A}_i^{(j)} = (\boldsymbol{I} - \beta_{i,j} \boldsymbol{k}_{i,j} \boldsymbol{k}_{i,j}^\top) \boldsymbol{A}_i^{(j-1)}
$$

But we never materialize the full $d_k \times d_k$ matrix $\boldsymbol{A}_i$. Instead, we use the **compact WY representation**:

$$
\boldsymbol{A}_i = \boldsymbol{I} - \boldsymbol{W}_i \boldsymbol{Y}_i^\top
$$

where $\boldsymbol{W}_i, \boldsymbol{Y}_i \in \mathbb{R}^{d_k \times n_h}$ are the WY factors for the product of $n_h$ Householder reflections. These are computed via the standard UT transform on the $n_h \times n_h$ inner Gram matrix:

$$
\boldsymbol{T}_i = (\boldsymbol{I} + \text{StrictTril}(\boldsymbol{K}_i \boldsymbol{K}_i^\top \odot \boldsymbol{\beta}_i))^{-1} \cdot \text{Diag}(\boldsymbol{\beta}_i) \in \mathbb{R}^{n_h \times n_h}
$$

$$
\boldsymbol{Y}_i = \boldsymbol{K}_i \in \mathbb{R}^{d_k \times n_h}, \quad \boldsymbol{W}_i = \boldsymbol{K}_i \boldsymbol{T}_i \in \mathbb{R}^{d_k \times n_h}
$$

**Key**: This inner UT transform is on an $n_h \times n_h$ matrix (typically $2 \times 2$ or $3 \times 3$), which is trivially computed in registers. No shared memory or HBM needed.

**Step 2: Accumulated input matrix.**

$$
\boldsymbol{B}_i = \boldsymbol{K}_i \boldsymbol{T}_i \boldsymbol{V}_i^\top = \boldsymbol{W}_i \boldsymbol{V}_i^\top \in \mathbb{R}^{d_k \times d_v}
$$

where $\boldsymbol{V}_i \in \mathbb{R}^{d_v \times n_h}$ stacks the $n_h$ value vectors.

This is a $d_k \times n_h$ times $n_h \times d_v$ matmul — trivially small.

**Step 3: Token-level chunkwise algorithm.**

Now we have a standard linear recurrence over $T$ tokens:

$$
\boldsymbol{H}_i = (\boldsymbol{I} - \boldsymbol{W}_i \boldsymbol{Y}_i^\top) \cdot g_i \cdot \boldsymbol{H}_{i-1} + \boldsymbol{B}_i
$$

The transition is now: diagonal gate $g_i$ (from Gated DeltaProduct) times a rank-$n_h$ correction. This is exactly the same structure as Gated DeltaNet's single-step case but with rank-$n_h$ WY factors instead of rank-1.

The UT transform at the chunk level now operates on the $C \times C$ **token-level** Gram matrix (not the $(n_h C) \times (n_h C)$ flattened one):

$$
\boldsymbol{M} = (\boldsymbol{I} + \text{StrictTril}(\boldsymbol{\hat{K}} \boldsymbol{\hat{K}}^\top \odot \hat{\boldsymbol{\beta}}))^{-1} \in \mathbb{R}^{C \times C}
$$

where $\boldsymbol{\hat{K}} \in \mathbb{R}^{C \times d_k}$ and $\hat{\boldsymbol{\beta}} \in \mathbb{R}^C$ are the token-level effective keys and betas, derived from the WY factors.

**Wait — there's a subtlety.** The token-level transition $\boldsymbol{A}_i = \boldsymbol{I} - \boldsymbol{W}_i \boldsymbol{Y}_i^\top$ is rank-$n_h$, not rank-1. The standard UT transform assumes rank-1 Householder reflections. For the token-level chunkwise algorithm, we need to handle rank-$n_h$ transitions.

**Resolution:** We don't need to modify the UT transform. Instead, we use the **generalized WY representation** (trick 145 / Schreiber-Van Loan): the product of all $C \cdot n_h$ Householder reflections across the chunk can be represented as $\boldsymbol{I} - \boldsymbol{\mathcal{W}} \boldsymbol{\mathcal{Y}}^\top$ where $\boldsymbol{\mathcal{W}}, \boldsymbol{\mathcal{Y}} \in \mathbb{R}^{d_k \times C n_h}$. But we can also **merge** the $n_h$ inner factors per token before constructing the chunk-level WY representation.

**Practical approach (two-level WY):**

1. **Inner level** (register-fused, per token): Compute $(\boldsymbol{W}_i, \boldsymbol{Y}_i) \in \mathbb{R}^{d_k \times n_h}$ for each token's $n_h$ Householder steps. Cost: $O(n_h^2 d_k)$ per token (tiny for $n_h = 2, 3$).

2. **Outer level** (chunk-level, tensor-core matmuls): Treat each token's transition as $\boldsymbol{I} - \boldsymbol{W}_i \boldsymbol{Y}_i^\top$ and build the chunk-level product. Since $n_h$ is small (2–3), we can expand the rank-$n_h$ transition into $n_h$ consecutive rank-1 reflections at the chunk level — but now with the crucial difference that the inner UT transform is already computed. The effective chunk sequence has length $C$ (not $n_h C$), and each position contributes $n_h$ reflections whose WY factors are precomputed.

**Simplified variant (for $n_h = 2$):**

For $n_h = 2$, each token has two keys $(\boldsymbol{k}_{i,1}, \boldsymbol{k}_{i,2})$ and two betas $(\beta_{i,1}, \beta_{i,2})$. The product is:

$$
\boldsymbol{A}_i = (\boldsymbol{I} - \beta_{i,2} \boldsymbol{k}_{i,2} \boldsymbol{k}_{i,2}^\top)(\boldsymbol{I} - \beta_{i,1} \boldsymbol{k}_{i,1} \boldsymbol{k}_{i,1}^\top) = \boldsymbol{I} - \boldsymbol{W}_i [\boldsymbol{k}_{i,1}, \boldsymbol{k}_{i,2}]^\top
$$

where $\boldsymbol{W}_i \in \mathbb{R}^{d_k \times 2}$ is computed from the $2 \times 2$ inner UT transform. The output and state update for the chunk can then use a "blocked" UT transform that processes pairs of reflections.

**Key Variables:**

- $n_h$ — Householder steps per token (2–4, the expressivity knob)
- $\boldsymbol{K}_i \in \mathbb{R}^{d_k \times n_h}$ — stacked keys for token $i$
- $\boldsymbol{V}_i \in \mathbb{R}^{d_v \times n_h}$ — stacked values for token $i$
- $\boldsymbol{\beta}_i \in \mathbb{R}^{n_h}$ — learning rates for token $i$
- $\boldsymbol{W}_i, \boldsymbol{Y}_i \in \mathbb{R}^{d_k \times n_h}$ — compact WY factors for token $i$'s product
- $\boldsymbol{T}_i \in \mathbb{R}^{n_h \times n_h}$ — inner UT transform (tiny, computed in registers)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Gated DeltaProduct (unchanged architecture) |
| Modification | Fused intra-token kernel replacing sequence flattening |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d_{\text{model}} = 768$–$2048$ |
| Heads | $H = 16$ (parameter-matched to DeltaNet 8 heads) |
| Head dim | $d_k = d_v = 128$ |
| $n_h$ | 2 (primary), 3 (secondary) |
| Chunk size | $C = 64$ |

### Baseline

1. **DeltaProduct (flattened)**: Current implementation that inflates sequence by $n_h \times$ and reuses DeltaNet's kernel. For $n_h = 2$: UT transform on $(2 \times 64) \times (2 \times 64) = 128 \times 128$ matrix. Throughput: ~35K tok/sec (16 heads, $d_h = 128$).
2. **DeltaNet ($n_h = 1$)**: Single Householder step. UT transform on $64 \times 64$. Throughput: ~40K tok/sec (8 heads, $d_h = 256$).
3. **Gated DeltaNet**: DeltaNet + scalar decay gate. Similar throughput to DeltaNet.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput ($n_h = 2$) | $\geq 45$K tok/sec | H100, 1.3B model, bf16 |
| Training throughput ($n_h = 3$) | $\geq 35$K tok/sec | H100, 1.3B model, bf16 |
| Model quality | Identical to flattened | Perplexity on WikiText-103 |
| UT transform time per chunk | $\leq 0.35\times$ flattened | Wall-clock μs on H100 |
| HBM traffic per chunk | $\leq 0.6\times$ flattened | Nsight Compute L2 sectors |

### Estimated Compute

**MVE (kernel microbenchmark)**: < 15 minutes on single H100
**Full kernel development + profiling**: ~20 GPU-hours
**End-to-end 1.3B training comparison**: ~200 GPU-hours on H100
**Total**: ~220 GPU-hours (medium-large scale)

## Expected Outcome

**If hypothesis is correct:**

- DeltaProduct$_2$ throughput increases from ~35K to ~45–50K tok/sec ($1.3$–$1.4\times$), because the UT transform cost drops from $O(128^2 \times 128) \approx 2M$ FLOPs to $O(64^2 \times 128 + 2 \times 64 \times 128) \approx 0.5M$ FLOPs per chunk — a $4\times$ reduction in the UT transform, which is ~30% of chunk time.
- DeltaProduct$_3$ throughput increases from ~18K to ~35–40K tok/sec ($2.0$–$2.2\times$), because the UT inflation is cubic in $n_h$ for the flattened case ($n_h^2$ factor in the transform).
- Model quality is **identical** — this is a pure kernel optimization with no algorithmic change to the recurrence.
- DeltaProduct$_2$ becomes faster than DeltaNet at parameter-matched configurations, because the throughput penalty of multi-step Householder is eliminated while the quality gains remain.

**If hypothesis is wrong:**

- **Scenario A: The inner UT transform + WY merge adds more overhead than it saves.** If the $n_h \times n_h$ inner UT transforms (even though tiny: $2 \times 2$ or $3 \times 3$) have non-negligible overhead due to register pressure or control flow, the savings from smaller chunk-level UT may not compensate. **Learn**: Profile the inner transform cost in isolation.
- **Scenario B: Rank-$n_h$ token transitions don't compose efficiently in the chunk-level UT transform.** If expanding rank-$n_h$ transitions into the chunk-level WY representation requires more computation than the flattened approach's uniform rank-1 stream, the overall cost may increase. **Learn**: Compare the number of FLOPs for both approaches at various $n_h$ and $C$.
- **Scenario C: Memory layout change hurts cache efficiency.** The fused approach reads $n_h$ keys/values per token from a different memory layout than the flattened sequence. If this layout is less cache-friendly, HBM traffic may not decrease as expected. **Learn**: Profile L2 cache hit rates.

## Minimum Viable Experiment

### Setup

- **Kernel**: Triton kernel implementing the two-level WY approach for DeltaProduct$_2$
- **Configuration**: $C = 64$, $d_k = d_v = 64$, $n_h = 2$, $H = 4$ heads, batch = 4
- **Comparison**: DeltaProduct flattened kernel (from fla-org) with equivalent configuration
- **Hardware**: Single H100 GPU
- **Compute**: < 15 minutes

### Implementation Sketch

```python
@triton.jit
def fused_deltaproduct_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, beta_ptr, gate_ptr,
    O_ptr, S_ptr,
    T, C: tl.constexpr, dk: tl.constexpr, dv: tl.constexpr,
    n_h: tl.constexpr,
):
    """
    Fused DeltaProduct chunkwise kernel.
    Key difference from DeltaNet: processes n_h keys/values per token
    in a register-resident inner loop, then builds the C×C chunk-level
    UT transform (not the (n_h*C)×(n_h*C) flattened one).
    """
    chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # For each token in the chunk:
    for t in range(C):
        # Load n_h keys, values, betas for this token (small: n_h × dk)
        # These stay in registers for the inner loop
        k_nh = load_nh_keys(K_ptr, chunk_id, t, head_id, n_h, dk)  # [n_h, dk]
        v_nh = load_nh_values(V_ptr, chunk_id, t, head_id, n_h, dv)  # [n_h, dv]
        beta_nh = load_nh_betas(beta_ptr, chunk_id, t, head_id, n_h)  # [n_h]

        # Inner UT transform: n_h × n_h (tiny, in registers)
        # Gram: G[j,l] = beta[j] * dot(k[j], k[l]) for j > l
        G_inner = tl.zeros([n_h, n_h])
        for j in range(n_h):
            for l in range(j):
                G_inner[j, l] = beta_nh[j] * tl.sum(k_nh[j] * k_nh[l])

        # Solve T_inner = (I + StrictTril(G))^{-1} @ diag(beta)
        # For n_h = 2: T_inner is 2x2, closed-form
        T_inner = solve_small_triangular(G_inner, beta_nh, n_h)

        # Compute WY factors: W_t = K_nh @ T_inner, Y_t = K_nh
        W_t = matmul_small(k_nh, T_inner, n_h, dk)  # [dk, n_h] in registers

        # Compute effective input: B_t = W_t @ V_nh^T
        B_t = matmul_small_outer(W_t, v_nh, n_h, dk, dv)  # [dk, dv]

        # Store W_t, Y_t (=K_nh) for chunk-level UT transform
        store_wy_factors(W_t, k_nh, chunk_id, t, head_id)
        store_input(B_t, chunk_id, t, head_id)

    # Now run the standard C×C chunk-level algorithm
    # using the precomputed rank-n_h WY factors per token
    # (This is the same as Gated DeltaNet but with rank-n_h transitions)
    run_chunk_level_ut_and_output(...)
```

### Success Criteria

- Fused kernel is $\geq 1.3\times$ faster than the flattened baseline for $n_h = 2$
- Produces bit-identical outputs (or within bf16 tolerance, $< 10^{-3}$ relative error)
- UT transform time decreases by $\geq 2\times$ (from $128 \times 128$ to $64 \times 64$ + small inner transforms)
- HBM traffic decreases by $\geq 30\%$ (no dummy outputs, no inflated key/value reads)

### Failure Criteria

- If the fused kernel is slower than flattened: the inner WY computation overhead dominates. Abandon the two-level approach.
- If outputs differ significantly: the mathematical reformulation has a bug (the two approaches should be algebraically identical).

### Why This Test Is Sufficient

- The kernel microbenchmark directly tests the core hypothesis: can we avoid the $n_h \times$ sequence inflation overhead? If the fused kernel shows speedup at the per-chunk level, the end-to-end model will proportionally benefit.
- The correctness test (bit-identical outputs) validates the mathematical reformulation — the two-level WY approach must produce exactly the same recurrence as the flattened approach.
- $n_h = 2$ is the most practical configuration (best expressivity/throughput tradeoff), so if the kernel works for $n_h = 2$, it covers the primary use case.

## Memory Access Pattern Analysis

**Flattened DeltaProduct (current):**
- Reads $n_h T$ keys, values, betas from HBM (inflated by $n_h \times$)
- UT transform involves $(n_h C)^2 / 2$ elements in the Gram matrix (inflated by $n_h^2 \times$)
- Computes $n_h T$ outputs, discards $n_h - 1$ per token (wasted compute + HBM writes)
- **Arithmetic intensity**: Same as DeltaNet per element, but more elements → lower effective utilization

**Fused DeltaProduct (proposed):**
- Reads $T \times n_h$ keys, values, betas (same data, but loaded into registers per-token, not materialized in sequence)
- Inner WY transform: $n_h \times n_h$ per token, entirely in registers (0 HBM)
- Chunk-level UT transform: $C \times C$ (not $(n_h C) \times (n_h C)$)
- Computes $T$ outputs (no waste)
- **Arithmetic intensity**: Higher because the same compute is done with less data movement

**Coalescing:**
- Keys/values for $n_h$ steps per token can be stored contiguously: $[k_{i,1}, k_{i,2}, \ldots, k_{i,n_h}]$ per token. Loading $n_h \times d_k$ contiguous elements per token is coalesced.
- The WY factors $W_t, Y_t$ (size $d_k \times n_h$) are written per token — small and contiguous.

## Parallelism Analysis

- **SM utilization**: Same as DeltaNet — each chunk × head pair is an independent program. $H \times (T/C) \times B$ programs total.
- **Warp divergence**: None within the inner $n_h$ loop (all warps execute the same $n_h$ steps).
- **Tensor core mapping**: The chunk-level UT transform ($C \times C$) and inter/intra-chunk matmuls use tensor cores as before. The inner $n_h \times n_h$ transforms are too small for tensor cores but are trivially computed in scalar ALU (only $4$ or $9$ FLOPs for $n_h = 2$ or $3$).
- **Register pressure**: The inner loop stores $n_h$ keys ($n_h \times d_k$) and WY factors ($d_k \times n_h$) in registers. For $n_h = 2, d_k = 64$: 256 bf16 values = 512 bytes per thread. H100 has 256 KB registers per SM — easily fits.

## Theoretical Analysis

| Operation | Flattened ($n_h \times$ seq) | Fused (proposed) |
|-----------|------------------------------|------------------|
| Sequence length | $n_h T$ | $T$ |
| UT transform per chunk | $O((n_h C)^2 d_k)$ | $O(C^2 d_k + C n_h^2 d_k)$ |
| KV reads from HBM | $O(n_h T (d_k + d_v))$ | $O(T n_h (d_k + d_v))$ (same data, but fused) |
| Outputs computed | $n_h T$ (discard $(n_h-1)T$) | $T$ |
| Inner WY transform | N/A (in flattened UT) | $O(C n_h^2 d_k)$ (registers) |
| Total FLOPs per chunk | $O(n_h^2 C^2 d + n_h C d^2)$ | $O(C^2 d + n_h C d + C n_h^2 d)$ |

**Speedup estimate for UT transform** ($n_h = 2, C = 64, d_k = 128$):

- Flattened: $(128)^2 \times 128 / 2 \approx 1.05M$ FLOPs
- Fused: $(64)^2 \times 128 / 2 + 64 \times 4 \times 128 \approx 0.26M + 0.03M \approx 0.29M$ FLOPs
- **UT speedup: $3.6\times$**

Since the UT transform is roughly 25–35% of per-chunk compute, this gives an overall per-chunk speedup of $\sim 1.3$–$1.5\times$.

## Risks & Limitations

1. **Rank-$n_h$ chunk-level recurrence is more complex than rank-1.** The standard DeltaNet kernel assumes rank-1 Householder transitions at each sequence position. With rank-$n_h$ transitions, the chunk-level Gram matrix and UT transform need modification. The simplest approach is to still "expand" the rank-$n_h$ transitions into $n_h$ rank-1 reflections at the chunk level, but with precomputed WY factors — this may partially re-introduce the flattened overhead.

2. **Backward pass complexity.** The fused forward pass computes compact WY factors per token. The backward pass needs to differentiate through these, requiring the chain rule through the inner UT transform. This is well-studied (trick 185, compact WY autodiff) but adds implementation complexity.

3. **Only benefits $n_h > 1$.** For $n_h = 1$ (standard DeltaNet), there's no inner loop to fuse. The optimization is only relevant for DeltaProduct with $n_h \geq 2$.

4. **Kernel implementation effort.** Writing a custom Triton kernel that handles the two-level WY factorization is significantly more complex than DeltaNet's single-level kernel. The MVE tests whether the approach is sound before committing to full kernel development.

5. **Small $n_h$ limits gains.** For $n_h = 2$, the UT transform inflation factor is only $4\times$ ($(2C)^2$ vs $C^2$). For $n_h = 3$, it's $9\times$. The larger $n_h$, the bigger the savings — but larger $n_h$ is also less commonly used in practice.

## Follow-up Experiments

1. **Combine with proposal 057 (FlashRNN inter-chunk scan)**: The fused intra-token kernel (this proposal) and the FlashRNN inter-chunk scan (proposal 057) optimize different parts of the chunkwise algorithm. Combining both could give a larger end-to-end speedup.

2. **Fused projection + inner WY**: The $n_h$ keys and values are projected from the input $\boldsymbol{x}_i$ via linear layers. Fusing the projection + inner WY computation into a single kernel would eliminate one more HBM round-trip (the materialization of the $n_h$ projected keys/values).

3. **Gated DeltaProduct with KDA-style per-channel decay**: Combine the fused DeltaProduct kernel with KDA's per-channel decay (instead of scalar gate). The per-channel decay would apply to the rank-$n_h$ transition as in KDA, and the inner WY factors would incorporate the decay.

4. **Autoregressive generation optimization**: During generation, the $n_h$ Householder steps per token are sequential but can be parallelized if the steps are independent (they operate on the same initial state). Explore whether the $n_h$ matvecs can be batched.

5. **Dynamic $n_h$**: Different tokens may benefit from different numbers of Householder steps. Use the fused kernel with a fixed maximum $n_h$ but allow early termination based on the effective $\beta$ (skip steps where $\beta_{i,j} \approx 0$).

## Human Review

(To be filled by reviewer)

## References

- Siems, J. et al. (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. NeurIPS 2025. arXiv:2502.10297.
- Pöppel, N., Beck, M. & Hochreiter, S. (2024). FlashRNN: I/O-Aware Optimization of Traditional RNNs on Modern Hardware. arXiv:2412.07752.
- Yang, S. et al. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.
- Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. SIAM J. Sci. Stat. Comput.
- Papanicolopulos, S.-A. (2022). Analytic Derivatives of the Compact WY QR Decomposition. arXiv:2209.15013.

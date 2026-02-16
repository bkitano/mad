---
status: ongoing
priority: high
created: 2026-02-15
based_on: newton-schulz-polar-orthogonalization (164), neumann-series-cwy-inverse-approximation (157), householder-product-parameterization (062), wy-representation (145), recursive-wy-merge-blocked-qr (159), chunkwise-parallel-scan (026), tfla-two-level-tiled-chunkwise-parallelism (158)
experiment_number: 043
experiment_log: experiment-log-043.md
---

# Newton-Schulz Orthogonal State Projection for DeltaNet/DeltaProduct

## Hypothesis

Replacing the sequential UT transform forward substitution in DeltaNet's chunkwise training with a **Newton-Schulz polar orthogonalization** of the accumulated per-chunk state transition will achieve **$\geq 1.3\times$ wall-clock speedup** for the per-chunk preprocessing step while maintaining equivalent model quality, by converting the only sequential bottleneck (the $O(C^2)$ triangular solve) into **pure tensor-core GEMMs** in bfloat16. The Newton-Schulz iteration's doubly-exponential convergence means only $q = 2$–$3$ iterations (6–9 matmuls of size $C \times C$) suffice for the $C \times C$ intra-chunk factor, compared to $C$ sequential steps of the UT forward substitution.

## Background

### The UT transform bottleneck in DeltaNet

DeltaNet (Yang et al., 2024) parameterizes state transitions as generalized Householder reflections $A_t = I - \beta_t k_t k_t^\top$. Within each chunk of $C$ tokens, the product of these reflections is accumulated via the **UT transform** (trick 145), which converts the sequential Householder product into a compact WY form $Q = I - Y T Y^\top$.

The UT transform requires computing the upper-triangular factor $T$ via:

$$
T^{-1} = I + L, \quad L_{ij} = -\beta_i (k_i^\top k_j) \text{ for } i > j
$$

Computing $T$ from $T^{-1} = I + L$ requires **forward substitution** — a sequential $O(C^2)$ operation where column $j$ depends on all previous columns. This is Level-2 BLAS (DTRSV), which runs at ~10% of tensor core peak on modern GPUs.

The Neumann series approximation (trick 157) replaces this with $(I + L)^{-1} \approx I - L + L^2 - \ldots$, but this requires $p$ sequential matmuls for $p$ terms, and the approximation error grows with $C$ (the truncation is exact only for $C = 1$).

### Why Newton-Schulz is better

Newton-Schulz polar orthogonalization (trick 164) provides a fundamentally different approach: instead of computing $T$ exactly and then forming $Q = I - YTY^\top$, we can:

1. Compute the "raw" accumulated transition $\tilde{A}_C = \prod_{t=1}^{C} (I - \beta_t k_t k_t^\top)$ via a simple left-multiplication chain
2. Apply $q$ steps of Newton-Schulz to **orthogonalize** $\tilde{A}_C$ into $\text{Polar}(\tilde{A}_C) = UV^\top$

The key advantages:

- **Pure GEMMs**: Each Newton-Schulz step is $A = XX^\top$ (symmetric GEMM), $B = bA + cA^2$ (GEMM + scale), $X \leftarrow aX + BX$ (GEMM + scale). All tensor-core-friendly.
- **bfloat16 safe**: Newton-Schulz is numerically stable in bf16 because the polynomial has only positive coefficients and the iteration is self-correcting (trick 164).
- **Doubly-exponential convergence**: With the quintic polynomial ($\kappa = 2$), orthogonality residual decays as $\delta \mapsto \delta^3$ per step. For $\delta_0 = 0.3$, after 2 steps: $\delta = 0.3^9 \approx 2 \times 10^{-5}$. For $\delta_0 = 0.5$, after 3 steps: $\delta = 0.5^{27} \approx 7 \times 10^{-9}$.
- **No sequential dependency**: All operations within a Newton-Schulz step are independent matmuls; steps are sequential but there are only $q = 2$–$3$ of them.

### The key insight: orthogonality is what matters

DeltaNet's Householder transitions $I - \beta_t k_t k_t^\top$ with $\beta_t \in [0, 2]$ produce near-orthogonal matrices. The product of $C$ such matrices is also near-orthogonal (since each factor has $\|I - \beta k k^\top\|_2 \leq 1$ for $\beta \in [0, 2]$). The UT transform computes this product **exactly**. But for pretraining, we don't need the exact product — we need a **good approximation** that preserves the orthogonal structure and gradient flow.

Newton-Schulz projects the accumulated product onto the nearest orthogonal matrix, which:
- Preserves the stability guarantee ($\|UV^\top\|_2 = 1$)
- Maintains gradient flow (orthogonal matrices have unit singular values)
- Introduces only a tiny projection error that vanishes with more NS steps

This trades **exactness** for **GPU efficiency** — a favorable trade for pretraining where stochastic gradient noise already dominates.

### What's different from existing proposals

- **Proposal 002** (SSD-DeltaNet WY Hybrid): Uses standard WY/UT transform for DeltaNet parallelization. Our approach replaces the WY/UT with Newton-Schulz.
- **Proposal 020** (OH-DeltaProduct): Combines oscillatory + Householder components. Still uses UT transform for the Householder part. Could benefit from our Newton-Schulz replacement.
- **Trick 157** (Neumann CWY Inverse): Approximates $(I+L)^{-1}$ via truncated Neumann series. Requires sequential matmul chain and has growing error with $C$. Newton-Schulz converges faster and is more GPU-friendly.

## Related Work

- **Muon optimizer (Jordan et al., 2024)**: Uses Newton-Schulz for orthogonalizing the **optimizer momentum** (gradient space), not the forward-pass state transition. Different application: optimizer step vs. recurrence computation.
- **UNSO (arXiv:2602.02500, Feb 2025)**: Unifies Newton-Schulz variants for optimization. Optimizer-focused; no application to recurrent forward passes.
- **CANS (arXiv:2506.10935, Jun 2025)**: Chebyshev-optimal Newton-Schulz polynomials. Improves NS convergence for optimization. Could be adopted in our approach for even faster convergence.
- **DeltaProduct (Keller et al., 2025)**: Products of Householder reflections for state-tracking. Uses standard UT transform. Our approach could accelerate DeltaProduct's per-chunk computation.
- **HOFT (Moreno Arcas et al., 2025)**: Uses Neumann approximation to CWY inverse for fine-tuning. Related approach but for PEFT, not recurrent state transitions.

**Gap**: No existing work applies Newton-Schulz polar orthogonalization to the **forward-pass state transition computation** in linear RNNs. All existing NS applications are in optimizer space (Muon) or weight reparameterization (HOFT). Using NS for the recurrence's accumulated transition matrix is novel.

## Mathematical Formulation

**Standard DeltaNet UT Transform (current approach):**

Within a chunk of $C$ tokens, the accumulated transition is:

$$
Q = \prod_{t=1}^{C} (I - \beta_t k_t k_t^\top) = I - YTY^\top
$$

where $Y = [k_1, \ldots, k_C] \in \mathbb{R}^{d \times C}$ and $T \in \mathbb{R}^{C \times C}$ is upper-triangular, computed via forward substitution of $(I + L)$:

$$
T_{ij} = \begin{cases} \beta_i & \text{if } i = j \\ -\sum_{l=i}^{j-1} T_{il} \beta_l (k_l^\top k_j) \beta_j & \text{if } i < j \end{cases}
$$

**Cost**: $O(C^2 d)$ for $K^\top K$ matmul (tensor core) + $O(C^2)$ sequential forward substitution (NOT tensor core).

**Proposed Newton-Schulz Approach:**

**Step 1**: Compute the raw transition product via sequential left-multiplication (within shared memory, fused):

$$
X_0 = \frac{1}{C} \sum_{t=1}^{C} (I - \beta_t k_t k_t^\top) \cdot e_t e_t^\top \quad \text{(conceptually)}
$$

More precisely, we compute $G = K^\top K \in \mathbb{R}^{C \times C}$ (one GEMM) and form the lower-triangular gram matrix $L_{ij} = -\beta_i (k_i^\top k_j)$ for $i > j$. Then:

$$
X_0 = (I + L) / \|I + L\|_F
$$

**Step 2**: Apply $q$ Newton-Schulz iterations (quintic, Muon-style):

$$
X_{j+1} = a X_j + (b X_j X_j^\top + c (X_j X_j^\top)^2) X_j
$$

with $(a, b, c) = (3.4445, -4.7750, 2.0315)$.

**Step 3**: Use $X_q \approx \text{Polar}(I + L) = UV^\top$ as the accumulated chunk transition.

**Cost**: $O(C^2 d)$ for $K^\top K$ + $q \times O(C^3)$ for NS iterations. Since $C \leq 64$ typically and $d = 64$–$256$, the $C^3$ NS cost is $O(64^3) = O(262K)$ FLOPs per iteration — negligible relative to the $O(C^2 d) = O(64^2 \times 256) \approx O(1M)$ GEMM.

**Alternative: Direct chunk-boundary propagation**

For inter-chunk state propagation, we need the accumulated transition across all $C$ tokens in a chunk. Instead of building the full $Q$ product, we can:

1. Compute the chunk's $C$-step product $\tilde{A} = \prod_{t} A_t$ lazily during the intra-chunk scan
2. At chunk boundaries, orthogonalize $\tilde{A}$ via Newton-Schulz before propagating to the next chunk

This amortizes the NS cost over $T/C$ chunk boundaries rather than every chunk.

**Key Variables:**
- $k_t \in \mathbb{R}^d$ — normalized key vector at time $t$
- $\beta_t \in [0, 2]$ — learning rate / reflection coefficient
- $C$ — chunk size (typically 64)
- $d$ — head dimension (typically 64–256)
- $q$ — number of Newton-Schulz iterations (2–3)
- $X_j \in \mathbb{R}^{C \times C}$ — Newton-Schulz iterate

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | DeltaNet / Gated DeltaNet |
| Layers | $L = 12$ |
| Hidden dim | $d_{\text{model}} = 768$ |
| Head dim | $d = 64$ |
| Heads | 12 |
| Chunk size | $C = 64$ |
| Newton-Schulz steps | $q = 2$ or $3$ |
| NS polynomial | Muon quintic $(a, b, c) = (3.4445, -4.7750, 2.0315)$ |

### Baseline
1. **DeltaNet with UT transform** (standard): Uses forward substitution for $T$ computation. Baseline includes the optimized Triton/CUDA kernel from Yang et al. (2024). The UT forward substitution is $O(C^2)$ sequential.
2. **DeltaNet with Neumann-2 approximation** (trick 157): $(I+L)^{-1} \approx I - L$. Same GEMM structure as our approach but lower accuracy.
3. **Mamba-2** (SSD): Diagonal SSM baseline for throughput comparison.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chunk preprocessing time | $\leq 0.7\times$ UT baseline | Wall-clock μs per chunk on H100 |
| End-to-end throughput | $\geq 1.15\times$ UT baseline | Tokens/sec on H100 |
| Orthogonality error | $\|I - X_q X_q^\top\|_F < 10^{-3}$ | Per-chunk max error |
| Perplexity (WikiText-103) | $\leq 1.01\times$ UT baseline | Validation perplexity ratio |
| $S_5$ state tracking | $\geq 95\%$ accuracy | 5-element permutation composition |
| Memory | Same as baseline | Peak GPU MB |

### Estimated Compute

**MVE**: < 10 minutes, single A100/H100
**Microbenchmark (chunk kernel)**: ~2 GPU-hours
**Full training (350M params, WikiText-103)**: ~100 GPU-hours on A100
**Total**: ~102 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- Newton-Schulz with $q = 2$ achieves orthogonality error $< 10^{-4}$ for typical DeltaNet key distributions (keys are normalized, so $\|L\|_F$ is bounded)
- Per-chunk preprocessing speedup of $1.3$–$1.5\times$ by eliminating the sequential forward substitution
- End-to-end training throughput improvement of $1.1$–$1.2\times$ (the UT preprocessing is ~15–25% of per-chunk time)
- Perplexity within 0.5% of exact UT transform (the projection error is dominated by SGD noise)
- On $S_5$ state tracking, accuracy within 1% of exact UT (orthogonal projection preserves group structure)

**If hypothesis is wrong:**
- If orthogonality error is too high ($> 10^{-2}$): The accumulated Householder product is too far from orthogonal for Newton-Schulz to converge in 2–3 steps. Would need $q = 4$–$5$ steps, reducing the speedup advantage. Investigate: what is the typical $\delta_0 = \|I - \tilde{A}\tilde{A}^\top\|_{\text{op}}$ for real DeltaNet checkpoints?
- If perplexity degrades significantly ($> 2\%$): The projection onto the nearest orthogonal matrix discards information that the exact Householder product encodes (e.g., the $\beta < 1$ contractions). Would need to preserve the contraction structure, perhaps via $X_q \cdot \text{diag}(\sigma_1, \ldots, \sigma_C)$ where $\sigma_i = \sigma_i(\tilde{A})$.
- If speedup is negligible: The UT forward substitution is already small relative to the $O(C^2 d)$ matmul, and the NS iterations' $O(C^3)$ cost doesn't compensate. This would indicate the bottleneck is elsewhere (HBM, not compute). The result would still be scientifically interesting: it would quantify the exact breakdown of DeltaNet's per-chunk computation.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer DeltaNet, $d = 32$, 2 heads, chunk size $C = 32$, ~80K params
- **Task**: $S_3$ permutation composition — compose sequences of 3-element permutations, predict resulting permutation
- **Data**: 5K sequences of length 128
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- Newton-Schulz ($q = 2$) achieves $> 90\%$ $S_3$ accuracy (matching exact UT transform within 3%)
- Orthogonality error $\|I - X_2 X_2^\top\|_F < 10^{-3}$ across all chunks during training
- Per-chunk kernel time (microbenchmark): NS variant $\leq 0.8\times$ UT variant on same GPU
- Training converges stably (no NaN/Inf) for 1000 steps

### Failure Criteria
- If $S_3$ accuracy drops below 70%: The Newton-Schulz projection destroys the Householder product structure needed for state tracking. The mechanism is broken for this application.
- If orthogonality error $> 10^{-1}$: The accumulated Householder product is far from orthogonal, and Newton-Schulz doesn't converge in 2 steps. Need to investigate why (likely $\beta$ values creating strong contractions).
- If NS kernel is slower than UT: The $C^3$ NS matmuls are not efficiently mapped to tensor cores at this small size ($C = 32$). Would need larger $C$ or batching across heads.

### Why This Test Is Sufficient
- $S_3$ directly tests the core capability (non-abelian state tracking). If the Newton-Schulz approximation preserves this at small scale, it will preserve it at large scale since the per-chunk computation is identical.
- The orthogonality error is a per-chunk quantity independent of model scale — if it's small at $C = 32$, it's small at $C = 64$.
- The microbenchmark directly measures the kernel speedup, which is the primary hypothesis.

## Memory Access Pattern Analysis

**UT forward substitution (current):**
- Access pattern: **Sequential column-by-column** — column $j$ requires reading all previous columns. Non-vectorizable, poor instruction-level parallelism.
- Arithmetic intensity: Low ($O(C)$ FLOPs per $O(C)$ memory access per column).
- Cache behavior: Good spatial locality (columns are contiguous) but sequential dependency prevents pipelining.

**Newton-Schulz (proposed):**
- Access pattern: **Batched GEMM** — $X_j X_j^\top$ is a symmetric rank-$C$ update, followed by GEMM. Fully coalesced, fully vectorized.
- Arithmetic intensity: High ($O(C^3)$ FLOPs for $O(C^2)$ data — arithmetic intensity $\sim C$).
- Cache behavior: $C \times C$ matrices fit entirely in shared memory for $C \leq 64$ ($64^2 \times 2 = 8$ KB in bf16). All NS iterations can stay in SRAM.

**Tensor core utilization:**
- UT forward substitution: **0%** tensor core utilization (scalar sequential ops)
- Newton-Schulz: **~80%** tensor core utilization (all matmuls, $C = 64$ matches MMA tile sizes well)

## Parallelism Analysis

- **Warp divergence**: None — all threads execute the same matmul instructions
- **Load imbalance**: None — batched across heads (12 heads = 12 independent NS computations)
- **Tensor core mapping**: $C \times C$ matmuls map directly to `mma.sync.m16n16k16` tiles for $C = 64$ (4 tiles per dimension)
- **Sequential bottleneck**: $q = 2$–$3$ NS iterations are sequential, but each is a single matmul — total serial depth is $O(q)$ matmuls vs $O(C)$ scalar ops for UT

## Theoretical Analysis

Complexity comparison (per chunk, per head):

| Operation | UT Transform | Newton-Schulz ($q$ steps) |
|-----------|-------------|--------------------------|
| Gram matrix $K^\top K$ | $O(C^2 d)$ GEMM ✓ | $O(C^2 d)$ GEMM ✓ |
| $T$ factor / Orthogonalization | $O(C^2)$ sequential ✗ | $q \times O(C^3)$ GEMM ✓ |
| Apply WY / Use result | $O(C^2 d)$ GEMM ✓ | $O(Cd)$ (direct use) ✓ |
| **BLAS level** | Mix of L2 + L3 | **All Level-3** |
| **Tensor core %** | ~80% (UT part: 0%) | **~100%** |

Crossover point: Newton-Schulz is faster when the tensor core throughput advantage ($\sim 16\times$ over scalar) outweighs the $q \times C$ additional FLOPs. For $C = 64$ and $q = 2$: NS does $2 \times 5 = 10$ matmuls of size $64^2$, totaling $\sim 5M$ FLOPs. UT forward substitution does $\sim 4K$ serial FLOPs but at $\sim 1/16\times$ throughput, equivalent to $\sim 64K$ "matmul-equivalent" FLOPs. **NS wins by a factor of $\sim 12\times$ in wall-clock for the $T$-computation step.**

However, the $T$-computation is only ~10–20% of the total per-chunk cost (dominated by the $O(C^2 d)$ matmuls). End-to-end speedup is therefore ~$1.1$–$1.2\times$.

## Risks & Limitations

1. **Contraction information loss**: DeltaNet with $\beta < 1$ produces contractive (not orthogonal) transitions. Newton-Schulz projects these onto the nearest orthogonal matrix, discarding the contraction. This may hurt quality for tasks requiring selective forgetting. **Mitigation**: Track the singular value distribution of accumulated transitions; if contractive, use a scaled version $X_q \cdot \sigma_{\min}(\tilde{A})$.

2. **Small-$C$ overhead**: For $C = 32$ or smaller, the $C \times C$ matmuls may be too small to saturate tensor cores efficiently. The 16×16 MMA tile means $C = 32$ gives only $2 \times 2$ tiles per matmul dimension. **Mitigation**: Batch multiple heads' NS computations into a single batched GEMM.

3. **Backward pass complexity**: The gradient through Newton-Schulz iterations requires differentiating through $q$ matmul chains. This is straightforward (chain rule through GEMMs) but adds $q$ backward GEMMs per forward NS step. Total backward cost: $\sim 3q$ additional $C \times C$ GEMMs per chunk.

4. **Not exact**: Unlike the UT transform which computes the exact Householder product, Newton-Schulz gives an approximation. For applications requiring exact orthogonal state tracking, this may not suffice. **Mitigation**: Use $q = 3$ for higher precision or fall back to UT for the final layer.

5. **Initial scaling sensitivity**: Newton-Schulz requires $\|X_0\|_{\text{op}} \leq 1$. The accumulated Householder product naturally satisfies this (each factor has $\|A_t\|_2 \leq 1$), but Frobenius-norm pre-scaling may slow convergence if $\sigma_{\max} \ll \sigma_{\min}$ (ill-conditioned).

## Follow-up Experiments

1. **Chebyshev-optimal NS (CANS)**: Replace the Muon quintic with the Chebyshev-optimal polynomial from arXiv:2506.10935 for faster convergence at the same iteration count. Expected: $q = 2$ with CANS matches $q = 3$ with Muon quintic.

2. **NS for DeltaProduct**: DeltaProduct applies $n_h > 1$ Householder reflections per step. The accumulated chunk product is a product of $C \times n_h$ reflections — more orthogonal than single-reflection DeltaNet. NS should converge even faster here ($\delta_0$ closer to 0).

3. **Gated NS-DeltaNet**: Combine with Gated DeltaNet's input-dependent gating. The gate introduces contractions that Newton-Schulz would project away. Test whether preserving the contraction via $X_q \cdot \text{diag}(\sigma_i)$ helps.

4. **Mixed-precision NS**: Use bfloat16 for the NS iterations but fp32 for the Gram matrix $K^\top K$. Test whether this improves the initial $\delta_0$ estimate.

5. **TFLA + NS**: Integrate Newton-Schulz into TFLA's two-level tiling (proposal 038). The inner-tile UT transform is the primary target for NS replacement, as inner tiles may have smaller $C_{\text{inner}}$ where the sequential overhead is proportionally larger.

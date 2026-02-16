---
status: ongoing
priority: high
created: 2026-02-15
based_on: fmmformer-near-far-field-attention, chunkwise-parallel-scan, linear-attention-approximation, cosine-reweighted-linear-attention, input-dependent-gating, io-aware-tiling, warp-specialized-pipelining, rfa-gated-random-feature-attention
experiment_number: 036
experiment_log: experiment-log-036.md
---

# Near-Far Field Decomposition for Chunkwise GLA Intra-Chunk Computation

## Hypothesis

Replacing the full $C \times C$ quadratic intra-chunk attention in chunkwise GLA/Mamba-2 with an **FMMformer-style near-field (banded, width $k$) + far-field (rank-$r$ linear attention) decomposition** will enable **2–4× larger chunk sizes** ($C = 256$–$512$ vs. $C = 64$) at equal or lower per-chunk compute cost, yielding $1.2$–$1.5\times$ end-to-end training throughput and improved quality (lower perplexity) because larger chunks reduce the number of inter-chunk scan steps and capture longer-range dependencies within the fast parallel phase, while the near-far decomposition keeps intra-chunk cost at $O(C(k + r)d)$ instead of $O(C^2 d)$.

## Background

### The chunk size dilemma in chunkwise SSMs

In chunkwise parallel SSMs (GLA, Mamba-2 SSD, Gated DeltaNet), the sequence of length $T$ is divided into chunks of size $C$. Within each chunk, a **quadratic** attention-like operation computes:

$$
O_j = (Q_j K_j^\top \odot M_j) V_j \quad \text{— } O(C^2 d) \text{ per chunk}
$$

Between chunks, a **linear** associative scan propagates boundary states:

$$
h_j = A_j^{(C)} h_{j-1} + h_j^{\text{local}} \quad \text{— } O(T/C) \text{ sequential steps}
$$

There is a fundamental tension:

| Chunk size $C$ | Intra-chunk cost | Inter-chunk cost | Quality |
|----------------|-----------------|-----------------|---------|
| Small ($C = 16$) | Low ($C^2 d = 16^2 \cdot d$) | High ($T/16$ scan steps) | Worse (more info lost at boundaries) |
| Medium ($C = 64$) | Moderate | Moderate | Good (current default) |
| Large ($C = 256$) | High ($C^2 d = 256^2 \cdot d$) | Low ($T/256$ scan steps) | Best (more context captured in parallel) |

**Current practice** uses $C = 64$–$128$ as a compromise. But this is dictated by the **quadratic cost** within each chunk, not by any fundamental architectural limitation. If we could make intra-chunk computation $O(Cd)$ instead of $O(C^2 d)$, we could use $C = 256$–$512$ "for free," getting both better quality (more context per chunk) and better throughput (fewer scan steps).

### FMMformer's insight applied to chunks

FMMformer (Nguyen et al., NeurIPS 2021) decomposes the $N \times N$ attention matrix into:

$$
\hat{V} = (w_1 D + w_2 L) V
$$

where $D$ is a banded (near-field) matrix of width $k$ and $L$ is a rank-$r$ (far-field) matrix. The key insight is that **attention matrices are approximately diagonal-plus-semi-separable** — after removing a banded matrix, the residual has very low numerical rank.

This insight applies **perfectly** to the intra-chunk $C \times C$ attention matrix in chunkwise GLA:

1. **Near-field:** Within a chunk, tokens close together (within $k$ positions) have strong, sharp attention patterns — these need full softmax-quality interaction. This is a sliding window of width $k$ within the chunk.

2. **Far-field:** Tokens far apart within the chunk (distance $> k$) have diffuse, low-rank attention patterns. These can be captured by $r$ feature maps (linear attention channels), which is what the inter-chunk recurrence already does globally.

The decomposition is especially natural for chunkwise SSMs because:
- The **existing recurrence** between chunks already provides a global far-field mechanism
- The **intra-chunk quadratic** is only needed for the near-field (local) interactions
- Making the intra-chunk near-field banded ($k \ll C$) reduces cost from $O(C^2 d)$ to $O(Ckd)$

### Why this hasn't been done

1. **FMMformer was applied to full-sequence Transformers**, not to the intra-chunk sub-problem of chunkwise SSMs. The connection between FMM decomposition and chunkwise partitioning hasn't been made explicitly.
2. **Chunkwise SSMs already have a "far-field"** (the inter-chunk recurrence), but no one has pointed out that the intra-chunk quadratic is wastefully computing far-field interactions that the recurrence handles anyway.
3. **The far-field within a chunk is redundant** with the recurrence's far-field — eliminating it trades a small quality loss within the chunk for a large compute savings, enabling larger chunks that more than compensate.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster?** Yes — replacing $C^2$ quadratic attention with $Ck$ banded attention is guaranteed cheaper for $k < C$. With $k = 16$, $C = 256$: the near-field costs $4096d$ vs. $65536d$ for full quadratic — a $16\times$ reduction in the dominant cost. Even with the far-field overhead ($rCd$ for $r = 2$), total is $\sim 5120d$ vs. $65536d$ — $12.8\times$ cheaper.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — the near-field is a sliding window attention (FlashAttention-2 with `window_size=k` — already implemented). The far-field is two small GEMMs (linear attention). Both are standard, well-optimized GPU primitives.

3. **Does it reduce HBM bandwidth?** Yes — the banded near-field never materializes the full $C \times C$ matrix (only the $C \times (2k+1)$ band). The far-field's intermediate is $d \times d$ (not $C \times C$). Total memory: $O(Ck + d^2)$ vs. $O(C^2)$.

### Memory access pattern analysis

**Near-field (banded):**
- Sliding window: each query attends to $2k+1$ contiguous keys — fully coalesced memory access
- FlashAttention-2's window mask is exactly this operation, running at near-peak throughput
- Tiles of $Q$ and $K$ overlap by $2k$ positions between adjacent query tiles — high cache reuse

**Far-field (linear attention):**
- Right-multiply-first: $\phi(K)^\top V \in \mathbb{R}^{d \times d}$ — small dense matmul, compute-bound
- Left-multiply: $\phi(Q) (\phi(K)^\top V) \in \mathbb{R}^{C \times d}$ — standard matmul, tensor-core friendly
- Feature map application ($\text{elu}+1$, $\tanh$): elementwise, bandwidth-bound but cheap

**Combined:**
- Near-field and far-field are independent and can execute **concurrently** on different SMs
- Final blending is elementwise (negligible cost)

### Parallelism analysis

**Intra-chunk:** Both near-field and far-field are fully parallelizable within a chunk:
- Near-field: each query's $2k+1$ attention scores are independent
- Far-field: each feature map's contribution is independent across $r$ kernels

**Inter-chunk:** Unchanged — the same associative scan propagates boundary states. But with larger $C$, there are **fewer chunks** ($T/C$ drops), so the sequential scan has fewer steps.

**Tensor Core mapping:**
- Near-field: maps to mma instructions via FlashAttention's tiling strategy
- Far-field: two standard dense GEMMs per kernel ($d \times C$ × $C \times d$ = $d \times d$ output) — direct tensor core use

## Related Work

- **[FMMformer](https://arxiv.org/abs/2108.02347)** (Nguyen et al., NeurIPS 2021): Proposed the near-field/far-field decomposition for full-sequence Transformers. Achieves 60.74% LRA accuracy (vs. 58.70% softmax). **Our approach**: Applies this decomposition to the intra-chunk sub-problem of chunkwise SSMs, not to full-sequence attention.
- **[GLA](https://arxiv.org/abs/2312.06635)** (Yang et al., 2024): Chunkwise parallel gated linear attention with hardware-efficient training. Uses full $C \times C$ quadratic intra-chunk. **Our approach**: Replaces the full quadratic with banded + linear, enabling larger chunks.
- **[Gated DeltaNet](https://arxiv.org/abs/2406.06484)** (Yang et al., ICLR 2025): Extends chunkwise parallel scan to delta-rule-based updates. Same $O(C^2 d)$ intra-chunk bottleneck. **Our approach**: Directly reduces this bottleneck.
- **[Griffin](https://arxiv.org/abs/2402.19427)** (De et al., 2024): Hybrid model mixing recurrent blocks with local (sliding window) attention. Uses sliding window as a separate layer type. **Our approach**: Integrates sliding window as the near-field of the intra-chunk computation, fused with the recurrent far-field.
- **[Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)** (Moonshot AI, 2025): Hybrid of delta attention + MLA with chunkwise training. **Our approach**: Orthogonal — could apply near-far decomposition to KDA's intra-chunk computation as well.

No prior work applies the FMM near-far decomposition specifically to the intra-chunk quadratic attention of chunkwise parallel SSMs.

## Mathematical Formulation

### Standard Chunkwise GLA Intra-Chunk

For chunk $j$ with $Q_j, K_j, V_j \in \mathbb{R}^{C \times d}$ and causal decay mask $M_j \in \mathbb{R}^{C \times C}$:

$$
O_j^{\text{intra}} = (Q_j K_j^\top \odot M_j) V_j
$$

where $M_{j,st} = \prod_{i=s+1}^{t} \alpha_{j,i}$ for $s \leq t$ (causal, with learned decay rates $\alpha_{j,i}$), and $M_{j,st} = 0$ for $s > t$.

**Cost:** $O(C^2 d)$ FLOPs, $O(C^2)$ memory for intermediate scores.

### Near-Far Field Intra-Chunk (Proposed)

Decompose the intra-chunk attention into:

$$
O_j^{\text{intra}} = w_1 \cdot D_j V_j + w_2 \cdot L_j V_j
$$

where $w_1, w_2 > 0$ are learnable blending weights.

**Near-field $D_j$ (banded, width $k$):**

$$
D_{j,st} = \begin{cases}
Q_{j,s}^\top K_{j,t} \cdot M_{j,st} & \text{if } 0 \leq s - t \leq k \text{ (causal band)} \\
0 & \text{otherwise}
\end{cases}
$$

With local softmax normalization within the band:

$$
\tilde{D}_{j,s\cdot} = \text{softmax}(D_{j,s,\max(0,s-k):s} / \sqrt{d})
$$

The causal banded matrix only includes $\min(k, s+1)$ entries per row $s$, so:

$$
D_j V_j \text{ costs } O(C \cdot k \cdot d) \text{ FLOPs}
$$

This is implemented as a **causal sliding window attention** of width $k$ within the chunk — directly maps to FlashAttention-2 with `window_size=k`.

**Far-field $L_j$ (rank-$r$ linear attention with gated decay):**

$$
L_j V_j = \sum_{l=1}^{r} \phi_l(Q_j) \left( \phi_l(K_j)^\top V_j \right)
$$

where $\phi_l$ are feature maps (e.g., $\phi_1(x) = \text{elu}(x) + 1$, $\phi_2(x) = \text{elu}(-x) + 1$).

For the **causal** case within a chunk, use the recurrent form with gated decay (RFA-Gate inspired):

$$
S_{j,t}^{(l)} = \alpha_{j,t} \cdot S_{j,t-1}^{(l)} + (1 - \alpha_{j,t}) \cdot \phi_l(K_{j,t}) V_{j,t}^\top
$$
$$
(L_j V_j)_t = \sum_{l=1}^{r} \phi_l(Q_{j,t})^\top S_{j,t}^{(l)}
$$

**Cost:** $O(r \cdot C \cdot d^2)$ for the parallel form (right-multiply-first), or $O(r \cdot C \cdot d)$ for the recurrent form per step. Since $C$ is small (even at 256–512), the parallel form is preferred: $r$ independent $d \times d$ matmuls.

Alternatively, use the **parallel form** with causal masking:

$$
(L_j V_j)_t = \sum_{l=1}^{r} \phi_l(Q_{j,t})^\top \sum_{i \leq t} \gamma_{t,i} \phi_l(K_{j,i}) V_{j,i}^\top
$$

where $\gamma_{t,i} = \prod_{s=i+1}^{t} \alpha_{j,s}$ is the cumulative decay — **this is exactly the existing inter-chunk recurrence, but applied within the chunk at far-field resolution.**

### Combined Cost

$$
\text{Cost}_{\text{near-far}} = \underbrace{O(C \cdot k \cdot d)}_{\text{near-field}} + \underbrace{O(r \cdot C \cdot d^2)}_{\text{far-field (parallel)}} = O(C d (k + r d))
$$

Compare to full quadratic:

$$
\text{Cost}_{\text{quadratic}} = O(C^2 d)
$$

**Crossover:** Near-far is cheaper when $k + rd < C$. For $k = 16$, $r = 2$, $d = 64$: $16 + 128 = 144$. So near-far is cheaper for $C > 144$. At $C = 256$: near-far is $256 \cdot 64 \cdot 144 = 2.36 \text{M}$ FLOPs vs. quadratic $256^2 \cdot 64 = 4.19 \text{M}$ — **1.78× cheaper**.

At $C = 512$: near-far is $512 \cdot 64 \cdot 144 = 4.72 \text{M}$ vs. quadratic $512^2 \cdot 64 = 16.78 \text{M}$ — **3.55× cheaper**.

### Impact on Inter-Chunk Scan

With larger chunks:

$$
\text{Scan steps} = T / C
$$

| $C$ | Scan steps ($T = 8192$) | Scan overhead |
|-----|------------------------|---------------|
| 64 | 128 | High |
| 128 | 64 | Moderate |
| 256 | 32 | Low |
| 512 | 16 | Very low |

Fewer scan steps means less sequential computation and fewer boundary state propagations — both throughput and quality improve.

### Key Variables

- $C$ — chunk size (proposed: 256–512, up from 64–128)
- $d$ — head dimension (64–128)
- $k$ — near-field bandwidth (hyperparameter, 8–32)
- $r$ — far-field rank (number of feature maps, 2–3)
- $w_1, w_2$ — learnable blending weights
- $\alpha_{j,t}$ — per-step decay rates (input-dependent)
- $\phi_l(\cdot)$ — feature maps for far-field linear attention
- $T$ — sequence length

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA with near-far field intra-chunk decomposition |
| Layers | $L = 24$ |
| Hidden dim | $d_{\text{model}} = 2048$ |
| Head dim | $d = 64$ |
| Heads | $H = 32$ |
| State dim | $n = 16$ per head |
| Chunk size | $C = 256$ (primary), $C \in \{64, 128, 256, 512\}$ (ablation) |
| Near-field $k$ | 16 (primary), $k \in \{8, 16, 32\}$ (ablation) |
| Far-field $r$ | 2 kernels: $\text{elu}+1$ and $\text{elu}(-\cdot)+1$ |
| Kernel | Triton: FlashAttention-2 (window) + linear attention |
| Precision | BF16 compute, FP32 accumulator |

### Baseline

1. **GLA (C=64)**: Standard chunkwise GLA with full quadratic intra-chunk — current practice
2. **GLA (C=128)**: Larger chunk baseline — shows quality gain from bigger chunks at quadratic cost
3. **GLA (C=256, full quadratic)**: Same chunk size as proposed but with $O(C^2 d)$ — isolates the near-far decomposition benefit from the chunk-size benefit
4. **Sliding window attention (Longformer-style)**: Pure local attention — shows what happens without the far-field

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.2\times$ GLA(C=64) | Tokens/sec, 8×A100 |
| Perplexity | $\leq$ GLA(C=64) | WikiText-103, SlimPajama |
| Intra-chunk latency | $< 0.5\times$ GLA(C=256,quad) | µs per chunk, NCU |
| Peak memory | $< 0.7\times$ GLA(C=256,quad) | Peak GPU memory at seq 8192 |
| Long-context quality | $> 1.05\times$ GLA(C=64) | SCROLLS, passkey retrieval |

### Estimated Compute

**MVE (kernel benchmark)**: ~30 minutes on single GPU (~$2) — benchmark banded + linear vs. full quadratic at various $C$
**Small-scale (370M model)**: 16 GPU-hours on A100 (~$64) — pretraining comparison (quality + speed)
**Full-scale (1.3B model)**: 128 GPU-hours on A100 (~$512) — end-to-end pretraining

## Expected Outcome

**If hypothesis is correct:**

- At $C = 256$, $k = 16$, $r = 2$: intra-chunk is $1.8\times$ cheaper than full quadratic, enabling $1.2$–$1.4\times$ training throughput
- Perplexity is equal to or better than GLA(C=64) because larger chunks capture more context in the parallel phase
- Long-context tasks (passkey retrieval, SCROLLS) show clear improvement from reduced boundary information loss
- Near-field bandwidth $k = 16$ captures 95%+ of the attention mass within chunks (validated by attention entropy analysis)
- Far-field with $r = 2$ captures the remaining diffuse interactions adequately

**If hypothesis is wrong:**

- **Scenario A**: Near-field bandwidth $k = 16$ is too narrow — important intra-chunk interactions happen at distance $> 16$. **Learn**: Need larger $k$ (or adaptive $k$ per layer/head). **Fix**: Use $k = 32$ or learn $k$ per head. If even $k = 32$ is insufficient, the near-far decomposition is wrong for this chunk size.
- **Scenario B**: Far-field linear attention with $r = 2$ feature maps is too coarse — quality degrades despite larger chunks. **Learn**: Simple elu-based feature maps lose too much information. **Fix**: Use FAVOR+ positive features ($r = 4$–$8$) or FAVOR# SADERF variant for data-adaptive feature maps.
- **Scenario C**: The inter-chunk recurrence already provides sufficient far-field, making the intra-chunk far-field redundant — removing it (pure banded + recurrence) works just as well. **Learn**: Simpler is better — just use banded intra-chunk attention. **Value**: This would be a positive result — even simpler than the full near-far proposal.
- **Scenario D**: Larger chunk sizes don't help quality — the quality bottleneck is the recurrence's state capacity, not chunk size. **Learn**: Invest in richer state transitions (Proposals 001–030) rather than larger chunks. **Value**: Negative but informative — redirects effort.

## Minimum Viable Experiment

### Setup
- **Model**: Tiny GLA (2 layers, $d_{\text{model}} = 128$, 4 heads, $d = 32$, ~2M params)
- **Task**: Synthetic associative recall at varying context lengths
- **Data**: 10K synthetic sequences of length 512–2048, with key-value pairs to recall
- **Compute**: Single GPU (A100), $< 10$ minutes

### Implementation Steps

1. **Kernel benchmark** (3 min): Compare wall-clock time for:
   - Full quadratic intra-chunk ($C = 256$, full $Q K^\top$ computation)
   - Near-far intra-chunk ($C = 256$, $k = 16$, $r = 2$)
   - Full quadratic ($C = 64$, baseline)

   On synthetic tensors $Q, K, V \in \mathbb{R}^{16 \times 4 \times C \times 32}$

2. **Quality test** (7 min): Train 2-layer GLA on associative recall task:
   - GLA(C=64, full quadratic) — baseline
   - GLA(C=256, near-far, $k=16$, $r=2$) — proposed
   - GLA(C=256, full quadratic) — quality ceiling

   Compare accuracy on recall at distances 16, 32, 64, 128, 256.

### Success Criteria

- Near-far intra-chunk ($C = 256$) is $> 1.3\times$ faster than full quadratic ($C = 256$)
- Near-far ($C = 256$) is at least as fast as full quadratic ($C = 64$) — larger chunk at no extra cost
- Associative recall accuracy at distance $\leq 16$ matches full quadratic ($C = 256$) — near-field captures local interactions
- Associative recall accuracy at distance 32–256 is within 10% of full quadratic ($C = 256$) — far-field captures long-range

### Failure Criteria

- **Kill if**: Near-far ($C = 256$) is slower than full quadratic ($C = 64$) — the overhead of the far-field GEMMs negates the bandwidth savings
- **Kill if**: Associative recall at distance 32–256 drops to random chance — the far-field completely fails to capture non-local interactions within the chunk
- **Investigate if**: Accuracy matches at short distances but degrades at medium distances (32–64) — increase $k$ or $r$

### Why This Test Is Sufficient

- Associative recall directly tests the ability to retrieve information at various distances — exactly the capability that the near-far decomposition must preserve
- If near-field ($k = 16$) captures local interactions and far-field ($r = 2$) captures diffuse interactions in a 2-layer model, the same mechanism will scale to deeper models where each layer adds to the overall context window
- The kernel benchmark directly validates the throughput hypothesis — if $C = 256$ near-far is cheaper than $C = 64$ full quadratic, the idea is worth scaling up

## Theoretical Analysis

### Complexity Comparison

| Operation | Full Quadratic ($C = 64$) | Full Quadratic ($C = 256$) | Near-Far ($C = 256$) |
|-----------|--------------------------|---------------------------|---------------------|
| Intra-chunk | $C^2 d = 262\text{K}$ | $C^2 d = 4.19\text{M}$ | $C(k + rd)d = 2.36\text{M}$ |
| Scan steps ($T = 8192$) | $128$ | $32$ | $32$ |
| Total intra + scan | $128 \times 262\text{K} + 128 \times n d$ | $32 \times 4.19\text{M} + 32 \times n d$ | $32 \times 2.36\text{M} + 32 \times n d$ |
| | $\approx 33.6\text{M}$ | $\approx 134\text{M}$ | $\approx 75.5\text{M}$ |

**Comparison with baseline ($C = 64$, full quad):**

At $C = 256$ with near-far: intra-chunk costs $2.36\text{M}$ per chunk × $32$ chunks = $75.5\text{M}$ total
At $C = 64$ with full quad: intra-chunk costs $262\text{K}$ per chunk × $128$ chunks = $33.6\text{M}$ total

Near-far at $C = 256$ costs $\sim 2.2\times$ more total intra-chunk FLOPs than $C = 64$ full quadratic. **However**, this is offset by:
1. **4× fewer scan steps** (128 → 32) — the scan's sequential overhead decreases proportionally
2. **Better quality** — longer chunks capture more context in the parallel (non-approximated) phase
3. **Better GPU utilization** — larger chunks have higher arithmetic intensity and better tensor core occupancy

The true benefit is most visible at $C = 256$ vs. $C = 256$ full quadratic: **$1.78\times$ cheaper** for the same chunk size.

### Hardware-Specific Considerations

**A100 (Ampere):**
- FlashAttention-2 sliding window: already implemented and optimized for A100
- 192 KB shared memory: can hold $k \times d = 16 \times 64 \times 2 = 2$ KB of K/V per window — easily fits
- Linear attention GEMMs ($d \times d$): $64 \times 64$ matmul tiles map directly to mma.sync ($16 \times 16 \times 16$)
- Expected throughput gain: $1.2$–$1.4\times$ at $C = 256$

**H100 (Hopper):**
- FlashAttention-3 sliding window with warp specialization and TMA
- 256 KB shared memory: can double-buffer K/V windows
- WGMMA for async far-field matmuls overlapped with near-field computation
- Expected throughput gain: $1.3$–$1.5\times$ at $C = 256$

**Register pressure:**
- Near-field: standard FlashAttention register usage (accumulator for $B_r \times d$ output tile)
- Far-field: $r$ accumulators of size $d \times d$ = $2 \times 64 \times 64 \times 4 = 32$ KB — may need shared memory for $d > 64$
- **Solution**: Tile the far-field's $d \times d$ accumulator if it exceeds register budget

## Risks & Limitations

1. **Far-field quality with simple feature maps**: The elu-based feature maps ($\phi(x) = \text{elu}(x) + 1$) provide a coarse approximation. For intra-chunk interactions at distance $k < \delta < C$, this may be insufficient. **Mitigation**: Use FAVOR+ positive features or cosFormer's cosine reweighting (trick 031) for better approximation quality. If quality is still poor, fall back to pure banded attention (Scenario C above).

2. **Hyperparameter sensitivity ($k$, $r$)**: The near-field bandwidth $k$ and far-field rank $r$ are new hyperparameters. Optimal values may differ per layer, per head, or per task. **Mitigation**: Start with $k = 16$, $r = 2$ (conservative) and ablate. Consider learned $k$ per head (make attention weights' effective bandwidth a learned parameter).

3. **Backward pass complexity**: The backward pass through the near-far decomposition requires computing gradients through both the banded attention and the linear attention components. This adds engineering complexity but not algorithmic difficulty — both components have well-understood gradients. **Mitigation**: Use FlashAttention-2's backward for the near-field (already implemented) and standard autograd for the far-field GEMMs.

4. **Interaction with decay mask**: The causal decay mask $M_j$ must be applied consistently to both near-field and far-field components. The near-field naturally applies it (windowed softmax with decay). The far-field's recurrent form naturally includes decay via the $\alpha_{j,t}$ gating. **No additional complexity**.

5. **Not beneficial at small chunk sizes**: If $C \leq k + rd$, near-far is not cheaper than full quadratic. For $k = 16$, $r = 2$, $d = 64$: crossover at $C = 144$. Below this, full quadratic is better. **Mitigation**: Use full quadratic for small $C$; switch to near-far at $C \geq 256$.

## Follow-up Experiments

1. **Adaptive $k$ per head**: Learn the near-field bandwidth per head via a soft attention window size parameter ($k_h = \text{softplus}(w_h)$). Different heads may specialize: some with wide near-field (local experts), others with narrow near-field + strong far-field (global experts).

2. **FAVOR#-enhanced far-field**: Replace elu-based feature maps with FAVOR# SADERF (trick 150) for data-adaptive variance reduction. This should improve far-field quality at minimal cost ($O(Md)$ same complexity, better constants).

3. **Combine with Proposal 032 (Chimera fusion)**: Apply Chimera's optimal block ordering to the **near-field** banded matmul chain ($Q_k K_k^\top \to$ mask $\to$ $\times V_k$ within the window). The near-field is exactly a 2-GEMM chain with small intermediate — Chimera can optimize tile sizes analytically.

4. **Combine with Proposal 035 (transposable sparse projections)**: Near-far chunks + sparse projections = compound speedup on both the projection GEMMs and the intra-chunk attention.

5. **Pure banded baseline**: If Scenario C holds (far-field redundant), test pure banded intra-chunk attention (no linear attention far-field at all). This would be the simplest possible modification and may be sufficient.

6. **Long-context scaling**: Test with $T = 32768$–$131072$ where the scan overhead becomes dominant. Near-far with $C = 512$ should shine here: only $T/512 = 64$–$256$ scan steps vs. $T/64 = 512$–$2048$ with standard GLA.

## Human Review

(To be filled by reviewer)

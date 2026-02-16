---
status: ongoing
priority: high
created: 2026-02-15
based_on: transposable-nm-mask-min-cost-flow, tsenor-entropy-regularized-transposable-masks, smooth-ste-continuous-sparse-projection, two-four-structured-sparsity, vnm-hierarchical-structured-sparsity, bilinear-gating-glu, chunkwise-parallel-scan
experiment_number: 035
experiment_log: experiment-log-035.md
---

# Transposable N:M Sparse Projection Pretraining for Gated Linear Attention

## Hypothesis

Pretraining a GLA/Mamba-2-style gated linear attention model with **transposable 4:8 structured sparsity** on all projection matrices ($W_Q, W_K, W_V, W_O, W_{\text{gate}}$), using TSENOR for GPU-accelerated mask computation and S-STE for continuous sparse training, will achieve $1.4$–$1.8\times$ **training throughput** improvement over the dense baseline on A100/H100, with $< 2\%$ perplexity degradation, because transposable masks enable Sparse Tensor Core acceleration in **both forward and backward passes** (66% of GEMMs accelerated, vs. only 33% with standard N:M), and the projection GEMMs constitute $\sim 75\%$ of per-layer FLOPs.

## Background

### Gap in existing proposals

Proposal 031 (VNM Sparse SSM Projections) applies V:N:M sparsity to the same projection matrices but with **non-transposable masks** — meaning Sparse Tensor Core acceleration applies only to the forward pass. In the backward pass, the weight gradients require computing with $W^\top$, which does not satisfy the N:M constraint under a standard mask. This means:

| GEMM | Forward (standard N:M) | Backward (standard N:M) | Forward (transposable) | Backward (transposable) |
|------|----------------------|------------------------|----------------------|------------------------|
| $y = xW$ | ✓ Sparse TC | ✗ Dense | ✓ Sparse TC | ✗ Dense |
| $\nabla_x = \nabla_y W^\top$ | ✗ Dense | ✗ Dense | ✓ Sparse TC | ✗ Dense |
| $\nabla_W = x^\top \nabla_y$ | ✗ Dense | ✗ Dense | ✗ Dense | ✗ Dense |

With transposable masks, **2 of 3 GEMMs** per projection layer use Sparse Tensor Cores (forward $xW$ and backward input gradient $\nabla_y W^\top$), versus only 1 of 3 with standard masks. Since projections dominate training time, this doubles the sparse acceleration.

### Why this hasn't been done

1. **Transposable masks were tested only on vision models and BERT** (ResNet, ResNeXt, VGG in the original paper; LLaMA pruning in TSENOR). No work applies them to SSM/GLA pretraining from scratch.
2. **TSENOR is new (2025)** and solves the scalability problem — previous transposable mask algorithms (min-cost flow) were CPU-only and took hundreds of seconds for large layers. TSENOR runs on GPU in $< 0.12$s for 8192×8192 matrices, making periodic mask recomputation during training feasible.
3. **S-STE + transposable masks** have never been combined. S-STE provides continuous sparse training gradients, but its mask selection is magnitude-based (keeps top-2 per group of 4). Transposable masks require a global constraint (both $W$ and $W^\top$ must satisfy N:M), which conflicts with greedy per-group selection. The combination requires a two-phase approach: S-STE provides the continuous weight landscape, then TSENOR extracts the optimal transposable mask from those weights.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster?** Yes — transposable 4:8 sparsity is proven to give $1.49\times$ inference speedup on LLaMA-2-7B (TSENOR paper), and training has 3× the GEMMs (forward + 2 backward), so the training speedup should be even larger since 2/3 GEMMs are accelerated vs. 1/3 for inference.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — the GEMMs use cuSPARSELt/Spatha with standard 2:4 sparse tensor core instructions. The transposable constraint is enforced at the mask level, not the kernel level. TSENOR's Dykstra algorithm is pure element-wise + matrix-vector ops.

3. **Does it reduce HBM bandwidth?** Yes — 4:8 sparsity means 75% sparsity, so weights are $\sim 38\%$ the size (data + metadata). Loading 38% as much weight data per forward/backward pass directly reduces HBM pressure.

### Memory access pattern analysis

**Coalesced access:** Sparse Tensor Cores on Ampere/Hopper access compressed weight data in contiguous format (2 values + 2-bit metadata per group of 4). This is inherently coalesced — the hardware reads consecutive memory addresses.

**Cache-friendly:** The compressed format reduces total bytes loaded from HBM, improving effective bandwidth. For 4:8 sparsity at FP16: each group of 8 elements stores 4 values (8 bytes) + 4 bits metadata ≈ 8.5 bytes, vs. 16 bytes dense. ~47% compression.

**Arithmetic intensity:** Unchanged per-nonzero (same FMA operations), but fewer nonzeros per output element → lower AI. However, Sparse Tensor Cores maintain throughput by processing 2× as many tiles per cycle, so effective TFLOPS is similar to dense.

### Parallelism analysis

**Full SM saturation:** Projection GEMMs are large ($d_{\text{model}} \times d_{\text{model}}$ or similar). At $d = 2048$, even with 4:8 sparsity, each GEMM has $> 10^6$ output elements — far more than $108 \times 128$ SMs × warps.

**Tensor Core mapping:** Direct — cuSPARSELt/Spatha maps 2:4 sparse inner products to Sparse Tensor Core mma instructions. The 4:8 transposable constraint is enforced at the mask level; the actual kernel sees standard 2:4 patterns within each 8-element window.

**No warp divergence:** All threads execute the same sparse matmul instruction. The sparsity pattern is fixed (static mask), so no conditional branching.

**No sequential bottleneck:** TSENOR mask recomputation is periodic (every $K$ steps, e.g., $K = 1000$). Between recomputations, masks are static and impose zero overhead.

## Related Work

- **[Accelerating Transformer Pre-Training with 2:4 Sparsity](https://arxiv.org/abs/2404.01847)** (Lu et al., 2024): Applied S-STE to Transformer pretraining with standard (non-transposable) 2:4 masks. Achieved near-lossless training at 50% sparsity. **Our approach**: Uses transposable 4:8 masks for 2× more GEMM acceleration and 75% sparsity (vs. 50%).
- **[TSENOR](https://arxiv.org/abs/2505.23949)** (2025): GPU-accelerated transposable mask finding for LLM pruning (post-training). Applied to LLaMA-3 1B–8B with Wanda/SparseGPT. **Our approach**: Uses TSENOR during pretraining (not post-training), combined with S-STE's continuous sparse gradients.
- **[Structured Sparse Transition Matrices for SSMs](https://arxiv.org/abs/2509.22284)** (IBM, NeurIPS 2025): Applied structured sparsity to SSM state transitions, not projection layers. Focus on expressivity (FSA state tracking), not training throughput. **Our approach**: Targets projection layers (75% of FLOPs), not state transitions.
- **Proposal 031 (VNM Sparse SSM Projections)**: Applies VNM sparsity to projections with standard (non-transposable) masks. Forward-only Sparse TC acceleration (33% of GEMMs). **Our approach**: Transposable masks give 66% GEMM acceleration.

No prior work combines transposable N:M sparsity + S-STE continuous training + GLA/SSM projection pretraining.

## Mathematical Formulation

### Standard Dense GLA Projection Layer

For a single projection $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:

**Forward:**
$$
y = xW \quad \text{— GEMM: } O(B T d_{\text{in}} d_{\text{out}})
$$

**Backward (input gradient):**
$$
\nabla_x = \nabla_y W^\top \quad \text{— GEMM: } O(B T d_{\text{in}} d_{\text{out}})
$$

**Backward (weight gradient):**
$$
\nabla_W = x^\top \nabla_y \quad \text{— GEMM: } O(B T d_{\text{in}} d_{\text{out}})
$$

Total: 3 GEMMs per projection per training step. With 5 projections ($W_Q, W_K, W_V, W_{\text{gate}}, W_O$): **15 GEMMs** per layer.

### Transposable 4:8 Sparse Projections (Proposed)

**Step 1 — TSENOR Mask Computation (periodic, every $K$ steps):**

Partition $W$ into $M \times M$ blocks (with $M = 8$ for 4:8 pattern). For each block $W_{\text{block}} \in \mathbb{R}^{8 \times 8}$, find the transposable mask $S^* \in \{0, 1\}^{8 \times 8}$ via:

$$
\max_S \langle S, |W_{\text{block}}| \rangle \quad \text{s.t.} \quad S \mathbf{1}_8 = 4 \cdot \mathbf{1}_8, \quad S^\top \mathbf{1}_8 = 4 \cdot \mathbf{1}_8, \quad 0 \leq S \leq 1
$$

Solved by entropy-regularized Dykstra iteration:

$$
S \leftarrow \exp(\tau |W_{\text{block}}|)
$$
$$
\text{For } t = 0, \ldots, T_{\text{iter}}-1: \quad S \leftarrow \text{Diag}(4 / (S \mathbf{1}_8)) \cdot S \cdot \text{Diag}(4 / (S^\top \mathbf{1}_8))
$$

Followed by greedy rounding to obtain binary $S^*$.

**Step 2 — S-STE Training (every step):**

Given the current mask $S^*$, apply S-STE's continuous sparse projection:

$$
W_{\text{sparse}} = \beta \cdot S_{\text{soft}}(W_{\text{dense}} \odot S^*_{\text{expanded}})
$$

where $S^*_{\text{expanded}}$ lifts the $8 \times 8$ block mask to the full weight matrix, and $S_{\text{soft}}$ applies soft-thresholding within each group of 4 (the inner 2:4 structure):

$$
(S_{\text{soft}}(\mathbf{a}))_i = \text{sign}(a_i) \cdot \max(|a_i| - |a_{(2)}|, 0)
$$

The scaling factor $\beta$ is computed once per recomputation cycle:

$$
\beta = \frac{W_{\text{dense}}^\top S_{\text{soft}}(W_{\text{dense}} \odot S^*)}{||S_{\text{soft}}(W_{\text{dense}} \odot S^*)||^2}
$$

**Step 3 — Sparse GEMMs (every step):**

$$
y = x \cdot \text{SpMM}(W_{\text{sparse}}) \quad \text{(forward — Sparse TC)}
$$
$$
\nabla_x = \nabla_y \cdot \text{SpMM}(W_{\text{sparse}}^\top) \quad \text{(backward input grad — Sparse TC, transposable!)}
$$
$$
\nabla_W = S^* \odot (x^\top \nabla_y) \quad \text{(backward weight grad — dense GEMM + mask)}
$$

### Sparse Tensor Core Utilization

| GEMM | Dense | Standard 2:4 | Transposable 4:8 (Ours) |
|------|-------|-------------|------------------------|
| $xW$ (forward) | Dense TC | Sparse TC ✓ | Sparse TC ✓ |
| $\nabla_y W^\top$ (input grad) | Dense TC | Dense TC ✗ | Sparse TC ✓ |
| $x^\top \nabla_y$ (weight grad) | Dense TC | Dense TC ✗ | Dense TC (masked) |
| **Sparse TC fraction** | 0% | 33% | **66%** |

### FLOPs and Bandwidth Analysis

Per-projection per-step, with $d_{\text{in}} = d_{\text{out}} = d$:

**Dense:**
$$
\text{FLOPs}_{\text{dense}} = 3 \times 2 B T d^2 = 6BTd^2
$$

**Transposable 4:8 (75% sparsity → 25% nonzeros):**
$$
\text{FLOPs}_{\text{sparse}} = 2 \times 2BT \cdot \frac{d^2}{4} + 2BTd^2 = BTd^2 + 2BTd^2 = 3BTd^2
$$

(Two GEMMs at 4× sparse speedup + one dense GEMM for weight gradient.)

**Theoretical speedup:** $6BTd^2 / 3BTd^2 = 2\times$.

**Practical speedup (accounting for Sparse TC overhead):** 4:8 gives ~1.7× per sparse GEMM (not 4×), so:

$$
\text{Speedup} = \frac{3}{2/1.7 + 1} = \frac{3}{2.176} \approx 1.38\times
$$

Over 5 projections per layer (75% of layer FLOPs):

$$
\text{Layer speedup} = \frac{1}{0.25 + 0.75/1.38} = \frac{1}{0.25 + 0.543} = \frac{1}{0.793} \approx 1.26\times
$$

With additional bandwidth savings (loading 47% compressed weights from HBM):

$$
\text{Adjusted layer speedup} \approx 1.3\text{–}1.5\times
$$

### Key Variables

- $d_{\text{model}}$ — model hidden dimension (2048)
- $d_k, d_v$ — key/value head dimensions (64–128)
- $H$ — number of heads (16)
- $M = 8$ — transposable block size (4:8 pattern)
- $K$ — mask recomputation interval (steps between TSENOR runs)
- $\beta$ — S-STE scaling factor (frozen between mask updates)
- $\tau$ — TSENOR entropy regularization temperature
- $T_{\text{iter}}$ — Dykstra iteration count (typically 10–20)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA-style with transposable 4:8 sparse projections |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Head dim | $d_k = d_v = 128$ |
| Heads | $H = 16$ |
| State dim | $n = 16$ per head |
| Sparsity | 4:8 transposable (75%) on all projections |
| Training | S-STE continuous pruning + TSENOR mask updates every $K = 500$ steps |
| Precision | BF16 compute, FP32 accumulator, Sparse Tensor Cores |

### Baseline

1. **Dense GLA** (flash-linear-attention): Standard dense projections — baseline throughput
2. **Standard 2:4 sparse GLA** (S-STE, non-transposable): 50% sparsity, forward-only Sparse TC
3. **Proposal 031 (VNM 64:2:8)**: 75% sparsity, forward-only Sparse TC — most directly comparable
4. **Dense Transformer** (FlashAttention-2): Reference architecture

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.3\times$ dense GLA | Tokens/sec, 8×A100 |
| Backward throughput | $> 1.5\times$ dense GLA | ms/backward pass |
| Perplexity | $< 1.02\times$ dense | WikiText-103, SlimPajama |
| Sparse TC utilization | 66% of GEMMs | nsight systems trace |
| TSENOR overhead | $< 1\%$ of training time | Wall-clock fraction |
| Memory (weights) | $< 0.5\times$ dense | Peak GPU memory |

### Estimated Compute

**MVE (kernel benchmark)**: ~1 hour on single GPU (~$4) — benchmark sparse vs. dense GEMMs with transposable masks
**Small-scale (370M model)**: 8 GPU-hours on A100 (~$32) — verify quality on small pretraining run
**Full-scale (1.3B model)**: 96 GPU-hours on A100 (~$400) — end-to-end pretraining comparison

## Expected Outcome

**If hypothesis is correct:**

- $1.3$–$1.5\times$ training throughput improvement over dense GLA at $d = 2048$
- Backward pass specifically shows $1.5$–$1.7\times$ improvement (two sparse GEMMs vs. zero in standard N:M)
- Perplexity within 2% of dense baseline on SlimPajama pretraining
- TSENOR mask updates take $< 0.12$s per layer (negligible at $K = 500$ step intervals)
- Compounding with Proposal 033 (EVT fusion): elementwise ops fused + projections sparsified = $1.5$–$2.0\times$ total

**If hypothesis is wrong:**

- **Scenario A**: Transposable 4:8 mask diversity is insufficient — the constraint (row-sum = col-sum = 4) limits which weights can survive, causing quality degradation $> 5\%$. **Learn**: The transposable constraint is too restrictive for SSM projections. **Fix**: Use 4:12 or 4:16 (more mask diversity at lower sparsity) or relax to "approximately transposable" (allow $\pm 1$ deviation from exact row/column sums).
- **Scenario B**: cuSPARSELt cannot exploit transposable masks — current sparse libraries don't support the $W^\top$ sparse path without explicit transposition + re-compression. **Learn**: Need custom kernel or library update. **Fix**: Implement custom Sparse TC kernel that stores both $W$ and $W^\top$ in compressed format (2× metadata storage, but no online transposition needed).
- **Scenario C**: S-STE's per-group soft-thresholding fights with TSENOR's global block constraint — the two-phase approach (S-STE weights + TSENOR mask) produces suboptimal masks because S-STE's weight magnitudes don't align with TSENOR's global optimum. **Learn**: Need joint optimization, not two-phase. **Fix**: Use TSENOR's importance scores ($|W_{ij}|$) directly instead of S-STE's magnitude ranking.

## Minimum Viable Experiment

### Setup
- **Model**: Tiny GLA (2 layers, $d = 128$, 4 heads, $d_k = 32$, ~2M params)
- **Task**: Language modeling on TinyStories (10K samples)
- **Data**: Synthetic or TinyStories subset
- **Compute**: Single GPU (A100), $< 10$ minutes

### Implementation Steps

1. **Kernel benchmark** (5 min): Compare cuSPARSELt throughput for standard 2:4 vs. transposable 4:8 on $128 \times 128$ and $2048 \times 2048$ matrices — verify that $W^\top$ sparse path actually works
2. **TSENOR speed** (2 min): Run TSENOR on 5 projection matrices of size $128 \times 128$ — verify mask computation is fast ($< 1$ second)
3. **Quality test** (5 min): Train 2-layer GLA with transposable 4:8 projections on TinyStories for 1000 steps, compare loss curve to dense baseline

### Success Criteria

- cuSPARSELt sparse GEMM with $W^\top$ (backward input gradient) achieves $> 1.3\times$ speedup over dense GEMM
- TSENOR computes transposable masks in $< 1$s for $128 \times 128$ matrices
- Transposable 4:8 sparse GLA achieves $< 1.1\times$ the loss of dense GLA after 1000 steps on TinyStories
- Total training step time (forward + backward) is $> 1.2\times$ faster than dense

### Failure Criteria

- **Kill if**: cuSPARSELt cannot do sparse $W^\top$ matmul (backward path) — transposable masks are useless without backward sparse acceleration
- **Kill if**: Training loss diverges or exceeds $2\times$ dense baseline after 1000 steps — transposable constraints are too restrictive
- **Investigate if**: Forward is fast but backward is not — cuSPARSELt may need separate compressed storage for $W^\top$

### Why This Test Is Sufficient

- The MVE tests all three critical components: (1) backward sparse GEMM viability, (2) mask computation speed, (3) training quality with constrained masks
- If the backward sparse GEMM works at small scale, it works at large scale (same hardware instructions)
- If 2-layer GLA trains successfully with transposable masks, the mask constraint doesn't fundamentally break learning — scaling adds capacity, not fixes broken mechanisms

## Theoretical Analysis

### Complexity Comparison

| Operation | Dense | Standard 2:4 | Transposable 4:8 (Ours) |
|-----------|-------|-------------|------------------------|
| Forward GEMM | $2BTd^2$ | $BTd^2$ (Sparse TC) | $\frac{BTd^2}{2}$ (Sparse TC, 75%) |
| Backward input grad | $2BTd^2$ | $2BTd^2$ (Dense) | $\frac{BTd^2}{2}$ (Sparse TC!) |
| Backward weight grad | $2BTd^2$ | $2BTd^2$ (Dense) | $2BTd^2$ (Dense, masked) |
| **Total per projection** | $6BTd^2$ | $5BTd^2$ | $3BTd^2$ |
| **Speedup** | $1\times$ | $1.2\times$ | $\mathbf{2\times}$ (theoretical) |

### Crossover Analysis

Transposable masks introduce periodic TSENOR overhead:

$$
\text{Overhead per step} = \frac{\text{TSENOR time per layer} \times L}{K \times \text{step time}} = \frac{0.12 \times 24}{500 \times 0.5} \approx 0.012 = 1.2\%
$$

This is negligible. The crossover point (where sparse training becomes faster than dense) is at step 0 — there's no warmup penalty.

### Hardware-Specific Considerations

**A100 (Ampere):**
- Sparse Tensor Cores: 624 TOPS (INT8) / 312 TFLOPS (FP16) — 2× dense peak
- cuSPARSELt: supports 2:4 structured sparsity natively
- 4:8 transposable: decomposes to two rounds of 2:4 within each 8-element window
- Expected: 1.3–1.5× training speedup

**H100 (Hopper):**
- Sparse Tensor Cores: 1979 TOPS (INT8 w/ sparsity) / 989 TFLOPS (FP16 w/ sparsity)
- cuSPARSELt 0.6+: improved support for larger block sizes
- TMA + Sparse TC: async weight loading + sparse compute overlap
- Expected: 1.4–1.8× training speedup

## Risks & Limitations

1. **cuSPARSELt backward path**: Current cuSPARSELt may not support efficient sparse $W^\top$ multiplication without explicit transposition and re-compression. **Mitigation**: Store both $W_{\text{compressed}}$ and $W^\top_{\text{compressed}}$ — doubles metadata storage (< 1% of total memory) but avoids online transposition.

2. **Mask recomputation frequency**: If weights change significantly between mask updates (every $K$ steps), the mask becomes stale. **Mitigation**: Monitor mask stability (fraction of mask entries that change); increase $K$ if stable, decrease if volatile. TSENOR is fast enough ($< 0.12$s) to run frequently.

3. **4:8 vs. 2:4 mask diversity**: 4:8 transposable masks have fewer valid patterns than 2:4 non-transposable (mask diversity $\sim 10^{13}$ vs. $10^{12}$). This is actually favorable — 4:8 transposable has *more* diversity than 2:4 structured.

4. **Interaction with S-STE**: S-STE's soft-thresholding and TSENOR's optimal transport are two different optimization objectives. They may not align perfectly. **Mitigation**: Use TSENOR's importance scores (not S-STE's) for mask selection; use S-STE only for the continuous gradient computation.

5. **No weight gradient acceleration**: The weight gradient GEMM ($x^\top \nabla_y$) cannot use Sparse TC because it's a dense matmul masked afterward. This limits the maximum theoretical speedup to $2\times$ (not $3\times$). **Mitigation**: Weight gradient is typically the smallest cost (can use FP8 or gradient checkpointing to amortize).

## Follow-up Experiments

1. **Combine with Proposal 033 (EVT fusion)**: Sparse projections with fused SiLU/sigmoid epilogues — compound both optimizations for $1.6$–$2.2\times$ total speedup.
2. **Combine with Proposal 034 (BRGEMM state accumulation)**: Sparse projections produce smaller K/V vectors → state accumulation has fewer nonzeros → potential further speedup in the scan kernel.
3. **4:16 transposable**: Higher sparsity (87.5%) with TSENOR at larger block size — even more aggressive compression if quality holds.
4. **FP8 + transposable 4:8**: Combine quantization with structured sparsity for compound $4\times$ theoretical speedup.
5. **Dynamic mask schedule**: Start dense, gradually increase sparsity during training (warmup phase dense → 2:4 → 4:8 transposable), following the three-staged approach from VNM.

## Human Review

(To be filled by reviewer)

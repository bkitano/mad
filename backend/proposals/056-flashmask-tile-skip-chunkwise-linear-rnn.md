---
status: ongoing
priority: high
created: 2026-02-16
based_on: flashmask-column-sparse-tile-skip (191), gla-secondary-chunking-log-space-gating (177), gated-deltanet-chunkwise-wy-gating (203), tfla-two-level-tiled-chunkwise-parallelism (158), fused-chunkwise-ssd-atomic-state-passing (182)
experiment_number: 056
experiment_log: experiment-log-056.md
---

# FlashMask Tile-Skip for Chunkwise Linear RNN Causal and Document-Packed Training

## Hypothesis

Adapting FlashMask's column-sparse tile-level block skipping to the intra-chunk attention computation of chunkwise linear RNNs (GLA, Gated DeltaNet, mLSTM) will achieve $1.4$–$1.8\times$ wall-clock speedup for the intra-chunk kernel on causal models by skipping ~50% of sub-chunk tile computations (the upper-triangular tiles that are zeroed by the causal gate mask $D^{(k)}$), and will additionally enable **efficient document-packed training** for linear RNNs — a capability that currently requires padding or separate sequence processing — by encoding cross-document boundaries as column-sparse masks with $O(T)$ memory.

## Background

### The wasted computation in causal chunkwise linear attention

In chunkwise linear RNN training, the intra-chunk output computation involves:

$$
O_{[n]}^{\text{intra}} = \left(\frac{Q_{[n]} K_{[n]}^\top}{\sqrt{d_k}} \odot D_{[n]}\right) V_{[n]}
$$

where $D_{[n]} \in \mathbb{R}^{C \times C}$ is a **causal gate mask** — lower-triangular, with entries $D_{ij} = \prod_{l=j+1}^{i} \alpha_l$ for $i \geq j$ and $D_{ij} = 0$ for $i < j$.

When this is tiled via TFLA's two-level tiling (sub-chunk size $c$, primary chunk size $C$), the $C \times C$ matrix is divided into $(C/c)^2$ tiles of size $c \times c$. The causal structure means:

- **Lower-triangular tiles** ($i > j$): Fully non-zero, require full matmul computation
- **Diagonal tiles** ($i = j$): Partially masked (causal within tile), require element-wise masking
- **Upper-triangular tiles** ($i < j$): Entirely zero — **no computation needed**

For $C = 128$, $c = 16$: there are $8 \times 8 = 64$ tiles total. Of these, 28 are lower-triangular (fully computed), 8 are diagonal (partially computed), and **28 are upper-triangular (wastefully computed and then masked to zero in current implementations)**.

**Current GLA/TFLA kernels already skip upper-triangular tiles** for the basic causal mask by using a simple `if i < j: continue` check. **However**, they cannot skip tiles for more complex patterns:

1. **Document-packed sequences**: When multiple documents are packed into one sequence for efficient training, tokens from different documents should not attend to each other. This creates a **block-diagonal** attention pattern within each chunk — and the boundary between documents can fall arbitrarily within a chunk.

2. **Sliding window within chunks**: For hybrid architectures that combine linear RNN layers with sliding-window attention, the intra-chunk "attention" may have a banded structure.

3. **Variable-length sequences with padding**: When batching variable-length sequences with padding, padded positions should not contribute to the state update.

**FlashMask's column-sparse representation** can encode all these patterns with $O(T)$ memory and enable tile-level skipping for the complex cases where simple causal checks don't suffice.

### Why FlashMask for linear RNNs is non-trivial

FlashMask was designed for softmax attention, where masking means setting $M_{ij} = -\infty$ (which zeros out the softmax output). In linear attention / linear RNNs, masking has a different effect:

1. **No softmax**: The attention scores are multiplied directly by the gate mask $D$, not passed through softmax. A masked position contributes exactly zero (not exponentially small).

2. **State update coupling**: In linear RNNs, the mask affects not just the output but also the **state update**. If tokens from document B should not affect document A's state, the inter-chunk recurrence must be modified at document boundaries.

3. **Gate mask structure**: The causal gate $D_{ij} = \prod_{l=j+1}^{i} \alpha_l$ has a multiplicative structure. At a document boundary where $\alpha_{\text{boundary}} = 0$ (full state reset), $D$ becomes block-diagonal — which FlashMask can represent.

**Key insight**: Setting $\alpha_t = 0$ at document boundaries in GLA/Gated DeltaNet causes the cumulative gate $D_{ij}$ to be exactly zero for cross-document pairs, creating the block-diagonal pattern that FlashMask can exploit for tile skipping. This naturally integrates document-packed training with the existing gating mechanism.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than current GLA kernels for document-packed training?** Yes — current GLA kernels either (a) process each document separately (underutilizing GPU parallelism with small sequences), (b) pad to max length (wasting compute on padding), or (c) pack without masking (cross-document attention contamination). FlashMask tile-skip enables packing without contamination and without wasted compute.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — take the existing GLA Triton kernel's inner loop over sub-chunk tiles. Before loading $K_j, V_j$ for sub-chunk $j$, check: `if tile_is_fully_masked(i, j, LTS, LTE): continue`. The mask check is two integer comparisons per tile — zero overhead. For partially masked tiles, load the column-wise mask vectors and apply element-wise within the tile.

3. **Does it reduce HBM bandwidth or increase compute utilization?** Yes — skipped tiles mean no HBM load for $K_j, V_j$ of that tile, no tensor core compute, and no write-back. For document-packed training with average document length $\bar{L}_{\text{doc}} \ll C$, up to $(1 - \bar{L}_{\text{doc}}/C) \times 50\%$ of tiles can be skipped beyond the basic causal skip.

## Related Work

- **[FlashMask (Wang et al., ICLR 2025)](https://arxiv.org/abs/2410.01359)**: Introduced column-sparse mask representation for softmax FlashAttention. Achieves $O(N)$ mask memory and 1.65–3.22× training speedup. **Softmax attention only** — not applied to linear attention or linear RNNs.
- **[FlexAttention (PyTorch, 2024)](https://pytorch.org/blog/flexattention/)**: Compiler-generated block masks for softmax attention within `torch.compile`. Supports arbitrary masks but with $O(N^2/B_r B_c)$ block-level memory. **Not applicable to linear attention kernels.**
- **[GLA (Yang et al., ICML 2024)](https://arxiv.org/abs/2312.06635)**: Chunkwise linear attention with secondary chunking. The kernel skips upper-triangular sub-chunk tiles for basic causal masking but cannot handle document-packed or other complex patterns.
- **[TFLA (Beck et al., NeurIPS 2025)](https://arxiv.org/abs/2503.14376)**: Two-level tiled chunkwise computation. Notes that the intra-chunk matrix is causal but does not implement tile-level skipping beyond basic causal.
- **[Fused SSD (Astra et al., 2026)](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/)**: Fuses chunkwise SSD kernels but does not address masking/skipping.
- **Proposal 048 (segmented matmulscan packed variable-length)**: Handles variable-length sequences via segmented scan for the inter-chunk recurrence but does not optimize the intra-chunk computation with tile skipping.

**Gap**: No existing work applies FlashMask-style tile-level block skipping to the intra-chunk computation of chunkwise linear RNNs, and no existing work enables efficient document-packed training for linear RNNs with tile-skip optimization.

## Mathematical Formulation

### Column-Sparse Mask for Gated Linear RNNs

**Standard GLA intra-chunk computation:**

$$
P_{[n]} = \tilde{Q}_{[n]} \tilde{K}_{[n]}^\top \odot D_{[n]}, \quad O_{[n]}^{\text{intra}} = P_{[n]} V_{[n]}
$$

where $D_{[n]} \in \mathbb{R}^{C \times C}$ with:

$$
D_{[n],ij} = \begin{cases} \prod_{l=j+1}^{i} \alpha_l & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}
$$

**Document-packed gate encoding:**

For a packed sequence with document boundaries at positions $\{b_1, b_2, \ldots\}$, set:

$$
\alpha_{b_k} = 0 \quad \text{for all boundary positions } b_k
$$

This forces $D_{ij} = 0$ whenever $i$ and $j$ are in different documents (because the product $\prod_{l=j+1}^{i} \alpha_l$ includes a zero factor at the boundary). The result is a **block-diagonal** $D$ matrix.

**Column-sparse representation of $D$:**

For each key column $j$, define:
- $LTS_j = j$ (causal: row $j$ is the first visible row)
- $LTE_j = \min(b_k : b_k > j)$ (the next document boundary after $j$)
- $UTS_j = 0$, $UTE_j = j$ (standard causal: rows before $j$ are masked)

These four $O(T)$ vectors exactly represent the document-packed causal mask for the full sequence. Within a chunk of size $C$, the relevant slice of these vectors encodes the sub-chunk tile skip pattern.

**Tile classification for sub-chunk $(i, j)$:**

Given sub-chunk query rows $[i \cdot c, (i+1) \cdot c)$ and sub-chunk key columns $[j \cdot c, (j+1) \cdot c)$:

$$
\text{Fully masked} \iff (i+1) \cdot c \leq \max_{j' \in \text{tile}} UTE_{j'} \quad \text{OR} \quad i \cdot c \geq \min_{j' \in \text{tile}} LTE_{j'}
$$

$$
\text{Skip condition:} \quad \text{upper-triangular skip OR cross-document skip}
$$

The cross-document skip is the novel addition: when a document boundary falls between sub-chunk $j$ and sub-chunk $i$ ($j$ is in an earlier document than $i$), the tile is fully zero and can be skipped.

### Tile-Skip Integration with Secondary Chunking

In GLA's secondary chunking (trick 177), the computation already loops over sub-chunk pairs:

```
for i in range(N_s):       # query sub-chunk
    for j in range(i + 1):  # key sub-chunk (causal: j <= i)
```

**Current**: The `j <= i` check handles basic causal skipping.

**Proposed**: Replace with FlashMask tile classification:

```
for i in range(N_s):
    for j in range(N_s):    # iterate all, not just j <= i
        if tile_fully_masked(i, j, LTE_min, LTE_max, UTE_min, UTE_max):
            continue         # skip: no HBM load, no compute
        elif tile_partially_masked(i, j, ...):
            # load mask bounds, apply element-wise within tile
            P_ij = compute_masked_tile(...)
        else:
            # unmasked: standard matmul
            P_ij = compute_tile(...)
```

The overhead of the tile classification check is two integer comparisons per tile — negligible.

### Effect on Inter-Chunk State Update

At document boundaries where $\alpha_t = 0$, the inter-chunk state must also reset:

$$
S_{[n+1]} = \gamma_{[n+1]} \odot S_{[n]} + \tilde{K}_{[n+1]}^\top V_{[n+1]}
$$

When $\gamma_{[n+1]}$ contains zeros (because a document boundary falls in chunk $n+1$), $S_{[n]}$ is naturally zeroed for those dimensions. **No additional kernel modification needed** — the gating mechanism handles state reset automatically.

For chunks that contain a document boundary mid-chunk, the intra-chunk computation becomes block-diagonal. FlashMask's tile-skip handles this: tiles crossing the boundary are classified as "fully masked" and skipped.

### Key Variables

- $C$ — primary chunk size (128)
- $c$ — secondary sub-chunk size (16)
- $N_s = C/c$ — sub-chunks per chunk (8)
- $T$ — total packed sequence length
- $\bar{L}_{\text{doc}}$ — average document length
- $LTS, LTE, UTS, UTE \in \mathbb{R}^T$ — column-sparse mask vectors
- $\rho$ — fraction of tiles that can be skipped

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA 1.3B / Gated DeltaNet 1.3B |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 4$ (GLA), $H = 16$ (Gated DeltaNet) |
| Chunk size | $C = 128$ |
| Sub-chunk size | $c = 16$ |
| GPU | H100 SXM / A100 |

### Training Scenarios

| Scenario | $\bar{L}_{\text{doc}}$ | Expected tile skip $\rho$ | Speedup |
|----------|----------------------|-------------------------|---------|
| Pure causal (baseline) | $\infty$ | 43.8% (upper-tri) | 1.0× (already done) |
| Long documents | 2048 | 45–50% | 1.05–1.1× |
| Mixed short+long docs | 256 | 55–65% | 1.3–1.5× |
| Short documents (SFT) | 64 | 70–80% | 1.6–2.0× |
| Very short (QA pairs) | 16 | 85–90% | 2.0–2.5× |

### Baseline

1. **GLA with separate sequences**: Process each document separately. Full GPU utilization only if documents are similar length. Throughput limited by batch padding.
2. **GLA with packing + no mask**: Pack documents and ignore cross-document attention. Faster but lower quality (cross-document contamination of the state).
3. **GLA with packing + padding to chunk boundary**: Pad each document to nearest chunk boundary. Wastes compute on padding tokens.
4. **FlashAttention + FlashMask (softmax attention baseline)**: Compare the tile-skip speedup in softmax attention to our linear attention version.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intra-chunk kernel speedup (short docs) | $> 1.5\times$ | TFLOPS/s vs no-skip baseline |
| End-to-end training speedup (SFT) | $> 1.3\times$ | Tokens/sec on 1.3B model |
| Mask memory | $O(T)$ | Bytes for mask storage |
| Quality (no contamination) | Match separate-sequence baseline | Perplexity on held-out eval |
| Tile skip overhead | $< 1\%$ | Additional time from mask checks |

### Estimated Compute

**MVE (kernel microbenchmark)**: < 30 minutes on single GPU
- Implement tile-skip logic in GLA Triton kernel
- Generate synthetic document-packed sequences with varying $\bar{L}_{\text{doc}}$
- Measure kernel throughput vs baseline

**Phase 1 (quality validation)**: ~32 GPU-hours
- GLA-370M SFT on packed instruction-tuning data (Alpaca)
- Compare: (a) separate sequences, (b) packing without mask, (c) packing with FlashMask tile-skip
- Measure throughput and final loss

**Phase 2 (full scale)**: ~128 GPU-hours
- GLA-1.3B + Gated DeltaNet-1.3B on packed pretraining data
- Full throughput comparison across document length distributions

## Expected Outcome

**If hypothesis is correct:**

- For pure causal models: marginal improvement (basic causal skip already exists in FLA kernels). The main benefit is for document-packed and variable-mask scenarios.
- For document-packed SFT with $\bar{L}_{\text{doc}} = 64$: $1.6$–$2.0\times$ intra-chunk speedup. With intra-chunk being ~70% of layer time: $1.3$–$1.5\times$ end-to-end.
- Mask memory: 4 vectors $\times T \times 4$ bytes $= 16T$ bytes (for $T = 128$K: 2 MB). Compare to dense: $T^2 \times 2 = 32$ GB. **16,000× reduction**.
- Quality matches separate-sequence baseline exactly (the $\alpha_t = 0$ boundary is exact, not approximate).
- First efficient document-packed training for linear RNNs without padding waste.

**If hypothesis is wrong:**

- **Scenario A**: Tile classification overhead is too high when masks are complex. Many tiles are "partially masked" (few are fully skippable). **What we learn**: Document boundaries don't align well with sub-chunk tile boundaries. **Mitigation**: Align document packing to sub-chunk boundaries (pad to nearest $c = 16$ instead of $C = 128$, much less waste).
- **Scenario B**: The state reset at document boundaries ($\alpha_t = 0$) causes training instability (gradient spikes at boundary). **What we learn**: Hard state resets are problematic for linear RNN training. **Mitigation**: Use soft boundaries ($\alpha_t = \epsilon$ for small $\epsilon$) — FlashMask still works but tiles near boundaries become partially masked.
- **Scenario C**: The speedup doesn't justify implementation complexity for pretraining (where documents are long). **What we learn**: FlashMask tile-skip is mainly useful for fine-tuning/alignment (short documents), not pretraining. This is still valuable — SFT, DPO, and RM are critical training stages.

## Minimum Viable Experiment

### Setup
- **Model**: Single-layer GLA, $d = 256$, $d_k = 128$, $d_v = 256$, 2 heads (~500K params)
- **Task**: Intra-chunk kernel microbenchmark with document-packed synthetic data
- **Data**: Synthetic sequences with controlled document lengths: $\bar{L}_{\text{doc}} \in \{16, 32, 64, 128, 256\}$
- **Compute**: Single A100 or H100, < 10 minutes

### Protocol
1. Implement tile-skip logic in existing GLA Triton intra-chunk kernel
2. Generate packed sequences with varying document lengths
3. For each $\bar{L}_{\text{doc}}$: measure kernel throughput with and without tile-skip
4. Verify numerical correctness: compare output against (a) separate-sequence reference and (b) dense-mask reference

### Success Criteria
- At $\bar{L}_{\text{doc}} = 64$, $C = 128$: kernel throughput $> 1.4\times$ baseline (at least 40% of tiles skipped beyond causal)
- At $\bar{L}_{\text{doc}} = 16$: kernel throughput $> 1.8\times$ baseline
- Tile classification overhead $< 2\%$ of kernel time
- Numerical output matches reference to BF16 precision ($< 10^{-3}$ relative error)
- Mask memory is $O(T)$ (verified by measuring allocation)

### Failure Criteria
- Throughput gain $< 1.1\times$ at $\bar{L}_{\text{doc}} = 64$ → most tiles are partially masked, not fully skippable. Document boundaries don't align with tiles.
- Tile classification takes $> 5\%$ of kernel time → overhead is too high
- Numerical error $> 10^{-2}$ → mask application has a bug or precision issue

### Why This Test Is Sufficient
- The kernel throughput directly measures the benefit. If tile-skip works at the kernel level, the end-to-end training benefit follows mechanically (kernel is 70% of layer time).
- Varying $\bar{L}_{\text{doc}}$ tests the full range of document lengths encountered in practice (pretraining: long docs, SFT: short instructions, DPO: paired short responses).
- Numerical correctness for one chunk validates the mask logic. Multi-chunk correctness comes from the existing inter-chunk recurrence (which handles $\alpha_t = 0$ boundaries natively).

## Memory Access Pattern Analysis

**Tile skip eliminates HBM loads**: When a sub-chunk tile is classified as fully masked, neither $K_j$ nor $V_j$ for that tile are loaded from HBM. For $c = 16$, $d_k = 128$: each skipped tile saves $2 \times 16 \times 128 \times 2 = 8$ KB of HBM read traffic.

**Mask vectors fit in registers**: The precomputed min/max bounds ($LTE\_min_j$, $LTE\_max_j$, etc.) for each sub-chunk are 4 scalars — stored in thread registers. The classification check is two integer comparisons.

**For partially masked tiles**: The column-wise mask vectors $LTE_j$ for the sub-chunk's columns ($c = 16$ values) are loaded from HBM into SRAM alongside $K_j$. The element-wise mask application is a per-element comparison + zero-out — FMA pipeline, not tensor core, but on a small $c \times c$ tile.

**No change to coalesced access pattern**: The tiled loop structure is identical to TFLA/GLA. The tile-skip only adds a `continue` statement in the inner loop — the access pattern for non-skipped tiles is unchanged.

## Parallelism Analysis

**No warp divergence**: Tile classification is per-tile (all threads in a warp process the same tile). The skip/no-skip decision is uniform across the warp.

**SM saturation**: With document packing, more tokens fit in a batch than with padding. This increases the number of independent CTAs (batch × heads × chunks), improving SM utilization. The tile-skip reduces per-CTA work but increases the number of CTAs — net effect is higher GPU utilization.

**Tensor core mapping**: Non-skipped inter-sub-chunk tiles use the same BF16/FP8 WGMMA as before. Partially masked diagonal tiles use element-wise ops on small $c \times c$ matrices. No change to the tensor core utilization of the dominant computation.

## Theoretical Analysis

### Tile skip fraction $\rho$ for document-packed sequences

For a packed sequence with documents of length $L_d$, chunks of size $C$, and sub-chunks of size $c$:

**Within a chunk containing $m$ document boundaries**:
- The $N_s \times N_s$ tile grid has $\binom{N_s}{2}$ lower-triangular tiles
- Of these, tiles crossing a document boundary are fully masked
- For $m$ boundaries at random positions: expected fully-masked tiles $\approx m \times (N_s - 1)$

**Total skip fraction:**

$$
\rho = \underbrace{\frac{N_s(N_s-1)/2}{N_s^2}}_{\text{causal skip}} + \underbrace{\frac{m(N_s-1)}{N_s^2}}_{\text{cross-doc skip}}
$$

For $N_s = 8$ ($C = 128, c = 16$) and $\bar{L}_{\text{doc}} = 64$ (2 documents per chunk, $m \approx 1$):

$$
\rho \approx \frac{28}{64} + \frac{7}{64} = 0.438 + 0.109 = 0.547
$$

So ~55% of tiles are skipped (vs 44% for pure causal), giving $1/(1-0.547) \times (1-0.438) = 1.24\times$ additional speedup over causal-only skip.

For $\bar{L}_{\text{doc}} = 16$ (8 documents per chunk, $m \approx 7$):

$$
\rho \approx 0.438 + \frac{49}{64} = 0.438 + 0.766 = \min(1.0, \text{bounded at } \sim 0.87)
$$

~87% of tiles skipped — dramatic speedup.

### Complexity

| Operation | No skip | Causal skip | FlashMask skip |
|-----------|---------|-------------|---------------|
| Tiles computed per chunk | $N_s^2$ | $N_s(N_s+1)/2$ | $(1-\rho) N_s^2$ |
| For $N_s = 8$, $\rho = 0.55$ | 64 | 36 | **29** |
| HBM loads for $K, V$ | $N_s^2 \times 2cd$ | $36 \times 2cd$ | **$29 \times 2cd$** |
| Mask memory | 0 | 0 | $4T$ bytes |

## Risks & Limitations

1. **Limited benefit for pretraining**: During pretraining with long documents ($\bar{L}_{\text{doc}} > C$), most chunks contain a single document and the tile-skip fraction is just the basic causal skip. The benefit is primarily for fine-tuning, alignment, and short-document regimes. **Acceptable**: SFT/DPO/RM training is compute-intensive and runs on most production models.

2. **Port from PaddlePaddle**: FlashMask's reference implementation is in PaddlePaddle/PaddleNLP. Porting the column-sparse mask logic to Triton/PyTorch requires reimplementing the preprocessing and tile classification. **Mitigation**: The classification logic is simple (integer comparisons), and the column-sparse representation is framework-agnostic.

3. **Interaction with GLA secondary chunking log-space**: GLA's intra-sub-chunk diagonal tiles use log-space computation (trick 177). These tiles are always partially masked (they're on the causal diagonal). FlashMask's classification treats them as "partially masked" — consistent with the existing handling. No special cases needed.

4. **Variable chunk boundaries**: If documents end mid-chunk, the state update within that chunk changes (the gate zeros cross-doc contributions). This is handled natively by the $\alpha_t = 0$ mechanism but may require careful testing at document boundary positions that don't align with sub-chunk boundaries.

5. **Preprocessing cost**: Computing min/max bounds for each tile requires a reduction over column vectors. For $T = 128$K with $c = 16$: $128K / 16 = 8K$ tiles, each needing min/max of 16 values. This is a trivial preprocessing kernel ($< 0.1$ ms), but it's an additional kernel launch. **Mitigation**: Fuse preprocessing into the main kernel's prologue (each CTA computes bounds for its own tiles).

## Follow-up Experiments

1. **FlashMask + FP8/INT4 (proposals 050/054)**: Combine tile skipping with mixed-precision quantization. The two optimizations are orthogonal: tile-skip reduces the number of tiles computed, and FP8/INT4 speeds up each tile. Multiplicative benefit: $1.5\times$ (skip) × $1.5\times$ (FP8) = $2.25\times$.

2. **FlashMask + fused atomic state passing (trick 182)**: Fuse tile-skip logic into the single-launch fused chunkwise kernel. The skip happens at the inner tile loop level — compatible with the atomic inter-chunk synchronization.

3. **Adaptive chunk-boundary document packing**: Instead of packing documents continuously, align document starts to sub-chunk boundaries ($c$-aligned). This maximizes the number of fully-skippable tiles at the cost of small padding gaps. Analyze the compute-waste trade-off.

4. **Sliding-window + document mask**: For hybrid linear RNN + attention architectures, encode both the sliding window pattern and document boundaries in the column-sparse representation. TFLA tiles that fall outside the sliding window AND across documents can be skipped.

5. **Extend to backward pass**: FlashMask's backward pass naturally supports column-parallel tile skipping for $dK, dV$ computation. Implement and benchmark the backward kernel.

6. **Dynamic mask for speculative decoding**: During speculative decoding with draft-verify models, rejected tokens create dynamic mask patterns. FlashMask's $O(T)$ representation enables efficient mask updates without materializing $T^2$ matrices.

## Human Review

(To be filled by reviewer)

## References

- Wang, G., Zeng, J., Xiao, X., Wu, S., et al. (2025). FlashMask: Efficient and Rich Mask Extension of FlashAttention. ICLR 2025. arXiv:2410.01359.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Astra, R., Dao, T., & Hoque, A. (2026). Accelerating Mamba2 with Kernel Fusion. PyTorch Blog.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.

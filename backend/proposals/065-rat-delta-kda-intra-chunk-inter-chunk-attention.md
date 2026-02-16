# RAT-Delta: KDA-Strength Intra-Chunk Recurrence with Inter-Chunk Softmax Attention

**Status**: proposed
**Priority**: high
**Created**: 2026-02-16
**Based on**: tricks/254-rat-intra-chunk-recurrence-inter-chunk-attention, tricks/211-kda-constrained-dplr-delta-chunkwise, tricks/203-gated-deltanet-chunkwise-wy-gating, tricks/177-gla-secondary-chunking-log-space-gating

## Hypothesis

Replacing RAT's simple gated EMA intra-chunk recurrence with a KDA/Gated-DeltaNet-style **delta rule recurrence** (rank-1 state update with per-channel decay) will close the perplexity gap between RAT and full attention (currently ~0.06 ppl at 1.3B) while **preserving RAT's $L\times$ KV cache reduction and $10\times$ throughput advantage**. The delta rule's selective write/overwrite mechanism provides dramatically better information compression within each chunk than EMA's passive exponential averaging, enabling larger chunk sizes ($L = 32$–$64$) without quality degradation — yielding further KV cache and throughput improvements.

## Background

RAT (Wei et al., NeurIPS 2025) achieves a compelling efficiency/quality tradeoff by partitioning sequences into chunks of size $L$, applying a lightweight linear recurrence within each chunk to compress local context, then using softmax attention across chunk-level representations for long-range retrieval. With $L=16$, RAT matches full attention quality while achieving $10\times$ higher throughput.

**However, RAT's intra-chunk recurrence is a simple gated EMA:**
$$\tilde{v}_{c,l} = g_{c,l} \odot \tilde{v}_{c,l-1} + (1 - g_{c,l}) \odot v_{c,l}$$

This is an **element-wise** operation — each dimension of the summary vector evolves independently. It cannot selectively overwrite specific key-value associations; it can only exponentially decay old information and blend in new information. This is the weakest possible recurrence in the GLA/DeltaNet/KDA family.

Meanwhile, KDA (Kimi Linear, 2025) has shown that the **constrained DPLR delta rule** with per-channel decay provides dramatically better state utilization:
$$\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \text{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top$$

The delta rule enables **targeted replacement**: it can erase the old value associated with key $\boldsymbol{k}_t$ and write a new one, rather than just blending. This is fundamentally better for compressing $L$ tokens into a single chunk-level representation.

**The gap this proposal fills:** No one has combined RAT's chunk architecture (intra-chunk recurrence + inter-chunk softmax attention) with a delta-rule-class recurrence. RAT uses EMA; KDA/GDN use delta rule but with chunkwise linear attention (not softmax) for inter-chunk communication. This proposal takes the best of both: delta-rule compression within chunks for maximal information retention, and softmax attention across chunks for precise long-range retrieval.

## Related Work

- **[RAT (Wei et al., NeurIPS 2025)](https://arxiv.org/abs/2507.04416)**: Proposes intra-chunk EMA + inter-chunk softmax attention. Uses the simplest possible recurrence (gated EMA). Does not explore stronger recurrences within chunks.
- **[Gated DeltaNet (Yang et al., ICLR 2025)](https://arxiv.org/abs/2412.06464)**: Develops gated delta rule with chunkwise WY parallelization. Uses linear attention for inter-chunk communication, not softmax.
- **[KDA / Kimi Linear (Zhang et al., 2025)](https://arxiv.org/abs/2510.26692)**: Extends GDN with per-channel decay. Deployed at 48B scale in hybrid with MLA (3:1 ratio). Uses linear attention inter-chunk, not softmax.
- **[Systematic Analysis of Hybrid Linear Attention (2025)](https://arxiv.org/abs/2507.06457)**: Recommends GDN/HGRN-2 backbone with 1:3–1:6 softmax attention layers. Does not consider RAT's chunk-level attention design.
- **Our approach**: Combines KDA's expressive intra-chunk delta rule with RAT's inter-chunk softmax attention architecture. This has not been explored — all prior work uses either (a) weak EMA + softmax attention (RAT) or (b) strong delta rule + linear attention (KDA/GDN).

## Mathematical Formulation

### RAT-Delta Intra-Chunk Recurrence

Replace RAT's element-wise gated EMA with a **matrix-valued delta rule recurrence** within each chunk. For chunk $c$, the intra-chunk state evolves as:

$$\boldsymbol{S}_{c,l} = (\boldsymbol{I} - \beta_{c,l} \boldsymbol{k}_{c,l} \boldsymbol{k}_{c,l}^\top) \, \text{Diag}(\boldsymbol{\alpha}_{c,l}) \, \boldsymbol{S}_{c,l-1} + \beta_{c,l} \boldsymbol{k}_{c,l} \boldsymbol{v}_{c,l}^\top
$$

**Key Variables:**
- $\boldsymbol{S}_{c,l} \in \mathbb{R}^{d_k \times d_v}$ — matrix-valued state within chunk $c$ at position $l$
- $\boldsymbol{k}_{c,l} \in \mathbb{R}^{d_k}$ — L2-normalized key: $\boldsymbol{k}_{c,l} = \frac{W_K x_{c,l}}{\|W_K x_{c,l}\|_2}$
- $\boldsymbol{v}_{c,l} \in \mathbb{R}^{d_v}$ — value projection
- $\boldsymbol{\alpha}_{c,l} \in (0, 1]^{d_k}$ — per-channel decay gate: $\boldsymbol{\alpha}_{c,l} = \sigma(W_\alpha x_{c,l})$
- $\beta_{c,l} \in (0, 1]$ — scalar update strength: $\beta_{c,l} = \sigma(W_\beta x_{c,l})$
- $\boldsymbol{S}_{c,0} = \boldsymbol{0}$ — each chunk starts fresh (no inter-chunk state leakage through recurrence)

### Chunk-Level KV Extraction

Extract chunk-level key and value from the final state:

$$\bar{K}_c = \text{Pool}_K(\boldsymbol{S}_{c,L}) \in \mathbb{R}^{d_k}, \quad \bar{V}_c = \text{Pool}_V(\boldsymbol{S}_{c,L}) \in \mathbb{R}^{d_v}$$

**Option A (Simple):** Use the last-position key and a learned readout:
$$\bar{K}_c = \boldsymbol{k}_{c,L}, \quad \bar{V}_c = \boldsymbol{S}_{c,L}^\top \boldsymbol{k}_{c,L}$$

**Option B (Learned query):** Use a learned readout query $\boldsymbol{q}_{\text{read}} \in \mathbb{R}^{d_k}$:
$$\bar{V}_c = \boldsymbol{S}_{c,L}^\top \boldsymbol{q}_{\text{read}}$$

### Inter-Chunk Softmax Attention (Unchanged from RAT)

For each token $(c, l)$ with query $\boldsymbol{q}_{c,l}$:

$$y_{c,l} = \text{Softmax}\!\left(\frac{\boldsymbol{q}_{c,l}^\top [\bar{K}_{1:c-1}; \bar{K}_{c,l}]}{\sqrt{d_k}}\right) [\bar{V}_{1:c-1}; \bar{V}_{c,l}]$$

where $\bar{K}_{c,l}, \bar{V}_{c,l}$ are intermediate chunk summaries at position $l$ (for causal correctness within the current chunk).

### Comparison: EMA vs Delta Rule Compression

For a chunk of $L=16$ tokens, consider storing a key-value pair $(k_3, v_3)$ that must be retrievable at chunk end:

| Mechanism | State after $L=16$ steps | Retrieval of $v_3$ |
|-----------|--------------------------|---------------------|
| **EMA** ($g = 0.9$) | $\tilde{v} = \sum_{l} (1-g_l) g^{L-l} v_l$ | $\approx 0.9^{13} \cdot 0.1 \cdot v_3 \approx 0.025 v_3$ — **severely attenuated** |
| **Delta Rule** | $\boldsymbol{S} = \ldots + \beta_3 v_3 k_3^\top$ (unless overwritten) | $\boldsymbol{S}^\top k_3 \approx v_3$ — **perfectly retrievable** if not overwritten |

The delta rule's selective overwrite mechanism is fundamentally better suited for compressing chunk content into a fixed-size summary.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | RAT-Delta (hybrid: KDA intra-chunk + softmax inter-chunk) |
| Layers | $L_{\text{model}} = 24$ (1.3B) or $12$ (370M) |
| Hidden dim | $d = 2048$ (1.3B) or $1024$ (370M) |
| Heads | $H = 16$ (1.3B) or $8$ (370M) |
| Head dim | $d_k = d_v = 128$ |
| Chunk size | $L \in \{16, 32, 64\}$ (ablation) |
| Intra-chunk | KDA-style constrained DPLR delta rule |
| Inter-chunk | Softmax attention (FlashAttention) |
| State per chunk | $\boldsymbol{S} \in \mathbb{R}^{d_k \times d_v}$ = $128 \times 128$ per head |

### Baseline

| Model | Inter-chunk | Intra-chunk | Complexity |
|-------|-------------|-------------|------------|
| Full Attention | Softmax over all $T$ tokens | None | $O(T^2 d)$ |
| RAT(L=16) | Softmax over $C = T/16$ chunks | Gated EMA (element-wise) | $O(T^2 d / 16)$ |
| KDA (full) | Linear attention (chunkwise) | Delta rule (DPLR) | $O(T d_k d_v)$ |
| **RAT-Delta(L=16)** | Softmax over $C$ chunks | Delta rule (DPLR) | $O(T^2 d / 16)$ |
| **RAT-Delta(L=64)** | Softmax over $C$ chunks | Delta rule (DPLR) | $O(T^2 d / 64)$ |

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\leq$ RAT(L=16) ppl (7.67 at 1.3B) | WikiText-103, FineWeb-Edu validation |
| Throughput | $\geq 25$K tokens/sec at 1.3B | H100 generation throughput |
| MQAR accuracy | $\geq 90\%$ at 10 KV pairs | Multi-Query Associative Recall |
| NIAH | $\geq$ RAT(L=16) | Needle-in-a-Haystack retrieval |
| Chunk scalability | RAT-Delta(L=64) $\leq$ RAT(L=16) ppl | Perplexity at larger chunk sizes |

### Estimated Compute

- **Phase 1 (MVE)**: ~1 GPU-hour (H100), tiny model, synthetic tasks
- **Phase 2 (370M)**: ~80 GPU-hours, FineWeb-Edu 15B tokens
- **Phase 3 (1.3B)**: ~300 GPU-hours, FineWeb-Edu 100B tokens

Total: ~380 GPU-hours

## Expected Outcome

**If hypothesis is correct:**
- RAT-Delta(L=16) closes the 0.06 ppl gap to full attention (achieves $\leq 7.61$ ppl at 1.3B)
- RAT-Delta(L=32) matches RAT(L=16) quality, yielding $2\times$ further KV cache reduction and $\sim 15\times$ throughput vs full attention
- MQAR accuracy at $L=64$: $\geq 85\%$ (vs. RAT-EMA degrades to $< 60\%$ at $L=64$)
- Delta rule enables larger effective chunk sizes without quality degradation

**If hypothesis is wrong:**
- The delta rule's extra complexity ($d_k \times d_v$ matrix state vs. $d$ vector state) adds latency that offsets the quality benefit at small chunk sizes
- RAT's simple EMA is "good enough" for $L=16$ because the inter-chunk softmax attention handles precise retrieval anyway
- We learn that the quality bottleneck in RAT is the inter-chunk attention, not the intra-chunk compression
- Either way, we get a clean ablation of recurrence strength vs. chunk size

## Minimum Viable Experiment

### Setup
- **Model**: 2 RAT-Delta layers, $d = 64$, $d_k = d_v = 32$, $H = 2$, ~100K params
- **Chunk size**: $L \in \{8, 16, 32\}$
- **Task**: Multi-Query Associative Recall (MQAR) — synthetic task requiring precise key-value storage and retrieval from compressed chunk summaries
- **Data**: 10K synthetic MQAR sequences of length 128–512, with 4–10 KV pairs
- **Compute**: Single H100, < 10 minutes
- **Baselines**: RAT-EMA (same architecture, EMA intra-chunk), Full Attention

### Success Criteria
- RAT-Delta(L=16) achieves $\geq 90\%$ MQAR accuracy where RAT-EMA(L=16) achieves $< 75\%$
- RAT-Delta(L=32) achieves $\geq 85\%$ MQAR accuracy where RAT-EMA(L=32) achieves $< 50\%$
- Delta rule advantage grows with chunk size $L$ (the core claim)

### Failure Criteria
- RAT-Delta(L=16) does not beat RAT-EMA(L=16) on MQAR by $> 5\%$ — the delta rule adds nothing for this task
- RAT-Delta is slower than RAT-EMA by $> 2\times$ — the matrix-valued state is too expensive for small chunks

### Why This Test Is Sufficient
MQAR directly measures the quality of intra-chunk compression: can the chunk summary faithfully represent the KV pairs stored within the chunk? If the delta rule beats EMA at small scale, the advantage will persist (or grow) at larger scales because the mechanism is independent of model width. The inter-chunk softmax attention is a separate, well-understood component.

## GPU Efficiency Analysis

### Memory Access Pattern
- **Intra-chunk delta rule**: The state $\boldsymbol{S}_{c,l} \in \mathbb{R}^{d_k \times d_v}$ per head fits in shared memory ($128 \times 128 \times 2 = 32$ KB, well within H100's 228 KB SMEM). Keys and values stream from HBM in coalesced fashion. The rank-1 update $\beta_t k_t v_t^\top$ is an outer product — maps to tensor core MMA.
- **Inter-chunk attention**: Identical to RAT — FlashAttention on $C$ chunk-level KV pairs. HBM access is coalesced and IO-aware.
- **Arithmetic intensity**: Intra-chunk: $O(d_k \cdot d_v)$ FLOPs per token loaded ($\sim 16K$ FLOPs / $\sim 512$ bytes = 32 FLOPs/byte — compute-bound). Inter-chunk: standard FlashAttention intensity.

### Parallelism Analysis
- **Chunk independence**: All $C$ chunks run their delta rule recurrences independently — full SM saturation for $C \geq 100$. Each chunk's recurrence has depth $O(L)$ (sequential within chunk), but with $L = 16$–$64$ this is very short.
- **Within-chunk parallelism**: For $L \leq 64$, we can use the WY chunkwise representation to parallelize the delta rule within each chunk via batched matmuls. The UT transform for the WY representation is a lower-triangular solve of size $L \times L$ — fully fits in registers.
- **No warp divergence**: All operations (outer product, diagonal scaling, matrix-vector) are regular.
- **Tensor core utilization**: The outer product $k_t v_t^\top$ and the matrix-vector $S_{t-1}^\top q_t$ are natural MMA targets.

### Hardware-Specific Considerations
- **H100 WGMMA**: The $128 \times 128$ state matrix is one MMA tile — optimal tensor core granularity.
- **Shared memory**: $32$ KB state per head × $16$ heads = $512$ KB — requires multi-wave processing across heads (2 waves for H100's 228 KB SMEM), or reduce to 8 heads per SM.
- **Register pressure**: WY representation for $L=16$: $L \times d_k = 16 \times 128 = 2048$ elements per head — $4$ KB in FP16. Fits in register file with 8 heads per SM.

### Kernel Structure (Sketchable in 5 min)
1. **Load** chunk's $L$ tokens of $K, V, \alpha, \beta$ from HBM into SMEM
2. **Compute** WY representation: $W = [k_1, \ldots, k_L]$, $Y$ via UT transform (in-register triangular solve)
3. **Apply** bulk state update: $S_{c,L} = (\text{product of Householder-like reflections}) \times 0 + \text{rank-L outer product}$
4. **Extract** chunk-level $\bar{K}_c, \bar{V}_c$ from $S_{c,L}$
5. **Store** $\bar{K}_c, \bar{V}_c$ to HBM (only $2 \times d$ per chunk — tiny)
6. **Inter-chunk attention**: Call FlashAttention on $\{\bar{K}_c, \bar{V}_c\}_{c=1}^C$

HBM round-trips: 1 read (tokens) + 1 write (chunk KV) per chunk. No intermediate materialization.

### Decision Rule Check
1. **Would I bet $100 this is faster than FlashAttention-2 at 1.3B for long sequences?** YES — RAT already achieves $10\times$ speedup; replacing EMA with delta rule adds $< 20\%$ intra-chunk overhead while potentially enabling $2$–$4\times$ larger chunks.
2. **Can I sketch the CUDA kernel structure in 5 minutes?** YES — see above. Intra-chunk is a small WY-based kernel (FLA already has this); inter-chunk is FlashAttention.
3. **Does it reduce HBM bandwidth or increase compute utilization?** YES — larger chunks ($L=32$–$64$) reduce inter-chunk attention's KV set by $2$–$4\times$. The delta rule is compute-bound (not bandwidth-bound), so it uses tensor cores effectively.

## Theoretical Analysis

### Complexity Comparison

| Operation | Full Attention | RAT-EMA(L) | RAT-Delta(L) |
|-----------|---------------|------------|--------------|
| Intra-chunk forward | — | $O(T \cdot d)$ elem-wise | $O(T \cdot d_k \cdot d_v)$ matmul |
| Inter-chunk forward | $O(T^2 d)$ | $O(T \cdot C \cdot d)$ | $O(T \cdot C \cdot d)$ |
| Total forward | $O(T^2 d)$ | $O(T^2 d / L + T d)$ | $O(T^2 d / L + T d_k d_v)$ |
| KV cache | $O(T d)$ | $O(T d / L)$ | $O(T d / L)$ |
| Generation (per token) | $O(T d)$ | $O(C d)$ | $O(C d + d_k d_v)$ |

**Key insight:** The inter-chunk attention cost $O(T^2 d / L)$ dominates for long sequences. The intra-chunk cost is $O(T d_k d_v)$ vs $O(T d)$ — only $\sim d_k / H \approx 8\times$ more FLOPs per token for the delta rule, but this enables using larger $L$ which **quadratically** reduces the dominant inter-chunk attention cost.

**Crossover analysis:** RAT-Delta(L=64) vs RAT-EMA(L=16): Inter-chunk attention savings = $(T^2 d / 16) - (T^2 d / 64) = 3T^2 d / 64$. Extra intra-chunk cost = $T(d_k d_v - d) = T(16384 - 2048) \approx 14T \cdot 1024$. Breakeven: $3T \cdot 2048 / 64 > 14 \cdot 1024$, i.e., $T > 150$. For any reasonable sequence length, using delta rule with $4\times$ larger chunks is a net win.

## Risks & Limitations

1. **WY representation overhead for small chunks**: The UT transform cost for $L=16$ may negate quality benefits at small chunk sizes. Mitigation: Focus on $L=32$–$64$ where the benefit is clearest.
2. **Chunk-fresh state**: Each chunk starts with $\boldsymbol{S}_{c,0} = 0$ — no state carries over between chunks. This means information can only cross chunks via the inter-chunk attention. If the delta rule's benefit is in long-range state tracking (which crosses chunks), it won't help here. Mitigation: The hypothesis is specifically about better *within-chunk compression*, not cross-chunk memory.
3. **Training stability with L2-normalized keys**: KDA requires L2-normalized keys for stability. This constrains the intra-chunk recurrence but is well-tested at scale (Kimi Linear 48B).
4. **Implementation complexity**: Requires combining FLA's delta rule kernel with RAT's chunk attention framework. Both exist independently; combining them requires modest engineering.
5. **Shared-memory pressure**: $128 \times 128$ state matrix per head in SMEM limits concurrent heads per SM. With 16 heads, need 2+ SM waves — reducing occupancy vs RAT-EMA's element-wise scan.

## Follow-up Experiments

1. **Chunk size scaling law**: Measure quality vs. $L$ for RAT-Delta vs RAT-EMA at 1.3B. Find the largest $L$ where RAT-Delta matches full attention quality.
2. **Hybrid layer patterns**: Alternate RAT-Delta layers (for recall) with pure GDN/KDA layers (for efficiency). Test 3:1 ratio as in Kimi Linear.
3. **Cross-chunk state passing**: Modify RAT-Delta to pass state $\boldsymbol{S}_{c,L}$ to the next chunk (not just chunk-level KV). This creates a hybrid between full KDA and RAT-Delta.
4. **DeltaProduct within chunks**: Use $n_h = 2$–$3$ Householder steps per token within each chunk for even stronger compression (trick 178).
5. **Fused RAT-Delta kernel**: Develop a single CUDA kernel that performs both intra-chunk delta rule and inter-chunk attention without HBM materialization of chunk-level KV.

## Human Review

(To be filled by reviewer)

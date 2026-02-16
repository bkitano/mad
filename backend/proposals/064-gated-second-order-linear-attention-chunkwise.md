# Gated Second-Order Linear Attention with Hardware-Efficient Chunkwise Training

**Status**: proposed
**Priority**: high
**Created**: 2026-02-16
**Based on**: [222-higher-order-linear-attention], [177-gla-secondary-chunking-log-space-gating], [203-gated-deltanet-chunkwise-wy-gating], [225-hgrn2-outer-product-state-expansion], [211-kda-constrained-dplr-delta-chunkwise], [244-diagonal-ssd-sum-of-heads-decomposition]

## Hypothesis

Adding **data-dependent diagonal gating** $\text{Diag}(\alpha_t)$ to Higher-Order Linear Attention's (HLA) second-moment accumulator $S_t^K$ — creating a "Gated HLA" (GHLA) — will improve quality over both first-order GLA and ungated HLA, while admitting a **chunkwise-parallel training algorithm** based on GLA's secondary chunking trick. The data-dependent decay enables the second-order kernel metric $S_t^K$ to forget outdated key correlations, solving HLA's key limitation of unbounded accumulation, while the second-order state provides richer expressivity than first-order GLA — specifically, the ability to learn **data-adaptive attention kernels** rather than fixed feature maps.

## Background

**The expressivity gap in linear attention:** First-order linear attention (including GLA, Mamba-2, RetNet) maintains a state $S_t = \sum_i \gamma_i k_i v_i^\top \in \mathbb{R}^{d_k \times d_v}$ that acts as a fixed linear kernel on queries: $o_t = q_t^\top S_t$. The kernel function $K(q, q') = q^\top S q'$ doesn't adapt to the query distribution — it's determined entirely by past keys.

**HLA's second-order solution:** HLA (trick 222) adds a second-moment key accumulator $S_t^K = \sum_i k_i k_i^\top \in \mathbb{R}^{d_k \times d_k}$ and computes $o_t = q_t^\top S_t^K C_t^{QV}$ where $C_t^{QV} = \sum_i q_i v_i^\top$. The kernel becomes $K_t(q, q') = q^\top S_t^K q'$ — a **data-adaptive polynomial kernel** that changes based on key statistics. This is strictly more expressive: it captures second-order key correlations that first-order attention misses.

**HLA's missing piece — gating:** HLA's paper mentions an optional fixed exponential decay $\gamma \in (0,1)$ but does not explore **data-dependent** gating. Without gating, $S_t^K$ accumulates all past key outer products indefinitely, leading to:
1. The metric $S_t^K$ eventually converges to a fixed covariance, losing sensitivity to recent keys
2. No mechanism to "forget" when context changes (e.g., paragraph boundaries)
3. The cross-summary $G_t$ grows unboundedly, consuming memory and compute

**GLA's gating + HLA's second order = GHLA:** By applying GLA's per-dimension gating $\text{Diag}(\alpha_t)$ to HLA's accumulators, we get:

$$
S_t^K = \text{Diag}(\alpha_t) \, S_{t-1}^K \, \text{Diag}(\alpha_t) + k_t k_t^\top
$$

This is a **gated second-order recurrence** where the gate acts on both sides of the symmetric matrix (preserving symmetry). The GLA secondary chunking trick applies because:
1. The gate $\text{Diag}(\alpha_t)$ is diagonal → cumulative products in log-space are elementwise cumsum
2. Within each chunk, the inter-sub-chunk interactions become standard matmuls (tensor-core friendly)
3. Only the $c \times c$ diagonal sub-chunks require full-precision log-space computation

**Why this hasn't been done:** HLA was published in October 2025 (Zhang, Qin & Gu) and GLA in December 2023 / ICML 2024. Kimi Linear (October 2025) explored DPLR transitions for gated delta rules but didn't explore second-order attention. The combination of HLA's second-order accumulation with GLA's hardware-efficient gating is novel.

## Related Work

- **HLA (Zhang, Qin & Gu, 2025)**: Introduced second-order linear attention with causal correction via cross-summaries. Uses a fixed (non-data-dependent) decay $\gamma$ or no decay. Does not provide a chunkwise algorithm optimized for tensor cores. **Our approach adds data-dependent diagonal gating and derives a chunkwise-parallel training algorithm.**

- **GLA (Yang et al., ICML 2024)**: Data-dependent diagonal gating for first-order linear attention with secondary chunking for tensor core utilization. State is $S_t \in \mathbb{R}^{d_k \times d_v}$. **Our approach extends GLA's gating and chunking to second-order states $S_t^K \in \mathbb{R}^{d_k \times d_k}$.**

- **HGRN2 (Qin et al., 2024)**: Uses outer product $k_t k_t^\top$ for state expansion in gated linear RNNs. However, HGRN2 uses this outer product to expand a *first-order* hidden state, not to accumulate a *second-order kernel metric*. The outer product is computed once per token and used as input gating, not as a running key-key correlation. **Our approach uses the outer product $k_t k_t^\top$ as a running second-moment summary that shapes the attention kernel.**

- **Kimi Linear / KDA (Moonshot AI, 2025)**: Extends Gated DeltaNet with channel-wise gating and DPLR transitions. Uses rank-1 delta rule with chunkwise parallelization. Does not explore second-order attention or key-key correlations. **Our approach is orthogonal: GHLA could be combined with KDA's finer-grained channel gating.**

- **Gated Attention (Qiu et al., NeurIPS 2025 Oral)**: Post-attention sigmoid gating for non-linearity. Operates on attention outputs, not on the recurrence itself. **Complementary to our approach — GHLA's second-order kernel with post-attention gating is a natural combination.**

## Mathematical Formulation

**Standard HLA (Second-Order, with fixed decay):**

$$
S_t^K = \gamma S_{t-1}^K + k_t k_t^\top \in \mathbb{R}^{d_k \times d_k}
$$

$$
C_t^{QV} = \gamma C_{t-1}^{QV} + q_t v_t^\top \in \mathbb{R}^{d_k \times d_v}
$$

$$
G_t = \gamma G_{t-1} + k_t (k_t^\top C_{t-1}^{QV}) \in \mathbb{R}^{d_k \times d_v}
$$

$$
o_t = q_t^\top (S_t^K C_t^{QV} - G_t) \in \mathbb{R}^{d_v} \quad \text{(causal-masked second-order output)}
$$

**Proposed GHLA (Gated Second-Order Linear Attention):**

$$
S_t^K = \text{Diag}(\alpha_t^K) \, S_{t-1}^K \, \text{Diag}(\alpha_t^K) + k_t k_t^\top
$$

$$
C_t^{QV} = \text{Diag}(\alpha_t^C) \, C_{t-1}^{QV} + q_t v_t^\top
$$

$$
G_t = \text{Diag}(\alpha_t^K) \, G_{t-1} \, + \, k_t \left(k_t^\top \text{Diag}(\alpha_t^C) \, C_{t-1}^{QV}\right)
$$

$$
o_t = q_t^\top \left(S_t^K C_t^{QV} - G_t\right)
$$

where $\alpha_t^K, \alpha_t^C \in (0, 1)^{d_k}$ are data-dependent gates:

$$
\alpha_t^K = \sigma\left(\frac{x_t W_{\alpha_1}^K W_{\alpha_2}^K + b_\alpha^K}{\tau}\right)^{1/\tau}, \quad \alpha_t^C = \sigma\left(\frac{x_t W_{\alpha_1}^C W_{\alpha_2}^C + b_\alpha^C}{\tau}\right)^{1/\tau}
$$

**Note on symmetry:** The two-sided gating $\text{Diag}(\alpha) S \text{Diag}(\alpha)$ preserves the symmetry of $S_t^K$ (since $(k_t k_t^\top)^\top = k_t k_t^\top$). This means we can store $S_t^K$ as upper triangular ($d_k(d_k+1)/2$ entries), halving the state memory.

**Chunkwise Parallel Form:**

Within a chunk of size $C$, define cumulative log-gates:

$$
\log B_t^K = \sum_{j=1}^{t} \log \alpha_j^K, \quad \log B_t^C = \sum_{j=1}^{t} \log \alpha_j^C
$$

The intra-chunk second-order attention matrix becomes:

$$
P_{ij}^{(2)} = \sum_{a,b=1}^{d_k} Q_{ia} \left(\sum_{l=j}^{i} K_{la} K_{lb} \exp\left(2(\log B_{il}^K - \log B_{jl}^K)\right)\right) C_{jb}^{QV}
$$

This can be decomposed via secondary chunking into:

1. **Inter-sub-chunk blocks** ($i > j$): Factor out gate scalings as diagonal rescalings, yielding standard matmuls:

$$
P_{[i][j]}^{(2)} = \left(\tilde{Q}_{[i]} \odot \Lambda_{[i]}^K\right) \cdot \left(\sum_{l \in [j]} (K_l K_l^\top) \cdot \text{gate\_factors}\right) \cdot \left(\tilde{C}_{[j]}^{QV} \odot \Gamma_{[j]}^C\right)
$$

The key insight: $\sum_l K_l K_l^\top$ within each sub-chunk is a **batched outer product** that can be accumulated as a matmul: $K_{[j]}^\top K_{[j]} \in \mathbb{R}^{d_k \times d_k}$, which is a $(d_k \times c) \times (c \times d_k)$ GEMM — perfectly tensor-core friendly.

2. **Intra-sub-chunk blocks** ($i = j$): Require log-space full precision, computed as $c \times c$ blocks.

**Associative Scan Operator (inter-chunk):**

The 5-tuple scan from HLA extends with gating:

$$
(\alpha_A, S_A, C_A, m_A, G_A, h_A) \oplus (\alpha_B, S_B, C_B, m_B, G_B, h_B)
$$

$$
= \left(\alpha_A \alpha_B,\; \text{Diag}(\alpha_B)^2 S_A + S_B,\; \text{Diag}(\alpha_B) C_A + C_B,\; \ldots\right)
$$

The scan combines gate propagation with state aggregation. Each scan step involves $d_k \times d_k$ matrix operations — tensor-core-friendly at typical $d_k = 64$–$128$.

**Key Variables:**
- $S_t^K \in \mathbb{R}^{d_k \times d_k}$ — gated second-moment key metric (symmetric)
- $C_t^{QV} \in \mathbb{R}^{d_k \times d_v}$ — gated query-value accumulator
- $G_t \in \mathbb{R}^{d_k \times d_v}$ — gated cross-summary for causal correction
- $\alpha_t^K, \alpha_t^C \in (0,1)^{d_k}$ — data-dependent forget gates for key metric and QV accumulator
- $d_k$ — head key dimension (typically 64–128)
- $d_v$ — head value dimension (typically 64–128)
- $C$ — chunk size, $c$ — sub-chunk size

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GHLA (Gated Second-Order Linear Attention) |
| Layers | $L = 24$ |
| Hidden dim | $d_{\text{model}} = 1536$ |
| Heads | $H = 12$ |
| Key dim | $d_k = 64$ per head |
| Value dim | $d_v = 128$ per head |
| Chunk size | $C = 64$, sub-chunk $c = 16$ |
| State per head | $d_k^2 + d_k d_v + d_k d_v + d_k = d_k(d_k + 2d_v + 1) = 64(64 + 256 + 1) \approx 20.5K$ |
| Total params | ~350M (Phase 1), ~1.3B (Phase 2) |

### Baseline

1. **GLA (first-order, gated)**: $S_t = \text{Diag}(\alpha_t) S_{t-1} + k_t v_t^\top$. State: $d_k \times d_v$. Complexity: $O(T d_k d_v)$. The standard first-order baseline.
2. **HLA (second-order, ungated)**: $S_t^K = S_{t-1}^K + k_t k_t^\top$. State: $d_k^2 + d_k d_v$. Complexity: $O(T(d_k^2 + d_k d_v))$. Tests whether gating adds value.
3. **HLA with fixed decay**: $S_t^K = \gamma S_{t-1}^K + k_t k_t^\top$, $\gamma = 0.99$ fixed. Tests whether data-dependency matters.
4. **Transformer++ (FlashAttention-2)**: The quality ceiling.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $< $ GLA by $\geq 0.3$ pts | WikiText-103 / FineWeb validation |
| Throughput | $\geq 0.7 \times$ GLA | Tokens/sec on H100, same batch/seq |
| Memory | $< 1.5 \times$ GLA | Peak GPU memory per head |
| State-tracking | $> $ GLA, $\approx$ Gated DeltaNet | MQAR / induction head accuracy |
| Recall | $>$ GLA | FDA, SWDE retrieval benchmarks |

### Estimated Compute

- **MVE**: < 10 minutes on single GPU (~100K params)
- **Phase 1** (350M model, 15B tokens): ~200 GPU-hours on H100
- **Phase 2** (1.3B model, 30B tokens, if Phase 1 succeeds): ~600 GPU-hours on H100

## Expected Outcome

**If hypothesis is correct:**
- GHLA outperforms GLA by $\geq 0.3$ perplexity points at 350M scale, demonstrating the value of second-order key correlations with data-dependent gating
- GHLA outperforms ungated HLA on recall tasks, demonstrating that data-dependent forgetting prevents the key metric from converging to a fixed covariance
- The chunkwise training algorithm achieves $\geq 70\%$ of GLA's throughput — the extra $d_k \times d_k$ matmuls per chunk are offset by better quality per parameter
- GHLA particularly excels on tasks requiring **context-dependent similarity** — where "what's relevant" changes over the sequence (e.g., topic shifts, entity disambiguation)
- The learned $\alpha_t^K$ reveals interpretable patterns: low decay (long memory) for stable contexts, high decay (fast forgetting) at topic boundaries

**If hypothesis is wrong:**
- If GHLA matches GLA but not better: second-order correlations don't help in practice — the first-order outer product $k_t v_t^\top$ already captures sufficient structure. This would be an important negative result.
- If ungated HLA matches GHLA: data-dependent gating doesn't matter for the second-order accumulator — a fixed decay suffices. This would simplify the architecture.
- If throughput is $< 0.5 \times$ GLA: the $d_k \times d_k$ state overhead is too expensive for practical use at standard head dimensions. This would motivate exploring GHLA only with very small $d_k$ (e.g., 32).

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GHLA, $d_{\text{model}} = 64$, $d_k = 16$, $d_v = 32$, 2 heads, ~100K params
- **Task**: Multi-Query Associative Recall (MQAR) — synthetic task requiring storing and retrieving key-value associations
- **Data**: 10K synthetic sequences of length 128 with 8 key-value pairs each
- **Baselines**: (1) First-order GLA (same dims), (2) HLA ungated, (3) HLA fixed decay $\gamma = 0.99$
- **Compute**: Single GPU, < 10 minutes

### Success Criteria
- GHLA achieves $> 90\%$ MQAR accuracy at 8 associations where first-order GLA achieves $< 70\%$
- GHLA outperforms ungated HLA by $> 5\%$ accuracy (demonstrating gating value)
- GHLA matches or exceeds HLA with fixed decay (demonstrating data-dependent gating is at least as good)

### Failure Criteria
- If GHLA doesn't beat first-order GLA on MQAR: the second-order accumulation provides no benefit for associative recall. The core mechanism is broken — kill the idea.
- If all second-order variants (GHLA, HLA, HLA-decay) perform identically: the second-order term is not contributing. Check that $S_t^K$ actually evolves during training (not collapsing to identity).

### Why This Test Is Sufficient
- MQAR directly tests the key property: storing multiple key-value associations and selectively retrieving them. The second-order key metric $S_t^K$ should enable more precise query routing than first-order attention.
- At $d_k = 16$, the state $S_t^K$ is only $16 \times 16 = 256$ entries — tiny. If the mechanism works at this scale, scaling to $d_k = 64$ adds capacity, not capability.
- Data-dependent gating matters when different parts of the sequence require different retention policies. The MQAR task naturally has "store" and "retrieve" phases that benefit from adaptive gating.

## Theoretical Analysis

**Complexity comparison (per head, per token):**

| Operation | First-order GLA | GHLA (proposed) |
|-----------|----------------|-----------------|
| State update | $O(d_k d_v)$ | $O(d_k^2 + d_k d_v)$ |
| Output computation | $O(d_k d_v)$ | $O(d_k^2 + d_k d_v)$ |
| State memory | $O(d_k d_v)$ | $O(d_k^2 + 2 d_k d_v + d_k)$ |
| Gate parameters | $O(d_k)$ | $O(2 d_k)$ |

**Crossover analysis:** GHLA adds $O(d_k^2)$ per token. Compared to GLA's $O(d_k d_v)$:
- For $d_k = d_v$: overhead is $\sim 1.5\times$ ($d_k^2$ additional vs. $d_k d_v$ existing)
- For $d_k < d_v$ (common: $d_k = 64, d_v = 128$): overhead is $\sim 1.25\times$

The overhead is bounded and modest — less than the gap between GLA and Gated DeltaNet (which adds $O(C^2 d_k)$ for the UT transform).

**Chunkwise complexity:**

| Component | GLA | GHLA |
|-----------|-----|------|
| Intra-chunk (tensor core) | $O(C^2 d_k)$ | $O(C^2 d_k^2)$ |
| Intra-chunk (log-space) | $O(C c d_k)$ | $O(C c d_k^2)$ |
| Inter-chunk state update | $O(C d_k d_v)$ | $O(C d_k^2 + C d_k d_v)$ |
| Inter-chunk scan | $O(N_c d_k d_v)$ | $O(N_c d_k^2 + N_c d_k d_v)$ |

The extra $d_k$ factor in GHLA is the cost of the second-order key metric computation. At $d_k = 64$, $C = 64$: the intra-chunk second-order matmul is $(C \times d_k) \times (d_k \times C) \times d_k = 64 \times 64 \times 64 \times 64 = 16M$ FLOPs — well within tensor core throughput.

**Memory access pattern:** All new operations are matmuls or elementwise ops:
- $K_{[j]}^\top K_{[j]}$: $(d_k \times c) \times (c \times d_k)$ GEMM — tensor core
- $S_t^K C_t^{QV}$: $(d_k \times d_k) \times (d_k \times d_v)$ GEMM — tensor core
- Diagonal gating: elementwise multiply with broadcast — register ops

**Arithmetic intensity:** The $d_k \times d_k$ matmul for $K^\top K$ has arithmetic intensity $O(d_k)$ FLOPs/byte, which is compute-bound for $d_k \geq 32$.

## Risks & Limitations

1. **Two-sided gating complexity:** The gating $\text{Diag}(\alpha) S \text{Diag}(\alpha)$ requires gate factors on both dimensions of $S_t^K$. In the chunkwise form, this means cumulative gate products appear quadratically in the inter-sub-chunk computation. **Mitigation**: Use the same gate for both sides (we already do — $\alpha_t^K$ applies to both dimensions), so cumulative products are $\exp(2 \log B)$ — just doubling the log-gate.

2. **State size at large $d_k$:** For $d_k = 128$ (as in some Gated DeltaNet configs), $S_t^K$ is $128 \times 128 = 16K$ entries per head. With 16 heads, that's 256K entries ($\sim$512 KB in FP32) for the key metric alone — may not fit in registers. **Mitigation**: Use symmetric packing ($d_k(d_k+1)/2$), multi-query sharing ($S_t^K$ shared across heads when keys are shared), or reduce $d_k$ to 64.

3. **Cross-summary $G_t$ update cost:** The gated cross-summary $G_t$ involves $k_t (k_t^\top \text{Diag}(\alpha_t^C) C_{t-1}^{QV})$ — a rank-1 update of a $d_k \times d_v$ matrix, requiring a matvec ($O(d_k d_v)$) followed by an outer product ($O(d_k d_v)$). This is $2 d_k d_v$ FLOPs, same as ungated HLA.

4. **Scan operator size:** The inter-chunk parallel scan carries a 6-tuple (gate, $S^K$, $C^{QV}$, $m^Q$, $G$, $h$) — the $S^K$ term is $d_k \times d_k$ per chunk boundary. For $N_c = 32$ chunks and $d_k = 64$: 32 × 4K = 128K entries. This is manageable but increases scan communication cost vs. GLA.

5. **Unclear if second-order helps language modeling:** The second-order mechanism captures key-key correlations, which may matter more for retrieval/reasoning than for next-token prediction. The quality gain might be task-specific. **Mitigation**: Test on both language modeling (perplexity) and retrieval (MQAR, FDA, SWDE).

## Follow-up Experiments

1. **GHLA + Delta Rule (Gated Second-Order DeltaNet):** Replace the additive $k_t v_t^\top$ update in $C_t^{QV}$ with a delta rule $\beta_t (v_t - S_t^K C_{t-1}^{QV} q_t) k_t^\top$. This makes the QV accumulator do targeted replacement using the second-order key metric for collision detection.

2. **Multi-head sharing:** Share $S_t^K$ across all heads (since keys are often shared in multi-query attention), computing it once and using it for all heads' output computations. This amortizes the $O(d_k^2)$ cost across $H$ heads.

3. **Third-order extension:** HLA supports third-order attention at $O(d_k^3)$ state cost. For small $d_k = 32$, this is 32K entries — feasible. Gated third-order would capture three-way key interactions.

4. **GHLA + Peri-LN DyT (Proposal 063):** Combine with the normalization-free fused kernel from proposal 063 for maximum throughput.

5. **Kimi Linear + GHLA hybrid:** Use KDA's DPLR transitions for the first-order $C_t^{QV}$ accumulator while using GHLA's gated second-order $S_t^K$ for the key metric. This combines KDA's expressive state transitions with GHLA's adaptive kernel.

## Human Review



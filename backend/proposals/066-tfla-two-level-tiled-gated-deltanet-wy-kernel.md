# TFLA Two-Level Tiled Gated DeltaNet WY Kernel

**Status**: proposed
**Priority**: high
**Created**: 2026-02-16
**Based on**: [158-tfla-two-level-tiled-chunkwise-parallelism], [203-gated-deltanet-chunkwise-wy-gating], [211-kda-constrained-dplr-delta-chunkwise], [177-gla-secondary-chunking-log-space-gating], [212-flashrnn-io-aware-fused-recurrence], [202-thunderkittens-register-tile-lcsf]

## Hypothesis

Extending TFLA's two-level tiling algorithm to Gated DeltaNet's chunkwise WY kernel — by computing the UT/WY transform at the **outer chunk level** ($L = 256$–$512$) and tiling the intra-chunk attention matmuls at the **inner tile level** ($B_{Lhq} = B_{Lkv} = 64$) — will achieve **1.3–1.8× wall-clock training speedup** over the current single-level FLA kernel ($C = 64$) by reducing the number of inter-chunk state materializations by $4$–$8\times$ and increasing arithmetic intensity from $\sim C = 64$ to $\sim L = 256$+ FLOPs/byte. The key insight is that the WY transform's $O(L^2 d_k)$ forward substitution cost is **amortized** by the proportionally larger intra-chunk matmul savings, and the WY factors themselves can be **tiled** across the inner dimension to fit in SRAM.

## Background

### The single-level chunking bottleneck in Gated DeltaNet

Current Gated DeltaNet training (FLA implementation) uses a single-level chunk size of $C = 64$. This means:

1. **$T/C = T/64$ inter-chunk states** materialized in HBM, each of size $d_k \times d_v$ (128 × 128 = 32KB per head in BF16)
2. For $T = 8192$: 128 state tensors per head × 16 heads = 2048 state writes/reads to HBM
3. **Low arithmetic intensity**: Each chunk performs $O(C^2 d)$ FLOPs but loads/stores $O(C \cdot d + d_k \cdot d_v)$ bytes, giving AI $\approx C = 64$ — barely compute-bound on H100 (machine balance $\approx 100$)

TFLA demonstrated on mLSTM that increasing $L$ to 256–4096 via two-level tiling:
- Reduces state materializations by $L/C$ factor
- Increases arithmetic intensity to $\sim L$ FLOPs/byte
- Achieves **2× speedup over Mamba-2** and outperforms FlashAttention-3 at sequences $\geq 4K$

### The WY transform complication

Gated DeltaNet's chunkwise algorithm differs from simple gated linear attention (GLA/mLSTM) in that the intra-chunk computation involves a **UT (upper triangular) transform** — a forward substitution that converts the sequential Householder product into matmul form:

$$
\tilde{\mathbf{w}}_r = \beta_r \left(\mathbf{k}_r - \sum_{i=1}^{r-1} \tilde{\mathbf{w}}_i \left(\frac{\gamma_i}{\gamma_r} \mathbf{k}_i^\top \mathbf{k}_r\right)\right)
$$

This is inherently **sequential** within the chunk — each $\tilde{\mathbf{w}}_r$ depends on all previous $\tilde{\mathbf{w}}_i$. At chunk size $C = 64$, this is a $64 \times 64$ triangular solve over $d_k$-dimensional vectors. At $L = 256$, it becomes $256 \times 256$ — a $16\times$ increase in the sequential component.

### Why TFLA still works for Gated DeltaNet

The critical observation: **the UT transform can be computed hierarchically**.

1. **Inner WY at sub-chunk level** ($C_{\text{inner}} = 64$): Compute WY factors $\tilde{W}_{[j]}, \tilde{U}_{[j]}$ for each sub-chunk $j$ within the outer chunk. This is the existing GDN kernel computation — unchanged.

2. **Sub-chunk state merging**: Accumulate sub-chunk boundary states using matmul-based state update:
$$
S_{[j+1]} = S_{[j]} \cdot F_{[j]} + G_{[j]}
$$
where $F_{[j]} = \gamma_{[j]} (\mathbf{I} - \tilde{W}_{[j]}^\top K_{[j]})$ is the transition and $G_{[j]} = \tilde{U}_{[j]}^\top K_{[j]}$ is the input — both $(d_k \times d_v)$ matmuls.

3. **TFLA tiling of the attention matmuls**: The intra-chunk attention scores $S_{ij} = Q_i K_j^\top \odot D_{ij}$ and outputs $H_i = \sum_j S_{ij} \tilde{U}_j$ are tiled at the inner level, computed in SRAM tiles of size $B_{Lhq} \times B_{Lkv}$.

The net effect: the UT transform is computed at the sub-chunk level (64 × 64, same as before), while the attention matmuls benefit from TFLA's larger effective chunk size. The sub-chunk state merging adds $O(L/C_{\text{inner}} \times d_k \times d_v)$ FLOPs — negligible compared to the $O(L^2 d)$ intra-chunk savings.

### Why this hasn't been done

The TFLA paper (Beck et al., NeurIPS 2025) explicitly notes that DeltaNet adaptation is "non-trivial" due to the UT transform and implemented TFLA only for mLSTM. The key difficulty is that mLSTM uses **scalar gating** (a single value per timestep), while Gated DeltaNet uses **Householder-like transitions** with rank-1 key-key outer products. The hierarchical WY composition step we propose resolves this by keeping the WY computation at the sub-chunk level and only applying TFLA tiling to the subsequent attention matmuls.

## Related Work

- **TFLA (Beck, Pöppel, Lippe & Hochreiter, NeurIPS 2025)**: Introduced two-level tiling for mLSTM/linear attention. 2× faster than Mamba-2 on H100. **Only implemented for scalar-gated linear RNNs (mLSTM, GLA).** Explicitly calls out DeltaNet extension as open work. Our proposal resolves this.

- **Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025)**: Chunkwise WY algorithm for gated delta rule. Achieves ~45 Kt/s at 1.3B on H100 with $C = 64$. **Uses single-level chunking.** Our proposal adds the second level.

- **KDA / Kimi Linear (Moonshot AI, 2025)**: Constrained DPLR variant that eliminates the UT transform entirely by sharing the key vector between low-rank factors. **KDA doesn't need TFLA** — its simplified structure already avoids the UT bottleneck. Our proposal targets the more expressive GDN variant.

- **Proposal 057 (FlashRNN-style fused inter-chunk state)**: Optimizes inter-chunk state passing by keeping state in registers. **Complementary to TFLA** — 057 reduces state HBM traffic; TFLA reduces the number of state materializations. Together, they could compound: fewer state materializations, each faster.

- **Proposal 038 (CTA-swizzled TFLA L2-cache-optimized linear RNN)**: Proposes applying CTA tile swizzling to TFLA for L2 cache locality. **Complementary** — swizzling improves cache hit rates within each TFLA tile, while our proposal extends TFLA to GDN's WY structure.

- **Proposal 062 (fused intra-token DeltaProduct Householder steps)**: Fuses Householder product steps within each token. **Orthogonal** — fuses intra-token computation; our proposal optimizes inter-token tiling.

No prior work found combining TFLA two-level tiling with Gated DeltaNet's WY-based chunkwise algorithm.

## Mathematical Formulation

**Current single-level GDN chunkwise algorithm ($C = 64$):**

For each chunk $[t]$ of size $C$:

1. **UT Transform** (sequential within chunk):
$$
\tilde{W}_{[t]}, \tilde{U}_{[t]} = \text{WY\_Transform}(K_{[t]}, V_{[t]}, \beta_{[t]}, \gamma_{[t]}) \quad O(C^2 d_k)
$$

2. **Intra-chunk attention** (parallel within chunk):
$$
H_{[t]}^{\text{intra}} = \left(\frac{Q_{[t]} K_{[t]}^\top}{\sqrt{d_k}} \odot D_{[t]}\right) \tilde{U}_{[t]} \quad O(C^2 (d_k + d_v))
$$

3. **Inter-chunk state update** (sequential across chunks):
$$
S_{[t+1]} = S_{[t]} \cdot \gamma_{[t]}^C (\mathbf{I} - \tilde{W}_{[t]}^\top K_{[t]}) + \tilde{U}_{[t]}^\top K_{[t]} \quad O(d_k d_v)
$$

4. **Inter-chunk output** (parallel within chunk):
$$
H_{[t]}^{\text{inter}} = Q_{[t]} S_{[t]}^{(\text{scaled})} \quad O(C d_k d_v)
$$

Number of chunks: $N_c = T / C = T / 64$. Total HBM state materializations: $N_c \times d_k \times d_v$.

**Proposed two-level TFLA-GDN ($L = 256$, inner $C = 64$):**

The outer chunk size is $L = 256$ (4× larger). Each outer chunk is subdivided into $L/C = 4$ inner sub-chunks of size $C = 64$.

**Level 1: Inner sub-chunk WY transforms (unchanged)**

For each sub-chunk $j \in [1, L/C]$ within outer chunk $[t]$:

$$
\tilde{W}_{[t,j]}, \tilde{U}_{[t,j]} = \text{WY\_Transform}(K_{[t,j]}, V_{[t,j]}, \beta_{[t,j]}, \gamma_{[t,j]}) \quad O(C^2 d_k)
$$

This produces modified keys and values for each sub-chunk. Total: $(L/C) \times O(C^2 d_k) = O(L \cdot C \cdot d_k)$ — same total FLOPs as before.

**Level 2: TFLA-tiled intra-chunk attention**

The intra-chunk attention of the outer chunk now operates on $L \times L$ implicit attention matrix, but tiled into $(B_{Lhq} \times B_{Lkv})$ blocks:

$$
H_{[t]}[i \cdot B_{Lhq} : (i+1) \cdot B_{Lhq}] = \sum_{j \leq i} \underbrace{S_{ij}^{\text{GDN}}}_{B_{Lhq} \times B_{Lkv}} \tilde{U}_{[t]}[j \cdot B_{Lkv} : (j+1) \cdot B_{Lkv}] + \bar{Q}_i \cdot S_{[t-1]}
$$

where $S_{ij}^{\text{GDN}}$ incorporates both the causal gate mask $D$ and the WY-transformed keys:

$$
S_{ij}^{\text{GDN}} = \frac{Q_i K_j^{\prime\top}}{\sqrt{d_k}} \odot D_{ij}^{\text{gate}}
$$

Here $K_j^\prime$ are the WY-modified keys from the sub-chunk computation. The key insight: **$K_j^\prime$ and $\tilde{U}_j$ are pre-computed at Level 1 and loaded from HBM as needed** — exactly as TFLA loads $K$ and $V$ for standard linear attention.

**Within each tile** (SRAM computation):
$$
S_{ij}^{\text{tile}} = \sum_{b=1}^{d_k / B_{d_q}} Q_i^{(b)} K_j^{\prime(b)\top} \quad \text{(tensor core matmul)}
$$

$$
H_i^{\text{tile}} \mathrel{+}= S_{ij}^{\text{tile}} \odot D_{ij} \cdot \tilde{U}_j \quad \text{(tensor core matmul)}
$$

**Inter-outer-chunk state update:**

$$
S_{[t+1]} = S_{[t]} \cdot \prod_{j=1}^{L/C} F_{[t,j]} + \sum_{j=1}^{L/C} G_{[t,j]} \cdot \prod_{m=j+1}^{L/C} F_{[t,m]}
$$

where $F_{[t,j]} = \gamma_{[t,j]}^C (\mathbf{I} - \tilde{W}_{[t,j]}^\top K_{[t,j]})$ and $G_{[t,j]} = \tilde{U}_{[t,j]}^\top K_{[t,j]}^\prime$.

This is a **mini-scan** of $L/C = 4$ steps with $(d_k \times d_v)$ matmul operations — fast and parallelizable.

**Key Variables:**
- $L$ — outer chunk size (256–512, tunable)
- $C$ — inner sub-chunk size (64, fixed for WY stability)
- $B_{Lhq}, B_{Lkv}$ — TFLA tile sizes (64 or 128)
- $B_{d_q}, B_{d_{hv}}$ — head dimension tile sizes (32 or 64)
- $\tilde{W}_{[t,j]}, \tilde{U}_{[t,j]}$ — WY factors for sub-chunk $j$ within outer chunk $t$
- $F_{[t,j]}, G_{[t,j]}$ — sub-chunk transition and input matrices

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Gated DeltaNet |
| Layers | $L_{\text{model}} = 24$ |
| Hidden dim | $d = 2048$ |
| Head dim | $d_k = 128$, $d_v = 128$ |
| Heads | $H = 16$ |
| Inner sub-chunk | $C = 64$ |
| Outer chunk | $L = 256$ (tunable: 128, 256, 512) |
| TFLA tiles | $B_{Lhq} = B_{Lkv} = 64$, $B_{d_q} = B_{d_{hv}} = 64$ |
| Parameters | ~1.3B |

### Baseline

1. **FLA single-level GDN ($C = 64$)**: Current FLA implementation. $\lceil T/64 \rceil$ inter-chunk states. Kernel throughput ~45 Kt/s at 1.3B on H100.
2. **FLA single-level GDN ($C = 128$)**: Larger chunk baseline. Higher arithmetic intensity but 4× more intra-chunk FLOPs and higher SRAM pressure.
3. **TFLA mLSTM ($L = 256$)**: The mLSTM TFLA baseline from the NX-AI implementation. Shows the ceiling for two-level tiling on a simpler (non-WY) architecture.
4. **GLA with TFLA ($L = 256$)**: GLA is compatible with TFLA natively (no WY transform). Shows the benefit of TFLA on a model in the same family without the WY overhead.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Kernel throughput | $\geq 1.3 \times$ FLA GDN baseline | TFLOPs/s measured via NSight |
| Training throughput | $\geq 1.2 \times$ FLA GDN baseline | Tokens/sec on H100, 1.3B model |
| Memory (states) | $\leq 0.25 \times$ baseline | Peak state buffer memory |
| Bit-exactness | Match FLA GDN outputs | Maximum relative error $< 10^{-5}$ |
| Arithmetic intensity | $\geq 200$ FLOPs/byte | Profiled via NSight |

### Estimated Compute

- **MVE**: < 30 minutes on single H100 (kernel microbenchmark)
- **Phase 1** (kernel development + validation): ~40 GPU-hours (Triton implementation + testing)
- **Phase 2** (end-to-end training benchmark, 1.3B model): ~100 GPU-hours
- **Total**: ~140 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- TFLA-GDN at $L = 256$ achieves 1.3–1.8× kernel throughput over single-level $C = 64$
- The speedup comes from (a) 4× fewer inter-chunk state materializations and (b) higher arithmetic intensity at the tile level
- The WY transform overhead (computed at sub-chunk level) is amortized: it's $O(L \cdot C \cdot d_k)$ vs. the $O(L^2 \cdot d)$ intra-chunk attention savings
- End-to-end training throughput improves by 1.2× at 1.3B scale (the chunkwise kernel is 40–60% of layer time)
- The optimal $L^*$ for GDN is smaller than for mLSTM (due to WY overhead) but still significantly larger than $C = 64$

**If hypothesis is wrong:**
- If WY overhead dominates: the $O(L \cdot C \cdot d_k)$ sub-chunk WY computation + $(L/C)$-step mini-scan overhead exceeds the attention savings. This would tell us that GDN's WY structure fundamentally limits the effective chunk size. **Implication**: Use KDA (which eliminates WY) for architectures that need large chunk sizes.
- If register pressure prevents TFLA tiling: the WY factors ($\tilde{W}, \tilde{U}$) consume registers that would otherwise hold TFLA tile accumulators. This would motivate a hybrid approach where WY factors are staged through shared memory instead of registers.
- If the mini-scan for sub-chunk state merging introduces numerical instability: cumulative Householder products across $L/C$ sub-chunks may accumulate error. This would require re-orthogonalization at sub-chunk boundaries.

## Minimum Viable Experiment

### Setup
- **Kernel microbenchmark**: Compare single-level GDN kernel ($C = 64$) vs. TFLA-GDN ($L = 256, C_{\text{inner}} = 64$) on synthetic Q, K, V, gate tensors
- **Dimensions**: $B = 4$, $T = 8192$, $d_k = d_v = 128$, $H = 16$
- **Implementation**: Triton kernel extending FLA's existing GDN chunkwise implementation
- **Metrics**: Wall-clock kernel time, peak SRAM usage, output correctness vs. reference
- **Compute**: Single H100, < 30 minutes

### Success Criteria
- TFLA-GDN kernel is $\geq 1.2\times$ faster than single-level GDN kernel at $T = 8192$
- Output matches single-level GDN within relative error $< 10^{-4}$ (numerical equivalence)
- SRAM usage fits within 200KB (H100 shared memory budget per SM)
- The speedup increases with sequence length (1.2× at 4K, 1.5× at 16K, 1.8× at 32K)

### Failure Criteria
- If TFLA-GDN is slower than single-level at any $T$: the WY overhead or mini-scan serialization dominates. Try reducing $L$ to 128 before killing.
- If numerical error exceeds $10^{-3}$: the hierarchical WY composition introduces precision loss. Try FP32 for the mini-scan before killing.
- If SRAM exceeds 200KB: the WY factors plus TFLA tile accumulators don't fit. Reduce tile sizes before killing.

### Why This Test Is Sufficient
- The kernel microbenchmark directly measures the wall-clock benefit of two-level tiling, which is the core claim.
- If the kernel is faster at $T = 8192$, the benefit at longer sequences will only increase (more inter-chunk states eliminated).
- Numerical correctness at the kernel level guarantees correctness at the model level — the kernel is a drop-in replacement.
- SRAM fitting is the hard constraint; if it passes, the engineering risk is resolved.

## Memory Access Pattern Analysis

**Single-level GDN ($C = 64$, $T = 8192$):**

| Operation | HBM Traffic | Count |
|-----------|-------------|-------|
| Load Q, K, V per chunk | $3 \times C \times d \times 2$ B = 48 KB | $T/C = 128$ |
| Load gate/beta per chunk | $2 \times C \times 2$ B = 256 B | 128 |
| Write/read state $S$ | $d_k \times d_v \times 2$ B = 32 KB | 128 |
| Write output $H$ per chunk | $C \times d_v \times 2$ B = 16 KB | 128 |
| **Total HBM traffic** | **~12.3 MB per head** | |

**TFLA-GDN ($L = 256$, $C_{\text{inner}} = 64$, $T = 8192$):**

| Operation | HBM Traffic | Count |
|-----------|-------------|-------|
| Load Q, K, V per outer chunk | $3 \times L \times d \times 2$ B = 192 KB | $T/L = 32$ |
| Load gate/beta per outer chunk | $2 \times L \times 2$ B = 1 KB | 32 |
| Write/read state $S$ | $d_k \times d_v \times 2$ B = 32 KB | 32 |
| Write output $H$ per outer chunk | $L \times d_v \times 2$ B = 64 KB | 32 |
| WY factors (inner) | Computed in SRAM, not materialized | — |
| **Total HBM traffic** | **~9.2 MB per head** | |

**State HBM reduction**: 128 → 32 state writes/reads = **4× fewer state materializations**.

**Arithmetic intensity improvement**:

Single-level: $\text{AI} = \frac{2 C^2 d_k + 2 C d_k d_v}{3 C d \cdot 2 + 2 d_k d_v \cdot 2} \approx \frac{2 \times 64^2 \times 128}{3 \times 64 \times 256 \times 2 + 2 \times 128^2 \times 2} \approx 64$

TFLA: $\text{AI} = \frac{2 L^2 d_k}{3 L d \cdot 2 + 2 d_k d_v \cdot 2} \approx \frac{2 \times 256^2 \times 128}{3 \times 256 \times 256 \times 2 + 2 \times 128^2 \times 2} \approx 192$

**3× higher arithmetic intensity** → transitions from memory-bound to firmly compute-bound on H100.

## Parallelism Analysis

- **Outer chunks** ($T/L$): Fully independent, parallel across GPU thread block clusters
- **TFLA tiles within outer chunk** ($L/B_{Lhq} \times d_v/B_{d_{hv}}$): Fully independent, parallel across thread blocks. For $L = 256$, $B_{Lhq} = 64$: 4 tile rows × $d_v/B_{d_{hv}}$ = 4 × 2 = 8 thread blocks per outer chunk
- **Inner WY transforms** ($L/C$ sub-chunks): Independent across sub-chunks within outer chunk — parallel. Each sub-chunk's WY is $C \times C = 64 \times 64$ triangular solve on $d_k$-dim vectors.
- **Mini-scan** ($L/C = 4$ steps): Sequential, but only 4 matmul steps of size $d_k \times d_v$ — negligible latency ($\sim 4 \times 32\text{KB} = 128\text{KB}$ of compute, $< 1\mu s$ on H100)
- **Tensor core utilization**: All tile matmuls are $(B_{Lhq} \times B_{d_q}) \times (B_{d_q} \times B_{Lkv})$ = $(64 \times 64) \times (64 \times 64)$ — ideal for WGMMA on H100
- **No warp divergence**: All tile computations are regular matmuls with optional causal masking (handled by tile-level skip, same as FlashAttention)

## Theoretical Analysis

**FLOP comparison per head per layer:**

| Component | Single-level ($C = 64$) | TFLA ($L = 256$, $C = 64$) |
|-----------|------------------------|---------------------------|
| WY transform | $\frac{T}{C} \times C^2 d_k = T C d_k$ | $\frac{T}{C} \times C^2 d_k = T C d_k$ (same!) |
| Intra-chunk QK | $\frac{T}{C} \times C^2 d_k = T C d_k$ | $\frac{T}{L} \times L^2 d_k = T L d_k$ ($4\times$ more) |
| Intra-chunk SV | $\frac{T}{C} \times C^2 d_v = T C d_v$ | $\frac{T}{L} \times L^2 d_v = T L d_v$ ($4\times$ more) |
| Inter-chunk state | $\frac{T}{C} \times d_k d_v$ | $\frac{T}{L} \times d_k d_v$ ($4\times$ fewer) |
| Mini-scan (new) | — | $\frac{T}{L} \times \frac{L}{C} \times d_k d_v$ |
| **Total** | $T(2Cd + d_k d_v / C)$ | $T(Cd_k + 2Ld + d_k d_v / L)$ |

For $C = 64$, $L = 256$, $d_k = d_v = 128$:
- Single-level: $T(2 \times 64 \times 256 + 128^2 / 64) = T \times 33,024$
- TFLA: $T(64 \times 128 + 2 \times 256 \times 256 + 128^2 / 256) = T \times 139,328$

**TFLA does more FLOPs** ($4.2\times$) but with **3× higher arithmetic intensity**, meaning it's **more efficient on real hardware** despite more total compute. This is the same tradeoff as FlashAttention vs. standard attention: more FLOPs but fewer HBM accesses = faster.

**Crossover analysis:**

TFLA-GDN is faster when the HBM bandwidth savings exceed the extra compute cost:

$$
\frac{T_{\text{saved\_HBM}}}{B_{\text{HBM}}} > \frac{\Delta \text{FLOPs}}{P_{\text{compute}}}
$$

For H100: $B_{\text{HBM}} = 3.35$ TB/s, $P_{\text{compute}} = 989$ TFLOPS (BF16 tensor core). With HBM savings of $\sim 3$ MB/head and extra FLOPs of $\sim 100K \times T$/head:

$$
\frac{3 \times 10^6}{3.35 \times 10^{12}} \approx 0.9 \mu s > \frac{100K \times T}{989 \times 10^{12}} \approx 0.8 \mu s \quad \text{(for } T = 8192\text{)}
$$

The crossover is around $T \approx 4K$ — **TFLA-GDN is beneficial for all practical pretraining sequence lengths**.

## Risks & Limitations

1. **WY transform at sub-chunk level still sequential**: The UT transform within each sub-chunk ($C = 64$) remains a triangular solve. At $L = 256$ with 4 sub-chunks, the total sequential WY work is unchanged vs. single-level — it's just partitioned. **Not a risk — same total sequential work, just different scheduling.**

2. **Mini-scan numerical stability**: The 4-step mini-scan composes WY transition matrices $F_{[t,j]}$. Each $F_{[t,j]}$ is a scaled Householder product (orthogonal-like), so the composition should be well-conditioned. **Mitigation**: Monitor condition number of $\prod F_{[t,j]}$ during training; add QR re-orthogonalization if it exceeds $10^3$.

3. **SRAM pressure from WY factors + TFLA tiles**: Each TFLA tile needs $B_{Lhq} \times B_{Lkv} \times 2 = 8$KB for the attention tile plus $B_{Lhq} \times B_{d_{hv}} \times 2 = 8$KB for the output accumulator. The WY factors for the current sub-chunk need $C \times d_k \times 2 = 16$KB. Total: ~32KB — well within H100's 228KB shared memory. **Low risk.**

4. **Triton implementation complexity**: The two-level structure requires nested loops with different parallelization strategies. Level 1 (WY) uses sub-chunk parallelism; Level 2 (TFLA) uses tile parallelism. This requires careful Triton kernel design with multiple `tl.program_id` axes. **Mitigation**: Start with the NX-AI mlstm_kernels TFLA implementation as a template and add the WY sub-chunk computation.

5. **Backward pass complexity**: The TFLA backward pass already requires 4 separate kernels (Table 1 in TFLA paper). Adding the WY transform creates 2 additional backward kernels for WY gradient computation. **Mitigation**: Implement forward-only first (sufficient for throughput benchmarking), then add backward kernels incrementally.

6. **Diminishing returns at very large $L$**: As $L$ increases, the $O(L^2)$ intra-chunk FLOPs eventually dominate. The TFLA paper finds optimal $L^* \in [128, 256]$ for mLSTM on H100. For GDN, the WY overhead shifts the optimal $L^*$ lower — we predict $L^* \in [128, 256]$ (lower end). **Mitigation**: Sweep $L \in \{128, 192, 256, 384, 512\}$ to find the optimum empirically.

## Follow-up Experiments

1. **TFLA-KDA**: KDA eliminates the WY transform entirely, making it a cleaner target for TFLA. The constrained DPLR structure means the intra-chunk computation is simpler — just gated matmuls with channel-wise decay. TFLA-KDA should achieve the full 2× speedup seen for mLSTM.

2. **TFLA-GDN + CTA swizzling (Proposal 038)**: Apply CTA tile rasterization swizzling to the TFLA tile grid to improve L2 cache hit rates. This addresses the secondary memory bottleneck after HBM state materialization.

3. **TFLA-GDN + FlashRNN inter-chunk state (Proposal 057)**: Keep inter-chunk states in registers across the mini-scan. Since the mini-scan is only 4 steps with $d_k \times d_v$ state, this should eliminate the remaining state HBM traffic.

4. **Warp specialization for WY + TFLA pipelining**: Use H100's warp specialization (trick 141) to overlap WY computation (producer warps) with TFLA tile consumption (consumer warps). This would hide the WY latency behind the attention computation.

5. **mLSTMsig → GDN variant**: TFLA's mLSTMsig variant (sigmoid gate instead of exponential) is simpler because it avoids max-state tracking. GDN already uses sigmoid gating — this suggests a natural simplification where GDN's gate computation matches mLSTMsig's structure, enabling direct reuse of the mLSTMsig TFLA kernel for the gating component.

## Human Review



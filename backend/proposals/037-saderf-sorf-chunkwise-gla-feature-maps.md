---
status: ongoing
priority: high
created: 2026-02-15
based_on: dense-exponential-random-features-favor-sharp, structured-orthogonal-random-features-sorf, rfa-gated-random-feature-attention, chunkwise-parallel-scan, tfla-two-level-tiled-chunkwise-parallelism, cosine-reweighted-linear-attention, input-dependent-gating
experiment_number: 037
experiment_log: experiment-log-037.md
---

# SADERF-SORF Variance-Reduced Feature Maps for Chunkwise GLA

## Hypothesis

Replacing GLA's identity feature map $\phi(x) = x$ (which captures only linear kernel $q^\top k$) with **SADERF-calibrated positive random features** projected via **SORF** (Walsh-Hadamard) will approximate the softmax kernel within GLA's chunkwise framework, giving **3–8% perplexity improvement** over standard GLA at **$< 5\%$ throughput overhead** by enabling the intra-chunk attention to approximate softmax-quality interactions while preserving the $O(Td^2)$ linear recurrence between chunks. The SORF projection reduces the feature map cost from $O(Md)$ to $O(M \log d)$ per token, keeping the feature map computation negligible relative to the chunk matmuls.

## Background

### The feature map quality gap in GLA

Gated Linear Attention (Yang et al., 2024) computes attention as:

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}, \quad S_t = G_t S_{t-1} + \phi(k_t) v_t^\top
$$

where $G_t$ is an input-dependent diagonal decay gate. The standard GLA uses $\phi(x) = x$ (identity), which means the intra-chunk attention computes $QK^\top$ — a **linear kernel**, not a softmax kernel. This is a fundamental quality limitation: the linear kernel cannot produce the sharp, sparse attention patterns that softmax generates, leading to ~2–5% perplexity gap vs. softmax Transformers at matched scale.

**ReGLA** (Lu et al., NAACL 2025) explored alternative feature maps for GLA (elu+1, ReLU, identity+elu) and showed they significantly impact quality. However, ReGLA did not test **positive random features** (FAVOR+-style) or **variance-reduced** variants (FAVOR#/SADERF), which provide the tightest known approximation to the softmax kernel.

### Why SADERF + SORF is the right combination

1. **SADERF** (Likhosherstov et al., 2023) achieves $e^{10}\times$ variance reduction over FAVOR+ by optimally calibrating diagonal rescaling of queries and keys. The rescaling $\Psi^*_{l,l} = (\sum_j k_l^{(j)2} / \sum_i q_l^{(i)2})^{1/4}$ is computable in $O(dL)$ and only requires per-dimension statistics — no eigendecomposition. This makes it GPU-friendly.

2. **SORF** (Yu et al., 2016) replaces the dense $M \times d$ random projection with $HD_1HD_2HD_3$ (Walsh-Hadamard + sign flips), reducing projection cost from $O(Md)$ to $O(M \log d)$. For $d = 64$ and $M = 64$, this is a $\sim 10\times$ reduction in feature map FLOPs.

3. **Chunkwise compatibility**: In GLA's chunkwise formulation, the feature map $\phi$ is applied to every token's $q_t$ and $k_t$. The intra-chunk attention becomes $\phi(Q_j)\phi(K_j)^\top$ instead of $Q_j K_j^\top$. If $\phi$ approximates the softmax kernel well, the intra-chunk computation inherits softmax-quality attention patterns while the inter-chunk recurrence preserves the $O(nd^2)$ state propagation.

4. **TFLA synergy**: TFLA's two-level tiling decouples chunk size from SRAM, allowing chunks of $C = 256$–$1024$. Larger chunks mean the intra-chunk attention quality matters more (tokens interact within-chunk rather than across the lossy recurrence boundary). Higher-quality feature maps thus have outsized impact with TFLA.

### What's different from existing proposals

- **Proposal 029** (Circulant FAVOR+): Uses circulant FFT-based projection — a different structured projection than SORF. Circulant provides $O(d \log d)$ projection but requires FFT (complex arithmetic, less tensor-core-friendly). SORF uses Walsh-Hadamard which is real-valued and cache-friendly.
- **Proposal 036** (Near-Far Field GLA): Decomposes intra-chunk into banded + low-rank. Uses simple elu+1 features for the far-field. Our proposal improves the feature map quality itself rather than decomposing the attention.
- **Proposal 009** (Post-Sigmoid Gating): Adds gating at readout. Orthogonal — can be composed with our feature map improvement.

## Related Work

- **ReGLA** (Lu et al., NAACL 2025): Explored feature maps (elu+1, ReLU, identity+elu) for GLA but did NOT test positive random features or variance-reduced variants. Our approach uses SADERF, which is provably optimal for softmax kernel approximation.
- **FAVOR#** (Likhosherstov et al., 2023): Proposed SADERF for Performer-style attention but did NOT apply it to gated linear attention or chunkwise training. Our approach integrates SADERF into GLA's chunkwise framework.
- **Spectraformer** (Chen et al., 2024): Unified framework for random feature attention. Found OPRF-FastFoodL outperforms SADERF-ORF on some tasks. However, Spectraformer did not test within chunkwise GLA or with SORF projection.
- **TFLA** (Beck et al., NeurIPS 2025): Two-level tiling for linear RNNs. Uses identity feature map. Our approach upgrades the feature map quality.

**Gap**: No existing work combines variance-reduced random features (SADERF) with structured fast projection (SORF) within a chunkwise gated linear attention framework (GLA/TFLA).

## Mathematical Formulation

### Standard GLA Intra-Chunk Attention

Within chunk $j$ of size $C$:

$$
O_j = \underbrace{(Q_j K_j^\top \odot M_j)}_{\text{linear kernel}} V_j \quad \text{— } O(C^2 d) \text{ per chunk}
$$

where $M_j$ is the causal decay mask.

### Proposed: SADERF-SORF GLA Intra-Chunk Attention

$$
O_j = \underbrace{(\phi_{\text{SADERF}}(Q_j) \phi_{\text{SADERF}}(K_j)^\top \odot M_j)}_{\text{approx-softmax kernel}} V_j
$$

where the feature map $\phi_{\text{SADERF}}: \mathbb{R}^d \to \mathbb{R}^M$ is:

$$
\phi_{\text{SADERF}}(x) = \frac{D}{\sqrt{M}} \exp\left(W_{\text{SORF}} (\Psi x) \cdot B_{\text{scalar}} - \|\Psi x\|^2 / 2\right)
$$

with:
- $\Psi \in \mathbb{R}^{d \times d}$ — diagonal rescaling, $\Psi_{ll} = (\sum_j k_l^{(j)2} / \sum_i q_l^{(i)2})^{1/4}$
- $W_{\text{SORF}} = \sqrt{d} \cdot H D_1 H D_2 H D_3$ — SORF projection, applied in $O(d \log d)$
- $B_{\text{scalar}} = \sqrt{1 - 4A}$ — SADERF scalar from optimal $A$
- $D = (1 - 4A)^{d/4}$ — normalization constant
- $A = \frac{1}{16}(1 - 2\bar\phi - \sqrt{(2\bar\phi + 1)^2 + 8\bar\phi})$ — optimal parameter

For keys: $\phi_{\text{SADERF}}(k) = \frac{D}{\sqrt{M}} \exp(W_{\text{SORF}} (\Psi^{-1} k) \cdot B_{\text{scalar}} - \|\Psi^{-1} k\|^2 / 2)$

### Inter-Chunk Recurrence (unchanged)

$$
S_j = G_j^{(C)} S_{j-1} + \sum_{t \in \text{chunk}_j} G_j^{(C-t)} \phi(k_t) v_t^\top
$$

The inter-chunk state $S_j \in \mathbb{R}^{M \times d}$ now has dimension $M \times d$ instead of $d \times d$. With $M = d$, the state size is unchanged. With $M = 2d$, state doubles but the approximation quality improves.

### SADERF Parameter Computation

Per-chunk statistics (amortized over $C$ tokens):

$$
\bar{q}_l^2 = \frac{1}{C} \sum_{t=1}^{C} q_{t,l}^2, \quad \bar{k}_l^2 = \frac{1}{C} \sum_{t=1}^{C} k_{t,l}^2
$$

$$
\Psi_{ll} = \left(\frac{\bar{k}_l^2}{\bar{q}_l^2 + \epsilon}\right)^{1/4}
$$

Cost: $O(Cd)$ per chunk — negligible relative to chunk matmuls.

### Key Variables

- $x_t \in \mathbb{R}^{d_{\text{model}}}$ — input at position $t$
- $q_t, k_t, v_t \in \mathbb{R}^d$ — query, key, value (per head)
- $M$ — number of random features ($M = d$ or $M = 2d$)
- $C$ — chunk size (64–1024)
- $\Psi \in \mathbb{R}^{d \times d}$ — SADERF diagonal rescaling
- $D_1, D_2, D_3 \in \{-1, +1\}^d$ — SORF sign-flip vectors (fixed per head)
- $H \in \mathbb{R}^{d \times d}$ — Walsh-Hadamard matrix (implicit, applied via FWHT)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA Transformer (Yang et al., 2024) |
| Feature map | SADERF + SORF (proposed) |
| Chunk algorithm | TFLA two-level tiling |
| Layers | $L = 12$ |
| Hidden dim | $d_{\text{model}} = 768$ |
| Heads | $H = 12$ |
| Head dim | $d = 64$ |
| Random features | $M = 64$ (= $d$) and $M = 128$ (= $2d$) |
| Chunk size | $C = 64, 128, 256$ |

### Baseline

1. **GLA** (identity $\phi$): $O(C^2 d + Td^2)$ — standard chunkwise GLA
2. **GLA + elu+1**: $O(C^2 d + Td^2)$ — ReGLA's best feature map
3. **GLA + FAVOR+** (dense ORF): $O(CMd + TMd)$ — FAVOR+ features in GLA
4. **Softmax Transformer + FlashAttention-2**: $O(T^2 d)$ — quality upper bound

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\geq 3\%$ reduction vs. GLA baseline | WikiText-103, 125M params |
| Feature map throughput | $> 90\%$ of identity $\phi$ throughput | Tokens/sec on A100 |
| Kernel approx. quality | MSE $< 0.01$ vs. softmax kernel | $\|\hat{K} - K_{\text{softmax}}\|_F / \|K_{\text{softmax}}\|_F$ |
| MQAR accuracy | $> 80\%$ at 4 KV pairs | Multi-Query Associative Recall |
| Total training throughput | $> 95\%$ of GLA baseline | End-to-end tokens/sec |

### Estimated Compute

- **MVE**: ~8 minutes on single A100 (~$0.50)
- **Small-scale ablation**: 4 GPU-hours on A100 (~$16)
- **Full-scale**: 24 GPU-hours on A100 (~$100)

## Expected Outcome

**If hypothesis is correct:**

1. **Perplexity**: SADERF-SORF GLA achieves 3–8% lower perplexity than identity-$\phi$ GLA at 125M params, narrowing the gap with softmax Transformers by ~50%.

2. **Feature map cost**: SORF projection at $d = 64$ costs $\sim 3 \times 64 \times 6 = 1152$ FLOPs per token (3 FWHT passes), vs. $64 \times 64 = 4096$ FLOPs for dense projection. Net overhead vs. identity is ~$2\%$ of total layer cost.

3. **Scaling with chunk size**: The benefit of SADERF-SORF grows with chunk size $C$ because more tokens interact via the improved intra-chunk attention. At $C = 256$ with TFLA, we expect $5$–$8\%$ perplexity improvement vs. $3$–$5\%$ at $C = 64$.

4. **MQAR recall**: Softmax-approximating features enable sharper key retrieval, improving MQAR accuracy by $10$–$20\%$ over identity features at $d = 64$.

**If hypothesis is wrong:**

- **Scenario A**: SADERF features don't help over identity
  - **Learn**: The quality gap in GLA is NOT in the kernel type but in the state capacity or gating. The linear kernel $q^\top k$ is sufficient given enough state dimensions.
  - **Next**: Focus on state-level improvements (proposals 006, 022) rather than feature maps.

- **Scenario B**: SADERF helps but SORF degrades it
  - **Learn**: SORF's $O(1/\sqrt{d})$ bias matters at $d = 64$. Dense ORF needed.
  - **Next**: Use dense ORF with SADERF (slower but higher quality).

- **Scenario C**: Good approximation quality but no perplexity gain
  - **Learn**: GLA's gating mechanism already compensates for linear kernel limitations. The gate learns to sharpen attention patterns even with a weak kernel.
  - **Insight**: Gating > kernel quality for practical performance.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA with SADERF-SORF features ($d_{\text{model}} = 64$, $H = 4$, $d = 16$, ~80K params)
- **Feature dims**: $M = 16$ (= $d$) and $M = 32$ (= $2d$)
- **Task**: Multi-Query Associative Recall (MQAR) — 4 KV pairs, sequence length $T = 64$, vocab size 16
- **Data**: 10K synthetic sequences
- **Compute**: Single GPU, $< 8$ minutes
- **Baselines**: GLA with identity $\phi$, GLA with elu+1 $\phi$, GLA with FAVOR+ (dense)

### Success Criteria
- SADERF-SORF GLA achieves $> 80\%$ MQAR accuracy at 4 KV pairs with $d = 16$
- Identity-$\phi$ GLA achieves $< 60\%$ on the same task
- SADERF-SORF matches or exceeds dense FAVOR+ quality (within $3\%$)
- Wall-clock time overhead of SADERF-SORF vs. identity is $< 10\%$
- Results consistent across 3 random seeds

### Failure Criteria
- SADERF-SORF GLA performs within $5\%$ of identity-$\phi$ GLA — feature map quality doesn't matter for GLA
- SADERF-SORF is $> 20\%$ slower than identity — SORF overhead too large at $d = 16$
- Training instability (NaN/Inf) from exponential feature map

### Why This Test Is Sufficient
- **MQAR stresses kernel quality**: Retrieving specific stored associations requires distinguishing between similar keys, which is precisely where softmax-approximating features excel over linear kernels. If SADERF-SORF helps here, it generalizes to language modeling where sharp attention is beneficial.
- **Small $d = 16$ maximizes the quality gap**: With tiny head dimension, the linear kernel $q^\top k$ has very limited discriminative power, making the softmax approximation's benefit most visible.
- **Chunkwise structure preserved**: Even at small scale, the GLA uses chunks with intra-chunk attention + inter-chunk recurrence, so the feature map's role is architecturally identical to the full-scale model.

## Theoretical Analysis

### Complexity Comparison

| Operation | GLA (identity $\phi$) | GLA + SADERF-SORF | GLA + FAVOR+ (dense) |
|-----------|----------------------|-------------------|---------------------|
| Feature map | $O(0)$ | $O(TM \log d)$ | $O(TMd)$ |
| Intra-chunk attn | $O(TC^2 d / C) = O(TCd)$ | $O(TCMd/C) = O(TMd)$ | $O(TMd)$ |
| Inter-chunk state | $O(Td^2)$ | $O(TMd)$ | $O(TMd)$ |
| SADERF calibration | $O(0)$ | $O(Td)$ | $O(0)$ |
| **Total** | $O(T(Cd + d^2))$ | $O(T(Md + M\log d))$ | $O(T(Md + Md))$ |

With $M = d$ and $C = 64$: all three have similar total FLOPs $O(Td^2)$. The SORF feature map adds $O(Td \log d)$ — about $10\%$ of $O(Td^2)$.

### GPU Efficiency Analysis

**Memory Access Pattern:**
- SADERF calibration: Two reductions over chunk tokens (coalesced, $O(Cd)$ reads)
- SORF projection: Three in-place FWHT passes — butterfly pattern, cache-friendly, $O(d \log d)$ per token
- Exponential feature map: Elementwise $\exp()$ — perfect coalescing
- All operations fit in shared memory for $d \leq 128$

**Arithmetic Intensity:**
- FWHT at $d = 64$: $3 \times 6 \times 64 = 1152$ FLOPs reading $64 \times 2 = 128$ bytes (FP16) → $I = 9$ FLOPs/byte (compute-bound at this intensity)
- SADERF diagonal scaling: $d$ multiplies, $d$ reads → $I = 0.5$ FLOPs/byte (memory-bound, but fused with FWHT)

**Parallelism:**
- SORF projection: independent per token, per head → fully parallelizable across $B \times T \times H$ work items
- FWHT: warp-level butterfly — maps to warp shuffles on GPU, no warp divergence
- No sequential bottlenecks (SADERF calibration is a parallel reduction)

**Tensor Core Compatibility:**
- The FWHT itself does NOT use tensor cores (butterfly, not matmul)
- However, the subsequent intra-chunk attention $\phi(Q)\phi(K)^\top V$ remains a GEMM → tensor cores
- At $d = 64$, the FWHT costs $\sim 1\%$ of the intra-chunk GEMM — tensor core utilization stays high

**Fusion Opportunity:**
- SADERF diagonal scaling + SORF FWHT + exponential can be fused into a single Triton kernel
- This avoids 3 separate kernel launches and keeps data in registers/shared memory
- The fused kernel reads $q_t, k_t$ once and writes $\phi(q_t), \phi(k_t)$ once

### HBM Bandwidth Analysis

| Data movement | GLA (identity) | GLA + SADERF-SORF |
|---------------|----------------|-------------------|
| Read Q, K | $2Td \cdot 2$ bytes | $2Td \cdot 2$ bytes |
| Write $\phi(Q), \phi(K)$ | $0$ | $2TM \cdot 2$ bytes |
| Intra-chunk GEMM reads | $O(TC^2 d / B_{\text{SRAM}})$ | $O(TCMd / B_{\text{SRAM}})$ |

With $M = d$: the feature map adds $4Td$ bytes of HBM traffic (writing $\phi(Q), \phi(K)$). At $T = 4096, d = 64$: $4 \times 4096 \times 64 \times 2 = 2$ MB — negligible vs. the $\sim 100$ MB of intra-chunk GEMM traffic.

## Risks & Limitations

### Risk 1: SORF Bias at Small $d$
- **Issue**: SORF has $O(z/\sqrt{d})$ bias. At $d = 16$ (MVE), this could be significant.
- **Mitigation**: Test with $d = 32$ and $d = 64$ in ablation. If bias is the problem, use 2-block SORF ($HD_1HD_2$) which has lower bias.

### Risk 2: Feature Map Training Stability
- **Issue**: The exponential in the SADERF feature map can produce very large or very small values.
- **Mitigation**: Clamp feature map values to $[\exp(-10), \exp(10)]$. Use mixed-precision with FP32 accumulation for the feature map.

### Risk 3: SADERF Calibration Overhead
- **Issue**: Computing $\Psi$ requires per-chunk statistics of Q and K, adding a reduction pass.
- **Mitigation**: Use running exponential moving average of $\bar{q}_l^2, \bar{k}_l^2$ across chunks, avoiding per-chunk computation. Cost: $O(d)$ per chunk (negligible).

### Risk 4: Marginal Benefit Over ReGLA's elu+1
- **Issue**: ReGLA showed that simple elu+1 features already help GLA significantly. SADERF-SORF may not improve enough over elu+1 to justify the complexity.
- **Mitigation**: Include elu+1 as a baseline. If SADERF-SORF only marginally beats elu+1, the experiment still provides valuable signal about the importance of kernel approximation quality vs. nonlinearity.

### Risk 5: State Dimension Mismatch
- **Issue**: With $M > d$, the inter-chunk state $S \in \mathbb{R}^{M \times d}$ grows. For $M = 2d$, state doubles, potentially exceeding SRAM for large $d$.
- **Mitigation**: Start with $M = d$ (same state size). Only test $M = 2d$ if $M = d$ shows promise.

## Follow-up Experiments

### If Successful:
1. **Compose with post-sigmoid gating (proposal 009)**: SADERF features + sigmoid gate should compound benefits
2. **Scale to 350M/1B params**: Test if the perplexity gain holds at larger scale
3. **Learnable SADERF**: Make $\Psi$ a learned parameter (initialized from data statistics) rather than computed from running averages
4. **Test with TFLA at $C = 256$–$512$**: Larger chunks should amplify the feature map benefit
5. **Compare with Spectraformer's OPRF-FastFoodL**: Is SADERF-SORF the best structured feature map for GLA?
6. **Fused SADERF-SORF Triton kernel**: Implement fully fused feature map computation and measure wall-clock overhead precisely

### If Unsuccessful:
1. **Ablate SADERF vs. SORF independently**: Is it the variance reduction or the structured projection that fails?
2. **Test larger $M$**: Perhaps $M = 4d$ is needed for quality at $d = 64$
3. **Try SDERF instead of SADERF**: Full eigendecomposition gives better variance reduction — is GPU cost worth it?
4. **Analyze learned attention patterns**: Compare $\phi(Q)\phi(K)^\top$ vs. $QK^\top$ vs. $\text{softmax}(QK^\top/\sqrt{d})$ at convergence to understand what the model actually needs

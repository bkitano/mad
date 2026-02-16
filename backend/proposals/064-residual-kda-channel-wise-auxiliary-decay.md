---
status: ongoing
priority: high
created: 2026-02-16
based_on: residual-linear-attention (243), kda-constrained-dplr-delta-chunkwise (211), gated-deltanet-chunkwise-wy-gating (203), post-attention-sigmoid-gating (094), stablessm-gradient-balanced-reparameterization (233)
experiment_number: 064
experiment_log: experiment-log-064.md
---

# Residual KDA: Auxiliary Error-Correcting State with Channel-Wise Decay

## Hypothesis

Augmenting KDA (Kimi Delta Attention) with the Residual Linear Attention (RLA) auxiliary state — creating **Residual KDA (RKDA)** — will compound the expressivity gains of channel-wise decay (KDA's per-channel $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$) with second-order error correction (RLA's residual accumulation), achieving $0.5$–$1.0$ perplexity improvement over standalone KDA at 1.5B scale with $< 10\%$ wall-clock overhead. The key insight is that channel-wise decay in the auxiliary state $\boldsymbol{R}_t$ enables **per-feature forgetting of stale residuals** — something impossible with GDN's scalar decay, where the entire residual state decays uniformly. This targeted residual management should improve recall on long-range associations where residual errors are feature-specific.

## Background

### RLA's expressivity boost + KDA's efficiency: a natural pairing

**Residual Linear Attention** (trick 243) introduces an auxiliary state $\boldsymbol{R}_t$ that accumulates past prediction residuals $\boldsymbol{r}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t$. This provides a second-order correction to the base state's output, improving recall by $3$–$5\%$ on MQAR and $0.5$–$0.8$ perplexity on language modeling. The paper validated RLA on top of Gated Linear Attention (GLA) and Gated DeltaNet (GDN) — both using **scalar** decay gates.

**KDA** (trick 211) extends GDN by replacing the scalar decay $\alpha_t$ with a per-channel decay vector $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$. This is the most expressive chunkwise-parallel linear RNN to date, achieving SOTA quality in the Kimi Linear architecture at 48B scale.

**The gap**: RLA has never been tested with channel-wise decay. Scalar decay means the auxiliary residual state $\boldsymbol{R}_t$ forgets at a uniform rate — all residual features decay identically. But prediction residuals are inherently **feature-specific**: some key dimensions may have persistent errors (requiring slow decay) while others have transient errors (requiring fast decay). Channel-wise decay in $\boldsymbol{R}_t$ provides exactly this selective retention.

### Why this matters for recall

Consider a multi-query associative recall task with 16 key-value pairs. The base state $\boldsymbol{S}_t$ must store all 16 associations. When a new pair partially overwrites an old one (because the keys overlap in some dimensions), the residual $\boldsymbol{r}_t$ captures the overwrite error. With scalar decay, the *entire* residual decays at one rate — but the overwrite error is concentrated in specific key dimensions. Channel-wise decay allows the residual state to retain error information precisely in the affected dimensions while clearing noise from unaffected ones.

### GPU efficiency: zero kernel development required

The critical practical advantage: **KDA's chunkwise kernel from FLA already supports channel-wise decay**. The residual pass requires calling the *same kernel* a second time with the residual $\boldsymbol{r}_t$ as input and a separate channel-wise gate $\boldsymbol{\gamma}_t$. The only new computation is:

1. Computing $\boldsymbol{r}_t = \text{Clip}(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t)$ — available as a byproduct of the base pass's inter-chunk computation
2. A second call to the KDA chunkwise kernel with $(q, k, r, \boldsymbol{\alpha}^R, \gamma)$
3. Adding the two outputs

All operations are existing matmuls and elementwise ops. No new CUDA/Triton kernel development needed.

## Related Work

- **Residual Linear Attention (arXiv:2509.25223, 2025)**: Introduced the prediction-correction framework and auxiliary residual state. Validated on GLA (→ RLA) and GDN (→ RDN) with scalar gating. **Not tested with channel-wise decay.** Our approach extends RDN to KDA's per-channel gates.

- **Kimi Linear (arXiv:2510.26692, 2025)**: Introduced KDA with channel-wise decay and the constrained DPLR formulation. Validated at 48B scale. **Does not use residual fitting.** Our approach adds the auxiliary state to KDA.

- **Deep Delta Learning (arXiv:2601.00417, 2026)**: Applied the delta rule over the *depth* dimension (residual connections). Different axis of application than our temporal residual. Complementary technique.

- **Higher-Order Linear Attention (HLA, trick 222)**: Uses second-order key covariance $\boldsymbol{M}_t = \sum k_i k_i^\top$ for state correction. Different mechanism: HLA augments the key metric, while we augment the output via error residuals. HLA's $O(d^2)$ per-token overhead is much larger than our 2× state cost.

- **Log-Linear Attention (arXiv:2506.04761, 2025)**: Extended GDN with log-linear (multiplicative) gates. Orthogonal improvement — could be combined with RKDA.

No prior work combines residual error-correcting states with channel-wise decay gates.

## Mathematical Formulation

**Base KDA recurrence:**

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

$$
\boldsymbol{o}_t^{\text{base}} = \boldsymbol{S}_t^\top \boldsymbol{q}_t
$$

**Residual computation (byproduct of inter-chunk pass):**

$$
\boldsymbol{r}_t = \text{Clip}_{[-c, c]}\left(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t\right) \in \mathbb{R}^{d_v}
$$

where $c = 1$ is the clipping threshold.

**Auxiliary residual state with channel-wise decay (our contribution):**

$$
\boldsymbol{R}_t = (\boldsymbol{I} - \gamma_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t^R) \boldsymbol{R}_{t-1} + \gamma_t \boldsymbol{r}_t \boldsymbol{k}_t^\top
$$

$$
\boldsymbol{o}_t^{\text{residual}} = \boldsymbol{R}_t^\top \boldsymbol{q}_t
$$

**Combined output:**

$$
\boldsymbol{o}_t = \boldsymbol{o}_t^{\text{base}} + \boldsymbol{o}_t^{\text{residual}}
$$

**Key Definitions:**

- $\boldsymbol{S}_t \in \mathbb{R}^{d_k \times d_v}$ — primary state matrix
- $\boldsymbol{R}_t \in \mathbb{R}^{d_k \times d_v}$ — auxiliary residual state matrix
- $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$ — per-channel decay for primary state (learned via $\boldsymbol{\alpha}_t = \sigma(W_\alpha^{\downarrow} W_\alpha^{\uparrow} x_t)$)
- $\boldsymbol{\alpha}_t^R \in [0,1]^{d_k}$ — per-channel decay for residual state (separate learned parameters)
- $\beta_t \in [0,1]$ — primary learning rate gate
- $\gamma_t \in [0,1]$ — residual correction strength (separate from $\beta_t$)
- $\boldsymbol{r}_t \in \mathbb{R}^{d_v}$ — clipped prediction residual

**Gate parameterization:**

Primary decay: $\boldsymbol{\alpha}_t = \sigma(W_\alpha^{\downarrow} W_\alpha^{\uparrow} x_t)$ (low-rank projection, same as KDA).

Residual decay: $\boldsymbol{\alpha}_t^R = \sigma(W_{\alpha_R}^{\downarrow} W_{\alpha_R}^{\uparrow} x_t)$ (separate low-rank projection).

Correction gate: $\gamma_t = \sigma(w_\gamma^\top x_t + b_\gamma)$ (scalar, cheap).

**Total new parameters per head:** $2 d_k r_\alpha + d + 1$ for residual gate projections, where $r_\alpha$ is the low-rank dimension (typically 16–32). This is negligible compared to the base model.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | KDA (constrained DPLR) with residual fitting |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Head dim | $d_k = d_v = 128$ |
| Heads | $n = 8$ |
| Chunk size | $C = 64$ |
| Residual clip | $c = 1.0$ |
| State memory per head | $2 \times d_k \times d_v = 2 \times 128 \times 128 = 32\text{K}$ |
| Total parameters | ~370M |

### Baseline

Three baselines:
1. **KDA** (standalone, no residual): Kimi Delta Attention with channel-wise decay. $O(6Td_h^2 + 3TCd_h + TC^2)$ FLOPs per head.
2. **RDN** (GDN + residual, scalar decay): Residual Delta Net from the RLA paper. $2 \times O(\text{GDN})$ FLOPs.
3. **Standard softmax Transformer**: FlashAttention-2, $O(T^2 d_h)$ per head.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $\geq 0.90 \times$ KDA | Tokens/sec on H100, averaged over 1K steps |
| Quality | $\geq 0.5$ ppl improvement over KDA | WikiText-103 / SlimPajama validation |
| Recall | $\geq 5\%$ improvement over KDA on MQAR | 8K context MQAR accuracy |
| State overhead | $2 \times$ KDA state | $2 \times d_k \times d_v$ per head |

### Estimated Compute

**Small** ($< 50$ GPU-hours): 370M model, 10B tokens on SlimPajama, single H100.

## Expected Outcome

**If hypothesis is correct:**
- $0.5$–$1.0$ perplexity improvement over KDA from the residual error correction
- $5$–$10\%$ MQAR recall improvement, especially at long context ($\geq 4096$)
- Channel-wise residual decay outperforms scalar residual decay (ablation: RKDA vs RDN-with-KDA-base)
- Wall-clock overhead $< 10\%$ (the residual pass is a second kernel call with the same structure)

**If hypothesis is wrong:**
- If quality matches RDN (no benefit from channel-wise residual decay): the residual errors are sufficiently homogeneous across features that uniform decay is optimal. This would be useful knowledge about the error structure.
- If quality is worse than KDA (residual fitting hurts with channel-wise decay): the auxiliary state may interfere with KDA's already-strong memory management. This would suggest that channel-wise decay already captures what residual fitting provides with scalar decay.

## Minimum Viable Experiment

### Setup

- **Model**: 2-layer KDA, $d = 128$, $d_k = d_v = 64$, $n = 2$ heads (~80K params)
- **Task**: Multi-Query Associative Recall (MQAR) with 16 key-value pairs at length 256
- **Variants**: (1) KDA, (2) KDA + scalar-decay residual (RDN-style), (3) KDA + channel-wise-decay residual (RKDA)
- **Data**: 10K synthetic MQAR sequences
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria

- RKDA achieves $\geq 5\%$ absolute accuracy improvement over KDA on MQAR
- RKDA outperforms KDA + scalar-residual by $\geq 2\%$ (showing channel-wise benefit)
- Training convergence is stable (no NaN, no divergence)

### Failure Criteria

- If all three variants achieve identical accuracy ($< 1\%$ difference): the residual mechanism provides no benefit when combined with KDA's already-strong channel-wise gating
- If RKDA diverges during training: the channel-wise residual decay introduces numerical instability (unlikely given KDA's stability guarantees, but should be checked)

### Why This Test Is Sufficient

MQAR directly tests recall capacity — the core capability that residual fitting improves. Channel-wise decay's benefit should manifest as better per-feature recall discrimination. If RKDA shows $\geq 5\%$ improvement at toy scale, the mechanism works; scaling adds capacity. The 16-pair MQAR at length 256 is challenging enough to differentiate the variants.

## Memory Access Pattern Analysis

**Base KDA forward (pass 1):**
- Load $Q, K, V, \boldsymbol{\alpha}, \beta$ from HBM: $5 \times T \times d_k$ elements per head
- Intra-chunk: $C \times C$ matmuls in SRAM (chunk size $C = 64$ → $64 \times 64 \times 2 = 8$KB per head, trivially fits in shared memory)
- Inter-chunk: state $S \in \mathbb{R}^{d_k \times d_v}$ ($128 \times 128 \times 2 = 32$KB) fits in shared memory
- Arithmetic intensity: $\approx C$ FLOPs/byte (compute-bound for $C \geq 32$)

**Residual computation:**
- $\boldsymbol{r}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t$: the matvec $\boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t$ is already computed in the base pass's inter-chunk output. The subtraction and clipping are elementwise ops on data already in registers. **Zero additional HBM loads.**

**Residual pass (pass 2):**
- Load residuals $\boldsymbol{r}$ (replaces $V$), $\boldsymbol{\alpha}^R, \gamma$ from HBM: $3 \times T \times d_k$ elements per head
- Same kernel structure as pass 1: intra-chunk + inter-chunk
- Output added to base output (elementwise, in registers or L1)

**Total HBM traffic**: $\approx 1.6\times$ base KDA (the residual pass loads $r, \alpha^R, \gamma$ but reuses $Q, K$; $Q, K$ are already in L2 from pass 1 if sequence fits).

## Parallelism Analysis

- **Pass 1 and Pass 2 are sequential** (pass 2 depends on residuals from pass 1). No inter-pass parallelism.
- **Within each pass**: full chunkwise parallelism across chunks and heads. Identical SM utilization as standalone KDA.
- **Tensor core utilization**: Both passes use the same KDA kernel, which is matmul-dominated ($\geq 87\%$ tensor core FLOPs at $C = 64$, $d_k = 128$).
- **No warp divergence**: Both passes are regular matmul + elementwise ops.
- **No extra kernel launches**: residual computation fuses into the epilogue of pass 1.

## Theoretical Analysis

Per-token FLOPs comparison (single head):

| Operation | KDA | RKDA (proposed) |
|-----------|-----|-----------------|
| Base state update | $6d_h^2 + 3Cd_h + C^2$ | $6d_h^2 + 3Cd_h + C^2$ |
| Residual computation | — | $d_k + d_v$ (elementwise) |
| Auxiliary state update | — | $6d_h^2 + 3Cd_h + C^2$ |
| Output combination | — | $d_v$ (addition) |
| **Total per token** | $6d_h^2 + 3Cd_h + C^2$ | $2(6d_h^2 + 3Cd_h + C^2)$ |

For $d_h = 128$, $C = 64$: base = $98\text{K} + 24\text{K} + 4\text{K} = 126\text{K}$ FLOPs/token/head. RKDA = $252\text{K}$.

**However**: the projection GEMMs ($W_Q, W_K, W_V$ etc.) are unchanged and account for $40$–$60\%$ of total layer FLOPs. So the total layer FLOP overhead is $\approx 40$–$60\%$ of the scan portion, which is $\approx 20$–$30\%$ of total layer FLOPs. **Expected wall-clock overhead: $8$–$15\%$** (the kernel is compute-bound, and pass 2's $Q, K$ benefit from L2 cache warmth).

## Risks & Limitations

1. **Diminishing returns with KDA**: KDA's channel-wise decay may already provide the fine-grained memory management that residual fitting adds for scalar-decay models. If so, the quality gain may be smaller than the $0.5$–$0.8$ seen for RDN over GDN.

2. **2× state memory**: The auxiliary state doubles the per-head recurrent memory. For 8 heads at $d_k = d_v = 128$: $2 \times 8 \times 128 \times 128 \times 2 = 512\text{KB}$ total, which still fits comfortably in H100 shared memory (256KB per SM, can be split across SMs). But at larger head dimensions ($d_k = 256$), this may become a constraint.

3. **Residual clipping threshold**: The clip value $c$ is a hyperparameter. RLA used $c = 1.0$ universally, but KDA's different dynamic range (channel-wise decay can create larger/smaller residuals in different channels) may require channel-dependent clipping.

4. **Backward pass cost**: The gradient through the residual requires backpropagating through two sequential kernel calls. This approximately doubles the backward scan cost, but the projections (which dominate) are unchanged.

5. **Gate interaction**: The separate residual decay $\boldsymbol{\alpha}_t^R$ introduces $d_k$ additional learned parameters per head. If these gates learn the same values as $\boldsymbol{\alpha}_t$ (primary decay), the residual state becomes redundant. Monitoring $\|\boldsymbol{\alpha}_t - \boldsymbol{\alpha}_t^R\|$ during training will reveal whether the two states learn complementary dynamics.

## Follow-up Experiments

1. **Ablation: shared vs. independent residual gates**: Test whether $\boldsymbol{\alpha}_t^R = \boldsymbol{\alpha}_t$ (shared decay) or $\boldsymbol{\alpha}_t^R = f(\boldsymbol{\alpha}_t)$ (derived decay) performs comparably to fully independent gates. If so, the parameter overhead and one gate projection GEMM can be eliminated.

2. **Fused two-state kernel**: Fuse the base and residual passes into a single kernel that maintains both $\boldsymbol{S}_t$ and $\boldsymbol{R}_t$ in shared memory simultaneously. This eliminates the second HBM round-trip for $Q, K$ and could reduce overhead from $10\%$ to $3$–$5\%$.

3. **RKDA + post-sigmoid gating (proposal 060)**: Combine residual fitting with the post-readout sigmoid gate for compounding quality gains. The sigmoid gate operates on the *combined* output $\boldsymbol{o}_t$, so it can adaptively weight the base vs. residual contributions.

4. **Residual fitting for RWKV-7**: RWKV-7's generalized delta rule (trick 219) has vector-valued decay and a separate removal key $\hat{\kappa}_t$. Testing residual fitting on RWKV-7 would validate generality beyond the KDA/GDN family.

5. **Scaling to 3B–7B**: If RKDA shows quality gains at 370M/1.5B, validate at 3B–7B to confirm the improvement scales.

## Human Review

(To be filled by reviewer)

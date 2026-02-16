# DyT-Fused Normalization-Free Chunkwise Linear RNN

**Status**: proposed
**Priority**: high
**Created**: 2026-02-16
**Based on**: [241-dynamic-tanh-normalization-free], [242-peri-ln-peripheral-layer-normalization], [177-gla-secondary-chunking-log-space-gating], [203-gated-deltanet-chunkwise-wy-gating], [212-flashrnn-io-aware-fused-recurrence], [227-smooth-swiglu-per-channel-fp8-stability]

## Hypothesis

Replacing LayerNorm/RMSNorm with Dynamic Tanh (DyT) in the sub-layer normalization of chunkwise linear RNN architectures (GLA, Gated DeltaNet, Mamba-2) will enable **deeper kernel fusion** — specifically fusing the pre-norm and post-norm (Peri-LN) operations directly into the chunkwise attention kernel — eliminating two HBM round-trips per sub-layer and yielding a **5–15% wall-clock training throughput improvement** at sequence lengths $\geq 2048$.

## Background

Current chunkwise linear RNN training pipelines (GLA, Gated DeltaNet, Mamba-2) use the following per-block structure:

```
x → RMSNorm → Projections(Q,K,V,gates) → Chunkwise Kernel → Output Proj → [+ residual] → RMSNorm → SwiGLU FFN → [+ residual]
```

Each RMSNorm requires:
1. **Read** the full activation tensor from HBM ($O(BTd)$ bytes)
2. **Reduce** across the channel dimension $d$ to compute variance (warp-level reduction + cross-warp sync)
3. **Write** the normalized output back to HBM ($O(BTd)$ bytes)

With Peri-LN (trick 242), which adds post-module output normalization for stability, there are **4 RMSNorm calls per block** — each one a separate HBM round-trip with a cross-channel reduction barrier.

**The key insight:** DyT replaces RMSNorm with $\gamma \odot \tanh(\alpha x) + \beta$, which is a **purely elementwise** operation (no cross-channel reduction). This means:

1. **Pre-norm DyT can be fused into the Q/K/V projection kernels**: Instead of `RMSNorm(x) → matmul(W_Q)`, we compute `matmul(W_Q, tanh(α·x))` — the tanh is absorbed into the matmul's epilogue/prologue.

2. **Post-norm DyT (Peri-LN style) can be fused into the chunkwise kernel's output**: The chunkwise kernel already writes its output to HBM; applying DyT as an epilogue adds negligible FLOPs and saves a full HBM read+write cycle.

3. **The DyT after SwiGLU can be fused into the FFN's output epilogue** using epilogue visitor trees (trick 039).

Each fusion eliminates one HBM round-trip of $O(BTd)$ bytes. For a 1.3B model with $d = 2048$, $B = 8$, $T = 2048$, each round-trip is $\sim$64 MB in BF16 — at H100's 3.35 TB/s HBM bandwidth, that's $\sim$19 µs per norm. With 4 norms per block × 24 blocks = 96 eliminated round-trips, this saves $\sim$1.8 ms per step, or roughly 5–10% of a typical step time (~20–30 ms).

## Related Work

- **DyT (Zhu et al., CVPR 2025)**: Showed DyT replaces LN/RMSNorm without quality loss in Transformers, ViTs, and SSMs (including HyenaDNA). However, they acknowledge **no wall-clock speedup** when using torch.compile with optimized LN kernels. Our approach differs by targeting the **fusion opportunity** — DyT's elementwise nature enables it to be absorbed into adjacent kernels, which optimized LN cannot do because of the reduction.

- **Peri-LN (Kim et al., ICML 2025)**: Demonstrated that output normalization (after module, before residual add) substantially improves stability and quality. Used in Gemma 2/3 and OLMo 2. But the extra LN calls add HBM round-trips. Our approach makes Peri-LN **free** by fusing both pre-norm and post-norm DyT into the chunkwise kernel.

- **FlashRNN (Beck et al., 2025)**: Demonstrated IO-aware fused recurrence for RNNs, keeping state in SRAM. Our approach extends this principle to normalization — keeping DyT computation in SRAM/registers alongside the chunkwise recurrence.

- **Smooth SwiGLU FP8 (trick 227)**: Per-channel smooth quantization for FP8 training stability. DyT's tanh provides a complementary smoothing mechanism — it naturally clips extreme outliers, potentially improving FP8 compatibility.

- **FlashLinearAttention (Yang et al., ICML 2024)**: The GLA chunkwise kernel already fuses projection-free attention into a single kernel launch. Our proposal extends this fusion boundary to include normalization.

No directly related work found combining DyT with fused chunkwise linear RNN kernels.

## Mathematical Formulation

**Standard Peri-LN GLA Block:**

$$
\tilde{x}_t = \text{RMSNorm}(x_t) \quad \text{(pre-norm: HBM read, reduce over } d \text{, write)}
$$

$$
q_t, k_t, v_t, \alpha_t = \text{LinearProj}(\tilde{x}_t) \quad \text{(matmul)}
$$

$$
o_t = \text{ChunkwiseGLA}(q, k, v, \alpha) \quad \text{(chunkwise kernel)}
$$

$$
\hat{o}_t = \text{RMSNorm}(W_O o_t) \quad \text{(post-norm: HBM read, reduce, write)}
$$

$$
x_{t+1} = x_t + \hat{o}_t \quad \text{(residual add)}
$$

**Proposed DyT-Fused GLA Block:**

$$
\tilde{x}_t = \gamma_{\text{pre}} \odot \tanh(\alpha_{\text{pre}} x_t) + \beta_{\text{pre}} \quad \text{(elementwise, fused into projection kernel prologue)}
$$

$$
q_t, k_t, v_t, \alpha_t = W_{Q,K,V,\alpha} \tilde{x}_t \quad \text{(single fused matmul with DyT prologue)}
$$

$$
\hat{o}_t = \gamma_{\text{post}} \odot \tanh\left(\alpha_{\text{post}} \cdot W_O \cdot \text{ChunkwiseGLA}(q, k, v, \alpha)\right) + \beta_{\text{post}}
$$

$$
\quad \text{(DyT fused as epilogue of chunkwise kernel or output projection)}
$$

$$
x_{t+1} = x_t + \hat{o}_t \quad \text{(residual add)}
$$

**Key change:** The pre-norm DyT is fused into the input projection's load-compute pipeline. The post-norm DyT is fused into the output path. Both avoid standalone HBM read/write cycles.

**DyT parameters per block (Peri-LN configuration):**

For a model with hidden dimension $d$:
- Pre-attention DyT: $\alpha_{\text{pre}} \in \mathbb{R}$, $\gamma_{\text{pre}} \in \mathbb{R}^d$, $\beta_{\text{pre}} \in \mathbb{R}^d$ → $2d + 1$ params
- Post-attention DyT: $\alpha_{\text{post}} \in \mathbb{R}$, $\gamma_{\text{post}} \in \mathbb{R}^d$, $\beta_{\text{post}} \in \mathbb{R}^d$ → $2d + 1$ params
- Pre-FFN DyT: same → $2d + 1$ params
- Post-FFN DyT: same → $2d + 1$ params
- **Total**: $8d + 4$ params per block (identical to Peri-LN with RMSNorm)

**Key Variables:**
- $x_t \in \mathbb{R}^d$ — hidden state at position $t$
- $\alpha_{\text{pre}}, \alpha_{\text{post}} \in \mathbb{R}$ — learnable DyT scaling (initialized per width, see trick 241)
- $\gamma, \beta \in \mathbb{R}^d$ — per-channel affine parameters

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Gated DeltaNet (either) |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 4$ (for GLA) or $H = 16$ (for Gated DeltaNet) |
| Key dim | $d_k = 256$ (GLA) or $d_k = 128$ (Gated DeltaNet) |
| Chunk size | $C = 64$ |
| Normalization | DyT with Peri-LN placement |
| Parameters | ~1.3B |

### Baseline

1. **GLA + Pre-LN RMSNorm**: Standard configuration from Yang et al. (ICML 2024). Complexity: $O(TC(d_k + d_v) + Td_kd_v)$ per layer. This is the "normal" GLA.
2. **GLA + Peri-LN RMSNorm**: GLA with output normalization added. Same compute complexity + 2 extra LN calls per block.
3. **GLA + Pre-LN DyT (unfused)**: DyT replacing RMSNorm but without kernel fusion — tests quality equivalence.
4. **GLA + Peri-LN DyT (fused)**: Full proposal — DyT replacing RMSNorm with fusion into adjacent kernels.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.10 \times$ Peri-LN RMSNorm baseline | Tokens/sec on H100, steady-state |
| Perplexity | $\leq$ Peri-LN RMSNorm baseline | WikiText-103/FineWeb validation |
| Memory | $\leq$ baseline | Peak GPU memory (GB) |
| Training stability | 0 NaN/divergence | Over 5 seeds × 10B tokens |
| Kernel time breakdown | Norm time $\to 0$ | NSight profiling of norm vs. matmul time |

### Estimated Compute

- **MVE**: < 10 minutes on single GPU (~50K params)
- **Phase 1** (quality validation, 350M model, 10B tokens): ~80 GPU-hours on H100
- **Phase 2** (throughput benchmarking, 1.3B model): ~40 GPU-hours on H100
- **Total**: ~120 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- Fused DyT Peri-LN achieves 5–15% throughput improvement over Peri-LN RMSNorm at sequence length $T = 2048$
- The throughput gain increases at shorter sequences (where norm overhead is a larger fraction of step time)
- Quality (perplexity) matches or exceeds Peri-LN RMSNorm — DyT's tanh squashing provides comparable outlier suppression
- Training stability matches Peri-LN — DyT's $\alpha$ learns $\approx 1/\text{std}(x)$, providing implicit normalization
- FP8 compatibility improves: DyT's tanh naturally clips activations to $[-1, 1]$, reducing the dynamic range that FP8 must cover

**If hypothesis is wrong:**
- If quality degrades: DyT's global $\alpha$ (not per-token) may be insufficient for linear RNNs where different chunks have very different activation scales. This would motivate a per-chunk $\alpha$ variant.
- If throughput gain is < 5%: The norm overhead is already small relative to matmul cost. This would be a useful negative result quantifying the normalization overhead.
- If training diverges: DyT may not provide sufficient outlier suppression for the specific activation patterns of chunkwise linear RNNs (especially the WY transform in Gated DeltaNet). This would narrow DyT's applicability.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 64$, $d_k = 32$, $d_v = 64$, 2 heads, ~100K params
- **Task**: Autoregressive language modeling on a 5M token subset of WikiText-103
- **Data**: WikiText-103 train (first 5M tokens), eval on validation set
- **Normalization variants**: (1) Pre-LN RMSNorm, (2) Pre-LN DyT, (3) Peri-LN RMSNorm, (4) Peri-LN DyT
- **Compute**: Single GPU, < 10 minutes per variant

### Success Criteria
- DyT variants (2, 4) match RMSNorm variants (1, 3) within 0.5 perplexity points after 5M tokens
- Peri-LN DyT (4) achieves the best stability (lowest gradient norm variance across layers)
- No NaN/Inf in any variant over the full training run
- Activation distributions of DyT output approximate those of RMSNorm output (KS test $p > 0.01$)

### Failure Criteria
- If DyT perplexity exceeds RMSNorm by > 2 points: the elementwise squashing is inadequate for linear RNN activations. Kill the idea.
- If DyT training diverges within 1M tokens: the $\alpha$ initialization is wrong for this architecture. Try width-dependent $\alpha_0$ from trick 241 before killing.

### Why This Test Is Sufficient
- DyT's core property — replacing cross-channel normalization with elementwise tanh — is scale-independent. If it works at $d = 64$, it works at $d = 2048$.
- The quality comparison (DyT vs RMSNorm) is architecture-dependent but the mechanism is the same at all scales.
- The fusion benefit cannot be tested at MVE scale (too small for HBM bottleneck to matter), but quality equivalence is the prerequisite for the full experiment. If quality holds, the fusion engineering is straightforward.
- Peri-LN (Zhu et al., 2025) was validated on SSMs/RNNs (HyenaDNA, Caduceus) — the remaining question is whether it works specifically inside chunkwise linear attention blocks with their unique activation patterns (log-space gates, WY transforms).

## Theoretical Analysis

**HBM bandwidth savings per block:**

| Operation | Peri-LN RMSNorm | DyT Fused | Savings |
|-----------|----------------|-----------|---------|
| Pre-attention norm | $2BTd$ bytes (read+write) | 0 (fused into proj) | $2BTd$ |
| Post-attention norm | $2BTd$ bytes | 0 (fused into output) | $2BTd$ |
| Pre-FFN norm | $2BTd$ bytes | 0 (fused into FFN) | $2BTd$ |
| Post-FFN norm | $2BTd$ bytes | 0 (fused into FFN) | $2BTd$ |
| **Total per block** | $8BTd$ bytes | **0** | $8BTd$ |

For $B = 8$, $T = 2048$, $d = 2048$, BF16:

$$
\text{Savings per block} = 8 \times 8 \times 2048 \times 2048 \times 2 \text{ bytes} = 512 \text{ MB}
$$

With 24 blocks: $\sim$12 GB of HBM traffic eliminated per forward pass. At H100's 3.35 TB/s: $\sim$3.6 ms saved.

Typical forward pass time at 1.3B: $\sim$20–30 ms → **12–18% potential speedup** (upper bound, assuming norms are perfectly serialized).

**Compute overhead of DyT:**

$$
\text{DyT FLOPs per call} = 3 \times B \times T \times d \quad \text{(multiply by } \alpha \text{, tanh, multiply by } \gamma \text{, add } \beta \text{)}
$$

vs. RMSNorm:

$$
\text{RMSNorm FLOPs per call} = 3 \times B \times T \times d + B \times T \times \text{reduce}(d)
$$

DyT saves the reduce$(d)$ term. Both have the same elementwise FLOPs. DyT replaces division with tanh — tanh is slightly more expensive as a single op, but this is negligible in the fused setting.

## Risks & Limitations

1. **DyT's global $\alpha$ may be insufficient for chunkwise computation**: Different chunks may have very different activation scales (especially with data-dependent gating where $\alpha_t$ varies wildly). RMSNorm normalizes per-token, adapting to each token's scale. DyT uses a single global $\alpha$ for all tokens. **Mitigation**: Test per-chunk $\alpha$ variant (one learnable $\alpha$ per chunk position).

2. **Tanh saturation in BF16**: For large activations, $\tanh(\alpha x)$ saturates to $\pm 1$, losing gradient signal. In BF16, the transition region is narrower. **Mitigation**: Initialize $\alpha_0$ small enough that saturation is rare (use width-dependent init from trick 241).

3. **Kernel fusion engineering complexity**: Fusing DyT into the chunkwise GLA kernel requires modifying the Triton kernel code in FlashLinearAttention. The pre-norm fusion (into projection prologue) is straightforward; the post-norm fusion (into chunkwise kernel epilogue) requires careful register management. **Mitigation**: Start with unfused DyT to validate quality, then add fusion incrementally.

4. **Backward pass complications**: Fusing DyT into the forward kernel also affects the backward pass. The $\text{sech}^2(\alpha x)$ gradient of tanh must be computed during the backward kernel. **Mitigation**: Store $\tanh(\alpha x)$ in the forward pass (it's already needed for the output), then compute $1 - \tanh^2$ in the backward — this is a standard epilogue pattern.

5. **Interaction with log-space gates**: GLA's secondary chunking uses log-space gates for numerical stability. DyT's tanh squashing could interact poorly with the log-space computation if applied before gating projections. **Mitigation**: Apply DyT only to the main hidden state, not to gate computation paths.

## Follow-up Experiments

1. **DyT + FP8 chunkwise training**: Test whether DyT's natural activation clipping ($[-1, 1]$ output range) improves FP8 training stability (proposal 050). The clipped range maps well to FP8's limited dynamic range.

2. **Per-chunk adaptive $\alpha$**: Instead of a single global $\alpha$, learn $\alpha_c$ per chunk position or per head. This addresses risk #1 while maintaining the elementwise property.

3. **DyT + Titans momentum memory**: Titans' neural memory (trick 238) uses gradient-based updates that can produce extreme activations. DyT's squashing may be particularly beneficial for stabilizing the momentum term.

4. **DyT for inter-chunk state normalization**: The state $S_{[t]}$ passed between chunks can grow unboundedly. Applying DyT to the state (not just activations) could improve stability without the cost of state normalization.

5. **Kernel-level profiling**: Use NSight Compute to precisely quantify the HBM bandwidth savings from fusion and compare against theoretical predictions.

## Human Review



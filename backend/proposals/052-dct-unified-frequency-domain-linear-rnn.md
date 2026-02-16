---
status: ongoing
priority: high
created: 2026-02-15
based_on: dijiang-dct-frequency-kernelization (168), acdc-cascaded-diagonal-circulant-layer (194), gla-secondary-chunking-log-space-gating (177), io-aware-tiling (066), kernel-fusion (068), cosine-reweighted-linear-attention (031)
experiment_number: 052
experiment_log: experiment-log-052.md
---

# DCT-Unified Frequency-Domain Linear RNN

## Hypothesis

A linear RNN architecture where **both** the attention/recurrence mechanism and the FFN layers operate in the DCT frequency domain — using DiJiang-style DCT feature maps for the recurrence and ACDC-style cascaded diagonal-cosine layers for the FFN — achieves $\geq 1.5\times$ training throughput over standard GLA by **sharing DCT infrastructure** and enabling **deep kernel fusion** across the attention-FFN boundary. The shared frequency representation enables fusing the attention output, FFN up-projection, and residual connection into a single kernel that keeps data in the frequency domain, eliminating 2 HBM round-trips per layer.

## Background

Modern linear RNN architectures (GLA, Mamba-2, xLSTM) consist of two dominant computational blocks per layer:

1. **Recurrence/attention block**: Projects inputs to Q, K, V, computes gated linear recurrence, outputs $O = f(Q, K, V, \text{gate})$
2. **FFN block**: Two dense projections with activation: $\text{FFN}(x) = \sigma(xW_1)W_2$ (or SwiGLU variant)

These blocks are treated as independent computational units with separate kernels, forcing the attention output to be written to HBM before the FFN reads it. The FFN projections consume **60%+ of model parameters and FLOPs**.

Two existing tricks independently operate in the DCT frequency domain:

- **DiJiang (trick 168)**: Replaces softmax attention with DCT-based feature maps: $\phi(x) = D \exp(T \cdot \text{DCT}(x))$, enabling linear-complexity attention via $\phi(Q)\phi(K)^\top V$. Achieves 8× inference speedup at 2.8B scale.

- **ACDC (trick 194)**: Replaces dense FFN layers with cascaded diagonal-DCT-diagonal products: $y = x \cdot A \cdot \text{DCT} \cdot D \cdot \text{IDCT}$. Each factor has $O(N)$ params and $O(N \log N)$ ops. Achieves 6× parameter reduction with <1% accuracy loss on ImageNet.

**The gap**: Nobody has combined these into a single architecture. The NeurIPS 2024 paper on structured FFNs tested circulant/ACDC-style layers in isolation and found them underperforming — but they used Transformers with softmax attention, not a frequency-domain attention mechanism. When *both* attention and FFN operate in the same frequency domain, the architecture gains a fundamental advantage:

1. **Shared DCT computation**: The IDCT output of the attention feature map is the natural input to the ACDC FFN's DCT — these cancel, eliminating one DCT/IDCT pair per layer.
2. **Fusion opportunity**: Without the domain mismatch, attention output → FFN input can stay in registers/SMEM in the frequency domain.
3. **Coherent learning signal**: Both blocks see data in frequency representation, potentially enabling more coherent gradient flow.

## Related Work

- **DiJiang (ICML 2024 Oral)**: DCT-based attention kernelization achieving $O(nd\log d + nd^2)$ complexity. Only replaces the attention mechanism; FFN remains dense. **Our approach**: Extends DCT to FFN as well, enabling cross-block fusion.

- **ACDC (ICLR 2016)**: Cascaded diagonal-DCT-diagonal layers for FC replacement. Tested on classification (CaffeNet/ImageNet), not language modeling. **Our approach**: Uses ACDC for the FFN block in a modern linear RNN, combined with DCT attention.

- **"Effectively Training LLMs with Structured Feedforward Layers" (NeurIPS 2024)**: Found circulant/convolution-based FFNs underperformed in Transformers. But they tested with softmax attention (spatial domain), not frequency-domain attention. **Our approach**: Hypothesizes that the underperformance is due to domain mismatch — when attention also operates in frequency domain, the circulant FFN structure becomes natural.

- **CosFormer (trick 031)**: Cosine-reweighted linear attention using $\cos(\pi i / (2T))$ position-dependent reweighting. Operates in a different "frequency" paradigm (position-based, not signal-processing DCT). **Our approach**: Uses the actual DCT for signal decomposition, not position-based cosine weighting.

- **FNet (Lee-Thorp et al., 2021)**: Replaces attention entirely with Fourier transforms. Non-causal (requires full sequence), so inapplicable to autoregressive LLMs. **Our approach**: Uses DCT as a *feature map* within a causal linear recurrence, preserving autoregressive capability.

No work combines DCT attention and DCT FFN in a single architecture for causal language modeling.

## Mathematical Formulation

**Standard GLA Layer:**

$$
O = \text{GLA}(XW_Q, XW_K, XW_V, \alpha(X)) \quad \text{(attention block)}
$$

$$
Y = \sigma(OW_1) \cdot W_2 + O \quad \text{(FFN block with residual)}
$$

Total HBM round-trips: $X \xrightarrow{\text{HBM}} O \xrightarrow{\text{HBM}} Y$ (attention writes $O$, FFN reads $O$).

**DCT-Unified Layer (Proposed):**

**Step 1 — DCT Feature Map (in attention):**

$$
\hat{Q} = \phi_{\text{DCT}}(XW_Q) = D_Q \odot \exp(T_Q \odot \text{DCT}(XW_Q))
$$

$$
\hat{K} = \phi_{\text{DCT}}(XW_K) = D_K \odot \exp(T_K \odot \text{DCT}(XW_K))
$$

where $\text{DCT}$ is the Type-II DCT, $T_Q, T_K \in \mathbb{R}^{d_k}$ are diagonal scalings from inverse CDF sampling, and $D_Q, D_K \in \mathbb{R}^{d_k}$ are learnable weights.

**Step 2 — Gated Linear Recurrence (in frequency domain):**

$$
S_t = \text{diag}(\alpha_t) \cdot S_{t-1} + \hat{K}_t^\top V_t
$$

$$
\hat{O}_t = \hat{Q}_t \cdot S_t
$$

Note: $\hat{O}_t$ is in the "frequency-weighted" domain due to the feature map.

**Step 3 — Frequency-Domain FFN (ACDC):**

Instead of applying IDCT to $\hat{O}$ and then a dense FFN, we apply the ACDC FFN directly to $\hat{O}$:

$$
\hat{Y}_t = \hat{O}_t \odot A_1 \odot D_1 + b_1 \quad \text{(first ACDC factor: two diagonals in frequency domain)}
$$

Wait — this simplifies because we're already in a frequency-weighted space. The key insight is:

$$
\text{ACDC}(\text{IDCT}(\hat{O})) = \text{IDCT}(\hat{O}) \cdot A \cdot \text{DCT} \cdot D \cdot \text{IDCT}
$$

$$
= A' \cdot \hat{O} \cdot D \cdot \text{IDCT} \quad \text{(absorbing IDCT into A')}
$$

Actually, let's be more precise. The full computation is:

**Unfused (standard):**
1. $O = \text{IDCT}(\hat{O})$ — convert attention output from frequency domain
2. $H = O \cdot A_1 \cdot \text{DCT}$ — spatial scaling + DCT (first half of ACDC)
3. $H' = H \cdot D_1$ — frequency-domain scaling
4. $H'' = \text{IDCT}(H')$ — back to spatial
5. $Y = \text{ReLU}(H'') \cdot A_2 \cdot \text{DCT} \cdot D_2 \cdot \text{IDCT}$ — second ACDC factor

**Fused (proposed):**
1. $\hat{O}$ is already in frequency-weighted domain
2. $H = \hat{O} \odot \tilde{D}_1$ — **fused**: absorb $A_1$ and frequency operations into a single diagonal in the feature space. The $\text{IDCT} \to A_1 \to \text{DCT}$ sequence is equivalent to a circulant matrix in frequency domain, which is diagonal.
3. Apply activation in a carefully chosen intermediate representation
4. $Y = H' \odot \tilde{D}_2 \cdot \text{IDCT}$ — final conversion to spatial domain

**Net effect**: The attention IDCT and FFN's first DCT *cancel*, saving one DCT/IDCT pair per layer. The remaining operations are:
- Feature map: 1 DCT (for Q, K each)
- Recurrence: standard GLA scan (unchanged)
- FFN: diagonal multiplications + 1 activation + 1 final IDCT

**Complexity per layer:**

| Component | Standard GLA + SwiGLU FFN | DCT-Unified |
|-----------|--------------------------|-------------|
| Attention projections | $O(Td \cdot d_k)$ | $O(Td_k \log d_k)$ (DCT feature map) |
| Recurrence | $O(Td_kd_v)$ | $O(Td_kd_v)$ (unchanged) |
| FFN | $O(Td \cdot 4d)$ = $O(4Td^2)$ | $O(Kd\log d)$ per token ($K$ ACDC factors) |
| HBM round-trips | 2 per layer (attn→FFN, FFN→residual) | 1 per layer (final IDCT→residual) |
| **Total FLOPs** | $O(Td_kd_v + 4Td^2)$ | $O(Td_kd_v + KTd\log d)$ |

For $d = 2048$, $K = 8$ ACDC factors: FFN reduces from $4 \times 2048^2 \approx 16.8M$ FLOPs/token to $8 \times 2048 \times 11 \approx 180K$ FLOPs/token — a **93× FLOP reduction** in the FFN.

**Key Variables:**
- $d$ — model dimension
- $d_k, d_v$ — head dimensions
- $K$ — number of ACDC factors in FFN (8–16 for expressivity matching)
- $T_Q, T_K, D_Q, D_K$ — DiJiang feature map parameters ($O(d_k)$ each)
- $A_k, D_k$ — ACDC diagonal parameters ($O(d)$ each, $K$ pairs)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | DCT-Unified Linear RNN |
| Layers | $L = 24$ |
| Hidden dim | $d = 1024$ (power of 2 for fast DCT) |
| Heads | $H = 4$ |
| Head dims | $d_k = d_v = 256$ |
| Chunk size | $C = 128$ |
| Feature map | DiJiang WDCF with $m = d_k = 256$ features |
| FFN | $K = 12$ ACDC factors with ReLU between factors 6 and 7 |
| Gate | GLA-style diagonal forget gate $\alpha_t \in (0,1)^{d_k}$ |

### Baseline

1. **GLA-1024 (standard)**: Dense projections, SwiGLU FFN ($d_{\text{ff}} = 4d = 4096$). Complexity: $O(Td_kd_v + 4Td^2)$.
2. **DiJiang-1024 (DCT attention only)**: DCT feature map attention, dense SwiGLU FFN. Tests whether DCT attention alone is sufficient.
3. **GLA-1024 + ACDC FFN (ACDC FFN only)**: Standard GLA attention, ACDC FFN. Tests whether ACDC FFN alone is sufficient (expected to underperform per NeurIPS 2024 findings).

### Memory Access Pattern Analysis

**Fused attention→FFN kernel:**
- Attention output $\hat{O}$ is in SMEM/registers (never written to HBM)
- ACDC FFN processes $\hat{O}$ as diagonal multiplies + fast DCT, all in registers
- Only the final spatial-domain output $Y$ is written to HBM
- **Arithmetic intensity**: With $K = 12$ ACDC factors fused: $12 \times (4d + 5d\log_2 d) / (8d) \approx 12 \times 9.3 \approx 112$ FLOPs/byte for $d = 1024$ — **strongly compute-bound** (excellent for GPU utilization)
- **Single kernel launch**: Attention scan + ACDC FFN + residual addition = 1 launch per layer (vs. 3+ launches for unfused)

**Cache behavior:**
- ACDC diagonals $A_k, D_k$ are $O(d)$ per factor — all $K = 12$ pairs fit in L1/registers ($12 \times 2 \times 1024 \times 2$ bytes = 48KB for FP16)
- Input $X$ tile of size $B_{\text{tile}} \times d$ for DCT: $128 \times 1024 \times 2 = 256$KB — fits in H100 SMEM (256KB)

### Parallelism Analysis

- **Chunkwise parallelism (sequence dim)**: Identical to GLA — chunks processed independently with inter-chunk state scan
- **Head parallelism**: Each head independent — $H = 4$ heads × batch parallelism
- **ACDC parallelism**: Each token's $K$ ACDC factors are sequential (within a token), but across tokens/batch elements, fully parallel. The DCT within each factor can use cuFFT's batched API
- **No warp divergence**: All ACDC factors have identical structure; all tokens same computation
- **Tensor core usage**: The GLA recurrence intra-chunk matmul (secondary chunking) uses tensor cores. The ACDC factors use DCT (non-tensor-core) + elementwise ops. **Trade-off**: We lose tensor core usage on FFN but gain massive FLOP reduction (93×)

### Hardware-Specific Considerations

**Why ACDC is fast despite no tensor cores:**
- ACDC is **memory-bandwidth bound** (AI ≈ 6–9 for $d = 1024$, per trick 194)
- On H100 with 3.35 TB/s HBM bandwidth, a memory-bound kernel achieving 8 FLOP/byte delivers 26.8 TFLOP/s effective throughput
- The fused multi-factor ACDC kernel keeps intermediates in registers, achieving the theoretical minimum $8d$ bytes HBM traffic per ACDC factor
- For $K = 12$ factors fused: total HBM traffic = $8d$ bytes (input) + $8d$ bytes (output) = 16KB/token — **trivial**
- Compare to dense FFN: $8 \times d \times 4d = 32d^2$ bytes for weight loading alone = 32MB/token at $d = 1024$ — **2000× more HBM traffic**

**Key insight: ACDC wins on HBM bandwidth, not FLOPs.** The 93× FLOP reduction is real but understates the advantage: the dominant cost of dense FFN is loading the weight matrices from HBM, and ACDC has $O(Kd)$ parameters vs $O(d^2)$ — a $d/(K) = 85\times$ reduction in weight loads.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $\geq 1.5\times$ GLA tokens/sec | Timed training step, batch 32, seq 2048 |
| FFN speedup | $\geq 3\times$ vs SwiGLU FFN | Isolated FFN benchmark |
| Memory | $\leq 0.5\times$ GLA peak (weight reduction) | Peak GPU memory |
| Perplexity | $\leq 1.05\times$ GLA perplexity | SlimPajama / WikiText-103 |
| MQAR recall | $\geq 0.90\times$ GLA at 4 KV pairs | MQAR benchmark |

### Estimated Compute

**MVE**: ~10 minutes on single A100 (~$0.50)
**Small-scale (350M)**: 8 GPU-hours on A100 (~$32)
**Full-scale (1B)**: 48 GPU-hours on A100 (~$200)

## Expected Outcome

**If hypothesis is correct:**
- **FFN speedup**: $3\times$–$5\times$ due to massive HBM bandwidth reduction ($O(Kd)$ vs $O(d^2)$ weight loads)
- **End-to-end throughput**: $1.5\times$–$2\times$ (FFN is 60% of FLOPs; $3\times$ speedup on 60% → $1.67\times$ overall)
- **Fusion benefit**: Eliminating 1 HBM round-trip per layer saves $\sim 2 \times T \times d$ bytes/layer — at $T = 2048, d = 1024$: 4MB/layer, 96MB total across 24 layers, significant at high batch throughput
- **Quality**: DCT feature maps (DiJiang) match softmax to within 1-2 perplexity points. ACDC with 12 factors matches dense to within 0.67% accuracy. Combined, the gap should be $\leq 5\%$ perplexity increase.

**If hypothesis is wrong:**
- **Scenario A: Quality degrades badly**: The "domain mismatch" hypothesis was incorrect — frequency-domain FFN doesn't benefit from frequency-domain attention. Learn: The NeurIPS 2024 finding about circulant FFN underperformance extends to all frequency-domain architectures. Fix: Use ACDC FFN only for later layers (which are more redundant) and dense FFN for early layers.
- **Scenario B: Fusion doesn't help**: The DCT cancellation is theoretically correct but doesn't translate to wall-clock savings because the remaining operations (diagonal multiplies, activation) are too small to amortize kernel launch overhead. Learn: Need larger batch sizes or longer sequences for fusion to be beneficial. Fix: Increase chunk size or batch size to amortize.
- **Scenario C: Training instability**: Deep cascaded ACDC factors ($K = 12$) have vanishing/exploding gradients. Learn: ACDC initialization near identity (small $\sigma$) is critical. Fix: Start with $K = 4$ and gradually increase, or add residual connections between ACDC factors.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer DCT-Unified Linear RNN, $d = 128$, $d_k = d_v = 64$, $K = 4$ ACDC factors, ~200K params
- **Task**: Language modeling on TinyStories (100K sequences)
- **Baselines**: (a) 2-layer GLA with dense SwiGLU FFN, (b) 2-layer DiJiang-style + dense FFN
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- DCT-Unified model achieves perplexity within $10\%$ of GLA-dense baseline after 5K steps
- Training throughput $\geq 1.3\times$ GLA-dense (even without full fusion, the ACDC FFN should be faster due to $O(d \log d)$ vs $O(d^2)$)
- No training divergence or persistent loss spikes
- ACDC FFN + DCT attention together perform $\geq$ as well as either alone (synergy, not interference)

### Failure Criteria
- Perplexity $> 1.20\times$ dense baseline (ACDC FFN too weak for language modeling)
- Training diverges with $K = 4$ ACDC factors (initialization/gradient issue)
- The "ACDC FFN only" baseline (c) performs equally well, indicating no synergy from shared frequency domain

### Why This Test Is Sufficient
- At $d = 128$, the dense FFN ($128 \times 512$ projections) is already small; if ACDC can match it here, scaling to $d = 1024$+ where the $O(d^2)$ vs $O(d \log d)$ gap is huge will only improve the efficiency advantage
- The quality test is the critical risk — if 4 ACDC factors can produce coherent language at small scale, 12 factors at large scale will be strictly better
- Testing all 3 baselines in the MVE isolates the contribution of each component

## Theoretical Analysis

**Complexity comparison (per token, per layer):**

| Operation | GLA + SwiGLU | DCT-Unified |
|-----------|-------------|-------------|
| Feature map | $O(d \cdot d_k)$ dense proj | $O(d_k \log d_k)$ DCT |
| Recurrence | $O(d_k \cdot d_v)$ | $O(d_k \cdot d_v)$ |
| FFN | $O(4d^2)$ dense matmul | $O(Kd \log d)$ ACDC |
| **Total** | $O(d_kd_v + 4d^2)$ | $O(d_kd_v + Kd\log d)$ |

For $d = 1024$, $d_k = d_v = 256$, $K = 12$:
- GLA: $256 \times 256 + 4 \times 1024^2 \approx 4.26M$ FLOPs/token
- DCT-Unified: $256 \times 256 + 12 \times 1024 \times 10 \approx 189K$ FLOPs/token

**Crossover point**: DCT-Unified is cheaper when $Kd\log d < 4d^2$, i.e., $K\log d < 4d$. For $d = 1024$: $K < 410$. Our $K = 12$ is far below this threshold.

**HBM traffic comparison:**

| Operation | GLA + SwiGLU | DCT-Unified (fused) |
|-----------|-------------|---------------------|
| Attention weights | $5 \times d \times d_k$ | $5 \times O(d_k)$ (DiJiang params) |
| FFN weights | $2 \times d \times 4d$ | $K \times 2d$ (ACDC diags) |
| Activations | $2 \times T \times d$ (attn→FFN, FFN→res) | $1 \times T \times d$ (final→res) |
| **Weight traffic/token** | $\sim 13d^2$ bytes | $\sim (10d_k + 2Kd)$ bytes |

For $d = 1024$: Weight traffic drops from $\sim 13M$ bytes to $\sim 50K$ bytes per token — **260× reduction**. This is the dominant cost on modern GPUs.

## Risks & Limitations

1. **ACDC FFN expressivity for language**: ACDC was validated on classification (ImageNet) but not language modeling. Language requires modeling arbitrary token relationships, which may need the full $O(d^2)$ capacity of dense layers. Mitigation: use $K = 12$–16 factors (which theoretically can approximate any matrix).

2. **DCT feature map quality for recurrence**: DiJiang was designed for softmax attention approximation, not gated linear recurrence. The feature map $\phi(x) = D\exp(T \cdot \text{DCT}(x))$ may interact poorly with the multiplicative gate $\alpha_t$. Mitigation: ablate different feature map variants (with/without exp, with/without learned $D, T$).

3. **Activation function placement in ACDC**: Standard ACDC uses ReLU between factors. In the DCT-unified architecture, the activation must be compatible with the frequency domain. ReLU in frequency domain is not equivalent to ReLU in spatial domain. Mitigation: apply activation in spatial domain by inserting one IDCT→ReLU→DCT pair at the midpoint of the ACDC cascade.

4. **No tensor core usage for FFN**: ACDC's DCT + diagonal ops don't use tensor cores. If the model becomes compute-bound (large batch), the lack of tensor core usage may bottleneck. Mitigation: for compute-bound regimes, consider Monarch FFN instead of ACDC (Monarch uses BMM → tensor cores).

5. **Power-of-2 dimension constraint**: Fast DCT is most efficient for power-of-2 sizes. Non-power-of-2 model dimensions require padding. Mitigation: choose $d \in \{512, 1024, 2048, 4096\}$.

6. **Gradient flow through deep ACDC cascade**: $K = 12$ sequential diagonal-DCT factors may have vanishing gradients, especially through the frequency domain. Mitigation: identity initialization (diagonals near 1), residual connections every 4 factors, gradient clipping.

## Follow-up Experiments

1. **DCT fusion kernel implementation**: Write a Triton kernel that fuses the attention output (in feature space) → ACDC FFN → residual into a single launch, benchmarking the actual HBM savings.

2. **Hybrid ACDC + dense FFN**: Use ACDC for the first half of layers and dense SwiGLU for the second half. Test whether deeper layers (which learn more redundant features) tolerate ACDC better.

3. **Scaling to 3B+**: At larger model dimensions ($d = 4096$), the $O(d \log d)$ vs $O(d^2)$ advantage grows dramatically. Test whether the quality gap also grows or shrinks.

4. **Learned DCT basis**: Replace the fixed DCT with a learned orthogonal transform (parameterized via Cayley, trick 022). The attention and FFN would share a learned frequency basis, potentially improving quality at the cost of additional parameters.

5. **Combine with Proposal 051**: Use Monarch projections for $W_Q, W_K$ (via KS fused kernel) and ACDC for FFN. This stacks both efficiency gains: sub-quadratic projections AND sub-quadratic FFN.

## Human Review

(To be filled by reviewer)

## References

- Chen, Liu, Wang, Tian & Wang (2024). DiJiang: Efficient Large Language Models through Compact Kernelization. ICML 2024 Oral. arXiv:2403.19928.
- Moczulski, Denil, Appleyard & de Freitas (2016). ACDC: A Structured Efficient Linear Layer. ICLR 2016. arXiv:1511.05946.
- Yang, Wang, Shen, Panda & Kim (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Thangarasa, Gupta, Marshall, Li, Leong & DeVries (2024). Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers. NeurIPS 2024. arXiv:2406.16450.
- Huhtanen & Perämäki (2015). Factoring matrices into the product of circulant and diagonal matrices. J. Fourier Anal. Appl.
- Lee-Thorp, Ainslie, Eckstein & Ontanon (2021). FNet: Mixing Tokens with Fourier Transforms. arXiv:2105.03824.
- Dao & Gu (2024). Transformers are SSMs. ICML 2024.
- Hua, Dai, Liu & Le (2022). Transformer Quality in Linear Time. ICML 2022.

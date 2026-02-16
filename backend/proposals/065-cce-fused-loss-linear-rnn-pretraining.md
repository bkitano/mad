# CCE Fused Loss for Linear RNN Pretraining

**Status**: proposed
**Priority**: high
**Created**: 2026-02-16
**Based on**: [257-cut-cross-entropy-fused-lse], [221-kahan-compensated-summation-low-precision-training], [177-gla-secondary-chunking-log-space-gating], [203-gated-deltanet-chunkwise-wy-gating], [211-kda-constrained-dplr-delta-chunkwise]

## Hypothesis

Applying Cut Cross-Entropy (CCE) with Kahan summation to linear RNN pretraining (GLA, Gated DeltaNet, KDA) will yield a **disproportionately larger relative speedup** compared to its application in Transformers, because the loss layer consumes a **larger fraction of total training time and memory** in linear RNNs (where the sequence mixing layer is $O(Td^2)$ rather than $O(T^2d)$). Specifically, we predict:

1. **1.3–2× larger effective batch sizes** at fixed GPU memory (enabling 8–15% wall-clock training speedup via better GPU utilization)
2. The loss layer shifts from **25–40% of peak memory** (in linear RNNs with $V \geq 64K$) to **<1%**, making the projection layers the new bottleneck
3. CCE's gradient filtering provides an additional **1.5–2× backward speedup** on the loss layer specifically for fine-tuning, where softmax sparsity is highest

## Background

### The loss layer bottleneck is proportionally worse for linear RNNs

In softmax Transformers, the attention layer is the dominant cost: $O(T^2 d)$ FLOPs for self-attention vs. $O(T \cdot d \cdot |V|)$ FLOPs for the vocabulary projection. At sequence length $T = 2048$, $d = 2048$, $|V| = 128K$:

- Attention FLOPs per layer: $\sim 2T^2 d = 17.2 \times 10^9$
- Vocab projection FLOPs: $\sim 2Td|V| = 1.07 \times 10^{12}$

But with 24 layers, total attention FLOPs = $412 \times 10^9$ — comparable to the single vocab projection. The vocab projection is ~20% of total compute.

**For linear RNNs**, the attention-equivalent layer costs only $O(T d_k d_v)$ per layer ($\sim 67 \times 10^9$ for $d_k = d_v = 128$, 24 layers = $1.6 \times 10^9$). The vocab projection is now **40% or more of total forward FLOPs** and the **dominant memory consumer**.

### Memory breakdown for 1.3B linear RNN with $|V| = 128K$

| Component | Memory (BF16) | % of Total |
|-----------|---------------|------------|
| Logit matrix ($B \times T \times |V|$) | $8 \times 2048 \times 128K \times 2 = 4$ GB | 35–45% |
| Chunkwise states (24 layers) | $24 \times \lceil T/C \rceil \times d_k \times d_v \times 2 \approx 0.2$ GB | 2–3% |
| Activations (projections, FFN) | $\sim 4$ GB | 35–40% |
| Model weights | $\sim 2.6$ GB | 20–25% |

The logit matrix is the **single largest activation tensor** in the entire model. CCE eliminates it entirely.

### Why no one has studied this specifically

CCE (Apple, ICLR 2025) was validated on Transformer-based LLMs (Gemma 2, Llama, Mistral). Fused linear cross-entropy (Liger Kernel, JonasGeiping) is available as a library. Flash-linear-attention (FLA) includes a fused cross-entropy option. However, **no systematic study exists** of:

1. The **relative benefit** of CCE for linear RNNs vs. Transformers (where the fraction of compute/memory in the loss layer differs)
2. The **interaction between CCE and chunkwise kernel memory** — CCE's memory savings may enable larger chunk sizes $C$, which directly improve arithmetic intensity of the chunkwise scan
3. **Optimal batch size scaling** when CCE frees 4+ GB of activation memory — can we increase $B$ or $T$ to improve training efficiency?
4. The **Kahan summation requirement** for linear RNN pretraining — linear RNN gradients may have different numerical characteristics than Transformer gradients in the loss layer

## Related Work

- **Cut Cross-Entropy (Wijmans et al., ICLR 2025)**: Introduced CCE for Transformer LLMs. Showed 24,000× memory reduction, no convergence loss with Kahan summation. **Only tested on softmax Transformers.** Our work studies the proportional benefit for linear RNNs.

- **Liger Kernel Fused Cross-Entropy (Hsu et al., 2024)**: Chunked fused linear + cross-entropy. Available in FLA. Achieves 60–80% memory reduction (less than CCE's 99.99%). **No analysis of relative benefit for linear RNNs.**

- **FLA fused_cross_entropy option**: The flash-linear-attention library includes a fused cross-entropy flag, but it defaults to disabled and no published benchmarks exist comparing its impact on total training throughput for GLA/GDN/KDA models.

- **Proposals 050, 054 (FP8/INT4 chunkwise quantization)**: These optimize the chunkwise scan kernel. **CCE is orthogonal** — it optimizes the loss layer, not the sequence mixing layer. The two compose multiplicatively: CCE reduces memory → enables larger batch → FP8/INT4 increases throughput per batch.

- **Proposal 063 (MFA shared projections)**: Optimizes the Q/K/V projection FLOPs. Also orthogonal to CCE. With CCE + MFA, both the projection and loss layers are optimized, leaving only the chunkwise scan as the bottleneck.

No directly related work found studying Cut Cross-Entropy specifically in linear RNN architectures.

## Mathematical Formulation

**Standard Loss Layer (baseline):**

For the final hidden states $\mathbf{h} \in \mathbb{R}^{BT \times d}$ from the linear RNN stack, the output logits and loss are:

$$
\mathbf{Z} = \mathbf{h} \mathbf{W}_{\text{vocab}}^\top \in \mathbb{R}^{BT \times |V|} \quad \text{(materialized in HBM: } O(BT|V|) \text{ memory)}
$$

$$
\ell = -\frac{1}{BT} \sum_{i=1}^{BT} \log \frac{\exp(Z_{i, x_i})}{\sum_{j=1}^{|V|} \exp(Z_{ij})}
$$

**With CCE (proposed):**

$$
\ell = -\frac{1}{BT} \sum_{i=1}^{BT} \left[ \underbrace{\mathbf{W}_{x_i}^\top \mathbf{h}_i}_{\text{indexed dot product}} - \underbrace{\log \sum_{j=1}^{|V|} \exp(\mathbf{W}_j^\top \mathbf{h}_i)}_{\text{fused linear-LSE}} \right]
$$

The logit matrix $\mathbf{Z}$ is **never materialized**. Both terms are computed via tiled matmul + reduction in SRAM.

**Memory freed by CCE for linear RNN training:**

$$
\Delta M = BT|V| \times \text{bytes\_per\_element}
$$

For $B = 8$, $T = 2048$, $|V| = 128K$, BF16:

$$
\Delta M = 8 \times 2048 \times 128000 \times 2 = 4.19 \text{ GB}
$$

**Batch size increase enabled:**

Per-sample memory for the chunkwise linear RNN forward pass (excluding loss):

$$
M_{\text{per\_sample}} \approx L \times T \times d \times \text{activation\_factor}
$$

For $L = 24$, $T = 2048$, $d = 2048$, activation factor $\approx 4$ (Q, K, V, output):

$$
M_{\text{per\_sample}} \approx 24 \times 2048 \times 2048 \times 4 \times 2 \text{ B} \approx 0.8 \text{ GB}
$$

CCE frees 4.19 GB → enables $\lfloor 4.19 / 0.8 \rfloor = 5$ additional samples, increasing batch from 8 to 13 ($\mathbf{1.6\times}$).

**Kahan summation for CCE:**

For pretraining stability, the CCE-Kahan variant tracks compensation:

$$
\text{LSE}_n^{\text{new}} = \log(\exp(\text{LSE}_n) + \exp(\text{lse\_block}))
$$

$$
c_n \mathrel{+}= (\text{LSE}_n^{\text{new}} - \text{LSE}_n) - \text{lse\_block} \quad \text{(Kahan error term)}
$$

This adds $O(BT)$ memory for the compensation buffer — negligible compared to the 4 GB saved.

**Key Variables:**
- $\mathbf{h} \in \mathbb{R}^{BT \times d}$ — final hidden states from linear RNN
- $\mathbf{W}_{\text{vocab}} \in \mathbb{R}^{|V| \times d}$ — vocabulary embedding/classifier matrix
- $|V|$ — vocabulary size (32K–256K)
- $B$ — batch size, $T$ — sequence length
- $\text{LSE}_n$ — log-sum-exp accumulator per token

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / Gated DeltaNet / KDA (any chunkwise linear RNN) |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Vocabulary | $|V| = 128K$ (Llama-style tokenizer) |
| Chunk size | $C = 64$ (baseline), $C = 128$ or $256$ (with CCE memory savings) |
| Loss layer | CCE-Kahan-FullC (pretraining) / CCE-Filter (fine-tuning) |
| Parameters | ~1.3B |

### Baseline

1. **Standard cross-entropy (torch.compile)**: Full logit materialization, $O(BT|V|)$ memory. This is what FLA uses by default.
2. **Liger Kernel fused cross-entropy**: Chunked fusion, 60–80% memory reduction. Partial materialization.
3. **CCE-Kahan (proposed)**: No logit materialization, Kahan summation for stability.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Peak memory | $< 0.6 \times$ baseline | `torch.cuda.max_memory_allocated` |
| Batch size at 80GB | $\geq 1.5 \times$ baseline | Max $B$ before OOM |
| Training throughput | $\geq 1.08 \times$ baseline (same $B$) | Tokens/sec on H100 |
| Training throughput | $\geq 1.15 \times$ baseline (scaled $B$) | Tokens/sec at max batch |
| Perplexity | $\leq$ baseline (within 0.1 ppl) | WikiText-103 / FineWeb validation |
| Loss convergence | Match baseline | Training loss curves overlay |
| Backward speed (loss only) | $\geq 1.5 \times$ baseline (fine-tuning) | Kernel time via NSight |

### Estimated Compute

- **MVE**: < 30 minutes on single H100 (memory profiling + throughput benchmark)
- **Phase 1** (convergence validation, 350M model, 10B tokens): ~80 GPU-hours
- **Phase 2** (scaling study, 1.3B model, batch size sweep): ~50 GPU-hours
- **Total**: ~130 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- CCE eliminates the loss layer as the memory bottleneck for linear RNN training, freeing 4+ GB of activation memory
- The freed memory enables 1.5–2× larger batch sizes, translating to 8–15% training throughput improvement via better GPU utilization (more tokens per gradient step)
- Alternatively, the freed memory enables larger chunk sizes ($C = 128$ or $256$), which increases the arithmetic intensity of the chunkwise scan kernel, yielding 5–10% throughput improvement even at fixed batch size
- The relative benefit is larger for linear RNNs than Transformers (where attention activations are the bottleneck, not the loss layer)
- CCE-Kahan matches standard cross-entropy convergence exactly (as shown in the original paper for Transformers)

**If hypothesis is wrong:**
- If the batch size increase doesn't translate to throughput gains: GPU compute is already saturated at the baseline batch size, and the extra memory headroom doesn't help. This would tell us that the chunkwise scan, not memory, is the bottleneck. **Useful negative result.**
- If CCE-Kahan shows precision loss specific to linear RNNs: the gradient flow through the linear RNN stack creates different numerical characteristics in the loss layer that interact with CCE's tiled computation. This would motivate investigating precision requirements.
- If Liger Kernel's fused cross-entropy already captures most of the benefit: the full CCE approach (no materialization at all) provides only marginal improvement over partial materialization. This would be a useful calibration.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 256$, $d_k = 64$, $d_v = 128$, 4 heads, ~10M params
- **Vocabulary**: $|V| = 32K$ (reduced to keep model small)
- **Task**: Autoregressive LM on 5M tokens of WikiText-103
- **Loss variants**: (1) Standard CE, (2) Liger fused CE, (3) CCE-Kahan
- **Measurements**: Peak memory, throughput (tokens/sec), final loss value
- **Compute**: Single GPU, < 30 minutes for all 3 variants

### Success Criteria
- CCE reduces peak memory by $\geq 50\%$ compared to standard CE
- CCE throughput is within 5% of standard CE at the same batch size (no slowdown)
- With increased batch size, CCE achieves $\geq 10\%$ higher throughput
- Final loss value matches standard CE within 0.01 nats (convergence equivalence)

### Failure Criteria
- If CCE is slower than standard CE at the same batch size: the tiled matmul + atomic LSE overhead exceeds the HBM savings. Kill the idea — CCE's benefit is memory-only and doesn't improve throughput.
- If convergence diverges between CCE and standard CE: numerical stability issue specific to linear RNNs. Debug with precision analysis before killing.

### Why This Test Is Sufficient
- The memory reduction is a mechanical property of CCE — it doesn't depend on model scale. If it saves 50% at 10M params, it saves proportionally more at 1.3B (because $|V|$ dominates).
- Throughput comparison at fixed vs. scaled batch size directly tests both hypotheses (speed equivalence and batch scaling benefit).
- Convergence equivalence at small scale implies equivalence at large scale (the numerical computation is identical — just tiled differently).

## Memory Access Pattern Analysis

**Standard CE forward:**
1. Load $\mathbf{h}$ from HBM ($BT \times d$): sequential, coalesced
2. Compute $\mathbf{Z} = \mathbf{h} W^\top$: GEMM (tensor core), writes $BT \times |V|$ to HBM
3. Load $\mathbf{Z}$ for softmax: $BT \times |V|$ read from HBM
4. Compute loss: elementwise, negligible

**CCE forward:**
1. Load $\mathbf{h}$ from HBM (same): sequential, coalesced
2. **Never allocate** $\mathbf{Z}$: tiled GEMM blocks of size $V_B \times N_B$ computed in SRAM
3. Each SRAM tile: $(V_B \times D_B) \times (D_B \times N_B)$ matmul (tensor core)
4. LSE reduction: atomic log-add-exp across tiles (low contention)
5. Write only: $BT$ scalar loss values and $BT$ LSE values to HBM

**Net HBM traffic reduction:**

| Operation | Standard CE | CCE |
|-----------|-------------|-----|
| Forward write | $BT \times |V| \times 2$ B | $BT \times 2 \times 4$ B |
| Forward read | $BT \times |V| \times 2$ B (softmax) | 0 |
| Backward read | $BT \times |V| \times 2$ B (reuse logits) | Recomputed in SRAM |
| **Total HBM traffic** | $3 \times BT \times |V| \times 2$ | $\sim BT \times d \times 2$ |

For $B = 8$, $T = 2048$, $|V| = 128K$: Standard = $12.6$ GB, CCE $\approx 0.067$ GB. **188× reduction in HBM traffic for the loss layer.**

## Parallelism Analysis

- **Tiled GEMM**: Standard blocked matmul structure — saturates all SMs with independent $(V_B, N_B)$ tile work
- **Atomic LSE**: One atomic per $(V_B, N_B)$ tile per token — low contention for $|V|/V_B \gg 1$
- **No warp divergence**: All threads within a tile execute the same matmul + reduce pattern
- **Tensor core utilization**: The matmul tiles are standard $(V_B \times D_B) \times (D_B \times N_B)$ GEMMs — full WGMMA utilization on H100
- **No sequential bottleneck**: All tiles independent in forward; backward has tile-level gradient filtering

## Theoretical Analysis

**FLOP comparison:**

| Operation | Standard CE | CCE |
|-----------|-------------|-----|
| Forward matmul | $2 \times BT \times d \times |V|$ | $2 \times BT \times d \times |V|$ (same) |
| Softmax | $O(BT \times |V|)$ | $O(BT \times |V|)$ (fused into tiles) |
| Backward matmul | $2 \times BT \times d \times |V|$ | $2 \times BT \times d \times |V|$ (same) |
| Backward filtering | — | Skip $> 99.98\%$ of tiles (3.5× speedup) |
| **Total FLOPs** | $4BT \cdot d \cdot |V|$ | $4BT \cdot d \cdot |V|$ (forward) |
| | | $\sim BT \cdot d \cdot |V| / 100$ (backward, filtered) |

**FLOPs are identical in forward; backward has up to 3.5× reduction with gradient filtering (fine-tuning only).**

The real gain is **memory**, not FLOPs. Memory savings → larger batch → higher utilization → wall-clock speedup.

**Relative impact: linear RNN vs. Transformer:**

| Metric | Transformer (1.3B) | Linear RNN (1.3B) |
|--------|-------------------|-------------------|
| Sequence mixing FLOPs/layer | $4T^2 d = 34.4 \times 10^9$ | $2T d_k d_v = 0.067 \times 10^9$ |
| Total mixing FLOPs (24L) | $825 \times 10^9$ | $1.6 \times 10^9$ |
| Vocab proj FLOPs | $1.07 \times 10^{12}$ | $1.07 \times 10^{12}$ |
| **Vocab proj as % of total** | **~20%** | **~40%** |
| Logit memory as % of activations | ~15% | ~35% |

**The loss layer is proportionally 2× more important in linear RNNs**, making CCE's savings proportionally 2× more impactful.

## Risks & Limitations

1. **Kahan summation adds complexity**: The CCE-Kahan variant requires compensation buffers and modified accumulation logic. While the memory cost is negligible ($O(BT)$), the kernel implementation is more complex. **Mitigation**: Use Apple's open-source CCE implementation (github.com/apple/ml-cross-entropy) which includes the Kahan variant.

2. **Gradient filtering inapplicable to pretraining**: The 3.5× backward speedup from gradient filtering only applies to fine-tuning (where softmax is very sparse). For pretraining on diverse data, all vocabulary entries may receive meaningful gradients. **Mitigation**: Use CCE-Kahan-FullC for pretraining (no filtering), CCE-Filter for fine-tuning.

3. **Batch size scaling may not translate to throughput**: If the GPU is already compute-saturated at the baseline batch size, adding more samples doesn't help — they just queue. **Mitigation**: Profile compute utilization at baseline to determine headroom. If utilization is >90%, use the freed memory for larger chunk sizes instead.

4. **Integration with existing FLA codebase**: FLA already has a fused cross-entropy option. CCE may require replacing this with a different implementation (Apple's ml-cross-entropy or custom). **Mitigation**: FLA's fused CE is a simple flag; swapping implementations requires changing one function call.

5. **Vocabulary size dependency**: CCE's benefit scales with $|V|$. For models with small vocabularies ($|V| = 32K$), the memory savings are less dramatic (1 GB instead of 4 GB). **Mitigation**: Test across vocabulary sizes to establish the crossover point.

## Follow-up Experiments

1. **CCE + larger chunk sizes**: Instead of increasing batch size, use the freed memory to increase the chunkwise scan's chunk size from $C = 64$ to $C = 256$. This increases arithmetic intensity of the scan kernel, potentially yielding a different (and compounding) throughput gain.

2. **CCE + FP8 chunkwise training (Proposal 050)**: CCE frees memory headroom that can be used for FP8 master weight buffers or larger activation buffers for mixed-precision. Test whether the combined approach (CCE + FP8) gives multiplicative speedup.

3. **CCE for very large vocabularies**: Test with $|V| = 256K$ (byte-level tokenizers, multilingual models) where the logit matrix would consume 8+ GB — CCE's benefit should be even more dramatic.

4. **Gradient filtering analysis for linear RNN fine-tuning**: Profile the softmax sparsity pattern during fine-tuning of a pretrained linear RNN. If sparsity matches Transformer baselines ($> 99.98\%$), the 3.5× backward speedup applies directly.

5. **CCE + MFA projections (Proposal 063)**: With CCE eliminating the loss layer bottleneck and MFA reducing projection FLOPs, profile what remains as the new bottleneck for linear RNN training.

## Human Review



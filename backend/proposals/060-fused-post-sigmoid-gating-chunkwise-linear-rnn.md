---
status: ongoing
priority: high
created: 2026-02-16
based_on: post-attention-sigmoid-gating (094), gla-secondary-chunking-log-space-gating (177), tfla-two-level-tiled-chunkwise-parallelism (158), epilogue-visitor-tree-fusion (039), flashrnn-io-aware-fused-recurrence (212), bilinear-gating-glu (005), smooth-swiglu-per-channel-fp8-stability (227), kda-constrained-dplr-delta-chunkwise (211), input-dependent-gating (065)
experiment_number: 060
experiment_log: experiment-log-060.md
---

# Fused Post-Sigmoid Gating for Chunkwise Linear RNN Readout

## Hypothesis

Adding a **post-readout sigmoid gate** (from trick 094) to chunkwise linear RNNs (GLA, KDA, mLSTM) — where the gate is **fused into the chunkwise kernel's epilogue** to avoid any additional HBM round-trip — will improve language modeling perplexity by $0.3$–$0.8$ points at 370M–1.3B scale with **zero wall-clock overhead** (< 1% latency increase), because: (1) the gate breaks the low-rank bottleneck in the readout path $\boldsymbol{o}_t = \boldsymbol{q}_t^\top \boldsymbol{S}_t$ → $W_O$ (identical to how it helps softmax attention), (2) the gate introduces learnable sparsity that suppresses irrelevant state readouts, and (3) the elementwise sigmoid + multiply fuses trivially into the chunkwise kernel's output epilogue without extra memory traffic.

## Background

### Post-sigmoid gating: a proven win for softmax attention

Trick 094 (post-attention sigmoid gating) was the **NeurIPS 2025 Best Paper** and has been adopted in Qwen3-Next. The mechanism is simple: after computing the attention output $Y = \text{Attn}(Q, K, V) \in \mathbb{R}^{n \times d_k}$ per head, apply an elementwise gate:

$$
Y' = Y \odot \sigma(X W_\theta)
$$

where $W_\theta \in \mathbb{R}^{d_{\text{model}} \times d_k}$ is a learned projection and $\sigma$ is sigmoid.

The key insight is that this gate breaks the **low-rank bottleneck** created by the consecutive $W_V W_O$ linear projection in GQA-style attention: without the gate, the attention output undergoes a rank-$d_k$ linear mapping; with the gate, a position-dependent nonlinearity intervenes, enabling higher effective rank.

### The same bottleneck exists in linear RNNs

In GLA/KDA/mLSTM, the readout path is:

$$
\boldsymbol{o}_t = \boldsymbol{q}_t^\top \boldsymbol{S}_t \in \mathbb{R}^{d_v}, \quad y_t = W_O \boldsymbol{o}_t \in \mathbb{R}^d
$$

This has the **same low-rank bottleneck**: the state $\boldsymbol{S}_t$ is read via a rank-1 query ($\boldsymbol{q}_t$), producing a $d_v$-dimensional output that is then linearly projected to $d$. The composition $\boldsymbol{q}_t^\top \cdot W_O$ is a rank-$d_v$ mapping from the state to the output — identical to the $W_V W_O$ bottleneck in attention.

A post-readout sigmoid gate would break this bottleneck:

$$
\boldsymbol{o}_t' = \boldsymbol{o}_t \odot \sigma(x_t W_g), \quad y_t = W_O \boldsymbol{o}_t'
$$

### Why hasn't this been tried?

Post-sigmoid gating was introduced specifically for softmax attention and validated on Transformer architectures. The linear RNN community has focused on **state transition improvements** (gating, delta rule, state expansion) rather than readout improvements. Existing proposals in this collection (001-058) that touch readout (proposal 009) propose sigmoid gating but haven't analyzed the kernel fusion opportunity or the wall-clock cost.

### The kernel fusion opportunity

The key GPU insight: in the chunkwise kernel, the output $\boldsymbol{O}_{[n]}$ for each chunk is computed via:

$$
\boldsymbol{O}_{[n]} = \underbrace{(\boldsymbol{Q}_{[n]} \odot \Lambda_{[n]}) \boldsymbol{S}_{[n-1]}}_{\text{inter-chunk}} + \underbrace{P_{[n]} \boldsymbol{V}_{[n]}}_{\text{intra-chunk}}
$$

This output is written to HBM at the end of the chunkwise kernel. If we apply the sigmoid gate **before** writing to HBM:

$$
\boldsymbol{O}_{[n]}' = \boldsymbol{O}_{[n]} \odot \sigma(X_{[n]} W_g)
$$

Then the gate computation ($\sigma(X_{[n]} W_g)$) can be:
1. **Precomputed** outside the chunkwise kernel (a single GEMM: $X W_g$, same as the $Q, K, V$ projections)
2. **Loaded** into the chunkwise kernel as an additional input tensor alongside $Q, K, V$
3. **Applied** as an elementwise multiply in the kernel's output epilogue — **zero additional HBM round-trips**

This is exactly the **epilogue visitor tree fusion** pattern from trick 039: the gate is a leaf visitor in the kernel's epilogue DAG.

### Why this gives real GPU speedup (or at least zero slowdown)

1. **Would I bet $100 this is faster than baseline on A100?** Yes for quality; the wall-clock should be identical. The gate is a single elementwise multiply fused into the kernel epilogue. The $X W_g$ projection is computed in parallel with the $Q, K, V$ projections and adds < 2% to the projection FLOP cost.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes. In the Triton TFLA kernel, add one line before the final `tl.store`: `output = output * tl.sigmoid(gate_scores)` where `gate_scores` is loaded alongside Q from the same tile.

3. **Does it reduce HBM bandwidth?** Not directly, but it improves quality at identical bandwidth. The gate scores are loaded once (same HBM read pattern as Q/K/V) and the output is written once (same as baseline).

## Related Work

- **[Post-Attention Sigmoid Gating (Qiu et al., NeurIPS 2025 Best Paper)](https://arxiv.org/abs/2505.06708)**: Introduced sigmoid gating after softmax attention. Validated on Transformers at 1.7B dense and 15B MoE. **Our approach**: Applies the same mechanism to linear RNN readout, with a kernel fusion strategy specific to chunkwise computation.
- **[GLA (Yang et al., ICML 2024)](https://arxiv.org/abs/2312.06635)**: Chunkwise kernel with secondary chunking. No post-readout gating. **Our approach**: Adds sigmoid gating to GLA's output, fused into the kernel epilogue.
- **[Gated DeltaNet (Yang et al., ICLR 2025)](https://arxiv.org/abs/2412.06464)**: Uses gating on the state transition but not on the readout. **Our approach**: Adds output gating orthogonal to the state transition gating.
- **[RWKV-7 (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)**: Uses a output gate $g_t$ multiplied on the attention result, parameterized as $\sigma(x_t W_g)$. This is similar but RWKV-7's gate is applied **after** LayerNorm and before $W_O$, whereas our proposal applies it **before** LayerNorm. Also, RWKV-7 uses a custom WKV7 CUDA kernel, not a chunkwise matmul kernel — the fusion strategy differs.
- **[TFLA (Beck et al., NeurIPS 2025)](https://arxiv.org/abs/2503.14376)**: Two-level tiled chunkwise kernel. No output gating. **Our approach**: Extends TFLA's kernel with an epilogue gate.
- **[CUTLASS Epilogue Visitor Tree (NVIDIA)](https://github.com/NVIDIA/cutlass)**: Framework for fusing elementwise operations into GEMM epilogues. **Our approach**: Uses this pattern for the sigmoid gate fusion.

**Gap**: While RWKV-7 uses an output gate similar to our proposal, no existing work has: (a) systematically evaluated post-sigmoid gating on the **GLA/KDA/mLSTM family** of chunkwise linear RNNs, (b) fused the gate into the **chunkwise kernel epilogue** for zero overhead, or (c) ablated the gate position (before vs after LayerNorm, before vs after $W_O$).

## Mathematical Formulation

### Standard Chunkwise Linear RNN Readout (Baseline)

For GLA, the per-chunk output is:

$$
\boldsymbol{O}_{[n]} = \underbrace{(\boldsymbol{Q}_{[n]} \odot \Lambda_{[n]}) \boldsymbol{S}_{[n-1]}}_{\text{inter-chunk: } (C, d_k) \times (d_k, d_v) \to (C, d_v)} + \underbrace{\text{Tril}(P_{[n]}) \boldsymbol{V}_{[n]}}_{\text{intra-chunk: } (C, C) \times (C, d_v) \to (C, d_v)}
$$

The multi-head output is then:

$$
y_t = W_O [\boldsymbol{o}_t^{(1)}; \ldots; \boldsymbol{o}_t^{(H)}] \in \mathbb{R}^d
$$

### Post-Sigmoid Gated Readout (Proposed)

**Gate projection** (computed once, outside the chunkwise kernel):

$$
G = \sigma(X W_g) \in \mathbb{R}^{T \times H \times d_v}
$$

where $W_g \in \mathbb{R}^{d \times (H \cdot d_v)}$ is a single linear projection from the pre-norm input $X \in \mathbb{R}^{T \times d}$.

**Gated output** (fused into chunkwise kernel epilogue):

$$
\boldsymbol{O}_{[n]}' = \boldsymbol{O}_{[n]} \odot G_{[n]} \in \mathbb{R}^{C \times d_v}
$$

**Output projection** (unchanged):

$$
y_t = W_O [\boldsymbol{o}_t'^{(1)}; \ldots; \boldsymbol{o}_t'^{(H)}]
$$

### Gate Initialization

Following the original paper's recommendation: **zero-initialize** $W_g$ so that $\sigma(0) = 0.5$, making the gate start as a uniform $0.5\times$ scaling. This ensures the gated model starts near the baseline behavior (up to a constant factor absorbed by $W_O$).

### Key Insight: The Gate is Query-Dependent, Not State-Dependent

The gate $G_t = \sigma(x_t W_g)$ depends only on the current input $x_t$, not on the state $\boldsymbol{S}_t$ or the readout $\boldsymbol{o}_t$. This means:
1. It can be **precomputed** before the chunkwise kernel (in the same pass as $Q, K, V$ projections)
2. It adds **no sequential dependency** — the chunkwise kernel's parallelism is unchanged
3. It acts as an **input-dependent feature selector** on the readout, analogous to the gate in SwiGLU FFNs

### Key Variables

- $X \in \mathbb{R}^{T \times d}$ — pre-norm input hidden states
- $W_g \in \mathbb{R}^{d \times (H \cdot d_v)}$ — gate projection matrix (new parameter)
- $G \in \mathbb{R}^{T \times H \times d_v}$ — precomputed gate scores
- $\boldsymbol{O}_{[n]} \in \mathbb{R}^{C \times d_v}$ — per-chunk, per-head output from chunkwise kernel
- $\boldsymbol{O}_{[n]}' \in \mathbb{R}^{C \times d_v}$ — gated output (written to HBM)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / KDA / mLSTM + Post-Sigmoid Gate |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 16$ |
| Head dim | $d_k = d_v = 128$ |
| Chunk size | $C = 64$–$256$ |
| Gate | Per-head elementwise sigmoid, zero-initialized |
| New params per layer | $d \times (H \cdot d_v) = 2048 \times 2048 = 4.2M$ |

**Parameter overhead**: $4.2M \times 24 = 100M$ new parameters for a 1.3B model (~$7.7\%$ increase). This is comparable to the overhead in Transformer gated attention (Qiu et al. report $201M$ new params for a 15B model = $1.3\%$). The relative overhead is higher here because the base model is smaller, but the absolute cost is manageable.

**Alternative — headwise gating** (lower parameter overhead):

$$
G_t^{(h)} = \sigma(x_t w_g^{(h)}) \in \mathbb{R}^{1} \quad \text{(scalar per head)}
$$

with $w_g^{(h)} \in \mathbb{R}^{d}$. This adds only $d \times H = 2048 \times 16 = 32K$ new params per layer — negligible. Qiu et al. show headwise gating captures most of the benefit (Table 2 in their paper).

### Baseline

1. **GLA (no gate)**: Standard chunkwise GLA, fla library.
2. **GLA + separate sigmoid gate**: Gate applied as a separate unfused kernel after the chunkwise kernel. Tests the quality benefit without the fusion benefit.
3. **KDA (no gate)**: Standard Kimi Delta Attention.
4. **RWKV-7**: Has output gating but via a different kernel architecture (WKV7).

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity (370M, 10B tokens) | $\leq$ baseline $- 0.3$ | Validation PPL on SlimPajama |
| Perplexity (1.3B, 15B tokens) | $\leq$ baseline $- 0.5$ | Validation PPL on SlimPajama |
| Wall-clock throughput (fused) | $\geq 0.99\times$ baseline | Tokens/sec on H100 |
| Wall-clock throughput (unfused) | measured for comparison | Tokens/sec (expect 2-5% slower) |
| MQAR accuracy | $\geq$ baseline | Associative recall accuracy |
| Attention sink metric | Reduced vs baseline | Entropy of first-token attention weight |

### Estimated Compute

**MVE**: < 1 GPU-hour (tiny model, synthetic tasks)
**Phase 1 — Quality (370M)**: ~60 GPU-hours
**Phase 2 — Full (1.3B)**: ~200 GPU-hours
**Phase 3 — Ablations**: ~80 GPU-hours (gate positions, headwise vs elementwise, etc.)
**Total**: ~340 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**

- **Perplexity improvement**: $0.3$–$0.8$ points at 370M–1.3B scale, matching the improvements seen in Transformer gated attention.
- **Zero wall-clock overhead** with fused kernel: the gate is a single `multiply(output, sigmoid(gate_scores))` in the kernel epilogue — 2 FLOPs per element, overlapped with the output store.
- **Sparsity in gate values**: Following the observations in the original paper, gate scores should concentrate near 0 (mean ~$0.1$–$0.2$), creating effective sparsity that suppresses irrelevant readout dimensions. This could be especially beneficial for linear RNNs where the state $\boldsymbol{S}_t$ accumulates all past information — the gate selectively filters what's relevant for the current query.
- **Attention sink elimination**: Linear RNNs can develop "state sinks" where certain state dimensions dominate regardless of context. The gate can suppress these, analogous to how it eliminates attention sinks in Transformers.

**If hypothesis is wrong:**

- **No perplexity improvement**: Linear RNNs don't suffer from the same low-rank bottleneck because their state is already higher-rank than softmax attention's weighted average. **Learn**: The gate's value is specific to softmax attention's properties.
- **Gate converges to $\sigma(0) = 0.5$ everywhere**: The gate doesn't learn useful patterns. **Learn**: The readout $\boldsymbol{o}_t = \boldsymbol{q}_t^\top \boldsymbol{S}_t$ is already sufficiently diverse that no feature selection is needed. Try applying the gate in the FFN layer instead (between the chunkwise output and the FFN input).
- **Quality degrades**: The $0.5\times$ scaling from zero-initialized gates hurts early training. **Mitigation**: Initialize $W_g$ to produce $\sigma(\cdot) \approx 1$ (bias term with positive initialization).

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 128$, $H = 4$, $d_k = d_v = 32$ (~200K params + ~16K gate params)
- **Task**: (a) WikiText-2 language modeling (first 1M tokens), (b) Synthetic MQAR ($T = 256$, 8 pairs)
- **Gate variant**: Headwise scalar gate (minimal parameter overhead)
- **Compute**: Single GPU, < 10 minutes

### Implementation (no kernel fusion needed for MVE)

```python
class GatedGLALayer(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.gla = GLALayer(d_model, n_heads, d_head)  # from fla library
        # Post-readout gate (headwise)
        self.W_gate = nn.Linear(d_model, n_heads * d_head, bias=False)
        nn.init.zeros_(self.W_gate.weight)  # zero-init for residual-friendly start

    def forward(self, x):
        # Standard GLA readout
        o = self.gla(x)  # (B, T, d_model) — after W_O

        # Post-readout gate (applied before W_O ideally, but after for simplicity in MVE)
        gate = torch.sigmoid(self.W_gate(x))  # (B, T, n_heads * d_head)
        o_gated = o * gate  # elementwise

        return o_gated
```

### Success Criteria
- WikiText-2 perplexity improves by $\geq 0.5$ points over ungated baseline (at convergence)
- MQAR accuracy is $\geq$ baseline (gate doesn't hurt retrieval)
- Gate activation statistics show concentration near 0 (mean $< 0.3$) — indicating learned sparsity

### Failure Criteria
- Perplexity $\geq$ baseline: gate provides no benefit for linear RNN readout
- Gate activations are uniformly $\approx 0.5$: gate doesn't learn useful patterns
- Training diverges: zero-initialization causes instability (try constant-init at 1.0)

### Why This Test Is Sufficient
- The low-rank bottleneck is a property of the readout mechanism, not the model size. If the gate helps at tiny scale, it will help at larger scale (the bottleneck gets worse with more heads, as GQA-style sharing increases).
- WikiText-2 perplexity directly measures language modeling quality — the primary target.
- The MVE tests the algorithmic benefit without kernel fusion. The fusion is a pure engineering optimization that doesn't affect quality — it can be validated separately via a kernel microbenchmark.

## Memory Access Pattern Analysis

**Gate precomputation** ($G = \sigma(X W_g)$):
- Standard GEMM: $(T, d) \times (d, H \cdot d_v)$. Fully coalesced, tensor-core compatible.
- Can be **fused with the Q projection**: $[Q; G] = X [W_Q; W_g]$ — a single wider GEMM.
- Memory: $T \times H \times d_v \times 2$ bytes in BF16. For $T = 4096, H = 16, d_v = 128$: 16 MB — loaded once from HBM into the chunkwise kernel.

**Gate application in kernel epilogue**:
- The gate $G_{[n]}$ for chunk $n$ is a $(C, d_v)$ tile — same shape as the output $\boldsymbol{O}_{[n]}$.
- Loaded from HBM (or streamed via TMA) alongside the output store.
- The multiply-and-store replaces a bare store: `tl.store(out_ptr, output * sigmoid_gate)` vs `tl.store(out_ptr, output)`.
- **No additional HBM round-trip**: The gate scores are read once, the output is written once.

**Arithmetic intensity**: The gate application adds $2 \times C \times d_v$ FLOPs (multiply + sigmoid lookup) to the kernel epilogue. For $C = 64, d_v = 128$: 16K FLOPs. The intra-chunk matmuls are $\sim 1.5M$ FLOPs. Overhead: $< 1.1\%$.

## Parallelism Analysis

- **No warp divergence**: Elementwise operation, same across all threads.
- **No sequential dependency**: Gate depends on $x_t$ (precomputed), not on $\boldsymbol{S}_t$ or $\boldsymbol{O}$.
- **Tensor core mapping**: The gate GEMM ($X W_g$) uses tensor cores. The elementwise multiply does not, but it's in the epilogue (overlapped with store).
- **SM utilization**: Identical to baseline — the gate adds no new thread blocks or synchronization.

## Theoretical Analysis

| Operation | Baseline (no gate) | With fused gate |
|-----------|--------------------|-----------------|
| Projection FLOPs | $3 d^2$ per token | $3 d^2 + d \cdot H \cdot d_v$ = $4 d^2$ |
| Chunkwise kernel FLOPs | $O(C^2 d_k + C d_k d_v)$ | Same + $O(C d_v)$ |
| Output write | $T \times d_v$ per head | Same (gate fused into write) |
| New parameters | $0$ | $d \times H \times d_v$ per layer |

**Projection overhead**: One additional GEMM of the same size as $W_Q$ — $\sim 33\%$ increase in projection FLOPs. Since projections are typically $\sim 40\%$ of total layer FLOPs, this is $\sim 13\%$ layer-level overhead.

**However**: The gate GEMM can be fused with the Q projection into a single wider GEMM:

$$
[Q; G] = X [W_Q; W_g] \in \mathbb{R}^{T \times 2(H \cdot d_k)}
$$

The wider GEMM has better arithmetic intensity and only marginally increases GEMM time (GEMMs are compute-bound, and the wider output dimension doesn't increase HBM reads of $X$). Expected overhead: $\sim 5$–$8\%$ for the projection, near-zero for the chunkwise kernel.

## Risks & Limitations

1. **Parameter overhead**: The elementwise gate adds $d \times H \times d_v$ parameters per layer. For $d = 2048, H = 16, d_v = 128$: $4.2M$ per layer, $100M$ total for 24 layers. This is manageable but non-trivial for a 370M baseline. **Mitigation**: Use headwise scalar gate ($d \times H = 32K$ per layer) or low-rank gate ($d \times r + r \times H \times d_v$).

2. **Gate position**: Applying the gate **after** the full multi-head concatenation and $W_O$ projection (as in the MVE) is suboptimal — the original paper shows the gate should be applied **per-head, before $W_O$**. The fused kernel version correctly applies per-head gating. The MVE approximation may understate the benefit.

3. **RWKV-7 already does this**: RWKV-7 has an output gate $g_t \odot p_t$ applied to the readout. However, RWKV-7's gate uses a **SiLU** activation (not sigmoid), and it's applied **after** LayerNorm of the readout, not before. The comparison with RWKV-7 is interesting but the architectures differ enough that results may not transfer directly.

4. **Interaction with SwiGLU FFN**: Both the post-readout gate and the SwiGLU FFN use sigmoid/SiLU-style gating. There may be diminishing returns from double gating. **Mitigation**: Test with and without SwiGLU to isolate the gate's contribution.

5. **FP8 stability**: If combined with proposal 050 (FP8 chunkwise training), the gate multiplication in the epilogue introduces another potential source of precision loss. **Mitigation**: Apply gate in BF16 after FP8 dequantization of the chunkwise output.

## Follow-up Experiments

1. **Gate position ablation**: Compare gate applied (a) after chunkwise output, before $W_O$ (our proposal), (b) after $W_O$ but before residual, (c) after LayerNorm (like RWKV-7). Measure perplexity and gate sparsity for each.

2. **Gate architecture ablation**: Compare (a) headwise scalar $\sigma(x_t w_g^{(h)})$, (b) elementwise $\sigma(x_t W_g)$ per head, (c) low-rank $\sigma(x_t U_g V_g^{(h)})$ with shared low-rank basis.

3. **Combine with SO-KDA (proposal 059)**: Apply both the second-order key metric AND post-readout gating. The key metric improves the state transition; the gate improves the readout. These are orthogonal improvements.

4. **Fused kernel implementation**: Implement the gate fusion in Triton (extend the TFLA kernel's epilogue) and CUTLASS (epilogue visitor pattern). Benchmark wall-clock vs unfused baseline.

5. **Gate sparsity for inference speedup**: If gate values concentrate near 0, use the gate as a **dynamic pruning mask** during inference: skip state dimensions where the gate is $< \epsilon$. This could reduce the effective readout cost from $O(d_k d_v)$ to $O(d_k d_v')$ where $d_v'$ is the number of "active" dimensions.

6. **Application to mLSTM**: mLSTM from xLSTM also uses a chunkwise kernel (TFLA). Apply the same sigmoid gate to mLSTM's output. Since mLSTM has exponential gates that can cause output scale issues, the sigmoid gate may additionally serve as a **normalizer**.

## Human Review

(To be filled by reviewer)

## References

- Qiu, Z. et al. (2025). Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. NeurIPS 2025 Best Paper. arXiv:2505.06708.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692.
- Peng, B. et al. (2025). RWKV-7 "Goose" with Expressive Dynamic State Evolution. arXiv:2503.14456.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Pöppel, Beck & Hochreiter (2024). FlashRNN: I/O-Aware Optimization of Traditional RNNs. arXiv:2412.07752.
- NVIDIA. CUTLASS 3.x with Epilogue Visitor Tree. https://github.com/NVIDIA/cutlass

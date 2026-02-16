---
status: ongoing
priority: high
created: 2026-02-16
based_on: higher-order-linear-attention (222), kda-constrained-dplr-delta-chunkwise (211), gla-secondary-chunking-log-space-gating (177), hgrn2-outer-product-state-expansion (225), post-attention-sigmoid-gating (094), input-dependent-gating (065)
experiment_number: 059
experiment_log: experiment-log-059.md
---

# Second-Order KDA: Augmenting Delta Attention with HLA's Data-Adaptive Key Metric

## Hypothesis

Augmenting KDA's (Kimi Delta Attention) constrained DPLR state transition with HLA's (Higher-Order Linear Attention) **second-order key metric** $\mathbf{S}_t^K = \sum_{i \leq t} \gamma^{t-i} \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_k \times d_k}$ — so that the delta rule's state correction uses a **data-adaptive, history-dependent removal direction** instead of the current token's key alone — will improve associative recall accuracy by $> 10\%$ on MQAR at sequence length $T = 8192$ compared to standard KDA, while adding $< 15\%$ training wall-clock overhead, because the second-order key metric enables the model to selectively erase information based on the **statistical structure of all past keys** (not just the current one), providing a richer inductive bias for in-context learning.

## Background

### KDA's delta rule: powerful but myopic

KDA's state update is:

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

The rank-1 removal term $\beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top$ erases the component of the state along the **current key direction** before writing the new key-value pair. This is the delta rule from Widrow-Hoff: it removes the old value associated with $\boldsymbol{k}_t$ before inserting the new one.

**The limitation**: The removal direction is always $\boldsymbol{k}_t$ — the current token's key. The model cannot erase information along directions that are **correlated with past keys but not aligned with the current key**. For example, if the model needs to erase all information related to a topic (spanning multiple key directions), it can only erase along one direction per step.

### HLA's insight: a data-adaptive key metric

HLA maintains a **second-order prefix summary** $\mathbf{S}_t^K = \sum_{i \leq t} \mathbf{k}_i \mathbf{k}_i^\top$ — the running covariance of keys. This matrix encodes the **statistical structure** of all past keys: which directions in key space have been heavily used, which are orthogonal, and how keys cluster. HLA uses this to compute a data-adaptive polynomial kernel $\mathbf{q}_t^\top \mathbf{S}_t^K \mathbf{C}_t^{QV}$, where the kernel is conditioned on the history of all keys.

### The combination: data-adaptive delta rule removal

By replacing KDA's rank-1 removal $\beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top$ with a **rank-1 removal in the adapted key metric space** $\beta_t (\mathbf{S}_t^K \boldsymbol{k}_t)(\mathbf{S}_t^K \boldsymbol{k}_t)^\top / \|\mathbf{S}_t^K \boldsymbol{k}_t\|^2$, the model can erase information along directions that are **weighted by historical key importance**. If many past keys pointed in a similar direction (large component in $\mathbf{S}_t^K$), the removal adapts to erase more precisely in that subspace.

This is analogous to using the **Mahalanobis distance** instead of Euclidean distance for the delta rule correction — accounting for the covariance structure of past keys.

### Why this hasn't been done

- **HLA** was introduced as a standalone linear attention replacement (Zhang et al., 2025). It does not use the delta rule or state corrections.
- **KDA/Gated DeltaNet** use the delta rule but with a fixed (current-key) removal direction. No adaptive removal based on historical key statistics.
- **RWKV-7** has a similar diagonal-plus-rank-one transition but the rank-1 component uses a separate "removal key" $\hat{\kappa}_t$ — still a single per-token direction.
- **HGRN2** uses outer-product state expansion but with a simple forget gate, not a delta rule correction.

No existing work combines HLA's second-order key metric with the delta rule's state correction mechanism.

## Related Work

- **[Higher-Order Linear Attention (Zhang et al., 2025)](https://arxiv.org/abs/2510.27258)**: Introduced second-order prefix summaries for data-adaptive polynomial kernels. Applied as a standalone attention replacement. **Our approach**: Uses the second-order key metric not for the output kernel but to guide the delta rule's state correction — a fundamentally different application.
- **[Kimi Delta Attention (Kimi Team, 2025)](https://arxiv.org/abs/2510.26692)**: Constrained DPLR with $\boldsymbol{a} = \beta\boldsymbol{k}$ tying. Achieves 2× faster chunkwise kernel vs general DPLR. **Our approach**: Augments KDA with an additional $d_k \times d_k$ state that adapts the removal direction, preserving KDA's chunkwise efficiency for the main state while adding a parallel scan for the key metric.
- **[Gated DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2412.06464)**: Combines gating with delta rule. Uses scalar decay. **Our approach**: Uses per-channel decay (like KDA) plus history-adaptive removal.
- **[HGRN2 (Qin et al., 2024)](https://arxiv.org/abs/2404.07904)**: Outer-product state expansion creating $d \times d$ state. **Our approach**: The second-order key metric is a different kind of $d \times d$ state — it's a running covariance, not a key-value memory.
- **[A Systematic Analysis of Hybrid Linear Attention (Wang et al., 2025)](https://arxiv.org/abs/2507.06457)**: Surveys gating vs state-size tradeoffs. Does not explore second-order metrics combined with delta rule.

No directly related work found combining HLA's second-order key metric with delta-rule state corrections.

## Mathematical Formulation

### Standard KDA (Baseline)

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

$$
\boldsymbol{o}_t = \boldsymbol{S}_t^\top \boldsymbol{q}_t
$$

**State**: $\boldsymbol{S}_t \in \mathbb{R}^{d_k \times d_v}$ (key-value memory).

### Second-Order KDA (Proposed)

**Additional state — Key metric:**

$$
\boldsymbol{M}_t = \gamma_M \boldsymbol{M}_{t-1} + \boldsymbol{k}_t \boldsymbol{k}_t^\top \in \mathbb{R}^{d_k \times d_k}
$$

where $\gamma_M \in (0, 1)$ is a learnable (or fixed) decay for the key metric. This is a running exponentially-weighted covariance of keys.

**Adapted removal direction:**

$$
\tilde{\boldsymbol{k}}_t = \frac{\boldsymbol{M}_t \boldsymbol{k}_t}{\|\boldsymbol{M}_t \boldsymbol{k}_t\|_2 + \varepsilon}
$$

This projects $\boldsymbol{k}_t$ through the key metric, emphasizing directions that have been historically important (high variance in key space) and deemphasizing rarely-used directions.

**Modified state update:**

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \tilde{\boldsymbol{k}}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

Note: The removal uses $\tilde{\boldsymbol{k}}_t$ (adapted direction) but the key-value write still uses the original $\boldsymbol{k}_t$. This means the removal direction is history-aware while the insertion is standard.

**Output (unchanged):**

$$
\boldsymbol{o}_t = \boldsymbol{S}_t^\top \boldsymbol{q}_t
$$

### Chunkwise Parallel Formulation

The key metric $\boldsymbol{M}_t$ update is a simple additive scan with diagonal decay — parallelizable via standard prefix sum:

$$
\boldsymbol{M}_t = \gamma_M^t \boldsymbol{M}_0 + \sum_{i=1}^{t} \gamma_M^{t-i} \boldsymbol{k}_i \boldsymbol{k}_i^\top
$$

Within a chunk of size $C$, the key metric can be computed in parallel:

$$
\boldsymbol{M}_{[n]}^{\text{end}} = \gamma_M^C \boldsymbol{M}_{[n-1]}^{\text{end}} + \sum_{i=1}^{C} \gamma_M^{C-i} \boldsymbol{k}_{[n],i} \boldsymbol{k}_{[n],i}^\top
$$

The per-position key metrics $\boldsymbol{M}_{[n],i}$ within a chunk can be computed via a cumulative sum of outer products, which is a batched matrix addition — trivially parallelizable.

**Key insight for GPU efficiency**: The key metric update is a **rank-1 additive accumulation** (outer product of $\boldsymbol{k}_t$ with itself), NOT a matrix-matrix multiply. The per-chunk summary $\sum_{i} \gamma^{C-i} \boldsymbol{k}_i \boldsymbol{k}_i^\top$ is equivalent to $(K \odot \Gamma)^\top K$ — a standard $d_k \times C \times d_k$ matmul, fully tensor-core compatible.

The adapted key computation $\tilde{\boldsymbol{k}}_t = \boldsymbol{M}_t \boldsymbol{k}_t / \|\cdot\|$ is a $d_k \times d_k$ matvec per token — this is the main overhead. Within a chunk, this can be batched as a $d_k \times d_k \times C$ batched matvec, which maps to a GEMM of shape $(C, d_k) \times (d_k, d_k)$ — tensor-core friendly.

### Key Variables

- $\boldsymbol{S}_t \in \mathbb{R}^{d_k \times d_v}$ — key-value state (same as KDA)
- $\boldsymbol{M}_t \in \mathbb{R}^{d_k \times d_k}$ — key second-moment metric (NEW)
- $\tilde{\boldsymbol{k}}_t \in \mathbb{R}^{d_k}$ — adapted removal key (NEW)
- $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$ — per-channel decay (same as KDA)
- $\beta_t \in [0,1]$ — learning rate gate (same as KDA)
- $\gamma_M \in (0,1)$ — key metric decay (NEW, learnable scalar)
- $d_k, d_v$ — key and value dimensions (typically 128)
- $C$ — chunk size (typically 64)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Second-Order KDA (SO-KDA) |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 16$ |
| Head dim | $d_k = d_v = 128$ |
| Chunk size | $C = 64$ |
| Key metric decay | $\gamma_M = 0.99$ (learnable) |
| FFN | SwiGLU, $d_{\text{ff}} = 5504$ |

### Baseline

1. **Standard KDA** (Kimi Linear): Constrained DPLR with per-channel decay, same head dimension and chunk size.
2. **GLA** (Yang et al., 2024): Diagonal-only gating, no delta rule.
3. **Gated DeltaNet** (Yang et al., 2024): Scalar decay + delta rule.
4. **Softmax Transformer** (FlashAttention-2): Standard attention baseline.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MQAR accuracy ($T=8192$) | $> 85\%$ (vs KDA ~$75\%$) | Multi-Query Associative Recall |
| Perplexity (1.3B, 15B tokens) | $\leq$ KDA baseline | Validation PPL on SlimPajama |
| Training throughput | $\geq 0.85\times$ KDA | Tokens/sec on H100 |
| Peak memory | $\leq 1.15\times$ KDA | GPU memory (GB) |

### Estimated Compute

**MVE**: < 1 GPU-hour (synthetic MQAR, tiny model)
**Full (370M)**: ~80 GPU-hours
**Full (1.3B)**: ~300 GPU-hours
**Total**: ~380 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- MQAR at $T = 8192$: $> 85\%$ accuracy where standard KDA achieves ~$75\%$ and GLA achieves ~$60\%$. The adaptive removal direction enables more precise key-value overwriting, reducing interference between similar keys.
- Perplexity: Within $0.3$ points of standard KDA on language modeling (the key metric adds expressivity but doesn't change the model's capacity for soft aggregation).
- Training overhead: $< 15\%$ wall-clock because the key metric update is a single matmul per chunk ($O(C d_k^2)$) and the adapted key computation is another matmul ($O(C d_k^2)$), both tensor-core friendly.
- The $d_k \times d_k$ key metric state adds $128^2 = 16$K parameters per head — only $128^2 \times 2 = 32$ KB memory per head (negligible).

**If hypothesis is wrong:**
- **Scenario A: No MQAR improvement** — the standard delta rule's removal direction is already sufficient for associative recall. **Learn**: The bottleneck is state capacity (size of $\boldsymbol{S}_t$), not removal precision. Follow up with state expansion (more heads or larger $d_v$).
- **Scenario B: Perplexity degrades** — the normalized removal key $\tilde{\boldsymbol{k}}_t$ has poor gradient flow through the normalization. **Learn**: Try unnormalized adaptation $\tilde{\boldsymbol{k}}_t = \boldsymbol{k}_t + \eta (\boldsymbol{M}_t \boldsymbol{k}_t - \boldsymbol{k}_t)$ with learnable mixing $\eta$.
- **Scenario C: Training too slow** — the additional $d_k \times d_k$ matvecs dominate. **Learn**: Reduce the key metric to a low-rank approximation $\boldsymbol{M}_t \approx \boldsymbol{U}_t \boldsymbol{U}_t^\top$ with $\boldsymbol{U}_t \in \mathbb{R}^{d_k \times r}$ for $r \ll d_k$.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer SO-KDA, $d = 128$, $H = 4$, $d_k = d_v = 32$ (~200K params)
- **Task**: Multi-Query Associative Recall (MQAR) with 16 key-value pairs, vocabulary size 64
- **Data**: Synthetic MQAR, $T = 512$, 10K training samples
- **Compute**: Single GPU, < 10 minutes
- **Baselines**: Same model with (a) standard KDA, (b) GLA (no delta rule)

### Success Criteria
- $> 95\%$ MQAR accuracy at $T = 512$ with 16 pairs where:
  - Standard KDA achieves ~$85\%$ (known limitation at high pair count)
  - GLA achieves ~$70\%$
- The improvement persists when testing at $T = 1024$ (length generalization)

### Failure Criteria
- SO-KDA accuracy $\leq$ standard KDA accuracy: the key metric adds no benefit
- Training loss diverges: the $\boldsymbol{M}_t \boldsymbol{k}_t$ normalization is unstable
- Per-step time $> 2\times$ standard KDA: overhead too large even at small scale

### Why This Test Is Sufficient
- MQAR specifically tests the ability to **overwrite and retrieve** key-value pairs — the exact capability the adapted removal direction targets. If SO-KDA's adaptive removal doesn't help on MQAR, it won't help anywhere.
- At $d_k = 32$, the key metric $\boldsymbol{M}_t \in \mathbb{R}^{32 \times 32}$ is small enough that even a naive implementation runs fast, isolating the algorithmic benefit from GPU engineering.

## Memory Access Pattern Analysis

**Key metric update** ($\boldsymbol{M}_t = \gamma_M \boldsymbol{M}_{t-1} + \boldsymbol{k}_t \boldsymbol{k}_t^\top$):
- Per chunk: accumulate $(K \odot \Gamma)^\top K$ — a $(d_k, C) \times (C, d_k)$ matmul. **Tensor-core friendly**.
- Memory: $d_k \times d_k = 128^2 \times 2 = 32$ KB per head in bf16. Fits comfortably in SRAM (H100: 228 KB per SM).
- Arithmetic intensity: $2 C d_k^2 / (2 C d_k + d_k^2) \times 2$ bytes $\approx d_k = 128$ FLOPs/byte. **Compute-bound.**

**Adapted key computation** ($\tilde{\boldsymbol{k}}_t = \boldsymbol{M}_t \boldsymbol{k}_t / \|\cdot\|$):
- Batched over chunk: $(C, d_k) \times (d_k, d_k) = (C, d_k)$ matmul. **Tensor-core friendly**.
- The normalization is element-wise (L2 norm + divide) — negligible cost.
- Can be fused into the WY transform computation as an additional matmul.

**Coalesced**: All accesses are through contiguous matmuls — fully coalesced.

**Cache-friendly**: The key metric $\boldsymbol{M}_t$ persists across the chunk in SRAM — no HBM round-trips within a chunk.

## Parallelism Analysis

- **Tensor core mapping**: All new operations are matmuls (key metric accumulation and adapted key computation). Maps directly to WGMMA on H100.
- **No warp divergence**: Uniform computation across all tokens within a chunk.
- **SM saturation**: $H \times B$ independent programs (same as KDA). With $H = 16, B = 8$: 128 programs.
- **Sequential bottleneck**: The inter-chunk key metric propagation ($\boldsymbol{M}_{[n]} = \gamma_M^C \boldsymbol{M}_{[n-1]} + \hat{\boldsymbol{M}}_{[n]}$) is a scalar-times-matrix plus matrix addition — trivially cheap ($d_k^2$ FLOPs per chunk boundary).

## Theoretical Analysis

| Operation | KDA (baseline) | SO-KDA (proposed) | Overhead |
|-----------|---------------|-------------------|----------|
| Intra-chunk FLOPs | $O(C^2 d_k + C d_k d_v)$ | $O(C^2 d_k + C d_k d_v + C d_k^2)$ | $+O(C d_k^2)$ |
| Inter-chunk state | $O(d_k \times d_v)$ per head | $O(d_k \times d_v + d_k^2)$ per head | $+O(d_k^2)$ |
| Per-step inference | $O(d_k d_v)$ | $O(d_k d_v + d_k^2)$ | $+O(d_k^2)$ |

With $d_k = d_v = 128$, $C = 64$:
- KDA intra-chunk: $64^2 \times 128 + 64 \times 128^2 = 524K + 1049K = 1.57M$ FLOPs
- SO-KDA overhead: $64 \times 128^2 = 1.05M$ FLOPs
- **Relative overhead**: $1.05M / 1.57M \approx 67\%$ for the chunkwise kernel alone

However, the **full layer** includes input/output projections ($3 \times d^2 = 3 \times 2048^2 = 12.6M$ FLOPs per token) which dwarf the chunkwise kernel. At layer level, the overhead is:

$$
\text{Layer overhead} = \frac{C d_k^2 \times H}{3 d^2 + C^2 d_k + C d_k d_v} \approx \frac{1.05M \times 16}{12.6M + 1.57M} = \frac{16.8M}{14.2M} \approx 12\%
$$

This aligns with the $< 15\%$ target.

## Risks & Limitations

1. **Key metric memory**: The $d_k \times d_k$ metric doubles the per-head state from $d_k \times d_v$ to $d_k \times (d_v + d_k)$. For $d_k = d_v = 128$: from 32 KB to 64 KB per head. With 16 heads: 1 MB total state per layer. This is manageable but non-trivial for very long-context inference.

2. **Normalization instability**: Dividing by $\|\boldsymbol{M}_t \boldsymbol{k}_t\|$ could be unstable if $\boldsymbol{M}_t$ is near-singular (early in training when few keys have been seen). **Mitigation**: Add $\varepsilon = 10^{-6}$ and initialize $\boldsymbol{M}_0 = \varepsilon \boldsymbol{I}$.

3. **Key metric decay sensitivity**: If $\gamma_M$ is too high, $\boldsymbol{M}_t$ accumulates indefinitely and becomes dominated by early keys. If too low, it forgets too quickly. **Mitigation**: Learn $\gamma_M$ per-head or per-layer. Initialize at $0.99$.

4. **Backward pass complexity**: The gradient through $\tilde{\boldsymbol{k}}_t = \boldsymbol{M}_t \boldsymbol{k}_t / \|\boldsymbol{M}_t \boldsymbol{k}_t\|$ involves gradients through the normalization and the matrix-vector product. This is standard (analogous to LayerNorm gradients) but adds compute.

5. **No benefit for tasks without key collision**: If keys are sufficiently diverse (different tokens always produce orthogonal keys), the adapted removal direction $\tilde{\boldsymbol{k}}_t \approx \boldsymbol{k}_t$ (since $\boldsymbol{M}_t$ is approximately $\sigma^2 \boldsymbol{I}$). The benefit is specifically for tasks with **key reuse** (associative recall, in-context learning).

## Follow-up Experiments

1. **Low-rank key metric**: Approximate $\boldsymbol{M}_t$ with rank-$r$ factorization $\boldsymbol{M}_t \approx \boldsymbol{U}_t \boldsymbol{U}_t^\top$ via online SVD or running top-$r$ eigenvalue tracking. Reduces state from $O(d_k^2)$ to $O(r d_k)$.

2. **Multi-query key metric sharing**: Share $\boldsymbol{M}_t$ across heads (like multi-query attention shares keys). Since $\boldsymbol{M}_t$ captures global key statistics, sharing may suffice — reducing memory by $H\times$.

3. **Second-order KDA + HLA output**: Instead of just using $\boldsymbol{M}_t$ to adapt the removal, also use it for the output computation: $\boldsymbol{o}_t = (\boldsymbol{q}_t^\top \boldsymbol{M}_t) \boldsymbol{S}_t$. This gives a full second-order output like HLA, combined with the delta rule state correction.

4. **Kimi Linear hybrid integration**: Test SO-KDA in a 3:1 hybrid with MLA (as Kimi Linear does with standard KDA). Check if the improved recall reduces the need for MLA layers.

5. **Ablation — unnormalized vs normalized adaptation**: Compare $\tilde{\boldsymbol{k}}_t = \boldsymbol{M}_t \boldsymbol{k}_t$ (unnormalized), $\tilde{\boldsymbol{k}}_t = \boldsymbol{M}_t \boldsymbol{k}_t / \|\cdot\|$ (normalized), and $\tilde{\boldsymbol{k}}_t = \boldsymbol{k}_t + \eta (\boldsymbol{M}_t \boldsymbol{k}_t - \boldsymbol{k}_t)$ (residual mixing) to understand the best form of adaptation.

## Human Review

(To be filled by reviewer)

## References

- Zhang, Y., Qin, Z., & Gu, Q. (2025). Higher-order Linear Attention. arXiv:2510.27258.
- Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Qin, Z., Yang, S., Sun, W. et al. (2024). HGRN2: Gated Linear RNNs with State Expansion. COLM 2024. arXiv:2404.07904.
- Peng, B. et al. (2025). RWKV-7 "Goose" with Expressive Dynamic State Evolution. arXiv:2503.14456.
- Qiu, Z. et al. (2025). Gated Attention for Large Language Models. NeurIPS 2025 Best Paper. arXiv:2505.06708.
- Wang, Z. et al. (2025). A Systematic Analysis of Hybrid Linear Attention. arXiv:2507.06457.

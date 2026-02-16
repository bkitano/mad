---
status: ongoing
priority: high
created: 2026-02-16
based_on: multi-matrix-factorization-attention (229), kda-constrained-dplr-delta-chunkwise (211), gla-secondary-chunking-log-space-gating (177), blast-block-adaptive-structured-matrix (250), tfla-two-level-tiled-chunkwise-parallelism (158)
experiment_number: 063
experiment_log: experiment-log-063.md
---

# MFA-Style Shared Latent Projections for Linear RNN Training

## Hypothesis

Replacing the independent per-head $W_Q, W_K, W_V$ projection matrices in chunkwise linear RNNs (KDA, GLA, Gated DeltaNet) with **MFA-style shared latent projections** — where a single shared down-projection $S \in \mathbb{R}^{d \times C}$ maps inputs to a latent space of dimension $C$, and small per-head matrices $Q_c \in \mathbb{R}^{C \times C}$ provide head-specific differentiation — will reduce projection FLOP cost by $30$–$50\%$ and HBM bandwidth for weight loading by a comparable factor, yielding $10$–$20\%$ end-to-end training speedup with $< 0.3\%$ perplexity degradation. The key is that MFA's factorization **maps entirely to batched GEMMs** (unlike Monarch/ACDC which require FFT/permutations), and it **increases Total Effective Rank** (TER) per FLOP by allowing more heads at the same compute budget.

## Background

### The projection bottleneck in linear RNN training

In chunkwise linear RNN training (GLA, KDA, Gated DeltaNet, SSD), the per-layer forward pass decomposes into:

1. **Projection GEMMs** ($40$–$60\%$ of FLOPs): Computing $Q = XW_Q$, $K = XW_K$, $V = XW_V$, gate projections, and output projection $O = YW_O$. Each is a dense GEMM of shape $(B \cdot T, d) \times (d, d_h)$.

2. **Chunkwise scan/attention** ($40$–$60\%$ of FLOPs): Intra-chunk quadratic attention $O(C^2 d_h)$ and inter-chunk state recurrence $O(T d_k d_v / C)$.

Existing proposals (001–062) have focused almost exclusively on optimizing part (2) — the scan mechanics, kernel fusion, and numerical stability. Proposal 051 explored Monarch factorization for projections but relies on FFT-like butterfly operations that don't use tensor cores. **No prior work has applied MFA's pure-GEMM factorization to linear RNN projections.**

### Why MFA is a better fit than Monarch for linear RNN projections

MFA (trick 229) was designed for softmax attention and validated at 7B scale (ACL 2025). Its structure offers specific advantages over Monarch (trick 076) for linear RNN projections:

| Property | MFA | Monarch | ACDC |
|----------|-----|---------|------|
| Core ops | Batched GEMM | BMM + permutation | FFT + elementwise |
| Tensor core usage | $\checkmark$ Full | Partial (perm overhead) | $\times$ No |
| Learnable structure | Per-head $Q_c$ adapt freely | Fixed butterfly pattern | Fixed DCT |
| KV sharing | Built-in shared $S_k, S_v$ | Must be imposed | N/A |
| Proven at scale | 7B MoE (1T tokens) | 1.3B (NeurIPS 2024) | ViT only |
| Quality at 50% FLOPs | Exceeds MHA | Needs "self-guided training" | Underperforms |

The critical insight: **MFA's shared projection architecture is a natural match for linear RNNs**, which already have multi-head structure where heads share the same input $X$. The shared $S_k, S_v$ projections mean the keys and values are computed once in the latent space and shared across all heads, with per-head differentiation provided by small $C \times C$ matrices.

### What MFA provides that's novel for linear RNNs

1. **More heads at the same cost**: By sharing the expensive down-projection and using cheap per-head $Q_c \in \mathbb{R}^{C \times C}$, we can run $m \gg n_{\text{original}}$ heads within the same FLOP budget. More heads means the state $\sum_h S_t^{(h)}$ has higher effective rank, improving recall capacity.

2. **Reduced HBM bandwidth for weights**: The dominant weights $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$ are replaced by $S_q, S_k, S_v \in \mathbb{R}^{d \times C}$ (shared) plus $m$ small matrices $Q_c \in \mathbb{R}^{C \times C}$. Total weight memory: $3dC + mC^2 + mCd_v$ vs $3d \cdot h \cdot d_h$ for standard.

3. **Compatible with chunkwise kernels**: The per-head $Q_c, V_c$ matrices transform the latent $q, k, v$ before they enter the chunkwise kernel. The kernel itself is unchanged — it still receives per-head $q, k, v$ of dimension $C$ (or $d_v$).

## Related Work

- **MFA (ACL 2025, arXiv:2412.19255)**: Proposed the shared latent projection architecture for softmax attention. Achieved 49.9% avg accuracy vs MHA's 49.0% at 7B MoE scale with 87.5% KV cache reduction. **Only tested on softmax Transformers, never on linear RNNs.**

- **Kimi Linear (2025, arXiv:2510.26692)**: Uses KDA (constrained DPLR) in a 3:1 hybrid with MLA. The MLA layers use DeepSeek-style latent compression, but the KDA layers use standard independent projections. **No shared projection structure within the KDA layers themselves.**

- **Proposal 051 (KS-Fused Monarch Projections)**: Explored Monarch factorization for linear RNN projections. Different approach: Monarch uses butterfly patterns (FFT-like), which require permutations and don't fully utilize tensor cores. MFA uses pure GEMM structure.

- **Proposal 053 (MLA Latent State Compression)**: Applies MLA's weight absorption to compress the recurrent *state* during inference. Different target: we compress the *projection weights* during training, not the recurrent state at inference.

- **"Effectively Training LLMs with Structured Feedforward Layers" (NeurIPS 2024)**: Tested structured FFN layers (block-diagonal, Monarch) for Transformer training. Found that quality requires "self-guided training" (dense warm-start). **Our approach**: MFA has been shown to match/exceed MHA quality without warm-start at 7B scale, suggesting shared projections are a more natural structured form.

No directly related work found applying MFA-style shared latent projections to linear RNN (GLA/KDA/GDN) layers.

## Mathematical Formulation

**Standard GLA/KDA Projection (Per-Head Independent):**

For input $X \in \mathbb{R}^{T \times d}$, each of $n$ heads computes:

$$
Q^{(h)} = X W_Q^{(h)}, \quad K^{(h)} = X W_K^{(h)}, \quad V^{(h)} = X W_V^{(h)}
$$

where $W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d \times d_k}$, $W_V^{(h)} \in \mathbb{R}^{d \times d_v}$.

Total projection FLOPs: $n \cdot T \cdot d \cdot (2d_k + d_v)$ (or equivalently $T \cdot d \cdot (2d_k + d_v) \cdot n$).

In practice, these are concatenated into fused GEMMs: $Q = X W_Q$ where $W_Q \in \mathbb{R}^{d \times (n \cdot d_k)}$.

**Proposed: MFA-Factored Projections:**

Replace per-head projections with shared down-projections + per-head latent rotations:

$$
q_{\text{latent}} = X S_q, \quad k_{\text{latent}} = X S_k, \quad v_{\text{latent}} = X S_v
$$

where $S_q, S_k \in \mathbb{R}^{d \times C}$ and $S_v \in \mathbb{R}^{d \times C}$ are shared across all heads, and $C$ is the latent dimension (typically $C = 2d_k$ to $4d_k$).

Per-head differentiation:

$$
Q^{(h)} = q_{\text{latent}} \cdot Q_c^{(h)}, \quad K^{(h)} = k_{\text{latent}}, \quad V^{(h)} = v_{\text{latent}} \cdot V_c^{(h)}
$$

where $Q_c^{(h)} \in \mathbb{R}^{C \times d_k}$ and $V_c^{(h)} \in \mathbb{R}^{C \times d_v}$.

**Key observation**: Keys are shared across heads (like MQA/GQA) but in a *latent space*. Per-head queries and values provide differentiation via small matrices.

**FLOP comparison:**

| Component | Standard | MFA-Factored |
|-----------|----------|--------------|
| Shared down-proj | — | $3 \cdot T \cdot d \cdot C$ |
| Per-head $Q_c$ | — | $m \cdot T \cdot C \cdot d_k$ |
| Per-head $V_c$ | — | $m \cdot T \cdot C \cdot d_v$ |
| Standard proj | $T \cdot d \cdot n(2d_k + d_v)$ | — |
| **Total** | $T \cdot d \cdot n(2d_k + d_v)$ | $T(3dC + mC(d_k + d_v))$ |

For $d = 2048$, $n = 4$, $d_k = d_v = 128$, standard: $T \cdot 2048 \cdot 4 \cdot 384 = T \cdot 3.1\text{M}$.

With $C = 256$, $m = 8$ (double the heads!): $T(3 \cdot 2048 \cdot 256 + 8 \cdot 256 \cdot 256) = T(1.57\text{M} + 0.52\text{M}) = T \cdot 2.1\text{M}$.

**Result**: $33\%$ fewer projection FLOPs with $2\times$ more heads.

**Total Effective Rank comparison:**

$$
\text{TER}_{\text{standard}} = n \cdot d_k = 4 \times 128 = 512
$$

$$
\text{TER}_{\text{MFA}} = m \cdot C = 8 \times 256 = 2048 \quad (4\times \text{ higher})
$$

The higher TER means the aggregate state $\sum_h S_t^{(h)}$ across heads can represent richer key-value associations.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA or KDA (chunkwise linear RNN) |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Latent dim | $C = 256$ |
| Heads (standard) | $n = 4$, $d_k = d_v = 128$ |
| Heads (MFA) | $m = 8$, $d_k = d_v = 128$ |
| Chunk size | $C_{\text{chunk}} = 64$ |
| Parameters | ~370M (standard) vs ~360M (MFA) |

### Baseline

Standard GLA/KDA with independent per-head projections, using FlashLinearAttention (FLA) Triton kernels. Complexity: $O(T \cdot d \cdot n(2d_k + d_v) + T \cdot C_{\text{chunk}}^2 \cdot d_k \cdot n)$.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $> 1.1 \times$ baseline tokens/sec | Average over 1K training steps on H100 |
| Memory | $\leq$ baseline peak GPU memory | `torch.cuda.max_memory_allocated` |
| Quality | $\leq 0.3\%$ perplexity degradation | WikiText-103 validation perplexity |
| Recall | $\geq$ baseline on MQAR | Multi-query associative recall accuracy |

### Estimated Compute

**Small** ($< 100$ GPU-hours): 370M model, 10B tokens on SlimPajama, single H100.

## Expected Outcome

**If hypothesis is correct:**
- $10$–$20\%$ end-to-end training speedup from reduced projection FLOPs
- $4\times$ higher TER from $2\times$ more heads enables better recall with fewer total FLOPs
- The quality may *improve* relative to standard (as MFA showed for softmax attention) due to the increased TER

**If hypothesis is wrong:**
- If quality degrades: the shared key/value structure may be too constrained for linear RNNs where key normalization interacts with the delta rule. This would tell us that linear RNN heads need more key diversity than softmax heads.
- If no speedup: the per-head $Q_c, V_c$ matmuls may not be large enough to saturate tensor cores, losing the FLOP savings to underutilization. This would inform minimum head/latent dimension choices.

## Minimum Viable Experiment

### Setup

- **Model**: 2-layer GLA/KDA, $d = 128$, $C = 64$, $m = 4$ MFA heads vs $n = 2$ standard heads (~100K params)
- **Task**: Multi-Query Associative Recall (MQAR) — tests whether shared projections preserve key-value recall
- **Data**: 10K synthetic MQAR sequences (length 128, 8 key-value pairs)
- **Compute**: Single GPU, $< 5$ minutes

### Success Criteria

- MFA variant achieves $\geq 90\%$ MQAR accuracy matching the standard-projection baseline
- Forward pass wall-clock is $\leq$ baseline (no slowdown from the extra per-head matmuls)
- Both models converge in the same number of training steps (shared projections don't slow optimization)

### Failure Criteria

- If MFA variant achieves $< 70\%$ MQAR accuracy while standard achieves $> 90\%$, the shared key structure fundamentally breaks the delta rule's key-based memory addressing
- If per-head matmuls are slower than a single fused GEMM (likely if $C \times d_k$ is too small for tensor core tiles)

### Why This Test Is Sufficient

MQAR is the canonical recall task for linear RNNs. If shared projections can maintain recall accuracy at toy scale, the mechanism is fundamentally sound. Quality at scale depends on capacity (more heads = more capacity), so the MVE focuses on whether sharing hurts per-head recall — not absolute quality.

## Memory Access Pattern Analysis

**Standard projections**: Single fused GEMM $(BT, d) \times (d, n \cdot d_k)$. Coalesced, high arithmetic intensity $\approx d$ FLOPs/byte for large $d$. Weight matrix loaded once from HBM.

**MFA projections**: Three phases:
1. **Shared down-projection**: $(BT, d) \times (d, C)$ — single GEMM, coalesced, arithmetic intensity $\approx d$.
2. **Per-head rotation**: $(BT, C) \times (C, d_k)$ for each head — batched GEMM. Each $Q_c^{(h)}$ is $C \times d_k$ ($256 \times 128 = 32\text{K}$ elements = 64KB in FP16), fits in L2 cache.
3. **Cache behavior**: Shared latent $q_{\text{latent}} \in \mathbb{R}^{BT \times C}$ is computed once and reused $m$ times for per-head $Q_c$. With $BT = 2048$, this is $2048 \times 256 \times 2 = 1\text{MB}$ — fits in L2 cache on H100 (50MB).

**Net effect**: The shared down-projection reduces weight HBM loads from $O(d \cdot n \cdot d_k)$ to $O(dC + mCd_k)$. The per-head matrices are small enough to stay in L2 across the batch.

## Parallelism Analysis

- **Shared down-projection**: Single large GEMM — saturates all SMs, identical to standard projection.
- **Per-head rotations**: $m$ independent small GEMMs — can be batched via `torch.bmm` or fused via einsum. Each GEMM has dimensions $(BT, C) \times (C, d_k)$ which is $(2048, 256) \times (256, 128)$ — large enough for tensor core tile utilization (minimum $16 \times 16$).
- **No sequential bottlenecks**: All operations are independent matmuls.
- **Warp divergence**: None — all operations are regular matmuls.

## Theoretical Analysis

Complexity comparison (projection FLOPs per token per layer):

| Operation | Standard ($n=4$, $d_k=128$) | MFA ($m=8$, $C=256$, $d_k=128$) |
|-----------|-----|-----|
| QKV projection | $d \cdot n(2d_k + d_v) = 1.57\text{M}$ | $3dC + mC(d_k + d_v) = 2.10\text{M}$ |
| Output projection | $d \cdot n \cdot d_v = 1.05\text{M}$ | $d \cdot m \cdot d_v + \text{up-proj} = 0.79\text{M}$ |
| **Total** | $2.62\text{M}$ | $2.89\text{M}$ |
| **FLOPs per head** | $655\text{K}$ | $361\text{K}$ |
| **Heads** | 4 | 8 |
| **Total Effective Rank** | 512 | 2048 |

At matched FLOPs ($m = 4$ MFA heads, $C = 256$): projection cost is $T(3 \cdot 2048 \cdot 256 + 4 \cdot 256 \cdot 256) = T \cdot 1.83\text{M}$ vs $T \cdot 2.62\text{M}$ — **30% reduction**.

Crossover: MFA is always cheaper per-head; the question is whether shared keys hurt recall. The TER analysis predicts no loss because $\text{TER}_{\text{MFA}} = mC \geq \text{TER}_{\text{std}} = nd_k$ when $C \geq d_k$ (which holds by construction).

## Risks & Limitations

1. **Shared keys may hurt delta rule addressing**: The delta rule uses $k_t$ both for state update direction and removal direction. If all heads share the same $k_{\text{latent}}$, the per-head state updates may become too correlated, reducing the effective diversity of the memory. Mitigation: per-head $Q_c$ provides query differentiation; and MFA already showed this works for softmax attention where keys also serve as addressing vectors.

2. **Small per-head matmuls may underutilize tensor cores**: If $C \times d_k = 256 \times 128$ is below the GEMM efficiency threshold, the batched matmuls may run at low utilization. Mitigation: batch across all $m$ heads into a single $(m \cdot BT, C) \times (C, d_k)$ GEMM, or use `torch.bmm` which is well-optimized for this regime.

3. **Gate projections also need factorization**: KDA/GLA have additional gate projections ($W_\alpha, W_\beta$) that are also $\mathbb{R}^{d \times d_k}$. These should also be shared for consistent savings, but sharing gate projections may hurt per-head adaptivity.

4. **Not validated beyond 7B for MFA itself**: MFA (ACL 2025) was validated at 7B MoE; behavior at 70B+ for softmax attention is unknown. For linear RNNs, even 7B validation doesn't exist yet.

## Follow-up Experiments

1. **MFA-KR for linear RNNs**: If shared projections work, test MFA-Key-Reuse (re-parameterize $S_v$ from $S_k$) to halve the shared projection cost further. This is especially interesting for KDA where keys are L2-normalized — the key-value correlation may be higher.

2. **BLAST per-head matrices**: If the per-head $Q_c \in \mathbb{R}^{C \times C}$ are too expensive for many heads, replace them with BLAST-factored matrices (trick 250) for additional $2\times$ compression of the per-head parameters.

3. **Scaling to 3B–7B**: If the 370M experiment succeeds, test at 3B scale on SlimPajama to verify the quality/efficiency tradeoff holds with more layers and larger state.

4. **Hybrid MFA-KDA + MLA**: In Kimi Linear's 3:1 hybrid, the MLA layers already use latent compression. Applying MFA-style sharing to the KDA layers would create a fully latent-projection architecture.

## Human Review

(To be filled by reviewer)

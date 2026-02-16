---
status: ongoing
priority: high
created: 2026-02-15
based_on: mla-weight-absorption-latent-kv (197), tfla-two-level-tiled-chunkwise-parallelism (158), gla-secondary-chunking-log-space-gating (177), fused-chunkwise-ssd-atomic-state-passing (182), input-dependent-gating (065)
experiment_number: 053
experiment_log: experiment-log-053.md
---

# MLA-Inspired Latent State Compression for Linear RNN Inference

## Hypothesis

Compressing the $d_k \times d_v$ hidden state of a linear RNN (GLA, mLSTM, Gated DeltaNet) into a low-rank latent $c_t \in \mathbb{R}^{d_c}$ (with $d_c \ll d_k \cdot d_v$) during autoregressive inference, and using **weight absorption** (from DeepSeek-V2's MLA) to compute the output $\phi(q_t)^\top S_t$ directly in the latent space without ever decompressing the full state, will achieve $2$–$4\times$ higher generation throughput for linear RNN models by reducing the per-step HBM bandwidth from $O(d_k \cdot d_v)$ to $O(d_c)$ — the same mechanism that gives MLA its 57× KV cache bandwidth reduction applied to a fundamentally different architecture.

## Background

### The inference bandwidth bottleneck for linear RNNs

Linear RNNs (GLA, mLSTM, Mamba-2) maintain a matrix-valued hidden state $S_t \in \mathbb{R}^{d_k \times d_v}$ that acts as a compressed summary of the entire history. During autoregressive generation, every decode step must:

1. **Read** $S_{t-1}$ from HBM ($d_k \cdot d_v$ elements)
2. **Update** $S_t = g_t \cdot S_{t-1} + k_t v_t^\top$ ($O(d_k \cdot d_v)$ work)
3. **Read out** $o_t = \phi(q_t)^\top S_t$ ($O(d_k \cdot d_v)$ work)
4. **Write** $S_t$ back to HBM ($d_k \cdot d_v$ elements)

For GLA with 4 heads, $d_k = d/2 = 1024$, $d_v = d = 2048$: each head's state is $1024 \times 2048 \times 2 = 4$ MB in FP16. With 4 heads and 24 layers, the total state is $4 \times 4 \times 24 = 384$ MB. Each decode step reads and writes this entire state — **768 MB of HBM traffic per step**.

For comparison, standard Transformer MHA with $n_h = 32$, $d_h = 128$ has a KV cache of $32 \times 128 \times 2 = 8$ KB per token per layer. At 2048 tokens, this is $8 \times 2048 \times 24 = 384$ MB — the same order, but standard attention only reads the KV cache (not write), while the linear RNN must both read AND write its state.

**This makes linear RNN inference bandwidth-bound for the same reason Transformer inference is bandwidth-bound** — the dominant cost is loading the state/cache from HBM, not the computation itself.

### MLA's key insight applies here

MLA (trick 197) solves the Transformer bandwidth problem by:
1. Compressing KV into a latent $c_t^{KV} \in \mathbb{R}^{d_c}$ (where $d_c \ll 2 n_h d_h$)
2. Using **weight absorption** to compute attention scores directly on $c_t^{KV}$ without decompressing

The same principle can be applied to linear RNN states:
1. Compress $S_t \in \mathbb{R}^{d_k \times d_v}$ into a latent $c_t \in \mathbb{R}^{d_c}$
2. Use weight absorption to compute the readout $o_t = \phi(q_t)^\top S_t$ directly from $c_t$

The key difference: MLA's latent is per-token and independent (each token's KV is compressed independently). The linear RNN state is **accumulated over time** — the compression must be compatible with the recurrent update.

### Why this is feasible: low-rank structure in linear RNN states

The state $S_t = \sum_{i=1}^t g_{i:t} \cdot k_i v_i^\top$ is a sum of rank-1 outer products with exponential decay. By the Eckart-Young theorem, the best rank-$r$ approximation to $S_t$ captures the top-$r$ singular values. Empirically, linear RNN states are **strongly low-rank**: the effective rank (measured by participation ratio of singular values) is typically $r_{\text{eff}} \ll \min(d_k, d_v)$ because:

1. The exponential gating concentrates energy on recent tokens
2. The $k_t, v_t$ projections share a common low-dimensional subspace within each head
3. The state acts as an associative memory with capacity bounded by the effective rank

If the state has effective rank $r_{\text{eff}} \approx 32$–$128$ while $d_k \cdot d_v = 1024 \times 2048$, there is a $16$–$64\times$ compression opportunity.

### What hasn't been done

- **MLA weight absorption** has only been applied to softmax attention KV cache (DeepSeek-V2/V3)
- **Linear RNN state compression for inference** has not been studied — all existing work maintains the full $d_k \times d_v$ state
- **PALU** (KV cache compression) applies low-rank factorization to Transformer KV caches but not to recurrent states
- No proposal in the existing set (001-052) addresses inference-specific optimization for linear RNNs

## Related Work

- **DeepSeek-V2 MLA (2024)**: Introduced weight absorption for latent KV cache in Transformers. Achieves 93.3% KV cache reduction. **Our approach**: Applies the same weight absorption principle to linear RNN hidden states, which have a fundamentally different structure (accumulated over time, not per-token).
- **PALU (ICLR 2025)**: Post-training KV-cache compression via low-rank factorization. Compresses KV projections using SVD. **Our approach**: Applies low-rank compression to the recurrent state itself, not just KV projections. The compression must be compatible with the recurrent update rule.
- **X-EcoMLA (2025)**: Upcycles pre-trained MHA into MLA for KV compression. **Our approach**: Targets a completely different architecture (linear RNNs vs Transformers).
- **GLA (Yang et al., 2024)**: Defines the chunkwise parallel training algorithm for gated linear attention but doesn't address inference-specific state compression. Our approach adds a separate inference mode with compressed state.
- **Kimi's DPLR Linear (2025)**: Uses diagonal-plus-low-rank state transitions for expressivity but not for state compression during inference.

No directly related work found combining MLA-style weight absorption with linear RNN hidden state compression for inference.

## Mathematical Formulation

### Standard Linear RNN Inference (Baseline)

At each decode step $t$, GLA computes:

**State update:**
$$
S_t = \text{Diag}(\alpha_t) \cdot S_{t-1} + k_t v_t^\top \in \mathbb{R}^{d_k \times d_v}
$$

**Readout:**
$$
o_t = q_t^\top S_t \in \mathbb{R}^{d_v}
$$

**Output projection:**
$$
y_t = W_O \, o_t \in \mathbb{R}^d
$$

**HBM traffic per step per head:** $2 d_k d_v$ elements (read $S_{t-1}$, write $S_t$).

### Latent State Formulation

**Training (full state, unchanged):** Train the model normally with full $S_t \in \mathbb{R}^{d_k \times d_v}$.

**Inference (compressed state):** Define learned projections:

$$
W^{\text{down}} \in \mathbb{R}^{d_c \times (d_k \cdot d_v)}, \quad W^{\text{up}} \in \mathbb{R}^{(d_k \cdot d_v) \times d_c}
$$

The compressed state is:
$$
c_t = W^{\text{down}} \, \text{vec}(S_t) \in \mathbb{R}^{d_c}
$$

### Weight Absorption for the Readout

The readout $o_t = q_t^\top S_t$ can be rewritten:
$$
o_t = q_t^\top S_t = (q_t \otimes I_{d_v})^\top \text{vec}(S_t) \approx (q_t \otimes I_{d_v})^\top W^{\text{up}} c_t
$$

By absorbing $W^{\text{up}}$ into the query:
$$
o_t = \underbrace{(W^{\text{up},\top} (q_t \otimes I_{d_v}))^\top}_{\tilde{q}_t \in \mathbb{R}^{d_c}} c_t
$$

The absorbed query $\tilde{q}_t$ is computed from the query projection and the fixed $W^{\text{up}}$ matrix. This can be precomputed as a single GEMM:

$$
\tilde{W}^Q = W^{\text{up},\top} (W_Q \otimes I_{d_v}) \in \mathbb{R}^{d_c \times d}
$$

At inference:
$$
\tilde{q}_t = \tilde{W}^Q h_t \in \mathbb{R}^{d_c}, \quad o_t = \tilde{q}_t^\top c_t
$$

**No decompression of $c_t$ needed** — the readout is a $d_c$-dimensional dot product.

### Compressed State Update

The recurrent update in the latent space:
$$
c_t = W^{\text{down}} \, \text{vec}(\text{Diag}(\alpha_t) S_{t-1} + k_t v_t^\top)
$$

$$
= W^{\text{down}} \, (\text{Diag}(\alpha_t) \otimes I_{d_v}) \, \text{vec}(S_{t-1}) + W^{\text{down}} \, \text{vec}(k_t v_t^\top)
$$

$$
= \underbrace{W^{\text{down}} (\text{Diag}(\alpha_t) \otimes I_{d_v}) W^{\text{up}}}_{A_t^c \in \mathbb{R}^{d_c \times d_c}} c_{t-1} + \underbrace{W^{\text{down}} (k_t \otimes v_t)}_{b_t \in \mathbb{R}^{d_c}}
$$

**Problem:** The compressed transition $A_t^c = W^{\text{down}} (\text{Diag}(\alpha_t) \otimes I_{d_v}) W^{\text{up}}$ is input-dependent (through $\alpha_t$) and $d_c \times d_c$, requiring $O(d_c^2)$ per step.

**Solution: Exploit the diagonal structure of $\alpha_t$.**

Since $\alpha_t$ is diagonal ($d_k$ values), the Kronecker product $\text{Diag}(\alpha_t) \otimes I_{d_v}$ is also diagonal (with $d_k \cdot d_v$ values, each $\alpha_{t,j}$ repeated $d_v$ times). Therefore:

$$
A_t^c = W^{\text{down}} \, \text{Diag}(\alpha_t \otimes \mathbf{1}_{d_v}) \, W^{\text{up}}
$$

This can be decomposed: let $W^{\text{down}} = U$ and $W^{\text{up}} = V$. Then:

$$
A_t^c = U \, \text{Diag}(\alpha_t \otimes \mathbf{1}_{d_v}) \, V = \sum_{j=1}^{d_k} \alpha_{t,j} \, U_{:, j \cdot d_v : (j+1) \cdot d_v} \, V_{j \cdot d_v : (j+1) \cdot d_v, :}
$$

This is a **weighted sum of $d_k$ precomputed $d_c \times d_c$ matrices**, where the weights are the gate values $\alpha_{t,j}$. If $d_k$ is moderate (e.g., 128), this is:

$$
A_t^c = \sum_{j=1}^{d_k} \alpha_{t,j} \, M_j, \quad M_j = U_{:,j} V_{j,:} \in \mathbb{R}^{d_c \times d_c}
$$

where $M_j$ are precomputed at model load time.

**Per-step compute:** $O(d_k \cdot d_c^2)$ for the weighted sum + $O(d_c^2)$ for the matrix-vector product $A_t^c c_{t-1}$ + $O(d_c)$ for the input term. With $d_c = 64$ and $d_k = 128$: $128 \times 64^2 = 524K$ FLOPs — much less than the $d_k \times d_v = 1024 \times 2048 = 2M$ FLOPs for the full state update.

### Key Variables

- $S_t \in \mathbb{R}^{d_k \times d_v}$ — full hidden state (used during training)
- $c_t \in \mathbb{R}^{d_c}$ — compressed latent state (used during inference)
- $W^{\text{down}} \in \mathbb{R}^{d_c \times (d_k d_v)}$ — state down-projection
- $W^{\text{up}} \in \mathbb{R}^{(d_k d_v) \times d_c}$ — state up-projection
- $\alpha_t \in (0,1)^{d_k}$ — per-dimension forget gate (input-dependent)
- $d_c$ — latent state dimension (hyperparameter, e.g., 64–256)
- $M_j \in \mathbb{R}^{d_c \times d_c}$ — precomputed per-gate-dimension transition matrices

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / mLSTM / Gated DeltaNet |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| State dim | $d_k = 1024$, $d_v = 2048$ (per head) |
| Heads | $n_h = 4$ |
| Latent dim | $d_c \in \{32, 64, 128, 256\}$ |

### Training Procedure

1. **Phase 1:** Train the model normally (full $d_k \times d_v$ state) to convergence
2. **Phase 2:** Learn the compression matrices $W^{\text{down}}, W^{\text{up}}$ by minimizing:
$$
\mathcal{L} = \sum_t \| q_t^\top S_t - \tilde{q}_t^\top c_t \|^2 + \lambda \| S_t - W^{\text{up}} W^{\text{down}} \text{vec}(S_t) \|^2
$$
This can be initialized via SVD of the empirical state covariance $\mathbb{E}[\text{vec}(S_t) \text{vec}(S_t)^\top]$ on a calibration set, then fine-tuned end-to-end for a small number of steps.

### Baseline

Standard GLA/mLSTM inference with full $d_k \times d_v$ state: $O(d_k d_v)$ HBM bandwidth per step per head per layer.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Generation throughput | $> 2\times$ baseline | Tokens/sec on A100/H100 (batch=1, greedy decode) |
| Peak memory | $< 0.3\times$ state memory | GPU memory for states only |
| Quality | $< 1.0$ ppl degradation | Perplexity on validation set |
| Latency per step | $< 0.5\times$ baseline | Time per decode step (ms) |

### Estimated Compute

**Small:** MVE on toy model: < 1 GPU-hour on A100.
**Medium:** Full experiment on 1.3B GLA model: ~50 GPU-hours (10 hours training + calibration + benchmarking).

## Expected Outcome

**If hypothesis is correct:**
- $2$–$4\times$ generation throughput improvement at $d_c = 64$ (comparable to MLA's 5.76× improvement)
- State memory reduced by $16$–$32\times$ per head (from $d_k \times d_v = 2M$ to $d_c = 64$–$256$ elements)
- Quality preserved within 1 perplexity point via careful initialization from SVD + fine-tuning

**If hypothesis is wrong:**
- If quality degrades significantly (> 2 ppl), the state has higher effective rank than expected — we learn that linear RNN states require large capacity and cannot be compressed for free
- If throughput doesn't improve (state update compute dominates), we learn that the weighted-sum transition $A_t^c$ is too expensive — the Kronecker structure doesn't simplify enough on actual hardware

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 128$, $d_k = 64$, $d_v = 128$, 2 heads (~200K params)
- **Task**: Language modeling on a small corpus (WikiText-2, first 10K tokens)
- **Latent dims**: $d_c \in \{8, 16, 32, 64\}$
- **Compute**: Single GPU, < 10 minutes

### Protocol
1. Train the small GLA model to convergence (< 5 min)
2. Collect states $S_t$ on validation data
3. Compute SVD of empirical state covariance to check effective rank
4. Initialize $W^{\text{down}}, W^{\text{up}}$ from top-$d_c$ SVD components
5. Measure readout error: $\|q_t^\top S_t - \tilde{q}_t^\top c_t\|^2 / \|q_t^\top S_t\|^2$
6. Run compressed inference and measure perplexity

### Success Criteria
- Effective rank of states is $\ll \min(d_k, d_v)$ (e.g., top-16 singular values capture > 90% of energy)
- $d_c = 32$ achieves $< 5\%$ relative readout error
- Compressed inference perplexity within 2 points of full inference
- Per-step latency measurably decreases (even without optimized kernel)

### Failure Criteria
- State effective rank is $> 0.5 \times \min(d_k, d_v)$ → states are not compressible
- $d_c = 64$ gives $> 20\%$ readout error → weight absorption doesn't work for accumulated states
- Accumulated compression error diverges over sequence length → the recurrent update amplifies compression artifacts

### Why This Test Is Sufficient
The core question is whether linear RNN states are low-rank enough to compress without quality loss. If the SVD analysis shows strong low-rank structure at small scale, this property is expected to hold (or even improve) at larger scales because: (a) larger models have more redundancy, and (b) the gating mechanism creates an exponential recency bias that concentrates state energy.

## Theoretical Analysis

### Bandwidth comparison

| Operation | Full State | Latent State |
|-----------|-----------|--------------|
| State read | $d_k d_v$ | $d_c$ |
| State write | $d_k d_v$ | $d_c$ |
| Readout compute | $O(d_k d_v)$ | $O(d_c)$ |
| State update compute | $O(d_k d_v)$ | $O(d_k d_c^2 + d_c)$ |
| Total HBM per step | $O(d_k d_v)$ | $O(d_c + d_k d_c^2 / \text{AI})$ |

With $d_k = 1024$, $d_v = 2048$, $d_c = 64$:
- **Full:** $1024 \times 2048 = 2{,}097{,}152$ elements = 4 MB per head
- **Latent:** $64$ elements = 128 bytes per head
- **Bandwidth reduction:** $32{,}768\times$ per head

The bottleneck shifts to the transition computation $A_t^c c_{t-1}$ which is $O(d_k d_c^2)$ FLOPs. For $d_k = 1024$, $d_c = 64$: $1024 \times 64^2 = 4.2M$ FLOPs — compute-bound on modern GPUs, not memory-bound.

### Memory access pattern analysis

1. **Coalesced access:** The latent state $c_t$ is a contiguous vector — loads/stores are perfectly coalesced.
2. **Cache-friendly:** $d_c = 64$ means the entire state fits in L1 cache (128 bytes in FP16). No HBM round-trips for the state during a decode step.
3. **Arithmetic intensity:** The transition $A_t^c c_{t-1}$ has intensity $\approx d_c = 64$ FLOPs/byte — well into the compute-bound regime on A100 (machine balance ~200 FLOPs/byte).
4. **Tensor cores:** The weighted sum $\sum_j \alpha_{t,j} M_j$ followed by $A_t^c c_{t-1}$ can be expressed as a batched GEMV, mapping to tensor cores via appropriate batching.

### Parallelism analysis

- **Across heads/layers:** Each head's state is compressed independently — full parallelism across heads and layers.
- **Within a step:** The transition computation is a weighted sum of small matrices — can be expressed as a single BRGEMM.
- **No warp divergence:** All operations are dense and uniform.
- **No sequential bottleneck:** The compression only affects inference, not training (which uses the full chunkwise parallel algorithm).

## Risks & Limitations

1. **Accumulated compression error:** Unlike MLA where each token's KV is compressed independently, the linear RNN state is recurrently updated in the compressed space. Errors may accumulate over time. **Mitigation:** Periodic "refresh" by running a few steps with full state every $k$ steps, or using a correction term.

2. **Training-inference asymmetry:** Training uses full state, inference uses compressed state. This requires a calibration/fine-tuning step to learn the compression matrices. **Mitigation:** SVD initialization + brief fine-tuning (similar to post-training quantization).

3. **Compressed update cost:** The weighted sum $\sum_j \alpha_{t,j} M_j$ with $d_k$ terms of size $d_c \times d_c$ requires $d_k \cdot d_c^2$ FLOPs. If $d_c$ is too large (e.g., $d_c = 256$ with $d_k = 1024$), this is $67M$ FLOPs — potentially more than the original $d_k \cdot d_v = 2M$ FLOPs. **Sweet spot:** $d_c \leq \sqrt{d_v} \approx 45$ for net compute savings.

4. **RoPE-like position sensitivity:** If the linear RNN uses position-dependent gates that can't be absorbed (analogous to MLA's RoPE issue), additional per-step compute may be needed. GLA's input-dependent gates are already handled by the Kronecker decomposition above.

5. **Not applicable during training:** The compression only benefits inference. Training still requires the full chunkwise parallel algorithm with full $d_k \times d_v$ states.

## Follow-up Experiments

1. **Joint training with latent state:** Instead of post-training compression, train the model to use a compressed state from the start (analogous to training MLA end-to-end rather than compressing a pre-trained model).
2. **Adaptive $d_c$:** Use different compression ratios for different layers (lower layers may have higher effective rank).
3. **Periodic state refresh:** Alternate between compressed steps and full-state steps to bound accumulated error.
4. **Comparison with state quantization:** Compare latent compression with FP8/INT8 quantization of the full state — different points on the accuracy-bandwidth tradeoff curve.
5. **Application to DeltaNet/DeltaProduct:** These models have non-diagonal state transitions, making the Kronecker decomposition of the transition update more complex. The SVD-based compression may still work but with different structure.

## Human Review

(To be filled by reviewer)

## References

- DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Chang, T., et al. (2025). PALU: Compressing KV-Cache with Low-Rank Projection. ICLR 2025.
- Dao, T. & Gu, A. (2024). Transformers are SSMs. ICML 2024.

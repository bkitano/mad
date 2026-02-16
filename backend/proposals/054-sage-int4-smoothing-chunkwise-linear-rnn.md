---
status: ongoing
priority: high
created: 2026-02-15
based_on: sageattention2-per-thread-mixed-precision-tiled-attention (190), tfla-two-level-tiled-chunkwise-parallelism (158), gla-secondary-chunking-log-space-gating (177), fused-chunkwise-ssd-atomic-state-passing (182), batch-reduce-gemm (batch-reduce-gemm)
experiment_number: 054
experiment_log: experiment-log-054.md
---

# SageAttention2-Style INT4 Smoothing for Chunkwise Linear RNN Training

## Hypothesis

Applying SageAttention2's **per-thread INT4 quantization with Q+K smoothing** to the intra-chunk $QK^\top$ matmul of chunkwise linear RNNs (GLA, mLSTM, Mamba-2/SSD), combined with **FP8 quantization for the $SV$ matmul**, will achieve $1.8$–$2.5\times$ wall-clock speedup for the intra-chunk computation on Ada/Hopper GPUs while maintaining training quality ($< 0.5\%$ perplexity degradation), because: (a) the $QK^\top$ in linear attention has **no softmax normalization** — making it strictly simpler to quantize than softmax attention, (b) SageAttention2's smoothing technique removes channel outliers that would otherwise destroy INT4 accuracy, producing a correction term that vanishes under the causal gate mask (analogous to how it vanishes under softmax), and (c) INT4 `mma.m16n8k64` provides $4\times$ tensor core throughput over BF16 — double the $2\times$ from FP8 alone.

## Background

### Why INT4 is better than FP8 for $QK^\top$ in linear attention

Existing proposal 050 covers FP8 precision for chunkwise linear RNNs. This proposal goes further by applying **INT4** specifically to the $QK^\top$ matmul — the largest single FLOP contributor in the intra-chunk computation. The key differences:

| Aspect | Proposal 050 (FP8) | This Proposal (INT4+FP8) |
|--------|-------------------|-------------------------|
| $QK^\top$ precision | FP8 E4M3 (2× throughput) | **INT4 (4× throughput)** |
| $SV$ precision | FP8 E4M3 (2× throughput) | FP8 E4M3 (2× throughput) |
| Smoothing | Not addressed | **Q+K smoothing (SageAttn2)** |
| Quantization granularity | Per-tile | **Per-thread (zero overhead)** |
| Expected intra-chunk speedup | 1.4–1.7× | **1.8–2.5×** |

### Why linear attention is easier to quantize than softmax attention

SageAttention2 achieves INT4 $QK^\top$ with 99.46% cosine similarity for **softmax** attention — where the softmax amplifies quantization errors exponentially. In linear attention:

1. **No softmax amplification:** The attention scores $P = QK^\top \odot D$ are used directly (with a causal gate mask $D$), not passed through $\exp(\cdot)$. Quantization errors in $QK^\top$ are not exponentially amplified.

2. **Gate mask absorbs the smoothing correction:** SageAttention2's smoothing computes $Q_i K_j^\top = \gamma(Q_i)\gamma(K_j)^\top + \bar{q}_i \gamma(K_j)^\top + b$ where $b$ is a constant per row that vanishes after softmax. In linear attention with causal gating, the correction $b$ is a rank-1 bias that can be absorbed into the gate mask or handled as a separate cheap GEMV — it doesn't vanish automatically but has negligible cost.

3. **Secondary chunking boundary is the natural INT4/FP32 split:** GLA's secondary chunking (trick 177) already separates computation into inter-sub-chunk matmuls (tensor-core friendly) and intra-sub-chunk log-space blocks (FP32). The INT4 quantization applies only to the inter-sub-chunk matmuls, leaving the precision-sensitive log-space computation unchanged.

4. **Gating provides implicit regularization:** The data-dependent gate $\alpha_t \in (0,1)$ naturally bounds the range of effective attention scores, reducing the dynamic range that quantization must capture.

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than BF16 TFLA on H100?** Yes — INT4 `mma.m16n8k64` provides $4\times$ throughput over BF16 `mma.m16n8k16`. The inter-sub-chunk $QK^\top$ matmuls constitute $\sim 40\%$ of total intra-chunk FLOPs. Even accounting for smoothing overhead (3.7% per SageAttn2) and the FP8 $SV$ path (2× throughput for another $\sim 40\%$ of FLOPs), the blended speedup on intra-chunk computation should be $\sim 2\times$.

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — start from the GLA Triton kernel. Before the inter-sub-chunk `tl.dot(Q_tile, K_tile.T)`:
   - Subtract channel mean: `Q_smooth = Q_tile - Q_tile.mean(dim=0)`
   - Per-thread INT4 quantize: `Q_int4, scale_Q = quantize_per_thread(Q_smooth)`
   - Compute INT4 matmul: `S_tile = tl.dot(Q_int4, K_int4.T) * scale_Q * scale_K`
   - Add smoothing correction: `S_tile += q_bar @ K_smooth.T`
   - Apply gate mask and continue with FP8 `S @ V`

3. **Does it reduce HBM bandwidth or increase compute utilization?** Both — INT4 activations are $4\times$ smaller than BF16, reducing HBM read traffic for $Q, K$ tile loads by $4\times$. INT4 tensor cores achieve $4\times$ higher TFLOPS. This pushes even the compute-bound intra-chunk regime toward higher throughput.

## Related Work

- **SageAttention2 (Zhang et al., ICML 2025)**: Introduced per-thread INT4 quantization with Q+K smoothing for softmax attention. Achieves 3× speedup over FlashAttention2 on RTX4090. **Our approach**: Adapts per-thread INT4 + smoothing to chunkwise linear attention, which has no softmax (simpler quantization) but has gated interactions (requires adapting the smoothing correction).
- **Proposal 050 (FP8 chunkwise linear RNN)**: Proposes FP8 for chunkwise linear RNN matmuls. **Our approach**: Goes further with INT4 for $QK^\top$ (4× vs 2× throughput), adds Q+K smoothing for accuracy, and retains FP8 for $SV$.
- **FlashAttention-3 FP8 (Dao et al., 2024)**: FP8 softmax attention on Hopper. Uses per-tile scaling. **Our approach**: Uses per-thread scaling (finer granularity, zero dequant overhead) and INT4 (4× vs 2× throughput).
- **SageAttention3 (NeurIPS 2025 Spotlight)**: Extends SageAttention2 to Blackwell architecture. Does not address linear attention variants.
- **GLA secondary chunking (Yang et al., ICML 2024)**: Separates inter-sub-chunk (tensor-core) from intra-sub-chunk (FP32 log-space) computation. **Our approach**: Applies INT4 specifically to the inter-sub-chunk matmuls, which are the dominant FLOP component (87%+ of intra-chunk FLOPs).

No directly related work found applying per-thread INT4 quantization with smoothing to chunkwise linear attention kernels.

## Mathematical Formulation

### GLA Intra-Chunk Computation with INT4 Quantization

**Standard GLA inter-sub-chunk matmul (BF16):**

For sub-chunk blocks $(i, j)$ with $i > j$:
$$
P_{[i][j]} = \left(\tilde{Q}_{[i]} \odot \Lambda_{[i]}\right) \left(\tilde{K}_{[j]} \odot \Gamma_{[j]}\right)^\top \in \mathbb{R}^{c \times c}
$$

This is a matmul of shape $(c \times d_k) \times (d_k \times c)$, currently computed in BF16 on tensor cores.

**With INT4 + smoothing:**

**Step 1 — Q+K Smoothing (per-block mean subtraction):**

Let $Q_s = \tilde{Q}_{[i]} \odot \Lambda_{[i]} \in \mathbb{R}^{c \times d_k}$ and $K_s = \tilde{K}_{[j]} \odot \Gamma_{[j]} \in \mathbb{R}^{c \times d_k}$.

$$
\bar{q} = \frac{1}{c} \sum_{m=1}^{c} Q_{s,m} \in \mathbb{R}^{d_k}, \quad \bar{k} = \frac{1}{c} \sum_{n=1}^{c} K_{s,n} \in \mathbb{R}^{d_k}
$$

$$
\gamma(Q_s) = Q_s - \mathbf{1}_c \bar{q}^\top, \quad \gamma(K_s) = K_s - \mathbf{1}_c \bar{k}^\top
$$

**Step 2 — Per-thread INT4 quantization:**

Each thread in the `mma.m16n8k64` instruction handles a specific subset of elements, indexed by $i_{\delta q}$:

$$
\delta_Q[i_{\delta q}] = \frac{\max(|\gamma(Q_s)[q_i[i_{\delta q}]]|)}{7}, \quad \hat{Q}[q_i[i_{\delta q}]] = \left\lfloor \frac{\gamma(Q_s)[q_i[i_{\delta q}]]}{\delta_Q[i_{\delta q}]} \right\rceil
$$

Similarly for $\hat{K}$ with groups indexed by $i_{\delta k}$.

**Step 3 — INT4 matmul + dequantization + correction:**

$$
P_{[i][j]} = \underbrace{\psi^{-1}_{\delta_Q, \delta_K}\left(\hat{Q} \hat{K}^\top\right)}_{\text{INT4 tensor core}} + \underbrace{\bar{q} \cdot \gamma(K_s)^\top}_{\text{GEMV correction}} + \underbrace{\gamma(Q_s) \cdot \bar{k}^\top + \bar{q} \cdot \bar{k}^\top}_{b \in \mathbb{R}^{c \times c}}
$$

where $\psi^{-1}$ is per-thread dequantization: $\hat{Q}\hat{K}^\top \times \delta_Q \times \delta_K$.

**Handling the bias term $b$:** Unlike softmax attention where the row-constant term vanishes after normalization, in linear attention with gating, $b$ is a rank-2 matrix. However:
- $b = \gamma(Q_s) \bar{k}^\top + \bar{q} \bar{k}^\top$ is the sum of two outer products
- Cost: $O(c \cdot d_k)$ for the outer products, negligible compared to the $O(c^2 \cdot d_k)$ matmul
- Can be computed in BF16 and added to the FP32 accumulator

**Step 4 — FP8 for the $SV$ matmul:**

$$
O_{[i]}^{\text{intra}} += \text{dequant}\left(\text{FP8-WGMMA}\left(\text{quant}(P_{[i][j]}), \text{quant}(V_{[j]})\right)\right)
$$

Following SageAttention2, $P$ is statically scaled to FP8 E4M3 range ($\times 448$), and $V$ uses per-channel FP8 scaling.

### Two-Level Accumulation for $SV$

Following SageAttention2's discovery that FP8 `mma` accumulates in FP22 (not true FP32):

$$
R_{ij}(\text{FP22}) = \text{mma}(\hat{P}_{ij}, \hat{V}_j)
$$

$$
O_{ij}(\text{FP32}) = O_{i,j-1}(\text{FP32}) + R_{ij}(\text{FP22})
$$

Flush from FP22 inner accumulator to FP32 outer accumulator every $b_k$ KV sub-chunks.

### Key Variables

- $C$ — primary chunk size (e.g., 128–256)
- $c$ — secondary sub-chunk size (e.g., 16)
- $d_k$ — key dimension per head
- $d_v$ — value dimension per head
- $\delta_Q, \delta_K$ — per-thread INT4 quantization scales
- $\bar{q}, \bar{k}$ — per-block channel means for smoothing
- $b_k$ — FP8 accumulator flush frequency

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA (primary), mLSTM (secondary) |
| Layers | $L = 24$ (1.3B scale) |
| Hidden dim | $d = 2048$ |
| Key dim | $d_k = 1024$ (per head) |
| Value dim | $d_v = 2048$ (per head) |
| Heads | $n_h = 4$ |
| Chunk size | $C = 128$ |
| Sub-chunk size | $c = 16$ |

### Baseline

1. **GLA (BF16 TFLA):** Standard TFLA kernel with BF16 tensor cores for all matmuls. Throughput: ~45 Ktok/s on H100 at 1.3B, seq 2048 (from GLA paper).
2. **GLA (FP8, proposal 050):** FP8 E4M3 for inter-sub-chunk matmuls. Expected: ~63 Ktok/s (1.4× baseline).

### Experimental Variants

| Variant | $QK^\top$ | $SV$ | Smoothing | Expected Speedup |
|---------|----------|------|-----------|-----------------|
| BF16 baseline | BF16 | BF16 | No | 1.0× |
| FP8 (prop 050) | FP8 | FP8 | No | 1.4× |
| INT4+FP8 (ours, no smooth) | INT4 | FP8 | No | 1.8× |
| **INT4+FP8 (ours, smooth)** | **INT4** | **FP8** | **Yes** | **1.8–2.0×** |

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intra-chunk throughput | $> 1.8\times$ BF16 | TFLOPS on H100/RTX4090 |
| End-to-end training throughput | $> 1.3\times$ BF16 | Tokens/sec (1.3B GLA, seq 2048) |
| Peak memory | $\leq$ BF16 baseline | Peak GPU memory |
| Quality (perplexity) | $< 0.5\%$ degradation | Val perplexity on SlimPajama |
| Cosine similarity ($QK^\top$) | $> 99\%$ | Mean cosine sim vs BF16 reference |

### Estimated Compute

**Small:** MVE kernel microbenchmark: 2 GPU-hours on H100/RTX4090.
**Medium:** Full 1.3B training comparison: ~200 GPU-hours (100B tokens, comparing BF16 vs INT4+FP8).

## Expected Outcome

**If hypothesis is correct:**
- $1.8$–$2.5\times$ speedup on intra-chunk computation (INT4 $QK^\top$: 4× throughput, FP8 $SV$: 2× throughput, blended with overhead)
- $1.3$–$1.5\times$ end-to-end training speedup (intra-chunk is ~70% of layer time)
- $<0.5\%$ perplexity degradation with smoothing (vs potentially $>2\%$ without)
- Cosine similarity $> 99.4\%$ on $QK^\top$ (matching SageAttention2's softmax results, likely better since no softmax amplification)
- Demonstrates that linear attention is **more quantization-friendly** than softmax attention

**If hypothesis is wrong:**
- If quality degrades significantly despite smoothing → the gate mask $D$ amplifies quantization errors differently than softmax. We learn about the precision requirements of gated attention.
- If speedup is $<1.5\times$ → the smoothing + quantization overhead (mean computation, GEMV correction) is too large relative to the INT4 throughput gain. We learn where the Roofline crossover is for mixed-precision linear attention.
- If INT4 works but FP8 $SV$ doesn't → the attention weight distribution is harder to quantize than the $QK^\top$ distribution in linear attention (opposite of softmax, where $P$ is in $[0,1]$).

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 256$, $d_k = 128$, $d_v = 256$, 2 heads (~1M params)
- **Task**: Forward-pass kernel microbenchmark + quality check on copying task
- **Data**: Synthetic sequence copying (1K samples, seq length 512)
- **Compute**: Single H100 or RTX4090, < 10 minutes
- **GPU requirement**: Ada or Hopper (INT4 `mma.m16n8k64` required)

### Protocol
1. Implement GLA chunkwise forward kernel in Triton (BF16 baseline)
2. Add INT4 quantization path for inter-sub-chunk $QK^\top$ tiles
3. Add Q+K smoothing (mean subtraction + GEMV correction)
4. Add FP8 path for $SV$ tiles
5. Compare:
   - **Accuracy**: Cosine similarity of $QK^\top$ output vs BF16 reference across 100 random inputs
   - **Throughput**: Measure TFLOPS for intra-chunk kernel at various $d_k, C, c$ configurations
   - **Quality**: Train tiny model on copying task, compare loss curves

### Success Criteria
- INT4 $QK^\top$ cosine similarity $> 99\%$ with smoothing (vs $< 85\%$ without)
- Intra-chunk kernel throughput $> 1.5\times$ BF16 baseline on H100
- Copying task final loss within $5\%$ of BF16 baseline
- Smoothing overhead $< 5\%$ of total kernel time

### Failure Criteria
- Cosine similarity $< 95\%$ even with smoothing → INT4 is insufficient for this operation
- Throughput $< 1.2\times$ → overhead exceeds precision throughput gain
- Copying task fails to converge → accumulated quantization error breaks learning

### Why This Test Is Sufficient
The cosine similarity of $QK^\top$ directly measures the quantization quality of the dominant operation. If this is high ($>99\%$) and the throughput gain is significant ($>1.5\times$), scaling to full training at 1.3B will maintain both properties because: (a) quantization error is independent of model scale, and (b) the throughput gain is a kernel-level property determined by hardware, not model size.

## Theoretical Analysis

### FLOP and bandwidth comparison

For a single GLA chunk ($C = 128$, $c = 16$, $d_k = 128$, $d_v = 256$):

| Operation | BF16 | INT4+FP8 (ours) |
|-----------|------|-----------------|
| Inter-sub-chunk $QK^\top$ | $N_s(N_s-1)/2 \times c^2 \times d_k$ | Same FLOPs, **4× TFLOPS** |
| Intra-sub-chunk (FP32) | $N_s \times c^2 \times d_k$ | Same (unchanged) |
| $SV$ matmul | $C^2 \times d_v$ | Same FLOPs, **2× TFLOPS** |
| Smoothing overhead | — | $+3.7\%$ (mean + GEMV) |
| Quantization overhead | — | $+0.35\%$ (per-thread) |

where $N_s = C/c = 8$.

**Effective throughput gain:**

Inter-sub-chunk $QK^\top$ is $\sim 77\%$ of intra-chunk matmul FLOPs (28 lower-triangular blocks vs 8 diagonal blocks). $SV$ is another $\sim 20\%$.

$$
\text{Speedup} = \frac{1}{0.77/4 + 0.20/2 + 0.03/1} = \frac{1}{0.193 + 0.10 + 0.03} = \frac{1}{0.323} \approx 3.1\times
$$

This is the theoretical maximum; accounting for smoothing overhead, memory access, and imperfect tensor core utilization, the practical speedup should be $1.8$–$2.5\times$.

### Memory access pattern analysis

1. **Coalesced access:** $Q, K$ tiles are loaded contiguously from HBM. INT4 format reduces tile size by $4\times$ — 4× fewer HBM reads.
2. **Per-thread quantization:** Each thread's scale aligns with the MMA instruction's thread-to-data mapping. No gather/scatter for dequantization — the scale is a single scalar per thread.
3. **Smoothing is local:** Mean computation and GEMV correction operate on data already in SRAM — no additional HBM traffic.
4. **Shared memory fits:** With $c = 16$ and $d_k = 128$: each INT4 $Q$ tile is $16 \times 128 / 2 = 1$ KB, each INT4 $K$ tile is $16 \times 128 / 2 = 1$ KB. Total SRAM for a sub-chunk pair: $< 4$ KB. H100 shared memory is 228 KB — room for many tiles.

### Parallelism analysis

- **All dominant operations are tensor core matmuls:** INT4 `mma.m16n8k64` for $QK^\top$, FP8 `mma.f32.f8.f8.f32` for $SV$.
- **No warp divergence:** The INT4 vs FP32 split follows the fixed secondary chunking boundary — all threads in a warp take the same path.
- **Full SM saturation:** Multiple sub-chunk pairs computed in parallel across warps; multiple heads and batch elements across thread blocks.
- **TMA async loads:** On Hopper, TMA can asynchronously load INT4 tiles while previous tiles are being computed — full overlap of memory and compute.

### Hardware requirements

| Feature | Required | Available on |
|---------|----------|-------------|
| INT4 `mma.m16n8k64` | Yes | Ada (RTX4090), Hopper (H100) |
| FP8 `mma.f32.f8.f8.f32` | Yes | Ada, Hopper |
| TMA async loads | Optional (for pipelining) | Hopper only |

Not available on A100 (Ampere) — would fall back to INT8 (2× throughput) instead of INT4 (4×).

## Risks & Limitations

1. **Ada/Hopper-only:** INT4 `mma.m16n8k64` is not available on A100 (the most common research GPU as of 2026). The technique requires Ada or Hopper GPUs. **Mitigation:** Fall back to INT8 `mma.m16n8k32` on A100 for 2× throughput (vs 4× on Hopper).

2. **Smoothing correction is non-trivial for gated attention:** In softmax attention, the row-constant bias $b$ vanishes after normalization. In gated linear attention, $b$ is modulated by the gate mask $D$, producing a non-constant correction. **Mitigation:** Compute $b \odot D$ explicitly — it's a rank-2 outer product masked by the causal gate, costing $O(c^2)$ per block (negligible).

3. **Backward pass requires higher precision:** INT4 forward may need FP16 or FP8 backward for gradient accuracy. **Mitigation:** Use INT4 forward + BF16 backward (as in SageAttention2's training mode), or INT4 forward + FP8 backward.

4. **Dynamic range of gated Q/K:** After gate absorption ($\tilde{Q} = Q \odot \Lambda$), the dynamic range may change. If $\Lambda$ has extreme values, smoothing alone may not suffice. **Mitigation:** Monitor the distribution of gated Q/K and adjust smoothing if needed (per-sub-chunk mean instead of per-block).

5. **Triton INT4 support maturity:** Triton's INT4 `tl.dot` support is still evolving. May need to use inline PTX or CUTLASS for the INT4 matmul. **Mitigation:** Start with CUTLASS INT4 GEMM and wrap in a custom CUDA extension, or use Triton's native INT4 when available.

## Follow-up Experiments

1. **INT4 for training (forward + backward):** If INT4 forward works, test INT4 for the backward pass gradient computation ($dQ, dK$ involve similar matmuls). This could give up to $4\times$ speedup on the entire training step.

2. **Adaptive precision per layer:** Lower layers may tolerate more aggressive quantization than upper layers. Use INT4 for lower layers and FP8/BF16 for upper layers.

3. **Combined with TFLA two-level tiling:** Apply INT4 within TFLA's inner tiles. The two-level tiling provides larger effective chunk sizes (higher arithmetic intensity) while INT4 provides higher per-FLOP throughput — multiplicative benefits.

4. **Extend to DeltaNet/DeltaProduct:** These models have a UT transform step before the attention matmul. The UT-transformed $K, V$ matrices may have different quantization properties — test whether smoothing is still effective.

5. **INT4 for Mamba-2/SSD:** Mamba-2's SSD chunkwise computation has the same $CB^\top$ and $(CB^\top)X$ matmul structure. Apply INT4+FP8 to the SSD kernel (which already has a fused version from trick 182).

6. **Comparison with SageAttention3 (Blackwell):** On Blackwell GPUs with native FP4 tensor cores, the throughput gap may narrow or change character. Benchmark on B100/B200 when available.

## Human Review

(To be filled by reviewer)

## References

- Zhang, J., Huang, H., Zhang, P., Wei, J., Zhu, J., & Chen, J. (2025). SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization. ICML 2025. arXiv:2411.10958.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS 2024. arXiv:2407.08691.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060.
- Georganas, E. et al. (2019). High-Performance Deep Learning via a Single Building Block. arXiv:1906.06440.
- Astra, R., Dao, T., & Hoque, A. (2026). Accelerating Mamba2 with Kernel Fusion. PyTorch Blog.

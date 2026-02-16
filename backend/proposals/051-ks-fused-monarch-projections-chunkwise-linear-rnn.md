---
status: ongoing
priority: high
created: 2026-02-15
based_on: kronecker-sparse-output-stationary-fused-kernel (195), monarch-matrix-factorization (076), gla-secondary-chunking-log-space-gating (177), fused-chunkwise-ssd-atomic-state-passing (182), chunkwise-parallel-scan (026)
experiment_number: 051
experiment_log: experiment-log-051.md
---

# KS-Fused Monarch Projections for Chunkwise Linear RNN Training

## Hypothesis

Replacing the dense projection matrices ($W_Q, W_K, W_V, W_B, W_C$, gate projections) in chunkwise linear RNNs (GLA, Mamba-2/SSD) with **Monarch-factored matrices** computed via the **KS fused output-stationary kernel** (trick 195) during *training* reduces the projection GEMM cost by $\sim 40\%$ while maintaining $\geq 95\%$ of the model quality, yielding a $15$–$25\%$ end-to-end wall-clock training speedup. The key is that the fused kernel eliminates the permutation overhead that made Monarch training impractical, and the projections dominate the non-scan portion of the chunkwise forward/backward passes.

## Background

In chunkwise linear RNN training (GLA, Mamba-2, TFLA), the computational cost splits into two parts:

1. **Projection GEMMs**: Computing $Q = XW_Q$, $K = XW_K$, $V = XW_V$, plus gate/state projections. These are standard dense matrix multiplies of shape $(B \cdot T, d) \times (d, d_h)$ — **identical in structure to Transformer FFN projections**.

2. **Chunkwise scan/attention**: The intra-chunk quadratic attention ($O(C^2 d)$) and inter-chunk state recurrence ($O(T d_k d_v / C)$).

For typical configurations (e.g., GLA with $d = 2048$, 4 heads, $d_k = d_v = d$), the projection GEMMs account for **40–60% of forward pass FLOPs** and are purely GEMM operations — the dominant kernel type on modern GPUs. Yet all existing proposals (001–050) focus exclusively on optimizing part (2) — the scan/attention mechanics — while leaving part (1) as dense GEMMs.

The NeurIPS 2024 paper "Effectively Training LLMs with Structured Feedforward Layers" showed that Monarch/block-diagonal structures can replace FFN layers during training at scale (1.3B params), achieving 17% throughput improvement with comparable perplexity when combined with "self-guided training" (dense residual warm-start). However, they used standard BMM-based Monarch kernels, which suffer from the permutation overhead (up to 50% of runtime per trick 195's analysis), and they tested only on Transformer FFNs, **not** on linear RNN projection layers.

The KS fused kernel (trick 195) eliminates this permutation overhead via an output-stationary tiling strategy that fuses permute-GEMM-permute into a single kernel launch. It achieves **1.4× median speedup** over BMM on 600+ KS patterns and **22% end-to-end speedup** on ViT-S/16. But it was only tested for *inference* and only on dense Transformers. **No work has applied KS fused kernels to training, or to linear RNN projection layers.**

This proposal fills both gaps simultaneously.

## Related Work

- **"Effectively Training LLMs with Structured Feedforward Layers" (NeurIPS 2024)**: Tested Monarch/block-diagonal FFN replacements for Transformer training at 1.3B scale. Found that structured FFNs need "self-guided training" (warm-start from dense for stability). Used standard BMM kernels (not KS fused). Tested only on Transformers, not linear RNNs. Found circulant/ACDC structures underperformed. **Our approach**: Uses KS fused kernels (1.4× faster than BMM) and targets linear RNN projections, which have different dimension profiles than Transformer FFNs.

- **MonarchAttention (May 2025, arXiv:2505.18698)**: Approximates the *attention matrix* with Monarch structure for sub-quadratic attention. Different from our approach: we structure the *weight matrices*, not the attention pattern. MonarchAttention is for softmax Transformers; we target linear RNNs.

- **"Fast Inference with Kronecker-Sparse Matrices" (ICML 2025)**: Introduces the KS fused kernel achieving 1.4× over BMM, 6.5× over dense. Only tested for *inference*, with no gradient support. **Our approach**: Extends to *training* by implementing the backward pass through the KS fused kernel and applying to linear RNN projections.

- **Mamba-2/SSD (ICML 2024)**: Established the chunkwise parallel framework but uses dense projections throughout. No structured weight matrices explored.

- **GLA (ICML 2024)**: Hardware-efficient training with secondary chunking for tensor core utilization. Dense projections throughout.

## Mathematical Formulation

**Standard GLA/SSD Projection (Dense):**

For input $X \in \mathbb{R}^{T \times d}$, the projections compute:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$. Each is a dense GEMM: $O(T \cdot d \cdot d_k)$.

**Monarch-Factored Projection (Proposed):**

Replace each dense weight with a 2-factor Monarch product:

$$
W_Q = P_b^\top L_Q^{(2)} P_b R_Q^{(2)} \cdot P_b^\top L_Q^{(1)} P_b R_Q^{(1)}
$$

where $L_Q^{(i)} = \text{diag}(L_{Q,1}^{(i)}, \ldots, L_{Q,\sqrt{d}}^{(i)})$, $R_Q^{(i)} = \text{diag}(R_{Q,1}^{(i)}, \ldots, R_{Q,\sqrt{d}}^{(i)})$, with each block $L_{Q,j}^{(i)}, R_{Q,j}^{(i)} \in \mathbb{R}^{\sqrt{d} \times \sqrt{d}}$.

**KS Fused Computation:**

Each Monarch factor has KS pattern $\pi = (a, \sqrt{d}/a, \sqrt{d}/a, d/\sqrt{d})$ for appropriate $a$. The projection $Q = X \cdot M_Q$ is computed as:

$$
Q = \text{KS\_Fused}(X, \text{Factor}_2, \pi_2) \circ \text{KS\_Fused}(\cdot, \text{Factor}_1, \pi_1)
$$

where each `KS_Fused` call is the single-launch fused kernel from trick 195, computing:

$$
Y[:, \text{row}_{i,j}] = X[:, \text{col}_{i,j}] \cdot K^\top[\text{col}_{i,j}, \text{row}_{i,j}]
$$

for each tile $(i, j)$, with no explicit permutation.

**Self-Guided Warm-Start (from NeurIPS 2024):**

For the first $\tau$ training steps, use a dense residual:

$$
W_Q^{(\text{warm})} = \alpha \cdot M_Q + (1 - \alpha) \cdot W_Q^{(\text{dense})}
$$

with $\alpha$ linearly increasing from 0 to 1 over $\tau$ steps. After warm-up, discard $W_Q^{(\text{dense})}$ and train purely with Monarch weights.

**Key Variables:**
- $d$ — model dimension (e.g., 2048)
- $d_k, d_v$ — head dimensions
- $\sqrt{d}$ — Monarch block size (e.g., $\sqrt{2048} \approx 45$; use $d = 2048 = 32 \times 64$ for cleaner factorization)
- $T$ — sequence length
- $\tau$ — warm-up steps (typically 5–10% of total)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Base Model | GLA (Gated Linear Attention) |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ (factors as $32 \times 64$) |
| Heads | $H = 4$ |
| Head dims | $d_k = 1024$, $d_v = 2048$ |
| Chunk size | $C = 128$ |
| Monarch factors | $k = 2$ per projection |
| KS pattern | $(32, 64, 64, 1)$ for $d \to d$ projections |
| Warm-start | $\tau = 2000$ steps with dense residual |

### Projections Replaced

| Projection | Shape | Dense Params | Monarch Params | Compression |
|-----------|-------|-------------|----------------|-------------|
| $W_Q$ | $d \to d_k$ | $2048 \times 1024$ | $2 \times 2048\sqrt{1024}$ | $3.1\times$ |
| $W_K$ | $d \to d_k$ | $2048 \times 1024$ | $2 \times 2048\sqrt{1024}$ | $3.1\times$ |
| $W_V$ | $d \to d_v$ | $2048 \times 2048$ | $2 \times 2048\sqrt{2048}$ | $4.5\times$ |
| $W_G$ (gate) | $d \to d_k$ | $2048 \times 1024$ | $2 \times 2048\sqrt{1024}$ | $3.1\times$ |
| $W_O$ (output) | $d_v \to d$ | $2048 \times 2048$ | $2 \times 2048\sqrt{2048}$ | $4.5\times$ |

Total per-layer projection parameter reduction: $\sim 3.5\times$

### Baseline
1. **GLA-2048 (dense projections)**: Standard GLA with dense $W_Q, W_K, W_V, W_G, W_O$. Uses FlashLinearAttention kernel.
2. **GLA-2048 (BMM Monarch, no fusion)**: Monarch projections with standard 3-kernel BMM.
3. **Mamba-2 2048 (dense)**: For cross-architecture comparison.

### Memory Access Pattern Analysis

The KS fused kernel's access pattern:
- **Arithmetic intensity**: Each tile $(i, j)$ computes a dense $(B, c) \times (c, b) \to (B, b)$ matmul. For $c = b = \sqrt{d} = 45$, the AI is $\sim 45$ FLOPs/byte — **compute-bound** on A100 (312 TFLOP threshold) for batch sizes $B \geq 32$
- **Coalescing**: In batch-size-last (BSL) memory layout, reads of $X[:, \text{col}_{i,j}]$ are contiguous. In standard BSF layout, reads are strided with stride $d$. BSL gives 2× additional speedup per trick 195.
- **Shared memory**: Each tile loads $B \times c + c \times b$ elements to SMEM. For $c = b = 45, B = 128$: $128 \times 45 + 45 \times 45 \approx 8K$ elements = 16KB — well within H100's 256KB SMEM.

### Parallelism Analysis

- **Independent tiles**: Each Monarch factor decomposes into $a \times d$ independent tile matmuls. For $(a, b, c, d) = (32, 64, 64, 1)$: 32 independent tiles, each a dense $B \times 64 \times 64$ matmul — **saturates GPU SMs**.
- **Tensor core mapping**: Each tile is a standard dense GEMM of size $\geq 64$, **directly maps to WGMMA/MMA** instructions.
- **No warp divergence**: All tiles have identical computation; perfect load balance.
- **Sequential bottleneck**: Two Monarch factors must be applied sequentially (factor 1 output feeds factor 2 input). This is 2 kernel launches (or 1 with intermediate in SMEM if tile sizes align).

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $\geq 1.15\times$ GLA-dense tokens/sec | Timed training step, batch 32, seq 2048 |
| Projection speedup | $\geq 1.4\times$ per projection GEMM | Isolated projection benchmark |
| Memory | $\leq 0.85\times$ GLA-dense peak memory | Peak GPU memory during training |
| Perplexity | $\leq 1.03\times$ GLA-dense perplexity | WikiText-103/SlimPajama |
| MQAR recall | $\geq 0.95\times$ GLA-dense at 4 KV pairs | MQAR benchmark |

### Estimated Compute

**MVE**: ~10 minutes on single A100 (~$0.50)
**Small-scale (380M)**: 8 GPU-hours on A100 (~$32)
**Full-scale (1.3B)**: 64 GPU-hours on A100 (~$260)

## Expected Outcome

**If hypothesis is correct:**
- **Projection GEMM speedup**: $1.4\times$–$1.6\times$ per projection (matching trick 195's median, enhanced by training-specific batching)
- **End-to-end training speedup**: $15$–$25\%$ (projections are 40–60% of forward pass; $1.4\times$ speedup on 50% → $1.17\times$ overall)
- **Memory reduction**: $\sim 3.5\times$ fewer projection parameters → smaller optimizer state, enabling larger batch sizes
- **Quality preservation**: Monarch$^2$ can represent DFT/DCT/Hadamard and approximate arbitrary matrices; with warm-start, perplexity within 3% of dense

**If hypothesis is wrong:**
- **If quality degrades**: Monarch$^2$ expressivity is insufficient for the projection matrices in linear RNNs. Learn: linear RNN projections may need higher expressivity than Transformer FFNs (where Monarch works). Fix: increase to Monarch$^3$ or use hybrid (Monarch for $W_V, W_O$, dense for $W_Q, W_K$).
- **If speedup is small**: KS fused kernel backward pass overhead negates forward gains. Learn: the autograd through 2 Monarch factors has 4 backward KS calls (2 per factor, for input and weight gradients). Fix: fuse the forward+backward of a single Monarch factor into one kernel.
- **If warm-start is insufficient**: Training instability (loss spikes) when transitioning from dense to Monarch. Learn: the self-guided schedule needs tuning for linear RNNs. Fix: extend warm-up period or keep a small dense residual throughout training.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GLA, $d = 256$ ($16 \times 16$ Monarch blocks), $d_k = d_v = 128$, ~1M params
- **Task**: Language modeling on TinyStories (100K sequences, vocab 32K)
- **Monarch**: 2-factor Monarch for $W_Q, W_K, W_V$ projections only (simplest case)
- **No KS fusion**: Use standard BMM for MVE (test quality first, speed second)
- **Warm-start**: 200 steps with dense residual
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- Monarch-projection GLA achieves perplexity within $5\%$ of dense-projection GLA after 5K training steps
- Projection parameter count reduced by $\geq 2\times$
- No training instability (no loss spikes after warm-start period)

### Failure Criteria
- Perplexity $> 1.10\times$ dense after 5K steps (Monarch expressivity too low for projections)
- Training diverges or exhibits persistent loss spikes (optimization landscape issue)
- Warm-start doesn't converge in 200 steps (schedule needs fundamental rethinking)

### Why This Test Is Sufficient
- Quality is the primary risk; if Monarch projections can match dense quality at small scale, speed gains are guaranteed by the KS fused kernel benchmarks (already validated on 600+ patterns)
- The projection structure is scale-invariant: $(B, d) \times (d, d_k)$ with Monarch factorization works identically at any $d$, just with different block sizes
- TinyStories exercises real language modeling dynamics (unlike synthetic tasks) while being small enough for rapid iteration

## Theoretical Analysis

**Complexity comparison (per layer, forward pass):**

| Operation | Dense Projections | Monarch Projections (KS Fused) |
|-----------|------------------|-------------------------------|
| $W_Q$ projection | $O(T \cdot d \cdot d_k)$ | $O(2 \cdot T \cdot d \cdot \sqrt{d_k})$ |
| $W_K$ projection | $O(T \cdot d \cdot d_k)$ | $O(2 \cdot T \cdot d \cdot \sqrt{d_k})$ |
| $W_V$ projection | $O(T \cdot d \cdot d_v)$ | $O(2 \cdot T \cdot d \cdot \sqrt{d_v})$ |
| Gate projection | $O(T \cdot d \cdot d_k)$ | $O(2 \cdot T \cdot d \cdot \sqrt{d_k})$ |
| Output projection | $O(T \cdot d_v \cdot d)$ | $O(2 \cdot T \cdot d_v \cdot \sqrt{d})$ |
| **Total projections** | $O(T \cdot d \cdot (3d_k + 2d_v))$ | $O(2T \cdot d \cdot (3\sqrt{d_k} + 2\sqrt{d_v}))$ |
| Chunkwise attention | $O(T \cdot C \cdot d_k)$ | $O(T \cdot C \cdot d_k)$ (unchanged) |
| Inter-chunk scan | $O(T \cdot d_k \cdot d_v / C)$ | $O(T \cdot d_k \cdot d_v / C)$ (unchanged) |

**Crossover analysis:** For $d = 2048$, $d_k = 1024$:
- Dense: $T \cdot 2048 \cdot 1024 \approx 2.1M \cdot T$ FLOPs per projection
- Monarch: $2 \cdot T \cdot 2048 \cdot 32 \approx 131K \cdot T$ FLOPs per projection
- **FLOP reduction**: $16\times$ per projection

But the real speedup is lower due to:
1. Each Monarch factor is a batched small GEMM (lower hardware utilization than one large GEMM)
2. Two sequential factor applications
3. KS fused kernel overhead (tile index computation)

**Realistic speedup**: $1.4\times$–$1.6\times$ per projection (matching empirical KS fused results), yielding $1.15$–$1.25\times$ end-to-end.

**Memory analysis:**

| Component | Dense | Monarch |
|-----------|-------|---------|
| Projection weights per layer | $5 \cdot d^2 \approx 21M$ params | $5 \cdot 2 \cdot 2 \cdot d^{3/2} \approx 1.9M$ params |
| Optimizer states (AdamW) | $2 \times 21M = 42M$ floats | $2 \times 1.9M = 3.8M$ floats |
| Activation memory | $5 \cdot T \cdot d$ (unchanged) | $5 \cdot T \cdot d$ (unchanged) |

At $d = 2048$, $L = 24$ layers: weight memory drops from $504M \to 45M$ parameters ($11\times$), freeing $\sim 1.8$GB in float16.

## Hardware-Specific Considerations

### Tensor Core Utilization
- Each KS tile is a dense $B \times c \times b$ matmul where $b, c \geq 32$ — **directly maps to WGMMA** (H100) or **MMA** (A100)
- The $16 \times 16$ minimum MMA tile is satisfied for $\sqrt{d} \geq 16$ (i.e., $d \geq 256$)
- For $d = 2048$ with factorization $32 \times 64$: block sizes are 32 and 64, both excellent for tensor cores

### TMA Async Loads
- The KS fused kernel can use H100 TMA for the strided loads of $X[:, \text{col}_{i,j}]$
- BSL layout enables contiguous TMA descriptors

### Register Pressure
- Each tile accumulates a $B_{\text{warp}} \times b$ result in registers
- For $b = 64$, $B_{\text{warp}} = 4$: 256 registers per warp — within budget

### CUDA Kernel Sketch
```
// One thread block per Monarch tile (i, j)
// Total tiles: a * d per Monarch factor
__global__ void ks_monarch_fwd(X, L_blocks, R_blocks, Y, ...) {
    int tile_id = blockIdx.x;
    // Compute row/col indices from KS pattern
    // Load X[:, col] and block[j] to SMEM via TMA
    // Dense GEMM in registers via WGMMA
    // Store Y[:, row] to global mem
}
```

## Risks & Limitations

1. **Backward pass through KS fused kernel**: The existing KS fused kernel (trick 195) only supports forward inference. Training requires computing $\frac{\partial L}{\partial X}$ and $\frac{\partial L}{\partial W}$ through the Monarch structure. The backward is structurally identical (a KS matmul with transposed pattern), but implementing and optimizing it is engineering effort.

2. **Batch-size-first layout incompatibility**: PyTorch defaults to BSF, but KS fused kernel benefits from BSL. Switching layout mid-pipeline (between projection and scan) may negate some gains via layout conversion overhead.

3. **Warm-start overhead**: The first $\tau$ steps use dense+Monarch (roughly $1.5\times$ memory). This is a one-time cost but may limit maximum batch size during warm-up.

4. **Factorization constraints**: $d$ must factor as $p \times q$ for balanced Monarch. Non-square dimensions (e.g., $d = 2048, d_k = 1024$) require rectangular Monarch variants or padding.

5. **Limited to projection layers**: This only accelerates the GEMM portion. If the scan/attention portion dominates (e.g., very long sequences), the overall speedup diminishes.

## Follow-up Experiments

1. **KS fused backward kernel implementation**: Implement the backward pass through the KS fused kernel in Triton/CUDA, enabling end-to-end training without falling back to BMM for gradients.

2. **Fused projection + chunking**: Merge the Monarch projection and the chunkwise reshape into a single kernel — project and partition simultaneously, avoiding an intermediate HBM write of the full projected tensor.

3. **2:4 sparsity on Monarch blocks**: Apply 2:4 structured sparsity (trick 136) to the individual $\sqrt{d} \times \sqrt{d}$ Monarch blocks. Each block is small enough that 2:4 pruning should maintain quality, yielding an additional $2\times$ speedup via Sparse Tensor Cores composing with the Monarch structure.

4. **Mixed Monarch/dense layers**: Profile which layers benefit most from Monarch projections. Hypothesis: deeper layers (which learn more redundant features) tolerate Monarch better than early layers.

5. **Scaling to 7B+**: Test whether the 3.5× parameter reduction in projections enables training larger models within the same GPU memory budget.

## Human Review

(To be filled by reviewer)

## References

- Gonon, Zheng, Carrivain & Le (2025). Fast Inference with Kronecker-Sparse Matrices. ICML 2025. arXiv:2405.15013.
- Dao, Chen, et al. (2022). Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML 2022.
- Yang, Wang, Shen, Panda & Kim (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024.
- Dao & Gu (2024). Transformers are SSMs. ICML 2024.
- Thangarasa, Gupta, Marshall, Li, Leong & DeVries (2024). Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers. NeurIPS 2024. arXiv:2406.16450.
- Astra, Dao & Hoque (2026). Accelerating Mamba2 with Kernel Fusion. PyTorch Blog.
- MonarchAttention (2025). Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention. arXiv:2505.18698.

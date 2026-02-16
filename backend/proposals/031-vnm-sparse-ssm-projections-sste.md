---
status: ongoing
priority: high
created: 2026-02-15
based_on: vnm-hierarchical-structured-sparsity, smooth-ste-continuous-sparse-projection, two-four-structured-sparsity, bilinear-gating-glu, input-dependent-gating, chunkwise-parallel-scan, io-aware-tiling
experiment_number: 031
experiment_log: experiment-log-031.md
---

# V:N:M Sparse SSM Projections with S-STE Training

## Hypothesis

Applying V:N:M hierarchical structured sparsity (60–75%) to the **projection matrices** ($W_Q, W_K, W_V, W_O$, and gate projections) within a gated chunkwise SSM (Mamba-2/GLA-style), trained from scratch using S-STE continuous pruning, will achieve $1.3$–$1.7\times$ wall-clock training speedup and $1.5$–$2.0\times$ inference speedup on A100/H100 GPUs with $< 3\%$ quality degradation, because (a) projections dominate the parameter count and FLOP budget of modern SSM layers, (b) VNM sparsity at 60–75% produces real hardware speedups on Sparse Tensor Cores (unlike 2:4's marginal 1.1–1.3×), and (c) the SwiGLU gating structure naturally compensates for sparsity-induced information loss through its multiplicative feature selection.

## Background

### The real computational bottleneck in SSM layers

In Mamba-2 and GLA-style architectures, the layer computation breaks down as:

| Component | FLOPs (per token) | % of Layer |
|-----------|-------------------|------------|
| Input projections ($W_Q, W_K, W_V, W_{gate}$) | $4 \times 2Bd \cdot d_{model}$ | ~60% |
| State recurrence / scan | $O(Bn)$ or $O(Bn^2)$ | ~5–15% |
| Output projection ($W_O$) | $2Bd \cdot d_{model}$ | ~15% |
| FFN (SwiGLU $W_1, W_2, W_3$) | $3 \times 2d_{model} \cdot d_{ff}$ | (separate block) |

The **projection matrices** ($W_Q, W_K, W_V, W_O, W_{gate}$) constitute the majority of both parameters and FLOPs. Yet all existing SSM efficiency proposals (001–030) focus on the state transition structure, not the projections. This is the wrong target — the state transition is already efficient ($O(n)$ for diagonal, $O(n \log n)$ for circulant) while the projections remain fully dense $O(d_{model}^2)$ operations.

### Why VNM over plain 2:4

Plain 2:4 structured sparsity achieves only 50% sparsity and, critically, delivers only **1.1–1.3× actual speedup** on current cuSPARSELt implementations (far below the 2× theoretical peak). This is because:
1. The sparse metadata overhead erodes gains at small matrix dimensions
2. Library overheads for sparse format conversion
3. Limited occupancy at typical LLM layer sizes

VNM hierarchical sparsity addresses this by pushing sparsity to 60–75%, which yields **1.49–1.7× measured speedup** (Zhao et al., 2025) because:
1. Column pruning eliminates entire columns, reducing the effective matrix dimension (not just nonzeros)
2. The inner 2:4 pattern still maps to Sparse Tensor Cores
3. Higher sparsity = fewer HBM bytes loaded per token

### Why S-STE for training from scratch

V:N:M was originally validated as a **post-training** technique (prune a dense model, then fine-tune with LoRA). Training sparse from scratch is harder but more valuable — it avoids the expensive dense pre-training phase entirely. S-STE (smooth straight-through estimator) is the only pruning method that provides a **continuous, hyper-parameter-free** sparse projection, eliminating the mask oscillation that kills sparse-from-scratch training.

The key insight: S-STE's soft-thresholding naturally extends to VNM's two-level structure. First apply soft column importance scoring, then apply S-STE's continuous 2:4 projection within retained columns. Both operations are differentiable.

### Why SwiGLU gating compensates for sparsity

SwiGLU's multiplicative gating structure ($\text{Swish}(xW) \otimes xV$) is particularly resilient to sparsification because:
1. The gate path ($\sigma(xW)$) learns **which features to activate** — if some features are zeroed by sparsity, the gate learns to route around them
2. The value path ($xV$) provides redundant capacity — information lost in one projection can be recovered through the other
3. Empirically, GLU-gated networks tolerate pruning better than non-gated networks (observed in LLaMA pruning studies)

### Distinction from Proposal 024

Proposal 024 applies 2:4 sparsity specifically to the **state transition matrix** $A \in \mathbb{R}^{n \times n}$ with learned permutations via Sinkhorn. Our proposal is fundamentally different:

| Aspect | Proposal 024 | This Proposal (031) |
|--------|-------------|-------------------|
| **Target** | State transition $A$ ($n \times n$) | Projection matrices ($d_{model} \times d$) |
| **Sparsity level** | 50% (2:4 only) | 60–75% (VNM hierarchical) |
| **Training method** | S-STE + Sinkhorn permutation | S-STE + static VNM column selection |
| **FLOP impact** | Minor (scan is ~10% of layer) | Major (projections are ~75% of layer) |
| **Speedup target** | $1.5$–$1.8\times$ (scan only) | $1.3$–$1.7\times$ (whole layer) |
| **Permutation learning** | Learned (Sinkhorn) — iterative, GPU-unfriendly | Static (VNM) — one-time, offline |

## Mathematical Formulation

### Standard Gated SSM Layer (Mamba-2 / GLA style)

$$
Q_t = x_t W_Q, \quad K_t = x_t W_K, \quad V_t = x_t W_V
$$
$$
g_t = \text{Swish}(x_t W_{gate})
$$
$$
h_t = \text{diag}(\alpha_t) h_{t-1} + K_t^\top V_t \quad \text{(state update via scan)}
$$
$$
o_t = Q_t h_t \quad \text{(readout)}
$$
$$
y_t = (o_t \odot g_t) W_O \quad \text{(gated output projection)}
$$

where $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}$, $W_V \in \mathbb{R}^{d_{model} \times d_v}$, $W_{gate} \in \mathbb{R}^{d_{model} \times d_v}$, $W_O \in \mathbb{R}^{d_v \times d_{model}}$.

### VNM-Sparse Projections (Proposed)

Replace each dense projection with a VNM-sparse variant trained via S-STE:

**Step 1 — VNM Column Selection (per $V \times M$ block):**

For each $V \times M$ block of weight matrix $W$, compute column importance:

$$
\text{Imp}_j = \sum_{i=1}^{V} |W_{ij}| \cdot (\|X_j\|_2)^a
$$

Retain the top-4 columns by importance. This is a **soft selection** during training:

$$
\text{mask}_j^{col} = \text{TopK-STE}(\text{Imp}_j, k=4)
$$

using straight-through estimation for gradient flow.

**Step 2 — S-STE 2:4 Pruning (within retained columns):**

Within each row's 4 retained columns, apply S-STE soft-thresholding:

$$
(S_{\text{soft}}(\mathbf{a}))_i = \text{sign}(a_i) \cdot \max(|a_i| - |a_{(2)}|, 0)
$$

with frozen scaling factor:

$$
\tilde{W} = \beta \cdot S_{\text{soft}}(W \odot \text{mask}^{col})
$$

**Step 3 — Sparse MatMul on Tensor Cores:**

At inference, the VNM-sparse weight is compiled to cuSPARSELt format. The column pruning reduces the effective matrix dimensions, and the inner 2:4 pattern maps to Sparse Tensor Cores:

$$
Q_t = x_t \tilde{W}_Q \quad \text{via VNM-SpMM: } O\left(\frac{2}{M} \cdot d_{model} \cdot d_k\right) \text{ effective FLOPs}
$$

**Combined Layer (Training):**

$$
Q_t = x_t \cdot [\beta_Q \cdot S_{\text{soft}}(\text{VNM}(W_Q^{dense}))]
$$
$$
K_t = x_t \cdot [\beta_K \cdot S_{\text{soft}}(\text{VNM}(W_K^{dense}))]
$$
$$
V_t = x_t \cdot [\beta_V \cdot S_{\text{soft}}(\text{VNM}(W_V^{dense}))]
$$
$$
g_t = \text{Swish}\left(x_t \cdot [\beta_g \cdot S_{\text{soft}}(\text{VNM}(W_{gate}^{dense}))]\right)
$$
$$
h_t = \text{diag}(\alpha_t) h_{t-1} + K_t^\top V_t \quad \text{(scan — unchanged)}
$$
$$
o_t = Q_t h_t
$$
$$
y_t = (o_t \odot g_t) \cdot [\beta_O \cdot S_{\text{soft}}(\text{VNM}(W_O^{dense}))]
$$

**Inference (Sparse):**

All projections are hardened to VNM-sparse format. Dense weight copies are discarded.

### Memory Access Pattern Analysis

**Coalesced access:** VNM sparse matrices in cuSPARSELt format maintain coalesced memory access patterns because column pruning preserves the row-major layout (columns are contiguous in memory) and the inner 2:4 pattern uses hardware-native compressed storage.

**Arithmetic intensity:** For a VNM-sparse matmul at 75% sparsity (V:2:8):
$$
\text{AI} = \frac{2 \cdot d_{model} \cdot d_k \cdot (2/M)}{d_{model} \cdot d_k \cdot (2/M) \cdot 2} = \frac{d_{model} \cdot d_k / 4}{d_{model} \cdot d_k / 4} = 1
$$

This is memory-bound (AI ≈ 1), but with 4× fewer bytes loaded from HBM vs. dense. The **HBM bandwidth reduction** is the primary source of speedup, not compute reduction.

**Shared memory:** VNM sparse tiles fit comfortably in shared memory (256KB on H100). A tile of $128 \times 128$ at 75% sparsity requires $128 \times 32 \times 2 = 8$KB (FP16), well within SRAM capacity.

### Parallelism Analysis

**Tensor core mapping:** The inner 2:4 pattern maps directly to NVIDIA Sparse Tensor Core instructions (mma.sp). Column pruning reduces the $K$-dimension of the matmul, enabling standard dense Tensor Core execution on the retained columns. No warp divergence or load imbalance.

**No sequential bottleneck:** VNM pruning decisions are made once (at initialization for column masks, continuously for 2:4 masks during training). No iterative algorithms at inference time.

**SM saturation:** Projection matmuls ($d_{model} \times d_k$ with $d_{model} = 2048+$) produce enough tiles to saturate all SMs even after sparsification.

### Key Variables

- $W_Q, W_K, W_V, W_{gate}, W_O$ — projection weight matrices
- $V$ — VNM block height (typically 64)
- $M$ — VNM block width (5 for 60%, 8 for 75%)
- $N = 2$ — nonzeros per 4-element group (fixed)
- $\beta_{\{Q,K,V,g,O\}}$ — S-STE scaling factors (frozen after step 1)
- $\text{mask}^{col}$ — column selection mask (refreshed periodically during training)
- $a$ — activation importance exponent for RIA scoring

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Mamba-2 with VNM-sparse projections |
| Layers | $L = 12$ |
| Hidden dim | $d_{model} = 768$ |
| Head dim | $d_k = d_v = 64$ |
| Heads | $H = 12$ |
| State dim | $n = 16$ per head |
| Sparsity | V:2:8 (75%) for projections; dense for state transition |
| VNM params | $V = 64$, $M = 8$ |
| S-STE scaling | $\beta$ frozen after iteration 1 |
| Column mask refresh | Every 1000 steps (first 10% of training), then fixed |
| Parameters | ~125M total (~85M effective at inference) |

### Baseline

1. **Dense Mamba-2**: Full-rank projections, $O(d_{model}^2)$ — quality upper bound
2. **2:4 Sparse Mamba-2**: Standard 2:4 S-STE on projections — shows marginal speedup of 2:4 alone
3. **VNM Post-Training**: Train dense, then apply VNM pruning + LoRA fine-tune — shows gap from training-from-scratch
4. **Smaller Dense Mamba-2**: Reduce $d_{model}$ to match VNM parameter count — iso-parameter baseline

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.3\times$ dense Mamba-2 | Tokens/sec (A100, cuSPARSELt) |
| Inference throughput | $> 1.5\times$ dense Mamba-2 | Tokens/sec (A100, cuSPARSELt) |
| WikiText-103 PPL | Within 3% of dense | Language modeling, 125M params |
| MQAR accuracy | Within 5% of dense at 4 KV pairs | Associative recall |
| HBM bandwidth | $< 0.5\times$ dense | Measured via NCU profiling |
| Mask flip rate | Converging | $r_k = \|m_k \oplus m_{k-1}\|_1 / d$ |
| Tensor Core utilization | $> 50\%$ | NCU SM active cycles |

### Estimated Compute

**MVE**: ~10 minutes on single GPU (~$0.50)
**Small-scale**: 4 GPU-hours on A100 (~$16)
**Full-scale**: 30 GPU-hours on A100 (~$120)

## Expected Outcome

**If hypothesis is correct:**

- VNM-sparse (75%) Mamba-2 achieves $1.5\times$ inference throughput improvement over dense baseline (consistent with VNM paper's 1.7× on DeiT-base)
- Training speedup of $1.3\times$ (conservative: S-STE overhead during training partially offsets sparse acceleration)
- WikiText-103 perplexity within 3% of dense baseline (less degradation than iso-parameter dense model because VNM preserves salient weights)
- SwiGLU-gated layers degrade less than non-gated (GEGLU vs ReLU ablation shows $< 1\%$ gap for gated, $> 5\%$ for non-gated)
- HBM bandwidth reduced by 60–70% for projection matmuls (4× fewer weights loaded × amortization overhead)

**If hypothesis is wrong:**

- **Scenario A**: Training from scratch fails (quality $> 10\%$ worse than dense) — S-STE can't handle VNM's two-level pruning during training. **Learn**: VNM requires dense pre-training; the column-level structure is too aggressive for from-scratch optimization. **Follow-up**: Try staged approach (dense → 2:4 → VNM, gradual sparsification).
- **Scenario B**: Speedup is $< 1.2\times$ — cuSPARSELt overhead for VNM format conversion dominates. **Learn**: Need custom Triton kernel for VNM-aware matmul instead of relying on library. **Follow-up**: Write Triton kernel that fuses column gather + sparse matmul.
- **Scenario C**: SwiGLU gating doesn't compensate — gated and non-gated show same degradation. **Learn**: Gating compensation only works for unstructured sparsity, not structured (VNM removes entire columns, which gating can't route around). **Follow-up**: Use per-head sparsity allocation (some heads denser, some sparser).

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer Mamba-2 with VNM-sparse projections, $d_{model} = 128$, $H = 4$, $d_k = 32$, $n = 16$, ~200K params
- **Task**: Multi-Query Associative Recall (MQAR) — 4 KV pairs, sequence length 64, vocabulary 16
- **Data**: 10K synthetic sequences
- **Compute**: Single GPU, $< 10$ minutes
- **Sparsity levels tested**: Dense, 2:4 (50%), V:2:6 (67%), V:2:8 (75%)

### Implementation Sketch

```python
import torch
import torch.nn as nn

class VNMSparseLinear(nn.Module):
    """Linear layer with VNM sparse projection via S-STE."""

    def __init__(self, in_features, out_features, V=64, M=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.V = V
        self.M = M
        self.beta = None  # Frozen after first step

    def vnm_soft_threshold(self, W):
        """Two-level VNM pruning with S-STE."""
        out_f, in_f = W.shape
        # Reshape into V x M blocks along input dimension
        W_blocks = W.view(out_f, -1, self.M)  # (out, n_blocks, M)

        # Step 1: Column importance (L1 norm across V rows)
        # Simplified: use column magnitude for each M-group
        col_imp = W_blocks.abs().sum(dim=0)  # (n_blocks, M)
        _, top_idx = col_imp.topk(4, dim=-1)  # Keep top-4 columns per block
        col_mask = torch.zeros_like(col_imp).scatter_(-1, top_idx, 1.0)
        col_mask = col_mask.unsqueeze(0).expand_as(W_blocks)

        # Apply column mask
        W_masked = W_blocks * col_mask

        # Step 2: S-STE 2:4 within retained columns
        # Reshape retained columns into groups of 4
        # Apply soft-thresholding per group
        W_flat = W_masked.reshape(-1, 4)
        abs_vals = W_flat.abs()
        sorted_abs, _ = abs_vals.sort(dim=-1)
        threshold = sorted_abs[:, 1:2]  # 2nd smallest
        W_soft = torch.sign(W_flat) * torch.relu(abs_vals - threshold)
        W_soft = W_soft.view_as(W)

        # Compute frozen beta on first call
        if self.beta is None:
            self.beta = (W * W_soft).sum() / (W_soft * W_soft).sum()
            self.beta = self.beta.detach()

        return self.beta * W_soft

    def forward(self, x):
        W_sparse = self.vnm_soft_threshold(self.weight)
        return x @ W_sparse.T  # STE: gradient flows through to dense weight
```

### Success Criteria

- VNM-sparse (75%) Mamba-2 achieves $> 80\%$ accuracy on MQAR at 4 KV pairs
- Dense Mamba-2 achieves $> 95\%$ accuracy (sanity check)
- 2:4 sparse Mamba-2 achieves $> 90\%$ accuracy (shows VNM is harder but feasible)
- Iso-parameter dense (smaller $d_{model}$) achieves $< 70\%$ (shows sparse > small-dense)
- S-STE mask flip rate converges within 500 training steps

### Failure Criteria

- **Kill if**: VNM-sparse (75%) achieves $< 50\%$ on MQAR — the two-level pruning destroys too much information in projections
- **Kill if**: VNM-sparse performs worse than iso-parameter dense — structured sparsity is less efficient than simply reducing model size
- **Investigate if**: 2:4 and VNM perform identically — the column-level pruning adds no benefit over standard 2:4

### Why This Test Is Sufficient

- MQAR requires precise key-value association, which stresses projection quality. If VNM-sparse projections can accurately project queries and keys to enable retrieval, the mechanism works.
- Testing at $d_k = 32$ with 75% sparsity means only 8 effective weights per row — highly aggressive. Success here implies success at larger scales where the ratio is more favorable.
- The ablation structure (4 sparsity levels + iso-parameter) gives clean signal about where the quality-efficiency frontier lies.
- The gating mechanism (SwiGLU) can be ablated (remove gate → use plain ReLU) to isolate its compensation effect.

## Theoretical Analysis

### Complexity Comparison

| Operation | Dense | 2:4 Sparse | VNM (V:2:8, 75%) |
|-----------|-------|-----------|-------------------|
| Projection FLOPs | $2 d_{model} \cdot d_k$ | $d_{model} \cdot d_k$ | $0.5 \cdot d_{model} \cdot d_k$ |
| Projection params | $d_{model} \cdot d_k$ | $0.5 \cdot d_{model} \cdot d_k$ | $0.25 \cdot d_{model} \cdot d_k$ |
| HBM bytes (weight) | $2 d_{model} \cdot d_k$ | $d_{model} \cdot d_k + \text{meta}$ | $0.5 \cdot d_{model} \cdot d_k + \text{meta}$ |
| Tensor Core speedup | 1× | 1.1–1.3× | **1.5–1.7×** |
| Total layer speedup | 1× | 1.05–1.15× | **1.3–1.5×** |

### HBM Bandwidth Analysis

For a Mamba-2 layer with $d_{model} = 2048$, $H = 16$, $d_k = 128$, batch $B = 16$, seq $T = 2048$:

- **Dense projection** HBM load per forward: $5 \times 2048 \times 128 \times 16 \times 2 = 41.9$ MB (5 projections, FP16)
- **VNM 75% sparse**: $5 \times 2048 \times 128 \times 16 \times 2 \times 0.25 + \text{meta} \approx 12.5$ MB
- **HBM reduction**: $\sim 3.3\times$ fewer bytes loaded

On A100 (2TB/s HBM bandwidth), this translates from 20.9 µs to 6.3 µs for weight loading — a significant fraction of total inference time at small batch sizes.

### Hardware-Specific Considerations

**A100 (Ampere):**
- Sparse Tensor Cores support 2:4 natively via mma.sp instructions
- cuSPARSELt provides VNM-compatible sparse GEMM (column selection is a dimension reduction)
- 40 MB L2 cache can hold several VNM-sparse projection matrices

**H100 (Hopper):**
- TMA (Tensor Memory Accelerator) can async-load sparse tiles
- 256 KB shared memory per SM — comfortably holds VNM sparse tiles
- WGMMA instructions work with sparse formats via cuSPARSELt
- FP8 + VNM sparsity could yield $3\times$ combined speedup (multiplicative)

## Risks & Limitations

1. **cuSPARSELt maturity**: VNM-specific acceleration may not be fully optimized in current cuSPARSELt. May need custom Triton kernel for the column-gather + 2:4 sparse matmul fusion.

2. **Training overhead**: Maintaining dense copies of all projection matrices during training doubles memory usage. Gradient computation flows through dense weights. Sparse acceleration only benefits forward pass during training (backward remains dense).

3. **Column mask stability**: Unlike 2:4 masks which change per-element, VNM column masks affect entire columns. If column importance shifts during training, mask refreshes cause larger discontinuities. Mitigation: use exponential moving average of column importance, refresh only in early training.

4. **Interaction with GQA/MQA**: Grouped-query attention shares $K, V$ projections across heads. VNM sparsity applied to shared projections affects all heads equally — no per-head adaptation. May need per-group sparsity budgets.

5. **Backward pass**: 2:4 sparsity of $W$ enables sparse forward ($y = Wx$) but the backward ($\nabla_x = W^T \nabla_y$) requires the transpose to also be 2:4. VNM column pruning naturally satisfies this (transposing a column-pruned matrix gives a row-pruned matrix), but the inner 2:4 pattern may not be transposable. Consider using transposable 2:4 masks (TSENOR trick) for the inner pattern.

## Follow-up Experiments

1. **FP8 + VNM**: Combine VNM sparsity with FP8 quantization for multiplicative speedup ($1.7\times \cdot 2\times = 3.4\times$ theoretical).
2. **Per-head sparsity allocation**: Learn different VNM configurations per attention head — some heads at 50% (2:4), others at 75% (V:2:8), based on head importance.
3. **Extend to FFN layers**: Apply same VNM-S-STE recipe to the SwiGLU FFN block (which has 3 projections — even more parameter-heavy).
4. **Combine with Proposal 009**: VNM-sparse projections + post-readout sigmoid gating — gating compensates for sparse readout while projections are hardware-accelerated.
5. **Progressive sparsification**: Start training at 2:4, gradually increase to V:2:6, then V:2:8 during training, following a cosine schedule.
6. **Triton kernel**: Write a fused Triton kernel that performs column-gather + 2:4 sparse matmul in one kernel launch, avoiding the column selection overhead of generic cuSPARSELt.

## Human Review

(To be filled by reviewer)

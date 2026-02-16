---
status: ongoing
priority: high
created: 2026-02-15
based_on: dijiang-dct-frequency-kernelization (168), chunkwise-parallel-scan (026), tfla-two-level-tiled-chunkwise-parallelism (158), input-dependent-gating (065), cosine-reweighted-linear-attention (031), structured-orthogonal-random-features-sorf (155), rfa-gated-random-feature-attention (156)
experiment_number: 045
experiment_log: experiment-log-045.md
---

# DCT Frequency-Domain Kernel Feature Maps for Chunkwise GLA

## Hypothesis

Replacing GLA's identity feature map $\phi(x) = x$ (linear kernel) with DiJiang's **Weighted DCT Feature map** (WDCF) will approximate the softmax kernel within GLA's chunkwise parallel framework, achieving **3–8% perplexity improvement** over standard GLA at **$< 3\%$ throughput overhead**, because:
1. The WDCF projection is deterministic $O(d \log d)$ per token via fast DCT — cheaper than SORF's three FWHT passes and without the random permutation that breaks coalescing in Fastfood
2. The DCT's energy compaction property means $m = d$ features suffice for high-quality kernel approximation ($O(1/m)$ convergence vs. FAVOR+'s $O(1/\sqrt{m})$)
3. The dominant cost remains the two tensor-core GEMMs for linear attention ($\phi(Q) \cdot [\phi(K)^\top V]$), which are unchanged

## Background

### The feature map quality gap in GLA

Gated Linear Attention (Yang et al., 2024) uses the chunkwise parallel formulation:

$$
O_j = \underbrace{(\phi(Q_j) \phi(K_j)^\top \odot M_j) V_j}_{\text{intra-chunk: } O(C^2 d)} + \underbrace{\phi(Q_j) S_{j-1}}_{\text{inter-chunk: } O(Cd^2)}
$$

$$
S_j = G_j S_{j-1} + \phi(K_j)^\top V_j
$$

where $G_j$ is the diagonal decay gate and $M_j$ is the causal gate mask. Standard GLA uses $\phi(x) = x$ (identity), meaning the intra-chunk attention computes $Q_j K_j^\top$ — a **linear kernel**. This cannot produce the sharp, sparse attention patterns that softmax generates, causing a 2–5% perplexity gap at matched scale.

**ReGLA** (Lu et al., NAACL 2025) explored deterministic feature maps (elu+1, ReLU) for GLA but did not test frequency-domain kernelized feature maps. **Proposal 037** tests SORF+SADERF random features for GLA, and **Proposal 029** tests circulant FAVOR+ — both are stochastic approximations requiring random projection matrices. Neither uses the deterministic DCT approach.

### Why DCT feature maps are the right choice for GLA

1. **Deterministic**: Unlike SORF (3 random sign diagonals) or FAVOR+ (random Gaussian matrix), the DCT matrix $\mathcal{C}$ is fixed and data-independent. This eliminates variance from random features and enables reproducible training.

2. **$O(1/m)$ convergence**: DiJiang's WQMC-based approach achieves $O(1/m)$ kernel approximation error (Theorem 3.2 of Chen et al., 2024), compared to $O(1/\sqrt{m})$ for FAVOR+/SORF. At $m = d = 64$ (typical head dimension), this is a $\sim 8\times$ error reduction.

3. **No power-of-2 constraint**: Unlike SORF and Fastfood (which require Walsh-Hadamard transforms with $d = 2^k$), the DCT works for any dimension $d$, avoiding the padding overhead that SORF incurs for non-power-of-2 head dimensions.

4. **GPU-friendly**: The feature map is: (a) fast DCT $O(d \log d)$ via cuFFT, (b) diagonal scaling (elementwise), (c) elementwise exp. All operations are memory-coalesced and can be fused into a single kernel. The dominant operations remain the two tensor-core GEMMs for linear attention.

5. **Learnable parameters**: The weight vector $D \in \mathbb{R}^d$ adapts per layer, providing data-dependent quality tuning with only $O(d)$ extra parameters — negligible.

6. **Chunkwise compatibility**: The feature map $\phi_{\text{WDCF}}$ is applied token-wise and doesn't interact with the chunkwise structure. It can be dropped into any chunkwise GLA/TFLA kernel by replacing the identity feature map in the $Q_j, K_j$ preprocessing.

### What's different from existing proposals

- **Proposal 037** (SADERF-SORF GLA): Uses stochastic SORF projection + SADERF calibration. Requires random sign matrices and three FWHT passes per token. Our approach is deterministic (single fast DCT) with better convergence rate.
- **Proposal 029** (Circulant FAVOR+): Uses circulant FFT-based projection (complex arithmetic). Our DCT is real-valued, avoiding complex number overhead.
- **Proposal 036** (Near-Far Field GLA): Decomposes attention into banded + low-rank. Uses simple elu+1 for far-field features. Orthogonal to our approach — could use our DCT features for the far-field component.
- **Proposal 008** (Cosine Log-Linear): Adds cosine reweighting to log-linear attention. Different mechanism (position-based decay vs. kernel approximation).

## Related Work

- **DiJiang** (Chen et al., ICML 2024 Oral): Proposed the WDCF feature map for converting softmax Transformers to linear attention. Demonstrated on Pythia (70M–2.8B) and LLaMA2-7B, achieving comparable quality with 10x training speedup. However, DiJiang applies WDCF to **standard (non-gated) linear attention** without chunkwise parallelism. It does not address gated linear attention (GLA), input-dependent decay gates, or chunkwise-parallel training.
- **ReGLA** (Lu et al., NAACL 2025): Explored feature maps for GLA (elu+1, ReLU, identity+elu, safe exp). Did NOT test DCT or frequency-domain feature maps. Found that feature map choice significantly impacts quality.
- **FAVOR#/SADERF** (Likhosherstov et al., 2023): Variance-reduced random features. Stochastic approach — fundamentally different from our deterministic DCT.
- **Spectraformer** (Chen et al., 2024): Unified random feature framework. Tested OPRF-FastFoodL and SADERF-ORF but NOT DCT-based features, and NOT within chunkwise GLA.
- **DCT-Former** (Scribano et al., 2023): Uses DCT for attention approximation in vision/NLP but via spectral truncation (keep top-$k$ DCT coefficients of the attention matrix), NOT via kernel feature maps. Different mechanism entirely.

**Gap**: No existing work applies DiJiang's DCT frequency-domain kernel feature map within a gated linear attention framework with chunkwise-parallel training (GLA/TFLA).

## Mathematical Formulation

### Standard GLA Intra-Chunk Computation

$$
O_j = (Q_j K_j^\top \odot M_j) V_j + \bar{Q}_j S_{j-1}
$$

where $Q_j, K_j \in \mathbb{R}^{C \times d_k}$, $V_j \in \mathbb{R}^{C \times d_v}$, $M_j \in \mathbb{R}^{C \times C}$ is the causal gate mask.

### Proposed: DCT-GLA Intra-Chunk Computation

$$
O_j = (\phi(Q_j) \phi(K_j)^\top \odot M_j) V_j + \phi(\bar{Q}_j) S_{j-1}
$$

$$
S_j = G_j S_{j-1} + \phi(K_j)^\top V_j
$$

where $\phi = \phi_{\text{WDCF}}$ is the Weighted DCT Feature map:

$$
\phi_{\text{WDCF}}(x) = D \odot \exp(T \cdot \text{DCT}(x))
$$

**Key Variables:**
- $x \in \mathbb{R}^{d_k}$ — input query or key vector (per head)
- $\text{DCT}(x) = \mathcal{C} x \in \mathbb{R}^{d_k}$ — Type-II DCT of $x$, computed via fast DCT in $O(d_k \log d_k)$
- $T = \text{diag}(t_1, \ldots, t_{d_k})$ — diagonal scaling from inverse CDF sampling (fixed per head, initialized once)
- $D \in \mathbb{R}^{d_k}$ — learnable weight vector (per layer, per head)
- $\mathcal{C} \in \mathbb{R}^{d_k \times d_k}$ — DCT-II coefficient matrix: $\mathcal{C}_{kn} = s_k \cos\left(\frac{\pi(2n+1)k}{2d_k}\right)$

### Feature Map Positivity

The exponential ensures $\phi_{\text{WDCF}}(x) > 0$ elementwise (assuming $D > 0$, enforced via softplus). This is critical for:
1. Non-negative kernel: $\langle \phi(q), \phi(k) \rangle > 0$ always, avoiding the negative-attention instabilities of non-positive feature maps
2. Compatibility with GLA's gating: the gate mask $M_j$ applies elementwise to non-negative attention scores

### Kernel Approximation Quality

The WDCF feature map approximates the Gaussian kernel:

$$
\langle \phi_{\text{WDCF}}(q), \phi_{\text{WDCF}}(k) \rangle \approx C \cdot e^{-\|q - k\|^2 / 2}
$$

which itself approximates the softmax kernel $e^{q^\top k}$ (after appropriate normalization). The approximation error is:

$$
\left| \hat{K}(q, k) - K(q, k) \right| = O(1/d_k)
$$

For $d_k = 64$: error $\sim 1.5\%$. For $d_k = 128$: error $\sim 0.8\%$.

### Interaction with TFLA Tiling

In TFLA's two-level tiling (trick 158), the intra-chunk computation is tiled along the sequence dimension:

$$
O_j[i \cdot B_{Lhq} : (i+1) \cdot B_{Lhq}, :] = \sum_l S_{il}^{(j)} V_j^{(l)} + \bar{Q}_j^{(i)} S_{j-1}
$$

With our DCT feature map, $S_{il}^{(j)} = \phi(Q_j^{(i)}) \phi(K_j^{(l)})^\top \odot M_{il}^{(j)}$. The feature map is applied **before** the tiled matmuls, so the tiling structure is unchanged. The feature map cost is $O(B_{Lhq} \cdot d_k \log d_k)$ per tile — negligible compared to the $O(B_{Lhq} \cdot B_{Lkv} \cdot d_k)$ matmul cost.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA (Gated Linear Attention) with DCT feature maps |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d_{\text{model}} = 768$–$2048$ |
| Head dim | $d_k = d_v = 64$ or $128$ |
| Heads | $h = 12$–$16$ |
| Chunk size | $C = 64$–$256$ (TFLA tiling for $C > 64$) |
| Feature map dim | $m = d_k$ (same as head dim) |

### Baseline

1. **Standard GLA** ($\phi = \text{id}$): Identity feature map, $O(TC^2 d)$ intra-chunk, current default
2. **ReGLA** ($\phi = \text{elu+1}$): Best deterministic feature map from Lu et al.
3. **GLA + SORF features** (Proposal 037): Stochastic SORF projection (if implemented)

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\geq 3\%$ improvement vs. GLA-identity | WikiText-103, SlimPajama validation |
| Throughput | $> 97\%$ of GLA-identity throughput | Tokens/sec on A100, $T = 2048$–$8192$ |
| Memory | $\leq 1.01\times$ GLA-identity | Peak GPU memory |
| Feature map overhead | $< 3\%$ of total forward time | Profiled via nsight |

### Estimated Compute

**Small**: 4–8 A100 GPU-hours for MVE + ablations at 125M–350M scale.
**Medium**: 32–64 A100 GPU-hours for full 350M–1.3B comparison.

## Expected Outcome

**If hypothesis is correct:**
- $\geq 3\%$ perplexity improvement over GLA-identity at 350M scale on SlimPajama
- Throughput within 97% of baseline (DCT feature map adds $< 3\%$ overhead since it's $O(d \log d)$ vs. the $O(C d^2)$ matmul cost)
- Matches or exceeds ReGLA's best feature map (elu+1) quality

**If hypothesis is wrong:**
- We learn that the softmax kernel approximation doesn't help within GLA's gated framework — the decay gates already compensate for the linear kernel's weakness
- This would suggest that GLA's quality gap vs. softmax attention is NOT primarily due to the feature map, but due to the finite state capacity ($d \times d$ matrix)
- Negative result still valuable: rules out kernel approximation as the improvement axis for GLA

## Minimum Viable Experiment

### Setup
- **Model**: Tiny GLA model (2 layers, $d = 64$, $d_k = d_v = 32$, 4 heads, ~500K params)
- **Task**: Synthetic associative recall — given key-value pairs followed by a query, retrieve the correct value. This directly tests whether the DCT feature map enables sharper attention patterns than the linear kernel.
- **Data**: 10K synthetic sequences, length 128, vocabulary size 256, 16 key-value pairs per sequence
- **Compute**: Single GPU, $< 5$ minutes training

### Success Criteria
- DCT feature map achieves $> 85\%$ associative recall accuracy where identity-GLA achieves $< 65\%$
- The gap between DCT-GLA and softmax attention on this task is $< 10\%$ (softmax should be $> 95\%$)

### Failure Criteria
- DCT feature map performs $\leq 5\%$ better than identity on associative recall → the kernel approximation doesn't help within GLA's gated framework
- DCT feature map causes training instability (loss divergence) → numerical issues with the exponential in WDCF

### Why This Test Is Sufficient
- Associative recall is the canonical task where softmax attention dominates linear attention (Sun et al., 2023)
- If DCT features close the gap on this task, they're approximating softmax well enough for the feature map to provide value
- GLA's decay gates should still help with non-associative-recall tasks, so success here implies broader improvement

## Theoretical Analysis

Complexity comparison (per layer, sequence length $T$, chunk size $C$):

| Operation | GLA-identity | GLA-DCT (proposed) |
|-----------|-------------|---------------------|
| Feature map (per token) | $O(1)$ | $O(d_k \log d_k)$ |
| Feature map (total) | $0$ | $O(T d_k \log d_k)$ |
| Intra-chunk attention | $O(\frac{T}{C} C^2 d_k)$ | $O(\frac{T}{C} C^2 d_k)$ |
| Inter-chunk state update | $O(\frac{T}{C} d_k d_v)$ | $O(\frac{T}{C} d_k d_v)$ |
| **Total** | $O(T C d_k + T d_k^2)$ | $O(T C d_k + T d_k^2 + T d_k \log d_k)$ |

The feature map overhead is $O(T d_k \log d_k)$, which is dominated by the $O(T C d_k)$ intra-chunk cost for $C \geq \log d_k$ (always true since $C \geq 16$, $\log d_k \leq 7$).

**Arithmetic intensity analysis:**
- Feature map computation: $d_k \log d_k$ FLOPs reading $d_k$ elements → AI = $\log d_k \approx 6$. Memory-bound on GPU, but the data is already in registers from the preceding Q/K projection GEMM. Can be fused into the projection epilogue via EVT (trick 039).
- The linear attention GEMMs ($\phi(Q) \cdot [\phi(K)^\top V]$) remain tensor-core-friendly: shapes $(n, d_k) \times (d_k, d_v)$ and $(d_k, n) \times (n, d_v)$ map directly to WGMMA.

**Memory access pattern:**
- Fast DCT is a structured transform with sequential butterfly-like access patterns — cache-friendly
- Diagonal scaling and exp are elementwise — fully coalesced
- No gather/scatter operations (unlike Fastfood's random permutation or SORF's sign diagonals)

## Risks & Limitations

1. **The exponential in WDCF may cause numerical overflow**: For large $\|x\|$, $\exp(T \cdot \text{DCT}(x))$ can overflow float16/bfloat16. Mitigation: use the numerically stable form $\phi(x) = D \odot \exp(T \cdot \text{DCT}(x) - \|x\|^2/2)$ (the Gaussian envelope absorbs the growth).

2. **GLA's decay gates may already solve the quality gap**: If the input-dependent gates $G_t$ provide sufficient content-based selection, the feature map may not matter. This would mean the identity feature map is "good enough" within GLA's framework.

3. **cuFFT DCT overhead**: While fast DCT is $O(d \log d)$, the actual cuFFT kernel launch overhead may dominate for small $d$ (e.g., $d = 64$). For such small $d$, we should implement the DCT as a dense $64 \times 64$ matmul on tensor cores instead.

4. **Interaction with TFLA**: The feature map must be applied before TFLA's tiling loop. If this prevents fusion of the feature map into the TFLA kernel, there's an extra HBM round-trip for the transformed Q/K. Mitigation: apply feature map as part of the Q/K projection (fuse DCT + diag + exp into the projection GEMM epilogue).

5. **Learnable $D$ and $T$ add hyperparameters**: The per-layer, per-head $D$ and $T$ vectors add $O(L \cdot h \cdot d_k)$ parameters. For $L = 24, h = 16, d_k = 64$: 24,576 extra parameters — negligible.

## Follow-up Experiments

1. **Ablate feature map components**: Test DCT-only (no learnable $D$), DCT + $D$ (no $T$ scaling), and full WDCF. This isolates the contribution of each component.

2. **Compare with SORF (Proposal 037)**: Head-to-head comparison of DCT vs. SORF feature maps within GLA's chunkwise framework. Test variance, quality, and throughput.

3. **Scale to 1.3B**: If MVE succeeds, train 1.3B-parameter DCT-GLA on SlimPajama and compare with GLA, Mamba-2, and softmax Transformer baselines.

4. **Combine with post-sigmoid gating (Proposal 009)**: The DCT feature map (kernel approximation quality) and sigmoid gating (readout nonlinearity) address orthogonal bottlenecks. Test their composition.

5. **Vary chunk size with TFLA**: With better feature maps, intra-chunk quality improves. Test whether DCT features enable larger optimal chunk sizes ($C = 256$–$512$) by reducing the quality loss from linear-kernel intra-chunk attention.

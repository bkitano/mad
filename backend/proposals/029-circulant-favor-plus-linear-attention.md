---
status: completed
priority: high
created: 2026-02-15
based_on: favor-plus-positive-orthogonal-random-features, circulant-binary-embedding, cosine-reweighted-linear-attention, post-attention-sigmoid-gating, io-aware-tiling
experiment_number: 029
results_file: 029_results.md
---

# Circulant FAVOR+ for Linear Attention and SSM Readout

## Hypothesis

Replacing the dense random projection matrix in FAVOR+ with a learnable circulant projection (from the CBE trick) will reduce the feature-map computation from $O(md)$ to $O(d \log d)$ while preserving softmax kernel approximation quality, and combining this with cosine reweighting and post-attention sigmoid gating will yield a linear attention mechanism that matches softmax quality at $O(T d \log d)$ total cost.

## Background

FAVOR+ (Choromanski et al., 2021) approximates softmax attention via positive random features:

$$
\phi^+(\mathbf{x}) = \frac{\exp(-\|\mathbf{x}\|^2/2)}{\sqrt{m}} \left(\exp(\omega_1^\top \mathbf{x}), \ldots, \exp(\omega_m^\top \mathbf{x})\right)
$$

where $\omega_i \in \mathbb{R}^d$ are random projection vectors. The feature map $\phi^+$ enables linear attention: $\hat{\mathbf{D}}^{-1}(\mathbf{Q}'(\mathbf{K}')^\top \mathbf{V})$ in $O(Tmd)$ time. The bottleneck is computing the projections $\omega_i^\top \mathbf{x}$ for all $m$ features and all $T$ tokens, which requires a dense $m \times d$ matrix multiply costing $O(md)$ per token.

The Circulant Binary Embedding (CBE) trick (Yu et al., 2014) shows that circulant random projections preserve angular distances as well as dense random projections. A circulant matrix $\text{circ}(\mathbf{r})$ applied to a vector costs $O(d \log d)$ via FFT, with only $O(d)$ storage. Crucially, the CBE paper proves that despite the row dependence in circulant matrices, the variance of angle preservation is empirically identical to independent projections.

**The gap**: No one has combined these two tricks — using circulant structure to accelerate the random feature map in FAVOR+. This is a natural pairing because:

1. FAVOR+ needs $m = O(d)$ random projections, so the dense projection costs $O(d^2)$ — exactly the regime where circulant gives $O(d \log d)$ speedup
2. The circulant preserves angular distances, which is precisely what the softmax kernel $\exp(\mathbf{q}^\top \mathbf{k})$ depends on
3. The circulant matrix can be made **learnable** (CBE-opt uses time-frequency alternating optimization), enabling the projection to adapt to the data distribution

Furthermore, two additional quality improvements can be composed orthogonally:

- **Cosine reweighting** (cosFormer): Adds a locality bias $\cos(\frac{\pi}{2} \cdot \frac{i-j}{M})$ via Ptolemy decomposition, improving causal language modeling quality without extra asymptotic cost
- **Post-attention sigmoid gating** (NeurIPS 2025 Best Paper): Breaks the low-rank bottleneck of $W_V W_O$ readout, which is especially severe in linear attention where the state is already rank-constrained

**Why this matters for SSMs**: Linear attention is mathematically equivalent to a matrix-valued SSM recurrence $S_t = \lambda S_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$. The random feature map determines the quality of the KV state, and an efficient feature map directly improves the SSM readout. Models like GLA and RWKV-v6 use input-dependent linear attention variants where the feature map quality is critical.

**What's different from existing proposals**:
- Proposal 008 (cos-LogLinear) combines cosine + log-linear but doesn't address feature map efficiency
- Proposal 009 (post-sigmoid gating) adds gating to linear attention/SSM readout but uses standard feature maps
- This proposal targets the *feature map computation itself* — an orthogonal axis of improvement

## Mathematical Formulation

**Standard FAVOR+ Feature Map:**

$$
\phi^+(\mathbf{x}) = \frac{\exp(-\|\mathbf{x}\|^2/2)}{\sqrt{m}} \left(\exp(\omega_1^\top \mathbf{x}), \ldots, \exp(\omega_m^\top \mathbf{x})\right), \quad \omega_i \sim \mathcal{N}(0, \mathbf{I}_d)
$$

Cost: $O(md)$ per token for the projection $\Omega \mathbf{x}$ where $\Omega \in \mathbb{R}^{m \times d}$.

**Circulant FAVOR+ (C-FAVOR+) Feature Map:**

Replace the dense projection matrix $\Omega$ with a product of circulant and diagonal sign-flip matrices:

$$
\tilde{\Omega} = \text{circ}(\mathbf{r}) \cdot \text{diag}(\mathbf{s})
$$

where $\mathbf{r} \in \mathbb{R}^d$ defines the circulant and $\mathbf{s} \in \{-1, +1\}^d$ is a fixed random sign vector. The projection becomes:

$$
\tilde{\Omega} \mathbf{x} = \mathcal{F}^{-1}\left(\mathcal{F}(\mathbf{r}) \circ \mathcal{F}(\mathbf{s} \odot \mathbf{x})\right)
$$

Cost: $O(d \log d)$ via two FFTs.

For $m = d$ random features (which FAVOR+ typically needs for good approximation), we use the full $d$-dimensional circulant output. For $m < d$, take the first $m$ elements. For $m > d$, stack multiple circulant projections with independent $\mathbf{r}_i, \mathbf{s}_i$:

$$
\tilde{\Omega} = \begin{bmatrix} \text{circ}(\mathbf{r}_1) \cdot \text{diag}(\mathbf{s}_1) \\ \vdots \\ \text{circ}(\mathbf{r}_{\lceil m/d \rceil}) \cdot \text{diag}(\mathbf{s}_{\lceil m/d \rceil}) \end{bmatrix}
$$

Cost: $O(\lceil m/d \rceil \cdot d \log d) = O(m \log d)$.

**The C-FAVOR+ feature map is then:**

$$
\phi_{\text{C-FAVOR}}^+(\mathbf{x}) = \frac{\exp(-\|\mathbf{x}\|^2/2)}{\sqrt{m}} \left(\exp([\tilde{\Omega}\mathbf{x}]_1), \ldots, \exp([\tilde{\Omega}\mathbf{x}]_m)\right)
$$

**Learnable variant (LC-FAVOR+):** Make $\mathbf{r}$ a learnable parameter (following CBE-opt). Training updates $\mathbf{r}$ via backpropagation through the FFT. The sign vector $\mathbf{s}$ remains fixed (critical for breaking circulant degeneracy, per CBE theory). The gradient through the FFT is simply the adjoint FFT:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{r}} = \mathcal{F}^{-1}\left(\overline{\mathcal{F}(\mathbf{s} \odot \mathbf{x})} \circ \frac{\partial \mathcal{L}}{\partial (\tilde{\Omega} \mathbf{x})_{\text{FFT}}}\right)
$$

**Combined with Cosine Reweighting:**

Apply cosFormer's Ptolemy decomposition to C-FAVOR+ features:

$$
\hat{\text{Att}}(Q, K, V)_i = \frac{\sum_j \phi_{\text{C-FAVOR}}^+(q_i)^\top \phi_{\text{C-FAVOR}}^+(k_j) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i-j}{M}\right) \cdot v_j}{\sum_j \phi_{\text{C-FAVOR}}^+(q_i)^\top \phi_{\text{C-FAVOR}}^+(k_j) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i-j}{M}\right)}
$$

Via Ptolemy's identity $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$, this decomposes into two independent linear attention computations:

$$
\hat{\text{Att}} = \frac{Q^{\cos}(K^{\cos\top}V) + Q^{\sin}(K^{\sin\top}V)}{Q^{\cos}(K^{\cos\top}\mathbf{1}) + Q^{\sin}(K^{\sin\top}\mathbf{1})}
$$

where $Q^{\cos}_i = \phi_{\text{C-FAVOR}}^+(q_i) \cdot \cos(\frac{\pi i}{2M})$ and similarly for sin terms.

**Combined with Post-Attention Sigmoid Gating:**

$$
Y' = Y \odot \sigma(X W_\theta)
$$

Applied after the linear attention output $Y$ but before the output projection, breaking the low-rank bottleneck.

**Full Pipeline:**

$$
\text{C-CosGate-Attn}(Q, K, V, X) = \sigma(X W_\theta) \odot \left[\hat{D}^{-1}\left(Q'^{\cos}(K'^{\cos\top}V) + Q'^{\sin}(K'^{\sin\top}V)\right)\right]
$$

where $Q' = \phi_{\text{C-FAVOR}}^+(Q)$, $K' = \phi_{\text{C-FAVOR}}^+(K)$.

**Key Variables:**

- $\mathbf{x} \in \mathbb{R}^d$ — query/key vector
- $\mathbf{r} \in \mathbb{R}^d$ — circulant defining vector (learnable)
- $\mathbf{s} \in \{-1, +1\}^d$ — fixed random sign vector
- $m$ — number of random features ($m = d$ by default)
- $T$ — sequence length
- $d$ — head dimension

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Transformer with C-CosGate-Attention |
| Layers | $L = 6$ |
| Hidden dim | $d_{\text{model}} = 256$ |
| Heads | $H = 4$, $d_k = 64$ per head |
| Random features | $m = d_k = 64$ (one circulant per head) |
| Cosine reweighting | Yes (Ptolemy decomposition) |
| Sigmoid gating | Yes (per-head, $< 2\%$ param overhead) |
| FFN | SwiGLU, $d_{ff} = 682$ (isoparametric with $d_{ff} = 1024$ ReLU) |
| Parameters | ~8M |

### Baseline

1. **Standard softmax attention**: $O(T^2 d)$ — quality ceiling
2. **Vanilla FAVOR+** (dense random features): $O(Tmd)$ — same quality, slower feature map
3. **cosFormer** (ReLU + cosine reweighting): $O(Td^2)$ — no random features, different kernel
4. **GLA/RWKV-style linear attention**: $O(Td^2)$ — input-dependent decay, no random features
5. **Mamba-2** (SSD): $O(Tnd)$ — SSM approach with diagonal state

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Feature map throughput | $> 2\times$ over dense FAVOR+ | Tokens/sec for feature computation only |
| End-to-end throughput | $> 1.3\times$ over dense FAVOR+ | Tokens/sec full model |
| WikiText-103 PPL | $\leq$ dense FAVOR+ PPL | Perplexity at convergence |
| LRA average | $> 55\%$ (cosFormer-level) | Long Range Arena benchmark |
| Quality vs softmax | $< 10\%$ PPL gap | Relative perplexity degradation |

### Estimated Compute

**Small.** Feature map benchmarking: < 1 GPU-hour. LRA suite: ~10 GPU-hours. WikiText-103 8M model: ~20 GPU-hours. Total: < 32 GPU-hours on single A100.

## Expected Outcome

**If hypothesis is correct:**

- C-FAVOR+ feature map is $2$–$4\times$ faster than dense FAVOR+ at $d = 64$–$128$ due to FFT vs dense matmul
- The angular preservation property of circulant projections (proven in CBE) translates to equivalent kernel approximation quality: $< 5\%$ PPL difference between C-FAVOR+ and dense FAVOR+
- Cosine reweighting adds $0.5$–$2.0$ PPL improvement over plain FAVOR+ (matching cosFormer's gains)
- Sigmoid gating adds $0.5$–$1.5$ PPL improvement (matching Csordas et al.'s results)
- The combined C-CosGate model approaches softmax quality (within $5\%$ PPL gap) at $O(T d \log d)$ cost
- Learnable circulant (LC-FAVOR+) outperforms random circulant by adapting the projection to data distribution

**If hypothesis is wrong:**

- **Scenario A**: Circulant projection doesn't preserve softmax kernel quality → the row dependence matters more than CBE theory predicts for positive (exponential) features. We learn the boundary of CBE's applicability.
- **Scenario B**: FFT overhead at $d = 64$ negates speedup → the crossover point is at larger $d$; the trick is only useful for large-head attention ($d > 128$)
- **Scenario C**: Learnable circulant overfits to training distribution → random circulant is preferable, but periodic re-sampling (as in original FAVOR+) is needed

## Minimum Viable Experiment

### Setup

- **Model**: Single-layer, single-head linear attention, $d = 32$, ~10K params
- **Task**: Synthetic associative recall — given $(k_1, v_1, \ldots, k_n, v_n, k_q)$, output $v_q$
- **Data**: 5K sequences of length 64, vocabulary size 16, 8 key-value pairs per sequence
- **Compute**: Single GPU, < 5 minutes

### Comparisons

1. Dense FAVOR+ (32 random features, dense $32 \times 32$ projection)
2. C-FAVOR+ (circulant $32 \times 32$ projection via FFT)
3. ReLU linear attention (no feature map)
4. Softmax attention (quality ceiling)

### Success Criteria

- C-FAVOR+ achieves $> 90\%$ associative recall accuracy (matching dense FAVOR+)
- C-FAVOR+ feature map computation is measurably faster than dense FAVOR+ (at $d = 32$, the speedup may be marginal — the *existence* of comparable quality is the key signal)
- Both FAVOR+ variants significantly outperform ReLU linear attention ($> 20\%$ accuracy gap)

### Failure Criteria

- C-FAVOR+ accuracy is $> 10\%$ worse than dense FAVOR+ → circulant projection fundamentally breaks the positive feature approximation
- C-FAVOR+ accuracy is not better than ReLU linear attention → the feature map adds no value with circulant projection

### Why This Test Is Sufficient

- Associative recall is the canonical task for testing attention kernel quality — it requires precise key-value binding that depends on the kernel's ability to distinguish similar keys
- If the circulant projection preserves this discrimination ability, it will preserve it for general sequence modeling tasks
- The test isolates the feature map quality from other architectural choices (no gating, no cosine reweighting) — those are additive improvements that can be validated separately

## Theoretical Analysis

**Complexity Comparison:**

| Operation | Softmax Attn | Dense FAVOR+ | C-FAVOR+ | cosFormer |
|-----------|-------------|--------------|----------|-----------|
| Feature map | — | $O(md)$ | $O(d \log d)$ | $O(d)$ (ReLU) |
| Attention | $O(T^2 d)$ | $O(Tmd)$ | $O(Td \log d + Tmd)$ | $O(Td^2)$ |
| Total (per layer) | $O(T^2 d)$ | $O(Tmd)$ | $O(Td \log d)$ when $m = d$ | $O(Td^2)$ |
| Parameters for feature map | $0$ | $O(md)$ | $O(d)$ | $0$ |

**When $m = d$ (standard setting):**

| | Softmax | Dense FAVOR+ | C-FAVOR+ |
|---|---|---|---|
| Feature map | — | $O(d^2)$ per token | $O(d \log d)$ per token |
| Total | $O(T^2 d)$ | $O(Td^2)$ | $O(Td \log d)$ |
| Speedup over softmax | $1\times$ | $T/d$ | $T \cdot d / \log d$ |

**Crossover point**: C-FAVOR+ beats dense FAVOR+ when $d \log d < d^2$, i.e., $\log d < d$, which is always true. In practice, the FFT constant factor means the actual crossover is around $d \geq 32$–$64$.

**Approximation Quality:**

The CBE angle preservation guarantee states that for circulant projection with Gaussian $\mathbf{r}$:

$$
\mathbb{E}[\mathcal{H}_d(\mathbf{x}_1, \mathbf{x}_2)] = \frac{\theta(\mathbf{x}_1, \mathbf{x}_2)}{\pi}
$$

with variance matching independent projections. Since the FAVOR+ approximation quality depends on the angular preservation of the random projection (the softmax kernel $\exp(\mathbf{q}^\top \mathbf{k})$ depends only on $\|\mathbf{q}\|, \|\mathbf{k}\|$, and the angle between them), the circulant projection should provide equivalent kernel approximation.

**However**, there is a subtlety: FAVOR+ uses positive features $\exp(\omega_i^\top \mathbf{x})$, which are sensitive to the *signed projections* $\omega_i^\top \mathbf{x}$, not just their angles. The exponential amplifies projection magnitudes, so the row dependence of the circulant matrix might matter more than for binary (sign-based) embeddings. This is the key theoretical uncertainty that the MVE must resolve.

## Risks & Limitations

1. **Positive feature sensitivity**: Unlike binary hashing (where only the sign matters), FAVOR+ uses $\exp(\omega_i^\top \mathbf{x})$ which is sensitive to projection magnitudes. Circulant row dependence might cause correlated features that reduce approximation quality.

2. **FFT constant factor**: At typical head dimensions ($d = 64$–$128$), the cuFFT overhead may dominate. The technique shines at larger $d$ or when batch-parallelized.

3. **Gradient through FFT**: Learnable circulant requires backpropagation through FFT. While the adjoint FFT is well-defined, the optimization landscape in Fourier domain may have different convergence properties than direct weight learning.

4. **Interaction with ORF**: FAVOR+ uses orthogonal random features for variance reduction. Circulant matrices have dependent rows, so the ORF variance reduction guarantee (proven for isotropic distributions) may not hold. We may need to use multiple independent circulant blocks.

5. **Re-drawing features**: FAVOR+ benefits from periodic re-drawing of random features during training. With learnable circulant, this isn't applicable — the projection adapts continuously instead. Need to verify this doesn't cause overfitting.

6. **Cosine reweighting doubles computation**: The Ptolemy decomposition requires two independent linear attention computations (cos and sin branches), doubling the constant factor. Combined with the circulant speedup, the total may not be faster than simpler approaches.

## Follow-up Experiments

1. **Scale to 1B+ parameters**: If C-FAVOR+ works at 8M scale, test on Mamba-2-scale language models as a drop-in replacement for the linear attention head
2. **Multiple circulant blocks**: Stack 2–4 independent circulant projections (each $O(d \log d)$) for $m = 2d$–$4d$ features, testing whether more features compensate for row dependence
3. **Structured orthogonal circulant**: Enforce approximate orthogonality between circulant blocks via the optimal circulant approximation trick — project the stacked circulant onto the nearest orthogonal structure
4. **Application to SSM readout**: Use C-FAVOR+ as the feature map in GLA-style SSMs, replacing the standard linear attention kernel with circulant random features for the KV state
5. **Hybrid heads**: Mix C-FAVOR+ heads (cheap, long-range) with softmax heads (expensive, precise) within the same layer, using expert-choice routing to select which heads process which tokens
6. **Integration with FlashAttention-style tiling**: Develop an IO-aware implementation that fuses the FFT projection with the linear attention accumulation, minimizing HBM reads

## Human Review

(To be filled by reviewer)

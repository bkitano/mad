---
status: failed
priority: high
created: 2026-02-15
based_on: blackbox-randomized-hss-compression, telescopic-decomposition-hss, semiseparable-block-decomposition, chunkwise-parallel-scan, log-linear-attention
experiment_number: 021
experiment_log: experiment-log-021.md
results_file: 021_results.md
---

# Black-Box HSS Compression for Adaptive Hierarchical Linear Attention

## Hypothesis

Compressing the cumulative attention/state matrix of a linear attention model into HSS form via black-box randomized compression — and then computing readout via the telescopic decomposition — will discover **adaptive multi-resolution temporal structure** that outperforms the fixed Fenwick-tree hierarchy of log-linear attention, achieving comparable $O(T \log T)$ quality at lower wall-clock cost by exploiting hardware-friendly $O(k^2 T)$ HSS matvecs rather than $O(\log T)$ independent scan passes.

## Background

### The Fixed vs. Adaptive Hierarchy Problem

Log-linear attention (Proposal 008, Gated DeltaNet paper) achieves $O(T \log T)$ training by partitioning the sequence into $O(\log T)$ levels of a Fenwick tree, with each level maintaining an independent hidden state summarizing tokens at that temporal resolution. This works well but imposes a **fixed, predetermined hierarchical structure**: the partition boundaries are determined by binary decomposition of position indices, not by the content of the tokens.

Real sequences have **non-uniform temporal structure**: in language, paragraph boundaries, topic shifts, and dialogue turns create natural groupings at various scales. A coding file has hierarchical scope (function → class → module) that doesn't align with power-of-two boundaries. The Fenwick tree cannot adapt to this.

### HSS Matrices Capture Adaptive Hierarchies

The HSS (Hierarchically Semiseparable) framework provides exactly the mathematical structure needed: a binary tree where each level captures interactions at a different spatial scale, with **adaptive rank** at each node. The key insight from Mamba-2's SSD framework is that SSM output matrices are semiseparable — and HSS generalizes semiseparable with a hierarchical tree that can be adapted to data.

### Black-Box Compression: Learning Structure from Matvecs

The black-box HSS compression algorithm (Levitt & Martinsson 2024) can discover HSS structure using only $O(k)$ matrix-vector products — no individual matrix entries needed. For attention matrices, the "matvec" $y = \text{Attn}(Q, K, V) \cdot x$ is exactly what we compute during training. This means we can:

1. **Discover** the hierarchical low-rank structure of the attention pattern from forward passes
2. **Compress** it into an HSS representation with $O(kT)$ parameters
3. **Apply** it via telescopic decomposition for efficient readout at $O(k^2 T)$ cost

### Why This Combination Is Novel

- **Proposal 005** (Segmented HSS Linear Attention) parameterizes the state matrix $S_t$ in HSS form but uses a *fixed* HSS tree and requires explicit HSS structure in the model. Our approach discovers the tree adaptively.
- **Proposal 008** (Cos-LogLinear) uses a fixed Fenwick hierarchy. Our approach replaces this with data-dependent hierarchy.
- **Proposal 002** (SSD-DeltaNet WY Hybrid) exploits semiseparable structure but only at the 1-semiseparable level. HSS captures higher-order hierarchical rank structure.

The unique combination here is: **black-box compression discovers the hierarchy**, **telescopic decomposition makes matrix functions efficient on it**, and **chunkwise parallel scan provides the training framework**.

## Mathematical Formulation

### Linear Attention as Matrix Application

In linear attention with feature map $\phi$, the output at position $t$ is:

$$
o_t = \frac{\sum_{j=1}^{t} \phi(q_t)^\top \phi(k_j) \cdot v_j}{\sum_{j=1}^{t} \phi(q_t)^\top \phi(k_j)}
$$

The unnormalized attention matrix is $M \in \mathbb{R}^{T \times T}$ with $M_{ij} = \phi(q_i)^\top \phi(k_j)$ for $i \geq j$ (causal mask). This is a rank-$d_k$ semiseparable matrix when $\phi$ has dimension $d_k$.

### HSS Representation of Attention

We represent $M$ in HSS form via a binary cluster tree $\mathcal{T}$ of depth $L = O(\log T)$:

$$
M = \mathbf{D}^{(L)} + \mathbf{U}^{(L)} \tilde{M}^{(L-1)} (\mathbf{V}^{(L)})^\top
$$

where recursively $\tilde{M}^{(\ell)} = \mathbf{D}^{(\ell)} + \mathbf{U}^{(\ell)} \tilde{M}^{(\ell-1)} (\mathbf{V}^{(\ell)})^\top$.

**Key variables:**
- $\mathbf{U}^{(\ell)}, \mathbf{V}^{(\ell)} \in \mathbb{R}^{T_\ell \times r_\ell}$ — block-diagonal basis matrices at level $\ell$ (learned or compressed)
- $\mathbf{D}^{(\ell)}$ — block-diagonal discrepancy at level $\ell$ (captures local interactions)
- $r_\ell$ — rank at level $\ell$ (adaptive: coarser levels may need higher rank)
- $L = O(\log T)$ — tree depth

### Black-Box Compression Step

Given the implicit linear attention matrix $M$ (accessible only via matvec $y = Mv$), we compress:

1. **Sample**: Draw $\Omega, \Psi \in \mathbb{R}^{T \times s}$ with $s = 3r$ Gaussian columns
2. **Probe**: Compute $Y = M\Omega$ and $Z = M^\top \Psi$ via standard linear attention forward passes
3. **Compress**: Apply Levitt-Martinsson Algorithm 4.1 to extract $\{U_\tau, V_\tau, D_\tau\}$ for all tree nodes $\tau$

The compression requires only $O(r)$ forward passes of the linear attention layer — amortized over training, this is negligible.

### Telescopic Readout

Once the HSS representation is obtained, compute the attention output $o = Mv$ via the telescopic decomposition:

$$
o = \mathbf{D}^{(L)} v + \mathbf{U}^{(L)} \left( \mathbf{D}^{(L-1)} (\mathbf{V}^{(L)})^\top v + \mathbf{U}^{(L-1)} \left( \cdots \right) \right)
$$

This is an upward pass (compress $v$ through $V$ bases) followed by a downward pass (expand through $U$ bases), costing $O(rT)$ per level × $O(\log T)$ levels = $O(rT \log T)$ total.

### Learnable HSS Attention Layer

Rather than compressing post-hoc, we propose a **learnable HSS attention layer** where the tree structure and basis matrices are parameterized and trained end-to-end:

$$
o_t = \sum_{\ell=0}^{L} \mathbf{U}_t^{(\ell)} \left( \sum_{j \in \text{bucket}(\ell, t)} D_{tj}^{(\ell)} \cdot v_j \right)
$$

**Parameterization:**
- $\mathbf{U}^{(\ell)}, \mathbf{V}^{(\ell)}$: Learned projection matrices per level (shared across positions within a block)
- $D^{(\ell)}$: Intra-block attention at level $\ell$, computed as small dense attention within each block of size $T/2^\ell$
- Gating: $\lambda_t^{(\ell)} = \sigma(x_t W_\lambda^{(\ell)})$ gates the contribution of each level (like log-linear attention's $\lambda$ weights)

**Combined readout:**

$$
o_t = \sum_{\ell=0}^{L} \lambda_t^{(\ell)} \cdot (\mathbf{U}^{(\ell)})_t \cdot S_t^{(\ell)}
$$

where $S_t^{(\ell)} = (\mathbf{V}^{(\ell)})^\top_{\text{block}(t)} \cdot \text{Attn}_{\text{local}}^{(\ell)}(q_t, K, V)$ is the compressed state at level $\ell$.

### Training via Chunkwise HSS Scan

For training, we use a chunkwise approach analogous to SSD:

1. **Intra-chunk** (chunk size $C$): Dense quadratic attention within each chunk — $O(C^2 d)$ matmuls on tensor cores
2. **Inter-chunk HSS scan**: Instead of a single linear scan (SSD) or $O(\log T)$ independent scans (log-linear), perform a single HSS-structured scan that maintains $r$-dimensional compressed states at each tree level

The inter-chunk scan uses the HSS structure:

$$
S_{c+1}^{(\ell)} = A_c^{(\ell)} S_c^{(\ell)} + B_c^{(\ell)} x_c^{(\ell)}
$$

where $A_c^{(\ell)} \in \mathbb{R}^{r \times r}$ is the inter-block transition at level $\ell$ and $x_c^{(\ell)}$ is the compressed chunk output projected into level $\ell$'s basis.

**Total training cost:** $O(T C d + (T/C) \cdot r^2 \cdot L)$ where $L = O(\log T)$.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | HSS-Attention Transformer |
| Layers | $L_{\text{model}} = 12$ |
| Model dim | $d = 768$ |
| Heads | $H = 12$ |
| Head dim | $d_k = 64$ |
| HSS tree depth | $L_{\text{tree}} = \lceil \log_2(T/C) \rceil$ |
| HSS rank per level | $r = 16$ (uniform) or adaptive |
| Chunk size | $C = 64$ |
| Attention type | cosFormer-style (ReLU + cosine reweighting) within chunks |

### Baseline

1. **Standard linear attention** (GLA/RetNet): $O(Td^2)$, fixed $O(1)$ state
2. **Log-linear attention**: $O(T \log T \cdot d)$, $O(\log T)$ states, fixed Fenwick hierarchy
3. **Softmax Transformer + FlashAttention-2**: $O(T^2 d)$, quadratic
4. **Mamba-2 (SSD)**: $O(TC + (T/C) \cdot N)$, 1-semiseparable

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MQAR accuracy (8 KV, T=4096) | $\geq$ log-linear | Synthetic retrieval task |
| WikiText-103 PPL (350M) | $\leq$ log-linear + 1.0 | Validation perplexity |
| Needle-in-Haystack (T=32K) | $> 95\%$ | Retrieval depth accuracy |
| Throughput | $> 0.8\times$ log-linear | Tokens/sec on A100 |
| Memory | $\leq 1.2\times$ linear attention | Peak GPU memory |

### Estimated Compute

**Full experiment**: ~200 GPU-hours (A100)
- 350M parameter model, 15B tokens
- Baselines: ~150 GPU-hours additional
- Total: ~350 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- **MQAR**: Matches or exceeds log-linear attention at $T = 4096$, because adaptive hierarchy places bucket boundaries at semantically meaningful positions rather than power-of-two positions
- **Perplexity**: Within 0.5 of log-linear (the adaptive hierarchy should strictly dominate the fixed one)
- **Needle-in-Haystack**: Better at non-power-of-two depths (where Fenwick tree has resolution gaps)
- **Throughput**: Within $0.8\times$ of log-linear (HSS matvec is $O(rT)$ per level vs log-linear's $O(Td^2/H)$ per level; with $r < d^2/H$ this is faster)

**If hypothesis is wrong:**
- **If adaptive hierarchy doesn't help**: The Fenwick tree's fixed structure may already be sufficient for most tasks. This would mean sequence structure is more uniform than expected, and the overhead of learning the hierarchy isn't worth it.
- **If compression quality is poor**: The attention matrix may not have hierarchical low-rank structure beyond what 1-semiseparable captures. This would validate SSD's approach and suggest HSS is overkill.
- **Either way, we learn**: Whether attention matrices have *hierarchical* low-rank structure (HSS) or only *flat* low-rank structure (semiseparable), and whether adaptive hierarchies help for language modeling.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer HSS-Attention, $d = 64$, $H = 4$, $d_k = 16$, $r = 4$, tree depth 3 (~120K params)
- **Task**: Multi-Query Associative Recall (MQAR) with **non-uniform key placement** — keys are clustered at random positions (simulating paragraph structure) rather than uniformly distributed
- **Data**: Synthetic MQAR, $T = 256$, 8 KV pairs placed in 2-3 clusters, 10K training samples
- **Compute**: Single GPU, $< 10$ minutes
- **Comparison**: Fixed Fenwick (log-linear style) vs adaptive HSS hierarchy on the same task

### Success Criteria
- $> 90\%$ MQAR accuracy where fixed Fenwick achieves $< 80\%$ on the clustered-key variant
- The learned HSS tree boundaries correlate with the key cluster positions (measured by mutual information between tree partition and cluster assignment)

### Failure Criteria
- If HSS-Attention performs no better than fixed Fenwick on clustered keys, the adaptive hierarchy provides no benefit over simple binary decomposition
- If the black-box compression requires $> 20$ matvecs to converge to a useful HSS ($> 5\times$ overhead), the discovery cost is prohibitive

### Why This Test Is Sufficient
- The clustered-key MQAR task is specifically designed to expose the weakness of fixed hierarchies: if keys cluster near positions 50-60 and 180-200, a power-of-two partition at positions 64 and 128 will split one cluster and miss the other. An adaptive HSS tree can place boundaries at 50 and 180 instead.
- If the tiny model demonstrates adaptive boundary placement on this synthetic task, the same mechanism will help with natural language where topic boundaries are non-uniform.
- The HSS rank $r = 4$ is small enough that the overhead is minimal, making the comparison fair.

## Theoretical Analysis

**Complexity comparison:**

| Operation | Linear Attention | Log-Linear | HSS-Attention (this) |
|-----------|-----------------|------------|---------------------|
| Forward pass | $O(T d_k^2)$ | $O(T \log T \cdot d_k^2)$ | $O(TC d_k + \frac{T}{C} r^2 \log T)$ |
| Backward pass | $O(T d_k^2)$ | $O(T \log T \cdot d_k^2)$ | $O(TC d_k + \frac{T}{C} r^2 \log T)$ |
| Decoding memory | $O(d_k^2)$ | $O(\log T \cdot d_k^2)$ | $O(r \log T)$ |
| Decoding time/step | $O(d_k^2)$ | $O(\log T \cdot d_k^2)$ | $O(r^2 \log T + d_k)$ |

**Crossover point:** HSS-Attention is faster than log-linear when $r^2 < d_k^2$, i.e., $r < d_k$. With $d_k = 64$ and $r = 16$, this gives a $16\times$ reduction in inter-chunk scan cost.

**Key advantage:** The decoding memory is $O(r \log T)$ vs $O(\log T \cdot d_k^2)$ for log-linear — a factor of $d_k^2 / r$ improvement. For $d_k = 64, r = 16$, this is $256\times$ less memory per head.

## Risks & Limitations

1. **Black-box compression overhead during training**: The $O(r)$ matvecs needed per compression step add $\sim 3r \times$ the cost of a single forward pass. Mitigation: compress only every $K$ training steps (e.g., every 100 steps) and reuse the HSS structure between compressions. Alternatively, learn the HSS parameters directly via gradient descent, using compression only for initialization.

2. **Tree structure must be differentiable**: The binary tree partition is discrete. Mitigation: fix the tree structure (balanced binary) and only learn the rank allocation $r_\ell$ and basis matrices $U, V$ at each level. The adaptivity comes from the learned bases, not the tree topology.

3. **HSS rank uniformity assumption**: Real attention matrices may have wildly varying ranks across tree levels. Mitigation: use the Hutchinson trace estimator (Proposal 018) to estimate per-level rank importance and allocate adaptively.

4. **Not GPU-native**: HSS tree traversal is inherently sequential across levels. Mitigation: with $L = O(\log T) \leq 12$ levels, the sequential overhead is small. Each level's computation is embarrassingly parallel across nodes.

5. **Comparison fairness**: Log-linear attention has been heavily optimized; our HSS variant starts unoptimized. Mitigation: focus the MVE on quality (MQAR accuracy) rather than throughput.

## Follow-up Experiments

1. **Adaptive tree topology**: Learn the tree structure itself (not just basis matrices) using a differentiable tree construction algorithm
2. **HSS + tropical attention**: Replace the dense intra-block attention with tropical attention (Proposal 015) for sharp retrieval within blocks
3. **Compression-free variant**: Remove the black-box compression step entirely; instead, parameterize $U^{(\ell)}, V^{(\ell)}$ as learned projections (making the model a "structured multi-resolution attention" without explicit HSS)
4. **Scale to 1.3B**: Test whether adaptive hierarchy benefits increase with model scale (more parameters → more capacity to exploit hierarchical structure)
5. **Visualization**: Analyze what temporal hierarchies the model learns on natural language — do HSS tree boundaries correlate with sentence/paragraph structure?

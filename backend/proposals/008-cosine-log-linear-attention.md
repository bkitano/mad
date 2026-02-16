---
status: completed
priority: medium
created: 2026-02-15
based_on: 
experiment_number: 008
experiment_log: experiment-log-008.md
results_file: 008_results.md
---

# Cosine-Reweighted Log-Linear Attention (cos-LogLinear)

- cosine-reweighted-linear-attention
- log-linear-attention
- online-softmax
- chunkwise-parallel-scan
- io-aware-tiling

## Hypothesis

Composing cosFormer's cosine re-weighting (which provides locality bias + non-negativity via ReLU, matching softmax quality at $O(Td^2)$) with log-linear attention's hierarchical multi-resolution states (which expand recall capacity from $O(1)$ to $O(\log T)$ states) will produce an attention mechanism that:
1. **Improves recall** over cosFormer alone (which, like all linear attention, is limited by its fixed $d \times d$ state)
2. **Improves quality** over log-linear attention with vanilla linear attention kernels (which lacks locality bias)
3. **Maintains $O(T \log T)$ training and $O(\log T)$ decoding** — better than softmax ($O(T^2)$, $O(T)$) at long sequences

**Core insight**: cosFormer and log-linear attention address orthogonal weaknesses of linear attention. cosFormer fixes *quality* (via locality bias) but not *capacity* (still one $d \times d$ state). Log-linear attention fixes *capacity* (via $O(\log T)$ hierarchical states) but inherits whatever quality the base kernel provides. Composing them should yield compounding improvements — better quality *per state* AND more states.

## Background

**Linear attention** computes $O_i = \phi(q_i)^\top S_i$ where $S_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{d \times d}$ is a fixed-size state. Two fundamental limitations:

1. **Quality gap**: Without locality bias or non-negativity, linear attention produces diffuse, uniform-ish weight distributions that degrade perplexity relative to softmax. cosFormer addresses this via ReLU + cosine re-weighting, achieving comparable quality.

2. **Capacity gap**: A single $d \times d$ state must compress the entire history $\{(k_j, v_j)\}_{j < i}$ — an information bottleneck when $T \gg d^2$. Log-linear attention addresses this by maintaining $O(\log T)$ states at different temporal resolutions via Fenwick tree partitioning.

**Neither alone is sufficient:**
- cosFormer with one state still has bounded recall — it cannot remember $T \gg d^2$ distinct KV pairs
- Log-linear attention with a weak base kernel (e.g., vanilla $\phi(x) = \text{elu}(x) + 1$) has $O(\log T)$ states but each state has poor quality

**No existing proposal or paper combines these techniques.** The closest is:
- Log-linear Mamba-2 / Gated DeltaNet (in the log-linear attention paper), which applies hierarchical masking to specific SSM kernels — but not to cosFormer-style reweighted attention
- Proposal 005 (Segmented-HSS Linear Attention), which addresses variable-length batching with HSS structure — a different problem (batching efficiency vs. recall quality)

## Mathematical Formulation

### cosFormer Recap

cosFormer defines the attention weight between positions $i$ and $j$ as:

$$
s(Q'_i, K'_j) = Q'^{\top}_i K'_j \cos\left(\frac{\pi}{2} \cdot \frac{i - j}{M}\right)
$$

where $Q' = \text{ReLU}(Q)$, $K' = \text{ReLU}(K)$, and $M \geq T$.

Via the Ptolemy identity $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$, this decomposes into two linear attention terms:

$$
O_i = \frac{(Q_i^{\cos})^\top S_i^{\cos} + (Q_i^{\sin})^\top S_i^{\sin}}{(Q_i^{\cos})^\top z_i^{\cos} + (Q_i^{\sin})^\top z_i^{\sin}}
$$

where (causal form):

$$
S_i^{\cos} = \sum_{j \leq i} K_j^{\cos} V_j^\top, \quad S_i^{\sin} = \sum_{j \leq i} K_j^{\sin} V_j^\top
$$

$$
z_i^{\cos} = \sum_{j \leq i} K_j^{\cos}, \quad z_i^{\sin} = \sum_{j \leq i} K_j^{\sin}
$$

and the modulated queries/keys are:

$$
Q_i^{\cos} = Q'_i \cos\tfrac{\pi i}{2M}, \quad Q_i^{\sin} = Q'_i \sin\tfrac{\pi i}{2M}
$$
$$
K_j^{\cos} = K'_j \cos\tfrac{\pi j}{2M}, \quad K_j^{\sin} = K'_j \sin\tfrac{\pi j}{2M}
$$

**State count**: 2 matrices ($S^{\cos}, S^{\sin}$) each $d \times d$, plus 2 vectors ($z^{\cos}, z^{\sin}$) each $d$-dimensional. Total: $2d^2 + 2d$.

### Log-Linear Extension

Log-linear attention replaces the single cumulative state with $L = \lceil \log_2(T+1) \rceil + 1$ levels of hierarchical states, partitioned via a Fenwick tree. The hierarchical mask is:

$$
\mathbf{M}^{\mathcal{H}}_{ts} = \begin{cases} \lambda_t^{\ell(t,s)} & \text{if } s \leq t \\ 0 & \text{otherwise} \end{cases}
$$

where $\lambda_t^{(\ell)} \geq 0$ are learned input-dependent weights controlling attention to each temporal scale.

### Proposed: cos-LogLinear Attention

**Compose the cosFormer kernel with the hierarchical mask:**

$$
O_i = \frac{\sum_{\ell=0}^{L-1} \lambda_i^{(\ell)} \left[ (Q_i^{\cos})^\top S_i^{(\ell, \cos)} + (Q_i^{\sin})^\top S_i^{(\ell, \sin)} \right]}{\sum_{\ell=0}^{L-1} \lambda_i^{(\ell)} \left[ (Q_i^{\cos})^\top z_i^{(\ell, \cos)} + (Q_i^{\sin})^\top z_i^{(\ell, \sin)} \right]}
$$

where at each level $\ell$, we maintain separate cosine and sine states:

$$
S_i^{(\ell, \cos)} = \sum_{j \in \mathcal{B}_i^{(\ell)}} K_j^{\cos} V_j^\top \in \mathbb{R}^{d \times d}
$$

$$
S_i^{(\ell, \sin)} = \sum_{j \in \mathcal{B}_i^{(\ell)}} K_j^{\sin} V_j^\top \in \mathbb{R}^{d \times d}
$$

$$
z_i^{(\ell, \cos)} = \sum_{j \in \mathcal{B}_i^{(\ell)}} K_j^{\cos} \in \mathbb{R}^d, \quad z_i^{(\ell, \sin)} = \sum_{j \in \mathcal{B}_i^{(\ell)}} K_j^{\sin} \in \mathbb{R}^d
$$

**Key Variables:**
- $Q, K, V \in \mathbb{R}^{T \times d}$ — query, key, value matrices
- $T$ — sequence length
- $d$ — head dimension
- $L = O(\log T)$ — number of hierarchical levels
- $M \geq T$ — cosine re-weighting scale
- $\lambda_i^{(\ell)} \in \mathbb{R}_{\geq 0}$ — learned per-level attention weights
- $\mathcal{B}_i^{(\ell)}$ — Fenwick tree bucket at level $\ell$ for position $i$

**Decoding (per token):**

At each step $t$, maintain $L$ pairs of states $(S_t^{(\ell, \cos)}, S_t^{(\ell, \sin)})$ and normalization vectors $(z_t^{(\ell, \cos)}, z_t^{(\ell, \sin)})$. Update the states at the appropriate level when Fenwick tree boundaries are crossed (same protocol as log-linear attention).

**State count (decoding):** $2L$ matrices of size $d \times d$ plus $2L$ vectors of size $d$:

$$
\text{Memory} = L \cdot (2d^2 + 2d) = O(d^2 \log T)
$$

Compare:
- cosFormer: $O(d^2)$ — fixed
- Log-linear (vanilla kernel): $O(d^2 \log T)$ — same
- Softmax attention: $O(Td)$ — linear in $T$

### Training: Chunkwise Hierarchical Scan

**Parallel training** follows the log-linear attention framework. The full attention matrix is:

$$
O = \left(Q^{\cos} (K^{\cos})^\top + Q^{\sin} (K^{\sin})^\top\right) \odot \mathbf{M}^{\mathcal{H}} \cdot V
$$

This decomposes into:

$$
O = \underbrace{\text{diag-block term}}_{\text{intra-chunk, } O(C^2 d)} + \sum_{\ell=1}^{L-1} \underbrace{\text{level-}\ell \text{ term}}_{\text{inter-chunk scan, } O(Td)}
$$

**Intra-chunk** (chunk size $C$): Standard cosFormer attention within each chunk, $O(C^2 d)$ per chunk. Since $C$ is small (e.g., 64–256), this is efficient and can use FlashAttention-style tiling.

**Inter-chunk** (per level $\ell$): A scan over $T/C$ chunk-level cosine and sine states. Each scan is a simple cumulative sum of $d \times d$ matrices — $O(Td^2)$ per level. With $L = O(\log T/C)$ levels:

$$
\text{Total inter-chunk} = O(Td^2 \log(T/C))
$$

**Total training cost**: $O(Td^2 \log T + T C d)$ — the $O(Td^2)$ base cost of cosFormer multiplied by the $O(\log T)$ factor from the hierarchical decomposition.

### Why cosFormer + Log-Linear Is Natural

The composition works cleanly because:

1. **cosFormer's decomposition is linear**: The Ptolemy split gives two independent linear attention streams. Each stream can independently be extended to log-linear by replacing the single cumulative state with $L$ hierarchical states.

2. **Log-linear attention is kernel-agnostic**: The hierarchical mask $\mathbf{M}^{\mathcal{H}}$ composes with *any* linear attention masking matrix via element-wise product. cosFormer's implicit mask (the cosine re-weighting) factors into position-dependent query/key modulations that are compatible.

3. **Normalization composes correctly**: cosFormer requires per-position normalization $O_i / z_i$. With hierarchical states, the numerator and denominator both sum over levels:

$$
O_i = \frac{\sum_\ell \lambda_i^{(\ell)} \text{num}_i^{(\ell)}}{\sum_\ell \lambda_i^{(\ell)} \text{den}_i^{(\ell)}}
$$

This is well-defined as long as the denominator is positive — guaranteed by ReLU non-negativity of the keys and the non-negativity of $\lambda_i^{(\ell)}$.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | cos-LogLinear Attention (drop-in for attention layers) |
| Layers | $L_{\text{model}} = 12$ |
| Hidden dim | $d_{\text{model}} = 768$ |
| Heads | $H = 12$ |
| Head dim | $d = 64$ |
| Log-linear levels | $L = \lceil \log_2(T/C) \rceil + 1$ |
| Chunk size | $C = 64$ |
| Sequence length | $T = 2048$ (training), up to 8192 (eval) |

### Baseline

1. **cosFormer** (standard, single state): $O(Td^2)$ — tests whether hierarchical states add value
2. **Log-linear GLA** (hierarchical, no cosine reweighting): $O(Td^2 \log T)$ — tests whether cosine reweighting adds value
3. **Softmax attention** (FlashAttention-2): $O(T^2 d)$ — gold standard for quality
4. **Vanilla linear attention** ($\phi = \text{elu}+1$): $O(Td^2)$ — weakest baseline

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\leq 1.05 \times$ softmax | WikiText-103, 125M param model |
| MQAR recall | $> 85\%$ at 8 KV pairs, seq 4096 | Multi-Query Associative Recall |
| Throughput | $> 1.5 \times$ FlashAttn at $T = 8192$ | Tokens/sec on A100 |
| Decoding memory | $< 0.1 \times$ softmax KV cache | Peak memory per token at $T = 8192$ |
| Quality vs. state count | Pareto-dominant over cosFormer | Accuracy vs. memory scatter plot |

### Estimated Compute

**MVE**: ~10 minutes on single GPU (~$0.50)
**Small-scale**: 16 GPU-hours on A100 (~$64)
**Full-scale**: 100 GPU-hours on A100 (~$400)

## Expected Outcome

**If hypothesis is correct:**
- **Recall improvement over cosFormer**: On MQAR with 8 KV pairs at $T = 4096$, cos-LogLinear achieves $> 85\%$ recall vs. cosFormer's $< 60\%$ (capacity-limited). The $O(\log T)$ hierarchical states allow the model to store more KV associations without interference.
- **Quality improvement over vanilla log-linear**: On WikiText-103, cos-LogLinear achieves $\leq 1.05 \times$ softmax perplexity, while log-linear with $\phi = \text{elu}+1$ achieves $\sim 1.10 \times$ softmax. The cosine locality bias concentrates attention weight distributions, improving language modeling quality.
- **Compounding**: The improvements are additive — cos-LogLinear achieves both the quality of cosFormer and the recall of log-linear, dominating both on a quality-vs-memory Pareto frontier.

**If hypothesis is wrong:**
- **Scenario A**: cos-LogLinear matches cosFormer but doesn't improve over log-linear GLA
  - **Learn**: The quality improvement from cosine reweighting is negligible when sufficient state capacity is available — log-linear's extra states compensate for the missing locality bias
  - **Insight**: For high-capacity models, kernel choice matters less than state count
- **Scenario B**: The $2\times$ state overhead from Ptolemy (two states per level) makes cos-LogLinear's memory worse than a simpler kernel with more levels
  - **Learn**: The Ptolemy decomposition's constant factor is not justified when budget-constrained
  - **Fix**: Use a single feature map (e.g., just ReLU, no cosine) with $2L$ levels instead of $2 \times L$ states
- **Scenario C**: Normalization instability at long sequences
  - **Learn**: Summing normalized terms across levels can create numerical issues when level weights $\lambda_i^{(\ell)}$ span many orders of magnitude
  - **Fix**: Apply online softmax over the level dimension: normalize $\lambda_i^{(\ell)}$ before combining

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer cos-LogLinear attention model ($d = 32$, 2 heads of dim 16, ~80K params)
- **Task**: **Multi-Query Associative Recall (MQAR)** — given sequence `k1 v1 k2 v2 ... kN vN [SEP] k3 k1`, recall `v3 v1`. Use $N = 8$ KV pairs at sequence length $T = 128$.
- **Data**: 5K synthetic sequences
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- cos-LogLinear achieves $> 80\%$ accuracy on MQAR with 8 KV pairs at $T = 128$
- Standard cosFormer (single state, same head dim) achieves $< 50\%$ accuracy (capacity-limited at 8 pairs with $d = 16$)
- Vanilla log-linear attention (elu+1 kernel, same number of levels) achieves $60$–$70\%$ (benefits from capacity but lacks locality bias)
- Training completes without NaN/Inf (normalization is stable)

### Failure Criteria
- cos-LogLinear cannot beat cosFormer by more than 5% on MQAR — hierarchical states don't help with cosine kernel
- cos-LogLinear is worse than vanilla log-linear — cosine reweighting actively hurts when combined with hierarchical structure
- Normalization produces NaN at $T = 128$ — fundamental numerical issue

### Why This Test Is Sufficient
- MQAR directly tests recall capacity, which is the core motivation for adding log-linear states to cosFormer. 8 KV pairs at $d = 16$ exceeds the capacity of a single $16 \times 16$ state matrix, forcing the model to use hierarchical states.
- The head dimension $d = 16$ is deliberately small to stress-test state capacity — at larger $d$, any method would have enough capacity for 8 pairs.
- If the tiny model shows clear stratification (cos-LogLinear > log-linear > cosFormer > vanilla linear), the principle generalizes to larger models where the advantages compound.

## Theoretical Analysis

### Complexity Comparison

| Operation | Softmax | cosFormer | Log-linear (vanilla) | cos-LogLinear |
|-----------|---------|-----------|----------------------|---------------|
| Training | $O(T^2 d)$ | $O(Td^2)$ | $O(Td^2 \log T)$ | $O(Td^2 \log T)$ |
| Decoding (per step) | $O(Td)$ | $O(d^2)$ | $O(d^2 \log T)$ | $O(d^2 \log T)$ |
| Decoding memory | $O(Td)$ | $O(d^2)$ | $O(d^2 \log T)$ | $O(2d^2 \log T)$ |
| Quality (empirical) | Best | Near-softmax | Good | Near-softmax? |
| Recall capacity | $O(T)$ | $O(d^2)$ | $O(d^2 \log T)$ | $O(d^2 \log T)$ |

**Crossover with softmax:**
- Training: cos-LogLinear is faster when $T \log T < T^2 / d$, i.e., $T > d \log T$. For $d = 64$: $T > 64 \cdot 6 \approx 400$ — practically always faster.
- Decoding: cos-LogLinear uses $O(d^2 \log T)$ memory vs. $O(Td)$ for KV cache. Better when $d \log T < T$, i.e., $T > d \log T$ — again, practically always.

**Crossover with cosFormer (single state):**
- cos-LogLinear pays $O(\log T)$ factor in both compute and memory. This is justified if the recall improvement is significant (i.e., MQAR accuracy gap).
- At $T = 8192$, $\log T \approx 13$, so cos-LogLinear uses ~13× more states but may achieve dramatically better recall.

### Information-Theoretic Argument

A single $d \times d$ state matrix has $O(d^2)$ parameters, limiting the number of KV pairs it can faithfully store to $O(d^2 / d) = O(d)$ pairs (rate-distortion bound). With $L = O(\log T)$ states, the capacity grows to $O(d \log T)$ pairs — still sub-linear in $T$ but sufficient for typical associative recall tasks.

The cosine reweighting ensures that the stored information is *weighted* by recency, so the most important (recent) information occupies the highest-resolution state, while distant context is progressively compressed into lower-resolution states. This matches the empirical attention distribution of language models (sharp for nearby tokens, diffuse for distant ones).

## Risks & Limitations

### Risk 1: Ptolemy Decomposition Doubles State Count
- **Issue**: cosFormer requires 2 states (cos, sin) per level, so the total is $2L$ states vs. $L$ for vanilla log-linear. This may not be worth the quality gain.
- **Mitigation**: If the $2\times$ overhead matters, use only the cosine term (drop sine), getting locality bias at $1\times$ state count but losing the symmetric decomposition. Alternatively, use a single learned positional modulation $\alpha_i$ instead of the cos/sin pair.
- **Quantification**: For $d = 64$, $L = 13$: cosFormer log-linear uses $2 \times 13 \times 64^2 \approx 106K$ parameters in states vs. $13 \times 64^2 \approx 53K$ for vanilla log-linear. Both are negligible compared to model parameters.

### Risk 2: Cosine Reweighting Interacts Poorly with Hierarchical Buckets
- **Issue**: The cosine reweighting applies a smooth distance-based decay $\cos(\pi(i-j)/2M)$, but the Fenwick tree partitioning creates discrete, power-of-two-sized buckets. The smooth decay within each bucket may be redundant if bucket sizes already handle the multi-resolution aspect.
- **Mitigation**: This is an empirical question. The cosine decay operates *within* each bucket (intra-bucket locality), while the Fenwick structure handles *between* buckets (inter-bucket temporal resolution). These may be complementary rather than redundant.
- **Ablation**: Compare cos-LogLinear vs. ReLU-only log-linear (drop cosine, keep non-negativity) to isolate the contribution of locality bias.

### Risk 3: Level-Weight Normalization
- **Issue**: The per-level weights $\lambda_i^{(\ell)}$ must be combined with the cosFormer normalization. If $\lambda_i^{(\ell)}$ values span many orders of magnitude, the combined denominator $\sum_\ell \lambda_i^{(\ell)} \text{den}_i^{(\ell)}$ may be numerically unstable.
- **Mitigation**: Apply softmax over levels: $\tilde{\lambda}_i^{(\ell)} = \text{softmax}(\lambda_i^{(0)}, \ldots, \lambda_i^{(L-1)})_\ell$, ensuring the weights sum to 1 and are well-conditioned.
- **Alternative**: Use the online softmax trick to compute the level-wise normalization incrementally, avoiding catastrophic cancellation.

### Risk 4: Implementation Complexity
- **Issue**: Combining cosFormer's Ptolemy decomposition with log-linear's Fenwick tree scan requires careful engineering — both have non-trivial implementations.
- **Mitigation**: Start with a pure-PyTorch reference implementation (no custom kernels) for the MVE. Profile bottlenecks before optimizing.
- **Note**: Both cosFormer and log-linear attention have open-source implementations. The composition is a straightforward modification of log-linear's kernel function.

## Follow-up Experiments

### If Successful:
1. **Scale to 800M params**: Match the log-linear attention paper's scale (50B tokens) with cosine reweighting to measure perplexity improvement
2. **Needle-in-a-Haystack evaluation**: Test recall at extreme sequence lengths (32K–128K) where the $O(\log T)$ state advantage is maximal
3. **IO-aware tiling for cos-LogLinear**: Design a FlashAttention-style kernel that fuses the Ptolemy decomposition with the hierarchical scan, minimizing HBM round-trips for the intra-chunk quadratic computation
4. **Alternative locality functions**: Replace cosine with other decomposable locality biases — e.g., $\exp(-|i-j|/\tau)$ (exponential decay), which decomposes as a product of per-position terms and may give different inductive biases
5. **Adaptive level count**: Make $L$ data-dependent — use fewer levels for short sequences and more for long ones, saving compute on shorter inputs

### If Unsuccessful:
1. **Ablate cosine vs. ReLU-only**: Test if non-negativity alone (without cosine locality bias) captures most of the quality improvement
2. **Profile state utilization**: Measure the effective rank of each hierarchical state — are all $2L$ states being used, or do some collapse?
3. **Compare with FAVOR+ log-linear**: Replace cosFormer kernel with FAVOR+ (random feature softmax approximation) — is the quality gap due to cosine specifically or just having a better kernel?

## References to Tricks

- **cosine-reweighted-linear-attention**: Core kernel providing quality via locality bias and non-negativity
- **log-linear-attention**: Hierarchical multi-resolution state structure via Fenwick tree
- **online-softmax**: Monoid structure for composable normalization across levels and chunks
- **chunkwise-parallel-scan**: Training parallelization for the inter-chunk hierarchical scan
- **io-aware-tiling**: Design principle for future kernel optimization of intra-chunk computation

## Connection to Existing Proposals

- **Distinct from 005 (Segmented-HSS Linear Attention)**: Proposal 005 addresses *batching efficiency* (variable-length sequences without padding) using HSS for the state matrix structure. This proposal addresses *recall quality* (more states + better kernel) using Fenwick tree decomposition for the temporal structure. Different axes of improvement.
- **Orthogonal to 004/007 (Oscillatory SSMs)**: Those proposals improve SSM architectures; this proposal improves attention architectures. Could be used in the same model (SSM layers + cos-LogLinear attention layers).
- **Builds on log-linear attention paper**: The log-linear attention paper already demonstrates log-linear variants of Mamba-2 and Gated DeltaNet. This proposal applies the same framework to cosFormer — a natural next step that the authors did not explore.

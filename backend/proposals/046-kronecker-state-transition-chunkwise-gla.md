---
status: ongoing
priority: high
created: 2026-02-15
based_on: generalized-kronecker-product-decomposition (053), chunkwise-parallel-scan (026), tfla-two-level-tiled-chunkwise-parallelism (158), input-dependent-gating (065), batch-reduce-gemm (batch-reduce-gemm), matmulscan-tcu-parallel-scan (167), diagonal-plus-low-rank-dplr (035)
experiment_number: 046
experiment_log: experiment-log-046.md
---

# Kronecker-Factored State Transitions for Chunkwise GLA

## Hypothesis

Factoring GLA's inter-chunk state transition gate $G_k \in \mathbb{R}^{d_k \times d_k}$ as a **Kronecker product** $G_k = G_k^{(1)} \otimes G_k^{(2)}$ (where $G_k^{(1)} \in \mathbb{R}^{p \times p}$, $G_k^{(2)} \in \mathbb{R}^{q \times q}$, $pq = d_k$) and correspondingly reshaping the state $S_k \in \mathbb{R}^{d_k \times d_v}$ into a tensor $\mathcal{S}_k \in \mathbb{R}^{p \times q \times d_v}$ will achieve $1.3$–$2\times$ **wall-clock speedup on the inter-chunk scan** while maintaining model quality, because: (a) the state update $\mathcal{S}_k = (G_k^{(1)} \otimes G_k^{(2)}) \text{vec}(S_{k-1}) + \ldots$ decomposes into two smaller GEMMs of sizes $p \times p$ and $q \times q$ instead of one $d_k \times d_k$ GEMM, reducing the scan's per-step compute from $O(d_k^2 d_v)$ to $O((p^2 + q^2) d_v)$; (b) both small GEMMs are tensor-core-friendly and can be pipelined; and (c) the Kronecker structure provides a natural "multi-resolution" state representation where $G^{(1)}$ controls coarse dynamics and $G^{(2)}$ controls fine dynamics.

## Background

### The inter-chunk scan cost in GLA

In GLA's chunkwise parallel formulation, the sequence of $T$ tokens is divided into $N_c = T/C$ chunks. The inter-chunk state propagation is:

$$
S_k = G_k \cdot S_{k-1} + S_k^{\text{local}}
$$

where $S_k \in \mathbb{R}^{d_k \times d_v}$ is the chunk boundary state, $G_k \in \mathbb{R}^{d_k \times d_k}$ is the accumulated per-chunk decay gate (diagonal in standard GLA: $G_k = \text{diag}(\gamma_1^{(k)}, \ldots, \gamma_{d_k}^{(k)})$), and $S_k^{\text{local}}$ is the intra-chunk contribution.

For **diagonal** $G_k$ (standard GLA/Mamba-2), the scan is cheap: $d_k$ independent scalar scans, each requiring $O(N_c)$ work. The total inter-chunk cost is $O(N_c d_k d_v)$ — linear in everything.

However, several recent works argue for **non-diagonal** state transitions to increase expressivity:
- **DeltaNet**: Uses Householder reflections $A_t = I - \beta_t k_t k_t^\top$ — full rank-1 updates to an orthogonal matrix
- **DeltaProduct**: Uses products of Householder reflections — dense orthogonal transitions
- **Kimi Linear (DPLR)**: Uses diagonal-plus-low-rank $A_t = D_t + u_t v_t^\top$ — adds rank-1 off-diagonal interaction

All of these make the inter-chunk scan more expensive: the scan operator $(A, b) \circ (A', b') = (A \cdot A', A \cdot b' + b)$ now involves $d_k \times d_k$ matrix multiplication at every scan step, costing $O(N_c d_k^2 d_v)$ total.

### Kronecker factorization: the sweet spot

A Kronecker-factored transition $G_k = G_k^{(1)} \otimes G_k^{(2)}$ sits between diagonal (too restrictive) and dense (too expensive):

| Structure | Parameters per step | Scan cost per step | Expressivity |
|-----------|--------------------|--------------------|-------------|
| Diagonal | $d_k$ | $O(d_k d_v)$ | Low (independent channels) |
| **Kronecker** ($p \times q$) | $p^2 + q^2$ | $O((p^2 + q^2) d_v)$ | Medium (structured coupling) |
| Dense | $d_k^2$ | $O(d_k^2 d_v)$ | High (full coupling) |

For $d_k = 64$ with $p = q = 8$: Kronecker uses $128$ parameters (vs. $64$ diagonal, $4096$ dense) and costs $O(128 d_v)$ per scan step (vs. $O(64 d_v)$ diagonal, $O(4096 d_v)$ dense). This is a $32\times$ reduction vs. dense with only $2\times$ cost vs. diagonal.

The Kronecker structure provides **structured channel coupling**: $G^{(1)}$ mixes groups of $q$ channels, while $G^{(2)}$ mixes within each group. This is analogous to grouped convolutions (ResNeXt) or the "group and shuffle" pattern in ShuffleNet — both proven effective in vision.

### GPU efficiency of Kronecker state update

The key operation is the state update:

$$
S_k = (G_k^{(1)} \otimes G_k^{(2)}) \text{vec}(S_{k-1}) + S_k^{\text{local}}
$$

By the Kronecker identity $(A \otimes B) \text{vec}(X) = \text{vec}(B X A^\top)$, this becomes:

$$
S_k^{\text{reshaped}} = G_k^{(2)} \cdot S_{k-1}^{\text{reshaped}} \cdot (G_k^{(1)})^\top + S_k^{\text{local, reshaped}}
$$

where $S^{\text{reshaped}} \in \mathbb{R}^{q \times (p \cdot d_v)}$ (or equivalently as a tensor $\mathbb{R}^{q \times p \times d_v}$).

This is **two GEMMs**:
1. Left multiply: $G_k^{(2)} \cdot S_{k-1}^{\text{reshaped}}$ — shape $(q, q) \times (q, p \cdot d_v)$ — tensor-core GEMM
2. Right multiply: $\text{result} \cdot (G_k^{(1)})^\top$ — shape $(q \cdot d_v, p) \times (p, p)$ — tensor-core GEMM

Both are standard dense matmuls on small matrices. For $p = q = 8, d_v = 64$: sizes are $(8, 512) \times (8, 8)$ and $(512, 8) \times (8, 8)$ — perfectly sized for tensor cores.

### The parallel scan with Kronecker structure

The associative scan operator for Kronecker transitions is:

$$
(G_k^{(1)} \otimes G_k^{(2)}, b_k) \circ (G_j^{(1)} \otimes G_j^{(2)}, b_j) = ((G_k^{(1)} G_j^{(1)}) \otimes (G_k^{(2)} G_j^{(2)}), (G_k^{(1)} \otimes G_k^{(2)}) b_j + b_k)
$$

The key insight: **the Kronecker product of Kronecker products is a Kronecker product**:

$$
(G_k^{(1)} \otimes G_k^{(2)}) \cdot (G_j^{(1)} \otimes G_j^{(2)}) = (G_k^{(1)} G_j^{(1)}) \otimes (G_k^{(2)} G_j^{(2)})
$$

So the $A$-component of the scan remains Kronecker throughout — we never need to form the full $d_k \times d_k$ product. The scan reduces to two parallel scans of $p \times p$ and $q \times q$ matrix products, plus the $b$-component update.

### Memory access pattern analysis

**Coalesced access**: The Kronecker state $\mathcal{S} \in \mathbb{R}^{q \times p \times d_v}$ is stored in row-major order. The left-multiply by $G^{(2)}$ contracts the first dimension — this is a standard batched GEMM with contiguous memory access. The right-multiply by $(G^{(1)})^\top$ contracts the second dimension — this requires a transpose or can be computed via a second batched GEMM with stride-$d_v$ access (still coalesced within warps for $d_v \geq 32$).

**Cache-friendly**: The factor matrices $G^{(1)}, G^{(2)}$ are $p^2 + q^2 = 128$ elements — fit entirely in L1/registers. The state $\mathcal{S}$ is $q \times p \times d_v = 8 \times 8 \times 64 = 4096$ elements at FP16 = 8 KB — fits in shared memory.

**Arithmetic intensity**: For the left-multiply: $q^2 \times p \times d_v$ FLOPs accessing $q \times p \times d_v + q^2$ elements → AI $= q \approx 8$. Low but comparable to the diagonal case (AI $= 1$). Both are memory-bound; Kronecker is $8\times$ more compute per byte.

### What's different from existing proposals

- **Proposal 003** (DPLR Column-Sparse Cauchy): Uses diagonal-plus-low-rank transitions but with Cauchy kernel for the SSM. Different structure (DPLR vs. Kronecker) and different architecture (SSM vs. GLA).
- **Proposal 004** (Oscillatory DPLR SSM): Adds oscillatory eigenvalues to DPLR. Different mechanism entirely.
- **Proposal 022** (Displacement-Rank SSM): Uses displacement rank structure for transitions. More general but less GPU-friendly (requires solving Sylvester equations).
- **Proposal 006** (Monarch Gated SSM): Uses Monarch matrix factorization ($B_1 D B_2$) for transitions. Monarch is a block-diagonal-plus-permutation structure — different from Kronecker. Monarch requires gather/scatter for the permutation step.
- **Our approach**: Kronecker factorization has no permutation, no gather/scatter — just two small GEMMs. It's the simplest structured transition that adds channel coupling while remaining tensor-core-friendly.

## Related Work

- **Kronecker Recurrent Units** (Jose et al., ICML 2018): Used Kronecker products to compress vanilla RNN weight matrices. Applied to standard RNNs (not linear RNNs), without chunkwise parallelism, and on older hardware without tensor cores. Our approach applies Kronecker structure specifically to the gated state transition in modern linear RNNs, exploiting chunkwise-parallel training.
- **Kimi Linear / DPLR** (Kimi team, 2025): Uses diagonal-plus-low-rank transitions for enhanced expressivity. DPLR adds a rank-$r$ correction to diagonal: $A = D + uv^\top$. Kronecker is a different factorization that provides structured full-rank coupling without the rank constraint.
- **Monarch Matrices** (Dao et al., 2022): Factorizes matrices as $M = P B_1 D B_2$ (block-diagonal + permutation). Used in proposal 006 for SSM state transitions. Kronecker avoids the permutation step that breaks memory coalescing.
- **Block-Diagonal SSMs**: Several works use block-diagonal transitions (e.g., Mamba-2's multi-head SSM). Kronecker generalizes block-diagonal: $\text{diag}(G^{(2)}, \ldots, G^{(2)})$ is a special case of $I_p \otimes G^{(2)}$, but $G^{(1)} \otimes G^{(2)}$ with non-identity $G^{(1)}$ adds cross-block coupling.

**Gap**: No existing work applies Kronecker-factored state transitions to modern gated linear attention architectures (GLA, mLSTM) with chunkwise-parallel training.

## Mathematical Formulation

### Standard GLA Inter-Chunk Recurrence

$$
S_k = \text{diag}(\gamma_k) \cdot S_{k-1} + S_k^{\text{local}}, \quad S_k \in \mathbb{R}^{d_k \times d_v}
$$

Cost: $O(d_k d_v)$ per chunk (elementwise scaling).

### Proposed: Kronecker-GLA Inter-Chunk Recurrence

Factor $d_k = p \cdot q$ and reshape state: $\mathcal{S}_k \in \mathbb{R}^{q \times p \times d_v}$.

$$
\mathcal{S}_k = G_k^{(2)} \cdot \mathcal{S}_{k-1} \cdot_2 (G_k^{(1)})^\top + \mathcal{S}_k^{\text{local}}
$$

where $\cdot_2$ denotes contraction along the second mode (the $p$-dimension). Equivalently:

$$
\text{For each } l = 1, \ldots, d_v: \quad S_k[:, :, l] = G_k^{(2)} \cdot S_{k-1}[:, :, l] \cdot (G_k^{(1)})^\top
$$

**Gate parameterization**: The factors $G_k^{(1)}, G_k^{(2)}$ are computed as input-dependent functions:

$$
G_k^{(1)} = \sigma(W_1 \bar{x}_k + b_1) \in (0, 1)^{p \times p}, \quad G_k^{(2)} = \sigma(W_2 \bar{x}_k + b_2) \in (0, 1)^{q \times q}
$$

where $\bar{x}_k$ is the chunk-level summary (e.g., mean of chunk $k$'s hidden states), $W_1 \in \mathbb{R}^{(p^2) \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{(q^2) \times d_{\text{model}}}$, and $\sigma$ is the sigmoid function.

**Key Variables:**
- $S_k \in \mathbb{R}^{d_k \times d_v}$ — inter-chunk state ($d_k = pq$)
- $\mathcal{S}_k \in \mathbb{R}^{q \times p \times d_v}$ — reshaped state
- $G_k^{(1)} \in \mathbb{R}^{p \times p}$ — coarse-grained gate factor
- $G_k^{(2)} \in \mathbb{R}^{q \times q}$ — fine-grained gate factor
- $p, q$ — factorization dimensions, $pq = d_k$, typically $p = q = \sqrt{d_k}$

### Complexity Comparison

| Operation | Diagonal GLA | Kronecker-GLA | Dense (DeltaNet) |
|-----------|-------------|---------------|-----------------|
| State update per chunk | $O(d_k d_v)$ | $O((p^2 + q^2) d_v)$ | $O(d_k^2 d_v)$ |
| Gate parameters | $d_k$ | $p^2 + q^2$ | $d_k^2$ |
| Scan operator (A-component) | $O(d_k)$ | $O(p^3 + q^3)$ | $O(d_k^3)$ |
| State memory per chunk | $d_k d_v$ | $d_k d_v$ (same) | $d_k d_v$ (same) |

For $d_k = 64, p = q = 8$:
- Diagonal: $64 d_v$ per chunk
- **Kronecker: $128 d_v$ per chunk** ($2\times$ diagonal)
- Dense: $4096 d_v$ per chunk ($64\times$ diagonal)

The Kronecker approach adds only $2\times$ cost over diagonal while providing structured channel coupling that dense transitions offer at $64\times$ cost.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA with Kronecker state transitions |
| Layers | $L = 12$ |
| Hidden dim | $d_{\text{model}} = 768$ |
| Head dim | $d_k = d_v = 64$ ($p = q = 8$) |
| Heads | $h = 12$ |
| Chunk size | $C = 64$–$128$ |
| Kronecker rank | $\hat{R} = 1$ (single Kronecker product) |

### Baseline
1. **Standard GLA** (diagonal gates): $O(N_c d_k d_v)$ inter-chunk cost
2. **DeltaNet** (Householder transitions): $O(N_c d_k^2 d_v)$ inter-chunk cost
3. **Kimi Linear DPLR** (diagonal + rank-1): $O(N_c d_k d_v + N_c d_v)$ inter-chunk cost

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | Within 1% of DeltaNet quality | WikiText-103, SlimPajama |
| Inter-chunk scan speedup | $1.3$–$2\times$ vs. DeltaNet | Profiled scan kernel time |
| End-to-end throughput | $\geq 1.1\times$ vs. DeltaNet | Tokens/sec on A100 |
| Memory | $\leq 1.05\times$ diagonal GLA | Peak GPU memory |
| Recall quality | $> 80\%$ on MQAR | Multi-query associative recall benchmark |

### Estimated Compute

**Small**: 4–8 A100 GPU-hours for MVE at 125M scale.
**Medium**: 32–64 A100 GPU-hours for 350M–1.3B comparison.

## Expected Outcome

**If hypothesis is correct:**
- Kronecker-GLA matches DeltaNet quality (within 1% perplexity) at $2$–$4\times$ lower inter-chunk scan cost
- The structured channel coupling enables associative recall comparable to dense transitions ($> 80\%$ on MQAR)
- End-to-end training is $1.1$–$1.2\times$ faster than DeltaNet due to cheaper inter-chunk scan

**If hypothesis is wrong:**
- **Scenario A**: Kronecker coupling is too weak — the $p \times p$ and $q \times q$ factors can't represent the token interactions that dense Householder transitions enable. This would mean the expressivity bottleneck in GLA is specifically in the dense cross-channel interactions, ruling out all low-parameter structured approaches.
- **Scenario B**: The inter-chunk scan is not the bottleneck — the intra-chunk $O(C^2 d)$ computation dominates, making the scan optimization irrelevant. This would redirect effort toward intra-chunk improvements (larger $C$ via TFLA, better feature maps, etc.).

## Minimum Viable Experiment

### Setup
- **Model**: Tiny GLA model with Kronecker transitions (2 layers, $d = 64$, $d_k = d_v = 16$, $p = q = 4$, 4 heads, ~200K params)
- **Task**: Multi-query associative recall (MQAR) — requires the model to store and retrieve multiple key-value pairs, testing state transition quality
- **Data**: 10K synthetic sequences, length 256, 8 key-value pairs, vocabulary 128
- **Compute**: Single GPU, $< 5$ minutes

### Success Criteria
- Kronecker-GLA ($p = q = 4$) achieves $> 75\%$ MQAR accuracy, significantly exceeding diagonal-GLA ($< 50\%$ expected for this configuration)
- Kronecker-GLA performs within 10% of dense-transition GLA on MQAR

### Failure Criteria
- Kronecker-GLA performs $\leq 5\%$ better than diagonal-GLA on MQAR → the Kronecker coupling doesn't help with recall
- Training diverges → the Kronecker gate parameterization has stability issues

### Why This Test Is Sufficient
- MQAR specifically tests the quality of cross-channel state interactions during state propagation across chunk boundaries
- If Kronecker coupling helps on MQAR at tiny scale, the mechanism works — scaling adds capacity, not capability
- The inter-chunk scan cost savings are architecture-independent and don't need scale validation

## Theoretical Analysis

### Why Kronecker coupling helps expressivity

Consider a 2-head GLA with $d_k = 4$ and $p = q = 2$. The diagonal gate $\text{diag}(\gamma_1, \gamma_2, \gamma_3, \gamma_4)$ treats all 4 state dimensions independently — there's no way to create interference between state channels.

The Kronecker gate $G^{(1)} \otimes G^{(2)}$ with:
$$
G^{(1)} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad G^{(2)} = \begin{pmatrix} e & f \\ g & h \end{pmatrix}
$$

produces the full transition:
$$
G = \begin{pmatrix} ae & af & be & bf \\ ag & ah & bg & bh \\ ce & cf & de & df \\ cg & ch & dg & dh \end{pmatrix}
$$

This has **rank $\leq 4$** (equal to $pq$) but provides structured cross-channel coupling via the product structure. The number of free parameters is only $p^2 + q^2 = 8$ instead of $d_k^2 = 16$, but the transition is full-rank and can represent rotations, reflections, and selective mixing patterns.

### Scan complexity comparison

For parallel scan over $N_c$ chunks:

| Transition type | Per-step A-mult | Scan depth | Total work |
|-----------------|----------------|------------|------------|
| Diagonal | $O(d_k)$ | $O(\log N_c)$ | $O(d_k N_c)$ |
| **Kronecker** | $O(p^3 + q^3)$ | $O(\log N_c)$ | $O((p^3 + q^3) N_c)$ |
| Dense | $O(d_k^3)$ | $O(\log N_c)$ | $O(d_k^3 N_c)$ |
| DPLR (rank-1) | $O(d_k^2)$ | $O(\log N_c)$ | $O(d_k^2 N_c)$ |

For $d_k = 64, p = q = 8$: Kronecker costs $O(1024)$ per step vs. dense $O(262144)$ — a **$256\times$ reduction**.

## Risks & Limitations

1. **Expressivity ceiling**: Kronecker-factored matrices have $p^2 + q^2$ free parameters, which may not capture the complex token-interaction patterns that dense Householder transitions provide. The Kronecker structure imposes a specific pattern of channel coupling that may be too rigid.

2. **Factorization choice**: $p = q = \sqrt{d_k}$ is the natural balanced choice, but other factorizations ($p = d_k/4, q = 4$) may be better. This introduces a hyperparameter.

3. **Gate stability**: Sigmoid-gated $p \times p$ matrices have spectral radius $< p$ (since all entries $\in (0, 1)$). For $p = 8$, $\rho(G^{(1)}) < 8$ — this may cause gradient explosion during long scans. Mitigation: normalize $G^{(1)}, G^{(2)}$ to have spectral radius $< 1$ via division by their Frobenius norm.

4. **Interaction with intra-chunk computation**: The Kronecker structure is only for inter-chunk transitions. The intra-chunk attention $Q K^\top V$ is unchanged. If intra-chunk computation dominates (large $C$), the scan optimization may not matter.

5. **Not directly comparable to DeltaNet**: DeltaNet's Householder transitions $I - \beta k k^\top$ have a specific algebraic structure (near-orthogonal, rank-1 update) that Kronecker products don't replicate. The comparison is more like "structured-dense vs. structured-sparse" rather than apples-to-apples.

## Follow-up Experiments

1. **Rank-$\hat{R}$ Kronecker**: Test $G = \sum_{r=1}^{\hat{R}} G^{(1)}_r \otimes G^{(2)}_r$ with $\hat{R} = 2, 4$. More expressive but higher cost.

2. **Asymmetric factorization**: Test $p \neq q$ (e.g., $p = 16, q = 4$ for $d_k = 64$). This creates an asymmetric multi-resolution structure.

3. **Kronecker + diagonal**: $G = (G^{(1)} \otimes G^{(2)}) + \text{diag}(\gamma)$ — a "Kronecker-plus-diagonal" transition that generalizes both approaches. Scan requires maintaining both components.

4. **Kronecker + TFLA**: Apply TFLA tiling (trick 158) for larger chunk sizes. With cheaper inter-chunk scans from Kronecker, the optimal chunk size may shift upward.

5. **Scale to 1.3B**: If MVE succeeds, train 1.3B-parameter Kronecker-GLA on SlimPajama and compare with GLA, DeltaNet, and Mamba-2.

6. **Stability analysis**: Study the eigenvalue distribution of the Kronecker transition during training. Compare with Cayley-parameterized orthogonal transitions (trick 022) for stability.

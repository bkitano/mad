---
status: completed
priority: high
created: 2026-02-15
based_on: nystrom-landmark-attention, chunkwise-parallel-scan, semiseparable-block-decomposition, schur-complement-block-inversion, linear-attention-approximation, favor-plus-positive-orthogonal-random-features, io-aware-tiling
experiment_number: 025
results_file: 025_results.md
---

# Nyström Landmark Compression for Chunkwise SSM Inter-Chunk State Transfer

## Hypothesis

In chunkwise-parallel SSM training (Mamba-2/SSD), the inter-chunk state transfer involves materializing and propagating dense $n \times n$ state-transition products across chunk boundaries, which becomes the memory and compute bottleneck for large state dimensions. By applying **Nyström landmark compression** to the accumulated inter-chunk transition matrices — selecting $m \ll n$ "landmark" state dimensions via learned projections and reconstructing the full $n \times n$ transfer via the Nyström formula $\hat{T} = \tilde{F} \tilde{A}^+ \tilde{B}$ — we can reduce inter-chunk communication from $O(n^2)$ to $O(nm)$ while preserving the essential low-rank structure that empirically dominates SSM state transitions.

## Background

Modern SSM architectures like Mamba-2 use the SSD (State Space Duality) framework for efficient training: sequences are divided into chunks of size $C$, where intra-chunk computation uses quadratic attention ($O(C^2 d)$), and inter-chunk state propagation uses the linear recurrence $h_{t+C} = T_{t:t+C} h_t + b_{t:t+C}$, where $T_{t:t+C} = \prod_{s=t}^{t+C-1} A_s$ is the cumulative state-transition product over the chunk.

**The bottleneck**: For input-dependent (selective) SSMs, each $A_t$ is a dense or structured $n \times n$ matrix, and the cumulative product $T_{t:t+C}$ is fully dense. Computing, storing, and applying these inter-chunk transition matrices costs $O(n^2)$ memory per chunk boundary and $O(n^3)$ for the matrix products (or $O(n^2 C)$ if computed via scan). When $n$ is large (e.g., $n = 64$–$256$ in multi-head settings), this dominates the overall cost.

**Key observation**: Empirically, SSM state-transition products $T_{t:t+C}$ are **approximately low-rank**. This is because:
1. Selective SSMs use diagonal-dominant $A_t$ matrices (Mamba: diagonal; DeltaNet: identity + rank-1), so products of many such matrices have rapidly decaying singular values
2. The contractive nature of well-trained SSMs ($\|A_t\|_2 \leq 1$) means singular values of the product decay exponentially
3. DPLR-structured SSMs (S4, S5) have transition matrices with displacement rank $\leq 2r$, and products of displacement-rank matrices grow rank slowly

**The Nyström connection**: The Nyström method reconstructs a low-rank approximation of a matrix from a small sampled submatrix and its interactions with all rows/columns. The approximation error equals the Schur complement of the sampled block — exactly zero when the matrix has rank $\leq m$. For approximately low-rank transition matrices, this gives a controlled approximation with error proportional to the $(m+1)$-th singular value.

**Gap filled**:
- Proposal 002 (SSD-DeltaNet) focuses on making the intra-chunk computation efficient via WY representation but doesn't address inter-chunk compression
- Proposal 021 (HSS attention compression) uses HSS structure for attention matrices, not for state-transition products
- Proposal 018 (Hutchinson adaptive rank) uses stochastic trace estimation to *detect* low rank but doesn't exploit it for compression
- No existing proposal applies Nyström approximation to SSM inter-chunk state transfer

## Mathematical Formulation

**Standard Chunkwise SSM (Mamba-2/SSD):**

Given chunk boundaries at positions $0, C, 2C, \ldots$, the inter-chunk recurrence is:

$$
h_{(k+1)C} = T_k \, h_{kC} + b_k
$$

where the cumulative transition matrix over chunk $k$ is:

$$
T_k = \prod_{t=kC}^{(k+1)C-1} A_t \in \mathbb{R}^{n \times n}
$$

and $b_k = \sum_{t=kC}^{(k+1)C-1} \left(\prod_{s=t+1}^{(k+1)C-1} A_s\right) B_t x_t$ is the accumulated input.

**Nyström Compression of $T_k$:**

Select $m$ landmark indices $\mathcal{L} = \{l_1, \ldots, l_m\} \subset \{1, \ldots, n\}$ (either fixed or learned). Partition $T_k$ as:

$$
T_k = \begin{bmatrix} T_{k,\mathcal{L}\mathcal{L}} & T_{k,\mathcal{L}\bar{\mathcal{L}}} \\ T_{k,\bar{\mathcal{L}}\mathcal{L}} & T_{k,\bar{\mathcal{L}}\bar{\mathcal{L}}} \end{bmatrix}
$$

The Nyström approximation:

$$
\hat{T}_k = \begin{bmatrix} T_{k,\mathcal{L}\mathcal{L}} \\ T_{k,\bar{\mathcal{L}}\mathcal{L}} \end{bmatrix} T_{k,\mathcal{L}\mathcal{L}}^{+} \begin{bmatrix} T_{k,\mathcal{L}\mathcal{L}} & T_{k,\mathcal{L}\bar{\mathcal{L}}} \end{bmatrix}
$$

**In factored form (avoiding full $n \times n$ materialization):**

Define projection matrices:

$$
R_k = T_k[:, \mathcal{L}] \in \mathbb{R}^{n \times m}, \quad C_k = T_k[\mathcal{L}, :] \in \mathbb{R}^{m \times n}, \quad W_k = T_k[\mathcal{L}, \mathcal{L}] \in \mathbb{R}^{m \times m}
$$

Then:

$$
\hat{T}_k = R_k \, W_k^{+} \, C_k
$$

**Efficient inter-chunk propagation:**

Instead of propagating the full $n$-dimensional state, propagate in the compressed form:

$$
h_{(k+1)C} = R_k \, W_k^{+} \, (C_k \, h_{kC}) + b_k
$$

This requires:
1. $C_k \, h_{kC}$: $O(mn)$ — project state to $m$-dimensional landmark space
2. $W_k^{+} \, (\cdot)$: $O(m^2)$ — apply pseudoinverse in landmark space
3. $R_k \, (\cdot)$: $O(mn)$ — project back to full state space

**Total per chunk boundary**: $O(mn)$ instead of $O(n^2)$.

**Learned Landmark Selection:**

Instead of fixed indices, learn a projection $P \in \mathbb{R}^{m \times n}$ that extracts the most informative state dimensions:

$$
C_k = P \, T_k, \quad R_k = T_k \, P^\top, \quad W_k = P \, T_k \, P^\top
$$

The projection $P$ can be parameterized as:
- **Segment-means** (à la Nyströmformer): average $n/m$ consecutive state dimensions
- **Learned linear projection**: $P = \text{softmax}(W_P / \tau)$ with temperature annealing
- **Random positive features** (FAVOR+): $P_{ij} = \frac{1}{\sqrt{m}} \exp(\omega_i^\top e_j - \|\omega_i\|^2/2)$ where $\omega_i$ are orthogonal random vectors

**Iterative Pseudoinverse (from Nyströmformer):**

Avoid SVD for $W_k^+$ using the third-order iterative method:

$$
Z_{j+1} = \frac{1}{4} Z_j (13I - W_k Z_j(15I - W_k Z_j(7I - W_k Z_j)))
$$

with $Z_0 = W_k^\top / (\|W_k\|_1 \|W_k\|_\infty)$. Converges in 6 iterations for well-conditioned $W_k$.

**Error Bound (Proposition):**

Let $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n$ be the singular values of $T_k$. Then:

$$
\|T_k - \hat{T}_k\|_F \leq \left(\sum_{i=m+1}^{n} \sigma_i^2\right)^{1/2} + O\left(\frac{\sigma_{m+1}}{\sigma_m}\right)
$$

The first term is the optimal rank-$m$ approximation error (Eckart-Young), and the second term is the Nyström overhead from sampling (vanishes when the spectrum has a clear gap at rank $m$).

**For contractive SSMs** ($\|A_t\|_2 \leq 1 - \epsilon$ with $\epsilon > 0$):

$$
\sigma_i(T_k) \leq (1 - \epsilon)^C \cdot \sigma_i(T_k / \|T_k\|_2)
$$

So the product's singular values decay exponentially with chunk size $C$, making the low-rank approximation increasingly accurate for larger chunks.

**Key Variables:**
- $T_k \in \mathbb{R}^{n \times n}$ — cumulative state-transition product over chunk $k$
- $n$ — state dimension
- $m$ — number of Nyström landmarks ($m \ll n$)
- $\mathcal{L}$ — landmark index set
- $R_k \in \mathbb{R}^{n \times m}$, $C_k \in \mathbb{R}^{m \times n}$ — row/column factors
- $W_k \in \mathbb{R}^{m \times m}$ — landmark kernel matrix
- $P \in \mathbb{R}^{m \times n}$ — learned projection matrix
- $C$ — chunk size

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Mamba-2 / SSD with Nyström inter-chunk compression |
| Layers | $L = 6$–$12$ |
| Hidden dim | $d = 256$–$512$ |
| State dim | $n = 64$–$128$ per head |
| Landmarks | $m = 8$–$16$ (compression ratio $4\times$–$8\times$) |
| Chunk size | $C = 64$–$256$ |
| Heads | 8 |
| Pseudoinverse | Iterative (6 iterations) |
| Landmark selection | Learned linear projection |

### Baseline

1. **Mamba-2 (full inter-chunk)**: Standard SSD with $O(n^2)$ inter-chunk state transfer. Complexity: $O(TC^2d + (T/C)n^2)$ for training
2. **Mamba-2 (diagonal only)**: Drop the low-rank component, use only diagonal $A_t$. Complexity: $O(TC^2d + (T/C)n)$ but loses expressivity
3. **Random projection baseline**: Replace Nyström with random Gaussian projection $P$ (not learned). Same $O(mn)$ cost but no adaptation
4. **Truncated SVD baseline**: Compute exact rank-$m$ SVD of $T_k$ at each chunk boundary. Complexity: $O(n^2 m)$ — more expensive but optimal quality

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity gap vs full | $< 0.5$ PPL | WikiText-103 validation |
| Inter-chunk memory | $< 0.25 \times$ baseline | Peak GPU memory for state transfer |
| Training throughput | $> 1.2 \times$ Mamba-2 | Tokens/sec on A100 at $n = 128$ |
| MQAR accuracy | $\geq 95\%$ of full | Multi-query associative recall |
| Approximation error | $\|T - \hat{T}\|_F / \|T\|_F < 0.1$ | Measured at convergence |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (synthetic + MQAR): ~20 GPU-hours on A100
**Phase 2** (language modeling, 350M params): ~80 GPU-hours on A100
**Total**: ~100 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- Nyström-compressed Mamba-2 with $m = n/4$ landmarks matches full Mamba-2 perplexity within $0.3$ PPL on WikiText-103
- Training throughput improves by $1.3\times$–$1.5\times$ for $n = 128$ (inter-chunk cost drops from $O(n^2)$ to $O(mn)$)
- The learned projection $P$ discovers interpretable structure: landmarks correspond to the most "information-rich" state dimensions (highest variance activations)
- MQAR accuracy is fully preserved because the associative recall signal lives in a low-rank subspace of the state
- The approximation error $\|T_k - \hat{T}_k\|_F$ decreases during training as the model learns to concentrate state information in the top-$m$ singular values (co-adaptation between model and compression)

**If hypothesis is wrong:**
- If perplexity degrades significantly ($> 1.0$ PPL gap): the state-transition products are not low-rank in practice, and the model relies on high-rank state interactions. This would imply that selective SSMs fundamentally need dense state coupling, validating the design choice of small $n$ in Mamba-2
- If random projection matches learned: the compression is not data-dependent, and any projection captures the low-rank structure. This would simplify the method (no learned $P$) but still validate the compression idea
- If truncated SVD significantly outperforms Nyström: the sampling-based approximation introduces harmful bias. This would motivate using more expensive but exact low-rank factorizations

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer Mamba-2 style model, $d = 64$, $n = 32$ (state dim), $m = 8$ landmarks, $C = 32$ chunk size, ~80K params
- **Task**: Synthetic copying task with delay — copy a sequence of tokens after a gap of $G$ positions, requiring state to maintain information across chunk boundaries
- **Data**: 5K sequences of length 256 (8 chunks of size 32), with copy delay $G = 64$ (spanning 2 chunk boundaries)
- **Compute**: Single GPU, < 10 minutes

### Success Criteria
- Nyström-compressed model ($m = 8$, compression $4\times$) achieves $> 90\%$ copy accuracy at delay $G = 64$
- Full model ($m = n = 32$, no compression) achieves $> 95\%$ copy accuracy
- Gap is $< 5\%$ (the lost information is in negligible singular values)
- Memory for inter-chunk state transfer is verified at $O(mn) = O(256)$ vs $O(n^2) = O(1024)$

### Failure Criteria
- If compressed model achieves $< 70\%$ copy accuracy: the state information required for copying is spread across all $n$ dimensions and cannot be compressed to $m$. The mechanism is broken for this fundamental task.
- If there's no memory or speed improvement at this scale: the overhead of the Nyström computation (pseudoinverse, projections) exceeds the savings from compression at small $n$. Would need larger $n$ to see benefit.

### Why This Test Is Sufficient
- The copying task directly exercises inter-chunk state transfer: the model must carry token information from input chunk to output chunk via the state recurrence. If Nyström compression loses copying ability, it's destroying essential state information.
- The delay $G = 64$ spans exactly 2 chunk boundaries, testing that the compressed state transfer preserves information across multiple hops.
- At full scale ($n = 128$), the compression ratio is $8\times$, making the memory and compute savings much more significant. If the mechanism works at $4\times$ compression, scaling to $8\times$ only requires the rank assumption to hold more strongly — which it does because larger chunks produce more contractive products.

## Theoretical Analysis

Complexity comparison (for $T$-length sequence, $Q$ chunks of size $C = T/Q$):

| Operation | Mamba-2 (full) | Nyström-compressed |
|-----------|---------------|-------------------|
| Intra-chunk (quadratic) | $O(TC^2 d)$ | $O(TC^2 d)$ (unchanged) |
| Inter-chunk transition product | $O(TCn)$ (diagonal scan) | $O(TCn)$ (unchanged) |
| Inter-chunk state propagation | $O(Qn^2)$ | $O(Qmn)$ |
| Inter-chunk memory | $O(Qn^2)$ | $O(Qmn + Qm^2)$ |
| Pseudoinverse per chunk | — | $O(Qm^3)$ (6 iterations) |
| **Total inter-chunk** | $O(Qn^2)$ | $O(Qmn + Qm^3)$ |

Crossover point: Nyström is beneficial when $mn + m^3 < n^2$, i.e., $m < n - m^3/n$. For typical $m = n/4$: $mn = n^2/4$ and $m^3 = n^3/64$, so the savings are $\approx 3n^2/4$ — a $4\times$ improvement in inter-chunk cost.

**When inter-chunk dominates**: For large $n$ and small $C$ (many chunks), inter-chunk cost $Qn^2 = (T/C)n^2$ can dominate intra-chunk cost $TC^2 d$. This happens when $n^2/C > C^2 d$, i.e., $n > C \sqrt{Cd}$. For $C = 64$, $d = 512$: $n > 64\sqrt{32768} \approx 11585$ — so inter-chunk dominates only for very large $n$. However, even when it doesn't dominate, reducing it by $4\times$ provides meaningful end-to-end speedup.

## Risks & Limitations

1. **Rank assumption may fail**: If selective SSMs learn state-transition matrices that are *not* low-rank (e.g., the input-dependent gating creates rapidly rotating, full-rank transitions), the Nyström approximation will be poor. Mitigation: add a nuclear norm regularizer $\lambda \|T_k\|_*$ to encourage low-rank transitions during training.

2. **Pseudoinverse instability**: If $W_k = T_k[\mathcal{L}, \mathcal{L}]$ is ill-conditioned, the iterative pseudoinverse diverges. Mitigation: add ridge regularization $W_k + \delta I$ or use the Moore-Penrose formula with SVD truncation for the small $m \times m$ matrix.

3. **Gradient flow through Nyström**: The approximation $\hat{T}_k = R_k W_k^+ C_k$ introduces a non-standard computational graph for backpropagation through the inter-chunk connection. The gradient through $W_k^+$ involves $\partial W^+ / \partial W = -(W^+)^\top \otimes W^+ + \ldots$, which can amplify small perturbations. Mitigation: use the straight-through estimator for the pseudoinverse (compute gradient as if $W^+ = W^{-1}$).

4. **Overhead at small $n$**: For Mamba-2's default $n = 16$ per head, the Nyström overhead (projection + pseudoinverse) may exceed the savings from compression. The benefit appears primarily at $n \geq 64$.

5. **Non-composability with DPLR**: If $A_t$ is already DPLR-structured (diagonal + low-rank), the product $T_k$ inherits structured low-rank properties that can be exploited more directly than through Nyström. The Nyström approach is most useful for unstructured selective SSMs (Mamba-style diagonal with varying parameters).

## Follow-up Experiments

1. **Adaptive landmark count**: Use Hutchinson trace estimation (Proposal 018) to estimate the effective rank of $T_k$ on-the-fly, and choose $m$ adaptively per chunk. High-rank chunks get more landmarks; low-rank chunks get fewer.

2. **FAVOR+ features for landmarks**: Instead of learned linear projections, use positive orthogonal random features from FAVOR+ as the landmark projection. This would give an unbiased estimator with provably low variance and no learned parameters.

3. **Hierarchical Nyström (HSS connection)**: Apply the Nyström compression recursively — compress chunk transitions to rank $m$, then compress groups of compressed transitions further. This creates an HSS-like hierarchy (connecting to Proposal 005 and 021).

4. **Nyström for DeltaNet inter-chunk**: Apply to DeltaNet/DeltaProduct (Proposal 001) where the transition matrices are $I + \beta_t v_t k_t^\top$ (rank-1 updates to identity). Products of $C$ such matrices have rank at most $C$, making Nyström with $m \ll C$ a natural fit.

5. **Joint compression of transition + input**: Compress not just $T_k$ but also the accumulated input $b_k$ using the same landmark basis, further reducing inter-chunk communication.

## Human Review

(To be filled by reviewer)

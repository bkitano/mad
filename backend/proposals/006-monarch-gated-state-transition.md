---
status: ongoing
priority: high
created: 2026-02-15
based_on: monarch-matrix-factorization, input-dependent-gating, chunkwise-parallel-scan, semiseparable-block-decomposition, two-four-structured-sparsity
experiment_number: 006
experiment_log: experiment-log-006.md
results_file: 006_results.md
---

# Monarch-Gated State Transition SSM

## Hypothesis

Replacing diagonal state transitions in SSMs with **input-dependent Monarch-factored transitions** achieves expressivity comparable to full dense transitions ($O(n^2)$) while maintaining near-diagonal computational cost ($O(n\sqrt{n})$ per step), and the BMM (batch matrix multiply) structure enables 2–4$\times$ higher GPU utilization than custom scan kernels.

## Background

Modern efficient SSMs (Mamba, Mamba-2, GLA) use **diagonal** state transitions $A_t = \text{diag}(\alpha_t)$, which are $O(n)$ per step but fundamentally limited in expressivity — they cannot perform any coordinate mixing. DeltaNet uses rank-1 updates ($I + kv^\top$) via Householder reflections, gaining associative memory but with limited mixing per step. Full dense transitions ($A_t \in \mathbb{R}^{n \times n}$) are $O(n^2)$ per step and impractical.

**Monarch matrices fill exactly this gap.** A single Monarch factor $M = P_b^\top L P_b R$ has:
- $O(n\sqrt{n})$ parameters and computation (sub-quadratic)
- Direct BMM implementation achieving near-peak GPU throughput
- Proven ability to represent DFT, DCT, Hadamard, and other fast transforms
- Built-in coordinate mixing via the permutation $P_b$

No existing proposal explores Monarch-structured transitions in an SSM with input-dependent gating. The closest is proposal 003 (DPLR + column-sparse), which adds a single permutation to DPLR but doesn't leverage the block-diagonal structure of Monarch that enables BMM.

**Key insight**: The Monarch factorization decomposes coordinate mixing into two levels — local mixing within blocks (via $L, R$) and global mixing between blocks (via $P_b$). This is analogous to how chunkwise parallel scan decomposes temporal processing into local (intra-chunk) and global (inter-chunk) — suggesting a natural compositional structure.

## Mathematical Formulation

**Standard Diagonal SSM (Mamba-2):**

$$
h_t = \text{diag}(\alpha_t) \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t
$$

where $\alpha_t = \sigma(W_\alpha x_t) \in (0, 1)^n$ are input-dependent decay gates.

**Monarch-Gated SSM (Proposed):**

$$
h_t = M(x_t) \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t
$$

where the Monarch transition is:

$$
M(x_t) = P_b^\top \cdot L(x_t) \cdot P_b \cdot R(x_t)
$$

with input-dependent block-diagonal factors:

$$
L(x_t) = \text{diag}\left(\alpha_{1}(x_t) \cdot L_1, \ldots, \alpha_{\sqrt{n}}(x_t) \cdot L_{\sqrt{n}}\right)
$$

$$
R(x_t) = \text{diag}\left(\beta_{1}(x_t) \cdot R_1, \ldots, \beta_{\sqrt{n}}(x_t) \cdot R_{\sqrt{n}}\right)
$$

where:
- $L_i, R_i \in \mathbb{R}^{\sqrt{n} \times \sqrt{n}}$ are **fixed** (LTI) orthogonal block matrices
- $\alpha_i(x_t), \beta_i(x_t) \in (0, 1)$ are **input-dependent** scalar gates per block
- $P_b$ is the fixed stride permutation (reshape transpose)

**Key Variables:**
- $h_t \in \mathbb{R}^n$ — hidden state
- $x_t \in \mathbb{R}^d$ — input at time $t$
- $n$ — state dimension (assumed to be a perfect square, e.g., $n = 256 = 16^2$)
- $\sqrt{n}$ — block count and block size
- $L_i, R_i \in \mathbb{R}^{\sqrt{n} \times \sqrt{n}}$ — orthogonal blocks (parameterized via Cayley)
- $\alpha_i, \beta_i \in (0, 1)$ — per-block input-dependent decay gates

**Gating Mechanism:**

$$
[\alpha_1, \ldots, \alpha_{\sqrt{n}}, \beta_1, \ldots, \beta_{\sqrt{n}}] = \sigma(W_g x_t + b_g) \in (0, 1)^{2\sqrt{n}}
$$

where $W_g \in \mathbb{R}^{2\sqrt{n} \times d}$ — only $O(d\sqrt{n})$ additional parameters.

**Stability Guarantee:**

Since each $L_i, R_i$ is orthogonal ($\|L_i\| = \|R_i\| = 1$) and gates $\alpha_i, \beta_i \in (0, 1)$:

$$
\|M(x_t)\| \leq \max_i \alpha_i \cdot \max_j \beta_j < 1
$$

The state is **contractive by construction** — no eigenvalue clipping or projection needed.

**Chunkwise Parallel Form:**

Within a chunk of size $C$, the cumulative transition is:

$$
M_{s:t} = \prod_{i=s+1}^{t} M(x_i) = \prod_{i=s+1}^{t} P_b^\top L(x_i) P_b R(x_i)
$$

Since each $M(x_i)$ is a Monarch matrix, the product of $C$ Monarch factors is a degree-$C$ Monarch product, which can be computed via $C$ sequential BMM operations (each $O(n\sqrt{n})$) within a chunk, and the inter-chunk scan operates on the resulting $n \times n$ cumulative matrices.

**SSD-Compatible Block Form:**

For intra-chunk computation with block size $Q$, the output submatrix for positions $i, j$ within a chunk is:

$$
\tilde{M}_{ij} = C_i^\top \left(\prod_{k=j+1}^{i} M(x_k)\right) B_j
$$

Each factor in the product is a Monarch matrix, so the product is computed via a sequence of BMM calls.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Monarch-Gated SSM |
| Layers | $L = 12$ |
| Hidden dim | $d = 768$ |
| State dim | $n = 256$ ($16 \times 16$ blocks) |
| Block size | $b = \sqrt{n} = 16$ |
| Chunk size | $C = 256$ |
| Gate params | $2\sqrt{n} = 32$ per head |
| Orthogonal blocks | Cayley-parameterized |

### Baseline

1. **Mamba-2** (diagonal gated SSM): $O(Tn)$ per layer, SSD algorithm
2. **Gated DeltaNet** (rank-1 Householder update): $O(Tnd)$ per layer, chunkwise parallel
3. **Dense-transition SSM** (upper bound): $O(Tn^2)$ per layer

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $> 0.7\times$ Mamba-2 tokens/sec | Timed forward pass, batch 16, seq 2048 |
| Memory | $< 1.3\times$ Mamba-2 | Peak GPU memory during training |
| MQAR accuracy | $> 95\%$ at 4 KV pairs | Multi-Query Associative Recall benchmark |
| State tracking | $> 90\%$ on $S_5$ permutation group | 5-element permutation composition |
| Perplexity | $\leq 1.05\times$ Mamba-2 | WikiText-103, 380M param model |

### Estimated Compute

**MVE**: ~10 minutes on single A100 (~$0.50)
**Small-scale**: 4 GPU-hours on A100 (~$16)
**Full-scale**: 32 GPU-hours on A100 (~$130)

## Expected Outcome

**If hypothesis is correct:**
- Monarch-Gated SSM achieves $> 90\%$ accuracy on $S_5$ permutation group tracking where Mamba-2 (diagonal) achieves $< 60\%$
- MQAR recall improves by $10$–$20\%$ over Mamba-2 at $\geq 4$ KV pairs due to coordinate mixing
- Throughput overhead is $< 30\%$ vs. Mamba-2 (BMM is highly optimized)
- Perplexity matches or beats Mamba-2 on WikiText-103 at matched parameter count

**If hypothesis is wrong:**
- If Monarch transitions don't improve state tracking: the $O(\sqrt{n})$-size blocks may be too small for meaningful coordinate mixing, suggesting the need for Monarch$^2$ (stacked) transitions at higher cost
- If throughput is much worse: the sequential dependence of $C$ Monarch multiplications within a chunk may dominate, suggesting a hybrid approach where only every $k$-th step uses Monarch and others use diagonal

## Minimum Viable Experiment

### Setup
- **Model**: 2 layers, $d = 64$, $n = 64$ ($8 \times 8$ blocks), ~120K params
- **Task**: $S_5$ permutation group composition — given a sequence of generators of $S_5$, predict the resulting permutation
- **Data**: 10K synthetic sequences of length 10–50, each a product of random transpositions from $S_5$
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- Monarch-Gated SSM achieves $> 85\%$ accuracy on $S_5$ composition with sequence length 20
- Diagonal SSM baseline achieves $< 50\%$ accuracy on the same task
- Forward pass of Monarch-Gated SSM is $< 3\times$ slower than diagonal SSM

### Failure Criteria
- If Monarch-Gated SSM cannot beat diagonal SSM on $S_5$ composition at any sequence length, the coordinate mixing from Monarch permutations is insufficient
- If forward pass is $> 5\times$ slower than diagonal, the BMM overhead negates any expressivity gains

### Why This Test Is Sufficient
- Permutation group composition is the canonical test for coordinate mixing ability — diagonal SSMs provably cannot solve it (they operate independently per coordinate)
- Monarch's built-in permutation $P_b$ should enable coordinate routing needed for group composition
- If coordinate mixing works at $n = 64$, scaling to $n = 256$ adds capacity (larger blocks) not qualitatively new capability
- The $S_5$ task directly tests the mechanism ($P_b$ enables coordinate routing) rather than a downstream proxy

## Theoretical Analysis

**Complexity comparison:**

| Operation | Diagonal SSM | Monarch SSM | Dense SSM |
|-----------|-------------|-------------|-----------|
| Forward (per step) | $O(n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Forward (full seq) | $O(Tn)$ | $O(Tn\sqrt{n})$ | $O(Tn^2)$ |
| Parameters (transition) | $O(n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Memory (state) | $O(n)$ | $O(n)$ | $O(n)$ |
| GPU utilization | Low (element-wise) | High (BMM) | High (GEMM) |

**Crossover analysis:**

For state dim $n = 256$: Monarch is $\sqrt{256} = 16\times$ more expensive per step than diagonal but $256 / 16 = 16\times$ cheaper than dense. The BMM implementation achieves ~2$\times$ higher FLOP utilization than element-wise operations, making the effective overhead ~$8\times$ vs. diagonal.

At $n = 256$ with chunk size $C = 256$: intra-chunk Monarch products cost $C \cdot n\sqrt{n} = 256 \cdot 256 \cdot 16 \approx 1M$ FLOPs per chunk (via BMM at ~300 TFLOPS), while inter-chunk scan over $T/C$ states costs $T/C \cdot n^2$ FLOPs for the cumulative transition products.

**Expressivity analysis:**

A single Monarch factor $M = P_b^\top L P_b R$ can represent:
- All diagonal matrices (setting $L = I$ or $R = I$)
- All permutation-conjugated block-diagonal matrices
- DFT, DCT, Hadamard at Monarch$^2$

Critically, Monarch matrices are **closed under multiplication** (product of Monarch matrices can be written as higher-order Monarch), meaning $T$-step cumulative transitions stay within the Monarch class. This is not true for DPLR or column-sparse transitions.

## Risks & Limitations

1. **Sequential Monarch products within chunks**: Unlike diagonal transitions where $\prod A_i = \text{diag}(\prod \alpha_i)$ (trivially parallel), Monarch products require sequential BMM. This may limit chunk-level parallelism.

2. **Fixed permutation $P_b$**: The stride permutation is fixed by Monarch convention. If the task requires different permutation structure, a single Monarch factor may be insufficient. Mitigation: use Monarch$^2$ (two factors) or learn $P_b$ via Sinkhorn (see proposal 007).

3. **Block size constraints**: $n$ must be a perfect square for balanced factorization. Non-square $n$ requires padding.

4. **Interaction with SSD duality**: The semiseparable structure in SSD relies on scalar/diagonal transitions for 1-semiseparable form. Monarch transitions produce $\sqrt{n}$-semiseparable matrices, which may not decompose as cleanly.

5. **Gradient flow through Monarch products**: Backpropagation through $C$ sequential Monarch multiplications may suffer from gradient issues despite the contractive guarantee. Checkpoint every $C/4$ steps to manage.

## Follow-up Experiments

1. **Monarch$^2$ transitions**: Stack two Monarch factors per step for expressivity equivalent to Monarch$^2$ (covers DFT, DCT). Cost doubles to $O(2n\sqrt{n})$ but still sub-quadratic.

2. **2:4 sparsity on Monarch blocks**: Apply 2:4 structured sparsity to the $\sqrt{n} \times \sqrt{n}$ block matrices $L_i, R_i$, reducing per-block cost by $2\times$ via Sparse Tensor Cores. Since blocks are small ($16 \times 16$), 2:4 pruning may tolerate the constraint well.

3. **Learned permutations via Sinkhorn**: Replace fixed $P_b$ with Gumbel-Sinkhorn learned permutation (see proposal 007 for full treatment).

4. **Hybrid Monarch-diagonal layers**: Alternate Monarch-gated layers (for mixing) with cheaper diagonal-gated layers (for per-coordinate processing), similar to how Transformers alternate attention (mixing) with FFN (per-position).

5. **Log-linear Monarch SSM**: Compose the Monarch transition mask with log-linear attention's hierarchical mask for $O(T \log T)$ training with sub-quadratic state transitions.

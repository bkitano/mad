---
status: ongoing
priority: high
created: 2026-02-15
based_on: smooth-ste-continuous-sparse-projection, blockwise-sinkhorn-channel-permutation, two-four-structured-sparsity, input-dependent-gating, chunkwise-parallel-scan, permutation-augmented-structured-sparsity, cayley-contractive-parameterization
experiment_number: 024
experiment_log: experiment-log-024.md
---

# 2:4 Sparse SSM State Transitions via S-STE + Blockwise Sinkhorn

## Hypothesis

Training SSM state transition matrices with 2:4 structured sparsity from scratch, using S-STE's continuous pruning function and blockwise Sinkhorn-learned channel permutations, will achieve expressivity comparable to dense transitions at $1.5$–$2\times$ training speedup and $2\times$ inference speedup, with less than $5\%$ quality degradation on state-tracking benchmarks.

## Background

**Current state of sparse SSMs:** Structured sparsity (2:4, N:M) has been extensively validated for Transformer FFN layers and attention projections, achieving near-lossless compression. However, **no prior work applies N:M sparsity directly to SSM state transition matrices** — the core computational bottleneck that determines SSM expressivity.

This gap exists because SSM transitions have unique constraints that FFN weights don't:
1. **Stability**: The state transition $A$ must have bounded spectral norm ($\|A\| \leq 1$) to prevent divergence — sparsification could violate this
2. **Sequential composition**: The transition is applied recurrently ($h_t = A_t h_{t-1} + Bu$), so approximation error compounds across time steps
3. **Expressivity dependence**: The transition structure directly determines what automata the SSM can simulate — random sparsification destroys group structure

**Why the combination works:** Two tricks solve these challenges:

1. **S-STE (Smooth Straight-Through Estimator)** provides a *continuous* pruning function that eliminates the mask oscillation, incorrect descent, and instability problems of hard-thresholding STE methods. S-STE's soft-thresholding $S_{\text{soft}}$ smoothly transitions between sparsity patterns, making the loss landscape for sparse training as well-behaved as dense training. The frozen scaling factor $\beta$ preserves relative weight magnitudes. **Key: S-STE has only been tested on FFN layers; applying it to recurrent state matrices is completely unexplored.**

2. **Blockwise Sinkhorn channel permutation (PermLLM)** learns to reorder state channels so that salient weights cluster into the same 4-element groups before 2:4 pruning is applied. This recovers expressivity that random sparsification destroys. **Key: PermLLM is a post-training technique; using learned permutations during training from scratch is unexplored.** By combining with S-STE, the permutation can co-evolve with the sparse mask, producing better sparsity patterns than either technique alone.

**The critical insight:** In a dense SSM, the transition matrix $A \in \mathbb{R}^{n \times n}$ has $n^2$ parameters. With 2:4 sparsity, only $n^2/2$ parameters are nonzero. But *which* $n^2/2$ parameters are nonzero matters enormously — a random sparsity pattern destroys the state-mixing structure needed for group computation. Blockwise Sinkhorn permutation optimizes which parameters survive pruning, while S-STE ensures the pruning is smooth enough to train through.

**Connection to existing proposals:** Proposal 010 (Sparse Monarch SSM with PA-DST) applies 2:4 sparsity to Monarch factors, but this is fundamentally different: Monarch factors are already structured (block-diagonal × permutation), so sparsifying them is a secondary optimization. Our proposal sparsifies the *raw transition matrix* with learned permutations, which is a more fundamental architectural change. The S-STE trick (not used in Proposal 010) is critical because it provides the continuous optimization landscape needed for training stability.

## Mathematical Formulation

**Dense SSM State Transition:**

$$
h_t = A(x_t) \cdot h_{t-1} + B \cdot x_t
$$

where $A(x_t) \in \mathbb{R}^{n \times n}$ is input-dependent. Complexity: $O(n^2)$ per step.

**Sparse-Permuted SSM (Proposed):**

$$
h_t = \underbrace{\beta \cdot S_{\text{soft}}(P_B^T \cdot A_{\text{dense}}(x_t) \cdot P_B)}_{\text{2:4 sparse transition}} \cdot h_{t-1} + B \cdot x_t
$$

where:

**Step 1 — Channel Permutation:**

The block-diagonal permutation $P_B = \text{diag}(P_1, P_2, \ldots, P_{N_B})$ reorders state channels to cluster salient weights:

$$
\tilde{A}(x_t) = P_B^T \cdot A_{\text{dense}}(x_t) \cdot P_B
$$

Each block $P_i \in \{0,1\}^{B \times B}$ is learned via Sinkhorn relaxation:

$$
\hat{P}_i = \text{Sinkhorn}(W_P^{(i)} / \tau, L), \quad P_i = \text{Hungarian}(\hat{P}_i) + (\hat{P}_i - \hat{P}_i^{\text{detach}}) \quad \text{(STE)}
$$

**Step 2 — Continuous 2:4 Pruning (S-STE):**

For each row of $\tilde{A}$, partition into groups of 4 and apply soft-thresholding:

$$
(S_{\text{soft}}(\mathbf{a}))_i = \text{sign}(a_i) \cdot \max(|a_i| - |a_{(2)}|, 0)
$$

where $|a_{(2)}|$ is the 2nd-smallest magnitude in the group. Scale by frozen $\beta$:

$$
A_{\text{sparse}}(x_t) = \beta \cdot S_{\text{soft}}(\tilde{A}(x_t))
$$

**Step 3 — Stability Enforcement:**

After sparsification, project onto the contractive set:

$$
A_{\text{stable}}(x_t) = \frac{A_{\text{sparse}}(x_t)}{\max(1, \|A_{\text{sparse}}(x_t)\|_2 / \gamma)}
$$

where $\gamma < 1$ is a spectral norm bound (e.g., $\gamma = 0.99$). For efficiency, use the power iteration estimate of $\|A_{\text{sparse}}\|_2$ with the sparse structure exploited.

**Input-Dependent Dense Transition (before sparsification):**

$$
A_{\text{dense}}(x_t) = W_A + \text{diag}(\sigma(x_t W_g)) \cdot W_A'
$$

where $W_A, W_A' \in \mathbb{R}^{n \times n}$ are dense weight matrices (maintained in dense form during training per S-STE), and $x_t W_g$ provides input-dependent gating.

**Forward Pass (Training):**

$$
A_t = \beta \cdot S_{\text{soft}}(P_B^T (W_A + \text{diag}(\sigma(x_t W_g)) W_A') P_B)
$$

$$
h_t = A_t h_{t-1} + B x_t
$$

**Backward Pass:** Gradients flow through S-STE (identity approximation) and Sinkhorn (soft matrix in backward), updating:
- Dense weights $W_A, W_A'$ — via STE
- Permutation cost matrices $W_P^{(i)}$ — via Sinkhorn + Hungarian STE
- Gating weights $W_g$ — standard backprop

**Inference (Sparse):**

At inference, $P_B$ is hardened to a fixed permutation (integer index array), $S_{\text{soft}}$ becomes hard 2:4 mask, and the sparse matvec uses Sparse Tensor Cores:

$$
A_{\text{infer}} = M_{2:4} \odot (P_B^T W_A P_B), \quad h_t = A_{\text{infer}} h_{t-1} + B x_t
$$

Complexity: $O(n^2 / 2)$ per step via 2:4 spMM on Sparse Tensor Cores.

**Key Variables:**

- $A_{\text{dense}} \in \mathbb{R}^{n \times n}$ — dense weight (maintained during training)
- $P_B \in \{0,1\}^{n \times n}$ — block-diagonal permutation matrix
- $W_P^{(i)} \in \mathbb{R}^{B \times B}$ — learnable cost matrix for block $i$
- $\beta \in \mathbb{R}$ — frozen MSE-optimal scaling factor
- $S_{\text{soft}}$ — continuous soft-thresholding 2:4 projection
- $B$ — block size for permutation ($B = 16$–$32$ for SSM state dims $n = 64$–$256$)
- $N_B = n / B$ — number of permutation blocks

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Sparse-Permuted SSM |
| Layers | $L = 6$ |
| Hidden dim | $d_{\text{model}} = 256$ |
| State dim | $n = 64$ (multi-head: $H = 4$, $n/H = 16$) |
| Sparsity | 2:4 structured |
| Permutation block size | $B = 16$ |
| S-STE scaling | $\beta$ frozen after iteration 1 |
| Sinkhorn iterations | $L_S = 5$ |
| Sinkhorn temperature | $\tau$: linearly decay $1.0 \to 0.1$ over training |
| Parameters | ~8M (dense during training, sparse at inference) |

### Baseline

1. **Dense SSM**: Full $n \times n$ transition, $O(n^2)$ — upper bound on quality
2. **Random 2:4 Sparse SSM**: Hard-threshold STE (SR-STE) with random mask — shows quality of naive sparsification
3. **S-STE-only Sparse SSM**: S-STE without learned permutation — isolates S-STE contribution
4. **PermLLM post-hoc Sparse SSM**: Train dense, then apply PermLLM + Wanda — shows post-training baseline
5. **Diagonal SSM (Mamba-style)**: $O(n)$, already "sparse" by construction — lower bound on mixing

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ state tracking | $> 85\%$ (dense: $> 95\%$) | Synthetic benchmark |
| WikiText-103 perplexity | Within 5% of dense | Language modeling |
| Training throughput | $> 1.3\times$ dense SSM | Tokens/sec (A100, SpTC) |
| Inference throughput | $> 1.8\times$ dense SSM | Tokens/sec (A100, SpTC) |
| Mask flip rate | Converging (not oscillating) | $r_k = \|m_k \oplus m_{k-1}\|_1 / d$ |

### Estimated Compute

**Small–Medium**. Synthetic experiments: < 10 GPU-hours. WikiText-103: < 30 GPU-hours. Ablations (5 baselines × 3 seeds): < 100 GPU-hours total.

## Expected Outcome

**If hypothesis is correct:**

- S-STE + Sinkhorn Sparse SSM achieves $> 85\%$ on $S_5$ tracking (vs $> 95\%$ dense, $< 60\%$ random sparse)
- S-STE alone achieves $\sim 70\%$ — learned permutations add $15$+ percentage points
- Mask flip rate converges to $< 1\%$ by end of training (S-STE solves oscillation)
- $1.8\times$ inference speedup on A100 with 2:4 Sparse Tensor Cores
- Language model perplexity within 5% of dense baseline

**If hypothesis is wrong:**

- **Scenario A**: Error compounds across recurrent steps — sparse transition is too lossy for $T > 100$ → learn that sparsity must be adapted to sequence length, perhaps with sequence-dependent masks
- **Scenario B**: S-STE works but Sinkhorn permutation adds no value → channel ordering doesn't matter for SSM transitions (unlike FFN weights), suggesting SSM weights have more uniform saliency distribution
- **Scenario C**: Training instability from combining S-STE + Sinkhorn + stability projection → too many gradient approximations stacked; simplify to just S-STE

## Minimum Viable Experiment

### Setup

- **Model**: 1-layer Sparse-Permuted SSM, $n = 16$, $d_{\text{model}} = 32$, ~25K params
- **Task**: $S_3$ (symmetric group on 3 elements) state tracking
- **Data**: 5K sequences of length 16
- **Compute**: Single GPU, $< 5$ minutes

### Implementation

```python
def sparse_permuted_ssm_step(h, x, W_A, W_P_blocks, beta, W_b, tau):
    """One step of Sparse-Permuted SSM."""
    n = h.shape[-1]

    # 1. Build block-diagonal permutation via Sinkhorn + Hungarian STE
    P_blocks = []
    for W_P in W_P_blocks:
        P_soft = sinkhorn(W_P / tau, n_iters=5)
        P_hard = hungarian_ste(P_soft)
        P_blocks.append(P_hard)
    P_B = torch.block_diag(*P_blocks)

    # 2. Permute dense transition
    A_permuted = P_B.T @ W_A @ P_B

    # 3. Apply S-STE continuous 2:4 pruning
    A_sparse = beta * soft_threshold_2_4(A_permuted)

    # 4. Stability projection
    s_norm = torch.linalg.matrix_norm(A_sparse, ord=2)
    A_stable = A_sparse / max(1.0, s_norm / 0.99)

    # 5. Recurrence
    h_new = A_stable @ h + x @ W_b
    return h_new
```

### Success Criteria

- Sparse-Permuted SSM achieves $> 90\%$ on $S_3$ state tracking
- Dense SSM achieves $> 98\%$ (sanity check)
- Random 2:4 sparse SSM achieves $< 50\%$ (shows sparsification is non-trivial)
- S-STE-only sparse SSM achieves $65$–$80\%$ (shows permutation helps)
- Mask flip rate decreases monotonically during training

### Failure Criteria

- **Kill if**: Sparse-Permuted SSM achieves $< 50\%$ on $S_3$ → 2:4 sparsity fundamentally destroys state-tracking capability
- **Kill if**: Mask flip rate oscillates without convergence after 500 steps → S-STE doesn't work for recurrent weights
- **Investigate if**: S-STE-only matches Sparse-Permuted → permutation is not needed (simplifies the method)

### Why This Test Is Sufficient

- $S_3$ state tracking requires non-trivial state mixing (non-abelian group). If 2:4 sparsity can preserve this, it preserves the core mechanism
- The test directly measures whether sparsification destroys group computation capability
- If it works at $n = 16$ (very aggressive sparsity — only 8 nonzeros per row of 16), it will work at $n = 64$+ where 2:4 leaves more relative capacity
- The ablation structure (4 models, $< 20$ min total) gives clear signal about which component matters

## Theoretical Analysis

**Complexity Comparison:**

| Operation | Dense SSM | Sparse SSM (2:4) | Diagonal SSM |
|-----------|----------|-------------------|-------------|
| Forward matvec | $O(n^2)$ | $O(n^2/2)$ | $O(n)$ |
| Backward matvec | $O(n^2)$ | $O(n^2/2)$ | $O(n)$ |
| Parameters | $O(n^2)$ | $O(n^2/2)$ at inference | $O(n)$ |
| Training memory | $O(n^2)$ | $O(n^2)$ (dense copy) | $O(n)$ |
| Inference memory | $O(n^2)$ | $O(n^2/2)$ | $O(n)$ |
| Permutation learning | — | $O(n \cdot B)$ extra params | — |

**Sparse Tensor Core Speedup:**

On NVIDIA A100/H100 with Sparse Tensor Cores, 2:4 spMM achieves:
- Theoretical: $2\times$ throughput (TFLOPS)
- Practical: $1.5$–$1.8\times$ (cuSPARSELt overhead)

**Error Accumulation Bound:**

Let $\epsilon_t = \|A_{\text{sparse},t} - A_{\text{dense},t}\|_2$ be the per-step sparsification error. For a sequence of length $T$:

$$
\|h_T^{\text{sparse}} - h_T^{\text{dense}}\|_2 \leq \sum_{t=1}^T \gamma^{T-t} \epsilon_t \|h_{t-1}\|_2 \leq \frac{\epsilon_{\max}}{1 - \gamma} \cdot \max_t \|h_t\|_2
$$

where $\gamma = 0.99$ is the spectral norm bound. This shows error is bounded and doesn't explode, but may grow linearly with $\epsilon_{\max}$.

## Risks & Limitations

1. **Training overhead**: Maintaining dense weights ($2\times$ memory) + Sinkhorn + Hungarian during training. At inference, only sparse weights + index array needed.

2. **Small state dimensions**: For Mamba's $n = 16$, a 2:4 sparse row has only 8 nonzeros out of 16 — very aggressive. May need to use 4:8 sparsity for small $n$, or apply only to layers with $n \geq 64$.

3. **Interaction with chunkwise scan**: The sparse transition matrix must be materialized at chunk boundaries. If chunk size $C$ is small, the materialization overhead dominates. With $C = 64$, amortized over 64 steps.

4. **Spectral norm estimation cost**: Computing $\|A_{\text{sparse}}\|_2$ via power iteration adds $O(n^2)$ cost per step. Alternative: constrain diagonal/off-diagonal entries directly.

5. **Gradient bias**: Three levels of gradient approximation stacked: S-STE (identity), Sinkhorn-Hungarian (STE), and stability projection (projection gradient). May require careful hyperparameter tuning.

6. **Hardware dependency**: Sparse Tensor Core acceleration is NVIDIA-specific (A100/H100). On other hardware, no speedup.

## Follow-up Experiments

1. **V:N:M hierarchical sparsity**: Apply column-level pruning + 2:4 row-level pruning for $3$–$4\times$ sparsity, following the VNM trick
2. **Transposable masks**: Use transposable 2:4 masks (TSENOR) to enable sparse forward AND backward passes
3. **Sparse + diagonal hybrid**: Keep some heads as diagonal (maximally sparse), others as 2:4 sparse (moderate mixing)
4. **Scale to Mamba-2**: Apply to Mamba-2's SSD framework where the transition is already expressed as matrix operations
5. **Dynamic sparsity**: Allow the 2:4 mask to be input-dependent (different masks for different tokens), trading hardware efficiency for expressivity
6. **Combine with Proposal 023 (CD-SSM)**: Apply 2:4 sparsity to the diagonal factors of CD-SSM transitions

## Human Review

(To be filled by reviewer)

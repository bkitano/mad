---
status: ongoing
priority: high
created: 2026-02-15
based_on: tropical-attention-hilbert-projective, semiring-monoid-lifting, input-dependent-gating, recurrence-to-scan-reduction, blelloch-work-efficient-scan, online-softmax, chunkwise-parallel-scan, simd2-semiring-matrix-acceleration
experiment_number: 015
---

# Tropical-Gated SSM: Max-Plus Parallel Scan with Input-Dependent Selectivity

## Hypothesis

Replacing the standard $(\mathbb{R}, +, \times)$ semiring in SSM state recurrences with the **tropical semiring** $(\mathbb{R} \cup \{-\infty\}, \max, +)$ — combined with input-dependent gating — produces a recurrent model with **hard-winner-take-all state dynamics** that (1) maintains sharper, non-fading memory over long sequences than softmax-based or standard linear attention, (2) achieves perfect length generalization on algorithmic tasks, and (3) is naturally 1-Lipschitz stable without requiring eigenvalue constraints.

## Background

### The Attention-Fading and State-Dilution Problem

Both softmax attention and standard SSMs suffer from **state dilution** as sequence length grows. In linear attention, the state matrix $S_t = \sum_i \phi(k_i) v_i^\top$ accumulates all past key-value pairs additively; as $t$ grows, any individual contribution $k_j v_j^\top$ is drowned out by the sum. Softmax attention avoids this via normalization but at $O(T^2)$ cost. SSMs with $|a_t| < 1$ decay old information exponentially — useful for some tasks but fatal for tasks requiring precise long-range recall.

The documented **tropical attention** trick (Hashemi et al., NeurIPS 2025) shows that replacing the softmax kernel with the tropical Hilbert projective metric and max-plus aggregation yields **scale-invariant, non-fading attention** that generalizes from length 8 to length 1024 without degradation. But tropical attention operates as a quadratic attention mechanism — it hasn't been formulated as a linear-cost recurrence.

### The Key Insight: Tropical Recurrences Are Parallelizable

The tropical semiring $(\max, +)$ satisfies the requirements for a parallel scan:

1. **Associativity**: $\max(a, \max(b, c)) = \max(\max(a, b), c)$ ✓
2. **Identity element**: $-\infty$ is the identity for $\max$ ✓
3. **Distributivity**: $a + \max(b, c) = \max(a+b, a+c)$ ✓

Therefore, the recurrence $\ell_t = \max(a_t + \ell_{t-1}, \; b_t)$ can be computed via a parallel prefix scan with $O(\log T)$ depth, exactly like standard SSM scans. The scan operator on pairs is:

$$
(a_1, b_1) \otimes (a_2, b_2) = (a_1 + a_2, \; \max(a_2 + b_1, \; b_2))
$$

This is the tropical analog of the affine-composition scan in standard SSMs.

### How This Differs From Proposal 014 (Log-Semiring SSM)

Proposal 014 uses the **logarithmic semiring** $(\text{logsumexp}, +)$ which smoothly interpolates between standard addition ($\mu \to 0$) and tropical max ($\mu \to \infty$). Our proposal operates at the **hard tropical limit** ($\mu = \infty$), which gives three distinct advantages:

| Property | Log-Semiring (014) | Tropical (this) |
|----------|-------------------|-----------------|
| Aggregation | Smooth $\text{logsumexp}$ | Hard $\max$ (winner-take-all) |
| Gradients | Dense (softmax-distributed) | Sparse (only argmax gets gradient) |
| Stability | Requires numerical care for $e^{\mu a}$ | Piecewise-linear, no exp/log |
| Length invariance | Approximate (depends on $\mu$) | Exact (max is scale-invariant) |
| Hardware | CUDA cores only ($\text{logsumexp}$) | CUDA cores (today); SIMD² $\text{maxplus}$ (future) |

The sparse gradients of $\max$ are a double-edged sword: they can cause training difficulty but also force the model to learn **discrete, interpretable routing** — exactly what's needed for algorithmic reasoning. We address the training challenge via a temperature-annealing schedule (see Method).

### Hardware Considerations

Currently, max-plus operations run on CUDA cores at ~$16\times$ lower throughput than tensor-core GEMM. However:

1. The **SIMD² proposal** (Zhang et al., ISCA 2022) shows that adding max-plus support to matrix units requires only 5% chip area overhead and yields $8\text{–}16\times$ speedup over CUDA cores — bringing tropical matmul to parity with standard GEMM.
2. Even on current hardware, the **per-element** nature of the tropical scan (no matrix operations needed) means it is memory-bandwidth-bound, not compute-bound. A scalar tropical scan is faster than a matrix-valued standard scan.
3. The tropical scan operator requires only **additions and comparisons** — the cheapest operations on any hardware. No multiplications, no divisions, no transcendentals.

## Mathematical Formulation

### Scalar Tropical SSM Recurrence

For a single state dimension per head:

$$
\ell_t = \max\big(a_t + \ell_{t-1}, \;\; b_t\big)
$$

where:
- $\ell_t \in \mathbb{R}$ — hidden state (log-space score of the "best" past input)
- $a_t = -\text{softplus}(W_a x_t + c_a) \in (-\infty, 0]$ — input-dependent decay (negative ⟹ contractive)
- $b_t = q_t^\top k_t \in \mathbb{R}$ — input "bid" (dot-product score of current input)

**Unrolled interpretation:**

$$
\ell_t = \max_{0 \leq j \leq t} \left(\sum_{s=j+1}^{t} a_s + b_j\right)
$$

This is the **maximum cumulative-decayed score** over all past inputs. The winner is the input $j^*$ that maximizes $b_j - \sum_{s=j+1}^{t} |a_s|$ — high initial score ($b_j$) minus cumulative decay.

### Vector-Valued Tropical SSM (Multi-Head)

For $H$ heads with $d_k$-dimensional keys/values:

$$
\ell_t^{(h)} = \max\big(a_t^{(h)} + \ell_{t-1}^{(h)}, \;\; b_t^{(h)}\big) \quad \text{for } h = 1, \ldots, H
$$

**Output computation:**

$$
y_t = \sum_{h=1}^{H} v_{j_h^*(t)} \cdot W_O^{(h)}
$$

where $j_h^*(t) = \arg\max_{0 \leq j \leq t} \left(\sum_{s=j+1}^{t} a_s^{(h)} + b_j^{(h)}\right)$ is the "winning" past position for head $h$.

**Key challenge:** The argmax-based value retrieval is non-differentiable. We address this via:

1. **Training**: Use the **log-semiring annealed to tropical** — start with $\text{logsumexp}_\mu$ at moderate $\mu$ (softmax-like) and anneal $\mu \to \infty$ during training (approaching hard max). This is the tropical analog of Gumbel-softmax temperature annealing.
2. **Inference**: Use hard max for exact, deterministic retrieval.

### Annealed Log-to-Tropical Training Schedule

During training, use the parametric recurrence:

$$
\ell_t^{(\mu)} = \frac{1}{\mu} \log\left(e^{\mu(a_t + \ell_{t-1}^{(\mu)})} + e^{\mu b_t}\right)
$$

with annealing schedule $\mu(s) = \mu_0 + (\mu_{\max} - \mu_0) \cdot \min(1, s / s_{\text{anneal}})$ where $s$ is the training step. At $\mu \to \infty$, this converges to $\max(a_t + \ell_{t-1}, b_t)$.

**Gradient through the annealed recurrence:**

$$
\frac{\partial \ell_t^{(\mu)}}{\partial a_t} = \frac{e^{\mu(a_t + \ell_{t-1})}}{e^{\mu(a_t + \ell_{t-1})} + e^{\mu b_t}} = \sigma\big(\mu(a_t + \ell_{t-1} - b_t)\big)
$$

This is a sigmoid with temperature $1/\mu$ — at high $\mu$, gradients are nearly binary, flowing almost entirely to the winner.

### Parallel Scan Operator

The tropical affine scan operator $\otimes$ on pairs $(a, \ell)$:

$$
(a_1, \ell_1) \otimes (a_2, \ell_2) = \big(a_1 + a_2, \;\; \max(a_2 + \ell_1, \;\; \ell_2)\big)
$$

**Proof of associativity:**

$$
\big((a_1, \ell_1) \otimes (a_2, \ell_2)\big) \otimes (a_3, \ell_3) = \big(a_1 + a_2 + a_3, \;\; \max(a_3 + a_2 + \ell_1, \; a_3 + \ell_2, \; \ell_3)\big)
$$

$$
(a_1, \ell_1) \otimes \big((a_2, \ell_2) \otimes (a_3, \ell_3)\big) = \big(a_1 + a_2 + a_3, \;\; \max(a_2 + a_3 + \ell_1, \; a_3 + \ell_2, \; \ell_3)\big)
$$

Both yield the same result. ✓

**Identity element:** $(0, -\infty)$ — zero decay, negative-infinity score.

### Tropical Attention Duality

The unrolled tropical SSM computes:

$$
y_t = v_{j^*(t)}, \quad j^*(t) = \arg\max_j \left(\underbrace{\sum_{s=j+1}^t a_s}_{\text{cumulative decay}} + \underbrace{b_j}_{\text{input score}}\right)
$$

This is equivalent to a **causal tropical attention** mechanism:

$$
y_t = V[j^*(t), :], \quad j^*(t) = \arg\max_j \left(-d_{\text{decay}}(j, t) + \text{score}(j)\right)
$$

where $d_{\text{decay}}$ is an input-dependent "distance" and $\text{score}$ is the key-query match. Unlike softmax attention which computes a weighted average (lossy), tropical attention **retrieves the single best-matching entry** (lossless for the winner).

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Tropical-Gated SSM (TG-SSM) |
| Layers | $L = 12$ |
| Model dim | $d_{\text{model}} = 768$ |
| Heads | $H = 12$ |
| Head dim | $d_k = 64$ |
| Scan semiring | Tropical $(\max, +)$ with log-semiring annealing |
| FFN | SwiGLU, $d_{\text{ff}} = 2048$ |

**Per-head computation:**

1. Project: $q_t, k_t, v_t = x_t W_Q, x_t W_K, x_t W_V$ (each $\in \mathbb{R}^{d_k}$)
2. Compute bid: $b_t = q_t^\top k_t / \sqrt{d_k}$
3. Compute decay: $a_t = -\text{softplus}(x_t W_a + c_a) \leq 0$
4. Tropical scan: $\ell_t = \max(a_t + \ell_{t-1}, b_t)$
5. Retrieve: $\hat{y}_t = \text{softmax}_\mu(\text{scores over history}) \cdot V$ (soft during training, hard at inference)
6. Gate output: $y_t = \hat{y}_t \odot \sigma(x_t W_g)$ (post-sigmoid gating from trick 009)

### Baseline

1. **Standard linear attention** (RetNet-style): $O(Td^2)$, standard semiring scan
2. **Mamba-2**: $O(Td)$, diagonal input-dependent SSM
3. **Log-semiring SSM** (Proposal 014): $O(Td)$, logsumexp scan
4. **Softmax Transformer**: $O(T^2 d)$, quadratic attention

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MQAR (Multi-Query Associative Recall) | $> 95\%$ at $T = 8192$ | Accuracy on synthetic retrieval |
| Needle-in-Haystack | $> 98\%$ at $T = 32768$ | Retrieval accuracy at depth |
| Length generalization | Train $T = 512$, test $T = 16384$ | MQAR accuracy at unseen lengths |
| WikiText-103 perplexity | $\leq$ Mamba-2 baseline | Validation PPL at 350M params |
| Throughput | $> 0.5\times$ Mamba-2 | Tokens/sec on A100 |

### Estimated Compute

**Full experiment**: ~150 GPU-hours (A100)
- 350M parameter model, 15B tokens training
- Baseline comparisons add ~100 GPU-hours
- Total: ~250 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- **MQAR**: $> 95\%$ accuracy at $T = 8192$ where linear attention degrades to $< 70\%$ (because tropical retrieval is exact, not averaged)
- **Length generalization**: Near-zero degradation from $T = 512$ to $T = 16384$ (max is scale-invariant)
- **Perplexity**: Within 5% of Mamba-2 on WikiText-103 (the model trades soft aggregation for hard retrieval; language modeling may slightly suffer but retrieval tasks should dominate)
- **Throughput**: $0.5\text{–}0.8\times$ Mamba-2 on current hardware (CUDA cores for max-plus); $\geq 1\times$ with hypothetical SIMD² support

**If hypothesis is wrong:**
- **Training instability from sparse gradients**: If the annealing schedule fails to produce a good curriculum from soft to hard, the model may converge to degenerate solutions (all heads attending to the same position). This would indicate that the smooth-to-hard transition requires more sophisticated techniques (e.g., Gumbel noise injection).
- **Perplexity catastrophe**: If hard-max aggregation is fundamentally inappropriate for language modeling (where soft mixture is essential), the model will have significantly worse perplexity ($> 20\%$ gap). This would suggest tropical SSMs are only suitable for algorithmic/retrieval tasks, not general language.
- **Either way, we learn**: Whether max-based aggregation can serve as a general-purpose sequence model or is restricted to reasoning/retrieval niches.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer TG-SSM, $d = 64$, $H = 4$, $d_k = 16$ (~150K params)
- **Task**: Multi-Query Associative Recall (MQAR) — given key-value pairs followed by queries, retrieve the correct value for each query key
- **Data**: Synthetic MQAR with $T = 256$, 8 key-value pairs, vocabulary size 64, 10K training samples
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- $> 95\%$ MQAR accuracy at $T = 256$ where a standard linear attention baseline achieves $< 80\%$
- Successful length generalization: train on $T = 256$, test on $T = 1024$ with $< 5\%$ accuracy drop

### Failure Criteria
- If MQAR accuracy is $< 60\%$ (below random retrieval with 8 items), the tropical scan mechanism is fundamentally broken
- If annealing from $\mu = 1$ to $\mu = 100$ causes training loss to diverge, the smooth-to-hard curriculum fails

### Why This Test Is Sufficient
- MQAR directly tests the core capability: **precise retrieval from long-range memory**. This is exactly where tropical (hard-max) aggregation should dominate over additive (soft) aggregation.
- If the tiny model achieves near-perfect MQAR with length generalization, it validates that (a) the tropical scan computes meaningful attention, (b) the annealing schedule enables learning, and (c) the max-based retrieval doesn't degrade with sequence length.
- Language modeling quality requires scaling; MQAR validates the core mechanism at toy scale.

## Theoretical Analysis

**Complexity comparison:**

| Operation | Linear Attention | Tropical SSM (this) | Softmax Attention |
|-----------|-----------------|---------------------|-------------------|
| Forward pass | $O(T d_k^2)$ | $O(T H)$ per scan + $O(T d_k)$ projections | $O(T^2 d_k)$ |
| Backward pass | $O(T d_k^2)$ | $O(T H)$ per scan + $O(T d_k)$ projections | $O(T^2 d_k)$ |
| Memory | $O(d_k^2)$ state | $O(H)$ scalar state + $O(T d_k)$ for argmax retrieval | $O(T^2)$ |
| Retrieval precision | Approximate (rank-$d_k$) | Exact (winner-take-all) | Exact (softmax-weighted) |

**Key advantage**: The tropical scan state is **scalar per head** ($O(H)$ total) vs $O(d_k^2)$ for linear attention. The retrieval requires storing the argmax indices and value cache, which is $O(T d_k)$ — same as KV cache in standard attention.

**Crossover analysis**: The tropical SSM is faster than softmax attention when $T > d_k$ (which is nearly always: $T \geq 512$, $d_k = 64$). It is faster than linear attention in the scan itself but requires KV cache for retrieval, making total memory comparable.

## Risks & Limitations

1. **Sparse gradients may cause training difficulty**: The max operator only passes gradients to the winning element. Mitigation: annealing schedule and multiple heads ensure gradient diversity.
2. **Hard retrieval may be too restrictive for language**: Language modeling often benefits from soft mixture of past tokens. Mitigation: mix tropical heads with standard linear attention heads in a hybrid architecture.
3. **KV cache requirement**: Unlike pure SSMs that discard past tokens, tropical retrieval needs to store all past values for the argmax lookup. This makes the model more like a cached attention model than a pure recurrent model. Mitigation: use sliding-window truncation for very long sequences.
4. **No tensor-core acceleration on current hardware**: Max-plus runs on CUDA cores. Mitigation: the scalar-per-head state means the scan is memory-bound, not compute-bound; CUDA core throughput is sufficient for scalar operations.
5. **Argmax ties**: When multiple past positions have identical scores, the argmax is non-unique. Mitigation: add small Gumbel noise during training for tie-breaking.

## Follow-up Experiments

1. **Hybrid tropical + linear attention**: Allocate some heads to tropical (retrieval) and some to standard linear (aggregation), allowing the model to both retrieve precisely and aggregate softly.
2. **Multi-value tropical retrieval (top-$k$)**: Instead of retrieving only the argmax, retrieve the top-$k$ values and average them — a smooth interpolation between tropical and softmax.
3. **Tropical SSM + log-linear attention**: Combine the tropical scan (for hard retrieval at recent positions) with log-linear attention's hierarchical states (for multi-resolution soft aggregation of distant history).
4. **SIMD² benchmarking**: If/when SIMD²-style hardware becomes available, benchmark the tropical scan against tensor-core standard scans to quantify the hardware acceleration potential.
5. **Tropical state-tracking**: Test on finite automaton / state-tracking tasks (S5 composition, group word problems) where the max-based dynamics might enable tracking multiple competing hypotheses.

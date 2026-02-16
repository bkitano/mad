---
status: ongoing
priority: high
created: 2026-02-15
based_on: signed-permutation-hyperoctahedral-group, wreath-product-fft-hyperoctahedral, cayley-contractive-parameterization, input-dependent-gating, chunkwise-parallel-scan, gumbel-softmax-reparameterization, krohn-rhodes-monoid-decomposition
experiment_number: 017
---

# Hyperoctahedral Signed-Permutation SSM State Transitions

## Hypothesis

A state-space model whose state transition matrices are parameterized as input-dependent signed permutations (elements of the hyperoctahedral group $B_n = \mathbb{Z}_2 \wr S_n$) will achieve strictly higher state-tracking expressivity than both diagonal SSMs and pure-permutation SSMs, while maintaining $O(n)$ per-step cost and stability $\|A_t\|_2 = 1$ by construction.

## Background

The existing proposals explore several structured state-transition classes:
- **Diagonal** (Mamba, S4D): $O(n)$ but abelian — cannot represent non-commutative groups.
- **Householder products** (DeltaProduct, proposals 001/002): expressive but sequential in $n_h$, losing parallelization.
- **Monarch factored** (proposals 006, 010, 016): $O(n\sqrt{n})$ with BMM but requires multiple factors.
- **Monomial** (proposal 016): $O(n)$ per-step, group closure, but block-diagonal structure limits cross-block mixing per factor.

**The gap**: None of the proposals exploit the **hyperoctahedral group** $B_n = \mathbb{Z}_2^n \rtimes S_n$ as a state-transition class. This is surprising because:

1. **$B_n$ is the maximal finite subgroup of $O(n)$** — it contains ALL permutation matrices AND all coordinate reflections, making it strictly more expressive than $S_n$ while remaining finite.
2. **Signed permutations have $O(n)$ storage and $O(n)$ matvec** — $M x = \text{signs} \odot x[\text{perm}]$ — matching diagonal cost.
3. **Krohn-Rhodes decomposition tells us** that the group component of any finite automaton's transition monoid decomposes into simple groups. For many tasks, $B_n$ provides the right "group envelope" because it captures both transposition and sign-flip dynamics.
4. **The D₄ dihedral group** (order 8) — which proposal 001 targets — embeds naturally in $B_2$ (signed permutations on 2 elements). More generally, all finite Coxeter groups embed in $B_n$ for appropriate $n$.

The key challenge is **differentiability**: $B_n$ is discrete ($2^n n!$ elements), so we need a continuous relaxation. We propose factoring $B_n = \mathbb{Z}_2^n \rtimes S_n$ into:
- **Signs**: $s \in \{-1, +1\}^n$ — relaxed via sigmoid: $\hat{s}_i = 2\sigma(\alpha_i) - 1$, giving continuous signs in $[-1, 1]$.
- **Permutation**: $\pi \in S_n$ — relaxed via Gumbel-Sinkhorn (Mena et al., 2018) with ST hardening.

This decomposition is algebraically exact ($M = \text{diag}(s) \cdot P_\pi$) and each component has a well-studied differentiable relaxation. The input-dependent version makes both $s(x_t)$ and $\pi(x_t)$ functions of the input, creating an LTV system.

## Mathematical Formulation

**Standard SSM recurrence:**

$$
h_t = A_t h_{t-1} + B x_t
$$

**Signed-Permutation State Transition:**

$$
A_t = D(x_t) \cdot P(x_t) \in B_n
$$

where:
- $D(x_t) = \text{diag}(s_1(x_t), \ldots, s_n(x_t))$ with $s_i(x_t) \in \{-1, +1\}$
- $P(x_t) \in S_n$ is a permutation matrix

**Continuous Relaxation:**

**Sign component:**

$$
\hat{s}_i(x_t) = 2\sigma(W_s x_t + b_s)_i - 1 \in [-1, 1]
$$

Training uses the soft $\hat{s}_i$; at inference, harden to $\text{sign}(\hat{s}_i)$.

**Permutation component (Gumbel-Sinkhorn):**

$$
\hat{P}(x_t) = \text{Sinkhorn}^{(L)}\left(\frac{W_P x_t}{\tau}\right) \in \mathcal{DS}_n
$$

where $W_P \in \mathbb{R}^{n \times n \times d}$ projects input to an $n \times n$ cost matrix, and $\mathcal{DS}_n$ is the set of doubly stochastic matrices. The ST Gumbel-Sinkhorn uses Hungarian hardening in forward:

$$
P_t^{\text{hard}} = \text{Hungarian}(\hat{P}(x_t)), \quad P_t^{\text{ST}} = P_t^{\text{hard}} + (\hat{P}(x_t) - \hat{P}(x_t)^{\text{detach}})
$$

**Combined action:**

$$
A_t x = \hat{D}(x_t) \cdot \hat{P}(x_t) \cdot x = \hat{s}(x_t) \odot (\hat{P}(x_t) \cdot x)
$$

**Key Properties:**
- $\|A_t\|_2 = 1$ when $\hat{s}_i \in \{-1, +1\}$ and $P$ is a permutation (orthogonal by construction)
- During training with soft relaxation: $\|A_t\|_2 \leq 1$ (contractive) since $|\hat{s}_i| \leq 1$ and $\|\hat{P}\|_2 \leq 1$ for doubly stochastic $\hat{P}$
- Captures all of $B_n$ at convergence (when $\tau \to 0$ and sigmoids saturate)

**Gating for Forgetting (Krohn-Rhodes aperiodic component):**

To handle tasks requiring information decay (the aperiodic $U_2$ component from Krohn-Rhodes), add a scalar gate:

$$
h_t = \gamma(x_t) \odot (A_t h_{t-1}) + (1 - \gamma(x_t)) \odot B x_t
$$

where $\gamma(x_t) = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)^n$. This interpolates between the signed-permutation dynamics (group component, $\gamma = 1$) and input injection (aperiodic reset, $\gamma = 0$).

**Key Variables:**
- $x_t \in \mathbb{R}^d$ — input at time $t$
- $h_t \in \mathbb{R}^n$ — hidden state
- $A_t \in B_n$ — signed permutation state transition (or its continuous relaxation)
- $D(x_t) \in \mathbb{R}^{n \times n}$ — diagonal sign matrix
- $P(x_t) \in \mathbb{R}^{n \times n}$ — permutation matrix
- $\gamma(x_t) \in (0,1)^n$ — forget gate

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | HyperSSM (Hyperoctahedral SSM) |
| Layers | $L = 4$–$8$ |
| Hidden dim | $d = 128$–$512$ |
| State dim per head | $n = 8$–$16$ |
| Heads | $H = d / n$ |
| Permutation relaxation | Gumbel-Sinkhorn ($L_{\text{Sink}} = 5$, $\tau$: $1.0 \to 0.1$) |
| Sign relaxation | Sigmoid with STE hardening |

### Baseline

1. **Diagonal SSM** (Mamba-2 style): $A_t = \text{diag}(\alpha(x_t))$, complexity $O(Tn)$
2. **DeltaProduct** ($n_h = 4$ Householder reflections): $O(Tn \cdot n_h)$ per step
3. **GS-Monomial** (proposal 016): $O(Tn\sqrt{n})$ per step
4. **Dense input-dependent**: $A_t = f(x_t) \in \mathbb{R}^{n \times n}$, complexity $O(Tn^2)$

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $B_n$ state tracking accuracy | $> 90\%$ | Signed permutation composition task |
| $S_5$ accuracy | $\geq$ GS-Monomial | Permutation group word problem |
| Throughput | $> 0.8\times$ Mamba-2 | Tokens/sec on A100 |
| Memory | $\leq 1.2\times$ Mamba-2 | Peak GPU memory |
| Stability | 0 NaN/Inf | Over 100K training steps |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (state tracking tasks): ~20 GPU-hours on A100
**Phase 2** (language modeling): ~100 GPU-hours on A100
**Total**: ~120 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- $> 95\%$ accuracy on $B_4$ (signed permutation) composition task where diagonal SSM achieves $< 30\%$ and pure $S_n$ SSM achieves $< 70\%$ (missing sign-flip dynamics)
- Matches or exceeds DeltaProduct on $S_5$ while being faster (no sequential Householder chain)
- $> 0.8\times$ Mamba-2 throughput because the core operation ($\text{signs} \odot x[\text{perm}]$) is a simple gather + multiply, tensor-core-free but very fast on GPU

**If hypothesis is wrong:**
- If Gumbel-Sinkhorn fails to learn useful permutations: the soft permutation gradient signal is too noisy at small $n$, suggesting the continuous relaxation of discrete groups has fundamental limitations for SSM training
- If signed permutations don't help over plain permutations: the sign component is redundant because Householder products already achieve sign flips implicitly, and the explicit $\mathbb{Z}_2^n$ factor adds no expressivity in practice
- Either outcome informs the broader question of whether finite group structure can be directly imposed on SSM transitions

## Minimum Viable Experiment

### Setup
- **Model**: 1-layer HyperSSM, $n = 8$ state dim, $d = 32$ input dim, ~50K params
- **Task**: $B_3$ composition — input is a sequence of generators of the hyperoctahedral group $B_3$ (3 adjacent transpositions + 1 sign flip), output is the resulting signed permutation applied to a test vector
- **Data**: 10K sequences of length 8–16, synthetic
- **Compute**: Single GPU, < 8 minutes

### Success Criteria
- $> 90\%$ accuracy on $B_3$ composition (HyperSSM)
- $< 60\%$ accuracy for diagonal SSM baseline on same task
- $< 75\%$ accuracy for permutation-only SSM (no signs) — demonstrating the value of the $\mathbb{Z}_2^n$ component

### Failure Criteria
- If HyperSSM accuracy $< 70\%$: the Gumbel-Sinkhorn relaxation fails to learn discrete permutations at this scale
- If permutation-only SSM matches HyperSSM: the sign component is unnecessary, reducing this to a known approach

### Why This Test Is Sufficient
- $B_3$ composition requires both permutation and sign-flip tracking — if the model can learn the group operation on $B_3$ (order $2^3 \cdot 3! = 48$), the core mechanism works. Scaling to larger $n$ adds capacity, not fundamentally different capability.
- The task directly tests whether the $\mathbb{Z}_2^n$ component provides value over pure permutations, which is the novel claim.

## Theoretical Analysis

Complexity comparison:

| Operation | Diagonal SSM | HyperSSM | DeltaProduct ($n_h$) | Dense |
|-----------|-------------|----------|---------------------|-------|
| Forward pass (per step) | $O(n)$ | $O(n^2)$ (Sinkhorn) or $O(n)$ (hard perm) | $O(n \cdot n_h)$ | $O(n^2)$ |
| Backward pass (per step) | $O(n)$ | $O(n^2)$ (Sinkhorn grad) | $O(n \cdot n_h)$ | $O(n^2)$ |
| Memory (per step) | $O(n)$ | $O(n^2)$ (cost matrix) | $O(n \cdot n_h)$ | $O(n^2)$ |
| Expressivity (group) | Abelian $(\mathbb{R}_{>0}, \times)$ | $B_n = \mathbb{Z}_2 \wr S_n$ | $O(n)$ (reaches all of $O(n)$) | $GL(n, \mathbb{R})$ |

**Training-time bottleneck**: The Sinkhorn normalization costs $O(n^2)$ per step per head. With small state dimension $n = 8$–$16$ and many heads, this is dominated by the $O(d)$ projections.

**Inference optimization**: At inference, the hard permutation $P_t$ can be represented as an integer index array + sign array, making the state update a simple gather-and-sign-flip at $O(n)$ cost.

**Crossover point**: HyperSSM is better than DeltaProduct when the task requires coordinate mixing that DeltaProduct's sequential Householder chain handles slowly (large $n_h$), and better than diagonal when the task requires non-commutative state tracking.

## Risks & Limitations

1. **Gumbel-Sinkhorn gradient quality**: The STE through Hungarian hardening introduces gradient bias. For small state dimensions ($n = 8$), there are only $n! = 40320$ permutations but the soft relaxation lives in $\mathbb{R}^{n \times n}$ — the landscape may have many local optima.
2. **Chunkwise parallelization**: Signed permutation matrices do NOT decompose into element-wise operations, so the standard diagonal parallel scan doesn't apply. Must use matrix-valued parallel scan, which costs $O(n^2)$ per scan step — acceptable for small $n$ but limits scaling.
3. **Temperature scheduling**: The Sinkhorn temperature $\tau$ must be annealed carefully; too fast $\to$ gradient collapse, too slow $\to$ never converges to discrete permutations.
4. **Comparison fairness**: DeltaProduct with $n_h \geq n$ can express ALL of $O(n) \supset B_n$, so the advantage of HyperSSM is efficiency, not expressivity. The claim is that $B_n$ is a "sweet spot" — more expressive than diagonal, cheaper than full $O(n)$.
5. **Scaling**: For large $n$, the $O(n^2)$ Sinkhorn cost may dominate. Blockwise Sinkhorn (from PermLLM trick) could help, at the cost of restricting to block-diagonal permutations.

## Follow-up Experiments

1. **Blockwise Sinkhorn for larger $n$**: Apply the PermLLM-style block-diagonal permutation to scale to $n = 64$–$256$ with $O(n \cdot b)$ Sinkhorn cost per block of size $b$.
2. **Wreath product structure**: Exploit the $B_n = \mathbb{Z}_2 \wr S_n$ decomposition more deeply — first transform the $\mathbb{Z}_2^n$ "fiber," then the $S_n$ "base" — using the wreath product FFT for spectral analysis of learned transitions.
3. **Comparison with proposal 016 (GS-Monomial)**: GS-Monomial uses monomial matrices (signed permutations!) but with block-diagonal + shuffle structure. Direct head-to-head comparison to understand whether the full $B_n$ or the structured GS decomposition is better.
4. **Cayley relaxation alternative**: Instead of Gumbel-Sinkhorn, parameterize the permutation component via Cayley transform of a skew-symmetric matrix restricted to $\{0, \pm 1\}$ entries — giving a differentiable path through $O(n)$ that passes through $B_n$.
5. **Language modeling scale**: If MVE succeeds, train a 125M-parameter model on OpenWebText with HyperSSM layers replacing Mamba-2, measuring perplexity and throughput.

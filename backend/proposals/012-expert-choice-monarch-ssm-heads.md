---
status: failed
priority: high
created: 2026-02-15
based_on: expert-choice-routing, monarch-matrix-factorization, input-dependent-gating, chunkwise-parallel-scan, semiseparable-block-decomposition, bilinear-gating-glu, kernel-fusion
experiment_number: 012
experiment_log: experiment-log-012.md
results_file: 012_results.md
---

# Expert-Choice Routing for Monarch-Factored SSM State Heads

## Hypothesis

Applying expert-choice routing to SSM state transition heads — where each "expert" is a distinct Monarch-factored state transition matrix that selects which tokens to process — will produce a multi-head SSM that achieves (a) $2\times$ better state-tracking expressivity than uniform multi-head SSMs (all heads process all tokens), (b) perfect computational load balancing by construction, and (c) implicit content-based specialization where different heads learn to handle different input patterns, with total compute matching a standard multi-head SSM at capacity factor $c = 1$.

## Background

### Multi-Head SSMs: Wasted Uniform Processing

Modern SSMs (Mamba, Mamba-2, GLA) use multiple heads, where each head maintains its own state and processes every token in the sequence. This is analogous to multi-head attention, where every head computes attention over all tokens. But expert-choice routing (Zhou et al. 2022) demonstrated for MoE FFN layers that *not all tokens need the same processing*: letting experts select their tokens achieves better results than uniform processing at the same compute budget, because tokens naturally cluster by difficulty and type.

**The gap**: Expert-choice routing has only been applied to FFN layers in Transformers. No one has applied it to SSM state heads, where the "experts" would be different state transition dynamics rather than different FFN weights. This is a natural fit because:

1. SSM heads already process tokens independently — each head has its own $A_h, B_h, C_h$
2. Different heads could specialize in different temporal patterns (fast decay vs. long memory, oscillatory vs. monotone)
3. The routing creates an implicit content-dependent selection: some tokens get processed by many specialized heads, others by few — a form of *adaptive computation* for sequence modeling

### Monarch Factorization for Diverse Head Dynamics

Proposal 006 explored Monarch-factored state transitions but applied them uniformly to all tokens. Here, we combine Monarch factorization with expert-choice routing: each head has a distinct Monarch-factored $A_h = P_b^T L_h P_b R_h$ (or simpler diagonal + permutation structure), and the expert-choice mechanism determines which tokens each head processes.

The Monarch factorization is particularly well-suited for this because:
- **Diverse expressivity**: Different $L_h, R_h$ can capture fundamentally different transition dynamics
- **Efficient computation**: Monarch matmul is $O(N\sqrt{N})$, so adding heads doesn't blow up cost
- **BMM-friendly**: Multiple heads naturally batch as block matrix multiplies — the same pattern that makes expert-choice routing efficient

### Connection to Krohn-Rhodes Decomposition

The Krohn-Rhodes theorem tells us that any finite-state computation decomposes into an alternating wreath product of simple group components and aperiodic ($U_2$) components. Different SSM heads can be understood as implementing different components of this decomposition — some heads handle group dynamics (orthogonal/oscillatory transitions), others handle gating/forgetting (contractive transitions). Expert-choice routing lets the model learn *which tokens need which decomposition component*, rather than applying all components to all tokens uniformly.

## Mathematical Formulation

**Standard Multi-Head SSM (Mamba-2 style):**

For $H$ heads, each processing all $T$ tokens:

$$
h_t^{(i)} = A^{(i)} h_{t-1}^{(i)} + B^{(i)} x_t, \quad o_t^{(i)} = C^{(i)} h_t^{(i)}, \quad i = 1, \ldots, H
$$

$$
y_t = \sum_{i=1}^{H} o_t^{(i)} \quad \text{(or concatenation + projection)}
$$

Total cost: $O(T \cdot H \cdot N^2)$ where $N$ is state dimension per head.

**Expert-Choice Monarch SSM (proposed):**

**Step 1: Compute token-to-head affinity.**

Given input $X \in \mathbb{R}^{T \times d}$ and head embeddings $W_g \in \mathbb{R}^{d \times H}$:

$$
S = \text{Softmax}(X \cdot W_g) \in \mathbb{R}^{T \times H}
$$

**Step 2: Expert (head) selects top-$k$ tokens.**

$$
G, I = \text{TopK}(S^\top, k), \quad k = \frac{T \cdot c}{H}
$$

- $I \in \mathbb{Z}^{H \times k}$ — indices of selected tokens per head
- $G \in \mathbb{R}^{H \times k}$ — gating weights

**Step 3: Per-head Monarch-factored SSM on selected tokens.**

For each head $i$, process only the $k$ selected tokens $\{x_{I[i,1]}, \ldots, x_{I[i,k]}\}$:

$$
h_j^{(i)} = A^{(i)} h_{j-1}^{(i)} + B^{(i)} x_{I[i,j]}, \quad j = 1, \ldots, k
$$

where $A^{(i)} = P_b^T L^{(i)} P_b R^{(i)}$ is the Monarch-factored transition for head $i$.

**Step 4: Scatter back with gating weights.**

$$
y_t = \sum_{\substack{i, j \\ I[i,j] = t}} G[i,j] \cdot C^{(i)} h_j^{(i)}
$$

**Monarch-Factored Transition Matrix:**

Each head's state transition is:

$$
A^{(i)} = (P_b^T L^{(i)} P_b) \cdot R^{(i)} \in \mathbb{R}^{N \times N}
$$

where:
- $P_b \in \{0,1\}^{N \times N}$ — fixed block permutation (shared across heads)
- $L^{(i)} = \text{BlockDiag}(L_1^{(i)}, \ldots, L_{\sqrt{N}}^{(i)})$ — block-diagonal, each block $\sqrt{N} \times \sqrt{N}$
- $R^{(i)} = \text{BlockDiag}(R_1^{(i)}, \ldots, R_{\sqrt{N}}^{(i)})$ — block-diagonal

Parameters per head: $2N$ (vs. $N^2$ for dense), giving $O(N\sqrt{N})$ MVM cost.

**Input-Dependent Gating (optional extension):**

Make the gating input-dependent within each head:

$$
\alpha_t^{(i)} = \sigma(w_\alpha^{(i) \top} x_t) \in (0, 1)
$$

$$
h_j^{(i)} = \alpha_t^{(i)} \cdot A^{(i)} h_{j-1}^{(i)} + B^{(i)} x_{I[i,j]}
$$

This adds selectivity within each head's assigned tokens.

**Key Variables:**

- $H$ — number of SSM heads ("experts")
- $k = Tc/H$ — tokens per head (expert capacity)
- $c$ — capacity factor ($c = 1$: each token sees 1 head on average; $c = 2$: two heads)
- $N$ — state dimension per head
- $S \in \mathbb{R}^{T \times H}$ — token-to-head affinity scores
- $G \in \mathbb{R}^{H \times k}$ — gating weights for selected tokens
- $I \in \mathbb{Z}^{H \times k}$ — token index assignments
- $A^{(i)}$ — Monarch-factored state transition for head $i$

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Base model | Multi-head SSM (Mamba-2 backbone) |
| Routing | Expert-choice: heads select tokens |
| State transition | Monarch-factored per head |
| Heads | $H = 8$–$16$ |
| State dim per head | $N = 16$–$64$ |
| Hidden dim | $d = 256$–$512$ |
| Layers | $L = 6$–$12$ |
| Capacity factor | $c = 1$ or $2$ |
| Parameters | ~10M–50M |

**Head diversity initialization:**

Initialize different heads with different dynamics to encourage specialization:
- Heads 1–$H/4$: Long-memory (eigenvalues near $|\lambda| = 1$, oscillatory via LinOSS initialization)
- Heads $H/4$–$H/2$: Short-memory (eigenvalues $|\lambda| \approx 0.5$–$0.9$, fast decay)
- Heads $H/2$–$3H/4$: Permutation-like (Monarch blocks initialized as near-permutations)
- Heads $3H/4$–$H$: Random (standard Monarch initialization)

### Baseline

1. **Standard multi-head SSM**: All heads process all tokens. Diagonal $A$, $H$ heads. Complexity: $O(THN)$.
2. **Monarch multi-head SSM (Proposal 006)**: All heads process all tokens, Monarch-factored $A$. Complexity: $O(THN\sqrt{N})$.
3. **Token-choice routed SSM**: Tokens select their heads (standard MoE routing applied to SSM heads). Same compute as expert-choice, but with load imbalance.
4. **Dense single-head SSM**: One large head with state dim $HN$, processing all tokens. Complexity: $O(T(HN)^2)$.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity (WikiText-103) | $> 5\%$ improvement vs baseline 1 at same compute | Validation perplexity |
| MQAR recall | $> 90\%$ at $T = 2048$ | Multi-Query Associative Recall |
| State tracking ($S_5$) | $> 85\%$ | Synthetic automaton benchmark |
| Head specialization | Entropy of routing distribution $< H/2$ | Measure how concentrated token-head assignments are |
| Load balance | Perfect ($k$ tokens per head) | Verify by construction |
| Throughput | $\geq 0.9\times$ standard multi-head | Tokens/sec on A100 |

### Estimated Compute

**Small to Medium**.

- Synthetic benchmarks (MQAR, state tracking): ~4 GPU-hours per configuration, 5 configurations = ~20 GPU-hours
- WikiText-103 training: ~24 GPU-hours per configuration, 5 configurations = ~120 GPU-hours
- Head specialization analysis: ~4 GPU-hours
- Ablation over $c \in \{0.5, 1, 2, 4\}$: ~48 GPU-hours

**Total: ~192 GPU-hours on A100.**

## Expected Outcome

**If hypothesis is correct:**

- **State tracking**: Expert-choice Monarch SSM significantly outperforms standard multi-head SSM on $S_5$ and other non-abelian group tasks, because specialized heads handle different group components (the Krohn-Rhodes decomposition emerges naturally through routing)
- **MQAR recall**: Improvement over fixed multi-head, because heads specialize — some become "memory heads" (long-range), others become "computation heads" (state transformation)
- **Language modeling**: $5$–$10\%$ perplexity improvement at matched compute, similar to the $2\times$ convergence speedup seen in expert-choice MoE for FFNs
- **Head specialization**: Different heads learn to process different token types — function words routed to fast-decay heads, content words to long-memory heads, punctuation/structure tokens to permutation heads
- **Scaling**: Benefit increases with $H$ (more heads = more opportunity for specialization)

**If hypothesis is wrong:**

- **No specialization emerges**: All heads process similar token distributions (routing entropy $\approx \log H$), suggesting SSM state dynamics don't benefit from token-level routing. This would be a useful negative result showing that SSMs need uniform processing unlike FFNs.
- **Quality degrades**: Tokens receiving zero heads (unprocessed) lose critical state information, and the model can't compensate. This would suggest a residual connection or minimum coverage constraint is needed.

## Minimum Viable Experiment

**Goal**: Demonstrate that expert-choice routing of SSM heads leads to meaningful head specialization and improved state-tracking accuracy compared to uniform multi-head processing.

### Setup

| Component | Configuration |
|-----------|---------------|
| Model | 2-layer Expert-Choice Monarch SSM |
| Heads | $H = 4$ |
| State dim per head | $N = 8$ |
| Hidden dim | $d = 32$ |
| Parameters | ~50K |
| Capacity factor | $c = 1$ (each token processed by 1 head on average) |
| Task | Multi-pattern state tracking |
| Data | 5K sequences, length 32 |
| Compute | Single GPU, $< 10$ minutes |

**Task: Multi-Pattern State Tracking**

Design a synthetic task that *requires* different processing for different tokens:

- **Pattern A tokens** (50%): Increment a counter modulo 5 (requires long-memory, additive dynamics)
- **Pattern B tokens** (30%): Apply a permutation from $S_3$ (requires group dynamics)
- **Pattern C tokens** (20%): Reset state to a fixed value (requires gating/forgetting)

Each token is labeled with its pattern type. The model must output the current state (counter value $\times$ permutation element $\times$ reset flag).

This task is inherently multi-modal: a single SSM head cannot efficiently handle all three patterns, but specialized heads can.

### Success Criteria

| Model | Expected Accuracy |
|-------|-------------------|
| Single-head SSM ($N = 32$) | $\sim 40$–$60\%$ (can handle some patterns, not all) |
| Uniform 4-head SSM ($N = 8$ each) | $\sim 50$–$70\%$ (redundant processing, no specialization) |
| **Expert-choice 4-head SSM** | $> 85\%$ (heads specialize per pattern type) |

**The idea works if:**
1. Expert-choice model achieves $> 85\%$ accuracy while uniform multi-head achieves $< 75\%$
2. Routing analysis shows $> 70\%$ of Pattern A tokens go to the same head(s), and similarly for B and C (measurable specialization)

### Failure Criteria

- **Kill the idea if**: Expert-choice performs no better than uniform multi-head (no specialization benefit)
- **Kill the idea if**: Training is unstable — expert-choice routing creates discontinuities that prevent SSM convergence
- **Pause and investigate if**: Expert-choice works but heads don't specialize (accuracy improves but routing is uniform) — suggests benefit comes from gating weights $G$, not routing

### Why This Test Is Sufficient

1. **Multi-pattern task forces specialization**: If the mechanism works, it must show specialization on this task — a single-head model cannot solve it efficiently
2. **Small scale tests the core mechanism**: The routing and Monarch factorization are scale-independent — if they work at $N = 8$, scaling adds capacity
3. **Measurable specialization**: We can directly inspect which heads process which token types, providing interpretable evidence
4. **10 minutes to signal**: Fast enough for rapid ablation of capacity factor, head count, and Monarch vs. diagonal transitions

### Implementation Sketch

```python
class ExpertChoiceSSMHead(nn.Module):
    """Expert-choice routing applied to SSM state heads."""
    def __init__(self, d_model, n_heads, state_dim, capacity_factor=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.capacity_factor = capacity_factor

        # Routing gate
        self.gate = nn.Linear(d_model, n_heads, bias=False)

        # Per-head Monarch-factored SSM parameters
        sqrt_n = int(state_dim ** 0.5)
        self.L_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(n_heads, sqrt_n, sqrt_n))
        ])
        self.R_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(n_heads, sqrt_n, sqrt_n))
        ])
        self.B = nn.Parameter(torch.randn(n_heads, state_dim, d_model))
        self.C = nn.Parameter(torch.randn(n_heads, d_model, state_dim))

    def forward(self, x):
        T, d = x.shape

        # Step 1: Token-to-head affinity
        S = F.softmax(self.gate(x), dim=-1)  # (T, H)

        # Step 2: Expert choice — heads select tokens
        k = int(T * self.capacity_factor / self.n_heads)
        G, I = torch.topk(S.T, k=k, dim=-1)  # (H, k)

        # Step 3: Per-head SSM on selected tokens
        outputs = []
        for h in range(self.n_heads):
            x_h = x[I[h]]  # (k, d) — selected tokens
            # Run SSM with Monarch-factored A
            o_h = self.run_ssm(h, x_h)  # (k, d)
            outputs.append(o_h * G[h].unsqueeze(-1))  # gate

        # Step 4: Scatter back
        y = torch.zeros_like(x)
        for h in range(self.n_heads):
            y.scatter_add_(0, I[h].unsqueeze(-1).expand(-1, d), outputs[h])

        return y
```

## Theoretical Analysis

**Complexity comparison:**

| Operation | Uniform Multi-Head | Expert-Choice Multi-Head |
|-----------|--------------------|--------------------------|
| Routing | $0$ | $O(THd_g)$ for affinity + $O(TH)$ for top-k |
| Per-head SSM | $O(TN\sqrt{N})$ per head | $O(kN\sqrt{N})$ per head, $k = Tc/H$ |
| Total SSM compute | $O(THN\sqrt{N})$ | $O(TcN\sqrt{N})$ |
| Scatter | $0$ | $O(THk) = O(T^2 c/H)$ |
| **Total** | $O(THN\sqrt{N})$ | $O(TcN\sqrt{N} + THd_g)$ |

**Key insight**: At $c = 1$, expert-choice SSM has total SSM compute $O(TN\sqrt{N})$ — independent of $H$! This means we can add more heads (each more specialized) without increasing total SSM computation. The only cost that grows with $H$ is the routing overhead $O(THd_g)$, which is small when $d_g \ll N\sqrt{N}$.

**Expressivity analysis:**

With $H$ specialized heads and capacity $c = 1$:
- Each token is processed by $\sim 1$ head on average (some by 0, some by 2+)
- The model has $H$ distinct state dynamics to choose from
- This is analogous to a mixture-of-SSMs with $H$ components and learned routing

By the Krohn-Rhodes theorem, any finite automaton decomposes into at most $O(|M| \cdot 2^{|M|})$ simple group + aperiodic layers. With $H$ specialized heads, the model can allocate different heads to different components of this decomposition, potentially simulating more complex automata than a monolithic SSM.

## Risks & Limitations

1. **Causal violation in routing**: Expert-choice routing requires seeing all tokens in the sequence (or at least a chunk) to compute affinities. For autoregressive training this is fine (full sequence is available), but for autoregressive *inference*, the model must either route based on past tokens only or use a chunk-based routing scheme. Mitigation: apply expert-choice routing at the chunk level during inference.

2. **Tokens with zero heads**: At $c = 1$, ~37% of tokens may receive zero heads (Poisson statistics). These tokens contribute nothing to the output. Mitigation: add a residual connection $y_t = x_t + \text{ExpertChoiceSSM}(x)_t$ so unprocessed tokens pass through unchanged.

3. **Routing gradient**: The top-$k$ operation in expert-choice is not differentiable. The gating weights $G$ provide gradient to the routing gate, but index selection $I$ does not. This is the same limitation as in MoE — it works in practice but the routing may be suboptimal.

4. **Sequence order**: Standard SSMs process tokens in order. Expert-choice routing gathers non-contiguous tokens to each head. The SSM must be applied to these tokens *in their original sequential order* (not in the order selected), requiring sorting by position index within each head's selected set.

5. **Memory**: Storing affinity scores $S \in \mathbb{R}^{T \times H}$ and indices $I \in \mathbb{Z}^{H \times k}$ adds $O(TH)$ memory. For large $T$ and $H$, this may compete with state memory.

6. **Comparison fairness**: At $c = 1$, expert-choice SSM uses less total SSM compute than uniform multi-head. For fair comparison, the baseline should either have fewer heads or smaller state dimension to match total FLOPs.

## Follow-up Experiments

1. **Compose with Proposal 009 (Post-Sigmoid Gating)**: Apply post-readout sigmoid gating to the scattered output. This addresses the low-rank bottleneck in the scatter-back step where multiple heads' outputs are summed.

2. **Adaptive capacity factor**: Learn $c$ per layer — early layers may benefit from more uniform processing ($c = 2$), while later layers may benefit from sharp specialization ($c = 0.5$).

3. **Hierarchical routing + Log-Linear Attention**: Combine expert-choice SSM heads with log-linear attention's hierarchical structure. Different heads could operate at different temporal resolutions in the Fenwick tree hierarchy.

4. **Scale to 1B+ parameters**: Test whether head specialization patterns persist at scale, and whether the $2\times$ convergence speedup observed in MoE FFN transfers to MoE SSM heads.

5. **Interpretability**: Analyze what linguistic properties (POS, syntax, semantics) drive routing decisions — this could reveal what SSM dynamics are needed for different aspects of language.

6. **Dynamic head count**: Use a learned "number of heads" per token (via Gumbel-softmax on the affinity distribution) to implement truly adaptive computation depth for sequence modeling.

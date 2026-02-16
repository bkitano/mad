---
status: ongoing
priority: high
created: 2026-02-10
based_on: column-sparse-transition-matrices, negative-eigenvalue-extension, input-dependent-gating, wy-representation
experiment_number: 001
---

# Column-Sparse Negative-Eigenvalue DeltaNet

## Hypothesis

Combining column-sparse transition matrices (PD-SSM) with the negative eigenvalue extension in a DeltaNet-style architecture will achieve strictly greater expressivity than either technique alone, enabling simulation of automata over non-solvable groups with sign-flipping dynamics, while maintaining $O(N)$ per-step complexity.

## Background

Two recent tricks independently expand SSM expressivity for state-tracking tasks, but target different bottlenecks:

1. **Column-sparse transition matrices** (PD-SSM) replace diagonal $A$ with column-sparse $P \cdot D$ factorization, enabling permutation routing between state dimensions. This lets the model emulate non-solvable group automata (e.g., $S_5$), which diagonal SSMs provably cannot represent. However, PD-SSM's diagonal component $D$ still has entries with $|D_{ii}| < 1$, restricting eigenvalues to contraction.

2. **Negative eigenvalue extension** multiplies DeltaNet's $\beta$ by 2, extending the range to $(0, 2)$ so that state matrix eigenvalues can become negative. This unlocks $\text{NC}^1$ state tracking (e.g., permutation composition) for DeltaNet. However, it operates within DeltaNet's rank-1 outer-product update structure, which cannot represent arbitrary permutations.

**The gap**: Neither trick alone gives both (a) permutation routing between states AND (b) sign-flipping/oscillatory dynamics within state channels. Combining them could produce a model that can track automata requiring both capabilities — for example, dihedral groups $D_n$ (rotations + reflections) or wreath products where state routing and sign changes interact.

**Why DeltaNet as the base**: DeltaNet's delta-rule update $S_t = S_{t-1} + \beta_t k_t(v_t - S_t^\top k_t)^\top$ provides a natural place to inject both tricks. The column-sparse structure can replace the implicit state routing, while the negative eigenvalue extension affects the $\beta$ gating. The WY representation enables efficient chunkwise computation.

## Mathematical Formulation

**Standard DeltaNet:**

$$
S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top
$$

**CS-NEG-DeltaNet (proposed):**

$$
S_t = P(x_t) \cdot S_{t-1} + \beta_t \cdot k_t (v_t - S_{t-1}^\top k_t)^\top
$$

**Key Variables:**

- $P(x_t) \in \{0,1\}^{d \times d}$ — input-dependent column one-hot permutation matrix (from PD-SSM), trained via Gumbel-softmax
- $\beta_t = 2 \cdot \sigma(x_t W_\beta) \in (0, 2)$ — extended learning rate (from negative eigenvalue extension)
- $k_t, v_t \in \mathbb{R}^d$ — key/value projections as in standard DeltaNet
- $S_t \in \mathbb{R}^{d \times d}$ — state matrix

The permutation $P$ acts as a "state routing" step before the delta-rule update, enabling the model to reindex which memory slots get updated. The extended $\beta$ allows the update itself to overshoot ($\beta > 1$), creating negative eigenvalues in the effective state transition.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Base model | DeltaNet |
| State routing | Column-sparse $P(x_t)$ via Gumbel-softmax |
| Learning rate | $\beta_t \in (0, 2)$ (extended) |
| State dim | $d = 128$–$256$ |
| Layers | $L = 4$–$8$ |
| Parameters | ~5M–20M |

**WY Representation for Efficiency:**

Within each chunk of size $C$, accumulate the delta-rule updates as thin $W, Y$ factors:
$$
S_t = I + W_t Y_t^\top
$$

At chunk boundaries, materialize $S_t$ and apply the permutation routing.

### Baseline

1. **Standard DeltaNet** ($\beta \in (0,1)$, no permutation routing)
2. **NEG-DeltaNet** ($\beta \in (0,2)$, no permutation routing) — isolates the negative eigenvalue contribution
3. **PD-SSM (Mamba-style)** (column-sparse $A$, no delta rule) — isolates the column-sparse contribution
4. **Gated DeltaNet** — current SOTA DeltaNet variant

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ composition accuracy | $> 95\%$ | Synthetic benchmark |
| $D_8$ (dihedral) accuracy | $> 90\%$ | Requires both tricks |
| Perplexity (WikiText-103) | Competitive | Language modeling sanity check |
| LRA average | $\geq 85\%$ | Long-range benchmark |

**State tracking benchmarks:**
- $S_5$ (symmetric group on 5 elements) — non-solvable, requires permutation expressivity
- $D_8$ (dihedral group of order 8) — requires both permutation AND sign-flip
- Modular arithmetic ($\mathbb{Z}/n\mathbb{Z}$) — abelian baseline, should not need either trick
- Wreath product $S_3 \wr \mathbb{Z}/2\mathbb{Z}$ — requires interaction of routing and sign

### Estimated Compute

**Small**. All evaluations use small models ($d_{\text{model}} = 128$–$256$, 4–8 layers, ~5M–20M parameters). Synthetic automaton tasks train in $<1$ GPU-hour on A100. WikiText-103 and LRA are standard small-scale benchmarks ($<24$ GPU-hours total).

## Expected Outcome

**If hypothesis is correct:**

- On **$S_5$ composition**: PD-SSM baseline achieves near-perfect; NEG-DeltaNet achieves moderate accuracy; CS-NEG-DeltaNet matches or exceeds PD-SSM
- On **$D_8$ (dihedral)**: CS-NEG-DeltaNet significantly outperforms both individual techniques, because $D_8$ requires both permutation routing (reflections swap elements) AND sign changes (orientation flips)
- On **wreath products**: CS-NEG-DeltaNet outperforms both baselines, demonstrating compounding expressivity
- On **language modeling**: Small improvement or parity with Gated DeltaNet — expressivity gains are most visible on structured tasks
- On **LRA**: Competitive or slightly better due to richer state dynamics

**If hypothesis is wrong:**

- The $D_8$ and wreath product results would show no improvement over individual techniques, suggesting the two expressivity tricks are redundant or interfere with each other

## Minimum Viable Experiment

**Goal**: Demonstrate that the combination of column-sparse routing + negative eigenvalues enables capabilities that neither trick alone provides, using the smallest possible setup.

### Setup

| Component | Configuration |
|-----------|---------------|
| Model | 1-layer CS-NEG-DeltaNet |
| State dim | $d = 16$ |
| Hidden dim | $d_{\text{model}} = 32$ |
| Parameters | ~10K |
| Task | $D_4$ (dihedral group of order 4) state tracking |
| Data | 5K sequences of length 20 |
| Compute | Single GPU, $< 5$ minutes |

**Why $D_4$?** The dihedral group $D_4$ (symmetries of a square) is the smallest group requiring both permutation (rotations swap corners) AND sign-flipping (reflections change orientation). It has only 8 elements, making it trivial to enumerate and verify.

### Task Definition

Input: Sequence of $D_4$ generators $(r, r, s, r, s, \ldots)$ where $r$ = rotation, $s$ = reflection.

Output: Current group element after applying all generators.

$$
g_t = g_{t-1} \cdot x_t, \quad g_0 = e \text{ (identity)}
$$

### Success Criteria

| Model | Expected Accuracy |
|-------|-------------------|
| Standard DeltaNet | $< 50\%$ (random guessing = 12.5%) |
| NEG-DeltaNet | $\sim 60$–$70\%$ (can handle sign, not routing) |
| PD-SSM | $\sim 60$–$70\%$ (can handle routing, not sign) |
| **CS-NEG-DeltaNet** | $> 90\%$ (can handle both) |

**The idea works if**: CS-NEG-DeltaNet achieves $> 90\%$ accuracy while both ablations (NEG-only, CS-only) achieve $< 75\%$.

### Failure Criteria

- **Kill the idea if**: CS-NEG-DeltaNet performs no better than the best single-trick baseline
- **Kill the idea if**: Training is unstable and doesn't converge within 1000 steps
- **Pause and investigate if**: CS-NEG works but so does one of the ablations (suggests $D_4$ isn't the right test)

### Why This Test Is Sufficient

1. **$D_4$ is the minimal test case**: It's the smallest non-abelian group with both rotation and reflection structure
2. **If the mechanism works at $d = 16$, scaling adds capacity**: The expressivity gap is fundamental, not a matter of scale
3. **5 minutes to signal**: We can iterate quickly on architecture choices before committing to full experiments
4. **Clear ablation structure**: Running 4 models (full, NEG-only, CS-only, baseline) takes $< 20$ minutes total

### Implementation Sketch

```python
# D_4 group elements: {e, r, r², r³, s, sr, sr², sr³}
# r = 90° rotation, s = reflection
# Relations: r⁴ = e, s² = e, srs = r⁻¹

def d4_multiply(g1, g2):
    # Implement D_4 group multiplication table
    ...

def generate_d4_sequence(length=20):
    generators = ['r', 's']
    seq = [random.choice(generators) for _ in range(length)]
    target = reduce(d4_multiply, seq, 'e')
    return seq, target

# Train tiny CS-NEG-DeltaNet on this task
```

## Theoretical Analysis

**Expressivity Comparison:**

| Model | Eigenvalue Range | Group Expressivity |
|-------|------------------|-------------------|
| Diagonal SSM | $[0, 1]$ | Abelian only ($\text{TC}^0$) |
| NEG-DeltaNet | $(-1, 1)$ | $\text{NC}^1$ (permutation composition) |
| PD-SSM | $[0, 1]$ with routing | Non-solvable groups ($S_5$) |
| **CS-NEG-DeltaNet** | $(-1, 1)$ with routing | Non-solvable + signed ($D_n$, wreath) |

## Risks & Limitations

1. **Training instability**: Combining Gumbel-softmax (for $P$) with extended $\beta$ range could create optimization challenges. Mitigation: warmup $\beta$ from $(0,1)$ to $(0,2)$ gradually; anneal Gumbel temperature.

2. **Interaction effects**: The permutation $P$ and delta-rule update may interfere — if $P$ scrambles state dimensions right before an update, the key-value associative structure of DeltaNet could be disrupted. Need to ablate $P$ placement (before vs. after delta update).

3. **WY representation compatibility**: The WY factorization assumes updates are rank-1 outer products. The permutation $P$ is a structural change, not a rank-1 update, so it may not fold cleanly into the WY form. May need to materialize at every chunk boundary where $P$ changes, reducing efficiency.

4. **Gumbel-softmax quality**: Discrete permutation selection via Gumbel-softmax may not train well at large $N$. The original PD-SSM paper may use straight-through estimators; need to compare both.

## Follow-up Experiments

1. **Ablate $P$ placement**: Test $P$ before delta update vs. $P$ after delta update vs. $P$ at chunk boundaries only
2. **Continuous relaxation**: Replace column one-hot $P$ with doubly stochastic matrix (Sinkhorn) to remove discrete training challenges, measure expressivity trade-off
3. **Scale to language modeling**: If synthetic results are positive, test on larger LM benchmarks (The Pile, 350M–1.3B params)
4. **Theoretical analysis**: Characterize the formal language class recognizable by CS-NEG-DeltaNet — is it strictly larger than $\text{NC}^1$?
5. **Combine with PTD initialization**: If the model uses HiPPO-style state initialization, apply PTD for better conditioning

## Human Review

By Cartan-Dieudonne, any $n \times n$ orthogonal matrix is the composition of at most $n$ Householder transformations. So in regular DeltaNet, since we're applying a single Householder transformation at each state update, we're learning a composition of them as we unroll all the states, and so we're *already* learning an orthogonal matrix, it's just spread across steps.

The $P(x_t)$ input-dependent permutation doesn't readily admit a matmul form that's efficient for training. So we'd rely on efficient approximations, one being Householder decomposition. In that case, at each step $t$, we'd do

$$
P(x_t) = \prod_{j=1}^{n_h} (I - \beta_t k_{j,t} k_{j,t}^\top)
$$

This is *exactly* DeltaProduct.

So Deltanet applies permutations, just slowly; DeltaProduct applies permutations quickly, but doesn't have the ability to parallelize in the sequence dimension (necessary for efficient training).

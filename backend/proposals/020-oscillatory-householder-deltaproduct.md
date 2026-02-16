---
status: ongoing
priority: high
created: 2026-02-15
based_on: oscillatory-eigenvalue-stability, householder-product-parameterization, negative-eigenvalue-extension, wy-representation, ut-transform-householder-accumulation, cayley-contractive-parameterization, chunkwise-parallel-scan, cartan-dieudonne-decomposition
experiment_number: 020
---

# Oscillatory Householder DeltaProduct (OH-DeltaProduct)

## Hypothesis

Decomposing the DeltaProduct state-transition matrix into an explicit **oscillatory component** (physics-based stable rotations) and a **reflective component** (Householder products for state-tracking) — where the oscillatory part handles long-range memory retention through unit-eigenvalue energy conservation while the reflective part handles discrete state manipulation through sign-flipping — will achieve both DeltaProduct-level state-tracking expressivity ($S_5$ composition) AND LinOSS-level stability (no NaN/divergence), while the two components' interactions remain efficient through the WY/UT transform machinery.

## Background

There is a fundamental tension in linear RNN design between **stability** and **state-tracking expressivity**:

- **LinOSS** (oscillatory SSM) achieves provable stability ($|\lambda| \leq 1$ by construction) through the physics of damped harmonic oscillators, but its diagonal-per-oscillator structure limits it to $\text{TC}^0$ — it cannot simulate non-abelian group automata like $S_5$.
- **DeltaProduct** achieves $\text{NC}^1$ state-tracking via products of Householder reflections, but stability depends on careful $\beta$ parameterization and can fail (the $\beta \in (0, 2)$ extension for negative eigenvalues increases instability risk). As noted in the human review of Proposal 001, DeltaNet already learns orthogonal matrices through composed reflections across steps — DeltaProduct accelerates this by doing $n_h$ reflections per step.
- **Gated DeltaNet** (Proposal 007, OscGate-SSM) makes oscillatory parameters input-dependent but doesn't incorporate the Householder product structure that enables efficient state-tracking.

**Key insight from Cartan-Dieudonné**: Every orthogonal matrix decomposes into at most $n$ Householder reflections. But orthogonal matrices form a group, and the group decomposes further: every element of $O(n)$ is either a pure rotation ($\det = +1$, element of $SO(n)$) or a rotation composed with a single reflection ($\det = -1$). The oscillatory parameterization naturally captures the rotation part (conjugate eigenvalue pairs on the unit circle), while Householder reflections capture the reflection part (eigenvalue $-1$).

**This proposal makes the decomposition explicit**: use LinOSS oscillators for the rotational part and DeltaProduct Householder reflections for the reflective/state-routing part. The two components compose multiplicatively, and the product inherits stability from the oscillatory factor and expressivity from the reflective factor.

**Gap filled**: No existing proposal combines oscillatory stability with Householder expressivity. Proposal 004 uses oscillatory DPLR but with fixed (not input-dependent) matrices and no Householder structure. Proposal 007 makes oscillatory parameters input-dependent but stays diagonal. Proposal 001 uses negative eigenvalues but without the oscillatory stability guarantee.

## Mathematical Formulation

**LinOSS Oscillatory Component (rotation):**

For $m$ oscillators with implicit discretization:

$$
R_t = \begin{pmatrix} S_t & -\Delta t \cdot A_t \cdot S_t \\ \Delta t \cdot S_t & S_t \end{pmatrix} \in \mathbb{R}^{2m \times 2m}
$$

where $A_t = \text{ReLU}(\hat{A}(x_t)) \geq 0$ is input-dependent (selective), $S_t = (I + \Delta t^2 A_t)^{-1}$ is diagonal, and eigenvalues satisfy $|\lambda_j| = \sqrt{S_{kk}} \leq 1$ by construction.

This is a block-diagonal matrix of $2 \times 2$ rotation-contraction blocks:

$$
R_t = \text{diag}\left( \begin{pmatrix} s_k & -\Delta t \cdot a_k \cdot s_k \\ \Delta t \cdot s_k & s_k \end{pmatrix} \right)_{k=1}^{m}
$$

**Householder Reflective Component (state-tracking):**

Apply $n_h$ input-dependent Householder reflections:

$$
H_t = \prod_{j=1}^{n_h} (I - \beta_{t,j} \, k_{t,j} \, k_{t,j}^\top)
$$

where $k_{t,j} = \text{normalize}(W_k^{(j)} x_t) \in \mathbb{R}^{2m}$ and $\beta_{t,j} = 2\sigma(w_\beta^{(j)} \cdot x_t) \in (0, 2)$.

With $\beta \in (0, 2)$: eigenvalues of each factor are $1 - \beta$ (along $k$) and $1$ (orthogonal complement). The product $H_t$ has eigenvalues with $|\lambda| \leq 1$ when $\beta \in [0, 2]$.

**Composed State Transition:**

$$
\tilde{A}_t = H_t \cdot R_t
$$

This is the product of a near-orthogonal matrix ($H_t$) and a contractive rotation ($R_t$).

**Stability Guarantee (Proposition):**

$$
\|\tilde{A}_t\|_2 = \|H_t R_t\|_2 \leq \|H_t\|_2 \cdot \|R_t\|_2 \leq 1 \cdot 1 = 1
$$

since:
- $\|H_t\|_2 \leq \prod_j \|I - \beta_{t,j} k_{t,j} k_{t,j}^\top\|_2 \leq \prod_j \max(|1 - \beta_{t,j}|, 1) \leq 1$ when $\beta_{t,j} \in [0, 2]$
- $\|R_t\|_2 = \max_k \sqrt{S_{kk}} \leq 1$ (LinOSS construction)

Therefore: $|\lambda(\tilde{A}_t)| \leq \|\tilde{A}_t\|_2 \leq 1$ — stability by construction.

**Expressivity Argument:**

- $R_t$ alone can represent any rotation within the $2 \times 2$ block structure (capturing frequency content)
- $H_t$ with $n_h \geq m$ reflections can represent any element of $O(2m)$ (Cartan-Dieudonné)
- The composition $H_t R_t$ can represent any product of an orthogonal and a contraction — sufficient for $\text{NC}^1$ state-tracking since $S_5$ embeds in $O(5)$ via the Cayley representation, and $O(5) \subset O(2m)$ for $m \geq 3$

**State Update (Full Recurrence):**

$$
h_t = \tilde{A}_t h_{t-1} + B_t x_t = H_t R_t h_{t-1} + B_t x_t
$$

**Chunkwise Parallel Training via WY + UT:**

Within a chunk of size $C$, the Householder component uses the WY representation:

$$
H_{t_0:t_0+C} = I + W_{1:C} Y_{1:C}^\top
$$

where $W, Y \in \mathbb{R}^{2m \times C}$ are accumulated via the UT transform (converting sequential dot products to tensor-core matmuls). The oscillatory component $R_t$ is diagonal and parallelizable via standard scan.

The full chunk computation:
1. **Oscillatory scan**: Parallel scan over $(R_{t_0}, \ldots, R_{t_0+C-1})$ — diagonal, $O(m \cdot C)$ via standard Blelloch scan
2. **Householder accumulation**: UT transform to build $W, Y$ — $O(C^2 \cdot 2m)$ matmuls on tensor cores
3. **Compose**: Apply $H \cdot R$ per-chunk product at chunk boundaries — $O(m^2 \cdot C/Q)$ inter-chunk products
4. **Output**: $y_t = C_t h_t$ — standard projection

**Key Variables:**
- $h_t \in \mathbb{R}^{2m}$ — state vector ($2m$ for position + velocity of $m$ oscillators)
- $R_t \in \mathbb{R}^{2m \times 2m}$ — oscillatory rotation-contraction (block-diagonal)
- $H_t \in \mathbb{R}^{2m \times 2m}$ — Householder product ($n_h$ reflections)
- $\tilde{A}_t = H_t R_t$ — composed transition
- $n_h$ — number of Householder reflections per step (controls expressivity/compute trade-off)
- $m$ — number of oscillators (state dim is $2m$)
- $C$ — chunk size for parallel training

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | OH-DeltaProduct |
| Layers | $L = 6$–$12$ |
| Hidden dim | $d = 256$–$512$ |
| Oscillators | $m = 32$ per head (state dim $2m = 64$) |
| Heads | 8 |
| Householder reflections | $n_h = 4$ (sufficient for $S_5$) |
| Chunk size | $C = 64$ |
| $\beta$ range | $(0, 2)$ (negative eigenvalue extension) |
| Oscillatory $A_t$ | Input-dependent: $A_t = \text{ReLU}(W_A x_t)$ |
| Readout | $y_t = C h_t + D x_t$ (skip connection) |

### Baseline

1. **DeltaProduct** ($n_h = 4$, no oscillatory component): Tests whether the oscillatory component adds value beyond Householder products alone. Complexity: $O(T \cdot 2m \cdot n_h)$
2. **LinOSS** ($m = 32$, no Householder): Tests whether Householder adds value beyond oscillatory alone. Complexity: $O(T \cdot 2m)$
3. **Gated DeltaNet** (single reflection, input-dependent gating): Standard baseline from Yang et al. Complexity: $O(T \cdot d)$
4. **Mamba-2** (diagonal selective SSM): Standard efficiency baseline. Complexity: $O(T \cdot n)$
5. **OH-DeltaProduct ablation** ($\beta \in (0, 1)$ only, no negative eigenvalues): Tests necessity of the negative eigenvalue extension

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ composition accuracy | $> 95\%$ (1 layer) | 5-element permutation group |
| $D_{16}$ dihedral group | $> 90\%$ | Dihedral group word problem |
| WikiText-103 perplexity | $\leq$ DeltaProduct | Validation perplexity |
| Training stability | 0 NaN/Inf | Count over 100K steps |
| Throughput | $\geq 0.7\times$ Mamba-2 | Tokens/sec on A100 |
| Length generalization | Train 512 → eval 2048 | Accuracy retention |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (state-tracking benchmarks): ~30 GPU-hours on A100
**Phase 2** (language modeling, 350M params): ~120 GPU-hours on A100
**Total**: ~150 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- OH-DeltaProduct matches DeltaProduct on $S_5$ while having zero NaN events (DeltaProduct with $\beta \in (0,2)$ occasionally diverges)
- OH-DeltaProduct significantly outperforms LinOSS on state-tracking ($> 30\%$ gap on $S_5$)
- Length generalization is superior to DeltaProduct: the oscillatory component maintains energy at long sequences while Householder products provide the discrete routing
- Perplexity matches or exceeds DeltaProduct: the explicit rotational component is a better inductive bias for periodic/rhythmic patterns in language than implicit rotations from composed reflections
- The oscillatory and reflective components specialize: $R_t$ captures smooth temporal dynamics, $H_t$ activates for discrete state changes (e.g., entity tracking, scope changes)

**If hypothesis is wrong:**
- If DeltaProduct alone matches: the oscillatory component is redundant because Householder products already implicitly learn rotations. This would confirm that the Cartan-Dieudonné decomposition into rotations + reflections doesn't provide a useful inductive bias
- If LinOSS alone matches on non-tracking tasks: Householder products only help for state-tracking, which isn't needed for language modeling. This would narrow the use case to algorithmic reasoning
- If throughput is too low ($< 0.5\times$ Mamba-2): the per-step cost of $n_h$ reflections plus oscillatory computation is too high for practical use. This would motivate exploring cheaper approximations (e.g., $n_h = 1$ with oscillatory, or shared $k$ vectors across steps)

## Minimum Viable Experiment

### Setup
- **Model**: 1-layer OH-DeltaProduct, $m = 8$ oscillators (state dim $2m = 16$), $n_h = 2$ reflections, $d = 32$, ~50K params
- **Task**: $S_3$ permutation composition — compose sequences of 3-element permutations, predict resulting permutation
- **Data**: 5K sequences of length 32 (each step applies a random generator of $S_3$)
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- OH-DeltaProduct achieves $> 95\%$ accuracy on $S_3$ composition
- Pure LinOSS (same $m = 8$, $n_h = 0$) achieves $< 40\%$ (cannot do non-abelian state-tracking)
- Pure DeltaProduct ($n_h = 2$, no oscillatory) also achieves $> 90\%$ BUT has $> 5\%$ NaN rate when $\beta \in (0, 2)$
- OH-DeltaProduct has $0\%$ NaN rate (stability from oscillatory factor)
- The ablation with $\beta \in (0, 1)$ only achieves $< 60\%$ (confirms need for negative eigenvalues)

### Failure Criteria
- If OH-DeltaProduct can't solve $S_3$: the composition $H_t R_t$ somehow interferes with state-tracking (e.g., the oscillatory rotation "smears" the Householder reflection's discrete routing). The mechanism is broken.
- If DeltaProduct never produces NaN at this scale: the stability benefit can't be demonstrated. Scale up to longer sequences ($T = 256$) where instability is more likely.

### Why This Test Is Sufficient
- $S_3$ is the simplest non-abelian group, requiring negative eigenvalues and non-commutative state manipulation. If the mechanism handles $S_3$, scaling to $S_5$ adds capacity not capability.
- The stability test is meaningful even at small scale: if $\beta \in (0, 2)$ causes NaN at $T = 32$, the oscillatory stabilization is immediately relevant.
- The UT transform and WY representation work identically at any scale — if chunked training works for $C = 16$, it works for $C = 64$.
- The oscillatory + Householder composition is a per-step operation — scale doesn't change the mechanism, only the capacity.

## Theoretical Analysis

Complexity comparison (per token, single head):

| Operation | LinOSS | DeltaProduct ($n_h$ refl) | OH-DeltaProduct |
|-----------|--------|--------------------------|-----------------|
| State transition | $O(m)$ | $O(m \cdot n_h)$ | $O(m + m \cdot n_h)$ |
| Stability | By construction | Requires $\beta$ control | By construction |
| State-tracking | $\text{TC}^0$ | $\text{NC}^1$ | $\text{NC}^1$ |
| Parameters per step | $O(m \cdot p)$ | $O(m \cdot n_h)$ | $O(m(p + n_h))$ |
| Chunkwise parallel | $O(C \cdot m)$ scan | $O(C^2 \cdot m)$ UT | $O(C \cdot m + C^2 \cdot m)$ |

where $p$ = input projection dimension for oscillatory frequencies.

**Expressivity analysis:**

The reachable set of $\tilde{A}_t = H_t R_t$ where $H_t \in O(2m)$ (via $n_h = 2m$ reflections) and $R_t$ is the LinOSS contraction is:

$$
\mathcal{R} = \{H \cdot R : H \in O(2m), R \in \mathcal{R}_{\text{osc}}\}
$$

where $\mathcal{R}_{\text{osc}} = \{R : R \text{ is block-diagonal LinOSS with } \|R\|_2 \leq 1\}$. Since $O(2m)$ acts transitively on the unit sphere and $\mathcal{R}_{\text{osc}}$ includes the identity (when $A_k = 0$), we have $\mathcal{R} \supseteq O(2m)$ — the full orthogonal group is reachable.

**Gradient flow analysis:**

The gradient through the composed transition at step $t$ involves:

$$
\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^{T} \tilde{A}_t^\top \cdot \frac{\partial \mathcal{L}}{\partial h_T}
$$

Since $\|\tilde{A}_t\|_2 \leq 1$ for all $t$, the gradient norm is non-increasing: $\|\nabla_{h_0} \mathcal{L}\| \leq \|\nabla_{h_T} \mathcal{L}\|$. This prevents gradient explosion. Gradient vanishing is mitigated when $\|\tilde{A}_t\|_2 \approx 1$ (near-orthogonal), which the oscillatory component encourages when $A_k$ is small (near-energy-conserving).

## Risks & Limitations

1. **Interaction between $R_t$ and $H_t$**: The oscillatory rotation might "rotate away" from the subspace the Householder reflections are targeting, causing the two components to fight rather than cooperate. Mitigation: initialize $A_k$ small so $R_t \approx I$ initially, letting Householder products dominate early training; the oscillatory component develops gradually.

2. **Parameter overhead**: OH-DeltaProduct uses parameters for both oscillatory frequencies ($O(m)$) and Householder vectors ($O(m \cdot n_h)$). Total: $O(m(1 + n_h))$. For $n_h = 4$, this is $5\times$ the parameter cost of LinOSS. Mitigation: the increased expressivity should justify the cost; compare at equal total parameters by reducing $m$.

3. **UT transform compatibility**: The UT transform assumes updates of the form $I - \beta k k^\top$. In OH-DeltaProduct, the Householder reflections act on $R_t h_{t-1}$ rather than $h_{t-1}$ directly. The recurrence is:
   $$h_t = H_t R_t h_{t-1} + B_t x_t$$
   This requires interleaving the diagonal scan ($R_t$) with the Householder accumulation ($H_t$). Within a chunk, we can first apply the oscillatory scan to get intermediate states $\tilde{h}_{t} = R_t h_{t-1}$, then apply the Householder WY representation. This is a two-pass algorithm: scan then reflect.

4. **Doubled state dimension**: Like LinOSS, the oscillatory component doubles the effective state dimension ($2m$ instead of $m$). Combined with $n_h$ reflections, the cost is $O(2m \cdot n_h)$ per step vs $O(m \cdot n_h)$ for pure DeltaProduct.

5. **Chunk boundary cost**: At chunk boundaries, the composed state $\tilde{A}_{t:t+C} = H_{t:t+C} R_{t:t+C}$ must be materialized as a $2m \times 2m$ matrix. For $2m = 64$, this is 4K entries — manageable but nontrivial.

## Follow-up Experiments

1. **IMEX variant (symplectic)**: Use the IMEX discretization instead of IM for the oscillatory component, giving $|\lambda| = 1$ exactly (energy-conserving). Combined with $\beta = 2$ Householder reflections (also $|\lambda| = 1$), the entire model becomes volume-preserving — a discrete symplectic map. This could excel at tasks requiring perfect long-range memory.

2. **Gated decomposition**: Instead of fixed $H_t R_t$ composition, learn a gating parameter $\gamma_t \in [0, 1]$ that interpolates: $\tilde{A}_t = \gamma_t H_t + (1 - \gamma_t) R_t$. This lets the model choose per-token whether to rotate (smooth dynamics) or reflect (discrete state change).

3. **Wreath product interpretation**: From Krohn-Rhodes, the oscillatory component captures the group part (rotation) and the Householder reflection with $\beta \in (0, 2)$ captures both group and aperiodic parts (since $\beta = 1$ gives the rank-1 idempotent $I - kk^\top$, analogous to $U_2$). The composition $H_t R_t$ is a wreath-product-like structure — formalize this connection and test whether it helps on automata tasks beyond $S_5$.

4. **Scale-dependent $n_h$**: Use $n_h = 1$ in early layers (cheap, sufficient for simple patterns) and $n_h = 4$ in later layers (expensive, needed for complex state-tracking). Guided by the importance scoring from Proposal 018.

5. **OH-DeltaProduct + SSD**: Integrate with Proposal 002's SSD-DeltaNet framework to get efficient chunkwise parallelism via semiseparable block decomposition, treating $R_t$ as the diagonal part and $H_t - I$ as the low-rank correction.

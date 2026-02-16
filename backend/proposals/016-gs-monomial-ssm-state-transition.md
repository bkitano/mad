---
status: ongoing
priority: high
created: 2026-02-15
based_on: group-and-shuffle-matrices, monomial-matrix-closure, cayley-contractive-parameterization, input-dependent-gating, chunkwise-parallel-scan, recurrence-to-scan-reduction, blelloch-work-efficient-scan, signed-permutation-hyperoctahedral-group
experiment_number: 016
---

# Group-and-Shuffle Monomial SSM: Structured Orthogonal State Transitions via GS Factorization

## Hypothesis

Parameterizing SSM state transitions as **Group-and-Shuffle (GS) monomial matrices** — block-diagonal monomial (permutation $\times$ diagonal) factors interleaved with a fixed shuffle permutation — achieves the coordinate-mixing expressivity of dense transitions and the parallel-scan efficiency of diagonal SSMs, at $O(n \sqrt{n})$ cost per step with only **2 factors** (vs 6 for block butterfly), by exploiting the $O(n)$ closure of monomial matrix products within each block and the GS-proven completeness guarantee for orthogonal coverage.

## Background

### The State-Transition Expressivity Hierarchy

Existing proposals for non-diagonal SSM state transitions form a clear hierarchy:

| Method | Cost/step | Scan cost | Expressivity | Proposals |
|--------|-----------|-----------|-------------|-----------|
| Diagonal | $O(n)$ | $O(Tn \log T)$ | Abelian only (solvable groups) | Mamba, S4D, S5 |
| Column-sparse (PD-SSM) | $O(n)$ | $O(Tn \log T)$ | Full $S_n$ (via monomial closure) | 001, 003 |
| Monarch | $O(n\sqrt{n})$ | $O(Tn\sqrt{n} \log T)$ | BMM-structured dense | 006, 010, 012 |
| Circulant (FFT) | $O(n \log n)$ | $O(Tn \log n \log T)$ | Commutative (cyclic $\mathbb{Z}_n$) | 013 |
| Dense | $O(n^2)$ | $O(Tn^2 \log T)$ | Full $\text{GL}(n)$ | — |

**The gap**: Column-sparse (monomial) matrices achieve $O(n)$ but can only route one state dimension to one other — no weighted mixing. Monarch matrices achieve rich mixing at $O(n\sqrt{n})$ but (a) require $O(n)$ parameters per block and (b) products of Monarch matrices are NOT closed (the product of two Monarch matrices is generally dense), preventing efficient cumulative-product scans. **No existing proposal achieves both sub-quadratic mixing AND efficient scan closure.**

### The GS Insight: Fewer Factors, Same Coverage

The Group-and-Shuffle (GS) framework (Gorbunov et al., NeurIPS 2024) proves that for $n = b \cdot r$ (block size $b$, number of blocks $r$):

$$
m = 1 + \lceil \log_b r \rceil \text{ GS factors suffice to cover all orthogonal matrices}
$$

For $n = 256$ with $b = 16$ ($r = 16$): GS needs only $m = 2$ factors vs block butterfly's $m = 6$. Each GS factor is $A_i = P_{L,i}(L_i P_i R_i) P_{R,i}$ with block-diagonal $L_i, R_i$.

### The Monomial Closure Insight

The documented monomial-matrix-closure trick shows that monomial matrices $(P \cdot D)$ are closed under multiplication in $O(n)$:

$$
(P_2 D_2)(P_1 D_1) = P_{2 \circ 1} \cdot D_{2 \circ 1}, \quad \text{where } \sigma_{2 \circ 1} = \sigma_2 \circ \sigma_1, \;\; d_{2 \circ 1}[j] = d_2[\sigma_1(j)] \cdot d_1[j]
$$

### Our Key Idea: GS with Monomial Blocks

We propose making the block-diagonal factors $L_i, R_i$ in the GS decomposition themselves be **monomial** (block-permutation $\times$ block-diagonal) rather than dense. This gives:

1. **Each block is monomial** ($b \times b$ with $O(b)$ params and $O(b)$ mat-vec)
2. **The shuffle $P$ provides cross-block mixing** (the GS completeness guarantee ensures this suffices)
3. **Products of GS-monomial matrices remain GS-monomial** within each block, enabling efficient cumulative products

The critical innovation: while the full $n \times n$ GS matrix product is NOT closed in the GS class (just as Monarch products aren't closed), we can **compute the cumulative product in a structured way** by tracking per-block monomial states and inter-block shuffle compositions separately.

### Why This Is Different From Existing Proposals

- **Proposal 001** (CS-NEG-DeltaNet): Uses column-sparse (monomial) transitions but without cross-block mixing — pure $O(n)$ routing, no weighted combining within blocks.
- **Proposal 006** (Monarch-Gated SSM): Uses dense block-diagonal Monarch factors — richer per-block but $O(b^2)$ per block instead of $O(b)$, and no efficient cumulative product.
- **Proposal 013** (Circulant SSM): Uses FFT-diagonal representation — commutative, cannot represent non-cyclic permutations.
- **This proposal**: Monomial blocks give $O(b)$ per-block cost, the GS shuffle gives cross-block mixing, and the monomial closure enables efficient scan composition.

## Mathematical Formulation

### GS-Monomial State Transition

At each timestep $t$, the state transition matrix is:

$$
A_t = \underbrace{L_t}_{b\text{-block monomial}} \cdot \underbrace{P_{\text{shuffle}}}_{n \times n \text{ fixed permutation}} \cdot \underbrace{R_t}_{b\text{-block monomial}}
$$

where:
- $R_t = \text{diag}(\pi_{R,t}^{(1)} D_{R,t}^{(1)}, \ldots, \pi_{R,t}^{(r)} D_{R,t}^{(r)})$ — block-diagonal with monomial blocks
- $L_t = \text{diag}(\pi_{L,t}^{(1)} D_{L,t}^{(1)}, \ldots, \pi_{L,t}^{(r)} D_{L,t}^{(r)})$ — block-diagonal with monomial blocks
- $P_{\text{shuffle}} \in \{0,1\}^{n \times n}$ — fixed stride permutation (bit-reversal or deinterleave)
- $\pi_{R,t}^{(i)}, \pi_{L,t}^{(i)} \in S_b$ — per-block permutations (input-dependent)
- $D_{R,t}^{(i)}, D_{L,t}^{(i)} \in \mathbb{R}^{b \times b}$ — per-block diagonal matrices (input-dependent)

**Input-dependent parameterization:**

$$
D_{R,t}^{(i)} = \text{diag}\big(\alpha_t^{(i)} \odot \tanh(x_t W_{D,R}^{(i)})\big), \quad \alpha_t^{(i)} = \sigma(x_t W_{\alpha}^{(i)}) \in (0, 1)^b
$$

$$
\pi_{R,t}^{(i)} = \text{Hungarian}\big(\text{Sinkhorn}(x_t W_{\pi,R}^{(i)} / \tau)\big) \quad \text{(hard in forward, soft grad in backward)}
$$

The diagonal uses $\alpha \cdot \tanh$ to allow values in $(-1, 1)$ — enabling both contraction (positive decay) and sign-flipping (negative eigenvalues), drawing on the negative-eigenvalue-extension trick.

### Structured Cumulative Product via Block-Tracking

**Key observation:** The product of two GS-monomial matrices $A_2 A_1 = (L_2 P L_1)(R_2 P R_1)$ is NOT a single GS factor. However, we can track the cumulative product **blockwise** using the following strategy:

**Within each chunk of size $C$:** Materialize the cumulative product as a sequence of $C$ GS applications. Each GS application on a vector costs $O(n \cdot b)$ (two block-monomial mat-vecs + one permutation):

$$
h_t = L_t \cdot P_{\text{shuffle}} \cdot R_t \cdot h_{t-1} + B_t x_t
$$

**Between chunks:** The inter-chunk state propagation requires composing $C$ GS factors into a single $n \times n$ matrix — which in general requires materializing the dense product. However, we can avoid this by:

1. **Keeping the inter-chunk state as a vector** $h \in \mathbb{R}^n$ (not a matrix)
2. **Propagating only the vector state** across chunks, using chunkwise parallel scan with the scan operator being "apply GS-monomial transition to vector"

The associative scan operator for GS-monomial transitions on vectors:

$$
(A_1, b_1) \otimes (A_2, b_2) = (A_2 A_1, A_2 b_1 + b_2)
$$

where $A_i$ are represented in GS-monomial form. The composition $A_2 A_1$ is NOT GS-monomial, but we can represent the inter-chunk accumulated transition as a **product of GS factors** — storing $C$ individual GS factors per chunk boundary.

**Alternative (more practical):** Use a **hybrid approach**:
- **Intra-chunk**: Sequential GS-monomial mat-vec, cost $O(C \cdot n \cdot b)$ per chunk
- **Inter-chunk**: Materialize the $n \times n$ chunk-boundary transition and use dense scan, cost $O((T/C) \cdot n^2 \log(T/C))$

For $n = 256$, $C = 64$, $T = 4096$: intra-chunk cost = $64 \times 256 \times 16 = 262K$ FLOPs per chunk; inter-chunk cost = $64 \times 256^2 \times 6 = 25M$ FLOPs total. The inter-chunk cost is dominated by intra-chunk since there are $T/C = 64$ chunks.

### Stability Analysis

**Per-block spectral norm:**

For a monomial block $\pi D$ with $|D_{jj}| \leq \alpha_{\max} < 1$:

$$
\|\pi D\|_2 = \|D\|_2 = \max_j |D_{jj}| \leq \alpha_{\max} < 1
$$

**Full GS factor spectral norm:**

$$
\|L P_{\text{shuffle}} R\|_2 \leq \|L\|_2 \cdot \|P_{\text{shuffle}}\|_2 \cdot \|R\|_2 = \|L\|_2 \cdot \|R\|_2
$$

Since $L$ and $R$ are block-diagonal with monomial blocks, $\|L\|_2 = \max_i \|L^{(i)}\|_2 \leq \alpha_{\max}$, and similarly for $R$. Thus:

$$
\|A_t\|_2 \leq \alpha_{\max}^2 < 1
$$

**Stability is guaranteed by construction** — the product of contractive GS-monomial matrices is contractive.

**Allowing $|D_{jj}| = 1$ (orthogonal mode):** By setting $\alpha_t = 1$ (removing contraction) and using Cayley-parameterized block rotations instead of diagonal, each block becomes orthogonal and the full GS factor becomes exactly orthogonal. This mode preserves information perfectly and is suitable for state-tracking tasks.

### Connection to Signed Permutation Group

When each monomial block has $D_{jj} \in \{-1, +1\}$, the block matrices are elements of the hyperoctahedral group $B_b = \mathbb{Z}_2 \wr S_b$. The full GS-monomial matrix is then an element of a subgroup of $B_n$ that respects the block structure. This connection means:

- The state transition can represent **signed permutations** — routing state dimensions with optional sign flips
- The $2^b \cdot b!$ elements per block give enormous expressivity with $O(b)$ cost
- The signed permutation structure is the maximal finite subgroup of $O(b)$, ensuring we capture the discrete backbone of any orthogonal transformation

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GS-Monomial SSM |
| Layers | $L = 12$ |
| Model dim | $d_{\text{model}} = 768$ |
| State dim | $n = 256$ |
| Block size | $b = 16$ (so $r = 16$ blocks) |
| GS factors | $m = 2$ (L-shuffle-R) |
| Chunk size | $C = 64$ |
| Permutation learning | Sinkhorn + Hungarian (STE) |
| Block diagonal | $\alpha \cdot \tanh$ parameterization |

**Parameter count per layer (state transition only):**
- Per block: $b$ (permutation logits) + $b$ (diagonal) = $2b = 32$ params
- Per GS factor: $2 \times r \times 2b = 2 \times 16 \times 32 = 1024$ params
- Per layer: $2 \times 1024 = 2048$ params for the transition (vs $n^2 = 65536$ for dense)
- **Compression ratio**: $32\times$ vs dense

**Plus** input-dependent projection weights: $d_{\text{model}} \times 2 \times r \times 2b \approx 768 \times 2048 = 1.5M$ per layer.

### Baseline

1. **Diagonal SSM** (Mamba-2 style): $O(n)$ per step, abelian, $n = 256$
2. **Column-sparse SSM** (PD-SSM): $O(n)$ per step, full $S_n$, $n = 256$
3. **Monarch SSM** (Proposal 006): $O(n\sqrt{n})$ per step, dense blocks, $n = 256$
4. **GLA** (Gated Linear Attention): $O(Td^2)$, linear attention with input-dependent gating
5. **Dense SSM**: $O(n^2)$ per step (upper bound on expressivity, intractable at scale)

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| S5 permutation composition | $> 95\%$ accuracy | Standard state-tracking benchmark |
| Dihedral group $D_{16}$ tracking | $> 90\%$ accuracy | New benchmark: rotation + reflection |
| MQAR (T=4096) | $> 90\%$ | Associative recall |
| WikiText-103 PPL | $\leq$ PD-SSM baseline | Language modeling quality |
| Throughput | $> 0.7\times$ Mamba-2 | Tokens/sec, A100 |
| Memory | $< 1.5\times$ Mamba-2 | Peak GPU memory |

### Estimated Compute

**Full experiment**: ~180 GPU-hours (A100)
- 350M parameter models (5 variants) × 15B tokens each
- Ablation studies: ~80 GPU-hours
- Total: ~260 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- **State tracking**: $> 95\%$ on S5 permutation composition (matching PD-SSM) AND $> 90\%$ on $D_{16}$ (where PD-SSM without sign-flip may degrade)
- **Quality**: Within 3% perplexity of Mamba-2 on WikiText-103, with measurably better MQAR recall (due to cross-block mixing enabling richer state representations)
- **Efficiency**: $0.7\text{–}0.9\times$ Mamba-2 throughput (the GS factor adds $O(\sqrt{n})$ overhead but the BMM structure maps well to tensor cores for the block-diagonal components)
- **Compression**: $32\times$ fewer state-transition parameters than a hypothetical dense SSM, enabling much larger state dimensions

**If hypothesis is wrong:**
- **Monomial blocks too restrictive**: If the $O(b)$-parameter monomial blocks cannot capture the within-block dynamics needed for language modeling (requiring smooth mixing rather than permutation routing), perplexity will be significantly worse ($> 10\%$ gap). This would suggest replacing monomial blocks with dense orthogonal blocks (Cayley-parameterized), reverting to standard GS at $O(b^2)$ per block.
- **Inter-chunk materialization bottleneck**: If the dense $n \times n$ inter-chunk state propagation dominates wall-clock time, the practical speedup will be negligible. This would motivate developing a specialized inter-chunk scan that exploits the GS structure.
- **Permutation learning instability**: If Sinkhorn + STE fails to learn useful block permutations (converging to identity or cycling), the model reduces to a block-diagonal SSM without cross-coordinate mixing. This would motivate replacing Sinkhorn with STEAM (STE-based auction) or OT4P (orthogonal relaxation).

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer GS-Monomial SSM, $n = 16$, $b = 4$ ($r = 4$ blocks), $d_{\text{model}} = 32$ (~80K params)
- **Task**: S5 permutation composition (5-element symmetric group) — input a sequence of group elements, predict their composition
- **Data**: 10K synthetic training samples, $T = 64$ (each sample is a sequence of 64 permutations to compose)
- **Compute**: Single GPU, < 8 minutes

### Success Criteria
- $> 90\%$ accuracy on S5 composition where a diagonal SSM baseline achieves $< 30\%$ (since diagonal SSMs provably cannot represent $S_5$, a non-solvable group)
- GS-Monomial SSM matches or exceeds column-sparse PD-SSM baseline ($> 85\%$)
- Cross-block mixing provides measurable benefit: GS-Monomial outperforms a block-diagonal-only ablation (same architecture but with $P_{\text{shuffle}}$ removed) by $> 10\%$

### Failure Criteria
- If accuracy is $< 50\%$ on S5 composition, the GS-monomial mechanism cannot represent non-abelian group operations despite theoretical coverage — the learning dynamics are broken
- If removing $P_{\text{shuffle}}$ makes $< 5\%$ difference, the cross-block mixing provides no value and the architecture reduces to independent block-monomial SSMs

### Why This Test Is Sufficient
- S5 permutation composition is the **canonical benchmark** for state-tracking expressivity in SSMs (Merrill et al., ICML 2024). It requires representing non-solvable groups, which diagonal SSMs provably cannot do. If GS-monomial SSMs match PD-SSM on this task while additionally benefiting from cross-block mixing, the architecture is viable.
- The cross-block mixing ablation directly tests whether the GS shuffle adds value over independent block processing. This is the core architectural innovation.
- At $n = 16$ with $b = 4$, the model is small enough for rapid iteration but large enough to have non-trivial block structure (4 blocks of 4).

## Theoretical Analysis

**Complexity comparison:**

| Operation | Diagonal SSM | Column-Sparse (PD-SSM) | GS-Monomial (this) | Monarch SSM (006) | Dense |
|-----------|-------------|----------------------|--------------------|--------------------|-------|
| Mat-vec $A_t x$ | $O(n)$ | $O(n)$ | $O(n \cdot b)$ | $O(n \cdot b)$ | $O(n^2)$ |
| Mat-mat $A_2 A_1$ | $O(n)$ | $O(n)$ | $O(n \cdot b)$ | $O(n \cdot b^2)$ | $O(n^3)$ |
| Intra-chunk (size $C$) | $O(Cn)$ | $O(Cn)$ | $O(Cnb)$ | $O(Cnb)$ | $O(Cn^2)$ |
| Inter-chunk scan | $O(\frac{T}{C} n \log \frac{T}{C})$ | $O(\frac{T}{C} n \log \frac{T}{C})$ | $O(\frac{T}{C} n^2 \log \frac{T}{C})$ | $O(\frac{T}{C} n^2 \log \frac{T}{C})$ | $O(\frac{T}{C} n^3 \log \frac{T}{C})$ |
| Parameters per transition | $O(n)$ | $O(2n)$ | $O(4rb) = O(4n)$ | $O(2rb^2) = O(2n \cdot b)$ | $O(n^2)$ |
| Group expressivity | $(F^\times)^n$ (abelian) | $F^\times \wr S_n$ (wreath) | $(B_b)^r \rtimes \langle P_{\text{shuffle}} \rangle$ | Dense blocks | $\text{GL}(n)$ |

**Crossover point with PD-SSM**: The GS-monomial overhead is the factor $b$ in intra-chunk cost. For $b = 16$, GS-monomial is $16\times$ more expensive per step than PD-SSM. The benefit is cross-block mixing. The crossover is justified when cross-block mixing improves quality enough to compensate — which we expect on tasks requiring coordination between state dimensions (language modeling, multi-entity tracking).

**Crossover point with Monarch SSM**: Both have $O(nb)$ mat-vec cost. But GS-monomial has $O(nb)$ mat-mat product (because monomial blocks compose in $O(b)$) while Monarch has $O(nb^2)$ (dense block products are $O(b^2)$ per block). For $b = 16$, this is a $16\times$ advantage in inter-block scan composition. The tradeoff: monomial blocks are less expressive than dense blocks per factor, but GS theory shows $m = 2$ monomial factors suffice for orthogonal coverage where $m = 6$ butterfly factors are needed.

## Risks & Limitations

1. **Monomial blocks are strictly less expressive than dense blocks**: Each monomial block has $O(b)$ parameters vs $O(b^2)$ for dense. The GS completeness guarantee requires stacking $m$ factors — with $m = 2$, coverage of all orthogonal matrices is achieved, but individual step expressivity is limited. Mitigation: increase to $m = 3$ factors if needed ($3\times$ parameter cost but still much less than dense).

2. **Permutation learning is noisy**: Sinkhorn + STE has known issues with gradient bias and convergence to local optima. Mitigation: compare Sinkhorn vs STEAM vs OT4P relaxations in the ablation study; use multiple random restarts.

3. **Inter-chunk dense materialization**: The $O(n^2)$ inter-chunk state costs $n^2 = 65536$ FLOPs per chunk boundary for $n = 256$. For $T/C = 64$ chunks, total is $\sim 4M$ FLOPs — small compared to intra-chunk but requires materializing a dense $n \times n$ matrix. Mitigation: use larger chunks ($C = 128$ or $C = 256$) to reduce the number of chunk boundaries.

4. **Fixed shuffle permutation**: The inter-block shuffle $P_{\text{shuffle}}$ is fixed (not learned), which may limit expressivity. Mitigation: experiment with bit-reversal, deinterleave, and random fixed permutations; consider learned $P_{\text{shuffle}}$ via STEAM as a follow-up.

5. **Block size sensitivity**: The architecture requires choosing $b$ (block size), which creates a hyperparameter. Too small ($b = 4$): monomial blocks are trivial permutations. Too large ($b = 64$): overhead approaches dense. Mitigation: sweep $b \in \{8, 16, 32\}$ in ablation.

## Follow-up Experiments

1. **Dense GS blocks (Cayley-parameterized)**: Replace monomial blocks with $b \times b$ orthogonal blocks via Cayley transform — trades $O(b)$ for $O(b^2)$ per block but gains continuous mixing within blocks. Comparison reveals whether discrete (monomial) or continuous (orthogonal) intra-block dynamics matter more.

2. **Learned shuffle via STEAM**: Replace the fixed $P_{\text{shuffle}}$ with a STEAM-learned permutation, potentially discovering task-optimal cross-block mixing patterns.

3. **GS-Monomial + DeltaNet hybrid**: Use GS-monomial for the state transition $A_t$ and delta-rule for the state update $B_t x_t$, combining the routing expressivity of GS with the associative memory of the delta rule. The WY representation from DeltaNet could be adapted to GS-monomial matrices.

4. **Hierarchical GS**: Nest the GS structure recursively — each "block" is itself a smaller GS-monomial matrix — creating a hierarchical state transition with $O(n \log n)$ cost and $O(\log n)$ levels of mixing. This connects to HSS matrices (Proposal 005).

5. **GS-Monomial for attention projections**: Use GS-monomial matrices for $W_Q, W_K, W_V$ in attention layers — structured orthogonal projections at $32\times$ compression with provable coverage of the orthogonal group.

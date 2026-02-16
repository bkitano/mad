---
status: ongoing
priority: high
created: 2026-02-15
based_on: block-circulant-matrices, input-dependent-gating, chunkwise-parallel-scan, recurrence-to-scan-reduction, blelloch-work-efficient-scan, cauchy-kernel-trick
experiment_number: 013
experiment_log: experiment-log-013.md
---

# Circulant SSM: Fourier-Domain Parallel Scan with Block-Circulant State Transitions

## Hypothesis

Block-circulant state transitions, when diagonalized via FFT into the Fourier domain, enable **element-wise parallel scans in frequency space** — recovering the $O(\log T)$-depth parallelism of diagonal SSMs while achieving the coordinate-mixing expressivity of dense transitions, at $O(n \log n)$ cost per step.

## Background

The fundamental tension in SSM design is:

| Architecture | Per-step cost | Coordinate mixing | Parallel scan? |
|---|---|---|---|
| Diagonal (Mamba-2) | $O(n)$ | None | ✓ (element-wise) |
| DPLR (S4) | $O(rn)$ | Rank-$r$ | ✓ (via Cauchy kernel) |
| Monarch (Proposal 006) | $O(n\sqrt{n})$ | Full via BMM | ✗ (sequential products) |
| DeltaNet | $O(nd)$ | Rank-1 per step | Partial (WY + chunk) |
| Dense | $O(n^2)$ | Full | ✗ |

**Block-circulant matrices fill a unique gap**: they are $O(n \log n)$ per step — cheaper than Monarch — but because circulant matrices are diagonalized by the DFT, the state transition in the Fourier domain becomes **element-wise multiplication**, enabling the same parallel scan structure as diagonal SSMs.

Concretely, if $A = F^{-1} \text{diag}(\hat{a}) F$ is a circulant matrix with FFT coefficients $\hat{a}$, then the recurrence $h_t = A h_{t-1} + B x_t$ becomes, in the Fourier domain $\hat{h}_t = F h_t$:

$$
\hat{h}_t = \text{diag}(\hat{a}) \hat{h}_{t-1} + \hat{B} \hat{x}_t
$$

This is a **diagonal recurrence in frequency space** — exactly the form that admits $O(\log T)$-depth parallel scans via Blelloch's algorithm. Yet in the spatial domain, the circulant matrix performs full coordinate mixing (every output coordinate depends on every input coordinate).

**Why this hasn't been explored:** Most SSM work has focused on diagonal/DPLR parameterizations in the *state* basis, not in a transform domain. The block-circulant trick has been applied to weight compression (CirCNN) and convolutional layers, but never to SSM state transitions with input-dependent gating.

**Key novelty over Proposal 006 (Monarch SSM):** Monarch transitions require sequential BMM products within chunks — the cumulative product $\prod_i M(x_i)$ cannot be parallelized. Circulant transitions avoid this entirely: the parallel scan happens in frequency space at $O(n)$ cost per step (element-wise), with only an $O(n \log n)$ FFT/IFFT at the input/output boundaries.

## Mathematical Formulation

**Standard Diagonal SSM (Mamba-2):**

$$
h_t = \text{diag}(\alpha_t) \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t
$$

**Block-Circulant SSM (Proposed):**

$$
h_t = A(x_t) \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t
$$

where $A(x_t)$ is a block-circulant matrix parameterized by its defining vectors:

$$
A(x_t) = \text{BlockCirc}\left(\{w_{ij}(x_t)\}_{i=1}^{p}, {}_{j=1}^{q}\right)
$$

with $p \times q$ circulant blocks of size $k \times k$, and $n = pk = qk$.

**Fourier-Domain Reformulation:**

Apply the block DFT $\hat{h}_t = (I_p \otimes F_k) h_t$ where $F_k$ is the $k \times k$ DFT matrix:

$$
\hat{h}_t = \hat{D}(x_t) \hat{h}_{t-1} + \hat{B}_t \hat{x}_t
$$

where $\hat{D}(x_t) \in \mathbb{C}^{n \times n}$ is **block-diagonal** with $k$ blocks of size $p \times p$:

$$
\hat{D}(x_t) = \text{diag}(\hat{W}_1(x_t), \hat{W}_2(x_t), \ldots, \hat{W}_k(x_t))
$$

Here $\hat{W}_\ell(x_t) \in \mathbb{C}^{p \times p}$ is the $\ell$-th frequency component of the block-circulant structure.

**Simplified Case ($p = 1$, pure circulant):**

When $A(x_t)$ is a single $n \times n$ circulant matrix (not block-structured), the Fourier-domain representation is fully diagonal:

$$
\hat{h}_t = \text{diag}(\hat{a}(x_t)) \hat{h}_{t-1} + \hat{B}_t \hat{x}_t
$$

This is the key insight: **circulant $\implies$ diagonal in Fourier domain $\implies$ element-wise parallel scan**.

**Input-Dependent Gating:**

The circulant defining vector is input-dependent:

$$
a(x_t) = \sigma(W_a x_t + b_a) \in \mathbb{R}^n
$$

where $\sigma$ is applied element-wise to ensure stability. The Fourier coefficients are:

$$
\hat{a}(x_t) = \text{FFT}(a(x_t)) \in \mathbb{C}^n
$$

**Stability Constraint:**

For stability we need $\rho(A(x_t)) < 1$, which in Fourier domain means $|\hat{a}_i(x_t)| < 1$ for all $i$. This is enforced by:

$$
a(x_t) = \gamma \cdot \tanh(W_a x_t + b_a), \quad \gamma \in (0, 1)
$$

Since $\|a\|_\infty \leq \gamma < 1$ and the DFT preserves spectral radius for circulant matrices, all Fourier eigenvalues satisfy $|\hat{a}_i| \leq n \gamma$. For strict stability, we can instead directly parameterize the Fourier coefficients:

$$
\hat{a}_i(x_t) = r_i(x_t) \cdot e^{j\theta_i(x_t)}, \quad r_i = \sigma(w_r^\top x_t) \in (0, 1), \quad \theta_i = w_\theta^\top x_t
$$

This gives direct control over magnitude and phase per frequency, with stability by construction.

**Parallel Scan in Fourier Domain:**

The scan operates on tuples $(\hat{a}_i(x_t), \hat{b}_i(x_t))$ for each frequency index $i \in \{1, \ldots, n\}$:

$$
\hat{h}_{t,i} = \hat{a}_i(x_t) \hat{h}_{t-1,i} + \hat{b}_i(x_t)
$$

This is $n$ independent scalar recurrences, each amenable to Blelloch parallel scan with $O(\log T)$ depth. Total work: $O(Tn)$, same as diagonal SSM.

**Full Pipeline:**

1. **Input projection**: $a(x_t), B_t x_t$ — standard linear projections, $O(nd)$
2. **FFT**: $\hat{a}(x_t) = \text{FFT}(a(x_t))$, $\hat{u}_t = \text{FFT}(B_t x_t)$ — $O(n \log n)$ per step
3. **Parallel scan**: $n$ element-wise scans in Fourier domain — $O(Tn)$ work, $O(\log T)$ depth
4. **IFFT**: $h_t = \text{IFFT}(\hat{h}_t)$ — $O(n \log n)$ per step
5. **Output projection**: $y_t = C_t^\top h_t$ — $O(nd)$

**Key Variables:**
- $h_t \in \mathbb{R}^n$ — hidden state (spatial domain)
- $\hat{h}_t \in \mathbb{C}^n$ — hidden state (frequency domain)
- $a(x_t) \in \mathbb{R}^n$ — circulant defining vector (input-dependent)
- $\hat{a}(x_t) \in \mathbb{C}^n$ — eigenvalues (Fourier coefficients)
- $n$ — state dimension
- $k$ — block size (for block-circulant; $k = n$ for pure circulant)
- $F_k$ — DFT matrix of size $k$

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Circulant SSM (Circ-SSM) |
| Layers | $L = 12$ |
| Hidden dim | $d = 768$ |
| State dim | $n = 256$ |
| Circulant block size | $k = 256$ (pure circulant) and $k = 16$ (block-circulant) |
| Parameterization | Direct Fourier-domain (magnitude + phase) |
| Scan algorithm | Blelloch parallel scan on $n$ frequency channels |

### Baseline

1. **Mamba-2** (diagonal SSM): $O(Tn)$ per layer, element-wise parallel scan
2. **Monarch-Gated SSM** (Proposal 006): $O(Tn\sqrt{n})$ per layer, sequential BMM
3. **S4 (DPLR)**: $O(Tn)$ per layer via Cauchy kernel, convolutional mode only (LTI)
4. **Gated DeltaNet**: $O(Tnd)$ per layer, chunkwise parallel

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $\geq 0.9\times$ Mamba-2 tokens/sec | Timed forward pass, batch 16, seq 2048 |
| Memory | $\leq 1.1\times$ Mamba-2 | Peak GPU memory during training |
| MQAR accuracy | $> 85\%$ at 4 KV pairs | Multi-Query Associative Recall |
| State tracking | $> 70\%$ on cyclic groups $\mathbb{Z}_n$ | Cyclic group composition |
| Perplexity | $\leq 1.03\times$ Mamba-2 | WikiText-103, 380M params |

### Estimated Compute

**MVE**: ~5 minutes on single GPU (~$0.25)
**Small-scale**: 2 GPU-hours on A100 (~$8)
**Full-scale**: 24 GPU-hours on A100 (~$100)

## Expected Outcome

**If hypothesis is correct:**
- Circ-SSM achieves $> 70\%$ accuracy on cyclic group $\mathbb{Z}_8$ composition where diagonal Mamba-2 achieves $< 50\%$ (cyclic structure is natural for circulant matrices)
- Throughput is within $10\%$ of Mamba-2 (the $O(n \log n)$ FFT overhead is amortized over long sequences)
- MQAR improves by $5$–$15\%$ over Mamba-2 due to cross-coordinate information flow in the circulant structure
- Perplexity matches Mamba-2 on language modeling (circulant mixing adds capacity without hurting optimization)

**If hypothesis is wrong:**
- If cyclic group tracking doesn't improve: circulant's cyclic shift structure may be too restrictive — only cyclic permutations are natural, not general $S_n$. This would motivate block-circulant variants where each block handles a sub-permutation.
- If FFT overhead dominates: for small $n$ (e.g., $n = 64$), the FFT constant factor may exceed the parallelism benefit. This establishes a crossover point $n^*$ below which diagonal is preferred.
- If Fourier-domain scan has numerical issues: complex arithmetic in BF16 may introduce errors that accumulate over long sequences. This motivates real-valued parameterization via DCT instead of DFT.

## Minimum Viable Experiment

### Setup
- **Model**: 2 layers, $d = 64$, $n = 64$, ~100K params
- **Task**: Cyclic group $\mathbb{Z}_8$ composition — given a sequence of elements from $\{0, 1, \ldots, 7\}$ interpreted as $+k \mod 8$ operations, predict the cumulative result
- **Data**: 10K synthetic sequences of length 16–64
- **Compute**: Single GPU, $< 5$ minutes

### Success Criteria
- Circ-SSM achieves $> 90\%$ accuracy on $\mathbb{Z}_8$ composition at sequence length 32
- Diagonal SSM baseline achieves $< 60\%$ accuracy on the same task
- Forward pass throughput of Circ-SSM is $> 0.5\times$ diagonal SSM (FFT overhead is bounded)
- Numerical error: $\|h_t^{\text{spatial}} - \text{IFFT}(\hat{h}_t^{\text{scan}})\|_\infty < 10^{-4}$ in FP32

### Failure Criteria
- If Circ-SSM cannot beat diagonal on $\mathbb{Z}_8$ (a cyclic group — the *most* favorable case for circulant structure), the approach is fundamentally flawed
- If FFT overhead makes Circ-SSM $> 5\times$ slower than diagonal at $n = 64$, the constant factors are too large for practical use

### Why This Test Is Sufficient
- Cyclic group composition is the **canonical task** for circulant matrices — circulant matrices represent exactly the cyclic convolution algebra, so they should excel at cyclic group operations
- The Fourier-domain parallel scan is the core mechanism; if it works numerically at $n = 64$, scaling to $n = 256$ only changes the FFT size (well-optimized by cuFFT)
- The task directly tests whether Fourier-domain eigenvalues can learn to represent cyclic group structure, which is the mathematical foundation of the approach
- If the idea works for $\mathbb{Z}_8$, follow-up experiments test whether block-circulant structure extends to non-cyclic groups ($S_n$, $D_n$)

## Theoretical Analysis

**Complexity comparison:**

| Operation | Diagonal SSM | Circulant SSM | Monarch SSM | Dense SSM |
|-----------|-------------|---------------|-------------|-----------|
| Forward (per step) | $O(n)$ | $O(n \log n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Forward (full seq) | $O(Tn)$ | $O(Tn \log n)$ | $O(Tn\sqrt{n})$ | $O(Tn^2)$ |
| Parameters (transition) | $O(n)$ | $O(n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Parallel scan depth | $O(\log T)$ | $O(\log T)$ | $O(T)$ ‡ | $O(T)$ |
| Memory (state) | $O(n)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| Coordinate mixing | None | Full (cyclic) | Full (block) | Full |

‡ Monarch products within chunks are sequential.

**Crossover point vs. diagonal:**

The overhead of Circ-SSM over diagonal is the $\log n$ factor from FFT. For $n = 256$, this is $\sim 8\times$ more FLOPs per step. However:
1. FFT is highly optimized on GPU (cuFFT achieves near-peak memory bandwidth)
2. The parallel scan structure is identical — no sequential bottleneck
3. The real cost metric is wall-clock time, not FLOPs

Circ-SSM is preferred when the task requires coordinate mixing (state tracking, associative recall) and $T$ is large enough that the $O(\log T)$ scan depth dominates over the per-step FFT.

**Expressivity analysis:**

A circulant matrix over $\mathbb{R}^n$ has exactly $n$ free parameters (its defining vector $a$) and eigenvalues $\hat{a}_0, \ldots, \hat{a}_{n-1}$ that are the DFT of $a$. This means:
- Circulant matrices can represent any cyclic convolution
- They form a commutative algebra (all circulant matrices commute)
- They **cannot** represent non-cyclic permutations (e.g., transpositions)

This is a real expressivity limitation vs. Monarch/dense. However, the commutativity means products of circulant matrices are also circulant, which is why the parallel scan works — the cumulative product $\prod A_i$ is just element-wise product $\prod \hat{a}_i$ in Fourier space.

**Comparison to DPLR/S4:**

S4 uses DPLR structure with the Cauchy kernel trick for convolutional (LTI) computation. Circ-SSM uses circulant structure with FFT for scan-based (LTV) computation. The key difference: S4 is LTI (state transition doesn't depend on input), while Circ-SSM supports input-dependent gating.

## Risks & Limitations

1. **Commutative algebra**: Circulant matrices commute, meaning the order of operations is lost in the cumulative product. For tasks requiring non-commutative state tracking (most real-world tasks), this is a fundamental limitation. Mitigation: use block-circulant with non-commuting blocks.

2. **Complex arithmetic overhead**: The Fourier-domain scan operates on complex numbers, doubling the memory for state storage and requiring complex multiplication. In BF16, complex arithmetic is not natively supported and requires emulation.

3. **Cyclic boundary effects**: Circulant convolution wraps around, which may introduce artifacts for non-periodic data. The state vector "wraps" during mixing, potentially coupling the first and last state dimensions inappropriately.

4. **FFT dimension constraints**: FFT is most efficient for power-of-2 sizes. Non-power-of-2 state dimensions require padding, wasting compute.

5. **Limited to cyclic group structure**: Unlike Monarch (which can represent DFT, DCT, Hadamard, and general permutations), circulant is restricted to cyclic convolutions. For general state tracking, this may be insufficient.

## Follow-up Experiments

1. **Block-circulant with non-commuting blocks**: Use block-circulant structure where each $p \times p$ block matrix is unconstrained, giving non-commutative mixing between blocks while retaining FFT-based scan within each block. The Fourier-domain representation becomes block-diagonal (not scalar-diagonal), requiring block-level parallel scan.

2. **DCT-domain scan**: Replace complex FFT with real-valued DCT/DST, avoiding complex arithmetic overhead. Circulant-like structure with real eigenvalues via symmetric/antisymmetric defining vectors.

3. **Hybrid circulant-diagonal layers**: Alternate circulant layers (for mixing) with diagonal layers (for per-coordinate gating), analogous to the Transformer's attention+FFN alternation.

4. **Learnable permutation pre-conditioning**: Apply a learned permutation before the circulant transition: $A(x_t) = P^\top \text{Circ}(a(x_t)) P$, breaking the cyclic constraint while retaining FFT-based computation (the permutation commutes with the scan since it's fixed).

5. **Connection to Monarch**: A Monarch matrix $P_b^\top L P_b R$ where $L, R$ are block-circulant would combine Monarch's coordinate routing with circulant's FFT efficiency. Each block in $L, R$ would be an FFT-diagonalizable circulant, and the global permutation $P_b$ provides non-cyclic mixing.

## References

- Ding, C., et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017)
- Gu, A., Goel, K., & Ré, C. "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022)
- Blelloch, G. (1993). "Prefix Sums and Their Applications"
- Mamba-2: Dao & Gu (2024). "Transformers are SSMs"

---
status: completed
priority: high
created: 2026-02-15
based_on: 
experiment_number: 007
results_file: 007_results.md
---

# Oscillatory-Gated Selective SSM (OscGate-SSM)

- oscillatory-eigenvalue-stability
- input-dependent-gating
- chunkwise-parallel-scan
- recurrence-to-scan-reduction
- cayley-contractive-parameterization

## Hypothesis

Making the oscillatory parameters (frequency $\omega_i$ and damping $\zeta_i$) **input-dependent** — i.e., $\omega_i(x_t)$ and $\zeta_i(x_t)$ — while preserving the eigenvalue stability guarantee $|\lambda| \leq 1$ *by construction*, will produce an SSM that simultaneously:
1. **Matches Mamba's selectivity** (content-dependent gating) without sacrificing stability
2. **Avoids the stability-expressivity tradeoff** that plagues existing selective SSMs (Mamba requires heuristic eigenvalue clipping; DeltaNet's WY norm can grow)
3. **Outperforms fixed-parameter oscillatory SSMs (LinOSS)** on tasks requiring content-dependent temporal dynamics (e.g., selective copying, MQAR)

**Core insight**: LinOSS achieves stability by construction but uses fixed $\omega, \zeta$ (LTI), limiting expressivity. Mamba achieves selectivity via input-dependent gates but requires ad-hoc stability constraints. OscGate-SSM threads the needle: input-dependent oscillatory parameters inherit stability from the physics of harmonic oscillators regardless of what values $\omega(x_t), \zeta(x_t)$ take, as long as $\omega > 0$ and $\zeta \geq 0$.

## Background

**Existing landscape:**
- **LinOSS** (Rusch & Rus, 2025): Fixed oscillatory parameters give stability-by-construction, but LTI dynamics cannot perform selective operations (copying, routing)
- **Mamba/Mamba-2** (Gu & Dao, 2023/2024): Input-dependent diagonal gates give selectivity, but stability relies on sigmoid-bounded decay $\alpha_t \in (0, 1)$ — a soft constraint that doesn't extend to non-diagonal transitions
- **Proposal 004 (Oscillatory-DPLR)**: Combines oscillatory eigenvalues with DPLR structure but keeps parameters **fixed** (LTI) — explicitly noted as a follow-up direction: "Make $\omega_i(x_t)$, $\zeta_i(x_t)$ input-dependent"

**Gap**: No architecture provides input-dependent (selective) dynamics with stability guaranteed by construction from the physics of the discretization. This is distinct from:
- Proposal 004, which is LTI (no selectivity)
- Proposal 006 (Monarch-Gated), which achieves stability via orthogonal blocks + sigmoid gates (an algebraic constraint, not a physics-based one)

**Why this matters**: As SSMs scale to billions of parameters, stability failures become harder to diagnose and fix. A model where stability is mathematically guaranteed for any input sequence eliminates an entire class of training failures.

## Mathematical Formulation

### LinOSS Background (Fixed Parameters)

The oscillatory SSM discretizes a second-order ODE system:

$$
\mathbf{y}''(t) = -\mathbf{A}\mathbf{y}(t) + \mathbf{B}\mathbf{u}(t)
$$

where $\mathbf{A} = \text{diag}(a_1, \ldots, a_m)$ with $a_k \geq 0$. The implicit (IM) discretization produces:

$$
\mathbf{x}_n = \mathbf{M}^{IM} \mathbf{x}_{n-1} + \mathbf{F}_n, \quad \mathbf{x}_n = [\mathbf{z}_n, \mathbf{y}_n]^\top \in \mathbb{R}^{2m}
$$

$$
\mathbf{M}^{IM} = \begin{bmatrix} \mathbf{S} & -\Delta t \mathbf{A}\mathbf{S} \\ \Delta t \mathbf{S} & \mathbf{S} \end{bmatrix}, \quad \mathbf{S} = (I + \Delta t^2 \mathbf{A})^{-1}
$$

**Eigenvalue bound**: $|\lambda_j|^2 = S_{kk} = \frac{1}{1 + \Delta t^2 a_k} \leq 1$ for all $a_k \geq 0$.

### Proposed: Input-Dependent Oscillatory Parameters

**Step 1**: Generate input-dependent frequency and damping:

$$
\omega_k(x_t) = \text{softplus}(W_\omega x_t + b_\omega)_k > 0
$$

$$
\zeta_k(x_t) = \sigma(W_\zeta x_t + b_\zeta)_k \in (0, 1)
$$

where $W_\omega, W_\zeta \in \mathbb{R}^{m \times d}$ are learnable projections.

**Step 2**: Construct input-dependent diagonal matrix:

$$
\mathbf{A}(x_t) = \text{diag}\left(\omega_1(x_t)^2, \ldots, \omega_m(x_t)^2\right)
$$

Note: $\mathbf{A}(x_t) \geq 0$ is guaranteed by squaring, regardless of $x_t$.

**Step 3**: Apply damped implicit discretization:

$$
\mathbf{M}_t = \begin{bmatrix} \mathbf{S}_t & -\Delta t \, \mathbf{A}(x_t) \mathbf{S}_t \\ \Delta t \, \mathbf{S}_t & \mathbf{S}_t \end{bmatrix}, \quad \mathbf{S}_t = \left(I + \Delta t^2 \mathbf{A}(x_t)\right)^{-1}
$$

**Step 4**: Add damping via input-dependent decay:

$$
\mathbf{M}_t^{\text{damped}} = \text{diag}(\mathbf{d}_t) \cdot \mathbf{M}_t
$$

$$
\mathbf{d}_t = [1 - \zeta_1(x_t), \ldots, 1 - \zeta_m(x_t), 1 - \zeta_1(x_t), \ldots, 1 - \zeta_m(x_t)] \in (0, 1)^{2m}
$$

**Step 5**: The recurrence becomes:

$$
\mathbf{x}_t = \mathbf{M}_t^{\text{damped}} \cdot \mathbf{x}_{t-1} + \mathbf{F}_t(x_t)
$$

### Stability Guarantee (Key Theorem)

**Claim**: For any input sequence $\{x_t\}_{t=1}^T$ and any learnable parameters $W_\omega, W_\zeta, b_\omega, b_\zeta$:

$$
\|\mathbf{M}_t^{\text{damped}}\|_2 \leq \max_k (1 - \zeta_k(x_t)) \cdot \sqrt{S_{kk}(x_t)} < 1
$$

**Proof sketch**:
- $\|\mathbf{M}_t\|_2 = \max_k \sqrt{S_{kk}} \leq 1$ (from LinOSS Proposition 3.1, since $\mathbf{A}(x_t) \geq 0$)
- $\|\text{diag}(\mathbf{d}_t)\|_2 = \max_k (1 - \zeta_k(x_t)) < 1$ (since $\zeta_k \in (0, 1)$ via sigmoid)
- $\|\mathbf{M}_t^{\text{damped}}\|_2 \leq \|\text{diag}(\mathbf{d}_t)\|_2 \cdot \|\mathbf{M}_t\|_2 < 1$

This holds **for any input** and **any learned parameters** — stability is an invariant of the architecture, not a property that must be maintained during training.

### Parallel Scan Compatibility

The time-varying recurrence $\mathbf{x}_t = \mathbf{M}_t^{\text{damped}} \mathbf{x}_{t-1} + \mathbf{F}_t$ is a first-order linear recurrence with input-dependent coefficients. Via the recurrence-to-scan reduction:

$$
(\mathbf{M}_i, \mathbf{F}_i) \bullet (\mathbf{M}_j, \mathbf{F}_j) = (\mathbf{M}_j \mathbf{M}_i, \; \mathbf{M}_j \mathbf{F}_i + \mathbf{F}_j)
$$

**Critical efficiency observation**: Since $\mathbf{M}_t$ has $2 \times 2$ block-diagonal structure (each block is $m \times m$ diagonal), the matrix-matrix product $\mathbf{M}_j \mathbf{M}_i$ decomposes into $O(m)$ element-wise operations (not $O(m^2)$ or $O(m^3)$):

$$
\mathbf{M}_j \mathbf{M}_i = \begin{bmatrix} S_j S_i - \Delta t^2 A_j S_j S_i & -\Delta t(A_i S_i S_j + S_j A_j S_i - \Delta t^2 A_j S_j A_i S_i) \\ \Delta t(S_j S_i + S_j S_i - \Delta t^2 A_j S_j S_i) & S_j S_i - \Delta t^2 S_j A_i S_i \end{bmatrix}
$$

Wait — this is not block-diagonal after multiplication. The product of two $2 \times 2$ block matrices gives a full $2 \times 2$ block matrix. However, since $\mathbf{A}$ and $\mathbf{S}$ are both diagonal, each of the four $m \times m$ blocks in the product is still diagonal. So the product is a $2m \times 2m$ matrix that is block-structured as a $2 \times 2$ matrix of $m \times m$ diagonal blocks.

**Actual cost**: The scan operation on pairs $(\mathbf{M}_t \in \mathbb{R}^{2m \times 2m}, \mathbf{F}_t \in \mathbb{R}^{2m})$ with block-diagonal structure costs $O(m)$ per composition (4 element-wise multiplications + 2 additions for the $2 \times 2$ block product, each on $m$-dimensional diagonal blocks), yielding $O(mT)$ total for the full scan with $O(\log T)$ parallel depth — **identical** to the cost of standard diagonal SSM scans, just with a $4\times$ constant factor.

### Chunkwise Parallel Implementation

Using the chunkwise parallel scan trick with chunk size $C$:

**Intra-chunk** ($O(Cm)$ per chunk, parallelized over chunks):
- Compute $\omega_k(x_t), \zeta_k(x_t)$ for all $t$ in chunk — simple linear projections
- Form $\mathbf{M}_t^{\text{damped}}$ for all $t$ — element-wise operations on diagonals
- Compute cumulative products within chunk via scan

**Inter-chunk** ($O(\frac{T}{C} \cdot m \log \frac{T}{C})$):
- Scan over chunk boundary states

**Total**: $O(Tm)$ work, $O(\log T)$ parallel depth — same asymptotic complexity as Mamba's diagonal scan.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | OscGate-SSM |
| Layers | $L = 8$ |
| Hidden dim | $d = 512$ |
| Oscillator dim | $m = 128$ (state dim $n = 2m = 256$) |
| Chunk size | $C = 128$ |
| Gating params | $W_\omega, W_\zeta \in \mathbb{R}^{m \times d}$ (131K per layer) |
| Total params | ~50M |

### Baseline

1. **LinOSS** (fixed oscillatory, LTI): Same $2m$ state, $O(Tm)$ per layer — tests whether input-dependence adds value
2. **Mamba-2** (diagonal gated, LTV): $O(Tn)$ per layer with $n = 256$ — tests whether oscillatory structure helps vs. pure diagonal
3. **S5** (diagonal fixed, LTI): $O(Tn)$ per layer — weakest baseline, no selectivity or oscillatory structure

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Selective copying | $> 95\%$ acc at length 256 | Synthetic selective copy task |
| MQAR recall | $> 90\%$ at 4 KV pairs | Multi-Query Associative Recall |
| Throughput | $\geq 0.8 \times$ Mamba-2 | Tokens/sec on A100 |
| Memory | $\leq 1.1 \times$ Mamba-2 | Peak GPU memory |
| Stability | 0 NaN/Inf over 100K steps | Monitor $\|\mathbf{x}_t\|$ during training |

### Estimated Compute

**MVE**: ~10 minutes on single GPU (~$0.50)
**Small-scale**: 8 GPU-hours on A100 (~$32)
**Full-scale**: 48 GPU-hours on A100 (~$200)

## Expected Outcome

**If hypothesis is correct:**
- OscGate-SSM matches Mamba-2's selective copying accuracy ($> 95\%$) where LinOSS (LTI) achieves $< 30\%$ — proving that input-dependence is necessary and our formulation achieves it
- OscGate-SSM experiences **zero stability failures** over 100K training steps, while Mamba-2 may require gradient clipping for certain learning rates
- On MQAR with 4+ KV pairs, OscGate-SSM matches or exceeds Mamba-2 accuracy ($> 90\%$) — oscillatory structure provides richer dynamics than scalar decay
- Throughput penalty is $< 20\%$ vs. Mamba-2 (the $4\times$ constant factor on scan is amortized by the projection costs)

**If hypothesis is wrong:**
- **Scenario A**: OscGate-SSM matches LinOSS but not Mamba-2 on selective tasks
  - **Learn**: The $2 \times 2$ block structure of the oscillatory transition limits effective selectivity compared to unconstrained diagonal gating
  - **Fix**: Decouple the damping from the oscillatory structure; use independent scalar gates per state dimension like Mamba, with oscillatory only for the transition structure
- **Scenario B**: OscGate-SSM is much slower than expected
  - **Learn**: The $4\times$ constant factor on the $2 \times 2$ block scan is worse than expected due to memory bandwidth (4 diagonal multiplies + 2 additions vs. 1 multiply + 1 addition for diagonal)
  - **Fix**: Fuse the $2 \times 2$ block operations into a custom CUDA kernel; or use the IMEX variant which has simpler (symplectic) block structure
- **Scenario C**: Stability is guaranteed but quality is worse than Mamba-2
  - **Learn**: The oscillatory inductive bias (eigenvalues in conjugate pairs) is too restrictive for general sequence modeling
  - **Insight**: Still valuable for domains where oscillatory dynamics are natural (time series, physics, audio)

## Minimum Viable Experiment

### Setup
- **Model**: 1-layer OscGate-SSM ($d = 64$, $m = 32$, state dim $n = 64$, ~30K params)
- **Task**: **Selective Copying** — given an input sequence `a b c d [SEP] _ _ 3 _`, output the token at the specified index (3rd token = `c`). Sequence length 16–32, vocabulary size 16.
- **Data**: 10K synthetic sequences
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- OscGate-SSM achieves $> 90\%$ accuracy on selective copying at length 32
- Fixed-parameter LinOSS (same architecture, no input-dependent gating) achieves $< 40\%$ accuracy
- No NaN/Inf values encountered during training (stability guarantee holds)
- Forward pass wall-clock time is $< 3\times$ that of a diagonal SSM of equal state dimension

### Failure Criteria
- OscGate-SSM cannot beat a fixed LinOSS on selective copying — input-dependent mechanism is broken
- NaN/Inf values appear despite the theoretical stability guarantee — implementation bug or numerical precision issue
- Forward pass is $> 5\times$ slower than diagonal SSM — $2 \times 2$ block overhead is impractical

### Why This Test Is Sufficient
- Selective copying is the canonical task that separates LTV (selective) from LTI (fixed) models. If the input-dependent gating of $\omega, \zeta$ enables selective copying, the core mechanism works.
- LinOSS provably cannot solve selective copying (it's LTI), so a clear positive result isolates the contribution of input-dependent oscillatory parameters.
- The stability guarantee is a mathematical invariant — if it holds at tiny scale, it holds at any scale. A single successful training run validates this.
- If the tiny model succeeds, scaling adds capacity (more oscillators, more layers) but the fundamental mechanism is validated.

## Theoretical Analysis

### Complexity Comparison

| Operation | LinOSS (LTI) | Mamba-2 (diagonal) | OscGate-SSM |
|-----------|--------------|---------------------|-------------|
| State dim for $m$ oscillators | $2m$ | $n = 2m$ | $2m$ |
| Forward step | $O(m)$ | $O(n) = O(2m)$ | $O(m)$ + gate projection |
| Parallel scan cost | $O(m \log T)$ | $O(n \log T)$ | $O(4m \log T)$ |
| Gate projection | None | $O(nd)$ | $O(2md)$ |
| Stability enforcement | Free (by construction) | Sigmoid bound | Free (by construction) |
| Selectivity | None (LTI) | Full | Full |

**Key tradeoff**: OscGate-SSM pays a $4\times$ constant in the scan (due to $2 \times 2$ block structure) but gains unconditional stability and oscillatory dynamics. The gate projection cost $O(2md)$ is comparable to Mamba-2's $O(nd)$ since $n = 2m$.

### Expressivity Analysis

**What LinOSS can't do (LTI limitation)**:
- Cannot selectively attend to specific tokens in context
- Cannot perform content-dependent routing or gating
- Example: Given `a b c [SEP] 2`, cannot select the 2nd token — fixed dynamics treat all inputs identically

**What OscGate-SSM adds**:
- Input-dependent frequency: $\omega(x_t)$ allows the model to "tune in" to different temporal scales per token
- Input-dependent damping: $\zeta(x_t)$ allows the model to selectively retain or forget state per token
- Combined: Equivalent to a selective SSM with structured (oscillatory) transitions instead of unconstrained diagonal

**Comparison with Mamba's selectivity**:
- Mamba: $\alpha_t = \sigma(W x_t) \in (0, 1)^n$ — $n$ independent decay rates
- OscGate-SSM: $\omega_t, \zeta_t$ give $2m$ parameters controlling $2m$ coupled eigenvalues ($n = 2m$)
- Mamba's eigenvalues are real and independent; OscGate-SSM's eigenvalues come in conjugate pairs with physical meaning
- This coupling is a **constraint** (fewer degrees of freedom) but also a **regularizer** (prevents pathological eigenvalue configurations)

## Risks & Limitations

### Risk 1: Oscillatory Inductive Bias Too Restrictive
- **Issue**: Forcing eigenvalues into conjugate pairs from oscillatory ODEs may not suit tasks without periodic structure
- **Mitigation**: Include "degenerate" oscillators where $\omega \to 0$ (reduces to exponential decay, recovering Mamba-like behavior). The softplus ensures $\omega > 0$ but allows $\omega \approx 0$
- **Fallback**: Use a mixture of oscillatory and pure-diagonal state dimensions

### Risk 2: Gate Projection Overhead
- **Issue**: Computing $\omega(x_t), \zeta(x_t)$ via linear projections $W_\omega, W_\zeta \in \mathbb{R}^{m \times d}$ adds $O(md)$ compute per step
- **Mitigation**: This is identical to Mamba-2's gate projection cost — not a new bottleneck
- **Optimization**: Share $W_\omega$ and $W_\zeta$ partially (e.g., shared trunk with separate heads)

### Risk 3: Chunkwise Implementation Complexity
- **Issue**: The $2 \times 2$ block-diagonal structure of $\mathbf{M}_t$ requires a custom scan operator, not available in standard libraries
- **Mitigation**: Implement as 4 parallel element-wise scans with cross-connections (can be written as a simple PyTorch custom op)
- **Alternative**: Flatten the $2 \times 2$ block structure into a size-$4m$ diagonal scan with redundant entries (trades compute for implementation simplicity)

### Risk 4: Numerical Precision in $S_t = (I + \Delta t^2 A(x_t))^{-1}$
- **Issue**: When $\omega(x_t) \to \infty$, $S_t \to 0$ — may underflow in fp16
- **Mitigation**: Clamp $\omega(x_t) \leq \omega_{\max}$ (e.g., $\omega_{\max} = 100$); or compute $S_t$ in fp32
- **Note**: This is a precision issue, not a stability issue — eigenvalues remain bounded regardless

## Follow-up Experiments

### If Successful:
1. **Scale to 1B+ parameters**: Test if stability guarantee eliminates the need for gradient clipping at scale
2. **Multi-scale oscillatory banks**: Partition oscillators into groups (slow $\omega \in [0.001, 0.01]$, medium $\omega \in [0.01, 0.1]$, fast $\omega \in [0.1, 1]$) with input-dependent routing between banks
3. **Oscillatory-Monarch hybrid**: Replace the $2 \times 2$ diagonal-block transition with a $2 \times 2$ Monarch-block transition — combines oscillatory stability with Monarch's coordinate mixing (connects to Proposal 006)
4. **Application to audio/speech**: Oscillatory dynamics are natural for audio signals; test on speech recognition where frequency-adaptive processing is physically motivated
5. **Combine with log-linear attention**: Apply the hierarchical mask from log-linear attention to the oscillatory state, giving $O(\log T)$ multi-resolution oscillatory states

### If Unsuccessful:
1. **Ablate oscillatory vs. plain gating**: Replace oscillatory structure with plain diagonal decay (Mamba-style) but keep the damping parameterization — is the failure from oscillatory structure or something else?
2. **Analyze learned $\omega, \zeta$ distributions**: Do they collapse to a degenerate regime? If $\omega \to 0$ everywhere, the model is "learning away" the oscillatory structure
3. **Try IMEX discretization**: The symplectic (IMEX) variant has $|\lambda| = 1$ exactly — does the energy-conserving property help or hurt?

## Connection to Existing Proposals

- **Distinct from 004 (Oscillatory-DPLR)**: Proposal 004 is LTI with DPLR structure for convolutional mode. This proposal is LTV with input-dependent oscillatory parameters for selective mode. They target different modalities of the SSM (convolution vs. recurrence).
- **Complementary to 006 (Monarch-Gated)**: Proposal 006 achieves coordinate mixing via Monarch structure. OscGate-SSM achieves stability via oscillatory physics. A future hybrid (Risk 3 follow-up) could combine both.
- **Extends 001's insight**: The human review on Proposal 001 noted that expressivity often involves composing Householder reflections. OscGate-SSM sidesteps this by operating in the oscillatory eigenvalue space, where stability is free and selectivity comes from input-dependent frequency/damping.

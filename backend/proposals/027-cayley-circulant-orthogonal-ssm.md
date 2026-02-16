---
status: ongoing
priority: high
created: 2026-02-15
based_on: cayley-contractive-parameterization, cdflow-circulant-diagonal-invertible, circulant-cycle-decomposition, block-circulant-matrices, recurrence-to-scan-reduction, chunkwise-parallel-scan, input-dependent-gating
experiment_number: 027
---

# Cayley-Parameterized Circulant-Diagonal Orthogonal SSM

## Hypothesis

A state transition matrix parameterized as a Cayley transform of a skew-circulant-diagonal product will achieve (1) exact orthogonality by construction with no clipping or projection, (2) $O(n \log n)$ per-step cost via FFT, and (3) superior long-range memory retention compared to both diagonal SSMs (S4D/Mamba) and oscillatory SSMs (LinOSS), while maintaining parallelizable training via associative scan.

## Background

### The Stability-Expressivity Gap

Current SSM state transitions face a fundamental tradeoff:

- **Diagonal SSMs** (S4D, Mamba): $O(n)$ per step, easily parallelizable, but diagonal structure limits expressivity -- cannot simulate non-abelian groups or perform coordinate mixing.
- **Dense orthogonal SSMs** (scoRNN/Cayley): Exact $|\lambda| = 1$ guarantees perfect information retention, but $O(n^2)$ matvec and $O(n^3)$ Cayley inversion make them impractical.
- **Oscillatory SSMs** (LinOSS, Proposal 004/007): Stability by construction with $|\lambda| \leq 1$, but eigenvalues are *damped* (strictly inside unit circle), so information decays over time. The damping is a feature for forgetting but a bug for tasks requiring perfect long-range recall.
- **Circulant-Diagonal SSMs** (Proposal 023): $O(n \log n)$ via FFT with full coordinate mixing, but no stability guarantee -- eigenvalues can escape the unit circle during training.

**No existing proposal combines exact orthogonality with sub-quadratic per-step cost.**

### The Key Insight

The Cayley transform $W = (I + A)^{-1}(I - A)$ maps skew-symmetric matrices to orthogonal matrices. If $A$ is structured (e.g., skew-circulant or skew-circulant-diagonal), then $(I + A)^{-1}$ can be computed in $O(n \log n)$ via FFT, because circulant matrices are diagonalized by the DFT. This gives an orthogonal matrix with $O(n)$ parameters and $O(n \log n)$ matvec -- the best of both worlds.

Specifically, a skew-circulant matrix $A = \text{circ}_{\text{skew}}(a)$ satisfies $A^T = -A$ automatically when the generating vector $a$ obeys the skew-symmetry constraint $a_j = -a_{n-j}$. The Cayley transform of such a matrix yields an orthogonal circulant-like matrix with eigenvalues exactly on the unit circle, computable entirely in the Fourier domain.

### What's New vs. Existing Proposals

| Proposal | Stability | Per-step | Mixing | Params |
|----------|-----------|----------|--------|--------|
| 004 (Oscillatory DPLR) | $|\lambda| \leq 1$ (damped) | $O(n)$ | Low-rank | $O(n)$ |
| 013 (Circulant SSM) | None guaranteed | $O(n \log n)$ | Full | $O(n)$ |
| 023 (CD-SSM) | None guaranteed | $O(n \log n)$ | Full | $O(n)$ |
| **027 (This)** | $|\lambda| = 1$ (exact) | $O(n \log n)$ | Full | $O(n)$ |

## Mathematical Formulation

### Core Parameterization

**Free parameters:** A skew-symmetric generating vector $a \in \mathbb{R}^{\lfloor n/2 \rfloor}$ (exploiting the constraint $a_j = -a_{n-j}$, the first half determines the full vector).

**Skew-circulant construction:**

$$
A = F^{-1} \text{diag}(\hat{a}) F, \quad \hat{a} = \text{FFT}(a_{\text{full}})
$$

where $a_{\text{full}} = [0, a_1, a_2, \ldots, a_{\lfloor n/2 \rfloor}, -a_{\lfloor n/2 \rfloor}, \ldots, -a_1]$ ensures $A^T = -A$.

Since $A$ is skew-symmetric, its eigenvalues $\hat{a}_j$ are purely imaginary: $\hat{a}_j = i \omega_j$ for real $\omega_j$.

**Cayley transform in Fourier domain:**

$$
W = (I + A)^{-1}(I - A) = F^{-1} \text{diag}\left(\frac{1 - i\omega_j}{1 + i\omega_j}\right) F
$$

Each diagonal entry is:

$$
\lambda_j = \frac{1 - i\omega_j}{1 + i\omega_j} = e^{-2i \arctan(\omega_j)}
$$

which has $|\lambda_j| = 1$ exactly. The matrix $W$ is orthogonal, circulant, and computed entirely via FFT.

**Matvec:**

$$
W x = \text{IFFT}\left(\frac{1 - i\omega}{1 + i\omega} \odot \text{FFT}(x)\right)
$$

Cost: $O(n \log n)$ via two FFTs + $O(n)$ element-wise ops.

### Input-Dependent Gating Extension

To make the model selective (like Mamba), we gate the free parameters:

$$
\omega(x_t) = \text{Linear}(x_t) \in \mathbb{R}^{\lfloor n/2 \rfloor}
$$

$$
A_t = W(\omega(x_t)), \quad h_t = W(\omega(x_t)) h_{t-1} + B x_t
$$

Each $W(\omega(x_t))$ is orthogonal by construction regardless of input.

### Parallel Scan Compatibility

For the associative scan, we need to compose pairs $(A_t, b_t)$ where $A_t$ is the transition and $b_t = B x_t$. The composition is:

$$
(A_2, b_2) \bullet (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)
$$

The product $A_2 A_1$ of two orthogonal circulant matrices is another orthogonal circulant matrix, computed in $O(n \log n)$:

$$
A_2 A_1 = F^{-1} \text{diag}(\hat{\lambda}^{(2)} \odot \hat{\lambda}^{(1)}) F
$$

So the scan operator costs $O(n \log n)$ per composition (element-wise multiply of Fourier eigenvalues) instead of $O(n^2)$ for dense matrices. The scan over $T$ steps costs $O(Tn \log n \cdot \log T)$ total.

### Forgetting via Controlled Contraction (Optional)

Pure orthogonality means no forgetting. For tasks that need selective forgetting, introduce a learned per-frequency damping:

$$
\lambda_j(x_t) = \sigma(\gamma_j(x_t)) \cdot e^{-2i \arctan(\omega_j(x_t))}
$$

where $\sigma(\gamma_j) \in (0, 1]$ is a sigmoid gate. This gives $|\lambda_j| \leq 1$ with the option to be exactly 1 (no forgetting) when $\gamma \to \infty$.

**Key Variables:**
- $x_t \in \mathbb{R}^d$ -- input at time $t$
- $h_t \in \mathbb{R}^n$ -- hidden state (may be complex-valued in Fourier domain)
- $\omega \in \mathbb{R}^{\lfloor n/2 \rfloor}$ -- free parameters of skew-circulant
- $\lambda_j \in \mathbb{C}$, $|\lambda_j| = 1$ -- eigenvalues of transition matrix
- $W \in \mathbb{R}^{n \times n}$ -- orthogonal circulant transition matrix

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Cayley-Circulant SSM (CC-SSM) |
| Layers | $L = 12$ |
| Hidden dim | $d = 256$ |
| State dim | $n = 64$ |
| Heads | $H = 4$ (each with $n/H = 16$ state dims) |
| Gating | Input-dependent $\omega(x_t)$ + optional damping $\gamma(x_t)$ |
| Scan | Chunkwise parallel, chunk size $C = 64$ |

### Baseline

| Model | Per-step | Stability | Mixing |
|-------|----------|-----------|--------|
| S4D (diagonal) | $O(n)$ | $|\lambda| \leq 1$ via clipping | None |
| Mamba-2 (SSD) | $O(Qn)$ | Sigmoid gate | Semiseparable |
| LinOSS (oscillatory) | $O(n)$ | $|\lambda| \leq 1$ by construction | 2x2 block |
| Circulant SSM (Prop. 013) | $O(n \log n)$ | None | Full circulant |
| **CC-SSM (this)** | $O(n \log n)$ | $|\lambda| = 1$ by construction | Full circulant |

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| S5 State Tracking (MQAR) | $> 95\%$ at $T = 512$ | Accuracy on multi-query associative recall |
| Long-Range Arena | $> 86\%$ avg | Standard LRA benchmark suite |
| Copy Task | $> 99\%$ at delay $T = 1000$ | Accuracy on delayed copy |
| Throughput | $> 0.7\times$ Mamba | Tokens/sec on A100 |
| Memory | $< 1.5\times$ diagonal SSM | Peak GPU memory |

### Estimated Compute

**Small:** 2-4 GPU-hours on single A100. MVE under 10 minutes.

## Expected Outcome

**If hypothesis is correct:**
- CC-SSM matches or exceeds LinOSS on long-range tasks (LRA, copy) due to exact $|\lambda| = 1$
- CC-SSM significantly outperforms diagonal SSMs on state-tracking (MQAR) due to full coordinate mixing
- $< 2\times$ overhead vs diagonal SSMs (the $\log n$ factor in FFT is small for $n = 64$)
- The damped variant (with $\gamma$ gate) matches Mamba-quality on language modeling while retaining stability

**If hypothesis is wrong:**
- If orthogonal circulant mixing is insufficient for state tracking, it reveals that the *structure* of mixing matters beyond just "full vs diagonal" -- i.e., circulant mixing is too regular/symmetric to simulate arbitrary permutations
- If the FFT overhead dominates, it establishes the practical crossover point where structured matvec beats dense matvec for SSM state dimensions
- If exact orthogonality hurts language modeling (no forgetting), it quantifies the importance of adaptive decay

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer CC-SSM, $d = 64$, $n = 32$, ~80K params
- **Task**: Delayed copy task: input sequence of $k$ tokens, wait $T$ steps of padding, reproduce the $k$ tokens
- **Data**: Synthetic, vocabulary size 8, $k = 5$, delay $T \in \{50, 100, 200, 500\}$
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- $> 99\%$ copy accuracy at delay $T = 500$ where diagonal SSM (S4D baseline) drops below $80\%$
- Orthogonal variant retains information perfectly (by construction), so near-100% is expected

### Failure Criteria
- If CC-SSM cannot achieve $> 90\%$ at $T = 200$, the implementation has a bug (the math guarantees perfect retention)
- If training is $> 10\times$ slower than diagonal SSM, the FFT overhead is too large for practical use

### Why This Test Is Sufficient
- The delayed copy task directly tests long-range memory retention, which is the core capability enabled by $|\lambda| = 1$
- If a tiny model can retain information across 500 steps with exact orthogonality, the mechanism is validated; scaling adds capacity
- The diagonal SSM baseline provides a clear "how much does orthogonal circulant mixing help?" comparison

## Theoretical Analysis

Complexity comparison:

| Operation | Diagonal SSM | CC-SSM (this) | Dense Cayley |
|-----------|-------------|---------------|--------------|
| Transition matvec | $O(n)$ | $O(n \log n)$ | $O(n^2)$ |
| Cayley construction | N/A | $O(n)$ (Fourier) | $O(n^3)$ |
| Scan composition | $O(n)$ | $O(n \log n)$ | $O(n^2)$ |
| Parameters | $O(n)$ | $O(n)$ | $O(n^2/2)$ |
| Full scan ($T$ steps) | $O(Tn \log T)$ | $O(Tn \log n \log T)$ | $O(Tn^2 \log T)$ |

Crossover: CC-SSM vs diagonal is always $\log n$ overhead. For $n = 64$, $\log_2 n = 6$, so ~$6\times$ more work per step but with full coordinate mixing. For $n = 16$, overhead is ~$4\times$.

## Risks & Limitations

1. **Circulant structure is too symmetric**: Circulant matrices have a very specific algebraic structure (every row is a cyclic shift). This may be insufficiently expressive for tasks requiring arbitrary permutation-like mixing (e.g., $S_5$ state tracking). The group of circulant permutations is $\mathbb{Z}_n$, which is abelian -- so this cannot track non-abelian automata.

2. **Complex arithmetic in practice**: The Fourier-domain computation involves complex numbers. While the input/output are real, intermediate FFTs are complex, requiring careful handling in BF16/FP16.

3. **Input-dependent gating breaks circulant closure**: When $\omega(x_t)$ varies per token, the product of two input-dependent Cayley-circulant matrices is still circulant (products of circulants are circulant), so closure is maintained. However, the Fourier-domain composition requires element-wise multiply of complex eigenvalues, which must be tracked through the scan.

4. **Forgetting may be essential**: Pure $|\lambda| = 1$ means the model never forgets, which may cause interference between old and new information on realistic NLP tasks.

## Follow-up Experiments

1. **CC-SSM + DPLR hybrid**: Use Cayley-circulant for the base and add a low-rank perturbation (DPLR-style) for controlled deviation from orthogonality: $W = \text{Cayley}(A_{\text{circ}}) + PQ^*$. This breaks orthogonality but adds expressivity.

2. **Multi-head with different damping**: Some heads use exact orthogonality ($|\lambda| = 1$, long-range recall), others use strong damping ($|\lambda| \ll 1$, local processing). Tests whether heterogeneous stability profiles improve overall performance.

3. **Block-circulant extension**: Replace single circulant with block-circulant (blocks of size $b$), giving $O(bn \log(n/b))$ cost with richer structure. Each block can have its own Cayley parameterization.

4. **Cayley-circulant as DeltaNet state transition**: Use CC-SSM transition matrix as the state update in DeltaNet's rank-1 update framework, replacing the Householder reflection with a circulant rotation.

## Human Review

(To be filled by reviewer)

## References

- Helfrich, Willmott, and Ye (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML.
- Huhtanen and Peramaki (2015). Factoring matrices into the product of circulant and diagonal matrices. J. Fourier Anal. Appl.
- Gu, Goel, and Re (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4).
- Rusch et al. (2024). Oscillatory State-Space Models (LinOSS).
- Dao and Gu (2024). Transformers are SSMs (Mamba-2/SSD).

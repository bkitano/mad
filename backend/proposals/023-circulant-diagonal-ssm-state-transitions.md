---
status: ongoing
priority: high
created: 2026-02-15
based_on: cdflow-circulant-diagonal-invertible, input-dependent-gating, chunkwise-parallel-scan, recurrence-to-scan-reduction, optimal-circulant-approximation, block-circulant-matrices, cauchy-kernel-trick
experiment_number: 023
experiment_log: experiment-log-023.md
---

# Circulant-Diagonal SSM State Transitions

## Hypothesis

Parameterizing SSM state transitions as input-dependent circulant-diagonal (CD) products $A_t = D_1(x_t) \cdot C(x_t) \cdot D_2(x_t)$ will achieve expressivity comparable to full dense transitions at $O(n \log n)$ per-step cost, while preserving parallel scan compatibility through efficient composition of CD factors in the Fourier domain.

## Background

A fundamental tension in SSM design is the tradeoff between state-mixing expressivity and computational efficiency:

- **Diagonal SSMs** (S4D, Mamba): $O(n)$ per step, but no cross-state interaction — provably limited to abelian group automata ($\text{TC}^0$)
- **Full dense transitions** (DeltaNet, DeltaProduct): $O(n^2)$ per step, can represent arbitrary orthogonal transformations and non-solvable groups
- **Structured alternatives** (Monarch, DPLR, HSS): Various $O(n\sqrt{n})$ to $O(n \log n)$ options with different expressivity-efficiency tradeoffs

The CDFlow trick (Feng & Liao 2025) demonstrates that **any** $n \times n$ matrix can be expressed as at most $2n-1$ alternating circulant and diagonal factors (Huhtanen-Perämäki theorem). In practice, just $m=2$ diagonal + $m-1=1$ circulant factors ($W = D_1 C D_2$, 3 total factors) provide competitive quality. Each factor has trivial inverse and determinant, and the product's matvec costs only $O(n \log n)$ via FFT.

**This decomposition has never been applied to SSM state transitions.** It offers a compelling sweet spot:

1. **Full coordinate mixing** via circulant convolution — unlike diagonal SSMs
2. **$O(n \log n)$ matvec** via FFT — cheaper than $O(n\sqrt{n})$ Monarch or $O(n^2)$ dense
3. **Trivial stability control** — constrain $|d_i| \leq 1$ and $|\hat{c}_i| \leq 1$ to ensure $\|A\|_2 \leq 1$
4. **Natural Fourier-domain composition** — for parallel scan, the product of two CD matrices $(D_1 C D_2)(D_1' C' D_2') = D_1 C (D_2 D_1') C' D_2'$ has $O(n \log n)$ evaluation via FFT

**Critical insight for parallel scan**: The binary associative operator $\bullet$ for the scan requires composing pairs of CD matrices. Two CD products compose as:

$$
(D_1 C D_2) \cdot (D_1' C' D_2') = D_1 \cdot C \cdot \underbrace{(D_2 \cdot D_1')}_{\text{diagonal}} \cdot C' \cdot D_2'
$$

This is a $D C D C D$ structure (5 factors) — still a CD product with $m=3$. After $\log_2 T$ scan steps, the accumulated product has $O(\log T)$ factors. We can either:
- (a) Keep the growing product in factored form and apply in $O(n \log n \cdot \log T)$ time
- (b) Periodically collapse via FFT: represent the full product as a single dense matrix in $O(n^2)$ at chunk boundaries (amortized over chunk size $C$)
- (c) Use a **truncated CD approximation**: after each composition, project back to a fixed number of factors using optimal circulant approximation (T. Chan's preconditioner)

Option (c) is the most interesting: after composing two 3-factor CD products into a 5-factor product, we approximate the result as a 3-factor product using the optimal circulant approximation trick. This maintains $O(n \log n)$ cost throughout the scan at the cost of controlled approximation error.

**Gap filled**: No existing proposal combines the CDFlow decomposition with SSM recurrences. Proposal 013 (Circulant SSM) uses plain circulant transitions but lacks the diagonal factors that give CDFlow its universal expressivity. CDFlow's alternating structure strictly subsumes circulant-only transitions.

## Mathematical Formulation

**Standard Diagonal SSM Recurrence:**

$$
h_t = \Lambda h_{t-1} + B x_t, \quad y_t = C h_t
$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$. Complexity: $O(n)$ per step, no cross-state mixing.

**CD-SSM (Proposed):**

$$
h_t = A(x_t) h_{t-1} + B x_t, \quad y_t = C h_t
$$

where the state transition is a circulant-diagonal product:

$$
A(x_t) = D_1(x_t) \cdot \text{circ}(\hat{c}(x_t)) \cdot D_2(x_t)
$$

**Input-Dependent Parameterization:**

$$
D_1(x_t) = \text{diag}(\sigma(x_t W_{d_1})), \quad D_2(x_t) = \text{diag}(\sigma(x_t W_{d_2}))
$$

$$
\hat{c}(x_t) = \tanh(x_t W_c) \quad \text{(Fourier-domain circulant eigenvalues)}
$$

where $\sigma(\cdot)$ is sigmoid (ensures $|d_i| \leq 1$), $\tanh$ ensures $|\hat{c}_i| \leq 1$.

**Stability Guarantee:**

$$
\|A(x_t)\|_2 = \|D_1 \cdot C \cdot D_2\|_2 \leq \|D_1\|_2 \cdot \|C\|_2 \cdot \|D_2\|_2 \leq 1
$$

since $\|C\|_2 = \max_i |\hat{c}_i| \leq 1$ (circulant spectral norm = max eigenvalue magnitude).

**Matvec Computation:**

$$
A(x_t) h_{t-1} = D_1 \cdot \text{IFFT}\left(\hat{c} \odot \text{FFT}(D_2 \cdot h_{t-1})\right)
$$

Cost: $O(n \log n)$ per step (two element-wise products + FFT + IFFT).

**Parallel Scan via Chunkwise CD Composition:**

For chunk size $C$, define the affine scan operator on $(A, b)$ pairs:

$$
(A_2, b_2) \bullet (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)
$$

Within each chunk, compose $C$ CD products. The composition of two CD(3) products yields a CD(5) product. After composing $C$ products, the result has up to $2C+1$ factors.

**Truncated CD Approximation (using optimal circulant approximation):**

After composing $k$ CD products, approximate the result as a single CD(3) product by:

1. Evaluate the composed product on $n$ standard basis vectors: $M = \prod_{i=1}^k (D_1^{(i)} C^{(i)} D_2^{(i)})$ — costs $O(kn \log n)$
2. Extract optimal circulant approximation $C^* = \text{circ}(c^*)$ where $c^*_m = \frac{1}{n}\sum_j M_{j,(j+m) \bmod n}$ (diagonal averaging) — costs $O(n^2)$
3. Solve for diagonal corrections $D_1^*, D_2^*$ to minimize $\|M - D_1^* C^* D_2^*\|_F$ — costs $O(n^2)$

This is done once per chunk boundary, amortized over $C$ steps.

**Alternative: Direct Fourier-Domain Scan:**

Since circulant matrices diagonalize in the Fourier basis, we can work entirely in the frequency domain within chunks:

$$
\hat{h}_t = \hat{D}_1(x_t) \odot \hat{c}(x_t) \odot \hat{D}_2(x_t) \odot \hat{h}_{t-1} + \hat{B} \hat{x}_t
$$

where $\hat{D}_i(x_t) = \text{FFT}(\text{diag}(d_i(x_t)) \cdot \text{IFFT}(\cdot))$ — but diagonal matrices do NOT diagonalize under FFT, so this approach doesn't simplify cleanly. The chunkwise materialization approach (option b) is more practical.

**Key Variables:**

- $h_t \in \mathbb{R}^n$ — hidden state at time $t$
- $D_1, D_2 \in \mathbb{R}^{n \times n}$ — input-dependent diagonal matrices (element-wise gating)
- $C \in \mathbb{R}^{n \times n}$ — input-dependent circulant matrix (state mixing via convolution)
- $\hat{c} \in \mathbb{C}^n$ — Fourier-domain eigenvalues of $C$ (learnable)
- $n$ — state dimension
- $T$ — sequence length
- $C$ — chunk size for parallel scan

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | CD-SSM (circulant-diagonal state transitions) |
| Layers | $L = 6$ |
| Hidden dim | $d_{\text{model}} = 256$ |
| State dim | $n = 64$ |
| Heads | $H = 4$ (multi-head, $n/H = 16$ per head) |
| CD factors | $m = 2$ (3 total factors: $D_1 C D_2$) |
| Chunk size | $C = 64$ |
| Parameters | ~8M |
| Gating | Input-dependent $D_1, D_2, \hat{c}$ with sigmoid/tanh activation |

### Baseline

1. **Diagonal SSM (S4D/Mamba-style)**: $A = \text{diag}(\lambda)$, $O(n)$ per step — no mixing
2. **Circulant-only SSM** (Proposal 013): $A = C$, $O(n \log n)$ per step — mixing but no selective gating
3. **Monarch-SSM** (Proposal 006): $A = P_b^T L P_b R$, $O(n\sqrt{n})$ per step — structured mixing
4. **Full dense DeltaNet**: $O(n^2)$ per step — maximal expressivity

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ state tracking | $> 90\%$ accuracy | Synthetic benchmark |
| Modular arithmetic | $> 95\%$ | Abelian baseline |
| Throughput | $> 1.5\times$ diagonal SSM baseline | Tokens/sec on A100 |
| Memory | $< 1.5\times$ diagonal SSM | Peak GPU memory |
| WikiText-103 perplexity | Competitive with Mamba | LM quality check |
| LRA average | $\geq 85\%$ | Long-range benchmark |

### Estimated Compute

**Small**. Synthetic state-tracking experiments: < 5 GPU-hours. WikiText-103 + LRA: < 50 GPU-hours total. All on single A100.

## Expected Outcome

**If hypothesis is correct:**

- CD-SSM achieves $> 90\%$ on $S_5$ state tracking (vs $< 50\%$ for diagonal SSM), demonstrating that CD products provide sufficient mixing for non-solvable group simulation
- CD-SSM matches or exceeds Monarch-SSM on expressivity benchmarks while being $1.2$–$1.5\times$ faster due to FFT-based $O(n \log n)$ vs butterfly $O(n\sqrt{n})$
- The chunkwise scan with periodic CD re-approximation introduces $< 5\%$ quality degradation vs exact (dense materialization at every chunk boundary)
- On language modeling: competitive perplexity with Mamba, slightly worse than full dense DeltaNet

**If hypothesis is wrong:**

- **Scenario A**: CD approximation error accumulates across scan steps, degrading long-sequence quality → learn that truncated CD composition is not viable, try exact Fourier-domain scan instead
- **Scenario B**: CD mixing is insufficient for $S_5$ despite theoretical universality → suggests 3 factors are not enough for input-dependent transitions; try $m=3$ (5 factors)
- **Scenario C**: FFT overhead dominates at small $n$, no speedup over Monarch → learn the crossover point $n^*$ where CD becomes faster

## Minimum Viable Experiment

### Setup

- **Model**: 1-layer CD-SSM, $n = 32$, $d_{\text{model}} = 64$, ~50K params
- **Task**: $S_3$ (symmetric group on 3 elements) state tracking — simplest non-abelian group
- **Data**: 5K sequences of length 32, each a random product of $S_3$ generators
- **Compute**: Single GPU, $< 5$ minutes

### Implementation

```python
# CD-SSM forward step
def cd_ssm_step(h, x, W_d1, W_d2, W_c, W_b):
    """One step of CD-SSM recurrence."""
    d1 = torch.sigmoid(x @ W_d1)  # (batch, n)
    d2 = torch.sigmoid(x @ W_d2)  # (batch, n)
    c_hat = torch.tanh(x @ W_c)   # (batch, n) Fourier eigenvalues

    # A(x) @ h = D1 * IFFT(c_hat * FFT(D2 * h))
    h_gated = d2 * h
    h_fft = torch.fft.fft(h_gated, dim=-1)
    h_mixed = torch.fft.ifft(c_hat * h_fft, dim=-1).real
    h_new = d1 * h_mixed + x @ W_b
    return h_new
```

### Success Criteria

- CD-SSM achieves $> 95\%$ accuracy on $S_3$ state tracking
- Diagonal SSM baseline achieves $< 40\%$ (near random = 16.7%)
- Circulant-only SSM achieves intermediate accuracy ($60$–$80\%$)
- Training converges within 500 steps

### Failure Criteria

- If CD-SSM achieves $< 50\%$ on $S_3$, the circulant mixing is fundamentally broken for state tracking → kill the idea
- If circulant-only matches CD-SSM, diagonal factors add no value → simplify to circulant-only (Proposal 013)

### Why This Test Is Sufficient

- $S_3$ is the simplest non-abelian group (6 elements, 2 generators). If CD mixing can't track $S_3$ states, it can't track anything requiring cross-state interaction
- The test isolates the core mechanism: does the $D_1 C D_2$ product provide sufficient state mixing for non-trivial group computation?
- Success on $S_3$ strongly predicts success on larger groups ($S_5$) because the mixing mechanism is the same — only the state dimension changes

## Theoretical Analysis

**Complexity Comparison:**

| Operation | Diagonal SSM | CD-SSM ($m=2$) | Monarch SSM | Dense SSM |
|-----------|-------------|----------------|-------------|-----------|
| Per-step matvec | $O(n)$ | $O(n \log n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Scan composition | $O(n)$ | $O(n \log n)$ or $O(n^2)$ | $O(n^{3/2})$ | $O(n^2)$ |
| Parameters per head | $O(n)$ | $O(n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Expressivity | Abelian only | Universal (HP theorem) | $\geq$ butterfly | Universal |
| Stability control | Trivial | Trivial | Non-trivial | Non-trivial |

**Crossover Analysis:**

CD-SSM's $O(n \log n)$ beats Monarch's $O(n\sqrt{n})$ when $\log n < \sqrt{n}$, i.e., always for $n > 1$. In practice, FFT has larger constants than butterfly, so the actual crossover depends on implementation. At $n = 64$ (typical state dim), $\log_2 64 = 6$ vs $\sqrt{64} = 8$, suggesting CD-SSM should be faster.

CD-SSM beats dense when $n \log n < n^2$, i.e., $\log n < n$, which is always true. Practical crossover is at $n \approx 16$–$32$ due to FFT overhead.

**Expressivity Analysis:**

By the Huhtanen-Perämäki theorem, $2n-1$ alternating circulant-diagonal factors can represent any $n \times n$ matrix. With $m=2$ (3 factors), the representable set is a strict subset but empirically sufficient for most structured matrices. The key question is whether 3 factors suffice for the *input-dependent* transitions needed in selective SSMs.

## Risks & Limitations

1. **FFT overhead at small $n$**: For $n = 16$ (common in Mamba), the FFT overhead may negate the theoretical advantage. Need to benchmark carefully.

2. **Complex arithmetic**: FFT involves complex numbers internally. While IFFT output is real for real-valued circulant kernels, intermediate complex computation may be slower on some hardware.

3. **Scan composition blowup**: If we don't truncate, the number of CD factors doubles at each scan level. With chunk size $C = 64$ and $\log_2 C = 6$ scan levels, we'd have $2^6 + 1 = 65$ factors — worse than dense. The truncated approximation is essential.

4. **Circulant bias**: Circulant matrices impose translation-equivariance within the state space. If the optimal state transition has very non-circulant structure, the CD approximation with few factors may be poor. The diagonal factors help break this symmetry.

5. **Training dynamics**: Input-dependent CD parameters may be harder to optimize than input-dependent diagonal parameters. The FFT introduces non-trivial gradient dynamics.

## Follow-up Experiments

1. **Factor count ablation**: Compare $m = 1$ (circulant only), $m = 2$ (3 factors), $m = 3$ (5 factors), and $m = 4$ (7 factors) to find the expressivity-efficiency sweet spot
2. **g-circulant extension**: Replace standard circulant ($g=1$ shift) with $g$-circulant (shift by $g$ positions) using the block-g-circulant-dct-dst trick for potentially richer eigenvalue structure
3. **Real-only arithmetic**: Use DCT/DST decomposition instead of FFT to avoid complex arithmetic, following the real-block-circulant-dct-dst trick
4. **Hybrid CD + diagonal**: Use CD transitions in some heads and diagonal in others (cheap heads for local, CD heads for global mixing)
5. **CDFlow initialization**: Initialize CD parameters from the optimal circulant approximation of HiPPO or S4 initialization matrices
6. **Scale to 1B+ params**: If small-scale results are positive, scale to Mamba-2 style architectures

## Human Review

(To be filled by reviewer)

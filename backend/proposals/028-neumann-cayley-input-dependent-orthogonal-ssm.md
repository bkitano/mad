---
status: ongoing
priority: high
created: 2026-02-15
based_on: cayley-contractive-parameterization, neumann-series-approximate-inverse, input-dependent-gating, recurrence-to-scan-reduction, chunkwise-parallel-scan, householder-product-parameterization, smooth-ste-continuous-sparse-projection
experiment_number: 028
---

# Neumann-Accelerated Input-Dependent Cayley Orthogonal SSM

## Hypothesis

Replacing the exact $O(n^3)$ matrix inversion in the Cayley transform with a $k$-term Neumann series approximation will enable **input-dependent orthogonal state transitions** at $O(kn^2)$ per token (dominated by $k$ GEMMs), achieving near-exact orthogonality ($\|W^TW - I\| < 10^{-4}$ for $k=4$) while being fully parallelizable via chunkwise scan. This will match or exceed DeltaProduct's expressivity on state-tracking tasks while providing stability guarantees that DeltaProduct lacks, at comparable or lower per-step cost.

## Background

### The Core Problem: Input-Dependent Orthogonality is Expensive

The critical insight from the human review of Proposal 001 is that DeltaNet *already* learns orthogonal matrices through composed Householder reflections, but does so "slowly" (one reflection per step). DeltaProduct applies multiple reflections per step for faster convergence to arbitrary orthogonal matrices, but loses parallelizability.

The ideal would be: **an input-dependent orthogonal transition that is cheap per-step and parallelizable**. The Cayley transform $W(x_t) = (I + A(x_t))^{-1}(I - A(x_t))$ achieves exact orthogonality for any skew-symmetric $A(x_t)$, but the per-token matrix inversion $(I + A(x_t))^{-1}$ costs $O(n^3)$ -- making it impractical when $A$ varies per token.

### The Neumann Connection

For skew-symmetric $A$ with bounded spectral norm $\|A\|_2 < 1$, the matrix $I + A$ is well-conditioned, and:

$$
(I + A)^{-1} = \sum_{j=0}^{\infty} (-A)^j = I - A + A^2 - A^3 + \cdots
$$

converges because $\rho(-A) = \|A\|_2 < 1$. Truncating at $k$ terms gives:

$$
(I + A)^{-1} \approx S_k(-A) = \sum_{j=0}^{k-1} (-A)^j
$$

The approximate Cayley transform becomes:

$$
\tilde{W}_k = S_k(-A) \cdot (I - A)
$$

For $k = 4$ and $\|A\|_2 \leq 0.5$, the error is $\|W - \tilde{W}_k\|_2 \leq \|A\|_2^k / (1 - \|A\|_2) < 0.125$, and crucially, the approximate matrix is still *nearly* orthogonal: $\|\tilde{W}_k^T \tilde{W}_k - I\| = O(\|A\|_2^k)$.

### Why This Combination is Novel

| Approach | Orthogonality | Per-token cost | Input-dependent | Parallelizable |
|----------|--------------|----------------|-----------------|----------------|
| Cayley (exact) | Exact | $O(n^3)$ | Yes | No (inversion) |
| Householder product ($n_h$) | Exact | $O(n \cdot n_h)$ | Yes | No (sequential) |
| DeltaNet (1 Householder) | Approximate | $O(n)$ | Yes | Yes (WY/UT) |
| DeltaProduct ($n_h$ Householder) | Approximate | $O(n \cdot n_h)$ | Yes | No |
| Diagonal + clipping | Approximate | $O(n)$ | Yes | Yes |
| **Neumann-Cayley (this)** | Near-exact | $O(kn^2)$ | Yes | Yes (scan) |

**Key insight**: Proposal 011 already proposed using Neumann series for the Woodbury resolvent in DPLR-SSMs. This proposal is fundamentally different: we use Neumann to approximate the *Cayley transform itself*, which gives near-orthogonal (not just stable) transitions. The Cayley structure guarantees that even the approximate matrix has eigenvalues near the unit circle, while DPLR + Neumann only guarantees eigenvalues inside the unit disk.

## Mathematical Formulation

### Step 1: Input-Dependent Skew-Symmetric Matrix

Given input $x_t \in \mathbb{R}^d$, project to skew-symmetric parameters:

$$
v_t = W_{\text{proj}} x_t \in \mathbb{R}^{n(n-1)/2}
$$

Construct skew-symmetric $A(x_t) \in \mathbb{R}^{n \times n}$:

$$
A(x_t)_{ij} = \begin{cases} v_t^{(\text{idx}(i,j))} & \text{if } i < j \\ -v_t^{(\text{idx}(j,i))} & \text{if } i > j \\ 0 & \text{if } i = j \end{cases}
$$

Apply spectral norm scaling to ensure $\|A(x_t)\|_2 \leq \rho_{\max} < 1$:

$$
A_{\text{scaled}}(x_t) = \rho_{\max} \cdot \frac{A(x_t)}{\max(\|A(x_t)\|_2, \rho_{\max})}
$$

where $\rho_{\max} \in (0, 1)$ is a hyperparameter (e.g., $0.5$). The spectral norm is estimated via 1-2 power iterations (amortized across tokens).

### Step 2: Neumann-Approximate Cayley Transform

$$
\tilde{W}_k(x_t) = \left(\sum_{j=0}^{k-1} (-A_t)^j\right) (I - A_t) = S_k(-A_t) (I - A_t)
$$

Using radix-2 binary splitting for efficiency:

$$
S_{2m}(-A) = S_m(-A) \cdot (I + A^m)
$$

For $k = 4$: compute $A^2$ (1 GEMM), then $S_4 = (I - A)(I + A^2)$ (1 GEMM), then $\tilde{W}_4 = S_4 (I - A)$ (1 GEMM). Total: **3 GEMMs** for a near-orthogonal transition.

### Step 3: State Update and Scan

The recurrence is:

$$
h_t = \tilde{W}_k(x_t) h_{t-1} + B x_t
$$

For the associative scan, define tuples $(M_t, b_t)$ with operator:

$$
(M_2, b_2) \bullet (M_1, b_1) = (M_2 M_1, M_2 b_1 + b_2)
$$

Each composition requires one $n \times n$ GEMM ($M_2 M_1$) and one GEMV ($M_2 b_1$).

### Orthogonality Error Analysis

**Lemma**: For $\|A\|_2 \leq \rho$ and $k$-term Neumann approximation:

$$
\|\tilde{W}_k^T \tilde{W}_k - I\|_2 \leq \frac{2\rho^k}{1 - \rho} + \frac{\rho^{2k}}{(1 - \rho)^2}
$$

| $\rho_{\max}$ | $k = 2$ | $k = 4$ | $k = 6$ |
|---------------|---------|---------|---------|
| 0.3 | $3.7 \times 10^{-1}$ | $1.7 \times 10^{-2}$ | $7.8 \times 10^{-4}$ |
| 0.5 | $1.0$ | $1.7 \times 10^{-1}$ | $3.9 \times 10^{-2}$ |
| 0.7 | $3.3$ | $1.1$ | $4.0 \times 10^{-1}$ |

With $\rho_{\max} = 0.3$ and $k = 4$, orthogonality deviation is $< 2\%$ -- sufficient for stability during training, and the gradient signal from the near-orthogonal constraint naturally regularizes.

### Optional: Banded Skew-Symmetric for Lower Cost

Instead of full $n \times n$ skew-symmetric, use bandwidth-$b$ banded skew-symmetric:

$$
A(x_t)_{ij} = 0 \quad \text{if } |i - j| > b
$$

This gives:
- $O(nb)$ parameters (vs $O(n^2/2)$ full)
- $O(n^2 b)$ per GEMM (vs $O(n^3)$ dense) for banded matrix products
- Sufficient expressivity when $b \geq 4$-$8$ (empirical)

**Key Variables:**
- $x_t \in \mathbb{R}^d$ -- input at time $t$
- $h_t \in \mathbb{R}^n$ -- hidden state
- $A_t \in \mathbb{R}^{n \times n}$ -- input-dependent skew-symmetric matrix, $\|A_t\|_2 \leq \rho_{\max}$
- $\tilde{W}_k(x_t) \in \mathbb{R}^{n \times n}$ -- near-orthogonal transition matrix ($k$-term Neumann-Cayley)
- $k$ -- Neumann truncation order (typically 3-6)
- $\rho_{\max}$ -- spectral radius bound (hyperparameter, typically 0.3-0.5)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Neumann-Cayley SSM (NC-SSM) |
| Layers | $L = 12$ |
| Hidden dim | $d = 256$ |
| State dim | $n = 16$ (small, since per-step is $O(kn^2)$) |
| Neumann order | $k = 4$ |
| Spectral bound | $\rho_{\max} = 0.3$ |
| Heads | $H = 16$ (each with $n = 16$ state dims) |
| Scan | Chunkwise parallel, chunk size $C = 64$ |
| Gating | SwiGLU output gate (standard) |

### Baseline

| Model | Per-step | Stability | Mixing | Params |
|-------|----------|-----------|--------|--------|
| S4D/Mamba (diagonal) | $O(n)$ | $|\lambda| \leq 1$ (clip) | None | $O(n)$ |
| DeltaNet (1 Householder) | $O(n)$ | Approximate ortho | Rank-1/step | $O(n)$ |
| DeltaProduct ($n_h = 4$) | $O(4n)$ | Approximate ortho | Rank-4/step | $O(4n)$ |
| LinOSS (oscillatory) | $O(n)$ | $|\lambda| \leq 1$ by constr. | $2 \times 2$ block | $O(n)$ |
| **NC-SSM (this)** | $O(kn^2) = O(4 \cdot 16^2)$ | Near-ortho ($< 2\%$ error) | Full dense | $O(n^2)$ |

Note: With $n = 16$ and $k = 4$, the per-step cost is $4 \times 256 = 1024$ FLOPs -- comparable to DeltaProduct with $n_h = 4$ on state dim $n = 256$ ($4 \times 256 = 1024$ FLOPs). The key difference is that NC-SSM achieves *full* mixing instead of rank-4 mixing.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ State Tracking | $> 90\%$ accuracy | MAD synthetic benchmark |
| MQAR (Multi-Query Assoc. Recall) | $> 90\%$ at $T = 512$ | Standard MQAR benchmark |
| Copy Task (delay 1000) | $> 95\%$ | Synthetic delayed copy |
| Orthogonality deviation | $\|\tilde{W}^T\tilde{W} - I\| < 0.02$ | Monitored during training |
| Throughput | $> 0.5\times$ Mamba | Tokens/sec on A100 |
| Memory | $< 2\times$ Mamba | Peak GPU memory |

### Estimated Compute

**Small:** 4-8 GPU-hours on single A100. MVE under 10 minutes.

## Expected Outcome

**If hypothesis is correct:**
- NC-SSM achieves $> 90\%$ on $S_5$ state tracking (vs $< 40\%$ for diagonal SSMs, $\sim 80\%$ for DeltaNet)
- Near-orthogonal transitions provide robust long-range recall on copy tasks (within 5% of exact orthogonal baseline)
- The $O(kn^2)$ cost with small $n = 16$ is practical -- comparable to DeltaProduct's $O(n_h \cdot n)$ with $n_h = 4, n = 256$
- Chunkwise parallel scan makes training efficient despite dense state transitions

**If hypothesis is wrong:**
- If $k = 4$ Neumann terms at $\rho = 0.3$ produces training instability, it establishes the minimum orthogonality precision needed for stable SSM training (valuable for all future orthogonal SSM designs)
- If the small state dimension $n = 16$ limits capacity, it reveals that the bottleneck is state *size*, not mixing *quality* -- guiding future work toward efficient large-state methods
- If full dense mixing at $n = 16$ underperforms DeltaNet's rank-1 mixing at $n = 256$, it demonstrates that **dimension beats mixing quality** for SSMs

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer NC-SSM, $d = 64$, $n = 8$, $k = 4$, $\rho_{\max} = 0.3$, ~60K params
- **Task**: Permutation composition -- input two permutations of $\{1, \ldots, 5\}$, output their product (i.e., $S_5$ group operation)
- **Data**: 10K random pairs of $S_5$ elements (120 group elements, so task is learnable)
- **Compute**: Single GPU, < 5 minutes

### Success Criteria
- $> 80\%$ accuracy on $S_5$ composition where a diagonal SSM baseline achieves $< 30\%$
- Orthogonality deviation $\|\tilde{W}_k^T \tilde{W}_k - I\|_F < 0.1$ maintained throughout training
- Training loss converges (no divergence from approximate orthogonality)

### Failure Criteria
- If the model diverges (NaN/Inf) within 1000 steps, the Neumann approximation is too loose for stability at these hyperparameters
- If $S_5$ accuracy is $< 40\%$ even with 5000 training steps, the combination of small state dim + approximate orthogonality cannot represent the necessary group structure

### Why This Test Is Sufficient
- $S_5$ state tracking is the canonical test for "does this SSM architecture go beyond diagonal/abelian capabilities" (established by Merrill et al.)
- If a tiny model can learn $S_5$ composition with Neumann-Cayley transitions, the core mechanism (input-dependent near-orthogonal mixing) is validated
- The diagonal SSM baseline provides a clear expressivity gap to close

## Theoretical Analysis

Complexity comparison:

| Operation | Diagonal SSM | DeltaProduct ($n_h$) | NC-SSM ($k$ terms) | Exact Cayley |
|-----------|-------------|---------------------|---------------------|--------------|
| Transition matvec | $O(n)$ | $O(n \cdot n_h)$ | $O(k n^2)$ | $O(n^3)$ |
| Scan composition | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ |
| Parameters/step | $O(n)$ | $O(n \cdot n_h)$ | $O(n^2/2)$ | $O(n^2/2)$ |
| Stability | Clipped | Approximate | $O(\rho^k)$-near | Exact |
| State tracking | Abelian only | Up to $S_{n_h+1}$ | Full $O(n)$ | Full $O(n)$ |

**Effective comparison point**: NC-SSM with $n = 16$ vs DeltaProduct with $n = 256, n_h = 4$:
- NC-SSM: $4 \times 16^2 = 1024$ FLOPs/step, full mixing of 16 dims
- DeltaProduct: $4 \times 256 = 1024$ FLOPs/step, rank-4 mixing of 256 dims

Both have comparable FLOP cost but fundamentally different representational properties.

**Crossover**: NC-SSM is preferable when full mixing at smaller state dim is more valuable than partial mixing at larger state dim -- i.e., when the task requires tracking complex state interactions (permutations, automata) rather than memorizing many independent facts.

## Risks & Limitations

1. **Spectral norm estimation overhead**: Per-token power iteration adds $O(n^2)$ cost. Can amortize by estimating norm every $m$ tokens and using cached scaling.

2. **Small state dimension**: $n = 16$ limits the number of "state slots" available. May need to compensate with more heads or layers. The total state across $H = 16$ heads is $H \times n = 256$ dimensions, matching Mamba's typical state size.

3. **Accumulation of orthogonality error over long sequences**: Each step introduces $O(\rho^k)$ error. Over $T$ steps of scan composition, the accumulated error in the composed transition matrix could be $O(T \cdot \rho^k)$. For $T = 8192, \rho = 0.3, k = 4$: accumulated $\approx 8192 \times 0.0081 \approx 66$ -- potentially problematic. **Mitigation**: Periodic re-orthogonalization at chunk boundaries (every $C = 64$ steps) via QR or Cayley re-projection, costing $O(n^3)$ but amortized over $C$ steps.

4. **Gradient through Neumann series**: Backpropagation through $k$ matrix products. Memory cost: $O(k \cdot n^2)$ per token for intermediate matrices. Can use gradient checkpointing.

5. **The banded variant may lose expressivity**: Banded skew-symmetric limits the rotation axes available. Need to verify empirically whether bandwidth $b = 8$ suffices.

## Follow-up Experiments

1. **Neumann-Cayley + Circulant**: Combine with Proposal 027 -- use Neumann to approximate Cayley of a skew-circulant matrix for $O(kn \log n)$ per step. Best of both worlds: near-exact orthogonality + sub-quadratic cost.

2. **Adaptive $k$ per layer**: Use Hutchinson trace estimator (Proposal 018) to monitor $\|\tilde{W}^T \tilde{W} - I\|$ per layer and allocate more Neumann terms to layers with larger $\|A\|_2$.

3. **Cayley-DeltaNet hybrid**: Use Neumann-Cayley as the "fast" path and DeltaNet-style rank-1 updates as a "slow" correction, combining the stability of near-orthogonal transitions with the fine-grained memory write capability of DeltaNet.

4. **Temperature annealing on $\rho_{\max}$**: Start training with small $\rho_{\max} = 0.1$ (highly accurate Neumann, small rotations) and anneal to $\rho_{\max} = 0.5$ (larger rotations, less accurate) as training progresses.

5. **Sparse Neumann-Cayley**: Apply 2:4 structured sparsity (Proposal 024) to the skew-symmetric matrix $A$, giving sparse GEMMs in the Neumann series with hardware acceleration.

## Human Review

(To be filled by reviewer)

## References

- Helfrich, Willmott, and Ye (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML.
- Sao, P. (2025). Fast Evaluation of Truncated Neumann Series by Low-Product Radix Kernels. arXiv:2602.11843.
- Gu, Goel, and Re (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4).
- Yang et al. (2024). Gated DeltaNet with chunkwise parallel training.
- Kerg et al. (2019). Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics.
- Merrill et al. (2024). The Illusion of State in State-Space Models.

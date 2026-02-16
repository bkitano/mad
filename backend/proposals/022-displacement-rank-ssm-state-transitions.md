---
status: completed
priority: high
created: 2026-02-15
based_on: displacement-rank-cauchy-like-matrices, cauchy-kernel-trick, recurrence-to-scan-reduction, chunkwise-parallel-scan, semiseparable-block-decomposition, woodbury-resolvent-identity, fast-kernel-transform
experiment_number: 022
results_file: 022_results.md
---

# Displacement-Rank SSM: Cauchy-Like State Transitions with Structured Parallel Scan

## Hypothesis

Parameterizing SSM state transition matrices as **Cauchy-like matrices with low displacement rank** $\alpha$ — instead of diagonal (Mamba), diagonal-plus-low-rank (S4), or Monarch-factored (Proposal 006) — enables a new point in the expressivity-efficiency tradeoff: $O(\alpha n \log n)$ per-step cost with **full coordinate mixing** (not just block-diagonal or cyclic), **closure under composition** (critical for parallel scan), and **native Cauchy kernel trick compatibility** for $O(\alpha (n + L) \log^2(n + L))$ convolutional training mode.

## Background

### The State Transition Parameterization Spectrum

The expressivity of an SSM is determined by its state transition class. The documented tricks reveal a clear hierarchy:

| Transition Class | Cost per Step | Mixing | Parallel Scan | Convolutional |
|-----------------|---------------|--------|---------------|---------------|
| Diagonal (Mamba-2) | $O(n)$ | None | ✓ (scalar) | ✓ (FFT) |
| DPLR (S4) | $O(rn)$ | Rank-$r$ | ✓ (via Woodbury) | ✓ (Cauchy) |
| Monarch (Prop 006) | $O(n\sqrt{n})$ | Block | ✓ (BMM) | ✗ |
| Circulant (Prop 013) | $O(n \log n)$ | Cyclic | ✓ (FFT domain) | ✓ (FFT) |
| Column-sparse | $O(n)$ | Permutation | ✓ (sparse) | ✗ |
| Dense | $O(n^2)$ | Full | ✓ (but expensive) | ✗ |

**The gap**: There is no parameterization that achieves (1) full non-cyclic coordinate mixing, (2) sub-quadratic cost, (3) closure under composition, AND (4) compatibility with the Cauchy kernel trick for convolutional training. Cauchy-like matrices with displacement rank $\alpha$ fill this gap.

### Why Displacement Rank Is Natural for SSMs

The displacement rank framework (Kailath, Kung & Morf 1979; Thomas, Gu, Dao et al. 2019) provides a unified view of structured matrices. A matrix $A$ has displacement rank $\alpha$ with respect to operators $(M, N)$ if:

$$
MA - AN = GH^\top, \quad G, H \in \mathbb{R}^{n \times \alpha}
$$

The critical property for SSMs is **closure under composition**: if $A_1$ has displacement rank $\alpha_1$ and $A_2$ has displacement rank $\alpha_2$ (with respect to the same operators), then $A_1 A_2$ has displacement rank $\leq \alpha_1 + \alpha_2$. This means the parallel scan operator — which composes state transitions — produces matrices whose displacement rank grows **additively** per composition, not multiplicatively.

For a chunkwise scan with chunk size $C$, the cumulative transition within a chunk is:

$$
\bar{A}_c = A_{cC+C} \cdot A_{cC+C-1} \cdots A_{cC+1}
$$

If each $A_t$ has displacement rank $\alpha$, then $\bar{A}_c$ has displacement rank $\leq C\alpha$. With rank truncation back to $\alpha$ at chunk boundaries (analogous to how Mamba-2 contracts the state), the effective rank stays bounded.

### Connection to S4's Cauchy Kernel

S4 already uses the displacement rank framework implicitly: its DPLR state matrix $A = \Lambda - PQ^*$ is a rank-$r$ perturbation of a diagonal, which is Cauchy-like with displacement rank $r$ when $M = N = \text{diag}(\lambda)$. The Cauchy kernel trick evaluates the transfer function:

$$
\hat{K}(\omega) = C^\top (e^{i\omega} I - A)^{-1} B = \sum_{j} \frac{\tilde{C}_j \tilde{B}_j}{\omega - \lambda_j} + \text{low-rank correction}
$$

Our proposal generalizes this: when $A$ is Cauchy-like with displacement operators $M = \text{diag}(x)$, $N = \text{diag}(y)$ and generators $(G, H)$ of rank $\alpha$, the resolvent $(zI - A)^{-1}$ can be computed via a Cauchy-like system solve in $O(\alpha^2 n \log^2 n)$ — enabling convolutional training for general Cauchy-like transitions, not just DPLR.

### What This Adds Beyond Existing Proposals

- **Proposal 003** (DPLR + Column-Sparse): Adds permutation routing to DPLR but doesn't exploit displacement rank structure for the composition.
- **Proposal 011** (Neumann Resolvent): Approximates the Woodbury resolvent but stays within the DPLR framework.
- **Proposal 013** (Circulant SSM): Uses FFT-diagonalizable transitions, which are displacement rank 2 w.r.t. shift operators — a special case of our framework.

Our proposal treats **displacement rank $\alpha$ as the fundamental capacity knob**, interpolating from diagonal ($\alpha = 0$), through DPLR ($\alpha = r$, diagonal operators) and circulant ($\alpha = 2$, shift operators), all the way to dense ($\alpha = n$).

## Mathematical Formulation

### Displacement-Rank State Transition

At each time step, the state transition is a Cauchy-like matrix:

$$
(A_t)_{ij} = \delta_{ij} \cdot d_i(x_t) + \sum_{k=1}^{\alpha} \frac{G_{ik}(x_t) \cdot H_{jk}(x_t)}{s_i - s_j}
$$

where:
- $d_i(x_t) \in \mathbb{R}$ — input-dependent diagonal (decay/gate)
- $G(x_t), H(x_t) \in \mathbb{R}^{n \times \alpha}$ — input-dependent generators
- $s \in \mathbb{R}^n$ — fixed displacement nodes (chosen at initialization, e.g., roots of unity or Chebyshev nodes)
- $\alpha$ — displacement rank (the capacity parameter)

**Key variables:**
- $A_t \in \mathbb{R}^{n \times n}$ — state transition (Cauchy-like, displacement rank $\alpha$)
- $h_t \in \mathbb{R}^n$ — hidden state
- $B_t \in \mathbb{R}^n$ — input projection
- $C \in \mathbb{R}^n$ — output projection
- $s \in \mathbb{R}^n$ — displacement nodes (fixed)

### Recurrence

$$
h_t = A_t h_{t-1} + B_t x_t
$$

### Input-Dependent Parameterization

$$
d(x_t) = \sigma(x_t W_d) \in (0, 1)^n \quad \text{(diagonal gate)}
$$
$$
G(x_t) = x_t W_G \in \mathbb{R}^{n \times \alpha}, \quad H(x_t) = x_t W_H \in \mathbb{R}^{n \times \alpha} \quad \text{(generators)}
$$

Total per-step parameters for transition: $d_{\text{model}} \times (n + 2n\alpha) = O(d \cdot n \alpha)$.

### Fast Matrix-Vector Product

The Cauchy-like matvec $A_t h_{t-1}$ decomposes as:

$$
A_t h_{t-1} = \underbrace{d(x_t) \odot h_{t-1}}_{\text{diagonal: } O(n)} + \underbrace{\sum_{k=1}^{\alpha} g_k(x_t) \cdot \text{Cauchy}_{s}(h_k(x_t) \odot h_{t-1})}_{\text{Cauchy matvecs: } O(\alpha n \log n)}
$$

where $\text{Cauchy}_s(v)_i = \sum_j v_j / (s_i - s_j)$ is a Cauchy matrix-vector product computable in $O(n \log n)$ via fast multipole or $O(n \log^2 n)$ via the Fast Kernel Transform.

### Parallel Scan via Generator Composition

The scan operator composes affine maps $(A_1, b_1) \bullet (A_2, b_2) = (A_2 A_1, A_2 b_1 + b_2)$. The product $A_2 A_1$ of two Cauchy-like matrices with displacement rank $\alpha$ has displacement rank $\leq 2\alpha$. After each composition, we **truncate** the generators back to rank $\alpha$:

$$
\text{truncate}_\alpha(G' H'^\top) = G'_{:,1:\alpha} H'_{:,1:\alpha}^\top
$$

where $G', H'$ are the rank-$2\alpha$ generators of the product, and truncation keeps the top-$\alpha$ components (via SVD of the $2\alpha \times 2\alpha$ gram matrix — $O(\alpha^3)$ cost, independent of $n$).

**Scan operator on Cauchy-like generators:**

$$
(d_1, G_1, H_1, b_1) \bullet (d_2, G_2, H_2, b_2) = (d_{12}, G_{12}, H_{12}, b_{12})
$$

where:
- $d_{12} = d_2 \odot d_1$ — diagonal composition: $O(n)$
- $(G_{12}, H_{12}) = \text{truncate}_\alpha(\text{compose}(d_1, G_1, H_1, d_2, G_2, H_2))$ — generator composition + truncation: $O(\alpha^2 n \log n + \alpha^3)$
- $b_{12} = A_2 b_1 + b_2$ — affine term: $O(\alpha n \log n)$

**Total scan cost**: $O(T/p + \log p) \times O(\alpha^2 n \log n)$ with $p$ processors.

### Chunkwise Training

For training, use chunkwise parallel scan with chunk size $C$:

1. **Intra-chunk**: Materialize the $C \times C$ attention matrix within each chunk via semiseparable structure (SSD-style). Each entry:

$$
M_{ij}^{(\text{chunk})} = C_i^\top \bar{A}_{i:j} B_j
$$

where $\bar{A}_{i:j}$ is the cumulative Cauchy-like transition. Since displacement rank grows as $O(C\alpha)$ within a chunk, and we only need the matrix entries (not the full product), use the Cauchy structure: $O(C^2 \alpha \log n)$ per chunk.

2. **Inter-chunk scan**: Compose $T/C$ chunk-level transitions using the generator-based scan: $O((T/C) \alpha^2 n \log n)$.

3. **Output**: Expand inter-chunk states via $C$ projections per chunk: $O(TC n)$.

### Convolutional Mode (LTI)

When the transition is time-invariant ($A_t = A$ for all $t$), the transfer function is:

$$
\hat{K}(\omega_j) = C^\top (e^{i\omega_j} I - A)^{-1} B
$$

For Cauchy-like $A$ with displacement operators $\text{diag}(s)$, the resolvent $(zI - A)^{-1}$ is itself Cauchy-like with displacement rank $\alpha + 1$ (since $zI$ adds 1 to the rank). Evaluating at $L$ frequencies costs $O(\alpha^2 n \log^2 n + L \log L)$ via fast Cauchy-like system solving.

This provides a convolutional training path for the LTI warmup phase.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | DR-SSM (Displacement-Rank SSM) |
| Layers | $L = 12$ |
| Model dim | $d = 768$ |
| State dim | $n = 64$ |
| Displacement rank | $\alpha = 4$ |
| Displacement nodes | Chebyshev nodes on $[-1, 1]$ |
| Chunk size | $C = 64$ |
| FFN | SwiGLU, $d_{\text{ff}} = 2048$ |
| Gating | Post-readout sigmoid gate (from Proposal 009) |

### Initialization

1. **Displacement nodes**: $s_i = \cos(\pi(2i-1)/(2n))$ (Chebyshev nodes — well-separated, good for Cauchy stability)
2. **Diagonal**: Initialize $d_i = e^{-\Delta / \tau_i}$ with $\tau_i$ log-spaced from 1 to $n$ (multi-scale decay, following S4)
3. **Generators**: Initialize $G, H \sim \mathcal{N}(0, 1/\sqrt{n\alpha})$ (small perturbation)
4. **LTI warmup**: Train in convolutional mode for first 10% of steps, then switch to input-dependent mode

### Baseline

1. **S4 (DPLR)**: Displacement rank $r = 1$ with diagonal operators — special case of our framework
2. **Mamba-2 (diagonal)**: Displacement rank 0 — degenerate case
3. **Monarch-SSM (Proposal 006)**: $O(n\sqrt{n})$ via BMM — different factorization family
4. **Circulant SSM (Proposal 013)**: Displacement rank 2 with shift operators — special case
5. **Dense transition SSM**: $O(n^2)$ — upper bound on expressivity

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| S5 composition (state tracking) | $> 90\%$ accuracy | Synthetic group composition task |
| WikiText-103 PPL (350M) | $\leq$ Mamba-2 | Validation perplexity |
| Long Range Arena | $> 85\%$ average | 6-task benchmark |
| Throughput | $> 0.5\times$ Mamba-2 | Tokens/sec on A100 |
| Parameter efficiency | $< 1.1\times$ S4 params | Total parameter count |

### Estimated Compute

**Full experiment**: ~300 GPU-hours (A100)
- 350M model, 15B tokens: ~150 GPU-hours
- Baselines (S4, Mamba-2, dense): ~100 GPU-hours
- Ablations ($\alpha \in \{1, 2, 4, 8\}$): ~50 GPU-hours
- Total: ~300 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- **State tracking**: $> 90\%$ on $S_5$ composition at $n = 64$, matching dense transitions but at $O(\alpha n \log n) \ll O(n^2)$ cost. The displacement rank $\alpha = 4$ should provide sufficient mixing (S4 with $\alpha = 1$ already solves many tasks; $\alpha = 4$ adds richer coupling).
- **Perplexity**: Within 2% of Mamba-2, because the additional coordinate mixing from Cauchy structure helps with multi-token dependencies.
- **Throughput**: $0.5$–$0.8\times$ Mamba-2 for $\alpha = 4$ (the $\alpha$ Cauchy matvecs add $\alpha \times O(n \log n)$ vs Mamba's $O(n)$ per step, but the ratio is $\alpha \log n \approx 24$ vs $n = 64$, so only ~$1.5\times$ slower per step).
- **Scaling**: Increasing $\alpha$ from 1 to 8 should show diminishing returns on perplexity but monotone improvement on state-tracking tasks, revealing the expressivity-efficiency tradeoff curve.

**If hypothesis is wrong:**
- **If Cauchy matvecs are too slow**: The $O(n \log n)$ Cauchy matvec may have high constant factors (FFT overhead) that make it slower than the $O(n\sqrt{n})$ BMM of Monarch for practical $n$. This would indicate that Monarch (Proposal 006) is the better hardware-matched solution.
- **If rank truncation in scan loses too much**: The generator truncation from $2\alpha$ to $\alpha$ at each scan step may accumulate errors, degrading quality on long sequences. This would indicate that the displacement rank framework requires exact composition (not truncated), limiting it to convolutional (LTI) mode only.
- **If $\alpha = 1$ (S4) is already sufficient**: If increasing $\alpha$ beyond 1 doesn't help on any task, then the DPLR parameterization is already optimal and the additional Cauchy structure is unnecessary.
- **Either way, we learn**: Where displacement rank sits in the expressivity hierarchy relative to Monarch, circulant, and dense transitions.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer DR-SSM, $d = 64$, $n = 16$, $\alpha \in \{1, 2, 4\}$ (~80K params)
- **Task**: $S_5$ permutation composition — input a sequence of $S_5$ generators, output the composed permutation
- **Data**: 10K sequences of length 20, vocabulary = 2 generators of $S_5$
- **Compute**: Single GPU, $< 10$ minutes
- **Key comparison**: $\alpha = 1$ (S4-equivalent) vs $\alpha = 4$ (richer Cauchy structure) vs diagonal ($\alpha = 0$, Mamba-equivalent) vs dense ($\alpha = n$, upper bound)

### Success Criteria
- **Rank-scaling signal**: $\alpha = 4$ achieves $> 85\%$ accuracy on $S_5$ while $\alpha = 1$ achieves $< 70\%$ and $\alpha = 0$ achieves $< 50\%$
- **Efficiency**: $\alpha = 4$ at $n = 16$ trains at $> 0.6\times$ the speed of dense ($\alpha = n = 16$) while matching its accuracy within 5%
- The Cauchy matvec via `torch.fft` achieves $> 0.3\times$ the throughput of dense matvec at $n = 16$ (to validate the implementation isn't hopelessly slow)

### Failure Criteria
- If $\alpha = 4$ doesn't outperform $\alpha = 1$ on $S_5$, the additional Cauchy mixing capacity is wasted — kill the idea
- If the generator truncation during scan produces $> 10\%$ relative error after 20 composition steps at $\alpha = 4$, truncation is too lossy — pause and investigate alternative compression strategies (e.g., adaptive rank)
- If the `torch.fft`-based Cauchy matvec is $> 10\times$ slower than dense matvec at $n = 16$, the constant factor overhead is too high for practical use

### Why This Test Is Sufficient
- $S_5$ is the canonical benchmark for non-abelian state tracking (used in proposals 001, 006, 010, 016, 017). If displacement rank $\alpha$ controls expressivity for $S_5$, it will also control it for language modeling.
- The $n = 16$ state dimension is small enough that both Cauchy matvec and dense matvec are fast, enabling clean comparison of accuracy without hardware confounds.
- The $\alpha$-scaling experiment ($\alpha \in \{0, 1, 2, 4, 16\}$) directly maps out the expressivity curve, which is the core scientific question.

### Implementation Sketch

```python
import torch
import torch.fft

def cauchy_matvec(s, d, G, H, h):
    """
    Compute A @ h where A_ij = d_i * delta_ij + sum_k G_ik H_jk / (s_i - s_j)

    s: (n,) displacement nodes
    d: (n,) diagonal
    G: (n, alpha) row generators
    H: (n, alpha) col generators
    h: (n,) input vector

    Returns: (n,) output
    Complexity: O(alpha * n * log n) via fast Cauchy transform
    """
    n, alpha = G.shape
    result = d * h  # diagonal part: O(n)

    for k in range(alpha):
        # Cauchy MVM: sum_j H[j,k] * h[j] / (s[i] - s[j])
        # Use partial fraction + FFT for O(n log n)
        # For MVE: use naive O(n^2) first, optimize later
        weighted = H[:, k] * h  # (n,)
        diffs = s[:, None] - s[None, :]  # (n, n)
        diffs.fill_diagonal_(1.0)  # avoid div by zero on diagonal
        cauchy_col = (weighted[None, :] / diffs).sum(dim=1)  # (n,)
        result += G[:, k] * cauchy_col

    return result

def dr_ssm_scan(xs, ds, Gs, Hs, Bs, C, s):
    """
    Displacement-rank SSM forward pass.
    xs: (T, d_model)  input sequence
    ds: (T, n)         diagonal gates
    Gs: (T, n, alpha)  row generators
    Hs: (T, n, alpha)  col generators
    Bs: (T, n)         input projections
    C:  (n,)           output projection
    s:  (n,)           displacement nodes
    """
    T, n = ds.shape
    h = torch.zeros(n)
    outputs = []

    for t in range(T):
        h = cauchy_matvec(s, ds[t], Gs[t], Hs[t], h) + Bs[t] * xs[t, 0]
        outputs.append(C @ h)

    return torch.stack(outputs)
```

## Theoretical Analysis

**Complexity comparison:**

| Operation | Diagonal (Mamba) | DPLR (S4) | DR-SSM (this) | Monarch | Dense |
|-----------|-----------------|-----------|---------------|---------|-------|
| Per-step matvec | $O(n)$ | $O(rn)$ | $O(\alpha n \log n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Scan composition | $O(n)$ | $O(rn + r^3)$ | $O(\alpha^2 n \log n + \alpha^3)$ | $O(n\sqrt{n})$ | $O(n^3)$ |
| Convolutional mode | $O(n + L\log L)$ | $O(rn\log n + L\log L)$ | $O(\alpha^2 n \log^2 n + L\log L)$ | ✗ | ✗ |
| Parameters per step | $O(n)$ | $O(rn)$ | $O(\alpha n)$ | $O(n\sqrt{n})$ | $O(n^2)$ |
| Displacement rank | 0 | $r$ | $\alpha$ | N/A | $n$ |

**Crossover analysis:**
- DR-SSM beats dense when $\alpha \log n < n$, i.e., $\alpha < n / \log n$. For $n = 64$: $\alpha < 10$.
- DR-SSM beats Monarch when $\alpha \log n < \sqrt{n}$, i.e., $\alpha < \sqrt{n}/\log n$. For $n = 64$: $\alpha < 1.3$. This means for $\alpha > 1$, Monarch may be faster per-step — but DR-SSM has the convolutional training advantage.
- DR-SSM's convolutional mode enables $O(L \log L)$ training for LTI warmup, which Monarch cannot do.

**Expressivity hierarchy (formal):**

$$
\text{Diagonal} \subsetneq \text{DPLR}(\alpha=r) \subsetneq \text{Cauchy-like}(\alpha) \subseteq \text{Dense}
$$

The inclusion is strict because Cauchy-like matrices with displacement rank $\alpha$ form a $O(\alpha n)$-dimensional manifold in the $n^2$-dimensional space of matrices, and DPLR is a special case with diagonal displacement operators.

## Risks & Limitations

1. **Cauchy singularity at $s_i = s_j$**: When displacement nodes coincide, the Cauchy matrix is undefined. Mitigation: use well-separated nodes (Chebyshev) and add small $\epsilon$ regularization: $1/(s_i - s_j + \epsilon)$.

2. **Generator truncation error in scan**: Truncating from rank $2\alpha$ to $\alpha$ at each scan step loses information. Over $O(\log T)$ scan levels, the cumulative error could be $O((\sigma_{\alpha+1}/\sigma_\alpha)^{\log T})$ where $\sigma_k$ is the $k$-th singular value. Mitigation: use overparameterized rank $\alpha' = 2\alpha$ during scan, truncate only at chunk boundaries.

3. **FFT overhead for small $n$**: The $O(n \log n)$ Cauchy matvec has high constant factors from FFT. For $n = 64$, a dense $O(n^2) = 4096$ matvec may be faster than $O(64 \cdot 6) = 384$ with FFT overhead. Mitigation: use naive $O(n^2)$ Cauchy matvec for small $n$; the structure benefits appear at $n > 256$.

4. **Stability of input-dependent generators**: Large generator values can produce ill-conditioned Cauchy matrices. Mitigation: normalize generators via $G \leftarrow G / \|G\|_F \cdot \sqrt{\alpha}$ and bound diagonal gates to $(0, 1)$.

5. **Not directly BMM-compatible**: Unlike Monarch, Cauchy matvecs don't map to batch matrix multiply. On tensor-core GPUs, this is a hardware mismatch. Mitigation: implement as specialized CUDA kernel using shared memory for the FFT-based Cauchy transform.

## Follow-up Experiments

1. **Displacement operator learning**: Instead of fixing $s$ at Chebyshev nodes, learn the displacement nodes $s$ via gradient descent. This could discover data-dependent structure (e.g., if $s_i$ cluster, the transition becomes approximately block-diagonal).

2. **Hybrid DR + Monarch**: Use Cauchy-like transitions for layers that need convolutional warmup (early layers) and Monarch for layers that don't (later layers). This combines the convolutional training advantage with Monarch's hardware efficiency.

3. **Stein-type displacement**: Replace Sylvester displacement $MA - AN = GH^\top$ with Stein displacement $A - MAN = GH^\top$. This changes the matrix structure and may be better suited for the contraction $\|A\| < 1$ requirement.

4. **Adaptive rank allocation**: Use Hutchinson trace estimation (Proposal 018) to determine which layers benefit from higher displacement rank $\alpha$, allocating rank budget across the model.

5. **Displacement-rank sparsity**: Apply 2:4 structured sparsity (S-STE) to the generators $G, H$, reducing per-step cost by $2\times$ while maintaining the Cauchy structure.

6. **Connection to Monarch**: Investigate whether Monarch factorizations can be reinterpreted as Cauchy-like matrices with specific displacement operators. If so, this unifies the two frameworks and enables convolutional training for Monarch-SSMs.

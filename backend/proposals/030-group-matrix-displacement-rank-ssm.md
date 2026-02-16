---
status: ongoing
priority: high
created: 2026-02-15
based_on: group-matrix-displacement-equivariance, displacement-rank-cauchy-like-matrices, signed-permutation-hyperoctahedral-group, input-dependent-gating, recurrence-to-scan-reduction, cayley-contractive-parameterization, blockwise-sinkhorn-channel-permutation
experiment_number: 030
---

# Group-Matrix Displacement Rank SSM

## Hypothesis

Parameterizing SSM state transitions as **low displacement rank group matrices** for the hyperoctahedral group $B_n = \mathbb{Z}_2^n \rtimes S_n$ will enable efficient approximately-equivariant state dynamics that combine the expressivity of signed permutation state-tracking (non-abelian group simulation) with $O(|G| \cdot r)$ parameter efficiency ($r \ll |G|$), while providing a principled mechanism to control the deviation from exact equivariance via the displacement rank $r$.

## Background

### The State-Tracking Expressivity Problem

Diagonal SSMs (Mamba, S4D) are limited to abelian group automata ($\text{TC}^0$), because diagonal matrices commute. To simulate non-abelian groups like $S_5$ (required for state-tracking, in-context learning, algorithmic reasoning), the state transition must perform **coordinate permutation** — non-commutative mixing of state dimensions.

Proposal 017 addresses this by parameterizing state transitions as signed permutations in the hyperoctahedral group $B_n = \mathbb{Z}_2^n \rtimes S_n$, which is the maximal finite subgroup of $O(n)$. However, 017 uses Gumbel-Sinkhorn relaxation to learn the permutations, which has two problems: (1) the Sinkhorn iterations add computational overhead, and (2) exact $B_n$ elements are too rigid — real-world state transitions may only be *approximately* equivariant to $B_n$.

### The Group Matrix Framework

The group-matrix displacement framework (Samudre et al., AISTATS 2025) provides an elegant solution. For any finite group $G$ with $N = |G|$ elements, a **group matrix** $M$ encodes an exact group convolution:

$$
M = \sum_{g \in G} \text{diag}(F_g) B_g
$$

where $B_g$ is the group diagonal (permutation matrix for left-multiplication by $g$), and $F_g$ are diagonal coefficient vectors. When $G = \mathbb{Z}/n\mathbb{Z}$ (cyclic), this reduces to a circulant matrix.

**The displacement rank** $\text{DR}(M) = \text{rank}(\text{D}(M))$ measures how far $M$ deviates from being an exact group matrix. At $\text{DR} = 0$, $M$ is exactly $G$-equivariant; at $\text{DR} = r > 0$, $M$ is approximately equivariant with controlled error $O(r/N)$. The parameterization requires only $N_k + N \cdot r$ parameters (kernel size + displacement perturbation) vs $N^2$ for dense.

### The Key Insight

**For SSM state transitions, exact group equivariance is too rigid, but structured deviations from equivariance are beneficial.** The displacement rank provides a *learnable knob* for controlling this deviation:

- At initialization, set $r = 0$ (exact group convolution) — the model starts with strong inductive bias
- During training, the displacement perturbation vectors $\mathbf{a}_1, \ldots, \mathbf{a}_r$ learn task-specific deviations
- The bound $\text{dist}(M, \mathcal{GM}(G)) = O(r \cdot \|\mathbf{a}\|)$ ensures deviations are controlled

### What's Different from Existing Proposals

- **Proposal 017** (Hyperoctahedral SSM): Uses Gumbel-Sinkhorn to learn *exact* signed permutations. No displacement rank, no continuous relaxation of equivariance. Our approach subsumes theirs: at $r = 0$, we recover exact $B_n$ equivariance.
- **Proposal 022** (Displacement Rank SSM): Uses classical LDR (cyclic group displacement) for Toeplitz/Cauchy-like structure. Our approach generalizes to *arbitrary* finite groups, specifically targeting $B_n$ which captures both permutation routing and sign-flipping.
- **Proposal 023** (CD-SSM): Uses circulant-diagonal products ($G = \mathbb{Z}/n\mathbb{Z}$). Our approach uses $G = B_n$, which includes the cyclic group as a subgroup but adds permutation and sign-flip structure.
- **Proposal 016** (GS-Monomial): Uses group-and-shuffle monomial matrices. Our approach uses the *group matrix* framework which provides principled displacement rank control, rather than ad-hoc factored structure.

### Why $B_n$ Specifically?

The hyperoctahedral group $B_n = \mathbb{Z}_2^n \rtimes S_n$ has $|B_n| = 2^n \cdot n!$ elements and captures:

1. **Permutation routing** ($S_n$ part): Arbitrary reordering of state dimensions — needed for non-abelian group simulation
2. **Sign flipping** ($\mathbb{Z}_2^n$ part): Negation of individual dimensions — needed for reflection-like operations and increased expressivity (Krohn-Rhodes decomposition requires both group and aperiodic components)
3. **Orthogonality**: Every element of $B_n$ is an orthogonal matrix ($B_n \subset O(n)$), so state transitions preserve norms — stability by construction

The semi-direct product structure $B_n = \mathbb{Z}_2^n \rtimes S_n$ means we can construct $B_n$ group matrices via the Kronecker product formula:

$$
B_{(g,h)}^{B_n} = (P_h B_g^{\mathbb{Z}_2^n}) \otimes B_h^{S_n}
$$

This construction is efficient because we build $B_n$ from smaller component groups.

## Mathematical Formulation

**Standard Diagonal SSM:**

$$
h_t = \Lambda h_{t-1} + B x_t, \quad \Lambda = \text{diag}(\lambda_i), \quad O(n) \text{ per step}
$$

**Group-Matrix Displacement Rank SSM (GM-DR-SSM):**

$$
h_t = A(x_t) h_{t-1} + B x_t
$$

where the state transition is a **low displacement rank group matrix** for a finite group $G$:

$$
A(x_t) = \underbrace{\sum_{g \in \mathcal{N}} \alpha_g(x_t) B_g}_{\text{group convolution kernel}} + \underbrace{\sum_{i=1}^{r} \text{diag}(\mathbf{a}_i(x_t)) \cdot B_{g_i}}_{\text{displacement perturbation}}
$$

**Input-dependent parameterization:**

$$
\alpha_g(x_t) = \sigma(x_t w_g) \quad \text{(kernel weights, } |\mathcal{N}| \text{ scalars)}
$$

$$
\mathbf{a}_i(x_t) = \tanh(x_t W_{a_i}) \quad \text{(perturbation vectors, } r \text{ vectors of dim } n \text{)}
$$

Here $\mathcal{N} \subset G$ is a small "neighborhood" of the identity (kernel support), and $g_1, \ldots, g_r$ are fixed group elements that anchor the perturbation directions.

**Practical Instantiation for Small $n$:**

For SSM state dimensions $n = 4$–$16$ (typical for Mamba), $|S_n| = n!$ is tractable only for small $n$. We use $B_n$ on a *partitioned* state: divide $n$-dimensional state into $b$ blocks of size $p$, and apply $B_p$ independently to each block, with learned inter-block coupling via the displacement perturbation.

**Block-partitioned GM-DR-SSM:**

$$
A(x_t) = \underbrace{\text{blockdiag}(A_1^{B_p}(x_t), \ldots, A_b^{B_p}(x_t))}_{\text{intra-block } B_p \text{ group conv}} + \underbrace{\sum_{i=1}^{r} \mathbf{a}_i(x_t) \mathbf{e}_{g_i}^\top}_{\text{inter-block coupling (disp. rank } r\text{)}}
$$

For $p = 4$: $|B_4| = 2^4 \cdot 4! = 384$ elements, manageable for pre-computing group diagonals. With kernel size $k = 2$ (identity + one generator), only $2 \times b$ kernel weights per block.

**Stability via Cayley Constraint:**

The group diagonal matrices $B_g$ are orthogonal by definition (permutation matrices). The kernel weights $\alpha_g$ are constrained so that $\sum_g |\alpha_g| \leq 1$ (convex combination ensures contraction). The displacement perturbation is bounded by $\|\mathbf{a}_i\| \leq \epsilon$ via tanh activation, ensuring:

$$
\|A(x_t)\|_2 \leq \underbrace{\sum_{g \in \mathcal{N}} |\alpha_g|}_{\leq 1} + \underbrace{\sum_{i=1}^{r} \|\mathbf{a}_i\|_2}_{\leq r\epsilon} \leq 1 + r\epsilon
$$

For stability, set $\epsilon$ small enough that $r\epsilon < \delta$ for some margin $\delta$. Alternatively, apply Cayley parameterization to the overall skew-symmetric part.

**Parallel Scan Compatibility:**

The scan operator $(A_2, b_2) \bullet (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)$ requires composing state matrices. Two group matrices compose as:

$$
\left(\sum_g \alpha_g B_g + \Delta_1\right) \cdot \left(\sum_h \beta_h B_h + \Delta_2\right) = \sum_{g,h} \alpha_g \beta_h B_{gh} + \text{cross terms}
$$

The product $B_g B_h = B_{gh}$ stays in the group diagonal basis (by the Cayley table), so the group convolution part composes exactly. The displacement perturbation cross-terms increase the displacement rank by at most $r_1 + r_2$, which can be truncated back to $r$ via projection after each composition step.

**Truncated composition (key for scan efficiency):**

After composing two rank-$r$ GM-DR matrices, the result has rank $\leq 2r$. Project back to rank $r$ by keeping the $r$ largest-magnitude perturbation vectors. This introduces approximation error bounded by the discarded perturbation magnitudes — analogous to truncated SVD.

**Key Variables:**

- $h_t \in \mathbb{R}^n$ — hidden state ($n = b \times p$, partitioned into $b$ blocks of size $p$)
- $G = B_p$ — hyperoctahedral group on $p$ elements
- $B_g \in \{0, \pm 1\}^{p \times p}$ — group diagonal (signed permutation matrix)
- $\mathcal{N} \subset G$ — kernel neighborhood ($|\mathcal{N}|$ generators)
- $r$ — displacement rank (controls equivariance deviation)
- $\alpha_g(x_t) \in \mathbb{R}$ — input-dependent kernel weights
- $\mathbf{a}_i(x_t) \in \mathbb{R}^n$ — input-dependent perturbation vectors

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GM-DR-SSM with $B_4$ group structure |
| Layers | $L = 6$ |
| Hidden dim | $d_{\text{model}} = 256$ |
| State dim | $n = 16$ (4 blocks of $p = 4$) |
| Group | $B_4 = \mathbb{Z}_2^4 \rtimes S_4$ ($|B_4| = 384$) |
| Kernel size | $|\mathcal{N}| = 3$ (identity + 2 generators) |
| Displacement rank | $r \in \{0, 1, 2, 4\}$ (ablation) |
| Heads | $H = 4$ (multi-head, $n/H = 4$ per head) |
| Parameters | ~8M |

### Baseline

1. **Diagonal SSM (S4D/Mamba)**: $A = \text{diag}(\lambda)$, $O(n)$ — no mixing, abelian only
2. **Column-sparse SSM (Proposal 001)**: $A = P \cdot D$, $O(n)$ — permutation routing, but no signs
3. **Signed permutation SSM (Proposal 017)**: $A \in B_n$ exactly via Gumbel-Sinkhorn — full $B_n$ but no displacement
4. **Monarch SSM (Proposal 006)**: $A = P_b^\top L P_b R$, $O(n\sqrt{n})$ — butterfly mixing
5. **CD-SSM (Proposal 023)**: $A = D_1 C D_2$, $O(n \log n)$ — circulant mixing

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| $S_5$ state tracking | $> 90\%$ accuracy | Synthetic benchmark ($S_5$ group products) |
| $D_4$ (dihedral) tracking | $> 95\%$ accuracy | Permutation + reflection tracking |
| Modular arithmetic | $> 95\%$ | Abelian sanity check |
| Throughput | $> 1.2\times$ over Gumbel-Sinkhorn baseline | Tokens/sec on A100 |
| LRA average | $\geq 85\%$ | Long-range benchmark |
| WikiText-103 perplexity | Competitive with Mamba-style | Language model quality |

### Estimated Compute

**Small.** Group diagonal precomputation (one-time): $< 1$ minute. State-tracking experiments: $< 10$ GPU-hours. WikiText-103: $< 30$ GPU-hours. LRA: $< 10$ GPU-hours. Total: $< 50$ GPU-hours on single A100.

## Expected Outcome

**If hypothesis is correct:**

- GM-DR-SSM with $r = 2$ achieves $> 90\%$ on $S_5$ state tracking, matching or exceeding Proposal 017 (exact $B_n$)
- The displacement rank ablation shows clear progression: $r = 0$ (exact group conv) struggles with tasks requiring non-equivariant dynamics; $r = 1$–$2$ hits sweet spot; $r \geq 4$ overfits at small scale
- Throughput is $1.2$–$1.5\times$ faster than Gumbel-Sinkhorn (Proposal 017) because we avoid iterative Sinkhorn normalization and Hungarian hardening
- On language modeling: competitive with Mamba at $n = 16$, with the group structure providing better in-context learning capabilities
- The learned displacement perturbations are interpretable — they break equivariance in task-specific ways (e.g., position-dependent state routing)

**If hypothesis is wrong:**

- **Scenario A**: $B_4$ group is too small for $S_5$ tracking (need $p \geq 5$) → try $B_5$ ($|B_5| = 3840$) or subgroup approach
- **Scenario B**: Displacement rank perturbation destabilizes the orthogonal structure → eigenvalues leak outside unit circle, gradient explosion. Fix: stronger Cayley constraint on the perturbation.
- **Scenario C**: Group matrix matvec with precomputed permutations is not faster than Gumbel-Sinkhorn due to memory access patterns → learn that the theoretical advantage doesn't translate to GPU efficiency at small $n$. Useful negative result about the gap between algebraic structure and hardware efficiency.

## Minimum Viable Experiment

### Setup

- **Model**: 1-layer GM-DR-SSM, $n = 4$, $p = 4$ (single block), $d_{\text{model}} = 32$, ~5K params
- **Task**: $S_3$ and $D_4$ (dihedral group of order 8) state tracking
- **Data**: 5K sequences of length 32, random products of group generators
- **Compute**: Single GPU, < 5 minutes

### Implementation

```python
import torch
import torch.nn as nn
from itertools import permutations

def make_B4_group_diagonals():
    """Precompute all 384 group diagonal matrices for B_4.
    Each is a 4x4 signed permutation matrix."""
    diags = []
    for perm in permutations(range(4)):
        for signs in range(2**4):
            mat = torch.zeros(4, 4)
            for i, j in enumerate(perm):
                sign = 1 if (signs >> i) & 1 == 0 else -1
                mat[i, j] = sign
            diags.append(mat)
    return torch.stack(diags)  # (384, 4, 4)

class GMDR_SSM_Step(nn.Module):
    def __init__(self, n=4, kernel_size=3, disp_rank=2, d_in=32):
        super().__init__()
        self.n = n
        self.r = disp_rank
        self.group_diags = make_B4_group_diagonals()  # (384, 4, 4)
        # Select kernel neighborhood (identity + generators)
        self.kernel_idx = [0, 1, 24]  # identity, one transposition, one sign flip
        # Input-dependent kernel weights
        self.W_alpha = nn.Linear(d_in, kernel_size)
        # Input-dependent displacement perturbation
        if disp_rank > 0:
            self.W_a = nn.Linear(d_in, disp_rank * n)
            # Fixed anchor group elements for perturbation
            self.anchor_idx = list(range(1, disp_rank + 1))

    def forward(self, h, x):
        """h: (batch, n), x: (batch, d_in)"""
        # Group convolution kernel
        alpha = torch.softmax(self.W_alpha(x), dim=-1)  # (batch, k)
        A = sum(alpha[:, i:i+1, None] * self.group_diags[self.kernel_idx[i]]
                for i in range(len(self.kernel_idx)))  # (batch, n, n)
        # Displacement perturbation
        if self.r > 0:
            a = torch.tanh(self.W_a(x)).view(-1, self.r, self.n)  # (batch, r, n)
            for i in range(self.r):
                A = A + torch.diag_embed(a[:, i]) @ self.group_diags[self.anchor_idx[i]]
        h_new = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)
        return h_new
```

### Success Criteria

- GM-DR-SSM ($r = 2$) achieves $> 95\%$ accuracy on $S_3$ state tracking (6 states)
- GM-DR-SSM ($r = 2$) achieves $> 90\%$ accuracy on $D_4$ state tracking (8 states)
- Diagonal SSM baseline achieves $< 30\%$ on both (near random)
- Ablation shows $r = 0$ (exact equivariance) underperforms $r = 2$ by $> 10\%$, demonstrating the value of controlled symmetry-breaking

### Failure Criteria

- If GM-DR-SSM ($r = 2$) achieves $< 50\%$ on $S_3$ → the group matrix parameterization fundamentally doesn't enable state tracking, kill the idea
- If $r = 0$ matches $r = 2$ → displacement rank adds no value; simplify to exact group conv (subset of Proposal 017)
- If the model doesn't converge within 1000 steps → optimization landscape of group matrix weights is pathological

### Why This Test Is Sufficient

- $S_3$ (6 elements, 2 generators) is the simplest non-abelian group. If the parameterization can't track $S_3$, it can't track anything non-commutative.
- $D_4$ (8 elements, reflections + rotations) tests the sign-flipping capability of $B_n$ — this is the core value add over plain $S_n$ permutations.
- The displacement rank ablation ($r = 0$ vs $r = 2$) tests the central hypothesis: controlled symmetry-breaking improves expressivity beyond exact equivariance.
- At $n = 4$, the entire $B_4$ group is enumerable (384 elements), so we can verify the computation exactly.

## Theoretical Analysis

**Complexity Comparison:**

| Operation | Diagonal SSM | GM-DR-SSM ($B_p$, $b$ blocks) | Signed Perm SSM (017) | Dense SSM |
|-----------|-------------|-------------------------------|----------------------|-----------|
| Per-step matvec | $O(n)$ | $O(b \cdot |\mathcal{N}| \cdot p + r \cdot n)$ | $O(n^2)$ (Sinkhorn) | $O(n^2)$ |
| Scan composition | $O(n)$ | $O(b \cdot |\mathcal{N}|^2 \cdot p + r^2 \cdot n)$ | $O(n^2)$ | $O(n^2)$ |
| Parameters per head | $O(n)$ | $O(b \cdot |\mathcal{N}| + r \cdot n)$ | $O(n^2)$ | $O(n^2)$ |
| Expressivity | Abelian only | Approx $B_p$-equivariant | Exact $B_n$ | Universal |
| Stability control | Trivial | Near-trivial (orthogonal basis) | Non-trivial | Non-trivial |

For typical values $n = 16$, $b = 4$, $p = 4$, $|\mathcal{N}| = 3$, $r = 2$:

- Per-step: $O(4 \times 3 \times 4 + 2 \times 16) = O(80)$ vs $O(16)$ diagonal vs $O(256)$ dense
- Parameters: $O(4 \times 3 + 2 \times 16) = O(44)$ vs $O(16)$ diagonal vs $O(256)$ dense

**Equivariance Error Bound:**

From Samudre et al. (AISTATS 2025), Proposition 3.1:

$$
\text{dist}(M_1 M_2, \mathcal{GM}(G)) \leq \max\{\|M_1\|, \|M_2\|\} \cdot (\text{dist}(M_1, \mathcal{GM}) + \text{dist}(M_2, \mathcal{GM}))
$$

For a sequence of $T$ compositions with displacement rank $r$ and perturbation magnitude $\epsilon$:

$$
\text{dist}(\prod_{t=1}^T A_t, \mathcal{GM}(G)) \leq T \cdot r \cdot \epsilon
$$

This grows linearly with sequence length, suggesting that for long sequences, the model will naturally learn to reduce perturbation magnitudes (use smaller $\epsilon$) or the chunkwise approach should periodically re-project onto exact group matrices.

**Expressivity Analysis:**

At displacement rank $r = 0$: the transition is an exact $B_p$ group convolution. This can simulate any automaton whose transformation monoid is a subgroup of $B_p$.

At displacement rank $r > 0$: the transition can approximate any $n \times n$ matrix (by the universality of group matrices with sufficient displacement rank). The displacement rank controls the tradeoff:

- $r = 0$: exact equivariance, limited to $B_p$ subgroup dynamics
- $r = O(1)$: approximately equivariant, can represent tasks with soft symmetry
- $r = O(n)$: effectively unconstrained, universal approximation

## Risks & Limitations

1. **Group size scaling**: $|B_p| = 2^p \cdot p!$ grows super-exponentially. For $p = 4$: 384 elements (fine). For $p = 5$: 3840 (borderline). For $p \geq 6$: impractical to enumerate all group diagonals. Must use block-partitioned approach or work with subgroups.

2. **Scan composition approximation**: Truncating the displacement rank after each scan composition step introduces cumulative error. Need to verify this doesn't degrade long-sequence quality.

3. **Block partitioning limits cross-block interaction**: With $b$ independent blocks, cross-block state mixing relies entirely on the displacement perturbation. If $r$ is too small, the model may fail to learn inter-block coordination.

4. **Group diagonal precomputation**: Storing all 384 $B_4$ group diagonals as a $(384, 4, 4)$ tensor requires 6K floats — negligible. But for $B_5$, this grows to $(3840, 5, 5) = 96$K floats per head, which may cause cache pressure during the matvec.

5. **Kernel neighborhood selection**: The choice of $\mathcal{N}$ (which group elements to include in the kernel) significantly affects expressivity. Poor choices (e.g., all from the same cyclic subgroup) would limit the model to abelian dynamics within $B_p$.

6. **Hardware efficiency**: The group convolution involves gathering and scattering along permutation indices, which has irregular memory access patterns. This may be slower than regular matrix operations despite having fewer FLOPs.

## Follow-up Experiments

1. **Kernel neighborhood ablation**: Systematically vary $\mathcal{N}$ — try minimal generators, random subsets, and learned neighborhood selection via attention over group elements
2. **Displacement rank curriculum**: Start training at $r = 0$ (exact equivariance), then gradually increase $r$ during training — similar to complexity curriculum learning
3. **Wreath product FFT acceleration**: For $B_n = \mathbb{Z}_2 \wr S_n$, the wreath product FFT (Clausen-Baum algorithm, from wreath-product-fft-hyperoctahedral trick) computes the DFT over $B_n$ in $O(|B_n| \log |B_n|)$. If we can reformulate the group convolution in the frequency domain, this would enable $O(n \log(n!))$ matvec instead of $O(n \cdot |\mathcal{N}|)$
4. **Integration with blockwise Sinkhorn**: Use the blockwise Sinkhorn channel permutation trick to learn the block partitioning itself — which state dimensions should share a $B_p$ group?
5. **Comparison with Krohn-Rhodes decomposition**: The Krohn-Rhodes theorem decomposes any transformation monoid into simple groups and aperiodic components. Compare the learned GM-DR-SSM factorization with the theoretically optimal Krohn-Rhodes decomposition for specific tasks
6. **Scale test**: If small-scale results are positive, apply GM-DR-SSM as a drop-in replacement for diagonal state matrices in Mamba-2, testing on standard LM benchmarks

## Human Review

(To be filled by reviewer)

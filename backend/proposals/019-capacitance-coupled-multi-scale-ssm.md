---
status: ongoing
priority: high
created: 2026-02-15
based_on: capacitance-matrix-method, hierarchically-semiseparable-hss-matrices, diagonal-plus-low-rank-(dplr), schur-complement-block-inversion, woodbury-resolvent-identity, chunkwise-parallel-scan, input-dependent-gating
experiment_number: 019
experiment_log: experiment-log-019.md
---

# Capacitance-Coupled Multi-Scale SSM

## Hypothesis

A multi-scale SSM architecture where independent SSM blocks operating at different timescales are coupled through a small, learned **capacitance matrix** will achieve better long-range modeling quality than a monolithic SSM of the same total state dimension, while maintaining near-linear complexity — because the capacitance matrix captures cross-scale interactions sparsely ($O(k^2)$ where $k$ is the number of scales) rather than densely ($O(n^2)$).

## Background

Current SSM architectures (Mamba, S4, S5, LinOSS) use a single state transition matrix $A_t \in \mathbb{R}^{n \times n}$ operating at one implicit timescale per head. While DPLR decomposition and input-dependent gating allow some multi-scale behavior through different eigenvalues, the coupling between slow and fast dynamics happens through the same dense matrix — there's no principled separation.

The **capacitance matrix method** from computational physics offers an elegant solution: it decomposes a problem with complex boundary interactions into independent subproblems on regular domains connected through a small "capacitance" matrix at the interfaces. This is precisely the structure we need: independent SSM blocks (each a "subdomain" operating at its own timescale) connected by a small coupling matrix at block boundaries.

**Key gap**: No existing proposal addresses multi-scale temporal structure through principled domain decomposition. Proposals 005 (HSS linear attention) and 008 (cosine-log-linear) hint at hierarchical time but don't decompose the *state space* into coupled scales. Proposal 018 (adaptive rank) adjusts DPLR rank per layer but not per timescale within a layer.

**Why this matters**: Real-world sequences (language, audio, time series) contain nested temporal structures — phonemes within words within sentences within documents. A single SSM blends all scales into one state, forcing the model to allocate capacity across scales implicitly. Explicit scale separation with sparse coupling should be both more interpretable and more efficient.

## Mathematical Formulation

**Standard Single-Scale SSM:**

$$
h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t
$$

with $A_t \in \mathbb{R}^{n \times n}$, $h_t \in \mathbb{R}^n$.

**Proposed Multi-Scale Decomposition:**

Partition the state space into $k$ blocks of sizes $n_1, \ldots, n_k$ (where $\sum_i n_i = n$), each operating at a different timescale $\Delta t_i$. The full state transition becomes:

$$
\underbrace{\begin{pmatrix} h_t^{(1)} \\ h_t^{(2)} \\ \vdots \\ h_t^{(k)} \end{pmatrix}}_{h_t} = \underbrace{\begin{pmatrix} A^{(1)}_t & & & \\ & A^{(2)}_t & & \\ & & \ddots & \\ & & & A^{(k)}_t \end{pmatrix}}_{\text{block-diagonal: independent scales}} \underbrace{\begin{pmatrix} h_{t-1}^{(1)} \\ h_{t-1}^{(2)} \\ \vdots \\ h_{t-1}^{(k)} \end{pmatrix}}_{h_{t-1}} + \underbrace{U \cdot \mathcal{C}_t \cdot V^T}_{\text{capacitance coupling}} h_{t-1} + B_t x_t
$$

where:
- $A^{(i)}_t \in \mathbb{R}^{n_i \times n_i}$ — per-scale diagonal (or DPLR) transition, discretized at timescale $\Delta t_i$
- $\mathcal{C}_t \in \mathbb{R}^{k \times k}$ — the **capacitance matrix**, a small learned coupling matrix
- $U \in \mathbb{R}^{n \times k}$, $V \in \mathbb{R}^{n \times k}$ — interface projection matrices (each column selects/projects onto the "boundary" of one scale block)

**Capacitance Matrix Parameterization:**

The capacitance matrix has special structure:

$$
\mathcal{C}_t = \text{Diag}(c^{(1)}_t, \ldots, c^{(k)}_t) + \mathcal{C}_{\text{off}}(x_t)
$$

where:
- Diagonal $c^{(i)}_t = \sigma(w_c^{(i)} \cdot x_t) \in (0, 1)$ — self-coupling (gating) per scale, input-dependent
- $\mathcal{C}_{\text{off}}(x_t) \in \mathbb{R}^{k \times k}$ — off-diagonal cross-scale coupling, with $|\mathcal{C}_{\text{off}, ij}| \leq 0$ (following the physical constraint that off-diagonal capacitance coefficients are non-positive)

In practice, parameterize $\mathcal{C}_{\text{off}, ij} = -\text{softplus}(w_{ij} \cdot x_t)$ for $i \neq j$, ensuring the capacitance matrix is diagonally dominant and well-conditioned.

**Woodbury-Efficient Recurrence:**

The full transition $\tilde{A}_t = A_{\text{diag},t} + U \mathcal{C}_t V^T$ is DPLR with block-diagonal base and rank-$k$ correction. The resolvent (needed for convolutional mode) is:

$$
(zI - \tilde{A}_t)^{-1} = D_z - D_z U (\mathcal{C}_t^{-1} + V^T D_z U)^{-1} V^T D_z
$$

where $D_z = (zI - A_{\text{diag},t})^{-1}$ is block-diagonal (each block inverts independently). The inner system is only $k \times k$, so the total cost is:

$$
O(n + k^3) \quad \text{per evaluation point}
$$

**Timescale Separation:**

Each block $A^{(i)}_t$ has its own discretization step $\Delta t_i$, parameterized geometrically:

$$
\Delta t_i = \Delta t_{\min} \cdot \rho^{i-1}, \quad \rho = \left(\frac{\Delta t_{\max}}{\Delta t_{\min}}\right)^{1/(k-1)}
$$

Block $i=1$ captures fast dynamics (small $\Delta t$), block $i=k$ captures slow dynamics (large $\Delta t$). Each block's eigenvalues are scaled accordingly:

$$
A^{(i)}_t = \exp(-\Delta t_i \cdot \Lambda^{(i)})
$$

where $\Lambda^{(i)}$ is learnable per block.

**Readout via Schur Complement:**

The output mixes information across scales:

$$
y_t = \sum_{i=1}^k \alpha_t^{(i)} C^{(i)} h_t^{(i)}
$$

where $\alpha_t^{(i)} = \text{softmax}(w_\alpha \cdot x_t)_i$ is an input-dependent scale-selection gate.

**Key Variables:**
- $h_t^{(i)} \in \mathbb{R}^{n_i}$ — state at scale $i$
- $A^{(i)}_t \in \mathbb{R}^{n_i \times n_i}$ — per-scale transition (diagonal or DPLR)
- $\mathcal{C}_t \in \mathbb{R}^{k \times k}$ — capacitance coupling matrix
- $U, V \in \mathbb{R}^{n \times k}$ — interface projections
- $\Delta t_i$ — timescale for scale $i$
- $k$ — number of scales (typically 3–6)
- $n = \sum_i n_i$ — total state dimension

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Multi-Scale Capacitance SSM (MC-SSM) |
| Layers | $L = 6$–$12$ |
| Hidden dim | $d = 256$–$512$ |
| Total state dim | $n = 64$ per head |
| Number of scales | $k = 4$ |
| Per-scale state dims | $n_1 = n_2 = n_3 = n_4 = 16$ |
| Timescale ratio | $\rho = 4$ ($\Delta t_i \in \{1, 4, 16, 64\}$) |
| Per-scale transition | Diagonal (like Mamba-2) |
| Capacitance coupling | Input-dependent $k \times k$ with diag-dominant constraint |
| Parallel training | Chunkwise parallel scan (SSD-compatible) |

### Baseline

1. **Mamba-2** (monolithic diagonal SSM): Same total state dim $n = 64$, single timescale — standard SSD scan. Complexity: $O(Tn)$
2. **Multi-head SSM** ($k=4$ heads, $n_i=16$ each, no coupling): Same block-diagonal structure but $\mathcal{C} = 0$ — ablation for whether coupling matters
3. **Dense-coupled multi-scale** ($k=4$ heads with dense $n \times n$ coupling): Upper bound on quality, $O(Tn^2)$ cost — tests whether sparse capacitance captures enough cross-scale interaction
4. **Log-linear attention** (from Proposal 008): Alternative multi-scale approach via Fenwick tree

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity | $\leq$ Mamba-2 baseline | WikiText-103 validation |
| MQAR recall | $> 80\%$ at 8 KV pairs, $T=4096$ | Multi-query associative recall |
| Needle-in-haystack | $> 85\%$ at $T=16384$ | Single-needle retrieval |
| Throughput | $\geq 0.9\times$ Mamba-2 | Tokens/sec on A100 |
| Cross-scale coupling | Non-trivial $\mathcal{C}_{\text{off}}$ | $\|\mathcal{C}_{\text{off}}\|_F / \|\mathcal{C}_{\text{diag}}\|_F > 0.1$ |

### Estimated Compute

**MVE**: < 10 minutes, single GPU
**Phase 1** (synthetic tasks): ~20 GPU-hours on A100
**Phase 2** (language modeling at 350M params): ~100 GPU-hours on A100
**Total**: ~120 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**
- MC-SSM outperforms monolithic Mamba-2 by $1$–$3\%$ perplexity at same total state dimension
- MQAR recall significantly better (multi-scale allows simultaneous short and long-range retrieval)
- The capacitance matrix learns interpretable structure: fast→slow coupling dominant (information flows from fine to coarse scales)
- Uncoupled multi-head baseline ($\mathcal{C} = 0$) significantly worse, validating that the coupling is essential
- Throughput within $10\%$ of Mamba-2 (the $k \times k$ coupling is negligible overhead)

**If hypothesis is wrong:**
- If uncoupled multi-head matches coupled: cross-scale interactions aren't important for these tasks, and simple independent heads suffice. This would validate the Mamba-2 multi-head design
- If monolithic outperforms coupled: the timescale separation is too rigid, and the model benefits from letting all eigenvalues interact freely. This would suggest that learned implicit timescales (via eigenvalue spread) are superior to explicit separation
- If dense coupling is needed: the capacitance sparsity ($k \times k$) is too restrictive; cross-scale interactions are fundamentally dense. This would point toward HSS-style hierarchical coupling (Proposal 005) rather than flat capacitance

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer MC-SSM, $d = 64$, $n = 32$ total state ($k = 4$ scales, $n_i = 8$), ~90K params
- **Task**: **Nested periodic detection** — synthetic task with signals at multiple frequencies: $x_t = \sin(2\pi f_1 t) + 0.5 \sin(2\pi f_2 t) + 0.25 \sin(2\pi f_3 t) + \text{noise}$, with $f_1 = 1/8, f_2 = 1/32, f_3 = 1/128$. Model must predict which frequencies are present in each window.
- **Data**: 10K synthetic sequences of length 512
- **Compute**: Single GPU, < 10 minutes

### Success Criteria
- MC-SSM with $k=4$ scales achieves $> 90\%$ accuracy on 3-frequency classification
- Monolithic baseline with same total $n=32$ achieves $< 75\%$ (struggles with the slowest frequency)
- Uncoupled ($\mathcal{C} = 0$) baseline achieves $70$–$80\%$ (detects individual scales but can't integrate)
- The learned $\Delta t_i$ values separate into distinct timescales (ratio between fastest and slowest $> 10\times$)

### Failure Criteria
- If monolithic SSM matches MC-SSM: the frequency separation is too easy and doesn't need explicit multi-scale structure
- If uncoupled matches coupled: cross-scale coupling provides no benefit on this task
- If training is unstable: the capacitance coupling introduces optimization difficulties

### Why This Test Is Sufficient
- The nested frequency task directly exercises the core multi-scale hypothesis: each scale block should lock onto one frequency, and the capacitance coupling should integrate them
- If scale separation helps at toy size, it should help more at scale where sequence contain richer multi-scale structure (phonemes/words/phrases in language)
- The capacitance matrix computation ($4 \times 4$) is identical at toy and full scale — the mechanism doesn't change with scale

## Theoretical Analysis

Complexity comparison:

| Operation | Monolithic ($n$) | Multi-scale ($k$ blocks, $n/k$ each) | MC-SSM (with coupling) |
|-----------|-----------------|--------------------------------------|------------------------|
| Per-step recurrence | $O(n)$ diagonal | $O(n)$ block-diagonal | $O(n + k^2)$ |
| Woodbury resolvent | $O(n + r^2)$ (rank-$r$ DPLR) | $O(n)$ (diagonal blocks) | $O(n + k^3)$ |
| Chunkwise scan (SSD) | $O(TQ + Tn/Q)$ | $O(TQ + Tn/Q)$ | $O(TQ + T(n+k^2)/Q)$ |
| Memory per step | $O(n)$ | $O(n)$ | $O(n + k^2)$ |

Crossover: For $k \ll n$ (typical: $k = 4, n = 64$), the coupling overhead $k^2 = 16 \ll n = 64$ is negligible. The main cost remains the block-diagonal scan.

**Information-theoretic argument**: A monolithic SSM with diagonal $A$ of dimension $n$ has $n$ independent decay/frequency channels. The MC-SSM has $n$ channels organized into $k$ groups with $k(k-1)/2$ cross-group coupling parameters. The coupling enriches the effective dynamics at a cost of $O(k^2)$ additional parameters — a favorable trade when cross-scale interactions matter.

## Risks & Limitations

1. **Timescale rigidity**: Fixing geometric $\Delta t_i$ may not match the true multi-scale structure of the data. Mitigation: make $\Delta t_i$ learnable (but risk that all scales collapse to same timescale).
2. **Scale collapse**: The model might learn to route all information through one scale, making the others redundant. Mitigation: add a load-balancing loss $\mathcal{L}_{\text{bal}} = \text{Var}(\alpha^{(i)})$ encouraging equal scale usage.
3. **Training dynamics**: The capacitance coupling introduces an additional low-rank term in the recurrence, which could interact poorly with gradient flow through long sequences. Mitigation: initialize $\mathcal{C}_{\text{off}} \approx 0$ (warm-start with independent scales).
4. **Overhead at small scale**: For $n = 16$ (small models), $k^2$ isn't negligible relative to $n$. The benefit appears mainly at larger state dimensions.
5. **SSD compatibility**: The rank-$k$ coupling breaks the pure diagonal structure needed for 1-semiseparable SSD. The recurrence $\tilde{A}_t = A_{\text{diag}} + UV^T$ has semiseparable rank $k$ instead of 1, requiring $k$-semiseparable SSD which costs $O(TQk)$ per block — still efficient but requires Mamba-2 kernel modifications.

## Follow-up Experiments

1. **Learnable timescale allocation**: Let $\Delta t_i$ be learnable with a repulsion loss $\sum_{i < j} 1/|\log \Delta t_i - \log \Delta t_j|$ to prevent scale collapse. Compare against geometric spacing.
2. **Hierarchical capacitance (HSS coupling)**: Instead of flat $k \times k$ coupling, use an HSS-structured coupling matrix with tree depth $\log k$ — connects to Proposal 005 (segmented HSS linear attention).
3. **Expert-choice scale routing**: Apply expert-choice routing (Proposal 012) to select which scale processes each token — tokens with long-range dependencies get routed to slow scales.
4. **Oscillatory multi-scale**: Use LinOSS oscillators (Proposal 007) as the per-scale dynamics, with capacitance coupling between oscillator banks at different frequencies — directly mirrors how physical multi-scale systems (acoustics, optics) are analyzed.
5. **Multi-scale DeltaProduct**: Use Householder products (DeltaProduct) within each scale for state-tracking, with capacitance coupling for cross-scale information flow — combines expressivity from Proposal 001 with multi-scale structure.

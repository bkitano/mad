---
status: ongoing
priority: medium
created: 2026-02-10
based_on: column-sparse-transition-matrices, diagonal-plus-low-rank-(dplr), woodbury-resolvent-identity, cauchy-kernel-trick, perturb-then-diagonalize, input-dependent-gating
experiment_number: 003
experiment_log: experiment-log-003.md
---

# DPLR Column-Sparse SSM: Expressivity Meets Efficient Convolution

## Hypothesis

A state transition matrix parameterized as $A = P(\Lambda + \mathbf{p}\mathbf{q}^*)P^\top$ — a column-sparse permutation applied to a DPLR core — retains the Cauchy kernel trick for efficient convolutional (LTI) training while gaining the permutation-routing expressivity of column-sparse matrices, creating an SSM that bridges the efficiency of S4-style convolution with the state-tracking power of PD-SSM.

## Background

The current SSM landscape has a sharp divide between two families:

**Convolutional SSMs (S4, S4D, S5)**: Use fixed (LTI) state matrices with DPLR structure. The Woodbury resolvent identity and Cauchy kernel trick enable $O(NL)$ training via convolution in the frequency domain. Excellent at long-range dependencies but provably limited in state-tracking expressivity — diagonal $A$ cannot represent permutations.

**Selective SSMs (Mamba, PD-SSM)**: Use input-dependent (LTV) state matrices. PD-SSM's column-sparse structure enables permutation routing, solving state-tracking tasks diagonal SSMs cannot. But input-dependent $A$ breaks the convolutional mode entirely — no frequency-domain tricks apply, requiring sequential scan or chunkwise parallel scan for training.

**The gap**: No architecture combines permutation-routing expressivity with frequency-domain convolutional efficiency. This experiment asks: can we design a **structured $A$ matrix** that (a) has column-sparse permutation structure for expressivity, (b) has DPLR-like decomposition for Cauchy kernel acceleration, and (c) optionally transitions to input-dependent gating via a hybrid architecture?

**Key insight**: A column-sparse matrix where each non-zero column entry comes from a DPLR structure can be viewed as $\Lambda_{\text{perm}} + \text{low-rank}$, where $\Lambda_{\text{perm}}$ is a permuted diagonal. The resolvent of a permuted-diagonal-plus-low-rank matrix can still be decomposed via Woodbury — the permutation simply reindexes the Cauchy kernel evaluation points.

## Mathematical Formulation

**DPLR-CS-SSM parameterizes the state transition as:**

$$
A = P \cdot (\Lambda + \mathbf{p}\mathbf{q}^*) \cdot P^\top
$$

**Key Definitions:**

- $P \in \{0,1\}^{N \times N}$ — fixed (LTI) permutation matrix, chosen at initialization
- $\Lambda \in \mathbb{C}^{N \times N}$ — diagonal with eigenvalues on/inside the unit circle
- $\mathbf{p}, \mathbf{q} \in \mathbb{C}^N$ — rank-1 correction vectors (as in S4's HiPPO decomposition)

**Why This Works for Cauchy Kernels:**

The resolvent becomes:

$$
(zI - A)^{-1} = P \cdot (zI - \Lambda - \mathbf{p}\mathbf{q}^*)^{-1} \cdot P^\top
$$

Since $P$ is orthogonal ($P^\top = P^{-1}$), the resolvent factors through:

1. Permute $\tilde{B} = P^\top B$ (reindex input projection)
2. Evaluate standard DPLR resolvent via Woodbury + Cauchy: $(zI - \Lambda - \mathbf{p}\mathbf{q}^*)^{-1}$
3. Permute output $\tilde{C} = C P$ (reindex output projection)

**This is algebraically equivalent to a DPLR SSM with permuted $B$, $C$ projections**, but the permutation $P$ creates coupling between state dimensions that diagonal SSMs cannot represent.

**Hybrid Mode (Input-Dependent):**

For tasks requiring input-dependence, extend to:

$$
A(x_t) = P(x_t) \cdot (\Lambda + \mathbf{p}\mathbf{q}^*) \cdot P(x_t)^\top
$$

where $P(x_t)$ is input-dependent (via Gumbel-softmax). This loses the convolutional mode but retains the structural benefits for chunkwise parallel scan (the DPLR core ensures well-conditioned products).

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| State matrix | $A = P(\Lambda + \mathbf{p}\mathbf{q}^*)P^\top$ |
| Permutation $P$ | Fixed (Phase 1) or input-dependent (Phase 2) |
| State dim | $N = 64$–$256$ |
| Rank | $r = 1$ or $2$ |

### Experiment Design

**Phase 1: Fixed permutation (LTI mode)**

Test whether a single well-chosen permutation $P$ improves over standard DPLR:

- $P = $ **cyclic shift**: Simplest non-trivial permutation, rotates state dimensions
- $P = $ **bit-reversal**: Maximizes mixing between state dimensions
- $P = $ **learned**: Initialize randomly, optimize $P$ (continuous relaxation via doubly-stochastic → project to permutation after training)
- $P = I$: Baseline (standard DPLR/S4)

Evaluate using the Cauchy kernel trick for training (convolutional mode).

**Phase 2: Input-dependent permutation (LTV mode)**

Replace fixed $P$ with $P(x_t)$, using the chunkwise parallel scan. Compare against:
- PD-SSM (column-sparse without DPLR core)
- Mamba-2 (scalar diagonal, input-dependent gating)

**Phase 3: PTD initialization**

Apply Perturb-Then-Diagonalize to the DPLR core of DPLR-CS-SSM:
- Start from HiPPO matrix decomposition
- Apply PTD perturbation to get well-conditioned eigenvectors
- Apply fixed permutation $P$ to the PTD-initialized matrix
- Compare against standard S5-PTD (no permutation)

### Baseline

1. **S4 / S4D** — standard DPLR without permutation ($P = I$)
2. **S5-PTD** — diagonal with PTD initialization
3. **PD-SSM** — column-sparse without DPLR structure
4. **Mamba-2** — scalar diagonal with input-dependent gating

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| LRA accuracy | $> 86\%$ avg | All 5 tasks |
| $S_5$ composition | Near-perfect | State tracking expressivity |
| Training speed | = S4 (Phase 1) | Wall-clock comparison |
| Condition number | $\kappa < 10^3$ | Eigenvector matrix |

### Estimated Compute

**Small–Medium**.

- Phase 1 (LTI, fixed $P$): Standard S4-scale experiments. 4 permutation choices × 5 LRA tasks × 3 seeds = 60 runs, ~2 GPU-hours each = **120 GPU-hours**
- Phase 2 (LTV, input-dependent $P$): Requires chunkwise parallel scan. 3 state-tracking tasks × 4 models × 3 seeds = 36 runs, ~1 GPU-hour each = **36 GPU-hours**
- Phase 3 (PTD): Same as Phase 1 with PTD init. Additional 60 runs = **120 GPU-hours**
- **Total**: ~280 A100 GPU-hours

## Expected Outcome

**Phase 1 (fixed $P$):**

- Cyclic and bit-reversal permutations provide modest improvement (0.5–2%) on LRA tasks like PathFinder and Retrieval where inter-dimension coupling helps
- Learned $P$ provides the largest gain (~1–3%) by discovering task-optimal state routing
- No improvement on state-tracking tasks (fixed $P$ cannot represent input-dependent permutations)
- Training speed is identical to S4 (permutation is absorbed into $B$, $C$ projections)

**Phase 2 (input-dependent $P$):**

- DPLR-CS-SSM outperforms PD-SSM on LRA tasks because the DPLR core provides better long-range dynamics
- DPLR-CS-SSM matches or exceeds PD-SSM on state-tracking tasks (same permutation expressivity)
- The DPLR core provides better conditioning for chunkwise parallel scan products, reducing numerical errors at long sequences

**Phase 3 (PTD + permutation):**

- S4-PTD-CS (PTD init + fixed permutation) achieves the best LRA results, combining PTD's robustness with permutation coupling
- The combination is particularly strong on tasks with both long-range and structural dependencies

**Key negative result to watch for:** If fixed permutation provides zero benefit in Phase 1, the permutation routing is only useful in the input-dependent (LTV) setting, and the LTI combination isn't valuable — still a useful finding.

## Minimum Viable Experiment

**Goal**: Test whether a fixed permutation $P$ applied to DPLR provides any measurable benefit on a task requiring inter-dimension coupling, before committing to the full experiment suite.

### Setup

| Component | Configuration |
|-----------|---------------|
| Model | 1-layer SSM with DPLR state matrix |
| State dim | $N = 16$ |
| Hidden dim | $d_{\text{model}} = 32$ |
| Parameters | ~5K |
| Task | Parity computation (requires dimension coupling) |
| Data | 10K sequences of length 32 |
| Compute | Single GPU, $< 5$ minutes |

**Why Parity?** Computing parity of a bit sequence requires information from all positions to propagate to a single output dimension. A diagonal SSM can only track each dimension independently — it cannot aggregate across dimensions without $B$ and $C$ doing all the work. A permutation-coupled SSM should learn this more easily.

### Task Definition

Input: Binary sequence $x_1, \ldots, x_T \in \{0, 1\}$

Output: $\bigoplus_{t=1}^T x_t$ (XOR of all bits)

```python
def generate_parity_data(T=32, n_samples=10000):
    X = torch.randint(0, 2, (n_samples, T)).float()
    Y = X.sum(dim=1) % 2  # Parity
    return X, Y
```

### Models to Compare

1. **DPLR ($P = I$)**: Standard diagonal + rank-1, no permutation
2. **DPLR-CS (cyclic $P$)**: Cyclic shift permutation
3. **DPLR-CS (bit-reversal $P$)**: Bit-reversal permutation (maximizes mixing)
4. **DPLR-CS (learned $P$)**: Doubly-stochastic relaxation, project to permutation

### Success Criteria

| Model | Expected Accuracy |
|-------|-------------------|
| DPLR ($P = I$) | $\sim 50$–$70\%$ (struggles without coupling) |
| DPLR-CS (any $P$) | $> 90\%$ |

**The idea works if**: Any non-identity permutation achieves $> 90\%$ accuracy while $P = I$ achieves $< 75\%$, demonstrating that the permutation provides inter-dimension coupling that the baseline lacks.

### Failure Criteria

- **Kill the idea if**: All permutations perform the same as $P = I$ (permutation is absorbed into $B/C$ reparameterization)
- **Kill the idea if**: Learned $P$ doesn't outperform random fixed $P$ (learning the permutation adds no value)
- **Pause and investigate if**: All models achieve $> 90\%$ (parity is too easy for this state size; need harder task)

### Why This Test Is Sufficient

1. **Parity isolates coupling**: The task specifically tests whether dimensions can communicate through the state matrix, not just through input/output projections
2. **Small scale reveals mechanism**: If permutation coupling helps at $N = 16$, it will help at $N = 256$ — this is a structural property
3. **5 minutes per model**: Can test all 4 variants in $< 20$ minutes total
4. **Binary outcome**: Either permutation helps or it doesn't — no ambiguity about whether to proceed

### Implementation Sketch

```python
class DPLR_CS_SSM(nn.Module):
    def __init__(self, N=16, rank=1, P_type='identity'):
        super().__init__()
        # DPLR core: Lambda (diagonal) + p q^* (rank-1)
        self.Lambda = nn.Parameter(torch.randn(N) * 0.1 - 0.5)
        self.p = nn.Parameter(torch.randn(N) * 0.01)
        self.q = nn.Parameter(torch.randn(N) * 0.01)

        # Permutation (fixed)
        if P_type == 'identity':
            P = torch.eye(N)
        elif P_type == 'cyclic':
            P = torch.roll(torch.eye(N), 1, dims=0)
        elif P_type == 'bit_reversal':
            P = bit_reversal_permutation(N)
        self.register_buffer('P', P)

    def get_A(self):
        # A = P (Lambda + p q^T) P^T
        Lambda_diag = torch.diag(self.Lambda)
        low_rank = torch.outer(self.p, self.q)
        core = Lambda_diag + low_rank
        return self.P @ core @ self.P.T
```

## Theoretical Analysis

**Complexity Comparison:**

| Mode | Standard DPLR | DPLR-CS (fixed $P$) | DPLR-CS (input $P$) |
|------|---------------|---------------------|---------------------|
| Training | $O(NL + L \log L)$ conv | $O(NL + L \log L)$ conv | $O(TN)$ scan |
| Per-step | $O(N)$ | $O(N)$ | $O(N)$ |
| Expressivity | Diagonal only | Permuted diagonal | Full column-sparse |

**The permutation adds no computational cost in LTI mode** — it's absorbed into the $B$, $C$ projections.

## Risks & Limitations

1. **Fixed permutation may be trivially absorbed**: Since $P \cdot (\Lambda + \mathbf{p}\mathbf{q}^*) \cdot P^\top$ with fixed $P$ is equivalent to $\Lambda_{\text{perm}} + (P\mathbf{p})(P\mathbf{q})^*$, the model might learn equivalent dynamics by just reparameterizing $B$ and $C$. The permutation only adds value if it creates coupling that the optimizer wouldn't find otherwise. Need to carefully verify this isn't just a reparameterization.

2. **DPLR rank-1 correction may be insufficient**: The DPLR structure uses rank-1 (or rank-2) correction. Column-sparse PD-SSM's diagonal component is full-rank within its sparse support. The low-rank DPLR core may be too restrictive for state-tracking tasks. Mitigation: test rank-2 and rank-4 corrections.

3. **PTD is specific to HiPPO**: The PTD initialization only helps when starting from HiPPO matrices. If the optimal initialization for DPLR-CS-SSM is different (e.g., random eigenvalues), PTD doesn't apply.

4. **Input-dependent Gumbel-softmax permutation**: Same differentiability challenges as PD-SSM. The DPLR core doesn't inherently solve this.

5. **Limited novelty in LTI mode**: If the permutation is fully absorbed into $B/C$, Phase 1 reduces to a reparameterization study with limited architectural insight. Phase 2 (LTV) is where the real novelty lies.

## Follow-up Experiments

1. **Higher-rank DPLR core**: Test rank-2, 4, 8 corrections within the permuted DPLR structure — at what rank does the Cauchy kernel speedup become negligible?
2. **Block-permutation structure**: Instead of full permutation, use block-diagonal permutations where each block is a small ($4 \times 4$) permutation — intermediate between diagonal and full permutation in expressivity and complexity
3. **Hybrid convolutional + selective layers**: Alternate DPLR-CS-SSM layers (convolutional, efficient) with Mamba-2 layers (selective, expressive) in a single model — can each layer type specialize?
4. **Frequency-domain analysis**: Analyze the learned convolution kernels' frequency response to understand what the permutation changes about the model's spectral properties
5. **Connection to group SSMs**: Investigate whether DPLR-CS-SSM's permutation structure relates to group-equivariant SSMs or symmetry-preserving architectures

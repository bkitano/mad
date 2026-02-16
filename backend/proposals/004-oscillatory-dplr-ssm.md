---
status: completed
priority: high
created: 2026-02-15
based_on: 
experiment_number: 004
results_file: 004_results.md
---

# Oscillatory-DPLR SSM: Constraint-Free Stable State Spaces

- oscillatory-eigenvalue-stability
- diagonal-plus-low-rank-(dplr)
- woodbury-resolvent-identity
- cauchy-kernel-trick
- perturb-then-diagonalize

## Hypothesis

Combining oscillatory discretization (which guarantees |λ|≤1 by construction from second-order ODEs) with DPLR structure (which enables O(n) Cauchy kernel convolution) will produce an SSM that is:
1. **More stable** than S4/S5 without requiring eigenvalue constraints or careful initialization
2. **Equally efficient** as S4D in convolution mode (O(T log T) via FFT)
3. **More trainable** because oscillatory parameters (ω, ζ) have direct physical interpretation

**Key insight**: S4/S5 impose stability via projection/constraints; oscillatory SSMs achieve it *by construction* from physics. DPLR provides efficiency. Their combination should yield robust initialization with S4-level speed.

## Background

**Current landscape:**
- **S4**: DPLR + HiPPO initialization, but requires careful eigenvalue initialization and PTD robustness fixes
- **S5**: Simplified to pure diagonal, but loses low-rank expressivity and still needs eigenvalue constraints
- **Oscillatory SSMs**: Stable by construction from harmonic oscillator discretization, but original formulation uses dense matrices (O(n²) memory)

**Gap**: No architecture combines oscillatory stability guarantees with structured (DPLR) efficiency.

**Why this matters**:
- Initialization robustness is critical for scaling SSMs to billions of parameters
- Oscillatory parameters (frequency ω, damping ζ) are interpretable; diagonal eigenvalues in S4 are not
- Low-rank component in DPLR captures interactions that pure diagonal (S5) misses

## Mathematical Formulation

### Oscillatory Second-Order ODE

Start from damped harmonic oscillator:

$$
\ddot{y}(t) + 2\zeta\omega\dot{y}(t) + \omega^2 y(t) = u(t)
$$

**Key variables:**
- $\omega > 0$ — natural frequency (oscillation rate)
- $\zeta \geq 0$ — damping ratio (stability control)
- $u(t)$ — forcing input

Convert to first-order system $\mathbf{x} = [y, \dot{y}]^T$:

$$
\frac{d}{dt}\begin{bmatrix} y \\ \dot{y} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\omega^2 & -2\zeta\omega \end{bmatrix} \begin{bmatrix} y \\ \dot{y} \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} u(t)
$$

**Eigenvalues**: $\lambda = -\zeta\omega \pm i\omega\sqrt{1 - \zeta^2}$ (when $\zeta < 1$)

**Stability guarantee**: $\text{Re}(\lambda) = -\zeta\omega \leq 0$ always, so $|\lambda| \leq \omega$ for all $\zeta \geq 0$.

### Bilinear Discretization (Tustin's Method)

Discretize with step size $\Delta$:

$$
A_d = (I - \frac{\Delta}{2}A_c)^{-1}(I + \frac{\Delta}{2}A_c)
$$

**Property**: If $\text{Re}(\lambda_c) \leq 0$, then $|\lambda_d| \leq 1$ (maps left half-plane to unit disk).

**Result**: Discrete eigenvalues are **guaranteed** on unit circle when $\zeta = 0$ (undamped), inside when $\zeta > 0$ (damped).

### DPLR Structure

Parameterize state matrix as:

$$
A = \Lambda + PQ^T
$$

where:
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ — diagonal component (oscillatory eigenvalues)
- $P, Q \in \mathbb{C}^{n \times r}$ — low-rank factors with $r \ll n$

### Proposed Oscillatory-DPLR Parameterization

**Step 1**: Generate $n$ oscillatory eigenvalues from learnable parameters:

$$
\lambda_i = -\zeta_i\omega_i + i\omega_i\sqrt{1 - \zeta_i^2}
$$

with $\omega_i > 0$, $\zeta_i \in [0, 1]$ (parameterized via softplus/sigmoid to enforce constraints).

**Step 2**: Construct diagonal oscillatory matrix:

$$
\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)
$$

**Step 3**: Add low-rank correction:

$$
A = \Lambda + PQ^T
$$

**Step 4**: Discretize via bilinear transform:

$$
A_d = (I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)
$$

**Step 5**: Apply Cauchy kernel trick for convolution (S4 algorithm):

$$
\bar{K} = (C^T \odot Q^T)(I - A_d^L)^{-1}(B \odot P) + C^T(\bar{A}_d^L - I)(I - \bar{A}_d)^{-1}B
$$

where $\bar{A}_d = \text{diag}((A_d)_{11}, \ldots, (A_d)_{nn})$ and $\odot$ is element-wise product.

**Complexity**: O(nL + n^2) where typically $r \ll n$, so dominated by O(nL) Cauchy kernel evaluation.

### Standard S4 Approach (for comparison)

$$
A = \text{diag}(\lambda_1, \ldots, \lambda_n) + PQ^T
$$

where $\lambda_i$ initialized from HiPPO, then:
- Apply PTD (perturb-then-diagonalize) to avoid ill-conditioning
- Project eigenvalues to ensure $|\lambda_i| < 1$ during training
- Hope initialization transfers well to downstream tasks

**Problem**: HiPPO eigenvalues are empirically chosen; no guarantee they're optimal for all tasks.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Oscillatory-DPLR SSM |
| Layers | $L = 6$ |
| State dimension | $n = 256$ |
| Low-rank | $r = 16$ |
| Hidden dimension | $d = 512$ |
| Sequence length | $T = 1024$ (training), up to 16384 (extrapolation test) |

### Baseline

**Primary**: S4D (pure diagonal DPLR with HiPPO+PTD initialization)
- Complexity: O(T log T) via FFT convolution

**Secondary**: S5 (simplified diagonal SSM)
- Complexity: O(T log T) via FFT convolution

**Tertiary**: Original Oscillatory SSM (dense matrix, if implementation available)
- Complexity: O(Tn²) (dense matrix multiplication)

### Training Details

- **Initialization**:
  - $\omega_i$: Log-uniform in [0.001, 0.1] (covers slow to fast oscillations)
  - $\zeta_i$: Uniform in [0, 1] (from undamped to critically damped)
  - $P, Q$: Glorot initialization
  - $B, C$: Glorot initialization

- **Discretization step**: $\Delta \in \{0.001, 0.01, 0.1\}$ (ablation)

- **Optimizer**: AdamW, lr=5e-4, warmup=5k steps, cosine decay

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Accuracy | $\geq$ S4D | Test perplexity/accuracy on LRA |
| Throughput | $\geq 0.9 \times$ S4D | Tokens/sec on A100 |
| Memory | $\leq$ S4D | Peak GPU memory during training |
| Initialization robustness | $> 0.8$ test acc in 10% of S4D training steps | Early-training learning curve |
| Extrapolation | $\leq 1.2 \times$ perplexity at $4\times$ training length | Test on 4096 when trained on 1024 |

### Estimated Compute

**Medium**: ~150 GPU-hours on A100
- 6 LRA tasks × 3 seeds × 2 models (Osc-DPLR, S4D) = 36 runs
- ~4 GPU-hours per run (shorter sequences, 6-layer models)

## Expected Outcome

**If hypothesis is correct:**

1. **Initialization robustness**: Osc-DPLR reaches $>80\%$ of final test accuracy in $<10\%$ of training steps, while S4D takes $>20\%$
   - **Why**: Oscillatory parameters start in physically meaningful regime; S4D relies on HiPPO which may not transfer

2. **Extrapolation**: At $4\times$ training length, Osc-DPLR maintains $\leq 1.2\times$ perplexity, S4D degrades $\geq 1.5\times$
   - **Why**: Oscillatory frequencies are continuous; learned $\omega_i$ naturally extend to longer sequences

3. **Long-range tasks**: Osc-DPLR matches or exceeds S4D on LRA Path-X and ListOps ($\geq +2\%$ accuracy)
   - **Why**: Low-rank component captures interactions that pure diagonal (S5) misses

**If hypothesis is wrong:**

- **Scenario A**: Low-rank component destabilizes oscillatory eigenvalues
  - **Learn**: Oscillatory guarantee only holds for diagonal; DPLR perturbation breaks it
  - **Fix**: Constrain $\|PQ^T\|_2 \ll \min_i |\lambda_i|$ to ensure perturbation is small

- **Scenario B**: Oscillatory initialization converges to same eigenvalue distribution as S4D
  - **Learn**: Architecture matters more than initialization; oscillatory is just a reparameterization
  - **Insight**: Still valuable for interpretability ($\omega, \zeta$ have physical meaning)

- **Scenario C**: Osc-DPLR is slower than S4D due to bilinear discretization overhead
  - **Learn**: Bilinear transform $(I - \frac{\Delta}{2}A)^{-1}$ is expensive compared to ZOH exponential
  - **Fix**: Precompute discretization for fixed $\Delta$, or use caching

## Minimum Viable Experiment

**CRITICAL**: Before running full LRA suite, validate core mechanism on synthetic task.

### Setup
- **Model**: Tiny Osc-DPLR (1 layer, $n=16$, $r=2$, $d=32$, ~5K params)
- **Task**: **Damped Oscillation Extrapolation**
  - Input: Random impulse $u_t = \delta_{t=0}$
  - Target: Generate damped sinusoid $y_t = A e^{-\zeta\omega t}\sin(\omega\sqrt{1-\zeta^2} t + \phi)$ with random $\omega \sim U(0.01, 0.1)$, $\zeta \sim U(0.2, 0.8)$
  - Train on $T=128$, test on $T=512$ (4× extrapolation)
- **Data**: 10K synthetic sequences
- **Compute**: Single A100, $< 5$ minutes

### Success Criteria
- **Training fit**: MSE $< 10^{-3}$ on training sequences (model can fit oscillatory patterns)
- **Extrapolation**: MSE $< 10^{-2}$ on $4\times$ longer test sequences (learned $\omega, \zeta$ generalize)
- **Interpretability**: Learned $\omega_i$ cluster near ground-truth frequency range [0.01, 0.1]

### Failure Criteria
- MSE $> 10^{-1}$ on training (cannot fit basic oscillations)
- Extrapolation MSE $> 10\times$ training MSE (complete failure to generalize)
- Learned $\omega_i$ collapse to single value or diverge outside [0.001, 1] (parameterization broken)

### Why This Test Is Sufficient
- **Oscillatory mechanism validation**: If tiny model can learn and extrapolate damped sinusoids, the oscillatory eigenvalue parameterization works
- **DPLR structure validation**: Low-rank component $r=2$ should capture phase relationships between sine/cosine components
- **Physical interpretability**: If learned $\omega_i$ match ground truth, the parameterization is meaningful
- **Failure fast**: 5 minutes reveals fundamental issues before investing 150 GPU-hours

**Decision rule**:
- ✅ All success criteria met → Proceed to full LRA experiments
- ❌ Any failure criterion → Debug parameterization (check $\omega, \zeta$ gradients, discretization numerics) before scaling

## Theoretical Analysis

### Complexity Comparison

| Operation | S4D | Oscillatory-DPLR | Notes |
|-----------|-----|------------------|-------|
| Forward pass (train) | $O(T \log T)$ | $O(T \log T)$ | Both use FFT convolution |
| Forward pass (inference) | $O(Td)$ | $O(Td)$ | Recurrent mode |
| Backward pass | $O(T \log T)$ | $O(T \log T)$ | FFT + autodiff |
| Memory | $O(T + n)$ | $O(T + n + 2nr)$ | Extra $P, Q$ storage |
| Initialization | $O(n^3)$ (PTD) | $O(n)$ (set $\omega, \zeta$) | **Osc-DPLR advantage** |

**Crossover point**: $2nr \ll n \Rightarrow r \ll n/2$. For $n=256$, $r=16$: $2 \cdot 256 \cdot 16 = 8192$ vs $256$ (negligible overhead).

### Stability Analysis

**S4D**: Eigenvalues $\lambda_i$ initialized from HiPPO, then projected to unit circle during training.
- **Guarantee**: $|\lambda_i| \leq 1 - \epsilon$ via projection
- **Issue**: Projection interferes with gradient flow; need careful learning rate tuning

**Oscillatory-DPLR**: Eigenvalues $\lambda_i = -\zeta_i\omega_i + i\omega_i\sqrt{1-\zeta_i^2}$ with $\zeta_i \in [0,1]$.
- **Guarantee**: $|\lambda_i| = \omega_i \leq 1$ (by parameterization)
- **Advantage**: No projection needed; gradients flow naturally through $\omega_i, \zeta_i$

**DPLR perturbation**: $A = \Lambda + PQ^T$ has eigenvalues $\tilde{\lambda}_i = \lambda_i + O(\|PQ^T\|)$.
- **Stability maintained if**: $\|PQ^T\|_2 \ll \min_i |\text{Re}(\lambda_i)| = \min_i \zeta_i\omega_i$
- **Practical**: Initialize $P, Q$ small (e.g., Glorot with scale 0.1); monitor $\|PQ^T\|_2$ during training

### Expressivity Analysis

**Pure diagonal (S5)**: Can only represent independent exponential decays.
- **Cannot**: Model interactions between state dimensions
- **Example failure**: Parity (requires XOR-like interactions)

**Oscillatory diagonal**: Can represent complex eigenvalues (oscillations + decay).
- **Can**: Model periodic patterns, multi-timescale dynamics
- **Cannot**: Model interactions (same as S5)

**Oscillatory-DPLR**: Diagonal oscillations + low-rank interactions.
- **Can**: Model periodic patterns + cross-state interactions via $PQ^T$
- **Hypothesis**: Sufficient for LRA tasks (Path-X requires long-range dependencies + interactions)

## Risks & Limitations

### Risk 1: Bilinear Discretization Cost
- **Issue**: $(I - \frac{\Delta}{2}A)^{-1}$ requires matrix inversion at initialization
- **Mitigation**: Precompute and cache discretized $A_d$ for fixed $\Delta$; use closed-form for DPLR structure
- **Fallback**: Use zero-order hold (ZOH) discretization instead (faster but loses bilinear stability guarantee)

### Risk 2: Low-Rank Destabilization
- **Issue**: $PQ^T$ perturbation may push eigenvalues outside unit circle
- **Mitigation**:
  - Initialize $P, Q$ with small magnitude (scale 0.01-0.1)
  - Add regularization: $\mathcal{L}_{\text{reg}} = \lambda \|PQ^T\|_F^2$ to keep low-rank small
  - Monitor $\|\tilde{A}\|_2$ during training; clip if $> 1$
- **Fallback**: Use $A = \Lambda + \epsilon PQ^T$ with learnable $\epsilon \in [0, 0.1]$ to control perturbation strength

### Risk 3: Oscillatory Parameters May Not Be Optimal
- **Issue**: Physical oscillators may not be the right inductive bias for language/vision tasks
- **Mitigation**: Compare to S4D on same tasks; if worse, analyze learned $\omega_i$ distribution (did they converge to non-oscillatory regime?)
- **Insight**: Even if performance is equal, interpretability ($\omega = $ frequency) is valuable for analysis

### Risk 4: Implementation Complexity
- **Issue**: Bilinear discretization + Cauchy kernel + complex eigenvalues = many moving parts
- **Mitigation**:
  - Use existing S4 codebase (minimal changes: swap HiPPO initialization for oscillatory)
  - Extensive unit tests for discretization correctness
  - Sanity check: Does $|\lambda_d| \leq 1$ hold after discretization?

## Follow-up Experiments

### If Successful (meets target metrics):

1. **Scale to Mamba-size models** (1.3B params)
   - Test if initialization robustness holds at scale
   - Compare to S6 (Mamba's parallel scan version)

2. **Input-dependent oscillatory SSM**
   - Make $\omega_i(x_t)$, $\zeta_i(x_t)$ input-dependent (like Mamba's selective scan)
   - Hypothesis: Dynamic frequency/damping enables richer temporal patterns

3. **Ablation: Oscillatory vs HiPPO initialization**
   - Train S4 (DPLR) with both initializations
   - Isolate effect of oscillatory parameterization vs DPLR structure

4. **Multi-scale oscillatory banks**
   - Use $k$ groups of oscillators at different frequency ranges (slow/medium/fast)
   - Like multi-head attention but for temporal scales

### If Unsuccessful (fails target metrics):

1. **Ablate low-rank component**
   - Test pure oscillatory diagonal (Osc-S5) vs S5
   - Isolate whether oscillatory parameterization alone helps

2. **Analyze learned eigenvalue distribution**
   - Visualize $\lambda_i$ in complex plane after training
   - Did oscillatory structure collapse to S4D-like distribution?

3. **Try alternative discretization**
   - Zero-order hold (ZOH): $A_d = \exp(\Delta A)$
   - Euler: $A_d = I + \Delta A$ (fast but unstable)
   - Compare stability vs speed trade-off

4. **Hybrid initialization**
   - Start from oscillatory, gradually relax to unconstrained eigenvalues
   - Test if warm-start improves S4 training

## References to Tricks

- **oscillatory-eigenvalue-stability**: Core parameterization ensuring |λ|≤1 by construction
- **diagonal-plus-low-rank-(dplr)**: Structure enabling O(n) Cauchy kernel convolution
- **woodbury-resolvent-identity**: Algebraic trick for fast $(I - A)^{-1}$ with DPLR
- **cauchy-kernel-trick**: Reduces convolution to Cauchy dot products (S4 algorithm)
- **perturb-then-diagonalize**: Compared as baseline initialization strategy

## Connection to Existing Proposals

- **Compared to 001 (CS-NEG-DeltaNet)**: Focuses on stability/initialization rather than expressivity; orthogonal direction
- **Compared to 002 (SSD-DeltaNet)**: Targets convolutional SSMs (S4 family) not recurrent SSMs (DeltaNet family)
- **Compared to 003 (DPLR-CS-SSM)**: Uses oscillatory diagonal instead of permutation matrices; both explore DPLR extensions

**Unique contribution**: Only proposal addressing initialization robustness (Gap 3) with principled physics-based approach.

---
status: completed
priority: high
created: 2026-02-15
based_on: neumann-series-approximate-inverse, diagonal-plus-low-rank-(dplr), woodbury-resolvent-identity, chunkwise-parallel-scan, cauchy-kernel-trick, io-aware-tiling, kernel-fusion
experiment_number: 011
results_file: 011_results.md
---

# Neumann-Approximate Resolvent for Chunkwise DPLR SSM Training

## Hypothesis

Replacing the exact Woodbury resolvent $(zI - A)^{-1}$ in DPLR SSM kernel computation with a truncated Neumann series of order $k = 4$–$8$ (accelerated via radix-$m$ kernels) will yield a training method that is (a) numerically more stable near resonant frequencies where $z \approx \lambda_i$, (b) $1.5$–$2\times$ faster in wall-clock time for chunkwise parallel training due to eliminating elementwise divisions in favor of fused GEMMs, and (c) within $< 1\%$ perplexity of the exact method.

## Background

### The Resolvent Bottleneck in DPLR SSMs

S4 and its descendants parameterize the state transition matrix as $A = \Lambda + PQ^*$ (diagonal plus low-rank). The convolutional training mode requires evaluating the SSM kernel at $L$ frequencies via the generating function:

$$
\hat{K}(\omega_j) = C(e^{i\omega_j} I - \bar{A})^{-1} B
$$

The Woodbury identity decomposes this as:

$$
(zI - \Lambda + PQ^*)^{-1} = D_z - D_z P (I + Q^* D_z P)^{-1} Q^* D_z
$$

where $D_z = \text{diag}(1/(z - \lambda_i))$. This is efficient ($O(N)$ per frequency for rank-$r$ correction with fixed $r$) but has a critical weakness: **when $z \approx \lambda_i$, the diagonal resolvent $D_z$ has entries approaching infinity**, causing numerical instability. This instability forces practitioners to use FP32 precision for SSM kernel computation even when the rest of the model runs in FP16/BF16, creating a mixed-precision bottleneck.

### The Neumann Alternative

For the resolvent of a DPLR matrix, we can write:

$$
(zI - A)^{-1} = (zI - \Lambda)^{-1} (I - (zI - \Lambda)^{-1} PQ^*)^{-1}
$$

Setting $E = (zI - \Lambda)^{-1} PQ^*$ (a rank-$r$ matrix), the second factor becomes:

$$
(I - E)^{-1} \approx S_k(E) = I + E + E^2 + \cdots + E^{k-1}
$$

**Key insight**: The Neumann series avoids computing $D_z$ entries individually. Instead, the critical quantity is $\|E\|$, which depends on the *product* $\|(zI - \Lambda)^{-1} PQ^*\|$. When $P, Q$ have small norm (the low-rank perturbation is small), this product can be well-conditioned even when individual $D_z$ entries are large, because the problematic eigenvalues may not align with the low-rank directions $P, Q$.

Moreover, the Neumann series computation is entirely GEMM-based: each term $E^j$ is a matrix-matrix product, which maps perfectly to tensor cores and can be fused into a single kernel. The radix-$m$ kernel acceleration (Sao 2025) reduces the number of required GEMMs from $k-1$ to $\sim 1.54 \log_2 k$.

### Why This Hasn't Been Tried

The existing Cauchy kernel trick + Woodbury pipeline is well-established and works well in FP32. The motivation for the Neumann alternative comes from three converging trends:
1. **Chunkwise training** (Mamba-2, DeltaNet) favors matmul-heavy computation over elementwise ops — Neumann is pure matmul
2. **BF16 training** — Woodbury's elementwise divisions are numerically fragile in BF16; Neumann's GEMMs are robust
3. **Kernel fusion** — Neumann terms can be fused into a single IO-aware kernel, while Woodbury requires storing intermediate diagonal-scaled vectors

### Gap in Existing Proposals

Proposal 003 combines DPLR with column-sparse matrices and the Cauchy kernel trick. Proposal 004 combines oscillatory stability with DPLR. Neither addresses the *numerical method* for evaluating the resolvent itself. This proposal is orthogonal: it replaces the resolvent evaluation subroutine, and could be composed with any of the DPLR-based proposals.

## Mathematical Formulation

**Standard DPLR Resolvent (Woodbury):**

$$
R(z) := (zI - A)^{-1} = D_z - D_z P(I + Q^* D_z P)^{-1} Q^* D_z
$$

where $D_z = \text{diag}\left(\frac{1}{z - \lambda_i}\right)_{i=1}^N$.

**Problem**: When $|z - \lambda_i| < \epsilon$, $D_z$ entries exceed $1/\epsilon$, amplifying FP16 rounding errors.

**Proposed Neumann Resolvent:**

Factor the resolvent as:

$$
R(z) = D_z \cdot (I - E_z)^{-1}, \quad E_z := D_z P Q^* \in \mathbb{R}^{N \times N}
$$

Note that $E_z$ has rank $\leq r$ (typically $r = 1$ or $2$), so:

$$
E_z^j = D_z P (Q^* D_z P)^{j-1} Q^* \quad \text{for } j \geq 1
$$

Each power $E_z^j$ costs only $O(Nr + r^3)$ because we never form the full $N \times N$ product — we keep it in factored form.

**Truncated Neumann approximation:**

$$
R_k(z) = D_z \cdot S_k(E_z) = D_z \left(I + \sum_{j=1}^{k-1} D_z P (Q^* D_z P)^{j-1} Q^*\right)
$$

**Radix-$m$ acceleration for the core:**

The inner computation reduces to evaluating $S_k(F)$ where $F = Q^* D_z P \in \mathbb{C}^{r \times r}$ is a tiny $r \times r$ matrix. For $r = 1$, this is a geometric series $S_k(f) = (1 - f^k)/(1 - f)$ — exact and trivial. For $r = 2$, binary splitting gives $S_{2n}(F) = S_n(F)(I + F^n)$ in $2\log_2 k$ products of $2 \times 2$ matrices.

**Full kernel computation:**

$$
\hat{K}_k(\omega_j) = C \cdot R_k(e^{i\omega_j}) \cdot B = C D_z B + \sum_{m=1}^{k-1} (C D_z P) \cdot (Q^* D_z P)^{m-1} \cdot (Q^* D_z B)
$$

This is a sum of $k$ terms, each involving Cauchy-like dot products (the $D_z$ scaling), but accumulated via matrix products rather than division.

**Convergence condition:**

$$
\|E_z\|_2 = \|D_z P Q^*\|_2 \leq \|D_z P\|_2 \|Q^*\|_2 < 1
$$

For the HiPPO-LegS initialization, $\|P\|_2 \|Q\|_2 = O(1)$, so convergence holds whenever $\|D_z\|_2$ (the inverse minimum distance to the spectrum) satisfies $\|D_z\|_2 < 1/(\|P\|_2 \|Q\|_2)$.

**Handling near-resonance ($z \approx \lambda_i$):**

When $z$ is close to an eigenvalue $\lambda_i$, we can apply a **spectral shift**: split $\Lambda = \Lambda_{\text{safe}} \oplus \lambda_i$ and use exact inversion for the single problematic eigenvalue combined with Neumann for the rest. This hybrid approach retains GEMM dominance while handling edge cases.

**Key Variables:**

- $A = \Lambda + PQ^* \in \mathbb{C}^{N \times N}$ — DPLR state matrix
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ — diagonal component
- $P, Q \in \mathbb{C}^{N \times r}$ — low-rank factors ($r = 1$ or $2$ for HiPPO)
- $D_z = (zI - \Lambda)^{-1}$ — diagonal resolvent
- $E_z = D_z PQ^*$ — rank-$r$ perturbation matrix
- $F = Q^* D_z P \in \mathbb{C}^{r \times r}$ — core $r \times r$ matrix
- $k$ — truncation order (number of Neumann terms)
- $S_k(E_z)$ — truncated Neumann sum

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | S4/S5-style SSM with DPLR state matrix |
| Resolvent method | Neumann series (order $k = 4$–$8$) with radix-$m$ acceleration |
| Layers | $L = 4$–$8$ |
| State dim | $N = 64$–$256$ |
| Hidden dim | $d = 128$–$512$ |
| Low-rank | $r = 1$ or $2$ (HiPPO-LegS) |
| Precision | BF16 throughout (no FP32 fallback) |
| Training mode | Chunkwise parallel (chunk size $C = 256$) |

### Baseline

1. **Exact Woodbury resolvent (FP32)**: Standard S4 kernel computation — $O(N + r^2)$ per frequency, FP32 precision. Complexity: $O((N + L) \log L)$ total via FFT.
2. **Exact Woodbury resolvent (BF16)**: Same algorithm but in BF16 — expected to show instability near resonant frequencies.
3. **Cauchy kernel trick**: S4's standard Cauchy kernel evaluation at roots of unity.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity gap vs exact | $< 1\%$ relative | WikiText-103 validation perplexity |
| Numerical stability | No NaN/Inf in BF16 | Monitor gradient norms and loss divergence |
| Wall-clock speedup | $> 1.5\times$ vs FP32 Woodbury | Time per training step on A100 |
| Kernel accuracy | $\|\hat{K}_k - \hat{K}_\text{exact}\|_2 / \|\hat{K}_\text{exact}\|_2 < 10^{-3}$ | Relative error of SSM kernel |

### Estimated Compute

**Small**. The experiment compares resolvent evaluation methods on a fixed S4 architecture. The model sizes are small (4–8 layers, 128–512 hidden dim, ~5M–30M parameters). Key experiments:
- Kernel accuracy sweep (varying $k$): < 1 GPU-hour
- WikiText-103 training comparison: ~24 GPU-hours per configuration, 4 configurations = ~96 GPU-hours total
- Numerical stability profiling: < 4 GPU-hours

**Total: ~100 GPU-hours on A100.**

## Expected Outcome

**If hypothesis is correct:**

- **Numerical stability**: Neumann resolvent in BF16 produces stable training (no NaN/Inf) while Woodbury in BF16 diverges after $\sim 10K$ steps when eigenvalues cluster
- **Speed**: $1.5$–$2\times$ wall-clock speedup from (a) BF16 tensor core throughput ($2\times$ vs FP32) and (b) kernel fusion of Neumann GEMMs
- **Quality**: Perplexity within $1\%$ of exact FP32 Woodbury at truncation order $k = 4$–$8$
- **Scaling**: Benefit increases with state dimension $N$ because the GEMM advantage over elementwise division grows

**If hypothesis is wrong:**

- If $k = 8$ is insufficient for convergence (kernel error $> 1\%$), this reveals that the low-rank perturbation $PQ^*$ is too large for practical Neumann truncation — a useful negative result that constrains the design space for approximate resolvent methods
- If BF16 Woodbury works fine in practice (no instability), the numerical stability motivation is moot, but the GEMM-friendly computation may still offer speed benefits via fusion

## Minimum Viable Experiment

**CRITICAL**: Before running the full training experiment, validate that the Neumann resolvent produces accurate SSM kernels.

### Setup

- **Model**: No model needed — this is a kernel accuracy test
- **Task**: Compute the SSM kernel $\hat{K}(\omega_j)$ for $j = 1, \ldots, L$ using both exact Woodbury and Neumann approximation
- **Data**: Use HiPPO-LegS initialization ($N = 64$, $r = 1$) with $L = 1024$ frequencies
- **Compute**: Single GPU (or CPU), $< 2$ minutes
- **Sweep**: Vary truncation order $k \in \{2, 4, 6, 8, 12, 16\}$

### Success Criteria

- **Relative kernel error**: $\|\hat{K}_k - \hat{K}_\text{exact}\|_2 / \|\hat{K}_\text{exact}\|_2 < 10^{-3}$ for $k \leq 8$
- **Near-resonance robustness**: When we artificially set $z_j = \lambda_i + \epsilon$ with $\epsilon = 10^{-3}$, the Neumann resolvent in BF16 produces finite results while exact Woodbury in BF16 overflows
- **GEMM speed**: Neumann evaluation is faster than Woodbury evaluation in BF16 for $N \geq 64$

### Failure Criteria

- **Kill the idea if**: Neumann requires $k > 16$ for $10^{-3}$ accuracy — the GEMM count makes it slower than exact methods
- **Kill the idea if**: The spectral radius $\|E_z\|_2 > 1$ for a significant fraction ($> 10\%$) of frequencies with HiPPO initialization — convergence is not guaranteed without extensive preconditioning
- **Pause and investigate if**: Neumann is accurate but not faster — fusion overhead or small matrix size may negate the GEMM advantage

### Why This Test Is Sufficient

1. **Kernel accuracy is the core question**: If the Neumann series can't approximate the SSM kernel accurately, no amount of training will fix the approximation error
2. **Near-resonance behavior is testable without training**: We can synthetically construct worst-case frequencies
3. **Speed comparison is architecture-independent**: GEMM vs elementwise division performance ratios transfer to any model size
4. **2 minutes to signal**: Fast iteration on truncation order and precision

## Theoretical Analysis

**Complexity comparison per frequency $\omega_j$:**

| Operation | Exact Woodbury | Neumann ($k$ terms) |
|-----------|---------------|---------------------|
| Diagonal inverse $D_z$ | $N$ divisions (FP32 needed) | $N$ divisions (but only for stable frequencies) |
| Core computation | $O(Nr + r^3)$ mixed ops | $O(kr^3)$ GEMMs on $r \times r$ matrices |
| Kernel evaluation | $O(Nr)$ Cauchy dot products | $O(kNr)$ GEMM-like accumulations |
| Total per frequency | $O(Nr + r^3)$ | $O(kNr + kr^3)$ |
| Precision required | FP32 for $D_z$ | BF16 throughout |

**Crossover analysis:**

For $r = 1$ (rank-1 HiPPO), the exact method costs $O(N)$ per frequency. The Neumann method costs $O(kN)$ per frequency — $k\times$ more work. However:

1. BF16 tensor core throughput is $2\times$ FP32 throughput → net factor: $k/2$
2. Kernel fusion eliminates $k-1$ intermediate memory round-trips → estimated $1.5\times$ additional speedup
3. For $k = 4$: net cost ratio $\approx 4/(2 \times 1.5) = 1.33\times$ — slightly more compute but in a fusion-friendly form

The real win is for $r = 2$ or higher rank, where the $r \times r$ matrix operations dominate and GEMM acceleration is more impactful.

**Approximation error bound:**

$$
\|R_k(z) - R(z)\| \leq \|D_z\| \cdot \frac{\|E_z\|^k}{1 - \|E_z\|} \leq \frac{\|E_z\|^k}{(1 - \|E_z\|) \min_i |z - \lambda_i|}
$$

For HiPPO-LegS with $N = 64$: empirically $\|E_z\|_2 \approx 0.3$–$0.5$ for most frequencies, giving $\|E_z\|^4 \approx 10^{-2}$–$10^{-1}$ at $k = 4$.

## Risks & Limitations

1. **Spectral radius > 1**: If the low-rank perturbation $PQ^*$ is large relative to the eigenvalue gaps, $\|E_z\| \geq 1$ and the Neumann series diverges. Mitigation: use spectral shift for problematic frequencies, or increase the gap via initialization constraints (PTD).

2. **Small state dimension $N$**: For $N = 16$–$32$ (common in Mamba), the overhead of Neumann terms may exceed the benefit of BF16 + fusion. The technique is most promising for $N \geq 64$.

3. **Training dynamics**: Even if the kernel approximation is accurate, the gradient of the Neumann approximation may have different conditioning than the exact gradient, affecting optimizer convergence.

4. **Limited to convolutional mode**: Neumann resolvent accelerates the kernel computation used in convolutional training. For recurrent inference, the resolvent is not computed (state updates are direct), so no benefit during inference.

5. **HiPPO-specific**: The convergence properties depend on the initialization. Non-HiPPO initializations (random, learned) may have larger $\|PQ^*\|$ and worse convergence.

## Follow-up Experiments

1. **Compose with Proposal 004 (Oscillatory-DPLR)**: The oscillatory parameterization guarantees $|\lambda_i| \leq 1$ by construction. Test whether this also helps control $\|E_z\|$, making the Neumann series converge faster.

2. **Adaptive truncation order**: Use a cheap spectral radius estimate (power iteration, 2–3 steps) to dynamically select $k$ per frequency — low $k$ for well-separated eigenvalues, high $k$ near resonances.

3. **Neumann for DeltaNet chunkwise**: In DeltaNet's chunkwise training, the WY representation involves similar resolvent-like computations. Test whether Neumann approximation accelerates DeltaNet chunk processing.

4. **Gradient analysis**: Compare gradient variance between Neumann and exact resolvent to understand if the approximation acts as implicit regularization (similar to dropout or stochastic depth).

5. **Scale to Mamba-2**: Apply Neumann resolvent to Mamba-2's SSD algorithm, where the semiseparable decomposition involves resolvent-like block inversions that could benefit from GEMM-friendly approximation.

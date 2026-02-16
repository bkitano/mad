# 240: Predictability-Conditioned Parallel Evaluation of Nonlinear SSMs

**Category**: parallelization
**Gain type**: efficiency
**Source**: Gonzalez, Kozachkov, Zoltowski, Clarkson & Linderman, "Predictability Enables Parallelization of Nonlinear State Space Models" (arXiv:2508.16817, NeurIPS 2025)
**Paper**: papers/predictability-parallel-nonlinear-ssm.pdf
**Documented**: 2026-02-15

## Description

This work establishes a precise theoretical connection between the **predictability** of a nonlinear state space model (measured by its Largest Lyapunov Exponent, LLE) and the **conditioning** of the optimization problem used to parallelize its evaluation. The key insight: DEER/DeepPCR-style parallelization (which recasts sequential recurrence evaluation as optimization) converges fast for **predictable** (contracting) systems ($\lambda < 0$) and fails for **chaotic** systems ($\lambda > 0$), with a sharp phase transition at $\lambda = 0$.

For predictable systems, the state trajectory can be computed in $O((\log T)^2)$ parallel time — a major improvement over $O(T)$ sequential evaluation. This provides a principled **design criterion** for parallelizable nonlinear RNNs: parameterize models to be contractive (negative LLE), and parallel training via Gauss-Newton + parallel scan will converge rapidly.

This is directly actionable for architecture design: models like LrcSSM and ParaRNN that enforce contractivity via weight norm clipping or spectral parameterization benefit from this guarantee.

## Mathematical Form

**Nonlinear State Space Model:**

$$
s_t = f_t(s_{t-1}) \in \mathbb{R}^D
$$

where $f_t$ are input-dependent nonlinear maps (e.g., $f_t(s) = \tanh(Ws + Bu_t)$).

**Merit Function (Residual Optimization):**

Stack all states into $\mathbf{s} = \text{vec}([s_1, \ldots, s_T]) \in \mathbb{R}^{TD}$ and define:

$$
\mathbf{r}(\mathbf{s}) := \text{vec}([s_1 - f_1(s_0), \ldots, s_T - f_T(s_{T-1})]) \in \mathbb{R}^{TD}
$$

$$
\mathcal{L}(\mathbf{s}) := \frac{1}{2} \|\mathbf{r}(\mathbf{s})\|_2^2
$$

The true trajectory $\mathbf{s}^*$ is the unique global minimum with $\mathcal{L}(\mathbf{s}^*) = 0$.

**Gauss-Newton (DEER) Update:**

$$
\mathbf{s}^{(i+1)} = \mathbf{s}^{(i)} - \mathbf{J}(\mathbf{s}^{(i)})^{-1} \mathbf{r}(\mathbf{s}^{(i)})
$$

where $\mathbf{J}$ is the block bidiagonal Jacobian:

$$
\mathbf{J}(\mathbf{s}^{(i)}) = \begin{pmatrix} I_D & 0 & \cdots & 0 \\ -J_2^{(i)} & I_D & \cdots & 0 \\ \vdots & \ddots & \ddots & \vdots \\ 0 & \cdots & -J_T^{(i)} & I_D \end{pmatrix}, \quad J_t^{(i)} := \frac{\partial f_t}{\partial s_{t-1}}(s_{t-1}^{(i)})
$$

The block bidiagonal structure means solving $\mathbf{J}^{-1}\mathbf{r}$ is equivalent to a **linear recurrence** — computable in $O(\log T)$ time via parallel scan.

**Largest Lyapunov Exponent (LLE):**

$$
\text{LLE} := \lim_{T \to \infty} \frac{1}{T} \log(\|J_T J_{T-1} \cdots J_1\|) = \lambda
$$

- $\lambda < 0$: **Predictable** (contracting) — trajectories converge
- $\lambda > 0$: **Unpredictable** (chaotic) — trajectories diverge exponentially

**Key Definitions:**

- $s_t \in \mathbb{R}^D$ — hidden state at time $t$
- $f_t : \mathbb{R}^D \to \mathbb{R}^D$ — input-conditioned transition function
- $J_t = \partial f_t / \partial s_{t-1}$ — Jacobian of the dynamics
- $\mu$ — Polyak-Łojasiewicz (PL) constant of $\mathcal{L}$

**Theorem 2 (PL Constant ↔ Lyapunov Exponent):**

Assume the regularity condition: $\forall t > 1, \forall k \geq 0$,

$$
b \, e^{\lambda k} \leq \|J_{t+k-1} J_{t+k-2} \cdots J_t\| \leq a \, e^{\lambda k}
$$

where $a \geq 1, b \leq 1$. Then the PL constant $\mu$ satisfies:

$$
\frac{1}{a} \cdot \frac{e^{\lambda} - 1}{e^{\lambda T} - 1} \leq \sqrt{\mu} \leq \min\left(\frac{1}{b} \cdot \frac{1}{e^{\lambda(T-1)}}, 1\right)
$$

For predictable systems ($\lambda < 0$): $\mu$ is bounded away from zero independent of $T$ → well-conditioned.
For unpredictable systems ($\lambda > 0$): $\mu \to 0$ exponentially in $T$ → ill-conditioned, flat merit landscape.

**Theorem 4 (Global Linear Convergence of DEER):**

$$
\|\mathbf{e}^{(i)}\|_2 \leq \chi_w \, \beta^i \|\mathbf{e}^{(0)}\|_2
$$

where $0 < \beta < 1$ and $\chi_w \geq 1$ is a transient growth constant. When $\lambda$ is sufficiently negative, $\chi_w$ is small and DEER converges with little-to-no overshoot.

**Theorem 5 (Basin of Quadratic Convergence):**

If the residual satisfies:

$$
\|\mathbf{r}(\mathbf{s}^{(i)})\|_2 < \frac{2\mu}{L}
$$

then $\mathbf{s}^{(i)}$ is in the basin of quadratic convergence (superlinear rate). In terms of the LLE:

$$
\|\mathbf{r}(\mathbf{s}^{(i)})\|_2 < \frac{2}{a^2 L} \cdot \left(\frac{e^{\lambda} - 1}{e^{\lambda T} - 1}\right)^2
$$

**Total Parallel Time Complexity:**

Each Gauss-Newton step takes $O(\log T)$ via parallel scan. For predictable systems, $O(\log T)$ steps suffice → total $O((\log T)^2)$ parallel time.

## Complexity

| Operation | Sequential | DEER (Predictable, $\lambda < 0$) | DEER (Chaotic, $\lambda > 0$) |
|-----------|-----------|-----------------------------------|-------------------------------|
| State evaluation | $O(T)$ | $O((\log T)^2)$ | $O(T)$ or worse |
| Per Gauss-Newton step | $O(T)$ | $O(\log T)$ via scan | $O(\log T)$ via scan |
| Number of GN steps | — | $O(\log T)$ | $O(T)$ (diverges) |

**Memory:** $O(TD)$ for the full trajectory. Each scan stores $O(TD)$ intermediate states. Quasi-Newton variants reduce memory at the cost of more steps.

## Applicability

- **Nonlinear RNN design**: Provides a design principle — parameterize RNNs to be contractive (clip Jacobian spectral norms, use spectral parameterization) to guarantee efficient parallelization
- **LrcSSM / ParaRNN models**: These architectures enforce contractivity and benefit directly — verified at 7B parameter scale with competitive perplexity
- **Nonlinear SSMs**: Any model of the form $s_t = f_t(s_{t-1})$ where $f_t$ has bounded Jacobian norms
- **Chaotic observers**: Stable observers of chaotic systems (which have negative observer LLE despite positive system LLE) converge in 2-3 DEER steps
- **Diffusion model sampling**: Sequential denoising steps are contractive, suggesting DEER can parallelize sampling

## Limitations

- **Only helps predictable systems**: Chaotic or unstable dynamics (positive LLE) lead to ill-conditioned optimization — parallelization converges too slowly to be useful. This is a **fundamental** limitation, not an algorithm deficiency
- **Memory overhead**: DEER materializes the full $TD$-dimensional trajectory, which can be large. Quasi-Newton methods reduce this but need more iterations
- **Linearity in each GN step**: Each iteration solves a *linear* system via parallel scan, requiring Jacobian computation ($D \times D$ per timestep) — this adds overhead vs. simple linear recurrence
- **Transient growth**: The constant $\chi_w$ can be large for systems with transient instability (even if asymptotically stable), requiring more iterations in practice
- **Not for linear SSMs**: Linear SSMs are already parallelizable via associative scan in one pass; this technique targets the gap for *nonlinear* models

## GPU Efficiency Analysis

**Memory Access Pattern**: The Jacobian $\mathbf{J}$ is block bidiagonal — each Gauss-Newton step reduces to a parallel scan over $D \times D$ matrix-vector products. These are coalesced, cache-friendly operations when states are stored contiguously per timestep.

**Tensor Core Utilization**: Each scan step involves $D \times D$ matrix multiplications (for the Jacobian products). For typical hidden sizes ($D = 256$–$4096$), these map well to tensor core tiles. The GEMM-dominated workload has high arithmetic intensity.

**Parallelism**: The parallel scan saturates all SMs with independent work across the $O(\log T)$ levels. For sequences of length $T = 8192$ with 13 scan levels, each level has thousands of independent $D \times D$ matmuls.

**Sequential Bottleneck**: $O(\log T)$ Gauss-Newton iterations, each requiring a full parallel scan — typically 5-15 iterations for well-conditioned systems. Each iteration is a separate kernel launch, but the scan itself is a single fused kernel.

**Practical Results**: On H100, DEER achieves order-of-magnitude speedups over sequential evaluation for predictable RNNs. Models at 7B parameters trained at competitive throughput with Mamba-2 and Transformers.

## Implementation Notes

```python
# DEER/Gauss-Newton parallel evaluation of nonlinear SSMs
import jax
import jax.numpy as jnp
from jax.lax import associative_scan

def deer_parallel_eval(f_list, s0, num_iters=10, tol=1e-6):
    """
    f_list: list of T functions f_t(s_{t-1}) -> s_t
    s0: initial state (D,)
    Returns: state trajectory (T, D)
    """
    T, D = len(f_list), s0.shape[0]

    # Initialize trajectory (e.g., zeros or forward prediction)
    s = jnp.zeros((T, D))

    for i in range(num_iters):
        # Compute residuals: r_t = s_t - f_t(s_{t-1})
        s_prev = jnp.concatenate([s0[None], s[:-1]], axis=0)
        f_vals = jnp.stack([f_t(sp) for f_t, sp in zip(f_list, s_prev)])
        r = s - f_vals  # (T, D)

        # Compute Jacobians: J_t = df_t/ds_{t-1}
        J = jnp.stack([jax.jacobian(f_t)(sp)
                        for f_t, sp in zip(f_list, s_prev)])  # (T, D, D)

        # Solve block bidiagonal system J^{-1} r via parallel scan
        # Linear recurrence: delta_t = J_t @ delta_{t-1} + r_t
        def scan_fn(carry_a, carry_b):
            # (matrix, vector) pairs composed associatively
            A_a, b_a = carry_a
            A_b, b_b = carry_b
            return (A_b @ A_a, A_b @ b_a + b_b)

        elements = (J, r)  # T pairs of (D×D matrix, D vector)
        _, delta = associative_scan(scan_fn, elements)

        # Gauss-Newton update
        s = s - delta

        if jnp.linalg.norm(r) < tol:
            break

    return s
```

## References

- Gonzalez, X., Kozachkov, L., Zoltowski, D.M., Clarkson, K.L. & Linderman, S.W. "Predictability Enables Parallelization of Nonlinear State Space Models." NeurIPS 2025. arXiv:2508.16817.
- Lim, S. et al. "DEER: A Parallel Algorithm for Nonlinear SSMs." 2024.
- Danieli, D. et al. "DeepPCR: Parallelizing Sequential Operations in Neural Networks." ICML 2024.
- Gonzalez, X. et al. "Parallel Evaluation of Nonlinear Dynamical Systems." 2024.
- Farsang, D. et al. "LrcSSM: Contractive Nonlinear State Space Models." 2024.
- Zoltowski, D.M. et al. "Parallelizing MCMC Across the Sequence Length." NeurIPS 2025.

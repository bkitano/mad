# 253: LrcSSM — Diagonal Jacobian Nonlinear Scan

**Category**: parallelization
**Gain type**: efficiency
**Source**: Farsang, Hasani, Rus & Grosu (2025). Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling. arXiv:2505.21717.
**Paper**: papers/lrcssm-diagonal-jacobian-nonlinear-scan.pdf
**Documented**: 2026-02-16

## Description

Standard nonlinear RNNs (LSTMs, GRUs, Liquid Networks) have dense state-transition Jacobians, meaning their recurrence $s_t = f(s_{t-1}, u_t)$ cannot be directly parallelized via associative scan. Prior work (DEER, quasi-DEER, ELK) attempts to parallelize these by iteratively solving a fixed-point problem with Newton's method, but this requires multiple sequential iterations and may diverge. The key trick in LrcSSM is **architectural**: instead of trying to parallelize an arbitrary nonlinear recurrence after-the-fact, **redesign the model so its Jacobian $\partial f / \partial s$ is inherently diagonal** by construction.

The idea is deceptively simple: in a bio-inspired Liquid-Resistance Liquid-Capacitance (LRC) neural network, each neuron's state-dependent dynamics are restricted to self-loops only (no cross-state synaptic connections), while input-dependent dynamics retain full cross-neuron connectivity. Since the Jacobian of the recurrence with respect to the previous state depends only on the state-dependent terms, eliminating cross-state connections forces $\mathbf{A}(\mathbf{x}, \mathbf{u}) = \text{diag}(\ldots)$ — a diagonal matrix at every timestep.

With a diagonal Jacobian, each Newton iteration in the DEER/ELK parallelization reduces to a **scalar** parallel scan per state dimension (not a matrix scan), making the iterative updates exact rather than approximate. This is the critical difference from quasi-DEER (which approximates by extracting the diagonal) — LrcSSM's diagonal is the **true** Jacobian, giving exact parallel updates.

## Mathematical Form

**Continuous-Time LRC Model:**

$$
\dot{x}_i = (-\sigma(f_i^*) \sigma(\epsilon_i^*)) \, x_i + \tau(z_i^*) \sigma(\epsilon_i^*) \, e_i^{leak}
$$

where each neuron $i$ has state-dependent terms $f_i^*, z_i^*, \epsilon_i^*$ that depend on the current state $x_i$ and input $\mathbf{u}$.

**Key Separation (State vs. Input Dependent):**

$$
f_i^*(x_i, \mathbf{u}) = g_i^{max,x} \sigma(a_i^x x_i + b_i^x) + g_i^{max,u} \sigma\!\left(\sum_{j=1}^{n} a_{ji}^u u_j + b_j^u\right) + g_i^{leak}
$$

$$
z_i^*(x_i, \mathbf{u}) = k_i^{max,x} \sigma(a_i^x x_i + b_i^x) + k_i^{max,u} \sigma\!\left(\sum_{j=1}^{n} a_{ji}^u u_j + b_j^u\right) + g_i^{leak}
$$

$$
\epsilon_i^*(x_i, \mathbf{u}) = w_i^x x_i + v_i^x + \sum_{j=1}^{n} w_{ji}^u u_j + v_j^u
$$

**Discretized SSM Form (Euler):**

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta t \, \dot{\mathbf{x}}_{t-1}
$$

which gives:

$$
\dot{\mathbf{x}} = \mathbf{A}(\mathbf{x}, \mathbf{u})\mathbf{x} + \mathbf{b}(\mathbf{x}, \mathbf{u})
$$

**Diagonal State-Transition Matrix:**

$$
\mathbf{A}(\mathbf{x}, \mathbf{u}) = \text{diag}\left[-\sigma(f_1^*)\sigma(\epsilon_1^*), \ldots, -\sigma(f_m^*)\sigma(\epsilon_m^*)\right]
$$

**Input-Transition Vector:**

$$
\mathbf{b}(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \tau(z_1^*)\sigma(\epsilon_1^*) \, e_1^{leak} \\ \vdots \\ \tau(z_m^*)\sigma(\epsilon_m^*) \, e_m^{leak} \end{bmatrix}
$$

**Key Insight:** Because $\mathbf{A}$ is diagonal, the Jacobian $\frac{\partial f_t}{\partial s_{t-1}}$ is diagonal. In DEER's Newton iteration, solving $\mathbf{J}^{-1}\mathbf{r}$ where $\mathbf{J}$ is block bidiagonal with diagonal blocks reduces to $D$ **independent scalar** parallel scans — each of depth $O(\log T)$.

**Parallelization via DEER with Diagonal Jacobian:**

At each Newton iteration $i$:

$$
J_{s,t}^{(i)} = \text{diag}\!\left(\frac{\partial f_t}{\partial s_{t-1}}\Big|_{s^{(i)}}\right) \in \mathbb{R}^{D \times D}
$$

Since $J_{s,t}^{(i)}$ is diagonal, each dimension $d$ is an independent scalar recurrence:

$$
\delta_{t,d} = J_{s,t,d}^{(i)} \cdot \delta_{t-1,d} + r_{t,d}
$$

solvable via a scalar parallel prefix scan in $O(\log T)$ steps.

## Complexity

| Operation | Sequential RNN | LrcSSM (Parallel) | Quasi-DEER (Dense) |
|-----------|---------------|--------------------|--------------------|
| Per Newton step | $O(TD)$ | $O(TD)$ work, $O(\log T)$ depth | $O(TD^2)$ work |
| State update | $O(D)$ per step | $D$ independent scalar scans | $D \times D$ matrix scan |
| Total (k iters) | $O(TD)$ | $O(kTD)$ work, $O(k \log T)$ depth | $O(kTD^2)$ work |
| Forward+backward | $O(TDL)$ | $O(TDL)$ FLOPs | $O(TD^2L)$ FLOPs |

**Memory:** $O(TD)$ for full trajectory materialization during parallel evaluation. Same as sequential RNN.

**Sequential Depth:** $O(k \log T)$ where $k$ is the number of Newton iterations (typically 3-8 for stable systems).

## Applicability

- **Nonlinear RNNs that need parallelization**: Any nonlinear recurrence can be made parallelizable by restricting cross-state dependencies to the input pathway only, keeping the Jacobian diagonal. The paper demonstrates this for MGU, GRU, and LSTM variants (MguSSM, GruSSM, LstmSSM)
- **Long-horizon sequence classification**: LrcSSM outperforms LRU, S5, S6, Mamba, and Transformers on long-horizon benchmarks (EthanolConcentration, MotorImagery, EigenWorms)
- **Bio-inspired recurrent models**: LRC/LTC networks from computational neuroscience that model chemical synapse dynamics
- **Any SSM with input-dependent dynamics**: The diagonal Jacobian trick generalizes to arbitrary nonlinear SSMs where one can separate state-state interactions from input-state interactions

## Limitations

- **Multiple Newton iterations required**: Unlike linear SSMs which parallelize in a single scan pass, LrcSSM needs $k = 3$–$8$ DEER iterations, each launching a parallel scan. This adds kernel launch overhead and total FLOPs compared to linear SSMs
- **Reduced expressivity in state coupling**: Removing cross-state synaptic connections restricts the model's ability to capture multi-neuron interaction dynamics directly through the state. The paper shows no empirical performance loss, but the theoretical expressivity is reduced
- **Not validated at LLM scale**: Experiments are on UEA-MTSCA classification benchmarks with small models (64-unit states). No language modeling results at 1B+ scale
- **Explicit Euler only**: Currently uses explicit Euler integration; implicit integration schemes (IMEX) may further improve accuracy but complicate the diagonal Jacobian structure
- **Still iterative**: Each Newton iteration is a kernel launch; for large $k$, the overhead of multiple scan passes may reduce the practical speedup vs. sequential evaluation

## GPU Efficiency Analysis

**Memory Access Pattern**: Each Newton iteration performs $D$ independent scalar parallel scans over sequence length $T$. The scalars $J_{s,t,d}$ and $r_{t,d}$ are stored contiguously per dimension, giving coalesced access patterns. The diagonal structure eliminates the $D \times D$ matrix storage per timestep required by dense DEER.

**Tensor Core Utilization**: The scalar scans themselves do not use tensor cores (they are element-wise multiply-add operations). However, the function evaluations $f_t(s_{t-1}, u_t)$ involve $\sigma(\cdot)$ applied to input projections $\sum_j a_{ji}^u u_j + b_j^u$, which are matmuls that can use tensor cores. The MLP layers surrounding the SSM block also use tensor cores.

**Parallelism**: The $D$ scalar scans are fully independent and can saturate GPU SMs. For $D = 64$–$256$ and $T = 1000$–$18000$, this provides ample parallelism. The prefix scan structure has $O(\log T)$ sequential levels.

**Arithmetic Intensity**: Lower than matmul-based approaches since the core scan is element-wise scalar operations. The bottleneck is memory bandwidth for loading/storing the trajectory and intermediate Jacobian values. Fusing the function evaluation, Jacobian computation, and scan update into a single kernel would improve intensity.

**Practical Speedup**: Not explicitly benchmarked against sequential evaluation in the paper. The theoretical $O(\log T)$ depth improvement is promising, but the constant factor from multiple Newton iterations and low arithmetic intensity of scalar scans may limit real GPU speedup. Best suited for very long sequences where $T \gg D$.

## Implementation Notes

```python
# LrcSSM: Diagonal Jacobian enables exact parallel nonlinear recurrence
import torch
from torch import Tensor

def lrcssm_parallel_eval(
    x0: Tensor,       # (B, D) initial state
    u: Tensor,         # (B, T, n_inputs) input sequence
    params: dict,      # diagonal LRC parameters
    n_iters: int = 5,
    dt: float = 0.01,
) -> Tensor:
    """
    Parallel evaluation of LrcSSM via DEER with diagonal Jacobian.
    Each Newton iteration reduces to D independent scalar prefix scans.
    """
    B, T, _ = u.shape
    D = x0.shape[-1]

    # Initialize trajectory guess (e.g., repeat initial state)
    states = x0.unsqueeze(1).expand(B, T, D).clone()  # (B, T, D)

    for iteration in range(n_iters):
        # Compute f(s_{t-1}, u_t) and diagonal Jacobian df/ds for all t
        s_prev = torch.cat([x0.unsqueeze(1), states[:, :-1]], dim=1)

        # f_star, z_star, eps_star: state + input dependent terms
        # Key: state-dependent part uses ONLY self-loop (diagonal)
        f_star = compute_f_star_diagonal(s_prev, u, params)  # (B, T, D)
        z_star = compute_z_star_diagonal(s_prev, u, params)
        eps_star = compute_eps_star_diagonal(s_prev, u, params)

        # A is diagonal: A_diag[t,d] = -sigma(f_star[t,d]) * sigma(eps_star[t,d])
        A_diag = -torch.sigmoid(f_star) * torch.sigmoid(eps_star)  # (B, T, D)

        # b[t,d] = tau(z_star[t,d]) * sigma(eps_star[t,d]) * e_leak[d]
        b_vec = torch.tanh(z_star) * torch.sigmoid(eps_star) * params['e_leak']

        # x_dot = A * x + b (element-wise, A is diagonal)
        x_dot = A_diag * s_prev + b_vec
        f_vals = s_prev + dt * x_dot  # Euler step

        # Residual
        r = states - f_vals  # (B, T, D)

        # Diagonal Jacobian: J_diag[t,d] = d(f_vals[t,d]) / d(s_{t-1}[t,d])
        # This is a SCALAR per (t, d) — no D×D matrix!
        J_diag = compute_diagonal_jacobian(s_prev, u, params, dt)  # (B, T, D)

        # Solve block-bidiagonal system via D independent SCALAR prefix scans
        # delta[t,d] = J_diag[t,d] * delta[t-1,d] + r[t,d]
        # This is the key efficiency gain: scalar scan, not matrix scan
        delta = scalar_parallel_scan(J_diag, r)  # (B, T, D)

        states = states - delta

    return states

def scalar_parallel_scan(a: Tensor, b: Tensor) -> Tensor:
    """
    Parallel prefix scan for scalar recurrence: y[t] = a[t]*y[t-1] + b[t]
    a, b: (B, T, D) — D independent scalar scans
    Uses standard Blelloch scan, O(T) work, O(log T) depth.
    """
    # Reshape to (B*D, T) for batched scalar scan
    BD = a.shape[0] * a.shape[2]
    a_flat = a.permute(0, 2, 1).reshape(BD, -1)
    b_flat = b.permute(0, 2, 1).reshape(BD, -1)

    # Associative scan with binary operator (a1, b1) * (a2, b2) = (a2*a1, a2*b1 + b2)
    # Can use torch.compile or custom CUDA kernel
    result = torch._associative_scan(
        lambda x, y: (y[0] * x[0], y[0] * x[1] + y[1]),
        (a_flat, b_flat),
        dim=1
    )

    return result[1].reshape(a.shape[0], a.shape[2], -1).permute(0, 2, 1)
```

## References

- Farsang, M., Hasani, R., Rus, D. & Grosu, R. "Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling." arXiv:2505.21717, 2025.
- Gonzalez, X., Warrington, A., Smith, J. & Linderman, S. "Towards Scalable and Stable Parallelization of Nonlinear RNNs." NeurIPS 2024.
- Lim, S. et al. "DEER: A Parallel Algorithm for Nonlinear SSMs." 2024.
- Farsang, M., Hasani, R. & Grosu, R. "Liquid Resistance Liquid Capacitance Networks." NeurAI@NeurIPS 2024.
- Hasani, R. et al. "Liquid Time-Constant Networks." AAAI 2021.

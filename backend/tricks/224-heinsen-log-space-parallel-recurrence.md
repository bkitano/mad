# 224: Heinsen Log-Space Parallel Recurrence

**Category**: parallelization
**Gain type**: efficiency
**Source**: Heinsen, "Efficient Parallelization of a Ubiquitous Sequential Computation" (2023)
**Paper**: papers/heinsen-parallel-recurrence.pdf
**Documented**: 2026-02-15

## Description

Heinsen's method converts the ubiquitous first-order linear recurrence $x_t = a_t x_{t-1} + b_t$ into a fully parallel computation using exactly **two prefix sums in log-space**. The key insight is that by taking logarithms, the multiplicative accumulation of decay coefficients $a_t$ becomes an additive cumulative sum, and the input contributions $b_t$ (normalized by cumulative decay) can be accumulated via `logcumsumexp` — a numerically stable, parallelizable primitive available in most deep learning frameworks.

Unlike Blelloch's general formulation (which requires defining custom associative binary operators), Heinsen's approach works specifically with real-valued scalar/vector recurrences and reduces entirely to two standard prefix sum operations that map directly to highly optimized GPU primitives (`torch.cumsum` and `torch.logcumsumexp`). This makes it trivial to implement in PyTorch/JAX with no custom CUDA kernels needed.

This technique is foundational to the parallelization of modern linear RNNs (minGRU, GateLoop, RWKV) and related to the parallel scan used in Mamba/SSMs, achieving $n / \log n$ speedup over sequential computation.

## Mathematical Form

**Core Recurrence:**

$$
x_t = a_t x_{t-1} + b_t, \quad t = 1, 2, \ldots, n
$$

where $a_t \in \mathbb{R}^n$, $b_t \in \mathbb{R}^n$, and initial value $x_0 \in \mathbb{R}$.

**Step 1: Expand and factor.** Unrolling the recurrence:

$$
x_1 = a_1\left(x_0 + \frac{b_1}{a_1}\right), \quad x_2 = a_1 a_2 \left(x_0 + \frac{b_1}{a_1} + \frac{b_2}{a_1 a_2}\right), \quad \ldots
$$

In general:

$$
x_t = \left(\prod_{s=1}^{t} a_s\right) \left(x_0 + \sum_{s=1}^{t} \frac{b_s}{\prod_{r=1}^{s} a_r}\right)
$$

**Step 2: Take logarithms.** Define two prefix sums:

$$
a_t^* = \sum_{s=1}^{t} \log a_s \qquad \text{(cumulative log-decay)}
$$

$$
b_t^* = \sum_{s=1}^{t} e^{\log b_s - a_s^*} \qquad \text{(normalized cumulative input)}
$$

Then the log of the result is:

$$
\log x_t = a_t^* + \log(x_0 + b_t^*)
$$

**Step 3: Recover result via exponentiation:**

$$
x_t = e^{a_t^* + \log(x_0 + b_t^*)}
$$

**Numerically stable implementation (Eq. 7 from paper):**

$$
x_t = \exp\!\Big(a_t^* + \mathrm{tail}\!\big(\mathrm{LCSE}(\mathrm{cat}(\log x_0, \log b_t - a_t^*))\big)\Big)
$$

where $\mathrm{LCSE}(\cdot) := \log \sum^{\mathrm{cum}} \exp(\cdot)$ is the LogCumSumExp function, which internally applies the log-sum-exp trick for numerical stability.

**Key Definitions:**

- $a_t \in \mathbb{R}^n$ — decay (multiplicative) coefficients at each step
- $b_t \in \mathbb{R}^n$ — additive input at each step
- $x_0 \in \mathbb{R}$ — initial value
- $a_t^* \in \mathbb{R}^n$ — cumulative sum of $\log a_t$ (prefix sum #1)
- $b_t^* \in \mathbb{R}^n$ — cumulative sum of $e^{\log b_t - a_t^*}$ (prefix sum #2, via logcumsumexp)
- $\mathrm{LCSE}$ — LogCumSumExp, i.e., $\log \circ \mathrm{cumsum} \circ \exp$, with log-sum-exp stabilization

## Complexity

| Operation | Sequential | Heinsen Parallel |
|-----------|-----------|-----------------|
| Time (n processors) | $O(n)$ | $O(\log n)$ |
| Total work | $O(n)$ | $O(n)$ |
| Space | $O(n)$ | $O(n)$ |
| Speedup factor | — | $\frac{n}{\log n}$ |

**Memory:** $O(n)$ — same as sequential. No additional memory overhead beyond storing the prefix sum intermediates.

**GPU considerations:**
- Both `torch.cumsum` and `torch.logcumsumexp` are highly optimized GPU primitives with coalesced memory access
- Elementwise `log` and `exp` operations are trivially parallel with full GPU saturation
- No custom CUDA kernels needed — works out-of-the-box with PyTorch/JAX
- Arithmetic intensity is moderate (elementwise ops are memory-bound), but the $O(\log n)$ depth is the critical advantage for long sequences
- Compatible with tensor cores when batched across multiple independent sequences (batch × heads dimension)

## Applicability

- **Linear RNNs / Gated Linear Recurrences**: minGRU, minLSTM, GateLoop, HGRN — any model with the form $h_t = a_t \odot h_{t-1} + b_t$ where gates don't depend on $h_{t-1}$
- **State Space Models**: S4, S5, Mamba — the selective scan in Mamba computes an equivalent recurrence, though Mamba uses a more hardware-optimized variant with kernel fusion
- **RWKV**: The WKV computation in RWKV-4/5/6 involves first-order recurrences in log-space
- **Linear Attention**: Any linear attention variant that can be expressed as a cumulative sum of outer products with decay factors
- **Signal processing**: Exponential moving averages, IIR filters, discounted cumulative rewards in RL
- **Financial modeling**: Compound interest, portfolio value with varying returns

## Limitations

- **Scalar recurrences only**: Heinsen's formulation handles $x_t = a_t x_{t-1} + b_t$ where the "multiplication" is scalar (elementwise). It does **not** directly handle matrix-valued recurrences $H_t = A_t H_{t-1} + B_t$ where $A_t$ is a full matrix — those require the more general Blelloch associative scan with $2 \times 2$ matrix tuples.
- **Requires $a_t, b_t$ independent of $x_{t-1}$**: The method precomputes all coefficients before the scan. If gates depend on the hidden state (as in classical LSTM/GRU), the recurrence is nonlinear and this trick does not apply.
- **Log-space numerical issues**: When $a_t < 0$ or $b_t < 0$, the intermediate logarithms become complex. The implementation handles this via `complex_log`, but this adds overhead and can introduce precision issues in float16.
- **`logcumsumexp` availability**: Not all frameworks have a fused, numerically stable `logcumsumexp`. JAX provides `jax.lax.associative_scan` which subsumes this, but some frameworks may require manual implementation.
- **Memory-bound on GPU**: The elementwise log/exp operations are memory-bandwidth-bound, not compute-bound. For short sequences ($n < 1000$), the kernel launch overhead may negate the parallelism benefit.
- **Not fused**: Unlike FlashRNN or Mamba's fused selective scan kernel, this approach uses separate kernel launches for cumsum, logcumsumexp, and elementwise ops. A fused kernel would further improve throughput by reducing HBM round-trips.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def heinsen_parallel_recurrence(log_coeffs, log_values, x0):
    """
    Parallel computation of x_t = a_t * x_{t-1} + b_t using Heinsen's method.

    Args:
        log_coeffs: [B, T] or [B, T, D] — log(a_t), the log of decay coefficients
        log_values: [B, T] or [B, T, D] — log(b_t), the log of input values
        x0: scalar or [B, D] — initial value

    Returns:
        x: [B, T] or [B, T, D] — the full sequence x_1, ..., x_T
    """
    # Prefix sum #1: cumulative log-decay
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-2), (0, 0, 1, 0))  # [B, T+1, ...]

    # Prefix sum #2: logcumsumexp of normalized inputs
    # Prepend log(x0) and compute LCSE over [log(x0), log(b_1) - a*_1, ..., log(b_T) - a*_T]
    log_x0 = torch.log(torch.abs(x0)).unsqueeze(-2)  # [B, 1, ...]
    log_adjusted = log_values - a_star[:, 1:]          # normalize by cumulative decay
    log_x0_plus_b_star = torch.logcumsumexp(
        torch.cat([log_x0, log_adjusted], dim=-2), dim=-2
    )

    # Combine: x_t = exp(a*_t + log(x0 + b*_t))
    log_x = a_star[:, 1:] + log_x0_plus_b_star[:, 1:]  # drop the x0 position
    return torch.exp(log_x)

# For gated linear RNNs (e.g., minGRU):
# h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
# => log_coeffs = log(1 - z), log_values = log(z * h_tilde)

# The key advantage: this uses ONLY torch.cumsum and torch.logcumsumexp,
# both of which are highly optimized GPU primitives with no custom CUDA needed.
```

## References

- Heinsen, F.A. "Efficient Parallelization of a Ubiquitous Sequential Computation." arXiv:2311.06281, 2023.
- GitHub: https://github.com/glassroom/heinsen_sequence
- Blelloch, G.E. "Prefix Sums and Their Applications." CMU-CS-90-190, 1990.
- Ladner, R.E. & Fischer, M.J. "Parallel Prefix Computation." J. ACM 27(4):831-838, 1980.
- Feng et al. "Were RNNs All We Needed?" arXiv:2410.01201, 2024. (minGRU/minLSTM using this technique)
- Katsch. "GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling." arXiv:2311.01927, 2023.
- Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.

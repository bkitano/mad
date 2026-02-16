# 223: Rational Transfer Function (RTF) State-Free Parallel SSM Inference

**Category**: decomposition
**Gain type**: efficiency
**Source**: Parnichkun, Massaroli, Moro et al. (2024) — "State-Free Inference of State-Space Models: The Transfer Function Approach" (ICML 2024, arXiv:2405.06147)
**Paper**: [papers/rtf-state-free-ssm-inference.pdf]
**Documented**: 2026-02-15

## Description

Standard diagonal SSMs (S4, S4D, S5, LRU, Mamba) parameterize the state-space matrices $(A, B, C, h_0)$ and compute the convolutional kernel $h_t = C A^{t-1} B$ either via (a) a parallel scan over the state recurrence ($O(\ell \, n)$ time, $O(\ell \, n)$ memory — **state-multiplicative**), or (b) specialized Cauchy/Vandermonde kernel evaluation ($O((\ell + n) \log^2(\ell + n))$ time — complex to implement, platform-specific). Both approaches suffer from **memory costs that grow with state dimension $n$**, because they materialize the $n$-dimensional state across the sequence length.

The **Rational Transfer Function (RTF)** approach reparameterizes the SSM through its frequency-domain dual — the $\mathcal{Z}$-transform transfer function:

$$
H(z) = h_0 + \frac{b_1 z^{-1} + \cdots + b_n z^{-n}}{1 + a_1 z^{-1} + \cdots + a_n z^{-n}}
$$

where $(a, b, h_0)$ are learnable polynomial coefficients with $2n + 1$ total parameters. The key insight is that computing the convolutional kernel's spectrum reduces to **evaluating a rational function at the roots of unity**, which is equivalent to **two FFTs** (one for numerator, one for denominator) followed by an element-wise division:

$$
H_\ell(z^t) = \frac{\text{FFT}_\ell(\bar{b})_t}{\text{FFT}_\ell(\bar{a})_t} + h_0
$$

The convolution kernel is then recovered via a single inverse FFT. This yields:

- **State-free** parallel inference: $O(\ell)$ space, $O(\ell \log \ell)$ time — independent of state size $n$
- **35% training speedup** over S4/S4D on Long Range Arena
- **Constant inference latency** as state size grows (vs. linear growth for scan-based methods)
- **Full expressivity** of any LTI system (unlike diagonal parameterizations which cannot represent repeated poles)

For autoregressive inference, RTF translates to a **companion-form recurrence** with $O(n)$ time and memory per step (shift + dot product, no matrix exponentials).

## Mathematical Form

**Transfer function representation:**

Any linear time-invariant SSM with state-space matrices $(A, B, C, h_0)$ has an equivalent transfer function:

$$
H(z) = h_0 + C(zI - A)^{-1}B = h_0 + \frac{b_1 z^{-1} + \cdots + b_n z^{-n}}{1 + a_1 z^{-1} + \cdots + a_n z^{-n}}
$$

**Key Definitions:**

- $a = (a_1, \ldots, a_n) \in \mathbb{R}^n$ — denominator polynomial coefficients (learnable)
- $b = (b_1, \ldots, b_n) \in \mathbb{R}^n$ — numerator polynomial coefficients (learnable)
- $h_0 \in \mathbb{R}$ — feedforward/direct term (learnable)
- $n$ — state dimension (order of the system)
- $\ell$ — sequence length
- $h_t$ — impulse response (convolutional kernel) at time $t$

**Coordinate invariance (Lemma 3.1):** The transfer function coefficients $(a, b)$ are invariant under any invertible change of state-space basis $\hat{x} = Kx$:

$$
\hat{A} = KAK^{-1}, \quad \hat{B} = KB, \quad \hat{C} = CK^{-1} \implies \hat{H}(z) = H(z)
$$

This means diagonal, DPLR, companion, and dense parameterizations that share the same $(a, b)$ are functionally equivalent — motivating direct learning in transfer function space.

**Expressivity (Eq. 5):** Given state-space parameters, the transfer function coefficients are:

$$
a = \text{poly}(\text{eig}(A)), \quad b = \text{poly}(\text{eig}(A - BC))(h_0 - 1)
$$

where $\text{poly}(r)$ computes polynomial coefficients from roots. RTF with $2n+1$ parameters covers any impulse response representable by dense $n^2 + 2n + 1$ parameter state-space models.

**State-free parallel inference algorithm (Algorithm 1):**

1. **Pad** $b$ and $a$ to length $\ell$: $\bar{b} = \text{pad}(b, (1, \ell - n - 1))$, $\bar{a} = \text{pad}(a, (1, \ell - n - 1))$, set $\bar{a}_0 = 1$
2. **FFT** both: $B = \text{FFT}_\ell(\bar{b})$, $A = \text{FFT}_\ell(\bar{a})$
3. **Rational evaluation**: $H = B / A + h_0$
4. **Recover kernel**: $h = \text{iFFT}_\ell(H)$

Then convolve with input: $y = \text{iFFT}(\text{FFT}(h) \odot \text{FFT}(u))$.

**Mathematical justification (Lemma 3.2):** Evaluating the length-$\ell$ truncated transfer function at the $m$-th roots of unity $\mathbb{T}_m = \{z^k : z = e^{2\pi i/m}\}$ for $m \geq \ell$ yields exactly the spectrum of the impulse response:

$$
(H_\ell(z))_{z \in \mathbb{T}_m} = \text{FFT}_m(h)
$$

**Truncation tail (Lemma 3.3):** The "tail" of an infinite impulse response $\tilde{H}_\ell(z) = \sum_{t=\ell+1}^{\infty} CA^{t-1}Bz^{-t}$ satisfies:

$$
\tilde{H}_\ell(z) = CA^\ell z^{-\ell}(zI - A)^{-1}B
$$

This allows computing a correction $\tilde{C} = C(I - A^\ell) := \tilde{b}$ for truncation, applied only at deployment (not during training).

**Companion-form recurrence (for autoregressive inference):**

$$
x_{t+1} = \begin{bmatrix} -a_1 & -a_2 & \cdots & -a_n \\ 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{bmatrix} x_t + \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} u_t, \quad y_t = \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix} x_t + h_0 \, u_t
$$

The companion structure reduces the recurrence to a **shift + dot product**: $O(n)$ time and memory per step (shift the state vector down by one, compute $-\sum a_i x_t^i + u_t$ for the top entry, and read out $\sum b_i x_t^i + h_0 u_t$).

**Stable parameterization (Eq. 17–18):** For stability, the roots of the denominator polynomial must satisfy $|r| \leq 1$. RTF uses **zero initialization** ($a = b = \mathbf{0}$), which places all poles at the origin, avoiding the restricted stable region of Montel's method used by SpaceTime.

## Complexity

| Operation | S5 (Scan-based) | S4/S4D (Cauchy) | RTF (This work) |
|-----------|-----------------|-----------------|-----------------|
| Parallel training time | $O(\ell \, n)$ | $O((\ell + n) \log^2(\ell + n))$ | $O(\ell \log \ell)$ |
| Parallel training space | $O(\ell \, n)$ | $O(\ell + n)$ | $O(\ell)$ |
| Autoregressive time/step | $O(n)$ | $O(n)$ | $O(n)$ |
| Autoregressive space | $O(n)$ | $O(n)$ | $O(n)$ |
| Prefilling time | $O(\ell \, n)$ | — | $O(\ell \log_2 \ell)$ via companion |

**Memory:** $O(\ell)$ for parallel inference — no state materialization. Scan-based methods require $O(\ell \, n)$, which becomes prohibitive at large state sizes (see Figure 1 in paper: at $n = 2^{16}$, S5 uses $>10^4$ MB while RTF stays flat).

**Parameters:** $2n + 1$ per channel (vs. $n^2 + 2n + 1$ for dense, $4n + 1$ for diagonal + DPLR).

## Applicability

- **Long-context LTI sequence layers** (Hyena, S4, S4D, H3): RTF is a drop-in replacement for the convolutional filter parameterization. Hyena-RTF achieves 18.0 perplexity on WikiText-103 (vs. 18.5 for Hyena, 18.3 for Hyena-S5).
- **Large state sizes**: RTF's memory cost is independent of $n$, making very large state dimensions feasible for training. This is critical for tasks requiring high memory capacity (e.g., copying, delay tasks — RTF-1024 achieves 100% accuracy on Copying vs. S4-1024's 33.2%).
- **Multi-SISO architectures**: RTF applies independently per channel, fitting naturally into the multi-head SISO pattern used by S4, H3, and Hyena.
- **Time-invariant (LTI) layers only**: RTF parameterizes fixed convolutional filters, not input-dependent (selective) SSMs. It complements but does not replace Mamba-style selective mechanisms.
- **Distillation from MLP-parameterized filters**: RTF distills Hyena's implicit MLP filters into compact rational functions, achieving lower MSE than diagonal SSMs at small state sizes (Table 3).

## Limitations

- **LTI only**: RTF parameterizes time-invariant convolutions. It does **not** apply to selective/input-dependent SSMs (Mamba, GLA, DeltaNet) where $A_t$ varies per token. For modern architectures that rely on input-dependent gating, RTF is limited to the non-selective layers.
- **Stability enforcement is non-trivial**: Ensuring all poles lie inside the unit circle requires careful initialization. The zero initialization works well but constrains the initial dynamics.
- **FFT-based**: The parallel inference algorithm relies on FFT, which is well-optimized on GPUs but has lower arithmetic intensity than matmul-based operations. On tensor-core-heavy hardware (H100), FFT may not fully utilize WGMMA units.
- **Companion recurrence for inference** is sequential per token — the shift+dot-product structure is simple but not parallelizable across tokens.
- **Moderate empirical scale**: Experiments are on 160M-parameter models and LRA. Scaling behavior at 1B+ parameters is not yet validated.
- **Truncation correction** ($\tilde{b}$ computation) requires computing $A^\ell$ at deployment, which involves the companion matrix power — a one-time $O(n^2 \log \ell)$ cost.

## Implementation Notes

```python
import torch
import torch.fft as fft

def rtf_kernel_generation(a, b, h0, seq_len):
    """
    Generate convolutional kernel from RTF parameters.
    State-free: O(ell) space, O(ell log ell) time.

    Args:
        a: (n,) denominator polynomial coefficients
        b: (n,) numerator polynomial coefficients
        h0: scalar feedforward term
        seq_len: sequence length ell
    Returns:
        h: (seq_len,) convolutional kernel
    """
    n = a.shape[0]

    # Pad to sequence length
    a_padded = torch.zeros(seq_len, dtype=a.dtype, device=a.device)
    a_padded[0] = 1.0  # Monic denominator: 1 + a_1 z^{-1} + ...
    a_padded[1:n+1] = a

    b_padded = torch.zeros(seq_len, dtype=b.dtype, device=b.device)
    b_padded[1:n+1] = b  # b_0 = 0 (strictly proper part)

    # FFT of both polynomials
    A_freq = fft.rfft(a_padded)
    B_freq = fft.rfft(b_padded)

    # Rational function evaluation at roots of unity
    H_freq = B_freq / A_freq + h0

    # Inverse FFT to get impulse response
    h = fft.irfft(H_freq, n=seq_len)
    return h

def rtf_forward(u, a, b, h0):
    """
    Full RTF forward pass: convolve input with RTF kernel.
    Total: 3 FFTs + 1 iFFT + elementwise ops.
    """
    seq_len = u.shape[-1]
    h = rtf_kernel_generation(a, b, h0, seq_len)

    # FFT-based causal convolution
    U_freq = fft.rfft(u, n=2*seq_len)  # zero-pad for linear conv
    H_freq = fft.rfft(h, n=2*seq_len)
    y = fft.irfft(U_freq * H_freq, n=2*seq_len)[..., :seq_len]
    return y

def rtf_companion_step(x, u_t, a, b, h0):
    """
    Single autoregressive step using companion-form recurrence.
    O(n) time and memory per step.

    Args:
        x: (n,) state vector
        u_t: scalar input
        a: (n,) denominator coefficients
        b: (n,) numerator coefficients
        h0: scalar feedforward term
    Returns:
        y_t: scalar output
        x_new: (n,) updated state
    """
    # Output: y_t = b^T x + h0 * u_t
    y_t = (b * x).sum() + h0 * u_t

    # State update: shift down + feed in
    x_new = torch.zeros_like(x)
    x_new[0] = -(a * x).sum() + u_t  # Top entry
    x_new[1:] = x[:-1]  # Shift down by one
    return y_t, x_new
```

**GPU efficiency notes:**
- **Training:** 3 FFTs + 1 iFFT per layer, all well-optimized via cuFFT. The FFT-conv pattern is the same as FlashFFTConv (Fu et al., 2024), enabling direct reuse of that kernel.
- **Memory:** Only $O(\ell)$ buffers for the frequency-domain computation — no $O(\ell \, n)$ state tensor. This frees memory for larger batch sizes.
- **Arithmetic intensity:** FFT is $O(\ell \log \ell)$ FLOPs over $O(\ell)$ data = $O(\log \ell)$ FLOPs/byte, which is compute-bound for $\ell \geq 2^{10}$. The element-wise division is bandwidth-bound but negligible cost.
- **Inference:** Companion recurrence is a simple shift + dot product — maps to a tight CUDA kernel with no divergence, fully coalesced memory access for the state vector.

## References

- Parnichkun, R. N., Massaroli, S., Moro, A., Smith, J. T. H., Hasani, R., Lechner, M., An, Q., Ré, C., Asama, H., Ermon, S., Suzuki, T., Yamashita, A., & Poli, M. (2024). State-Free Inference of State-Space Models: The Transfer Function Approach. ICML 2024. arXiv:2405.06147. [https://arxiv.org/abs/2405.06147](https://arxiv.org/abs/2405.06147)
- Code: [https://github.com/rukelire/RTF](https://github.com/rukelire/RTF)
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022.
- Gupta, A., Gu, A., & Berant, J. (2022). Diagonal State Spaces (S4D/DSS). NeurIPS 2022.
- Smith, J. T. H., Warrington, A., & Linderman, S. (2023). Simplified State Space Layers (S5). ICLR 2023.
- Fu, D. Y., Kumbong, H., Nguyen, E., & Ré, C. (2024). FlashFFTConv: Efficient FFT convolutions for long sequences with tensor cores. ICLR 2024.
- Poli, M. et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. ICML 2023.

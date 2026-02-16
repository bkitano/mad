# 216: S4D Vandermonde Kernel — Diagonal SSM to Vandermonde Product

**Category**: decomposition
**Gain type**: efficiency
**Source**: Gu, Gupta, Goel & Ré (2022) — "On the Parameterization and Initialization of Diagonal State Space Models" (NeurIPS 2022)
**Paper**: [papers/s4d-diagonal-ssm-vandermonde.pdf]
**Documented**: 2026-02-15

## Description

The original S4 model computes the SSM convolution kernel via a sophisticated algorithm involving DPLR (diagonal plus low-rank) decomposition, Cauchy kernel evaluation, and Woodbury correction — requiring custom CUDA kernels and hundreds of lines of code. S4D shows that **restricting the state matrix $A$ to be diagonal** reduces the entire kernel computation to a **Vandermonde matrix-vector product**, implementable in just 2 lines of code while matching S4's performance.

The key insight: for a diagonal state matrix $A = \text{diag}(\bar{A}_0, \ldots, \bar{A}_{N-1})$, the discretized SSM kernel $\bar{K}_\ell = \sum_n C_n \bar{A}_n^\ell \bar{B}_n$ factors as a Hadamard product followed by a Vandermonde matrix-vector product. The Vandermonde matrix $\mathcal{V}_L(\bar{A})$ has entries $\mathcal{V}_{n,\ell} = \bar{A}_n^\ell$, and the multiplication $(\bar{B}^\top \circ C) \cdot \mathcal{V}_L(\bar{A})$ yields the full length-$L$ kernel. This eliminates the Cauchy kernel trick, Woodbury identity, and all DPLR-specific machinery.

The theoretical justification is that the diagonal restriction of S4's HiPPO-LegS matrix recovers the same basis functions as $N \to \infty$ (Theorem 3), and almost all SSMs are equivalent to a diagonal SSM over $\mathbb{C}$ (Proposition 2). The practical impact is enormous: S4D matches S4 on nearly all benchmarks while being dramatically simpler.

## Mathematical Form

**Continuous-time SSM:**

$$
x'(t) = Ax(t) + Bu(t), \quad y(t) = Cx(t)
$$

where $A \in \mathbb{C}^{N \times N}$, $B \in \mathbb{C}^{N \times 1}$, $C \in \mathbb{C}^{1 \times N}$.

**Convolution kernel:**

$$
K(t) = Ce^{tA}B, \quad y(t) = (K * u)(t)
$$

**Basis kernel decomposition:**

$$
K(t) = \sum_{n=0}^{N-1} C_n K_n(t), \quad K_n(t) := e_n^\top e^{tA} B
$$

**Discretization (bilinear or ZOH):**

For bilinear: $\bar{A} = (I - \Delta/2 \cdot A)^{-1}(I + \Delta/2 \cdot A)$, $\bar{B} = (I - \Delta/2 \cdot A)^{-1} \cdot \Delta B$

For ZOH: $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta \cdot A) - I) \cdot \Delta B$

**Discrete kernel:**

$$
y = u * \bar{K}, \quad \bar{K} = (C\bar{B}, C\bar{A}\bar{B}, \ldots, C\bar{A}^{L-1}\bar{B})
$$

**Core Operation — Vandermonde Product (when $A$ is diagonal):**

$$
\bar{K}_\ell = \sum_{n=0}^{N-1} C_n \bar{A}_n^\ell \bar{B}_n \implies \bar{K} = (\bar{B}^\top \circ C) \cdot \mathcal{V}_L(\bar{A})
$$

where $\circ$ is the Hadamard product and $\mathcal{V}_L(\bar{A})$ is the **Vandermonde matrix**:

$$
\mathcal{V}_L(\bar{A}) = \begin{pmatrix} 1 & \bar{A}_0 & \bar{A}_0^2 & \cdots & \bar{A}_0^{L-1} \\ 1 & \bar{A}_1 & \bar{A}_1^2 & \cdots & \bar{A}_1^{L-1} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \bar{A}_{N-1} & \bar{A}_{N-1}^2 & \cdots & \bar{A}_{N-1}^{L-1} \end{pmatrix} \in \mathbb{C}^{N \times L}
$$

**Key Definitions:**

- $A \in \mathbb{C}^{N \times N}$ — diagonal state matrix (complex-valued for expressivity)
- $\bar{A}_n \in \mathbb{C}$ — $n$-th diagonal entry of the discretized state matrix
- $B, C \in \mathbb{C}^N$ — input/output projections (vectors since $A$ is diagonal)
- $L$ — sequence length
- $N$ — state dimension (typically 64)
- $\mathcal{V}_L(\bar{A}) \in \mathbb{C}^{N \times L}$ — Vandermonde matrix

**Conjugate symmetry trick (real outputs from complex parameters):**

To parameterize a real SSM with state size $N$, use a complex SSM with state size $N/2$ and take $2 \cdot \text{Re}(\cdot)$ of the kernel. Parameters come in conjugate pairs, so the Vandermonde sum implicitly adds conjugate terms:

$$
\bar{K} = 2 \cdot \text{Re}\left((\bar{B}^\top \circ C) \cdot \mathcal{V}_L(\bar{A})\right)
$$

**HiPPO-D initialization (S4D-LegS):**

$$
A_n^{(D)} = \text{diag}(A^{(N)}) \quad \text{where } A_{nk}^{(N)} = -\begin{cases} (n+\tfrac{1}{2})^{1/2}(k+\tfrac{1}{2})^{1/2} & n > k \\ \tfrac{1}{2} & n = k \\ (n+\tfrac{1}{2})^{1/2}(k+\tfrac{1}{2})^{1/2} & n < k \end{cases}
$$

**Simplified initializations:**

$$
\text{(S4D-Inv)} \quad A_n = -\frac{1}{2} + i\frac{N}{\pi}\left(\frac{N}{2n+1} - 1\right) \qquad \text{(S4D-Lin)} \quad A_n = -\frac{1}{2} + i\pi n
$$

## Complexity

| Operation | S4 (DPLR + Cauchy) | S4D (Diagonal + Vandermonde) |
|-----------|---------------------|-------------------------------|
| Kernel computation | $\tilde{O}(N + L)$ via Cauchy kernel | $\tilde{O}(N + L)$ via Vandermonde |
| Naive GPU kernel | $O(NL)$ matmul | $O(NL)$ broadcast multiply |
| Convolution | $O(L \log L)$ via FFT | $O(L \log L)$ via FFT |
| Recurrence (inference) | $O(N)$ per step | $O(N)$ per step |
| Implementation complexity | ~300 lines, custom CUDA | **2 lines**, pure PyTorch |

**Memory:** $O(N + L)$ for both methods (without materializing the full $N \times L$ Vandermonde matrix). On GPU, the naive $O(NL)$ materialization uses $O(N + L)$ space via broadcasting.

**Theoretical vs practical:** On modern GPUs, the naive $O(NL)$ computation via broadcasting and summation (without the fast $\tilde{O}(N+L)$ Vandermonde algorithm) is typically sufficient since $N$ is small (64–256) and the operation maps to efficient elementwise + reduction kernels. The $O(L \log L)$ FFT convolution dominates.

## Applicability

- **Direct replacement for S4:** S4D is a drop-in replacement that eliminates S4's complex Cauchy kernel machinery while matching performance on Long Range Arena, speech commands, sequential CIFAR, and medical time series
- **Foundation for all subsequent diagonal SSMs:** S5, Mamba, Mamba-2, and most modern SSMs use diagonal state matrices, making S4D's Vandermonde insight the standard computational primitive
- **Any linear time-invariant (LTI) SSM:** The trick applies whenever the SSM has a fixed (non-input-dependent) diagonal state matrix and uses the convolutional mode for parallel training
- **Enables easy hybridization:** The simplicity of S4D makes it trivial to combine with attention, gating, and other mechanisms (H3, Hyena, etc.)

## Limitations

- **Does not apply to selective/input-dependent SSMs:** Mamba and Mamba-2 have input-dependent $A$, $B$, $C$ matrices, so the kernel cannot be precomputed as a single convolution. The Vandermonde trick only works in the LTI (convolutional) mode
- **Complex arithmetic required:** The diagonal parameterization requires complex-valued parameters and arithmetic, which has ~2× overhead vs real. Mitigated by conjugate symmetry (halving the state dimension)
- **Initialization is critical:** Random diagonal matrices perform poorly. The HiPPO-D initialization (or S4D-Lin/Inv approximations) is essential for long-range tasks. The real part controls decay rate ($-\frac{1}{2}$ is a good default), and imaginary parts must be spread out to capture different frequencies
- **Convolutional mode only:** During autoregressive inference, the model switches to the recurrent form $x_t = \bar{A} x_{t-1} + \bar{B} u_t$, which doesn't use the Vandermonde structure at all
- **Superseded by selective SSMs:** For state-of-the-art language modeling, input-dependent SSMs (Mamba) outperform fixed-kernel models like S4D. The Vandermonde trick is most relevant for understanding the theoretical foundations and for LTI applications (audio, time series)

## Implementation Notes

```python
# S4D kernel computation — the entire thing in 2 lines
# (from Figure 1 of S4D paper, ZOH discretization)
def s4d_kernel(A, B, C, dt, L):
    vandermonde = exp(arange(L)[:, None] * dt * A)  # (L, N) Vandermonde matrix
    return sum(vandermonde * B * C * (exp(dt * A) - 1) / A)  # (L,) kernel

# Full S4D-Lin implementation in NumPy (Listing 1 from paper):
def parameters(N, dt_min=1e-3, dt_max=1e-1):
    # Initialization
    log_dt = np.random.rand() * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
    A = -0.5 + 1j * np.pi * np.arange(N // 2)  # S4D-Lin initialization
    B = np.ones(N // 2) + 0j
    C = np.random.randn(N // 2) + 1j * np.random.randn(N)  # Variance preserving
    return log_dt, np.log(-A.real), A.imag, B, C

def kernel(L, log_dt, log_A_real, A_imag, B, C):
    # Discretize (bilinear transform)
    dt, A = np.exp(log_dt), -np.exp(log_A_real) + 1j * A_imag
    dA, dB = (1 + dt * A / 2) / (1 - dt * A / 2), dt * B / (1 - dt * A / 2)

    # Vandermonde matrix multiplication — THE CORE TRICK
    # Return twice the real part — same as adding conjugate pairs
    return 2 * ((B * C) @ (dA[:, None] ** np.arange(L))).real  # (L,) kernel

def forward(u, parameters):
    L = u.shape[-1]
    K = kernel(L, *parameters)
    # Convolve y = u * K using FFT
    K_f, u_f = np.fft.fft(K, n=2*L), np.fft.fft(u, n=2*L)
    return np.fft.ifft(K_f * u_f, n=2*L)[..., :L]
```

**GPU Efficiency Analysis:**

- **Memory access:** The Vandermonde product `dA[:, None] ** np.arange(L)` is a broadcast operation — coalesced memory access, high arithmetic intensity. On GPU, this becomes an elementwise kernel with no irregular access patterns
- **Tensor core utilization:** The core operation is *not* a standard matmul (it's elementwise power + reduce), so it does not directly use tensor cores. However, the subsequent FFT convolution is highly optimized on GPU
- **Arithmetic intensity:** For the naive $O(NL)$ approach: $N \cdot L$ complex multiplications and $N$ reductions. With $N = 64$, $L = 8192$, this is ~1M FLOPs — negligible compared to the FFT and the rest of the model
- **Practical bottleneck:** The FFT convolution ($O(L \log L)$) dominates wall-clock time, not the Vandermonde product. S4D's win is primarily in **implementation simplicity** and **code maintainability**, not raw kernel speed over S4
- **Why it matters for GPU:** By eliminating the Cauchy kernel + Woodbury correction, S4D removes two custom CUDA kernels and replaces them with standard PyTorch operations that benefit from framework-level optimizations (autograd, mixed precision, compilation)

## References

- Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). On the Parameterization and Initialization of Diagonal State Space Models. NeurIPS 2022. arXiv:2206.11893.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022. arXiv:2111.00396.
- Gupta, A., Gu, A., & Berant, J. (2022). Diagonal State Spaces are as Effective as Structured State Spaces (DSS). NeurIPS 2022. arXiv:2203.14343.
- Smith, J. T. H., Warrington, A., & Linderman, S. W. (2023). Simplified State Space Layers for Sequence Modeling (S5). ICLR 2023. arXiv:2208.04933.

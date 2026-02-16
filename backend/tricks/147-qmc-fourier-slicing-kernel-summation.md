# 147: QMC Fourier-Slicing Kernel Summation

**Category**: approximation
**Gain type**: efficiency
**Source**: Hertrich, Jahn & Quellmalz (ICLR 2025); Hertrich (SIAM J. Math. Data Sci. 2024)
**Paper**: [papers/fast-fourier-slicing-kernel-summation.pdf], [papers/fast-kernel-summation-high-dim-slicing.pdf]
**Documented**: 2026-02-15

## Description

This trick reduces $d$-dimensional radial kernel summation to a collection of **one-dimensional** kernel summations via random projections ("slicing"), then solves each 1D problem using the **non-equispaced fast Fourier transform (NFFT)**. By using quasi-Monte Carlo (QMC) sequences on the sphere instead of random directions, the convergence rate improves from $O(1/\sqrt{P})$ to $O(P^{-s/(d-1)})$ where $s$ depends on the kernel smoothness.

The key insight is that any radial kernel $K(x,y) = F(\|x - y\|)$ can be written as an expectation of 1D kernel evaluations over random projection directions on the unit sphere $\mathbb{S}^{d-1}$. The 1D "sliced" kernel $\mathsf{k}(x,y) = f(|x-y|)$ is related to the $d$-dimensional kernel via the **generalized Riemann-Liouville fractional integral transform**. Crucially, for the Gaussian kernel, this error bound is **dimension-independent**: the slicing variance is bounded by a constant $C = 1$ regardless of $d$.

This is highly GPU-friendly: the core operations are (1) matrix-vector products $\xi_p^T x_n$ which are batched GEMMs, and (2) 1D NFFTs which are well-optimized on GPU (torch-NFFT, cuFFT). The $P$ slicing directions are completely independent, enabling embarrassing parallelism.

## Mathematical Form

**Core Problem:**

Compute the kernel sums:

$$
s_m = \sum_{n=1}^{N} w_n K(x_n, y_m), \quad m = 1, \ldots, M
$$

where $K(x,y) = F(\|x - y\|)$ is a radial kernel with $x_n, y_m \in \mathbb{R}^d$.

**Slicing Representation:**

Any radial kernel admits the representation:

$$
K(x, y) = \mathbb{E}_{\xi \sim \mathcal{U}_{\mathbb{S}^{d-1}}} \left[ \mathsf{k}(\langle \xi, x \rangle, \langle \xi, y \rangle) \right]
$$

where $\mathsf{k}: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ is a one-dimensional kernel with basis function $f$ satisfying $\mathsf{k}(x,y) = f(|x-y|)$.

**Riemann-Liouville Fractional Integral Transform:**

The relation between $d$-dimensional basis function $F$ and 1D basis function $f$ is:

$$
F(s) = \frac{2\Gamma(\frac{d}{2})}{\sqrt{\pi}\Gamma(\frac{d-1}{2})} \int_0^1 f(ts)(1 - t^2)^{\frac{d-3}{2}} \, \mathrm{d}t
$$

For analytic $F(x) = \sum_{n=0}^{\infty} a_n x^n$, the 1D basis function has coefficients:

$$
f(x) = \sum_{n=0}^{\infty} b_n x^n, \quad b_n = \frac{\sqrt{\pi}\,\Gamma(\frac{n+d}{2})}{\Gamma(\frac{d}{2})\Gamma(\frac{n+1}{2})} a_n
$$

**Discretized Approximation (QMC Slicing):**

Choose $P$ directions $\xi_1, \ldots, \xi_P \in \mathbb{S}^{d-1}$ via QMC design, then:

$$
s_m \approx \frac{1}{P} \sum_{p=1}^{P} \sum_{n=1}^{N} w_n f(|\langle \xi_p, x_n - y_m \rangle|)
$$

**1D Fast Fourier Summation:**

For each slice $p$, compute the 1D kernel sum via truncated Fourier series. The 1D kernel is periodized and expanded:

$$
f(|x|) \approx \phi(x) = \sum_{k \in \mathcal{C}} c_k \exp(-2\pi i k x)
$$

where $\mathcal{C} = \{k \in \mathbb{Z} : -K_{\max} \leq k \leq K_{\max}, |c_k| > \epsilon\}$. Then the 1D sums become:

$$
t_m = \sum_{n=1}^{N} w_n \phi(\tilde{x}_n - \tilde{y}_m) = \sum_{k \in \mathcal{C}} c_k \left(\sum_{n=1}^{N} w_n e^{-2\pi i k \tilde{x}_n}\right) e^{2\pi i k \tilde{y}_m}
$$

This factorizes as: $t = \mathcal{F}_{\mathcal{C},y}(c \odot (\mathcal{F}_{\mathcal{C},x}^H w))$, computed via adjoint NFFT + NFFT.

**Gaussian Kernel Fourier Coefficients:**

For the Gaussian kernel, the 1D counterpart $f_\sigma(x) = {}_1F_1(\frac{d}{2}; \frac{1}{2}; \frac{-x^2}{2\sigma^2})$ has closed-form Fourier transform:

$$
\hat{f}_\sigma(k) = \frac{d\pi\sigma \exp(-2\pi^2\sigma^2 k^2)(2\pi^2\sigma^2 k^2)^{(d-1)/2}}{\sqrt{2}\,\Gamma(\frac{d+2}{2})}
$$

These coefficients decay rapidly (only a few $k$ have $|c_k| > \epsilon$), so $|\mathcal{C}|$ is small.

**Key Definitions:**

- $\xi_p \in \mathbb{S}^{d-1}$ — projection directions (QMC or random)
- $P$ — number of slicing directions (controls accuracy)
- $\mathcal{C}$ — set of significant Fourier coefficients
- $\mathcal{F}_{\mathcal{C},x}^H$ — adjoint NFFT (type-2 nonuniform FFT)
- $\mathcal{F}_{\mathcal{C},y}$ — NFFT (type-1 nonuniform FFT)

## Complexity

| Operation | Naive | With Slicing + NFFT |
|-----------|-------|---------------------|
| Kernel summation | $O(MN)$ | $O(P(M + N + N_{\text{ft}} \log N_{\text{ft}}))$ |
| Per-slice cost | — | $O(M + N + |\mathcal{C}|\log|\mathcal{C}|)$ |
| Total | $O(N^2)$ for $M=N$ | $O(PN)$ for $M=N$, small $|\mathcal{C}|$ |

**Memory:** $O(P(M + N) + |\mathcal{C}|)$ vs $O(MN)$

**Error rates:**

- Random slicing: $O(1/\sqrt{P})$ — dimension-independent for Gaussian kernel
- QMC distance design: $O(P^{-d/(2(d-1))})$ for Gaussian/Matérn kernels
- Spherical $t$-designs: $O(P^{-s/(d-1)})$ for kernels in Sobolev space $H^s(\mathbb{S}^{d-1})$

**Practical numbers:** For $d = 50$, $N = 60000$ (MNIST), the method with $P = 5000$ QMC Fourier slices achieves relative $L^1$ error $< 10^{-3}$ in seconds on a single CPU thread, vs. minutes for naive computation.

## Applicability

- **Kernel attention mechanisms**: Any attention variant using radial kernels (Gaussian, Laplace, Matérn) can use sliced fast summation. The operation $\text{softmax}(QK^T/\sqrt{d})V$ with a Gaussian kernel in the exponent can be reformulated and accelerated.
- **SSM kernel evaluation**: The Cauchy kernel sums $\sum_i v_i / (\omega_j - \lambda_i)$ appearing in S4 can potentially be sliced if reformulated as a radial kernel on appropriate embeddings.
- **Maximum Mean Discrepancy (MMD)**: Fast kernel summation directly accelerates MMD gradient flows used in generative modeling and distribution matching losses.
- **Gaussian process inference**: Kernel matrix-vector products in GP training with conjugate gradients — the dominant cost — are directly accelerated.
- **Stein Variational Gradient Descent**: Kernel sums appear in SVGD particle updates; slicing + NFFT makes this scale to large particle counts.
- **Kernel density estimation**: t-SNE/UMAP gradient computations involve kernel sums that this method directly accelerates.

## Limitations

- **Not faster than FlashAttention for standard softmax attention**: Standard dot-product attention is not a radial kernel sum, so this doesn't directly replace FlashAttention. It applies only to kernel-based attention variants.
- **Accuracy depends on number of slices $P$**: Error decays as $O(1/\sqrt{P})$ for random or $O(P^{-r})$ for QMC; high precision requires many slices, each adding cost.
- **QMC direction computation**: Constructing optimal QMC sequences on $\mathbb{S}^{d-1}$ via energy minimization (Adam optimizer on PyKeOps) takes minutes to hours for large $P$. However, this is a one-time precomputation.
- **Smooth kernels only for NFFT path**: Non-smooth kernels (Laplacian, negative distance) need the sorting algorithm fallback, which requires $O((M+N)\log(M+N))$ per slice — still fast but not as clean.
- **1D NFFT overhead**: The NFFT has constant-factor overhead; for very small $N$ or $M$, direct computation may be faster.
- **Spherical designs intractable for $d > 4$**: The best QMC sequences (spherical $t$-designs) are only computed for $\mathbb{S}^2$ and $\mathbb{S}^3$; for higher dimensions, heuristic QMC (Sobol, distance minimization) must be used.

## Implementation Notes

```python
# QMC Fourier-Slicing Fast Kernel Summation
# Reference implementation: github.com/johertrich/fastsum_QMC_slicing
import torch
import torch.fft

def qmc_fourier_slicing_kernel_sum(
    x: torch.Tensor,      # (N, d) source points
    y: torch.Tensor,      # (M, d) target points
    w: torch.Tensor,      # (N,) weights
    xi: torch.Tensor,     # (P, d) QMC directions on S^{d-1}
    fourier_coeffs: torch.Tensor,  # (|C|,) Fourier coefficients c_k
    freq_indices: torch.Tensor,    # (|C|,) frequency indices k in C
) -> torch.Tensor:
    """
    Compute s_m = sum_n w_n K(x_n, y_m) via sliced Fourier summation.

    Core operations: batched matmul (GPU GEMM) + 1D NFFT (GPU FFT).
    All P slices are processed in parallel.
    """
    P, d = xi.shape
    N = x.shape[0]
    M = y.shape[0]

    # Step 1: Project to 1D — batched GEMM, (P, N) and (P, M)
    proj_x = xi @ x.T  # (P, N)
    proj_y = xi @ y.T  # (P, M)

    # Step 2: Rescale to [-1/2, 1/2] for NFFT
    all_proj = torch.cat([proj_x, proj_y], dim=1)  # (P, N+M)
    c_min = all_proj.min(dim=1, keepdim=True).values
    c_max = all_proj.max(dim=1, keepdim=True).values
    scale = 0.5 / (c_max - c_min + 1e-8)
    proj_x_scaled = (proj_x - (c_min + c_max) / 2) * scale
    proj_y_scaled = (proj_y - (c_min + c_max) / 2) * scale

    # Step 3: Adjoint NFFT — source to frequency domain
    # w_hat[p,k] = sum_n w_n * exp(-2*pi*i*k*proj_x_scaled[p,n])
    # Shape: (P, |C|)
    phases_x = -2j * torch.pi * freq_indices[None, :, None] * proj_x_scaled[:, None, :]
    # For efficiency, use actual NFFT library (torch_nfft)
    w_hat = (w[None, None, :] * torch.exp(phases_x)).sum(dim=-1)  # (P, |C|)

    # Step 4: Multiply by Fourier coefficients
    w_hat_c = fourier_coeffs[None, :] * w_hat  # (P, |C|)

    # Step 5: NFFT — frequency to target points
    # t[p,m] = sum_k c_k * w_hat[p,k] * exp(2*pi*i*k*proj_y_scaled[p,m])
    phases_y = 2j * torch.pi * freq_indices[None, :, None] * proj_y_scaled[:, None, :]
    t = (w_hat_c[:, :, None] * torch.exp(phases_y)).sum(dim=1).real  # (P, M)

    # Step 6: Average over slices
    s = t.mean(dim=0)  # (M,)

    return s

# For GPU: use torch_nfft (github.com/dominikbuenger/torch_nfft)
# or the simple_torch_NFFT package for batched 1D NFFTs.
# The batched projection (Step 1) maps naturally to tensor core GEMMs.
```

## References

- Hertrich, J., Jahn, T., & Quellmalz, M. (2025). Fast Summation of Radial Kernels via QMC Slicing. ICLR 2025. arXiv:2410.01316.
- Hertrich, J. (2024). Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms. SIAM J. Math. Data Sci. arXiv:2401.08260.
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007.
- Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). Orthogonal Random Features. NeurIPS 2016.
- Potts, D. & Steidl, G. (2004). Fast Summation at Nonequispaced Knots by NFFTs. SIAM J. Sci. Comput.
- Keiner, J., Kunis, S., & Potts, D. (2009). Using NFFT 3 — a software library for various nonequispaced fast Fourier transforms. ACM TOMS.

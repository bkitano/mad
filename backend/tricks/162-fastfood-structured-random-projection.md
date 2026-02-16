# 162: Fastfood Structured Random Projection

**Category**: approximation
**Gain type**: efficiency
**Source**: Le, Sarlós & Smola (2013), "Fastfood — Approximating Kernel Expansions in Loglinear Time" (ICML 2013)
**Paper**: [papers/fastfood-kernel-approximation.pdf]
**Documented**: 2026-02-15

## Description

Fastfood replaces the dense Gaussian random matrix $\mathbf{Z} \in \mathbb{R}^{n \times d}$ used in Random Kitchen Sinks (Rahimi & Recht 2007) with a structured product of diagonal and Hadamard matrices: $\mathbf{V} = \frac{1}{\sigma\sqrt{d}}\mathbf{S}\mathbf{H}\mathbf{G}\boldsymbol{\Pi}\mathbf{H}\mathbf{B}$. This reduces the random projection cost from $O(nd)$ to $O(n \log d)$ per observation and storage from $O(nd)$ to $O(n)$, while preserving the unbiasedness and low variance of the kernel approximation.

The key insight is that Hadamard matrices, when combined with diagonal Gaussian matrices, produce pseudo-random vectors that are approximately spherically uniform — closely mimicking the properties of dense Gaussian random matrices. The structured matrix product uses two Hadamard transforms (applied via FWHT in $O(d \log d)$), a random permutation (to break alignment between the two Hadamard stages), a diagonal Gaussian matrix (to inject continuous randomness), and diagonal sign/scaling matrices.

Fastfood is the direct predecessor to SORF (trick 155). While SORF uses three $\mathbf{HD}$ blocks to achieve near-orthogonality, Fastfood uses two $\mathbf{H}$ stages interleaved with a permutation and Gaussian diagonal to achieve near-Gaussianity. Both achieve $O(d \log d)$ projection time, but Fastfood explicitly models the row-norm distribution via a $\chi_d$-scaling matrix $\mathbf{S}$, while SORF removes the permutation and Gaussian diagonal in favor of a simpler three-block Rademacher structure with reduced variance.

For FAVOR+/Performer-style linear attention, Fastfood provides the same projection acceleration as SORF: the random projection $\mathbf{W}\mathbf{x}$ in the feature map $\phi(\mathbf{x}) = \exp(\mathbf{W}\mathbf{x} - \|\mathbf{x}\|^2/2)/\sqrt{M}$ is reduced from $O(Md)$ to $O(M \log d)$.

## Mathematical Form

**Core Operation:**

The Fastfood matrix replacing a dense Gaussian $\mathbf{Z} \in \mathbb{R}^{d \times d}$ is:

$$
\mathbf{V} = \frac{1}{\sigma\sqrt{d}} \mathbf{S} \mathbf{H} \mathbf{G} \boldsymbol{\Pi} \mathbf{H} \mathbf{B}
$$

**Key Definitions:**

- $\mathbf{H} \in \mathbb{R}^{d \times d}$ — normalized Walsh-Hadamard matrix, defined recursively: $H_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $H_{2d} = \begin{bmatrix} H_d & H_d \\ H_d & -H_d \end{bmatrix}$
- $\mathbf{B} \in \mathbb{R}^{d \times d}$ — diagonal matrix with $B_{ii} \in \{-1, +1\}$ drawn i.i.d. uniformly (Rademacher). Acts as a randomized preconditioner: $d^{-1/2}\mathbf{H}\mathbf{B}$ densifies the input
- $\boldsymbol{\Pi} \in \{0,1\}^{d \times d}$ — random permutation matrix. Ensures rows of the two Hadamard stages are incoherent with each other
- $\mathbf{G} \in \mathbb{R}^{d \times d}$ — diagonal matrix with $G_{ii} \sim \mathcal{N}(0, 1)$ i.i.d. Injects continuous (Gaussian) randomness so each row of $\mathbf{H}\mathbf{G}\boldsymbol{\Pi}\mathbf{H}\mathbf{B}$ is marginally Gaussian
- $\mathbf{S} \in \mathbb{R}^{d \times d}$ — diagonal scaling matrix: $S_{ii} = s_i \|\mathbf{G}\|_{\text{Frob}}^{-1/2} d^{-1/2}$ where $s_i$ is drawn from the $\chi_d$ distribution to match Gaussian row-norm distribution
- $\sigma$ — kernel bandwidth parameter
- $d$ — input dimension (padded to power of 2 for Hadamard)

**Feature Map:**

For $n > d$ random features, stack $n/d$ independent blocks $\mathbf{V}_1, \ldots, \mathbf{V}_{n/d}$:

$$
\mathbf{V}^T = [\mathbf{V}_1, \mathbf{V}_2, \ldots, \mathbf{V}_{n/d}]^T
$$

The Fastfood feature map for the Gaussian RBF kernel is:

$$
\phi_j(\mathbf{x}) = n^{-1/2} \exp(i [\mathbf{V}\mathbf{x}]_j)
$$

so that $\mathbb{E}[\phi(\mathbf{x})^\top \phi(\mathbf{x}')] = e^{-\|\mathbf{x} - \mathbf{x}'\|^2 / 2\sigma^2}$.

**Unbiasedness (Lemma 7):**

Each row of $\mathbf{S}\mathbf{H}\mathbf{G}\boldsymbol{\Pi}\mathbf{H}\mathbf{B}$ is Gaussian with distribution $\mathcal{N}(0, \sigma^{-2} I_d)$:

$$
\mathbb{E}_{\mathbf{S},\mathbf{G},\mathbf{B},\boldsymbol{\Pi}}\left[\phi(\mathbf{x})^\top \phi(\mathbf{x}')\right] = e^{-\|\mathbf{x} - \mathbf{x}'\|^2 / 2\sigma^2}
$$

**Variance Bound (Theorem 9):**

For a single $d \times d$ block with $v = (\mathbf{x} - \mathbf{x}')/\sigma$:

$$
\text{Var}\left[\psi_j(v)\right] = \frac{1}{2}\left(1 - e^{-\|v\|^2}\right)^2
$$

$$
\text{Var}\left[\sum_{j=1}^{d} \psi_j(v)\right] \leq \frac{d}{2}\left(1 - e^{-\|v\|^2}\right)^2 + dC(\|v\|)
$$

where $C(\alpha) = 6\alpha^4\left[e^{-\alpha^2} + \frac{\alpha^2}{3}\right]$ captures inter-feature correlation. For $n/d$ stacked blocks, the variance scales as $O(2/n)$.

**Row Properties:**

1. All rows of $\mathbf{H}\mathbf{G}\boldsymbol{\Pi}\mathbf{H}\mathbf{B}$ have identical length: $l^2 = \|G\|_{\text{Frob}}^2 \cdot d$
2. Any given row is marginally i.i.d. Gaussian: $[\mathbf{H}\mathbf{G}\boldsymbol{\Pi}\mathbf{H}\mathbf{B}]_{ij} \sim \mathcal{N}(0, d)$
3. After $\mathbf{S}$ rescaling, rows have $\chi_d$-distributed norms matching true Gaussian row norms

**Concentration (Theorem 11):**

$$
\mathbb{P}\left[\left|\hat{k}(\mathbf{x}, \mathbf{x}') - k(\mathbf{x}, \mathbf{x}')\right| \geq 2\sigma^{-1} d^{-1/2} \|\mathbf{x} - \mathbf{x}'\| \sqrt{\log(2/\delta)\log(2d/\delta)}\right] \leq 2\delta
$$

This is only logarithmically weaker than the $O(m^{-1/2})$ concentration of Random Kitchen Sinks.

## Complexity

| Operation | Random Kitchen Sinks | Fastfood |
|-----------|---------------------|----------|
| Projection $\mathbf{Z}\mathbf{x}$ (per obs.) | $O(nd)$ | $O(n \log d)$ |
| Storage of $\mathbf{Z}$ | $O(nd)$ | $O(n)$ |
| Feature map generation | $O(nd)$ | $O(n)$ (diag. sampling) |
| Training cost | $O(m^\beta n \rho d)$ | $O(m^\beta n \log d)$ |
| Test-time prediction | $O(n \rho d)$ | $O(n \log d)$ |
| Test-time RAM | $O(nd)$ | $O(n)$ |

**Empirical speedups** (from paper, CPU, Spiral library):
- $d = 1024, n = 16384$: **24×** faster, **256×** less RAM
- $d = 4096, n = 32768$: **89×** faster, **1024×** less RAM
- $d = 8192, n = 65536$: **199×** faster, **2048×** less RAM

**Memory:** $O(n)$ total — only $3d$ diagonal entries ($\mathbf{S}, \mathbf{G}, \mathbf{B}$) plus a permutation lookup table $\boldsymbol{\Pi}$ per block. The Hadamard matrix is never stored.

## Applicability

- **Direct acceleration of FAVOR+/Performer**: Same role as SORF (trick 155) — accelerates the random projection $\mathbf{W}\mathbf{x}$ from $O(Md)$ to $O(M \log d)$ per token. Fastfood's explicit $\chi_d$-scaling may provide slightly better kernel approximation quality than SORF's simpler structure at the cost of one extra random permutation
- **Any random-feature kernel approximation**: Gaussian RBF, Matérn, polynomial (via spherical harmonics), and any dot-product kernel $k(\langle x, x'\rangle)$
- **Kernel spectrum control**: The diagonal scaling matrix $\mathbf{S}$ can be tuned to change the kernel spectrum without changing the structured projection, enabling Matérn and other non-RBF kernels at no extra projection cost
- **GPU considerations**: The projection involves two FWHT passes, one random permutation (gather operation), and three elementwise diagonal multiplications. The permutation $\boldsymbol{\Pi}$ is a gather/scatter operation that breaks memory coalescing — this is the main GPU efficiency concern relative to SORF which avoids permutations entirely. On modern GPUs, the gather for $\boldsymbol{\Pi}$ through shared memory is feasible for $d \leq 8192$ but adds latency vs. pure FWHT+diagonal
- **Tensor core caveat**: Like SORF, replaces a single GEMM with structured operations. For small $d$ (64–128), the dense GEMM underutilizes tensor cores anyway, so Fastfood may be competitive. For large $d$ or large batch, dense GEMM on tensor cores may win

## Limitations

- **Random permutation breaks coalescing**: The permutation $\boldsymbol{\Pi}$ between the two Hadamard stages is a gather operation that can break memory coalescing on GPUs. SORF (trick 155) avoids this by using three $\mathbf{HD}$ blocks instead of two $\mathbf{H}$ stages with a permutation
- **Requires power-of-2 dimension**: Like SORF, the Walsh-Hadamard transform requires $d = 2^k$
- **$\chi_d$ sampling overhead**: The scaling matrix $\mathbf{S}$ requires sampling from the $\chi_d$ distribution, which is more expensive than Rademacher sampling (though it's a one-time cost)
- **Correlated rows within a block**: Unlike i.i.d. Random Kitchen Sinks, the $d$ rows within a single Fastfood block are correlated (they share the same $\mathbf{G}, \boldsymbol{\Pi}, \mathbf{B}$). This correlation is mild and controlled by the permutation, but means the effective variance reduction per feature is slightly less than for truly independent features
- **Not exactly orthogonal**: Unlike SORF/ORF which enforce row orthogonality, Fastfood rows are only approximately decorrelated. The variance reduction from orthogonality (which SORF explicitly provides) is absent
- **Two sequential FWHT + permutation**: Three sequential barriers (FWHT, permutation, FWHT) vs. SORF's three FWHT barriers. The permutation is cheaper than FWHT but has worse memory access patterns

## Implementation Notes

```python
import torch
import math

def fastfood_projection(x, S, G, Pi, B, sigma=1.0):
    """Apply Fastfood projection: (1/sigma*sqrt(d)) * S H G Pi H B x.

    Args:
        x: (..., d) input tensor
        S: (d,) chi_d scaling diagonal
        G: (d,) Gaussian diagonal entries
        Pi: (d,) permutation indices (integer tensor)
        B: (d,) Rademacher diagonal (+/-1)
        sigma: kernel bandwidth
    Returns:
        (..., d) projected tensor
    """
    d = x.shape[-1]

    # Step 1: Apply B (Rademacher sign flip) + first Hadamard
    y = x * B
    y = fast_walsh_hadamard_transform(y)  # O(d log d)

    # Step 2: Apply permutation Pi (gather — this is the GPU-unfriendly step)
    y = y[..., Pi]

    # Step 3: Apply Gaussian diagonal G
    y = y * G

    # Step 4: Second Hadamard
    y = fast_walsh_hadamard_transform(y)  # O(d log d)

    # Step 5: Apply scaling S and normalization
    y = y * S / (sigma * math.sqrt(d))

    return y

def fast_walsh_hadamard_transform(x):
    """In-place FWHT along last dimension."""
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a, b = x[..., 0, :].clone(), x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        h *= 2
    return x.view(*x.shape[:-3], d) / math.sqrt(d)

def init_fastfood_params(d, device='cuda'):
    """Initialize Fastfood parameters for one d×d block."""
    B = torch.randint(0, 2, (d,), device=device).float() * 2 - 1
    Pi = torch.randperm(d, device=device)
    G = torch.randn(d, device=device)

    # S: chi_d-distributed scaling
    # chi_d = norm of d-dimensional Gaussian vector
    G_frob = G.norm()
    s = torch.distributions.Chi2(torch.tensor(float(d))).sample((d,)).sqrt().to(device)
    S = s / (G_frob * math.sqrt(d))

    return S, G, Pi, B

# Comparison: Fastfood vs SORF vs Dense
# Dense RKS:  Z @ x          -> O(nd) FLOPS, O(nd) storage
# Fastfood:   SHGΠHB @ x     -> O(n log d) FLOPS, O(n) storage
# SORF:       HD₁HD₂HD₃ @ x  -> O(n log d) FLOPS, O(n) storage
#
# Key difference: Fastfood uses permutation Π (gather, bad for GPU)
#                 SORF uses three HD blocks (all FWHT+diag, GPU-friendly)
#                 Fastfood has explicit χ_d norm scaling (better kernel approx)
#                 SORF has orthogonal rows (lower variance)
```

## References

- Le, Q. V., Sarlós, T., & Smola, A. J. (2013). Fastfood — Approximating Kernel Expansions in Loglinear Time. ICML 2013. arXiv:1408.3060.
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007.
- Ailon, N. & Chazelle, B. (2009). The Fast Johnson-Lindenstrauss Transform and Approximate Nearest Neighbors. SIAM Journal on Computing, 39(1).
- Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). Orthogonal Random Features. NeurIPS 2016.
- Choromanski, K., Likhosherstov, V., et al. (2021). Rethinking Attention with Performers. ICLR 2021.

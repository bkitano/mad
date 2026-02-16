# 155: Structured Orthogonal Random Features (SORF)

**Category**: approximation
**Gain type**: efficiency
**Source**: Yu, Suresh, Choromanski, Holtmann-Rice & Kumar (2016), "Orthogonal Random Features" (NeurIPS 2016)
**Paper**: [papers/structured-orthogonal-random-features.pdf]
**Documented**: 2026-02-15

## Description

Structured Orthogonal Random Features (SORF) accelerates the random projection step in kernel approximation (and by extension, in FAVOR+/Performer-style linear attention) by replacing the dense random orthogonal matrix $\mathbf{Q}$ with a structured product of Walsh-Hadamard matrices and diagonal sign-flip matrices: $\mathbf{H}\mathbf{D}_1\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_3$. This reduces the projection cost from $O(Dd)$ to $O(D \log d)$ per token, where $D$ is the number of random features and $d$ is the input dimension, while achieving nearly identical kernel approximation quality to dense orthogonal random features (ORF).

The key insight is twofold. First, **orthogonal random features** (ORF) — where the projection matrix rows are exactly orthogonal — provably reduce kernel approximation variance compared to i.i.d. random Fourier features (RFF). Second, the orthogonal matrix can be replaced by a **structured** one (the $\mathbf{HD}_1\mathbf{HD}_2\mathbf{HD}_3$ product) that is "near-orthogonal" and can be applied via the Fast Walsh-Hadamard Transform (FWHT) in $O(d \log d)$ time. The FWHT is an in-place, cache-friendly operation well-suited to GPUs.

For FAVOR+/Performer attention, the bottleneck in the feature map computation $\phi(\mathbf{x}) = \exp(\mathbf{W}\mathbf{x} - \|\mathbf{x}\|^2/2)/\sqrt{M}$ is the matrix-vector product $\mathbf{W}\mathbf{x}$, which SORF replaces with three sequential Hadamard-diagonal products. This is directly applicable as an acceleration of the random projection step in any random-feature-based linear attention method (FAVOR+, FAVOR++, FAVOR#, RFA).

## Mathematical Form

**Core Operation:**

The SORF transformation matrix replaces the dense random orthogonal projection:

$$
\mathbf{W}_{\text{SORF}} = \frac{\sqrt{d}}{\sigma}\mathbf{H}\mathbf{D}_1\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_3
$$

where:
- $\mathbf{H} \in \mathbb{R}^{d \times d}$ is the normalized Walsh-Hadamard matrix ($H_{ij} = d^{-1/2}(-1)^{\langle i,j \rangle}$)
- $\mathbf{D}_i \in \mathbb{R}^{d \times d}$, $i = 1, 2, 3$, are diagonal "sign-flipping" matrices with diagonal entries sampled i.i.d. from the Rademacher distribution ($\pm 1$ with equal probability)
- $\sigma$ is the kernel bandwidth parameter

**Key Definitions:**

- $d$ — input dimension (must be padded to power of 2 for Hadamard)
- $D$ — number of random features ($D \geq d$ typically; for $D > d$, use multiple independent blocks)
- $\mathbf{Q} \in \mathbb{R}^{d \times d}$ — uniformly distributed random orthogonal matrix (in ORF)
- $\mathbf{S} \in \mathbb{R}^{d \times d}$ — diagonal matrix with entries from $\chi_d$ distribution (to match Gaussian row norms)

**ORF Construction (baseline):**

The dense Orthogonal Random Features matrix is:

$$
\mathbf{W}_{\text{ORF}} = \frac{1}{\sigma}\mathbf{S}\mathbf{Q}
$$

where $\mathbf{Q}$ is a uniformly random orthogonal matrix (from QR decomposition of a Gaussian matrix) and $\mathbf{S}$ rescales rows to have $\chi_d$-distributed norms.

**ORF Variance Reduction (Theorem 1):**

ORF is unbiased: $\mathbb{E}[K_{\text{ORF}}(\mathbf{x}, \mathbf{y})] = e^{-\|\mathbf{x}-\mathbf{y}\|^2/2\sigma^2}$, and its variance satisfies:

$$
\text{Var}(K_{\text{ORF}}(\mathbf{x}, \mathbf{y})) \leq \frac{1}{2D}\left(\left(1 - e^{-z^2}\right)^2 - \frac{D-1}{d}e^{-z^2}z^4\right) + \frac{f(z)}{d^2}
$$

where $z = \|\mathbf{x} - \mathbf{y}\|/\sigma$ and $f$ is a function bounded for all $z$. The second term is strictly negative for $z > 0$, giving variance strictly less than RFF.

**Variance ratio (large $d$, $D = d$):**

$$
\frac{\text{Var}(K_{\text{ORF}})}{\text{Var}(K_{\text{RFF}})} \approx 1 - \frac{(D-1)e^{-z^2}z^4}{d(1 - e^{-z^2})^2}
$$

This ratio is always $< 1$, meaning ORF always improves over RFF.

**SORF Bias Bound (Theorem 3):**

SORF introduces a small bias (unlike ORF which is exactly unbiased):

$$
\left|\mathbb{E}[K_{\text{SORF}}(\mathbf{x}, \mathbf{y})] - e^{-z^2/2}\right| \leq \frac{6z}{\sqrt{d}}
$$

This bias vanishes as $O(1/\sqrt{d})$, and is negligible for typical Transformer head dimensions ($d = 64$ or $128$).

**Near-Orthogonality (Theorem 4):**

The SORF matrix $\sqrt{d}\mathbf{H}\mathbf{D}_1\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_3$ can be decomposed as $\bar{\mathbf{R}}\bar{\mathbf{g}}$ where rows of $\bar{\mathbf{R}}$ have norm $\|\mathbf{z}\|_2$ and for any $t \geq 1/d$, the inner product between any two rows is at most $t\|\mathbf{z}\|_2$ with probability $1 - de^{-ct^{2/3}d^{1/3}}$.

## Complexity

| Operation | RFF (dense) | ORF (dense) | SORF |
|-----------|------------|------------|------|
| Projection $\mathbf{W}\mathbf{x}$ | $O(Dd)$ | $O(Dd)$ | $O(D \log d)$ |
| Storage of $\mathbf{W}$ | $O(Dd)$ | $O(Dd)$ | $O(D)$ or $O(1)$ in-place |
| Generating $\mathbf{W}$ | $O(Dd)$ | $O(d^3)$ (QR) | $O(D)$ (Rademacher samples) |
| Kernel MSE | baseline | lower | ≈ same as ORF |

**Speedup factor:** $O(d / \log d)$ over RFF/ORF for the projection step. For $d = 64$, this is roughly $10\times$; for $d = 128$, roughly $18\times$.

**Per-token feature map cost in FAVOR+:**
- Dense: $O(Md)$ where $M$ is number of random features
- With SORF: $O(M \log d)$

**Memory:** SORF requires only $O(D)$ random bits (for the three diagonal matrices), vs $O(Dd)$ for a dense random matrix. The Walsh-Hadamard transform is applied in-place with $O(1)$ extra memory.

## Applicability

- **Direct acceleration of FAVOR+/FAVOR++/FAVOR#**: The random projection $\mathbf{W}\mathbf{x}$ is the main cost in computing positive random feature maps $\phi(\mathbf{x}) = \exp(\mathbf{W}\mathbf{x} - \|\mathbf{x}\|^2/2)/\sqrt{M}$. SORF reduces this from $O(Md)$ to $O(M\log d)$ per token
- **Any random-feature-based kernel approximation**: Gaussian kernels, softmax kernels, arc-cosine kernels
- **GPU-friendly**: The Walsh-Hadamard transform is a butterfly-like operation with regular memory access patterns, maps well to both shared memory and warp-level operations. FWHT implementations exist for CUDA (e.g., in the `hadamard_transform` package). Three sequential FWHT + diagonal sign-flip is amenable to kernel fusion
- **Tensor core caveat**: SORF replaces a single GEMM (which maps perfectly to tensor cores) with 3 sequential FWHT passes. For small $d$ (64–128, typical in attention heads), the GEMM is small enough that tensor core utilization may be low anyway, making SORF competitive. For large $d$ or batched projections, dense GEMM on tensor cores may still win
- **Combines with all FAVOR variants**: SORF is orthogonal to the choice of feature map (positive, OPRF, GERF, SDERF) — it only accelerates the projection step

## Limitations

- **Requires power-of-2 dimension**: The Walsh-Hadamard transform requires $d = 2^k$. Non-power-of-2 dimensions must be zero-padded, wasting some computation
- **Slight bias**: Unlike dense ORF, SORF is not exactly unbiased. The bias is $O(z/\sqrt{d})$ which is small for $d \geq 32$ but not zero
- **Three sequential passes**: Each $\mathbf{H}\mathbf{D}_i$ is a sequential barrier. For very small $d$, the launch overhead of three passes may not beat a single small GEMM
- **GPU efficiency question**: On modern GPUs (A100/H100), a single $64 \times 64$ or $128 \times 128$ GEMM using tensor cores may be faster than 3 FWHT passes, even though SORF has asymptotically fewer FLOPs. The crossover depends on batch size and $d$. For $d \geq 256$, SORF is likely faster; for $d = 64$, empirical testing is needed
- **Reduces to 2 blocks (HDHD) with similar quality**: The paper shows that using only 2 Hadamard-diagonal blocks instead of 3 provides similar kernel approximation quality, saving one FWHT pass
- **Not beneficial for very small $M$**: If the number of random features $M$ is very small (e.g., $M = 8$), the projection cost is already negligible and SORF provides no practical speedup

## Implementation Notes

```python
import torch
import math

def fast_walsh_hadamard_transform(x):
    """In-place Fast Walsh-Hadamard Transform.

    Args:
        x: (..., d) tensor where d must be a power of 2
    Returns:
        (..., d) tensor with WHT applied along last dimension
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, "d must be a power of 2"

    h = 1
    while h < d:
        # Butterfly operation
        x_even = x[..., 0::2*h, None]  # Not quite right for in-place
        # More practical: reshape-based implementation
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]  # even indices
        b = x[..., 1, :]  # odd indices
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        h *= 2

    return x.view(*x.shape[:-3], d) / math.sqrt(d)

def sorf_projection(x, D1, D2, D3):
    """Apply SORF projection: sqrt(d) * H D1 H D2 H D3 x.

    Args:
        x: (..., d) input tensor
        D1, D2, D3: (d,) diagonal sign-flip vectors (+/-1)
    Returns:
        (..., d) projected tensor
    """
    d = x.shape[-1]

    # Apply D3, then H, then D2, then H, then D1, then H
    y = x * D3                          # (*, d) elementwise
    y = fast_walsh_hadamard_transform(y)  # O(d log d)
    y = y * D2                          # (*, d) elementwise
    y = fast_walsh_hadamard_transform(y)  # O(d log d)
    y = y * D1                          # (*, d) elementwise
    y = fast_walsh_hadamard_transform(y)  # O(d log d)

    return y * math.sqrt(d)

def favor_plus_with_sorf(Q, K, V, M=None):
    """FAVOR+ attention using SORF for fast random projection.

    Replaces the O(Md) dense projection with O(M log d) SORF.
    """
    L, d = Q.shape
    if M is None:
        M = d

    # Pad d to next power of 2 if needed
    d_pad = 1 << (d - 1).bit_length()
    if d_pad != d:
        Q = torch.nn.functional.pad(Q, (0, d_pad - d))
        K = torch.nn.functional.pad(K, (0, d_pad - d))

    # Generate multiple SORF blocks if M > d_pad
    n_blocks = (M + d_pad - 1) // d_pad
    features_q, features_k = [], []

    for _ in range(n_blocks):
        # Random sign-flip diagonals (only 3*d_pad random bits needed!)
        D1 = torch.randint(0, 2, (d_pad,), device=Q.device) * 2 - 1.0
        D2 = torch.randint(0, 2, (d_pad,), device=Q.device) * 2 - 1.0
        D3 = torch.randint(0, 2, (d_pad,), device=Q.device) * 2 - 1.0

        # SORF projection: O(L * d_pad * log(d_pad)) instead of O(L * d_pad^2)
        proj_q = sorf_projection(Q, D1, D2, D3)  # (L, d_pad)
        proj_k = sorf_projection(K, D1, D2, D3)  # (L, d_pad)

        # Positive feature map (same as FAVOR+)
        q_norm_sq = (Q ** 2).sum(dim=-1, keepdim=True)
        k_norm_sq = (K ** 2).sum(dim=-1, keepdim=True)

        features_q.append(torch.exp(proj_q - q_norm_sq / 2))
        features_k.append(torch.exp(proj_k - k_norm_sq / 2))

    Q_prime = torch.cat(features_q, dim=-1)[:, :M] / math.sqrt(M)
    K_prime = torch.cat(features_k, dim=-1)[:, :M] / math.sqrt(M)

    # Standard linear attention
    KV = K_prime.T @ V
    K_sum = K_prime.sum(dim=0)
    num = Q_prime @ KV
    den = Q_prime @ K_sum
    return num / den.unsqueeze(-1)
```

## References

- Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). Orthogonal Random Features. NeurIPS 2016. arXiv:1610.09072.
- Choromanski, K., Likhosherstov, V., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Ailon, N. & Chazelle, B. (2006). Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. STOC 2006.
- Le, Q., Sarlós, T., & Smola, A. (2013). Fastfood — approximating kernel expansions in loglinear time. ICML 2013.
- Fino, B. J. & Algazi, V. R. (1976). Unified matrix treatment of the fast Walsh-Hadamard transform. IEEE Transactions on Computers.

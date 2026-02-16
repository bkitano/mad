# 038: Doubly Block Circulant SVD for Convolutional Layers

**Category**: decomposition
**Gain type**: efficiency
**Source**: Sedghi, Gupta & Long "The Singular Values of Convolutional Layers" (ICLR 2019)
**Paper**: [papers/doubly-block-circulant-svd-conv.pdf]
**Documented**: 2026-02-15

## Description

The linear transformation computed by a 2D multi-channel convolutional layer (with circular/wrap-around padding) is represented by a matrix $M$ that has **three levels of circulant structure**: it is a block matrix of doubly block circulant sub-matrices. This hierarchical circulant structure means the entire transformation can be **diagonalized by Fourier transforms**, enabling exact computation of all singular values in $O(n^2 m^2(m + \log n))$ time — orders of magnitude faster than the brute-force $O(n^6 m^3)$ SVD of the full matrix.

The key insight is:
1. **Single-channel case**: The conv layer's matrix is **doubly block circulant**, and its eigenvectors are columns of $Q = \frac{1}{n}(F \otimes F)$ where $F$ is the DFT matrix. Thus the eigenvalues (and hence singular values, since the matrix is normal) are exactly the entries of $F^T K F$, the **2D Fourier transform of the kernel**.
2. **Multi-channel case**: The full matrix $M$ is a block matrix of $m \times m$ doubly block circulant blocks $B_{cd}$. Each $B_{cd}$ can be diagonalized by $Q$, yielding diagonal matrices $D_{cd}$. The singular values of $M$ equal those of a block-diagonal matrix $L$ whose $n^2$ diagonal blocks are $m \times m$ matrices $P^{(u,v)} = (F^T K_{:,:,c,d} F)_{uv}$ formed by picking the $(u,v)$-th Fourier coefficient from each channel pair. Thus $\sigma(M) = \bigcup_{u,v \in [n]} \sigma(P^{(u,v)})$.

This reduces computing all $mn^2$ singular values to: (a) $m^2$ 2D FFTs of size $n \times n$, then (b) $n^2$ SVDs of $m \times m$ matrices. The FFTs and SVDs are all independent and fully parallelizable on a GPU.

The practical payoff is **exact spectral normalization** — projecting convolutional layers onto operator-norm balls by clipping singular values — which improves generalization (CIFAR-10 error 6.2% → 5.3% on ResNet-32) and provides Lipschitz guarantees for robustness.

## Mathematical Form

**Single-Channel Doubly Block Circulant Matrix:**

For a kernel $K \in \mathbb{R}^{n \times n}$ (zero-padded from $k \times k$), the conv layer's linear transformation on $n \times n$ feature maps with circular padding is:

$$
A = \begin{bmatrix} \text{circ}(K_{0,:}) & \text{circ}(K_{1,:}) & \cdots & \text{circ}(K_{n-1,:}) \\ \text{circ}(K_{n-1,:}) & \text{circ}(K_{0,:}) & \cdots & \text{circ}(K_{n-2,:}) \\ \vdots & \vdots & \ddots & \vdots \\ \text{circ}(K_{1,:}) & \text{circ}(K_{2,:}) & \cdots & \text{circ}(K_{0,:}) \end{bmatrix}
$$

where each $\text{circ}(K_{i,:})$ is an $n \times n$ circulant matrix built from row $i$ of $K$.

**Eigenvalue Characterization (Theorem 5):**

Let $F$ be the $n \times n$ DFT matrix with $F_{ij} = \omega^{ij}$, $\omega = e^{2\pi i / n}$. Let $Q = \frac{1}{n}(F \otimes F)$.

$$
\text{Eigenvalues of } A = \left\{ (F^T K F)_{u,v} \;\middle|\; u, v \in [n] \right\}
$$

Since $A$ is normal ($A^T A = A A^T$), the singular values are the magnitudes of eigenvalues:

$$
\sigma(A) = \left\{ \left| (F^T K F)_{u,v} \right| \;\middle|\; u, v \in [n] \right\}
$$

Note that $F^T K F$ is exactly the **2D DFT of the kernel** $K$.

**Multi-Channel Case (Theorem 6):**

For kernel tensor $K \in \mathbb{R}^{n \times n \times m \times m}$ with $m$ input and $m$ output channels, the full transformation matrix is:

$$
M = \begin{bmatrix} B_{00} & B_{01} & \cdots & B_{0(m-1)} \\ B_{10} & B_{11} & \cdots & B_{1(m-1)} \\ \vdots & \vdots & \ddots & \vdots \\ B_{(m-1)0} & B_{(m-1)1} & \cdots & B_{(m-1)(m-1)} \end{bmatrix}
$$

where each $B_{cd}$ is the $n^2 \times n^2$ doubly block circulant matrix for input channel $d$ to output channel $c$.

For each spatial frequency $(u, v) \in [n] \times [n]$, define the $m \times m$ matrix:

$$
P^{(u,v)}_{cd} = (F^T K_{:,:,c,d} F)_{uv}
$$

Then:

$$
\sigma(M) = \bigcup_{u \in [n], v \in [n]} \sigma\left(P^{(u,v)}\right)
$$

The total set of $mn^2$ singular values is the union of the singular values of $n^2$ separate $m \times m$ matrices, each formed by picking the $(u,v)$-th 2D Fourier coefficient from all $m^2$ channel pairs.

**Operator Norm Projection (Proposition 9):**

Given SVD $A = UDV^\top$, define $\tilde{D}_{ii} = \min(D_{ii}, c)$ for clipping threshold $c$. Then $\tilde{A} = U\tilde{D}V^\top$ is the Frobenius-norm projection onto $\{X \mid \|X\|_2 \le c\}$.

In practice, this is implemented as:

$$
\tilde{K} = \text{IFFT}_{2D}\left( U \cdot \min(D, c) \cdot V^\top \right)
$$

where the FFT/IFFT operate on the spatial dimensions and the SVD operates on the $m \times m$ channel matrices.

**Key Definitions:**

- $n \times n$ — spatial dimension of the feature map
- $k \times k$ — kernel size (padded to $n \times n$)
- $m$ — number of input/output channels
- $F \in \mathbb{C}^{n \times n}$ — DFT matrix, $F_{ij} = \omega^{ij}$
- $Q = \frac{1}{n}(F \otimes F)$ — eigenvector matrix for doubly block circulant matrices
- $P^{(u,v)} \in \mathbb{C}^{m \times m}$ — channel mixing matrix at spatial frequency $(u,v)$
- $\sigma(\cdot)$ — multiset of singular values

## Complexity

| Operation | Brute Force (full matrix SVD) | Reshape $K$ heuristic | **This Method** |
|-----------|-------|------------|----------------|
| Compute all singular values | $O((n^2 m)^3) = O(n^6 m^3)$ | $O(m^3 k^2)$ (approximate only) | $O(n^2 m^2 (m + \log n))$ |
| Compute operator norm only | $O(n^2 m^2 k^2)$ (power iteration) | $O(m^3 k^2)$ (approximate) | $O(n^2 m^2 (m + \log n))$ (exact) |
| Singular value clip + project | Not feasible | Approximate | $O(m^2 n^2 \log n + n^2 m^3)$ |

**Breakdown of $O(n^2 m^2(m + \log n))$:**
- $m^2$ independent 2D FFTs of size $n \times n$: $O(m^2 n^2 \log n)$
- $n^2$ independent SVDs of $m \times m$ matrices: $O(n^2 m^3)$
- Both stages are fully parallelizable on GPU

**Memory:** $O(m^2 n^2)$ for the Fourier-domain channel matrices (same as storing the kernel).

**Practical timing (from paper):**
- 3×3 kernel, 16×16 feature map, 1000 channels: full matrix method ~1000s, this method ~2s (TensorFlow on GPU)
- 11×11 kernel, 64×64 feature map, 250 channels: full matrix method infeasible, this method ~100s (NumPy), ~10s (TensorFlow on GPU)
- Operator norm clipping every 100 steps: 25% faster than the reshape $K$ heuristic on GPU

## Applicability

- **Spectral normalization / Lipschitz regularization**: Exact projection of convolutional layers onto operator-norm balls. Improves CIFAR-10 test error from 6.2% to 5.3% on ResNet-32 with batch normalization. Complements (does not replace) batch normalization
- **Adversarial robustness**: Bounding the Lipschitz constant of each layer provides certified robustness guarantees via product of per-layer operator norms
- **Generalization bounds**: Spectrally-normalized margin bounds (Bartlett et al. 2017) require the operator norm of each layer — this provides the exact value rather than an upper bound
- **Neural network analysis**: Plotting singular value spectra of convolutional layers reveals effective rank, conditioning, and how information flows through the network (demonstrated on ResNet-v2)
- **Training stability**: Preventing singular values from growing too large prevents gradient explosion in deep networks; clipping to a ball is a principled alternative to gradient clipping
- **State-space models with circular convolutions**: Any model that uses circular convolution kernels (including Toeplitz-structured SSMs with circulant embedding) benefits from this exact spectral characterization

## Limitations

- **Circular padding only**: The exact eigenvalue characterization assumes wrap-around (circular) padding. For zero-padded or no-padding convolutions, the matrix is Toeplitz rather than circulant, and the eigenvalue structure is only approximate. Recent work (Araujo et al. 2024) addresses this gap
- **Square feature maps assumed**: The analysis assumes $n \times n$ spatial dimensions; rectangular feature maps require straightforward extension with different DFT sizes per dimension
- **$m \times m$ SVDs still cubic in $m$**: For very large channel counts ($m > 1000$), the $n^2$ SVDs of $m \times m$ matrices dominate. However, they are fully independent and parallelizable
- **Projection changes kernel support**: Clipping singular values and projecting back via IFFT may produce a kernel with support larger than the original $k \times k$ region (up to $n \times n$). The paper suggests re-projecting onto the $k \times k$ support set, but alternating these two projections is needed for exactness
- **Not a training speedup**: The method speeds up spectral analysis and regularization, not the forward/backward convolution itself

## Implementation Notes

```python
import numpy as np
import torch

def singular_values_conv(kernel, input_shape):
    """Compute all singular values of a convolutional layer.

    Uses the doubly block circulant structure: the singular values
    are the union of SVDs of n^2 small m×m matrices formed by
    2D-FFT of each channel pair.

    Args:
        kernel: (k_h, k_w, m_in, m_out) or (k_h, k_w, m, m) tensor
        input_shape: (n_h, n_w) spatial size of input feature map

    Returns:
        1D array of all singular values
    """
    # Step 1: 2D FFT of each channel's kernel (padded to input size)
    # transforms shape: (n_h, n_w, m_in, m_out) complex
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])

    # Step 2: For each spatial frequency (u,v), form m×m matrix
    # and compute its SVD. This is vectorized by np.linalg.svd.
    # transforms[u,v,:,:] is the m_in × m_out matrix P^{(u,v)}
    return np.linalg.svd(transforms, compute_uv=False)


def clip_operator_norm(kernel, input_shape, clip_to):
    """Project a convolutional kernel onto the operator-norm ball.

    Clips all singular values to at most clip_to, then reconstructs
    the kernel via inverse 2D FFT.

    Args:
        kernel: (k_h, k_w, m, m) kernel tensor
        input_shape: (n_h, n_w) spatial size
        clip_to: maximum allowed operator norm

    Returns:
        Clipped kernel of same shape
    """
    # Step 1: 2D FFT of kernel
    transform_coefficients = np.fft.fft2(kernel, input_shape, axes=[0, 1])

    # Step 2: SVD of each (u,v) channel matrix
    U, D, V = np.linalg.svd(
        transform_coefficients, compute_uv=True, full_matrices=False
    )

    # Step 3: Clip singular values
    D_clipped = np.minimum(D, clip_to)

    # Step 4: Reconstruct clipped Fourier coefficients
    if kernel.shape[2] > kernel.shape[3]:
        clipped = np.matmul(U * D_clipped[..., None, :], V)
    else:
        clipped = np.matmul(U * D_clipped[..., None], V)

    # Step 5: Inverse 2D FFT back to spatial domain
    clipped_kernel = np.fft.ifft2(clipped, axes=[0, 1]).real

    # Step 6: Truncate to original kernel support
    return clipped_kernel[np.ix_(
        *[range(d) for d in kernel.shape]
    )]


# PyTorch version for use during training
class SpectrallyNormalizedConv2d(torch.nn.Module):
    """Conv2d with periodic operator-norm projection.

    Every `project_every` steps, clips the layer's singular values
    to at most `max_norm`, using the doubly block circulant SVD.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 input_size, max_norm=1.0, project_every=100):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, padding_mode='circular'
        )
        self.input_size = input_size
        self.max_norm = max_norm
        self.project_every = project_every
        self._step = 0

    def forward(self, x):
        self._step += 1
        if self._step % self.project_every == 0:
            self._project()
        return self.conv(x)

    @torch.no_grad()
    def _project(self):
        # Get kernel: (out_ch, in_ch, kh, kw)
        w = self.conv.weight.data
        kh, kw = w.shape[2], w.shape[3]

        # Reshape to (kh, kw, in_ch, out_ch) for FFT
        w_np = w.permute(2, 3, 1, 0).cpu().numpy()

        # Clip and reconstruct
        w_clipped = clip_operator_norm(
            w_np, self.input_size, self.max_norm
        )

        # Write back
        w_clipped_t = torch.from_numpy(w_clipped).permute(3, 2, 0, 1)
        self.conv.weight.data.copy_(w_clipped_t.to(w.device))
```

## References

- Sedghi, H., Gupta, V. & Long, P.M. "The Singular Values of Convolutional Layers" ICLR 2019. arXiv:1805.10408
- Chao, C. "A note on block circulant matrices" Kyungpook Mathematical Journal, 14:97–100, 1974
- Jain, A.K. "Fundamentals of Digital Image Processing" Prentice Hall, 1989 (Section 5.5)
- Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. "Spectral Normalization for Generative Adversarial Networks" ICLR 2018
- Bartlett, P.L., Foster, D.J. & Telgarsky, M.J. "Spectrally-normalized margin bounds for neural networks" NeurIPS 2017
- Araujo, A. et al. "Spectral Norm of Convolutional Layers with Circular and Zero Paddings" arXiv:2402.00240, 2024
- Bibi, A., Ghanem, B., Koltun, V. & Ranftl, R. "Deep Layers as Stochastic Solvers" ICLR 2019

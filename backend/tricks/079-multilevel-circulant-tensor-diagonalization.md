# 079: Multilevel Circulant Tensor Diagonalization

**Category**: decomposition
**Gain type**: efficiency
**Source**: Rezghi & Eldén "Diagonalization of Tensors with Circulant Structure" (Linear Algebra and its Applications, 2011)
**Paper**: [papers/multilevel-circulant-kronecker-diag.pdf]
**Documented**: 2026-02-15

## Description

A tensor (multi-dimensional array) that is **circulant with respect to pairs of modes** can be diagonalized in those modes by applying Fourier matrices. This generalizes the classical result that a circulant matrix $C = F^* \Lambda F$ to arbitrary-order tensors: if a tensor $\mathcal{A}$ is $\{l, k\}$-circulant (i.e., its slices along modes $l$ and $k$ are cyclic shifts of each other), then applying the DFT matrix $F$ in mode $l$ and $F^*$ in mode $k$ produces an $\{l, k\}$-diagonal tensor.

The power of this result is **composability**: if a tensor is circulant in multiple disjoint pairs of modes (e.g., $\{1,3\}$ and $\{2,4\}$), then applying Fourier transforms in all circulant modes simultaneously diagonalizes all pairs at once. This yields the **multilevel circulant diagonalization**: a tensor that is $\{i, i+N\}$-circulant for $i = 1, \ldots, N$ (the structure arising in $N$-dimensional convolution with periodic boundary conditions) can be fully diagonalized by $2N$ Fourier transforms, reducing the tensor-tensor linear system $\mathcal{Y} = \langle \mathcal{A}, \mathcal{X} \rangle$ to an element-wise division $\bar{\mathcal{X}} = \bar{\mathcal{Y}} \mathbin{./} \mathcal{D}$ in the Fourier domain.

The key insight for neural networks is: **any $N$-dimensional circular convolution, when expressed as a tensor equation, has multilevel circulant structure and can be solved/applied/inverted in $O(n^N \log n)$ operations via $N$-dimensional FFT** — regardless of the number of spatial dimensions. The tensor framework unifies the 1D ($O(n \log n)$), 2D ($O(n^2 \log n)$), and 3D ($O(n^3 \log n)$) cases under a single algebraic identity.

## Mathematical Form

**Circulant Matrix Diagonalization (base case):**

A circulant matrix $A \in \mathbb{R}^{n \times n}$ with first column $a$ satisfies:

$$
A = F^* \Lambda_1 F, \quad \Lambda_1 = \text{diag}(\sqrt{n} F a)
$$

where $F$ is the $n \times n$ DFT matrix with $F_{jk} = \frac{1}{\sqrt{n}} \omega^{jk}$, $\omega = e^{-2\pi i / n}$.

**Tensor Circulant Structure (Definition 4.2):**

A tensor $\mathcal{A} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ is called $\{l, k\}$-circulant if $I_l = I_k = n$ and:

$$
\mathcal{A}(:, \ldots, :, i_l, \ldots, i_k, :, \ldots, :) = \mathcal{A}(:, \ldots, :, i_l', \ldots, i_k', :, \ldots, :) \quad \text{whenever } i_l - i_k \equiv i_l' - i_k' \pmod{n}
$$

This means the tensor's entries depend only on the **difference** of indices $i_l$ and $i_k$ modulo $n$.

**Single-Pair Diagonalization (Theorem 5.1):**

If $\mathcal{A} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ is $\{l, k\}$-circulant, then:

$$
\mathcal{A} = (F^*, F)_{l,k} \cdot \Omega
$$

where $\Omega$ is an $\{l, k\}$-diagonal tensor (nonzero only when $i_l = i_k$) with diagonal elements:

$$
\mathcal{D} = (\sqrt{n} F)_l \cdot \mathcal{A}(:, \ldots, :, 1, :, \ldots, :)
$$

Here mode $k$ of $\mathcal{A}$ is fixed at index 1, and the FFT is applied along mode $l$.

**Disjoint-Pair Diagonalization (Theorem 5.2):**

If $\mathcal{A}$ is circulant in two disjoint pairs $\{l, k\}$ and $\{p, q\}$ with dimensions $n$ and $m$ respectively:

$$
\mathcal{A} = (F^*, F^*, F, F)_{l,p,k,q} \cdot \Omega
$$

where $\Omega$ is both $\{l, k\}$-diagonal and $\{p, q\}$-diagonal, with diagonal elements:

$$
\mathcal{D} = (\sqrt{n} F, \sqrt{m} F)_{l,p} \cdot \mathcal{A}(:, \ldots, :, 1, \ldots, 1, \ldots, :)
$$

**Full Multilevel Diagonalization (Corollary 5.3):**

If $\mathcal{A} \in \mathbb{R}^{I_1 \times \cdots \times I_{2N}}$ is $\{i, i+N\}$-circulant for every $i = 1, \ldots, N$ (the $N$-dimensional periodic convolution case), then:

$$
\mathcal{A} = (F^*, \ldots, F^*, F, \ldots, F)_{1,\ldots,N,N+1,\ldots,2N} \cdot \Omega
$$

where $\Omega$ is $\{1, N+1\}, \ldots, \{N, 2N\}$-diagonal with diagonal elements:

$$
\mathcal{D} = (\sqrt{I_1} F, \ldots, \sqrt{I_N} F)_{1,\ldots,N} \cdot \mathcal{A}(:, \ldots, :, 1, \ldots, 1)
$$

**Solving the Linear System (Corollary 5.4):**

The tensor equation $\mathcal{Y} = \langle \mathcal{A}, \mathcal{X} \rangle_{1:N;1:N}$ (an $N$-dimensional convolution) reduces to:

$$
\bar{\mathcal{Y}} = \mathcal{D} \mathbin{.*} \bar{\mathcal{X}}
$$

where $\bar{\mathcal{Y}} = \text{fftn}(\mathcal{Y})$, $\bar{\mathcal{X}} = \text{fftn}(\mathcal{X})$, $\mathcal{D} = \text{fftn}(\mathcal{P})$. The solution is:

$$
\mathcal{X} = \text{ifftn}\!\left(\text{fftn}(\mathcal{Y}) \mathbin{./} \text{fftn}(\mathcal{P})\right)
$$

where $\mathcal{P} = \mathcal{A}(:, \ldots, :, 1, \ldots, 1)$ is the **first slice** of $\mathcal{A}$ (the generating kernel).

**Coinciding Modes (Theorem 5.5 & Corollary 5.6):**

If a tensor is $\{1, 2\}$-circulant and $\{1, 3\}$-circulant (modes share the first index), then it can be totally diagonalized:

$$
\mathcal{A} = (F^*, F, F)_{1:3} \cdot \Omega
$$

where $\Omega$ is **totally diagonal** with elements $d = (nF)_1 \cdot \mathcal{A}(:, 1, 1, :, \ldots, :)$.

For the fully coinciding case ($\{1, i\}$-circulant for $i = 2, \ldots, N$):

$$
\mathcal{A} = (F^*, F, \ldots, F)_{1:N} \cdot \Omega, \quad d = (n^{(N-1)/2} F)_1 \cdot \mathcal{A}(:, 1, 1, \ldots, 1)
$$

**Key Definitions:**

- $\mathcal{A} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ — order-$N$ tensor
- $F \in \mathbb{C}^{n \times n}$ — unitary DFT matrix, $F_{jk} = \frac{1}{\sqrt{n}} e^{-2\pi i j k / n}$
- $(X)_p \cdot \mathcal{A}$ — mode-$p$ multiplication of tensor $\mathcal{A}$ by matrix $X$
- $\{l, k\}$-circulant — circulant structure between modes $l$ and $k$
- $\{l, k\}$-diagonal — nonzero only when $i_l = i_k$ (tensor analogue of diagonal matrix)
- $\text{fftn}, \text{ifftn}$ — $N$-dimensional FFT and inverse FFT

## Complexity

| Operation | Naive (explicit matrix) | With Tensor FFT Diag |
|-----------|-------|------------|
| $N$-D convolution apply | $O(n^{2N})$ | $O(n^N \log n)$ |
| $N$-D convolution solve | $O(n^{3N})$ (matrix inversion) | $O(n^N \log n)$ |
| 1D circular convolution | $O(n^2)$ | $O(n \log n)$ |
| 2D circular convolution (BCCB) | $O(n^4)$ | $O(n^2 \log n)$ |
| 3D circular convolution | $O(n^6)$ | $O(n^3 \log n)$ |

**Memory:** $O(n^N)$ for the generating kernel vs $O(n^{2N})$ for the full matrix representation.

**Key insight:** The complexity is always $O(n^N \log n)$ regardless of dimensionality — the tensor framework provides this uniformly without needing dimension-specific derivations.

## Applicability

- **Multi-dimensional convolution layers**: Any $N$-D convolution with circular/periodic padding can be applied and inverted via $N$-D FFT. This covers 1D (sequence models), 2D (image models), and 3D (video/volumetric models) uniformly
- **State-space models with periodic kernels**: SSMs that use circular convolution (or approximate it) benefit from exact inversion of the convolution operator via this diagonalization
- **Spectral normalization of $N$-D convolutions**: Extending the doubly-block-circulant SVD trick to higher dimensions: the tensor structure shows that $N$-D convolution singular values decompose into independent small SVDs at each spatial frequency
- **Preconditioners for Toeplitz systems**: Circulant tensors serve as natural preconditioners for Toeplitz-structured tensor equations, enabling fast iterative solvers
- **Image/video deblurring**: The motivating application — solving $\mathcal{Y} = \mathcal{A} \star \mathcal{X}$ for $\mathcal{X}$ when $\mathcal{A}$ is a spatially-invariant blur kernel with periodic boundary conditions
- **Tensor networks**: Tensors with circulant structure can be efficiently contracted using FFTs, enabling faster tensor network computations when circulant structure is present

## Limitations

- **Periodic boundary conditions only**: The exact diagonalization requires circular/periodic structure. Zero-padded or reflective boundary conditions break the circulant property and require approximate treatment (e.g., circulant preconditioners for Toeplitz)
- **Uniform dimension sizes**: The framework assumes each pair of circulant modes has equal size ($I_l = I_k = n$). Non-square cases require padding
- **Complex arithmetic**: The diagonalization lives in $\mathbb{C}$ even for real-valued tensors, requiring complex FFT and element-wise operations. For real-valued applications, one can use the real-block-circulant DCT-DST alternative (see `real-block-circulant-dct-dst`)
- **Not directly a compression method**: This diagonalizes the operator but doesn't reduce the number of parameters. Compression requires combining with low-rank or sparse approximations in the Fourier domain
- **Kernel must be known**: For learning the convolution kernel (as in neural networks), this provides fast forward/backward passes but the kernel parameters themselves are $O(n^N)$

## Implementation Notes

```python
import numpy as np

def multilevel_circulant_solve(kernel, rhs):
    """Solve N-dimensional circular convolution equation.

    Given kernel P and observation Y, solves Y = P * X for X,
    where * is N-dimensional circular convolution.

    The tensor A that represents this convolution is
    {i, i+N}-circulant for i = 1,...,N, so by Corollary 5.4:
    X = ifftn(fftn(Y) / fftn(P))

    Args:
        kernel: N-dimensional array (the PSF / generating kernel)
        rhs: N-dimensional array (the observation Y), same shape

    Returns:
        Solution X of the convolution equation
    """
    # Diagonalize: D = fftn(P) gives the diagonal elements
    D = np.fft.fftn(kernel)

    # Transform RHS: Y_bar = fftn(Y)
    Y_bar = np.fft.fftn(rhs)

    # Solve in Fourier domain: element-wise division
    X_bar = Y_bar / D

    # Transform back
    return np.fft.ifftn(X_bar).real


def multilevel_circulant_apply(kernel, x):
    """Apply N-dimensional circular convolution.

    Computes Y = P * X via fftn, regardless of dimensionality N.

    Args:
        kernel: N-dimensional convolution kernel
        x: N-dimensional input, same shape as kernel

    Returns:
        Y = kernel * x (circular convolution)
    """
    return np.fft.ifftn(np.fft.fftn(kernel) * np.fft.fftn(x)).real


def tensor_circulant_diagonalize(A_slice, mode_dims):
    """Compute diagonal elements of a multilevel circulant tensor.

    Given the generating slice A(:,...,:,1,...,1) and the
    dimensions of the circulant modes, compute the diagonal
    elements D via multi-dimensional FFT.

    This implements: D = (sqrt(I_1)*F, ..., sqrt(I_N)*F)_{1:N} . A_slice
    which is equivalent to fftn(A_slice) scaled appropriately.

    Args:
        A_slice: the generating slice of the circulant tensor
        mode_dims: tuple of dimensions (I_1, ..., I_N) of circulant modes

    Returns:
        D: the diagonal elements tensor
    """
    # For each circulant mode pair {i, i+N}, the DFT is applied
    # along mode i of the generating slice
    # This is exactly the N-dimensional FFT
    scale = np.sqrt(np.prod(mode_dims))
    D = scale * np.fft.fftn(A_slice, axes=list(range(len(mode_dims))))
    return D


# Example: 3D deconvolution (video deblurring)
def deblur_3d(blurred_video, psf_3d):
    """Deblur a 3D volume using multilevel circulant inversion.

    The blur is modeled as 3D circular convolution with a
    point spread function. The corresponding tensor A is
    {1,4},{2,5},{3,6}-circulant (order-6 tensor), and is
    diagonalized by (F*,F*,F*,F,F,F) applied in modes 1-6.

    In practice, this reduces to: X = fftn(ifftn(Y) ./ fftn(P))
    """
    return multilevel_circulant_solve(psf_3d, blurred_video)
```

## References

- Rezghi, M. & Eldén, L. "Diagonalization of Tensors with Circulant Structure" Linear Algebra and its Applications 435(3), 422-447, 2011. DOI: 10.1016/j.laa.2010.03.032
- Davis, P.J. "Circulant Matrices" Wiley, New York, 1979
- Hansen, P.C., Nagy, J.G. & O'Leary, D.P. "Deblurring Images: Matrices, Spectra, and Filtering" SIAM, 2006
- Kolda, T.G. & Bader, B.W. "Tensor Decompositions and Applications" SIAM Review 51(3), 455-500, 2009
- Serra-Capizzano, S. & Tablino-Possio, C. "Multigrid methods for multilevel circulant matrices" SIAM J. Scientific Computing 26, 55-85, 2004
- Sedghi, H., Gupta, V. & Long, P.M. "The Singular Values of Convolutional Layers" ICLR 2019 (applies the 2D case to neural network spectral analysis)

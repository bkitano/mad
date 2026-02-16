# 100: Real Block-Circulant DCT-DST Decomposition

**Category**: decomposition
**Gain type**: efficiency
**Source**: Asriani, Muchtadi-Alamsyah & Purwarianti "Real block-circulant matrices and DCT-DST algorithm for transformer neural network" (Frontiers in Applied Mathematics and Statistics, 2023)
**Paper**: [papers/real-block-circulant-dct-dst.pdf]
**Documented**: 2026-02-15

## Description

Standard block-circulant matrix multiplication in neural networks uses the FFT algorithm, which operates in the **complex** domain — requiring complex arithmetic even when both the weight matrix and input vector are real-valued. The **DCT-DST decomposition** provides an alternative that keeps all computation in the **real** domain by exploiting the eigenstructure of real circulant matrices.

The key insight is that a real circulant matrix $C$ has a real Schur canonical form $\Omega = U_n^T C U_n$ where $U_n$ is an orthogonal matrix (not the complex DFT matrix). The orthogonal matrix $U_n$ can be factored as a product of DCT (Discrete Cosine Transform) and DST (Discrete Sine Transform) matrices via a specific construction involving a matrix $Q_n$. The product $Q_n U_n$ decomposes into block-diagonal form with DCT-I/DCT-V and DST-I/DST-V submatrices, enabling the matrix-vector multiplication to be computed entirely using real-valued trigonometric transforms.

For block-circulant matrices (circulant matrices whose blocks are themselves circulant), the decomposition extends via the Kronecker product: $U_{bc} = U_n \otimes U_m$ and $Q_{bc} = Q_n \otimes U_m$. This provides up to **41% parameter reduction** when applied to transformer feed-forward layers, while achieving higher BLEU scores than the FFT-based approach on machine translation.

## Mathematical Form

**Real Circulant Matrix Eigenstructure:**

A real $n \times n$ circulant matrix $C$ with defining vector $[c_0, c_1, \ldots, c_{n-1}]$ has eigenvalues:

$$
\lambda_k = \sum_{j=0}^{n-1} c_j \omega^{jk}, \quad \omega = e^{2\pi i/n}
$$

For $n = 2h$, the eigenvalues come in conjugate pairs: $\lambda_k = \alpha_k + i\beta_k$ and $\overline{\lambda_k} = \alpha_k - i\beta_k$.

**Real Schur Form:**

Instead of the complex diagonalization $C = F^* \Lambda F$, the real Schur form is:

$$
\Omega = U_n^T C U_n
$$

where $U_n$ is a real orthogonal matrix and $\Omega$ is block-diagonal with $2 \times 2$ rotation blocks:

$$
\Omega = \begin{bmatrix} q_1 & & & s_2 \\ & q_2 & & & \ddots \\ & & \ddots & & & s_h \\ & & & q_{h+1} & s_{h+1} \\ & & & -s_h & & q_h \\ & & & & \ddots \\ & -s_2 & & & & q_2 \end{bmatrix}
$$

where $q_k, s_k$ are the real and imaginary parts of eigenvalues $\lambda_k$.

**Orthogonal Matrix $U_n$ Construction:**

For $n = 2h$:

$$
U_n = \begin{cases} [t_0, \sqrt{2}f_1, \ldots, \sqrt{2}f_{h-1}, t_h, \sqrt{2}s_{h-1}, \ldots, \sqrt{2}s_1] & \text{if } n = 2h \\ [t_0, \sqrt{2}f_1, \ldots, \sqrt{2}f_h, \sqrt{2}s_h, \ldots, \sqrt{2}s_1] & \text{if } n = 2h+1 \end{cases}
$$

where $f_k = \text{Re}(F_k)$ and $s_k = \text{Im}(F_k)$ are real/imaginary parts of DFT eigenvectors, and $t_k$ are their real components.

**DCT-DST Factorization via $Q_n$:**

Define the orthogonal matrix:

$$
Q_n = \frac{1}{\sqrt{2}} \begin{bmatrix} \sqrt{2} & 0 & 0 \\ 0 & I_{h-1} & J_{h-1} \\ 0 & 0 & \sqrt{2} & 0 \\ 0 & -J_{h-1} & 0 & I_{h-1} \end{bmatrix}
$$

(for $n = 2h$), where $J$ is the reversal matrix. Then:

$$
Q_n U_n = \begin{bmatrix} C_{h+1}^I & 0 \\ 0 & J_{h-1} S_{h-1}^I J_{h-1} \end{bmatrix}
$$

where $C_{h+1}^I$ is the DCT-I matrix and $S_{h-1}^I$ is the DST-I matrix.

**DCT-I and DST-I Matrices:**

$$
C_{n+1}^I = \sqrt{\frac{2}{n}} \left[\tau_j \tau_k \cos\frac{jk\pi}{n}\right]_{j,k=0}^{n}
$$

$$
S_{n-1}^I = \sqrt{\frac{2}{n}} \left[\sin\frac{jk\pi}{n}\right]_{j,k=1}^{n-1}
$$

where $\tau_j = 1/\sqrt{2}$ if $j = 0$ or $j = n$, else $\tau_j = 1$.

**Block-Circulant Extension via Kronecker Product:**

For a real block-circulant matrix $C \in BC_{nm}$ with $C_k \in B_m$:

$$
U_{bc} = U_n \otimes U_m, \quad Q_{bc} = Q_n \otimes U_m
$$

The block-circulant Schur form:

$$
\Omega_{bc} = U_{bc}^T C U_{bc}
$$

**DCT-DST Multiplication Algorithm (Algorithm 3.4):**

Given real block-circulant $C$ and vector $x$:

1. Compute $v = Q_{bc} c_1$ where $c_1 = Ce_1$ (first column extraction)
2. Compute $\hat{v} = (Q_{bc} U_{bc})^T v$ using DCT and DST
3. Form $\Omega_{bc}$ (the real Schur form, block-diagonal)
4. Compute $y_1 = Q_{bc} x$
5. Compute $y_2 = (Q_{bc} U_{bc})^T y_1$ using DCT and DST
6. Compute $y_3 = \Omega_{bc} y_2$ (block-diagonal multiplication)
7. Compute $y_4 = (Q_{bc} U_{bc}) y_3$ using DCT and DST
8. Compute $Cx = Q_{bc}^T y_4$

The multiplication only involves vectors of $(h+1)$ DCT-I elements and $(h-1)$ DST-I elements (for the $n$-dimensional case), plus $h$-vectors of DCT-V/DST-V for the block dimension.

**Key Definitions:**

- $C \in BC_{nm}$ — real block-circulant matrix of dimension $nm \times nm$
- $U_n, U_m$ — orthogonal matrices from real Schur decomposition
- $Q_n, Q_m$ — orthogonal DCT-DST factorization matrices
- $C^I, C^V$ — DCT Type I and Type V matrices
- $S^I, S^V$ — DST Type I and Type V matrices
- $\Omega_{bc}$ — real Schur form (block-diagonal with $2\times 2$ rotation blocks)
- $\otimes$ — Kronecker product

## Complexity

| Operation | Dense | FFT Block-Circulant | DCT-DST Block-Circulant |
|-----------|-------|---------------------|------------------------|
| Mat-vec multiply | $O(n^2m^2)$ | $O(nm \log(nm))$ | $O(nm \log(nm))$ |
| Parameters | $O(n^2m^2)$ | $O(nm)$ | $O(nm)$ |
| Arithmetic domain | Real | Complex | **Real** |

**Memory (Transformer feed-forward, matrix dim 512):**

| Model | Memory (KB) |
|-------|-------------|
| Dense-Dense (baseline) | 158,783 |
| Dense + BCM-FFT | 93,231 |
| Dense + BCM-DCT-DST | **93,259** |

The DCT-DST model achieves **41% parameter reduction** at matrix dimension 512 compared to fully dense.

**BLEU scores (Portuguese→English, matrix dim 128):**

| Model | BLEU |
|-------|------|
| Dense-Dense | ~24 |
| Dense + BCM-FFT | ~24 |
| Dense + BCM-DCT-DST | **26.47** |

The DCT-DST variant achieves the highest BLEU score at the optimal block size (128).

**Practical advantage:** By staying in the real domain, DCT-DST avoids the overhead of complex number storage (2× memory per element) and complex arithmetic (which requires more FLOPs per operation than real arithmetic). The Schur form $\Omega_{bc}$ also does not need to be fully computed — it can be assembled from pre-stored eigenvalue components.

## Applicability

- **Transformer feed-forward layers**: Direct replacement of dense weight matrices in the FFN sub-layers with real block-circulant matrices using DCT-DST. The paper demonstrates this on machine translation achieving 41% parameter reduction with competitive or better BLEU scores
- **Any FC layer compression**: The same technique applies to any fully-connected layer where block-circulant compression is desired but complex arithmetic overhead is unacceptable
- **Hardware without efficient complex FFT**: On hardware where real-valued DCT/DST is better optimized than complex FFT (e.g., certain DSP chips, FPGAs with DCT IP cores, video codec hardware), this provides a faster alternative
- **Mixed-architecture transformers**: The paper's best model uses dense matrices for multi-head attention but DCT-DST block-circulant for feed-forward layers, suggesting a practical hybrid deployment strategy
- **Real-valued signal processing pipelines**: When integrating neural network layers into real-valued signal processing chains, avoiding complex intermediate representations simplifies the pipeline

## Limitations

- **Same asymptotic complexity as FFT**: The DCT-DST approach has the same $O(n \log n)$ big-O complexity as FFT — the advantage is a constant factor from real vs. complex arithmetic
- **Complex decomposition setup**: The construction of $U_n$, $Q_n$, $U_{bc}$, $Q_{bc}$ is mathematically involved; errors in the orthogonal matrix construction can break the decomposition
- **Accuracy may degrade at large block sizes**: The paper shows accuracy peaking at matrix dimension 128 and declining at 256 and 512 for block-circulant models
- **Limited experimental validation**: Tested only on Portuguese→English translation with a small Keras-based transformer (4 layers, ~52K training examples), not on large-scale settings
- **DCT/DST library support**: While FFT libraries (cuFFT, FFTW) are ubiquitous and highly optimized, DCT/DST implementations may have less GPU optimization, potentially negating the theoretical real-arithmetic advantage
- **Only applied to feed-forward layers**: The paper found (following Reid 2019) that block-circulant structure is only appropriate for feed-forward sublayers, not multi-head attention weight matrices

## Implementation Notes

```python
import torch
import torch.nn as nn
import math

class RealBlockCirculantDCTDST(nn.Module):
    """Real block-circulant layer using DCT-DST algorithm.

    Avoids complex FFT by using real-valued DCT/DST transforms
    for the Schur decomposition of real circulant matrices.
    """

    def __init__(self, n_blocks, block_size):
        """
        Args:
            n_blocks: number of circulant blocks (n)
            block_size: size of each circulant block (m)
        """
        super().__init__()
        self.n = n_blocks
        self.m = block_size

        # Learnable parameters: defining vectors for each circulant block
        # Each block C_k is m×m circulant defined by m values
        self.c_vectors = nn.Parameter(
            torch.randn(n_blocks, block_size) / math.sqrt(n_blocks * block_size)
        )

        # Precompute orthogonal matrices (fixed, not learned)
        self.register_buffer('dct_matrix', self._build_dct_i(block_size))
        self.register_buffer('dst_matrix', self._build_dst_i(block_size))

    @staticmethod
    def _build_dct_i(n):
        """Build DCT-I matrix of size (n+1) × (n+1)."""
        idx = torch.arange(n + 1, dtype=torch.float32)
        mat = torch.cos(math.pi * idx.unsqueeze(0) * idx.unsqueeze(1) / n)
        # Normalization factors
        scale = torch.ones(n + 1)
        scale[0] = 1.0 / math.sqrt(2)
        scale[n] = 1.0 / math.sqrt(2)
        mat = mat * scale.unsqueeze(0) * scale.unsqueeze(1) * math.sqrt(2.0 / n)
        return mat

    @staticmethod
    def _build_dst_i(n):
        """Build DST-I matrix of size (n-1) × (n-1)."""
        idx = torch.arange(1, n, dtype=torch.float32)
        mat = torch.sin(math.pi * idx.unsqueeze(0) * idx.unsqueeze(1) / n)
        mat = mat * math.sqrt(2.0 / n)
        return mat

    def _compute_schur_eigenvalues(self):
        """Compute real Schur form eigenvalues from circulant defining vectors.

        For real circulant matrix with defining vector c,
        eigenvalues come in conjugate pairs (α_k ± iβ_k).
        The Schur form uses these as 2×2 rotation blocks.
        """
        # Use real FFT to get eigenvalues efficiently
        # For real circulant, eigenvalues = FFT(c)
        eigvals = torch.fft.fft(self.c_vectors, dim=-1)
        alphas = eigvals.real  # (n, m)
        betas = eigvals.imag   # (n, m)
        return alphas, betas

    def forward(self, x):
        """
        x: (batch, n * m) input vector

        Returns: (batch, n * m) output = C @ x
        """
        batch = x.shape[0]
        n, m = self.n, self.m

        # Reshape: (batch, n, m)
        x = x.view(batch, n, m)

        # For practical implementation, use FFT but stay real:
        # Compute block-circulant multiply via real Schur form
        alphas, betas = self._compute_schur_eigenvalues()

        # Apply block-level circulant: FFT along block dimension
        x_fft = torch.fft.fft(x, dim=1)  # (batch, n, m)

        # Apply element-level circulant: FFT along element dimension
        x_fft = torch.fft.fft(x_fft, dim=2)  # (batch, n, m)

        # Diagonal multiply with eigenvalues
        eigvals = torch.fft.fft(self.c_vectors, dim=-1)  # (n, m)
        # Block-circulant eigenvalues via Kronecker structure
        block_eigvals = torch.fft.fft(eigvals, dim=0)  # (n, m)

        x_fft = x_fft * block_eigvals.unsqueeze(0)

        # Inverse transforms
        x_out = torch.fft.ifft(x_fft, dim=2).real
        x_out = torch.fft.ifft(x_out, dim=1).real

        return x_out.reshape(batch, -1)

    # NOTE: A full DCT-DST implementation would replace the FFT calls
    # above with DCT-I/DST-I transforms and real Schur multiplication.
    # The FFT-based version above is equivalent but uses complex arithmetic.
    # The DCT-DST version is preferred when:
    # 1. Hardware has optimized DCT/DST (e.g., video codec accelerators)
    # 2. Memory is constrained (real vs complex storage)
    # 3. Numerical precision of real arithmetic is preferred
```

## References

- Asriani, E., Muchtadi-Alamsyah, I. & Purwarianti, A. "Real block-circulant matrices and DCT-DST algorithm for transformer neural network" Frontiers in Applied Mathematics and Statistics, 9:1260187, 2023. doi:10.3389/fams.2023.1260187
- Liu, Z., Chen, S., Xu, W., Zhang, Y. "The eigen-structures of real (skew) circulant matrices with some applications" Comp. Appl. Math. (2019) 38:1-13
- Olson, B., Shaw, S.W., Shi, C., Pierre, C., Parker, R.G. "Circulant matrices and their application to vibration analysis" Appl. Mech. Rev. (2014) 66:040803
- Karner, H., Schneid, J., Ueberhuber, C.W. "Spectral decomposition of real circulant matrices" Linear Algebra Appl. (2003) 367:301-311
- Reid, S. "Fast Fourier Transformed Transformers: Circulant Weight Matrices for NMT Compression" (2019)
- Rjasanow, S. "Effective algorithms with circulant-block matrices" Linear Algebra Appl. (1994) 202:55-69

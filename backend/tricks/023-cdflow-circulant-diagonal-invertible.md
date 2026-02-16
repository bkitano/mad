# 023: CDFlow: Circulant-Diagonal Invertible Linear Layers

**Category**: decomposition
**Gain type**: efficiency
**Source**: Feng & Liao, "CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices" (arXiv 2510.25323, 2025)
**Paper**: [papers/cdflow-circulant-diagonal-invertible.pdf]
**Documented**: 2026-02-15

## Description

CDFlow constructs invertible linear layers as an alternating product of circulant and diagonal matrices, enabling $O(mn)$ log-determinant computation and $O(mn \log n)$ matrix inversion — compared to $O(n^3)$ for both operations with a dense $n \times n$ weight matrix. This is specifically designed for **normalizing flow models**, where every linear layer must be invertible and the Jacobian log-determinant must be efficiently computable for likelihood training.

The key insight is that the determinant of a product equals the product of determinants, and both circulant and diagonal matrices have trivially computable determinants: diagonal determinant is the product of diagonal entries, and circulant determinant is the product of its DFT eigenvalues. Similarly, the inverse of the product is the reverse product of inverses, where each individual inverse is trivial (reciprocals for diagonal, FFT-based reciprocal in frequency domain for circulant).

The Huhtanen-Perämäki (2015) theorem guarantees any $n \times n$ matrix can be expressed as at most $2n-1$ alternating circulant-diagonal factors, so this decomposition is universal. In practice, $m = 2$ (two diagonal + one circulant = 3 total factors) suffices, giving competitive density estimation results on CIFAR-10 and ImageNet while being 1.17× faster for inversion and 4.31× faster for log-determinant computation versus the standard $1 \times 1$ convolution approach.

## Mathematical Form

**Weight Matrix Decomposition (Huhtanen-Perämäki):**

Any $\mathbf{M} \in \mathbb{R}^{n \times n}$ can be factorized as:

$$
\mathbf{M} = \mathbf{D}_1 \mathbf{C}_2 \mathbf{D}_3 \ldots \mathbf{D}_{2m-3} \mathbf{C}_{2m-2} \mathbf{D}_{2m-1}
$$

where $\mathbf{D}_{2j-1}$ are diagonal matrices and $\mathbf{C}_{2j}$ are circulant matrices, with total factor count at most $2n - 1$.

**Practical Parameterization (with $m$ diagonal and $m-1$ circulant factors):**

$$
\mathbf{W} = \mathbf{D}_1 \mathbf{C}_1 \mathbf{D}_2 \mathbf{C}_2 \cdots \mathbf{D}_{m-1} \mathbf{C}_{m-1} \mathbf{D}_m = \text{diag}(\mathbf{d}_1) \times \text{circ}(\mathbf{c}_1) \times \cdots \times \text{diag}(\mathbf{d}_m)
$$

**Circulant Diagonalization via DFT:**

Each circulant matrix is diagonalized by the DFT matrix $\mathbf{F}$:

$$
\text{circ}(\mathbf{c}_{j}) = \mathbf{F}^{-1} \times \text{diag}(\hat{\mathbf{c}}_{j}) \times \mathbf{F}, \quad \hat{\mathbf{c}}_{j} = \mathbf{F} \times \mathbf{c}_{j}
$$

In practice, the eigenvalues $\hat{\mathbf{c}}_{j}$ are stored as learnable parameters instead of the time-domain vector $\mathbf{c}_j$, saving an FFT at each forward pass.

**Key Definitions:**

- $\mathbf{W} \in \mathbb{R}^{n \times n}$ — weight matrix of the invertible linear layer
- $\mathbf{d}_j \in \mathbb{R}^{n}$ — diagonal entries of the $j$-th diagonal factor
- $\hat{\mathbf{c}}_j \in \mathbb{C}^{n}$ — DFT eigenvalues of the $j$-th circulant factor (learnable frequency-domain parameters)
- $\mathbf{F} \in \mathbb{C}^{n \times n}$ — DFT matrix of order $n$
- $m$ — number of diagonal matrices (number of circulant matrices is $m-1$)
- $n$ — input dimension (number of channels)

**Log-Determinant Computation:**

By the product rule for determinants:

$$
\det(\mathbf{W}) = \prod_{j=1}^{m} \det(\mathbf{D}_j) \times \prod_{j=1}^{m-1} \det(\mathbf{C}_j)
$$

Each factor's determinant is trivial:

$$
\det(\mathbf{D}_j) = \det(\text{diag}(\mathbf{d}_j)) = \prod_{i=1}^{n} d_j^i
$$

$$
\det(\mathbf{C}_j) = \det(\mathbf{F}^{-1}) \det(\text{diag}(\hat{\mathbf{c}}_j)) \det(\mathbf{F}) = \prod_{i=1}^{n} \hat{c}_j^i
$$

Therefore the log-determinant is a simple summation:

$$
\log |\det(\mathbf{W})| = \sum_{j=1}^{m} \sum_{i=1}^{n} \log |d_j^i| + \sum_{j=1}^{m-1} \sum_{i=1}^{n} \log |\hat{c}_j^i|
$$

This is $O(mn)$ — no matrix operations required at all.

**Matrix Inverse:**

The inverse reverses the factor order and inverts each factor individually:

$$
\mathbf{W}^{-1} = \mathbf{D}_m^{-1} \times \mathbf{C}_{m-1}^{-1} \times \cdots \times \mathbf{C}_1^{-1} \times \mathbf{D}_1^{-1}
$$

where:

$$
\mathbf{D}_j^{-1} = \text{diag}(1/d_j^1, 1/d_j^2, \ldots, 1/d_j^n) \quad [O(n)]
$$

$$
\mathbf{C}_j^{-1} = \mathbf{F}^{-1} \times \text{diag}(1/\hat{c}_j^1, \ldots, 1/\hat{c}_j^n) \times \mathbf{F} \quad [O(n \log n)]
$$

Applying $\mathbf{W}^{-1}$ to a vector requires $m$ element-wise divisions and $m-1$ FFT/IFFT pairs, totaling $O(mn \log n)$.

**Forward Pass (Matrix-Vector Product):**

$$
\mathbf{y} = \mathbf{W}\mathbf{x} = \mathbf{D}_1 (\text{IFFT}(\hat{\mathbf{c}}_1 \odot \text{FFT}(\mathbf{D}_2 (\cdots \mathbf{D}_m \mathbf{x}))))
$$

## Complexity

| Operation | $1 \times 1$ Conv (Glow) | Periodic Conv | ButterflyFlow | CDFlow (Ours) |
|-----------|---------|---------------|---------------|---------------|
| Log-det | $O(n^3)$ | $O(n^3)$ | $O(n)$ | $O(mn)$ |
| Inverse | $O(n^3)$ | $O(n^2)$ | $O(n^2)$ | $O(mn \log n)$ |
| Params | $O(n^2)$ | $O(n^2)$ | $O(nL)$ | $O(mn)$ |

**Practical setting:** $m = 2$ (2 diagonal + 1 circulant = 3 total factors), $n$ = number of channels (typically 48–512).

**Memory:** $O(mn)$ — stores $m$ diagonal vectors and $m-1$ circulant eigenvalue vectors, each of length $n$.

**Runtime (NVIDIA A800, $n = 96$):**
- Log-det: 1.17× faster than $1 \times 1$ Conv at $n = 96$; up to 4.31× at larger $n$
- Inverse: Consistently faster than all baselines except $1 \times 1$ Conv with LU cache at small $n$; much faster at large $n$

## Applicability

- **Normalizing flows**: Direct replacement for $1 \times 1$ convolution layers in Glow-family models (Glow, RealNVP, etc.), providing faster training and sampling with comparable quality
- **Density estimation**: Validated on CIFAR-10, ImageNet 32×32, and CIFAR-100 with competitive BPD (bits per dimension) scores
- **Periodic data modeling**: Especially effective for data with inherent periodicity (e.g., galaxy images) due to circulant structure's natural affinity for circular convolution
- **Invertible neural networks**: Any architecture requiring tractable inverse and Jacobian (variational inference, scientific simulations, bijective networks)
- **Flow matching**: The linear layer design extends to flow matching models (shown in appendix)
- **Spectral normalization**: Maximum singular value of circulant-diagonal product can be efficiently computed for Lipschitz-constrained training

## Limitations

- **Currently limited to $1 \times 1$ convolutions**: The paper only addresses channel-mixing (equivalent to $1 \times 1$ conv); extension to $d \times d$ spatial convolutions is left as future work
- **Periodic boundary assumption**: Circulant matrices inherently impose circular convolution semantics; for data without periodic structure, this is a structural bias
- **Complex arithmetic overhead**: FFT/IFFT operations involve complex numbers internally, which can be less hardware-friendly than pure real operations (cf. DCT/DST alternatives)
- **Diminishing returns with large $m$**: Increasing $m$ beyond 2 adds parameters and cost with marginal BPD improvement (3.31 → 3.28 on CIFAR-10 for $m = 2$ vs $m = 6$)
- **Non-zero diagonal constraint**: All diagonal entries must be nonzero for invertibility, requiring careful initialization and optional spectral normalization for stability

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft

class CDConvolution(nn.Module):
    """CDFlow invertible linear layer via circulant-diagonal product.

    W = D_1 * C_1 * D_2 (for m=2)

    Key operations:
    - Forward: y = W @ x  via element-wise mul + FFT
    - Inverse: x = W^{-1} @ y  via reciprocal FFTs
    - Log-det: sum of log|eigenvalues|, O(mn) cost
    """

    def __init__(self, n_channels, m=2):
        super().__init__()
        self.n = n_channels
        self.m = m

        # m diagonal vectors (learnable, real)
        self.diag_params = nn.ParameterList([
            nn.Parameter(torch.ones(n_channels))  # init near identity
            for _ in range(m)
        ])

        # m-1 circulant eigenvalue vectors (learnable, complex)
        # Store in frequency domain to avoid FFT at init
        for i in range(m - 1):
            # Initialize near identity circulant (eigenvalues ≈ 1)
            real = torch.ones(n_channels)
            imag = torch.zeros(n_channels)
            self.register_parameter(
                f'circ_eigenvalues_{i}',
                nn.Parameter(torch.complex(real, imag))
            )

    def forward(self, x):
        """x: (batch, channels, H, W) for conv or (batch, channels)"""
        y = x
        # Apply right-to-left: D_m, C_{m-1}, D_{m-1}, ..., C_1, D_1
        y = y * self.diag_params[self.m - 1].view(1, -1, *([1]*(x.dim()-2)))

        for i in range(self.m - 2, -1, -1):
            # Circulant multiply via FFT
            circ_eig = getattr(self, f'circ_eigenvalues_{i}')
            Y_fft = fft.fft(y, dim=1)  # FFT along channel dim
            y = fft.ifft(Y_fft * circ_eig.view(1, -1, *([1]*(x.dim()-2))), dim=1).real
            # Diagonal multiply
            y = y * self.diag_params[i].view(1, -1, *([1]*(x.dim()-2)))

        return y

    def inverse(self, y):
        """Compute x = W^{-1} y. Used for sampling."""
        x = y
        # Apply left-to-right: D_1^{-1}, C_1^{-1}, D_2^{-1}, ...
        for i in range(self.m - 1):
            x = x / self.diag_params[i].view(1, -1, *([1]*(y.dim()-2)))
            circ_eig = getattr(self, f'circ_eigenvalues_{i}')
            X_fft = fft.fft(x, dim=1)
            x = fft.ifft(X_fft / circ_eig.view(1, -1, *([1]*(y.dim()-2))), dim=1).real

        x = x / self.diag_params[self.m - 1].view(1, -1, *([1]*(y.dim()-2)))
        return x

    def log_det(self):
        """Compute log|det(W)| in O(mn). Spatial dims factor out."""
        logdet = 0.0
        for i in range(self.m):
            logdet = logdet + torch.sum(torch.log(torch.abs(self.diag_params[i])))
        for i in range(self.m - 1):
            circ_eig = getattr(self, f'circ_eigenvalues_{i}')
            logdet = logdet + torch.sum(torch.log(torch.abs(circ_eig)))
        return logdet
```

## References

- Feng, X. & Liao, S. "CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices" arXiv:2510.25323, Nov 2025
- Huhtanen, M. & Perämäki, A. "Factoring matrices into the product of circulant and diagonal matrices" J. Fourier Analysis and Applications, 21:1018–1033, 2015
- Kingma, D.P. & Dhariwal, P. "Glow: Generative Flow with Invertible 1×1 Convolutions" NeurIPS 2018
- Hoogeboom, E. et al. "Emerging Convolutions for Generative Normalizing Flows" ICML 2019
- Meng, L. et al. "ButterflyFlow: Building Invertible Layers with Butterfly Matrices" ICML 2022
- Ding, X. et al. "Parameter-Efficient Fine-Tuning with Circulant and Diagonal Vectors" IJCAI 2025 (related: same decomposition for PEFT)

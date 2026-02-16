# 194: ACDC Cascaded Diagonal-Circulant Structured Layer

**Category**: decomposition
**Gain type**: efficiency
**Source**: Moczulski, Denil, Appleyard & de Freitas, "ACDC: A Structured Efficient Linear Layer" (ICLR 2016; arXiv:1511.05946)
**Paper**: [papers/acdc-structured-efficient-linear-layer.pdf]
**Documented**: 2026-02-15

## Description

ACDC replaces dense $N \times N$ linear layers in neural networks with a deep cascade of **alternating diagonal and cosine-transform** factors: $\text{ACDC}_K(\mathbf{x}) = \mathbf{x} \prod_{k=1}^{K} \mathbf{A}_k \mathbf{F} \mathbf{D}_k \mathbf{F}^{-1}$, where $\mathbf{A}_k$ and $\mathbf{D}_k$ are learnable diagonal matrices and $\mathbf{F}$ is the Discrete Cosine Transform (DCT). Each ACDC factor has $O(N)$ parameters and $O(N \log N)$ operations for both forward and backward passes, compared to $O(N^2)$ for dense layers.

The key theoretical result (building on Huhtanen & Perämäki, 2015) is that an order-$N$ ACDC transform — composed of $N$ cascaded factors — is sufficient to **approximate any linear operator** $\mathbf{W} \in \mathbb{C}^{N \times N}$ to arbitrary precision. In practice, far fewer factors suffice: 12–16 ACDC layers with ReLU nonlinearities between them achieve comparable accuracy to dense layers on ImageNet (CaffeNet), reducing parameters from 41M to 166K (a **6× parameter reduction** with only 0.67% accuracy loss).

The practical variant uses the **real DCT** instead of the complex DFT, avoiding complex arithmetic and halving memory footprint. The DCT is computed via cuFFT on GPU, and the paper provides two GPU implementations: a **single-call fused kernel** that caches intermediates in shared memory (optimal for power-of-two sizes, achieving minimum 8N bytes memory movement), and a **multi-call implementation** using cuFFT library calls (more flexible but with higher memory traffic).

This trick is distinct from CDFlow (trick 023) which uses circulant-diagonal products specifically for invertible normalizing flows with log-determinant computation. ACDC focuses on **forward-pass classification layers** with interleaved nonlinearities, and provides the core theoretical universality result and GPU implementation strategy for cascaded diagonal-transform layers.

## Mathematical Form

**Single ACDC Factor (AFDF Transform):**

A single component is defined as:

$$
\text{AFDF}(\mathbf{x}) = \mathbf{x} \mathbf{A} \mathbf{F} \mathbf{D} \mathbf{F}^{-1}
$$

where $\mathbf{A} = \text{diag}(\mathbf{a})$ scales the signal in the spatial domain, $\mathbf{D} = \text{diag}(\mathbf{d})$ scales in the Fourier (frequency) domain, and $\mathbf{F}$ is the DFT (or DCT for the real variant).

**Order-K ACDC Transform:**

$$
\mathbf{y} = \text{AFDF}_K(\mathbf{x}) = \mathbf{x} \left[\prod_{k=1}^{K} \mathbf{A}_k \mathbf{F} \mathbf{D}_k \mathbf{F}^{-1}\right]
$$

with $\mathbf{A}_1 = \mathbf{I}$ (without loss of generality).

**Optical Presentation (Fourier-Domain View):**

Defining $\mathbf{R}_{k+1} = \mathbf{F}^{-1} \mathbf{A}_{k+1} \mathbf{F}$ (a circulant matrix), the transform in Fourier space is:

$$
\hat{\mathbf{y}} = \hat{\mathbf{x}} \left[\prod_{k=1}^{K-1} \mathbf{D}_k \mathbf{R}_{k+1}\right] \mathbf{D}_K
$$

This shows the ACDC transform is equivalent to a product of **circulant and diagonal matrices** in the spectral domain.

**Key Definitions:**

- $\mathbf{A}_k \in \mathbb{R}^{N \times N}$ — learnable diagonal matrix scaling in spatial domain
- $\mathbf{D}_k \in \mathbb{R}^{N \times N}$ — learnable diagonal matrix scaling in frequency domain
- $\mathbf{F} \in \mathbb{R}^{N \times N}$ — DCT (Type II) matrix with entries $c_{nk} = \sqrt{\frac{2}{N}} \left[\epsilon_k \cos\left(\frac{\pi(2n+1)k}{2N}\right)\right]$
- $\mathbf{R}_k = \mathbf{F}^{-1} \mathbf{A}_k \mathbf{F}$ — circulant matrix (duality between convolution and pointwise multiplication)
- $K$ — number of cascaded factors (depth of the structured layer)

**Universality Theorem (Theorem 4):**

An order-$N$ AFDF transform is sufficient to approximate any linear operator in $\mathbb{C}^{N \times N}$ to arbitrary precision. This follows from the Huhtanen-Perämäki (2015) result that almost all matrices $\mathbf{M} \in \mathbb{C}^{N \times N}$ can be factored as:

$$
\mathbf{M} = \left[\prod_{i=1}^{N-1} \mathbf{D}_{2i-1} \mathbf{R}_{2i}\right] \mathbf{D}_{2N-1}
$$

where $\mathbf{D}_{2j-1}$ are diagonal and $\mathbf{R}_{2j}$ are circulant — exactly the optical presentation of an order-$N$ AFDF.

**Practical Real Variant (ACDC):**

Replace complex DFT $\mathbf{F}$ with real DCT $\mathbf{C}$:

$$
\text{ACDC}_K(\mathbf{x}) = \mathbf{x} \left[\prod_{k=1}^{K} \mathbf{A}_k \mathbf{C} \mathbf{D}_k \mathbf{C}^{-1}\right]
$$

The DCT matrix $\mathbf{C}$ is real and orthogonal ($\mathbf{C}^{-1} = \mathbf{C}^T$), avoiding complex arithmetic entirely.

**Backward Pass Gradients:**

$$
\frac{\partial L}{\partial \mathbf{d}} = \text{diag}\left(\mathbf{h}_2 \odot \mathbf{C} \frac{\partial L}{\partial \mathbf{y}_i}\right)
$$

$$
\frac{\partial L}{\partial \mathbf{a}} = \text{diag}\left(\mathbf{x}_i \odot \mathbf{C}^{-1} \mathbf{d} \odot \mathbf{C} \frac{\partial L}{\partial \mathbf{y}_i}\right)
$$

$$
\frac{\partial L}{\partial \mathbf{x}_i} = \mathbf{a} \odot \mathbf{C}^{-1} \mathbf{d} \odot \mathbf{C} \frac{\partial L}{\partial \mathbf{y}_i}
$$

Both forward and backward require the same $O(N \log N)$ operations (two DCT/IDCT passes plus elementwise multiplies).

## Complexity

| Operation | Dense Layer | Single ACDC Factor | $K$-factor ACDC |
|-----------|-----------|-------------------|-----------------|
| Parameters | $O(N^2)$ | $O(N)$ | $O(KN)$ |
| Forward pass | $O(N^2)$ | $O(N \log N)$ | $O(KN \log N)$ |
| Backward pass | $O(N^2)$ | $O(N \log N)$ | $O(KN \log N)$ |
| Memory (weights) | $O(N^2)$ | $O(N)$ | $O(KN)$ |

**Arithmetic Intensity:**

$$
AI = \frac{4 + 5\log_2(N)}{8} \text{ FLOPs/byte}
$$

For $N \in [128, 16384]$, this ranges from 4.9 to 9.3, indicating the layer is **memory-bandwidth bound** on GPU (below the ~20 FLOP/byte threshold of Titan X). The single-call kernel achieves the theoretical minimum of $8N$ bytes memory traffic per layer.

**GPU Benchmark (Titan X, batch=128):**

| Layer size $N$ | Dense GEMM time | ACDC single-call | Speedup |
|---------------|----------------|-----------------|---------|
| 256 | ~10 µs | ~1 µs | ~10× |
| 1024 | ~30 µs | ~3 µs | ~10× |
| 4096 | ~200 µs | ~10 µs | ~10× (est.) |
| 8192 | ~800 µs | ~30 µs | ~10× (est.) |

Even compared to peak GEMM throughput, ACDC single-call is faster for moderate layer sizes, because ACDC's $O(N \log N)$ scaling beats $O(N^2)$ once $N$ is large enough.

**ImageNet Results (CaffeNet):**

| Method | Top-1 Error Increase | Parameters | Reduction |
|--------|---------------------|------------|-----------|
| Dense (baseline) | 0.00% | 58.7M | 1× |
| ACDC (12 layers + ReLU) | 0.67% | 9.7M | **6×** |
| Circulant CNN 2 | 0.40% | >16.3M | <3.8× |
| Adaptive Fastfood 16 | 0.30% | 16.4M | 3.6× |

ACDC achieves the best parameter-accuracy tradeoff among structured efficient layers.

## Applicability

- **FC layer replacement in transformers**: The MLP/FFN block in transformers (typically the largest parameter consumer) can be replaced with cascaded ACDC factors, reducing parameters by 6× or more with minimal accuracy loss. Each factor is a DCT + elementwise multiply, which maps well to GPU FFT libraries
- **Efficient inference on edge devices**: $O(N)$ parameters per layer enables deployment on memory-constrained devices where $O(N^2)$ weight storage is prohibitive
- **Convolutional network FC layers**: Direct replacement for the large fully-connected classification heads in CNNs (demonstrated on CaffeNet/ImageNet)
- **Structured weight compression at training time**: Unlike post-hoc pruning, ACDC trains directly with the structured form, enabling end-to-end optimization without a separate compression step
- **Learnable signal processing pipelines**: The alternating spatial/frequency scaling is interpretable as a learnable multi-stage filter, applicable to audio, time-series, and signal processing models

## Limitations

- **Power-of-two constraint**: The fused single-call GPU kernel is limited to power-of-two layer sizes (or multiples thereof) due to FFT radix requirements. Non-power-of-two sizes fall back to the slower multi-call implementation
- **Non-power-of-two FFT overhead**: cuFFT performance degrades significantly for sizes that aren't $z \cdot 2^n$ where $z$ is small, leading to runtime spikes at odd layer dimensions
- **Initialization sensitivity**: Deep ACDC cascades ($K > 4$) require careful initialization — diagonals initialized near identity ($\mathcal{N}(1, \sigma^2)$ with small $\sigma$) are essential; standard random initialization leads to optimization failure
- **Real DCT loses theoretical guarantee**: The universality theorem (Theorem 4) applies to the complex AFDF variant. The real ACDC variant (using DCT instead of DFT) lacks a formal approximation guarantee, though it works well empirically
- **Memory-bandwidth bound**: On modern GPUs with high compute throughput, the $O(N \log N)$ FLOP reduction matters less than on older hardware — the bottleneck is HBM bandwidth for moving the $2N$ parameters (A and D vectors) per factor. Fusing multiple ACDC factors into a single kernel (keeping intermediates in registers/SMEM) is critical
- **No tensor core utilization**: DCT/elementwise operations don't map to tensor cores (MMA instructions), limiting peak throughput on modern hardware. Monarch matrices (trick 076) are preferred when tensor core utilization matters

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft

class ACDCLayer(nn.Module):
    """Single ACDC factor: y = x * A * DCT * D * IDCT.

    Uses real FFT to compute DCT via Makhoul's method:
    DCT(x) can be computed from FFT of a reordered version of x.

    For GPU efficiency, the single-call fused kernel keeps
    intermediates in shared memory, achieving minimum 8N bytes
    memory traffic. This implementation uses cuFFT for portability.
    """

    def __init__(self, N, init_std=0.061):
        super().__init__()
        self.N = N
        # Spatial scaling (A diagonal)
        # Initialize near identity for deep cascades
        self.a = nn.Parameter(torch.ones(N) +
                              torch.randn(N) * init_std)
        # Frequency scaling (D diagonal)
        self.d = nn.Parameter(torch.ones(N) +
                              torch.randn(N) * init_std)
        # Bias (added after D, before IDCT)
        self.bias = nn.Parameter(torch.zeros(N))

    def dct(self, x):
        """Type-II DCT via FFT (Makhoul's method)."""
        N = x.shape[-1]
        # Reorder: even indices first, then odd indices reversed
        v = torch.zeros_like(x)
        v[..., :N//2+N%2] = x[..., 0::2]           # even indices
        v[..., N//2+N%2:] = x[..., 2*(N//2)-1::(-2)] # odd reversed
        V = fft.rfft(v, dim=-1)
        # Twiddle factors
        k = torch.arange(V.shape[-1], device=x.device, dtype=x.dtype)
        W = torch.exp(-1j * torch.pi * k / (2 * N))
        return (V * W).real * (2.0 / N) ** 0.5

    def idct(self, X):
        """Inverse Type-II DCT via IFFT."""
        N = X.shape[-1]
        k = torch.arange(N, device=X.device, dtype=X.dtype)
        W = torch.exp(1j * torch.pi * k / (2 * N))
        V = X.to(torch.complex64) * W * (N / 2.0) ** 0.5
        v = fft.irfft(V, n=N, dim=-1)
        # Reverse reordering
        x = torch.zeros_like(v)
        x[..., 0::2] = v[..., :N//2+N%2]
        x[..., 1::2] = v[..., N-1:N//2+N%2-1:-1] if N > 1 else v[..., :0]
        return x

    def forward(self, x):
        """x: (batch, N) -> y: (batch, N)"""
        # Step 1: Scale in spatial domain
        h1 = x * self.a
        # Step 2: DCT to frequency domain
        h2 = self.dct(h1)
        # Step 3: Scale in frequency domain + bias
        h3 = h2 * self.d + self.bias
        # Step 4: IDCT back to spatial domain
        y = self.idct(h3)
        return y


class ACDCStack(nn.Module):
    """K cascaded ACDC factors with interleaved ReLU.

    Architecture: ACDC -> ReLU -> Permute -> ACDC -> ReLU -> ...

    Permutations between layers break coherence and improve
    expressivity. 12-16 factors suffice for ImageNet-scale tasks.
    """

    def __init__(self, N, K=12, use_permutations=True):
        super().__init__()
        self.layers = nn.ModuleList([ACDCLayer(N) for _ in range(K)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(K - 1)])

        # Random fixed permutations between layers
        self.permutations = []
        if use_permutations:
            for _ in range(K - 1):
                perm = torch.randperm(N)
                self.register_buffer(
                    f'perm_{len(self.permutations)}', perm)
                self.permutations.append(perm)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relus[i](x)
                if self.permutations:
                    perm = getattr(self, f'perm_{i}')
                    x = x[:, perm]
        return x


# Arithmetic intensity analysis
def compute_arithmetic_intensity(N):
    """Compute AI for ACDC layer (FLOPs per byte loaded from HBM)."""
    flops = 4 * N + 5 * N * torch.log2(torch.tensor(float(N)))
    bytes_moved = 8 * N  # minimum: 2 diag vectors * 4 bytes each
    return (flops / bytes_moved).item()

# For N=1024: AI ≈ 6.8 → memory-bound on modern GPUs
# For N=4096: AI ≈ 8.0 → still memory-bound
# Key insight: fuse multiple ACDC factors to amortize memory loads
```

## References

- Moczulski, M., Denil, M., Appleyard, J. & de Freitas, N. "ACDC: A Structured Efficient Linear Layer" ICLR 2016. arXiv:1511.05946
- Huhtanen, M. & Perämäki, A. "Factoring matrices into the product of circulant and diagonal matrices" J. Fourier Analysis and Applications, 21:1018–1033, 2015
- Araujo, A. et al. "Understanding and Training Deep Diagonal Circulant Neural Networks" ECAI 2020. arXiv:1901.10255
- Cheng, Y. et al. "An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections" ICCV 2015
- Feng, X. & Liao, S. "CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices" arXiv:2510.25323, 2025
- Dao, T. et al. "Monarch: Expressive Structured Matrices for Efficient and Accurate Training" ICML 2022

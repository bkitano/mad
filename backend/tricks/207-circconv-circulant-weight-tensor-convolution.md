# 207: CircConv: Circulant Weight Tensor Convolution

**Category**: decomposition
**Gain type**: efficiency
**Source**: Liao, Li, Zhao, Qiu, Wang & Yuan "CircConv: A Structured Convolution with Low Complexity" (AAAI 2019, arXiv:1902.11268)
**Paper**: [papers/circconv-structured-convolution.pdf]
**Documented**: 2026-02-16

## Description

CircConv imposes **circulant structure on the 4D weight tensor** of convolutional layers along the input/output channel dimensions. The key idea: given a convolutional kernel tensor $\mathcal{W} \in \mathbb{R}^{W_1 \times H_1 \times C_0 \times C_2}$ (spatial width × height × input channels × output channels), the weight tensor is constrained so that its channel-dimension slices are **cyclic shifts** of each other. This means the entire weight tensor is defined by a smaller **base tensor** $\mathcal{W}' \in \mathbb{R}^{W_1 \times H_1 \times N \times S}$ where $C_0 = RN$ and $C_2 = SN$, reducing parameters by a factor of $N$.

The circulant structure along channel dimensions enables the forward and backward passes to be computed using the **t-product** (tensor-tensor product based on circular convolution along tubal fibers), which can be accelerated via **FFT along the circulant dimension**. This reduces the time complexity of the channel-dimension multiplication from $O(N^2)$ to $O(N \log N)$ per spatial position.

The approach differs from block-circulant weight matrices (trick 013) by operating directly on the **4D weight tensor** without flattening to 2D, preserving the spatial structure and enabling both train-from-scratch and re-training from pre-trained non-circulant models via a conversion algorithm.

## Mathematical Form

**Standard Convolution:**

For input $\mathcal{X} \in \mathbb{R}^{W_0 \times H_0 \times C_0}$, kernel $\mathcal{W} \in \mathbb{R}^{W_1 \times H_1 \times C_0 \times C_2}$, output $\mathcal{Y} \in \mathbb{R}^{W_2 \times H_2 \times C_2}$:

$$
\mathcal{Y}(w_2, h_2, c_2) = \sum_{w_1=1}^{W_1} \sum_{h_1=1}^{H_1} \sum_{c_0=1}^{C_0} \mathcal{X}(w_2 - w_1, h_2 - h_1, c_0) \cdot \mathcal{W}(w_1, h_1, c_0, c_2)
$$

**Circulant Weight Tensor Construction:**

Let $N$ be the circulant block size. Partition $C_0 = R \times N$ and $C_2 = S \times N$. The full weight tensor $\mathcal{W}$ is defined by a base tensor $\mathcal{W}' \in \mathbb{R}^{W_1 \times H_1 \times RN \times S}$:

$$
\mathcal{W}(w_1, h_1, c_0, c_2) = \mathcal{W}'(w_1, h_1, p, q)
$$

where $p = c_0$, $q = \lfloor c_2 / N \rfloor$, subject to the circulant constraint $c_0 - c_2 \equiv p \pmod{N}$. In other words, the $(c_0, c_2)$ slice of $\mathcal{W}$ depends only on $(c_0 - c_2) \bmod N$, making the channel-to-channel mapping circulant.

**Parameter Reduction:**

$$
\text{Parameters}_{\text{standard}} = W_1 \times H_1 \times C_0 \times C_2
$$

$$
\text{Parameters}_{\text{CircConv}} = W_1 \times H_1 \times R \times N \times S = \frac{W_1 \times H_1 \times C_0 \times C_2}{N}
$$

Compression ratio: $N\times$ fewer parameters.

**Fast Forward Propagation via FFT:**

Expressing convolution as fiber-slice products, define $N_k = ((k-1)N+1, \ldots, kN)$ for $k = 1, \ldots, \max(R, S)$:

$$
\mathcal{Y}(w_2, h_2, N_i) = \sum_{w_1=1}^{W_1} \sum_{h_1=1}^{H_1} \sum_{j=1}^{R} \mathcal{X}(w_2 - w_1, h_2 - h_1, N_j) * \mathcal{W}(w_1, h_1, N_j, N_i)
$$

where $*$ denotes circular convolution along the $N$-dimensional fiber. Since $\mathcal{W}$ is circulant in this dimension, the circular convolution becomes pointwise multiplication in the Fourier domain:

$$
\mathcal{Y}(w_2, h_2, N_i) = \text{ifft}\left(\sum_{w_1=1}^{W_1} \sum_{h_1=1}^{H_1} \sum_{j=1}^{R} \text{fft}\big(\mathcal{X}(w_2 - w_1, h_2 - h_1, N_j)\big) \circ \text{fft}\big(\mathcal{W}'(w_1, h_1, N_j, N_i)\big)\right)
$$

where $\circ$ is element-wise multiplication and fft/ifft are 1D transforms along the $N$-dimensional channel fiber.

**Time complexity reduction:** The circular convolution fiber product costs $O(N^2)$ naively but $O(N \log N)$ via FFT.

**Fast Backward Propagation:**

Weight gradient:

$$
\frac{\partial L}{\partial \mathcal{W}'(w_1, h_1, N_j, i)} = \text{ifft}\left(\sum_{w_2=1}^{W_2} \sum_{h_2=1}^{H_2} \text{fft}\left(\frac{\partial L}{\partial \mathcal{Y}(w_2, h_2, N_i)}\right) \circ \text{fft}(\mathbf{x}_j')\right)
$$

Input gradient:

$$
\frac{\partial L}{\partial \mathcal{X}(x, y, N_j)} = \text{ifft}\left(\sum_{w_1=1}^{W_1} \sum_{h_1=1}^{H_1} \sum_{i=1}^{S} \text{fft}\left(\frac{\partial L}{\partial \mathcal{Y}(w_1 + x, h_1 + y, N_i)}\right) \circ \text{fft}(\mathbf{w}_{j,i}')\right)
$$

Both backward passes also benefit from $O(N \log N)$ circular convolution via FFT.

**Converting Pre-trained Models to CircConv:**

Given a non-circulant weight tensor $\mathcal{W}$, the circulant approximation is obtained by averaging over cyclic shifts:

$$
\mathcal{W}'(w_1, h_1, p, q) = \frac{1}{N} \sum_{k=0}^{N-1} \mathcal{W}(w_1, h_1, (p+k) \bmod C_0, (q \cdot N + k) \bmod C_2)
$$

This is the **optimal Frobenius-norm circulant approximation** (analogous to T. Chan's optimal circulant preconditioner for Toeplitz matrices — see trick 084).

**Key Definitions:**

- $\mathcal{W} \in \mathbb{R}^{W_1 \times H_1 \times C_0 \times C_2}$ — 4D convolutional weight tensor
- $\mathcal{W}' \in \mathbb{R}^{W_1 \times H_1 \times RN \times S}$ — base (compressed) weight tensor
- $N$ — circulant block size (compression factor)
- $R = C_0 / N$, $S = C_2 / N$ — number of input/output channel blocks
- $*$ — circular convolution along channel fiber
- $\circ$ — element-wise (Hadamard) product

## Complexity

| Operation | Standard Conv | CircConv (naive) | CircConv (FFT) |
|-----------|-------------|-----------------|---------------|
| Parameters | $W_1 H_1 C_0 C_2$ | $W_1 H_1 C_0 C_2 / N$ | $W_1 H_1 C_0 C_2 / N$ |
| Forward FLOPs | $W_1 H_1 C_0 C_2 W_2 H_2$ | $W_1 H_1 R S N^2 W_2 H_2$ | $W_1 H_1 R S N \log N \cdot W_2 H_2$ |
| Compression ratio | $1\times$ | $N\times$ (params) | $N\times$ (params), $\frac{N}{\log N}\times$ (FLOPs) |

**FLOP reduction factor:** $N / \log N$ for the channel-dimension operations.

**Memory:** $O(W_1 H_1 C_0 C_2 / N)$ for weight storage — $N\times$ reduction.

**Measured results (from paper):**

| Model | Method | Top-1 Accuracy | Params (M) | FLOPs (G) |
|-------|--------|---------------|------------|-----------|
| WRN-22 (CIFAR-10) | Baseline | 95.72% | 17.16 | 3.23 |
| WRN-22 (CIFAR-10) | CircConv N=4 | 95.55% | 4.29 | 0.84 |
| WRN-22 (CIFAR-10) | CircConv N=8 | 95.15% | 2.15 | 0.44 |
| WRN-22 (CIFAR-10) | CircConv N=16 | 94.37% | 1.07 | 0.24 |

For WRN-22 with $N = 4$: **4× parameter reduction** and **3.85× FLOP reduction** with only 0.17% accuracy drop.

## Applicability

- **CNN compression for deployment**: Direct plug-in replacement for any convolutional layer. The circulant structure can be imposed at initialization (train from scratch) or via re-training from a pre-trained model, making it practical for compressing existing architectures
- **Vision Transformer convolutional stems**: The convolutional patch embedding and downsampling layers in hybrid ViTs can use CircConv for parameter-efficient operation
- **Sequence model token mixers with channel structure**: Any operation that mixes channels via a structured matrix can use the circulant constraint along the channel dimension, benefiting from FFT-accelerated channel mixing
- **Lightweight/mobile networks**: The $N\times$ parameter and FLOP reduction is complementary to other efficiency techniques (depthwise separable convolution, quantization) and can be stacked
- **Efficient fine-tuning**: The circulant structure can be imposed as a constraint during fine-tuning, providing parameter-efficient adaptation similar to CDVFT (trick 024) but at the convolution level rather than fully-connected level
- **Multi-head attention weight compression**: Attention projection matrices ($W_Q, W_K, W_V, W_O$) have a natural "head" dimension that maps to the circulant block structure

## Limitations

- **Accuracy degradation at high compression**: At $N = 16$ ($16\times$ compression), accuracy drops by ~1.35% on CIFAR-10 for WRN-22. The circulant constraint limits expressiveness — not all weight patterns can be well-approximated by circulant tensors
- **Requires $N | C_0$ and $N | C_2$**: The circulant block size must divide both channel dimensions. When it does not, zero-padding is needed, which wastes compute and memory
- **FFT overhead for small $N$**: When $N$ is small (e.g., $N = 2$ or $N = 4$), the FFT overhead may not justify the reduction compared to direct computation. The benefit is most pronounced for $N \geq 8$
- **Not directly compatible with depthwise convolution**: Depthwise separable convolution has $C_0 = C_2 = 1$ per group, eliminating the channel dimension where circulant structure operates. CircConv is complementary but not applicable to the depthwise step
- **Irregular memory access patterns**: The circulant indexing pattern $(c_0 - c_2) \bmod N$ introduces non-contiguous memory access when computing without FFT. With FFT, the access pattern is contiguous but requires complex arithmetic
- **No tensor core utilization**: The FFT-based channel mixing cannot use tensor cores (which require GEMM structure). For moderate channel counts where tensor-core GEMM would be fast, CircConv's FFT path may be slower despite fewer FLOPs

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft
import math

class CircConv2d(nn.Module):
    """Circulant convolutional layer.

    Imposes circulant structure on the weight tensor along
    input/output channel dimensions, reducing parameters by
    factor N and enabling FFT-accelerated channel mixing.

    Args:
        in_channels: number of input channels (must be divisible by N)
        out_channels: number of output channels (must be divisible by N)
        kernel_size: spatial kernel size
        N: circulant block size (compression factor)
        padding: spatial padding
        use_fft: use FFT for channel mixing (True) or naive (False)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 N=4, padding=0, use_fft=True):
        super().__init__()
        assert in_channels % N == 0, f"in_channels must be divisible by N={N}"
        assert out_channels % N == 0, f"out_channels must be divisible by N={N}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N = N
        self.R = in_channels // N   # number of input channel blocks
        self.S = out_channels // N   # number of output channel blocks
        self.padding = padding
        self.use_fft = use_fft

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Base weight tensor: (kH, kW, R*N, S) = (kH, kW, C_0, S)
        # Only N independent parameters per (R, S) block
        self.base_weight = nn.Parameter(
            torch.randn(kernel_size[0], kernel_size[1], in_channels, self.S)
            / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _expand_circulant_weight(self):
        """Expand base tensor to full circulant weight tensor.

        The full weight W(w1, h1, c0, c2) = W'(w1, h1, c0, c2 // N)
        subject to c0 - c2 ≡ c0 (mod N).

        For FFT mode, we work with the base tensor directly.
        """
        kH, kW = self.kernel_size
        W_full = torch.zeros(kH, kW, self.in_channels, self.out_channels,
                            device=self.base_weight.device)

        for s in range(self.S):
            for shift in range(self.N):
                # Output channel index: s * N + shift
                c2 = s * self.N + shift
                # Input channel mapping: circulant shift
                c0_indices = torch.arange(self.in_channels,
                                         device=self.base_weight.device)
                # The circulant constraint: W[c0, c2] = W'[c0, s]
                # when c0 mod N == shift
                mask = (c0_indices % self.N) == (shift % self.N)
                # For circulant: W(c0, c2) = W'(c0_shifted, s)
                c0_shifted = (c0_indices - shift) % self.in_channels
                W_full[:, :, :, c2] = self.base_weight[:, :,
                                                        c0_shifted, s]

        return W_full

    def forward_fft(self, x):
        """FFT-accelerated forward pass.

        Computes convolution using circular convolution along
        channel fibers, accelerated by FFT.

        Args:
            x: (B, C_0, H, W) input tensor

        Returns:
            y: (B, C_2, H_out, W_out) output tensor
        """
        B, C0, H, W = x.shape
        N = self.N
        R, S = self.R, self.S
        kH, kW = self.kernel_size

        # Reshape input to expose circulant blocks: (B, R, N, H, W)
        x_blocks = x.view(B, R, N, H, W)

        # FFT along circulant dimension (dim=2)
        x_fft = fft.fft(x_blocks, dim=2)  # (B, R, N, H, W) complex

        # Reshape base weight to (kH, kW, R, N, S)
        w_blocks = self.base_weight.view(kH, kW, R, N, S)

        # FFT along circulant dimension of weight
        w_fft = fft.fft(w_blocks, dim=3)  # (kH, kW, R, N, S) complex

        # Spatial convolution + channel mixing in Fourier domain
        # For each frequency bin n in [0, N):
        #   Y_fft[:, s, n, :, :] = sum_r conv2d(X_fft[:, r, n], W_fft[:,:, r, n, s])
        y_fft = torch.zeros(B, S, N, H - kH + 1 + 2*self.padding,
                           W - kW + 1 + 2*self.padding,
                           dtype=torch.complex64, device=x.device)

        # Pad input for spatial convolution
        if self.padding > 0:
            x_fft = torch.nn.functional.pad(
                x_fft, [self.padding]*4, mode='constant', value=0
            )

        # Pointwise multiply in frequency domain (replaces circular conv)
        for n in range(N):
            for r in range(R):
                for s in range(S):
                    # Spatial conv of frequency-domain slices
                    x_slice = x_fft[:, r, n]  # (B, H, W)
                    w_slice = w_fft[:, :, r, n, s]  # (kH, kW)
                    # Use unfold for spatial convolution
                    conv_result = torch.nn.functional.conv2d(
                        x_slice.unsqueeze(1).real,
                        w_slice.unsqueeze(0).unsqueeze(0).real,
                        padding=0
                    ) - torch.nn.functional.conv2d(
                        x_slice.unsqueeze(1).imag,
                        w_slice.unsqueeze(0).unsqueeze(0).imag,
                        padding=0
                    ) + 1j * (torch.nn.functional.conv2d(
                        x_slice.unsqueeze(1).real,
                        w_slice.unsqueeze(0).unsqueeze(0).imag,
                        padding=0
                    ) + torch.nn.functional.conv2d(
                        x_slice.unsqueeze(1).imag,
                        w_slice.unsqueeze(0).unsqueeze(0).real,
                        padding=0
                    ))
                    y_fft[:, s, n] += conv_result.squeeze(1)

        # IFFT to get output in spatial domain
        y_blocks = fft.ifft(y_fft, dim=2).real  # (B, S, N, H_out, W_out)

        # Reshape to (B, C_2, H_out, W_out)
        y = y_blocks.reshape(B, self.out_channels, *y_blocks.shape[3:])

        return y + self.bias.view(1, -1, 1, 1)

    def forward(self, x):
        if self.use_fft and self.N >= 4:
            return self.forward_fft(x)
        else:
            # Fallback: expand to full weight and use standard conv
            W_full = self._expand_circulant_weight()
            # Reshape to standard conv weight: (C_2, C_0, kH, kW)
            W_conv = W_full.permute(3, 2, 0, 1)
            return torch.nn.functional.conv2d(
                x, W_conv, self.bias, padding=self.padding
            )


@torch.no_grad()
def convert_to_circulant(conv_layer, N=4):
    """Convert a pre-trained Conv2d to CircConv2d.

    Uses the optimal Frobenius-norm circulant approximation:
    average over cyclic shifts along channel dimensions.

    Args:
        conv_layer: nn.Conv2d to convert
        N: circulant block size

    Returns:
        CircConv2d with approximated weights
    """
    C2, C0, kH, kW = conv_layer.weight.shape

    circ_conv = CircConv2d(
        C0, C2, (kH, kW), N=N,
        padding=conv_layer.padding[0],
        use_fft=True
    )

    # Compute optimal circulant approximation
    W = conv_layer.weight.data  # (C2, C0, kH, kW)
    W = W.permute(2, 3, 1, 0)  # (kH, kW, C0, C2)

    S = C2 // N
    base = torch.zeros(kH, kW, C0, S, device=W.device)

    for s in range(S):
        for shift in range(N):
            c2 = s * N + shift
            # Average over cyclic positions
            c0_shifted = (torch.arange(C0, device=W.device) + shift) % C0
            base[:, :, :, s] += W[:, :, c0_shifted, c2] / N

    circ_conv.base_weight.data = base

    if conv_layer.bias is not None:
        circ_conv.bias.data = conv_layer.bias.data.clone()

    return circ_conv
```

## References

- Liao, S., Li, Z., Zhao, L., Qiu, Q., Wang, Y. & Yuan, B. "CircConv: A Structured Convolution with Low Complexity" Proceedings of AAAI-19, 2019. arXiv:1902.11268
- Kilmer, M.E. & Martin, C.D. "Factorization strategies for third-order tensors" Linear Algebra Appl. 435:641-658, 2011
- Cheng, Y., Yu, F.X., Feris, R.S., Kumar, S., Choudhary, A. & Chang, S.F. "An exploration of parameter redundancy in deep networks with circulant projections" ICCV 2015
- Pan, V. "Structured Matrices and Polynomials: Unified Superfast Algorithms" Birkhäuser, 2001
- Ding, C. et al. "CirCNN: Accelerating and compressing deep neural networks using block-circulant weight matrices" MICRO 2017

# 093: PolyHankel: Polynomial Multiplication Derived Convolution

**Category**: kernel
**Gain type**: efficiency
**Source**: Xu, Zhang, Cheng & Li "An Efficient Polynomial Multiplication Derived Implementation of Convolution in Neural Networks" (CGO 2025)
**Paper**: [papers/polyhankel-polynomial-convolution.pdf]
**Documented**: 2026-02-15

## Description

PolyHankel reinterprets 2D convolution in neural networks as a **polynomial multiplication coefficient-finding problem**, exploiting the doubly blocked Hankel structure of the im2col matrix. The key insight is that the im2col matrix — the standard data layout transformation used by cuDNN's GEMM-based convolution — is a **doubly blocked Hankel matrix** (a block matrix where each block is itself a Hankel matrix, with blocks arranged in a Hankel pattern). This structure enables a carefully constructed polynomial mapping where:

1. The input image elements become coefficients of an **input polynomial** $A(t)$
2. The kernel elements become coefficients of a **kernel polynomial** $U(t)$ (constructed using a mirror-symmetry property of Hankel matrices)
3. The convolution output elements are specific coefficients of the **product polynomial** $P(t) = A(t) \times U(t)$

The polynomial multiplication is then solved via a **single 1D FFT** on the input polynomial, a **single 1D FFT** on the kernel polynomial, element-wise multiplication, and a **single 1D IFFT** — avoiding the data redundancy of im2col (which duplicates input elements across overlapping patches) and the multiple 2D FFT passes of traditional FFT-based convolution (which requires separate FFTs for each row and column).

The result is an algorithm that achieves up to **34.6%** speedup over the next best cuDNN method on RTX 3090Ti, **43.1%** on A10G, and **33.6%** on V100, while using fewer floating point operations and fewer memory transactions than all competing methods.

## Mathematical Form

**Naive 2D Convolution:**

For input $A$ of dimension $[I_h, I_w]$ and kernel $U$ of dimension $[K_h, K_w]$:

$$
D_{i,j} = \text{conv}_{2D}(A, U)_{(i,j)} = \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} A[i+u, j+v] \cdot U[u, v]
$$

Output dimensions: $O_h = I_h - K_h + 1$, $O_w = I_w - K_w + 1$.

**Step 1: Input Polynomial Construction**

Map each element $A_{i,j}$ to coefficient of $t^k$ where $k = i \times I_w + j$:

$$
A(t) = \sum_{i=0}^{O_h + K_h - 2} \sum_{j=0}^{O_w + K_w - 2} a_{i,j} \, t^{(O_w + K_w - 1) \times i + j}
$$

For a $5 \times 5$ input with $3 \times 3$ kernel:

$$
A(t) = a_{0,0}t^0 + a_{0,1}t^1 + a_{0,2}t^2 + a_{0,3}t^3 + a_{0,4}t^4 + a_{1,0}t^5 + a_{1,1}t^6 + \cdots + a_{4,4}t^{24}
$$

**Step 2: Kernel Polynomial Construction via Hankel Mirror Symmetry**

The kernel polynomial is constructed by exploiting the Hankel matrix property that element indices along each row are mirror-symmetric to the reverse of the first row's degree sequence. Specifically:

$$
U(t) = \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} u_{i,j} \, t^{-(O_w + K_w - 1) \times i - j + (O_w + K_w - 1) \times K_h - O_h - 1}
$$

The construction uses the **reverse of the first-row degree vector**: if the first row of the im2col matrix $A_{im2col}^t$ has degree sequence $\overrightarrow{RD}_{1st}$, then:

$$
U_{im2col}^t = U_{im2col} \odot \text{reverse}(\overrightarrow{RD}_{1st})
$$

where $\odot$ is element-wise product. This ensures that the inner product of each row of $A_{im2col}^t$ with $U_{im2col}^t$ produces a **single unique polynomial term** per row, with the coefficient being exactly the corresponding convolution output value.

**Step 3: Polynomial Multiplication via FFT**

$$
P(t) = A(t) \times U(t)
$$

Solved via the standard FFT-based polynomial multiplication:

$$
R = \text{IFFT}(\text{FFT}(P) \cdot \text{FFT}(Q))
$$

where $P = [p_0, p_1, \ldots, p_{N-1}, \underbrace{0, \ldots, 0}_{M-1}]$ and $Q = [q_0, q_1, \ldots, q_{M-1}, \underbrace{0, \ldots, 0}_{N-1}]$ are zero-padded coefficient vectors.

**Step 4: Extract Convolution Output**

The output $D$ is extracted from specific coefficients of $R$:

$$
d_{i,j} = p_{(O_w + K_w - 1) \times i + j + (O_w + K_w - 1) \times K_h - O_w}
$$

for $0 \le i < O_h$, $0 \le j < O_w$.

**General Formulas (arbitrary input/kernel size):**

$$
A(t) = \sum_{i=0}^{O_h + K_h - 2} \sum_{j=0}^{O_w + K_w - 2} a_{i,j} \, t^{(O_w + K_w - 1) \times i + j}
$$

$$
U(t) = \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} u_{i,j} \, t^{-(O_w + K_w - 1) \times i - j + (O_w + K_w - 1) \times K_h - O_h - 1}
$$

**Key Definitions:**

- $I_h, I_w$ — input height and width
- $K_h, K_w$ — kernel height and width
- $O_h = I_h - K_h + 1$, $O_w = I_w - K_w + 1$ — output height and width
- $C$ — number of input channels
- $K$ — number of output kernels
- $P$ — padding size
- $N$ — mini-batch size

## Complexity

**Time Complexity:**

| Method | Time Complexity |
|--------|----------------|
| im2col + GEMM | $K_h \times K_w \times O_h \times O_w$ |
| Traditional FFT | $(I_w + K_w)(I_h + K_h)(\log(I_h + K_h) + \log(I_w + K_w)) \times 2 + (I_h + K_h)(I_w + K_w) + (I_w + K_w)(I_h + K_h)(\log(I_h + K_h) + \log(I_w + K_w))$ |
| **PolyHankel** | $(I_h \times I_w + K_h \times I_w) \log(I_h \times I_w + K_h \times I_w) + (I_h \times I_w + K_h \times I_w) \log(I_h \times I_w + K_h \times I_w) + (I_h \times I_w + K_h \times I_w)$ |

Simplified: PolyHankel requires **two 1D FFTs** (on input and kernel polynomials), **one element-wise multiplication**, and **one 1D IFFT** — all on vectors of size $\approx I_h \times I_w + K_h \times I_w$.

**Space Complexity:**

| Method | Space Complexity |
|--------|-----------------|
| im2col + GEMM | $K_h \times K_w \times O_h \times O_w$ (expanded im2col matrix) |
| Traditional FFT | $(I_h + K_h)(I_w + K_w) + (I_h + K_h)(I_w + K_w) + (I_h + K_h)(I_w + K_w)$ |
| **PolyHankel** | $(I_h \times I_w + K_h \times I_w) + (I_h \times I_w + K_h \times I_w) + (I_h \times I_w + K_h \times I_w)$ |

PolyHankel has lower space complexity than both FFT (which pads to $(I_h + K_h)(I_w + K_w)$ in 2D) and im2col (which expands by factor $K_h \times K_w$).

**Multi-channel complexity:** For $C$ input channels, PolyHankel performs FFT on separate channels and sums outputs:

$$
O(C \cdot K_h I_w \log(K_h I_w) + C \cdot K_h I_w \log(K_h I_w) + C \cdot O_h)
$$

**Measured Speedups (API-level, kernel size 3–15, batch 128):**

| GPU | Max Speedup over Next Best | Average End-to-End Speedup |
|-----|---------------------------|---------------------------|
| GeForce 3090Ti | 34.6% | 1.36× |
| A10G | 43.1% | 1.59× |
| V100 | 33.6%–48.9% | 2.08× |

## Applicability

- **CNN convolution layers**: Direct replacement for cuDNN's im2col+GEMM or FFT convolution. Most beneficial for input sizes > 100 and kernel sizes 3–15 (the most common range in practical CNNs)
- **Graph Neural Networks (GNNs)**: GNN applications where convolution is the performance bottleneck can benefit from the same polynomial reformulation
- **Multi-channel convolution**: Handles multiple input channels by performing FFT on separate channels and summing, with kernel polynomials merged across channels using non-overlapping degree assignments
- **Streaming / overlap-save**: The method naturally supports the overlap-save technique for batched processing, with zero-padding at batch boundaries
- **Vision Transformers with convolutional stems**: The convolutional layers in hybrid ViT architectures (patch embedding, downsampling) can use PolyHankel
- **Any layer using im2col internally**: Since the method exploits the doubly blocked Hankel structure of im2col matrices, it applies wherever im2col-based convolution is used

## Limitations

- **Not universally fastest**: For very small input sizes (< 100) or very large kernel sizes (> 15), im2col+GEMM or traditional FFT may still be faster — no single convolution method dominates all parameter configurations
- **No Tensor Core support (yet)**: The current implementation does not exploit Tensor Cores, which cuDNN's GEMM-based methods can leverage. Future work could implement FFT and element-wise multiplication via matrix form on Tensor Cores
- **FFT size sensitivity**: Performance depends on the FFT size, which is determined by $I_h \times I_w + K_h \times I_w$. When this size crosses a power-of-two boundary, the FFT must pad to the next $2^a \times 3^b \times 5^c \times 7^d$ for cuFFT optimality, causing performance "jumps"
- **1D FFT only**: The method converts 2D convolution into a single 1D polynomial multiplication, so it requires a 1D FFT of size proportional to the total input area — for very high-resolution inputs, this 1D FFT can become large
- **Implementation complexity**: Constructing the polynomial degree mappings (the "map" data structure) requires careful index arithmetic based on the doubly blocked Hankel structure

## Implementation Notes

```python
import torch
import torch.fft as fft
import math

class PolyHankelConv2d(torch.nn.Module):
    """2D Convolution via PolyHankel polynomial multiplication.

    Converts 2D convolution into polynomial multiplication using
    the doubly blocked Hankel structure of the im2col matrix.
    Only requires: 1 FFT(input) + 1 FFT(kernel) + 1 IFFT(product).
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.Kh, self.Kw = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, self.Kh, self.Kw)
            / math.sqrt(in_channels * self.Kh * self.Kw)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def _build_degree_map(self, Ih, Iw, Kh, Kw):
        """Build the polynomial degree mapping from doubly blocked
        Hankel structure.

        The map assigns each unique element in the im2col matrix
        an incrementing integer, traversing in an L-shaped path:
        first row left-to-right, then rightmost column top-to-bottom,
        for each block and across blocks.
        """
        Ow = Iw - Kw + 1
        Oh = Ih - Kh + 1
        stride = Ow + Kw - 1  # = Iw (for no padding)

        # Input polynomial: A(t) = sum a_{i,j} * t^(stride*i + j)
        # for 0 <= i < Ih, 0 <= j < Iw
        input_degrees = {}
        for i in range(Ih):
            for j in range(Iw):
                input_degrees[(i, j)] = stride * i + j

        # Kernel polynomial: constructed via reverse of first-row degrees
        # The first row of A^t_{im2col} has degrees from the last column
        max_input_deg = stride * (Ih - 1) + (Iw - 1)
        kernel_degrees = {}
        for i in range(Kh):
            for j in range(Kw):
                # Mirror symmetry from Hankel structure
                kernel_degrees[(i, j)] = max_input_deg - (stride * i + j)

        # Result extraction: d_{i,j} = coefficient of t^target_deg
        result_degrees = {}
        for i in range(Oh):
            for j in range(Ow):
                result_degrees[(i, j)] = max_input_deg + (stride * i + j) \
                    - (stride * (Kh - 1) + (Kw - 1))  # simplified offset

        return input_degrees, kernel_degrees, result_degrees

    def forward(self, x):
        """
        x: (N, C, Ih, Iw) input tensor
        Returns: (N, K, Oh, Ow) output tensor
        """
        N, C, Ih, Iw = x.shape
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x, [self.padding]*4, mode='constant', value=0
            )
            Ih, Iw = Ih + 2*self.padding, Iw + 2*self.padding

        Oh = Ih - self.Kh + 1
        Ow = Iw - self.Kw + 1
        stride = Iw  # = Ow + Kw - 1

        # Polynomial size (padded to efficient FFT length)
        poly_size = stride * Ih + self.Kh * Iw  # approximate
        fft_size = 1
        while fft_size < poly_size:
            fft_size *= 2  # pad to power of 2 for cuFFT efficiency

        # Build input polynomial: flatten with stride-based indexing
        # Each pixel a_{i,j} -> coefficient of t^(stride*i + j)
        input_poly = torch.zeros(N, C, fft_size, device=x.device)
        for i in range(Ih):
            input_poly[:, :, stride*i : stride*i + Iw] = x[:, :, i, :]

        # Build kernel polynomial using Hankel mirror symmetry
        max_deg = stride * (Ih - 1) + (Iw - 1)
        kernel_poly = torch.zeros(self.out_channels, C, fft_size,
                                   device=x.device)
        for i in range(self.Kh):
            for j in range(self.Kw):
                deg = max_deg - (stride * i + j)
                kernel_poly[:, :, deg] = self.weight[:, :, i, j]

        # FFT-based polynomial multiplication
        # P_hat = FFT(input), Q_hat = FFT(kernel)
        P_hat = fft.rfft(input_poly, n=fft_size, dim=-1)    # (N, C, F)
        Q_hat = fft.rfft(kernel_poly, n=fft_size, dim=-1)    # (K, C, F)

        # R = IFFT(P_hat * Q_hat), summed over channels
        # (N, 1, C, F) * (1, K, C, F) -> sum over C -> (N, K, F)
        R_hat = (P_hat.unsqueeze(1) * Q_hat.unsqueeze(0)).sum(dim=2)
        R = fft.irfft(R_hat, n=fft_size, dim=-1)

        # Extract convolution output from result polynomial coefficients
        output = torch.zeros(N, self.out_channels, Oh, Ow, device=x.device)
        for i in range(Oh):
            output[:, :, i, :] = R[:, :, max_deg + stride*i :
                                        max_deg + stride*i + Ow]

        return output + self.bias.view(1, -1, 1, 1)

# NOTE: The actual high-performance implementation uses:
# 1. Pre-computed degree maps stored as index arrays
# 2. cuFFT for FFT/IFFT operations
# 3. Overlap-save for batched processing
# 4. Merged kernel polynomials across channels with non-overlapping degrees
# 5. FFT sizes padded to nearest multiple of 2 (empirically optimal for cuFFT)
```

## References

- Xu, H., Zhang, Y., Cheng, Z. & Li, X. "An Efficient Polynomial Multiplication Derived Implementation of Convolution in Neural Networks" Proceedings of CGO '25, March 2025, Las Vegas. ACM. doi:10.1145/3696443.3708947
- Zhang, Y. & Li, X. "Fast Convolutional Neural Networks with Fine-Grained FFTs" Proceedings of PACT '20, 2020. ACM. doi:10.1145/3410463.3414642
- Partington, J.R. "An Introduction to Hankel Operators" Cambridge University Press, 1988
- Lee, D. "Fast Multiplication of a Recursive Block Toeplitz Matrix by a Vector and its Application" Journal of Complexity 2, 295–305, 1986
- Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. "Introduction to Algorithms" 2nd ed., MIT Press, 2001

# 014: Block-Circulant Power-of-Two Quantization (Multiplier-Free Inference)

**Category**: kernel
**Gain type**: efficiency
**Source**: Qin et al. "Accelerating Deep Neural Networks by Combining Block-Circulant Matrices and Low-Precision Weights" (Electronics, MDPI, 2019); Wang et al. "Energy-Efficient, High-Performance, Highly-Compressed DNN Design Using Block-Circulant Matrices" (2018)
**Paper**: [papers/block-circulant-low-precision.pdf]
**Documented**: 2026-02-15

## Description

Standard block-circulant neural network layers (CirCNN) use FFT to accelerate matrix-vector multiplication, reducing complexity from $O(n^2)$ to $O(n \log n)$. However, **FFT requires complex floating-point arithmetic**, which limits quantization to at least 12 bits and demands dedicated FFT hardware (multipliers, butterfly units). The **block-circulant power-of-two (PoT) quantization** trick eliminates both problems by directly computing the circulant matrix-vector product in the spatial domain with **power-of-two quantized weights**, replacing all multiplications with **bit-shift operations**.

The key insight is a two-level synergy between circulant structure and PoT quantization:

1. **Circulant structure** reduces the number of unique weight parameters from $O(n^2)$ to $O(n)$ — each $k \times k$ circulant block is defined by a single $k$-vector. This is a **parameter-level** compression (e.g., $16\times$ for block size 16).

2. **Power-of-two quantization** represents each weight as $\hat{w} = \text{sign}(w) \cdot 2^n$ where $n$ is a small integer (typically encoded in 3–4 bits). This enables **multiplier-free computation**: the product $x \cdot \hat{w}$ becomes a bit-shift of $x$ by $n$ positions, plus a sign flip. This is a **bit-level** compression (e.g., $8\times$ from 32-bit to 4-bit).

The combined compression is **multiplicative**: block size $k$ with $b$-bit PoT encoding gives a total compression ratio of $k \times (32/b)$. For $k = 16$ and $b = 4$, this yields $128\times$ compression with less than 1% accuracy loss on MNIST and CIFAR-10.

Crucially, unlike FFT-based block-circulant inference, the PoT approach operates **entirely in the spatial domain** using direct circulant matrix-vector products. The circulant structure is exploited through a **block pseudo-circulant reorganization**: each $k \times k$ circulant block is subdivided into $\alpha \times \alpha$ sub-blocks (where $\alpha = k/16$ for a 16-wide SIMD), and the cyclic shift structure is implemented by a hardware circular shift register feeding row processing elements.

## Mathematical Form

**Block-Circulant FC Layer:**

For weight matrix $W \in \mathbb{R}^{a \times b}$ partitioned into $p \times q$ circulant blocks of size $k \times k$ (where $p = a/k$, $q = b/k$):

$$
W = \begin{bmatrix} W_{1,1}^c & W_{1,2}^c & \cdots & W_{1,q}^c \\ W_{2,1}^c & W_{2,2}^c & \cdots & W_{2,q}^c \\ \vdots & \vdots & \ddots & \vdots \\ W_{p,1}^c & W_{p,2}^c & \cdots & W_{p,q}^c \end{bmatrix}
$$

Each $W_{i,j}^c \in \mathbb{R}^{k \times k}$ is a circulant matrix defined by primitive vector $w_{ij} = (w_{ij}^1, w_{ij}^2, \ldots, w_{ij}^k)$.

**Power-of-Two Quantization:**

Each weight $w$ is quantized to:

$$
\hat{w} = \begin{cases} 0 & \text{if } w = 0 \\ \text{sign}(w) \cdot 2^n & \text{otherwise} \end{cases}
$$

where:

$$
n = \text{round}(\log_2 |w|), \quad n \in [n_1, n_2]
$$

The codebook for $b$-bit encoding with weights in $[-1, 1]$ is:

$$
S_n = \{0, \pm 2^{n_1}, \pm 2^{n_1+1}, \ldots, \pm 2^{n_2}\}
$$

For 4-bit indices with $n_2 = 0$ (max weight $= 1$): $S = \{0, \pm 2^{-6}, \pm 2^{-5}, \pm 2^{-4}, \pm 2^{-3}, \pm 2^{-2}, \pm 2^{-1}, \pm 1\}$.

**Multiplier-Free MAC Operation:**

The multiply-accumulate $P_{sum} = \sum_{i=1}^{m} x[i] \cdot w[i]$ becomes:

$$
P_{sum} = \sum_{i=1}^{m} x[i] \ll w_{shift}[i] \cdot (-1)^{w_{sign}[i]}
$$

where $\ll$ denotes bit-shift, $w_{shift}$ is the 3-bit shift amount, and $w_{sign}$ is the 1-bit sign. Each multiplication is replaced by a **Basic Shift Unit (BSU)**: a barrel shifter controlled by the weight index.

**Block Pseudo-Circulant Reorganization:**

For large block sizes ($k > 16$), each circulant sub-matrix $W_{ij}^c$ is partitioned into $\alpha^2$ sub-blocks of size $\gamma \times \gamma$ (where $\gamma = k/\alpha$, typically $\gamma = 16$):

$$
\begin{bmatrix} Y_i^1 \\ Y_i^2 \\ \vdots \\ Y_i^\alpha \end{bmatrix} = \begin{bmatrix} P_{ij}^1 & P_{ij}^2 & \cdots & P_{ij}^\alpha \\ S_\gamma P_{ij}^\alpha & P_{ij}^1 & \cdots & P_{ij}^{\alpha-1} \\ S_\gamma P_{ij}^{\alpha-1} & S_\gamma P_{ij}^\alpha & \cdots & P_{ij}^{\alpha-2} \\ \vdots & \vdots & \ddots & \vdots \\ S_\gamma P_{ij}^2 & S_\gamma P_{ij}^3 & \cdots & P_{ij}^1 \end{bmatrix} \begin{bmatrix} X_j^1 \\ X_j^2 \\ \vdots \\ X_j^\alpha \end{bmatrix}
$$

where $P_{ij}^\beta$ are $\gamma \times \gamma$ "short circulant" sub-blocks and $S_\gamma$ is the $\gamma \times \gamma$ cyclic shift operator. Each $P_{ij}^\beta X_j^\beta$ is a small circulant matrix-vector product (C-MV) computable by the Block-MV hardware unit.

**Training Strategy (Two-Stage):**

1. **Stage 1**: Train with full-precision block-circulant weights (no FFT needed — direct spatial domain computation)
2. **Stage 2**: Quantize weights to PoT: $W^q = \text{PoT}(W^t)$, then retrain with quantized weights using straight-through estimator for gradients

**Key Definitions:**

- $k$ — circulant block size (compression = $k\times$ at the parameter level)
- $b$ — number of bits per weight index (compression = $32/b$ at the bit level)
- $n_1, n_2$ — range of PoT exponents ($n \in [n_1, n_2]$)
- $S_n$ — PoT codebook: $\{0, \pm 2^{n_1}, \ldots, \pm 2^{n_2}\}$ with $2^b - 1$ entries
- $w_{sign}$ — 1-bit sign indicator
- $w_{shift}$ — $(b-1)$-bit shift amount
- BSU — Basic Shift Unit (barrel shifter replacing a multiplier)
- B-MV — Block Matrix-Vector unit (processes $16 \times 16$ sub-blocks)
- $\gamma$ — sub-block size for reorganization (typically 16)
- $\alpha = k / \gamma$ — number of sub-blocks per dimension

## Complexity

| Operation | Dense (FP32) | Block Circ + FFT | Block Circ + PoT |
|-----------|-------------|------------------|------------------|
| Parameters | $O(ab)$ | $O(ab/k)$ | $O(ab/k)$ |
| Storage | $32 \cdot ab$ bits | $32 \cdot ab/k$ bits | $b \cdot ab/k$ bits |
| Multiplications | $ab$ | $\sim ab \log k / k$ | **0** |
| Additions | $ab$ | $\sim ab \log k / k$ | $ab/k$ |
| Hardware | Multiplier array | FFT butterfly | **Shift + Add only** |

**Compression ratios (from paper):**

| Dataset | Block $k$ | Bits $b$ | Compression | Accuracy (baseline → compressed) |
|---------|-----------|----------|-------------|-----------------------------------|
| MNIST | 16 | 3 | 171× | 98.47% → 97.06% |
| CIFAR-10 | 256 | 3 | 2731× | 92.00% → 91.20% |
| ImageNet | 16 | 4 | 128× | 56.3%/79.0% → 49.3%/75.4% (Top-1/5) |

**Hardware performance (BPCA accelerator, TSMC 40nm @ 800 MHz):**

| Metric | Value |
|--------|-------|
| Throughput | 409.6 GOPS |
| Energy efficiency | 5.3 TOPS/W |
| Logic area | 0.16 mm² |
| Total area (with SRAM) | 12.88 mm² |
| Power | 77.09 mW |

**Processing time per layer:**

$$
t_l = \frac{ab}{\gamma^2 f_{clk}}
$$

For a $4096 \times 4096$ layer with $\gamma = 16$ at 800 MHz: $t_l = 81.9 \mu s$.

**Memory:** Storage per layer is $b \cdot pqk$ bits $= b \cdot ab/k$ bits. For 4-bit weights with $k = 16$: $ab/4$ bits, a $128\times$ reduction over 32-bit dense.

## Applicability

- **Edge / embedded inference**: The multiplier-free design is ideal for FPGAs, ASICs, and microcontrollers where multiplier units are expensive. The configurable block size allows trading accuracy for compression on a per-layer basis
- **Transformer feedforward layers**: The two large projections ($W_1, W_2$) in transformer FFN blocks are natural candidates — they account for 2/3 of transformer parameters and can tolerate block-circulant structure with minimal accuracy loss
- **Fully-connected classifiers**: The final FC layers in CNNs (which are the most parameter-heavy, e.g., 96% of AlexNet parameters) benefit most from this combined compression
- **On-device LLM inference**: For deploying language models on mobile/edge devices, this trick enables extreme compression of the linear projection layers while maintaining shift-only arithmetic
- **Energy-constrained applications**: Bit-shift operations consume ~5.5× less energy than fixed-point multiplications and ~100× less than floating-point multiplications, making this critical for battery-powered devices
- **RNN/LSTM weight matrices**: Recurrent weight matrices can be block-circulant + PoT quantized, enabling low-power sequential inference

## Limitations

- **Incompatible with FFT acceleration**: The PoT quantization must operate in the spatial domain — FFT transforms the weights into the Fourier domain where the PoT structure is destroyed. This means the $O(n \log n)$ FFT speedup is sacrificed in favor of multiplier-free $O(n^2/k)$ spatial-domain computation
- **Accuracy degrades for large block sizes**: Block size $k = 256$ with 3-bit quantization loses 0.8% accuracy on CIFAR-10, while $k = 16$ loses only 0.4%. ImageNet with $k = 16$ and 4-bit loses 7% Top-1 accuracy
- **Two-stage training**: Requires first training with full-precision block-circulant weights, then quantizing and retraining — adding training complexity
- **Limited to FC/linear layers**: The paper applies this only to fully-connected layers. Convolutional layers with small kernels ($3 \times 3$) don't benefit as much from circulant structure
- **Custom hardware required**: The full benefit requires a custom ASIC or FPGA with BSU units. On GPUs, bit-shift operations don't map well to the CUDA execution model (which is optimized for FP16/INT8 multiply-accumulate)
- **Fixed codebook**: The PoT codebook $S_n$ is fixed and may not match the weight distribution well. Adaptive codebooks (e.g., from learned quantization) could improve accuracy but would sacrifice the shift-only hardware benefit

## Implementation Notes

```python
import torch
import torch.nn as nn
import math

def power_of_two_quantize(w, n_bits=4):
    """Quantize weights to nearest power of two.

    Each weight becomes sign(w) * 2^n where n is integer.

    Args:
        w: weight tensor (any shape)
        n_bits: bits per weight index (1 sign + (n_bits-1) shift)

    Returns:
        w_q: quantized weight tensor
        indices: integer indices into the PoT codebook
    """
    # Determine range: n2 = 0 (max |w| = 1), n1 = n2 - 2^(n_bits-1) + 2
    n2 = 0
    n1 = n2 - 2 ** (n_bits - 1) + 2  # e.g., -6 for 4-bit

    # Normalize weights to [-1, 1]
    w_max = w.abs().max()
    if w_max > 0:
        w_norm = w / w_max
    else:
        w_norm = w

    # Compute log2 of absolute values
    w_abs = w_norm.abs().clamp(min=2.0 ** n1)  # avoid log(0)
    log_w = torch.log2(w_abs)

    # Round to nearest integer and clamp to [n1, n2]
    n = torch.round(log_w).clamp(n1, n2).long()

    # Reconstruct quantized weights
    w_q = torch.sign(w_norm) * (2.0 ** n.float())

    # Zero out very small weights
    w_q[w_norm.abs() < 2.0 ** (n1 - 1)] = 0

    # Rescale
    w_q = w_q * w_max

    return w_q


class BlockCirculantPoTLinear(nn.Module):
    """Block-circulant linear layer with power-of-two quantized weights.

    Multiplier-free inference: all multiplications are replaced
    by bit-shift operations.

    Args:
        in_features: input dimension
        out_features: output dimension
        block_size: circulant block size k
        n_bits: bits per quantized weight (default 4)
        quantize: whether to apply PoT quantization (False during warmup)
    """

    def __init__(self, in_features, out_features, block_size=16,
                 n_bits=4, quantize=False):
        super().__init__()
        self.k = block_size
        self.q = in_features // block_size   # input blocks
        self.p = out_features // block_size   # output blocks
        self.n_bits = n_bits
        self.quantize = quantize

        # Store only defining vectors: (p, q, k) parameters
        self.w = nn.Parameter(
            torch.randn(self.p, self.q, self.k) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        x: (batch, in_features)

        Compute y = W_circulant @ x + bias
        In spatial domain (no FFT) for PoT compatibility.
        """
        batch = x.shape[0]
        p, q, k = self.p, self.q, self.k

        # Optionally quantize weights
        w = self.w
        if self.quantize:
            # Straight-through estimator: quantize forward, copy grad backward
            w = w + (power_of_two_quantize(w, self.n_bits) - w).detach()

        # Reshape input: (batch, q, k)
        x_blocks = x.view(batch, q, k)

        # Direct circulant matrix-vector product in spatial domain
        # For each output block i, input block j:
        #   y_i = sum_j circ(w_ij) @ x_j
        #
        # circ(w) @ x can be computed as:
        #   y[r] = sum_{s=0}^{k-1} w[(r-s) mod k] * x[s]

        # Build full circulant indices for vectorized computation
        # row r, col s -> weight index (r-s) mod k
        r = torch.arange(k, device=x.device)
        s = torch.arange(k, device=x.device)
        idx = (r.unsqueeze(1) - s.unsqueeze(0)) % k  # (k, k)

        # Gather weights to form circulant matrices: (p, q, k, k)
        W_circ = w[:, :, idx]  # (p, q, k, k)

        # Matrix multiply: (batch, p, q, k, k) @ (batch, 1, q, k, 1)
        # -> (batch, p, q, k, 1) -> sum over q -> (batch, p, k)
        x_expand = x_blocks.unsqueeze(1).unsqueeze(-1)  # (batch, 1, q, k, 1)
        y_blocks = (W_circ.unsqueeze(0) @ x_expand).squeeze(-1)  # (batch, p, q, k)
        y_blocks = y_blocks.sum(dim=2)  # (batch, p, k)

        return y_blocks.reshape(batch, -1) + self.bias

    def compression_ratio(self):
        """Compute the total compression ratio."""
        dense_bits = self.p * self.q * self.k * self.k * 32
        compressed_bits = self.p * self.q * self.k * self.n_bits
        return dense_bits / compressed_bits

    def enable_quantization(self):
        """Enable PoT quantization (call after warmup training)."""
        self.quantize = True


# Simulated shift-and-add MAC (for hardware modeling)
def shift_add_mac(x, w_sign, w_shift):
    """Multiply-accumulate using only shifts and adds.

    x: (m,) activation vector (fixed-point)
    w_sign: (m,) sign bits (0 or 1)
    w_shift: (m,) shift amounts (integers)

    Returns: scalar result
    """
    # In hardware, this is a parallel array of barrel shifters
    # followed by an adder tree
    shifted = torch.where(
        w_shift >= 0,
        x * (2.0 ** w_shift),    # left shift (multiply by power of 2)
        x / (2.0 ** (-w_shift))  # right shift (divide by power of 2)
    )
    signed = torch.where(w_sign == 0, shifted, -shifted)
    return signed.sum()


# Example usage
def demo():
    layer = BlockCirculantPoTLinear(
        in_features=1024, out_features=512,
        block_size=16, n_bits=4, quantize=True
    )
    x = torch.randn(32, 1024)
    y = layer(x)
    print(f"Output shape: {y.shape}")
    print(f"Compression ratio: {layer.compression_ratio():.0f}x")
    # Expected: 128x compression (16x circulant * 8x quantization)
```

## References

- Qin, Z., Zhu, D., Zhu, X., Chen, X., Shi, Y., Gao, Y., Lu, Z., Shen, Q., Li, L. & Pan, H. "Accelerating Deep Neural Networks by Combining Block-Circulant Matrices and Low-Precision Weights" Electronics 8(1):78, 2019. doi:10.3390/electronics8010078
- Ding, C. et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" MICRO 2017. arXiv:1708.08917
- Wang, B. et al. "Energy-Efficient Neural Networks Using Block-Circulant Weight Matrices" arXiv:1702.03440, 2018
- Liao, Q. et al. "Block-Circulant Matrix Based Neural Network Training and Inference on FPGA" ACM Trans. Reconfigurable Technol. Syst., 2019
- Li, F. & Liu, B. "Ternary Weight Networks" NeurIPS Workshop, 2016
- Chen, H., Wang, Y., Xu, C., Shi, B. & Han, J. "AdderNet: Do We Really Need Multiplications in Deep Learning?" CVPR 2020

# Im2col Convolution Lowering

**Category**: algebraic
**Gain type**: efficiency
**Source**: Chellapilla et al. (2006), systematized by Anderson et al. (2017)
**Paper**: [papers/im2col-convolution-lowering.pdf]
**Documented**: 2026-02-14

## Description

Convert convolution into matrix multiplication by unrolling input patches into columns of a matrix. Instead of implementing convolution as a 7-deep loop nest, im2col "lowers" it to a single GEMM call — the most optimized operation on any hardware (CPU BLAS, GPU cuBLAS, TPU MXU). This is the canonical example of "reshape your problem to fit the hardware's sweet spot." The trade-off is memory expansion ($K^2 \times$ for the patch matrix), but the resulting GEMM runs at near-peak throughput because BLAS Level-3 operations have $O(n^3)$ compute with $O(n^2)$ data — optimal arithmetic intensity.

## Mathematical Form

**Core Operation:**

A convolution layer with input $\mathcal{I} \in \mathbb{R}^{H \times W \times C}$, kernels $\mathcal{K} \in \mathbb{R}^{M \times K \times K \times C}$, producing output $\mathcal{O} \in \mathbb{R}^{H' \times W' \times M}$:

**Step 1 — Build patch matrix (im2col):**

Extract all $K \times K \times C$ patches from the input and arrange as columns:

$$
\hat{\mathcal{I}} \in \mathbb{R}^{(H'W') \times (CK^2)}
$$

Each row of $\hat{\mathcal{I}}$ is one flattened receptive field.

**Step 2 — Reshape kernel:**

$$
\hat{\mathcal{K}} \in \mathbb{R}^{M \times (CK^2)}
$$

Each row of $\hat{\mathcal{K}}$ is one flattened filter.

**Step 3 — GEMM:**

$$
\mathcal{O}_{\text{flat}} = \hat{\mathcal{I}} \cdot \hat{\mathcal{K}}^T \in \mathbb{R}^{(H'W') \times M}
$$

Reshape to get $\mathcal{O} \in \mathbb{R}^{H' \times W' \times M}$.

**Key Definitions:**

- $H, W$ — input spatial dimensions
- $C$ — input channels
- $M$ — output channels (number of filters)
- $K$ — kernel size (assuming square)
- $H' = \lfloor(H - K + 2P)/S\rfloor + 1$, $W' = \lfloor(W - K + 2P)/S\rfloor + 1$ — output spatial dimensions
- $P$ — padding, $S$ — stride

**Memory-Efficient Variants:**

*kn2row* — eliminates the $K^2$ patch matrix entirely by decomposing into $K^2$ separate $1 \times 1$ convolutions (each is a GEMM without data replication), then shift-and-accumulate:

$$
\mathcal{O} = \sum_{r=0}^{K-1} \sum_{s=0}^{K-1} \text{shift}_{(r,s)}\left(\mathcal{K}_{:,:,r,s} \cdot \mathcal{I}\right)
$$

Memory: $O(MHW)$ additional vs. $O(CK^2HW)$ for im2col.

*Accumulating GEMM (k2r-aa)* — uses the GEMM accumulation parameter $C = \alpha AB + \beta C$ to accumulate shifted partial results directly:

$$
\mathcal{O} \mathrel{+}= \mathcal{K}_{:,:,r,s} \cdot \text{shift}_{(r,s)}(\mathcal{I})
$$

Memory: only $O(KW)$ additional.

## Complexity

| Variant | Compute | Additional Memory | Data Locality |
|---------|---------|-------------------|---------------|
| Direct loop nest | $O(MCHW K^2)$ | $O(1)$ | Poor |
| im2col + GEMM | $O(MCHW K^2)$ | $O(CK^2 HW)$ | Excellent (BLAS-3) |
| kn2row | $O(MCHW K^2)$ | $O(MHW)$ | Good |
| kn2row-aa (accum.) | $O(MCHW K^2)$ | $O(KW)$ | Good |
| MEC | $O(MCHW K^2)$ | $O(KCHW)$ | Good |

**Key insight:** All variants have the *same* arithmetic complexity. The difference is entirely in how well they map to optimized GEMM, which achieves near-peak FLOPS due to $O(n)$ compute-to-memory ratio.

**Memory:** Classical im2col expands input by $K^2\times$. For VGG-16 conv3.1 ($256 \times 56 \times 56$ input, $3 \times 3$ kernel): patch matrix is $9\times$ the input size.

## Applicability

- The **default** convolution implementation in most deep learning frameworks (PyTorch, TensorFlow, Caffe)
- All CNN architectures: VGG, ResNet, EfficientNet, ConvNeXt, etc.
- Available as `torch.nn.functional.unfold` (im2col) and `torch.nn.functional.fold` (col2im)
- Memory-efficient variants (kn2row, MEC) essential for edge/embedded deployment
- Implicit GEMM (computing patches on-the-fly without materializing) used in cuDNN for GPU

## Limitations

- Classical im2col has $O(K^2)$ memory blowup — problematic for large kernels or memory-constrained devices
- Patch building itself takes time — can be 10-30% of total convolution time
- Redundant data: overlapping patches copy the same input values multiple times
- Strided convolutions reduce patch matrix density, hurting GEMM efficiency
- For $1 \times 1$ convolutions ($K = 1$), im2col is unnecessary — convolution is already a GEMM
- Depthwise convolutions cannot benefit (no channel accumulation → no GEMM structure)

## Implementation Notes

```python
# im2col convolution lowering — PyTorch
import torch
import torch.nn.functional as F

def conv2d_via_im2col(input, weight, bias=None, stride=1, padding=0):
    """
    input:  [N, C, H, W]
    weight: [M, C, K, K]
    """
    N, C, H, W = input.shape
    M, _, K, _ = weight.shape

    # Step 1: im2col — unfold input into patch matrix
    # Output: [N, C*K*K, H'*W']
    patches = F.unfold(input, kernel_size=K, stride=stride, padding=padding)

    # Step 2: Reshape kernel to [M, C*K*K]
    kernel_flat = weight.view(M, -1)

    # Step 3: GEMM — [M, C*K*K] x [C*K*K, H'*W'] = [M, H'*W']
    output_flat = kernel_flat @ patches  # Batched over N via broadcasting

    # Step 4: Reshape back
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    output = output_flat.view(N, M, H_out, W_out)

    if bias is not None:
        output += bias.view(1, M, 1, 1)
    return output

# Memory-efficient kn2row variant
def conv2d_kn2row(input, weight, bias=None, padding=0):
    """K^2 separate 1x1 GEMMs with shift-and-accumulate."""
    N, C, H, W = input.shape
    M, _, K, _ = weight.shape
    pad_input = F.pad(input, [padding]*4)
    output = torch.zeros(N, M, H, W)
    for r in range(K):
        for s in range(K):
            # 1x1 conv = GEMM: [M, C] x [C, H*W]
            partial = weight[:, :, r, s] @ pad_input[:, :, r:r+H, s:s+W].reshape(N, C, -1)
            output += partial.view(N, M, H, W)
    return output
```

## References

- Chellapilla, K., Puri, S., Simard, P. (2006). High Performance Convolutional Neural Networks for Document Processing.
- Anderson, A., Vasudevan, A., Keane, C., Gregg, D. (2017). Low-memory GEMM-based convolution algorithms for deep neural networks. arXiv:1709.03395.
- Cho, M. & Brand, D. (2017). MEC: Memory-efficient convolution for deep neural network. ICML.
- NVIDIA cuDNN — Implicit GEMM convolution implementation.

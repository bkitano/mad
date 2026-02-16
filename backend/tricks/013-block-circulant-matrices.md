# 013: Block Circulant Matrices (FFT-Based Linear Layers)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ding et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017)
**Paper**: [papers/circnn-block-circulant.pdf]
**Documented**: 2025-02-11

## Description

Block circulant matrices replace arbitrary dense weight matrices with structured matrices composed of circulant sub-blocks. A circulant matrix is fully determined by its first row — each subsequent row is a cyclic shift of the previous one. This structure enables matrix-vector multiplication via the **circulant convolution theorem**: multiply in the Fourier domain using FFT, element-wise multiplication, and IFFT, reducing complexity from $O(n^2)$ to $O(n \log n)$.

The key GPU-relevant insight is that **FFT is a well-optimized GPU primitive** (cuFFT), so replacing dense GEMM with FFT-based circulant multiplication trades one GPU-friendly operation (GEMM) for another (FFT + element-wise multiply). The block structure provides a tunable knob: larger block sizes yield higher compression but potentially more accuracy loss, while smaller blocks preserve accuracy at lower compression.

Unlike Monarch matrices (which use BMM as the primitive) or butterfly matrices (which use sparse structured products), block circulant matrices leverage FFT as the fundamental compute kernel, making them especially efficient on hardware with dedicated FFT units.

## Mathematical Form

**Circulant Matrix:**

A circulant matrix $C \in \mathbb{R}^{k \times k}$ defined by vector $w = [w_0, w_1, \ldots, w_{k-1}]$:

$$
C = \text{circ}(w) = \begin{bmatrix} w_0 & w_1 & \cdots & w_{k-1} \\ w_{k-1} & w_0 & \cdots & w_{k-2} \\ \vdots & \vdots & \ddots & \vdots \\ w_1 & w_2 & \cdots & w_0 \end{bmatrix}
$$

**Circulant Convolution Theorem:**

$$
Cx = \text{IFFT}(\text{FFT}(w) \circ \text{FFT}(x))
$$

where $\circ$ denotes element-wise (Hadamard) multiplication.

**Block Circulant Weight Matrix:**

For weight matrix $W \in \mathbb{R}^{m \times n}$, partition into $p \times q$ blocks of size $k \times k$ where $p = m/k$, $q = n/k$:

$$
W = \begin{bmatrix} W_{11} & W_{12} & \cdots & W_{1q} \\ W_{21} & W_{22} & \cdots & W_{2q} \\ \vdots & \vdots & \ddots & \vdots \\ W_{p1} & W_{p2} & \cdots & W_{pq} \end{bmatrix}
$$

where each $W_{ij} = \text{circ}(w_{ij})$ is a $k \times k$ circulant matrix defined by vector $w_{ij} \in \mathbb{R}^k$.

**Forward propagation (FC layer):**

Partition input $x = [x_1^\top, x_2^\top, \ldots, x_q^\top]^\top$ into $q$ blocks of size $k$:

$$
a_i = \sum_{j=1}^{q} W_{ij} x_j = \sum_{j=1}^{q} \text{IFFT}(\text{FFT}(w_{ij}) \circ \text{FFT}(x_j))
$$

**Backward propagation:**

$$
\frac{\partial L}{\partial w_{ij}} = \text{IFFT}\left(\text{FFT}\left(\frac{\partial L}{\partial a_i}\right) \circ \text{FFT}(x_j')\right)
$$

$$
\frac{\partial L}{\partial x_j} = \sum_{i=1}^{p} \text{IFFT}\left(\text{FFT}\left(\frac{\partial L}{\partial a_i}\right) \circ \text{FFT}(w_{ij}')\right)
$$

where $x_j'$ and $w_{ij}'$ are reversed versions used due to the cross-correlation structure of the gradient.

**Key Definitions:**

- $k$ — block (circulant) size, the tunable compression parameter
- $p = m/k$, $q = n/k$ — number of blocks in each dimension
- $w_{ij} \in \mathbb{R}^k$ — defining vector for circulant sub-matrix $W_{ij}$
- FFT, IFFT — Fast Fourier Transform and its inverse

**Diagonalization form:**

Every circulant matrix can be diagonalized by the DFT matrix:

$$
C = F^{-1} \text{diag}(\text{FFT}(w)) F
$$

where $F$ is the $k \times k$ DFT matrix. This connects to the DPLR trick — circulant matrices are a special case of diagonal matrices in Fourier space.

## Complexity

| Operation | Dense | Block Circulant |
|-----------|-------|-----------------|
| Mat-vec multiply | $O(mn)$ = $O(n^2)$ | $O(pqk \log k)$ = $O(n \log k)$ |
| Parameters | $O(mn)$ = $O(n^2)$ | $O(pqk)$ = $O(n)$ |
| Training (per layer) | $O(n^2)$ | $O(n \log k)$ |

**Memory:** $O(n)$ parameters vs $O(n^2)$ for dense — compression ratio is $k\times$.

**Compression ratio** is directly controlled by block size $k$: larger $k$ gives $k\times$ compression. For $k = n$ (single circulant), get $n\times$ compression but may lose accuracy. For $k = 1$ (trivial blocks), no compression.

## Applicability

- **Fully-connected layers**: Direct replacement — largest storage savings since FC layers are most parameter-heavy
- **Convolutional layers**: Represent the filter matrix (across input/output channels) as block-circulant, applying FFT across channel dimension
- **Transformer FFN layers**: The two large linear projections ($W_1, W_2$) in feed-forward blocks are natural candidates
- **RNN/LSTM layers**: Weight matrices in recurrent cells can be block-circulant
- **Edge deployment**: Particularly valuable for inference on resource-constrained devices (FPGA, embedded processors)
- **Training acceleration**: Unlike many compression methods, CirCNN accelerates both training AND inference

## Limitations

- Circulant structure is a strong constraint — expressivity is fundamentally limited by the circulant eigenvalue structure (spectrum must be a DFT of real vectors)
- Block size $k$ must divide both $m$ and $n$; requires padding if dimensions don't align
- GPU FFT (cuFFT) is well-optimized but may not match peak GEMM throughput for small sizes (GEMM benefits from Tensor Cores, FFT does not)
- For very small block sizes, the overhead of FFT/IFFT calls can dominate
- No hardware-native support (unlike 2:4 sparsity which has dedicated Tensor Core instructions)
- Accuracy degradation is more noticeable on tasks requiring fine-grained weight structure (e.g., NLP may be more sensitive than vision)

## Implementation Notes

```python
import torch
import torch.fft as fft

class BlockCirculantLinear(torch.nn.Module):
    """Linear layer using block-circulant weight matrices."""

    def __init__(self, in_features, out_features, block_size):
        super().__init__()
        self.k = block_size
        self.q = in_features // block_size   # input blocks
        self.p = out_features // block_size   # output blocks

        # Store only the defining vectors (not full matrices)
        # Shape: (p, q, k) — one k-vector per circulant block
        self.w = torch.nn.Parameter(
            torch.randn(self.p, self.q, self.k) / (in_features ** 0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        # Pre-compute FFT of weights (cache for inference)
        self._w_fft = None

    def forward(self, x):
        # x: (batch, in_features)
        batch = x.shape[0]

        # Reshape input into blocks: (batch, q, k)
        x_blocks = x.view(batch, self.q, self.k)

        # FFT of input blocks: (batch, q, k)
        x_fft = fft.fft(x_blocks, dim=-1)

        # FFT of weight vectors: (p, q, k)
        w_fft = fft.fft(self.w, dim=-1)

        # Element-wise multiply and sum over q blocks
        # (batch, 1, q, k) * (1, p, q, k) -> sum over q -> (batch, p, k)
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)

        # IFFT to get output: (batch, p, k)
        out = fft.ifft(out_fft, dim=-1).real

        # Reshape to (batch, out_features)
        return out.reshape(batch, -1) + self.bias
```

## References

- Ding, C., et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017). arXiv:1708.08917
- Cheng, Y., et al. "An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections" (ICCV 2015)
- Trabelsi, C., et al. "Deep Complex Networks" (ICLR 2018)
- Sindhwani, V., et al. "Structured Transforms for Small-Footprint Deep Learning" (NeurIPS 2015)
- Suleiman, A. & Sze, V. "Real Block-Circulant Matrices and DCT-DST Algorithm for Transformer Neural Network" (Frontiers in Applied Math, 2023)

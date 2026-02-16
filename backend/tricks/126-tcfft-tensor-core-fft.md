# 126: tcFFT: Tensor Core Accelerated FFT for Block Circulant Layers

**Category**: kernel
**Gain type**: efficiency
**Source**: Li et al. "tcFFT: Accelerating Half-Precision FFT through Tensor Cores" (2021)
**Paper**: [papers/tcfft-tensor-core-fft.pdf]
**Documented**: 2026-02-15

## Description

tcFFT accelerates half-precision FFT computation by reformulating FFT merging stages as matrix multiplications that can execute on NVIDIA Tensor Cores. Standard FFT implementations (e.g., cuFFT) use CUDA cores for the butterfly operations, leaving Tensor Cores idle. Since Tensor Cores deliver 8–16× higher throughput than CUDA cores for FP16 operations (125 TFLOPS on V100, 312 TFLOPS on A100), mapping FFT onto them yields significant speedups.

**Why this matters for block circulant layers:** Block circulant matrix-vector multiplication is computed as $\text{IFFT}(\text{FFT}(w) \circ \text{FFT}(x))$, making FFT the core compute kernel. Any speedup in FFT directly translates to faster block circulant forward and backward passes. By using tcFFT instead of cuFFT, block circulant layers can leverage Tensor Cores — the same hardware units that make dense GEMM fast — closing the performance gap between structured (circulant) and dense layers.

**Key innovations:**

1. **Radix-16 base computation**: Uses $16 \times 16$ DFT as the base case, matching Tensor Core WMMA fragment dimensions. Each radix-16 DFT is computed as a $16 \times 16$ matrix multiplication via `wmma::mma_sync`.

2. **Single-element fragment manipulation**: Implements twiddle factor multiplication and data reordering by directly manipulating individual elements within Tensor Core fragments, avoiding costly global memory round-trips.

3. **In-place computation with changing data order**: Instead of maintaining a fixed data layout (which requires explicit bit-reversal permutations), tcFFT keeps data in whatever order each stage produces and adjusts the next stage's access pattern accordingly. This preserves memory coalescing without extra shuffle kernels.

4. **Batched execution**: Supports batched 1D and 2D FFT, which is exactly what block circulant layers need (batched FFT across all $p \times q$ circulant blocks simultaneously).

## Mathematical Form

**Core Idea — FFT merging as matrix multiplication:**

A length-$N$ FFT is decomposed into stages. The radix-16 merging step for a group of 16 sub-FFTs of size $N/16$ each can be written:

$$
X_{\text{merged}} = T \cdot (F_{16} \otimes I_{N/16}) \cdot X_{\text{sub}}
$$

where:
- $F_{16}$ is the $16 \times 16$ DFT matrix
- $T$ is a diagonal matrix of twiddle factors $e^{-2\pi i k / N}$
- $I_{N/16}$ is the identity (each sub-FFT is independent)

The key insight is that the $F_{16}$ DFT is a $16 \times 16$ complex matrix multiply, which maps directly to Tensor Core's `wmma::mma_sync` with $16 \times 16 \times 16$ fragments.

**Radix-16 DFT via Tensor Cores:**

$$
X[k] = \sum_{n=0}^{15} x[n] \cdot W_{16}^{nk}, \quad k = 0, 1, \ldots, 15
$$

where $W_{16} = e^{-2\pi i / 16}$. In matrix form:

$$
\mathbf{X} = F_{16} \cdot \mathbf{x}
$$

This is computed as two real matrix multiplications (for real and imaginary parts):

$$
\text{Re}(\mathbf{X}) = \text{Re}(F_{16}) \cdot \text{Re}(\mathbf{x}) - \text{Im}(F_{16}) \cdot \text{Im}(\mathbf{x})
$$
$$
\text{Im}(\mathbf{X}) = \text{Re}(F_{16}) \cdot \text{Im}(\mathbf{x}) + \text{Im}(F_{16}) \cdot \text{Re}(\mathbf{x})
$$

Each of these is a $16 \times 16$ half-precision matmul → 4 `wmma::mma_sync` calls per complex DFT.

**Merging kernel for larger FFTs:**

For radix-$r$ merging of $r$ sub-FFTs (where $r = 2$ for radix-2 stages after the base):

$$
X[k] = \sum_{j=0}^{r-1} W_N^{jk} \cdot X_j[k \bmod (N/r)]
$$

The radix-2 merging is also mapped to Tensor Cores using a $16 \times 16$ batched butterfly structure.

**Application to block circulant multiply:**

For block circulant layer with block size $k$, the forward pass becomes:

$$
a_i = \text{tcIFFT}\left(\sum_{j=1}^{q} \text{tcFFT}(w_{ij}) \circ \text{tcFFT}(x_j)\right)
$$

where `tcFFT` and `tcIFFT` use Tensor Cores instead of CUDA cores.

**Key Definitions:**

- $N$ — FFT length (equals block size $k$ in circulant context)
- $F_{16}$ — the $16 \times 16$ DFT matrix (fixed constant)
- $W_N^k = e^{-2\pi i k / N}$ — twiddle factors
- WMMA — Warp Matrix Multiply-Accumulate (Tensor Core API)
- Fragment — $16 \times 16$ half-precision matrix tile stored across 32 threads in a warp

## Complexity

| Operation | cuFFT (CUDA Cores) | tcFFT (Tensor Cores) |
|-----------|---------------------|----------------------|
| FP16 FFT of length $N$ | $O(N \log N)$ FLOPs at CUDA core rate | $O(N \log N)$ FLOPs at Tensor Core rate |
| Peak hardware throughput (V100) | 15.7 TFLOPS (FP16 CUDA) | 125 TFLOPS (FP16 TC) |
| Peak hardware throughput (A100) | 78 TFLOPS (FP16 CUDA) | 312 TFLOPS (FP16 TC) |
| Batched FFT ($B$ transforms) | $O(BN \log N)$ | $O(BN \log N)$ with better HW utilization |

**Measured speedups over cuFFT:**
- V100: **1.29×–3.24×** across FFT sizes
- A100: **1.10×–3.03×** across FFT sizes

**Memory:** Same asymptotic memory as cuFFT. In-place computation avoids extra buffers for bit-reversal permutations.

**Precision:** ~1.76% relative error (comparable to cuFFT's half-precision mode).

## Applicability

- **Block circulant linear layers**: Direct drop-in replacement of cuFFT calls with tcFFT for both forward and backward passes, benefiting from Tensor Core throughput
- **FFT-based convolutions**: Any layer using spectral-domain convolution (e.g., FNet, AFNO/FNO for PDEs)
- **State space models**: S4/S5 models using FFT for computing convolution kernels from DPLR parameters
- **Signal processing layers**: Learned spectral filters, frequency-domain attention
- **Pretraining acceleration**: Half-precision FFT is naturally compatible with mixed-precision training (FP16 forward, FP32 accumulation for gradients)

**Best block sizes for Tensor Core utilization:**
- Powers of 2 that are multiples of 16: $k \in \{16, 32, 64, 128, 256, 512, 1024, \ldots\}$
- Block size $k = 16$ is the sweet spot for maximum Tensor Core utilization (single radix-16 DFT, no merging stages needed)

## Limitations

- **Half-precision only**: tcFFT operates in FP16; not suitable for operations requiring FP32/FP64 precision (though FP16 is standard for pretraining)
- **Size constraints**: Best performance for FFT lengths that are powers of 2 and ≥ 16; odd or prime-length FFTs not supported
- **NVIDIA-specific**: Requires Tensor Core hardware (Volta/V100 or newer); not portable to AMD or other accelerators
- **Library maturity**: Research prototype (open source) vs. cuFFT's production-grade library; may require integration effort
- **Radix-16 overhead**: For very small FFTs ($N < 16$), the radix-16 base case is wasteful; cuFFT may be faster for $N < 32$
- **Twiddle factor precision**: Half-precision twiddle factors accumulate rounding errors for very large $N$; accuracy degrades for $N > 8192$
- **A100 gains smaller than V100**: cuFFT on A100 already utilizes some Tensor Core paths, reducing tcFFT's relative advantage

## GPU Efficiency Analysis

### Memory Access Pattern
- **Coalesced access**: In-place computation with changing data order ensures each stage reads/writes consecutive memory locations within warps
- **Cache-friendly**: Radix-16 base operates on 16-element vectors that fit in registers; merging stages access contiguous blocks
- **Arithmetic intensity**: High — Tensor Core matmuls have $O(N^3)$ FLOPs on $O(N^2)$ data for each $16 \times 16$ tile
- **Shared memory**: Intermediate FFT stages use shared memory for inter-warp communication; fits comfortably in SM shared memory

### Parallelism
- **SM saturation**: Batched FFT (across $p \times q$ circulant blocks × batch size) provides abundant independent work
- **No warp divergence**: All threads in a warp execute the same WMMA instruction
- **Tensor Core mapping**: By design — radix-16 DFT IS a $16 \times 16$ matmul
- **No sequential bottlenecks**: All FFT stages are parallel across independent transforms in the batch

### Baseline Comparison
- **Measured against cuFFT** (the production GPU FFT library) — not a toy baseline
- **1.29×–3.24× speedup on V100**, **1.10×–3.03× on A100** (real hardware measurements)
- Largest gains on medium-sized batched FFTs (the exact use case for block circulant layers)

### Hardware Considerations
- **Tensor Core utilization**: Core design principle — maps FFT butterfly to WMMA $16 \times 16 \times 16$ fragments
- **Warp specialization**: Single-element fragment manipulation avoids store-load-modify pattern
- **Register pressure**: Moderate — each warp holds 2–4 fragments ($16 \times 16$ FP16 tiles)
- **Occupancy**: Good for batched transforms; each block processes one or more independent FFTs

## Implementation Notes

```python
import torch
import torch.fft as fft

# Conceptual integration: tcFFT-accelerated block circulant layer
# In practice, tcFFT is a C++/CUDA library called via custom extension

class TensorCoreBlockCirculantLinear(torch.nn.Module):
    """Block circulant linear layer using Tensor Core FFT.

    Key insight: replace torch.fft.fft/ifft with tcFFT calls
    for FP16 inputs, gaining Tensor Core throughput.

    Block size should be a multiple of 16 for optimal TC utilization.
    """

    def __init__(self, in_features, out_features, block_size=64):
        super().__init__()
        assert block_size % 16 == 0, "Block size must be multiple of 16 for Tensor Cores"
        self.k = block_size
        self.q = in_features // block_size
        self.p = out_features // block_size

        # Store defining vectors in FP16 for Tensor Core compatibility
        self.w = torch.nn.Parameter(
            torch.randn(self.p, self.q, self.k, dtype=torch.float16) / (in_features ** 0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))

    def forward(self, x):
        # x: (batch, in_features) in FP16
        batch = x.shape[0]

        # Reshape input into blocks: (batch, q, k)
        x_blocks = x.view(batch, self.q, self.k)

        # === These FFT calls would be replaced by tcFFT ===
        # tcFFT uses Tensor Cores for the butterfly operations
        # API: tcfft_batched_1d(input, n, batch_count)
        x_fft = fft.fft(x_blocks, dim=-1)  # → tcFFT call
        w_fft = fft.fft(self.w, dim=-1)     # → tcFFT call (can be cached)

        # Element-wise multiply (Hadamard product in frequency domain)
        # This is a simple pointwise op — already fast on GPU
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)

        # IFFT to get output — also via Tensor Cores
        out = fft.ifft(out_fft, dim=-1).real  # → tcIFFT call

        return out.reshape(batch, -1) + self.bias

# Performance expectation for block circulant layer:
# - FFT/IFFT are ~60-70% of total compute in block circulant multiply
# - tcFFT gives 1.3-3.2x speedup on FFT portion
# - Net layer speedup: ~1.2-2.2x over cuFFT-based implementation
# - Best gains when: block_size in {64, 128, 256}, large batch, FP16 training
```

## References

- Li, S., et al. "tcFFT: Accelerating Half-Precision FFT through Tensor Cores" (2021). arXiv:2104.11471
- Ding, C., et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017)
- NVIDIA. "WMMA (Warp Matrix Multiply-Accumulate) API Documentation"
- Cotter, F. & Kingsbury, N. "A Learnable ScatterNet: Locally Invariant Convolutional Layers" (2019) — FFT in neural networks
- NVIDIA. "cuFFT Library User's Guide" — baseline comparison

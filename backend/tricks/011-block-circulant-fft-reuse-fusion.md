# 011: Block Circulant FFT Reuse and Kernel Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Dong, S. "Exploring GPU Acceleration of DNNs using Block Circulant Matrices" (PhD Thesis, Northeastern University, 2020)
**Paper**: [papers/gpu-block-circulant-training.pdf]
**Documented**: 2026-02-15

## Description

Training neural networks with block circulant weight matrices involves computing many batched FFTs, element-wise multiplications, and IFFTs across forward and backward passes. The naive implementation launches separate GPU kernels for each FFT, Hadamard product, IFFT, and summation, leading to excessive kernel launch overhead, redundant memory accesses, and poor GPU occupancy.

This trick identifies three **flow-level optimizations** that reduce redundant computation in the block circulant forward/backward pass, and two **kernel-level optimizations** that improve GPU hardware utilization. Together, they address the key bottleneck: block circulant training is **memory-bandwidth bound** with many small operations, not compute-bound like dense GEMM.

**Why this matters for pretraining:** The naive block circulant implementation can be slower than dense GEMM despite lower FLOP count because it suffers from:
1. Many kernel launches (≥5 per layer per direction: FFT, Hadamard, IFFT, sum, reorder)
2. Poor memory coalescing from strided access patterns
3. Low SM occupancy when block size $k$ is not a multiple of 32
4. Redundant FFT computations between forward and backward passes

The optimizations in this trick directly address these GPU-specific bottlenecks, making block circulant layers competitive with dense layers in wall-clock time while retaining their parameter efficiency advantage.

## Mathematical Form

**Standard block circulant forward pass:**

For weight matrix partitioned into $p \times q$ circulant blocks of size $k$, input batch of size $B$:

$$
a_{b,i} = \sum_{j=1}^{q} \text{IFFT}\left(\text{FFT}(w_{ij}) \circ \text{FFT}(x_{b,j})\right), \quad i = 1, \ldots, p
$$

**Standard backward pass (weight gradient):**

$$
\frac{\partial L}{\partial w_{ij}} = \sum_{b=1}^{B} \text{IFFT}\left(\text{FFT}\left(\frac{\partial L}{\partial a_{b,i}}\right) \circ \text{FFT}(x'_{b,j})\right)
$$

**Standard backward pass (input gradient):**

$$
\frac{\partial L}{\partial x_{b,j}} = \sum_{i=1}^{p} \text{IFFT}\left(\text{FFT}\left(\frac{\partial L}{\partial a_{b,i}}\right) \circ \text{FFT}(w'_{ij})\right)
$$

where $x'_{b,j}$ and $w'_{ij}$ denote reversed (flipped) vectors needed for the cross-correlation gradient.

---

### Flow Optimization 1: Swap IFFT and Summation

**Key identity:** IFFT is linear, so summation and IFFT commute:

$$
a_{b,i} = \sum_{j=1}^{q} \text{IFFT}(\hat{w}_{ij} \circ \hat{x}_{b,j}) = \text{IFFT}\left(\sum_{j=1}^{q} \hat{w}_{ij} \circ \hat{x}_{b,j}\right)
$$

where $\hat{w}_{ij} = \text{FFT}(w_{ij})$ and $\hat{x}_{b,j} = \text{FFT}(x_{b,j})$.

**Savings:** Reduces from $q$ IFFT calls to **1 IFFT call** per output block $i$. Total IFFTs reduced from $pq$ to $p$ per batch element.

Similarly for the backward pass:

$$
\frac{\partial L}{\partial w_{ij}} = \text{IFFT}\left(\sum_{b=1}^{B} \widehat{\delta}_{b,i} \circ \hat{x}'_{b,j}\right)
$$

$$
\frac{\partial L}{\partial x_{b,j}} = \text{IFFT}\left(\sum_{i=1}^{p} \widehat{\delta}_{b,i} \circ \hat{w}'_{ij}\right)
$$

where $\widehat{\delta}_{b,i} = \text{FFT}(\partial L / \partial a_{b,i})$.

### Flow Optimization 2: Eliminate Reversal via Conjugate FFT Property

The backward pass requires reversed vectors $x'$ and $w'$. Naively, this means an explicit reorder operation (non-coalesced memory access). Instead, use the conjugate property:

$$
\text{FFT}(x') = \overline{\text{FFT}(x)}
$$

where $\overline{\cdot}$ denotes complex conjugation (flip sign of imaginary part).

**Savings:** Replaces an explicit memory reorder kernel with a sign flip on the imaginary component during the Hadamard product — essentially **free** (folded into the element-wise multiply).

### Flow Optimization 3: Reuse FFT Results Across Forward/Backward

The forward pass computes $\hat{w}_{ij} = \text{FFT}(w_{ij})$ and $\hat{x}_{b,j} = \text{FFT}(x_{b,j})$. The backward pass needs these same FFT results (or their conjugates). Instead of recomputing, **cache the FFT outputs** from the forward pass:

**Forward pass computes and caches:**
- $\hat{w}_{ij} = \text{FFT}(w_{ij})$ for all $i, j$ — reused in input gradient
- $\hat{x}_{b,j} = \text{FFT}(x_{b,j})$ for all $b, j$ — reused in weight gradient

**Backward pass reuses:**
- Weight grad: uses cached $\hat{x}_{b,j}$ (conjugated) and freshly computed $\widehat{\delta}_{b,i}$
- Input grad: uses cached $\hat{w}_{ij}$ (conjugated) and the same $\widehat{\delta}_{b,i}$

**Savings:** Eliminates $pq + Bq$ FFT computations in the backward pass. The $\widehat{\delta}_{b,i}$ (FFT of output gradients) is computed once and shared between weight and input gradient computation.

**Combined optimized dataflow:**

$$
\text{Forward: } a_{b,i} = \text{IFFT}\left(\sum_{j} \hat{w}_{ij} \circ \hat{x}_{b,j}\right), \quad \text{cache } \hat{w}, \hat{x}
$$

$$
\text{Backward (weights): } \nabla_{w_{ij}} = \text{IFFT}\left(\sum_{b} \widehat{\delta}_{b,i} \circ \overline{\hat{x}_{b,j}}\right)
$$

$$
\text{Backward (inputs): } \nabla_{x_{b,j}} = \text{IFFT}\left(\sum_{i} \widehat{\delta}_{b,i} \circ \overline{\hat{w}_{ij}}\right)
$$

---

### Kernel Optimization: Memory Layout and Occupancy

**GPU occupancy problem:** FFT output has $\lfloor k/2 \rfloor + 1$ complex values (Hermitian symmetry for real inputs). When $k$ is not a power of 2 minus 1, this value is not a multiple of 32 (warp size), causing warp underutilization.

**Data Reuse metric (DR):** Maximum number of threads that can reuse the same data element:

$$
\text{DR} = \max_{\text{data element } d} |\{t : \text{thread } t \text{ accesses } d\}|
$$

**Memory Distance metric (MD):** Average distance between memory accesses by consecutive threads:

$$
\text{MD} = \frac{1}{T} \sum_{t=0}^{T-2} |\text{addr}(t+1) - \text{addr}(t)|
$$

Low MD (ideally MD = 1) indicates coalesced access.

**Two kernel optimization strategies:**

**O1 (Occupancy-First):** Set thread block size = $\text{lcm}(\lfloor k/2 \rfloor + 1, 32)$ to ensure full warp utilization. Then optimize data layout within this constraint for reuse.

**O2 (Data-Reuse-First):** Set thread block size to maximize DR, then pad/align to minimize warp divergence. Better when data reuse dominates over occupancy.

**Memory layout reorganization:** Rearrange the $p \times q \times (k/2+1)$ tensor of FFT coefficients so that the Hadamard product accesses consecutive memory addresses within each warp. Instead of storing as `[block_i][block_j][freq]`, store as `[freq_chunk][block_i][block_j]` where each chunk is 32-aligned.

## Complexity

| Operation | Naive BCM | Optimized BCM | Savings Factor |
|-----------|-----------|---------------|----------------|
| Forward IFFTs | $Bpq$ | $Bp$ | $q\times$ fewer |
| Backward FFTs (recompute) | $Bq + pq$ | $Bp$ (only $\delta$) | $(q + p)/p \approx q\times$ |
| Reorder operations | $Bq + pq$ | $0$ | Eliminated |
| Total FFT/IFFT calls per training step | $O(Bpq + Bp + Bq + pq)$ | $O(Bp + Bq + pq)$ | Up to $2\times$ reduction |
| Kernel launches per layer | $\geq 5$ per direction | $2$–$3$ (fused) | $2$–$3\times$ fewer |

**Memory:** Caching FFT results costs $O(pq(k/2+1) + Bq(k/2+1))$ extra memory in complex FP16. For typical transformer FFN dimensions ($m = n = 4096$, $k = 64$, $B = 32$), this is ~70 MB — acceptable given GPU HBM capacity.

**Net throughput improvement:** The flow optimizations alone reduce total FFT/IFFT operations by ~40–60%. Combined with kernel fusion (fewer launches, better coalescing), real-world speedups of **1.5×–2.5×** over naive block circulant implementation are achievable.

## Applicability

- **Block circulant linear layers**: Direct application to any BCM-based layer (FFN projections, attention projections, embedding layers)
- **Transformer pretraining**: FFN layers in transformers are the primary target — two large linear projections per block that can be replaced with block circulant layers using these optimizations
- **Mixed-precision training**: All optimizations are compatible with FP16 forward / FP32 gradient accumulation
- **Large batch training**: FFT reuse savings scale with batch size $B$ — larger batches amortize the cached FFT memory cost
- **Combinable with tcFFT**: The flow optimizations (swap IFFT/sum, eliminate reversal, cache FFTs) are orthogonal to the choice of FFT implementation — can use tcFFT for the individual FFT calls to stack both speedups

## Limitations

- **Memory overhead from caching**: Storing FFT results for backward pass increases peak memory by $O((pq + Bq) \cdot k)$ complex values. For very large models or batch sizes, this trades memory for compute.
- **Implementation complexity**: Requires custom CUDA kernels for the fused Hadamard-sum-IFFT operation; not a simple drop-in with PyTorch's autograd
- **Block size constraints**: Occupancy optimization (O1) works best when $\lfloor k/2 \rfloor + 1$ has small LCM with 32; pathological block sizes (e.g., $k = 62$, giving $\lfloor k/2 \rfloor + 1 = 32$) can cause very large thread blocks
- **Kernel fusion limits**: Fusing FFT + Hadamard + sum + IFFT into a single kernel is only practical for small $k$ (where the entire FFT fits in shared memory); for $k > 1024$, must fall back to separate kernel launches
- **Not a standard library**: These optimizations require custom kernel development; no off-the-shelf library provides them (unlike cuBLAS for GEMM)

## GPU Efficiency Analysis

### Memory Access Pattern
- **Coalesced access**: Memory layout reorganization ensures consecutive threads access consecutive frequency-domain elements during Hadamard product
- **Cache-friendly**: FFT results cached in HBM are accessed in a streaming pattern (each used once in weight gradient, once in input gradient) — good L2 cache behavior
- **Arithmetic intensity**: Low for individual operations (element-wise multiply is $O(1)$ FLOPs per element loaded), but kernel fusion amortizes memory round-trips by chaining FFT → multiply → sum → IFFT in shared memory
- **HBM bandwidth savings**: Eliminating reversal operations and reducing IFFT count directly reduces HBM traffic — the primary bottleneck for block circulant layers

### Parallelism
- **SM saturation**: $B \times p$ independent output blocks provide ample parallel work for forward pass; $p \times q$ independent weight gradients for backward
- **No warp divergence**: Occupancy-first (O1) strategy ensures thread blocks are multiples of 32
- **Tensor Core compatibility**: Orthogonal — these optimizations reduce the number of FFT calls, while tcFFT accelerates each individual FFT call
- **No sequential bottlenecks**: All three flow optimizations preserve full parallelism; summation across $q$ or $B$ is a parallel reduction

### Hardware Considerations
- **Shared memory**: For $k \leq 256$, the entire FFT + Hadamard + partial sum can be computed in shared memory (256 KB on H100)
- **Register pressure**: Moderate — each thread holds $O(1)$ complex values during element-wise operations
- **Kernel launch reduction**: From $\geq 15$ launches per layer (5 per direction × 3 directions) to $\leq 9$ with fusion — significant for small layers where launch overhead dominates

## Implementation Notes

```python
import torch
import torch.fft as fft

class OptimizedBlockCirculantLinear(torch.nn.Module):
    """Block circulant layer with FFT reuse and kernel fusion optimizations.

    Implements three flow optimizations from Dong (2020):
    1. Swap IFFT/sum: single IFFT after accumulating in frequency domain
    2. Conjugate trick: eliminate explicit reversal operations
    3. FFT caching: reuse forward-pass FFTs in backward pass
    """

    def __init__(self, in_features, out_features, block_size=64):
        super().__init__()
        self.k = block_size
        self.q = in_features // block_size
        self.p = out_features // block_size
        self.w = torch.nn.Parameter(
            torch.randn(self.p, self.q, self.k) / (in_features ** 0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        batch = x.shape[0]
        x_blocks = x.view(batch, self.q, self.k)

        # Compute FFTs (these get cached for backward via autograd)
        x_fft = fft.rfft(x_blocks, dim=-1)       # (B, q, k//2+1) complex
        w_fft = fft.rfft(self.w, dim=-1)          # (p, q, k//2+1) complex

        # === Optimization 1: Accumulate in frequency domain, SINGLE IFFT ===
        # Instead of q separate IFFTs per output block, do Hadamard + sum first
        # (B, 1, q, k//2+1) * (1, p, q, k//2+1) -> sum over q -> (B, p, k//2+1)
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)

        # Single IFFT per output block (not q IFFTs)
        out = fft.irfft(out_fft, n=self.k, dim=-1)  # (B, p, k)

        return out.reshape(batch, -1) + self.bias


class OptimizedBlockCirculantFunction(torch.autograd.Function):
    """Custom autograd function implementing all three flow optimizations."""

    @staticmethod
    def forward(ctx, x, w, bias, k, p, q):
        batch = x.shape[0]
        x_blocks = x.view(batch, q, k)

        # Compute and CACHE FFTs for backward pass (Optimization 3)
        x_fft = fft.rfft(x_blocks, dim=-1)
        w_fft = fft.rfft(w, dim=-1)

        # Optimization 1: accumulate in freq domain, single IFFT
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)
        out = fft.irfft(out_fft, n=k, dim=-1)

        # Cache FFT results — NOT the original tensors
        ctx.save_for_backward(x_fft, w_fft)
        ctx.k, ctx.p, ctx.q, ctx.batch = k, p, q, batch

        return out.reshape(batch, -1) + bias

    @staticmethod
    def backward(ctx, grad_output):
        x_fft, w_fft = ctx.saved_tensors  # Reuse cached FFTs!
        k, p, q, batch = ctx.k, ctx.p, ctx.q, ctx.batch

        grad = grad_output.view(batch, p, k)
        grad_fft = fft.rfft(grad, dim=-1)  # Only NEW FFT needed

        # === Weight gradient ===
        # Optimization 2: use conjugate instead of reversal
        # grad_w_ij = IFFT(sum_b(grad_fft_bi * conj(x_fft_bj)))
        # (B, p, 1, freq) * (B, 1, q, freq).conj() -> sum over B -> (p, q, freq)
        w_grad_fft = (grad_fft.unsqueeze(2) * x_fft.unsqueeze(1).conj()).sum(dim=0)
        w_grad = fft.irfft(w_grad_fft, n=k, dim=-1)  # Optimization 1: single IFFT

        # === Input gradient ===
        # Optimization 2: use conjugate instead of reversal
        # grad_x_bj = IFFT(sum_i(grad_fft_bi * conj(w_fft_ij)))
        # (B, p, 1, freq) * (1, p, q, freq).conj() -> sum over p -> (B, q, freq)
        x_grad_fft = (grad_fft.unsqueeze(2) * w_fft.unsqueeze(0).conj()).sum(dim=1)
        x_grad = fft.irfft(x_grad_fft, n=k, dim=-1)  # Optimization 1: single IFFT

        # Bias gradient
        bias_grad = grad_output.sum(dim=0)

        return x_grad.reshape(batch, -1), w_grad, bias_grad, None, None, None
```

## References

- Dong, S. "Algorithm-Hardware Co-Optimization of Neural Network Architectures" (PhD Thesis, Northeastern University, 2020). Chapter 7: "Exploring GPU Acceleration of DNNs using Block Circulant Matrices"
- Ding, C., et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" (MICRO 2017)
- Li, S., et al. "tcFFT: Accelerating Half-Precision FFT through Tensor Cores" (2021). arXiv:2104.11471 — complementary acceleration of individual FFT calls
- Wang, B., et al. "C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs" (FPGA 2018) — block circulant RNNs
- NVIDIA. "CUDA C++ Best Practices Guide" — occupancy, coalescing, shared memory guidelines

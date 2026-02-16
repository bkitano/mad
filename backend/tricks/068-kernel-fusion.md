# 068: Kernel Fusion (Operator Fusion)

**Category**: kernel
**Gain type**: efficiency
**Source**: General GPU optimization technique; formalized in ML context by various works including FlashAttention (Dao et al., 2022), TVM (Chen et al., 2018)
**Paper**: [papers/flash-attention-io-aware-tiling.pdf]
**Documented**: 2025-02-11

## Description

Kernel fusion combines multiple sequential GPU operations into a single GPU kernel launch, eliminating intermediate reads/writes to high-bandwidth memory (HBM). On modern GPUs, most neural network operations are **memory-bound** — the time is dominated by data movement, not arithmetic. When operations are executed separately, each one reads inputs from HBM, computes, and writes results back to HBM. By fusing them into a single kernel, intermediate results stay in fast on-chip SRAM/registers, dramatically reducing memory traffic.

The fundamental insight is that **GPU memory bandwidth is the bottleneck**, not compute. A modern A100 GPU can perform ~312 TFLOPS (FP16) but has only ~2 TB/s memory bandwidth. For element-wise operations like activation functions, dropout, or layer norm, the arithmetic intensity (FLOPs/byte) is very low, making them almost entirely memory-bound. Fusing $k$ such operations reduces HBM traffic by up to $k\times$.

Kernel fusion is the enabling implementation technique behind FlashAttention, which fuses matmul → softmax → masking → dropout → matmul into a single kernel. It also underpins efficient implementations of normalization layers, activation functions, and position encoding.

## Mathematical Form

**Unfused computation** (standard):

For a chain of operations $f_1, f_2, \ldots, f_k$ applied to input $X$:

$$
Y_1 = f_1(X), \quad Y_2 = f_2(Y_1), \quad \ldots, \quad Y_k = f_k(Y_{k-1})
$$

Each $Y_i$ is written to HBM and read back for the next operation. Total HBM traffic:

$$
\text{HBM}_{\text{unfused}} = \sum_{i=0}^{k} |Y_i| \cdot (\text{read} + \text{write}) = 2(k+1) \cdot |X| \text{ (for same-size tensors)}
$$

**Fused computation:**

Compose operations into a single kernel $g = f_k \circ f_{k-1} \circ \cdots \circ f_1$:

$$
Y_k = g(X) = f_k(f_{k-1}(\cdots f_1(X)))
$$

Intermediates $Y_1, \ldots, Y_{k-1}$ stay in SRAM/registers. Total HBM traffic:

$$
\text{HBM}_{\text{fused}} = |X| + |Y_k| = 2|X|
$$

**Speedup factor** (for memory-bound operations):

$$
\text{Speedup} \approx \frac{\text{HBM}_{\text{unfused}}}{\text{HBM}_{\text{fused}}} = \frac{2(k+1)}{2} = k + 1
$$

**Arithmetic intensity analysis:**

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes accessed}}
$$

- Element-wise ops (ReLU, GELU, dropout): $\sim 1$ FLOP/element → AI $\approx 0.5$ (heavily memory-bound)
- Reduction ops (softmax, layernorm): $\sim 5$ FLOPs/element → AI $\approx 2.5$ (memory-bound)
- Matrix multiply: $\sim 2N$ FLOPs/element → AI $\approx N$ (compute-bound for large $N$)

**Fusing element-wise ops with matmul** raises the overall arithmetic intensity, moving the composite operation toward compute-bound territory.

**Key Definitions:**

- HBM — High Bandwidth Memory (GPU main memory, ~40-80 GB, ~2 TB/s)
- SRAM — On-chip Static RAM (~20 MB total across SMs, ~19 TB/s)
- Registers — Per-thread storage (~256 KB per SM, effectively infinite bandwidth)
- Kernel — A single GPU program launch that runs across streaming multiprocessors
- Arithmetic intensity — FLOPs per byte of memory accessed (determines if operation is memory- or compute-bound)

## Complexity

| Scenario | HBM Reads/Writes | Kernel Launches |
|----------|-----------------|-----------------|
| $k$ separate ops | $2(k+1) \cdot |X|$ | $k$ |
| Fused single kernel | $2 \cdot |X|$ | $1$ |
| Fused with recomputation | $2 \cdot |X|$ + recompute FLOPs | $1$ |

**Memory:** Eliminates $O(k \cdot |X|)$ intermediate storage in HBM.

**Kernel launch overhead:** Each launch has ~5-10 μs overhead; fusing $k$ kernels saves $\sim k \times 5$ μs. For small tensors, this can be significant.

## Applicability

- **Attention layers**: FlashAttention fuses QK^T → softmax → mask → dropout → ×V into one kernel
- **Normalization + activation**: Fuse LayerNorm/RMSNorm with subsequent activation (GELU/SiLU/ReLU)
- **Residual connections**: Fuse bias addition + residual add + dropout into one kernel
- **Position encoding**: Fuse RoPE or ALiBi position encoding with attention score computation
- **Loss computation**: Fuse softmax + cross-entropy into numerically stable single-kernel implementation
- **Quantization-aware ops**: Fuse dequantize → matmul → quantize into one kernel
- **SSM/linear recurrence**: Fuse gate computation + state update + output projection

## Limitations

- Requires custom CUDA/Triton kernel development (cannot be done purely from Python)
- Fused kernels are less composable/modular — changes to the computation require rewriting the kernel
- SRAM is limited (~100-192 KB per SM); fusion is bounded by how much intermediate data fits on-chip
- Autograd compatibility: fused forward pass needs a corresponding fused backward kernel
- Trade-off with recomputation: sometimes it's better to recompute intermediates than to store them (FlashAttention does this)
- Compiler-based fusion (TorchInductor, XLA, TVM) can handle simple cases but struggles with complex patterns like attention
- Different GPU architectures may need different fusion strategies (A100 vs H100 vs MI300)

## Implementation Notes

```python
import triton
import triton.language as tl

# Example: Fused bias + GELU + dropout in Triton
@triton.jit
def fused_bias_gelu_dropout_kernel(
    x_ptr, bias_ptr, out_ptr,
    seed, p_drop,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    """Single kernel: load x, add bias, apply GELU, apply dropout, store."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Single HBM read
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % bias_ptr.shape[0], mask=mask)

    # All computation in registers (no HBM writes for intermediates)
    x = x + bias                                          # bias add
    x = x * 0.5 * (1.0 + tl.math.erf(x / 1.4142135))   # GELU

    # Dropout
    random = tl.rand(seed, offsets)
    x = tl.where(random > p_drop, x / (1 - p_drop), 0.0)

    # Single HBM write
    tl.store(out_ptr + offsets, x, mask=mask)

# Without fusion: 3 separate kernel launches, 3 HBM round-trips
# With fusion: 1 kernel launch, 1 HBM round-trip
# Typical speedup: 2-3x for this pattern
```

## References

- Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022). arXiv:2205.14135
- Chen, T., et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (OSDI 2018)
- NVIDIA. "Matrix Multiplication Background User's Guide" — NVIDIA Deep Learning Performance Documentation
- Zheng, S., et al. "Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion" (2024)
- Li, A., et al. "Automatic Horizontal Fusion for GPU Kernels" (CGO 2022)
- Tillet, P., et al. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MLSys 2019)

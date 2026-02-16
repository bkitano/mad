# 136: 4 Structured Sparsity (Sparse Tensor Cores)

**Category**: kernel
**Gain type**: efficiency
**Source**: Mishra et al. "Accelerating Sparse Deep Neural Networks" (NVIDIA, 2021)
**Paper**: [papers/accelerating-sparse-dnn-2-4.pdf]
**Documented**: 2025-02-11

## Description

2:4 structured sparsity is a hardware-native sparsity pattern built into NVIDIA Ampere (and later) Tensor Cores. The constraint is simple: in every contiguous group of 4 elements, at least 2 must be zero. This yields exactly 50% sparsity with a **regular, predictable pattern** that hardware can exploit directly — unlike unstructured sparsity, which suffers from irregular memory access and poor hardware utilization.

The key insight is that this is a **co-design** of the sparsity pattern and the hardware: the 2:4 constraint is specifically chosen because it enables 2× math throughput on Sparse Tensor Cores while being fine-grained enough to maintain model accuracy with a simple prune-then-retrain workflow. The compressed format stores only the 2 nonzero values plus 2-bit metadata per group, achieving ~44% memory savings for 16-bit values.

This is a prime example of designing mathematical objects (sparse matrices) to match GPU hardware constraints, then showing that neural networks are robust enough to work within those constraints.

## Mathematical Form

**2:4 Sparsity Constraint:**

For a weight matrix $W \in \mathbb{R}^{m \times n}$, every contiguous group of 4 elements along a row satisfies:

$$
|\{i \in \{4k, 4k+1, 4k+2, 4k+3\} : W_{r,i} \neq 0\}| \leq 2 \quad \forall r, \forall k
$$

**Compressed Storage Format:**

A sparse matrix $W \in \mathbb{R}^{R \times C}$ is stored as:
- Nonzero values: $W_{\text{compressed}} \in \mathbb{R}^{R \times C/2}$ (the 2 nonzero values per group)
- Metadata: $\text{idx} \in \{0,1,2,3\}^{R \times C/2}$ (2-bit index per nonzero, encoding position within group of 4)

**Sparse GEMM Operation:**

For dense matrix $B \in \mathbb{R}^{C \times N}$ and sparse $W$:

$$
C = W \cdot B
$$

The Sparse Tensor Core:
1. Reads compressed $W_{\text{compressed}}$ ($R \times C/2$ elements)
2. Uses metadata to select $C/2$ matching elements from each column of $B$
3. Computes $R \times (C/2) \times N$ multiply-accumulate operations (half the dense work)

**Pruning criterion (magnitude-based):**

For each group of 4 weights $\{w_{4k}, w_{4k+1}, w_{4k+2}, w_{4k+3}\}$, keep the 2 with largest $|w_i|$, zero the other 2.

**Training workflow:**

$$
W_{\text{dense}} \xrightarrow{\text{train}} W_{\text{trained}} \xrightarrow{\text{prune 2:4}} W_{\text{sparse}} \xrightarrow{\text{retrain (same hyperparams)}} W_{\text{final}}
$$

**Key Definitions:**

- $R \times C$ — weight matrix dimensions
- $C/2$ — compressed column dimension (50% of original)
- 2-bit metadata — encodes position of each nonzero within its group of 4

## Complexity

| Operation | Dense Tensor Core | Sparse Tensor Core |
|-----------|------------------|-------------------|
| Math throughput | $T$ ops/sec | $2T$ ops/sec |
| Weight memory | $R \times C$ elements | $R \times C/2$ elements + metadata |
| Memory savings (FP16) | baseline | ~44% reduction |
| Memory savings (INT8) | baseline | ~38% reduction |
| Metadata overhead (FP16) | 0% | 12.5% |

**Storage:** For 16-bit values, dense requires $4 \times 16 = 64$ bits per group of 4, while 2:4 sparse requires $2 \times 16 + 2 \times 2 = 36$ bits.

**Performance:** Up to 2× speedup on matrix multiply operations, translating to 30-36% end-to-end performance/watt gain on models like ResNeXt-101.

## Applicability

- **Inference acceleration**: Direct 2× speedup on all GEMM-based layers (linear, conv, attention projections) on Ampere+ GPUs
- **Training acceleration**: The 2:4 pattern must also hold for $W^\top$ to accelerate both forward and backward passes
- **Transformer models**: All linear projections ($W_Q, W_K, W_V, W_O$, FFN layers) are candidates
- **Convolutional networks**: Conv layers expressed as im2col + GEMM benefit directly
- **Quantization-friendly**: Composes with INT8/FP8 quantization for additional speedup
- **PyTorch native**: Available via `torch.sparse.to_sparse_semi_structured()` since PyTorch 2.1

## Limitations

- Only 50% sparsity — less aggressive than unstructured pruning (which can reach 90%+)
- Requires NVIDIA Ampere or later GPU hardware (A100, H100, etc.)
- Not all layers tolerate 2:4 pruning equally — embedding layers and first/last layers may need to stay dense
- The 2:4 constraint on transposed weights (for backward pass) is more restrictive than forward-only
- Group size of 4 is hardware-fixed; cannot adjust granularity
- Pruning criterion is typically simple magnitude-based; more sophisticated criteria add complexity

## Implementation Notes

```python
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Enable cuSPARSELt backend for best performance
SparseSemiStructuredTensor._FORCE_CUTLASS = False

def apply_2_4_sparsity(model):
    """Apply 2:4 structured sparsity to linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Magnitude-based pruning: zero smallest 2 of every 4
            weight = module.weight.data
            # Reshape to groups of 4
            w = weight.view(-1, 4)
            # Find indices of 2 smallest magnitudes per group
            _, idx = w.abs().topk(2, dim=1, largest=False)
            mask = torch.ones_like(w, dtype=torch.bool)
            mask.scatter_(1, idx, False)
            weight.view(-1, 4).mul_(mask)

            # Convert to semi-structured sparse format
            module.weight = torch.nn.Parameter(
                to_sparse_semi_structured(weight)
            )
    return model

# Workflow: train dense → prune → retrain → deploy
# 1. model = train(model, data, epochs=90)
# 2. model = apply_2_4_sparsity(model)
# 3. model = train(model, data, epochs=90)  # same hyperparams
# 4. deploy with Sparse Tensor Cores
```

## References

- Mishra, A., et al. "Accelerating Sparse Deep Neural Networks" (2021). arXiv:2104.08378
- NVIDIA. "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT" (2021). NVIDIA Technical Blog.
- Pool, J. & Yu, C. "Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity" (PyTorch Blog, 2023).
- Hubara, I., et al. "Accelerating Transformer Pre-training with 2:4 Sparsity" (2024). arXiv:2404.01847
- Li, Z., et al. "Fused3S: Fast Sparse Attention on Tensor Cores" (ICS 2025). arXiv:2505.08098

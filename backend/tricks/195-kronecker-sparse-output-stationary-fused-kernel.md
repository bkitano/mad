# 195: Kronecker-Sparse Output-Stationary Fused Kernel

**Category**: kernel
**Gain type**: efficiency
**Source**: Gonon, Zheng, Carrivain & Le, "Fast Inference with Kronecker-Sparse Matrices" (ICML 2025; arXiv:2405.15013)
**Paper**: [papers/kronecker-sparse-fast-inference.pdf]
**Documented**: 2026-02-15

## Description

Kronecker-sparse (KS) matrices — whose sparsity pattern is a Kronecker product of identity and all-ones blocks — underpin the Butterfly, Monarch, and Kaleidoscope matrix families used for efficient neural network layers. Standard GPU implementations for KS matrix-vector products follow a three-kernel strategy: (1) **permute** input columns to make blocks contiguous, (2) **batched GEMM** on the block-diagonal form, (3) **inverse permute** output columns back. Profiling reveals that the two permutation kernels — which are pure memory-rewriting operations — consume **up to 50%** of total runtime.

This trick introduces an **output-stationary tiling strategy** that fuses all three operations into a **single CUDA kernel**, eliminating two global-memory round trips. The key insight is a mathematically equivalent reformulation of the permutation-based algorithm: instead of permuting data to create contiguous blocks, the kernel directly computes each output tile by loading the correct (non-contiguous) input columns and KS-matrix sub-blocks into shared memory via a **Kronecker-adapted reading pattern**, performing the dense multiply in registers, and writing the result directly to the correct (non-contiguous) output location via a **Kronecker-adapted writing pattern**.

Across 600+ KS patterns spanning 6 orders of magnitude in matrix size, the fused kernel achieves a **median 1.4× speedup** over the best existing KS baseline (BMM) and **6.5× over dense/sparse baselines** in FP32. A simple heuristic $h(b,c) = \frac{b+c}{bc}$ predicts when the fused kernel outperforms — maximizing $h(b,c)$ maximizes the fused kernel's advantage. End-to-end, replacing dense layers in ViT-S/16 and GPT-2 Medium with KS factors using this kernel yields **22% and 16% wall-clock inference speedups** respectively.

## Mathematical Form

**KS Pattern Definition:**

A KS pattern is a 4-tuple $\pi = (a, b, c, d)$. Its support is:

$$
\mathbf{S}_\pi = \mathbf{I}_a \otimes \mathbf{1}_{b \times c} \otimes \mathbf{I}_d
$$

where $\mathbf{I}_n$ is the $n \times n$ identity, $\mathbf{1}_{b \times c}$ is a $b \times c$ all-ones matrix, and $\otimes$ is the Kronecker product, giving a matrix of size $abd \times acd$.

A matrix $\mathbf{K} \in \mathcal{K}_\pi$ if $\text{supp}(\mathbf{K}) \subseteq \text{supp}(\mathbf{S}_\pi)$.

**Standard 3-Kernel Algorithm (Permute-GEMM-Permute):**

Given $\mathbf{K} \in \mathbb{R}^{abd \times acd}$ and inputs $\mathbf{X} \in \mathbb{R}^{B \times acd}$, compute $\mathbf{Y} = \mathbf{X}\mathbf{K}^\top$:

$$
\mathbf{Y} = \mathbf{X}\mathbf{Q}^\top \underbrace{\mathbf{Q}\mathbf{K}^\top\mathbf{P}}_{\bar{\mathbf{K}}^\top} \mathbf{P}^\top
$$

where $\mathbf{P}, \mathbf{Q}$ are permutations making $\bar{\mathbf{K}} = \mathbf{P}^\top\mathbf{K}\mathbf{Q}^\top$ block-diagonal with $a$ dense blocks of size $bd \times cd$.

**Output-Stationary Reformulation (Algorithm 2):**

Partition row indices $\{0, \ldots, abd-1\}$ into tiles:

$$
\text{row}_{i,j} := \left\{i\frac{M}{a} + j + kd : 0 \leq k < b\right\}, \quad i = 0, \ldots, a-1, \; j = 0, \ldots, d-1
$$

$$
\text{col}_{i,j} := \left\{i\frac{N}{a} + j + \ell d : 0 \leq \ell < c\right\}
$$

Then the full multiplication decomposes into $a \cdot d$ independent dense sub-products:

$$
\mathbf{Y}[:, \text{row}_{i,j}] = \mathbf{X}[:, \text{col}_{i,j}] \cdot \mathbf{K}^\top[\text{col}_{i,j}, \text{row}_{i,j}]
$$

Each tile is a dense $B \times c \to B \times b$ matmul — no permutations needed.

**Key Definitions:**

- $\pi = (a, b, c, d)$ — KS pattern parameters
- $M = abd$ — output dimension
- $N = acd$ — input dimension
- $\text{nnz} = abcd$ — number of nonzeros
- Sparsity $= 1 - \frac{1}{ad}$ — fraction of zeros in $\mathbf{K}$
- $h(b,c) = \frac{b+c}{bc}$ — memory overhead heuristic (higher → more benefit from fusion)
- $\mathbf{K}^\top[\text{col}_{i,j}, \text{row}_{i,j}]$ — the $c \times b$ dense sub-block for tile $(i,j)$

**Connection to Butterfly/Monarch:**

The $N \times N$ DFT matrix factors into $L = \log_2 N$ KS factors, each with pattern $(a, b, c, d) = (2^{\ell-1}, 2, 2, 2^{L-\ell})$. Monarch matrices (Dao et al., 2022) use KS factors with $(a, \sqrt{N}, \sqrt{N}, 1)$. The circulant-diagonal product $\mathbf{F}^{-1}\mathbf{D}\mathbf{F}$ decomposes as KS factor products interleaved with diagonal scaling.

## Complexity

| Operation | Dense | BMM (3-kernel) | Fused KERNEL (this trick) |
|-----------|-------|----------------|--------------------------|
| Useful FLOPs | $O(BMN)$ | $O(B \cdot \text{nnz})$ | $O(B \cdot \text{nnz})$ |
| Global memory ops | $B(M+N)$ | $3B(N+M)$ | $B(N+M)$ |
| Kernel launches | 1 | 3 | **1** |
| Memory overhead ratio | — | $h(b,c) = \frac{b+c}{bc}$ | **0** (eliminated) |

**Wasted memory traffic eliminated:**

The baseline wastes $2B(N+M)$ global memory operations on the two permutation passes. The fused kernel eliminates this entirely, saving:

$$
\text{fraction saved} = \frac{2B(N+M)}{3B(N+M)} = \frac{2}{3} \approx 67\%
$$

of total baseline memory traffic (in the ideal case).

**Benchmark Results (600 KS patterns, FP32, A100):**

| Comparison | Win Rate | Median Speedup |
|-----------|---------|---------------|
| KERNEL vs Dense/Sparse | 98.1% | **6.5×** |
| KERNEL vs BMM/EINSUM/BSR | 85.5% | **1.39×** |
| BMM vs all others | 90.0% | 1.36× |

**Speedup Regression Model:**

$$
\log(\text{speedup}) \approx 1.69 - 0.031 \log(\text{density}) + 0.325 \log(h(b,c))
$$

with adjusted $R^2 = 0.697$. The heuristic $h(b,c)$ is **10× more influential** than density in predicting speedup.

**Energy Reduction:** Median 15% energy saving on V100 GPU, predicted by $d \times h(b,c)$.

**End-to-End Inference (FP32):**

| Model | Sparsity | time(BMM)/time(Dense) | time(KERNEL)/time(Dense) |
|-------|---------|----------------------|-------------------------|
| ViT-S/16 | 57.1% | 0.89 | **0.78** (22% speedup) |
| GPT-2 Medium | 24.8% | 0.88 | **0.84** (16% speedup) |

## Applicability

- **Inference acceleration for KS-structured models**: Drop-in replacement (`KSLinear` PyTorch module) for any Butterfly/Monarch/Kaleidoscope layer, providing 1.4× median speedup over BMM with no accuracy change. Released as open-source CUDA/OpenCL kernels at github.com/PascalCarrivain/ksmm
- **Transformer linear layers**: FC layers in ViT and GPT models are natural candidates — they account for 30–60% of inference time, and KS factorization can replace them with 2 KS factors per layer
- **Post-training KS compression**: After training a dense model, weight matrices can be approximated by KS products and deployed with the fused kernel for faster inference
- **Circulant-diagonal cascades (ACDC/CDFlow)**: The FFT that implements each circulant multiply is itself a KS product — this kernel directly accelerates the FFT butterfly factors within circulant-diagonal pipelines
- **Multi-factor structured layers**: For cascaded KS products $\mathbf{K}_1 \cdots \mathbf{K}_L$, each factor gets the fusion benefit. Combined with the batch-size-last memory layout, gains compound across factors
- **Design rule for KS pattern selection**: When choosing among multiple valid KS factorizations, maximize $h(b,c) = \frac{b+c}{bc}$ to get the most benefit from the fused kernel. This is a one-line rule practitioners can use

## Limitations

- **FP32 only (for now)**: Gains in FP16/FP8 are smaller (Appendix E.9) because tensor core operations are already highly optimized and the permutation overhead is relatively smaller. Extending to tensor-core-aware tiling is future work
- **Batch-size-first layout suboptimal**: PyTorch defaults to batch-size-first (BSF) memory layout, where the strided access pattern of the Kronecker-adapted reading causes non-coalesced loads. Batch-size-last (BSL) layout gives coalesced access and up to 2× additional speedup, but switching the full pipeline to BSL is non-trivial
- **Single-factor multiplication only**: The kernel handles one KS factor at a time. Fusing multiple sequential KS factors (e.g., the full FFT butterfly chain) into a single kernel would further reduce HBM round trips but is not addressed
- **No training support**: Focus is on inference — gradient computation through the fused kernel is not provided. Training still uses the standard BMM approach
- **Pattern-specific tuning**: Optimal kernel hyperparameters (thread count, tile shape) depend on the specific KS pattern $(a,b,c,d)$. Pre-tuned configurations are provided for 600 patterns; novel patterns may need re-tuning
- **Not applicable to unstructured sparsity**: Only works for Kronecker-structured sparsity patterns, not general sparse matrices

## Implementation Notes

```python
import torch

# The key insight: avoid explicit permutations by computing
# tile indices directly from the KS pattern (a, b, c, d)

def compute_tile_indices(a, b, c, d):
    """Compute row/col index sets for each tile (i, j).

    For tile (i, j):
      row_{i,j} = {i*M/a + j + k*d : 0 <= k < b}  (b indices, stride d)
      col_{i,j} = {i*N/a + j + l*d : 0 <= l < c}  (c indices, stride d)

    Total tiles: a * d (one dense b×c matmul each)
    """
    M = a * b * d  # output dim
    N = a * c * d  # input dim

    tiles = []
    for i in range(a):
        for j in range(d):
            row = [i * (M // a) + j + k * d for k in range(b)]
            col = [i * (N // a) + j + l * d for l in range(c)]
            tiles.append((row, col))
    return tiles


def ks_matmul_fused(X, K_tiles, tiles, M):
    """Fused KS matrix-vector product (Python reference).

    Equivalent to Y = X @ K^T but without explicit permutations.
    Each tile is an independent dense c->b matmul.

    Args:
        X: (B, N) input batch
        K_tiles: list of (c, b) dense sub-blocks, one per tile
        tiles: list of (row_indices, col_indices) from compute_tile_indices
        M: output dimension

    Returns:
        Y: (B, M) output
    """
    B = X.shape[0]
    Y = torch.zeros(B, M, device=X.device, dtype=X.dtype)

    for (row, col), K_tile in zip(tiles, K_tiles):
        # Load X columns (strided access in BSF layout)
        X_tile = X[:, col]          # (B, c)
        # Dense matmul: (B, c) @ (c, b) -> (B, b)
        Y_tile = X_tile @ K_tile    # (B, b)
        # Accumulate into output (strided write in BSF layout)
        Y[:, row] += Y_tile

    return Y


# CUDA kernel sketch (see Algorithm 3 in paper):
#
# __global__ void ks_kernel(X, K, Y, a, b, c, d, B) {
#     // Identify tile (i, j) for this thread block
#     int tile_id = blockIdx.x;
#     int i = tile_id / d, j = tile_id % d;
#
#     // Compute strided row/col index sets
#     // row = {i*M/a + j + k*d : k in 0..b-1}
#     // col = {i*N/a + j + l*d : l in 0..c-1}
#
#     // Cooperative load: X[:, col] and K^T[col, row] -> shared mem
#     // (Kronecker-adapted strided reads, coalesced in BSL layout)
#
#     for (subtile in range(c)):
#         // shared -> registers: load X and K subtiles
#         // Multiply-and-accumulate in registers
#         // Prefetch next subtile (double buffering)
#
#     // Cooperative store: registers -> Y[:, row] in global mem
#     // (Kronecker-adapted strided writes)
# }
#
# Key optimizations:
# - Double buffering for memory latency hiding
# - Vectorized memory access (128-bit loads where possible)
# - Warp-level parallelism across batch dimension
# - Output-stationary: each thread block owns its output tile,
#   no atomics or synchronization between blocks needed


def select_ks_pattern(N, M, target_sparsity=0.9):
    """Heuristic: choose KS pattern (a,b,c,d) maximizing h(b,c).

    Given target dimensions N (input) and M (output),
    find the pattern that maximizes the fused kernel benefit.

    Rule: maximize h(b,c) = (b+c)/(b*c) subject to:
      M = a*b*d, N = a*c*d, sparsity = 1 - 1/(a*d) >= target
    """
    best_h = 0
    best_pattern = None

    for a in range(1, min(M, N) + 1):
        for d in range(1, min(M // a, N // a) + 1):
            if M % (a * d) != 0 or N % (a * d) != 0:
                continue
            b = M // (a * d)
            c = N // (a * d)
            sparsity = 1.0 - 1.0 / (a * d)
            if sparsity < target_sparsity:
                continue
            h = (b + c) / (b * c)
            if h > best_h:
                best_h = h
                best_pattern = (a, b, c, d)

    return best_pattern, best_h
```

## References

- Gonon, A., Zheng, L., Carrivain, P. & Le, Q.-T. "Fast Inference with Kronecker-Sparse Matrices" ICML 2025. arXiv:2405.15013
- Dao, T. et al. "Monarch: Expressive Structured Matrices for Efficient and Accurate Training" ICML 2022
- Dao, T. et al. "Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations" ICML 2019
- Moczulski, M. et al. "ACDC: A Structured Efficient Linear Layer" ICLR 2016
- Fu, D.Y. et al. "Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture" NeurIPS 2023
- Le, Q.-T. et al. "Butterfly Factorization with Error Guarantees" arXiv:2411.04506, 2024

# 113: SIMD² Semiring Matrix Acceleration

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhang, Tsai, and Tseng (2022). SIMD²: A Generalized Matrix Instruction Set for Accelerating Tensor Computation beyond GEMM. ISCA '22.
**Paper**: papers/simd2-semiring-matrix-instructions.pdf
**Documented**: 2026-02-15

## Description

SIMD² extends GPU matrix-multiplication units (MXUs, e.g., Tensor Cores) to support **eight additional semiring-like matrix operations** beyond standard GEMM $(+, \times)$. The key insight is that MXUs are successful not because of multiply-add per se, but because of the **semiring structure** — the tiling pattern that enables both parallelism and data reuse. By adding minimal hardware (5% chip area overhead), SIMD² enables hardware-accelerated semiring matrix operations like min-plus, max-plus, min-mul, max-mul, min-max, max-min, or-and, and add-norm, achieving up to 38.59× speedup over optimized CUDA implementations.

This directly addresses the fundamental bottleneck identified in the semiring-monoid-lifting trick: semiring operations that don't match the standard $(+, \times)$ pattern must fall back to CUDA cores at ~16× lower throughput. SIMD² closes this gap by generalizing the hardware itself.

## Mathematical Form

**Core Semiring-Like Structure:**

All supported operations follow the common pattern:

$$
D = C \oplus (A \otimes B)
$$

where $\oplus$ is the **accumulation** (reduction) operator and $\otimes$ is the **element-wise** (product) operator, applied in the standard tiled matrix-multiply pattern:

$$
d(i, j) = c(i, j) \oplus \bigoplus_{k=0}^{N} \left( a(i,k) \otimes b(k,j) \right)
$$

**Supported Semiring-Like Operations:**

| Instruction | $\oplus$ (1st OP) | $\otimes$ (2nd OP) | Algorithm |
|---|---|---|---|
| `SIMD².mma` | $+$ | $\times$ | Standard GEMM |
| `SIMD².minplus` | $\min$ | $+$ | All-pairs shortest path (APSP) |
| `SIMD².maxplus` | $\max$ | $+$ | Maximum cost (critical path) |
| `SIMD².minmul` | $\min$ | $\times$ | Minimum reliability path |
| `SIMD².maxmul` | $\max$ | $\times$ | Maximum reliability path |
| `SIMD².minmax` | $\min$ | $\max$ | Minimum spanning tree |
| `SIMD².maxmin` | $\max$ | $\min$ | Maximum capacity path |
| `SIMD².orand` | $\lor$ | $\land$ | Transitive/reflexive closure |
| `SIMD².addnorm` | $+$ | $\|a - b\|^2$ | L2 distance (KNN) |

**Example — All-Pairs Shortest Path (Min-Plus Semiring):**

The Floyd-Warshall/Bellman-Ford APSP is expressed as iterative min-plus matrix "multiplication":

$$
d(i, j) = \min\left(c(i, j), \min_{k=0}^{N}\{c(i, k) + a(k, j)\}\right)
$$

which is exactly $D = C \oplus_{\min} (C \otimes_{+} A)$ — a single `SIMD².minplus` instruction per tile.

**Hardware Architecture:**

The SIMD² unit extends the MXU ALU array:
- The **$\otimes$ ALU** supports: multiply, min/max, add/and, and L2 distance
- The **$\oplus$ ALU** supports: add, min/max, or, and subtract
- Both are configured by decoding SIMD² instruction opcodes
- The broadcast-and-accumulate data flow pattern is **identical** to standard GEMM

## Complexity

| Operation | CUDA Cores | SIMD² (MXU) | Speedup |
|-----------|-----------|-------------|---------|
| Min-Plus matmul (APSP) | $O(N^3)$ on scalar ALUs | $O(N^3)$ on MXU tiles | $7.9\times - 15.8\times$ |
| Max-Plus matmul | $O(N^3)$ on scalar ALUs | $O(N^3)$ on MXU tiles | $7.9\times - 15.8\times$ |
| Min-Max (MST) | $O(N^3)$ on scalar ALUs | $O(N^3)$ on MXU tiles | up to $38.59\times$ |
| Or-And (GTC) | $O(N^3)$ on scalar ALUs | $O(N^3)$ on MXU tiles | $7.1\times - 23.8\times$ |
| Add-Norm (KNN) | $O(N^2 d)$ on scalar ALUs | $O(N^2 d)$ on MXU tiles | $4.8\times - 6.4\times$ |

**Key point:** The asymptotic complexity is unchanged — the speedup comes from **hardware throughput**. MXUs achieve higher throughput than CUDA cores because they exploit the semiring structure for data reuse (broadcast pattern) and parallelism (tiled accumulation), which CUDA cores cannot exploit.

**Area Overhead:** 69% over baseline MMA unit, but only **5% of total chip area** (0.378 mm² on Samsung 8nm).

**Memory:** Same $O(N^2)$ for input/output matrices. The tiled MXU approach inherits the favorable compute-to-memory ratio: $O(N^3)$ compute with $O(N^2)$ data transfer.

## Applicability

- **Sequence models with non-standard scans**: SSMs and linear RNNs that use parallel scans could benefit from hardware-accelerated semiring operations. A tropical-semiring scan (max-plus) would enable hardware-efficient "tropical SSMs."
- **Tropical attention**: The max-plus matmul instruction (`SIMD².maxplus`) directly accelerates the core operation in tropical attention mechanisms, where $C_{ij} = \max_k(Q_{ik} + K_{jk})$ replaces softmax dot-product attention.
- **Graph neural networks**: Many GNN message-passing operations are naturally expressed as semiring matrix operations (min-plus for shortest paths, max-min for bottleneck paths).
- **Dynamic programming alignment**: Sequence alignment (Needleman-Wunsch, Smith-Waterman) uses min/max-plus operations that map directly to SIMD² instructions.
- **Shortest-path layers**: Neural network layers that embed shortest-path computations can use `SIMD².minplus` for hardware-accelerated inference.

## Limitations

- **Hardware not yet deployed**: SIMD² is validated via emulation on NVIDIA GPUs using Tensor Cores — no commercial chip implements SIMD² natively yet. The paper demonstrates feasibility and area cost, but actual silicon doesn't exist.
- **Fixed tile size**: SIMD² operates on fixed 16×16 tiles in FP16→FP32, matching existing MXU constraints. Non-tile-aligned problems require padding.
- **No gradient support**: The hardware computes forward-pass semiring operations only. Backward passes for tropical operations (sparse argmax gradients) still require CUDA cores or custom implementations.
- **Limited to two-operator semirings**: SIMD² supports $(\oplus, \otimes)$ pairs but not more complex algebraic structures (e.g., semirings with additional operations or multi-sorted algebras).
- **Convergence checks on CPU**: For iterative algorithms like APSP, convergence checks require CPU-side logic or conventional GPU cores, creating synchronization overhead.

## Implementation Notes

```python
# Pseudocode showing the SIMD² programming model for min-plus matrix multiply

# Low-level SIMD² API (C++ PTX-level)
# simd2::matrix<half> mat_A, mat_B;     # 16x16 FP16 input tiles
# simd2::matrix<float> mat_C;           # 16x16 FP32 accumulator
# simd2::loadmatrix(mat_A, A, lda);     # Load tile from shared memory
# simd2::loadmatrix(mat_B, B, ldb);
# simd2::loadmatrix(mat_C, C, ldc);     # Load current partial result
# simd2::mmo(mat_C, mat_A, mat_B, mat_C, minplus);  # D = C ⊕_min (A ⊗_+ B)
# simd2::storematrix(D, mat_C, ldd);    # Store result

# The key insight: SIMD² reuses the SAME tiling/broadcast/accumulate
# data flow as GEMM, just swaps the ALU operations.
# This means existing CUTLASS-style tiling strategies apply directly.

# High-level usage for APSP:
# while not converged:
#     D = simd2_minplus(C, A)  # One "iteration" of Floyd-Warshall
#     converged = check_convergence(D, C)
#     C = D

# For neural network tropical attention:
# scores = simd2_maxplus(Q, K.T)  # Tropical similarity: max_k(q_ik + k_jk)
# output = simd2_maxplus(scores, V)  # Tropical aggregation: max_j(s_ij + v_j)
```

## References

- Zhang, Tsai, and Tseng (2022). SIMD²: A Generalized Matrix Instruction Set for Accelerating Tensor Computation beyond GEMM. ISCA '22. arXiv:2205.01252.
- cuASR: CUDA Algebra for Semirings. https://github.com/hpcgarage/cuASR
- SIMD² code repository: https://github.com/escalab/SIMD2
- GraphBLAS specification (semiring operations for graph analytics)

# 039: Epilogue Visitor Tree (EVT) Fusion

**Category**: kernel
**Gain type**: efficiency
**Source**: Chen, Kerr, Cai, Kosaian, Wu, Ding, Xie — "EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree" (ASPLOS 2024)
**Paper**: N/A (ACM ASPLOS 2024, DOI: 10.1145/3620666.3651369; no open-access PDF available)
**Documented**: 2025-06-15

## Description

The Epilogue Visitor Tree (EVT) is a compiler abstraction and code generation framework that enables composable fusion of arbitrary post-GEMM operations into a single GPU kernel. In neural networks, GEMM (matrix multiply) is typically followed by a chain of element-wise, broadcast, and reduction operations — bias addition, activation functions (GELU, SiLU, ReLU), residual connections, loss computation, and quantization. Without fusion, each operation requires a separate kernel launch with full HBM round-trips for intermediate results. EVT solves this by representing the post-GEMM computation as a **tree of visitor nodes** that execute within the GEMM kernel's epilogue phase, while the accumulator data is still in registers or shared memory.

The fundamental problem EVT addresses is **combinatorial explosion**: there are hundreds of possible epilogue patterns across different neural network layers (forward and backward passes), and manually writing fused kernels for each combination is impractical. EVT provides a composable type-level abstraction where leaf nodes represent data sources (accumulator, auxiliary matrices, scalars, broadcast vectors) and internal nodes represent operations (element-wise compute, reductions, stores). Any valid tree composition automatically generates a correct, high-performance fused kernel.

A critical innovation is the **ILP-based graph partitioner** for training workloads. During training, the joint forward-backward computation graph contains complex data dependencies (including DAG structures, not just trees). The EVT partitioner uses integer linear programming to find the optimal partitioning of this graph into fusible epilogue subgraphs, maximizing fusion opportunities while respecting hardware constraints (register pressure, shared memory capacity).

EVT achieves 1.26–3.1× speedup on diverse training workloads by eliminating intermediate HBM traffic and kernel launch overhead, while maintaining full numerical equivalence — no approximation is involved.

## Mathematical Form

**Standard unfused GEMM + epilogue:**

Given GEMM output $\mathbf{D}_{\text{acc}} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{M \times N}$ (in registers after mainloop), a typical post-processing chain is:

$$
\mathbf{D} = \text{activation}\!\left(\alpha \cdot \mathbf{D}_{\text{acc}} + \beta \cdot \mathbf{C} + \mathbf{b}\right)
$$

where $\mathbf{C} \in \mathbb{R}^{M \times N}$ is a source matrix, $\mathbf{b} \in \mathbb{R}^{1 \times N}$ is a bias (row-broadcast), and $\alpha, \beta$ are scalars. Without fusion, this requires 3 separate kernels after GEMM:

$$
\text{Kernel 1: } \mathbf{T}_1 = \alpha \cdot \mathbf{D}_{\text{acc}} \quad (\text{write to HBM})
$$
$$
\text{Kernel 2: } \mathbf{T}_2 = \mathbf{T}_1 + \beta \cdot \mathbf{C} + \mathbf{b} \quad (\text{read } \mathbf{T}_1, \mathbf{C}, \mathbf{b} \text{ from HBM, write } \mathbf{T}_2)
$$
$$
\text{Kernel 3: } \mathbf{D} = \text{activation}(\mathbf{T}_2) \quad (\text{read } \mathbf{T}_2 \text{ from HBM, write } \mathbf{D})
$$

Total HBM traffic: $\sim 6 \cdot M \cdot N$ elements (each intermediate read + written).

**EVT fused epilogue:**

All operations compose into a single epilogue executed while $\mathbf{D}_{\text{acc}}$ is still in registers:

$$
\mathbf{D} = \text{activation}\!\left(\alpha \cdot \mathbf{D}_{\text{acc}} + \beta \cdot \mathbf{C} + \mathbf{b}\right)
$$

Total HBM traffic: $M \cdot N$ (read $\mathbf{C}$) + $N$ (read $\mathbf{b}$) + $M \cdot N$ (write $\mathbf{D}$) $\approx 2 \cdot M \cdot N$.

**Tree structure:**

The EVT for the above computation is:

$$
\texttt{EVT}\!\left(\text{activation},\; \texttt{EVT}\!\left(\text{add},\; \texttt{EVT}\!\left(\text{fma},\; \texttt{Scalar}(\alpha),\; \texttt{AccFetch},\; \texttt{EVT}\!\left(\text{mul},\; \texttt{Scalar}(\beta),\; \texttt{SrcFetch}(\mathbf{C})\right)\right),\; \texttt{RowBroadcast}(\mathbf{b})\right)\right)
$$

**Node types (CUTLASS Hopper implementation):**

*Leaf nodes (data sources):*
- $\texttt{AccFetch}$ — reads GEMM accumulator fragment from registers
- $\texttt{SrcFetch}(\mathbf{C})$ — loads matrix $\mathbf{C}$ from global memory via TMA
- $\texttt{ScalarBroadcast}(\alpha)$ — broadcasts a scalar to all elements
- $\texttt{RowBroadcast}(\mathbf{b})$ — broadcasts a row vector $\mathbf{b} \in \mathbb{R}^{1 \times N}$
- $\texttt{ColBroadcast}(\mathbf{c})$ — broadcasts a column vector $\mathbf{c} \in \mathbb{R}^{M \times 1}$
- $\texttt{AuxLoad}(\mathbf{X})$ — loads an auxiliary matrix via TMA

*Internal nodes (operations):*
- $\texttt{Compute}(f, \text{children})$ — applies element-wise function $f$ (ReLU, GELU, SiLU, sigmoid, multiply, add, FMA, etc.)
- $\texttt{ScalarReduction}(\oplus)$ — reduces all elements to a scalar using $\oplus$
- $\texttt{RowReduction}(\oplus)$ — reduces along rows to produce a column vector
- $\texttt{ColReduction}(\oplus)$ — reduces along columns to produce a row vector
- $\texttt{AuxStore}(\mathbf{Y})$ — stores intermediate result to an auxiliary output

**Execution model:**

The epilogue processes data hierarchically within each CTA:

$$
\text{Tile} \to \text{Subtiles}_{(i,j)} \to \text{Fragments}_{v}
$$

For each fragment $v$ in subtile $(i, j)$:
1. **visit()**: Each leaf/internal node produces its fragment output
2. Tree evaluation proceeds bottom-up: children visited before parents
3. **end_loop()**: After all fragments in a subtile, reduction nodes finalize

**Gated-SiLU (SwiGLU) fusion example:**

The gated linear unit $\mathbf{D} = \text{SiLU}(\mathbf{G}) \odot \mathbf{U}$ where $[\mathbf{G} \| \mathbf{U}] = \mathbf{A}\mathbf{B}$ requires pairing adjacent output columns. The EVT handles this by interleaving the weight matrix columns during packing so that gate-up pairs land in the same thread's fragment:

$$
\mathbf{D}_i = \text{SiLU}(\mathbf{D}_{\text{acc}}[2i]) \cdot \mathbf{D}_{\text{acc}}[2i + 1], \quad i = 0, \ldots, N/2 - 1
$$

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**ILP-based graph partitioning (for training):**

Given the joint forward-backward computation DAG $G = (V, E)$, the partitioner solves:

$$
\min \sum_{(u,v) \in E} x_{uv} \quad \text{s.t.} \quad x_{uv} \geq |p_u - p_v|, \;\; \sum_{v \in S} r_v \leq R_{\max} \;\; \forall \text{ partition } S
$$

where $p_v$ assigns each node to a partition, $x_{uv}$ indicates cross-partition edges (requiring HBM materialization), $r_v$ is the register cost of node $v$, and $R_{\max}$ is the register budget per kernel. The objective minimizes the number of cross-partition edges, i.e., the amount of data materialized to HBM.

**Key Definitions:**

- Epilogue — The phase of a GEMM kernel after the mainloop (MAC iterations) completes, where the accumulator is post-processed and written to global memory
- Visitor — A node in the EVT that implements `visit()` (per-fragment) and optionally `end_loop()` (per-subtile) callbacks
- Fragment — A small tile of data (e.g., $8 \times 8$) owned by a single thread, the finest granularity of EVT processing
- Mainloop — The inner loop of GEMM that performs the tiled multiply-accumulate; EVT operates on its output
- TMA — Tensor Memory Accelerator; hardware unit for async global↔shared memory transfers (Hopper+)

## Complexity

| Scenario | HBM Traffic | Kernel Launches |
|----------|------------|-----------------|
| GEMM + $k$ unfused epilogue ops | $\sim 2(k+1) \cdot MN$ | $k + 1$ |
| GEMM + EVT fused epilogue | $\sim 2 \cdot MN + \text{aux inputs}$ | $1$ |
| Training (forward + backward, unfused) | $\sim 4k \cdot MN$ | $\sim 2k$ |
| Training (EVT-partitioned) | optimal partition-dependent | $\leq 2k$ |

**Memory:** Eliminates $O(k \cdot MN)$ intermediate tensor storage. The EVT uses only registers and shared memory for intermediate values within the epilogue.

**Performance (from paper):**

| Workload | Speedup over Unfused |
|----------|---------------------|
| Fused bias + GELU (inference) | 1.26× |
| Fused gated-SiLU (SwiGLU) | ~1.28× (166 μs saved per layer) |
| Binary cross-entropy loss (training) | up to 3.1× |
| General training epilogues | 1.26–3.1× |

**Quantization amplification:** When combined with output quantization (FP8/NVFP4), the fused epilogue avoids writing wide BF16 intermediates, reducing write traffic by up to 8×.

## Applicability

- **Transformer MLP layers**: Linear → activation (GELU/SiLU) → linear, with bias, residual, and dropout all fused into GEMM epilogues
- **Gated linear units (SwiGLU/GeGLU)**: The gate × up-projection pattern fuses naturally into a single GEMM epilogue
- **Attention output projection**: Fuse the final linear projection with residual add and layer norm
- **Loss computation**: Fuse softmax → cross-entropy or sigmoid → BCE directly into the final GEMM's epilogue
- **Training backward pass**: EVT's ILP partitioner identifies optimal fusion opportunities in joint forward-backward graphs, critical for training throughput
- **Quantization-aware training**: Fuse fake-quantization (scale → clamp → round) into GEMM epilogues, reducing per-layer kernel count by 4×
- **Any GEMM-centric layer**: Any computation of the form $\mathbf{D} = f(\mathbf{A}\mathbf{B}, \text{aux inputs})$ where $f$ is composable from element-wise, broadcast, and reduction operations

## Limitations

- **Only epilogue operations**: EVT cannot fuse operations that precede the GEMM mainloop (e.g., input preprocessing, embedding lookups)
- **Element-wise and simple reductions only**: Complex operations requiring cross-tile communication (e.g., full attention softmax) cannot be expressed as EVT nodes
- **Register pressure**: Each fused operation adds register usage; over-fusion can cause register spilling and degrade mainloop performance
- **Hardware requirement**: The Hopper-optimized EVT nodes (TMA-based loads, warp-specialized epilogues) require NVIDIA H100 or later; earlier architectures use simpler but less performant EVT variants
- **Compile-time composition**: The EVT tree structure is determined at compile time via C++ templates; dynamic epilogue selection requires multiple compiled kernel variants
- **DAG limitations**: Standard EVTs enforce tree structure (each node visited once); DAG patterns (multiple consumers of same intermediate) require the topological visitor extension, which adds complexity
- **CUTLASS-specific**: The EVT abstraction is tightly integrated with NVIDIA's CUTLASS library; porting to other frameworks (Triton, ROCm) requires reimplementation

## Implementation Notes

```python
# Pseudocode for EVT-fused GEMM epilogue
# This shows the conceptual execution model, not the actual C++ template metaprogramming

class EVT_LinCombEltAct:
    """EVT for: D = activation(alpha * AccFetch + beta * SrcFetch(C))"""

    def __init__(self, alpha, beta, activation_fn):
        # Tree structure (bottom-up):
        #   Compute(activation_fn,
        #     Compute(fma,
        #       ScalarBroadcast(alpha),
        #       AccFetch,
        #       Compute(mul,
        #         ScalarBroadcast(beta),
        #         SrcFetch(C))))
        self.alpha = alpha
        self.beta = beta
        self.activation = activation_fn

    def visit(self, acc_fragment, src_fragment):
        """Called per-fragment within the epilogue.
        acc_fragment: GEMM accumulator data (still in registers!)
        src_fragment: loaded from C via TMA (in SMEM → registers)
        """
        # All computation happens in registers — no HBM intermediates
        result = self.alpha * acc_fragment + self.beta * src_fragment
        result = self.activation(result)  # e.g., GELU, SiLU, ReLU
        return result

    def store(self, result_fragment, D_ptr):
        """Write final result to global memory (single HBM write)."""
        tma_store(D_ptr, result_fragment)


# Gated-SiLU epilogue (SwiGLU fusion)
class EVT_GatedSiLU:
    """EVT for: D = SiLU(acc[:, :N//2]) * acc[:, N//2:]
    Requires weight packing to interleave gate/up pairs."""

    def visit(self, acc_fragment):
        """Process interleaved gate-up pairs in registers."""
        output = []
        for i in range(0, len(acc_fragment), 2):
            gate = acc_fragment[i]
            up = acc_fragment[i + 1]
            silu_gate = gate / (1.0 + exp(-gate))
            output.append(silu_gate * up)
        return output

    def end_loop(self, subtile_results, D_ptr):
        """After all fragments: write half-width output."""
        # Output is N//2 wide (gate and up consumed pairwise)
        tma_store(D_ptr, subtile_results)


# CUTLASS 3.x C++ template composition (actual usage pattern):
#
# using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
#     Sm90Compute<cutlass::epilogue::thread::GELU>,    // activation
#     Sm90EVT<
#         Sm90Compute<cutlass::multiplies>,             // alpha * acc
#         Sm90ScalarBroadcast<float>,                   // alpha
#         Sm90AccFetch                                  // GEMM accumulator
#     >,
#     Sm90EVT<
#         Sm90Compute<cutlass::multiplies>,             // beta * C
#         Sm90ScalarBroadcast<float>,                   // beta
#         Sm90SrcFetch                                  // source matrix C
#     >
# >;
#
# Arguments mirror the tree but with operations last:
# EpilogueEVT::Arguments args = {
#     {{alpha_args}, {acc_args}, {beta_args, src_args}},
#     {activation_args}
# };
```

## References

- Chen, Z., Kerr, A., Cai, R., Kosaian, J., Wu, H., Ding, Y., Xie, Y. "EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree." ASPLOS 2024. DOI: 10.1145/3620666.3651369
- Colfax Research. "Epilogue Fusion in CUTLASS with Epilogue Visitor Trees." 2024. https://research.colfax-intl.com/epilogue_visitor_tree/
- fal.ai. "Crafting Efficient Kernels with Epilogue Fusion." 2024. https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/
- NVIDIA. "CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design." NVIDIA Developer Blog, 2024.
- NVIDIA CUTLASS. Epilogue fusion operations: `cutlass/epilogue/fusion/operations.hpp`. https://github.com/NVIDIA/cutlass

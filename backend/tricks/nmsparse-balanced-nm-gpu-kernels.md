# nmSPARSE: Balanced N:M Sparsity GPU Kernels via Conflict-Free Shared Memory

**Category**: kernel
**Gain type**: efficiency
**Source**: Lin, Zheng, Wang, Cao, Ma, Zhang, Zhu, Cao, Xue, Yang & Yang (Microsoft Research Asia & Tsinghua, MLSys 2023)
**Paper**: [papers/nmsparse-balanced-nm-gpu-kernels.pdf]
**Documented**: 2026-02-15

## Description

N:M sparsity patterns — where exactly $N$ out of every $M$ consecutive weights are nonzero — impose an **intrinsic balance** constraint: every row of the sparse matrix has the same number of nonzeros per window of $M$ elements. This balance is the key property that nmSPARSE exploits to build highly efficient GPU kernels for sparse matrix-vector (SpMV) and sparse matrix-matrix (SpMM) multiplication, achieving up to 5.2$\times$ and 6.0$\times$ speedups over the fastest baselines on A100.

The core challenge in sparse GEMM on GPUs is that nonzero positions are irregular, causing **scattered memory accesses** to the dense input matrix/vector $A$ and **workload imbalance** across threads. Traditional sparse formats (CSR, COO) are optimized for extremely sparse matrices ($>$95%) and perform poorly at DNN-typical sparsity ratios (50--90%). NVIDIA's cuSPARSELt only supports the specific 2:4 pattern on Ampere Sparse Tensor Cores.

nmSPARSE introduces three key innovations:

1. **Condensed Representation**: N:M sparse weights are compressed vertically along the reduction dimension $k$ into a dense 2D data array (with balanced dimensions) plus a compact index array (each index uses only $\lceil \log_2 M \rceil$ bits). This eliminates the storage overhead of general sparse formats while being decoding-friendly — indices directly map to addresses in the dense input.

2. **Conflict-Free Access Pattern (SpMV)**: For SpMV, each thread computes the dot product of a sub-column of $B$ with the corresponding elements of dense vector $A$. The N:M balance guarantees that when $A$ is loaded into shared memory partitioned by the sparse column structure, the 32 threads in a warp access 32 distinct memory banks — **zero bank conflicts**. Without the N:M constraint, threads accessing the same bank would serialize, destroying bandwidth.

3. **Conflict-Free Broadcast Access Pattern (SpMM)**: For SpMM, the dense $A$ tile is stored in shared memory row-major. Threads are mapped to columns of the sparse $B$ tile. The N:M balance ensures that within each warp, all 32 threads reading scattered positions in $A$'s shared memory either access **different banks** (conflict-free) or access the **exact same address** within a bank (broadcast, handled by hardware at no cost). This is because the N:M constraint restricts column indices within each window to $[0, M)$, bounding the range of concurrent read addresses to 32 entries — exactly matching the bank count.

4. **VW/BW-N:M Extensions**: By applying the N:M balance at vector-wise (VW) or block-wise (BW) granularity (e.g., VW 1$\times$4-N:M or BW 4$\times$4-N:M), nmSPARSE achieves even higher speedups through aligned memory loads and compatibility with Tensor Core MMA instructions (CUTLASS-based).

This trick is directly relevant to blockwise Sinkhorn channel permutation: the channel permutation reorders weights to maximize retained saliency under N:M constraints, but the *inference kernel* that actually executes the sparse multiply is where wall-clock speedup materializes. nmSPARSE provides the missing kernel piece that cuSPARSELt doesn't cover: general N:M ratios (not just 2:4), arbitrary $M$ (up to 32), and support for both SpMV (autoregressive decoding) and SpMM (batch inference).

## Mathematical Form

**N:M Sparsity Constraint:**

Given sparse weight matrix $B \in \mathbb{R}^{k \times n}$, the N:M constraint requires:

$$
\forall i \in [n], \forall j \in \{0, 1, \ldots, \lceil k/M \rceil - 1\}: \quad \|B_{jM:(j+1)M, i}\|_0 = N
$$

That is, in every consecutive window of $M$ elements along the reduction dimension $k$, exactly $N$ are nonzero.

**Condensed Representation:**

The data array $B_\text{data} \in \mathbb{R}^{(k \cdot N/M) \times n}$ stores only nonzero values, condensed vertically. The index array $B_\text{idx} \in \{0, \ldots, M-1\}^{(k \cdot N/M) \times n}$ records the position of each nonzero within its window of $M$.

Storage: $N/M$ fraction of dense for data, plus $\lceil\log_2 M\rceil$ bits per nonzero for indices.

**SpMV: $C = A \times B$ where $A \in \mathbb{R}^{1 \times k}$, $B$ is N:M sparse:**

Each output element $C_j$ is:

$$
C_j = \sum_{i=0}^{k \cdot N/M - 1} A[\text{idx}[i,j]] \cdot B_\text{data}[i,j]
$$

**Conflict-Free Bank Access (SpMV):**

Partition $A$ into segments of size $M$ stored in shared memory. For a warp of 32 threads computing 32 sub-columns of $B$ simultaneously, thread $t$ reads $A_\text{shared}[\text{idx}[i, t]]$ from the same segment.

With N:M balance: $\text{idx}[i,t] \in \{0, \ldots, M-1\}$ for all $t$. When $M \leq 32$ (the bank count), and the segment base is aligned, the 32 threads access addresses within a range of 32, mapping to distinct banks:

$$
\text{bank}(t) = (\text{base} + \text{idx}[i,t]) \bmod 32
$$

The N:M constraint ensures each thread's index is bounded, eliminating bank conflicts with high probability (and guaranteed for $M = 32$).

**SpMM Tiling: $C = A \times B$ where $A \in \mathbb{R}^{m \times k}$:**

$A$ tile ($m_t \times k_t$) is stored in shared memory row-major. Threads mapped to $B$ tile columns. For column index computations, the N:M balance restricts the range of column indices within each $B$ tile's window of $M$ to $[0, M)$.

Within a warp, the 32 threads reading row $r$ of $A$'s shared memory access addresses:

$$
\text{addr}(t) = r \cdot k_t + \text{idx}[i, t]
$$

Since $\text{idx}[i,t] \in \{0, \ldots, M-1\}$ and $M \leq 32$, threads either hit **distinct banks** (conflict-free) or hit the **same address** (broadcast). Both are served at full bandwidth by the hardware.

**VW/BW-N:M SpMM with Tensor Cores:**

For VW-N:M (vector-wise) with vector size $V$ and BW-N:M (block-wise) with block size $V \times V$: the condensed $A$ and $B$ tiles have regular shapes that map directly to MMA (matrix multiply-accumulate) instructions:

$$
C_\text{tile} = \sum_{s=0}^{k_t \cdot N / (M \cdot V)} A_\text{condensed}[s] \times B_\text{condensed}[s]
$$

where each $A_\text{condensed}[s]$ and $B_\text{condensed}[s]$ are dense tiles of size compatible with Tensor Core MMA (e.g., $16 \times 16$ for INT8).

## Complexity

| Operation | Dense cuBLAS | cuSPARSELt (2:4 only) | nmSPARSE EW-N:M | nmSPARSE VW/BW-N:M |
|-----------|-------------|----------------------|-----------------|---------------------|
| SpMV $(1 \times k) \times (k \times n)$ | $O(kn)$ | N/A (no SpMV) | $O(kn \cdot N/M)$ | N/A |
| SpMM $(m \times k) \times (k \times n)$ | $O(mkn)$ | $O(mkn/2)$ (2:4 only) | $O(mkn \cdot N/M)$ | $O(mkn \cdot N/M)$ + Tensor Cores |
| Memory for $B$ | $O(kn)$ | $O(kn/2 + kn \cdot 2/16)$ | $O(kn \cdot N/M + kn \cdot \lceil\log_2 M\rceil / (M \cdot 8))$ | Same |
| Supported N:M | N/A | 2:4 only | Any N:M, $M \leq 32$ | VW: $M \leq 32$, BW: $M \leq 64$ |
| Bank conflicts | None (dense) | None (HW) | **Zero** (N:M balance) | **Zero** + Tensor Core |

**Measured speedups on A100 (vs. cuBLAS):**

| Sparsity | SpMV avg (up to) | SpMM EW avg (up to) | SpMM VW32 avg (up to) | SpMM BW4x4 avg (up to) |
|----------|------------------|--------------------|-----------------------|----------------------|
| 50% | 1.4$\times$ (1.9$\times$) | 1.2$\times$ (1.7$\times$) | 2.4$\times$ (3.3$\times$) | 2.5$\times$ (3.5$\times$) |
| 75% | 2.1$\times$ (2.3$\times$) | 1.4$\times$ (1.9$\times$) | 2.7$\times$ (3.8$\times$) | 2.8$\times$ (4.0$\times$) |
| 90% | 3.5$\times$ (5.1$\times$) | 1.3$\times$ (2.1$\times$) | 4.0$\times$ (6.0$\times$) | 2.8$\times$ (4.0$\times$) |

**Kernel profiling (M25 shape, 90% sparsity):**

| Metric | cuSPARSE | nmSPARSE-EW | nmSPARSE-VW4 | nmSPARSE-VW32 |
|--------|----------|-------------|--------------|---------------|
| Latency (ms) | 6.79 | 0.24 | 0.12 | 0.08 |
| DRAM util (%) | 0.15 | 3.19 | 7.58 | 9.39 |
| Shared mem util (%) | 0.71 | 10.45 | 45.88 | 52.05 |
| FP unit util (%) | 2.87 | 18.72 | 21.79 | 43.13 |

## Applicability

- **Inference with channel-permuted N:M sparse LLMs:** After PermLLM or learnable permutation finds the optimal channel order and N:M mask, nmSPARSE provides the inference kernel that actually executes the sparse multiply at general N:M ratios (not just 2:4). This enables using 4:8, 2:8, or other ratios that may give better accuracy-speed tradeoffs.
- **Autoregressive decoding (SpMV):** Unlike cuSPARSELt which only supports SpMM, nmSPARSE's SpMV kernel handles the batch-size-1 decoding case critical for LLM inference. The conflict-free access pattern is especially effective here since SpMV is memory-bandwidth-bound.
- **Transformer attention and FFN layers:** Validated on BERT-Large (SQuAD-1.1) with 50--90% N:M sparsity at element, vector, and block granularities. Adding the N:M balance constraint to VW/BW sparsity has no observable accuracy impact vs. unconstrained VW/BW (Table 3 in paper).
- **Mixed-granularity per-layer sparsity:** Different layers can use different N:M configurations (EW, VW, BW) and sparsity ratios, all supported by nmSPARSE. This enables layer-wise sparsity optimization matching each layer's sensitivity.
- **Composable with Sparse Tensor Cores:** VW/BW-N:M kernels are built on CUTLASS MMA, directly leveraging Tensor Cores for INT8 precision. At 50% sparsity, cuSPARSELt (2:4 on Sparse Tensor Core) still wins; at 75%+ sparsity, nmSPARSE outperforms.
- **Integration with sparse compilers:** nmSPARSE is integrated into SparTA (sparse DNN compiler), enabling end-to-end sparse model inference with automatic kernel selection.

## Limitations

- At 50% sparsity, the hardware-native Sparse Tensor Core path (cuSPARSELt, 2:4 only) is faster than nmSPARSE's software kernels — nmSPARSE's advantage emerges at 75%+ sparsity
- Maximum window size $M = 32$ for EW/VW, $M = 64$ for BW — larger windows lose the bank-conflict-free guarantee as addresses span beyond 32 banks
- SpMV kernel is memory-bandwidth-bound; on small matrices, kernel launch overhead dominates and speedup is limited
- VW/BW-N:M requires model accuracy validation at coarser granularity — the paper shows no accuracy loss from adding N:M balance to VW/BW, but this was only validated on BERT-Large
- The condensed representation requires a preprocessing step to convert from dense+mask to condensed format; this is a one-time offline cost
- Only FP32 and INT8 precisions evaluated; FP16/BF16 support would require kernel modifications for modern mixed-precision training workflows
- No support for dynamic sparsity (mask changes each iteration) — assumes static masks computed once before inference

## Implementation Notes

```python
# nmSPARSE kernel design (pseudocode)

# === SpMV: Conflict-Free Access ===
# __global__ void SpMV(float *A, float *B, int *B_IDX, float *C) {
#     __shared__ float A_shared[SHARE_TILE_A_LEN];
#     float A_reg, B_reg;
#     float C_reg[MINIBATCH] = 0;
#
#     // Load A tile from global memory to shared memory
#     LoadTile(A_shared, A);
#
#     // Main loop: iterate over N:M windows
#     for (int i = 0; i < BANK_NUM * (1-SPARSITY); i++) {
#         LoadReg(B_reg, B);
#         access_idx = GetIndex(B_IDX);  // index within window [0, M)
#         for (j = 0; j < MINIBATCH; j++) {
#             LoadRegWithIdx(A_reg, A_shared, access_idx);  // conflict-free!
#             CalcuOnReg(A_reg, B_reg, C_reg);
#         }
#     }
#     Store(C, C_reg);
# }

# === SpMM: Conflict-Free Broadcast Access ===
# __global__ void SpMM(float *A, float *B, int *B_IDX, float *C) {
#     __shared__ float A_shared[SHARE_TILE_A_LEN];
#     __shared__ float B_shared[SHARE_TILE_B_LEN];
#     float A_reg, B_reg;
#     float C_reg[REG_TILE_A_LEN][REG_TILE_B_LEN] = 0;
#
#     for (; mz > 0; mz -= kBlockItemsK) {
#         LoadTile(A_shared, A);  // dense A tile -> shared mem (row-major)
#         LoadTile(B_shared, B);  // condensed B tile -> shared mem
#
#         for (int i = 0; i < K_BLOCK_TILE_LEN * (1-SPARSITY); i++) {
#             LoadReg(B_reg, B_shared);
#             access_idx = GetIndex(B_IDX);  // bounded to [0, M)
#             LoadRegWithIdx(A_reg, A_shared, access_idx);  // broadcast OK!
#             CalcuOnReg(A_reg, B_reg, C_reg);
#         }
#     }
#     Store(C, C_reg);
# }

# Key insight: N:M balance guarantees that within a warp of 32 threads,
# the scattered access indices into A_shared are bounded to [0, M).
# When M <= 32 (= number of shared memory banks):
# - Different indices -> different banks -> conflict-free
# - Same index -> same address -> broadcast (free)
# Without N:M balance, indices could span [0, k), causing arbitrary
# bank conflicts that serialize shared memory accesses.

# Python-level usage with SparTA integration:
import torch
# from sparta import nmSPARSE  # hypothetical API

def condense_nm_sparse(W, mask, M=4):
    """Convert dense weight + N:M mask to condensed representation."""
    k, n = W.shape
    N = int(mask.sum() / (k * n / M))  # infer N from mask

    # Condense: keep only nonzero values and their indices
    data = []
    indices = []
    for col in range(n):
        for window in range(k // M):
            start = window * M
            window_mask = mask[start:start+M, col]
            nonzero_pos = window_mask.nonzero(as_tuple=True)[0]
            data.extend(W[start + nonzero_pos, col].tolist())
            indices.extend(nonzero_pos.tolist())

    condensed_k = k * N // M
    data = torch.tensor(data).reshape(condensed_k, n)
    idx = torch.tensor(indices, dtype=torch.int8).reshape(condensed_k, n)
    return data, idx
```

## References

- Lin, B., Zheng, N., Wang, L., Cao, S., Ma, L., Zhang, Q., Zhu, Y., Cao, T., Xue, J., Yang, Y. & Yang, F. (2023). Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning. MLSys 2023.
- Mishra, A., Latorre, J.A., Pool, J., Stosic, D., Stosic, D., Venber, G. & Micikevicius, P. (2021). Accelerating Sparse Deep Neural Networks. arXiv:2104.08378.
- Pool, J. & Yu, C. (2021). Channel Permutations for N:M Sparsity. NeurIPS 2021.
- Zheng, N. et al. (2022). SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute. OSDI 2022.
- Cao, S. et al. (2019). Efficient and Effective Sparse LSTM on FPGA with Bank-Balanced Sparsity. ISPD 2019.

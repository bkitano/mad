# 050: Fused3S: Fused Sparse Attention on Tensor Cores

**Category**: kernel
**Gain type**: efficiency
**Source**: Li & Chandramowlishwaran "Fused3S: Fast Sparse Attention on Tensor Cores" (ICS 2025)
**Paper**: [papers/fused3s-fast-sparse-attention.pdf]
**Documented**: 2025-06-15

## Description

Fused3S is the first fully on-chip, fused sparse attention kernel designed for GPU tensor cores. Sparse attention — used in graph neural networks (GATs, Graph Transformers), sparse transformers, and models with dynamic sparsity masks — decomposes into a sequence of three sparse matrix operations (the **3S pattern**):

1. **SDDMM** (Sampled Dense-Dense Matrix Multiplication): Compute attention scores $S = QK^T \odot A$ only at positions where the sparse mask $A$ is nonzero
2. **Softmax**: Row-wise normalization $E = \text{softmax}(S)$
3. **SpMM** (Sparse Matrix-dense Matrix Multiplication): Aggregate features $O = EV$

Prior work either optimizes individual operations (SDDMM or SpMM separately) or fuses them on CUDA cores without leveraging tensor cores. The gap is that: (a) unfused kernels materialize the intermediate score matrix $S$ in global memory, causing high memory traffic; (b) CUDA-core fused kernels (DF-GNN) miss the 8× tensor core throughput improvement from V100 to H100.

Fused3S closes this gap with three core innovations:

1. **Binary Sparse Block (BSB) format**: A tensor-core-aligned sparse format that encodes the sparsity pattern with a compact bitmap (128 bits per 16×8 block) rather than explicit index arrays, reducing metadata overhead.

2. **Full 3S fusion**: SDDMM, softmax, and SpMM are fused into a single kernel. Intermediate attention scores ($S$) stay in registers/shared memory and are never written to HBM. Softmax uses the **online (blocked) algorithm** from FlashAttention-2 for numerical stability.

3. **Register remapping**: A permuted data layout for $Q$, $K$, and $V$ that converts scattered 16-bit loads into coalesced 128-bit loads, improving memory throughput for the irregular gather operations inherent in sparse attention.

## Mathematical Form

**Sparse Attention (3S Pattern):**

Given queries, keys, values $Q, K, V \in \mathbb{R}^{N \times d}$ and a sparse attention pattern $A \in \mathbb{R}^{N \times N}$ (adjacency matrix, mask, etc.):

$$
O = \text{softmax}(QK^T \odot A) \cdot V
$$

This decomposes into:

**Step 1 — SDDMM:** Compute attention scores sampled by sparsity pattern:

$$
S = QK^T \odot A
$$

Only the $z$ nonzero entries of $A$ are computed: $S_{ij} = \mathbf{q}_i^T \mathbf{k}_j$ for $(i,j)$ where $A_{ij} \neq 0$.

**Step 2 — Softmax:** Row-wise normalization with max-stabilization:

$$
E_{ij} = \frac{\exp(S_{ij} - \max_k S_{ik})}{\sum_k \exp(S_{ik} - \max_k S_{ik})}
$$

**Step 3 — SpMM:** Feature aggregation:

$$
O = E \cdot V
$$

**Online Softmax (Blocked):**

For numerical stability across blocks, Fused3S tracks running row-wise max $\mathbf{m}_o$ and normalization factor $\mathbf{l}_o$ across tensor core blocks (TCBs):

$$
\mathbf{m}_i = \max(\mathbf{m}_o, \text{rowmax}(S_i))
$$

$$
E_i = \exp(S_i - \mathbf{m}_i)
$$

$$
\mathbf{l}_o = \text{diag}(e^{\mathbf{m}_o - \mathbf{m}_i}) \mathbf{l}_o + \text{rowsum}(E_i)
$$

When processing block $j$, the running output is rescaled:

$$
O_i \leftarrow \text{diag}(e^{\mathbf{m}_o - \mathbf{m}_i}) O_i + E_i \hat{V}_j
$$

Final output: $O_i \leftarrow \text{diag}(\mathbf{l}_o)^{-1} O_i$

**Binary Sparse Block (BSB) Format:**

The sparse matrix $A$ is stored as:
- **Row windows (RW)**: $A$ is divided into row windows of height $r$
- **Tensor Core Blocks (TCB)**: Each RW is compacted (removing all-zero columns) and partitioned into blocks of shape $r \times c$ matching the MMA tile (e.g., $16 \times 8$)
- **Bitmap**: Each TCB's sparsity is encoded as a binary bitmap of $r \times c$ bits (e.g., 128 bits for $16 \times 8$)

Three metadata arrays:
- `tcb_row_offset (tro)`: Number of TCBs per row window
- `col_sparse_to_dense (sptd)`: Mapping from compacted column indices to original columns
- `bitmap`: Fixed-size bitmask per TCB encoding which elements are nonzero

**Mixed Precision Pipeline:**

| Matrix | Precision |
|--------|-----------|
| $Q, K, V$ (inputs) | fp16 |
| $S$ (SDDMM scores) | fp32 (accumulated) |
| Softmax computation | fp32 |
| $E$ (normalized scores) | fp32 → fp16 (cast for SpMM) |
| $O$ (output) | fp32 |

**Key Definitions:**

- $N$ — number of nodes/tokens
- $d$ — feature dimension
- $z$ — number of nonzero entries in $A$
- $r, c$ — tensor core block dimensions (row window height, TCB width)
- TCB — Tensor Core Block, the basic sparse tile unit
- RW — Row Window, a horizontal strip of the sparse matrix
- BSB — Binary Sparse Block format

## Complexity

| Operation | Unfused (separate kernels) | DF-GNN (CUDA fused) | Fused3S (TC fused) |
|-----------|--------------------------|---------------------|-------------------|
| SDDMM compute | TC or CUDA, separate | CUDA core, fused | **Tensor Core, fused** |
| Softmax | Separate kernel | Fused (CUDA) | **Fused (TC + fp32)** |
| SpMM compute | TC or CUDA, separate | CUDA core, fused | **Tensor Core, fused** |
| $S$ materialized in HBM | Yes | **No** | **No** |
| $E$ materialized in HBM | Yes | **No** | **No** |
| Precision | Varies | fp32 only | **fp16/fp32 mixed** |

**Memory:** BSB format footprint: $32(\frac{N}{r} + bc + brc)$ bits, where $b$ = number of blocks, $bc$ = stored columns after compaction, $rc$ = elements per block. The bitmap encoding is more compact than explicit index formats (ME-TCF, TCF) that store integer indices per nonzero.

**Speedups (geometric mean over graph datasets):**

| Baseline | A30 GPU | H100 GPU |
|----------|---------|----------|
| DF-GNN_tiling | 2.7× | 2.8× |
| DF-GNN_hyper | 1.7× | 2.2× |
| FlashSparse | 1.5× | 1.6× |
| FlashSparse (stable softmax) | 2.2× | 4.4× |
| PyG | 12.3× | 14.7× |

**End-to-end Graph Transformer inference:** 1.05–5.36× speedup over all 3S baselines across datasets and GPUs.

## Applicability

- **Graph Attention Networks (GATs)**: The 3S pattern is the core computation; sparse $A$ is the adjacency matrix
- **Graph Transformers**: Full-graph attention with structural bias encoded in $A$; Fused3S replaces the attention kernel in DGL implementations
- **Sparse Transformers**: Models using static or dynamic sparse masks (BigBird, Longformer) where $A$ defines the attention pattern
- **Dynamic sparse attention**: Models that generate $A$ on-the-fly (SEA, AGNN) — Fused3S handles arbitrary sparsity patterns via BSB format
- **Batched small graphs**: Graph property prediction tasks where many small graphs are batched together — Fused3S achieves up to 16.3× speedup on LRGB/OGB benchmarks
- **Hardware**: NVIDIA A30 (Ampere, 4 TCs per SM) and H100 (Hopper, 4 TCs per SM); uses PTX `mma` interface for direct register-to-register TC operations

## Limitations

- **Forward pass only**: Current implementation focuses on forward pass; backward pass (reverse SpMM + SDDMM) not yet optimized
- **Graph-focused evaluation**: Primarily benchmarked on GNN/graph transformer workloads; not yet evaluated on sequence-model sparse attention patterns (e.g., local + global windows)
- **Load imbalance on irregular graphs**: Power-law degree distributions cause high variance in TCBs per row window (CV up to 0.95 for Reddit dataset), leading to SM underutilization
- **No thread block cluster support**: Hopper's thread block clusters could enable synchronization across multiple thread blocks for better load balancing — left as future work
- **Block size sensitivity**: Performance depends on TCB size ($16 \times 8$) matching the sparsity structure; very high sparsity can lead to many near-empty TCBs
- **BSB preprocessing**: Sparse matrix must be converted to BSB format in preprocessing; adds negligible overhead for static graphs but non-trivial for dynamic patterns
- **Online softmax stability**: The blocked online softmax can be less stable than global softmax for small block sizes, though this is mitigated by fp32 accumulation

## Implementation Notes

```python
# Pseudocode for Fused3S algorithm (Algorithm 1 from paper)
# Open-source implementation: https://github.com/HPCForge/Fused3S

# === BSB Format Construction (preprocessing) ===
def construct_bsb(A, r, c):
    """Convert sparse matrix A to Binary Sparse Block format.

    1. Divide A into row windows (RW) of height r
    2. Within each RW, remove all-zero columns (compaction)
    3. Tile compacted RW into TCBs of shape r × c
    4. Encode each TCB as a fixed-size bitmap (r*c bits)
    """
    tcb_row_offset = []  # num TCBs per RW
    col_sparse_to_dense = []  # column index mapping
    bitmaps = []  # r*c bit patterns per TCB
    # ... (see paper Figure 1 for detailed format)
    return tcb_row_offset, col_sparse_to_dense, bitmaps

# === Fused3S Kernel (single GPU kernel) ===
def fused3s_kernel(Q, K, V, A_bsb):
    """Fused SDDMM + Softmax + SpMM on tensor cores.

    Key: intermediate S and E never leave on-chip memory.
    """
    tro, sptd, bitmap = A_bsb
    T_r = N // r  # number of row blocks

    for i in range(T_r):  # outer loop: row windows (thread blocks)
        # Initialize online softmax accumulators
        m_o = -inf  # running row-wise max
        l_o = 0     # running normalization factor
        O_i = 0     # running output accumulator

        Q_i = load_to_smem(Q[i*r:(i+1)*r])  # shared memory

        t = tro[i+1] - tro[i]  # number of TCBs in this RW
        c_indices = sptd[i]     # column indices for this RW

        # Gather relevant K, V rows based on column indices
        K_hat = K[c_indices]  # non-contiguous gather
        V_hat = V[c_indices]

        T_c = ceil(t / W)  # warp partitioning (split-column)

        for j in range(T_c):  # inner loop: TCBs (warps)
            # --- SDDMM via TBGemm (tensor core) ---
            S_i = TBGemm(Q_i, K_hat_j.T, 0)  # fp16→fp32
            S_i = apply_bitmap_mask(S_i, bitmap[j])

            # --- Online Softmax (numerically stable) ---
            m_i = max(m_o, rowmax(S_i))
            E_i = exp(S_i - m_i)  # fp32
            l_o = diag(exp(m_o - m_i)) * l_o + rowsum(E_i)

            E_i_fp16 = cast_to_fp16(E_i)  # store in SMEM

            # --- SpMM via TBGemm (tensor core) ---
            O_i = diag(exp(m_o - m_i)) * O_i  # rescale
            O_i += TBGemm(E_i_fp16, V_hat_j, O_i)  # fp16→fp32

            m_o = m_i

        # Final normalization
        O_i = diag(1.0 / l_o) * O_i
        write_to_hbm(O_i)

# === Register Remapping Optimization ===
# Problem: gathering K[c_indices] produces scattered 16-bit loads
# Solution: permute columns of K^T so each thread issues one
#           128-bit coalesced load instead of four 32-bit scattered loads
# Same permutation applied to Q columns for correctness
# Softmax unaffected (S already in registers after SDDMM)
# Reverse permutation applied when writing O back to HBM
```

## References

- Li, Z. & Chandramowlishwaran, A. "Fused3S: Fast Sparse Attention on Tensor Cores" (ICS 2025). arXiv:2505.08098. Code: https://github.com/HPCForge/Fused3S
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024). Online softmax algorithm.
- Dao, T., Fu, D.Y., Ermon, S., Rudra, A. & Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022). IO-aware tiling foundation.
- Liu, J., Cai, Z., Chen, Z. & Wang, M. "DF-GNN: Dynamic Fusion Framework for Attention Graph Neural Networks on GPUs" (TLoG 2024). CUDA-core fused 3S baseline.
- Shi, J., et al. "FlashSparse: Minimizing Computation Redundancy for Fast Sparse Matrix Multiplications on Tensor Cores" (2024). arXiv:2412.11007. TC-based separate SDDMM/SpMM baseline.
- Fan, R., Wang, W. & Chu, X. "DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication with Tensor Cores" (ASPLOS 2024). ME-TCF format for tensor core SpMM.
- Dwivedi, V.P. & Bresson, X. "A Generalization of Transformer Networks to Graphs" (AAAI DLG Workshop 2021). Graph Transformer architecture used in end-to-end evaluation.

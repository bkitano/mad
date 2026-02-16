# 066: IO-Aware Tiling (FlashAttention)

**Category**: kernel
**Gain type**: efficiency
**Source**: Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
**Paper**: [papers/flash-attention-io-aware-tiling.pdf]
**Documented**: 2025-02-11

## Description

IO-aware tiling exploits the GPU memory hierarchy to compute **exact** attention without materializing the full $N \times N$ attention matrix in high-bandwidth memory (HBM). Modern GPUs have a memory hierarchy: large but slow HBM (40-80 GB, ~1.5 TB/s) and small but fast on-chip SRAM (~20 MB, ~19 TB/s). Since compute speed has outpaced memory bandwidth, most Transformer operations are **memory-bound**, not compute-bound. IO-aware tiling restructures the attention computation to minimize reads/writes to HBM by performing all computation steps (matmul, softmax, masking, dropout, matmul) within a single fused CUDA kernel, loading tiles from HBM to SRAM and writing back only the final output.

The key insight is that reducing HBM accesses matters more than reducing FLOPs. FlashAttention actually performs **more** FLOPs than standard attention (due to recomputation in the backward pass), but runs 2-7× faster because it dramatically reduces HBM traffic.

## Mathematical Form

**Standard Attention (materialized):**

$$
S = QK^\top \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}
$$

Standard implementation writes $S$ and $P$ to HBM, requiring $\Theta(Nd + N^2)$ HBM accesses.

**IO-Aware Tiling (FlashAttention):**

Partition $Q, K, V$ into blocks of size $B_r \times d$ and $B_c \times d$:

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

where $M$ is the SRAM size.

**Core tiled computation (inner loop):**

For each block pair $(i, j)$:

1. Load $Q_i, K_j, V_j$ from HBM to SRAM
2. Compute local scores: $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$
3. Compute local softmax statistics:
$$
\tilde{m}_{ij} = \text{rowmax}(S_{ij}), \quad \tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}), \quad \tilde{\ell}_{ij} = \text{rowsum}(\tilde{P}_{ij})
$$
4. Update running statistics:
$$
m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij}), \quad \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\ell}_{ij}
$$
5. Rescale and accumulate output:
$$
O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1} \left( \text{diag}(\ell_i) e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j \right)
$$

**Key Definitions:**

- $Q, K, V \in \mathbb{R}^{N \times d}$ — query, key, value matrices
- $N$ — sequence length, $d$ — head dimension
- $M$ — SRAM size (bytes), typically ~100 KB per streaming multiprocessor
- $B_r, B_c$ — block sizes chosen to fit tiles in SRAM
- $m_i, \ell_i$ — running softmax statistics (max and sum of exponentials)

## Complexity

| Operation | Standard | IO-Aware Tiling |
|-----------|----------|-----------------|
| HBM accesses | $\Theta(Nd + N^2)$ | $\Theta(N^2 d^2 M^{-1})$ |
| FLOPs | $O(N^2 d)$ | $O(N^2 d)$ |
| Peak memory | $O(N^2)$ | $O(N)$ |

**Memory:** $O(N)$ additional memory vs $O(N^2)$ for standard attention.

**HBM reduction factor:** For $d = 64$ and $M = 100$ KB, the ratio is $\frac{d^2}{M} \approx \frac{4096}{100000} \approx 0.04$, meaning ~25× fewer HBM accesses.

**Optimality:** No exact attention algorithm can achieve $o(N^2 d^2 M^{-1})$ HBM accesses for all SRAM sizes $M \in [d, Nd]$ (proven lower bound).

## Applicability

- **Transformer self-attention**: Direct replacement, exact computation with no approximation
- **Multi-head attention**: Each head computed independently, naturally parallelizable
- **Long-context models**: Enables training with 4-16× longer sequences at same memory budget
- **Block-sparse attention**: Extends to sparse patterns with IO complexity proportional to sparsity ratio
- **Cross-attention, causal masking**: Supports all attention variants via masking within tiles

## Limitations

- Requires custom CUDA kernel (not achievable with standard PyTorch/TensorFlow ops)
- Block sizes must be tuned per GPU architecture (A100 vs H100 vs different SRAM sizes)
- Backward pass requires recomputation (trades FLOPs for memory), adding ~20% FLOPs
- Not beneficial when $d$ is very large relative to $M$ (the $d^2/M$ factor must be $\ll 1$)
- Tiling pattern is specific to attention; different fused kernels needed for other operations

## Implementation Notes

```python
# Pseudocode for FlashAttention forward pass
def flash_attention_forward(Q, K, V, M_sram):
    N, d = Q.shape
    Bc = M_sram // (4 * d)  # block size for K, V
    Br = min(Bc, d)          # block size for Q

    O = zeros(N, d)
    ell = zeros(N)  # running sum of exponentials
    m = full(N, -inf)  # running max

    for j in range(0, N, Bc):        # outer loop: K, V blocks
        Kj = K[j:j+Bc]               # load to SRAM
        Vj = V[j:j+Bc]               # load to SRAM
        for i in range(0, N, Br):     # inner loop: Q blocks
            Qi = Q[i:i+Br]            # load to SRAM

            # All computation happens on SRAM
            Sij = Qi @ Kj.T           # local attention scores
            m_new = maximum(m[i:i+Br], Sij.max(axis=1))
            P_tilde = exp(Sij - m_new[:, None])
            ell_new = exp(m[i:i+Br] - m_new) * ell[i:i+Br] + P_tilde.sum(axis=1)

            # Rescale previous output and add new contribution
            O[i:i+Br] = (diag(ell[i:i+Br]) * exp(m[i:i+Br] - m_new) @ O[i:i+Br]
                         + P_tilde @ Vj) / diag(ell_new)

            m[i:i+Br] = m_new
            ell[i:i+Br] = ell_new

    return O
```

## References

- Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Re, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022). arXiv:2205.14135
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024). arXiv:2307.08691
- Dao, T., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024). arXiv:2407.08691
- Milakov, M. & Gimelshein, N. "Online normalizer calculation for softmax" (2018). arXiv:1805.02867
- Rabe, M.N. & Staats, C. "Self-attention Does Not Need O(n²) Memory" (2021). arXiv:2112.05682

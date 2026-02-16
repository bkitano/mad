# Batch-Reduce GEMM (BRGEMM)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Georganas et al. (2019), Intel Corporation
**Paper**: [papers/batch-reduce-gemm.pdf]
**Documented**: 2026-02-14

## Description

Unify all deep learning primitives (LSTM, CNN, MLP) into a single optimized kernel: the batch-reduce GEMM. Instead of calling $N$ separate small GEMMs and accumulating results externally (which wastes memory bandwidth on intermediate stores/loads), BRGEMM multiplies $N$ pairs of sub-tensor blocks and *reduces* (accumulates) them into a single output block in-register. This is the WY trick generalized: convert $N$ rank-$k$ updates into one fused operation where the accumulation stays in fast registers, never spilling to memory. By having one highly-optimized kernel serve all DL workloads, library development reduces from $O(\text{architectures} \times \text{primitives})$ to $O(\text{architectures})$.

## Mathematical Form

**Core Operation:**

$$
C_j = \beta \cdot C_j + \alpha \sum_{i=0}^{N-1} A_i \cdot B_i
$$

where $A_i \in \mathbb{R}^{m \times k}$, $B_i \in \mathbb{R}^{k \times n}$, $C_j \in \mathbb{R}^{m \times n}$, $\alpha, \beta \in \mathbb{R}$.

**Key Definitions:**

- $A_i, B_i$ — input sub-tensor block pairs (can reside at arbitrary positions in tensors)
- $C_j$ — output accumulator block
- $N$ — number of block pairs to multiply and reduce
- $\alpha, \beta$ — scaling parameters (standard GEMM convention)

**Contrast with standard batched GEMM:**

Standard batched GEMM: $C_i = \beta \cdot C_i + \alpha \cdot A_i \cdot B_i$ (separate output per pair)

Batch-reduce GEMM: $C = \beta \cdot C + \alpha \sum_i A_i \cdot B_i$ (single accumulated output)

The critical difference is the **reduction** — partial products stay in accumulator registers and are never written to memory until the final result.

**Application to LSTM:**

For LSTM gates $i_t = \sigma(W_i \cdot x_t + R_i \cdot h_{t-1} + b_i)$, instead of two separate GEMMs:

$$
i_t = \sigma\left(\text{BRGEMM}\left(\{W_i, R_i\}, \{x_t, h_{t-1}\}\right) + b_i\right)
$$

The element-wise activation $\sigma$ is fused while the output block is still hot in cache.

**Application to CNN (direct convolution):**

For a convolution with $R \times S$ kernel, each output pixel requires summing $R \times S \times C_b$ partial products:

$$
O[n][k_b][oj][oi] = \text{BRGEMM}\left(\{W[k_b][c_b][r][s]\}, \{I[n][c_b][ii+r][ij+s]\}\right)
$$

where the batch dimension covers all $(r, s, c_b)$ combinations.

## Complexity

| Operation | Separate GEMMs | Batch-Reduce GEMM |
|-----------|---------------|-------------------|
| Memory writes | $N \times O(mn)$ intermediates | $O(mn)$ single output |
| Kernel launches | $N$ | $1$ |
| Register reuse | None (cold cache) | Full (accumulate in-register) |
| Bandwidth | $O(N \cdot mn)$ load/store | $O(mn)$ store, $O(N \cdot mk + N \cdot kn)$ load |

**Key insight:** Output data movement is reduced by $N\times$ since accumulation happens in registers. For LSTM with $C = K = 1024$, the BRGEMM runs at 93.3% of peak (2550 GFLOPS on Skylake-SP 8180).

**Memory:** Same as separate GEMMs for inputs; saves $O(N \cdot mn)$ for intermediates.

## Applicability

- **LSTM/GRU cells:** Fuse $W \cdot x_t + R \cdot h_{t-1}$ into one BRGEMM call per gate
- **Direct convolutions:** All spatial kernel positions accumulated without intermediate buffers
- **Fully connected / MLP layers:** Blocked matrix multiply with fused activation (activation applied while output still in cache)
- **Transformer attention:** Can express batched attention head computations
- **Tensor compilers:** Serves as the single building block for TVM, PlaidML, MLIR backends
- Achieves average 83% efficiency on ResNet-50 convolutions, outperforming vendor-optimized MKL-DNN

## Limitations

- Requires JIT code generation for peak performance (architecture-specific microkernel)
- Input sub-blocks must be addressable via pointer arrays (flexible but requires setup)
- The reduction dimension $N$ must be known at kernel launch
- Primarily optimized for CPUs (AVX-512, VNNI); GPU adaptation uses different blocking strategies
- Small batch sizes $b_n$ can still be handled via flexible blocking factors

## Implementation Notes

```python
# Batch-Reduce GEMM pseudocode
# Key: accumulation stays in registers across all N block pairs

def batch_reduce_gemm(A_ptrs, B_ptrs, C, N, m_b, n_b, k):
    """
    A_ptrs: list of N pointers to [m_b, k] blocks
    B_ptrs: list of N pointers to [k, n_b] blocks
    C:      output [m_b, n_b] accumulator
    """
    # Block C into register-sized sub-blocks
    for i_m in range(0, m_b, REG_BLOCK_M):
        for i_n in range(0, n_b, REG_BLOCK_N):
            # Load accumulator into vector registers
            acc_regs = load_registers(C[i_m:, i_n:])

            # Accumulate ALL N block-pairs in-register
            for i in range(N):
                for i_k in range(k):
                    # Outer product microkernel (FMA)
                    a_col = A_ptrs[i][i_m:, i_k]
                    b_row = B_ptrs[i][i_k, i_n:]
                    acc_regs += outer_product(a_col, b_row)

            # Store only once after all N reductions
            store_registers(C[i_m:, i_n:], acc_regs)

# LSTM example: fuse W*x + R*h into single BRGEMM
def lstm_cell_brgemm(W_i, R_i, x_t, h_prev, bias):
    A_ptrs = [&W_i[block_c], &R_i[block_c] for block_c in channel_blocks]
    B_ptrs = [&x_t[block_c],  &h_prev[block_c] for block_c in channel_blocks]
    out = zeros(hidden_size)
    batch_reduce_gemm(A_ptrs, B_ptrs, out, N=2*num_channel_blocks)
    return sigmoid(out + bias)  # Fused while out is hot in cache
```

## References

- Georganas, E. et al. (2019). High-Performance Deep Learning via a Single Building Block. arXiv:1906.06440.
- Intel oneDNN (MKL-DNN) documentation — BRGEMM kernel API.
- LIBXSMM library — JIT-based small GEMM and BRGEMM implementations.

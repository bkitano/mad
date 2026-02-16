# 134: TSQR (Tall-Skinny QR)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Demmel et al., "Communication-avoiding parallel and sequential QR factorizations"
**Paper**: [papers/tsqr-communication-avoiding.pdf]
**Documented**: 2026-02-15

## Description

TSQR (Tall-Skinny QR) is a communication-avoiding algorithm for computing QR factorization of matrices with many more rows than columns ($m \gg n$). It uses a tree-based reduction structure to minimize communication between processors (in parallel) or between levels of the memory hierarchy (in sequential), achieving logarithmic communication costs instead of linear.

The key insight: rather than applying Householder transformations sequentially down the matrix, TSQR factors local blocks in parallel, then combines the resulting $R$ factors recursively using a reduction tree. This restructuring reduces messages by a factor of $n$ and bandwidth by a factor of $n$ compared to ScaLAPACK.

## Mathematical Form

**Binary Tree Reduction:**

For $m \times n$ matrix $A$ with $P$ processors, partition into blocks:

$$
A = \begin{pmatrix} A_0 \\ A_1 \\ \vdots \\ A_{P-1} \end{pmatrix}
$$

**Stage 0** (local factorizations in parallel):

$$
A_i = Q_{i0} R_{i0} \quad \text{for } i = 0, 1, \ldots, P-1
$$

**Stage 1** (pairwise reduction):

$$
\begin{pmatrix} R_{00} \\ R_{10} \end{pmatrix} = Q_{01} R_{01}, \quad \begin{pmatrix} R_{20} \\ R_{30} \end{pmatrix} = Q_{11} R_{11}
$$

**Stage** $\log_2 P$:

$$
\begin{pmatrix} R_{01} \\ R_{11} \end{pmatrix} = Q_{02} R_{02}
$$

Final factorization:

$$
A = \begin{pmatrix} Q_{00} \\ & Q_{10} \\ & & Q_{20} \\ & & & Q_{30} \end{pmatrix} \begin{pmatrix} Q_{01} \\ & Q_{11} \end{pmatrix} \begin{pmatrix} Q_{02} \end{pmatrix} R_{02}
$$

**Flat Tree (Sequential):**

For sequential out-of-DRAM, use flat tree:

$$
A = Q_{00} \cdot Q_{01} \cdot Q_{02} \cdots Q_{0,P-1} \cdot R_{0P}
$$

where each $Q_{0j}$ combines $R_{0,j-1}$ with next block $A_j$.

## Complexity

### Parallel (Binary Tree on $P$ processors)

| Metric | ScaLAPACK | TSQR |
|--------|-----------|------|
| **Messages** | $2n \log(P)$ | $\log(P)$ |
| **Words** | $\frac{n^2 \log(P)}{P}$ | $\Theta\left(\frac{n^3}{W}\right)$ |
| **Flops** | $\frac{2mn^2}{P} + \frac{2}{3}n^3 \log(P)$ | $2mn^2 - \frac{2}{3}n^3$ |

**Speedup:** $O(n)$ fewer messages, same total bandwidth (but better locality)

### Sequential (Fast memory of size $W$)

| Metric | Standard QR | TSQR |
|--------|-------------|------|
| **Words transferred** | $\Theta(mn^2)$ | $2mn + \frac{mn^2}{W}$ |
| **Slow memory accesses** | $\Theta(mn^2/W)$ | $\Theta(mn/W)$ |

**Speedup:** Factor of $n$ reduction in communication when $W \geq n^2$

## Applicability

**Ideal for:**

1. **Tall-skinny matrices**: $m \gg n$ (many rows, few columns)
   - Block iterative methods (GMRES, MINRES, CG)
   - s-step Krylov methods
   - Panel factorization in 2D block cyclic QR
   - Principal Component Analysis (PCA)

2. **Communication-bound environments**:
   - Distributed memory clusters (reduce message latency)
   - Out-of-DRAM computation (minimize slow memory access)
   - GPU/accelerator-based systems (minimize PCIe transfers)
   - Multi-level memory hierarchies

3. **Modern supercomputers**: Where communication costs dominate computation

**Performance gains:**
- Speedup of 6.7x on 16-processor Pentium III cluster
- Speedup of 4x on 32-processor IBM BG/L
- Speedup of 13x on GPU implementations

## Limitations

- **Tall-skinny only**: Designed for $m \gg n$; general $m \times n$ requires CAQR (Communication-Avoiding QR)
- **Redundant computation**: Performs $2mn^2$ flops vs optimal $2mn^2 - \frac{2n^3}{3}$
- **Tree structure overhead**: Binary tree not optimal for all architectures
- **$Q$ factor storage**: Full $Q$ requires storing tree of local factors (memory overhead)
- **Load imbalance**: Binary tree assumes uniform block sizes and processor speeds

## Implementation Notes

```python
# Pseudocode for parallel TSQR on binary tree
def parallel_tsqr(A, num_procs, tree_structure='binary'):
    # Stage 0: Local QR factorizations
    R_factors = []
    for i in range(num_procs):
        Q_local, R_local = qr_factorization(A[i])
        R_factors.append(R_local)

    # Reduction stages
    log_stages = ceil(log2(num_procs))
    for stage in range(log_stages):
        new_R = []
        for pair in group_pairs(R_factors):
            R_stacked = vstack(pair)
            Q_reduce, R_combined = qr_factorization(R_stacked)
            new_R.append(R_combined)
        R_factors = new_R

    return R_factors[0]  # Final R factor

# Sequential TSQR (flat tree, out-of-DRAM)
def sequential_tsqr(A, block_size):
    num_blocks = ceil(m / block_size)

    # Initial block
    Q, R = qr_factorization(A[0:block_size])

    # Combine with successive blocks
    for i in range(1, num_blocks):
        A_block = A[i*block_size:(i+1)*block_size]
        R_stacked = vstack([R, A_block])
        Q_i, R = qr_factorization(R_stacked)

    return R
```

**Key optimizations:**

1. **Structured Householder QR**: Exploit upper triangular structure of stacked $R$ factors
   - Saves $5\times$ flops and $3\times$ storage in parallel case
   - Saves $2\times$ flops and $2\times$ storage in sequential case

2. **Recursive QR**: Use Elmroth-Gustavson recursive algorithm for local factorizations
   - 20% faster than LAPACK for square matrices
   - Up to 3x faster for tall-thin matrices

3. **Tree tuning**: Choose tree structure based on architecture
   - Binary tree: minimizes message count
   - Flat tree: minimizes bandwidth for sequential/out-of-DRAM
   - Hybrid trees: adapt to heterogeneous networks

4. **Integration with CAQR**: Use TSQR for panel factorization in general 2-D block QR

## References

- Demmel, Grigori, Hoemmen, Langou, "Communication-avoiding parallel and sequential QR factorizations", 2008
- Elmroth and Gustavson, "Applying recursion to serial and parallel QR factorization leads to better performance"
- Ballard et al., "Communication Optimal Parallel Multiplication of Sparse Random Matrices"

# 128: Tiled QR Factorization

**Category**: parallelization
**Gain type**: efficiency
**Source**: Buttari et al., "Parallel Tiled QR Factorization for Multicore Architectures", LAPACK Working Note #190
**Paper**: [papers/recursive-qr-factorization.pdf]
**Documented**: 2026-02-15

## Description

Tiled QR factorization is a fine-grained parallel algorithm that breaks QR decomposition into small tasks operating on square blocks (tiles) of data. These tasks are dynamically scheduled based on dependencies in a Directed Acyclic Graph (DAG), enabling out-of-order execution that completely hides sequential bottlenecks and maximizes parallelism on multicore architectures.

The key innovation is replacing Level 2 BLAS operations (which have poor memory performance) with Level 3 BLAS operations (matrix-matrix multiplication) at a cost of 25% more floating-point operations, achieving significantly better performance on modern memory hierarchies.

## Mathematical Form

**Core Operations:**

The matrix $A \in \mathbb{R}^{pb \times qb}$ is partitioned into blocks $A_{ij}$ of size $b \times b$:

$$
A = \begin{pmatrix}
A_{11} & A_{12} & \cdots & A_{1q} \\
A_{21} & A_{22} & \cdots & A_{2q} \\
\vdots & \vdots & \ddots & \vdots \\
A_{p1} & A_{p2} & \cdots & A_{pq}
\end{pmatrix}
$$

**Four Elementary Operations:**

1. **DGEQT2**: Panel factorization of diagonal block $A_{kk}$ produces:
   - $A_{kk} \leftarrow V_{kk}$ (unit lower triangular with Householder reflectors)
   - $R_{kk}$ (upper triangular)
   - $T_{kk}$ (compact WY representation)

2. **DLARFB**: Apply transformation to trailing block:
   $$A_{kj} \leftarrow (I - V_{kk}T_{kk}V_{kk}^T)A_{kj}$$

3. **DTSQT2**: QR factorization of coupled blocks:
   $$\begin{pmatrix} R_{kk} \\ A_{ik} \end{pmatrix} \leftarrow \begin{pmatrix} I \\ V_{ik} \end{pmatrix} \tilde{R}_{kk}, \quad T_{ik} \leftarrow T_{ik}$$

4. **DSSRFB**: Apply coupled transformation:
   $$\begin{pmatrix} A_{kj} \\ A_{ij} \end{pmatrix} \leftarrow \left(I - \begin{pmatrix} I \\ V_{ik} \end{pmatrix} \langle T_{ik} \rangle \cdot (I \, V_{ik}^T)\right) \begin{pmatrix} A_{kj} \\ A_{ij} \end{pmatrix}$$

**DAG Structure:**

Tasks are organized into a dependency graph where:
- DGEQT2 has highest priority (critical path)
- DTSQT2, DLARFB, DSSRFB have descending priorities
- Dependencies prevent violations, enabling asynchronous execution

## Complexity

| Operation | LAPACK | Tiled QR |
|-----------|--------|----------|
| Flops | $\frac{2}{3}n^2(m - \frac{n}{3})$ | $\frac{5}{6}n^2(m - \frac{n}{3})$ (25% overhead) |
| Panel factorization | $O(n^2)$ (sequential bottleneck) | Hidden by parallelism |
| Level 3 BLAS ratio | Low | High |

**Memory:** Block Data Layout (BDL) improves cache performance by storing each $b \times b$ block contiguously, rather than using column-major format.

**Parallelism:** Achieves near-linear speedup by:
- Eliminating idle time through dynamic scheduling
- Exploiting very fine granularity (tasks on $b \times b$ blocks where $b \ll n$)
- Overlapping panel factorization with trailing matrix updates

## Applicability

- **Multicore CPUs**: Designed for thread-level parallelism with shared memory
- **QR-based algorithms**: GMRES, QMR, CG, eigenvalue solvers, least squares
- **Architectures with memory hierarchy**: Benefits from Level 3 BLAS and cache locality
- **Any matrix size**: Algorithm scales with problem size and core count

**Particularly effective when:**
- Number of cores is high (4-16+)
- Matrix dimensions allow many $b \times b$ blocks
- Memory bandwidth is limited (common on modern CPUs)

## Limitations

- **25% computational overhead**: More FLOPs than LAPACK due to redundant computations
- **Block size tuning**: Requires choosing $b$ (typically 32-200) for architecture
- **Shared memory required**: Not directly applicable to distributed memory
- **Implementation complexity**: Requires DAG scheduler and task management
- **Small matrices**: Overhead dominates when matrix fits in cache

## Implementation Notes

```python
# Pseudocode for tiled QR algorithm
for k in range(num_blocks):
    # Panel factorization (highest priority)
    DGEQT2(A[k,k], T[k,k])

    # Apply to trailing blocks (parallel)
    for j in range(k+1, num_blocks):
        DLARFB(A[k,j], V[k,k], T[k,k])

    # Update below diagonal (parallel)
    for i in range(k+1, num_blocks):
        DTSQT2(R[k,k], A[i,k], T[i,k])

        # Update trailing submatrix (parallel)
        for j in range(k+1, num_blocks):
            DSSRFB(A[k,j], A[i,j], V[i,k], T[i,k])
```

**Key insights:**
- Use Block Data Layout (BDL) for better cache performance
- Priority-based dynamic scheduling: DGEQT2 > DTSQT2 > DLARFB > DSSRFB
- All operations use BLAS subroutines for portability
- Compact WY representation ($T$ matrices) enables efficient BLAS3 operations

**Performance:**
- Speedup of 1.97-3.97x on 2-4 cores (8-way dual Opteron)
- Performance scales with both problem size and core count
- Can outperform fork-join parallelism when granularity is high

## References

- Buttari et al., "Parallel Tiled QR Factorization for Multicore Architectures", LAPACK Working Note #190
- Schreiber and Van Loan, "A Storage-Efficient WY Representation for Products of Householder Transformations"

# 160: Detached Band Reduction with WY Representation for Tensor Cores

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhang, Shah, Ootomo, Yokota & Wu (PPoPP 2023); Wang, Shi, Duan, Wu, Guo & Zhang (arXiv 2410.02170, 2024)
**Paper**: [papers/wy-tensor-core-evd-zhang-ppopp2023.pdf]
**Documented**: 2026-02-15

## Description

In two-stage tridiagonalization (the dominant cost of symmetric eigenvalue decomposition), the first stage — **Successive Band Reduction (SBR)** — reduces a symmetric matrix to band form using blocked Householder reflections. The conventional approach accumulates reflections using the **ZY representation**, where the trailing matrix update takes the form $A \leftarrow A - ZY^\top - YZ^\top$. This produces **outer-product-shaped GEMMs** (tall-skinny $\times$ short-wide) that are unfavorable for tensor cores because they lack sufficient data reuse and parallelism.

The key trick is to switch to the **WY representation**, where the accumulated transformation becomes $(I - YW^\top) A (I - WY^\top)$. Under this formulation, the dominant GEMMs become **square-matrix $\times$ tall-skinny-matrix** multiplications ($A \cdot W$), which are much more favorable for tensor cores. The WY representation "reshapes" the computation into forms that tensor cores can execute efficiently.

The follow-up innovation — **Detached Band Reduction (DBR)** — decouples the bandwidth $b$ from the blocksize $nb$, allowing $nb \gg b$. This enables much larger GEMMs in the trailing matrix update ($\text{syr2k}$ with $k = nb$) while keeping the bandwidth $b$ small for fast bulge chasing. The recursive $\text{syr2k}$ further converts tall-skinny GEMMs into more square shapes.

**Why this matters for neural networks:** Any NN layer or training procedure requiring eigenvalue decomposition (spectral normalization, orthogonal weight parameterization, PCA whitening, neural ODEs with Jacobian eigenvalues) can benefit from 10× faster GPU tridiagonalization. The GEMM-reshaping insight also applies broadly: whenever a linear algebra routine produces unfavorably-shaped GEMMs, restructuring the accumulation representation can unlock tensor core throughput.

## Mathematical Form

**Conventional SBR with ZY representation:**

The trailing matrix update after factoring a panel of $b$ columns uses:

$$
A \leftarrow A - ZY^\top - YZ^\top
$$

where $Y \in \mathbb{R}^{n \times b}$ contains Householder vectors and $Z \in \mathbb{R}^{n \times b}$ is accumulated as:

$$
Z = A Y - \frac{1}{2} Y (Y^\top A Y)
$$

The $\text{syr2k}$ operation $A - ZY^\top - YZ^\top$ involves GEMMs of shape $(n \times b) \times (b \times n)$ — an outer product. When $b$ is small (typically 64–256), these are **bandwidth-bound** on tensor cores.

**WY representation alternative:**

$$
Q = I - Y W^\top
$$

The two-sided update becomes:

$$
A \leftarrow Q^\top A Q = (I - WY^\top) A (I - YW^\top)
$$

The dominant operations are now:

$$
AW: \quad (n \times n) \times (n \times b) \to (n \times b) \qquad \text{[square × tall-skinny, tensor-core friendly]}
$$

$$
AY: \quad (n \times n) \times (n \times b) \to (n \times b) \qquad \text{[same shape]}
$$

These are **symv-like** operations with high arithmetic intensity when $n$ is large.

**Detached Band Reduction (DBR):**

The key innovation is decoupling bandwidth $b$ from blocksize $nb$:

$$
b \leq nb, \quad \text{typically } b = 32, \; nb = 1024 \text{ or } 2048
$$

**Algorithm (DBR):**

Given $A \in \mathbb{R}^{n \times n}$ symmetric, bandwidth $b$, blocksize $nb$:

```
for i = 1 : nb : n do
    for j = i : b : nb do                    // Inner loop: accumulate reflections
        [W, Y, R] = QR(A_panel)              // Panel QR factorization
        if j + b < nb then
            Z = AW - ½ Y W^T A W             // Panel update only (small)
            A ← A - ZY^T - YZ^T              // Update needed panel
        end if
    end for
    Accumulate Y and Z matrices over all inner iterations
    A ← A - ZY^T - YZ^T                      // Trailing matrix update with k = nb
end for
```

The trailing matrix update now uses $\text{syr2k}$ with $k = nb$ (e.g., 1024–2048) instead of $k = b$ (e.g., 32–64), producing GEMMs large enough to saturate tensor cores.

**Recursive $\text{syr2k}$ decomposition (Eq. 1 in Wang et al.):**

$$
\begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix} = \begin{pmatrix} A_{11} \\ A_{21} \end{pmatrix} \begin{pmatrix} B_{11}^\top & B_{21}^\top \end{pmatrix} + \begin{pmatrix} B_{11} \\ B_{21} \end{pmatrix} \begin{pmatrix} A_{11}^\top & A_{21}^\top \end{pmatrix}
$$

This decomposes into:
- $C_{11} = A_{11} B_{11}^\top + B_{11} A_{11}^\top$ (sub-$\text{syr2k}$, square)
- $C_{21} = A_{21} B_{11}^\top + B_{21} A_{11}^\top$ (GEMM, more square)
- $C_{22} = A_{21} B_{21}^\top + B_{21} A_{21}^\top$ (sub-$\text{syr2k}$, square)

Applied recursively, this converts all tall-skinny GEMMs into progressively more square shapes.

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — symmetric input matrix
- $b$ — bandwidth (small, e.g., 16–64, controls bulge chasing cost)
- $nb$ — blocksize (large, e.g., 512–2048, controls GEMM size)
- $Y \in \mathbb{R}^{n \times nb}$ — accumulated Householder vectors
- $W \in \mathbb{R}^{n \times nb}$ — WY representation factor
- $Z \in \mathbb{R}^{n \times nb}$ — ZY representation factor

## Complexity

| Operation | Conventional SBR ($b = nb$) | DBR ($b \ll nb$) |
|-----------|---------------------------|-------------------|
| Panel QR | $O(nb^2)$ per panel | $O(nb^2)$ per panel (same) |
| Trailing update GEMM shape | $(n \times b) \times (b \times n)$, $k = b$ | $(n \times nb) \times (nb \times n)$, $k = nb$ |
| $\text{syr2k}$ tensor core utilization | < 30% peak (H100) | **29–46% peak** (H100) |
| Bulge chasing | $O(n b^2)$ | $O(n b^2)$ (same $b$, fast) |
| SBR speedup vs cuSOLVER | 1× baseline | **up to 10.1×** (H100) |
| End-to-end EVD speedup | 1× baseline | **up to 4.1×** (H100) |

**Performance (TFLOPs on H100, $n = 65536$):**

| Method | SBR TFLOPs | % of 67 TFLOP peak |
|--------|------------|---------------------|
| cuSOLVER | 1.8 | 2.8% |
| MAGMA | 4.7% | ~3.1 |
| DBR ($b = 32, nb = 2048$) | **19.6** | **29%** |

**Memory:** $O(n \cdot nb)$ for the accumulated $Y, W/Z$ matrices. Since $nb \leq n$, this is at most $O(n^2)$ — no additional asymptotic memory cost.

## Applicability

- **Spectral normalization in GANs/diffusion models:** Requires computing the largest singular value of weight matrices via power iteration or full SVD. Faster EVD/SVD directly accelerates training.

- **Orthogonal weight parameterization:** Models that constrain weights to the orthogonal group (orthogonal RNNs, Cayley parameterizations) sometimes use EVD for the orthogonal projection step. 10× faster tridiagonalization makes this practical at larger scales.

- **PCA whitening layers:** Layers that whiten activations using the covariance eigendecomposition (e.g., Decorrelated Batch Normalization) benefit directly.

- **Neural ODE Jacobian analysis:** Analyzing stability of neural ODEs requires eigenvalues of the Jacobian. Faster EVD enables larger-scale stability monitoring during training.

- **General GEMM shape optimization insight:** The principle of switching between equivalent algebraic representations (WY vs ZY) to improve GEMM shapes for tensor cores is broadly applicable. Any blocked algorithm producing outer-product-shaped GEMMs should consider reformulation.

- **Recursive $\text{syr2k}$ for any symmetric rank-$k$ update:** The recursive decomposition into more-square sub-GEMMs applies to any $\text{syr2k}$ call in neural network training (e.g., Hessian approximations, covariance updates).

## Limitations

- **Not directly applicable to non-symmetric matrices:** The band reduction and $\text{syr2k}$ formulation are specific to symmetric EVD. For general SVD, the bidiagonal reduction has a similar but distinct structure.

- **Tuning required:** Optimal $b$ and $nb$ depend on GPU architecture. On H100, $nb = 2048$ works well; on A100, $nb = 512$ is better. The bandwidth $b$ must balance SBR performance against bulge chasing cost.

- **WY representation has higher FLOP count:** Compared to ZY, the WY form requires additional computation to build $W$. The speedup comes entirely from better tensor core utilization (higher FLOP rate), not fewer FLOPs. On hardware without tensor cores, WY may be slower.

- **Large matrix sizes needed:** The tensor core advantage is most pronounced for $n \geq 8192$. For small matrices ($n < 4096$), the overhead of the blocked approach and the relatively small GEMMs limit the benefit.

- **Bulge chasing still memory-bound:** While DBR dramatically speeds up SBR, the bulge chasing stage remains memory-bound (though the GPU-parallel implementation with inter-sweep pipelining achieves 8× speedup over CPU-based MAGMA). At $n = 65536$ with $b = 128$, bulge chasing consumes ~25% of total tridiagonalization time.

## Implementation Notes

```python
import torch

def detached_band_reduction_sketch(A, b, nb):
    """
    Sketch of Detached Band Reduction (DBR).
    Decouples bandwidth b from blocksize nb for better tensor core utilization.

    A: (n, n) symmetric matrix
    b: bandwidth (small, e.g., 32)
    nb: blocksize (large, e.g., 1024-2048)
    """
    n = A.shape[0]
    assert nb >= b and nb % b == 0

    for i in range(0, n, nb):
        # Accumulate Y and Z across inner iterations
        Y_accum = []
        Z_accum = []

        for j in range(i, min(i + nb, n), b):
            # Panel QR factorization (tall-skinny QR, e.g., TSQR)
            panel = A[j:, j:j+b]
            W, Y, R = torch.linalg.qr(panel, mode='reduced')  # simplified

            if j + b < i + nb:
                # Panel update only (small, within the block)
                # Uses standard syr2k with small k = b
                Z = A[j:, j:] @ W - 0.5 * Y @ (W.T @ A[j:, j:] @ W)
                # Update panel region only
                A[j:, j:j+nb] -= Z @ Y[:nb-j+i].T + Y[:, :b] @ Z[:nb-j+i].T

            Y_accum.append(Y)
            Z_accum.append(Z)

        # === KEY: Trailing matrix update with k = nb ===
        # This syr2k now has k = nb (e.g., 1024), not k = b (e.g., 32)
        # Shape: (n × nb) × (nb × n) — large enough for tensor cores
        Y_big = torch.cat(Y_accum, dim=1)  # (n-i, nb)
        Z_big = torch.cat(Z_accum, dim=1)  # (n-i, nb)

        trailing = A[i+nb:, i+nb:]
        # syr2k: trailing -= Z_big @ Y_big.T + Y_big @ Z_big.T
        # Use recursive syr2k for even better GEMM shapes
        recursive_syr2k(trailing, Z_big[nb:], Y_big[nb:])

    return A


def recursive_syr2k(C, A_mat, B_mat):
    """
    Recursive symmetric rank-k update: C -= A B^T + B A^T

    Recursively decomposes into sub-syr2k and GEMM operations
    that are progressively more square-shaped.

    This is Algorithm 3 from Wang et al. (2024).
    """
    n, k = A_mat.shape
    if n <= k or n <= 256:  # Base case: use cuBLAS syr2k
        C -= A_mat @ B_mat.T + B_mat @ A_mat.T
        return

    mid = n // 2
    # C11 -= A1 B1^T + B1 A1^T  (sub-syr2k, more square)
    recursive_syr2k(C[:mid, :mid], A_mat[:mid], B_mat[:mid])

    # C21 -= A2 B1^T + B2 A1^T  (GEMM, square-ish)
    C[mid:, :mid] -= A_mat[mid:] @ B_mat[:mid].T + B_mat[mid:] @ A_mat[:mid].T

    # C22 -= A2 B2^T + B2 A2^T  (sub-syr2k, more square)
    recursive_syr2k(C[mid:, mid:], A_mat[mid:], B_mat[mid:])
```

**Key GPU optimization insights from the paper:**

1. **GEMM shape matters more than FLOP count on tensor cores.** The WY representation has ~17% more FLOPs than ZY, but achieves 3.7× speedup because the GEMM shapes (square × tall-skinny) have 10–16× higher throughput on tensor cores than the outer-product shapes of ZY.

2. **Decouple bandwidth from blocksize.** In conventional SBR, $b = nb$ forces a tradeoff: large $b$ helps SBR but hurts bulge chasing. DBR breaks this constraint, allowing $nb = 2048$ for large GEMMs while keeping $b = 32$ for fast bulge chasing.

3. **Recursive decomposition of $\text{syr2k}$** converts tall-skinny GEMMs into more square ones. For $b = 32, nb = 256$: non-recursive produces 7 GEMMs with $k = 32$; recursive produces 4 GEMMs with $k = 32$, 2 with $k = 64$, and 1 with $k = 128$.

4. **Roofline analysis (H100):** Peak FP64 = 67 TFLOPs, bandwidth = 3430 GB/s. The operational intensity crossover is at ~20 FLOP/byte. $\text{syr2k}$ with $k = 32$ achieves only 0.09 TFLOPs ($n = 4096$); with $k = 2048$ it achieves 34.6 TFLOPs (Table 1 in paper).

5. **GPU-parallel bulge chasing with pipelining:** Different sweeps of bulge chasing lack data dependencies and can execute in parallel via inter-kernel pipelining. Synchronization between adjacent sweeps uses lock flags in global memory. This achieves 8× speedup over CPU-based MAGMA.

## References

- Zhang, S., Shah, R., Ootomo, H., Yokota, R., & Wu, P. (2023). Fast Symmetric Eigenvalue Decomposition via WY Representation on Tensor Core. *PPoPP '23*, 301–312.
- Wang, H., Shi, L., Duan, Z., Wu, P., Guo, L., & Zhang, S. (2024). Extracting the Potential of Emerging Hardware Accelerators for Symmetric Eigenvalue Decomposition. arXiv:2410.02170.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. *SIAM J. Sci. Stat. Comput.*, 10(1), 53–57.
- Ootomo, H., Ozaki, K., & Yokota, R. (2024). DGEMM on Integer Matrix Multiplication Unit. *Intl. J. High Performance Computing Applications*.

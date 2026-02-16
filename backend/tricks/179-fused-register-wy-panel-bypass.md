# 179: Fused Register-Resident WY Panel Bypass

**Category**: kernel
**Gain type**: efficiency
**Source**: Abdelfattah, Tomov & Dongarra (ICCS 2022); MAGMA library
**Paper**: [papers/batched-qr-fused-wy-gpu.pdf]
**Documented**: 2026-02-15

## Description

When computing batched QR factorizations on GPUs (common in orthogonal weight construction, gradient orthogonalization, and low-rank approximations for neural networks), the standard LAPACK approach first factorizes a panel, then **explicitly constructs the $T$ factor** of the compact WY representation $Q = I - VTV^\top$, and finally applies the block reflector to the trailing matrix via BLAS-3 GEMMs. For small matrices ($n \leq 32$), the T-factor construction (`dlarft`) and associated auxiliary kernels (`dlacpy`, `dlaset`) consume **30–45% of total runtime** — dominating the actual computation.

The fused register-resident panel bypass eliminates the T-factor construction entirely by **keeping the factorized panel in GPU registers or shared memory** and applying individual Householder reflectors directly to the trailing matrix within a single fused kernel. Each reflector $(I - \tau_j v_j v_j^\top)$ is applied as a rank-1 update: $C \leftarrow C - \tau_j v_j (v_j^\top C)$, where both $v_j$ and $C$ are cached in registers. This avoids:

1. **Materializing $T$:** No upper-triangular $T$ matrix is ever formed
2. **Multiple kernel launches:** Panel factorization + T-factor + trailing update → one fused kernel
3. **HBM round-trips:** The matrix is read once from global memory, factorized in registers/shared memory, and written back once

The result is up to **3.22× speedup** over cuBLAS for tiny matrices ($n \leq 32$), and **12.9–57.1%** improvement over the standard MAGMA LAPACK design for square matrices, all on A100 GPUs.

**Why this matters for neural networks:** Small batched QR operations arise in:
- **Orthogonal RNN weight construction:** Periodically re-orthogonalizing small weight blocks
- **Low-rank adapter orthogonalization:** HOFT, OFT, and similar methods use QR on $r \times r$ blocks with $r \leq 128$
- **Per-head orthogonal projections:** Multi-head attention with orthogonal constraints on $d_h \times d_h$ matrices ($d_h = 64$ or $128$)
- **DeltaNet UT transform intra-chunk:** The forward substitution on $C \times C$ lower-triangular matrices ($C = 64$–$128$) is structurally similar to applying individual reflectors from registers

## Mathematical Form

**Standard LAPACK Blocked QR (for comparison):**

Given $A \in \mathbb{R}^{m \times n}$, the blocked approach proceeds in panels of width $nb$:

1. **Panel factorization** (`dgeqr2`): Factor $A_{:, j:j+nb}$ column-by-column using Householder reflections, producing vectors $v_1, \ldots, v_{nb}$ and scalars $\tau_1, \ldots, \tau_{nb}$.

2. **T-factor construction** (`dlarft`): Build the upper-triangular $T \in \mathbb{R}^{nb \times nb}$ such that:

$$
Q = I - V T V^\top, \quad V = [v_1, \ldots, v_{nb}]
$$

Classical `dlarft` (Algorithm 1 in paper): For each $j = 1, \ldots, nb$:

$$
T_{1:j-1, j} = -\tau_j T_{1:j-1, 1:j-1} \cdot V_{:, 1:j-1}^\top \cdot v_j \quad \text{(DGEMV + DTRMV)}
$$

$$
T_{j,j} = \tau_j
$$

This is **BLAS-2** (matrix-vector products), $O(m \cdot nb)$ per column, $O(m \cdot nb^2)$ total.

Improved `dlarft` (Algorithm 2, aggregating GEMVs into one GEMM):

$$
T_{1:nb, 1:n} = V_{1:nb, 1:n}^\top \times V_{1:n, 1:n} \quad \text{(one DGEMM)}
$$

then apply triangular multiplies. This is **BLAS-3** but requires auxiliary workspace and multiple kernel launches.

3. **Trailing matrix update**: Apply $Q^\top$ to $A_{:, j+nb:n}$ using DGEMM.

**Fused Register-Resident Approach (the trick):**

Skip steps 2 and 3 entirely. Instead, within a single GPU kernel:

1. Load the full matrix $A$ into **GPU registers** (one row per thread, compile-time width $nb$).

2. For each column $j = 1, \ldots, n$:
   - Compute the Householder vector $v_j$ and scalar $\tau_j$ via `dlarfg` (norm + sign + scaling — uses shared memory for tree reduction across threads).
   - **Immediately** apply the reflector to all remaining columns $j+1, \ldots, n$:

$$
A_{:, j+1:n} \leftarrow A_{:, j+1:n} - \tau_j v_j (v_j^\top A_{:, j+1:n})
$$

Since $v_j$ is in registers and $A$ is in registers, the inner product $v_j^\top A_{:, k}$ is a **reduction across threads** (using shared memory), and the rank-1 update is a **register-local multiply-add**.

3. Write the factorized matrix back to global memory **once**.

**The key insight:** For small matrices ($m \leq 1024$, $n \leq 32$), the entire matrix fits in registers across threads of a thread block. No intermediate results ever touch global memory or even shared memory (except for the cross-thread reductions). The T-factor is **never constructed** because reflectors are applied one-at-a-time from registers — achieving the same result as the blocked update but without the materialization overhead.

**For larger panels ($nb > 32$):** A shared-memory variant caches the panel in shared memory (256 KB on H100) instead of registers, relaxing the thread-count constraint. The trailing matrix update loops over sub-blocks of width $\bar{n}$ from global memory, applying reflectors from shared memory via `dlarf` device calls.

**Fused Trailing Update (Algorithm 3 in paper):**

```
pA[] ← read factorized panel into shared memory
pA[] ← dlaset(pA[], 'upper', 0, 'diag', 1)  // zero upper tri, ones on diag
for i = 1 to n̄ step ib do
    tA[] ← read next block of columns from trailing panel
    for j = 1 to ib do
        tA ← dlarf(pA(:,j), tA[])  // apply single reflector from shared mem
    end for
    write tA[] back to memory
end for
```

This applies reflectors **without forming $T$**, trading the BLAS-3 DGEMM for many BLAS-2-like `dlarf` calls. The tradeoff is favorable when:
- The matrix is small enough that `dlarf` runs at near-peak bandwidth from shared memory
- The T-factor overhead (auxiliary kernel launches, workspace allocation) dominates

**Key Definitions:**

- $m$ — number of rows (panel height)
- $n$ — number of columns (panel width / matrix size for square)
- $nb$ — blocking width (equals $n$ for the fully fused case)
- $\tau_j$ — Householder scalar for column $j$
- $v_j$ — Householder vector for column $j$ (unit lower triangular)
- $T \in \mathbb{R}^{nb \times nb}$ — the WY T-factor (upper triangular) — **not constructed** in this approach

## Complexity

| Operation | LAPACK Blocked QR | Fused Register-Resident |
|-----------|-------------------|----------------------|
| Panel factorization | `dgeqr2`: $O(mn^2)$, BLAS-2 | Same FLOPs, but in registers |
| T-factor construction | `dlarft`: $O(mn^2)$, BLAS-2/3 | **Eliminated** ($0$ FLOPs) |
| Trailing update | `dgemm`: $O(mn^2)$, BLAS-3 | `dlarf` × $n$: $O(mn^2)$, register-local |
| Auxiliary kernels | `dlacpy`+`dlaset`: $O(mn)$ | **Eliminated** |
| Kernel launches | 5–8 per panel | **1** (fully fused) |
| Global memory reads | $3mn$ words (panel + workspace + trailing) | $mn$ words (**1 read**) |
| Global memory writes | $3mn$ words | $mn$ words (**1 write**) |

**Memory traffic reduction:** The fused approach reduces global memory traffic by approximately **3×** for the panel factorization phase. Since small-matrix QR is memory-bandwidth-bound (not compute-bound), this directly translates to wall-clock speedup.

**Performance results (A100 GPU, double precision, batch size 1000):**

| Matrix Size | cuBLAS | KBLAS | MAGMA (semi-fused) | MAGMA (fully-fused) | Speedup |
|-------------|--------|-------|-------------------|--------------------|---------|
| $2 \times 2$ | ~10 GF/s | ~15 GF/s | ~20 GF/s | ~20 GF/s | 1.07× |
| $4 \times 4$ | ~40 GF/s | ~50 GF/s | ~70 GF/s | ~80 GF/s | 1.56× |
| $8 \times 8$ | ~80 GF/s | ~100 GF/s | ~150 GF/s | ~200 GF/s | 2.10× |
| $16 \times 16$ | ~100 GF/s | ~140 GF/s | ~280 GF/s | ~350 GF/s | 3.22× |
| $32 \times 32$ | ~150 GF/s | ~200 GF/s | ~350 GF/s | ~350 GF/s | 2.15× |

The **3.22× speedup at $16 \times 16$** is the sweet spot where the register-resident approach maximally benefits from eliminating T-factor overhead.

**Time breakdown (A100, standard MAGMA vs. fused):**

| Component | $(128,16)$ std | $(128,16)$ fused | Reduction |
|-----------|---------------|-----------------|-----------|
| `dgeqr2` kernels | 64.4% | 24.4% | 2.6× |
| `dgemm` | 12.5% | 26.4% | (larger fraction of reduced total) |
| Auxiliary (`dlacpy`, etc.) | 21.3% | 45.3% | Now the bottleneck |
| `trmv` (for `dlarft`) | 1.9% | 4.0% | |

After fusing the panel kernel, the auxiliary kernels become the new bottleneck — motivating the fully-fused approach that eliminates them too.

## Applicability

- **Orthogonal fine-tuning (OFT/HOFT/BOFT):** These methods maintain orthogonal adapter matrices via products of Householder reflections. The CWY representation requires computing $S^{-1}$ (or the T-factor). For small block sizes ($r = 4$–$64$), the fused approach eliminates the T-factor entirely, applying reflectors directly from registers. This directly accelerates the HOFT method (trick 157).

- **Per-head orthogonal projections:** In multi-head attention with orthogonal constraints (e.g., Parseval networks), each head has a $d_h \times d_h$ orthogonal matrix ($d_h = 64$ or $128$). Batched QR across all heads and all layers is naturally a "batch of small matrices" workload — the fused approach's sweet spot.

- **DeltaNet/DeltaProduct chunkwise UT transform:** The UT transform's forward substitution on $C \times C$ matrices ($C = 64$–$128$) is structurally similar to applying individual reflectors. A fused kernel that keeps the UT matrix in registers and applies it without materializing intermediate results would follow the same design pattern.

- **Gradient orthogonalization:** Methods like SOAP that periodically QR-factorize gradient accumulation matrices can use batched fused QR when the gradients are partitioned into small blocks.

- **Low-rank SVD via QR:** Computing truncated SVD of activation matrices (for LoRA, PCA whitening) often starts with a QR step on small matrices. The fused approach accelerates this.

## Limitations

- **Size constraints:** The fully-fused register-resident approach is limited to $n \leq nb_{\max}$, where $nb_{\max}$ depends on the register file size and compute precision. On A100: $nb_{\max} \approx 32$ for FP64, $\approx 64$ for FP32, $\approx 128$ for FP16. Matrices larger than this must use the shared-memory variant or fall back to the LAPACK blocked design.

- **Panel height constraint:** The register-resident kernel uses one thread per row, so panels taller than the maximum thread block size (1024 threads) require the shared-memory variant, which has lower register-level data reuse.

- **Not BLAS-3:** The fused approach applies individual reflectors (BLAS-2-like), not block reflectors (BLAS-3). For large matrices where BLAS-3 GEMMs achieve high tensor-core utilization, the standard blocked approach with T-factor is faster. The crossover point is architecture-dependent: approximately $n = 192$ on A100.

- **Compile-time specialization:** The panel width $nb$ must be known at compile time for register allocation. This requires instantiating the kernel template for each possible width, increasing binary size. In practice, widths 1–32 (in steps of 1) are pre-compiled.

- **No tensor core usage:** The fused approach uses CUDA cores (scalar FMA) rather than tensor cores, since the operations are rank-1 updates, not matrix-matrix multiplications. For FP16/BF16 workloads where tensor cores provide 16× throughput advantage, the fused approach may lose to the T-factor-based blocked approach even at small sizes.

## Implementation Notes

```python
# Pseudocode for the fused register-resident QR kernel
# In practice, this is implemented in CUDA C++ with compile-time templates

def fused_register_qr_kernel(A_batch, m, n):
    """
    Fused QR factorization for a batch of small m × n matrices.
    Each matrix is processed by one thread block.
    Each thread owns one row of the matrix (stored in registers).

    Args:
        A_batch: (batch, m, n) - batch of matrices in global memory
        m: number of rows (m <= 1024)
        n: number of columns (n <= 32, compile-time constant)
    """
    # Thread ID determines which row this thread owns
    tid = threadIdx.x  # 0 <= tid < m
    bid = blockIdx.x   # which matrix in the batch

    # Step 1: Load one row of A into n registers
    # reg[0..n-1] = A_batch[bid, tid, 0..n-1]
    reg = load_row(A_batch, bid, tid)  # n registers per thread

    # Step 2: Column-by-column Householder QR
    for j in range(n):
        # 2a: Compute norm of column j below diagonal
        # Each thread contributes reg[j]^2 if tid >= j
        local_sq = reg[j] ** 2 if tid >= j else 0.0

        # Tree reduction in shared memory to get column norm
        col_norm_sq = shared_memory_reduce_sum(local_sq)
        col_norm = sqrt(col_norm_sq)

        # 2b: Compute Householder vector v_j and scalar tau_j
        # Only thread tid == j modifies the pivot element
        if tid == j:
            alpha = -sign(reg[j]) * col_norm
            v_j_pivot = reg[j] - alpha
            tau_j = (alpha - reg[j]) / alpha
            reg[j] = alpha  # Store R[j,j]
            # Normalize: v[j] = 1 (implicit), v[i>j] = reg[i] / v_j_pivot
        else:
            # Threads with tid > j: their entry in v_j is reg[j] / v_j_pivot
            pass

        # Broadcast tau_j and v_j_pivot via shared memory
        tau_j = shared_broadcast(tau_j, source=j)
        v_j_pivot = shared_broadcast(v_j_pivot, source=j)

        # Normalize v entries: v[tid] = reg[j] / v_j_pivot for tid > j
        v_entry = reg[j] / v_j_pivot if tid > j else (1.0 if tid == j else 0.0)

        # 2c: Apply reflector to remaining columns k = j+1, ..., n-1
        # For each column k: compute dot = v^T * A[:,k], then A[:,k] -= tau * v * dot
        for k in range(j + 1, n):
            # Inner product v^T * A[:,k] via reduction
            local_prod = v_entry * reg[k] if tid >= j else 0.0
            dot = shared_memory_reduce_sum(local_prod)

            # Rank-1 update: A[tid, k] -= tau * v[tid] * dot
            if tid >= j:
                reg[k] -= tau_j * v_entry * dot

        # Store v_j in the lower triangular part of reg[j]
        if tid > j:
            reg[j] = v_entry  # Overwrite A[tid, j] with v_j[tid]

    # Step 3: Write factorized matrix back to global memory (ONCE)
    store_row(A_batch, bid, tid, reg)


# For the fused trailing update variant (larger matrices):
def fused_trailing_update_kernel(V_panel, tau, trailing_A, m, nb, n_trail):
    """
    Apply Householder reflectors from shared memory to trailing matrix,
    WITHOUT constructing the T factor.

    V_panel: (m, nb) - Householder vectors (in shared memory)
    tau: (nb,) - Householder scalars (in shared memory)
    trailing_A: (m, n_trail) - trailing panel (loaded in blocks from global mem)

    Each reflector is applied individually via dlarf.
    """
    # Load V_panel into shared memory
    smem_V = load_to_shared_memory(V_panel)
    smem_tau = load_to_shared_memory(tau)

    # Process trailing matrix in blocks of width ib
    ib = 16  # sub-block width, tunable
    for i in range(0, n_trail, ib):
        # Load sub-block into registers
        tA = load_to_registers(trailing_A[:, i:i+ib])

        # Apply each reflector from shared memory
        for j in range(nb):
            v_j = smem_V[:, j]
            tau_j = smem_tau[j]
            # dlarf: tA -= tau_j * v_j * (v_j^T * tA)
            for k in range(ib):
                dot = reduce_sum(v_j * tA[:, k])  # via shared memory
                tA[:, k] -= tau_j * v_j * dot

        # Write sub-block back to global memory
        store_from_registers(trailing_A[:, i:i+ib], tA)
```

**GPU efficiency analysis:**

1. **Memory traffic is the bottleneck, not FLOPs.** For $16 \times 16$ matrices in FP64, each matrix is 2 KB. The T-factor is 2 KB more. Auxiliary copies (`dlacpy`, `dlaset`) add another 4 KB. The fused approach eliminates 6 KB of unnecessary traffic per matrix — a 4× reduction. At batch size 10K, this saves 60 MB of HBM bandwidth.

2. **Kernel launch overhead matters at small sizes.** The LAPACK design launches 5–8 kernels per panel iteration. At batch size 10K with $n = 16$, each kernel launch takes ~5 µs, totaling ~40 µs of pure launch overhead vs. ~100 µs of actual computation. The fused approach uses **1 kernel launch**.

3. **Register pressure is manageable.** Each thread stores $n$ FP64 values. At $n = 32$, this is 256 bytes = 64 registers per thread. With 256 threads per block (for $m = 256$), total register usage is 16K registers per block — well within the A100's 65K register limit per SM.

4. **Shared memory for reductions.** Cross-thread reductions for column norms and inner products use $m$ words of shared memory. At $m = 256$, $n = 32$, FP64: 2 KB for reduction workspace. The shared memory variant adds $m \times nb = 64$ KB for the panel — fits within A100's 192 KB shared memory.

5. **Arithmetic intensity.** The fused approach has arithmetic intensity ~$n/2$ FLOP/byte (column norms + reflector applications over $n$ columns, reading/writing each matrix element twice). At $n = 16$: AI = 8 FLOP/byte, which is near the roofline crossover for A100 FP64 (AI* ≈ 20). This confirms the operation is bandwidth-bound, making memory traffic reduction the primary optimization lever.

## References

- Abdelfattah, A., Tomov, S., & Dongarra, J. (2022). Batch QR Factorization on GPUs: Design, Optimization, and Tuning. *ICCS 2022*, LNCS 13350, pp. 60–74. Springer.
- Haidar, A., Dong, T. T., Luszczek, P., Tomov, S., & Dongarra, J. (2015). Batched Matrix Computations on Hardware Accelerators Based on GPUs. *IJHPCA*, 29(2), 193–208.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. *SIAM J. Sci. Stat. Comput.*, 10(1), 53–57.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Dongarra, J. & MAGMA team. MAGMA: Matrix Algebra on GPU and Multicore Architectures. https://icl.utk.edu/magma/

# 172: Segmented Matmul Scan (SCD / SSCR)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Sobczyk, Sorrentino & Zouzias (arXiv:2506.23906, 2025)
**Paper**: [papers/segmented-matmul-scan-scd.pdf]
**Documented**: 2026-02-15

## Description

Segmented scan — the operation that simultaneously performs independent prefix scans over arbitrary contiguous segments of an array, restarting accumulation at each segment boundary — is a fundamental primitive for batched variable-length sequence processing in SSMs, linear attention, and sparse matrix-vector multiplication. The classical approach (trick 107) uses flag-value pair operator transformation and maps to scalar/vector GPU units. However, on modern AI accelerators with Matrix Multiplication Units (MMUs — NVIDIA tensor cores, Google TPU MXUs, Huawei Ascend cube units), the classical approach leaves the most powerful compute unit entirely idle.

This trick introduces two algorithms — **SCD** (Scan, Compress, Differentiation) for segmented *sum* and **SSCR** (Scan, Scan, Compress, Revert) for segmented *scan* — that exploit MMUs by performing **speculative unsegmented scans** via matrix multiplication, then correcting the speculation using lightweight vector operations. The key insight is that an unsegmented scan can be computed efficiently on the MMU (using MatMulScan, trick 167), and the segmented version can be recovered from it by a sequence of compress (gather segment endpoints) and differentiation/revert (vector correction) steps.

The authors propose the **MMV-RAM** computational model that formalizes the dual-unit architecture (MMU + VCU) of modern AI accelerators, prove that MMU+VCU algorithms achieve provable theoretical speedups over VCU-only algorithms for segmented operations, and validate with implementations on the Ascend 910B accelerator showing significant speedups over vector-only baselines.

## Mathematical Form

**Problem Definition (Segmented Scan):**

Given data vector $\mathbf{x}$ of length $n$ and boolean flag vector $\mathbf{f}$ (where $\mathbf{f}(i) = 1$ marks the start of a new segment):

$$
\mathbf{z}(i) = \begin{cases} \mathbf{x}(i) & \text{if } i = 0 \text{ or } \mathbf{f}(i) = 1 \\ \mathbf{z}(i-1) + \mathbf{x}(i) & \text{otherwise} \end{cases}
$$

**Segmented Sum** returns only the last element of each segment's scan. For example:

$$
\mathbf{x} = (2, 2, 3, 3, 1, 3, 1, 2), \quad \mathbf{f} = (1, 0, 1, 0, 0, 1, 0, 0)
$$
$$
\text{segscan}(\mathbf{x}, \mathbf{f}) = (2, 4, 3, 6, 7, 3, 4, 6), \quad \text{segsum}(\mathbf{x}, \mathbf{f}) = (4, 7, 6)
$$

**Algorithm 4.2: SCD (Segmented Sum) ≅ DIFF ∘ COMPRESS ∘ SCAN**

1. **SCAN** (speculative): Compute the *unsegmented* prefix scan of $\mathbf{x}$ using the MMU:
$$
\hat{\mathbf{x}} \leftarrow \text{SCAN}(\mathbf{x})
$$
This uses MatMulScan (trick 167): reshape into $s \times s$ blocks, multiply by $\mathbf{U}_s$ (upper-triangular all-ones matrix), recursively aggregate via Brent-Kung-style upsweep/downsweep. All heavy computation is matrix multiplication.

2. **COMPRESS**: Shift $\mathbf{f}$ left by one position and append 1, giving $\mathbf{f}^-$ which marks segment *endpoints*. Gather the scanned values at segment endpoints:
$$
\mathbf{z} \leftarrow \text{COMPRESS}(\hat{\mathbf{x}}, \mathbf{f}^-)
$$
This collects only the elements of $\hat{\mathbf{x}}$ where $\mathbf{f}^-$ is set to 1.

3. **DIFF** (vector differentiation): The speculative scan over-accumulated across segment boundaries. Correct by differencing:
$$
\mathbf{z}_{\text{diff}}(i) = \mathbf{z}(i) - \mathbf{z}(i-1)
$$
assuming $\mathbf{z}(-1) = 0$. This can be implemented via the differentiation matrix:

$$
\mathbf{D}_s := \begin{pmatrix} 1 & -1 & 0 & \cdots & 0 \\ 0 & 1 & -1 & \cdots & 0 \\ 0 & 0 & 1 & \ddots & 0 \\ \vdots & & & \ddots & -1 \\ 0 & 0 & 0 & \cdots & 1 \end{pmatrix}_{s \times s}
$$

The matrix product $\mathbf{A}\mathbf{D}_s$ computes differences on each $s$-block boundary.

**Algorithm 4.3: SSCR (Segmented Scan) ≅ REVERT ∘ COMPRESS ∘ (SCAN, SCAN)**

1. Compute both $\hat{\mathbf{x}} \leftarrow \text{SCAN}(\mathbf{x})$ and $\hat{\mathbf{f}} \leftarrow \text{SCAN}(\mathbf{f})$ using the MMU
2. Shift flags: $\mathbf{f}^- \leftarrow \mathbf{f}(1:) \text{ appended with } 1$
3. Compress: $\mathbf{w} \leftarrow \text{COMPRESS}(\hat{\mathbf{x}}, \mathbf{f}^-)$
4. Revert speculation: For each position $i$, subtract the last element of the previous segment:
$$
\mathbf{z}(i) = \hat{\mathbf{x}}(i) - (\hat{\mathbf{f}}(i) = \text{idx}+1) \cdot \mathbf{w}(\text{idx})
$$

**The Speculative Computation Insight:**

The unsegmented scan $\hat{\mathbf{x}}$ "over-accumulates" by summing across segment boundaries. But the error is structured: for any position $i$ in segment $k$, the error is exactly the cumulative sum up to the end of segment $k-1$. The COMPRESS step extracts these boundary sums, and DIFF/REVERT corrects them. This decomposition separates the heavy arithmetic (SCAN via MMU) from the lightweight correction (COMPRESS + DIFF via VCU).

**Block-Recursive Improvement (Section 4.2, Appendix A.2):**

The basic SCD/SSCR has $O(n^2)$ work due to the COMPRESS gather. This is improved to $O(n)$ work via a block-recursive approach:

1. Partition $\mathbf{x}$ and $\mathbf{f}$ into blocks of size $s$
2. Compute local scans within each block using MMU
3. Use VCU to revert mis-speculated values within each block
4. Collect block boundary values into reduced vector $\mathbf{x}_s$ of size $\lceil n/s \rceil$
5. Create block-level flags $\mathbf{f}_s$ via logical OR within each block
6. Recursively compute segmented scan on $(\mathbf{x}_s, \mathbf{f}_s)$
7. Propagate corrections from block boundaries back to within-block elements

## Complexity

**Theorem 4.2 (SCD & SSCR):** Both algorithms achieve:

| Metric | Value |
|--------|-------|
| Steps (depth) | $O(\log_s(n))$ |
| Work | $O\left(\mathcal{M}(\frac{n}{s}) + nB(s + \frac{B}{s})\right)$ |
| Bits per element | $B \in O(\log(nM))$ |

where $\mathcal{M}(n)$ is the cost of matrix multiplications on $n/s$ blocks, $s$ is the MMU tile size, and $B$ is the bits per element.

**Lower bound:** Any VCU-only algorithm that uses only vector operations and executes $n^{O(1)}$ work requires $\Omega\left(\frac{\log(n)}{\log(\log(n))}\right)$ steps — strictly more than the $O(\log_s(n))$ achieved with the MMU.

**Applications complexity (Table 4.1):**

| Problem | Steps (with MMU) | Steps (VCU only) |
|---------|-------------------|-------------------|
| Segmented sum/scan | $O(\log_s(n))$ | $\Omega\left(\frac{\log(n)}{\log(\log(n))}\right)$ |
| Element-wise vector product | $O(\log_s(B))$ | $\Omega\left(\frac{\log(B)}{\log(\log(B))}\right)$ |
| Matrix product ($n \times n$) | $O(\log_s(nB))$ | $\Omega\left(\frac{\log(n)}{\log(\log(n))}\right)$ |
| SpMV (CSR) | $O(\log_s(nB))$ | $\Omega\left(\frac{\log(n)}{\log(\log(n))}\right)$ |

**Memory:** $O(n)$ — the algorithm operates on the input arrays plus $O(n/s)$ for reduced boundary vectors.

**Experimental results (Ascend 910B, single AI-core, $s = 128$):**

- MMU+Vector implementation is significantly faster than Vector-only baseline across all tested sparse matrices (36K×36K to 5.6M×5.6M)
- Reaches and sometimes surpasses single-thread CPU performance
- For segmented scan with segment density ≈ 0.1%, achieves ~90% of unsegmented scan bandwidth
- Bandwidth degrades to ~50% for denser segments (0.3-0.5% density), remaining within ≥20% of unsegmented performance in all cases
- COMPRESS is the dominant bottleneck (~50% of SCD time, ~33% of SSCR time) due to irregular memory access patterns
- SCAN operator achieves close to peak memory bandwidth (800 GB/s on Ascend 910B)

**SpMV results (multi-core, sparse-attention matrices):**

- Up to 3.44× speedup vs. Intel MKL (32 threads)
- Up to 6.39× speedup vs. Eigen (32 threads)
- Best performance on block-sparse attention patterns (b-64_r-2): exactly the sparsity structure used in production sparse-attention models

## Applicability

- **Batched SSM training with variable-length sequences:** Pack multiple variable-length sequences into a single array with segment flags. Run one segmented scan (via SCD/SSCR) instead of padding to max length. The MMU acceleration means this now runs at near-unsegmented speed on tensor-core hardware, eliminating the main objection to packing-based approaches.

- **Sparse attention pattern computation (SpMV):** Sparse-attention matrices in CSR format can be multiplied by dense vectors using Algorithm 4.4 (GATHER → MULT → SCD). The segmented sum collects per-row reductions. This achieves 3-6× speedup over CPU baselines on block-sparse attention patterns.

- **Linear attention with variable-length batches:** The KV state accumulation $S_t = \lambda_t S_{t-1} + v_t k_t^\top$ across packed variable-length sequences uses segmented scan as the core primitive.

- **Integer/element-wise vector multiplication on MMUs:** Using segmented scan as a subroutine, the paper shows how to compute element-wise products of integer vectors — relevant for quantized operations on integer-only NPUs.

- **Dense matrix multiplication via segmented operations:** The paper proves matrix product can be computed in $O(\log_s(nB))$ MMV-RAM steps, providing theoretical speedup bounds for reduction-heavy matrix algorithms.

## Limitations

- **COMPRESS is the bottleneck:** The gather/compress operation requires irregular memory access patterns (reading at positions marked by flags), which achieves poor memory bandwidth on all tested hardware. This is inherent to segmented operations and cannot be eliminated by the MMU.

- **Segment density sensitivity:** Performance degrades for denser segments (many short segments). At 1% density, bandwidth drops to ~50% of unsegmented scan. For SSM batching where segments are entire sequences (density ≪ 0.1%), this is not a concern; for SpMV with many short rows, it matters.

- **$O(n^2)$ work in basic SCD/SSCR:** The straightforward algorithm has quadratic work. The block-recursive improvement achieves $O(n)$ work but requires specialized VCU circuits (REVSPEC instruction) that may not be available on all hardware — "at least at the time of this writing, this improvement is mainly of theoretical interest."

- **Ascend-specific implementation:** The proof-of-concept is implemented only on Ascend 910B using AscendC. Porting to NVIDIA GPUs (tensor cores + CUDA cores) or TPUs (MXU + vector unit) requires re-implementation, though the algorithmic structure is hardware-agnostic.

- **No fusion of SCD primitives:** The current implementation runs SCAN, COMPRESS, and DIFF as separate kernel invocations. Fusing them into a single kernel (as done by FlashAttention for attention) could significantly reduce memory traffic but is left as future work.

- **Floating-point non-associativity:** The speculative unsegmented scan followed by differentiation introduces different rounding behavior compared to a direct segmented scan. For neural network training (where floating-point tolerance is acceptable) this is fine; for exact integer applications, the paper's analysis ensures correctness via bit-width bounds.

## Implementation Notes

```python
import torch

def scd_segmented_sum(x, f, s=128):
    """
    SCD: Segmented Sum via Scan-Compress-Differentiation.
    x: (n,) data vector
    f: (n,) boolean segment start flags (1 = new segment)
    s: MMU tile size (match hardware: 16 for NVIDIA, 128 for Ascend)
    Returns: segment sums (one per segment)
    """
    n = len(x)

    # Step 1: SCAN — speculative unsegmented prefix scan via MMU
    # (In practice, this uses MatMulScan with U_s matrices)
    x_hat = torch.cumsum(x, dim=0)  # MMU-accelerated scan

    # Step 2: Shift flags to mark segment endpoints
    f_minus = torch.cat([f[1:], torch.ones(1, dtype=f.dtype, device=f.device)])

    # Step 3: COMPRESS — gather scanned values at segment endpoints
    endpoint_indices = torch.nonzero(f_minus).squeeze()
    z = x_hat[endpoint_indices]

    # Step 4: DIFF — vector differentiation to correct speculation
    z_diff = torch.zeros_like(z)
    z_diff[0] = z[0]
    z_diff[1:] = z[1:] - z[:-1]

    return z_diff


def sscr_segmented_scan(x, f, s=128):
    """
    SSCR: Segmented Scan via Scan-Scan-Compress-Revert.
    x: (n,) data vector
    f: (n,) boolean segment start flags
    Returns: (n,) segmented prefix scan
    """
    n = len(x)

    # Step 1: Two speculative scans via MMU
    x_hat = torch.cumsum(x, dim=0)       # scan of data
    f_hat = torch.cumsum(f.float(), dim=0)  # scan of flags (gives segment index)

    # Step 2: Shift flags for endpoint marking
    f_minus = torch.cat([f[1:], torch.ones(1, dtype=f.dtype, device=f.device)])

    # Step 3: COMPRESS — gather endpoint values
    endpoint_indices = torch.nonzero(f_minus).squeeze()
    w = x_hat[endpoint_indices]  # cumulative sums at segment endpoints

    # Step 4: REVERT — subtract previous segment's total from each position
    # For each position i, find its segment index and subtract the
    # cumulative sum at the end of the previous segment
    segment_idx = (f_hat - 1).long()  # 0-indexed segment for each position
    prev_seg_sum = torch.zeros(n, dtype=x.dtype, device=x.device)
    mask = segment_idx > 0
    prev_seg_sum[mask] = w[segment_idx[mask] - 1]

    z = x_hat - prev_seg_sum

    return z


def spmv_via_scd(A_row, A_col, A_val, x, n):
    """
    Sparse matrix-vector multiplication using SCD (Algorithm 4.4).
    A is in CSR format: A_row (row pointers), A_col (column indices), A_val (values).
    x: dense vector
    Returns: y = Ax
    """
    # Step 1: GATHER — collect x values at column indices
    w = x[A_col]  # w(i) = x(A_col(i))

    # Step 2: MULT — element-wise multiply with matrix values
    z = w * A_val

    # Step 3: SCD — segmented sum to reduce each row
    # Convert row pointers to segment flags
    nnz = len(A_val)
    f = torch.zeros(nnz, dtype=torch.long)
    f[A_row[:-1]] = 1  # mark start of each row's nonzeros

    y = scd_segmented_sum(z, f)

    return y
```

**Hardware mapping:**

| MMV-RAM Component | NVIDIA GPU | Huawei Ascend | Google TPU |
|-------------------|-----------|---------------|------------|
| MMU (matrix mult) | Tensor Cores | Cube Unit | MXU |
| VCU (vector ops) | CUDA Cores | Vector Unit | Vector Proc. Unit |
| Scalar Unit | SM scalar pipe | Scalar Unit | Scalar pipe |
| $s$ (tile size) | 16 | 128 | 128–256 |

**Key engineering insight:** The COMPRESS operation's irregular gather pattern is the performance bottleneck. On Ascend, this uses the VCU's scatter/gather capabilities. On NVIDIA GPUs, shared memory staging with warp-level ballot/shuffle instructions could accelerate this step. Fusing SCAN + COMPRESS into a single kernel pass (avoiding the HBM round-trip for $\hat{\mathbf{x}}$) is the main optimization opportunity.

## References

- Sobczyk, A., Sorrentino, G., and Zouzias, A. (2025). Segmented Operations using Matrix Multiplications. arXiv:2506.23906v2.
- Zouzias, A. and McColl, W.F. (2024). A Parallel Scan Algorithm in the Tensor Core Unit Model. arXiv:2411.17887.
- Wróblewski, B., Gottardo, G., and Zouzias, A. (2025). Parallel Scan on Ascend AI Accelerators. arXiv:2505.15112. IPDPS 2025.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.
- Sengupta, S., Harris, M., and Garland, M. (2008). Efficient Parallel Scan Algorithms for GPUs. NVIDIA Tech Report.
- Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality. (Mamba-2).

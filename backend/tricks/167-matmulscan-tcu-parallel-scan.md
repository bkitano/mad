# 167: MatMulScan — Parallel Scan in the Tensor Core Unit Model

**Category**: parallelization
**Gain type**: efficiency
**Source**: Zouzias & McColl (Euro-Par 2023 / arXiv:2411.17887, 2024)
**Paper**: [papers/tcu-parallel-scan.pdf]
**Documented**: 2026-02-15

## Description

MatMulScan is a parallel scan (prefix sum) algorithm designed for the **Tensor Core Unit (TCU) model** of computation, where multiplication of two square matrices of constant size $s$ is the basic operation. Unlike traditional PRAM-based scan algorithms (Blelloch, Brent-Kung) that count scalar binary operations, MatMulScan counts *matrix multiplications* — the primitive that maps directly to tensor cores (NVIDIA), TPUs (Google), and Ascend cube units (Huawei).

The algorithm is a **generalization of the Brent-Kung scan** to arbitrary radix $s$: when $s = 2$, it reduces exactly to the classical Brent-Kung construction. The key insight is that both the local prefix sum operation and the broadcast-and-add correction can be encoded as multiplication by two constant $s \times s$ matrices — a lower-triangular all-ones matrix $L_s$ (for prefix sums) and a broadcast matrix $B_s$ (for scalar/vector addition). This means the entire scan algorithm — both upsweep and downsweep — consists solely of batched matrix multiplications against these two fixed matrices.

This is directly relevant to SSM recurrences: the Mamba-2 paper explicitly identifies that "matmul FLOPs are up to 16x faster than non-matmul FLOPs" on modern accelerators. MatMulScan provides the theoretical foundation for why reformulating scans as matrix multiplications (as done by RandMScan, trick 099) yields speedups — it's not just a heuristic, but a provably work-efficient algorithm in the TCU computation model.

## Mathematical Form

**TCU Model:** An $(s^2, \ell)$-TCU is a standard RAM augmented with a circuit that performs multiplication of an $s \times s$ matrix by an $s \times m$ matrix ($m \geq s$) in time $O(ms + \ell)$, where $\ell \geq 0$ is the latency of initiating a matrix multiplication.

**Two constant matrices encode the scan:**

The lower-triangular all-ones matrix performs local prefix sums:

$$
L_s = \begin{bmatrix} 1 & 0 & 0 & \cdots & 0 \\ 1 & 1 & 0 & \cdots & 0 \\ 1 & 1 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ 1 & 1 & 1 & \cdots & 1 \end{bmatrix}
$$

Given vector $w$ of size $s$: $L_s w = \text{prefix\_sum}(w)$.

The broadcast matrix encodes scalar addition to a vector:

$$
B_s = \begin{bmatrix} 1 & 0 & 0 & \cdots & 0 \\ 1 & 1 & 0 & \cdots & 0 \\ 1 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ 1 & 0 & 0 & \cdots & 1 \end{bmatrix}
$$

Given column vector $[\alpha; q]$ (scalar $\alpha$ prepended to vector $q$ of size $s-1$): $B_s [\alpha; q] = q + \alpha \cdot \mathbf{1}_{s-1}$, i.e., it broadcasts the first element and adds it to all others.

**Algorithm 1 (MatMulScan):** Input: vector $x$ of size $n = s^k$, parameter $s$.

**Phase 1 — Upsweep** (compute local prefix sums at exponentially increasing strides):

For $t = 0, 1, \ldots, k-1$:
1. Gather: $y \leftarrow x[\text{start} :: s^t]$ where $\text{start} = s^t - 1$
2. Batch multiply: $z \leftarrow \text{BatchMatMul}(y, L_s)$
3. Scatter: $x[\text{start} :: s^t] \leftarrow z$

**Phase 2 — Downsweep** (broadcast and correct local prefix sums):

For $t = k-1, k-2, \ldots, 1$:
1. Gather: $y \leftarrow x[\text{start} :: s^{t-1}]$ where $\text{start} = s^t - 1$
2. Batch multiply: $z \leftarrow \text{BatchMatMul}(y, B_s)$
3. Scatter: $x[\text{start} :: s^{t-1}] \leftarrow z$

**BatchMatMul procedure:** Takes vector $y$ of length $m$ and matrix $A_s$ of size $s \times s$:
1. Zero-pad $y$ to length $s^2 \lceil m/s^2 \rceil$
2. Reshape as $(\lceil m/s^2 \rceil, s, s)$ tensor $T$
3. Compute $W = A_s \cdot T$ (batched matrix multiplication)
4. Flatten $W$ back to vector, drop padding

**Relationship to Brent-Kung:** When $s = 2$:
- $L_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ encodes a single binary addition
- $B_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = L_2$ (same matrix)
- The algorithm reduces to exactly the classical Brent-Kung scan with depth $2\log_2(n) - 1$ and work $2(n-1)$

## Complexity

**Lemma 1 (Power-of-$s$ inputs):** For $n = s^k$, Algorithm 1 has:

| Metric | Value |
|--------|-------|
| Depth (TCU model) | $2k - 1 = 2\log_s(n) - 1$ |
| Matrix multiplications | $\leq \lceil \frac{2n}{s(s-1)} \rceil + 2k - 2$ |
| Scalar additions (work) | $\lceil n(1 + s/2) \rceil + O(s^3 \log_s(n))$ |

**Theorem 1 (TCU time):** Algorithm 1 takes $O(n + \ell k)$ time in the $(s^2, \ell)$-TCU model.

**Corollary 1 (Arbitrary $n$):** There exists an algorithm with depth at most $2\lfloor \log_s(n) \rfloor$, taking $O(n + s\ell \log_s^2(n))$ time, performing $O(n/s^2)$ matrix multiplications.

**With $p$ parallel TCUs (Brent's theorem):** Runtime is $O(n(1 + \ell/s^2)/p + (s^2 + \ell)\log_s(n))$.

**Comparison table (from paper):**

| Method | Depth | Work | Model |
|--------|-------|------|-------|
| Sklansky | $\log_2(n)$ | $n\log_2(n)/2$ | PRAM |
| Hillis-Steele | $\log_2(n)$ | $n\log_2(n) - n + 1$ | PRAM |
| Blelloch | $2\log_2(n)$ | $2(n-1)$ | PRAM |
| Brent-Kung | $2\log_2(n) - 1$ | $2n - \log_2(n) - 2$ | PRAM |
| Dakkak et al. Alg. 7 | $5\lceil n/s^3 \rceil$ | $O(ns)$ | TCU / GEMMs |
| **MatMulScan** ($s=2$) | $2\log_2(n) - 1$ | $2n + O(\log_2(n))$ | TCU |
| **MatMulScan** ($s=4$) | $\log_2(n) - 1$ | $3n + O(\log_2(n))$ | TCU |
| **MatMulScan** (general) | $2\log_s(n) - 1$ | $n(1+s/2) + O(s^3\log_s(n))$ | TCU |

**Key trade-off:** Increasing $s$ reduces depth (fewer sequential steps) but increases work per step (each matmul involves $s(s-1)/2$ additions). For $s = 4$, depth is halved vs. $s = 2$ at the cost of 1.5x more work.

**Memory:** $O(n)$ — the algorithm operates in-place with $O(1)$ auxiliary matrices ($L_s$ and $B_s$ are constant and can be stored in registers/constant memory).

## Applicability

- **SSM recurrence parallelization:** The scan of $A_t$ matrices in SSM recurrences (Mamba, S5) can be reformulated as a MatMulScan where the "scalar" elements are replaced by $d \times d$ state transition matrices. The same up-sweep/down-sweep structure applies to matrix-valued scans via an associative binary operator.
- **Tensor core utilization:** The algorithm converts all scan operations into batched matrix multiplications against constant matrices, which directly maps to WMMA/WGMMA instructions on NVIDIA GPUs, MXU operations on TPUs, and cube unit operations on Ascend NPUs.
- **Mamba-2 SSD kernel:** The Mamba-2 paper's insight that "matmul FLOPs are up to 16x faster" is precisely the TCU model assumption. MatMulScan provides the theoretical work/depth analysis for this regime.
- **Choosing block size $s$:** The parameter $s$ should match the tensor core's native matrix size (e.g., $s = 16$ for NVIDIA's `mma.m16n16k16` instruction, $s = 128$ for TPU systolic arrays). Larger $s$ gives lower depth but higher total work.
- **Gradient boosting and parallel sorting:** The paper motivates MatMulScan via XGBoost binary tree splits (which require multiple prefix sums) and radix sort (which reduces to prefix sum via Blelloch's construction).

## Limitations

- **Low tensor core utilization for $L_s$:** The matrix $L_s$ is lower-triangular, meaning at most 50% of the tensor core's compute units are utilized during upsweep multiplications. The paper explicitly notes this as "roughly $O(1/s)$ utilization" for the broadcast matrix $B_s$ in the downsweep.
- **Gather/scatter overhead:** Each round requires strided gather and scatter operations. On GPUs, non-contiguous memory access patterns can degrade throughput. The paper acknowledges that "the scatter/gather memory operations could be a critical bottleneck."
- **No high-performance implementation yet:** The paper provides only a NumPy reference implementation and theoretical analysis — no optimized CUDA/TIK kernel. The authors note plans for an Ascend implementation as future work.
- **Work overhead for large $s$:** While depth decreases as $O(\log_s n)$, total work grows as $O(n \cdot s/2)$. For $s = 16$ (typical tensor core size), the work is $\sim 8n$ — significantly more than the $2n$ of Brent-Kung, though the faster matmul throughput may compensate.
- **Theoretical model gap:** The TCU model ignores memory hierarchy, bank conflicts, and warp scheduling. Real performance depends heavily on data layout and memory access patterns that the model abstracts away.

## Implementation Notes

```python
import numpy as np

def matmul_scan(x, s, k):
    """
    MatMulScan: Parallel scan via batched matrix multiplications.
    x: input vector of size n = s^k
    s: TCU matrix size parameter (radix)
    k: number of levels (n = s^k)

    Reference implementation from the paper (Appendix A).
    """
    # Constant matrices
    L_s = np.tril(np.ones(s))       # Lower-triangular all-ones (prefix sum)
    B_s = np.eye(s)
    B_s[:, 0] = 1                   # Broadcast matrix

    # Phase 1: Upsweep (local prefix sums at increasing strides)
    for t in range(k):
        start, step = s ** t - 1, s ** t
        y = x[start::step]
        z = batch_matmuls(y, L_s)
        x[start::step] = z

    # Phase 2: Downsweep (broadcast corrections at decreasing strides)
    for t in range(k - 1, 0, -1):
        start, step = s ** t - 1, s ** (t - 1)
        y = x[start::step]
        z = batch_matmuls(y, B_s)
        x[start::step] = z

    return x


def batch_matmuls(y, A_s):
    """
    Reshape vector y into batched s×s matrices,
    multiply each by constant matrix A_s.
    """
    m, s = len(y), A_s.shape[0]
    y = y.flatten()
    extra_pad = int((s ** 2) * np.ceil(m / s ** 2))
    y.resize(extra_pad)

    T = y.reshape((-1, s, s)).transpose((0, 2, 1))
    W = A_s @ T  # Batched matrix multiplication (the TCU primitive)
    z = W.reshape((-1, s ** 2), order='F').flatten()
    return z[:m]


# Example: scan of [1, 2, ..., 16] with s=4 (k=2)
x = np.arange(1, 17, dtype=float)
result = matmul_scan(x.copy(), s=4, k=2)
# result = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]
```

**Mapping to GPU tensor cores:**
- Set $s$ to match native MMA tile size (e.g., $s = 16$ for `mma.sync.m16n16k16`)
- $L_s$ and $B_s$ are constant — load once into shared memory or registers
- Each `BatchMatMul` call maps to a single batched WMMA/WGMMA invocation
- Gather/scatter between rounds: use shared memory staging with bank-conflict-free layout
- For non-power-of-$s$ input sizes: use the base-$s$ decomposition (Section 2.2 of paper) to handle arbitrary $n$

## References

- Zouzias, A. and McColl, W.F. (2024). A Parallel Scan Algorithm in the Tensor Core Unit Model. arXiv:2411.17887. (Also in Euro-Par 2023, Springer LNCS.)
- Chowdhury, R., Silvestri, F., and Vella, F. (2020). A Computational Model for Tensor Core Units. Proc. ACM SPAA.
- Chowdhury, R., Silvestri, F., and Vella, F. (2021). Algorithm Design for Tensor Units. Euro-Par 2021.
- Dakkak, A., Li, C., Xiong, J., Gelado, I., and Hwu, W.-m. (2019). Accelerating Reduction and Scan Using Tensor Core Units. Proc. ACM ICS.
- Brent, R.P. and Kung, H.T. (1982). A Regular Layout for Parallel Adders. IEEE Trans. Computers C-31(3).
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.
- Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality. (Mamba-2).

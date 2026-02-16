# 099: RandMScan: Matrix-Based Parallel Scan with Random-Jump Aggregation

**Category**: parallelization
**Gain type**: efficiency
**Source**: Peng et al. (Scientific Reports, 2025)
**Paper**: [papers/randmscan-parallel-scan-acceleration.pdf]
**Documented**: 2026-02-15

## Description

RandMScan reformulates parallel prefix scan (the core primitive underlying SSM recurrences, linear attention cumulative sums, and chunkwise parallelism) as a sequence of matrix multiplications, making it efficient on modern AI accelerators equipped with processing element (PE) arrays and tensor cores rather than traditional vector units. The algorithm operates in two stages: (1) a **local chunk stage** that computes intra-chunk scans purely via matrix operations using upper-triangular and lower-triangular matrices, and (2) a **global stage** using a lightweight **Random-Jump** communication strategy that propagates inter-chunk prefix sums without global barriers or hierarchical reduction trees.

This is directly relevant to column-sparse transition matrices: the parallel scan that computes cumulative products of $A_t = P_t D_t$ matrices is the computational bottleneck for parallelizing SSM recurrences. RandMScan accelerates precisely this scan operation, achieving 73--87% speedup on PE-array accelerators and 15--26% end-to-end latency reduction on downstream scan-dependent tasks.

## Mathematical Form

**Classical Scan Reformulation as Matrix Multiply:**

Given input sequence $z = [x_1, x_2, \ldots, x_n]$, the inclusive scan (prefix sum) can be expressed as:

$$
\text{scan}(z) = A \cdot U + L^{-} \cdot (A \cdot \mathbf{1})
$$

where:
- $A \in \mathbb{R}^{s \times s}$ is the input reshaped into a square matrix
- $U \in \mathbb{R}^{s \times s}$ is an all-ones upper triangular matrix
- $L^{-} \in \mathbb{R}^{s \times s}$ is a strictly lower triangular all-ones matrix
- $\mathbf{1} \in \mathbb{R}^{s \times 1}$ is an all-ones vector

The result $\text{scan}(z) \in \mathbb{R}^{s \times s}$ is flattened back to a 1D sequence.

**Local Chunk Stage (Algorithm 1):**

For a chunk of length $l = (k-1)k^2$, divide into $k-1$ blocks of size $k^2$:

1. Reshape each block $x^{(i)}$ into matrix $A^{(i)}_{k \times k}$
2. Compute row-wise prefix sums via upper-triangular multiply:
$$
B^{(i)} = A^{(i)} \cdot U_{k \times k}
$$
3. Stack last columns of all $B^{(i)}$ to form $V_{k \times (k-1)}$, prepend zero column to get $V'_{k \times k}$
4. Compute inter-block cumulative sums:
$$
C_{k \times k} = L_{k \times k} \cdot V'
$$
where $L$ is an all-ones lower triangular matrix
5. For each block $i$, down-shift column $i$ of $C$ and broadcast as correction:
$$
D^{(i)} = B^{(i)} + c'^{(i)}
$$
6. Extract inter-chunk boundary: $s = \text{last row of } C$, compute $s^{\text{scan}} = s \cdot U$

**Operation count for $k-1$ blocks:** $k+1$ matrix multiplications and $2k-2$ matrix additions, using only **two** special matrices ($U$ and $L$).

**Global Stage: Random-Jump Strategy (Algorithm 2):**

After local chunk scans, each chunk $j$ has:
- $\text{flag}_j \leftarrow 1$ (indicates local scan complete)
- $\text{data}_j \leftarrow y^{(j)}[-1]$ (last element = chunk sum)

The Random-Jump loop for chunk $j$:
```
while True:
    p = j - flag_j                    // target chunk to read from
    data_j += data_p                  // accumulate prefix
    flag_j += flag_p                  // advance jump distance
    if flag_p == p:                   // reached a prefix-complete chunk
        break
```

After convergence, $\text{data}_j$ contains the cumulative sum of chunks $1$ through $j$. Apply offset:

$$
Y^{(j)} = y^{(j)} + (\text{data}_j - y^{(j)}[-1])
$$

**Key Property:** Jump distances double each round (similar to pointer-jumping in parallel list ranking), so convergence takes $O(\log m)$ rounds for $m$ chunks, without requiring global barriers.

## Complexity

| Operation | Classical GPU Scan | Matrix-Based (Prior) | RandMScan |
|-----------|--------------------|---------------------|-----------|
| Local stage mat-muls | N/A (vector ops) | 3 per block | $\frac{k+1}{k-1} \approx 1$ per block |
| Auxiliary matrices | 0 | 3 ($U$, $L^-$, $\mathbf{1}$) | 2 ($U$, $L$) |
| Memory traffic (per element) | $\sim$16 bytes | $\sim$25 bytes | $\sim$8 bytes |
| Global stage passes | 2 (up-sweep + down-sweep) | 2+ (hierarchical) | $O(\log m)$ async rounds |
| Global barrier required | Yes | Yes | **No** |

**Speedup on PE arrays (SOFA accelerator):**

| Input size | Prior best (ms) | RandMScan (ms) | Speedup |
|------------|----------------|----------------|---------|
| $2^{16}$ | 0.07 | 0.04 | 75% |
| $2^{18}$ | 0.26 | 0.15 | 73% |
| $2^{20}$ | 0.90 | 0.48 | 87% |
| $2^{22}$ | 3.69 | 2.04 | 81% |
| $2^{24}$ | 14.94 | 8.31 | 79% |

**Downstream task latency reduction:** 15--26% for radix sort, CSR matrix addition, and top-$p$ sampling.

## Applicability

- **SSM parallel scans:** The cumulative product $A_{t:s} = A_t \cdots A_{s+1}$ in Mamba, S5, and PD-SSM is computed via parallel scan with an associative operator. RandMScan accelerates the scan primitive itself, especially on edge accelerators with matrix engines.
- **Linear attention cumulative sums:** The KV state accumulation $S_t = \lambda_t S_{t-1} + v_t k_t^\top$ reduces to a scan, directly accelerated by this method.
- **Top-$p$ sampling in LLM inference:** Sorting and cumulative-sum operations in nucleus sampling account for 19--39% of execution time; RandMScan reduces this by up to 26%.
- **Chunkwise parallel algorithms:** Any algorithm that uses a local-scan + global-aggregation pattern (e.g., chunkwise-parallel-scan for SSMs) can adopt RandMScan's two-stage framework.
- **Edge AI accelerators:** Particularly beneficial on hardware with PE arrays but limited vector units (Huawei Ascend, custom NPUs), where traditional GPU scan algorithms are inefficient.

## Limitations

- Designed primarily for **additive** scan (prefix sum); applying to general associative scans (e.g., matrix-matrix products in SSM recurrences) requires mapping to matrix multiplication, which may not always be natural for non-commutative operators
- Assumes rectangular PE arrays for the local chunk stage; non-rectangular or irregular PE layouts may require additional scheduling
- The Random-Jump global stage introduces minor floating-point non-associativity (relative deviations $\sim 10^{-6}$ to $10^{-7}$), acceptable for most neural network applications but not for exact integer computations requiring bitwise reproducibility
- On traditional GPUs with fast vector units (e.g., CUDA cores), the advantage over CUB's `DeviceScan` is marginal — the main benefit is on tensor-core / PE-array dominated hardware
- Does not currently exploit specialized hardware units like attention engines or convolution engines that some accelerators provide

## Implementation Notes

```python
import torch

def local_chunk_scan_matrix(x, k):
    """
    Matrix-based local chunk scan.
    x: (l,) input chunk of length l = (k-1) * k^2
    k: block size parameter (determined by PE array granularity)
    Returns: (l,) scanned output
    """
    num_blocks = k - 1
    block_size = k * k

    # Upper triangular all-ones matrix (for row-wise prefix sums)
    U = torch.triu(torch.ones(k, k, device=x.device))
    # Lower triangular all-ones matrix (for inter-block accumulation)
    L = torch.tril(torch.ones(k, k, device=x.device))

    # Step 1: Reshape blocks and compute row-wise prefix sums
    Bs = []
    for i in range(num_blocks):
        A_i = x[i * block_size : (i + 1) * block_size].reshape(k, k)
        B_i = A_i @ U  # Row-wise prefix sums via matrix multiply
        Bs.append(B_i)

    # Step 2: Stack last columns, compute inter-block cumulative sums
    V = torch.stack([B[:, -1] for B in Bs], dim=1)  # (k, num_blocks)
    V_prime = torch.cat([torch.zeros(k, 1, device=x.device), V], dim=1)  # (k, k)
    C = L @ V_prime  # Inter-block cumulative sums

    # Step 3: Broadcast corrections back to each block
    Ds = []
    for i in range(num_blocks):
        correction = C[: , i].unsqueeze(1).expand_as(Bs[i])  # Broadcast column
        Ds.append(Bs[i] + correction)

    # Step 4: Flatten
    y = torch.cat([D.reshape(-1) for D in Ds])

    # Step 5: Compute inter-chunk scan values for global stage
    s_scan = C[-1, :] @ U  # Last row of C through U

    return y, s_scan


def random_jump_global_scan(chunk_sums):
    """
    Random-Jump global aggregation (simplified sequential simulation).
    chunk_sums: (m,) last element of each chunk's local scan
    Returns: (m,) global prefix sums
    """
    m = len(chunk_sums)
    data = chunk_sums.clone()
    flag = torch.ones(m, dtype=torch.long)  # Initially: only local sum available

    # In hardware, this runs asynchronously per chunk
    converged = torch.zeros(m, dtype=torch.bool)
    while not converged.all():
        for j in range(1, m):
            if converged[j]:
                continue
            p = j - flag[j].item()
            if p >= 0:
                data[j] = data[j] + data[p]
                flag[j] = flag[j] + flag[p]
                if flag[p].item() == p + 1:  # Reached prefix-complete
                    converged[j] = True

    return data
```

## References

- Peng, S., Lin, X., Zhang, Y., Xiao, Y., and Hu, Y. (2025). RandMScan: Accelerating Parallel Scan via Matrix Computation and Random-Jump Strategy. Scientific Reports 16:2475.
- Dakkak, A., Li, C., Xiong, J., Gelado, I., and Hwu, W.-m. (2019). Accelerating Reduction and Scan Using Tensor Core Units. ACM SC 2019.
- Wroblewski, B., Gottardo, G., and Zouzias, A. (2025). Parallel Scan on Ascend AI Accelerators. arXiv:2505.15112.
- Harris, M., Sengupta, S., and Owens, J.D. (2007). Parallel Prefix Sum (Scan) with CUDA. GPU Gems 3.
- Merrill, D. and Garland, M. (2016). Single-Pass Parallel Prefix Scan with Decoupled Look-Back. NVIDIA Tech Report.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.

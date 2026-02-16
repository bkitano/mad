# 244: Diagonal SSD — Sum-of-Heads Attention Decomposition

**Category**: decomposition
**Gain type**: expressivity
**Source**: Hu, Zhang, ElSheikh, Wu & Liu (2025) — Northwestern University / UCSD / Ensemble AI
**Paper**: [papers/generalized-diagonal-ssd.pdf]
**Documented**: 2026-02-15

## Description

Structured State-Space Duality (SSD) establishes that a scalar-identity SSM ($A^t = a_t I_N$) is equivalent to a masked self-attention with a 1-semiseparable causal mask. However, this original duality covers only the simplest case where all $N$ state dimensions share the same decay rate.

**Diagonal SSD** extends this duality to **general diagonal SSMs** where $A^t = \text{diag}(a_{1,n}^t, \ldots, a_{N,n}^t)$ — each state dimension has its own independent, time-varying decay. The key mathematical result is that the sequence kernel $M$ of a diagonal SSM decomposes as a **sum of $N$ independent 1-semiseparable attention heads**:

$$
M = \sum_{n=1}^{N} M^n, \quad M^n = \text{1SS}(A_{n,n}^1, \ldots, A_{n,n}^T) \odot (b^n \cdot c^{n\top})
$$

where each $M^n$ corresponds to one state dimension operating as an independent scalar SSM (equivalently, a single attention head with its own 1-SS causal mask).

**Why this matters:** Diagonal SSMs are strictly more expressive than scalar-identity SSMs at the same state dimension $N$. A diagonal SSM with $N$ distinct decay rates can realize kernels that require $N$ "new columns" (independent directions) in the attention matrix — while a scalar SSM can produce at most 1 new column per position. In concrete terms: a 2D diagonal SSM with decay rates $\lambda_1 \neq \lambda_2$ can fit a mixture of two exponential signals, which no scalar SSM of dimension 2 can match.

**The efficiency result:** Despite this richer expressivity, diagonal SSD has the **same optimal $O(TN)$ computational complexity** as scalar SSD. The algorithm decomposes into $N$ independent scalar SSM computations (one per state dimension), each costing $O(T)$, yielding $O(TN)$ total — matching both the FLOPs and the constant factor (4$NTd$ FLOPs total) of the scalar case.

**Negative results:** The paper also proves: (i) SSD cannot extend to softmax attention due to rank explosion ($\text{Softmax}(QK^\top)$ becomes full-rank $T$, violating the $N$-semiseparable bound), and (ii) even low-rank non-diagonal SSMs may fail to have a 1-SS attention dual.

## Mathematical Form

**Time-varying diagonal SSM:**

The recurrence with diagonal state matrices:

$$
h_{t+1} = A^t h_{t-1} + b_t x_t, \quad y_t = c_t^\top h_t, \quad \text{for } t \in [T]
$$

where $h_t \in \mathbb{R}^{N \times d}$, $A^t = \text{diag}(A_{1,1}^t, \ldots, A_{N,N}^t) \in \mathbb{R}^{N \times N}$, $b_t \in \mathbb{R}^{N \times 1}$, $c_t \in \mathbb{R}^{N \times 1}$.

**SSM kernel (sequence mixing matrix):**

Unrolling the recurrence yields the input-output relation $y_t = \sum_{s=1}^{t} M_{t,s} x_s$ where:

$$
M_{j,i} = c_j^\top A^j \cdots A^{i+1} b_i, \quad \text{for } 1 \leq i \leq j \leq T
$$

Since each $A^t$ is diagonal, the product $A^j \cdots A^{i+1}$ is also diagonal, and $M_{j,i}$ decomposes by state dimension:

$$
M_{j,i} = \sum_{n=1}^{N} (c_j)_n \cdot (A_{n,n}^j \cdots A_{n,n}^{i+1}) \cdot (b_i)_n
$$

**Sum-of-heads decomposition:**

Defining per-dimension quantities:

- $b_t^n := (b_t)_n \in \mathbb{R}$ — $n$-th component of input projection at time $t$
- $c_t^n := (c_t)_n \in \mathbb{R}$ — $n$-th component of output projection at time $t$
- $a^t_{n} := A_{n,n}^t \in \mathbb{R}$ — $n$-th diagonal decay at time $t$

Each head $M^n$ is a rank-1 outer product masked by a 1-SS matrix:

$$
M^n = \text{1SS}(a_n^1, \ldots, a_n^T) \odot (b^n \cdot c^{n\top})
$$

where $\text{1SS}(a_1, \ldots, a_T)_{j,i} := a_j \cdots a_{i+1}$ for $j \geq i$ (product of decays), and $b^n, c^n \in \mathbb{R}^T$ collect the per-dimension input/output weights over time.

The full kernel is the sum:

$$
M = \sum_{n=1}^{N} M^n
$$

**Attention-like form:**

In matrix notation, the input-output map is:

$$
Y = M \cdot X = \sum_{n=1}^{N} \left[\text{1SS}(a_n^1, \ldots, a_n^T) \odot (b^n \cdot c^{n\top})\right] X
$$

This is equivalent to $N$ independent 1-SS masked attention operations summed together, where each "head" has its own causal mask (determined by its decay sequence $a_n^1, \ldots, a_n^T$) and its own rank-1 query-key structure ($c^n$ and $b^n$).

**Full-rank state matrices — 1-SS dual:**

When every $A^t$ has full rank (all diagonal entries nonzero), the attention-like dual simplifies further. Define rescaled projections:

$$
{b'}_t^n := (b_t)_n \cdot (A_{n,n}^1 \cdots A_{n,n}^t), \quad {c'}_t^n := (c_t)_n / (A_{n,n}^1 \cdots A_{n,n}^t)
$$

Then:

$$
M = \text{1SS}(1, 1, \ldots, 1) \odot (B' \cdot C'^\top) = \text{tril}(\mathbf{1}) \odot (B' C'^\top)
$$

where $B', C' \in \mathbb{R}^{T \times N}$ collect the rescaled projections. This shows that full-rank diagonal SSMs are equivalent to standard causal linear attention with transformed queries and keys — the decay information is absorbed into the projections.

**Expressivity advantage — "richer dynamics":**

A diagonal SSM with $N = 2$, $T = 4$, and distinct decay rates $\lambda_1 \neq \lambda_2$ produces a kernel:

$$
M = \begin{bmatrix} 2 & 0 & 0 & 0 \\ 1 & 2 & 0 & 0 \\ 0 & 1 & 2 & 0 \\ 0 & 0 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix} + \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

This kernel $M$ has 3 > 2 = $N$ new columns, so it **cannot** be realized by any scalar-identity SSM of dimension $N = 2$. Different decay rates enable independent time-scale dynamics within the same state size.

**Necessary and sufficient condition for 1-SS dual (Theorem 4.1):**

An $N$-SS matrix $M$ corresponding to an SSM has a 1-SS masked attention dual if and only if $M$ has block-diagonal structure where each diagonal block has at most $N$ new columns.

## Complexity

**Algorithm 1 — Diagonal SSD computation:**

```
procedure SSM(X):
    Z^n ← f(b^n, X)    for all n ∈ [N]     // Time O(NTd)
    H^n ← g(a^n, Z^n)   for all n ∈ [N]     // Time O(NTd)
    Y^n ← f(c^n, H^n)   for all n ∈ [N]     // Time O(NTd)
    Y ← Σ Y^n                                // Time O(NTd)
    return Y
```

where $f(x, Y)_{:,s} = x \odot Y_{:,s}$ (elementwise broadcast) and $g(x, Y)_{t+1,:} = x_{t+1} \cdot g(x, Y)_{t,:} + Y_{t+1,:}$ (cumulative product-sum scan).

| Operation | Scalar-Identity SSD | Diagonal SSD | Dense SSM |
|-----------|-------------------|-------------|-----------|
| Time complexity | $O(TN d)$ | $O(TN d)$ | $O(TN^2 d)$ |
| Constant factor | $4NTd$ FLOPs | $4NTd$ FLOPs | — |
| Memory | $O(NTd)$ | $O(NTd)$ | $O(N^2 T)$ |
| Parallelism | $Nd$ independent $O(T)$ processes | $Nd$ independent $O(T)$ processes | Sequential |
| Expressivity (new columns) | $\leq 1$ per position | Up to $N$ per position | Up to $N$ per position |

**Key result:** Diagonal SSMs achieve **strictly richer dynamics** (multiple independent decay rates) at **zero additional asymptotic cost** compared to scalar SSMs. The extra expressivity is "free" because the $N$ state dimensions are already computed independently.

## Applicability

- **Direct upgrade for Mamba-2/SSD**: Mamba-2 uses scalar-identity SSD ($A^t = a_t I_N$). Replacing this with diagonal $A^t = \text{diag}(a_1^t, \ldots, a_N^t)$ gives each state dimension its own decay rate, supporting richer multi-timescale dynamics at the same computational cost. The chunkwise SSD algorithm generalizes naturally — each chunk processes $N$ independent scalar scans.

- **Multi-timescale sequence modeling**: A diagonal SSM can simultaneously capture fast (small decay) and slow (large decay) patterns. This is useful for sequences with hierarchical temporal structure (e.g., character-level vs. phrase-level patterns in language, or fast oscillations vs. slow trends in time series).

- **Design space guidance**: The decomposition $M = \sum_n M^n$ shows that increasing the state dimension $N$ is equivalent to adding more independent attention heads, each with its own causal mask. This provides a principled lens for deciding between wider states vs. deeper stacking.

- **Theoretical foundation for linear attention design**: The necessary-and-sufficient condition for 1-SS duality (Theorem 4.1) constrains which SSM architectures can have efficient attention duals, guiding the design of new expressive-yet-efficient sequence models.

- **Impossibility result for softmax linearization**: The proof that $\text{Softmax}(QK^\top)$ has rank explosion (full rank $T$ for generic $Q, K$) formalizes why no finite-state SSM or linear RNN can exactly replicate softmax attention — useful for understanding the fundamental expressivity gap.

## Limitations

- **Theoretical paper — no optimized GPU kernel yet**: The authors explicitly note (Remark 4.3) that "there is no specialized kernel that directly accelerates diagonal-SSD computations" and that "realizing such performance requires dedicated diagonal-SSD kernel design, which we leave to future work." The algorithm matches scalar SSD's complexity in theory, but a practical implementation with fused chunkwise kernels does not yet exist.

- **Expressivity gains are task-dependent**: The multi-timescale advantage is most apparent when the data contains signals at multiple characteristic frequencies. For tasks where a single decay rate suffices, diagonal SSM offers no benefit over scalar SSM.

- **Does not address the fundamental recall limitation**: Both scalar and diagonal SSMs are bounded by the finite state size $N$. The duality shows this corresponds to attention with rank-$N$ masks, which cannot match the full-rank attention of softmax. The recall bottleneck of linear attention persists.

- **Non-diagonal SSMs may not have 1-SS duals**: The paper proves (Proposition 5.1) that even with low state dimension, general (non-diagonal) SSMs may not have a 1-SS attention dual. This limits the extension of SSD beyond diagonal structure.

- **Constant factor matters on GPU**: While the algorithm achieves $4NTd$ FLOPs (matching scalar SSD), the $N$ independent scans may have different memory access patterns than a single batched scan. GPU utilization depends on how well $N$ parallel scans map to the hardware — batching $N$ scans with different decay sequences may be less efficient than $N$ copies of the same scan.

## Implementation Notes

```python
# Diagonal SSD — Algorithm 1 from the paper
# Key insight: decomposes into N independent scalar SSM computations

def diagonal_ssd(X, A_diag, B, C):
    """
    Diagonal State-Space Dual computation.

    Args:
        X: (T, d) - input sequence
        A_diag: (T, N) - diagonal entries of state matrices [A^1_{n,n}, ..., A^T_{n,n}]
        B: (T, N) - input projection weights
        C: (T, N) - output projection weights

    Returns:
        Y: (T, d) - output sequence
    """
    T, d = X.shape
    N = A_diag.shape[1]

    Y = torch.zeros(T, d)

    # Each state dimension n is an independent scalar SSM
    # These N computations are embarrassingly parallel
    for n in range(N):  # Parallelizable over n
        # Step 1: f(b^n, X) — broadcast multiply input by b^n
        Z_n = B[:, n:n+1] * X  # (T, d) — elementwise broadcast

        # Step 2: g(a^n, Z^n) — cumulative product-sum scan
        # H^n_{t+1} = a^n_{t+1} * H^n_t + Z^n_{t+1}
        # This is a parallel scan with (multiply, add) semiring
        H_n = parallel_scan(
            decay=A_diag[:, n],  # (T,) per-dimension decay
            input=Z_n            # (T, d)
        )  # (T, d)

        # Step 3: f(c^n, H^n) — broadcast multiply output by c^n
        Y_n = C[:, n:n+1] * H_n  # (T, d)

        # Step 4: Accumulate
        Y = Y + Y_n

    return Y


def parallel_scan(decay, input):
    """
    Parallel prefix scan: H_t = decay_t * H_{t-1} + input_t

    On GPU, this maps to a work-efficient Blelloch scan
    or a chunkwise parallel algorithm (as in Mamba-2).
    Each scan is O(T) sequential, O(T log T) parallel work.
    """
    T, d = input.shape
    H = torch.zeros_like(input)
    H[0] = input[0]
    for t in range(1, T):
        H[t] = decay[t] * H[t-1] + input[t]
    return H


# Chunkwise variant (more GPU-friendly):
def diagonal_ssd_chunkwise(X, A_diag, B, C, chunk_size=256):
    """
    Chunkwise implementation of diagonal SSD.

    Within each chunk: quadratic attention-like computation
    Between chunks: linear state passing via N independent scans

    This is the natural extension of Mamba-2's SSD algorithm
    from scalar A to diagonal A.
    """
    T, d = X.shape
    N = A_diag.shape[1]
    num_chunks = T // chunk_size

    Y = torch.zeros(T, d)

    # For each state dimension, maintain a running state
    states = torch.zeros(N, d)  # (N, d)

    for chunk_idx in range(num_chunks):
        sl = slice(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size)

        # Extract chunk data
        X_c = X[sl]              # (C, d)
        A_c = A_diag[sl]         # (C, N)
        B_c = B[sl]              # (C, N)
        C_c = C[sl]              # (C, N)

        # For each state dimension n (parallelizable):
        for n in range(N):
            # Intra-chunk: compute 1-SS masked attention for head n
            # M^n_{j,i} = (a^n_j ... a^n_{i+1}) * b^n_i * c^n_j
            # This is a C×C attention matrix — compute via matmul
            log_decay = torch.log(A_c[:, n].abs())
            cum_decay = torch.cumsum(log_decay, dim=0)
            # L_{j,i} = exp(cum_decay_j - cum_decay_i) for j >= i
            L = torch.exp(cum_decay.unsqueeze(0) - cum_decay.unsqueeze(1))
            L = torch.tril(L)  # Causal mask

            # Scores: L ⊙ (c^n @ b^{n,T})
            scores = L * (C_c[:, n:n+1] @ B_c[:, n:n+1].T)  # (C, C)

            # Intra-chunk output
            Y_intra_n = scores @ X_c  # (C, d) — TENSOR CORE matmul

            # Inter-chunk: state contribution
            # Decay state to each position in chunk
            state_decay = torch.exp(cum_decay)  # (C,)
            Y_inter_n = C_c[:, n:n+1] * (state_decay.unsqueeze(1) *
                         (states[n:n+1] @ torch.ones(1, d)))

            Y[sl] += Y_intra_n + Y_inter_n

            # Update state for next chunk
            total_decay = torch.exp(cum_decay[-1])
            state_update = (B_c[:, n:n+1] *
                           torch.exp(cum_decay[-1] - cum_decay).unsqueeze(1)).T @ X_c
            states[n] = total_decay * states[n] + state_update.squeeze(0)

    return Y
```

**GPU efficiency analysis:**

1. **Embarrassingly parallel across state dimensions**: The $N$ independent scalar SSM computations have no data dependencies and can execute as a batched parallel scan. On GPU, this maps to $N \times d$ independent $O(T)$ scan lanes — excellent SM utilization when $Nd$ is large (typical: $N = 64$, $d = 128$ gives 8192 parallel lanes).

2. **Chunkwise algorithm natural extension**: The SSD chunkwise algorithm (intra-chunk quadratic attention + inter-chunk linear state passing) extends directly. Each state dimension $n$ contributes a rank-1 attention head within the chunk, and the $N$ heads sum. The intra-chunk computation is a sum of $N$ rank-1 outer products masked by different 1-SS matrices — these can be batched into a single matmul if the decay rates are handled via elementwise scaling.

3. **Same memory access pattern as scalar SSD**: The algorithm loads $X$ once and processes it with $N$ different decay/projection vectors. Since $B, C \in \mathbb{R}^{T \times N}$ are contiguous in memory, accessing different columns is coalesced. The extra memory for $N$ decay sequences is $O(TN)$ — negligible compared to the $O(TNd)$ for the data.

4. **No new kernel primitives required**: The building blocks are: (i) parallel scans (already optimized for Mamba-2), (ii) elementwise multiplies, and (iii) matmuls. A fused diagonal-SSD kernel would combine these, but even without fusion, the algorithm runs on existing primitives.

## References

- Hu, J.Y.-C., Zhang, X., ElSheikh, A., Wu, W., & Liu, H. (2025). On Structured State-Space Duality. arXiv:2510.04944.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. arXiv:2111.00396.
- Gupta, A., Gu, A., & Berant, J. (2022). Diagonal State Spaces are as Effective as Structured State Spaces. NeurIPS 2022. arXiv:2203.14343.
- Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). On the Parameterization and Initialization of Diagonal State Space Models. NeurIPS 2022. arXiv:2206.11893.
- Fasino, D. & Gemignani, L. (2002). Direct and Inverse Eigenvalue Problems for Diagonal-Plus-Semiseparable Matrices.
- Vandebril, R., Van Barel, M., & Mastronardi, N. (2005, 2008). Matrix Computations and Semiseparable Matrices (Vols. I & II).
- Code: https://github.com/MAGICS-LAB/state_space_duality

---
status: completed
priority: high
created: 2026-02-10
based_on: semiseparable-block-decomposition, wy-representation, chunkwise-parallel-scan, linear-attention-approximation
experiment_number: 002
results_file: 002_results.md
---

# SSD-DeltaNet: Semiseparable Block Decomposition for Delta Rule Models via WY Representation

## Hypothesis

The WY representation of DeltaNet's state matrix can be reformulated as a block-semiseparable matrix, enabling the SSD (Structured State Space Duality) algorithm to accelerate DeltaNet training by 2–4× via tensor-core-friendly matmul operations, without sacrificing the delta rule's superior associative memory properties.

## Background

**The efficiency gap**: DeltaNet maintains a $d \times d$ state matrix $S_t$ updated via rank-1 delta rules, giving it stronger associative memory than Mamba-2's scalar-state SSM. However, this comes at a cost: DeltaNet's chunkwise parallel training (via WY representation) requires $O(d^2 \cdot C)$ work per chunk, with operations that don't map well to tensor cores. Meanwhile, Mamba-2's SSD algorithm achieves 2–8× speedups by decomposing computation into matmul-heavy blocks, but it requires scalar/diagonal state transitions — it cannot handle DeltaNet's dense $d \times d$ state.

**The connection**: Both the SSD framework and the WY representation are block decomposition techniques that split sequences into chunks and separate intra-block from inter-block computation. The key insight is that DeltaNet's output within a chunk, when expressed through the WY representation, has a specific algebraic structure:

- The WY form $S_t = I + W_t Y_t^\top$ means the output $y_t = S_t v_t = v_t + W_t (Y_t^\top v_t)$
- Within a chunk, $W_t$ and $Y_t$ are built from accumulated $k$ and $v$ vectors
- The matrix mapping inputs to outputs within a chunk is lower-triangular and structured

**The question**: Can this within-chunk structure be expressed as a semiseparable matrix, enabling SSD-style block decomposition with matmul-based computation? If so, DeltaNet could achieve Mamba-2-level training speed while retaining its expressivity advantages.

## Mathematical Formulation

**DeltaNet Recurrence:**

$$
S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top
$$

**Block-level Output Matrix:**

For a chunk of $C$ tokens with keys $K = [k_1, \ldots, k_C]$ and values $V = [v_1, \ldots, v_C]$, the DeltaNet output matrix $M \in \mathbb{R}^{C \times C}$ has entries:

$$
M_{ij} = v_i^\top S_{j \to i} k_j \quad \text{for } i \geq j
$$

where $S_{j \to i}$ is the state accumulated from step $j$ to step $i$ via delta rule updates.

**WY-based Factorization:**

Using the WY representation, $S_{j \to i} = I + W_{j:i} Y_{j:i}^\top$ where $W, Y \in \mathbb{R}^{d \times (i-j)}$. This gives:

$$
M_{ij} = v_i^\top k_j + v_i^\top W_{j:i} Y_{j:i}^\top k_j
$$

The first term is a standard linear attention gram matrix ($VK^\top$), computable as a single matmul. The second term involves the accumulated WY factors.

**UT Transform: Converting Recurrence to Matmuls for Tensor Cores**

The naive WY recurrence for computing $W$ and $U$ matrices within a chunk is fully sequential:

$$
w^r_{[t]} = \beta^r_{[t]} \left( k^r_{[t]} - \sum_{i=1}^{r-1} w^i_{[t]} (k^{i\top}_{[t]} k^r_{[t]}) \right)
$$

$$
u^r_{[t]} = \beta^r_{[t]} \left( v^r_{[t]} - \sum_{i=1}^{r-1} u^i_{[t]} (k^{i\top}_{[t]} k^r_{[t]}) \right)
$$

This cannot use tensor cores because each step $r$ depends on all previous steps $1, \ldots, r-1$.

The **UT transform** (from [Joffrain et al. 2006](https://api.semanticscholar.org/CorpusID:15723171)) reformulates this recurrence into tensor-core-friendly matmuls:

$$
T_{[t]} = \left( I + \text{tril}(\text{diag}(\beta_{[t]}) K_{[t]} K^{\top}_{[t]}, -1) \right)^{-1} \text{diag}(\beta_{[t]})
$$

$$
W_{[t]} = T_{[t]} K_{[t]}, \quad U_{[t]} = T_{[t]} V_{[t]}
$$

**Why this enables tensor cores:**

1. **Build the lower-triangular matrix**: $L = \text{tril}(\text{diag}(\beta) K K^\top, -1)$ is a single matmul `K_beta @ K.T` followed by masking — fully parallelizable on tensor cores.

2. **Invert via forward substitution**: $(I + L)^{-1}$ is computed via vectorized forward substitution:
   ```python
   T = -(K_beta @ K.t()).tril(-1)  # tensor core matmul
   for i in range(1, C):
       T[i, :i] = T[i, :i] + (T[i, :, None] * T[:, :i]).sum(-2)  # small sequential loop
   T += torch.eye(C)
   ```
   The loop has $O(C)$ iterations but each iteration is $O(C)$ work with high parallelism.

3. **Final WY factors are pure matmuls**:
   ```python
   W = T @ K_beta  # tensor core matmul: (C×C) @ (C×d) → (C×d)
   U = T @ V_beta  # tensor core matmul: (C×C) @ (C×d) → (C×d)
   ```
   These are the dominant compute operations and map directly to tensor cores.

**Compute breakdown** (for chunk size $C$, head dim $d$):

| Operation | FLOPs | Tensor Core? |
|-----------|-------|--------------|
| Build $L = KK^\top$ | $O(C^2 d)$ | ✓ |
| Forward substitution | $O(C^2)$ | Partial |
| $W = TK$ | $O(C^2 d)$ | ✓ |
| $U = TV$ | $O(C^2 d)$ | ✓ |

The forward substitution is $O(C^2)$ while the matmuls are $O(C^2 d)$. Since $d \gg 1$ (typically 64–256), the matmuls dominate and achieve high tensor core utilization.

**SSD-style Decomposition:**

Split each chunk into sub-blocks of size $Q$:

1. **Intra-sub-block**: Compute the $Q \times Q$ output matrix densely (quadratic in $Q$ but $Q$ is small, ~32–64)
2. **Inter-sub-block**: The off-diagonal blocks factor through the compressed state at sub-block boundaries, computed via a short scan over $C/Q$ states

**Key Variables:**

- $M \in \mathbb{R}^{C \times C}$ — output (mixer) matrix within chunk
- $W_t, Y_t \in \mathbb{R}^{d \times t}$ — WY factors
- $Q$ — sub-block size (typically 32–64)
- $C$ — chunk size (typically 256)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Base model | DeltaNet |
| Chunk size | $C = 256$ |
| Sub-block size | $Q = 64$ |
| State dim | $d = 128$–$4096$ |
| Model sizes | 125M, 350M, 1.3B |

**Implementation (Pseudocode):**

```python
def ssd_deltanet_forward(K, V, beta, chunk_size=256, sub_block=64):
    T, d = K.shape
    outputs = []
    S = torch.zeros(d, d)  # inter-chunk state

    for chunk_start in range(0, T, chunk_size):
        chunk_K = K[chunk_start:chunk_start+chunk_size]
        chunk_V = V[chunk_start:chunk_start+chunk_size]
        chunk_beta = beta[chunk_start:chunk_start+chunk_size]

        # Intra-chunk: SSD-style block decomposition
        # Diagonal blocks: dense Q×Q matmul (tensor core friendly)
        # Off-diagonal: low-rank via WY state at sub-block boundaries
        chunk_out, S = ssd_chunk_forward(
            chunk_K, chunk_V, chunk_beta, S, sub_block
        )
        outputs.append(chunk_out)

    return torch.cat(outputs)
```

### Baseline

1. **DeltaNet (standard WY chunkwise)** — current best training algorithm for DeltaNet
2. **Mamba-2 (SSD)** — SSD with scalar state (faster but less expressive)
3. **Gated DeltaNet (GDN)** — DeltaNet + input-dependent gating
4. **Flash Linear Attention** — optimized linear attention training kernel

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.5\times$ DeltaNet | tokens/sec on A100/H100 |
| Memory usage | $\leq$ DeltaNet | Peak GPU memory vs sequence length |
| Perplexity | = DeltaNet | WikiText-103, The Pile |
| Associative recall | = DeltaNet | Synthetic benchmarks |
| FLOP utilization | $> 40\%$ | % of peak tensor core TFLOPS |

### Estimated Compute

**Medium**. Kernel development requires iterative profiling (~1 week of engineering). Benchmark evaluations: throughput profiling is cheap ($<1$ GPU-hour). Full training runs for quality verification: ~48–96 GPU-hours for 350M-param models on 15B tokens. **Total**: ~100–200 A100 GPU-hours.

## Expected Outcome

**If hypothesis is correct:**

- **Throughput**: SSD-DeltaNet achieves $1.5$–$3\times$ higher training throughput than standard WY-chunkwise DeltaNet, by converting most operations to matmuls
- **Memory**: Comparable to standard DeltaNet at short sequences; 20–40% reduction at long sequences ($>16$K) due to more efficient block structure
- **Quality**: Perplexity matches standard DeltaNet exactly (this is an algebraic reformulation, not an approximation)
- **FLOP utilization**: 40–60% of peak, up from ~15–25% for current DeltaNet

**If hypothesis is wrong:**

- The WY → semiseparable mapping doesn't hold cleanly, and speedups are minimal or come with quality degradation

## Minimum Viable Experiment

**Goal**: Demonstrate that SSD-style block decomposition provides measurable speedup for DeltaNet's WY representation before investing in full kernel development.

### Setup

| Component | Configuration |
|-----------|---------------|
| Model | Single DeltaNet layer (no full model) |
| State dim | $d = 64$ |
| Sequence length | $T = 512$ |
| Chunk size | $C = 64$ |
| Sub-block size | $Q = 16$ |
| Compute | Single GPU, $< 5$ minutes profiling |

**Implementation**: Compare two PyTorch implementations (no custom CUDA yet):

1. **Naive WY**: Standard loop accumulating WY factors, then materializing output
2. **Block-SSD**: Restructure computation into $Q \times Q$ matmul blocks + inter-block low-rank terms

### Task Definition

Pure forward pass throughput benchmark (no training, no backward pass):

```python
# Generate random K, V, beta tensors
K = torch.randn(T, d)
V = torch.randn(T, d)
beta = torch.sigmoid(torch.randn(T))

# Time both implementations over 100 iterations
naive_time = benchmark(naive_wy_forward, K, V, beta)
block_time = benchmark(block_ssd_forward, K, V, beta)
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Speedup | $> 1.3\times$ (block vs naive) |
| Numerical error | $\|y_{\text{naive}} - y_{\text{block}}\|_\infty < 10^{-5}$ |
| Matmul fraction | $> 60\%$ of FLOPs in matmul ops |

**The idea works if**: Block-SSD achieves $> 1.3\times$ speedup with negligible numerical error, even in pure PyTorch (before custom kernels). This proves the algebraic restructuring is sound and worth optimizing further.

### Failure Criteria

- **Kill the idea if**: Block-SSD is slower than naive (overhead exceeds benefit)
- **Kill the idea if**: Numerical error $> 10^{-3}$ (the decomposition is approximate, not exact)
- **Pause and investigate if**: Speedup is $< 1.1\times$ (restructuring may only help with custom kernels, higher risk)

### Why This Test Is Sufficient

1. **PyTorch speedup implies kernel speedup**: If matmul-heavy restructuring wins in PyTorch, custom tensor-core kernels will amplify the gap
2. **Numerical exactness is binary**: Either the semiseparable structure holds exactly, or it doesn't — a tiny model reveals this
3. **No training required**: This is a computational restructuring, not a learning algorithm — forward pass alone tests the core claim
4. **5 minutes to signal**: Fast iteration before committing to kernel engineering

### Implementation Sketch

```python
def naive_wy_forward(K, V, beta):
    """Standard WY accumulation."""
    T, d = K.shape
    W, Y = [], []
    outputs = []

    for t in range(T):
        # Compute S_t^T k_t via WY form
        if W:
            W_mat = torch.stack(W, dim=1)  # (d, t)
            Y_mat = torch.stack(Y, dim=1)  # (d, t)
            Stk = K[t] + Y_mat @ (W_mat.T @ K[t])
        else:
            Stk = K[t]

        error = V[t] - Stk
        W.append(beta[t] * K[t])
        Y.append(error)
        outputs.append(V[t] + sum(w * (y @ V[t]) for w, y in zip(W, Y)))

    return torch.stack(outputs)

def block_ssd_forward(K, V, beta, chunk_size=64, sub_block=16):
    """SSD-style block decomposition."""
    # Split into chunks, compute Q×Q diagonal blocks as matmuls
    # Off-diagonal blocks via low-rank WY state at sub-block boundaries
    ...
```

## Theoretical Analysis

**Complexity Comparison:**

| Operation | DeltaNet (WY) | SSD-DeltaNet |
|-----------|---------------|--------------|
| Intra-chunk work | $O(Cd^2)$ | $O(CQ) + O((C/Q)d^2)$ matmuls |
| Inter-chunk | $O(d^2)$ | $O(d^2)$ |
| Sequential steps | $O(T/C)$ | $O(T/C)$ |

**Speedup source:** Converting $O(Cd^2)$ scalar operations to $O(CQ)$ tensor-core matmuls. At $d = 256$, $C = 256$, $Q = 64$:

$$
\text{Speedup} \approx \frac{\text{Matmul TFLOPS}}{\text{Scalar TFLOPS}} \cdot \frac{Q}{d} = 16 \times \frac{64}{256} = 4\times
$$

## Risks & Limitations

1. **WY → semiseparable mapping may not be clean**: The delta rule's rank-1 update structure is more complex than SSM's scalar gating. The within-chunk output matrix may not be exactly semiseparable — it could be semiseparable only in an approximate sense, requiring error analysis.

2. **Sub-block size sensitivity**: The optimal $Q$ for DeltaNet may differ from Mamba-2's optimal $Q$ because the state dimension $d$ affects the rank of off-diagonal blocks differently. May need extensive tuning.

3. **Kernel engineering effort**: Custom CUDA/Triton kernels are required. The SSD kernel for Mamba-2 is ~1000 lines of Triton; DeltaNet's $d \times d$ state adds complexity. This is a significant engineering investment.

4. **Inter-chunk bottleneck**: The $d \times d$ state materialization at chunk boundaries is $O(d^2)$ and cannot be avoided. At large $d$ (e.g., $d = 4096$), this may dominate, limiting the speedup from intra-chunk optimization.

5. **Diminishing returns at small $d$**: If $d_{\text{model}}$ is small (128–256), the current WY implementation may already be fast enough that SSD-style restructuring adds overhead without meaningful speedup.

## Follow-up Experiments

1. **Approximate SSD-DeltaNet**: If exact semiseparable structure doesn't hold, test low-rank approximations of the within-chunk output matrix and measure quality degradation
2. **Hardware-aware $Q$ tuning**: Profile across A100, H100, and different $d_{\text{model}}$ values to find optimal sub-block sizes
3. **Combine with input-dependent gating**: Apply SSD-DeltaNet to Gated DeltaNet (GDN), which adds input-dependent decay — this is the most practically important variant
4. **Scale to 7B parameters**: If the kernel delivers on efficiency, test at scale to verify that DeltaNet's quality advantages persist and the speed gap vs. Mamba-2 is closed
5. **Backward pass optimization**: The backward pass through the SSD decomposition may have different bottlenecks; profile and optimize separately

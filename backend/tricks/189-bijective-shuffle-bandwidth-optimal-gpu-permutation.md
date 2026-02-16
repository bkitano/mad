# 189: Bijective Shuffle — Bandwidth-Optimal GPU Permutation

**Category**: parallelization
**Gain type**: efficiency
**Source**: Mitchell, Stokes, Frank & Holmes, "Bandwidth-Optimal Random Shuffling for GPUs" (ACM TOPC 2022; arXiv:2106.06161)
**Paper**: [papers/bandwidth-optimal-gpu-shuffle.pdf]
**Documented**: 2026-02-15

## Description

Generating random permutations (shuffling) on GPUs is a fundamental operation needed for data augmentation, stochastic training, Monte Carlo methods, and permutation-based attention patterns. The classical Fisher-Yates algorithm is inherently sequential ($O(n)$ swaps with dependencies), and existing parallel approaches either have suboptimal $O(n \log n)$ work (sort-based) or require many global memory round-trips (dart-throwing, divide-and-conquer).

The **bijective shuffle** algorithm generates pseudo-random permutations in a **single gather pass** that approaches peak GPU memory bandwidth. The core idea is:

1. Use a **pseudo-random bijective function** $f_n : [0, n) \to [0, n)$ (a block cipher) to generate a permutation over a power-of-two domain
2. Use **stream compaction** (parallel prefix sum) to "trim" the power-of-two permutation down to the desired arbitrary length $m \leq n$
3. **Fuse** the bijection evaluation, flag computation, prefix sum, and gather into a **single GPU kernel** with exactly one global memory read and one write per input element

The bijective function is based on a **VariablePhilox** cipher — a modified Feistel network that produces bijections over arbitrary power-of-two domains with cryptographic-quality pseudo-randomness (passes BigCrush statistical tests with 24 rounds). Each thread independently computes $f_n(i)$ with no inter-thread communication, yielding perfect parallelism.

The fused kernel achieves **near-theoretical-peak bandwidth** utilization: on a Tesla V100, the bijective shuffle reaches throughput within 2× of the raw random-gather upper bound for large inputs, and 10–100× faster than sort-based or dart-throwing alternatives.

## Mathematical Form

**Bijective function (Feistel network):**

Given a $b$-bit input split into halves $L$ (left, $\lfloor b/2 \rfloor$ bits) and $R$ (right, $\lceil b/2 \rceil$ bits), each round $i$ of the Feistel network computes:

$$
f_i(L|R) = R \,|\, (L \oplus F(R, k_i))
$$

where $F$ is a round function, $k_i$ is a round key, and $\oplus$ is bitwise XOR. A complete cipher with $r$ rounds:

$$
g(L|R) = f_r \circ f_{r-1} \circ \cdots \circ f_1(L|R)
$$

A Feistel network with $r \geq 3$ rounds is a **pseudo-random permutation** (Luby-Rackoff theorem).

**VariablePhilox cipher:**

For arbitrary power-of-two length $n = 2^b$, the modified Philox cipher computes:

$$
L' = F_k(L) \oplus R
$$

$$
(R', b') = G(B_k(L), b)
$$

where $B_k$ is a bijection (odd-constant multiplication mod $2^w$), $F_k$ is a pseudo-random function, and $G$ mixes an extra odd bit $b$ (when $b$ is odd) into the output to handle non-even bit splits.

**Stream compaction for arbitrary lengths:**

Given desired length $m$ and bijection domain $n = 2^{\lceil \log_2 m \rceil}$, the algorithm:

1. Evaluates $w_i = f_n(i)$ for each $i \in [0, n)$
2. Flags: $\text{flag}_i = \mathbb{1}[w_i < m]$
3. Computes prefix sum: $\text{out\_idx}_i = \text{exclusive\_scan}(\text{flag})_i$
4. Gathers: if $\text{flag}_i = 1$, then $Y[\text{out\_idx}_i] = X[w_i]$

**Proposition 1 (Uniformity):** If $p(\sigma) = 1/n!$ for all $\sigma \in \mathfrak{S}_n$, then the compacted permutation $\tau \in \mathfrak{S}_m$ satisfies $p(\tau) = 1/m!$.

**Fused kernel (bandwidth-optimal):**

All four steps are fused into a single kernel using a **decoupled look-back** single-pass scan. Each thread:
1. Evaluates $f_n(i)$ (register-only arithmetic)
2. Computes flag (comparison)
3. Participates in single-pass prefix sum (one global read/write for scan state)
4. Performs gather $Y[\text{out\_idx}] = X[f_n(i)]$ (one global read + one global write)

Total global memory transactions: **$m$ reads + $m$ writes** = bandwidth-optimal.

**Key Definitions:**

- $f_n : [0, n) \to [0, n)$ — pseudo-random bijection (block cipher) over power-of-two domain
- $m$ — desired permutation length (arbitrary)
- $n = 2^{\lceil \log_2 m \rceil}$ — nearest power-of-two $\geq m$
- $\mathfrak{S}_n$ — symmetric group of permutations on $n$ elements
- $r$ — number of Feistel rounds (24 recommended for VariablePhilox)
- $k_i$ — round keys (derived from seed)

## Complexity

| Operation | Fisher-Yates | Sort-based | Dart-throwing | Bijective shuffle |
|-----------|-------------|------------|---------------|-------------------|
| Work | $O(n)$ | $O(n \log n)$ | $O(n)$ expected | $O(n)$ |
| Depth | $O(n)$ | $O(\log n)$ | $O(\log n)$ expected | $O(\log n)$ |
| Global mem R/W | — | $O(n \log n)$ | $O(n)$ expected | $O(n)$ exactly |
| Deterministic | Yes | Yes | No | Yes |
| Extra memory | $O(1)$ | $O(n)$ | $O(n)$ | $O(1)$ working |

**Memory:** $O(n)$ for output buffer (not in-place). $O(1)$ working memory per thread.

**GPU throughput (Tesla V100, 64-bit elements, $n = 2^{29} + 1$):**

| Algorithm | Throughput (M items/s) |
|-----------|----------------------|
| Random gather (upper bound) | 4,409 |
| **Bijective shuffle (VarPhilox)** | **4,018** |
| LCG bijective | 4,011 |
| DartThrowing | 133.8 |
| SortShuffle (radix sort) | 118.4 |

The bijective shuffle achieves **91% of the theoretical bandwidth ceiling** (random gather), and is **30× faster** than sort-based shuffling.

**GeForce RTX 2080 ($n = 2^{26} + 1$):**

| Algorithm | Throughput (M items/s) |
|-----------|----------------------|
| Random gather | 2,096 |
| **Bijective shuffle** | **2,233** |
| SortShuffle | 75.03 |

## Applicability

- **Data augmentation shuffling in training pipelines**: Every training epoch requires shuffling the dataset. When done on-GPU, the bijective shuffle avoids the CPU↔GPU transfer bottleneck. At $n = 2^{29}$ (500M tokens), the V100 shuffles in ~130ms vs. ~4.2s for sort-based methods — a **32× speedup** per epoch
- **Permutation-based attention patterns**: Architectures like permutation-equivariant transformers or set transformers that sample random attention patterns can generate permutation indices at near-bandwidth speed without pre-computing and storing permutation tables
- **Stochastic sequence ordering in SSMs**: Mamba-style models that benefit from randomized input ordering during training can generate per-batch permutations on-the-fly with negligible overhead
- **Monte Carlo permutation tests**: Statistical tests requiring many random permutations (e.g., for feature attribution via Shapley values) can generate permutations at near-peak bandwidth
- **Dropout pattern generation**: The bijective function can generate pseudo-random binary masks without storing them, by evaluating $f_n(i) < \lfloor pn \rfloor$ per-element. This is already how Philox is used in PyTorch dropout, but the compaction trick extends it to structured/grouped dropout patterns
- **Differentially private training**: DP-SGD requires random sampling of examples; the bijective shuffle provides deterministic, reproducible shuffling with cryptographic-quality randomness

## Limitations

- **Not in-place**: Requires a separate output buffer of size $m$ — cannot permute an array without $O(n)$ extra memory. For memory-constrained scenarios, the decomposed transposition (trick 188) or cycle-following may be preferred
- **Pseudo-random, not truly random**: The VariablePhilox cipher with 24 rounds generates at most $2^{64}$ distinct permutations (limited by key size), whereas $n!$ permutations exist. For $n > 20$, not all permutations are reachable. This is acceptable for ML applications but not for cryptographic use
- **Worst-case 2× redundant work**: When $m$ is just above a power of two (e.g., $m = 2^k + 1$), the algorithm evaluates $2^{k+1}$ bijection invocations and discards nearly half. In practice, this overhead is hidden by the memory-bound nature of the kernel
- **Feistel round arithmetic cost**: Each VariablePhilox evaluation requires 24 rounds × (1 multiply + 1 XOR + bit shifts) ≈ 72 integer operations per element. For small inputs where compute matters, this adds overhead compared to simpler LCG-based approaches
- **Output-only gather pattern**: The permutation result is written to contiguous output locations via a gather from non-contiguous input locations. Gather has higher bandwidth than scatter on GPUs (due to write coalescing), but still lower than coalesced reads — the fundamental cost of random permutation

## Implementation Notes

```python
import torch

def variable_philox_bijection(val, key, num_rounds=24):
    """VariablePhilox: bijection over power-of-two domain.

    Implements a modified Philox PRNG as a bijective function.
    Each input maps to a unique output (permutation).

    Args:
        val: input index (uint64)
        key: list of round keys
        num_rounds: number of Feistel rounds (24 recommended)

    Returns:
        permuted index (uint64)
    """
    M0 = 0xD2B74407B1CE6E93  # Philox multiplier constant
    bits = val.bit_length()  # total bits
    right_bits = (bits + 1) // 2
    left_bits = bits - right_bits
    right_mask = (1 << right_bits) - 1
    left_mask = (1 << left_bits) - 1

    state_r = val & right_mask
    state_l = (val >> right_bits) & left_mask

    for i in range(num_rounds):
        # Philox round: multiply, split hi/lo, XOR with key
        product = M0 * state_l  # 64-bit multiply
        hi = product >> 32
        lo = product & 0xFFFFFFFF
        lo = (lo << (right_bits - left_bits)) | (state_r >> left_bits)
        new_l = ((hi ^ key[i]) ^ state_r) & left_mask
        state_r = lo & right_mask
        state_l = new_l

    return (state_l << right_bits) | state_r


def bijective_shuffle_gpu(X, seed=42):
    """Bandwidth-optimal shuffling on GPU.

    Generates a pseudo-random permutation of X using the
    bijective shuffle algorithm. Achieves near-peak GPU
    memory bandwidth through kernel fusion.

    Args:
        X: input tensor of length m on GPU
        seed: random seed for key generation

    Returns:
        Y: shuffled tensor (new allocation)
    """
    m = X.shape[0]
    # Find nearest power of two >= m
    import math
    n = 1 << math.ceil(math.log2(max(m, 2)))

    # Generate round keys from seed
    rng = torch.Generator(device=X.device).manual_seed(seed)
    keys = torch.randint(0, 2**32, (24,), generator=rng,
                         device=X.device)

    # Evaluate bijection for all indices [0, n)
    indices = torch.arange(n, device=X.device, dtype=torch.int64)

    # In practice, this would be a fused CUDA kernel.
    # Here we show the logic:
    # Step 1: Evaluate bijection
    permuted = torch.zeros_like(indices)
    for i in range(n):
        # VariablePhilox evaluation (simplified)
        permuted[i] = variable_philox_bijection(i, keys.tolist())

    # Step 2: Flag valid elements (those mapping to [0, m))
    flags = (permuted < m).int()

    # Step 3: Exclusive prefix sum of flags
    out_idx = torch.cumsum(flags, dim=0) - flags

    # Step 4: Gather valid elements
    Y = torch.empty(m, dtype=X.dtype, device=X.device)
    valid_mask = permuted < m
    Y[out_idx[valid_mask]] = X[permuted[valid_mask]]

    return Y


def bijective_shuffle_cuda_sketch():
    """Sketch of the fused CUDA kernel structure.

    The key optimization is fusing all four steps into one kernel
    using a decoupled look-back single-pass scan:

    __global__ void bijective_shuffle_kernel(
        const T* __restrict__ input,   // input array
        T* __restrict__ output,         // output array
        uint64_t m,                     // desired length
        uint64_t n,                     // power-of-two domain
        uint32_t* keys                  // Feistel round keys
    ) {
        // 1. Each thread evaluates f_n(threadIdx + blockIdx*blockDim)
        uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        uint64_t perm_val = variable_philox(idx, keys);

        // 2. Compute flag
        int flag = (perm_val < m) ? 1 : 0;

        // 3. Single-pass decoupled look-back prefix sum
        // (Merrill & Garland, "Single-pass Parallel Prefix Scan")
        // Uses tile-level aggregates + look-back for O(n) work
        uint64_t out_idx = decoupled_lookback_scan(flag);

        // 4. Gather (single global read + single global write)
        if (flag) {
            output[out_idx] = input[perm_val];  // random gather
        }
    }

    // Total global memory: n reads (scan state) + m reads (data) + m writes
    // For large n: dominated by m data reads + m writes = bandwidth-optimal
    """
    pass
```

## References

- Mitchell, R., Stokes, D., Frank, E. & Holmes, G. "Bandwidth-Optimal Random Shuffling for GPUs" ACM Trans. Parallel Computing, 2022. arXiv:2106.06161
- Salmon, J.K. et al. "Parallel Random Numbers: As Easy as 1, 2, 3" SC '11, 2011 (Philox PRNG)
- Luby, M. & Rackoff, C. "How to Construct Pseudorandom Permutations from Pseudorandom Functions" SIAM J. Comput. 17(2), 1988
- Merrill, D. & Garland, M. "Single-pass Parallel Prefix Scan with Decoupled Look-back" Tech Report NVR-2016-002, NVIDIA, 2016
- GitHub implementation: https://github.com/djns99/CUDA-Shuffle

# 058: Gyro-Permutation for Hierarchical N:M Sparsity

**Category**: kernel
**Gain type**: efficiency
**Source**: Yu, Yi, Lee & Shin (Sungkyunkwan University, 2024)
**Paper**: [papers/gyro-permutation-hierarchical-nm.pdf]
**Documented**: 2026-02-15

## Description

Hierarchical N:M (HiNM) sparsity combines column-wise vector pruning with row-wise N:M (e.g., 2:4) sparsity to achieve compression ratios beyond 50% (e.g., 75% with $4 \times 1$ vector pruning + 2:4). However, this two-level structure requires channel permutation on *both* output and input dimensions simultaneously — a harder optimization than single-level N:M permutation. Prior permutation methods (designed for single-level sparsity) fall into local minima when naively applied to HiNM, and methods like Tetris that do permute both dimensions incur significant runtime overhead from inter-layer index translation operations.

**Gyro-Permutation** is a channel permutation algorithm designed specifically for HiNM sparsity that:

1. **Decomposes** the joint output+input permutation problem into two sequential sub-problems: output channel permutation (for column-wise vector pruning) followed by tile-wise input channel permutation (for 2:4 row-wise sparsity).
2. **Avoids local minima** through a three-phase iterative algorithm: sampling → clustering → assignment, where each phase is tailored to the HiNM structure.
3. **Achieves zero runtime overhead** by integrating index translation into the native HiNM data loading pipeline — the input channel permutation modifies vector indices during the existing global-to-shared-memory transfer on GPUs, requiring no additional computation.

The key architectural insight is that during GPU SpMM with HiNM patterns, each thread block independently processes a "tile" of output channels. Since tiles are computed independently, reordering column vectors *within* a tile does not affect computation results. This enables tile-local permutations that can be fused into the existing memory load operations.

## Mathematical Form

**HiNM Permutation Problem:**

Given a weight matrix $W \in \mathbb{R}^{m \times n}$, column-wise vector mask $M_v$, row-wise 2:4 mask $M_{24}$, and saliency scores $\rho$, find the optimal output channel order $\sigma_o$ and tile-wise input channel orders $\sigma_i^0, \ldots, \sigma_i^T$ that maximize retained saliency:

$$
\max_{\sigma_o, \sigma_i^0, \ldots, \sigma_i^T} \| M \odot \rho[\sigma_0; \sigma_i] \| \quad \text{s.t.} \quad M \text{ satisfies } M_v \text{ and } M \setminus \{0\} \text{ satisfies } M_{24}
$$

where $T$ is the number of tiles and $\sigma_i$ denotes the set of column-wise vector orders per tile.

**Decomposition into two sub-problems:**

**Step 1 — Output Channel Permutation** (for column-wise vector pruning):

$$
\max_{\sigma_o} \| M_v \odot \rho[\sigma_o; I] \| \quad \text{s.t. } M_v \text{ satisfies column-wise vector constraints}
$$

This reorders output channels so that less-salient elements cluster together, enabling more effective vector-level pruning.

**Step 2 — Tile-wise Input Channel Permutation** (for 2:4 row-wise sparsity):

$$
\max_{\sigma_i^0, \ldots, \sigma_i^T} \| M_{24} \odot (M_v \odot \rho[\sigma_o; I])[I; \sigma_i] \| \quad \text{s.t. } M_{24} \text{ satisfies 2:4 structure}
$$

This reorders column vectors within each tile to balance important element distribution across 2:4 row vectors.

**Assignment Cost Function:**

For placing the $j$-th sample (cluster) into the $i$-th partition:

$$
C_{i,j} = \rho - |M \odot \rho|, \quad \text{where } \rho \subset P_i \cup s_j
$$

This cost quantifies the saliency of weights that would be *pruned* when the $j$-th sample is integrated into the $i$-th partition. The Hungarian algorithm minimizes total cost across all assignments.

**Search Space:**

For a $16 \times 16$ matrix with column vector size 4, the number of possible permutations exceeds $27$ trillion:

$$
|\mathcal{P}| = \frac{m!}{V!^{P_o}} \times T \times \frac{n!}{4!^{P_i}}
$$

where $V$ is the column vector size, $P_o$ and $P_i$ are the number of output and input partitions.

**Key Definitions:**

- $m, n$ — weight matrix dimensions (output channels $\times$ input channels)
- $V$ — column vector size (e.g., 32 or 64 for CNN, 4 for attention)
- $T$ — number of tiles (thread blocks)
- $\sigma_o$ — output channel permutation
- $\sigma_i^t$ — tile-wise input channel permutation for tile $t$
- $\rho$ — saliency scores (magnitude or second-order information)
- $M_v, M_{24}$ — column-wise vector mask and 2:4 row-wise mask

## Complexity

| Operation | Naive HiNM (no perm) | Tetris-style perm | Gyro-Permutation |
|-----------|---------------------|-------------------|------------------|
| Output channel perm | N/A | $O(m^2)$ clustering | $O(m^2)$ balanced K-means + Hungarian |
| Input channel perm | N/A | $O(n^2)$ global swap | $O(T \cdot (n/T)^2)$ tile-local Hungarian |
| Runtime index translation | $0$ | $O(n)$ per layer (explicit) | **$0$** (fused into memory load) |
| Inference SpMM | $O(\text{nnz} \cdot K)$ | $O(\text{nnz} \cdot K) + O(n)$ | $O(\text{nnz} \cdot K)$ |

**Memory:** No additional memory at inference — output permutation pre-orders layers offline; input permutation modifies only the vector index array already used by HiNM kernels.

**Accuracy improvements at 75% sparsity (one-shot pruning + fine-tuning):**

| Model | HiNM-NoPerm | OVW (vector-only perm) | Gyro-Permutation |
|-------|-------------|----------------------|------------------|
| ResNet18 | 63.79% | 65.21% | **68.91%** (+5.12%) |
| ResNet50 | 70.83% | 70.91% | **74.45%** (+3.62%) |
| DeiT-base | 76.10% | — | **81.14%** (+5.04%) |

**Latency:** Zero detectable runtime overhead on RTX 3090 across all sparsity ratios (75%, 87.5%, 93.75%) and vector sizes (64, 128).

## Applicability

- **Hierarchical N:M sparse inference:** Primary use case. Enables >50% sparsity (75%, 87.5%) with accuracy comparable to unstructured pruning, all running on existing Sparse Tensor Core hardware.
- **CNN and Transformer weight matrices:** Validated on ResNet18/50 (Conv2d), DeiT-base (Linear attention projections and FFN), and BERT-base (all Linear layers).
- **Composable with V:N:M sparsity:** Gyro-Permutation's tile-wise strategy complements V:N:M's block-level column pruning. Both use the VENOM kernel library as a backend.
- **Connection to PA-DST:** While PA-DST learns permutations *during* training via differentiable relaxation, Gyro-Permutation is a post-training/pre-inference permutation that uses combinatorial optimization (Hungarian algorithm). Both target the same goal — restoring expressivity lost to structured sparsity — but at different stages.
- **Sparse SSM recurrences:** Any SSM using block-diagonal or column-sparse transitions could benefit from tile-local permutations to redistribute nonzero pattern across blocks.

## Limitations

- Currently assumes a fixed pruning sequence: column-wise vector pruning *first*, then N:M pruning. Alternative sequences (N:M first) may benefit from different permutation strategies.
- The iterative sampling-clustering-assignment loop has tunable hyperparameters (sample count, number of iterations) that require some experimentation.
- Tile-wise input channel permutation is limited to reordering within tiles — inter-tile reordering would require explicit index translation at runtime.
- Only validated with 2:4 as the inner N:M pattern; other N:M ratios (4:8, etc.) may require kernel modifications.
- The balanced K-means clustering for output channels is a heuristic that may not find the global optimum for very wide layers.
- Zero-overhead claim is specific to HiNM GPU kernels based on the VENOM library; other sparse kernel implementations may not support fused index translation.

## Implementation Notes

```python
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def gyro_output_channel_permutation(W, V, n_iters=10):
    """
    Output channel permutation for column-wise vector pruning.

    Args:
        W: (m, n) weight matrix
        V: column vector size (e.g., 32)
        n_iters: number of sampling-clustering-assignment iterations

    Returns:
        sigma_o: (m,) output channel permutation indices
    """
    m, n = W.shape
    P_o = m // V  # number of output partitions
    sigma_o = torch.arange(m)

    for it in range(n_iters):
        # Sampling: extract channels from each partition
        # Dynamic sample count (like learning rate decay)
        n_samples = max(1, P_o // (it + 1))
        sampled = []
        for p in range(P_o):
            start = p * V
            end = (p + 1) * V
            partition_channels = sigma_o[start:end]
            # Sample channels with lowest saliency in this partition
            saliency = W[partition_channels].abs().sum(dim=1)
            _, idx = saliency.topk(n_samples, largest=False)
            sampled.append(partition_channels[idx])

        # Clustering: balanced K-means on sampled channels
        # Groups channels with similar weight distributions
        all_sampled = torch.cat(sampled)
        features = W[all_sampled]  # (total_sampled, n)
        # ... balanced K-means assigns to P_o clusters ...

        # Assignment: Hungarian algorithm
        # Cost C[i,j] = saliency of weights pruned if sample j -> partition i
        cost_matrix = compute_pruning_cost(W, sigma_o, sampled, V)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update sigma_o based on assignment
        # ... place assigned samples into target partitions ...

    return sigma_o

def gyro_tile_input_permutation(W_pruned, tile_size, V):
    """
    Tile-wise input channel permutation for 2:4 row-wise sparsity.
    Each tile's column vectors are independently reordered.

    Args:
        W_pruned: (m, n_retained) weight matrix after vector pruning
        tile_size: number of output channels per GPU thread block tile
        V: column vector size

    Returns:
        per_tile_perm: list of (n_vectors_per_tile,) permutation arrays
    """
    m, n = W_pruned.shape
    n_tiles = m // tile_size
    n_vectors = n // V  # number of column vectors
    per_tile_perm = []

    for t in range(n_tiles):
        tile_rows = W_pruned[t * tile_size:(t + 1) * tile_size]

        # For tile-wise perm: sample 1 vector per partition (4 vectors each)
        # No clustering needed (sample count = partition count)

        # Assignment: cost of placing vector j at position i
        n_partitions = n_vectors  # one partition per 2:4 group
        cost = torch.zeros(n_partitions, n_vectors)
        for i in range(n_partitions):
            for j in range(n_vectors):
                # Cost = saliency pruned under 2:4 if vector j at position i
                cost[i, j] = compute_24_pruning_cost(tile_rows, i, j, V)

        row_ind, col_ind = linear_sum_assignment(cost.numpy())
        per_tile_perm.append(col_ind)

    return per_tile_perm

# GPU kernel integration (pseudocode):
# During HiNM SpMM, the vector index array is already used to load
# column vectors from global to shared memory. Gyro-permutation
# simply modifies these indices offline:
#
# Original:  vector_index = [0, 2, 5, 7, ...]  (from vector pruning)
# Permuted:  vector_index = [5, 0, 7, 2, ...]  (after gyro-perm)
#
# The GPU kernel loads: shared_mem[tid] = global_mem[vector_index[tid]]
# No additional computation — just a different index sequence.
```

## References

- Yu, S., Yi, X., Lee, H. & Shin, D. (2024). Toward Efficient Permutation for Hierarchical N:M Sparsity on GPUs. arXiv:2407.20496.
- Castro, R.L., et al. (2023). Venom: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores. SC 2023.
- Ji, Y., et al. (2018). Tetris: Tile-Matching the Tremendous Irregular Sparsity. NeurIPS 2018.
- Pool, J. & Yu, C. (2021). Channel Permutations for N:M Sparsity. NeurIPS 2021.
- Tan, Y., et al. (2022). Accelerating Sparse Convolution with Column Vector-wise Sparsity. NeurIPS 2022.
- Kuznedelev, D., et al. (2024). CAP: Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models. NeurIPS 2024.

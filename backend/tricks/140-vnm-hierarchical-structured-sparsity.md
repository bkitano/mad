# 140: V:N:M Hierarchical Structured Sparsity

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhao, Yuan, Su et al. "Beyond 2:4: Exploring V:N:M Sparsity for Efficient Transformer Inference on GPUs" (Tsinghua / Beijing Jiaotong, 2025)
**Paper**: [papers/vnm-sparsity-beyond-2-4.pdf]
**Documented**: 2025-07-14

## Description

V:N:M sparsity is a hierarchical extension of standard 2:4 structured sparsity that enables **flexible sparsity ratios above 50%** while still leveraging Sparse Tensor Cores for hardware acceleration. The key limitation of 2:4 sparsity is threefold: (1) actual speedups are low (≤1.3x in practice despite 2x theoretical), (2) only 50% sparsity is accelerable (4:8, 8:16 etc. get no GPU speedup), and (3) models with high redundancy cannot exploit higher sparsity levels. V:N:M addresses all three.

**Two-level pruning structure:**
1. **Column pruning** (coarse): Divide the weight matrix into $V \times M$ blocks. Within each block, prune $(M-4)$ columns based on column-wise importance, retaining exactly 4 columns.
2. **2:4 sparsity** (fine): Apply standard 2:4 structured sparsity within the retained 4 columns.

The result is that each $V \times M$ block has only $V \times 2$ nonzero values (2 per row out of the original $M$), giving a sparsity ratio of $1 - 2/M$. For example, $V$:2:8 achieves 75% sparsity, $V$:2:16 achieves 87.5% sparsity. Crucially, because the inner pattern is always 2:4, **any GPU that supports 2:4 sparsity can accelerate V:N:M** — no new hardware is needed.

The paper introduces three key techniques: (1) heuristic V and M selection using mask diversity to find Pareto-optimal configurations, (2) V:N:M-specific channel permutation to maximize retained weight importance, and (3) three-staged LoRA training for efficient LLM fine-tuning with V:N:M sparsity.

## Mathematical Form

**V:N:M Pruning Process ($S_{V:N:M}$):**

Given a weight matrix $\mathbf{W} \in \mathbb{R}^{R \times C}$, the pruning has three steps:

**Step 1 — Importance scoring:**

For each weight $\mathbf{W}_{ij}$, compute an importance score. Two criteria are used:

*Absolute value (ABS):* $\text{score}_{ij} = |\mathbf{W}_{ij}|$

*Relative Importance and Activation (RIA):*

$$
RIA_{ij} = \left(\frac{|\mathbf{W}_{ij}|}{\sum_k |\mathbf{W}_{kj}|} + \frac{|\mathbf{W}_{ij}|}{\sum_k |\mathbf{W}_{ik}|}\right) \times (\|\mathbf{X}_i\|_2)^a
$$

where $\|\mathbf{X}_i\|_2$ is the L2 norm of the $i$-th input channel's activations and $a$ controls activation influence.

**Step 2 — Column pruning:**

Partition $\mathbf{W}$ into $V \times M$ blocks. For each block, compute the L1 norm of importance scores per column. Prune the $(M-4)$ columns with smallest L1 norms, retaining exactly 4 columns.

**Step 3 — 2:4 sparsity:**

Within each row's 4 retained columns, zero the 2 weights with smallest importance scores.

**Effective sparsity ratio:**

$$
\text{Sparsity} = 1 - \frac{N}{M} = 1 - \frac{2}{M}
$$

since $N=2$ is fixed and $M$ varies (5, 6, 7, 8, 12, 16, 24, ...).

**Mask Diversity (V and M Selection):**

The mask diversity of a V:N:M sparse transformer is:

$$
MD_f = \prod_l MD^l_{V:N:M}, \quad MD^l_{V:N:M} = \left[\binom{M}{4}^4 \cdot \binom{4}{2}^V\right]^{\frac{m \cdot n}{V \cdot M}} = K(V,M)^{mn}
$$

where $m, n$ are layer dimensions. The key insight: the relative ordering of $MD_f$ for different $(V, M)$ pairs depends **only** on $K(V, M)$, not on layer shapes. This enables fast Pareto-front computation.

**Heuristic selection**: Given a target speedup $s$, enumerate all $(V, M)$ combinations where $V \in \{2^k | k \in \mathbb{N}^+, k \geq 4\}$ and $M \in \mathbb{N}^+, M \geq 5$, filter by speedup constraint, then select the pair maximizing $K(V, M)$.

**V:N:M-Specific Channel Permutation:**

Both input and output permutations are applied:

$$
\mathbf{Y} = \mathbf{W}\mathbf{X} = \mathbf{P}_o^\top \mathbf{P}_o \mathbf{W} \mathbf{P}_i \mathbf{P}_i^\top \mathbf{X} = \mathbf{P}_o^\top \mathbf{W}_p \mathbf{P}_i^\top \mathbf{X}
$$

where $\mathbf{P}_o, \mathbf{P}_i$ are output and input channel permutation matrices, and $\mathbf{W}_p = \mathbf{P}_o \mathbf{W} \mathbf{P}_i$ is the permuted weight. The objective is:

$$
\arg\max_{\mathbf{P}_o, \mathbf{P}_i} \sum_{i,j} RIA_{ij}(S_{V:N:M}(\mathbf{W}_p))
$$

Solved via alternating optimization (2 iterations suffice):

$$
\mathbf{P}_i^{k+1} = \arg\max_{\mathbf{P}_i} \sum_{i,j} RIA_{ij}(S_{V:N:M}(\mathbf{P}_o^k \mathbf{W} \mathbf{P}_i))
$$

$$
\mathbf{P}_o^{k+1} = \arg\max_{\mathbf{P}_o} \sum_{i,j} RIA_{ij}(S_{V:N:M}(\mathbf{P}_o \mathbf{W} \mathbf{P}_i^k))
$$

Each subproblem is a linear sum assignment, solvable by the Hungarian algorithm in $O(n^3)$. At inference, $\mathbf{P}_o^\top$ and $\mathbf{P}_i^\top$ fuse into adjacent LayerNorm or preceding linear layers at zero cost.

**Three-Staged LoRA Training:**

For LLM fine-tuning with V:N:M sparsity, the sparse weight is:

$$
\mathbf{W}' = (\mathbf{W} + \mathbf{B}\mathbf{A}) \odot \mathbf{M}
$$

Three stages:
1. **Dense LoRA**: Standard LoRA fine-tuning with all-ones mask $\mathbf{M} = \mathbf{1}$ (warmup)
2. **Sparse LoRA with dynamic masks**: Masks $\mathbf{M}$ updated every 5 epochs; $\mathbf{B}, \mathbf{A}$ merged into $\mathbf{W}$ before re-pruning
3. **Sparse LoRA with fixed masks**: Masks frozen, LoRA continues fine-tuning to convergence

Stages 1+2 use ≤10% of total iterations; stage 3 uses the remaining ≥90%.

**Key Definitions:**

- $V$ — number of rows in each pruning block (vector dimension)
- $N = 2$ — nonzeros retained per row in the inner 2:4 pattern (fixed)
- $M$ — number of columns in each pruning block
- $K(V, M)$ — mask diversity function, determines accuracy-speedup tradeoff
- $\mathbf{P}_o, \mathbf{P}_i$ — output/input channel permutation matrices

## Complexity

| Configuration | Sparsity | Params Reduction | FLOPs Reduction | Practical Speedup |
|---------------|----------|-----------------|-----------------|-------------------|
| 2:4 | 50% | ~50% | ~50% | 1.1-1.3x |
| 64:2:5 (V:N:M) | 60% | ~58% | ~54% | 1.3-1.5x |
| 64:2:6 (V:N:M) | 67% | ~65% | ~60% | 1.5-1.7x |
| 64:2:8 (V:N:M) | 75% | ~74% | ~72% | **1.7x** |
| 128:2:6 (V:N:M) | 67% | ~65% | ~60% | 1.5-1.7x |

**Acceleration mechanism:** V:N:M-sparse MMs have fewer computations than 2:4-sparse MMs (at higher sparsity), so they achieve greater speedups on the **same** Sparse Tensor Core hardware. The kernel selects the 4 retained columns' data from input $\mathbf{B}$, then executes a standard 2:4-sparse MM.

**Compressed storage:**
- Nonzero values: $\mathbf{A}_n$ (same shape as 2:4 compressed format, but over fewer columns)
- Column indices: $\mathbf{A}_{i1}$ (which 4 of $M$ columns are retained per block)
- Position metadata: $\mathbf{A}_{i2}$ (2-bit indices for 2:4 positions within retained columns)

**Channel permutation overhead:** Hungarian algorithm: $O(n^3)$ per subproblem, 2 alternating iterations. Negligible compared to training. At inference: **zero** overhead (permutations fused into adjacent layers).

## Applicability

- **Vision Transformers (DeiT, Swin)**: DeiT-base at 64:2:8 achieves **lossless** accuracy (81.76% vs. 81.84% dense) with 73.8% parameter reduction and 1.7x speedup. DeiT-small at 64:2:5 is also lossless (79.65% vs. 79.85%)
- **Large Language Models (LLaMA)**: LLaMA2-7B at 64:2:5 with three-staged LoRA achieves PPL 9.97 (vs. 5.12 dense, 10.52 for 2:4 RIA) with **1.49x** speedup vs. 1.26x for 2:4
- **Object Detection (H-DETR)**: Swin-Tiny backbone at 32:2:5 achieves 47.8% mAP, outperforming dense equivalent trained for same epochs
- **Higher speedup than 2:4**: For models with significant redundancy, V:N:M consistently provides larger practical speedups than 2:4, with comparable or better accuracy after training
- **No hardware changes**: Runs on existing Ampere/Hopper GPUs via Spatha sparse library (Castro et al., 2023) — the inner 2:4 pattern ensures compatibility with existing Sparse Tensor Cores
- **Composable**: V:N:M can be combined with channel permutation, LoRA fine-tuning, knowledge distillation, and other compression techniques

## Limitations

- Requires training budget to restore accuracy — post-training V:N:M pruning alone incurs up to 5.87% accuracy drop; training reduces this to <1%
- Channel permutation requires careful handling — V:N:M affects both input and output channels (unlike 2:4, which only affects input channels)
- V and M selection requires evaluating speedup on target hardware (GPU profiling step)
- Not demonstrated on very large LLMs (>7B parameters) — larger models likely tolerate even higher sparsity but this is untested
- Dynamic mask training with V:N:M can cause gradient flow instability — the three-staged LoRA mitigates this but adds training complexity
- The column pruning step is coarser than element-wise pruning, potentially discarding individually important weights if their column has low aggregate importance
- $V$ must be a power of 2 (≥16) for GPU acceleration affinity; $M \geq 5$; $N$ is fixed at 2

## Implementation Notes

```python
import torch

def vnm_prune(W: torch.Tensor, V: int, M: int,
              X: torch.Tensor = None, a: float = 0.5) -> torch.Tensor:
    """Apply V:N:M structured sparsity to weight matrix.

    Args:
        W: (R, C) weight matrix
        V: block row dimension (power of 2, >= 16)
        M: block column dimension (>= 5)
        X: (C, ...) input activations for RIA scoring (optional)
        a: activation influence exponent for RIA

    Returns:
        mask: (R, C) binary mask with V:N:M sparsity pattern
    """
    R, C = W.shape
    N = 2  # fixed for 2:4 inner pattern

    # Step 1: Compute importance scores
    if X is not None:
        # RIA: relative importance + activation
        col_norm = W.abs().sum(dim=0, keepdim=True)  # (1, C)
        row_norm = W.abs().sum(dim=1, keepdim=True)  # (R, 1)
        act_norm = X.norm(dim=1) if X.dim() > 1 else X.abs()  # (C,)
        scores = (W.abs() / col_norm + W.abs() / row_norm) * (act_norm ** a)
    else:
        # ABS: simple magnitude
        scores = W.abs()

    mask = torch.ones_like(W, dtype=torch.bool)

    # Step 2: Column pruning within V x M blocks
    # Pad if needed so C is divisible by M and R by V
    for br in range(0, R, V):
        for bc in range(0, C, M):
            block_scores = scores[br:br+V, bc:bc+M]
            # L1 norm per column
            col_importance = block_scores.sum(dim=0)  # (M,)
            # Keep top 4 columns
            _, keep_cols = col_importance.topk(4)
            # Zero out pruned columns
            prune_cols = torch.ones(M, dtype=torch.bool)
            prune_cols[keep_cols] = False
            mask[br:br+V, bc + prune_cols.nonzero().squeeze()] = False

    # Step 3: 2:4 sparsity on retained columns (per row, groups of 4)
    # For each row, within each block's 4 retained columns,
    # zero the 2 smallest
    for br in range(0, R, V):
        for bc in range(0, C, M):
            block_scores = scores[br:br+V, bc:bc+M]
            col_importance = block_scores.sum(dim=0)
            _, keep_cols = col_importance.topk(4)
            keep_cols_sorted, _ = keep_cols.sort()

            for row in range(V):
                row_scores = scores[br+row, bc + keep_cols_sorted]
                _, bottom2 = row_scores.topk(2, largest=False)
                for idx in bottom2:
                    mask[br+row, bc + keep_cols_sorted[idx]] = False

    return mask


def select_vm(target_speedup: float, layer_shapes: list) -> tuple:
    """Heuristic V and M selection via mask diversity.

    Args:
        target_speedup: minimum required speedup (e.g., 1.5)
        layer_shapes: list of (R, C) tuples for each layer

    Returns:
        (V, M): optimal configuration
    """
    from math import comb, log

    best_K = -float('inf')
    best_vm = None

    for V in [16, 32, 64, 128]:
        for M in range(5, 33):
            sparsity = 1 - 2 / M
            # Estimate speedup (rough heuristic)
            est_speedup = 1.0 / (1.0 - sparsity * 0.8)  # approximate

            if est_speedup >= target_speedup:
                # Compute mask diversity K(V, M)
                K = log(comb(M, 4)) * 4 + log(comb(4, 2)) * V
                K = K / (V * M)  # normalize
                if K > best_K:
                    best_K = K
                    best_vm = (V, M)

    return best_vm


# Deployment: convert to compressed format for Sparse Tensor Cores
# 1. Zero-pad W so dims are divisible by V and M
# 2. Apply V:N:M pruning mask
# 3. Extract retained column indices (A_i1)
# 4. Convert retained 4 columns to 2:4 compressed format (A_n, A_i2)
# 5. Use Spatha library for V:N:M-sparse matrix multiplication
```

## References

- Zhao, K., Yuan, T., Su, Z., et al. "Beyond 2:4: Exploring V:N:M Sparsity for Efficient Transformer Inference on GPUs" (2025). arXiv:2410.16135
- Castro, R.L., et al. "VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores" (SC 2023) — Spatha library for V:N:M acceleration
- Mishra, A., et al. "Accelerating Sparse Deep Neural Networks" (2021) — foundational 2:4 sparsity work
- Zhou, A., et al. "Learning N:M Fine-grained Structured Sparse Neural Networks from Scratch" (ICLR 2021) — SR-STE framework adapted for V:N:M training
- Hubara, I., et al. "Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks" (NeurIPS 2021) — mask diversity metric
- Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — foundation for three-staged LoRA training
- Zhang, Y., et al. "Plug-and-play: An Efficient Post-training Pruning Method for Large Language Models" (ICLR 2024) — RIA importance criterion

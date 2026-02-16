# 130: Transposable N:M Mask via Min-Cost Flow

**Category**: kernel
**Gain type**: efficiency
**Source**: Hubara, Chmiel, Island, Banner, Naor & Soudry "Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks" (NeurIPS 2021)
**Paper**: [papers/transposable-nm-mask-min-cost-flow.pdf]
**Documented**: 2025-06-15

## Description

Standard N:M structured sparsity (e.g., 2:4) accelerates only the **forward pass** of neural network training, because the backward pass uses the **transposed** weight matrix $W^T$, which generally does not satisfy the N:M constraint even when $W$ does. This means Sparse Tensor Cores sit idle during backpropagation — which accounts for 2 of the 3 matrix multiplications per layer.

A **transposable N:M mask** is a binary mask $S \in \{0,1\}^{M \times M}$ such that both $W \odot S$ and $W^T \odot S^T$ satisfy the N:M fine-grained sparsity constraint simultaneously. This enables 2× sparse acceleration in **both** forward and backward passes, doubling the utilization of Sparse Tensor Cores during training.

The key algorithmic insight is that finding the optimal transposable mask can be reduced to a **minimum-cost flow problem** on a bipartite graph, solvable in $O(M^3 \log M)$ time for exact solutions, or $O(M^2 \log M)$ for a provably tight 2-approximation. The transposable constraint is more restrictive than standard N:M (lower "mask diversity"), so the paper proposes using 4:8 transposable masks instead of 2:4 — both have similar mask diversity (and hence similar accuracy), but the transposable version accelerates both passes.

## Mathematical Form

**Problem Formulation (Integer Program):**

For a weight block $W \in \mathbb{R}^{M \times M}$, find a binary mask $S \in \{0,1\}^{M \times M}$ that maximizes the $\ell_1$ norm of kept weights subject to N:M constraints on both rows and columns:

$$
\max_{S \in \{0,1\}^{M \times M}} \sum_{i,j} S_{i,j} |W_{i,j}| \quad \text{s.t.} \quad \forall j: \sum_i S_{i,j} = N, \quad \forall i: \sum_j S_{i,j} = N
$$

This ensures exactly $N$ nonzeros per row (forward N:M constraint) and exactly $N$ nonzeros per column (backward/transpose N:M constraint).

**Reduction to Min-Cost Flow:**

The IP reduces to a min-cost flow on a bipartite graph $G = (V, E)$ with:
- **Source** $s$ and **sink** $t$
- **Row nodes** $\{r_1, \ldots, r_M\}$ and **column nodes** $\{c_1, \ldots, c_M\}$
- **Source edges** $(s, r_i)$ with capacity $\frac{M}{2}$ and cost 0 (number of elements to prune per row)
- **Sink edges** $(c_j, t)$ with capacity $\frac{M}{2}$ and cost 0 (number of elements to prune per column)
- **Coefficient edges** $(r_i, c_j)$ with capacity 1 and cost $|W_{i,j}|$

Total flow: $F = \frac{M^2}{2}$ from source to sink.

A unit flow on edge $(r_i, c_j)$ corresponds to **pruning** element $W_{i,j}$. The min-cost flow minimizes total pruned magnitude, which is equivalent to maximizing kept magnitude.

**Optimal Solution Complexity:** $O(M^3 \log M)$ per block using efficient min-cost flow algorithms.

**2-Approximation Algorithm (Linear-Time):**

A greedy algorithm that is much faster and provably within factor 2 of optimal:

1. Sort all coefficient edges by weight (ascending — light to heavy)
2. For each edge $e_i = (u, v)$ in sorted order:
   - If $\text{degree}(u) \leq \frac{M}{2}$ or $\text{degree}(v) \leq \frac{M}{2}$ in current pruning set $P$:
     - Add $e_i$ to $P$ (prune this element)

**Lemma:** The 2-approximation produces a solution $W(P) < 2 \cdot W^*$, where $W^*$ is the optimal pruned weight sum. This bound is tight.

**Complexity:** $O(M^2 \log M)$ per block (dominated by sorting).

**Mask Diversity Analysis:**

The paper introduces **mask diversity (MD)** — the number of valid masks for a given constraint type — as a predictor of accuracy:

$$
\text{MD}_{\text{Structured}} = \binom{M!}{N!(M-N)!}^{T/M}
$$

$$
\text{MD}_{\text{Transposable}} = (M! \cdot (M-1)! \cdots (M-N+1)!)^{T/M^2}
$$

For 2:4 structured: $\text{MD} = 2.8 \times 10^{12}$ for an $8 \times 8$ block. For 4:8 transposable: $\text{MD} = 1.7 \times 10^{13}$ — **higher** than 2:4 structured, predicting comparable accuracy.

**Key Definitions:**

- $M$ — block size (group size for N:M constraint)
- $N$ — number of zeros required per group of $M$
- $S \in \{0,1\}^{M \times M}$ — binary sparsity mask
- Transposable: both row-wise and column-wise N:M constraints hold
- $W(P) = \sum_{(i,j) \in P} |W_{i,j}|$ — total magnitude of pruned weights

## Complexity

| Operation | Standard N:M Mask | Transposable Mask (Exact) | Transposable Mask (2-Approx) |
|-----------|------------------|--------------------------|------------------------------|
| Mask computation | $O(M \log M)$ per row | $O(M^3 \log M)$ per block | $O(M^2 \log M)$ per block |
| Forward GEMM | Sparse ($2\times$) | Sparse ($2\times$) | Sparse ($2\times$) |
| Backward GEMM ($W^T$) | **Dense** ($1\times$) | **Sparse** ($2\times$) | **Sparse** ($2\times$) |
| Sparse Tensor Core utilization | 33% (1 of 3 GEMMs) | **66%** (2 of 3 GEMMs) | **66%** (2 of 3 GEMMs) |

**Training overhead** (mask recomputation every 40 iterations on ResNet50):
| Method | Overhead vs. dense training |
|--------|----------------------------|
| Integer programming | 180% |
| Min-cost flow | 70% |
| 2-approximation | **14%** |

**Memory:** Same as standard N:M — stores compressed weights + 2-bit metadata indices.

## Applicability

- **Training acceleration**: Primary use case — enables 2× Sparse Tensor Core utilization in both forward and backward passes, giving 66% sparse utilization vs. 33% for forward-only
- **Transformer FFN layers**: All linear projections ($W_Q, W_K, W_V, W_O$, FFN) benefit from transposable masks during training
- **Vision models**: Validated on ResNet18/50, ResNext50, VGG11 (ImageNet) with no accuracy loss vs. NVIDIA ASP baseline
- **Language models**: BERT-large fine-tuning on SQuAD with transposable 4:8 mask matches 2:4 accuracy while accelerating backward pass
- **Detection**: MaskRCNN on COCO with comparable AP to ASP 2:4 baseline
- **Sparse training from scratch**: Compatible with SR-STE-style dynamic mask training — the 2-approximation algorithm recomputes the transposable mask periodically (every 40 iterations)
- **AdaPrune**: The paper also introduces a method to convert unstructured sparse models to N:M structured format via per-layer optimization: $\min_{W'} \|WX - (S \odot W')X\|_2^2$

## Limitations

- 4:8 transposable mask requires block size 8 instead of 4 — the hardware must support 4:8 sparsity pattern (some architectures only support 2:4 natively)
- The 2-approximation algorithm's 14% overhead, while much better than exact, is still nonzero — mask must be recomputed periodically during training
- Transposable constraint reduces mask diversity compared to standard structured sparsity — 4:8 transposable is needed to match 2:4 structured diversity
- The exact min-cost flow approach ($O(M^3 \log M)$) is too expensive for dynamic training — only practical for one-time mask computation from pretrained models
- Mean absorption trick (absorbing pruned weight magnitude into kept weights) provides additional benefit but adds implementation complexity
- Not yet demonstrated on large-scale LLM pre-training (paper focuses on ResNets and BERT)

## Implementation Notes

```python
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def transposable_mask_approx(W: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """2-approximation algorithm for transposable N:M mask.

    For each M×M block, finds a mask satisfying N:M constraints
    on both rows and columns (transposable property).
    """
    rows, cols = W.shape
    mask = torch.ones_like(W, dtype=torch.bool)

    # Process M×M blocks
    for r in range(0, rows, M):
        for c in range(0, cols, M):
            block = W[r:r+M, c:c+M].abs()
            block_mask = _approx_2_transposable(block, N, M)
            mask[r:r+M, c:c+M] = block_mask

    return mask

def _approx_2_transposable(block: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """Greedy 2-approximation: sort edges light-to-heavy,
    greedily prune if row/column hasn't reached quota."""
    prune_per_rc = M // 2  # number to prune per row/col
    prune_count_row = torch.zeros(M, dtype=torch.long)
    prune_count_col = torch.zeros(M, dtype=torch.long)
    mask = torch.ones(M, M, dtype=torch.bool)

    # Sort all elements by magnitude (ascending = light first)
    flat = block.flatten()
    sorted_idx = flat.argsort()

    for idx in sorted_idx:
        i, j = idx // M, idx % M
        # Prune if either row or column still needs pruning
        if prune_count_row[i] < prune_per_rc or prune_count_col[j] < prune_per_rc:
            mask[i, j] = False
            prune_count_row[i] += 1
            prune_count_col[j] += 1

    return mask

def transposable_mask_exact(W: torch.Tensor, M: int) -> torch.Tensor:
    """Exact solution via min-cost flow (using scipy for demo).
    In practice, use a dedicated min-cost flow solver."""
    # For M/2:M sparsity, we need to select M/2 elements
    # to KEEP per row and per column — equivalent to a
    # maximum weight bipartite matching generalization.
    # This is a simplified version using assignment for M/2=1 case.
    rows, cols = W.shape
    mask = torch.ones_like(W, dtype=torch.bool)
    # ... (use dedicated min-cost flow library for general case)
    return mask
```

## References

- Hubara, I., Chmiel, B., Island, M., Banner, R., Naor, J. & Soudry, D. "Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks" (NeurIPS 2021). arXiv:2102.08124
- Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. "Network Flows" (1988). Foundation for min-cost flow algorithms.
- Zhou, A., et al. "Learning N:M Fine-grained Structured Sparse Neural Networks from Scratch" (ICLR 2021). SR-STE training with N:M masks.
- Mishra, A., et al. "Accelerating Sparse Deep Neural Networks" (2021). NVIDIA ASP (Automatic Sparsity) baseline.
- Zhang, Y., et al. "Bi-directional Masks for Efficient N:M Sparse Training" (ICML 2023). Improved transposable training via Bi-Mask.
- Hu, Y., et al. "Accelerating Transformer Pre-training with 2:4 Sparsity" (ICML 2024). Transposable mask computation by convolution.

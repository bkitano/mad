# 133: TSENOR: Entropy-Regularized Optimal Transport for Transposable N:M Masks

**Category**: kernel
**Gain type**: efficiency
**Source**: Meng, Makni & Mazumder "TSENOR: Highly-Efficient Algorithm for Finding Transposable N:M Sparse Masks" (MIT, 2025)
**Paper**: [papers/tsenor-transposable-nm-masks.pdf]
**Documented**: 2025-07-14

## Description

TSENOR (Transposable N:M Sparsity with ENtropy regularization and Optimized Rounding) is a GPU-accelerated algorithm for finding high-quality transposable N:M sparse masks that scales to billion-parameter models. It improves upon the min-cost flow approach (see `transposable-nm-mask-min-cost-flow`) by reformulating the mask-finding problem as millions of small **optimal transport** problems that can be solved simultaneously on GPUs via entropy regularization and Dykstra's algorithm.

The key insight is a three-stage pipeline:
1. **Optimal transport reformulation**: The transposable N:M mask problem on each $M \times M$ block is equivalent to a capacitated optimal transport problem, where each row/column must "send/receive" exactly $N$ units of mass.
2. **Entropy-regularized relaxation**: Adding Shannon entropy $\frac{1}{\tau}H(\mathbf{S})$ to the objective converts the combinatorial problem into a smooth optimization solvable by Dykstra's algorithm — which only requires matrix-vector multiplications and element-wise operations, enabling full GPU vectorization across all blocks simultaneously.
3. **Greedy selection + local search rounding**: The fractional solution from step 2 is rounded to a binary mask via a greedy descending-magnitude assignment followed by swap-based local search that fixes unsaturated rows/columns.

This achieves up to **100x speedup** over prior methods (min-cost flow, cuPDLP) with only **1-10% relative error** vs. optimal solutions. Crucially, it handles **arbitrary N:M patterns** (not just 2:4), enabling transposable 8:16, 16:32, etc. — patterns where the performance gap between transposable and standard N:M diminishes to ~12% while enabling 2x acceleration in both forward and backward passes.

## Mathematical Form

**Core Optimization Problem:**

For weight matrix $\mathbf{W}$, partitioned into $M \times M$ blocks, find binary mask $\mathbf{S}$ maximizing kept weight magnitude under transposable N:M constraints:

$$
\max_{\mathbf{S} \in \{0,1\}^{M \times M}} \sum_{i,j} \mathbf{S}_{ij} |\mathbf{W}_{ij}| \quad \text{s.t. } \mathbf{S}\mathbf{1}_M = N\mathbf{1}_M, \quad \mathbf{S}^\top\mathbf{1}_M = N\mathbf{1}_M
$$

The row-sum constraint ensures N:M sparsity on rows (forward pass); the column-sum constraint ensures N:M sparsity on columns (backward pass via $\mathbf{W}^\top$).

**LP Relaxation as Optimal Transport:**

By bipartite matching polytope theory, the binary constraint relaxes to $\mathbf{S} \in [0,1]^{M \times M}$ without changing the optimal value:

$$
\max_{\mathbf{S}} \langle \mathbf{S}, |\mathbf{W}| \rangle \quad \text{s.t. } \mathbf{S}\mathbf{1}_M = N\mathbf{1}_M, \quad \mathbf{S}^\top\mathbf{1}_M = N\mathbf{1}_M, \quad \mathbf{0} \leq \mathbf{S} \leq \mathbf{1}
$$

This is a **capacitated optimal transport** problem: $\mathbf{S}$ is a transport plan where each row and column sends/receives exactly $N$ units of mass.

**Entropy-Regularized Formulation:**

$$
\max_{\mathbf{S}} \langle \mathbf{S}, |\mathbf{W}| \rangle + \frac{1}{\tau} H(\mathbf{S}) \quad \text{s.t. } \mathbf{S}\mathbf{1}_M = N\mathbf{1}_M, \quad \mathbf{S}^\top\mathbf{1}_M = N\mathbf{1}_M, \quad \mathbf{0} \leq \mathbf{S} \leq \mathbf{1}
$$

where $H(\mathbf{S}) = -\sum_{i,j} \mathbf{S}_{ij} \log(\mathbf{S}_{ij})$ is the Shannon entropy and $\tau > 0$ controls regularization strength. Small $\tau$ yields solutions that poorly approximate the original; excessively large $\tau$ impedes convergence.

**Dykstra's Algorithm (Algorithm 1):**

The entropy-regularized problem is interpreted as projecting $\mathbf{W}_\tau = \exp(\tau|\mathbf{W}|)$ onto the intersection of three constraint sets:

$$
\mathcal{C}_1 = \{\mathbf{S} \mid \mathbf{S}\mathbf{1}_M = N\mathbf{1}_M\}, \quad \mathcal{C}_2 = \{\mathbf{S} \mid \mathbf{S}^\top\mathbf{1}_M = N\mathbf{1}_M\}, \quad \mathcal{C}_3 = \{\mathbf{S} \mid \mathbf{0} \leq \mathbf{S} \leq \mathbf{1}\}
$$

Dykstra's algorithm iteratively projects onto each set:

1. Initialize $\mathbf{S}^{(0)} = \exp(\tau|\mathbf{W}|)$, dual variable $\mathbf{Q}^{(0)} = \mathbf{1}_{M \times M}$
2. For $t = 0, 1, \ldots, T-1$:
   - $\mathbf{S}^{(t)} \leftarrow \text{Diag}(N / (\mathbf{S}^{(t)}\mathbf{1}_M)) \, \mathbf{S}^{(t)}$ (project onto $\mathcal{C}_1$: row normalization)
   - $\mathbf{S}^{(t)} \leftarrow \mathbf{S}^{(t)} \text{Diag}(N / (\mathbf{S}^{(t)\top}\mathbf{1}_M))$ (project onto $\mathcal{C}_2$: column normalization)
   - $\mathbf{S}^{(t+1)} \leftarrow \min(\mathbf{S}^{(t)} \odot \mathbf{Q}^{(t)}, \mathbf{1})$ (project onto $\mathcal{C}_3$: box constraint)
   - $\mathbf{Q}^{(t+1)} \leftarrow \mathbf{Q}^{(t)} \odot (\mathbf{S}^{(t)} \oslash \mathbf{S}^{(t+1)})$ (update dual variable)

All operations are element-wise or matrix-vector, enabling full vectorization across millions of blocks.

**Rounding via Greedy Selection + Local Search (Algorithm 2):**

Given fractional solution $\mathbf{S}^a$ from Dykstra's algorithm:

1. **Greedy selection**: Sort all $M^2$ elements of $\mathbf{S}^a$ in descending order. Iterate through and assign $\mathbf{S}_{i,j} = 1$ if row $i$ has fewer than $N$ selected elements AND column $j$ has fewer than $N$ selected elements.

2. **Local search** (up to $L$ steps): For any unsaturated row $i$ and column $j$ (fewer than $N$ selected elements), find the swap $(i', j')$ that maximizes:

$$
\text{Swap}(i', j') := (|\mathbf{W}_{i,j'}| + |\mathbf{W}_{i',j}| - |\mathbf{W}_{i',j'}|) - \infty \cdot ((1 - \mathbf{S}_{i',j'}) + \mathbf{S}_{i,j'} + \mathbf{S}_{i',j})
$$

If $\text{Swap}(i', j') > 0$: insert elements $(i, j')$ and $(i', j)$, remove $(i', j')$.

**Key Definitions:**

- $M$ — block size for N:M sparsity constraint
- $N$ — number of nonzeros to keep per group of $M$
- $\tau$ — entropy regularization parameter (controls approximation quality vs. convergence speed)
- $T$ — maximum Dykstra iterations
- $L$ — number of local search steps in rounding
- $\mathbf{Q}$ — Dykstra dual variable tracking constraint violations

## Complexity

| Operation | Min-Cost Flow (Hubara et al.) | 2-Approximation | TSENOR |
|-----------|------------------------------|-----------------|--------|
| Per-block mask computation | $O(M^3 \log M)$ | $O(M^2 \log M)$ | $O(TM^2 + LM^2)$ |
| Parallelism | Sequential per block | Sequential per block | **All blocks simultaneously** (GPU) |
| Solution quality (rel. error) | Optimal (0%) | Up to 50% | **1-10%** |
| Wall-clock (8192x8192, 8:16) | 350s (CPU) | 3.23s (CPU) | **0.12s (H100)** |
| Wall-clock (512x512, 8:16) | 1.82s (CPU) | 0.13s (CPU) | **0.08s (H100)** |

**Speedup vs. baselines:**
- Up to **100x** faster than min-cost flow (Network Flow)
- Up to **300x** faster than cuPDLP (GPU-accelerated LP solver)
- Within **1-10%** of optimal solution quality

**Memory:** $O(B \cdot M^2)$ where $B$ is the number of $M \times M$ blocks — all blocks stored as a single batched tensor.

## Applicability

- **LLM pruning at scale**: Demonstrated on LLaMA 3.2 (1B-8B parameters) — the first transposable N:M mask solver that practically scales to billion-parameter models
- **Integration with pruning frameworks**: Plug-in replacement for the mask-finding step in Wanda, SparseGPT, and ALPS. With ALPS, TSENOR+ALPS produces transposable N:M models that slightly **outperform** standard (non-transposable) N:M via SparseGPT
- **Large M values (8:16, 16:32)**: The key practical finding — transposable sparsity with larger M values (e.g., 16:32) incurs only ~12% of the performance loss compared to smaller M (2:4), while delivering the same ~3.3x speedup at 75% sparsity
- **Training acceleration**: Transposable masks enable Sparse Tensor Core utilization in **both** forward and backward passes (2 of 3 GEMMs per layer), vs. only 1 of 3 for standard N:M
- **Fine-tuning transposable models**: TSENOR+ALPS followed by fine-tuning progressively outperforms Bi-NM baseline as M increases, with exact gradients and better parameter updates

## Limitations

- Requires tuning $\tau$ (entropy regularization strength) — too small gives poor approximation, too large impedes convergence
- Rounding step (greedy + local search) does not guarantee optimality — only 1-10% relative error in practice
- The fractional Dykstra solution requires rounding, adding implementation complexity beyond Sinkhorn-style iterations
- Only demonstrated for post-training pruning and fine-tuning — not yet validated for sparse training from scratch with dynamic mask updates
- GPU memory scales with $B \cdot M^2$ — very large block sizes could become memory-bound
- Currently limited to standard N:M patterns; does not directly handle V:N:M or other hierarchical sparsity patterns

## Implementation Notes

```python
import torch

def tsenor_dykstra(W_blocks: torch.Tensor, N: int, M: int,
                    tau: float = 10.0, T: int = 50) -> torch.Tensor:
    """Entropy-regularized optimal transport via Dykstra's algorithm.

    Args:
        W_blocks: (B, M, M) tensor of weight magnitude blocks
        N: number of nonzeros to keep per row/column
        tau: entropy regularization strength
        T: max Dykstra iterations

    Returns:
        S: (B, M, M) fractional approximate solution
    """
    B = W_blocks.shape[0]
    # Initialize S = exp(tau * |W|), Q = 1
    S = torch.exp(tau * W_blocks.abs())
    Q = torch.ones_like(S)

    for t in range(T):
        # Project onto C1: row sums = N
        row_sums = S.sum(dim=2, keepdim=True)  # (B, M, 1)
        S = S * (N / row_sums)

        # Project onto C2: column sums = N
        col_sums = S.sum(dim=1, keepdim=True)  # (B, 1, M)
        S = S * (N / col_sums)

        # Project onto C3: 0 <= S <= 1 (with Dykstra dual update)
        S_before = S * Q
        S = torch.clamp(S_before, max=1.0)
        Q = Q * (S_before / S.clamp(min=1e-10))

    return S


def tsenor_round(S_frac: torch.Tensor, W_blocks: torch.Tensor,
                  N: int, M: int, L: int = 10) -> torch.Tensor:
    """Round fractional solution to binary mask via greedy + local search.

    Args:
        S_frac: (B, M, M) fractional solution from Dykstra
        W_blocks: (B, M, M) original weight magnitudes
        N: nonzeros per row/column
        L: local search steps

    Returns:
        S_binary: (B, M, M) binary transposable N:M mask
    """
    B = S_frac.shape[0]
    S = torch.zeros(B, M, M, dtype=torch.bool, device=S_frac.device)

    # Greedy selection: sort descending, assign if row/col not saturated
    flat_vals = S_frac.reshape(B, -1)
    sorted_idx = flat_vals.argsort(dim=1, descending=True)

    row_counts = torch.zeros(B, M, dtype=torch.long, device=S_frac.device)
    col_counts = torch.zeros(B, M, dtype=torch.long, device=S_frac.device)

    for k in range(M * M):
        idx = sorted_idx[:, k]
        i = idx // M
        j = idx % M
        can_assign = (row_counts.gather(1, i.unsqueeze(1)).squeeze(1) < N) & \
                     (col_counts.gather(1, j.unsqueeze(1)).squeeze(1) < N)
        # Set mask for assignable elements
        S[torch.arange(B)[can_assign], i[can_assign], j[can_assign]] = True
        row_counts.scatter_add_(1, i[can_assign].unsqueeze(1),
                               torch.ones(can_assign.sum(), 1, dtype=torch.long,
                                          device=S.device))
        col_counts.scatter_add_(1, j[can_assign].unsqueeze(1),
                               torch.ones(can_assign.sum(), 1, dtype=torch.long,
                                          device=S.device))

    # Local search: swap to improve objective (simplified)
    # Full implementation enumerates swap candidates for unsaturated rows/cols
    return S


def tsenor_pipeline(W: torch.Tensor, N: int, M: int,
                     tau: float = 10.0, T: int = 50, L: int = 10):
    """Full TSENOR pipeline: partition -> Dykstra -> round."""
    R, C = W.shape
    # Partition into M x M blocks
    W_blocks = W.reshape(R // M, M, C // M, M).permute(0, 2, 1, 3)
    W_blocks = W_blocks.reshape(-1, M, M)  # (B, M, M)

    S_frac = tsenor_dykstra(W_blocks, N, M, tau, T)
    S_binary = tsenor_round(S_frac, W_blocks, N, M, L)

    # Reshape back to original dimensions
    B_r, B_c = R // M, C // M
    mask = S_binary.reshape(B_r, B_c, M, M).permute(0, 2, 1, 3)
    mask = mask.reshape(R, C)
    return mask
```

## References

- Meng, X., Makni, M. & Mazumder, R. "TSENOR: Highly-Efficient Algorithm for Finding Transposable N:M Sparse Masks" (2025). arXiv:2505.23949
- Hubara, I., et al. "Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks" (NeurIPS 2021). arXiv:2102.08124 — original min-cost flow approach
- Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013) — entropy regularization for OT
- Benamou, J.-D., et al. "Iterative Bregman Projections for Regularized Transportation Problems" (SIAM J. Sci. Comput. 2015) — Dykstra's algorithm for constrained OT
- Zhang, Y., et al. "Bi-directional Masks for Efficient N:M Sparse Training" (ICML 2023) — Bi-NM baseline
- Meng, X., et al. "ALPS" (2024a) — layer-wise pruning framework integrated with TSENOR

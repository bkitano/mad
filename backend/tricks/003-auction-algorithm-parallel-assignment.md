# 003: Auction Algorithm for Parallel Assignment

**Category**: parallelization
**Gain type**: efficiency
**Source**: Bertsekas (1979, 1992), Date & Nagi (2016) — GPU acceleration
**Paper**: [papers/auction-algorithm-bertsekas.pdf]
**Documented**: 2026-02-15

## Description

The **auction algorithm** is a massively parallelizable alternative to the Hungarian algorithm for solving the **linear assignment problem** (LAP): given an $n \times n$ benefit matrix $A = [a_{ij}]$, find a one-to-one assignment of persons to objects that maximizes $\sum_{i=1}^{n} a_{i j_i}$. In neural network contexts, LAP arises as the "hardening" step when converting a soft doubly stochastic matrix (from Sinkhorn normalization) into a hard permutation matrix — exactly the operation needed in blockwise channel permutation for N:M sparsity.

The Hungarian algorithm has $O(n^3)$ worst-case complexity and is inherently **sequential**: it maintains augmenting paths through a bipartite graph, with each path depending on previous ones. This makes it poorly suited for GPU execution. The auction algorithm, by contrast, is **naturally parallel**: in each round, all unassigned "bidders" can independently and simultaneously compute their bids, and all "objects" can independently award themselves to the highest bidder.

The algorithm operates via an economic metaphor: persons (rows) bid for objects (columns) by raising their prices, and objects are awarded to the highest bidder. The key innovation is **$\epsilon$-complementary slackness** — each bid must raise the object's price by at least $\epsilon > 0$, which breaks ties and guarantees finite termination. The resulting assignment is within $n\epsilon$ of optimal.

For exact optimality with integer costs, **$\epsilon$-scaling** applies the auction algorithm with geometrically decreasing $\epsilon$ values (e.g., $\epsilon_k = \epsilon_{k-1} / 2$), using the prices from each phase to warm-start the next. This yields $O(n^2 \log(nC))$ average complexity (vs. $O(n^3)$ for Hungarian), where $C = \max_{i,j} |a_{ij}|$.

In the context of blockwise Sinkhorn channel permutation (PermLLM), each block's Sinkhorn output $\hat{P}_i \in \mathbb{R}^{B \times B}$ must be hardened to a permutation via $\arg\max_{P \in \mathcal{P}} \text{Tr}(P^\top \hat{P}_i)$, which is exactly a LAP instance of size $B$. PermLLM uses the Hungarian algorithm at $O(B^3)$ per block; replacing it with the auction algorithm enables **parallel execution across all $N_B$ blocks and within each block**, potentially reducing the hardening bottleneck significantly on GPUs.

## Mathematical Form

**Assignment Problem:**

$$
\max_{\sigma \in S_n} \sum_{i=1}^{n} a_{i,\sigma(i)}
$$

where $S_n$ is the symmetric group of permutations of $\{1, \ldots, n\}$.

**Dual Problem (prices):**

$$
\min_{p_1, \ldots, p_n} \left\{ \sum_{j=1}^{n} p_j + \sum_{i=1}^{n} \max_j \{a_{ij} - p_j\} \right\}
$$

**$\epsilon$-Complementary Slackness:**

An assignment $\sigma$ and prices $\mathbf{p}$ satisfy $\epsilon$-CS if for all assigned pairs $(i, \sigma(i))$:

$$
a_{i,\sigma(i)} - p_{\sigma(i)} \geq \max_{j=1,\ldots,n} \{a_{ij} - p_j\} - \epsilon
$$

This means each person is assigned to an object whose value is within $\epsilon$ of the maximum.

**Bidding Step (parallelizable over all unassigned persons $i$):**

1. Find the best object: $j_i = \arg\max_j \{a_{ij} - p_j\}$

2. Compute the best value: $v_i = \max_j \{a_{ij} - p_j\}$

3. Compute the second-best value: $w_i = \max_{j \neq j_i} \{a_{ij} - p_j\}$

4. Compute the bid increment: $\gamma_i = v_i - w_i + \epsilon$

5. Place bid on object $j_i$ with new price: $p_{j_i} \leftarrow p_{j_i} + \gamma_i$

**Assignment Step (parallelizable over all objects $j$):**

Each object $j$ that received bids is assigned to the highest bidder.

**$\epsilon$-Scaling:**

Apply the auction algorithm in $K = \lceil \log_2(nC) \rceil$ phases with:

$$
\epsilon_k = \frac{C}{2^k}, \quad k = 0, 1, \ldots, K
$$

where the final phase has $\epsilon_K < 1/n$ (guaranteeing exact optimality for integer costs). The prices from phase $k$ warm-start phase $k+1$.

**Optimality Guarantee:**

If all benefits $a_{ij}$ are integer and $\epsilon < 1/n$, the assignment at termination is **optimal** (not just $\epsilon$-approximate).

## Complexity

| Operation | Hungarian | Auction (single $\epsilon$) | Auction ($\epsilon$-scaling) |
|-----------|-----------|--------------------------|------------------------------|
| Sequential | $O(n^3)$ | $O(n^2 C / \epsilon)$ | $O(n^2 \log(nC))$ |
| Parallel (Jacobi) | Hard to parallelize | $O(n \cdot C / \epsilon)$ with $n$ processors | $O(n \log(nC))$ with $n$ processors |
| GPU practical | Poor utilization | High utilization (all bidders parallel) | High utilization |

**For blockwise channel permutation ($N_B$ blocks of size $B$):**

| Method | Total Hardening Cost | GPU Parallelism |
|--------|---------------------|-----------------|
| Hungarian (PermLLM) | $O(N_B \cdot B^3)$ sequential per block | $N_B$ blocks independent, but each block sequential |
| Auction ($\epsilon$-scaling) | $O(N_B \cdot B^2 \log(BC))$ | Both across blocks AND within each block |

**Concrete numbers (LLaMA-2 7B, $C_\text{in} = 4096$, $B = 64$, $N_B = 64$):**
- Hungarian: $O(64 \times 64^3) = O(16.8\text{M})$ operations, sequential per block
- Auction: $O(64 \times 64^2 \times \log(64 \cdot C))$ operations, massively parallel within each block

**Memory:** $O(n)$ for the price vector $\mathbf{p}$ per block — much less than $O(n^2)$ for the full cost matrix (which is already available as the Sinkhorn output).

## Applicability

- **Blockwise Sinkhorn hardening:** Direct replacement for the Hungarian algorithm in PermLLM's STE hardening step. Each of the $N_B$ blocks computes $P_i = \arg\max_P \text{Tr}(P^\top \hat{P}_i)$ via auction instead of Hungarian. The auction's natural parallelism maps well to GPU warps.
- **Differentiable sorting/matching in neural networks:** Any pipeline using Sinkhorn + Hungarian (or Gumbel-Sinkhorn + Hungarian) for permutation learning can substitute the auction algorithm for the hardening step.
- **Optimal transport discretization:** When solving entropy-regularized OT via Sinkhorn and then rounding to a discrete transport plan, the auction algorithm provides a parallelizable rounding method.
- **GPU-accelerated combinatorial optimization:** The Jacobi (fully parallel) variant of the auction algorithm maps naturally to GPU execution, with all unassigned bidders executing simultaneously as GPU threads.
- **Sparse attention routing:** Learned token-to-expert or token-to-block assignments that require solving an assignment problem can benefit from the auction algorithm's parallelism.
- **Structured pruning:** Channel permutation for structured sparsity (N:M, block-sparse) requires repeated LAP solving during the permutation learning phase.

## Limitations

- **$\epsilon$-approximate:** Without $\epsilon$-scaling, the result is only $n\epsilon$-optimal. For the Sinkhorn hardening application, this may be acceptable since the Sinkhorn output is already soft — but if exact optimality matters, $\epsilon$-scaling adds a $\log(nC)$ factor.
- **Cost matrix range dependence:** The number of bidding rounds is proportional to $C/\epsilon$ where $C = \max|a_{ij}|$. For Sinkhorn outputs (entries in $[0, 1]$), $C = 1$, so this is favorable. But for general cost matrices, the algorithm can be slower than Hungarian.
- **Synchronization overhead:** In the parallel (Jacobi) variant, multiple bidders may bid on the same object, requiring conflict resolution. On GPUs, this translates to atomic operations or warp-level reductions.
- **Not differentiable:** Like the Hungarian algorithm, the auction algorithm produces a discrete permutation — it cannot directly replace the Sinkhorn operator for gradient flow. It is useful only for the hardening (forward-pass) step, combined with STE for backward pass.
- **Diminishing returns for small $n$:** For very small block sizes (e.g., $B = 4$), the overhead of the auction framework may exceed a simple brute-force or Hungarian approach. The advantage materializes for $B \geq 16$.
- **Implementation complexity:** A correct and efficient GPU auction implementation requires careful handling of atomic price updates and bidder conflict resolution, which adds engineering complexity vs. calling `scipy.optimize.linear_sum_assignment`.

## Implementation Notes

```python
import torch

def auction_lap(cost_matrix, epsilon=1e-3, max_iters=1000):
    """
    Auction algorithm for the linear assignment problem (LAP).
    Solves: max_sigma sum_i cost[i, sigma(i)]

    This is the Gauss-Seidel (sequential) version.
    For GPU, use the Jacobi variant where all unassigned
    persons bid simultaneously.

    Args:
        cost_matrix: (n, n) benefit matrix
        epsilon: minimum bid increment (controls optimality gap)
        max_iters: maximum auction rounds

    Returns:
        assignment: (n,) tensor of column indices
    """
    n = cost_matrix.shape[0]
    prices = torch.zeros(n, device=cost_matrix.device)
    person_to_obj = -torch.ones(n, dtype=torch.long,
                                 device=cost_matrix.device)
    obj_to_person = -torch.ones(n, dtype=torch.long,
                                 device=cost_matrix.device)

    for _ in range(max_iters):
        # Find unassigned persons
        unassigned = (person_to_obj == -1).nonzero(as_tuple=True)[0]
        if len(unassigned) == 0:
            break

        for i in unassigned:
            # Compute net values: a_ij - p_j
            values = cost_matrix[i] - prices

            # Best and second-best objects
            sorted_vals, sorted_idx = values.topk(2)
            j_best = sorted_idx[0]
            v_best = sorted_vals[0]
            w_second = sorted_vals[1]

            # Bid increment
            gamma = v_best - w_second + epsilon

            # Evict current owner of j_best (if any)
            prev_owner = obj_to_person[j_best].item()
            if prev_owner >= 0:
                person_to_obj[prev_owner] = -1

            # Assign person i to object j_best
            person_to_obj[i] = j_best
            obj_to_person[j_best] = i

            # Raise price
            prices[j_best] += gamma

    return person_to_obj


def auction_lap_jacobi(cost_matrix, epsilon=1e-3, max_iters=1000):
    """
    Jacobi (fully parallel) auction algorithm.
    All unassigned persons bid simultaneously — suitable for GPU.

    Note: multiple persons may bid on the same object; highest
    bidder wins via scatter_max (requires torch_scatter or atomics).
    """
    n = cost_matrix.shape[0]
    prices = torch.zeros(n, device=cost_matrix.device)
    assignment = -torch.ones(n, dtype=torch.long,
                              device=cost_matrix.device)

    for _ in range(max_iters):
        unassigned = (assignment == -1).nonzero(as_tuple=True)[0]
        if len(unassigned) == 0:
            break

        # All unassigned persons compute values simultaneously
        values = cost_matrix[unassigned] - prices.unsqueeze(0)
        top2 = values.topk(2, dim=-1)
        j_best = top2.indices[:, 0]  # best object for each person
        v_best = top2.values[:, 0]
        w_second = top2.values[:, 1]
        bids = v_best - w_second + epsilon

        # Resolve conflicts: for each object, keep highest bidder
        # (In CUDA, this would use atomicMax)
        for idx, person_idx in enumerate(unassigned):
            j = j_best[idx].item()
            bid = bids[idx].item()

            # Check if this is the highest bid for object j
            # (simplified; real GPU impl uses atomics)
            prev = assignment.clone()
            evict = (prev == j).nonzero(as_tuple=True)[0]
            if len(evict) > 0:
                assignment[evict[0]] = -1
            assignment[person_idx] = j
            prices[j] += bid

    return assignment


def blockwise_auction_hardening(P_soft_blocks, epsilon=1e-3):
    """
    Replace Hungarian with auction for blockwise Sinkhorn hardening.

    Args:
        P_soft_blocks: list of (B, B) doubly stochastic matrices
    Returns:
        P_hard_blocks: list of (B, B) hard permutation matrices
    """
    P_hard_blocks = []
    for P_soft in P_soft_blocks:
        # LAP: find permutation maximizing Tr(P^T @ P_soft)
        # = maximizing sum_i P_soft[i, sigma(i)]
        assignment = auction_lap(P_soft, epsilon=epsilon)

        P_hard = torch.zeros_like(P_soft)
        rows = torch.arange(P_soft.shape[0], device=P_soft.device)
        P_hard[rows, assignment] = 1.0
        P_hard_blocks.append(P_hard)

    return P_hard_blocks
```

## References

- Bertsekas, D. P. (1979). A Distributed Algorithm for the Assignment Problem. Lab. for Information and Decision Systems Working Paper, M.I.T.
- Bertsekas, D. P. (1992). Auction Algorithms for Network Flow Problems: A Tutorial Introduction. Computational Optimization and Applications, Vol. 1, pp. 7-66.
- Bertsekas, D. P. (1998). Network Optimization: Continuous and Discrete Problems. Athena Scientific.
- Bertsekas, D. P. & Castanon, D. A. (1989). The Auction Algorithm for Transportation Problems. Annals of Operations Research, Vol. 20, pp. 67-96.
- Bertsekas, D. P., Castanon, D. A., Eckstein, J., & Zenios, S. (1995). Parallel Computing in Network Optimization. Handbooks in OR and MS, Vol. 7.
- Date, K. & Nagi, R. (2016). GPU-accelerated Hungarian algorithms for the Linear Assignment Problem. Parallel Computing, 57, pp. 52-72.
- Lopes, P. A. et al. (2019). Fast block distributed CUDA implementation of the Hungarian algorithm. Journal of Parallel and Distributed Computing.
- Castanon, D. A. (1993). Reverse Auction Algorithms for Assignment Problems. In Algorithms for Network Flows and Matching.
- Johnson, B. (2019). auction-lap: Auction algorithm for solving LAP in PyTorch (GPU). GitHub: bkj/auction-lap.

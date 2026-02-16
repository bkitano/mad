# 110: ShuffleSoftSort: N-Parameter Permutation Learning

**Category**: approximation
**Gain type**: efficiency
**Source**: Barthel, Barthel & Eisert (HTW Berlin, Fraunhofer HHI, 2025)
**Paper**: [papers/softsort-n-parameter-permutation.pdf]
**Documented**: 2026-02-15

## Description

Differentiable permutation learning methods like Gumbel-Sinkhorn require $O(N^2)$ parameters to represent a soft permutation matrix over $N$ elements, making them impractical for large-scale problems ($N > 10{,}000$). Low-rank factorizations ("Kissing to Find a Match") reduce this to $O(NM)$ with $M \ll N$, but still struggle at very large scales and often fail to produce valid permutations. SoftSort provides a continuous relaxation of the argsort operator using only $N$ parameters (a 1D weight vector), but is inherently limited to one-dimensional sorting — it cannot perform complex permutations like 2D grid rearrangements or multi-criteria assignments.

**ShuffleSoftSort** extends SoftSort to handle complex, multi-dimensional permutation problems while retaining the $O(N)$ parameter count. The key idea is to **iteratively shuffle the indices and apply SoftSort on the shuffled order**. Each iteration:
1. Randomly permutes the element indices
2. Applies SoftSort optimization steps on the shuffled sequence
3. Reverse-shuffles to recover the original ordering
4. Computes the loss on the reverse-shuffled output

By repeatedly changing the 1D ordering in which SoftSort operates, elements that could never reach their optimal positions via monotonic sorting along a single axis can now be repositioned through multiple shuffled passes. The trainable weights $w \in \mathbb{R}^N$ are initialized linearly in ascending order for each shuffle iteration, with a small initial temperature $\tau$ that is gradually reduced.

This is directly relevant to channel permutation for N:M sparsity: when $N = C_\text{in}$ is large (e.g., 4096 or more), Gumbel-Sinkhorn requires $\sim$16M parameters per layer, while ShuffleSoftSort requires only 4096 — a **4096$\times$ parameter reduction**. The iterative shuffling mechanism can explore permutations that blockwise approaches cannot (cross-block moves), potentially combining the global reach of full permutation with the memory efficiency of blockwise methods.

## Mathematical Form

**SoftSort Operator:**

Given a weight vector $w \in \mathbb{R}^N$ and a sorted reference $w_\text{sort}$, the soft permutation matrix is:

$$
\text{SoftSort}_\tau^D(w) = \text{softmax}\left(\frac{-D(w_\text{sort}, w)}{\tau}\right)
$$

where $D(w_\text{sort}, w) \in \mathbb{R}^{N \times N}$ is the L1 distance matrix between sorted and unsorted elements:

$$
D_{ij} = |w_{\text{sort},i} - w_j|
$$

Row-wise softmax produces a doubly stochastic-like matrix where elements at matching positions have high values. As $\tau \to 0$, $P_\text{soft}$ converges to a hard permutation matrix.

**ShuffleSoftSort Algorithm:**

For $R$ outer iterations (epochs):
1. Anneal temperature: $\tau = \tau_\text{start} \cdot (\tau_\text{end}/\tau_\text{start})^{r/R}$ where $\tau_\text{start} = 1$, $\tau_\text{end} = 0.1$
2. Initialize weights linearly: $w = \text{arange}(0, N)$
3. Generate random shuffle: $\text{shuf\_idx} = \text{randperm}(N)$
4. Shuffle input: $x_\text{shuf} = x[\text{shuf\_idx}]$
5. For $I$ inner SoftSort iterations (default $I = 4$):
   - Increase $\tau_i$ from $0.2\tau$ to $\tau$ (small $\tau_i$ preserves initial order)
   - Compute soft permutation: $P_\text{soft} = \text{SoftSort}(w, \tau_i)$
   - Apply to shuffled input: $x_\text{sort,soft} = P_\text{soft} \, @ \, x_\text{shuf}$
   - Reverse shuffle: $x_\text{sort,soft}[\text{shuf\_idx}] = x_\text{sort,soft}$
   - Compute and backpropagate loss
6. Extract hard permutation: $\text{sort\_idx} = \arg\max(P_\text{soft}, -1)$
7. Reorder: $x_\text{sort} = x[\text{shuf\_idx}][\text{sort\_idx}]$

**Loss Function for Grid Sorting:**

$$
L(P) = L_\text{nbr}(P) + \lambda_s L_s(P) + \lambda_\sigma L_\sigma(P)
$$

where:

- $L_\text{nbr}(P)$: neighborhood loss (average distance of neighboring grid vectors)
- $L_s(P) = \frac{1}{N} \sum_j \left(\left(\sum_i P_{ij}\right) - 1\right)^2$: stochastic constraint loss ensuring doubly stochastic convergence
- $L_\sigma(P) = |\sigma_X - \sigma_Y| / \sigma_X$: standard deviation loss preserving output variance

Default: $\lambda_s = 1$, $\lambda_\sigma = 2$.

**Comparison of Parameter Counts:**

| Method | Parameters | Iterative Normalization | Quality | Stability |
|--------|-----------|------------------------|---------|-----------|
| Gumbel-Sinkhorn | $N^2$ | Yes | ++ | + |
| Kissing (low-rank) | $2NM$ | No | + | $-^*$ |
| SoftSort | $N$ | No | $-$ | ++ |
| **ShuffleSoftSort** | $N$ | No | ++ | ++ |

$^*$Kissing often fails to produce valid permutations.

## Complexity

| Operation | Gumbel-Sinkhorn | Blockwise Sinkhorn ($N_B$ blocks) | ShuffleSoftSort |
|-----------|----------------|----------------------------------|-----------------|
| Parameters | $N^2$ | $N_B \times B^2 = N \times B$ | $N$ |
| Memory | $O(N^2)$ | $O(N \times B)$ | $O(N^2)$ per SoftSort call$^*$ |
| Per-iteration cost | $O(N^2)$ Sinkhorn | $O(N \times B)$ Sinkhorn | $O(N^2)$ distance + softmax |
| Hardening | $O(N^3)$ Hungarian | $O(N \times B^2)$ Hungarian | $O(N \log N)$ argsort |

$^*$The $N \times N$ distance matrix is computed per SoftSort call, but only $N$ parameters are stored and optimized. The distance matrix computation can be done in tiles if $N$ is very large.

**Key advantage:** The $O(N \log N)$ hardening via argsort (instead of $O(N^3)$ Hungarian) and the $N$ parameter count make ShuffleSoftSort uniquely scalable to millions of elements. For $N = 1024$: Gumbel-Sinkhorn uses 1,048,576 parameters (1.0M floats); ShuffleSoftSort uses 1,024 (1024$\times$ reduction).

**Experimental results ($N = 1024$ RGB colors):**

| Method | Memory | Runtime (s) | Quality (DPQ$_{16}$) $\uparrow$ |
|--------|--------|------------|-------------------------------|
| Gumbel-Sinkhorn | 1,048,576 | 226.8 | 0.913 |
| Kissing | 26,624 | 114.4 | invalid$^*$ |
| SoftSort | 1,024 | 110.7 | 0.698 |
| **ShuffleSoftSort** | **1,024** | **98.0** | **0.892** |

## Applicability

- **Large-scale channel permutation:** For LLMs with very large hidden dimensions ($C_\text{in} \geq 8192$), even blockwise approaches with $B = 64$ require $C_\text{in} \times 64 = 524{,}288$ parameters per layer. ShuffleSoftSort needs only $C_\text{in}$ parameters, enabling full global permutation search without blockwise restrictions.
- **Cross-block permutation exploration:** Unlike blockwise Sinkhorn which restricts moves to within blocks, ShuffleSoftSort's random shuffling can discover beneficial cross-block channel reorderings, potentially capturing global structure that block-diagonal approaches miss.
- **3D Gaussian Splatting compression:** Self-Organizing Gaussians (SOG) sort millions of Gaussian splat attributes into 2D grids for image-codec compression. ShuffleSoftSort enables gradient-based end-to-end optimization of the sort order at scale ($N > 10^6$), achieving up to 40$\times$ storage reduction.
- **Visual data organization:** Grid-based image sorting for browsing/retrieval, where $N$ can be hundreds or thousands of images sorted by visual similarity.
- **Composable with N:M pruning:** The learned permutation from ShuffleSoftSort can be hardened via argsort and applied as channel permutation before N:M mask computation, with the same zero-overhead inference as other channel permutation methods.

## Limitations

- The $O(N^2)$ distance matrix computation per SoftSort call means each iteration still has quadratic compute cost, even though only $N$ parameters are stored — the parameter savings do not translate to proportional compute savings
- Quality (DPQ$_{16} = 0.892$) slightly below Gumbel-Sinkhorn (0.913) despite much better efficiency — the iterative shuffling is an approximation to full permutation optimization
- Random shuffling introduces stochasticity in the optimization trajectory; convergence quality depends on the number of shuffle iterations $R$ and inner steps $I$
- SoftSort's row-wise softmax over L1 distances is less theoretically grounded than Sinkhorn's convergence to the Birkhoff polytope — the soft matrix may not be exactly doubly stochastic
- Not yet validated for channel permutation in LLM pruning specifically — demonstrated on grid sorting and 3D Gaussian splatting; adaptation to N:M sparsity optimization requires a suitable saliency-based loss function
- The alternating horizontal/vertical sorting strategy for 2D grids may not directly generalize to higher-dimensional permutation structures

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def softsort(w, tau=1.0):
    """
    SoftSort: continuous relaxation of argsort.

    Args:
        w: (N,) learnable weight vector
        tau: temperature (lower = sharper)

    Returns:
        P_soft: (N, N) approximate permutation matrix
    """
    w_sorted, _ = torch.sort(w)
    # L1 distance matrix between sorted and unsorted
    D = torch.abs(w_sorted.unsqueeze(1) - w.unsqueeze(0))  # (N, N)
    P_soft = F.softmax(-D / tau, dim=-1)
    return P_soft

def shuffle_softsort(x, loss_fn, N, R=100, I=4,
                     tau_start=1.0, tau_end=0.1):
    """
    ShuffleSoftSort: iterative shuffled permutation learning.

    Args:
        x: (N, d) input vectors to sort/permute
        loss_fn: differentiable loss on permuted output
        N: number of elements
        R: outer iterations (shuffle rounds)
        I: inner SoftSort optimization steps per shuffle
        tau_start, tau_end: temperature schedule

    Returns:
        x_sorted: (N, d) optimally permuted output
    """
    for r in range(R):
        # Temperature annealing (geometric)
        tau = tau_start * (tau_end / tau_start) ** (r / R)

        # Initialize weights linearly
        w = torch.arange(N, dtype=torch.float32, requires_grad=True)

        # Random shuffle
        shuf_idx = torch.randperm(N)
        x_shuf = x[shuf_idx]

        for i in range(I):
            # Inner temperature schedule
            tau_i = 0.2 * tau + (tau - 0.2 * tau) * i / I

            # Compute soft permutation
            P_soft = softsort(w, tau_i)

            # Apply permutation to shuffled input
            x_sort_soft = P_soft @ x_shuf

            # Reverse shuffle for loss computation
            x_output = torch.zeros_like(x_sort_soft)
            x_output[shuf_idx] = x_sort_soft

            # Backprop through loss
            loss = loss_fn(x_output)
            loss.backward()
            # Update w (only N parameters!)
            with torch.no_grad():
                w -= lr * w.grad
                w.grad.zero_()

        # Extract hard permutation
        sort_idx = P_soft.argmax(dim=-1)
        x_sorted = x_shuf[sort_idx]

    return x_sorted

# For channel permutation in N:M sparsity:
# x = weight columns, loss_fn = saliency-after-pruning metric
# The N-parameter representation makes this feasible for
# C_in >= 8192 without block decomposition.
#
# Key insight: random shuffling explores cross-block moves
# that blockwise Sinkhorn cannot make, at the cost of
# stochastic optimization quality.
```

## References

- Barthel, K. U., Barthel, F. T. & Eisert, P. (2025). Permutation Learning with Only N Parameters: From SoftSort to Self-Organizing Gaussians. arXiv:2503.13051.
- Prillo, S. & Eisenschlos, J. M. (2020). SoftSort: A Continuous Relaxation for the argsort Operator. ICML 2020.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Droge, H., Lahner, Z., Bahat, Y., Martorell, O., Heide, F. & Moller, M. (2023). Kissing to Find a Match: Efficient Low-Rank Permutation Representation. NeurIPS 2023.
- Morgenstern, W., Barthel, F., Barthel, A., Hilsmann, A. & Eisert, P. (2025). Compact 3D Scene Representation via Self-Organizing Gaussian Grids. ECCV 2024.
- Barthel, K. U., Barthel, F. T., Eisert, P., Hezel, N. & Schall, K. (2024). Creating Sorted Grid Layouts Using Distance-Preservation Quality and Linear Assignment Sorting. ICMR 2024.

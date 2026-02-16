# 115: Sinkhorn Permutation Relaxation

**Category**: approximation
**Gain type**: flexibility
**Source**: Sinkhorn (1964), Adams & Zemel (2011), Mena et al. (2018) — Gumbel-Sinkhorn
**Paper**: [papers/sinkhorn-permutation-relaxation.pdf]
**Documented**: 2026-02-11

## Description

The Sinkhorn operator provides a **differentiable relaxation** of discrete permutation matrices onto the **Birkhoff polytope** (the set of doubly stochastic matrices), enabling gradient-based optimization over latent permutations. Since permutation matrices are the vertices of the Birkhoff polytope (Birkhoff-von Neumann theorem), the Sinkhorn operator acts as a continuous, differentiable "softmax over permutations" — iteratively normalizing rows and columns of an arbitrary matrix until it becomes doubly stochastic.

Combined with the Gumbel trick, this yields the **Gumbel-Sinkhorn** distribution: a differentiable analog of the Gumbel-Softmax for permutations that converges to a true permutation matrix as temperature $\tau \to 0$. This is the standard technique for making permutation learning differentiable in neural networks.

This trick is the **complementary approach** to Householder-product parameterization: while Householder products parameterize the continuous orthogonal group $O(n)$ and reach permutations as a special case, Sinkhorn relaxation directly targets the discrete permutation group $S_n$ via its convex hull. The two approaches can be combined — e.g., OT4P (NeurIPS 2024) uses the orthogonal group as an intermediate manifold for permutation relaxation.

## Mathematical Form

**Sinkhorn Operator:**

Given a matrix $X \in \mathbb{R}^{n \times n}$, the Sinkhorn operator $S(X)$ is defined by alternating row and column normalizations:

$$
S^0(X) = \exp(X)
$$
$$
S^{l+1}(X) = \mathcal{T}_c(\mathcal{T}_r(S^l(X)))
$$

where:
- $\mathcal{T}_r(M)_{ij} = M_{ij} / \sum_k M_{ik}$ — row normalization
- $\mathcal{T}_c(M)_{ij} = M_{ij} / \sum_k M_{kj}$ — column normalization

**Key Definitions:**

- **Birkhoff polytope** $\mathcal{B}_n = \{M \in \mathbb{R}_{\geq 0}^{n \times n} : M\mathbf{1} = \mathbf{1}, M^\top\mathbf{1} = \mathbf{1}\}$ — the set of doubly stochastic matrices
- **Birkhoff-von Neumann theorem:** The vertices of $\mathcal{B}_n$ are exactly the $n!$ permutation matrices
- $\lim_{L \to \infty} S^L(X) \in \mathcal{B}_n$ — the Sinkhorn iterates converge to a doubly stochastic matrix

**Temperature-controlled convergence to permutations:**

$$
S(X / \tau) \xrightarrow{\tau \to 0} P^* = \arg\max_{P \in S_n} \langle P, X \rangle_F
$$

As temperature $\tau$ decreases, the doubly stochastic output concentrates toward the permutation matrix that maximizes the inner product with $X$.

**Gumbel-Sinkhorn distribution:**

$$
\hat{P} = S\left(\frac{X + G}{\tau}\right), \quad G_{ij} \sim \text{Gumbel}(0, 1)
$$

where $G_{ij} = -\log(-\log(U_{ij}))$ with $U_{ij} \sim \text{Uniform}(0, 1)$.

This defines a differentiable distribution over (approximate) permutation matrices, enabling:
- **Reparameterization gradients** via the Sinkhorn iterates
- **Straight-through estimation** using hard permutation in forward, soft gradient in backward

**Connection to optimal transport:**

The Sinkhorn operator solves the entropy-regularized optimal transport problem:

$$
\hat{P} = \arg\min_{P \in \mathcal{B}_n} \langle C, P \rangle_F - \varepsilon H(P)
$$

where $H(P) = -\sum_{ij} P_{ij} \log P_{ij}$ is the entropy and $\varepsilon = \tau$.

## Complexity

| Operation | Exact Permutation | Sinkhorn Relaxation |
|-----------|-------------------|---------------------|
| Optimization | NP-hard (combinatorial) | $O(Ln^2)$ for $L$ Sinkhorn iterations |
| Gradient | Not defined (discrete) | $O(Ln^2)$ via backprop through iterations |
| Storage | $O(n)$ (index array) | $O(n^2)$ (doubly stochastic matrix) |
| Sampling | $O(n \log n)$ | $O(Ln^2)$ (Gumbel-Sinkhorn) |
| Hungarian matching (exact) | $O(n^3)$ | — |

**Sinkhorn convergence:** Typically $L = 10$–$20$ iterations suffice for practical convergence.

**Memory:** $O(n^2)$ — a significant limitation for large $n$, as both the doubly stochastic matrix and the Gumbel noise require $n^2$ storage.

## Applicability

- **Differentiable sorting:** Learning to sort sequences end-to-end (NeuralSort, SoftSort)
- **Matching and alignment:** Solving assignment problems within neural networks (e.g., object tracking, graph matching)
- **Sparse Sinkhorn Attention:** Tay et al. (2020) use Sinkhorn-based learned permutations to rearrange tokens before applying block-diagonal attention, reducing attention complexity
- **Jigsaw puzzles / combinatorial tasks:** Learning to reassemble permuted inputs
- **Structured matrices:** Learning the permutation components in Monarch and Group-and-Shuffle factorizations (currently an active research direction)
- **Latent variable models:** Gumbel-Sinkhorn enables variational inference with latent permutation variables

## Limitations

- **$O(n^2)$ memory:** The doubly stochastic matrix requires $n^2$ storage, limiting scalability to large permutations
- **Not exactly a permutation:** The output is a soft approximation; rounding to a hard permutation introduces bias and requires the Hungarian algorithm ($O(n^3)$)
- **Temperature sensitivity:** Too high $\tau$ gives a uniform matrix (no information); too low $\tau$ gives near-discrete outputs with vanishing gradients
- **Local optima:** The Birkhoff polytope relaxation is convex, but the overall optimization landscape (with the rest of the neural network) is not
- **No orthogonality guarantee:** Doubly stochastic matrices are generally not orthogonal (unlike permutation matrices), so this relaxation does not preserve the orthogonal structure
- **Comparison with Householder parameterization:** Householder products offer a natural smooth path through $O(n)$ to permutations, while Sinkhorn relaxes to the convex hull — these are complementary but different geometric approaches

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def sinkhorn(log_alpha, n_iters=20, tau=1.0):
    """
    Sinkhorn operator: maps log_alpha to a doubly stochastic matrix.

    Args:
        log_alpha: (n, n) - log-scores matrix
        n_iters: number of Sinkhorn iterations
        tau: temperature parameter

    Returns:
        (n, n) - doubly stochastic matrix
    """
    log_alpha = log_alpha / tau
    for _ in range(n_iters):
        # Row normalization (in log space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization (in log space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def gumbel_sinkhorn(log_alpha, n_iters=20, tau=1.0):
    """
    Gumbel-Sinkhorn: differentiable sampling of (approximate) permutations.
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(
        torch.rand_like(log_alpha) + 1e-20) + 1e-20)
    # Add noise and apply Sinkhorn
    return sinkhorn(log_alpha + gumbel_noise, n_iters, tau)


def sinkhorn_to_permutation(soft_perm):
    """
    Project doubly stochastic matrix to nearest permutation via
    row-wise argmax (greedy, not optimal) or Hungarian algorithm.
    """
    # Greedy: row-wise argmax
    hard_perm = torch.zeros_like(soft_perm)
    indices = soft_perm.argmax(dim=-1)
    hard_perm.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return hard_perm
```

## References

- Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices. Annals of Mathematical Statistics.
- Adams, R. & Zemel, R. (2011). Ranking via Sinkhorn Propagation. arXiv:1106.1925.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Tay, Y. et al. (2020). Sparse Sinkhorn Attention. ICML 2020.
- Guo et al. (2024). OT4P: Unlocking Effective Orthogonal Group Path for Permutation Relaxation. NeurIPS 2024.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.

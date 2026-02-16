# 062: Householder Product Parameterization

**Category**: decomposition
**Gain type**: expressivity
**Source**: Mhammedi et al. (2017), Schlag et al. (2025) — DeltaProduct
**Paper**: [papers/householder-product-parameterization.pdf] (DeltaProduct), [papers/householder-orthogonal-rnn.pdf] (Mhammedi et al.)
**Documented**: 2026-02-11

## Description

Parameterize an orthogonal (or near-orthogonal) state-transition matrix as a product of $n_h$ generalized Householder reflections, each of the form $I - \beta k k^\top$. This provides a smooth, tunable trade-off between expressivity and efficiency: with $n_h = n$ reflections any orthogonal matrix can be represented (Cartan-Dieudonné theorem), while fewer reflections give a structured subset with lower computational cost. In the context of linear RNNs (DeltaProduct), taking $n_h$ gradient-descent steps per token naturally yields state-transition matrices formed as products of $n_h$ generalized Householder transformations, enabling permutation simulation and state-tracking with controllable complexity.

## Mathematical Form

**Core Operation:**

$$
A(\mathbf{x}_i) = \prod_{j=1}^{n_h} \left( I - \beta_{i,j} \, \mathbf{k}_{i,j} \mathbf{k}_{i,j}^\top \right)
$$

where each factor is a generalized Householder transformation with:
- $\beta_{i,j} \in [0, 2]$ — learnable scalar (when $\beta = 2$ and $\|\mathbf{k}\| = 1$, this is a standard Householder reflection)
- $\mathbf{k}_{i,j} \in \mathbb{R}^d$ — normalized direction vector, $\|\mathbf{k}_{i,j}\| = 1$

**Key Definitions:**

- $A \in \mathbb{R}^{d \times d}$ — state-transition matrix, structured as a product of Householder factors
- $n_h$ — number of Householder reflections (tunable hyperparameter)
- $I - \beta k k^\top$ — generalized Householder transformation (reflection when $\beta = 2$)

**Properties of Individual Factors:**

Each factor $H_j = I - \beta_j k_j k_j^\top$ satisfies:
- Eigenvalues: $\{1 - \beta_j\}$ (multiplicity 1, along $k_j$) and $\{1\}$ (multiplicity $d-1$)
- $\|H_j\|_2 \leq 1$ when $\beta_j \in [0, 2]$, ensuring stability
- When $\beta_j = 2$: $H_j^2 = I$ (involution), $H_j = H_j^\top$ (symmetric), $H_j H_j^\top = I$ (orthogonal)

**Product Structure (DeltaProduct):**

The product $A = \prod_{j=1}^{n_h} H_j$ can be written as:

$$
A = I + W R
$$

where $W \in \mathbb{R}^{d \times n_h}$ and $R \in \mathbb{R}^{n_h \times d}$, i.e., identity plus a rank-$n_h$ perturbation.

**Input Term:**

$$
B(\mathbf{x}_i) = \sum_{j=1}^{n_h} \left(\prod_{k=j+1}^{n_h} H_{i,k}\right) \beta_{i,j} \mathbf{k}_{i,j} \mathbf{v}_{i,j}^\top
$$

**Special Cases of the Product:**

- **Identical direction vectors:** Product collapses to a single transformation
- **Orthogonal direction vectors:** Matrix becomes symmetric with purely real eigenvalues $\{1-\beta_1, \ldots, 1-\beta_{n_h}, 1, \ldots, 1\}$
- **Non-orthogonal (linearly dependent) vectors:** Permits complex eigenvalues, enabling rotations beyond pure reflections

## Complexity

| Operation | Dense Matrix | Householder Product ($n_h$ reflections) |
|-----------|-------------|----------------------------------------|
| Parameterization | $O(d^2)$ params | $O(d \cdot n_h)$ params |
| Matrix-vector product $Ax$ | $O(d^2)$ | $O(d \cdot n_h)$ |
| Training (per token) | $O(d^2)$ | $O(d \cdot n_h)$ per step, linear in $n_h$ |
| Full orthogonal group | $O(d^2)$ | $n_h = d$ reflections needed |

**Memory:** $O(d \cdot n_h)$ parameters vs $O(d^2)$ for a dense matrix

**Key trade-off:** Training/prefill time increases linearly with $n_h$; spectral norm $\leq 1$ is automatically guaranteed.

## Applicability

- **Linear RNNs / SSMs:** DeltaProduct uses this to form state-transition matrices with tunable expressivity
- **Orthogonal RNNs:** Mhammedi et al. (2017) used products of Householder reflections to parameterize orthogonal transition matrices, preventing exploding/vanishing gradients
- **State-tracking tasks:** DeltaProduct with $n_h$ reflections can simulate permutations of up to $(n_h + 1)$ elements in a single layer
  - $S_3$ (3-element permutations): requires $n_h = 2$
  - $S_5$ (5-element permutations): requires $n_h = 4$
  - $S_4$ exploits isomorphism to subgroups of $SO(3, \mathbb{R})$, needing only $n_h = 2$
- **Group word problems:** 1 layer with $n_h = n-1$, or 3 layers with $n_h > 1$, or 4 layers with $n_h = 1$
- **Regular languages:** For any $n_h \geq 1$, a finite number of Gated DeltaProduct layers can recognize any regular language

## Limitations

- Training cost scales linearly with $n_h$ — more reflections means proportionally more compute
- With $n_h \ll d$, only a structured subset of orthogonal matrices is reachable
- Non-orthogonal direction vectors needed for complex eigenvalues (rotations), which may be harder to learn
- The WY representation is needed for efficient chunkwise parallel training (see [wy-representation](wy-representation.md))
- Unlike diagonal state-transition matrices, does not admit element-wise parallel scans

## Implementation Notes

```python
import torch

def householder_product(keys, betas, x):
    """
    Apply product of n_h generalized Householder reflections to vector x.

    Args:
        keys: (n_h, d) - normalized direction vectors
        betas: (n_h,) - scalar coefficients in [0, 2]
        x: (d,) - input vector

    Returns:
        (d,) - A @ x where A = prod_j (I - beta_j k_j k_j^T)
    """
    result = x
    for j in range(keys.shape[0]):
        k_j = keys[j]
        beta_j = betas[j]
        # (I - beta * k k^T) x = x - beta * k * (k^T x)
        result = result - beta_j * k_j * torch.dot(k_j, result)
    return result


def householder_product_matrix(keys, betas):
    """
    Materialize the full product matrix (for analysis/debugging).
    Cost: O(d^2 * n_h)
    """
    d = keys.shape[1]
    A = torch.eye(d)
    for j in range(keys.shape[0]):
        k = keys[j].unsqueeze(1)  # (d, 1)
        H_j = torch.eye(d) - betas[j] * k @ k.T
        A = H_j @ A
    return A
```

## References

- Mhammedi, Hellicar, Rahman, Bailey (2017). Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections. ICML 2017.
- Schlag, Irie, Schmidhuber (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. ICLR 2025.
- Cartan (1938), Dieudonné (1940). The Cartan-Dieudonné theorem on decomposition of orthogonal transformations into reflections.
- Yang et al. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.

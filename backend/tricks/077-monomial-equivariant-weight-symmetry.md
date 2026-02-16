# 077: Monomial Matrix Group Equivariant Weight-Space Layers

**Category**: algebraic
**Gain type**: efficiency
**Source**: Tran, Vo, Tran-Huu, Nguyen & Nguyen (NeurIPS 2024)
**Paper**: [papers/monomial-nfn-equivariant.pdf] (Monomial Matrix Group Equivariant Neural Functional Networks)
**Documented**: 2026-02-15

## Description

Neural Functional Networks (NFNs) process the **weights** of neural networks as data — predicting generalization, editing weight spaces, or classifying implicit neural representations. The key insight of this trick is that the **symmetry group** acting on weight spaces of neural networks is not just the permutation group $S_n$ (neuron reordering) but the **monomial matrix group** $\mathcal{G}_n$, which includes both permutation and scaling/sign-flipping symmetries. For ReLU networks, this is the group $\mathcal{G}_n^{>0} = \Delta_n^{>0} \rtimes \mathcal{P}_n$ (positive scaling + permutations); for sin/tanh networks, it is $\mathcal{G}_n^{\pm 1} = \Delta_n^{\pm 1} \rtimes \mathcal{P}_n$ (sign-flipping + permutations) — the latter being precisely the **hyperoctahedral group** $B_n$.

By designing layers that are **equivariant to the full monomial matrix group** rather than just the permutation subgroup, the number of independent trainable parameters drops dramatically: from $O(cc'L^2)$ (permutation-only) to $O(cc'(L + n_0 + n_L))$ (monomial group), where $L$ is the number of network layers and $c, c'$ are channel/width dimensions. This is because the larger symmetry group creates more orbits, forcing more parameter sharing and zeroing out more degrees of freedom in the equivariant linear maps.

The construction separates into: (1) a **diagonal scaling-invariant** map $I_{\Delta^{>0}}$ using positively homogeneous functions of degree zero, and (2) a **permutation-equivariant** map $I_\mathcal{P}$ using row/column summation/averaging. Their composition, followed by an MLP, yields a $G$-invariant functional. For equivariant layers, parameter-sharing constraints derived from the monomial group action yield a linear map with parameters that are **linear** in $L, n_0, n_L$ rather than quadratic.

## Mathematical Form

**Weight Space of an $L$-layer FCNN:**

$$
\mathcal{U} = \mathcal{W} \times \mathcal{B}, \quad \mathcal{W} = \mathbb{R}^{n_L \times n_{L-1}} \times \cdots \times \mathbb{R}^{n_1 \times n_0}, \quad \mathcal{B} = \mathbb{R}^{n_L \times 1} \times \cdots \times \mathbb{R}^{n_1 \times 1}
$$

A weight-space element is $U = (W, b)$ where $W = \{W^{(i)} \in \mathbb{R}^{n_i \times n_{i-1}}\}_{i=1}^L$ and $b = \{b^{(i)} \in \mathbb{R}^{n_i}\}_{i=1}^L$.

**Monomial Matrix Group Action on Weight Spaces:**

The group acting on $\mathcal{U}$ is:

$$
\mathcal{G}_\mathcal{U} = \mathcal{G}_{n_L} \times \mathcal{G}_{n_{L-1}} \times \cdots \times \mathcal{G}_{n_1} \times \mathcal{G}_{n_0}
$$

where each $g^{(i)} = D^{(i)} \cdot P_{\pi_i} \in \mathcal{G}_{n_i}$ is a monomial matrix (diagonal $\times$ permutation). The group element $g = (g^{(L)}, \ldots, g^{(0)})$ acts on weights and biases as:

$$
(gW)^{(i)} = g^{(i)} \cdot W^{(i)} \cdot \left(g^{(i-1)}\right)^{-1}, \quad (gb)^{(i)} = g^{(i)} \cdot b^{(i)}
$$

In coordinates:

$$
(gW)^{(i)}_{jk} = \frac{d^{(i)}_j}{d^{(i-1)}_k} \cdot W^{(i)}_{\pi_i^{-1}(j), \pi_{i-1}^{-1}(k)}, \quad (gb)^{(i)}_j = d^{(i)}_j \cdot b^{(i)}_{\pi_i^{-1}(j)}
$$

**Key Symmetry Theorem (Proposition 4.4):**

For an FCNN or CNN $f(\mathbf{x}; U, \sigma)$ with activation $\sigma$:

- If $\sigma = \text{ReLU}$: $G = \{\text{id}\} \times \mathcal{G}_{n_{L-1}}^{>0} \times \cdots \times \mathcal{G}_{n_1}^{>0} \times \{\text{id}\}$ preserves $f$.
- If $\sigma = \sin$ or $\tanh$: $G = \{\text{id}\} \times \mathcal{G}_{n_{L-1}}^{\pm 1} \times \cdots \times \mathcal{G}_{n_1}^{\pm 1} \times \{\text{id}\}$ preserves $f$.

That is, $f(\mathbf{x}; U, \sigma) = f(\mathbf{x}; gU, \sigma)$ for all $g \in G$ and $U \in \mathcal{U}$. Moreover, $G$ is **maximal** (no larger monomial subgroup preserves $f$).

**Connection to Hyperoctahedral Group:**

For $\sin/\tanh$ networks, the sign-flipping symmetry group at each hidden layer is:

$$
\mathcal{G}_n^{\pm 1} = \Delta_n^{\pm 1} \rtimes_\varphi \mathcal{P}_n = B_n \quad \text{(hyperoctahedral group)}
$$

where $\Delta_n^{\pm 1} = \{D = \text{diag}(d_1, \ldots, d_n) : d_i \in \{-1, +1\}\} \cong \mathbb{Z}_2^n$ and $\mathcal{P}_n \cong S_n$.

**Equivariant Linear Layer:**

A $G$-equivariant linear map $E: \mathcal{U} \to \mathcal{U}'$ satisfying $E(gU) = gE(U)$ for all $g \in G$ has the form $E(U) = \mathfrak{a} \cdot \text{vec}(U) + \mathfrak{b}$ where the parameter-sharing constraints from monomial equivariance yield:

$$
W'^{(1)}_{jk} = \sum_{q=1}^{n_0} \mathfrak{p}^{1jk}_{1jq} W^{(1)}_{jq} + \mathfrak{a}^{1jk}_{1j} b^{(1)}_j
$$

$$
W'^{(i)}_{jk} = \mathfrak{p}^{ijk}_{1jk} W^{(i)}_{jk}, \quad 1 < i < L
$$

$$
W'^{(L)}_{jk} = \sum_{p=1}^{n_L} \mathfrak{p}^{Ljk}_{Lpk} W^{(L)}_{pk}
$$

**Invariant Layer Construction:**

The $G$-invariant map $I: \mathcal{U} \to \mathbb{R}^d$ decomposes as:

$$
I = \text{MLP} \circ I_\mathcal{P} \circ I_{\Delta^{>0}}
$$

where:
- $I_{\Delta^{>0}}$: applies positively homogeneous degree-zero functions $\alpha^{(i)}_{jk}$ to each weight/bias entry, achieving $\Delta^{>0}$-invariance
- $I_\mathcal{P}$: sums/averages over rows or columns of each weight matrix, achieving $\mathcal{P}$-invariance:

$$
I_\mathcal{P}(U) = \left(W^{(1)}_{\star, :}, W^{(L)}_{:, \star}, W^{(2)}_{\star, \star}, \ldots, W^{(L-1)}_{\star, \star}; v^{(L)}, v^{(1)}_\star, \ldots, v^{(L-1)}_\star\right)
$$

where $\star$ denotes summation/averaging over that index.

## Complexity

| Aspect | Permutation-Only NFN | Monomial-NFN |
|--------|---------------------|--------------|
| Equivariant layer params | $O(cc'L^2)$ | $O(cc'(L + n_0 + n_L))$ |
| Invariant layer params | $O(cc'L^2)$ | $O(cc'(L + n_0 + n_L))$ |
| Symmetry group size at layer $i$ | $n_i!$ | $2^{n_i} \cdot n_i!$ (sign-flip) or $\mathbb{R}_{>0}^{n_i} \cdot n_i!$ (scaling) |
| Parameters of weight in HNP baseline | $O(cc'(L + n_0 + n_L)^2)$ | $O(cc'(L + n_0 + n_L))$ |

**Memory:** The parameter count for Monomial-NFN equivariant layers is **linear** in $L, n_0, n_L$, compared to **quadratic** in prior work (Table 1 in paper). This is a direct consequence of the larger symmetry group creating more parameter-sharing constraints.

**Key efficiency gain:** Using only ~50% of parameters compared to the best permutation-equivariant baselines while achieving competitive or better performance.

## Applicability

- **Neural functional networks:** Processing weight spaces of trained neural networks for tasks like predicting generalization, weight-space style editing, and classifying implicit neural representations (INRs).
- **Hyperoctahedral equivariance for $\tanh$/$\sin$ networks:** The sign-flipping symmetry group $\mathcal{G}_n^{\pm 1} = B_n$ is the natural symmetry group for networks with odd activations, and designing layers equivariant to $B_n$ is more parameter-efficient than equivariance to $S_n$ alone.
- **Weight-space meta-learning:** Any meta-learning system that processes network weights as input can benefit from exploiting monomial matrix symmetries to reduce model size.
- **Structured pruning and model compression:** Understanding that neural networks have monomial matrix symmetries can inform which weights are truly redundant (related by symmetry transformations).
- **Equivariant architectures generally:** The decomposition $I = \text{MLP} \circ I_\mathcal{P} \circ I_\Delta$ provides a template for building equivariant/invariant layers for any group that decomposes as a semidirect product $\Delta \rtimes \mathcal{P}$.

## Limitations

- The large symmetry group means equivariant **linear** layers have limited expressivity — each weight is updated based only on its own value, not on other weights in the same layer (except at the first and last layers). Nonlinear equivariant layers are needed to capture cross-weight interactions.
- The framework assumes fixed activation function (all ReLU or all $\tanh$); mixed-activation networks require different symmetry analysis.
- Maximality of $G$ is only proven for specific network architectures (decreasing widths for ReLU, and $\tanh$); the question remains open for general architectures.
- Fourier Features (commonly used in NFNs for positional encoding) break $\Delta^{>0}$-invariance, so they cannot be used in the equivariant part of Monomial-NFNs — only after the invariant layer.
- The approach is specific to processing weight spaces; it does not directly apply to the data/activation spaces of the networks being analyzed.

## Implementation Notes

```python
import torch
import torch.nn as nn

class MonomialEquivariantLayer(nn.Module):
    """
    G-equivariant linear layer for weight spaces under monomial matrix group.

    For interior layers (1 < i < L), the equivariant map is diagonal:
        W'^{(i)}_{jk} = p^{ijk} * W^{(i)}_{jk}

    For first layer (i=1), it mixes across input dimension:
        W'^{(1)}_{jk} = sum_q p^{1jk}_{1jq} * W^{(1)}_{jq} + a^{1jk}_{1j} * b^{(1)}_j

    For last layer (i=L), it mixes across output dimension:
        W'^{(L)}_{jk} = sum_p p^{Ljk}_{Lpk} * W^{(L)}_{pk}

    This gives O(L + n_0 + n_L) parameters instead of O(L^2).
    """

    def __init__(self, num_layers, hidden_dims, n_0, n_L):
        super().__init__()
        self.num_layers = num_layers
        self.n_0 = n_0
        self.n_L = n_L

        # Interior layers: one scalar parameter per layer per (j,k) pattern
        # In practice, shared across j,k → one param per interior layer
        self.interior_scales = nn.Parameter(torch.ones(num_layers - 2))

        # First layer: mixing across input dimension
        self.first_layer_weight = nn.Parameter(torch.eye(n_0))  # n_0 x n_0
        self.first_layer_bias_mix = nn.Parameter(torch.zeros(n_0))

        # Last layer: mixing across output dimension
        self.last_layer_weight = nn.Parameter(torch.eye(n_L))  # n_L x n_L

    def forward(self, weights, biases):
        """
        Args:
            weights: list of L weight matrices [W^(1), ..., W^(L)]
            biases: list of L bias vectors [b^(1), ..., b^(L)]

        Returns:
            Transformed weights and biases
        """
        new_weights = []
        new_biases = []

        for i in range(self.num_layers):
            if i == 0:
                # First layer: mix across input dimension
                # W'^(1)_{jk} = sum_q p_{jq} W^(1)_{jq} + a_j b^(1)_j
                W = weights[i]  # (n_1, n_0)
                b = biases[i]  # (n_1,)
                W_new = W @ self.first_layer_weight + \
                        b.unsqueeze(1) * self.first_layer_bias_mix.unsqueeze(0)
                new_weights.append(W_new)
                new_biases.append(b.clone())

            elif i == self.num_layers - 1:
                # Last layer: mix across output dimension
                # W'^(L)_{jk} = sum_p p_{pk} W^(L)_{pk}
                W = weights[i]  # (n_L, n_{L-1})
                W_new = self.last_layer_weight @ W
                new_weights.append(W_new)
                new_biases.append(biases[i].clone())

            else:
                # Interior layers: diagonal scaling
                # W'^(i)_{jk} = scale_i * W^(i)_{jk}
                scale = self.interior_scales[i - 1]
                new_weights.append(scale * weights[i])
                new_biases.append(scale * biases[i])

        return new_weights, new_biases


class ScaleInvariantMap(nn.Module):
    """
    Delta^{>0}-invariant map using positively homogeneous degree-zero functions.

    For each weight entry W_{jk}, applies alpha(W_{jk}) where alpha is
    positively homogeneous of degree zero (e.g., sign function, or
    normalization by row/column norm).
    """

    def __init__(self, method='sign'):
        super().__init__()
        self.method = method

    def forward(self, weights, biases):
        if self.method == 'sign':
            # sign(x) is positively homogeneous of degree 0
            return [torch.sign(W) for W in weights], \
                   [torch.sign(b) for b in biases]
        elif self.method == 'normalize':
            # x / ||x|| is positively homogeneous of degree 0
            return [W / (W.norm(dim=1, keepdim=True) + 1e-8) for W in weights], \
                   [b / (b.norm() + 1e-8) for b in biases]


class PermutationInvariantPool(nn.Module):
    """
    P-invariant map via row/column averaging.
    I_P(U) = (W^(1)_{*,:}, W^(L)_{:,*}, W^(2)_{*,*}, ...; v^(L), v^(1)_*, ...)
    where * denotes averaging over that axis.
    """

    def forward(self, weights, biases):
        features = []

        # First layer: average over rows (permuted axis)
        features.append(weights[0].mean(dim=0))  # (n_0,)

        # Last layer: average over columns (permuted axis)
        features.append(weights[-1].mean(dim=1))  # (n_L,)

        # Interior layers: average over both axes → scalar per layer
        for W in weights[1:-1]:
            features.append(W.mean().unsqueeze(0))  # (1,)

        # Biases: last layer bias directly, others averaged
        features.append(biases[-1])  # (n_L,)
        for b in biases[:-1]:
            features.append(b.mean().unsqueeze(0))  # (1,)

        return torch.cat(features)
```

## References

- Tran, V.-H., Vo, T.N., Tran-Huu, T., Nguyen, A.T. & Nguyen, T.M. (2024). Monomial Matrix Group Equivariant Neural Functional Networks. *NeurIPS 2024*. arXiv:2409.11697.
- Zhou, A. et al. (2024). Permutation equivariant neural functionals. *NeurIPS 2023*.
- Zhou, A. et al. (2024). Neural functional transformers. *NeurIPS 2024*.
- Hecht-Nielsen, R. (1990). On the algebraic structure of feedforward network weight spaces. *Advanced Neural Computers*, Elsevier.
- Wikipedia: Hyperoctahedral group. https://en.wikipedia.org/wiki/Hyperoctahedral_group
- GitHub: https://github.com/MathematicalAI-NUS/Monomial-NFN

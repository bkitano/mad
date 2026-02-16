# 056: Group Matrix Displacement for Approximate Equivariance

**Category**: algebraic
**Gain type**: flexibility
**Source**: Samudre, Petrache, Nord & Trivedi (AISTATS 2025); Gader (1990); Waterhouse (1992); Thomas, Gu, Dao et al. (NeurIPS 2018)
**Paper**: [papers/symmetry-based-structured-group-matrices.pdf] (Samudre et al. — Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks)
**Documented**: 2026-02-15

## Description

**Group matrices (GMs)** generalize circulant matrices from cyclic groups to arbitrary finite groups, providing a structured matrix class that naturally encodes group convolution. Just as circulant matrices encode convolutions on $\mathbb{Z}/n\mathbb{Z}$ and are a subclass of Low Displacement Rank (LDR) matrices, group matrices encode convolutions on a general finite group $G$ and generalize the classical displacement rank framework.

The key insight is that any $|G| \times |G|$ matrix $M$ can be uniquely decomposed in the **group diagonal basis**:

$$M = \sum_{g \in G} \text{diag}(F_g) B_g$$

where $B_g$ is the **group diagonal** (a permutation matrix with exactly one nonzero entry per row, encoding multiplication by $g$), and $F_g$ are diagonal coefficient vectors. When only finitely many $F_g$ are nonzero, $M$ is a sparse group matrix encoding a group convolution with a small kernel. By allowing $M$ to deviate from being a perfect group matrix (measured by displacement rank), one obtains **approximately equivariant** layers with controlled error — crucial because real-world symmetries are rarely exact.

This framework is directly applicable to the hyperoctahedral group $B_n = \mathbb{Z}_2 \wr S_n$: one constructs group matrices for $B_n$ via Kronecker products of group matrices for the component groups ($\mathbb{Z}_2$ and $S_n$), enabling efficient approximately $B_n$-equivariant neural network layers with 1–2 orders of magnitude fewer parameters than dense layers.

## Mathematical Form

**Core Definition — Group Diagonal:**

For a finite group $G$ with $N = |G|$, and element $g \in G$, the **group diagonal** $B_g \in \mathbb{R}^{N \times N}$ is defined by:

$$
(B_g)_{h,h'} = \delta(h = gh')
$$

i.e., $B_g$ is a permutation matrix corresponding to left-multiplication by $g$. Group diagonals have exactly one nonzero entry per row and column.

**Group Convolution via Group Matrices:**

Standard group convolution of $\phi, \psi: G \to \mathbb{R}$:

$$
(\phi \star \psi)(x) = \sum_{g \in G} \phi(g) \psi(g^{-1}x)
$$

Re-expressed as a matrix–vector product using group diagonals:

$$
\text{Conv}_\psi \vec{\phi} := \vec{\phi} \star \vec{\psi} = \sum_{g \in G} \phi(g) B_g \vec{\psi}
$$

where $\vec{\phi}, \vec{\psi} \in \mathbb{R}^{|G|}$ are the vectorized functions.

**Group Diagonal Basis Decomposition:**

Any matrix $M \in \mathbb{R}^{N \times N}$ can be written as:

$$
M = \sum_{g \in G} \text{diag}(F_g) B_g, \quad \text{where } (F_g)_h := M_{h, hg^{-1}}
$$

The coefficients $F(M)$ (the $|G| \times |G|$ matrix with rows $F_g$) are obtained by $F(M) = M \vec{v}$ (shuffling entries of $M$).

**Displacement Operator for General Groups:**

$$
\text{D}(M) = \text{D}_P(M) := F(M) - P F(M)
$$

where $P(x_1, \ldots, x_N) = (x_2, \ldots, x_N, x_1)$ is a cyclic permutation. Then:

- $M$ is a group matrix (exact group convolution) $\iff$ $\text{D}(M) = 0$
- **Displacement rank:** $\text{DR}(\mathcal{M}) := \text{rank}(\text{D}(M))$

More generally, for each $g \in G$, one selects a cyclic permutation $\sigma_g \in \text{Perm}(G)$ and defines:

$$
[\text{D}_{\vec{P}}(M)]_{g,g'} := [F(M)]_{g,g'} - [F(M)]_{g, \sigma_g(g')}
$$

The **displacement dimension** is:

$$
\text{dim}_\text{D}(\mathcal{M}) := \dim_\mathbb{R}(\text{Span}(\{\text{D}(M) : M \in \mathcal{M}\}))
$$

**Low Displacement Rank Implementation:**

To allow controlled deviation from exact equivariance, parameterize:

$$
F(M) = \mathbf{1} \otimes \mathbf{b} + \sum_{i=1}^{r} \mathbf{a}_i \otimes \mathbf{1}
$$

where $\mathbf{b} \in \mathbb{R}^{|G|}$ encodes the base group convolution kernel and $\mathbf{a}_1, \ldots, \mathbf{a}_r \in \mathbb{R}^{|G|}$ are learnable perturbation vectors. This gives $\text{DR}(\mathcal{M}) \leq r$ and the parameter space has dimension $|G| \cdot r$.

**Constructing Group Matrices for Product Groups:**

For $G \times H$ (direct product): $B_{(g,h)}^{G \times H} = B_g^G \otimes B_h^H$

For $G \rtimes_\phi H$ (semi-direct product): $B_{(g,h)}^{G \rtimes_\phi H} = (P_h B_g^G) \otimes B_h^H$

where $P_h$ is the $|G| \times |G|$ permutation matrix induced by $\phi_h$.

**Application to $B_n = \mathbb{Z}_2 \wr S_n$:**

Group matrices for $B_n$ are constructed via the semi-direct product formula above, using $G = \mathbb{Z}_2^n$ and $H = S_n$ with the natural permutation action.

## Complexity

| Operation | Dense Layer | Group Matrix (exact) | GM + Displacement Rank $r$ |
|-----------|------------|---------------------|---------------------------|
| Parameters | $N^2$ | $N_k$ (kernel size) | $N_k + N \cdot r$ |
| Mat-vec | $O(N^2)$ | $O(N_k \cdot N)$ | $O((N_k + r) \cdot N)$ |
| Equivariance error | Arbitrary | $0$ (exact) | $O(r / N)$ (controlled) |

where $N = |G|$ and $N_k$ is the number of group elements within kernel radius $k$.

**Typical parameter savings:** For symmetry group $G = C_N \times C_N$ with neighborhood $k=1$ and displacement rank $r$: $(2k+1)^2 + Nr$ parameters vs. $N^2$ dense — often **1–2 orders of magnitude fewer** parameters.

**Memory:** $O(N_k + Nr)$ vs. $O(N^2)$.

## Applicability

- **Approximately equivariant CNNs (GM-CNNs):** Replace dense or circulant weight matrices with group matrices for any discrete symmetry group, achieving competitive accuracy with drastically reduced parameter counts
- **Hyperoctahedral equivariance:** For data with signed permutation symmetries (point clouds, molecular conformations), construct $B_n$-group matrices via the Kronecker product construction
- **State-space models:** Group matrices for $B_n$ can parameterize structured transition matrices that respect signed permutation symmetry while allowing learned deviations
- **Structured weight compression:** Generalizes the LDR framework (Toeplitz, circulant, Cauchy-like) from cyclic groups to arbitrary finite groups, providing a principled compression method for any layer with known (approximate) symmetry
- **Equivariant autoencoders:** The GMPool operation (pooling via subgroup cosets) enables group-equivariant encoder-decoder architectures

## Limitations

- Group matrices require $|G|$ to be tractable — for large groups (e.g., $|B_n| = 2^n n!$), the full group matrix is impractical; must work with cosets or subgroups
- The displacement rank framework for general groups lacks the elegant spectral theory available for cyclic groups (where LDR matrices are diagonalized by the DFT)
- No inherent steerability: unlike steerable equivariant networks, GM-CNNs cannot interpolate between group elements
- Error bounds on approximate equivariance under composition (stacking layers) are not tight — Proposition 3.1 gives multiplicative bounds: $\text{dist}(MN, \mathcal{GM}) \leq \max\{\|M\|, \|N\|\} (\text{dist}(M, \mathcal{GM}) + \text{dist}(M', \mathcal{GM}))$
- Constructing group matrices for non-abelian groups requires explicit knowledge of the Cayley table, which may be expensive for large groups

## Implementation Notes

```python
import torch
import torch.nn as nn
from itertools import product as iterproduct

class GroupDiagonal:
    """
    Represents a group diagonal B_g for a finite group G.
    B_g is a permutation matrix: (B_g)_{h, h'} = δ(h = g·h').
    Stored as an index array for O(n) mat-vec.
    """
    def __init__(self, group_cayley_table, g_idx):
        # cayley_table[g][h] = g*h
        self.perm = group_cayley_table[g_idx]

    def matvec(self, x):
        return x[self.perm]


class GMConv(nn.Module):
    """
    Group-matrix convolution layer.
    Implements Conv_ψ(φ) = Σ_{g in kernel} φ(g) B_g ψ
    with optional displacement-rank perturbation for approximate equivariance.
    """
    def __init__(self, group_size, kernel_indices, displacement_rank=0):
        super().__init__()
        self.N = group_size
        self.kernel_indices = kernel_indices  # indices of g within radius k
        N_k = len(kernel_indices)

        # Kernel weights: one scalar per group element in the neighborhood
        self.kernel_weights = nn.Parameter(torch.randn(N_k))

        # Displacement rank perturbation for approximate equivariance
        self.displacement_rank = displacement_rank
        if displacement_rank > 0:
            self.perturbation = nn.Parameter(
                torch.randn(displacement_rank, group_size) * 0.01
            )

    def forward(self, x, group_diag_perms):
        """
        x: (batch, N) — function on the group
        group_diag_perms: list of permutation index arrays for B_g
        """
        out = torch.zeros_like(x)
        for i, g_idx in enumerate(self.kernel_indices):
            perm = group_diag_perms[g_idx]
            out += self.kernel_weights[i] * x[:, perm]

        # Add displacement rank perturbation (breaks exact equivariance)
        if self.displacement_rank > 0:
            # perturbation acts as additional diagonal modulation
            for a in self.perturbation:
                out = out + a.unsqueeze(0) * x

        return out


def construct_product_group_diagonals(G_perms, H_perms):
    """
    Construct group diagonals for G × H via Kronecker product:
    B_{(g,h)}^{G×H} = B_g^G ⊗ B_h^H

    Args:
        G_perms: list of permutation arrays for group G
        H_perms: list of permutation arrays for group H

    Returns:
        list of permutation arrays for G × H
    """
    nG, nH = len(G_perms[0]), len(H_perms[0])
    product_perms = []
    for g_perm in G_perms:
        for h_perm in H_perms:
            # Kronecker product of permutations
            perm = []
            for i in range(nG):
                for j in range(nH):
                    perm.append(g_perm[i] * nH + h_perm[j])
            product_perms.append(perm)
    return product_perms
```

## References

- Samudre, A., Petrache, M., Nord, B.D. & Trivedi, S. (2025). Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks. *AISTATS 2025*. PMLR 258. arXiv:2409.11772.
- Gader, P. (1990). Group matrices and their relationship to group algebras. *Linear Algebra Appl.*
- Waterhouse, W. (1992). Group matrices and related topics. *Linear Algebra Appl.*
- Chalkley, R. (1981). Information about group matrices. *Linear Algebra Appl.* 38, 121–133.
- Thomas, A., Gu, A., Dao, T., Rudra, A. & Ré, C. (2018). Learning compressed transforms with low displacement rank. *NeurIPS 2018*.
- Sindhwani, V., Sainath, T. & Kumar, S. (2015). Structured transforms for small-footprint deep learning. *NeurIPS 2015*.
- Cohen, T. & Welling, M. (2016). Group equivariant convolutional networks. *ICML 2016*.

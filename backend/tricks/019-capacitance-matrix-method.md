# 019: Capacitance Matrix Method

**Category**: decomposition
**Gain type**: efficiency
**Source**: Maxwell's Treatise on Electricity and Magnetism (1873); computational applications in domain decomposition
**Paper**: [papers/capacitance-matrix-method.pdf]
**Documented**: 2026-02-15

## Description

The capacitance matrix method is a computational technique that transforms problems involving boundary conditions on irregular domains into problems on regular domains with modified equations. Originally from electrostatics (relating charges to potentials on conductors), it has been adapted to numerical linear algebra and domain decomposition methods. The key insight is that the capacitance matrix **C** is block-diagonal, symmetric, and relates to the Sherman-Morrison-Woodbury formula, enabling efficient inversion and solution of systems that would otherwise require handling complex boundary geometries.

In the context of neural architectures, this method can be applied to state space models or attention mechanisms where boundary/interface conditions between different components need efficient handling, particularly when the system has a natural block structure with sparse coupling between blocks.

## Mathematical Form

**Core Operation:**

For a system of $N$ conductors with charges $Q_i$ and potentials $\varphi_i$:

$$
Q_i = \sum_{j=1}^{N} C_{ij}(\varphi_j - \varphi_\infty)
$$

where $\varphi_\infty$ is the reference potential at infinity.

**Key Definitions:**

- $\mathbf{C} \in \mathbb{R}^{N \times N}$ — capacitance matrix relating charges to potential differences
- $C_{ij} := -\oint_{\mathcal{S}_i} n^\alpha \epsilon_a^b (\nabla_b u_j) \, da$ — capacitance between conductors $i$ and $j$
- $u_i$ — auxiliary potential function (solution to Dirichlet problem)
- $\mathcal{S}_i$ — boundary surface of conductor $i$

**Block-Diagonal Structure:**

$$
\mathbf{C} = \begin{pmatrix}
\mathbf{C}_{11} & 0 & \cdots & 0 \\
0 & \mathbf{C}_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \mathbf{C}_{MM}
\end{pmatrix}
$$

where each block corresponds to a connected component $\Omega_k$ of the domain.

**Properties:**

- Diagonal dominance: $\Delta_k := C_{kk} - \sum_{\ell=1, \ell \neq k}^{N} |C_{k\ell}| \geq 0$
- Symmetry: $C_{ij} = C_{ji}$
- Off-diagonal elements: $C_{ij} \leq 0$ for $i \neq j$
- Positive definiteness (under appropriate conditions): $C_{ii} \geq |C_{ij}|$

**Connection to Woodbury:**

The capacitance matrix method uses the Sherman-Morrison-Woodbury formula to avoid inverting the full matrix with boundary modifications:

$$
(\mathbf{A} + \mathbf{UCV}^T)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}^T\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{A}^{-1}
$$

where $\mathbf{C}$ is the capacitance matrix (typically much smaller than $\mathbf{A}$).

## Complexity

| Operation | Naive | With Capacitance Matrix |
|-----------|-------|------------------------|
| Dense system solve | $O(n^3)$ | $O(n^2 \log n)$ (with fast Poisson solver) |
| Capacitance matrix assembly | — | $O(N^2)$ where $N \ll n$ |
| Capacitance system solve | — | $O(N^3)$ (but $N \ll n$) |
| **Total for irregular domain** | $O(n^3)$ | $O(n^2 \log n + N^3)$ |

**Memory:** $O(n)$ for regular grid + $O(N^2)$ for capacitance matrix vs $O(n^2)$ for dense irregular system

**Key efficiency:** When $N \ll n$ (few conductors/boundaries relative to grid size), the capacitance matrix is small and cheap to invert, while the bulk of the computation uses fast structured solvers.

## Applicability

- **Domain decomposition methods**: Efficiently handle interface conditions between subdomains
- **Embedding methods**: Solve irregular domain problems on regular grids (enables FFT-based fast solvers)
- **State space models**: Systems with multiple coupled components where interface coupling is sparse
- **Hierarchical architectures**: Neural networks with modular structure and limited cross-module communication
- **Attention mechanisms**: Potentially applicable to block-sparse attention with boundary terms

The method is most effective when:
1. The domain has natural block structure (connected components)
2. The number of interface/boundary conditions is small relative to domain size
3. Fast solvers are available for the "bulk" problem on regular domains

## Limitations

- Requires explicit identification of conductor/boundary structure — not automatic
- Regularization issues when boundaries touch or have cavities (rank deficiency)
- Most computational savings require fast solvers for the bulk problem (e.g., FFT for Poisson equation)
- The capacitance matrix itself can be singular if improperly regularized
- Limited to problems with separable boundary and interior contributions

## Implementation Notes

```python
# Capacitance matrix method for solving (A + UCV^T)x = b
def capacitance_solve(A_inv_fn, U, C, V, b):
    """
    Solve (A + UCV^T)x = b using capacitance matrix method

    A_inv_fn: function that computes A^{-1}y efficiently
    U, V: low-rank interface matrices (n x k)
    C: capacitance matrix (k x k), k << n
    b: right-hand side (n,)
    """
    # Step 1: Compute A^{-1}b using fast solver
    y1 = A_inv_fn(b)  # O(n log n) for FFT-based solver

    # Step 2: Compute A^{-1}U (can be precomputed)
    A_inv_U = np.column_stack([A_inv_fn(U[:, i]) for i in range(U.shape[1])])

    # Step 3: Form capacitance system: (C^{-1} + V^T A^{-1} U)
    C_inv = np.linalg.inv(C)  # O(k^3), k << n
    cap_system = C_inv + V.T @ A_inv_U  # (k x k)

    # Step 4: Solve small capacitance system
    w = np.linalg.solve(cap_system, V.T @ y1)  # O(k^3)

    # Step 5: Apply correction
    x = y1 - A_inv_U @ w

    return x
```

**Regularization strategies:**
1. Remove rows/columns for bounded domains (version 1)
2. Use charge differences instead of absolute charges (version 2)
3. Rearrange to maintain symmetry while working with charge-potential decomposition (version 3)

## References

- Maxwell, J. C. (1873). A Treatise on Electricity and Magnetism. Oxford: Clarendon Press.
- Smolić, I. & Klajn, B. (2021). Capacitance matrix revisited. arXiv:2007.10251.
- Proskurowski, W. & Widlund, O. (1980). A Finite Element-Capacitance Matrix Method for the Neumann Problem for Laplace's Equation. SIAM Journal on Scientific Computing.
- Buzbee, B. L., Golub, G. H., & Nielson, C. W. (1970). On Direct Methods for Solving Poisson's Equations. SIAM Journal on Numerical Analysis.

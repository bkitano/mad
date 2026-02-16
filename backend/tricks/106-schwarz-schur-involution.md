# 106: Schwarz–Schur Involution

**Category**: parallelization
**Gain type**: efficiency
**Source**: Wang, Abulnaga, Balbastre & Fischl (ICML 2025); builds on nested dissection (George, 1973) and multifrontal solvers (Duff & Reid, 1983)
**Paper**: [papers/schwarz-schur-involution.pdf]
**Documented**: 2026-02-15

## Description

The Schwarz–Schur involution converts a large sparse linear system $\mathbf{A}\mathbf{x} = \mathbf{b}$ into "involuted" dense tensors $(\boldsymbol{\alpha}, \boldsymbol{\beta})$ that can be batch-inverted on GPU, achieving up to 1000× speedups over SciPy and 170× over CUDA sparse solvers. The key insight is to "condense" a sparse Laplacian (or similar pixel-affinity) matrix into a compact 4D tensor that batch-wise stores Dirichlet-to-Neumann (DtN) matrices — small dense systems capturing the interface behavior of each subdomain. The sparse solve is then reduced to recursively merging pairs of dense matrices via Schur complement elimination, where the batched small dense systems are sliced and inverted in parallel using highly optimized dense GPU BLAS kernels.

The algorithm has two phases: (1) a **Schwarz step** that decomposes the domain into patches and eliminates interior variables (reducing each patch to a "wire-frame" of boundary unknowns), and (2) recursive **Schur steps** that merge adjacent subdomains by Schur-complementing out shared boundary variables. The factorization and back-substitution both flow through the same tensorized representation, yielding a direct solver that is differentiable, zero-parameter, problem-independent, and integrable into neural network pipelines.

## Mathematical Form

**Core Setup:**

For an $H \times W$ image with $n = HW$ pixels, the sparse affinity matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ encodes pixel adjacency. The domain is divided into $2^k$ patches of size $p \times p$ (e.g., $5 \times 5$).

**Schwarz Step (Interior Elimination):**

For each patch $P$ with interior nodes $\mathbf{a}$ and boundary nodes $\mathbf{r}$, eliminate interior unknowns:

$$
\mathbf{A}_{\mathbf{rr}}^{\text{reduced}} = \mathbf{A}_{\mathbf{rr}} - \mathbf{A}_{\mathbf{ra}} \mathbf{A}_{\mathbf{aa}}^{-1} \mathbf{A}_{\mathbf{ar}}
$$

This is a standard Schur complement that reduces each patch to its boundary system (DtN map).

**Schur Step (Merging Adjacent Subdomains):**

Two adjacent subdomains $P, Q$ with system matrices $\mathbf{P}, \mathbf{Q}$ are merged into a joint domain $D$. After assembling the joint system:

$$
\begin{pmatrix} \mathbf{X} & \mathbf{Y} \\ \mathbf{Z} & \mathbf{W} \end{pmatrix} \begin{pmatrix} \mathbf{x} \\ \mathbf{u}_\star \end{pmatrix} = \begin{pmatrix} \mathbf{y} \\ \mathbf{w} \end{pmatrix}
$$

where $\mathbf{u}_\star$ contains the shared interface unknowns, eliminate $\mathbf{u}_\star$ via:

$$
\mathbf{u}_\star = \mathbf{W}^{-1}(\mathbf{w} - \mathbf{Z}\mathbf{x}) \quad \text{(back-fill)}
$$

$$
\mathbf{D} := \mathbf{X} - \mathbf{Y}\mathbf{W}^{-1}\mathbf{Z}, \quad \mathbf{d} := \mathbf{y} - \mathbf{Y}\mathbf{W}^{-1}\mathbf{w}
$$

where $\mathbf{D}$ is the Schur complement of the merged system.

**Tensorized Representation:**

The linear system at the $j$-th Schur step is stored as a 4D tensor $\boldsymbol{\alpha}^{(j)}$ of size $(2^{k/2-i}, 2^{k/2-i}, b_j, b_j)$ where $b_j$ is the boundary size at level $j$. The key batch operation for merging is:

$$
\mathbf{W} := \mathbf{P}[\cdot, \cdot, \gamma, \gamma] + \mathbf{J}^T \mathbf{Q}[\cdot, \cdot, \nu, \nu] \mathbf{J}
$$

$$
\mathbf{D} := \mathbf{X} - \mathbf{Y} \mathbf{W}^{-1} \mathbf{Z}
$$

computed via `D = X - Y * torch.linalg.inv(W) * W` across all subdomains simultaneously.

**Key Definitions:**

- $\boldsymbol{\alpha}^{(j)} \in \mathbb{R}^{2^{k/2-i} \times 2^{k/2-i} \times b_j \times b_j}$ — batch of DtN matrices (left-hand sides) at Schur level $j$
- $\boldsymbol{\beta}^{(j)} \in \mathbb{R}^{2^{k/2-i} \times 2^{k/2-i} \times b_j \times 1}$ — batch of right-hand sides at level $j$
- $\boldsymbol{\chi}^{(j)}$ — solution tensor at level $j$, computed as $\boldsymbol{\chi}^{(k)} := (\boldsymbol{\alpha}^{(k)})^{-1} \boldsymbol{\beta}^{(k)}$
- $\mathbf{J}$ — reverse permutation matrix for boundary node alignment during merging
- $k$ — number of Schur steps ($2^k$ patches total)

**Forward Pass (Factorization):**

$$
\boldsymbol{\alpha}^{(0)}, \boldsymbol{\beta}^{(0)} \leftarrow \text{Schwarz}(\boldsymbol{\alpha}^{(*)}, \boldsymbol{\beta}^{(*)})
$$

$$
\text{For } i = 0, \ldots, k/2 - 1: \quad \boldsymbol{\alpha}^{(2i+1)}, \boldsymbol{\beta}^{(2i+1)} \leftarrow \text{SchurH}(\boldsymbol{\alpha}^{(2i)}, \boldsymbol{\beta}^{(2i)})
$$

$$
\boldsymbol{\alpha}^{(2i+2)}, \boldsymbol{\beta}^{(2i+2)} \leftarrow \text{SchurV}(\boldsymbol{\alpha}^{(2i+1)}, \boldsymbol{\beta}^{(2i+1)})
$$

**Backward Pass (Back-Substitution):**

$$
\boldsymbol{\chi}^{(k)} \leftarrow (\boldsymbol{\alpha}^{(k)})^{-1} \boldsymbol{\beta}^{(k)}
$$

$$
\text{For } i = k/2, \ldots, 1: \quad \boldsymbol{\chi}^{(2i-1)} \leftarrow \text{SchurV}^{-1}(\boldsymbol{\chi}^{(2i)}), \quad \boldsymbol{\chi}^{(2i-2)} \leftarrow \text{SchurH}^{-1}(\boldsymbol{\chi}^{(2i-1)})
$$

$$
\boldsymbol{\chi}^{(*)} \leftarrow \text{Schwarz}^{-1}(\boldsymbol{\chi}^{(0)})
$$

## Complexity

| Operation | Naive (SciPy/CUDA) | With Schwarz–Schur Involution |
|-----------|---------------------|-------------------------------|
| Laplacian solve ($513^2$) | 143,100 ms (SciPy) / 36,926 ms (CUDA) | 220 ms |
| Laplacian solve ($2561^2$) | 253,318 ms (SciPy) / 36,926 ms (CUDA) | 220 ms |
| Laplacian solve ($257^2$) | 16,512 ms (SciPy) / 4,710 ms (CUDA) | 36.45 ms |
| Schwarz step (interior elim.) | — | $O(n)$ total, batch of $p^2$ inversions |
| Each Schur step | — | batch of $b \times b$ dense inversions via BLAS |

**Speedup:** 60–1000× over SciPy, 40–170× over CUDA (cuDSS), on NVIDIA A6000.

**Memory:** $O(n \cdot b^2)$ where $b$ is the boundary size per patch (e.g., $b = 24$ for $5 \times 5$ patches). The largest dense matrix inverted is at most $10^3 \times 10^3$ even for $10^6$-pixel images. More memory-intensive than sparse solvers for the same problem, but fits in GPU memory for typical image sizes.

**Parallelism:** All patches at each level are processed simultaneously via batch matrix operations. Logarithmic depth: $k$ Schur steps for $2^k$ patches, each step fully batched.

## Applicability

- **Differentiable PDE layers in neural networks**: The solver is fully differentiable (gradients via adjoint method on transposed system), enabling end-to-end training of networks that include sparse linear solves (e.g., physics-informed neural networks, solver-in-the-loop architectures)
- **Generalized deconvolution**: Solving $\mathbf{A}^{-1}\mathbf{v}$ for spatially varying convolution kernels — a fundamental operation in image restoration, inverse problems, and generative models
- **Spectral methods**: Computing eigenvectors of graph Laplacians for spectral clustering, spectral neural networks, and manifold learning at interactive rates
- **Graph neural networks and mesh-based architectures**: Any architecture operating on regular grids where the interaction matrix has local sparsity (e.g., graph/mesh/FEM neural networks)
- **Newton's method and Hessian solves**: Making second-order optimization tractable inside neural network training loops by efficiently solving Hessian systems
- **Real-time physical simulation**: Heat equation, wave equation, and diffusion PDE solvers embedded in differentiable rendering or robotics pipelines

## Limitations

- **Regularity assumption**: The method is optimized for regular 2D grids (images) where patches can be uniformly tiled. Extension to arbitrary meshes/graphs requires additional work
- **Memory overhead**: The dense tensor representation is more memory-intensive than sparse storage, limiting applicability to very large problems (currently practical for images up to ~$1024 \times 1024$)
- **2D only**: Currently demonstrated for 2D problems; 3D extension is noted as future work
- **Sparsity pattern**: Requires the sparse matrix to have local connectivity (bounded stencil); does not apply to arbitrary sparse matrices without local structure
- **Boundary size growth**: As patches get larger, the boundary system grows, potentially reducing the benefit of the condensation

## Implementation Notes

```python
import torch

def schwarz_schur_involution(alpha, beta, k):
    """
    Solve sparse system via Schwarz-Schur involution.

    alpha: (2^{k/2}, 2^{k/2}, b, b) - batch of DtN matrices per patch
    beta:  (2^{k/2}, 2^{k/2}, b, 1) - batch of right-hand sides
    k:     number of Schur levels

    Key insight: all operations are batched dense linear algebra
    on 4D tensors — no sparse operations at all.
    """
    # Forward pass: factorization
    a, b_rhs = schwarz_step_forward(alpha, beta)  # eliminate interiors

    for i in range(k // 2):
        a, b_rhs = schur_step_horizontal(a, b_rhs)  # merge pairs along x
        a, b_rhs = schur_step_vertical(a, b_rhs)    # merge pairs along y

    # Solve the coarsest level (single small dense system)
    chi = torch.linalg.solve(a, b_rhs)  # or torch.linalg.inv(a) @ b_rhs

    # Backward pass: back-substitution
    for i in range(k // 2 - 1, -1, -1):
        chi = schur_step_vertical_back(chi)
        chi = schur_step_horizontal_back(chi)

    chi = schwarz_step_backward(chi)
    return chi

def schur_merge(P, Q, p, q, J):
    """
    Merge two adjacent subdomains via Schur complement.

    P, Q: system matrices of two subdomains (batched)
    p, q: right-hand sides
    J:    reverse permutation for boundary alignment

    Returns: merged system (D, d)
    """
    # Assemble X, Y, Z, W from P, Q subblocks
    # (see paper Eq. 9-12 for precise index slicing)
    W = P[:, :, gamma, gamma] + J.T @ Q[:, :, nu, nu] @ J
    X = ...  # assembled from P, Q subblocks
    Y = ...  # assembled from P, Q subblocks
    Z = ...  # assembled from P, Q subblocks

    # Schur complement — the core operation, fully batched
    W_inv = torch.linalg.inv(W)
    D = X - Y @ W_inv @ Z
    d = y - Y @ W_inv @ w

    return D, d
```

## References

- Wang, Y., Abulnaga, S. M., Balbastre, Y., & Fischl, B. (2025). Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267.
- George, A. (1973). Nested dissection of a regular finite element mesh. *SIAM Journal on Numerical Analysis*, 10(2), 345–363.
- Duff, I. S. & Reid, J. K. (1983). The multifrontal solution of indefinite sparse symmetric linear. *ACM Transactions on Mathematical Software*, 9(3), 302–325.
- GitHub: https://github.com/wangyu9/Schwarz_Schur_Involution

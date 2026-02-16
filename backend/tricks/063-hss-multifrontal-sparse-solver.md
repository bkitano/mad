# 063: HSS-Structured Multifrontal Sparse Solver

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ghysels, Li, Rouet, Williams, Napov — "An efficient multi-core implementation of a novel HSS-structured multifrontal solver using randomized sampling" (2015/2018)
**Paper**: [papers/hss-multifrontal-strumpack.pdf]
**Documented**: 2026-02-15

## Description

The HSS-structured multifrontal solver accelerates sparse direct solvers by compressing the dense frontal matrices that arise during multifrontal Gaussian elimination using hierarchically semiseparable (HSS) matrices. In standard multifrontal methods, the computational bottleneck is the dense LU factorization of frontal matrices, which costs $O(N^3)$ per front. By recognizing that these frontal matrices possess low-rank off-diagonal structure (for PDE-derived problems), HSS compression reduces both the factorization cost and memory from $O(N^{3/2})$ to $O(N)$ for 2D elliptic problems. The key innovation is combining randomized HSS compression (needing only matrix-vector products, not explicit matrix entries) with a fast ULV factorization, all within a fully structured multifrontal framework that never forms dense frontal matrices explicitly.

## Mathematical Form

**Core Structure — Multifrontal Elimination:**

Each frontal matrix $\mathcal{F}_i$ is assembled from the sparse matrix $A$ and children's update (Schur complement) matrices:

$$
\mathcal{F}_i = A_i \stackrel{\leftarrow}{\downarrow} \mathcal{U}_{\nu_1} \stackrel{\leftarrow}{\downarrow} \mathcal{U}_{\nu_2} \stackrel{\leftarrow}{\downarrow} \cdots
$$

where $\stackrel{\leftarrow}{\downarrow}$ denotes the extend-add operation. The frontal matrix has $2 \times 2$ block structure:

$$
\mathcal{F}_i = \begin{bmatrix} F_{11} & F_{12} \\ F_{21} & F_{22} \end{bmatrix}
$$

**HSS Compression of Frontal Matrices:**

Off-diagonal blocks of $\mathcal{F}_i$ are approximated via the HSS representation:

$$
A_{\nu_1, \nu_2} \approx U_{\nu_1}^{\text{big}} B_{\nu_1, \nu_2} \left(V_{\nu_2}^{\text{big}}\right)^*
$$

with hierarchically nested bases:

$$
U_\tau^{\text{big}} = \begin{bmatrix} U_{\nu_1}^{\text{big}} & 0 \\ 0 & U_{\nu_2}^{\text{big}} \end{bmatrix} U_\tau, \quad V_\tau^{\text{big}} = \begin{bmatrix} V_{\nu_1}^{\text{big}} & 0 \\ 0 & V_{\nu_2}^{\text{big}} \end{bmatrix} V_\tau
$$

**Randomized HSS Construction:**

Given random matrix $R \in \mathbb{C}^{N \times d}$ with $d = r + p$ columns, compute samples:

$$
S^r = AR, \quad S^c = A^* R
$$

At each level $\ell$ of the HSS tree, construct the sample matrix isolating off-diagonal contributions:

$$
S^{(\ell)} = \left(A - D^{(\ell)}\right) R = S^r - D^{(\ell)} R
$$

where $D^{(\ell)} = \text{diag}(D_{\tau_1}, D_{\tau_2}, \ldots, D_{\tau_q})$ is the block diagonal at level $\ell$.

Compress via interpolative decomposition (ID):

$$
[U_\tau^*, J_\tau^r] = \text{ID}((S_\tau^r)^*, \varepsilon), \quad [V_\tau^*, J_\tau^c] = \text{ID}((S_\tau^c)^*, \varepsilon)
$$

yielding interpolative bases:

$$
U_\tau = \Pi_\tau^r \begin{bmatrix} I \\ E_\tau^r \end{bmatrix}, \quad V_\tau = \Pi_\tau^c \begin{bmatrix} I \\ E_\tau^c \end{bmatrix}
$$

**Skinny Extend-Add:**

Random matrices are merged hierarchically along the elimination tree. For parent node $i$ with children $\nu_1, \nu_2$:

$$
R_i(r, c) = \begin{cases} R_{\nu_1}(r, c) \equiv R_{\nu_2}(r, c) & \text{if } c < \min(d_{\nu_1}, d_{\nu_2}) \text{ and overlapping} \\ R_{\nu_k}(r, c) & \text{if row belongs to child } \nu_k \\ \text{random}(r, c) & \text{otherwise} \end{cases}
$$

This allows efficient evaluation $\mathcal{F}_i R_i$ without forming $\mathcal{F}_i$ explicitly.

**ULV Factorization and Low-Rank Schur Complement:**

After HSS compression, the Schur complement update is stored in low-rank form:

$$
\mathcal{U}_i = F_{i_{22}} - F_{i_{21}} F_{i_{11}}^{-1} F_{i_{12}} = F_{i_{22}} - \Theta_i^* \Phi_i
$$

where $\Theta_i^*$ and $\Phi_i$ are dense rectangular matrices. The multiplication with $\mathcal{U}_i$ is performed efficiently using HSS matrix-vector multiplication for $F_{i_{22}}$ and two dense rectangular matrix products.

**Key Definitions:**

- $N$ — total number of degrees of freedom (matrix dimension)
- $r$ — maximum HSS rank of the frontal matrices
- $\varepsilon$ — compression tolerance for the interpolative decomposition
- $\ell_s$ — switch level: fronts at levels $\ell \geq \ell_s$ use HSS; others use dense LU
- $d = r + p$ — number of random sampling vectors ($p$ is oversampling parameter)

## Complexity

| Operation | Standard Multifrontal | HSS Multifrontal |
|-----------|----------------------|------------------|
| 2D elliptic factor | $O(N^{3/2})$ | $O(N)$ |
| 2D elliptic memory | $O(N \log N)$ | $O(N)$ |
| 2D Helmholtz factor | $O(N^{3/2})$ | $O(N)$ |
| 3D elliptic factor | $O(N^2)$ | $O(N^{10/9} \log N)$ |
| 3D elliptic memory | $O(N^{4/3})$ | $O(N)$ |
| 3D Helmholtz factor | $O(N^2)$ | $O(N^{10/9} \log N)$ |

**Memory:** $O(N)$ for HSS-compressed factors vs $O(N \log N)$ (2D) or $O(N^{4/3})$ (3D) for standard multifrontal.

**HSS construction cost:** $O(Nr^2)$ per front using randomized sampling (vs $O(N^2 r)$ for direct compression).

## Applicability

- **Sparse linear systems from PDE discretizations**: Poisson, convection-diffusion, Helmholtz equations where frontal matrices have naturally low HSS rank
- **Preconditioners for iterative solvers**: Using loose compression tolerance $\varepsilon$ gives a fast approximate factorization; paired with GMRES for the outer solve
- **Neural network weight matrices**: Large structured weight matrices in physics-informed neural networks or PDE-based layers can benefit from HSS-structured factorization
- **State-space models**: The connection between SSMs and semiseparable matrices (Mamba-2/SSD) suggests that HSS multifrontal techniques could accelerate training of structured state-space layers with sparse transition matrices
- **Scalable direct solvers for kernel matrices**: Dense kernel matrices from attention-like mechanisms with geometric structure can be compressed via HSS within a multifrontal framework

## Limitations

- HSS rank depends on the problem: for 2D elliptic PDE the rank is $O(1)$, for 3D elliptic it is $O(k)$ where $k$ is the mesh size, and for Helmholtz it can be $O(\log k)$ to $O(k)$
- The adaptive rank scheme requires multiple passes when the HSS rank is not known a priori
- Speedups are most significant for 3D problems (up to 7×); for 2D problems the smaller fronts mean less room for compression savings
- Optimal switch level $\ell_s$ and compression tolerance $\varepsilon$ must be tuned experimentally
- As a preconditioner (loose $\varepsilon$), the number of GMRES iterations increases, creating a factorization-time vs. solve-time tradeoff

## Implementation Notes

```python
# Pseudocode for HSS-structured multifrontal factorization

def hss_multifrontal_factor(A, elimination_tree, ell_s, epsilon):
    """
    Factor sparse matrix A using HSS-compressed multifrontal method.

    Args:
        A: sparse matrix (N x N)
        elimination_tree: nested dissection elimination tree
        ell_s: switch level (levels >= ell_s use HSS compression)
        epsilon: HSS compression tolerance
    """
    # Bottom-up traversal of elimination tree
    for node_i in elimination_tree.postorder():
        # Assemble frontal matrix from A and children's update matrices
        # F_i = A_i extend-add U_child1 extend-add U_child2 ...

        if node_i.level < ell_s:
            # Dense factorization for small fronts near leaves
            F_11_factors = dense_lu(F_i.F11)
            U_i = F_i.F22 - F_i.F21 @ solve(F_11_factors, F_i.F12)
        else:
            # HSS compression: never form F_i explicitly
            # 1. Generate/merge random matrix R_i (skinny extend-add)
            R_i = skinny_extend_add(R_children, node_i)

            # 2. Compute samples S^r = F_i @ R_i, S^c = F_i^H @ R_i
            #    via recursive application through children
            Sr_i = apply_frontal_matrix(node_i, R_i)
            Sc_i = apply_frontal_matrix_adjoint(node_i, R_i)

            # 3. HSS compress using randomized sampling + ID
            hss_i = hss_compress(Sr_i, Sc_i, R_i, epsilon)  # adaptive rank

            # 4. ULV factorization of HSS-compressed F_11
            ulv_factors = hss_ulv_factor(hss_i.F11)

            # 5. Low-rank Schur complement: U_i = F_22 - Theta^* @ Phi
            Theta_i, Phi_i = compute_low_rank_schur(hss_i, ulv_factors)
            U_i = HSSSchurComplement(hss_i.F22, Theta_i, Phi_i)

    return factors

# Three levels of parallelism (key to scalability):
# 1. Tree parallelism: independent subtrees processed concurrently
# 2. HSS-tree parallelism: within each front, HSS hierarchy parallelism
# 3. Node parallelism: dense BLAS operations within HSS tree nodes
# Product of all three remains roughly constant = good load balance
```

## References

- Ghysels, Li, Rouet, Williams, Napov, "An efficient multi-core implementation of a novel HSS-structured multifrontal solver using randomized sampling," SIAM J. Sci. Comput. 38(5), 2016. arXiv:1502.07405
- Xia, Chandrasekaran, Gu, Li, "Fast algorithms for hierarchically semiseparable matrices," Numer. Linear Algebra Appl. 17(6), 2010
- Martinsson, "A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix," SIAM J. Matrix Anal. Appl. 32(4), 2011
- STRUMPACK software: https://portal.nersc.gov/project/sparse/strumpack/

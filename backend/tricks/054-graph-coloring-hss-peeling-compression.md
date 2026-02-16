# 054: Graph-Coloring Accelerated HSS Peeling Compression

**Category**: decomposition
**Gain type**: efficiency
**Source**: Levitt & Martinsson (2024), J. Comput. Appl. Math.
**Paper**: [papers/graph-coloring-hss-peeling.pdf]
**Documented**: 2026-02-15

## Description

Graph-coloring accelerated peeling is a black-box algorithm for computing data-sparse representations of rank-structured matrices ($\mathcal{H}^1$, uniform $\mathcal{H}^1$, and $\mathcal{H}^2$ formats) using only matrix-vector products. The key innovation is using graph coloring of a *constraint incompatibility graph* to design structured random test matrices that minimize the number of matrix-vector products needed to recover all low-rank blocks in the hierarchical representation.

The method works by processing the hierarchical tree level-by-level from coarsest to finest. At each level, it must sample all admissible (low-rank) blocks simultaneously. The challenge is that different blocks impose conflicting requirements on the test matrix — some rows must be random while others must be zero. The algorithm formulates these conflicts as a graph coloring problem: vertices represent distinct constraint sets, and edges connect incompatible pairs. Each color yields one test matrix, so the chromatic number of the graph determines the number of matrix-vector products per level.

For uniform trees in $d$ dimensions, this reduces the number of matrix-vector products from $\sim k \cdot 8^d \log N$ (original peeling algorithm) to $\sim k \cdot 6^d \log N$ for $\mathcal{H}^1$ matrices, and substantially more for non-uniform trees and $\mathcal{H}^2$ formats. The method is particularly effective for problems with intrinsic low-dimensional structure (e.g., surface integral equations in 3D), where graph coloring discovers and exploits this structure automatically.

## Mathematical Form

**Rank-Structured Matrix ($\mathcal{H}^1$):**

Given a matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ with an associated hierarchical tree of depth $L$, the $\mathcal{H}^1$ representation approximates each admissible (well-separated) block as:

$$
\mathbf{A}(I_\alpha, I_\beta) \approx \mathcal{U}_{\alpha,\beta} \mathbf{B}_{\alpha,\beta} \mathcal{V}_{\alpha,\beta}^*
$$

where $\mathcal{U}_{\alpha,\beta} \in \mathbb{R}^{|I_\alpha| \times k}$ and $\mathcal{V}_{\alpha,\beta} \in \mathbb{R}^{|I_\beta| \times k}$ are basis matrices, and $\mathbf{B}_{\alpha,\beta} \in \mathbb{R}^{k \times k}$ is a small coupling matrix.

**Level-$l$ Truncated Matrix:**

Define $\mathbf{A}^{(l)}$ as the matrix obtained by replacing with zeros every block of $\mathbf{A}$ corresponding to levels finer than $l$. The residual $\mathbf{A} - \mathbf{A}^{(l)}$ isolates the blocks at levels $l+1, \ldots, L$.

**Sampling Constraints:**

To sample admissible block $\mathbf{A}_{\alpha,\beta}$ of level $l$, we require a test matrix $\boldsymbol{\Omega}$ satisfying:

$$
\boldsymbol{\Omega}(I_\beta, :) = \mathbf{G}_\beta \quad \text{(random)}
$$
$$
\boldsymbol{\Omega}(I_\gamma, :) = \mathbf{0} \quad \text{for all } \gamma \in \mathcal{L}_\alpha^{\text{nei}} \cup \mathcal{L}_\alpha^{\text{int}} \setminus \{\beta\}
$$

where $\mathbf{G}_\beta$ is a random matrix, $\mathcal{L}_\alpha^{\text{nei}}$ is the neighbor list, and $\mathcal{L}_\alpha^{\text{int}}$ is the interaction list of box $\alpha$. If $\boldsymbol{\Omega}$ satisfies these constraints, then:

$$
\left(\mathbf{A}\boldsymbol{\Omega} - \mathbf{A}^{(l-1)}\boldsymbol{\Omega}\right)(I_\alpha, :) = \mathbf{A}_{\alpha,\beta} \mathbf{G}_\beta
$$

which provides a randomized sample of the column space of block $\mathbf{A}_{\alpha,\beta}$.

**Constraint Incompatibility Graph:**

Two constraint sets are *compatible* if they can be satisfied simultaneously by a single test matrix $\boldsymbol{\Omega}$. Two constraint sets are *incompatible* if their zero-out requirements conflict.

The *constraint incompatibility graph* for level $l$ has:
- Vertices: distinct constraint sets (one per admissible block or group of blocks sharing constraints)
- Edges: between pairs with incompatible constraints

A valid vertex coloring assigns a color to each vertex such that no adjacent vertices share a color. Each color $c$ defines one test matrix $\boldsymbol{\Omega}_c$.

**Graph Coloring via DSatur:**

The DSatur (degree of saturation) algorithm greedily assigns colors:

$$
T_{\text{color}} \sim \deg(G) \cdot |V| \cdot \log |V|
$$

where $\deg(G)$ is the maximum vertex degree. For bounded-degree graphs (typical in $d$-dimensional problems), this is nearly linear.

**Chromatic Number Bounds:**

For problems on a $d$-dimensional grid with uniform trees:

| Format | Admissible blocks per level | Inadmissible blocks per level |
|--------|----------------------------|------------------------------|
| $\mathcal{H}^1$ | $\chi_{\text{nonunif}} \leq 6^d$ | $\chi_{\text{leaf}} \leq 3^d$ |
| Uniform $\mathcal{H}^1$ | $\chi_{\text{unif}} \leq 5^d$ | $\chi_{\text{leaf}} \leq 3^d$ |

**Basis Matrix Recovery ($\mathcal{H}^1$):**

Given samples $\mathbf{Y}_i = \mathbf{A}\boldsymbol{\Omega}_i - \mathbf{A}^{(l-1)}\boldsymbol{\Omega}_i$, the column basis for block $\mathbf{A}_{\alpha,\beta}$ is:

$$
\mathcal{U}_{\alpha,\beta} = \texttt{qr}(\mathbf{Y}_i(I_\alpha, :), k)
$$

where $\boldsymbol{\Omega}_i$ is the test matrix containing the constraints for pair $(\alpha, \beta)$. Row bases $\mathcal{V}_{\alpha,\beta}$ are obtained similarly using $\mathbf{A}^*$.

**Coupling Matrix:**

$$
\mathbf{B}_{\alpha,\beta} = (\mathbf{G}_\alpha \mathcal{U}_\alpha)^\dagger \mathbf{G}_\alpha \mathbf{A}_{\alpha,\beta} \mathbf{G}_\beta (\mathcal{V}_\beta \mathbf{G}_\beta)^\dagger
$$

where the products $\mathbf{A}_{\alpha,\beta}\mathbf{G}_\beta$ have already been obtained from the samples.

**$\mathcal{H}^2$ Extension (Nested Bases):**

For $\mathcal{H}^2$ matrices, each box $\tau$ has uniform basis matrices satisfying a nestedness condition:

$$
\mathcal{U}_\tau = \begin{bmatrix} \mathcal{U}_\alpha & 0 \\ 0 & \mathcal{U}_\beta \end{bmatrix} \mathbf{U}_\tau
$$

where $\alpha, \beta$ are children of $\tau$, and $\mathbf{U}_\tau$ is a small "short" basis of size $2k \times k$. The sampling constraints for uniform basis computation are modified: rows corresponding to $\mathcal{L}_\alpha^{\text{int}}$ are filled with random values, while rows for $\mathcal{L}_\alpha^{\text{nei}}$ are zero:

$$
\boldsymbol{\Omega}(I_\beta, :) = \mathbf{G}_\beta \quad \text{for all } \beta \in \mathcal{L}_\alpha^{\text{int}}
$$
$$
\boldsymbol{\Omega}(I_\gamma, :) = \mathbf{0} \quad \text{for all } \gamma \in \mathcal{L}_\alpha^{\text{nei}}
$$

The column basis $\mathcal{U}_\alpha$ spans the column space of $\mathbf{A}(I_\alpha, \cup_{\beta \in \mathcal{L}_\alpha^{\text{int}}} I_\beta)$ — the interactions of $\alpha$ with all boxes in its interaction list collectively.

## Complexity

| Operation | Original Peeling | Graph-Coloring Peeling |
|-----------|-----------------|----------------------|
| Matvecs per level ($d$-dim, $\mathcal{H}^1$) | $\sim k \cdot 8^d$ | $\sim k \cdot 6^d$ (via $\chi_{\text{nonunif}}$) |
| Total matvecs ($\mathcal{H}^1$) | $\sim k \cdot 8^d \log N$ | $\sim k \cdot 6^d \log N$ |
| Total matvecs (uniform $\mathcal{H}^1$) | $\sim k \cdot 8^d \log N$ | $\sim k \cdot 5^d \log N$ |
| Non-uniform tree acceleration | Fixed pattern | Adaptive (often dramatic) |
| Total FLOPs | $O(\chi k^2 N (\log N)^2)$ | $O(\chi k^2 N (\log N)^2)$ |

**Detailed Compression Cost ($\mathcal{H}^1$):**

$$
T_{\text{compress}} \sim T_{\text{mult}} \times 2\chi_{\text{nonunif}} k \log N + T_{\text{flop}} \times 2\chi_{\text{nonunif}} k^2 N (\log N)^2
$$

where $T_{\text{mult}}$ is the cost of one matrix-vector product with $\mathbf{A}$, $\chi_{\text{nonunif}}$ is the chromatic number of the constraint incompatibility graph, and $k$ is the block rank.

**Memory:** $O(k^2 N \log N)$ for $\mathcal{H}^1$; $O(k^2 N)$ for $\mathcal{H}^2$ (due to nested bases).

**Key Advantage — Non-Uniform Trees:**

For problems with intrinsic low-dimensional structure (e.g., a 2D surface in 3D space), the graph coloring approach discovers this structure automatically. The incompatibility graph has lower connectivity than the worst-case bound, yielding dramatically fewer test matrices. For a surface integral equation on a 2D manifold in $\mathbb{R}^3$, the chromatic number is bounded by $\sim 6^2 = 36$ rather than $6^3 = 216$.

## Applicability

1. **Implicit Kernel Matrix Compression**: When the attention or kernel matrix $K_{ij} = \mathcal{K}(x_i, x_j)$ is defined by a kernel function and available only through matrix-vector products (e.g., via FMM), graph coloring minimizes the number of matvecs needed to build a compressed representation
2. **Sparse Direct Solver Acceleration**: Compressing dense Schur complements $\mathbf{S}_{22} = \mathbf{A}_{21}\mathbf{A}_{11}^{-1}\mathbf{A}_{12}$ in multifrontal solvers, where the Schur complement is available only as an implicit operator
3. **Adaptive Rank Discovery**: The graph-coloring framework naturally adapts to non-uniform rank distributions — regions with higher-dimensional interactions use more test matrices while low-dimensional regions need fewer
4. **Preconditioner Construction for Sequence Models**: Building hierarchical preconditioners for the implicit linear systems arising in implicit integration of SSMs or continuous-time attention models
5. **Batched Kernel Compression**: When compressing multiple related kernel matrices (e.g., across attention heads or time steps), the same graph coloring and test matrix patterns can be reused, amortizing the setup cost
6. **Strong Admissibility Problems**: Unlike the black-box HSS compression (which uses weak admissibility), this method handles $\mathcal{H}^1$ matrices with strong admissibility — a broader class including finite element and finite difference discretizations

## Limitations

1. **Known Tree Structure Required**: The hierarchical tree partition must be known a priori (typically from the geometry of the underlying problem). Does not discover the optimal partition
2. **Logarithmic Overhead**: Requires $O(k \log N)$ matrix-vector products total (vs. $O(k)$ for the black-box HSS method with weak admissibility), though the pre-factor is significantly smaller than the original peeling algorithm
3. **Graph Coloring Heuristic**: DSatur is a greedy heuristic that may not find the minimum chromatic number. For most practical problems with bounded geometry, the coloring is near-optimal, but pathological cases exist
4. **Reconstruction Quality**: The accuracy depends on the oversampling parameter $p$ in the randomized sampling; insufficient oversampling can lead to poor approximations
5. **$\mathcal{H}^1$ vs. $\mathcal{H}^2$ Trade-off**: The $\mathcal{H}^1$ format uses non-nested bases (more flexible, more storage), while $\mathcal{H}^2$ uses nested bases (less storage, but requires the additional SVD-based augmentation step for basis enrichment)
6. **Strong Admissibility Overhead**: Handling non-low-rank inadmissible blocks (neighbor interactions) requires additional test matrices beyond what the peeling of admissible blocks needs

## Implementation Notes

```python
# Graph-coloring accelerated peeling compression
# (Levitt & Martinsson, Algorithm 4.1)

def graph_coloring_compress(matvec, matvec_adj, N, tree, k, p=10):
    """
    Compress rank-structured matrix using graph-coloring peeling.

    Args:
        matvec: function computing A @ x
        matvec_adj: function computing A^* @ x
        N: matrix dimension
        tree: hierarchical tree with interaction/neighbor lists
        k: target rank
        p: oversampling parameter

    Returns:
        H1_repr: dict of {U, V, B} for each admissible pair,
                 plus inadmissible blocks at leaf level
    """
    r = k + p  # rank with oversampling
    A_approx_levels = {}  # level-l truncated matrix approximations

    for level in range(2, tree.depth + 1):
        # --- Step 1: Build constraint incompatibility graph ---
        constraints = {}
        for (alpha, beta) in tree.admissible_pairs(level):
            # Constraint: Omega[I_beta,:] = random,
            #             Omega[I_gamma,:] = 0 for gamma in nei(alpha) union int(alpha)\{beta}
            constraints[(alpha, beta)] = {
                'random': [beta],
                'zero': list(alpha.neighbors | alpha.interaction_list - {beta})
            }

        G_incompat = build_incompatibility_graph(constraints)

        # --- Step 2: Color the graph (DSatur) ---
        coloring = dsatur_coloring(G_incompat)
        n_colors = max(coloring.values()) + 1

        # --- Step 3: Build structured test matrices ---
        Omegas = []
        for color in range(n_colors):
            Omega = np.zeros((N, r))
            for (alpha, beta), c in coloring.items():
                if c == color:
                    Omega[tree.indices(beta), :] = np.random.randn(
                        len(tree.indices(beta)), r
                    )
            Omegas.append(Omega)

        # --- Step 4: Apply matrix and subtract known levels ---
        for i, Omega_i in enumerate(Omegas):
            Y_i = matvec(Omega_i)
            # Subtract contributions from previously compressed levels
            Y_i -= apply_truncated(A_approx_levels, level - 1, Omega_i)

            # Extract column bases for each admissible pair
            for (alpha, beta), c in coloring.items():
                if c == i:
                    sample = Y_i[tree.indices(alpha), :]
                    U_ab = np.linalg.qr(sample, mode='reduced')[0][:, :k]
                    store_column_basis(alpha, beta, U_ab)

        # --- Step 5: Similarly for row bases using A^* ---
        # (Same process with matvec_adj and transposed constraints)
        # ... (symmetric procedure)

        # --- Step 6: Compute coupling matrices B ---
        for (alpha, beta) in tree.admissible_pairs(level):
            B_ab = compute_coupling(alpha, beta)
            store_coupling(alpha, beta, B_ab)

    # --- Step 7: Extract inadmissible blocks at leaf level ---
    # (Requires separate graph coloring with fewer constraints)
    extract_inadmissible_blocks(matvec, tree, A_approx_levels)

    return H1_repr


def dsatur_coloring(graph):
    """
    Greedy graph coloring using Degree of Saturation heuristic.

    At each step, pick the uncolored vertex with highest saturation
    (most distinct colors among neighbors), breaking ties by degree.

    Complexity: O(deg(G) * |V| * log|V|) with priority queue
    """
    import heapq

    colors = {}
    sat = {v: 0 for v in graph.vertices}
    neighbor_colors = {v: set() for v in graph.vertices}

    # Priority queue: (-saturation, -degree, vertex)
    pq = [(-0, -graph.degree(v), v) for v in graph.vertices]
    heapq.heapify(pq)

    while pq:
        _, _, v = heapq.heappop(pq)
        if v in colors:
            continue

        # Find smallest available color
        used = neighbor_colors[v]
        color = 0
        while color in used:
            color += 1
        colors[v] = color

        # Update neighbors
        for w in graph.neighbors(v):
            if w not in colors:
                if color not in neighbor_colors[w]:
                    neighbor_colors[w].add(color)
                    sat[w] = len(neighbor_colors[w])
                    heapq.heappush(pq, (-sat[w], -graph.degree(w), w))

    return colors
```

**Key Implementation Insights:**

1. **Constraint Incompatibility = Graph Problem**: The fundamental insight is that finding the minimum number of test matrices is equivalent to finding a minimum vertex coloring of a constraint graph. This transforms a combinatorial problem into a well-studied graph theory problem with efficient heuristics
2. **Level-by-Level Peeling**: By processing coarse levels first and subtracting their contributions ($\mathbf{A} - \mathbf{A}^{(l)}$), each subsequent level only sees the residual blocks, enabling clean isolation of per-level information
3. **Shared Constraint Sets**: Multiple admissible blocks often share identical constraint sets (e.g., blocks in the same row that interact with the same neighbors). Deduplicating these reduces the graph size
4. **Non-Uniform Adaptivity**: For problems with lower intrinsic dimension, the incompatibility graph is sparser (fewer conflicts), so the chromatic number is automatically smaller — the algorithm discovers and exploits this without user intervention
5. **$\mathcal{H}^2$ Augmentation Trick**: For nested-basis ($\mathcal{H}^2$) compression, the sample is enriched with the parent's contributions: $[\mathbf{Y}_i(I_\alpha, :); \mathcal{U}_\tau(I_{\tau,\alpha}, :)\boldsymbol{\Sigma}_\tau^{\text{in}}\mathbf{G}_k]$, ensuring the child's basis spans a subspace compatible with the parent's

## References

- Levitt, J. & Martinsson, P. G. (2024). Randomized compression of rank-structured matrices accelerated with graph coloring. *Journal of Computational and Applied Mathematics*, 446, 116044. arXiv:2205.03406.
- Lin, L., Lu, J., & Ying, L. (2011). Fast construction of hierarchical matrix representation from matrix-vector multiplication. *Journal of Computational Physics*, 230(10), 4071-4087.
- Martinsson, P. G. (2011). A fast randomized algorithm for computing a hierarchically semiseparable representation of a matrix. *SIAM J. Matrix Anal. Appl.*, 32(4), 1251-1274.
- Brézinski, D. (1979). New aspects of the degree of saturation (DSatur) graph coloring algorithm. *Journal of Combinatorial Theory*, Series B, 26(2), 149-159.

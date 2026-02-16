# 088: Parallel HSS Context-Based Tree Algorithms

**Category**: parallelization
**Gain type**: efficiency
**Source**: Wang, Li, Xia, Situ & de Hoop — "Efficient Scalable Algorithms for Hierarchically Semiseparable Matrices" (2013)
**Paper**: [papers/parallel-hss-algorithms.pdf]
**Documented**: 2026-02-15

## Description

This trick provides a complete framework for parallelizing the three core HSS matrix operations — construction (compression), ULV factorization, and linear system solution — across distributed-memory parallel architectures. The key innovations are:

1. **Context-based tree parallelism**: Each node of the HSS binary tree is assigned a BLACS *context* (MPI sub-communicator), with process groups doubling as you move up the tree. This maps the recursive HSS tree structure onto a hierarchical process topology, enabling independent computation within subtrees with minimal inter-context communication.

2. **Level-wise traversal** instead of postorder: The serial HSS algorithms traverse the tree in postorder (children before parents), which is inherently sequential. The parallel version reorganizes to level-by-level traversal — all leaf nodes process simultaneously, then level-2 nodes, etc. — exposing $O(n/m)$ parallelism at the leaf level (where $m$ is the block size).

3. **Pairwise exchange redistribution**: Moving data between tree levels requires only pairwise exchanges between neighboring process contexts (e.g., process 0↔4, 1↔5, etc.), with total communication volume $O(rn)$ across all levels — linear in $n$.

4. **Visited-set column compression**: The column compression stage uses "visited sets" $\mathcal{V}_i$ and $\mathcal{W}_i$ to collect off-diagonal blocks from multiple tree levels into a single tall block $G_i$ for parallel RRQR compression, generalizing the symmetric-only technique to nonsymmetric matrices.

The framework achieves weak scaling factor $\approx 2\times$ and strong scaling demonstrated on matrices up to $n = 100{,}000$ with 1,024 processes.

## Mathematical Form

**HSS Generator Definition:**

For an $n \times n$ matrix $A$ with HSS tree $\mathcal{T}$ (postordered full binary tree with $2k-1$ nodes), the HSS form stores generators $\{D_i, U_i, V_i, R_i, W_i, B_i\}$ for each node $i \in \mathcal{T}$:

$$
D_i = A|_{t_i \times t_i}, \quad A|_{t_i \times (\mathcal{I} \setminus t_i)} \approx U_i \widehat{F}_i
$$

At internal nodes with children $c_1, c_2$:

$$
U_i = \begin{pmatrix} U_{c_1} R_{c_1} \\ U_{c_2} R_{c_2} \end{pmatrix}, \quad V_i = \begin{pmatrix} V_{c_1} W_{c_1} \\ V_{c_2} W_{c_2} \end{pmatrix}
$$

where $R_{c_j} \in \mathbb{R}^{r \times r}$ and $W_{c_j} \in \mathbb{R}^{r \times r}$ are translation operators that encode the nested basis structure.

**Parallel Row Compression (Level $\ell$):**

At each level $\ell$ of the tree, the block to be compressed for node $i$ with children $c_1, c_2$ is:

$$
F_i \equiv \begin{pmatrix} A|_{\hat{t}_{c_1} \times (\mathcal{I} \setminus t_i)} \\ A|_{\hat{t}_{c_2} \times (\mathcal{I} \setminus t_i)} \end{pmatrix}
$$

where $\hat{t}_{c_j}$ is the compressed index set after applying previous level's basis. This is approximated via parallel RRQR:

$$
F_i \approx \begin{pmatrix} R_{c_1} \\ R_{c_2} \end{pmatrix} A|_{\hat{t}_i \times (\mathcal{I} \setminus t_i)}
$$

**Parallel RRQR Factorization:**

Given a block $F$ of size $M \times N$ distributed across $P$ processes on a $\sqrt{P} \times \sqrt{P}$ grid, the Modified Gram-Schmidt RRQR computes $F \approx \widetilde{Q}\widetilde{T}$ where $\widetilde{Q} = (q_1, \ldots, q_r)$ and $\widetilde{T}^H = (t_1, \ldots, t_r)$:

```
for i = 1:r
  1. Find column f_j with maximum norm (parallel reduction)
  2. Interchange f_i ↔ f_j
  3. Compute t_{ii} = ||f_i||; stop if t_{ii}/t_{11} ≤ τ
  4. Normalize: q_i = f_i / ||f_i||
  5. Broadcast q_i within context
  6. Deflate: t_i^H = q_i^H (f_{i+1}, ..., f_N) via PBLAS2
  7. Rank-1 update: (f_{i+1},...,f_N) -= q_i t_i^H
```

Communication cost per RRQR:

$$
\text{Comm}_{RRQR} = \left[\log\sqrt{P}, \; \frac{M+N}{\sqrt{P}} \log\sqrt{P}\right] \cdot r
$$

**Parallel Column Compression via Visited Sets:**

The *left visited set* of node $i$ in a postordered tree $\mathcal{T}$:

$$
\mathcal{V}_i = \{j \mid j \text{ is a left node and } \text{sib}(j) \in \text{ancs}(i)\}
$$

The blocks gathered for column compression at leaf node $i$ form:

$$
G_i^H = \begin{pmatrix} A|_{\hat{t}_{j_1} \times t_i} \\ A|_{\hat{t}_{j_2} \times t_i} \\ \vdots \end{pmatrix}, \quad j_1, j_2, \ldots \in \mathcal{V}_i \cup \mathcal{W}_i
$$

where $\mathcal{W}_i$ is the analogous right visited set. The row dimension of $G_i$ is bounded by $r \log(n/m) \approx r \log P$, making each column RRQR cheap.

**Parallel ULV Factorization:**

Starting from a block $2 \times 2$ HSS form at children $c_1, c_2$:

$$
\begin{pmatrix} D_{c_1} & U_{c_1} B_{c_1} V_{c_2}^H \\ U_{c_2} B_{c_2} V_{c_1}^H & D_{c_2} \end{pmatrix}
$$

1. **Intra-context** QL factorization: $U_{c_j} = Q_{c_j} \begin{pmatrix} 0 \\ \widetilde{U}_{c_j} \end{pmatrix}$ (no communication)

2. **Intra-context** LQ factorization of transformed diagonal blocks (no communication)

3. **Inter-context** exchange to form new merged generators:

$$
D_i = \begin{pmatrix} \widehat{D}_{c_1;2,2} & \widetilde{U}_{c_1} B_{c_1} \widetilde{V}_{c_2;2}^H \\ \widetilde{U}_{c_2} B_{c_2} \widetilde{V}_{c_1;2}^H & \widehat{D}_{c_2;2,2} \end{pmatrix}
$$

Communication cost for entire factorization:

$$
\left[O\left(\frac{r}{b} \log P\right), \; O(r^2 \log P)\right]
$$

where $b$ is the ScaLAPACK block size.

**Key Definitions:**

- $n$ — matrix dimension
- $r$ — HSS rank (maximum rank of off-diagonal blocks)
- $m$ — leaf block size ($P \approx n/m$ processes)
- $P$ — number of processes
- $L = \log_2 P$ — number of tree levels
- $\tau$ — relative tolerance for RRQR truncation
- Context — BLACS sub-communicator mapped to an HSS tree node

## Complexity

| Operation | Serial | Parallel ($P$ processes) |
|-----------|--------|--------------------------|
| HSS construction (flops) | $O(rn^2)$ | $O(rn^2/P)$ |
| HSS factorization (flops) | $O(r^2 n)$ | $O(r^2 n / P)$ |
| HSS solution (flops) | $O(rn)$ | $O(rn/P)$ |

**Communication costs (total across all levels):**

| Phase | #Messages | #Words |
|-------|-----------|--------|
| HSS construction — redistribution | $O(\log P)$ | $O(rn)$ |
| HSS construction — RRQR | $O(r \log^2 P)$ | $O(rn \log P + r^2 \log^2 P)$ |
| ULV factorization | $O(\frac{r}{b} \log P)$ | $O(r^2 \log P)$ |
| HSS solution | $O(\log P)$ | $O(r \log P)$ |

**Flop-to-byte ratio:** $\approx n / (P \log P)$, indicating the algorithm is communication-bound rather than compute-bound for large $P$.

**Memory:** $O(rn / P)$ per process (perfect memory scaling).

**Demonstrated performance:**
- Weak scaling factor $\approx 2.0\times$ (ideal = 1.0) for construction
- Strong scaling: $90s \to 28s$ for $n = 100{,}000$ Toeplitz matrix (64 → 1024 processors)
- MF+HSS solver is 2-3× faster than exact multifrontal for 2D Helmholtz, and enables 3D problems that run out of memory with exact methods

## Applicability

- **Large-scale structured linear systems**: Any application where $A$ has HSS structure (PDEs, integral equations, kernel matrices) and $n$ is large enough to require distributed memory.
- **Parallel SSM training**: State-space models with large hidden dimensions where the transition matrix has HSS structure; parallel construction enables scaling to dimensions infeasible serially.
- **Structured parallel multifrontal solvers**: The parallel HSS operations serve as kernels inside multifrontal sparse solvers (STRUMPACK), compressing the dense frontal matrices that dominate cost.
- **Multi-frequency/multi-right-hand-side problems**: Helmholtz and wave equations requiring solves at many frequencies; the HSS factorization is reused across right-hand sides with only $O(rn/P)$ cost per solve.
- **Distributed attention kernels**: For attention matrices with hierarchical low-rank structure, the context-based tree parallelism maps naturally to multi-GPU/multi-node setups.

## Limitations

- **Communication-bound**: The flop-to-byte ratio $n/(P\log P)$ is small, meaning network bandwidth and latency dominate over compute for large $P$. Performance is more sensitive to interconnect speed than CPU speed.
- **Power-of-two constraint**: The current framework assumes $P$ is a power of 2 for clean binary tree mapping. Non-power-of-2 process counts require padding or load-balancing strategies.
- **2D block-cyclic overhead**: The ScaLAPACK 2D block-cyclic distribution introduces redistribution costs at every tree level. Modern communication-avoiding algorithms might reduce this.
- **Static tree structure**: The HSS tree and process assignment are fixed at construction time. Dynamic load balancing for matrices with non-uniform rank structure is not addressed.
- **Leaf block size tradeoff**: The block size $m$ must balance per-process work (larger $m$ = more flops per process) against parallelism (smaller $m$ = more processes). Optimal $m$ depends on $r$ and hardware characteristics.
- **BLAS/LAPACK dependency**: Heavy use of ScaLAPACK routines (PxGEMR2D, PxGEQP3) which may not map efficiently to GPU architectures without adaptation.

## Implementation Notes

```python
# Pseudocode for parallel HSS construction via context-based tree

from mpi4py import MPI
import numpy as np

def parallel_hss_construction(A_local, tree, rank, tol, comm):
    """
    Parallel HSS construction using BLACS-style contexts.

    A_local: local portion of matrix A on this process
    tree: HSS binary tree structure
    rank: target HSS rank r
    tol: RRQR truncation tolerance tau
    comm: MPI communicator (all processes)
    """
    P = comm.Get_size()
    my_rank = comm.Get_rank()
    L = int(np.log2(P))  # number of tree levels

    # Phase 1: Parallel row compression (bottom-up)
    for level in range(1, L + 1):
        # Each process owns one leaf node at level 1
        # At level ell, groups of 2^(ell-1) processes cooperate

        context_size = 2**(level - 1)
        context_id = my_rank // context_size
        local_rank_in_context = my_rank % context_size

        # Step 1a: Redistribute data between sibling contexts
        # Pairwise exchange: process i <-> process i XOR context_size
        partner = my_rank ^ context_size
        send_data = A_local.get_sibling_block()
        recv_data = comm.sendrecv(send_data, dest=partner, source=partner)

        # Step 1b: Form F_i by stacking compressed child blocks
        F_i = np.vstack([child1_compressed, child2_compressed])

        # Step 1c: Parallel RRQR within context
        # Uses context sub-communicator with 2D block-cyclic layout
        context_comm = comm.Split(color=context_id)
        Q, T, perm = parallel_rrqr(F_i, tol, context_comm)

        # Extract R generators (translation operators)
        R_child1 = T[:rank, :]  # transfer from child1 to parent
        R_child2 = T[rank:2*rank, :]  # transfer from child2 to parent

        # Store U generators
        U_node = Q[:, :rank]

    # Phase 2: Parallel column compression (top-down redistribution,
    #           then bottom-up compression)

    # Step 2a: Top-down redistribution of visited-set blocks
    for level in range(L, 0, -1):
        # Redistribute A|_{t_j x sib(t_j)} blocks downward
        # to the leaf contexts that need them
        partner = my_rank ^ (2**(level-1))
        visited_block = comm.sendrecv(block_data, dest=partner,
                                       source=partner)

    # Step 2b: Bottom-up column compression
    for level in range(1, L + 1):
        context_comm = comm.Split(color=my_rank // (2**(level-1)))

        # Form G_i from visited set blocks (row dim = r * log P)
        G_i = stack_visited_set_blocks(level)

        # Parallel RRQR on G_i
        Q, T, perm = parallel_rrqr(G_i, tol, context_comm)

        # Extract V and W generators
        V_node = Q[:, :rank]
        W_child = T[:rank, :]

    return hss_generators

def parallel_hss_solve(hss, b_local, comm):
    """
    Solve Ax = b using parallel ULV factorization + substitution.

    Phase 1 (bottom-up): QL factorizations, partial elimination
    Phase 2 (top-down): back-substitution with inter-context communication

    Cost: O(rn/P) flops + O(r * log P) communication words
    """
    P = comm.Get_size()
    L = int(np.log2(P))

    # Forward sweep (bottom-up)
    for level in range(1, L + 1):
        context_size = 2**(level - 1)
        context_comm = comm.Split(color=my_rank // context_size)

        # Intra-context: QL factorization of U, LQ of D (no communication)
        Q_local = ql_factorize(U_node)
        b_transformed = Q_local.T @ b_local

        # Intra-context: triangular solve for first components
        x_partial = triangular_solve(D_upper, b_transformed[:r])

        # Inter-context: exchange partial solutions for RHS update
        partner = my_rank ^ context_size
        x_partner = comm.sendrecv(x_partial, dest=partner, source=partner)

        # Update remaining RHS
        b_local -= U_tilde @ (B @ V_tilde.T @ x_partner)

    # Root solve (all processes cooperate)
    x_root = solve_root_system(D_root, b_root, comm)

    # Backward sweep (top-down)
    for level in range(L, 0, -1):
        # Recover full solution from partial solutions
        x_local = P_local.T @ np.concatenate([x_partial, x_remaining])

    return x_local
```

**Key Implementation Insights:**

1. **Context hierarchy**: Map each HSS tree node to a BLACS context. Leaf nodes get individual processes; internal nodes get the union of their children's contexts. This naturally decomposes the parallelism.

2. **Pairwise exchanges only**: All inter-context communication is pairwise (process $i \leftrightarrow i \oplus 2^{\ell-1}$), avoiding collective operations and enabling overlap with computation.

3. **ScaLAPACK integration**: Use `PxGEMR2D` for data redistribution between contexts and `PxGEQP3` for parallel RRQR. This provides portable, optimized implementations across architectures.

4. **Nonsymmetric generalization**: The visited-set concept ($\mathcal{V}_i, \mathcal{W}_i$) extends the symmetric-only tree techniques to general nonsymmetric HSS matrices, collecting blocks from ancestors' siblings for column compression.

## References

- Wang, S., Li, X. S., Xia, J., Situ, Y., & de Hoop, M. V. (2013). Efficient scalable algorithms for hierarchically semiseparable matrices. SIAM Journal on Scientific Computing, 35(6), C519-C544.
- Chandrasekaran, S., Gu, M., & Pals, T. (2006). A fast ULV decomposition solver for hierarchically semiseparable representations. SIAM J. Matrix Anal. Appl., 28(3), 603-622.
- Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices. Numerical Linear Algebra with Applications, 17(6), 953-976.
- Rouet, F.-H., Li, X. S., Ghysels, P., & Napov, A. (2016). A distributed-memory package for dense hierarchically semi-separable matrix computations using randomization. ACM Transactions on Mathematical Software, 42(4), 1-35.

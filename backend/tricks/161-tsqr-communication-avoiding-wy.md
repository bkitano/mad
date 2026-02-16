# 161: TSQR — Communication-Avoiding QR with Tree-Structured WY

**Category**: parallelization
**Gain type**: efficiency
**Source**: Demmel, Grigori, Hoemmen & Langou (2012), SIAM Review 54(1); Ballard, Demmel, Grigori, Jacquelin, Nguyen & Solomonik (2014)
**Paper**: [papers/tsqr-communication-avoiding-qr.pdf]
**Documented**: 2026-02-15

## Description

**Tall-Skinny QR (TSQR)** is a communication-avoiding QR factorization that replaces the conventional column-by-column Householder QR with a **tree-structured reduction** where each node performs a local QR of stacked triangular matrices. The Q factor is stored implicitly as a **tree of compact WY representations** — one $(Y_i, T_i)$ pair at each node — rather than a single monolithic WY form. This structure sends $\log P$ messages (optimal) vs. $2n \log P$ for ScaLAPACK's PDGEQRF, a factor of $2n$ fewer.

The key insight: QR factorization can be viewed as a **reduction operation** where the binary operator is "stack two $R$ factors vertically and QR the result." Since each local QR of a $2n \times n$ structured matrix (two stacked upper triangulars) costs only $O(n^3)$ and produces a compact WY representation of the local $Q$ factor, the entire tree of WY factors implicitly represents the global $Q$ at a cost of $O(\log P)$ messages vs. $O(n \log P)$ for conventional approaches.

**CAQR** (Communication-Avoiding QR) uses TSQR as the panel factorization within a blocked right-looking QR algorithm. The trailing matrix update traverses the same tree, applying each local WY factor using the formula $C \leftarrow (I + Y_i T_i Y_i^\top) C$. This sends $\Theta(\sqrt{mn/P})$ fewer messages than ScaLAPACK for general $m \times n$ matrices.

This is directly relevant to neural networks that use QR factorization (orthogonal weight parameterization, Householder-based RNNs, gradient orthogonalization) and to the chunkwise parallel structure in DeltaNet/DeltaProduct where WY factors from different chunks must be communicated and combined.

## Mathematical Form

**Binary-Tree TSQR (Parallel Case):**

Given $A \in \mathbb{R}^{m \times n}$ distributed across $P$ processors in block-row layout, $A = (A_0; A_1; \ldots; A_{P-1})$:

**Stage 0 — Independent local QR:**

$$
A_i = Q_{i,0} R_{i,0}, \quad i = 0, \ldots, P-1
$$

Each $Q_{i,0}$ stored as Householder vectors + $\tau$ scalars (or equivalently, compact WY form $(Y_{i,0}, T_{i,0})$).

**Stage $k$ ($k = 1, \ldots, \log_2 P$) — Pairwise combination:**

Pairs of $R$ factors are stacked and QR-factored:

$$
\begin{pmatrix} R_{i,k-1} \\ R_{j,k-1} \end{pmatrix} = Q_{i,k} R_{i,k}
$$

where $Q_{i,k}$ is stored as a compact WY pair $(Y_{i,k}, T_{i,k})$.

**Complete factorization:**

$$
A = \underbrace{\text{diag}(Q_{0,0}, Q_{1,0}, \ldots, Q_{P-1,0})}_{\text{Stage 0}} \cdot \underbrace{\text{diag}(Q_{0,1}, Q_{1,1}, \ldots)}_{\text{Stage 1}} \cdots \underbrace{Q_{0,\log P}}_{\text{Root}} \cdot R
$$

The product of all these block-diagonal orthogonal matrices is the global $Q$.

**Structured QR of Stacked Triangulars:**

The local QR at each tree node factors a $2n \times n$ matrix consisting of two stacked upper triangulars. The Householder vectors have a special structure (Equation 4 in paper):

$$
Y_{i,k} = \begin{pmatrix} I_n \\ L \end{pmatrix}, \quad L \in \mathbb{R}^{n \times n} \text{ lower triangular}
$$

This means only the $n \times n$ lower triangular $L$ needs to be stored and communicated (not the identity block).

**YT Representation Construction (Algorithm 2 in paper):**

Given $n$ Householder reflectors $\rho_j = I - \tau_j v_j v_j^\top$:

$$
\rho_1 \rho_2 \cdots \rho_n = I + Y T Y^\top
$$

where $Y = (v_1, v_2, \ldots, v_n)$ and $T$ is upper triangular, built column by column:

$$
T = \begin{pmatrix} -\tau_1 & & \\ & \ddots & \\ & & -\tau_n \end{pmatrix} + \text{(off-diagonal from } z_j = -\tau_j T_{1:j-1, 1:j-1} Y_{:,1:j-1}^\top v_j \text{)}
$$

Cost: $O(qn^3/3)$ for $T$ construction at a node with branching factor $q$.

**Trailing Matrix Update (for CAQR):**

At each tree node with WY factor $(Y_i, T_i)$, the update of trailing matrix blocks $(C_0; C_1)$ is:

$$
\begin{pmatrix} \hat{C}_0 \\ \hat{C}_1 \end{pmatrix} = \left(I + \begin{pmatrix} I \\ L \end{pmatrix} T^\top \begin{pmatrix} I & L^\top \end{pmatrix}\right) \begin{pmatrix} C_0 \\ C_1 \end{pmatrix}
$$

Expanding:

$$
W = T^\top (C_0 + L^\top C_1) \qquad \text{[inner product: GEMM]}
$$

$$
\hat{C}_0 = C_0 + W \qquad \text{[local update]}
$$

$$
\hat{C}_1 = C_1 + L W \qquad \text{[local update: GEMM]}
$$

For general branching factor $q$:

$$
D = C_0 + Y_1^\top C_1 + Y_2^\top C_2 + \cdots + Y_{q-1}^\top C_{q-1} \qquad \text{[all-reduce]}
$$

$$
\hat{C}_i = C_i - Y_i \cdot T^\top D, \quad i = 1, \ldots, q-1
$$

$$
\hat{C}_0 = C_0 - T^\top D
$$

**Key Definitions:**

- $m \times n$ — matrix dimensions ($m \gg n$ for tall-skinny)
- $P$ — number of processors (parallel) or blocks (sequential)
- $R_{i,k} \in \mathbb{R}^{n \times n}$ — upper triangular $R$ factor at processor $i$, tree level $k$
- $Q_{i,k}$ — local orthogonal factor, stored as $(Y_{i,k}, T_{i,k})$ compact WY pair
- $Y_{i,k} \in \mathbb{R}^{2n \times n}$ — structured Householder vector matrix (identity block + lower triangular)
- $T_{i,k} \in \mathbb{R}^{n \times n}$ — upper triangular WY factor
- $W$ — fast memory size (for sequential analysis)

## Complexity

**Parallel TSQR vs ScaLAPACK PDGEQRF ($m \times n$ matrix, $P$ processors, $m/P \geq n$):**

| Metric | TSQR | ScaLAPACK PDGEQRF | Lower Bound |
|--------|------|-------------------|-------------|
| FLOPs | $\frac{2mn^2}{P} + \frac{2n^3}{3} \log P$ | $\frac{2mn^2}{P} - \frac{2n^3}{3P}$ | $\Theta\!\left(\frac{mn^2}{P}\right)$ |
| Words | $\frac{n^2}{2} \log P$ | $\frac{n^2}{2} \log P$ | $\frac{n^2}{2} \log P$ |
| **Messages** | $\log P$ | $2n \log P$ | $\log P$ |

**TSQR achieves the optimal message count — a factor of $2n$ fewer messages than ScaLAPACK.**

**Sequential TSQR ($m \times n$ matrix, fast memory $W$ words):**

| Metric | Seq. TSQR | Blocked Householder QR | Lower Bound |
|--------|-----------|----------------------|-------------|
| FLOPs | $2mn^2$ | $2mn^2$ | $\Theta(mn^2)$ |
| **Words transferred** | $2mn$ | $\frac{m^2 n^2}{2W}$ | $2mn$ |
| **Messages** | $\frac{2mn}{\tilde{W}}$ | $\frac{mn^2}{2W}$ | $\frac{2mn}{W}$ |

where $\tilde{W} = W - n(n+1)/2$. Sequential TSQR transfers **optimal words** ($2mn$) and has $\Theta(n)$ times fewer messages.

**Parallel CAQR vs ScaLAPACK ($m \times n$ matrix, $P_r \times P_c$ processor grid, block size $b$):**

| Metric | CAQR | ScaLAPACK | Reduction |
|--------|------|-----------|-----------|
| Messages | $\frac{3n}{b} \log P_r + \frac{2n}{b} \log P_c$ | $3n \log P_r + \frac{2n}{b} \log P_c$ | $\Theta(b)$ fewer |
| Optimal msgs | $\Theta\!\left(\sqrt{\frac{nP}{m}} \cdot \text{polylog}\right)$ | $\Theta\!\left(n \cdot \text{polylog}\right)$ | $\Theta\!\left(\sqrt{\frac{mn}{P}}\right)$ fewer |

**Memory:** $O(mn/P + n^2)$ per processor — the $n^2$ term is for storing the local $R$ factor and the $T$ matrix at each tree node. The tree requires $O(\log P)$ WY pairs, each needing $n(n+1)/2$ storage for $T$ plus the structured $Y$ (just the lower triangular part, $n(n-1)/2$ entries).

**Numerical Stability:**

| Algorithm | $\|I - Q^\top Q\|_2$ bound |
|-----------|---------------------------|
| Householder QR | $O(\epsilon)$ |
| **TSQR** | $O(\epsilon)$ ✓ |
| Modified Gram-Schmidt | $O(\epsilon \cdot \kappa(A))$ |
| Cholesky QR | $O(\epsilon \cdot \kappa(A)^2)$ |

TSQR is **unconditionally numerically stable** — as stable as Householder QR, regardless of the condition number of $A$.

## Applicability

- **Parallel orthogonal weight construction:** When parameterizing neural network weights as products of Householder reflections distributed across multiple GPUs (tensor parallelism), TSQR provides the communication-optimal way to compute the QR factorization. The tree-structured WY avoids the $2n \log P$ message bottleneck of standard approaches.

- **DeltaNet / DeltaProduct inter-chunk merging:** In chunkwise parallel training, each chunk produces a compact WY representation of its state transition. Merging these across chunks (and across GPUs in sequence parallelism) follows the same tree-reduction pattern as TSQR. The structured WY factors at each tree node enable the merge via GEMMs (trick 159).

- **Gradient orthogonalization:** Methods like SOAP (Shampoo-style) that orthogonalize gradient matrices can use TSQR when the gradient is distributed across data-parallel workers. The $\log P$ message count (vs. $2n \log P$) becomes significant at large $P$.

- **Large-batch orthogonal projections:** When computing QR of activation matrices for whitening or decorrelation, the activations are naturally partitioned across the batch dimension (tall-skinny). TSQR is the optimal algorithm for this layout.

- **Sequential out-of-core QR:** For neural network layers too large to fit in GPU memory (e.g., very large embedding matrices), sequential TSQR with a flat tree achieves optimal data transfer between GPU memory and CPU/NVMe, transferring only $2mn$ words vs. $m^2 n^2 / (2W)$ for blocked Householder.

- **Cache-oblivious recursive variant:** The TSQR tree can be combined with the recursive WY merge (trick 159) at each node, making the entire algorithm cache-oblivious — no block-size tuning needed.

## Limitations

- **Tree overhead for small $n$:** Each tree node performs a QR of a $2n \times n$ structured matrix costing $(2/3)n^3$ FLOPs. With $\log P$ levels, the total tree cost is $(2/3)n^3 \log P$. For large $n$ relative to $m/P$, this term dominates and TSQR offers no benefit.

- **Implicit Q representation:** The Q factor is stored as a tree of $(Y_i, T_i)$ pairs, not as a single WY pair. Applying $Q$ or $Q^\top$ to a vector requires traversing the tree ($\log P$ steps), each involving a GEMM with the local $(Y_i, T_i)$. This is efficient for applying $Q$ to tall matrices (trailing matrix update) but has overhead for applying $Q$ to single vectors.

- **Reconstruction cost for explicit WY:** If a single compact WY representation is needed (e.g., for the parallel scan in trick 159), the tree must be "flattened" by successively merging WY pairs from leaves to root. This costs $O(mn^2 / P + n^3 \log P)$ — the same as the factorization itself. See Ballard et al. (2014) for the reconstruction algorithm.

- **Not directly applicable to non-QR decompositions:** TSQR exploits the specific algebraic structure of QR (the reduction property). LU, Cholesky, and eigenvalue decompositions have different communication patterns and cannot directly use this tree structure.

- **Synchronization at tree levels:** In the parallel binary tree, each level requires a synchronization point (processors must exchange $R$ factors). For GPUs connected via NVLink/NVSwitch, the $\log P$ synchronizations are fast; for distributed clusters, the latency per level may dominate for small $n$.

- **Structured QR exploits sparsity:** The $(2/3)n^3$ cost per node assumes exploiting the stacked-triangular structure. Without this (i.e., treating the $2n \times n$ input as dense), the cost is $2n^3$ — a 3× increase. Implementations must carefully handle the structure.

## Implementation Notes

```python
import torch

def tsqr_binary_tree(A_blocks):
    """
    Parallel TSQR via binary tree reduction.

    Args:
        A_blocks: list of P tensors, each (m/P, n) — block rows of A

    Returns:
        R: (n, n) upper triangular R factor
        tree: list of (Y, T) pairs at each tree level for implicit Q
    """
    P = len(A_blocks)
    n = A_blocks[0].shape[1]

    # Stage 0: Independent local QR on each block
    local_factors = []
    R_factors = []
    for i in range(P):
        Q_i, R_i = torch.linalg.qr(A_blocks[i])
        # Store Q as Householder vectors (compact WY form)
        # In practice, use torch.geqrf for (tau, Y) representation
        local_factors.append(Q_i)  # simplified
        R_factors.append(R_i)

    tree = [local_factors]  # Level 0

    # Stages 1 to log2(P): Pairwise combination
    level = 1
    while len(R_factors) > 1:
        new_R = []
        level_factors = []
        for i in range(0, len(R_factors), 2):
            if i + 1 < len(R_factors):
                # Stack two R factors (both n×n upper triangular)
                stacked = torch.cat([R_factors[i], R_factors[i+1]], dim=0)  # (2n, n)

                # Structured QR exploiting triangular structure
                Q_node, R_node = structured_qr_stacked_triangular(
                    R_factors[i], R_factors[i+1]
                )
                new_R.append(R_node)
                level_factors.append(Q_node)  # (Y, T) pair
            else:
                new_R.append(R_factors[i])
                level_factors.append(None)  # identity

        R_factors = new_R
        tree.append(level_factors)
        level += 1

    R = R_factors[0]
    return R, tree


def structured_qr_stacked_triangular(R1, R2):
    """
    QR factorization of (R1; R2) where R1, R2 are n×n upper triangular.

    Exploits structure: Householder vectors have form (e_i; l_i)
    where l_i has nonzeros only in positions 1:i.

    Cost: (2/3)n^3 flops (vs 2n^3 for unstructured)

    Returns:
        (Y, T): compact WY representation of Q
                 Y = (I; L) where L is n×n lower triangular
        R: n×n upper triangular R factor
    """
    n = R1.shape[0]
    # Working copy
    R = R1.clone()
    L = torch.zeros(n, n, dtype=R1.dtype, device=R1.device)

    taus = torch.zeros(n, dtype=R1.dtype, device=R1.device)

    for j in range(n):
        # Gather: entries from R[j,j] and R2[0:j+1, j]
        # (exploiting that R2 is upper triangular, only entries 0..j are nonzero)
        w = torch.cat([R[j:j+1, j], R2[:j+1, j]])

        # Compute Householder reflection
        tau, v = householder(w)
        taus[j] = tau

        # v[0] = 1 (by convention), store v[1:] in L
        L[:j+1, j] = v[1:]

        # Apply to remaining columns j+1:n
        # Only affects R[j, j+1:n] and R2[0:j+1, j+1:n]
        if j < n - 1:
            x = torch.cat([R[j:j+1, j+1:], R2[:j+1, j+1:]], dim=0)
            x -= tau * v.unsqueeze(1) @ (v.unsqueeze(0) @ x)
            R[j, j+1:] = x[0]
            R2[:j+1, j+1:] = x[1:]

        R[j, j] = w[0] - tau * v[0] * (v @ w)  # simplified

    # Build T matrix from taus and L (Algorithm 2)
    T = build_T_from_householder(L, taus)

    # Y = (I_n; L) — only L needs to be stored/communicated
    return (L, T), R


def householder(x):
    """Compute Householder reflection: H = I - tau * v * v^T such that Hx = ||x|| e_1"""
    sigma = torch.norm(x[1:])
    if sigma == 0:
        return 0.0, x
    norm_x = torch.sqrt(x[0]**2 + sigma**2)
    v = x.clone()
    if x[0] <= 0:
        v[0] = x[0] - norm_x
    else:
        v[0] = -sigma**2 / (x[0] + norm_x)
    tau = 2 * v[0]**2 / (sigma**2 + v[0]**2)
    v = v / v[0]  # normalize so v[0] = 1
    return tau, v


def build_T_from_householder(Y_lower, taus):
    """
    Build upper triangular T from Householder vectors and scalars.
    Algorithm 2 from Demmel et al.

    rho_1 * rho_2 * ... * rho_n = I + Y T Y^T
    """
    n = len(taus)
    T = torch.zeros(n, n, dtype=Y_lower.dtype, device=Y_lower.device)

    for j in range(n):
        if j == 0:
            T[0, 0] = -taus[0]
        else:
            # v_j has structure: (e_j; Y_lower[:j+1, j])
            # Y[:, :j] has structure: (I[:, :j]; Y_lower[:, :j])
            # Y[:,:j]^T v_j = I[:j, :]^T e_j + Y_lower[:,:j]^T Y_lower[:,j]
            #               = delta_{ij} + Y_lower[:,:j]^T Y_lower[:j+1,j]
            YtV = Y_lower[:j+1, :j].T @ Y_lower[:j+1, j]  # (j,)
            YtV[:j] += torch.eye(j, dtype=Y_lower.dtype, device=Y_lower.device)[:, min(j, j)]  # simplified

            z = -taus[j] * T[:j, :j] @ YtV
            T[:j, j] = z
            T[j, j] = -taus[j]

    return T


def apply_Q_transpose_tsqr(tree, x_blocks):
    """
    Apply Q^T to a vector/matrix distributed as x_blocks,
    traversing the TSQR tree from leaves to root.

    At each node, apply the local WY factor:
        (I + Y T Y^T) @ (x_parent; x_child)
    """
    # Level 0: apply local Q_i^T to each block
    for i in range(len(x_blocks)):
        if tree[0][i] is not None:
            x_blocks[i] = tree[0][i].T @ x_blocks[i]

    # Levels 1 to log P: apply tree WY factors
    stride = 1
    for level in range(1, len(tree)):
        for i in range(0, len(x_blocks), 2 * stride):
            j = i + stride
            if j < len(x_blocks) and tree[level][i // (2*stride)] is not None:
                L, T = tree[level][i // (2*stride)]
                n = T.shape[0]
                # Extract top n rows from each block
                c0 = x_blocks[i][:n]  # from parent R
                c1 = x_blocks[j][:n]  # from child R

                # W = T^T (c0 + L^T c1)  [GEMM]
                W = T.T @ (c0 + L.T @ c1)

                # Update
                x_blocks[i][:n] = c0 + W
                x_blocks[j][:n] = c1 + L @ W

        stride *= 2

    return x_blocks
```

**Communication pattern analysis for GPU clusters:**

1. **NVLink/NVSwitch (intra-node):** With 8 GPUs per node, the binary tree has 3 levels. Each level exchanges one $n \times n$ upper triangular matrix ($n^2/2$ words). Total: $3n^2/2$ words in 3 messages. For $n = 128$ (typical head dimension), this is ~98 KB — fits in a single NVLink transfer.

2. **Inter-node (InfiniBand):** For sequence parallelism across nodes, TSQR's $\log P$ messages (vs $2n \log P$) become critical. At $n = 128, P = 64$: TSQR sends 6 messages; ScaLAPACK sends 768 messages.

3. **Batched TSQR for multi-head attention:** With $H$ attention heads, all heads can perform TSQR simultaneously as a batched operation. The $n \times n$ R factors from all heads can be packed into a single communication buffer.

4. **Flat tree for sequential (out-of-core):** When the tall-skinny matrix doesn't fit in GPU memory, a flat tree (left-looking) processes one block at a time, achieving optimal data transfer of $2mn$ words between GPU HBM and CPU DRAM.

## References

- Demmel, J., Grigori, L., Hoemmen, M., & Langou, J. (2012). Communication-Optimal Parallel and Sequential QR and LU Factorizations. *SIAM Review*, 54(1), 3–49. (Extended version: UC Berkeley EECS Tech Report 2008-89, 2013.)
- Ballard, G., Demmel, J., Grigori, L., Jacquelin, M., Nguyen, H. D., & Solomonik, E. (2014). Reconstructing Householder Vectors from Tall-Skinny QR. *IPDPS 2014*, 1159–1170.
- Anderson, M., Ballard, G., Demmel, J., & Keutzer, K. (2011). Communication-Avoiding QR Decomposition for GPUs. *IPDPS 2011*, 48–58.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Schreiber, R. & Van Loan, C. (1989). A Storage-Efficient WY Representation for Products of Householder Transformations. *SIAM J. Sci. Stat. Comput.*, 10(1), 53–57.
- Elmroth, E. & Gustavson, F. G. (2000). Applying Recursion to Serial and Parallel QR Factorization Leads to Better Performance. *IBM J. Res. Develop.*, 44(4), 605–624.

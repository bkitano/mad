# 139: UT Transform for Householder Accumulation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Joffrain, Low, Quintana-Ortí, van de Geijn & Van Zee (2006); Yang, Wang, Zhang, Shen & Kim (2024)
**Paper**: [papers/ut-transform-joffrain-2006.pdf] (original), [papers/deltanet-chunkwise-parallel.pdf] (application to DeltaNet)
**Documented**: 2026-02-12

## Description

The UT transform converts a sequential recurrence for accumulating products of generalized Householder reflections into a system of equations solvable by **forward substitution on a lower-triangular matrix**, followed by **matrix multiplications**. This is the key trick that enables DeltaNet to leverage GPU tensor cores for training.

In DeltaNet, the state update $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ is a product of rank-1 modifications (generalized Householder transformations). The WY representation compactly stores these products as $P_n = I - \sum_{t=1}^n w_t k_t^\top$, but the vectors $w_t$ are defined by a **sequential recurrence** where each $w_r$ depends on all previous $w_1, \ldots, w_{r-1}$. This recurrence cannot use tensor cores.

The UT transform reformulates this recurrence as:
1. Build a **lower-triangular matrix** $L$ via a single matmul ($O(C^2 d)$, tensor-core friendly)
2. **Invert** $(I + L)$ via forward substitution ($O(C^2)$, cheap)
3. Compute final WY factors $W, U$ via **two matmuls** ($O(C^2 d)$ each, tensor-core friendly)

Since $d \gg 1$ (typically 64–256), the matmuls dominate and achieve high tensor-core utilization, while the $O(C^2)$ forward substitution is negligible.

## Mathematical Form

**The Problem: Sequential WY Recurrence**

Within a chunk of $C$ tokens, the WY vectors $w_r$ and $u_r$ are defined recursively (Eq. 7 in Yang et al.):

$$
w^r_{[t]} = \beta^r_{[t]} \left( k^r_{[t]} - \sum_{i=1}^{r-1} w^i_{[t]} (k^{i\top}_{[t]} k^r_{[t]}) \right)
$$

$$
u^r_{[t]} = \beta^r_{[t]} \left( v^r_{[t]} - \sum_{i=1}^{r-1} u^i_{[t]} (k^{i\top}_{[t]} k^r_{[t]}) \right)
$$

Each step $r$ depends on all previous steps $1, \ldots, r-1$. This is inherently sequential and **cannot use tensor cores**.

**The Solution: UT Transform (Eq. 10–11 in Yang et al.)**

Define the lower-triangular adjacency matrix:

$$
B_{[t]} = \text{diag}(\beta_{[t]})
$$

$$
L_{[t]} = \text{tril}(B_{[t]} K_{[t]} K_{[t]}^\top, -1)
$$

where $\text{tril}(\cdot, -1)$ extracts the strictly lower-triangular part. Then:

$$
T_{[t]} = (I + L_{[t]})^{-1} B_{[t]}
$$

$$
W_{[t]} = T_{[t]} K_{[t]}, \qquad U_{[t]} = T_{[t]} V_{[t]}
$$

**Derivation (from DeltaNet Appendix B.2):**

Writing the recurrence in matrix form for row $r$ of $W$:

$$
W_{[t]}[r, :] = \beta^r_{[t]} K_{[t]}[r, :] - \beta^r_{[t]} \sum_{i=1}^{r-1} W_{[t]}[i, :] \cdot (K_{[t]}[i, :] \cdot K_{[t]}[r, :]^\top)
$$

This is a linear system:

$$
W_{[t]} + L_{[t]} W_{[t]} = B_{[t]} K_{[t]}
$$

$$
(I + L_{[t]}) W_{[t]} = B_{[t]} K_{[t]}
$$

Since $L_{[t]}$ is strictly lower-triangular, $(I + L_{[t]})$ is unit lower-triangular and invertible by forward substitution. The same derivation applies to $U_{[t]}$ by replacing $K$ with $V$.

**Connection to Joffrain et al. (2006):**

Joffrain et al. proved that for a product of Householder reflections $(I - u_i u_i^\top / \tau_i)$, the compact WY form $I - U S U^\top$ has the property:

$$
S^{-1} + S^{-\top} = U^\top U
$$

They showed $S$ can be computed as: (1) compute $T = \text{striu}(U^\top U) + \frac{1}{2}\text{diag}(U^\top U)$, (2) invert $T$ to get $S = T^{-1}$. The **UT transform** eliminates the explicit inversion by computing $T^{-1}W$ via a triangular solve (DTRSM) instead of $SW$ via a triangular multiply (DTRMM). This saves $k^3$ FLOPs and is numerically more stable.

DeltaNet adapts this idea to the **lower-triangular** case (since the causal mask goes forward in time), giving the formulation above.

**Key Definitions:**

- $K_{[t]} \in \mathbb{R}^{C \times d}$ — stacked key vectors within chunk $t$
- $V_{[t]} \in \mathbb{R}^{C \times d}$ — stacked value vectors within chunk $t$
- $\beta_{[t]} \in \mathbb{R}^{C}$ — per-token learning rates within chunk $t$
- $B_{[t]} = \text{diag}(\beta_{[t]}) \in \mathbb{R}^{C \times C}$ — diagonal matrix of learning rates
- $L_{[t]} \in \mathbb{R}^{C \times C}$ — strictly lower-triangular "adjacency" matrix
- $T_{[t]} \in \mathbb{R}^{C \times C}$ — the UT transform matrix
- $W_{[t]} \in \mathbb{R}^{C \times d}$ — accumulated WY decay factors
- $U_{[t]} \in \mathbb{R}^{C \times d}$ — accumulated WY value factors
- $C$ — chunk size (typically 64 or 128)
- $d$ — head dimension (typically 64–256)

**Once $W$ and $U$ are computed, the chunkwise parallel output follows standard linear attention form (Eq. 8–9 in Yang et al.):**

$$
S_{[t+1]} = S_{[t]} + (U_{[t]} - W_{[t]} S_{[t]}^\top)^\top K_{[t]}
$$

$$
O_{[t]} = Q_{[t]} S_{[t]}^\top + (Q_{[t]} K_{[t]}^\top \odot M_C)(U_{[t]} - W_{[t]} S_{[t]}^\top)
$$

where $M_C$ is the $C \times C$ causal mask.

## Complexity

| Operation | Sequential WY | With UT Transform |
|-----------|---------------|-------------------|
| Build $L = \text{tril}(BKK^\top)$ | — | $O(C^2 d)$ matmul ✓ |
| Forward substitution $(I+L)^{-1}$ | — | $O(C^2)$ |
| Compute $W = TK$ | $O(C^2 d)$ sequential | $O(C^2 d)$ matmul ✓ |
| Compute $U = TV$ | $O(C^2 d)$ sequential | $O(C^2 d)$ matmul ✓ |
| **Total** | $O(C^2 d)$ **sequential** | $O(C^2 d)$ **parallel** + $O(C^2)$ seq |

✓ = maps to tensor cores

**The key insight:** Both approaches are $O(C^2 d)$ in FLOPs, but the UT transform converts the dominant computation from sequential scalar operations to parallel matrix multiplications. Since tensor cores deliver 10–16× higher throughput than scalar operations on modern GPUs (A100/H100), this translates to large wall-clock speedups.

**Memory:** $O(C^2)$ for the $T$ matrix (small since $C \leq 256$), plus $O(Cd)$ for $W, U$.

**Speedup (empirical, from Yang et al. Figure 1):**

| Sequence Length | Head Dim 64 | Head Dim 128 | Head Dim 256 |
|-----------------|-------------|--------------|--------------|
| 1K | ~3× | ~5× | ~8× |
| 4K | ~8× | ~15× | ~25× |
| 16K | ~15× | ~25× | ~35× |

Speedups grow with both sequence length $L$ and head dimension $d$ because larger dimensions make tensor-core matmuls more efficient.

## Applicability

- **DeltaNet** (primary application): Enables hardware-efficient chunkwise parallel training of linear attention with the delta rule
- **Gated DeltaNet**: Same UT transform applies with input-dependent gating $\alpha_t$, giving $S_t = S_{t-1} \alpha_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$
- **Any model with Householder-structured state updates**: The technique applies whenever the state transition involves products of rank-1 modifications $(I - \beta_t u_t u_t^\top)$
- **SSD-DeltaNet hybrid** (proposed): The UT transform is the intra-chunk component; combining with SSD-style sub-block decomposition may yield further speedups via $Q \times Q$ matmul tiles
- **Orthogonal RNNs**: Models parameterizing transition matrices as products of Householder reflections can use this for efficient training

## Limitations

- **Forward substitution is sequential**: The $O(C^2)$ forward substitution loop has $C$ sequential steps. While cheap relative to the $O(C^2 d)$ matmuls, it becomes a bottleneck at very small $d$ or very large $C$.
- **Chunk size constraints**: $C$ must be a multiple of 16 for tensor core alignment. Practical values are 64–256; larger $C$ increases the $O(C^2)$ sequential overhead.
- **Only for rank-1 updates**: The technique exploits the specific structure $I - \beta k k^\top$. More general state transitions (e.g., full matrix multiplication) cannot use this trick.
- **Numerical precision**: Forward substitution with lower-triangular matrices is well-conditioned, but mixed-precision (FP16/BF16) training may require careful handling of the $T$ matrix accumulation.
- **Inter-chunk state is $O(d^2)$**: The UT transform optimizes intra-chunk computation, but the $d \times d$ state materialization at chunk boundaries remains, limiting scalability for very large $d$.
- **Speed still lags GLA**: DeltaNet's training speed still falls behind GLA (Gated Linear Attention) because GLA's elementwise state transitions avoid the $d \times d$ state entirely. The UT transform narrows but does not close this gap.

## Implementation Notes

```python
# UT Transform for DeltaNet (from Yang et al. 2024, Appendix D)
# PyTorch-like pseudocode for the forward pass

def chunk_delta_rule_forward(Q, K, V, beta, C):
    """
    Q/K/V: query, key, value of shape [L, d]
    beta: learning rate of shape [L]
    C: chunk size
    """
    L, d = Q.shape

    # Reshape into chunks
    Q, K, V = map(lambda x: x.reshape(-1, C, d), [Q, K, V])
    beta = beta.reshape(-1, C)

    # Scale K and V by beta
    K_beta = K * beta.unsqueeze(-1)   # (n_chunks, C, d)
    V_beta = V * beta.unsqueeze(-1)   # (n_chunks, C, d)

    # === UT TRANSFORM (Eq. 10) ===
    # Step 1: Build L = tril(diag(beta) @ K @ K^T, -1) via matmul
    # This is the TENSOR CORE matmul: (C×d) @ (d×C) → (C×C)
    T = -(K_beta @ K.transpose(-1, -2)).tril(-1)

    # Step 2: Forward substitution to compute (I + L)^{-1}
    # O(C^2) sequential, but C is small (64-256)
    for i in range(1, C):
        T[:, i, :i] = T[:, i, :i] + (T[:, i, :, None] * T[:, :, :i]).sum(-2)
    T += torch.eye(C)

    # Step 3: Compute W = T @ K_beta, U = T @ V_beta (Eq. 11)
    # These are TENSOR CORE matmuls: (C×C) @ (C×d) → (C×d)
    W = T @ K_beta    # WY decay factors
    U = T @ V_beta    # WY value factors

    # === CHUNKWISE PARALLEL (Eq. 8-9) ===
    S = torch.zeros(d, d)   # inter-chunk state
    O = torch.empty_like(V)

    for i in range(L // C):
        q_i, k_i, w_i = Q[i], K[i], W[i]
        u_i = U[i] - w_i @ S          # corrected values

        o_inter = q_i @ S             # inter-chunk contribution

        # Intra-chunk: masked matmul (tensor core)
        A_i = (q_i @ k_i.t()).tril()  # (C×C) causal attention
        o_intra = A_i @ u_i           # (C×d) output

        S += k_i.t() @ u_i            # update state: (d×d)
        O[i] = o_intra + o_inter

    return O.reshape(L, d)
```

**Key implementation details:**

1. **The forward substitution loop** (lines 17-18) is the only sequential part of the UT transform. It runs for $C$ iterations, each doing $O(C)$ work. In Triton, this can be vectorized across the batch and head dimensions.

2. **Tensor core alignment**: $C$ should be a multiple of 16. The DeltaNet paper uses $C = 64$ or $C = 128$ in practice.

3. **The matmuls `K_beta @ K.T`, `T @ K_beta`, `T @ V_beta`** are the dominant operations and map directly to tensor cores (GEMM operations). These constitute $> 90\%$ of FLOPs when $d \geq 64$.

4. **Graph-theoretic interpretation** (from Songlin Yang's blog): The matrix $L_{ij} = -\beta_i (k_i^\top k_j)$ for $i > j$ can be viewed as edge weights in a directed acyclic graph, where entry $[i, j]$ of $(I + L)^{-1}$ gives the sum of weights of all paths from $j$ to $i$. The UT transform computes these "accumulated influence paths" efficiently.

## References

- Joffrain, T., Low, T. M., Quintana-Ortí, E. S., van de Geijn, R. A., & Van Zee, F. G. (2006). Accumulating Householder Transformations, Revisited. *ACM Trans. Math. Softw.*, 32(2), 169–179.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *NeurIPS 2024*. arXiv:2406.06484.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. *ICLR 2025*. arXiv:2412.06464.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. *SIAM J. Sci. Stat. Comput.*, 8(1), 2–13.
- Dominguez, A. E. T. & Quintana-Ortí, E. S. (2018). Fast Blocking of Householder Reflectors on Graphics Processors. *PDP 2018*.
- Yang, S. (2024). DeltaNet Explained (Part II). Blog post: https://sustcsonglin.github.io/blog/2024/deltanet-2/

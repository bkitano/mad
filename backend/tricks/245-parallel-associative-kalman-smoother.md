# 245: Parallel Associative Kalman Smoother

**Category**: parallelization
**Gain type**: efficiency
**Source**: Särkkä & García-Fernández (2020), "Temporal Parallelization of Bayesian Smoothers"; Särkkä & García-Fernández (2025), "On The Performance of Prefix-Sum Parallel Kalman Filters and Smoothers on GPUs"
**Paper**: [papers/parallel-associative-kalman-smoother.pdf], [papers/prefix-sum-parallel-kalman-gpu.pdf]
**Documented**: 2026-02-15

## Description

Reformulates the sequential Kalman filter and Rauch-Tung-Striebel (RTS) smoother as **parallel prefix sums over custom associative operators**, reducing temporal complexity from $O(T)$ to $O(\log T)$ span on $T$ parallel processors. This is the foundational technique that enables parallel training of deep state space models (S5, Mamba, etc.) over sequence length. The key insight is that the Bayesian filtering and smoothing recurrences — which are inherently sequential — can be recast as associative binary operations on tuples of sufficient statistics, making them compatible with parallel scan algorithms (Blelloch, Hillis-Steele, Ladner-Fischer).

This extends the scalar parallel scan (already used for diagonal SSMs) to **matrix-valued recurrences** where each element is parameterized by a 5-tuple $(A_k, b_k, C_k, \eta_k, J_k)$ and the binary operator involves matrix inversions and multiplications. A novel two-filter smoother variant enables even further parallelism by running forward and backward scans simultaneously on separate GPUs.

## Mathematical Form

**Core Recurrence (Linear/Gaussian State Space Model):**

$$
x_k = F_{k-1} x_{k-1} + u_{k-1} + q_{k-1}, \quad y_k = H_k x_k + d_k + r_k
$$

where $q_{k-1} \sim \mathcal{N}(0, Q_{k-1})$ and $r_k \sim \mathcal{N}(0, R_k)$.

**Filtering Element:**

Each time step $k$ is encoded as a tuple $a_k = (A_k, b_k, C_k, \eta_k, J_k)$ where:

- $S_k = H_k Q_{k-1} H_k^\top + R_k$
- $K_k = Q_{k-1} H_k^\top S_k^{-1}$
- $A_k = (I_{n_x} - K_k H_k) F_{k-1}$
- $b_k = u_{k-1} + K_k(y_k - H_k u_{k-1} - d_k)$
- $C_k = (I_{n_x} - K_k H_k) Q_{k-1}$
- $\eta_k = F_{k-1}^\top H_k^\top S_k^{-1}(y_k - H_k u_{k-1} - d_k)$
- $J_k = F_{k-1}^\top H_k^\top S_k^{-1} H_k F_{k-1}$

**Associative Filtering Operator $\otimes$:**

Given elements $(A_i, b_i, C_i, \eta_i, J_i)$ and $(A_j, b_j, C_j, \eta_j, J_j)$:

$$
A_{i,j} = A_j (I_{n_x} + C_i J_j)^{-1} A_i
$$

$$
b_{i,j} = A_j (I_{n_x} + C_i J_j)^{-1} (b_i + C_i \eta_j) + b_j
$$

$$
C_{i,j} = A_j (I_{n_x} + C_i J_j)^{-1} C_i A_j^\top + C_j
$$

$$
\eta_{i,j} = A_i^\top (I_{n_x} + J_j C_i)^{-1} (\eta_j - J_j b_i) + \eta_i
$$

$$
J_{i,j} = A_i^\top (I_{n_x} + J_j C_i)^{-1} J_j A_i + J_i
$$

This operator is **proven associative** (Appendix of Särkkä 2020), enabling parallel prefix sum computation.

**Smoothing Operator $\otimes$:**

For RTS smoothing, element $a_k = (E_k, g_k, L_k)$ with operator:

$$
E_{i,j} = E_i E_j, \quad g_{i,j} = E_i g_j + g_i, \quad L_{i,j} = E_i L_j E_i^\top + L_i
$$

**Two-Filter Smoother (Novel):**

Runs forward and backward Kalman filters **simultaneously** (on separate GPUs), then combines:

$$
\bar{x}_{k|T} = (I + P_{k|k} J_{k|k+1:T})^{-1}(\bar{x}_{k|k} + P_{k|k} \eta_{k|k+1:T})
$$

$$
P_{k|T} = (I + P_{k|k} J_{k|k+1:T})^{-1} P_{k|k}
$$

## Complexity

| Operation | Sequential | With Parallel Scan |
|-----------|-----------|-------------------|
| Kalman filter (span) | $O(T \cdot n_x^3)$ | $O(\log T \cdot n_x^3)$ |
| RTS smoother (span) | $O(T \cdot n_x^3)$ | $O(\log T \cdot n_x^3)$ |
| Two-filter smoother (span) | $O(T \cdot n_x^3)$ | $O(\log T \cdot n_x^3)$ |

**Work complexity:** $O(T \cdot n_x^3)$ for sequential vs $O(T \log T \cdot n_x^3)$ for parallel (constant factor ~4-8x more total FLOPs).

**Memory:** $O(T \cdot n_x^2)$ for storing all elements (matrices per time step).

**Key tradeoff:** Wall-clock time drops from $O(T)$ to $O(\log T)$ at the cost of ~4-8x more total FLOPs. This is favorable on GPUs where parallelism is cheap but sequential latency is expensive.

## Applicability

- **Deep state space models (S5, S4, Mamba)**: This is the core technique enabling $O(\log T)$ parallel training of linear recurrences over sequence length. S5 directly uses this for its parallel scan.
- **Linear attention models**: Any linear recurrence $M_t = \Theta_t \diamond M_{t-1} + \hat{M}_t$ can be parallelized via this framework.
- **Bidirectional sequence models**: The two-filter smoother enables true bidirectional processing with independent forward/backward scans.
- **Kalman filter layers in neural networks**: Deep Kalman filters or probabilistic SSM layers benefit directly.

## Limitations

- **Matrix-valued operator cost**: Each associative operator application involves matrix inversions $(I + CJ)^{-1}$ of size $n_x \times n_x$. For large state dimensions, this is $O(n_x^3)$ per operation, which can dominate.
- **Work overhead**: Total FLOPs increase by ~4-8x vs sequential. On hardware with limited parallelism, the overhead may not be recovered.
- **Numerical stability**: The matrix inversions $(I + C_i J_j)^{-1}$ can be ill-conditioned. The GPU paper uses QR factorization with 0.99 scaling for stability.
- **State dimension scaling**: Practical for small-to-moderate state dimensions ($n_x \leq 64$). For very large states, the $O(n_x^3)$ per-operator cost makes each scan step expensive and memory-heavy.
- **Diagonal SSMs avoid this entirely**: Models like S4D/Mamba use diagonal state matrices, reducing the matrix-valued scan to independent scalar scans — far cheaper on GPU. This trick is most relevant when full matrix recurrences are needed.

## GPU Efficiency Analysis

**Memory access pattern**: Each scan operator reads/writes 5 matrices of size $n_x \times n_x$ — for $n_x = 4$ this is ~320 bytes per element (fp32). For $n_x = 64$, this is ~80KB per element, potentially exceeding shared memory per thread.

**Parallelism**: The scan itself maps well to GPU warps (each warp handles one level of the tree). The Blelloch scan achieves $O(T)$ work with $O(\log T)$ span. Ladner-Fischer (in-place) variant saves memory.

**Tensor core utilization**: The matrix multiplications within the associative operator ($A_j \cdot (I + C_i J_j)^{-1} \cdot A_i$) can leverage tensor cores if $n_x$ is a multiple of 16. For small $n_x$ (typical in SSMs), this may not saturate tensor cores.

**Practical GPU results (Särkkä 2025)**: On NVIDIA A100 with CUDA/Julia, the parallel Kalman filter achieves speedups starting at $T \approx 20$ time steps, reaching significant speedups at $T = 10^4$--$10^6$. Blelloch and Ladner-Fischer algorithms perform best at large $T$.

## Implementation Notes

```python
# Pseudocode: Parallel Kalman Filter via Associative Scan
# Each element a_k = (A_k, b_k, C_k, eta_k, J_k)

def compute_initial_elements(y, F, H, Q, R, u, d, x0, P0):
    """Compute elements a_k for k=1..T in parallel."""
    elements = []
    for k in range(1, T+1):
        S_k = H[k] @ Q[k-1] @ H[k].T + R[k]
        K_k = Q[k-1] @ H[k].T @ inv(S_k)
        A_k = (I - K_k @ H[k]) @ F[k-1]
        b_k = u[k-1] + K_k @ (y[k] - H[k] @ u[k-1] - d[k])
        C_k = (I - K_k @ H[k]) @ Q[k-1]
        eta_k = F[k-1].T @ H[k].T @ inv(S_k) @ (y[k] - H[k] @ u[k-1] - d[k])
        J_k = F[k-1].T @ H[k].T @ inv(S_k) @ H[k] @ F[k-1]
        elements.append((A_k, b_k, C_k, eta_k, J_k))
    return elements

def filtering_operator(elem_i, elem_j):
    """Associative binary operator for Kalman filtering."""
    Ai, bi, Ci, etai, Ji = elem_i
    Aj, bj, Cj, etaj, Jj = elem_j
    M = inv(I + Ci @ Jj)
    A_ij = Aj @ M @ Ai
    b_ij = Aj @ M @ (bi + Ci @ etaj) + bj
    C_ij = Aj @ M @ Ci @ Aj.T + Cj
    eta_ij = Ai.T @ inv(I + Jj @ Ci) @ (etaj - Jj @ bi) + etai
    J_ij = Ai.T @ inv(I + Jj @ Ci) @ Jj @ Ai + Ji
    return (A_ij, b_ij, C_ij, eta_ij, J_ij)

# Run parallel prefix sum (e.g., Blelloch scan)
prefix_sums = parallel_scan(elements, filtering_operator)

# Extract Kalman filter means and covariances
for k in range(T):
    x_bar_k = prefix_sums[k].b  # filtering mean
    P_k = prefix_sums[k].C      # filtering covariance
```

## References

- Särkkä, S. & García-Fernández, Á.F. (2020). "Temporal Parallelization of Bayesian Smoothers." arXiv:1905.13002.
- Särkkä, S. & García-Fernández, Á.F. (2025). "On The Performance of Prefix-Sum Parallel Kalman Filters and Smoothers on GPUs." arXiv:2511.10363.
- Smith, J.T.H., Warrington, A., & Linderman, S. (2023). "Simplified State Space Layers for Sequence Modeling" (S5). ICLR 2023.
- Blelloch, G.E. (1989). "Scans as primitive parallel operations." IEEE Trans. Computers.
- Gu, A. & Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."

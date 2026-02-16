# 048: FlashSinkhorn: IO-Aware Entropic Optimal Transport

**Category**: kernel
**Gain type**: efficiency
**Source**: Ye, Li, Yu, Chang, Chu & Wertheimer (U. Albany, UNC Charlotte, IBM Research, 2026)
**Paper**: [papers/flashsinkhorn-io-aware-eot.pdf]
**Documented**: 2026-02-15

## Description

Sinkhorn iterations are the workhorse for entropy-regularized optimal transport (EOT) and for differentiable permutation learning (Gumbel-Sinkhorn, blockwise channel permutation, etc.). Standard GPU implementations suffer from a fundamental IO bottleneck: tensorized solvers materialize the dense $n \times m$ cost/transport matrix in HBM, incurring $O(nm)$ memory and quadratic HBM traffic per iteration. Online backends (KeOps) avoid dense storage but use generic map-reduce kernels with limited fusion, achieving low SM utilization.

FlashSinkhorn observes that for squared Euclidean cost, each stabilized log-domain Sinkhorn half-step is algebraically identical to a **row-wise LogSumExp (LSE) reduction of a biased dot-product score matrix** — the same normalization that underlies scaled dot-product attention. This structural equivalence enables direct transfer of the FlashAttention IO-aware tiling strategy: fused Triton kernels stream tiles of source/target points through on-chip SRAM, compute cost tiles on-the-fly, maintain online max/sumexp statistics per row, and write only the updated dual potential vectors back to HBM — never materializing the $n \times m$ score matrix.

The key insight is the **attention-form Sinkhorn update**: defining $Q = \sqrt{2}X$, $K = \sqrt{2}Y$, the row-logits become $S_X(\hat{\mathbf{g}}) = (QK^\top + \mathbf{1}_n(\hat{\mathbf{g}} + \boldsymbol{\delta}))/\varepsilon$, and the Sinkhorn update $\hat{\mathbf{f}} \leftarrow -\varepsilon \, \text{LSE}_\text{row}(S_X(\hat{\mathbf{g}}))$ is computed by the same online LSE algorithm as FlashAttention. Additionally, the transport matrix application $P^*V$ admits a streaming **matmul-softmax-matmul** fusion identical to attention output computation, and the Hessian-vector product (HVP) for second-order optimization decomposes into streaming transport-vector and transport-matrix products via a Schur complement formulation.

This directly accelerates any system using Sinkhorn iterations for permutation learning, including blockwise Sinkhorn channel permutation (PermLLM), learnable permutation cost bipartite matching, Sinkhorn attention, and Gumbel-Sinkhorn networks.

## Mathematical Form

**EOT Problem:**

$$
\text{OT}_\varepsilon(\boldsymbol{\mu}, \boldsymbol{\nu}) = \min_{P \in \Pi(\mathbf{a}, \mathbf{b})} \langle C, P \rangle + \varepsilon \, \text{KL}(P \| \mathbf{a} \otimes \mathbf{b})
$$

where $C_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|_2^2$ is the squared Euclidean cost, $\Pi(\mathbf{a}, \mathbf{b})$ is the transport polytope, and $\varepsilon > 0$ is the regularization strength.

**Stabilized Log-Domain Sinkhorn:**

Let $\mathbf{f} \in \mathbb{R}^n, \mathbf{g} \in \mathbb{R}^m$ be dual potentials. The alternating updates are:

$$
f_i \leftarrow -\varepsilon \, \text{LSE}_j \left[ \frac{g_j - C_{ij}}{\varepsilon} + \log b_j \right]
$$

$$
g_j \leftarrow -\varepsilon \, \text{LSE}_i \left[ \frac{f_i - C_{ij}}{\varepsilon} + \log a_i \right]
$$

**Attention-Form Rewriting (Proposition 1):**

Define shifted potentials $\hat{\mathbf{f}} := \mathbf{f} - \boldsymbol{\alpha}$, $\hat{\mathbf{g}} := \mathbf{g} - \boldsymbol{\beta}$ where $\alpha_i = \|\mathbf{x}_i\|_2^2$, $\beta_j = \|\mathbf{y}_j\|_2^2$. Set $Q := \sqrt{2}X$, $K := \sqrt{2}Y$. Define bias vectors $\boldsymbol{\delta} := \varepsilon \log \mathbf{b}$, $\boldsymbol{\gamma} := \varepsilon \log \mathbf{a}$.

Row-logits and column-logits:

$$
S_X(\hat{\mathbf{g}}) := \left(QK^\top + \mathbf{1}_n(\hat{\mathbf{g}} + \boldsymbol{\delta})\right) / \varepsilon
$$

$$
S_Y(\hat{\mathbf{f}}) := \left(KQ^\top + \mathbf{1}_m(\hat{\mathbf{f}} + \boldsymbol{\gamma})\right) / \varepsilon
$$

Then the stabilized Sinkhorn updates are equivalently:

$$
\hat{\mathbf{f}} \leftarrow -\varepsilon \, \text{LSE}_\text{row}(S_X(\hat{\mathbf{g}}))
$$

$$
\hat{\mathbf{g}} \leftarrow -\varepsilon \, \text{LSE}_\text{row}(S_Y(\hat{\mathbf{f}}))
$$

**Streaming Algorithm (Algorithm 1):**

For each row block $I$ of size $B_N$:
1. Load $X_I$ to on-chip SRAM
2. Initialize running max $\mathbf{m}_I \leftarrow -\infty$, sumexp $\mathbf{s}_I \leftarrow 0$
3. For each column block $J$ of size $B_M$:
   - Load $Y_J$, $\hat{\mathbf{g}}_J$, $\boldsymbol{\delta}_J$ to SRAM
   - Compute score tile: $S \leftarrow (2 X_I Y_J^\top + \hat{\mathbf{g}}_J + \boldsymbol{\delta}_J) / \varepsilon$
   - Update online LSE: $\mathbf{m}_\text{new} \leftarrow \max(\mathbf{m}_I, \text{rowmax}(S))$
   - $\mathbf{s}_I \leftarrow e^{\mathbf{m}_I - \mathbf{m}_\text{new}} \odot \mathbf{s}_I + \text{rowsum}(e^{S - \mathbf{m}_\text{new}})$
4. Write $\hat{\mathbf{f}}_I \leftarrow -\varepsilon(\mathbf{m}_I + \log \mathbf{s}_I)$ to HBM

**Transport Application as Attention (Proposition 3):**

The transport matrix $P_{ij}(\hat{\mathbf{f}}, \hat{\mathbf{g}}) = a_i b_j \exp\left(\frac{\hat{f}_i + \hat{g}_j + (QK^\top)_{ij}}{\varepsilon}\right)$ enables:

$$
PV = \text{diag}(\mathbf{r}) \, \text{Softmax}(S_X(\hat{\mathbf{g}})) \, V
$$

$$
P^\top U = \text{diag}(\mathbf{c}) \, \text{Softmax}(S_Y(\hat{\mathbf{f}})) \, U
$$

where $\mathbf{r}, \mathbf{c}$ are marginal correction vectors. This is computed by the same streaming matmul-softmax-matmul kernel as FlashAttention.

**Streaming HVP (Theorem 5):**

The Hessian-vector product $G = \mathcal{T} A$ decomposes via the Schur complement:

$$
S := \text{diag}(\mathbf{b}) - (P^*)^\top \text{diag}(\mathbf{a})^{-1} P^*
$$

Solved by CG using only streaming transport applications — memory $O((n+m)d)$, flops $O((K_{CG}+1) \, nmd)$.

## Complexity

| Operation | Tensorized | KeOps (Online) | FlashSinkhorn |
|-----------|-----------|----------------|---------------|
| HBM per iteration | $\Theta(nd + md + nm)$ | $\Theta(nd + md)$ | $\Theta\left(nd + md + \frac{nmd^2}{M}\right)$ |
| Memory | $O(nm)$ | $O((n+m)d)$ | $O((n+m)d)$ |
| $PV$ application | $O(nm + nmp)$ | $O(nmd + nmp)$ | $O\left((n+m)(d+p) + \frac{nm(d+p)^2}{M}\right)$ |
| HVP memory | $O(n^2d^2 + nm)$ | $O(nd)$ | $O((n+m)d)$ |

where $M$ is SRAM size, $d$ is point dimension, $p$ is the output dimension of $V$.

**Key result:** For SRAM $M \geq \min\{n,m\} \cdot d$, FlashSinkhorn achieves $\Theta(nd + md)$ HBM per iteration — matching the linear-memory online backends but with fused kernels that eliminate kernel-launch overhead and maximize SM utilization.

**Speedups (A100-80GB):**

| Setting ($n$, $d$) | Fwd vs KeOps | Fwd vs Tensorized | End-to-End vs KeOps |
|---------------------|-------------|-------------------|---------------------|
| 10k, 128 | 9.4$\times$ | 3.2$\times$ | 10.3$\times$ |
| 10k, 512 | 31.7$\times$ | 1.7$\times$ | 161$\times$ |
| 40k, 128 | 12.5$\times$ | OOM | 13.5$\times$ |
| 40k, 512 | OOT | OOM | OOT |

## Applicability

- **Blockwise Sinkhorn channel permutation:** Each block's Sinkhorn normalization (converting learnable $\mathbf{W}_P^i$ to doubly stochastic matrix) can use FlashSinkhorn's streaming kernel, particularly beneficial when block size $B$ is large ($B \geq 128$) or when many blocks must be processed in parallel across layers.
- **Differentiable permutation learning:** Any system using Gumbel-Sinkhorn (ICLR 2018) or entropy-regularized matching benefits from the IO-aware iteration, especially for large-dimensional permutations ($N \geq 1024$) where tensorized Sinkhorn is memory-bound.
- **Sinkhorn attention:** Sinkformers and Sparse Sinkhorn Attention replace row-softmax with doubly-stochastic normalization. FlashSinkhorn's attention-form update directly applies, enabling IO-aware doubly-stochastic attention with the same tiling strategy as FlashAttention.
- **OT-based training losses:** Wasserstein distance, Sinkhorn divergence, and OTDD dataset comparison losses all use Sinkhorn iterations. FlashSinkhorn makes these practical at scales previously infeasible ($n \geq 50$k).
- **Second-order optimization:** The streaming HVP enables Newton-CG for OT objectives at scale, useful for saddle-point escape in shuffled regression and calibration problems.

## Limitations

- Currently restricted to **squared Euclidean cost** $C_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|_2^2$; the dot-product decomposition that enables the attention-form rewriting does not extend to general costs (e.g., geodesic distance, learned costs)
- Requires Triton for fused kernel implementation — not portable to non-NVIDIA hardware without rewriting
- The symmetric update schedule (Eq. 4–5) uses a single fused kernel but requires both $\hat{\mathbf{f}}$ and $\hat{\mathbf{g}}$ simultaneously, increasing register pressure; the alternating schedule is preferred at large $n$ and high $d$
- For very small problems ($n < 1000$), kernel launch overhead dominates and tensorized Sinkhorn may be faster
- Does not address the convergence properties of Sinkhorn itself — only makes each iteration faster

## Implementation Notes

```python
import triton
import triton.language as tl
import torch

# High-level pseudocode for FlashSinkhorn f-hat update
# (actual Triton kernel uses tiled GEMM + online LSE)

def flash_sinkhorn_f_update(X, Y, g_hat, b, eps, B_N=64, B_M=64):
    """
    Streaming Sinkhorn f-hat update via IO-aware tiling.

    Args:
        X: (n, d) source points
        Y: (m, d) target points
        g_hat: (m,) shifted dual potential
        b: (m,) target weights
        eps: regularization strength
        B_N, B_M: tile sizes for row/column blocks

    Returns:
        f_hat: (n,) updated shifted dual potential
    """
    n, d = X.shape
    m = Y.shape[0]

    Q = X * (2.0 ** 0.5)  # sqrt(2) * X
    K = Y * (2.0 ** 0.5)  # sqrt(2) * Y
    delta = eps * torch.log(b)  # bias from target weights

    f_hat = torch.empty(n, device=X.device)

    for i_start in range(0, n, B_N):
        i_end = min(i_start + B_N, n)
        Q_I = Q[i_start:i_end]  # Load to SRAM

        m_I = torch.full((i_end - i_start,), float('-inf'), device=X.device)
        s_I = torch.zeros(i_end - i_start, device=X.device)

        for j_start in range(0, m, B_M):
            j_end = min(j_start + B_M, m)
            K_J = K[j_start:j_end]  # Stream from HBM
            g_J = g_hat[j_start:j_end]
            d_J = delta[j_start:j_end]

            # Score tile = (2 * X_I @ Y_J^T + g_hat_J + delta_J) / eps
            S = (Q_I @ K_J.T + g_J.unsqueeze(0) + d_J.unsqueeze(0)) / eps

            # Online LSE update (numerically stable)
            m_tile = S.max(dim=-1).values
            m_new = torch.maximum(m_I, m_tile)
            s_I = torch.exp(m_I - m_new) * s_I + (torch.exp(S - m_new.unsqueeze(-1))).sum(dim=-1)
            m_I = m_new

        # Write f_hat_I back to HBM
        f_hat[i_start:i_end] = -eps * (m_I + torch.log(s_I))

    return f_hat

# In practice, this is a single fused Triton kernel that
# never materializes the n x m score matrix:
# - Q_I stays in SRAM (outer loop)
# - K_J streams from HBM (inner loop)
# - Online max/sumexp maintained in registers
# - Only f_hat written back to HBM

# For Sinkhorn permutation learning, the iteration is:
# for t in range(T):
#     f_hat = flash_sinkhorn_f_update(X, Y, g_hat, b, eps)
#     g_hat = flash_sinkhorn_g_update(X, Y, f_hat, a, eps)
# P_soft = recover_transport(f_hat, g_hat, X, Y, a, b, eps)
```

## References

- Ye, F. X.-F., Li, X., Yu, A., Chang, M.-C., Chu, L. & Wertheimer, D. (2026). FlashSinkhorn: IO-Aware Entropic Optimal Transport. arXiv:2602.03067.
- Dao, T., Fu, D., Ermon, S., Rudra, A. & Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.
- Mena, G. et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018.
- Milakov, M. & Gimelshein, N. (2018). Online Normalizer Calculation for Softmax. arXiv:1805.02867.
- Feydy, J. (2020). Geometric Data Analysis, Beyond Convolutions. Applied Mathematics.
- Open-source implementation: https://github.com/ot-triton-lab/flash-sinkhorn

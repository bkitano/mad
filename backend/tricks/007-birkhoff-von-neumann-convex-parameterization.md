# 007: Birkhoff-von Neumann Convex Parameterization

**Category**: approximation
**Gain type**: efficiency
**Source**: Birkhoff (1946), von Neumann (1953), Yang & Gao (2026) — mHC-lite
**Paper**: [papers/mhc-lite-birkhoff-parameterization.pdf]
**Documented**: 2026-02-15

## Description

The Sinkhorn-Knopp (SK) algorithm is the standard method for projecting learnable matrices onto the Birkhoff polytope $\mathcal{B}_n$ (the set of $n \times n$ doubly stochastic matrices) in neural networks. However, finite SK iterations have two fundamental issues: (i) they do **not** guarantee exact doubly stochasticity — column/row sums can deviate from 1 by up to 220% after composing across 24 layers, and (ii) efficient SK implementation requires highly specialized fused CUDA kernels (the mHC paper reports needing custom kernels to amortize 20 repeated kernel launches).

The **Birkhoff-von Neumann convex parameterization** (mHC-lite) replaces iterative Sinkhorn projection with a **direct reparameterization**: any doubly stochastic matrix $X \in \mathcal{B}_n$ is expressed as a convex combination of the $n!$ permutation matrices $\{P_k\}_{k=1}^{n!}$, which are the vertices of the Birkhoff polytope. The weights are produced by a softmax over unconstrained parameters, so the output is **exactly doubly stochastic by construction** — no iterative normalization needed.

The key insight is that this shifts the computational paradigm from **"projection"** (iteratively normalizing toward the constraint set) to **"reparameterization"** (constructing outputs that satisfy the constraint by design). This eliminates the approximation gap entirely and enables implementation with only standard matrix operations (linear layers + softmax), removing the need for specialized CUDA kernels.

For the practical case of $n = 4$ residual streams (as in Hyper-Connections), $n! = 24$ permutation matrices can be enumerated and stored as a constant $n^2 \times n!$ matrix. The doubly stochastic matrix is then computed via a single matrix-vector product followed by reshaping.

## Mathematical Form

**Birkhoff-von Neumann Theorem:**

For any doubly stochastic matrix $X \in \mathcal{B}_n$, there exists a weight vector $\mathbf{a} = (a_1, \ldots, a_{n!}) \in \mathbb{R}^{n!}$ with $a_k \geq 0$ for all $k$ and $\|\mathbf{a}\|_1 = 1$ such that:

$$
X = \sum_{k=1}^{n!} a_k P_k
$$

where $\{P_k\}_{k=1}^{n!}$ is the set of all $n \times n$ permutation matrices.

**mHC-lite Parameterization:**

Given input features $\hat{x}_l' = \text{RMSNorm}(\hat{x}_l)$ at layer $l$, the residual matrix is constructed as:

$$
\mathbf{a}_l = \text{softmax}\left(\alpha_l^{\text{res}} \hat{x}_l' W_l^{\text{res}} + b_l^{\text{res}}\right) \in \mathbb{R}^{n!}
$$

$$
H_l^{\text{res}} = \sum_{k=1}^{n!} a_{l,k} P_k
$$

where $W_l^{\text{res}} \in \mathbb{R}^{nC \times n!}$ and $b_l^{\text{res}} \in \mathbb{R}^{1 \times n!}$ are learnable parameters and $\alpha_l^{\text{res}}$ is a learnable scalar.

**Implementation as matrix multiplication:**

In practice, the $n!$ permutation matrices are concatenated into a constant matrix $\mathbf{P} \in \mathbb{R}^{n^2 \times n!}$ where column $k$ is the vectorized permutation matrix $\text{vec}(P_k)$. Then:

$$
\text{vec}(H_l^{\text{res}}) = \mathbf{P} \cdot \mathbf{a}_l
$$

$$
H_l^{\text{res}} = \text{mat}(\mathbf{P} \cdot \mathbf{a}_l) \in \mathbb{R}^{n \times n}
$$

This is a single matrix-vector product followed by reshaping from $\mathbb{R}^{n^2}$ to $\mathbb{R}^{n \times n}$.

**Full Hyper-Connection update:**

$$
x_{l+1} = H_l^{\text{res}} x_l + H_l^{\text{post}} f(H_l^{\text{pre}} x_l; \mathcal{W}_l)
$$

where $H_l^{\text{pre}} = \text{sigmoid}(\alpha_l^{\text{pre}} \hat{x}_l' W_l^{\text{pre}} + b_l^{\text{pre}})$ and $H_l^{\text{post}} = 2 \cdot \text{sigmoid}(\alpha_l^{\text{post}} \hat{x}_l' W_l^{\text{post}} + b_l^{\text{post}})$.

**Stability guarantee:**

Since each $H_l^{\text{res}}$ is exactly doubly stochastic, its spectral norm satisfies $\|H_l^{\text{res}}\|_2 \leq 1$, and the product across layers is also doubly stochastic:

$$
\prod_{l} H_l^{\text{res}} \in \mathcal{B}_n
$$

This prevents gradient explosion through the residual path at any depth.

## Complexity

| Operation | Sinkhorn (mHC, $L$ iters) | Convex Param (mHC-lite) |
|-----------|---------------------------|------------------------|
| Forward pass | $O(L \cdot n^2)$ per token | $O(n! \cdot n^2)$ per token |
| Doubly stochastic? | Approximate | Exact by construction |
| Custom CUDA kernel? | Required for efficiency | Not needed (standard ops) |
| Gradient computation | Through $L$ unrolled iterations | Through softmax + matmul |
| Accumulation error | $O(\epsilon^L)$ per layer, compounds | Zero |

**Concrete numbers ($n = 4$, i.e., 4 residual streams):**

- Sinkhorn (mHC): 20 SK iterations per layer, requires fused CUDA kernel. DeepSeek reports 6.7% overhead relative to standard residual connections with their optimized kernel.
- Convex param (mHC-lite): Softmax over $n! = 24$ weights + matrix-vector product with $16 \times 24$ constant matrix. **Higher throughput than mHC even without any system optimization.**

**Throughput (M model, 0.12B params, 8× A100):**
- Standard residual: ~750k tokens/sec
- mHC (PyTorch, no custom kernel): ~230k tokens/sec
- mHC-lite (PyTorch, no custom kernel): ~500k tokens/sec
- HC (unconstrained): ~600k tokens/sec

**Memory:** $O(n! \cdot n^2)$ for the constant permutation matrix (384 floats for $n=4$) plus $O(n!)$ for the softmax weights per token. Negligible.

## Applicability

- **Hyper-Connections / multi-stream residuals:** Primary application. Drop-in replacement for mHC (DeepSeek-V3's manifold-constrained residual connections) that eliminates approximation error and specialized kernels. Validated at scales from 45M to 0.36B parameters with matching or better loss.
- **Blockwise Sinkhorn channel permutation (PermLLM):** The same reparameterization could replace the Sinkhorn normalization in each block's learnable permutation matrix. Instead of $L$ Sinkhorn iterations per block, express the block's doubly stochastic matrix as a convex combination of $B!$ permutation matrices. However, for block size $B = 64$, $B!$ is astronomically large — this approach is only practical for very small $n$ (e.g., $n \leq 6$).
- **Differentiable sorting/matching:** Any task requiring a differentiable doubly stochastic matrix with small $n$ benefits from exact parameterization.
- **Training stability:** The exact doubly stochasticity guarantee prevents the accumulation of normalization errors across depth, which is critical for very deep networks (1000+ layers in reinforcement learning).
- **Framework portability:** No specialized kernels means the technique works in any framework (PyTorch, JAX, TensorFlow) with standard matrix operations.

## Limitations

- **Factorial scaling:** The number of permutation matrices $n!$ grows superexponentially — $4! = 24$, $5! = 120$, $6! = 720$, $7! = 5040$. This limits applicability to small $n$ (the paper uses $n = 4$). For the blockwise Sinkhorn setting with $B = 64$, this is completely infeasible.
- **Subsampling mitigation:** The paper notes that for larger $n$, one can sample a subset of $K \ll n!$ permutation matrices, restricting to a subset of the Birkhoff polytope while maintaining exact doubly stochasticity. However, expressiveness is reduced.
- **Not a general replacement for Sinkhorn:** For large-$n$ permutation learning (e.g., the $64 \times 64$ blocks in PermLLM), iterative Sinkhorn remains the only practical approach.
- **Input-dependent weights:** Each token dynamically computes its own softmax weights $\mathbf{a}_l$, adding a linear layer with output dimension $n!$. For small $n$ this is negligible but could become significant for moderate $n$.
- **Requires enumeration:** All $n!$ permutation matrices must be enumerated and stored. This is a one-time cost but limits the approach to small $n$.

## Implementation Notes

```python
import torch
import torch.nn.functional as F
from itertools import permutations

def build_permutation_basis(n):
    """
    Enumerate all n! permutation matrices and stack into a
    constant matrix P of shape (n^2, n!).
    """
    perms = list(permutations(range(n)))
    n_fact = len(perms)  # n!
    P = torch.zeros(n * n, n_fact)
    for k, perm in enumerate(perms):
        for i, j in enumerate(perm):
            P[i * n + j, k] = 1.0
    return P  # (n^2, n!)

class MHCLiteResidual(torch.nn.Module):
    """
    mHC-lite: Birkhoff-von Neumann parameterization of
    doubly stochastic residual matrices.
    """
    def __init__(self, n, C):
        super().__init__()
        self.n = n
        self.C = C
        n_fact = 1
        for i in range(1, n + 1):
            n_fact *= i

        # Constant: all n! permutation matrices, shape (n^2, n!)
        self.register_buffer('P_basis', build_permutation_basis(n))

        # Learnable: linear layer to produce softmax weights
        self.alpha = torch.nn.Parameter(torch.tensor(0.01))
        self.W_res = torch.nn.Linear(n * C, n_fact, bias=True)

        # Initialize bias so identity permutation dominates
        with torch.no_grad():
            self.W_res.bias.fill_(-8.0)
            # Find identity permutation index and set high
            identity_idx = list(permutations(range(n))).index(
                tuple(range(n))
            )
            self.W_res.bias[identity_idx] = 0.0

    def forward(self, x_flat):
        """
        Args:
            x_flat: (batch, n*C) flattened input features
        Returns:
            H_res: (batch, n, n) doubly stochastic residual matrix
        """
        # Compute softmax weights over permutations
        a = F.softmax(self.alpha * self.W_res(x_flat), dim=-1)  # (batch, n!)

        # Convex combination: (n^2, n!) @ (n!, batch) -> (n^2, batch)
        H_vec = self.P_basis @ a.T  # (n^2, batch)
        H_res = H_vec.T.reshape(-1, self.n, self.n)  # (batch, n, n)

        return H_res  # Exactly doubly stochastic by construction
```

## References

- Birkhoff, G. (1946). Three observations on linear algebra. Univ. Nac. Tucuman Revista A., 5:147-151.
- von Neumann, J. (1953). A certain zero-sum two-person game equivalent to an optimal assignment problem. In Contributions to the Theory of Games, Vol. 2.
- Yang, Y. & Gao, J. (2026). mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations. arXiv:2601.05732.
- Xie, Z. et al. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Zhu, Z. et al. (2024). Hyper-Connections. arXiv.
- Knight, P. A. (2008). The Sinkhorn-Knopp Algorithm: Convergence and Applications. SIAM J. Matrix Anal. Appl.
- Chakrabarty, D. & Khanna, S. (2021). Better and Simpler Error Analysis of the Sinkhorn-Knopp Algorithm for Matrix Scaling. Math. Program.

# 250: BLAST Block-Level Adaptive Structured Matrix

**Category**: decomposition
**Gain type**: efficiency
**Source**: Lee et al., "BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference" (NeurIPS 2024)
**Paper**: [papers/blast-block-adaptive-structured-matrices.pdf]
**Documented**: 2026-02-16

## Description

BLAST (Block-Level Adaptive STructured) is a flexible block-structured matrix factorization that replaces dense weight matrices $A \in \mathbb{R}^{n \times n}$ in neural networks. The key idea: partition $A$ into $b \times b$ blocks, then factor each block $A_{i,j} = U_i \operatorname{diag}(s_{i,j}) V_j^T$. The left factors $U_i$ are **shared across rows** and right factors $V_j$ are **shared across columns**, while only the diagonal coupling vectors $s_{i,j}$ are block-specific.

This structure is a strict generalization of low-rank, block-diagonal, Monarch, and block low-rank matrices — it can learn which structure best fits each weight matrix. Critically, the factorization maps to **batched matrix multiplications** (via `torch.bmm`) that leverage tensor cores effectively, and the diagonal coupling avoids the need for QR decomposition or other sequential operations.

## Mathematical Form

**Core Operation:**

$$
A_{i,j} = U_i \operatorname{diag}(s_{i,j}) V_j^T
$$

where $A$ is partitioned into $b \times b$ blocks of size $p \times p$ (with $n = bp$).

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — the original dense weight matrix
- $U_i \in \mathbb{R}^{p \times r}$ — left factor shared across the $i$-th block row ($i = 1, \ldots, b$)
- $V_j \in \mathbb{R}^{p \times r}$ — right factor shared across the $j$-th block column ($j = 1, \ldots, b$)
- $s_{i,j} \in \mathbb{R}^r$ — diagonal coupling vector specific to block $(i,j)$
- $r$ — rank parameter controlling compression ratio

**Special Cases:**

- **Low-rank** ($s_{i,j} = \mathbf{1}_r$ for all $i,j$): reduces to $A = UV^T$ where $U = [U_1; \ldots; U_b]$, $V = [V_1; \ldots; V_b]$
- **Block-diagonal** ($s_{i,j} = \mathbf{0}$ for $i \neq j$, arbitrary for $i = j$): independent diagonal blocks
- **Monarch** (all blocks share the same rank $r$, specific diagonal pattern): block low-rank with equal ranks

**Matrix-Vector Product (Algorithm 1):**

$$
y_i = U_i \left( \sum_{j=1}^{b} s_{i,j} \odot (V_j^T x_j) \right), \quad i = 1, \ldots, b
$$

Step 1: Compute $z_j = V_j^T x_j$ for all $j$ (batched matmul, shared across all output blocks).
Step 2: For each output block $i$, accumulate $\sum_j s_{i,j} \odot z_j$ (elementwise multiply + sum).
Step 3: Apply $U_i$ to the accumulated result (batched matmul).

**Parameter Count:**

$$
|\text{params}| = 2nr + rb^2 = 2nr + rb^2
$$

vs $n^2$ for dense. Since typically $r \ll p$ and $b$ is small (3–16), this gives significant compression.

**BLAST Factorization (Compression of Pre-trained Weights):**

Minimize the Frobenius error via alternating preconditioned gradient descent:

$$
\ell(U_*, V_*, s_{*,*}) = \sum_{i=1}^{b} \sum_{j=1}^{b} \frac{1}{2} \|A_{i,j} - U_i \operatorname{diag}(s_{i,j}) V_j^T\|_F^2
$$

with preconditioners:

$$
P_{U_i} = (\tilde{V}_i^{(k)T} \tilde{V}_i^{(k)} + \delta I)^{-1}, \quad P_{V_j} = (\bar{U}_j^{(k)T} \bar{U}_j^{(k)} + \delta I)^{-1}
$$

## Complexity

| Operation | Dense | Low-Rank ($r$) | BLAST ($b$ blocks, rank $r$) |
|-----------|-------|----------------|------------------------------|
| Matvec FLOPs | $n^2$ | $2nr$ | $(2n + b^2)r$ |
| Parameters | $n^2$ | $2nr$ | $2nr + rb^2$ |
| Matmul ($n \times n$ by $n \times m$) | $n^2 m$ | $2nrm$ | $(2nm/b + b^2 m/b)r$ |

**Memory:** $O(nr + rb^2)$ vs $O(n^2)$ for dense. For typical settings ($b = 3$–$16$, $r = 8$–$64$), this is a 2×–5× reduction.

**Practical Compression Ratios:**
- ViT-Base on ImageNet: 79.3% accuracy at 27.8% relative FLOPs (vs 78.7% dense)
- GPT-2 on WikiText-103: Best perplexity-FLOPs tradeoff among all structured matrices tested
- Llama-7B: 50% compression with lowest performance degradation among all baselines
- DiT diffusion: 50% compression, FID 10.45 vs 48.07 for low-rank (vs 9.62 original)

## Applicability

- **Linear layer compression** in transformers (attention projections, FFN layers)
- **Training from scratch** with structured weights (replace dense layers with BLAST factors)
- **Post-training compression** of pre-trained foundation models (ViT, GPT-2, Llama-7B, DiT)
- **Any architecture** where dense weight matrices are the computational bottleneck
- Maps naturally to **batched GEMM on tensor cores** via `torch.bmm`

## Limitations

- **Block size selection**: The number of blocks $b$ and rank $r$ are hyperparameters. Using the same $r$ across all layers is suboptimal (unlike Gaudi-GBLR which learns per-layer budgets), though simpler to tune.
- **Not optimal for all structures**: If the true weight structure is neither low-rank nor block-diagonal (e.g., purely sparse), BLAST may not be the best fit.
- **Compression algorithm cost**: The preconditioned gradient descent factorization has complexity $O(nr^2 + r^3)$ per iteration, which can be expensive for large $r$, though $r \ll n$ in practice.
- **Rectangular matrices**: The formulation assumes $b$ divides both dimensions; non-divisible dimensions require padding.
- **Re-training recommended**: For best compression results, BLAST factorization should be followed by re-training on data, adding to the total compute budget.

## Implementation Notes

```python
import torch

class BLASTLayer(torch.nn.Module):
    """BLAST structured linear layer.

    Replaces a dense n×n weight matrix with BLAST factorization.
    Each block A_{i,j} = U_i @ diag(s_{i,j}) @ V_j^T
    """
    def __init__(self, n, b, r):
        super().__init__()
        self.n = n
        self.b = b
        self.r = r
        self.p = n // b  # block size

        # Shared left factors: b matrices of shape [p, r]
        self.U = torch.nn.Parameter(torch.randn(b, self.p, r))
        # Shared right factors: b matrices of shape [p, r]
        self.V = torch.nn.Parameter(torch.randn(b, self.p, r))
        # Block-specific diagonal coupling: b×b vectors of length r
        self.S = torch.nn.Parameter(torch.randn(b, b, r))

    def forward(self, x):
        # x: [..., n]
        *batch_dims, n = x.shape
        x_blocks = x.reshape(*batch_dims, self.b, self.p)  # [..., b, p]

        # Step 1: z_j = V_j^T @ x_j for all j (batched matmul)
        # x_blocks: [..., b, p] -> [..., b, p, 1] or use einsum
        z = torch.einsum('...bp, bpr -> ...br', x_blocks, self.V)  # [..., b, r]

        # Step 2: accumulate sum_j s_{i,j} * z_j for each i
        # S: [b_out, b_in, r], z: [..., b_in, r]
        accum = torch.einsum('ijr, ...jr -> ...ir', self.S, z)  # [..., b, r]

        # Step 3: y_i = U_i @ accum_i (batched matmul)
        y = torch.einsum('...br, bpr -> ...bp', accum, self.U)  # [..., b, p]

        return y.reshape(*batch_dims, n)

# Key GPU-friendly properties:
# 1. Steps 1 and 3 are batched GEMMs -> tensor core friendly
# 2. Step 2 is einsum (reducible to batched GEMM or fused kernel)
# 3. No sequential iterations, no gather/scatter
# 4. All operations have regular memory access patterns
```

## References

- Lee, C., Kwon, S. M., Qu, Q., & Kim, H.-S. (2024). BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference. NeurIPS 2024. arXiv:2410.21262.
- GitHub: https://github.com/changwoolee/BLAST
- Dao, T., et al. (2022). Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML 2022.
- Lee, C. & Kim, H.-S. (2024). Gaudi-GBLR: Generalized Block Low-Rank. ICLR 2024.

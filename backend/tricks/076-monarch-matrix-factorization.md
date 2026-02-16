# 076: Monarch Matrix Factorization

**Category**: decomposition
**Gain type**: efficiency
**Source**: Dao et al. (2022), ICML 2022
**Paper**: [papers/group-and-shuffle-matrices.pdf] (GS generalization of Monarch)
**Documented**: 2026-02-11

## Description

Monarch matrices are structured matrices parameterized as a product of two block-diagonal matrices interleaved with a fixed permutation: $M = P^\top L P R$, where $L$ and $R$ are block-diagonal and $P$ is a specific reshape permutation. This factorization is **hardware-efficient** — it maps directly to batch matrix multiply (BMM) on GPUs — while being **expressive enough** to represent many important fast transforms (DFT, DCT, Hadamard, and their inverses).

The key insight connecting Householder reflections and permutations is that Monarch matrices **separate the roles**: permutations handle coordinate mixing (shuffling entries between blocks), while block-diagonal matrices handle local orthogonal/linear transformations within blocks. A single Monarch factor has the structure of a two-layer "group-and-shuffle" network. By stacking 2–4 Monarch factors, one can approximate any dense matrix, achieving a practical $O(n^{3/2})$ parameterization with $O(n^{3/2})$ computation — sub-quadratic in the matrix dimension.

## Mathematical Form

**Core Operation:**

$$
M = P_b^\top \, L \, P_b \, R
$$

where:
- $R = \text{diag}(R_1, \ldots, R_{\sqrt{n}}) \in \mathbb{R}^{n \times n}$, each $R_i \in \mathbb{R}^{\sqrt{n} \times \sqrt{n}}$
- $L = \text{diag}(L_1, \ldots, L_{\sqrt{n}}) \in \mathbb{R}^{n \times n}$, each $L_i \in \mathbb{R}^{\sqrt{n} \times \sqrt{n}}$
- $P_b$ is the **block-reshape permutation** that maps index $(i, j)$ in a $\sqrt{n} \times \sqrt{n}$ grid to $(j, i)$

**Key Definitions:**

- $n$ — matrix dimension (assuming $n = p \cdot q$ for block sizes $p, q$)
- $P_b$ — the specific permutation that transposes a $p \times q$ matrix when viewed as a vector (stride permutation)
- $\sqrt{n}$ — typical block count and block size (for square factorization)

**Equivalent tensor view:**

Viewing $x \in \mathbb{R}^n$ as $X \in \mathbb{R}^{p \times q}$:

$$
Y = L \cdot (R \cdot X^\top)^\top
$$

Step 1: $R$ acts on $q$ groups of $p$ elements (rows of $X^\top$) — batch matrix multiply
Step 2: Transpose (the permutation $P_b$) — reshuffles entries
Step 3: $L$ acts on $p$ groups of $q$ elements — batch matrix multiply

**Stacked Monarch (Monarch$^k$):**

For $k$ Monarch factors:

$$
M^{(k)} = \prod_{i=1}^{k} \left( P_b^\top L^{(i)} P_b R^{(i)} \right)
$$

**Expressiveness hierarchy:**
- Monarch$^1$: represents all butterfly matrices with depth 2
- Monarch$^2$: represents DFT, DCT, Hadamard, and their inverses
- Monarch$^4$: can approximate arbitrary dense matrices

**Optimal Monarch approximation:**

Given a dense matrix $A$, the best rank-$\sqrt{n}$ Monarch approximation has an **analytical closed-form solution**:

$$
\min_{M \in \mathcal{M}} \|A - M\|_F^2
$$

is solved by reshaping $A$ into blocks and computing truncated SVDs — despite the non-convex parameterization.

## Complexity

| Operation | Dense Matrix | Monarch ($k$ factors) |
|-----------|-------------|----------------------|
| Parameters | $O(n^2)$ | $O(kn\sqrt{n})$ |
| Mat-vec | $O(n^2)$ | $O(kn\sqrt{n})$ via BMM |
| Training (GPU) | $O(n^2)$, memory-bound | $O(kn\sqrt{n})$, compute-efficient (BMM) |
| Expressiveness | Full | DFT etc. at $k=2$; dense approx. at $k=4$ |

**Concrete speedups (from paper):**

| Matrix size | Dense | Monarch | Speedup |
|-------------|-------|---------|---------|
| $n = 1024$ | 1.0x | ~2x | 2x faster |
| $n = 4096$ | 1.0x | ~2x | 2x faster |

**Hardware efficiency:** Monarch maps to `torch.bmm`, achieving near-peak GPU utilization, unlike butterfly matrices which require custom kernels.

**Memory:** $O(n\sqrt{n})$ per factor vs $O(n^2)$ for dense

## Applicability

- **Replacing dense linear layers:** Drop-in replacement for $n \times n$ weight matrices in transformers and other architectures, with ~2x speedup and comparable accuracy
- **Efficient training:** Used to speed up ViT and GPT-2 training on ImageNet and Wikitext-103 by 2x
- **PDE solving:** 40% error reduction on PDE tasks with structured Monarch layers
- **MRI reconstruction:** Similar improvements in structured signal processing tasks
- **Sequence model state transitions:** The permutation-interleaved-with-blocks structure is directly applicable to parameterizing state-transition matrices in linear RNNs
- **Foundation for GS and BOFT:** Group-and-Shuffle and Butterfly Orthogonal Fine-Tuning both generalize Monarch factorization

## Limitations

- The square-root factorization $n = p \times q$ requires $n$ to factor nicely; non-square dimensions need padding
- Fixed permutation $P_b$ limits expressivity compared to learned permutations (active research: Mohamed et al., 2025)
- Cannot represent all matrices with $k = 1$ factor; $k = 2$ covers common fast transforms but not all matrices
- The analytical approximation result assumes the specific Monarch structure; approximation quality degrades for matrices far from this class
- Block sizes must be chosen to match hardware (GPU warp size, memory alignment)

## Implementation Notes

```python
import torch

def monarch_forward(L_blocks, R_blocks, x, p, q):
    """
    Apply single Monarch factor M = P_b^T @ L @ P_b @ R to x.

    Args:
        L_blocks: (p, q, q) - p blocks of size q x q
        R_blocks: (q, p, p) - q blocks of size p x p
        x: (n,) where n = p * q
        p, q: block dimensions

    Returns:
        (n,) - M @ x
    """
    # Step 1: Reshape x to (q, p) and apply R via batch matmul
    X = x.reshape(q, p)
    X = torch.bmm(R_blocks, X.unsqueeze(-1)).squeeze(-1)  # (q, p)

    # Step 2: Transpose (= apply permutation P_b)
    X = X.T  # (p, q)

    # Step 3: Apply L via batch matmul
    X = torch.bmm(L_blocks, X.unsqueeze(-1)).squeeze(-1)  # (p, q)

    # Step 4: Transpose back
    X = X.T  # (q, p)

    return X.reshape(-1)


def monarch_approximation(A, p, q):
    """
    Find optimal Monarch approximation of dense matrix A.
    Analytical solution via block SVDs.

    Args:
        A: (n, n) dense matrix, n = p * q
        p, q: block dimensions

    Returns:
        L_blocks: (p, q, q), R_blocks: (q, p, p)
    """
    n = A.shape[0]
    # Reshape A into blocks
    A_blocks = A.reshape(p, q, p, q)

    # SVD-based optimal projection per block pair
    L_blocks = torch.zeros(p, q, q)
    R_blocks = torch.zeros(q, p, p)

    # Simplified: actual implementation uses coupled SVDs
    # across block rows and columns
    for i in range(p):
        for j in range(q):
            block = A_blocks[i, :, :, j]  # (q, p)
            U, S, Vh = torch.linalg.svd(block, full_matrices=False)
            # Distribute singular values
            L_blocks[i, :, j] = U[:, 0] * S[0].sqrt()
            R_blocks[j, :, i] = Vh[0, :] * S[0].sqrt()

    return L_blocks, R_blocks
```

## References

- Dao, Chen, Sohoni, Desai, Poli, Grogan, Liu, Rao, Rudra, Re (2022). Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML 2022.
- De Sa, Gu, Ré, Rudra (2018). Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations. ICML 2019.
- Gorbunov, Yudin, Dvurechensky (2024). Group and Shuffle: Efficient Structured Orthogonal Parametrization. NeurIPS 2024.
- Mohamed et al. (2025). Learning Permutations in Monarch Factorization. HAL preprint.

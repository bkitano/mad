# 055: Group-and-Shuffle Matrices

**Category**: decomposition
**Gain type**: efficiency
**Source**: Gorbunov et al. (2024), NeurIPS 2024
**Paper**: [papers/group-and-shuffle-matrices.pdf]
**Documented**: 2026-02-11

## Description

Group-and-Shuffle (GS) matrices are a structured matrix class parameterized as the product of two block-diagonal matrices interleaved with permutations: $A = P_L (L P R) P_R$. The key idea is a "group-and-shuffle" mechanism: a block-diagonal matrix $R$ applies fully-connected transformations within subgroups of coordinates, then a permutation $P$ shuffles entries into new subgroups, and a second block-diagonal matrix $L$ processes the reshuffled groups. This generalizes Monarch matrices and block butterfly matrices while requiring fewer matrix factors to form dense orthogonal matrices, directly connecting the expressivity of permutations and structured orthogonal transformations.

GS matrices provide a principled framework where permutations serve as the "mixing" mechanism between block-diagonal orthogonal transformations — exactly the interplay between permutations and Householder-like orthogonal operations that underlies efficient orthogonal parameterization.

## Mathematical Form

**Core Operation:**

$$
A = P_L \left( L \cdot P \cdot R \right) P_R
$$

where:
- $L = \text{diag}(L_1, L_2, \ldots, L_{k_L})$ — left block-diagonal, $L_i \in \mathbb{R}^{b_L \times b_L}$
- $R = \text{diag}(R_1, R_2, \ldots, R_{k_R})$ — right block-diagonal, $R_i \in \mathbb{R}^{b_R \times b_R}$
- $P_L, P, P_R \in \{0, 1\}^{n \times n}$ — permutation matrices
- Dimensional constraint: $b_L \cdot k_L = b_R \cdot k_R = s$ (intermediate dimension)

**Key Definitions:**

- $n$ — matrix dimension
- $k_L, k_R$ — number of blocks in left/right block-diagonal matrices
- $b_L, b_R$ — block sizes
- $\mathcal{GS}(P_L, P, P_R)$ — the class of all matrices of the form $P_L(LPR)P_R$

**Product of GS matrices (for dense coverage):**

To form dense $n \times n$ orthogonal matrices, stack $m$ GS factors:

$$
A = \prod_{i=1}^{m} \text{GS}_i = \prod_{i=1}^{m} P_{L,i} (L_i P_i R_i) P_{R,i}
$$

**Orthogonality (Theorem 1):**

Any orthogonal matrix $A \in \mathcal{GS}(P_L, P, P_R)$ admits a representation where all blocks $L_i, R_i$ are individually orthogonal. Orthogonality of the product reduces to orthogonality of the component blocks.

**Block orthogonality enforcement via Cayley parameterization:**

$$
Q_i = (I + K_i)(I - K_i)^{-1}
$$

where $K_i$ is a learnable skew-symmetric matrix, ensuring each block is orthogonal.

## Complexity

| Operation | Dense Orthogonal | Block Butterfly ($m$ factors) | GS ($m$ factors) |
|-----------|-----------------|------------------------------|-------------------|
| Factors needed for density | 1 | $m = 1 + \lceil \log_2 r \rceil$ | $m = 1 + \lceil \log_b r \rceil$ |
| Parameters per factor | $O(n^2)$ | $O(r \cdot b^2)$ | $O(r \cdot b^2)$ |
| Mat-vec per factor | $O(n^2)$ | $O(n \cdot b)$ | $O(n \cdot b)$ |

**Concrete example ($n = 1024$, block size $b = 32$, $r = n/b = 32$):**

| Method | Factors needed | Total parameters |
|--------|---------------|-----------------|
| Block Butterfly | $m = 6$ | $6 \times 32 \times 32^2 = 196{,}608$ |
| GS | $m = 2$ | $2 \times 32 \times 32^2 = 65{,}536$ |
| Dense | $m = 1$ | $1{,}048{,}576$ |

**Theorem 2 (Tightness):** For $m < 1 + \lceil \log_b r \rceil$, all matrices in the product class $\mathcal{GS}(P_{m+1}, \ldots, P_1)$ contain zero blocks, so the bound is tight.

**Memory:** $O(n \cdot b \cdot m)$ parameters vs $O(n^2)$ for dense

## Applicability

- **Parameter-efficient fine-tuning (PEFT):** The GSOFT method applies GS orthogonal matrices for efficient fine-tuning of large models, competing with LoRA, OFT, and BOFT
- **Text-to-image diffusion models:** Validated on adapting Stable Diffusion models with fewer trainable parameters than BOFT
- **Language model fine-tuning:** Demonstrated on downstream tasks with efficient orthogonal weight adaptation
- **Convolutional architectures:** Adapted for compressing and speeding up orthogonal convolution layers
- **Structured state-transition matrices:** The GS framework can parameterize orthogonal state transitions in linear RNNs using the permutation-interleaved-with-blocks pattern

## Limitations

- Permutations $P_L, P, P_R$ are typically fixed (not learned), limiting expressivity per factor
- Requires choosing block sizes and number of factors — not fully automatic
- The Cayley parameterization for block orthogonality introduces overhead (matrix inverse per block)
- Approximation quality depends on the choice of permutations; the projection algorithm (Algorithm 1 in paper) provides optimal Frobenius-norm approximation via SVD but is $O(n^2)$
- Not all orthogonal matrices can be efficiently approximated with a small number of GS factors

## Implementation Notes

```python
import torch

def gs_matvec(L_blocks, R_blocks, perm_L, perm, perm_R, x):
    """
    Apply GS matrix A = P_L @ L @ P @ R @ P_R to vector x.

    Args:
        L_blocks: list of (b_L, b_L) block matrices
        R_blocks: list of (b_R, b_R) block matrices
        perm_L, perm, perm_R: (n,) permutation index arrays
        x: (n,) input vector
    """
    n = x.shape[0]

    # Step 1: Apply P_R
    y = x[perm_R]

    # Step 2: Apply block-diagonal R
    b_R = R_blocks[0].shape[0]
    y = y.reshape(-1, b_R)
    y = torch.stack([R @ yi for R, yi in zip(R_blocks, y)])
    y = y.reshape(n)

    # Step 3: Apply permutation P (the "shuffle")
    y = y[perm]

    # Step 4: Apply block-diagonal L
    b_L = L_blocks[0].shape[0]
    y = y.reshape(-1, b_L)
    y = torch.stack([L @ yi for L, yi in zip(L_blocks, y)])
    y = y.reshape(n)

    # Step 5: Apply P_L
    y = y[perm_L]

    return y


def cayley_orthogonal(K):
    """
    Cayley parameterization: maps skew-symmetric K to orthogonal Q.
    Q = (I + K)(I - K)^{-1}
    """
    n = K.shape[0]
    I = torch.eye(n, device=K.device)
    return torch.linalg.solve(I - K, I + K)
```

## References

- Gorbunov, Yudin, Dvurechensky (2024). Group and Shuffle: Efficient Structured Orthogonal Parametrization. NeurIPS 2024.
- Dao, Chen, Sohoni et al. (2022). Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML 2022.
- Qiu et al. (2024). Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization (BOFT). ICLR 2024.
- Liu et al. (2023). Orthogonal Finetuning (OFT).

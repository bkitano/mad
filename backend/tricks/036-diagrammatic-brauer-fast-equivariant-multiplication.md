# 036: Diagrammatic Brauer Algebra Fast Equivariant Multiplication

**Category**: algebraic
**Gain type**: efficiency
**Source**: Pearce-Crump & Knottenbelt (2024)
**Paper**: [papers/diagrammatic-equivariant-fast-multiplication.pdf] (arXiv:2412.10837)
**Documented**: 2026-02-15

## Description

Group equivariant neural networks that use high-order tensor power spaces $(\mathbb{R}^n)^{\otimes k}$ as their layers face a computational bottleneck: applying an equivariant weight matrix $W$ to an input vector $v \in (\mathbb{R}^n)^{\otimes k}$ naively requires $O(n^{l+k})$ time (the weight matrix has $n^l \times n^k$ entries). This trick uses a **diagrammatic framework based on category theory** to factor each equivariant weight matrix into a sequence of smaller operations, achieving an **exponential reduction** in the Big-$O$ time complexity of the forward pass.

The key insight is that every equivariant weight matrix between tensor power spaces can be expressed as a linear combination of matrices that correspond to **set partition diagrams** (for $S_n$) or **Brauer diagrams** (for $O(n)$, $SO(n)$, $Sp(n)$) under a **monoidal functor** from a diagram category to the representation category. The monoidal property means that tensor products of diagrams map to Kronecker products of matrices, so **decomposing a diagram into a tensor product of smaller diagrams** yields a factorization of the matrix into a Kronecker product of smaller matrices that can be applied sequentially. By choosing an optimal "algorithmically planar" decomposition, the computation is broken into three phases — tensor contractions (bottom-row blocks), transfer operations (cross-row pairs), and index copies (top-row blocks) — applied right-to-left.

For the orthogonal group $O(n)$ (the continuous group containing the hyperoctahedral group $B_n$), the algorithm uses **Brauer diagrams** (set partition diagrams where every block has exactly size 2), and achieves complexity $O(n^{k-1})$ vs the naive $O(n^{l+k})$. This is directly relevant to signed permutation symmetries because $B_n \subset O(n)$, and $O(n)$-equivariant layers automatically respect $B_n$-symmetry.

## Mathematical Form

**Setup:**

A weight matrix $W \in \text{Hom}_{G(n)}((\mathbb{R}^n)^{\otimes k}, (\mathbb{R}^n)^{\otimes l})$ maps between tensor power layer spaces. For each of the four classical groups:

- $G(n) = S_n$: $W = \sum_{d_\pi} \lambda_\pi D_\pi$ where $d_\pi$ ranges over $(k,l)$-partition diagrams
- $G(n) = O(n)$: $W = \sum_{d_\beta} \lambda_\beta E_\beta$ where $d_\beta$ ranges over $(k,l)$-Brauer diagrams
- $G(n) = Sp(n)$: $W = \sum_{d_\beta} \lambda_\beta F_\beta$ where $d_\beta$ ranges over $(k,l)$-Brauer diagrams
- $G(n) = SO(n)$: $W = \sum_{d_\beta} \lambda_\beta E_\beta + \sum_{d_\alpha} \lambda_\alpha H_\alpha$ (Brauer + $(l+k)\backslash n$-diagrams)

**Monoidal Functor (key structural result):**

There exist full, strict $\mathbb{R}$-linear monoidal functors:

$$
\Theta : \mathcal{P}(n) \to \mathcal{C}(S_n), \quad \Phi : \mathcal{B}(n) \to \mathcal{C}(O(n)), \quad X : \mathcal{B}(n) \to \mathcal{C}(Sp(n))
$$

where $\mathcal{P}(n)$ is the partition category, $\mathcal{B}(n)$ is the Brauer category, and $\mathcal{C}(G(n))$ is the tensor power representation category.

The **monoidal property** $\Phi(f \otimes g) = \Phi(f) \otimes_\mathcal{D} \Phi(g)$ ensures:

$$
\text{If } d_\beta = d_1 \otimes d_2 \otimes \cdots \otimes d_m, \quad \text{then } E_\beta = E_1 \otimes E_2 \otimes \cdots \otimes E_m
$$

(tensor product of diagrams $\mapsto$ Kronecker product of matrices).

**Algorithm (MatrixMult):**

Given a group $G(n)$, an appropriate $(k,l)$-diagram $d_\pi$, and input $v \in (\mathbb{R}^n)^{\otimes k}$:

1. **Factor**: Decompose $d_\pi$ into three diagrams via string manipulation:
   - A $(k,k)$-permutation diagram $\sigma_k$
   - An algorithmically planar $(k,l)$-diagram $d_\pi'$
   - An $(l,l)$-permutation diagram $\sigma_l$

2. **Permute**: $v \leftarrow \text{Permute}(v, \sigma_k)$ — rearrange tensor indices (free)

3. **PlanarMult**: Apply the planar diagram via three sequential phases:
   - **Step 1 (Bottom-row contractions):** For each bottom-row block $B_i$, contract tensor indices by summing over shared dimensions
   - **Step 2 (Cross-row transfers):** Apply the middle diagram connecting top and bottom rows — for $O(n)$ this is the **identity** (zero cost!)
   - **Step 3 (Top-row copies):** Duplicate/broadcast tensor indices for each top-row block

4. **Permute**: $w \leftarrow \text{Permute}(w, \sigma_l)$ — final index rearrangement (free)

**For $O(n)$ (Brauer diagrams):**

A $(k,l)$-Brauer diagram $d_\beta$ decomposes into pairs: $t$ top-row pairs, $d$ cross-row pairs, $b$ bottom-row pairs, satisfying $2t + d = l$ and $2b + d = k$.

- **Bottom-row pairs** perform tensor contraction: $r_M = \sum_{j \in [n]} w_{M,j,j}$ (trace-like)
- **Cross-row pairs** are identity maps under $\Phi$ (zero cost!)
- **Top-row pairs** perform index duplication: $e_m \otimes e_m$ (copy)

**Key Definitions:**

- $(k,l)$-partition diagram $d_\pi$: bipartite graph with $l$ top vertices, $k$ bottom vertices, edges from set partition $\pi$ of $[l+k]$
- $(k,l)$-Brauer diagram $d_\beta$: partition diagram where every block has size exactly 2 (a perfect matching)
- Algorithmically planar: a diagram with bottom blocks right-aligned in decreasing size, top blocks left-aligned, and no crossing between cross-row pairs

## Complexity

| Group | Naive | With Diagrammatic Trick | Reduction |
|-------|-------|------------------------|-----------|
| $S_n$: partition diagram | $O(n^{l+k})$ | $O(n^k)$ (worst), $O(n)$ (best) | Exponential in $l$ |
| $O(n)$: Brauer diagram | $O(n^{l+k})$ | $O(n^{k-1})$ | Exponential in $l+1$ |
| $Sp(n)$: Brauer diagram | $O(n^{l+k})$ | $O(n^{k-1})$ | Exponential in $l+1$ |
| $SO(n)$: Brauer diagram | $O(n^{l+k})$ | $O(n^{k-1})$ | Exponential in $l+1$ |
| $SO(n)$: $(l+k)\backslash n$-diagram | $O(n^{l+k})$ | $O(n^{k-(n-s)}(n! + n^{s-1}))$ | Depends on $s$ |

**For the orthogonal group $O(n)$ specifically:**

- Step 1 (bottom-row contractions): $O(n^{k-1})$ — each contraction sums $n$ terms, reducing tensor order by 2
- Step 2 (cross-row transfers): $O(1)$ — identity transformation!
- Step 3 (top-row copies): $O(1)$ — array copies only

**Memory:** $O(n^{\max(k,l)})$ for the input/output vectors; the diagram itself requires $O(k+l)$ storage.

## Applicability

- **$O(n)$-equivariant neural networks with tensor power layers:** The primary application — any network using high-order tensor representations of $O(n)$ benefits from exponential speedup in the forward pass
- **Hyperoctahedral-equivariant models:** Since $B_n \subset O(n)$, any $O(n)$-equivariant layer is automatically $B_n$-equivariant; the Brauer algebra fast multiplication directly applies
- **Point cloud processing with reflection symmetry:** Data with signed permutation symmetries (e.g., molecules with chirality, crystallographic symmetries) can use $O(n)$-equivariant layers accelerated by this trick
- **Higher-order equivariant graph networks:** Architectures like Invariant Graph Networks (Maron et al. 2019) use tensor power spaces; this algorithm makes higher-order ($k \geq 3$) layers practical
- **Physics simulations with $O(n)$ symmetry:** Dynamics prediction, molecular property prediction, and particle physics applications where orthogonal symmetry is fundamental

## Limitations

- The paper covers $S_n$, $O(n)$, $SO(n)$, and $Sp(n)$ but does **not** directly address the hyperoctahedral group $B_n$ as a standalone group — $B_n$-specific diagram algebras (e.g., based on signed Brauer diagrams) could yield even tighter results
- The exponential improvement is in the Big-$O$ exponent; constant factors from the diagram decomposition and permutation steps may be significant for small $n$
- The algorithm processes each spanning set element independently; the full weight matrix $W = \sum \lambda_i M_i$ requires one fast multiplication per spanning set element, with parallelism across elements
- GPU-efficient implementation requires careful memory layout for the tensor contraction, transfer, and copy operations — the sequential right-to-left application may limit parallelism within each diagram
- For very high tensor orders $k$, even $O(n^{k-1})$ may be prohibitive; the technique is most impactful when $l$ is large (many top-row vertices to "remove")
- The algorithm assumes the weight matrix is expressed in the diagram basis; converting from other parameterizations adds overhead

## Implementation Notes

```python
import torch
from itertools import permutations

def brauer_fast_multiply(diagram, v, n):
    """
    Fast multiplication of Brauer diagram matrix E_β with vector v.

    For O(n)-equivariant layers: reduces O(n^{l+k}) to O(n^{k-1}).

    Args:
        diagram: dict with keys:
            'top_pairs': list of (i,j) pairs in top row
            'cross_pairs': list of (top_i, bot_j) pairs across rows
            'bottom_pairs': list of (i,j) pairs in bottom row
            'perm_k': permutation σ_k from Factor
            'perm_l': permutation σ_l from Factor
        v: tensor of shape (n,) * k — input in (R^n)^{⊗k}
        n: dimension of R^n

    Returns:
        w: tensor of shape (n,) * l — output in (R^n)^{⊗l}
    """
    k = v.ndim
    t = len(diagram['top_pairs'])      # top-row pairs
    d = len(diagram['cross_pairs'])     # cross-row pairs
    b = len(diagram['bottom_pairs'])    # bottom-row pairs
    l = 2 * t + d

    # Step 0: Permute input indices
    perm_k = diagram['perm_k']
    v = v.permute(perm_k)

    # Step 1: Contract bottom-row pairs (right to left, largest first)
    # Each bottom pair contracts two indices by tracing
    w = v
    for i, (idx1, idx2) in enumerate(reversed(diagram['bottom_pairs'])):
        # Contract: sum over the diagonal where idx1 == idx2
        # This reduces tensor order by 2
        w = torch.diagonal(w, dim1=idx1, dim2=idx2).sum(-1)
        # After contraction, remaining shape has 2 fewer dimensions

    # Step 2: Transfer (cross-row pairs) — identity for O(n)!
    # No computation needed.

    # Step 3: Copy top-row pairs (duplicate indices)
    # Each top pair adds two new dimensions with shared index
    for (idx1, idx2) in diagram['top_pairs']:
        # Expand: add dimension and repeat along diagonal
        w = w.unsqueeze(-1).expand(*w.shape, n)
        # Create diagonal embedding e_m ⊗ e_m

    # Step 4: Permute output indices
    perm_l = diagram['perm_l']
    w = w.permute(perm_l)

    return w


def factor_brauer_diagram(top_pairs, cross_pairs, bottom_pairs, k, l):
    """
    Factor a Brauer diagram into:
    1. Input permutation σ_k
    2. Algorithmically planar Brauer diagram
    3. Output permutation σ_l

    The planar diagram has:
    - Bottom pairs right-aligned
    - Top pairs left-aligned
    - Cross pairs in between, non-crossing
    """
    # Compute permutations that make the diagram algorithmically planar
    # This is the "string dragging" procedure from the paper

    # Bottom pairs: line up on far right, consecutive vertices
    bottom_positions = []
    pos = k - 1
    for pair in sorted(bottom_pairs, key=lambda p: -max(p)):
        bottom_positions.append((pos - 1, pos))
        pos -= 2

    # Cross pairs: arrange in middle, non-crossing
    # Top pairs: line up on far left, consecutive vertices

    return {
        'top_pairs': top_pairs,
        'cross_pairs': cross_pairs,
        'bottom_pairs': bottom_pairs,
        'perm_k': list(range(k)),  # Simplified; actual impl computes from diagram
        'perm_l': list(range(l)),
    }


# Example: For a (5,5)-Brauer diagram with b=1 bottom pair,
# d=1 cross pair, t=1 top pair:
# Naive: O(n^10)
# Fast:  O(n^4)  — one contraction step costs n^(k-2) = n^3,
#                  identity transfer, and free copy
```

## References

- Pearce-Crump, E. & Knottenbelt, W.J. (2024). A Diagrammatic Approach to Improve Computational Efficiency in Group Equivariant Neural Networks. arXiv:2412.10837.
- Pearce-Crump, E. (2023a). Brauer's Group Equivariant Neural Networks. ICML 2023.
- Pearce-Crump, E. (2023b). How Jellyfish Characterise Alternating Group Equivariant Neural Networks. ICML 2023.
- Godfrey, C., Rawson, M.G., Brown, D. & Kvinge, H. (2023). Fast computation of permutation equivariant layers with the partition algebra. ICLR 2023 Workshop.
- Maron, H., Ben-Hamu, H., Shamir, N. & Lipman, Y. (2019). Invariant and Equivariant Graph Networks. ICLR 2019.
- Brauer, R. (1937). On algebras which are connected with the semisimple continuous groups. *Annals of Mathematics*, 38:857–872.
- Lehrer, G.I. & Zhang, R.B. (2012). The Brauer category and invariant theory. *J. European Math. Soc.*, 17:2311–2351.

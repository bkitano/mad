# 144: Wreath Product FFT (Hyperoctahedral Group)

**Category**: algebraic
**Gain type**: efficiency
**Source**: Clausen (1989, 1995); Maslen & Rockmore (1997); Rockmore (2004)
**Paper**: [papers/wreath-product-fft-rockmore.pdf] (Rockmore — Recent Progress and Applications in Group FFTs)
**Documented**: 2026-02-15

## Description

The **wreath product FFT** is a generalization of the Cooley–Tukey FFT from cyclic groups to wreath products $G \wr S_n = G^n \rtimes S_n$. The most important special case is the **hyperoctahedral group** $B_n = \mathbb{Z}_2 \wr S_n$, the group of signed permutations. Whereas a naive DFT on a group $G$ of order $|G|$ requires $O(|G|^2)$ operations, the wreath product FFT computes the full Fourier transform in $O(|G| \log^c |G|)$ operations for small constant $c$ (typically $c \leq 4$).

The key idea is the **separation of variables** approach: exploit the chain of subgroups $G_n > G_{n-1} > \cdots > G_1 > \{e\}$ and the corresponding Bratteli diagram to factor the DFT matrix into a product of sparse matrices. For wreath products, the semidirect product structure $G^n \rtimes S_n$ allows a two-stage recursion: first transform over the "fiber" copies of $G$, then transform over the permutation action of $S_n$, analogous to how the Cooley–Tukey FFT splits a DFT of length $pq$ into $p$-point and $q$-point sub-transforms.

This is directly relevant to neural networks because: (1) **convolutions on hyperoctahedral groups** arise when building equivariant layers for data with signed permutation symmetries (e.g., point clouds under reflections, state machines tracking signed permutations); (2) the FFT enables fast evaluation of the group algebra $\mathbb{C}[B_n]$, converting convolution to pointwise multiplication in the Fourier domain; (3) it provides the algorithmic backbone for spectral methods on wreath product groups used in ranking, experimental design, and combinatorial optimization.

## Mathematical Form

**Core Operation — Group DFT:**

Given a function $f: G \to \mathbb{C}$ on a finite group $G$ with irreducible representations $\{\rho_\lambda\}_{\lambda \in \hat{G}}$, the Fourier transform is:

$$
\hat{f}(\lambda) = \sum_{g \in G} f(g) \rho_\lambda(g) \in \mathbb{C}^{d_\lambda \times d_\lambda}
$$

where $d_\lambda = \dim(\rho_\lambda)$.

**Convolution Theorem (noncommutative):**

$$
\widehat{f \star h}(\lambda)_{k,\ell} = \sum_m \hat{f}(\lambda)_{k,m} \hat{h}(\lambda)_{m,\ell}
$$

i.e., convolution in the spatial domain becomes matrix multiplication in each irreducible block of the Fourier domain.

**Key Definitions:**

- $B_n = \mathbb{Z}_2 \wr S_n = \mathbb{Z}_2^n \rtimes S_n$ — the hyperoctahedral group, $|B_n| = 2^n n!$
- $\hat{B}_n$ — the set of irreducible representations of $B_n$, indexed by pairs of partitions $(\alpha, \beta)$ with $|\alpha| + |\beta| = n$
- $d_{(\alpha,\beta)}$ — dimension of irreducible representation indexed by $(\alpha, \beta)$

**Wreath Product Structure:**

For $B_n = \mathbb{Z}_2 \wr S_n$, elements are pairs $(\mathbf{s}, \sigma)$ where $\mathbf{s} = (s_1, \ldots, s_n) \in \mathbb{Z}_2^n$ and $\sigma \in S_n$. The group operation is:

$$
(\mathbf{s}, \sigma) \cdot (\mathbf{t}, \tau) = (\mathbf{s} + \sigma(\mathbf{t}), \sigma \tau)
$$

where $\sigma(\mathbf{t})_i = t_{\sigma^{-1}(i)}$.

**Separation of Variables (Decimation in Time):**

The algorithm exploits the subgroup chain $B_n > B_{n-1} > \cdots > B_1 > \{e\}$ and uses Gel'fand–Tsetlin bases adapted to this chain. In the Bratteli diagram:

1. Nodes at level $k$ correspond to irreducible representations of $B_k$
2. Edges encode restriction multiplicities: $\rho|_{B_{k-1}}$
3. Paths from root to a node at level $n$ index basis vectors (Gel'fand–Tsetlin basis)

The DFT is factored into $n$ stages, where stage $k$ processes the inclusion $B_{k-1} \hookrightarrow B_k$ by computing over coset representatives.

**Two-Stage Recursion for Wreath Products:**

For $G \wr S_n$ with $|G| = m$:

- **Stage 1 (fiber transform):** Apply $n$ independent FFTs on $G$ (one per coordinate), requiring $n \cdot T_{\text{FFT}}(G)$ operations.
- **Stage 2 (permutation transform):** Apply an FFT on $S_n$ to combine the fiber-transformed data, requiring $T_{\text{FFT}}(S_n)$ operations.

For $B_n = \mathbb{Z}_2 \wr S_n$: Stage 1 consists of $n$ trivial 2-point transforms (on $\mathbb{Z}_2$), and Stage 2 is an FFT on $S_n$.

## Complexity

| Operation | Naive (DFT) | With Wreath FFT |
|-----------|-------------|-----------------|
| DFT on $B_n$ | $O(|B_n|^2) = O(4^n (n!)^2)$ | $O(|B_n| \log^4 |B_n|)$ |
| DFT on $S_n$ | $O((n!)^2)$ | $O((n!) \log^2(n!))$ (Maslen) |
| DFT on $\mathbb{Z}_2^n$ | $O(4^n)$ | $O(n \cdot 2^n)$ (standard FFT) |
| Group convolution on $B_n$ | $O(|B_n|^2)$ | $O(|B_n| \log^4 |B_n|)$ (FFT + pointwise + IFFT) |

**Memory:** $O(|B_n|)$ for the function values; $O(\sum_\lambda d_\lambda^2)$ for the Fourier coefficients (which equals $|B_n|$ by Plancherel).

**Note:** For the hyperoctahedral group $B_n$, $|B_n| = 2^n n!$ and $\log |B_n| = n \log 2 + \sum_{k=1}^n \log k \approx n \log n$, so the FFT complexity is roughly $O(2^n n! \cdot (n \log n)^4)$.

## Applicability

- **Equivariant neural networks:** Fast convolution layers for architectures with $B_n$-symmetry (e.g., point cloud processing with reflection symmetry, hyperoctahedral equivariant networks)
- **State-space models with signed permutation transitions:** When the state transition matrix lives in (or near) $B_n$, spectral analysis via the group FFT can accelerate training and inference
- **Combinatorial optimization:** Fourier analysis on $S_n$ and $B_n$ is used in ranking problems, multi-object tracking, and experimental design — all areas where learned representations benefit from spectral structure
- **Group convolution in GM-CNNs:** The wreath product FFT provides the efficient computational backbone for group-convolutional layers when the symmetry group is a wreath product (e.g., $B_n$, $D_n \wr S_k$)

## Limitations

- The group $B_n$ grows as $2^n n!$, which is superexponential — practical only for small $n$ (typically $n \leq 10$-$15$)
- The constant in the $\log^4$ factor can be large, making the asymptotic advantage kick in only for moderately large groups
- Irreducible representations of $B_n$ (indexed by pairs of partitions) have varying dimensions; the largest irreducibles have dimension $\sim \sqrt{n!}$, requiring significant memory for Fourier coefficients
- Implementation complexity: unlike the cyclic FFT (a single butterfly network), the group FFT requires Gel'fand–Tsetlin bases and Bratteli diagram traversal, making GPU-efficient implementations challenging
- The noncommutative convolution theorem involves matrix multiplication in each irreducible block, adding overhead beyond pointwise scaling

## Implementation Notes

```python
import numpy as np
from itertools import permutations, product

def hyperoctahedral_elements(n):
    """
    Generate all elements of B_n = Z_2^n ⋊ S_n.
    Each element is (signs, perm) where signs ∈ {+1,-1}^n, perm ∈ S_n.
    """
    for signs in product([-1, 1], repeat=n):
        for perm in permutations(range(n)):
            yield (signs, perm)

def hyperoctahedral_multiply(a, b, n):
    """
    Group multiplication in B_n:
    (s, σ) · (t, τ) = (s + σ(t), σ ∘ τ)
    where σ(t)_i = t_{σ^{-1}(i)} and + is componentwise in Z_2 (as ±1, so multiply).
    """
    s, sigma = a
    t, tau = b
    # σ acts on t by permuting coordinates
    sigma_t = tuple(t[sigma.index(i)] for i in range(n))
    # Signs compose multiplicatively (±1)
    new_signs = tuple(s[i] * sigma_t[i] for i in range(n))
    # Permutations compose
    new_perm = tuple(sigma[tau[i]] for i in range(n))
    return (new_signs, new_perm)

def naive_group_dft(f_values, group_elements, representations):
    """
    Naive DFT: hat{f}(λ) = Σ_g f(g) ρ_λ(g)
    O(|G|^2) — this is what the wreath product FFT replaces.
    """
    fourier = {}
    for lam, rho in representations.items():
        d = rho[group_elements[0]].shape[0]
        fourier[lam] = np.zeros((d, d), dtype=complex)
        for g, fg in zip(group_elements, f_values):
            fourier[lam] += fg * rho[g]
    return fourier

def group_convolution_fourier(f_hat, h_hat):
    """
    Convolution theorem: hat{f * h}(λ) = hat{f}(λ) @ hat{h}(λ)
    Apply after FFT, then IFFT to get spatial-domain result.
    """
    result = {}
    for lam in f_hat:
        result[lam] = f_hat[lam] @ h_hat[lam]
    return result

# The actual wreath product FFT would:
# 1. Apply n independent Z_2 FFTs (trivial: just sum/difference)
# 2. Apply S_n FFT using Gel'fand-Tsetlin bases
# 3. Combine using Bratteli diagram structure
# This requires specialized representation-theoretic machinery.
```

## References

- Clausen, M. (1989). Fast generalized Fourier transforms. *Theoret. Comput. Sci.* 67(1), 55–63.
- Clausen, M. & Baum, U. (1993). *Fast Fourier Transforms*. BI-Wissenschaftsverlag.
- Clausen, M. (1995). Fast Fourier transforms for wreath products. *Appl. Comput. Harmon. Anal.* 2(3), 34–55.
- Maslen, D. & Rockmore, D. (1997). Generalized FFTs — a survey of some recent results. *Groups and Computation, II*, AMS, 183–237.
- Maslen, D. & Rockmore, D. (1997). Separation of variables and the computation of Fourier transforms on finite groups, I. *J. Amer. Math. Soc.* 10(1), 169–214.
- Rockmore, D. (2004). Recent progress and applications in group FFTs. *NATO Science Series*, Springer.
- Wikipedia: Hyperoctahedral group. https://en.wikipedia.org/wiki/Hyperoctahedral_group

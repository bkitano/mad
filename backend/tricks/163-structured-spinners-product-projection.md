# 163: Structured Spinners Product Projection

**Category**: approximation
**Gain type**: efficiency|flexibility
**Source**: Bojarski, Choromanska, Choromanski et al. (2017), "Structured Adaptive and Random Spinners for Fast Machine Learning Computations" (AISTATS 2017)
**Paper**: [papers/structured-spinners.pdf]
**Documented**: 2026-02-15

## Description

Structured Spinners is a general framework that unifies and extends all structured random projection methods (Fastfood, SORF, circulant projections, etc.) into a single parametric family. Each structured spinner is a product of three structured matrix blocks $\mathbf{G}_{\text{struct}} = \mathbf{M}_3 \mathbf{M}_2 \mathbf{M}_1$, where each block plays a specific geometric role: $\mathbf{M}_1$ balances the input (no dimension dominates), $\mathbf{M}_2$ makes projections near-orthogonal, and $\mathbf{M}_3$ provides the random/adaptive directional structure. The framework comes with theoretical guarantees (via Berry-Esseen CLT for random vectors) showing that structured spinners can replace dense Gaussian matrices in essentially any machine learning algorithm with minimal accuracy loss.

The key theoretical contribution is **Theorem 1**: for any randomized algorithm $\mathcal{A}$ using unstructured Gaussian matrices $\mathbf{G}$, replacing $\mathbf{G}$ with a structured spinner $\mathbf{G}_{\text{struct}}$ from blocks of $m$ rows, the output distributions converge with probability $p_{\text{succ}} \geq 1 - 2p(n)d - 2\binom{md}{2}e^{-\Omega(\min(\cdot))}$, which approaches 1 for large $n$. This means structured spinners are drop-in replacements for dense random matrices in kernel approximation, dimensionality reduction, LSH, Newton sketches, neural network layers, and more.

Notable special cases in the family include:
- $\sqrt{n}\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (the SORF/cross-polytope LSH matrix)
- $\mathbf{G}_{\text{circ}}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (Gaussian circulant spinner)
- $\mathbf{G}_{\text{Toeplitz}}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (Gaussian Toeplitz spinner)
- $\mathbf{G}_{\text{skew-circ}}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (skew-circulant spinner)
- $\mathbf{H}\mathbf{D}_{g_1,\ldots,g_n}\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (Gaussian diagonal spinner)

In the **adaptive** (learnable) setting, the random parameters become trainable, turning structured spinners into parameter-efficient replacements for dense weight matrices in neural networks — $O(d)$ parameters instead of $O(d^2)$, with $O(d \log d)$ forward pass.

## Mathematical Form

**Core Operation:**

$$
\mathbf{G}_{\text{struct}} = \mathbf{M}_3 \mathbf{M}_2 \mathbf{M}_1 \in \mathbb{R}^{n \times n}
$$

where the three blocks satisfy:

**Block $\mathbf{M}_1$ — Balancedness:**

$\mathbf{M}_1$ and $\mathbf{M}_2\mathbf{M}_1$ are $(\delta(n), p(n))$-balanced isometries, meaning for every $\|\mathbf{x}\|_2 = 1$:

$$
\mathbb{P}\left[\|\mathbf{M}_1\mathbf{x}\|_\infty > \frac{\delta(n)}{\sqrt{n}}\right] \leq p(n)
$$

This ensures no single coordinate dominates after transformation. The canonical choice is $\mathbf{M}_1 = \mathbf{H}\mathbf{D}_1$ where $\mathbf{D}_1$ is a Rademacher diagonal. Then $\mathbf{M}_1$ is $(\log(n), 2ne^{-\frac{\log^2(n)}{8}})$-balanced (Remark 1, from Ailon & Chazelle 2006).

**Block $\mathbf{M}_2$ — Near-orthogonality:**

$\mathbf{M}_2 = \mathbf{V}(\mathbf{W}^1, \ldots, \mathbf{W}^n)\mathbf{D}_{\rho_1,\ldots,\rho_n}$ where:
- $\mathbf{W}^1, \ldots, \mathbf{W}^n \in \mathbb{R}^{k \times n}$ form a $(\Lambda_F, \Lambda_2)$-smooth set (columns have unit norm, are mutually orthogonal, bounded Frobenius/spectral norms of cross-products)
- $\rho_1, \ldots, \rho_n$ are i.i.d. sub-Gaussian random variables

The role of $\mathbf{M}_2$ is to transform vectors so that their projections of the Gaussian/Rademacher vector $\mathbf{r}$ onto different directions are near-orthogonal, i.e. approximately independent.

**Block $\mathbf{M}_3$ — Directional structure:**

$\mathbf{M}_3 = \mathbf{C}(\mathbf{r}, n) \in \mathbb{R}^{n \times nk}$ where $\mathbf{r} \in \mathbb{R}^k$ is either a random Rademacher/Gaussian vector (random setting) or a learnable parameter vector (adaptive setting).

**Canonical Instance ($\sqrt{n}\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$):**

$$
\mathbf{G}_{\text{struct}} = \sqrt{n}\,\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1
$$

where $\mathbf{D}_i$ are random diagonal $\pm 1$ matrices and $\mathbf{H}$ is the normalized Hadamard matrix. This is equivalent to the SORF matrix (trick 155).

**Unifying Lemma (Lemma 1):**

The following are all valid structured spinners with $\delta(n) = \log(n)$, $p(n) = 2ne^{-\frac{\log^2(n)}{8}}$, $K = 1$, $\Lambda_F = O(\sqrt{n})$, and $\Lambda_2 = O(1)$:

- $\sqrt{n}\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (three-block Hadamard-Rademacher)
- $\mathbf{G}_{\text{circ}}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (circulant + Hadamard)
- $\sqrt{n}\mathbf{H}\mathbf{D}_{g_1,\ldots,g_n}\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ (Gaussian diagonal + Hadamard)

The same holds if one replaces $\mathbf{G}_{\text{circ}}$ by a Gaussian Hankel or Toeplitz matrix.

**Main Approximation Theorem (Theorem 1, simplified):**

Let $\mathcal{A}$ be a randomized algorithm using Gaussian matrix $\mathbf{G}$, operating on functions $f_1, \ldots, f_s$ in $d$-dimensional spaces. Replace $\mathbf{G}$ with structured spinner blocks of $m$ rows. Then for $n$ large enough:

$$
p_{\text{succ}} \geq 1 - 2p(n)d - 2\binom{md}{2} e^{-\Omega\left(\min\left(\frac{i^2 n^2}{K^4 \Lambda_F^2 \delta^4(n)},\, \frac{t n}{K^2 \Lambda_2 \delta^2(n)}\right)\right)}
$$

and for any $b$-convex set $\mathcal{S}$:

$$
\left|\mathbb{P}[f_i(\mathbf{q}_{f_i}) \in \mathcal{S}] - \mathbb{P}[f_i(\mathbf{q}_{f'_i}) \in \mathcal{S}]\right| \leq b\eta
$$

where $\eta = \frac{\delta^3(n)}{n^{5/7}}$, showing the distributions converge as $n$ grows.

**Neural Network Theorem (Theorem 5, adaptive):**

For a neural network weight matrix $\mathbf{M} \in \mathbb{R}^{m \times n}$ between layers $l_0$ (size $n$) and $l_1$ (size $m$), if the input data lies in a $d$-dimensional space $\mathcal{L}$, there exists a structured spinner $\mathbf{M}^{\text{struct}} = \mathbf{M}_3\mathbf{M}_2\mathbf{M}_1$ with a learnable vector $\mathbf{r}$ such that $\mathbf{M}^{\text{struct}}$ equals $\mathbf{M}$ on $\mathcal{L}$ with probability $p_{\text{succ}}$ as above. This means structured spinners can replace FC layers with $O(n)$ parameters instead of $O(mn)$.

## Complexity

| Operation | Dense Gaussian | Structured Spinner |
|-----------|---------------|-------------------|
| Projection $\mathbf{G}\mathbf{x}$ | $O(mn)$ | $O(n \log m)$ or $O(n \log n)$ |
| Storage | $O(mn)$ | $O(n)$ to $O(n \log n)$ |
| Parameter count (adaptive) | $O(mn)$ | $O(n)$ (learnable $\mathbf{r}$ + diagonals) |
| Generation (random) | $O(mn)$ | $O(n)$ (diagonal sampling) |

**Empirical speedups** (from paper, Table 1, single-threaded CPU, kernel approximation):
- $d = 2^9$: $\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ gives **2.2×** speedup
- $d = 2^{10}$: **6.0×** speedup
- $d = 2^{11}$: **14.1×** speedup
- $d = 2^{12}$: **33.3×** speedup
- $d = 2^{13}$: **74.3×** speedup
- $d = 2^{14}$: **140.4×** speedup
- $d = 2^{15}$: **316.8×** speedup

These speedups grow as $O(d / \log d)$, confirming the asymptotic advantage.

**Neural network (MLP on MNIST, Table 2):**

For hidden size $h = 2^{10}$, the structured spinner MLP ($\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ replacing FC layers) runs in **117.4 μs** vs **99.9 μs** for unstructured — slightly slower due to launch overhead at small sizes. But at $h = 2^{12}$, structured takes **389.0 μs** vs **2317.4 μs** unstructured (**5.95× faster**), with comparable test error.

## Applicability

- **Unifying framework for random projections**: Any existing structured projection method (Fastfood, SORF, circulant projections, etc.) is a special case. New instances can be designed by choosing different $\mathbf{M}_1, \mathbf{M}_2, \mathbf{M}_3$ blocks
- **Drop-in replacement for dense random matrices**: Theorem 1 guarantees that *any* algorithm using Gaussian random matrices can use structured spinners with provably similar output distributions. This includes: kernel approximation, JL dimensionality reduction, cross-polytope LSH, Newton sketches for convex optimization, random projection trees
- **Learnable structured layers**: In adaptive mode, structured spinners replace dense FC layers with $O(n)$ parameters and $O(n \log n)$ forward/backward cost. Demonstrated on MNIST: comparable accuracy with far fewer parameters
- **Linear attention acceleration**: Can accelerate the random projection in FAVOR+/Performer attention just like Fastfood/SORF, but the framework allows choosing the best variant for the specific hardware
- **GPU considerations**: The $\mathbf{H}\mathbf{D}_3\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$ instance (three FWHT + elementwise ops) is the most GPU-friendly variant — no permutations or gathers, just FWHT (butterfly) + diagonal. The circulant variants ($\mathbf{G}_{\text{circ}}\mathbf{D}_2\mathbf{H}\mathbf{D}_1$) use FFT instead of one FWHT stage. On modern GPUs, the FWHT-only variant is preferred because it avoids complex-valued arithmetic and maps well to shared memory
- **Tensor core compatibility**: Like Fastfood/SORF, these structured projections replace a single GEMM with sequential structured operations. Best suited when the GEMM would be small ($d = 64$–$256$) and underutilize tensor cores, or when memory bandwidth (not compute) is the bottleneck

## Limitations

- **Framework generality vs. specific implementation**: The paper provides a broad theoretical framework but the optimal choice of blocks depends on the application and hardware. There is no single "best" structured spinner
- **Requires power-of-2 dimension** for Hadamard-based instances. Circulant-based variants work for any $n$ but use FFT (complex arithmetic)
- **Three sequential passes**: All instances require at least 3 sequential structured matrix-vector multiplications (one per block). This is 3 kernel launches or 3 synchronization barriers on GPU
- **Small-dimension regime**: For $d \leq 64$ (typical attention head dimension), the overhead of 3 FWHT passes may not beat a single optimized small GEMM on tensor cores
- **Adaptive mode training complexity**: In the learnable setting, the structured spinner has $O(n)$ parameters but the gradient computation requires backpropagating through the structured product, which may be less numerically stable than standard dense layer gradients
- **Correlation between features within a block**: Like Fastfood and SORF, features within a single $n \times n$ block are correlated (not independent). The theory bounds this correlation but it doesn't vanish

## Implementation Notes

```python
import torch
import math

class StructuredSpinner:
    """General structured spinner: G_struct = M3 @ M2 @ M1.

    Canonical instance: sqrt(n) * H D3 H D2 H D1
    - Most GPU-friendly: no permutations, no complex arithmetic
    - Three FWHT passes + three elementwise multiplications
    """

    def __init__(self, d, n_blocks=1, mode='random', device='cuda'):
        """
        Args:
            d: dimension (padded to power of 2)
            n_blocks: number of d×d blocks (for n > d features)
            mode: 'random' (fixed Rademacher diags) or 'adaptive' (learnable)
        """
        self.d = d
        self.d_pad = 1 << (d - 1).bit_length()  # next power of 2
        self.n_blocks = n_blocks

        # Each block has 3 diagonal sign vectors
        if mode == 'random':
            self.diags = [
                [torch.randint(0, 2, (self.d_pad,), device=device).float() * 2 - 1
                 for _ in range(3)]
                for _ in range(n_blocks)
            ]
        else:  # adaptive: learnable diagonals
            self.diags = [
                [torch.nn.Parameter(torch.randn(self.d_pad, device=device))
                 for _ in range(3)]
                for _ in range(n_blocks)
            ]

    def project(self, x):
        """Apply structured spinner projection.

        Args:
            x: (..., d) input
        Returns:
            (..., n_blocks * d_pad) projected features
        """
        # Pad to power of 2
        if x.shape[-1] < self.d_pad:
            x = torch.nn.functional.pad(x, (0, self.d_pad - x.shape[-1]))

        outputs = []
        for block_diags in self.diags:
            D1, D2, D3 = block_diags
            y = x * D1                          # M1: sign flip
            y = fwht(y)                          # M1: Hadamard
            y = y * D2                          # M2: sign flip
            y = fwht(y)                          # M2: Hadamard
            y = y * D3                          # M3: sign flip
            y = fwht(y)                          # M3: Hadamard
            outputs.append(y * math.sqrt(self.d_pad))

        return torch.cat(outputs, dim=-1)

def fwht(x):
    """Fast Walsh-Hadamard Transform (normalized)."""
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a, b = x[..., 0, :].clone(), x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        h *= 2
    return x.view(*x.shape[:-3], d) / math.sqrt(d)

# The key design space for structured spinners:
#
# Instance                     | M1       | M2              | M3         | GPU-friendly?
# ----------------------------|----------|-----------------|------------|---------------
# HD3 HD2 HD1 (SORF)          | HD1      | D2              | HD3        | YES (best)
# G_circ D2 HD1               | HD1      | D2              | G_circ     | OK (uses FFT)
# G_Toep D2 HD1               | HD1      | D2              | G_Toeplitz | OK (uses FFT)
# SHGΠHB (Fastfood)           | HB       | GΠ              | SH         | NO (permutation)
# Adaptive HD_learn HD2 HD1   | HD1      | D2              | HD_learn   | YES (learnable)
```

## References

- Bojarski, M., Choromanska, A., Choromanski, K., Fagan, F., Gouy-Pailler, C., Morvan, A., Sakr, N., Sarlós, T., & Atif, J. (2017). Structured Adaptive and Random Spinners for Fast Machine Learning Computations. AISTATS 2017. arXiv:1610.06209.
- Choromanski, K., Rowland, M., & Weller, A. (2017). The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings. NeurIPS 2017. arXiv:1703.00864.
- Le, Q. V., Sarlós, T., & Smola, A. J. (2013). Fastfood — Approximating Kernel Expansions in Loglinear Time. ICML 2013.
- Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D., & Kumar, S. (2016). Orthogonal Random Features. NeurIPS 2016.
- Ailon, N. & Chazelle, B. (2006). Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform. STOC 2006.
- Yang, Z., Moczulski, M., Denil, M., de Freitas, N., Smola, A., Song, L., & Wang, Z. (2015). Deep Fried Convnets. ICCV 2015.

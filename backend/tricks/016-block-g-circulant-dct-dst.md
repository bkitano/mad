# 016: Block g-Circulant Matrices with DCT-DST Multiplication

**Category**: decomposition
**Gain type**: expressivity
**Source**: Asriani, Muchtadi-Alamsyah & Purwarianti "On Block g-Circulant Matrices with Discrete Cosine and Sine Transforms for Transformer-Based Translation Machine" (Mathematics, MDPI, 2024)
**Paper**: [papers/block-g-circulant-dct-dst.pdf]
**Documented**: 2026-02-15

## Description

A **block $g$-circulant matrix** generalizes the standard block circulant matrix by introducing a **$g$-position cyclic shift** instead of a 1-position shift between consecutive block rows. Where a standard block circulant matrix has each block row as a single cyclic permutation of the row above, a $g$-circulant shifts by $g$ positions. This introduces richer interaction patterns within the same parameter budget, expanding the expressivity of the structured weight matrix beyond what standard circulant or block-circulant matrices can achieve.

The key algebraic insight is the **factorization via a permutation matrix**: a block $g$-circulant matrix $C_{nmg}$ can be written as $C_{nmg} = Z_{nmg} C_{nm}$, where $C_{nm}$ is a standard block circulant matrix ($g=1$) and $Z_{nmg} = Z_{n,g} \otimes Z_{m,g}$ is a Kronecker product of permutation matrices encoding the $g$-shift. This means:

1. The eigenvalue structure is a **twisted version** of standard block circulant eigenvalues — the eigenvalues of $C_{nmg}$ are the $s$-th roots of products of block circulant eigenvalues, where $s$ is the smallest positive integer such that $g^s \equiv 1 \pmod{nm}$.
2. The DCT-DST real-arithmetic multiplication algorithm extends from standard block circulant to block $g$-circulant via a **column permutation** of the orthogonal decomposition matrices.

When applied to transformer feedforward layers, the block $g$-circulant DCT-DST approach achieves 22.14% memory reduction over dense models with competitive or improved BLEU scores (26.47 vs 25.43 for dense at $d_{model}=128$), while the $g$ parameter provides a tunable knob trading expressivity against eigenvalue structure complexity.

## Mathematical Form

**Block $g$-Circulant Matrix:**

Let $C_i^{(p)}$ be $m \times m$ circulant blocks. A $nm \times nm$ block $g$-circulant matrix $C_{nmg}$ is generated from blocks $C_i^{(p)}$, $i = 1, \ldots, n$; $p = 1, \ldots, m$, with each row of blocks shifted $g$ positions to the right:

$$
C_{nmg} = Z_{nmg} C_{nm}
$$

where $C_{nm}$ is the standard block circulant ($g=1$) and $Z_{nmg}$ is the $g$-shift permutation.

**$g$-Shift Permutation Matrix:**

$$
Z_{nmg} = Z_{n,g} \otimes Z_{m,g}
$$

where:

$$
Z_{n,g} = [\delta_{(gr - s) \bmod n}]_{g,r=0}^{n-1}
$$

with $\delta_k = 1$ if $k \equiv 0 \pmod{n}$, else $0$.

**Diagonalization (Fourier Domain):**

$$
C_{nmg} = Z_{nmg}(F_n \otimes F_m) D_{nm} (F_n^* \otimes F_m^*)
$$

where $D_{nm}$ is the diagonal eigenvalue matrix and $F_n$ is the $n \times n$ DFT matrix.

**Eigenvalues ($g = 1$, standard block circulant):**

$$
\lambda_i^{(p)} = \sum_{k=1}^{n} \sum_{l=1}^{m} c_k^l \omega_{p-1}^{l-1} \omega_{i-1}^{k-1}
$$

with $i = 1, \ldots, n$ and $p = 1, \ldots, m$.

**Eigenvalue Modulus ($(nm, g) = 1$ case):**

When $nm$ and $g$ are coprime, let $s$ be the smallest positive integer with $g^s \equiv 1 \pmod{nm}$. Then:

$$
|\lambda_j^{(p)}(C_{nmg})| = \left| \sqrt[s]{\prod_{k=0}^{s-1} \lambda_{(g^k j) \bmod n}^{(g^k p) \bmod m}} \right|
$$

The eigenvalues of the $g$-circulant are $s$-th roots of products of standard circulant eigenvalues, traversing a $g$-orbit in index space.

**Real Orthogonal Decomposition (Schur Form):**

For real block $g$-circulant $C_{nmg} \in \mathbb{R}^{nm \times nm}$:

$$
C_{nmg} = U_{nmg} \Omega_{nmg} U_{nmg}^T
$$

where:

$$
U_{nmg} = (U_n \otimes U_m)(P_{n,n-g+1} \otimes P_{m,m-g+1})
$$

$U_n$ is the real Schur eigenvector matrix (constructed from real/imaginary parts of DFT columns), and $P_{t,t-g+1}$ is a permutation identity matrix that swaps columns $t$ and $t-g+1$.

**DCT-DST Factorization:**

The orthogonal matrix $Q_{nmg} = Q_n \otimes (U_m P_{m,m-g+1})$ enables multiplication via real-valued DCT and DST:

$$
Q_{nmg} U_{nmg} = (Q_n U_n P_{n,n-g+1}) \otimes (U_m P_{m,m-g+1})^2
$$

where:

$$
Q_n U_n = \begin{cases} \begin{bmatrix} \mathcal{C}_{h+1}^I & 0 \\ 0 & J_{h-1} \mathcal{S}_{h-1}^I J_{h-1} \end{bmatrix} & \text{if } n = 2h \\ \begin{bmatrix} \mathcal{C}_{h+1}^V & 0 \\ 0 & J_h \mathcal{S}_h^V J_h \end{bmatrix} & \text{if } n = 2h+1 \end{cases}
$$

$\mathcal{C}^I, \mathcal{C}^V$ are DCT Type I and V matrices; $\mathcal{S}^I, \mathcal{S}^V$ are DST Type I and V matrices.

**DCT-DST Multiplication Algorithm (8 steps):**

Given block $g$-circulant $C_{nmg}$ and input vector $x$:

1. Compute $v = Q_{nmg} c_1$ where $c_1 = C_{nmg} e_1$ (first column)
2. Compute $\hat{v} = (Q_{nmg} U_{nmg})^T v$ via DCT and DST
3. Form $\Omega_{nmg}$ (real Schur form, block-diagonal with $2 \times 2$ rotation blocks)
4. Compute $y_1 = Q_{nmg} x$
5. Compute $y_2 = (Q_{nmg} U_{nmg})^T y_1$ via DCT and DST
6. Compute $y_3 = \Omega_{nmg} y_2$ (block-diagonal multiply)
7. Compute $y_4 = (Q_{nmg} U_{nmg}) y_3$ via DCT and DST
8. Compute $C_{nmg} x = Q_{nmg}^T y_4$

**Key Definitions:**

- $g$ — the cyclic shift parameter (number of positions shifted per block row)
- $n$ — number of blocks; $m$ — block size; matrix is $nm \times nm$
- $(nm, g)$ — greatest common divisor of $nm$ and $g$
- $s$ — order of $g$ modulo $nm$: smallest $s > 0$ with $g^s \equiv 1 \pmod{nm}$
- $Z_{nmg}$ — $g$-shift permutation matrix (Kronecker product structure)
- $U_n, U_m$ — real Schur eigenvector matrices
- $Q_n, Q_m$ — DCT-DST factorization matrices
- $P_{t,t-g+1}$ — column permutation identity matrix
- $\mathcal{C}^I, \mathcal{S}^I$ — DCT/DST Type I matrices
- $\mathcal{C}^V, \mathcal{S}^V$ — DCT/DST Type V matrices

## Complexity

| Operation | Dense | Block Circulant ($g=1$) | Block $g$-Circulant |
|-----------|-------|------------------------|---------------------|
| Mat-vec multiply | $O(n^2 m^2)$ | $O(nm \log(nm))$ | $O(nm \log(nm))$ |
| Parameters | $O(n^2 m^2)$ | $O(nm)$ | $O(nm)$ |
| Arithmetic domain (DCT-DST) | Real | Real | Real |
| Eigenvalue computation | $O(n^3 m^3)$ | $O(nm \log(nm))$ | $O(s \cdot nm)$ where $g^s \equiv 1$ |

**Memory (Transformer FFN, from paper, $d_{model}=128$):**

| Model | $g$ | BLEU (%) | Memory (KB) |
|-------|-----|----------|-------------|
| Dense-Dense (A) | 0 | 25.43 | 18,394 |
| Dense-Block 1-Circ DCT-DST (B) | 1 | **26.47** | **14,322** |
| Dense-Block 2-Circ DCT-DST (C) | 2 | 21.69 | 14,340 |
| Dense-Block 3-Circ DCT-DST (D) | 3 | 24.12 | 14,340 |

The block 1-circulant ($g=1$) achieves the best BLEU (+1.04 over dense) with 22.14% memory reduction. Higher $g$ values show more varied performance.

**Parameter compression:** Same as standard block circulant — $nm$ parameters instead of $(nm)^2$, giving compression ratio of $nm$.

## Applicability

- **Transformer feedforward layers**: Direct replacement of dense $W_1, W_2$ matrices in FFN sublayers with block $g$-circulant weight matrices + DCT-DST multiplication. The paper demonstrates this on Portuguese-English translation
- **Expressivity beyond standard circulant**: The $g$ parameter provides an additional degree of freedom — different $g$ values create different interaction patterns (different eigenvalue orbits), potentially matching different data distributions better than fixed $g=1$
- **Low Displacement Rank (LDR) framework**: Block $g$-circulant matrices belong to the LDR matrix family, providing a principled structured matrix framework for neural network compression
- **Product of $g$-circulants**: By Lemma 9, if $C_{nmg}$ is a $g$-block circulant and $C_{nmh}$ is an $h$-block circulant, then $C_{nmg} C_{nmh}$ is a $gh$-block circulant. This closure under multiplication means stacking $g$-circulant layers preserves structure
- **Real-arithmetic hardware**: The DCT-DST algorithm avoids complex arithmetic, making it suitable for hardware without efficient complex FFT support (video codec accelerators, certain DSPs, FPGAs)

## Limitations

- **Same asymptotic complexity as $g=1$**: The $g$-shift adds no computational benefit over standard block circulant — both are $O(nm \log(nm))$. The advantage is purely in expressivity
- **Eigenvalue structure depends on $(nm, g)$**: When $\gcd(nm, g) \neq 1$, the matrix $C_{nmg}$ becomes singular (has $nm - n_g m_g$ zero eigenvalues), limiting expressivity. Best results require coprime $nm$ and $g$
- **Hyperparameter search over $g$**: The optimal $g$ value is task-dependent — the paper shows $g=1$ wins on this dataset, but $g=3$ outperforms $g=2$. No principled way to choose $g$ a priori
- **Slower test-time inference**: Despite theoretical efficiency, the DCT-DST algorithm's procedural complexity led to longer test durations in practice (27.8s vs 8.9s for dense at $d_{model}=128$)
- **Small-scale validation only**: Tested only on Portuguese-English translation with a small transformer (4 layers, 8 heads, ~52K training examples, $d_{model} \leq 256$)
- **Column permutation overhead**: The $P_{t,t-g+1}$ permutation matrices required for $g > 1$ add implementation complexity to the DCT-DST algorithm compared to the standard $g=1$ case

## Implementation Notes

```python
import torch
import torch.fft as fft
import math

class BlockGCirculantLinear(torch.nn.Module):
    """Block g-circulant linear layer with FFT-based multiplication.

    Generalizes block circulant by shifting each block row by g positions
    instead of 1, providing richer interaction patterns.

    For g=1: standard block circulant (CirCNN).
    For g=0: rank-1 block structure (degenerate).
    For g coprime to nm: full-rank with twisted eigenvalue structure.
    """

    def __init__(self, n_blocks, block_size, g=1):
        """
        Args:
            n_blocks: number of circulant blocks (n)
            block_size: size of each circulant block (m)
            g: shift parameter (positions shifted per block row)
        """
        super().__init__()
        self.n = n_blocks
        self.m = block_size
        self.g = g
        self.dim = n_blocks * block_size

        # Learnable: defining vectors for each circulant block
        # Same parameters as standard block circulant
        self.c_vectors = torch.nn.Parameter(
            torch.randn(n_blocks, block_size) / math.sqrt(self.dim)
        )

        # Build g-shift permutation (fixed, not learned)
        self.register_buffer(
            'g_perm', self._build_g_permutation(n_blocks, g)
        )

    @staticmethod
    def _build_g_permutation(n, g):
        """Build the g-shift permutation for indices 0..n-1.

        Z_{n,g}[r, s] = 1 iff (g*r - s) mod n == 0
        i.e., s = (g*r) mod n
        """
        indices = torch.arange(n)
        perm = (g * indices) % n
        return perm

    def forward(self, x):
        """
        x: (batch, dim) where dim = n * m

        The g-circulant multiply: C_{nmg} x = Z_{nmg} C_{nm} x
        Step 1: Apply standard block circulant (FFT-based)
        Step 2: Apply g-shift permutation
        """
        batch = x.shape[0]
        n, m = self.n, self.m

        # Reshape to blocks: (batch, n, m)
        x_blocks = x.view(batch, n, m)

        # Step 1: Standard block circulant multiply via FFT
        # FFT along block dimension
        x_fft_n = fft.fft(x_blocks, dim=1)  # (batch, n, m)
        # FFT along element dimension
        x_fft = fft.fft(x_fft_n, dim=2)  # (batch, n, m)

        # Eigenvalues via 2D FFT of defining vectors
        w_fft = fft.fft(fft.fft(self.c_vectors, dim=0), dim=1)  # (n, m)

        # Diagonal multiply
        y_fft = x_fft * w_fft.unsqueeze(0)

        # Inverse FFT
        y = fft.ifft(fft.ifft(y_fft, dim=2), dim=1).real  # (batch, n, m)

        # Step 2: Apply g-shift permutation to blocks
        # Permute block indices: block i -> block (g*i) mod n
        y_permuted = y[:, self.g_perm, :]

        return y_permuted.reshape(batch, -1)


# Example: comparing g=1 (standard) vs g=2 (twisted)
def demo_g_circulant():
    n, m, g = 8, 16, 2  # 8 blocks of size 16, g=2 shift
    dim = n * m  # 128

    layer_g1 = BlockGCirculantLinear(n, m, g=1)
    layer_g2 = BlockGCirculantLinear(n, m, g=2)

    x = torch.randn(4, dim)
    y1 = layer_g1(x)  # standard block circulant
    y2 = layer_g2(x)  # g=2 block circulant (different mixing)

    print(f"Input:  {x.shape}")
    print(f"g=1 output: {y1.shape}, norm={y1.norm():.4f}")
    print(f"g=2 output: {y2.shape}, norm={y2.norm():.4f}")
    # Same parameter count, different interaction patterns
```

## References

- Asriani, E., Muchtadi-Alamsyah, I. & Purwarianti, A. "On Block g-Circulant Matrices with Discrete Cosine and Sine Transforms for Transformer-Based Translation Machine" Mathematics 2024, 12, 1697. doi:10.3390/math12111697
- Asriani, E., Muchtadi-Alamsyah, I. & Purwarianti, A. "Real Block-Circulant Matrices and DCT-DST Algorithm for Transformer Neural Network" Front. Appl. Math. Stat. 2023, 9, 1260187
- Serra-Capizzano, S. & Debora, S. "A note on the eigenvalues of g-circulants (and of g-Toeplitz, g-Hankel matrices)" Calcolo 2014, 51, 639-659
- Pan, V. "Structured Matrices and Polynomials: Unified Superfast Algorithms" Springer, 2001
- Davis, P.J. "Circulant Matrices" Wiley, 1979
- Liu, Z., Chen, S., Xu, W. & Zhang, Y. "The eigen-structures of real (skew) circulant matrices with some applications" Comp. Appl. Math. 2019, 38, 1-13

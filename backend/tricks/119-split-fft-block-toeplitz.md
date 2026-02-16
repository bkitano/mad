# 119: Split FFT Block Toeplitz Multiplication

**Category**: decomposition
**Gain type**: efficiency
**Source**: Siron & Molesky "A Split Fast Fourier Transform Algorithm for Block Toeplitz Matrix-Vector Multiplication" (arXiv 2024)
**Paper**: [papers/split-fft-block-toeplitz.pdf]
**Documented**: 2026-02-15

## Description

The standard method for multiplying a $d$-level block Toeplitz matrix by a vector is to embed the Toeplitz matrix into a larger circulant matrix at each level, then use FFT + diagonal multiply + IFFT. The problem: each embedding level doubles the vector size, so a $d$-dimensional block Toeplitz structure inflates the working vector to $2^d$ times its original size. The padding zeros waste both memory and computation — for a 3D problem, $7/8$ of the embedded vector is padding.

The **Split FFT** algorithm ("lazy embedding, eager projection") improves on this by exploiting a property of the DFT: after an FFT, the even and odd frequency coefficients correspond to the original vector and a phase-shifted copy respectively. Instead of fully embedding at every level and then projecting at the end, the algorithm **interleaves embedding and projection** — it embeds lazily (only when needed) and projects eagerly (as soon as possible). This converts the single large circulant multiplication into a binary tree of smaller operations, each operating on vectors of the original block size $n$ rather than the inflated size $2n$.

The result is an asymptotic reduction in both computation and memory by a factor that grows with dimensionality $d$, making it especially valuable for multi-dimensional block Toeplitz problems (2D images, 3D volumes) that arise in convolution-based neural networks and spatial state-space models.

## Mathematical Form

**Standard Circulant Embedding (Baseline):**

For a 1D Toeplitz matrix $T \in \mathbb{R}^{n \times n}$, embed into circulant $C \in \mathbb{R}^{2n \times 2n}$:

$$
Tx = [I_n \quad 0] \cdot \mathcal{F}^{-1}(\mathcal{F}(c) \circ \mathcal{F}(\tilde{v}))
$$

where $\tilde{v} = [v; 0_n]$ is zero-padded and $c$ is the first column of $C$.

For $d$ levels of block Toeplitz structure, the fully embedded vector has size $2^d s$ where $s = n^d$ is the original vector size. Complexity:

$$
C_{\text{embed}} = \underbrace{2^{d+1} s \log_2(2^d s)}_{\text{FFTs}} + \underbrace{2^d s}_{\text{multiplication}}
$$

**Split FFT Key Property:**

Let $\mathbf{v}$ be a vector of size $n$, and define $\tilde{\mathbf{v}} = [\mathbf{v}; \mathbf{0}]$ (zero-padded to size $2n$). Let $P = \text{diag}(1, e^{i\pi/n}, e^{2i\pi/n}, \ldots, e^{i\pi(n-1)/n})$ be the diagonal phase shift operator. Then:

$$
\mathcal{F}(\tilde{\mathbf{v}}) = \mathcal{F}(\tilde{\mathbf{v}})_{\text{even}} + \mathcal{F}(\tilde{\mathbf{v}})_{\text{odd}}
$$

where the even and odd Fourier coefficients satisfy:

$$
\mathcal{F}(\tilde{\mathbf{v}})_{\text{even}} = \mathcal{F}(\mathbf{v}), \quad \mathcal{F}(\tilde{\mathbf{v}})_{\text{odd}} = \mathcal{F}(P\mathbf{v})
$$

This means the circulant embedding at each level can be replaced by a **branching operation**: split the transformed vector into an "original" branch and a "phase-modified child" branch, each retaining the original vector size $n$.

**Merging (Projection Counterpart):**

After diagonal multiplication, branches are merged in reverse order:

$$
\tilde{\mathbf{v}} = \mathcal{F}^{-1}\left(\frac{1}{2}\left(\mathcal{F}(\mathbf{v}_{\text{even}}) + \bar{P}\mathcal{F}(\mathbf{v}_{\text{odd}})\right)\right)
$$

where $\bar{P}$ is the conjugate of the phase shift operator.

**Recursive Algorithm (Algorithm 1):**

For a $d$-level block Toeplitz matrix $T$ and input vector $v$:
1. FFT along the $d$-th dimension
2. Create phase-shifted copy (splitting)
3. Recursively process two branches at level $d+1$
4. Merge branches (even coefficients from one, odd from other)
5. Inverse FFT along the $d$-th dimension
6. At the deepest level ($d = d_{\max}$), perform diagonal multiplication with Toeplitz data

**Operational Complexity of Split Algorithm:**

$$
C_{\text{split}} = \underbrace{2 \sum_{l=1}^{d} 2^l s \log_2(n)}_{\text{FFTs}} + \underbrace{2^d s}_{\text{multiplication}} + \underbrace{2 \sum_{l=0}^{d-1} 2^l s}_{\text{phase shift}}
$$

$$
= 2(2^d - 1)s(2\log_2(n) + 1) + 2^d s
$$

**Key Definitions:**

- $d$ — number of levels (dimensionality) of the block Toeplitz structure
- $s = n^d$ — total size of the original vector
- $n$ — block size at each level
- $P$ — diagonal phase shift operator, $P = \text{diag}(1, e^{i\pi k/n})_{k=0}^{n-1}$
- $\mathcal{F}$ — discrete Fourier transform

## Complexity

**Computational complexity ratio** (standard embedding / split):

$$
R_c = \frac{d \log_2(2n) + 1}{(1 - 2^{-d})(2\log_2(n) + 1) + 1}
$$

As $n \to \infty$: $R_c \to \frac{d}{2 - 2^{-d+1}}$

| Dimensions $d$ | Asymptotic Speedup $R_c$ | Peak Memory Ratio $R_m$ | Memory Ratio (Symmetric) $R_{m,\text{sym}}$ |
|---|---|---|---|
| 2 | 1.33× | 1.14× | 1.25× |
| 3 | 1.71× | 1.33× | 1.80× |
| 4 | 2.13× | 1.52× | 2.83× |
| 5 | 2.58× | 1.68× | 4.71× |
| 6 | 3.05× | 1.80× | 8.13× |

**Peak memory ratio** (general case):

$$
R_m = \frac{2}{(d+1)2^{-d} + 1}
$$

**Peak memory ratio** (symmetric/skew-symmetric case):

$$
R_{m,\text{sym}} = \frac{2^d + 1}{d + 2}
$$

**Memory:** Standard embedding requires $O(2^d s)$ peak allocation; Split FFT requires $O((d+1)s)$ peak allocation.

## Applicability

- **Multi-dimensional convolution**: 2D (image) and 3D (video/volume) convolutions that are represented as block Toeplitz matrix-vector products benefit from $1.3\text{--}3\times$ speedup and reduced memory
- **Toeplitz-based sequence models**: The Toeplitz Neural Network (TNN) and related architectures that compute token mixing via circulant embedding can use this for the inner FFT computation, reducing the $2\times$ overhead of the standard embedding
- **State-space models**: SSMs like S4 that compute convolution kernels via Toeplitz structure could benefit, especially multi-dimensional extensions (2D S4, etc.)
- **Spatial attention with translation invariance**: Any attention mechanism approximated by Toeplitz/circulant structure (e.g., BCCB circulant attention for vision) can use this to reduce the embedding overhead
- **Physics simulations in neural networks**: Neural operators (FNO, etc.) that solve PDEs with Green function convolutions naturally produce block Toeplitz structure
- **Parallelization of large systems**: The branching tree structure naturally maps to multi-device parallelism — each branch can be assigned to a different device

## Limitations

- **Asymptotic improvement only**: For 1D ($d=1$), the improvement is marginal. The benefit grows significantly only for $d \geq 2$
- **Constant factor improvement**: The speedup is a constant factor (not a change in big-O), so it matters most for large-scale problems where the constant matters
- **Implementation complexity**: The recursive branching structure is more complex to implement than the straightforward embed-FFT-multiply-IFFT pipeline
- **Phase shift overhead**: The additional phase shift operations ($P\mathbf{v}$) add computation that partially offsets the savings from avoiding full embedding
- **FFT library dependency**: Performance depends heavily on the FFT library's handling of different vector sizes; small vectors may not see the theoretical speedup due to FFT library internals and thread scheduling
- **Not applicable to non-Toeplitz structure**: Only helps when the matrix has genuine multi-level block Toeplitz structure

## Implementation Notes

```python
import torch
import torch.fft as fft
import math

def split_fft_block_toeplitz_mv(toeplitz_data, v, dim=0, d_max=None):
    """Block Toeplitz matrix-vector multiply using Split FFT.

    Lazy embedding, eager projection algorithm.
    For d-level block Toeplitz, avoids 2^d size inflation.

    Args:
        toeplitz_data: Fourier-domain representation of Toeplitz data
                       (pre-computed FFT of defining vectors at each level)
        v: input vector, shape (n^d,) or reshaped as (n, n, ..., n)
        dim: current dimension being processed (0 to d_max)
        d_max: maximum dimensionality of block Toeplitz structure
    """
    n = v.shape[dim]

    if d_max is None:
        d_max = v.ndim - 1

    # Step 1: FFT along current dimension
    v_fft = fft.fft(v, dim=dim)

    if dim < d_max:
        # Step 2: Create phase-shifted copy (splitting)
        # P = diag(1, e^{iπ/n}, e^{2iπ/n}, ..., e^{i(n-1)π/n})
        phase = torch.exp(1j * math.pi * torch.arange(n, device=v.device) / n)
        # Reshape for broadcasting along the correct dimension
        shape = [1] * v.ndim
        shape[dim] = n
        phase = phase.reshape(shape)

        v_child = v_fft * phase  # phase-shifted copy (odd coefficients)

        # Step 3: Recurse on both branches
        v_fft = split_fft_block_toeplitz_mv(
            toeplitz_data, v_fft, dim + 1, d_max
        )  # even branch
        v_child = split_fft_block_toeplitz_mv(
            toeplitz_data, v_child, dim + 1, d_max
        )  # odd branch

        # Step 4: Merge branches
        # v_merged = F^{-1}(1/2 * (F(v_even) + conj(P) * F(v_odd)))
        v_fft = 0.5 * (v_fft + phase.conj() * v_child)
    else:
        # Base case: diagonal multiplication with Toeplitz data
        v_fft = v_fft * toeplitz_data

    # Step 5: Inverse FFT along current dimension
    v_out = fft.ifft(v_fft, dim=dim)

    return v_out


# Example: 2D block Toeplitz (e.g., 2D convolution)
def demo_2d_split_fft():
    n = 64  # block size per dimension
    # Standard embedding: operates on 2n × 2n = 128 × 128 = 16384 elements
    # Split FFT: operates on n × n = 64 × 64 = 4096 elements per branch
    # Memory ratio R_m = 2 / ((2+1)*2^{-2} + 1) = 2/1.75 = 1.14x savings
    # Compute ratio R_c ≈ 1.33x speedup

    v = torch.randn(n, n, dtype=torch.complex64)
    toeplitz_fft = torch.randn(n, n, dtype=torch.complex64)  # precomputed

    result = split_fft_block_toeplitz_mv(toeplitz_fft, v, dim=0, d_max=1)
    return result
```

## References

- Siron, A. & Molesky, S. "A Split Fast Fourier Transform Algorithm for Block Toeplitz Matrix-Vector Multiplication" arXiv:2406.17981, 2024
- Lee, D. "Fast Multiplication of a Recursive Block Toeplitz Matrix by a Vector and its Application" Journal of Complexity, Vol. 2, pp. 295-305, 1986
- Ferreira, P.J.S.G. & Domínguez, M.E. "Trading-off matrix size and matrix structure: Handling Toeplitz equations by embedding on a larger circulant set" Digital Signal Processing, Vol. 20, pp. 1711-1722, 2010
- Qin, Z. et al. "Toeplitz Neural Network for Sequence Modeling" (ICLR 2023). arXiv:2305.04749
- Julia implementation: https://github.com/alsirc/SplitFFT_lazyEmbed

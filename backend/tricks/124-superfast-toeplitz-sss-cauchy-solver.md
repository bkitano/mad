# 124: Superfast Toeplitz Solver via SSS Cauchy Transformation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Chandrasekaran, Gu, Sun, Xia, Zhu — "A Superfast Algorithm for Toeplitz Systems of Linear Equations" (2007)
**Paper**: [papers/superfast-toeplitz-sss-cauchy.pdf]
**Documented**: 2026-02-15

## Description

This trick solves Toeplitz linear systems $Tx = b$ in superfast $O(N \log N + Np^2)$ time by chaining three structural transformations: (1) transform the Toeplitz matrix to a Cauchy-like matrix via FFT and displacement structure, (2) exploit the low-rank off-diagonal property of the Cauchy-like matrix to construct a compact sequentially semiseparable (SSS) representation, and (3) solve the SSS system in linear time using a compression-and-merge ULV-like procedure. The critical insight is that Cauchy-like matrices obtained from Toeplitz matrices have off-diagonal blocks with small numerical ranks (bounded by a parameter $p$ independent of $N$), and the compressions needed for SSS construction can be precomputed from a single Cauchy matrix $C_0$ whose entries are independent of the actual Toeplitz entries. This separation of structure from data makes the precomputation reusable across multiple right-hand sides and even across different Toeplitz matrices of the same size.

## Mathematical Form

**Stage 1 — Displacement Structure and Cauchy Transformation:**

A Toeplitz matrix $T \in \mathbb{R}^{N \times N}$ satisfies the Sylvester-type displacement equation:

$$
Z_1 \hat{C} - \hat{C} Z_{-1} = UV
$$

where $Z_\delta$ is the $\delta$-circulant shift matrix and the displacement rank $\alpha = \text{rank}(UV) \leq 2$.

Applying the FFT, the Toeplitz system $Tx = b$ is transformed to a Cauchy-like system $C\bar{x} = \bar{b}$:

$$
C = \mathcal{F} \hat{C} D_0^{-1} \mathcal{F}^H
$$

where $\mathcal{F} = \frac{1}{\sqrt{N}} \left(\omega^{2(k-1)(j-1)}\right)_{1 \leq k,j \leq N}$ is the normalized inverse DFT matrix, $\omega = e^{\frac{\pi i}{N}}$, and $D_0 = \text{diag}(1, \omega, \ldots, \omega^{N-1})$.

The Cauchy-like matrix has entries:

$$
C_{ij} = \frac{u_i^T v_j}{\omega^{2i} - \omega^{2j+1}}, \quad u_i, v_j \in \mathbb{R}^\alpha
$$

satisfying the transformed displacement equation:

$$
\mathcal{D}_1 C - C \mathcal{D}_{-1} = (\mathcal{F}U)(V D_0^H \mathcal{F}^H)
$$

with $\mathcal{D}_1 = \text{diag}(\omega^2, \omega^4, \ldots, \omega^{2N})$ and $\mathcal{D}_{-1} = \text{diag}(\omega^3, \omega^5, \ldots, \omega^{2N+1})$.

**Stage 2 — Low-Rank Property of Cauchy-like Matrices:**

The pure Cauchy matrix $C_0 = \left(\frac{1}{\omega^{2i} - \omega^{2j+1}}\right)$ has off-diagonal blocks with numerical rank at most $r+1$ where the approximation error satisfies:

$$
\frac{1}{\lambda_i - \eta_j} = \sum_{k=0}^{r} \frac{(\eta_j - c)^k}{(\lambda_i - c)^{k+1}} + O\left(\left(\frac{\eta_j - c}{\lambda_i - c}\right)^{r+1}\right)
$$

for well-separated point sets $\{\lambda_k\} = \{\omega^{2k}\}_{k=1}^N$ and $\{\eta_k\} = \{\omega^{2k+1}\}_{k=1}^N$.

Any off-diagonal block $\hat{G}$ of the Cauchy-like matrix $C$ can be factored as:

$$
\hat{G} \approx X Y^H
$$

where $X$ and $Y$ are tall-and-skinny matrices with at most $\alpha(r+1)$ columns — the rank of $\hat{G}$ is bounded by $\alpha$ times the rank of the corresponding block of $C_0$.

**Key Theorem (Precomputable Compressions):** The compression of any off-diagonal block of $C$ can be obtained from the compression of the corresponding block of $C_0$, with the column dimension of $X$ no larger than twice that of $X_0$. This means compressions are independent of the Toeplitz entries.

**Stage 3 — SSS Representation and Solver:**

A matrix $A \in \mathbb{C}^{M \times M}$ in SSS form with block partition $\{m_i\}_{i=1}^n$ satisfies:

$$
A_{ij} = \begin{cases} D_i & \text{if } i = j \\ U_i W_{i+1} \cdots W_{j-1} V_j^H & \text{if } i < j \\ P_i R_{i-1} \cdots R_{j+1} Q_j^H & \text{if } i > j \end{cases}
$$

where $\{U_i, V_i, W_i, P_i, Q_i, R_i, D_i\}$ are the SSS generators.

**SSS Solve Algorithm (Compression + Merging):**

The solver alternates two stages recursively:

*Compression:* Apply unitary transformation $q_1^H$ to $U_1$ to introduce zeros:

$$
q_1^H U_1 = \begin{pmatrix} 0 \\ \tilde{U}_1 \end{pmatrix} \quad \begin{matrix} m_1 - k_1 \\ k_1 \end{matrix}
$$

Then lower-triangularize $q_1^H D_1$ via another unitary $w_1^H$, solve for the first components, and reduce to a smaller SSS system.

*Merging:* When $k_i$ is no longer smaller than $m_i$, merge adjacent block rows/columns:

$$
\hat{D}_1 = \begin{pmatrix} D_1 & U_1 V_2^H \\ P_2 Q_1^H & D_2 \end{pmatrix}, \quad \hat{U}_1 = \begin{pmatrix} U_1 W_2 \\ U_2 \end{pmatrix}, \quad \hat{Q}_1 = \begin{pmatrix} Q_1 R_2^H \\ Q_2 \end{pmatrix}
$$

reducing the number of block rows by one.

**Key Definitions:**

- $T \in \mathbb{R}^{N \times N}$ — Toeplitz matrix with entries $T_{ij} = t_{i-j}$
- $C \in \mathbb{C}^{N \times N}$ — Cauchy-like matrix obtained via FFT transformation
- $\alpha \leq 2$ — displacement rank of the Toeplitz matrix
- $p$ — maximum numerical rank of off-diagonal blocks of $C$ (the SSS complexity parameter)
- $d$ — SSS block size (tunable parameter)
- $n = N/d$ — number of SSS blocks

## Complexity

| Operation | Gaussian Elimination | Superfast SSS Solver |
|-----------|---------------------|---------------------|
| Toeplitz solve | $O(N^3)$ | $O(N \log N + Np^2)$ |
| Fast methods (GKO, Heinig) | $O(N^2)$ | $O(N \log N + Np^2)$ |
| Precomputation (one-time) | — | $O(N^2 p)$ or $O(N \log N)$ |
| Per-solve after precomp | — | $O(Np^2)$ |
| Storage | $O(N^2)$ | $O(Np)$ |

**Memory:** $O(Np)$ total storage — linear in $N$.

**Time scaling factor:** Empirically $\approx 2$ when $N$ doubles (near-linear time).

**Precomputation:** Can be reduced from $O(N^2 p)$ to $O(N \log N)$ via the shifting strategy that relates compressions at different hierarchical levels.

## Applicability

- **Toeplitz/Hankel linear systems**: Direct application to convolution-based operations in signal processing and time series models
- **Structured state-space models (S4/S5)**: SSMs with Toeplitz-structured transition matrices can use this solver for fast parameter updates during training
- **Convolutional layers**: 1D convolution operations expressed as Toeplitz matrix-vector products; the SSS solver enables fast backpropagation through implicit Toeplitz systems
- **Linear attention with Toeplitz structure**: Relative position encodings that create Toeplitz attention patterns can leverage this for efficient exact attention computation
- **Kernel methods with shift-invariant kernels**: Gram matrices for shift-invariant kernels (Gaussian, Matérn) on regular grids are Toeplitz; this solver enables exact kernel regression at scale
- **Preconditioners for sequence models**: The fast approximate solve (with loose $\tau$) plus iterative refinement provides an effective preconditioner for structured linear systems in long-sequence models

## Limitations

- The parameter $p$ (off-diagonal numerical rank) grows logarithmically with $N$ and depends on the compression tolerance $\tau$; for very high accuracy, $p$ can become large
- The Toeplitz matrix need not be symmetric or positive definite, but moderately ill-conditioned matrices may require more iterative refinement steps
- The current implementation uses Fortran 90 with non-optimized SSS data structures; practical speedups require careful memory management
- The precomputation (one-time $O(N^2 p)$ or $O(N \log N)$ with shifting) is amortized only when solving multiple systems of the same size
- The SSS solver is practically stable but lacks a formal backward stability proof
- Block size $d$ must be tuned: too large increases solve time, too small increases precomputation time (optimal $d$ derivations exist in the literature)

## Implementation Notes

```python
# Pseudocode for superfast Toeplitz solver via SSS/Cauchy

import numpy as np
from numpy.fft import fft, ifft

def superfast_toeplitz_solve(t, b, tau=1e-9, block_size=100):
    """
    Solve Tx = b where T is Toeplitz defined by vector t.

    Args:
        t: Toeplitz vector of length 2N-1 (t[-(N-1)], ..., t[0], ..., t[N-1])
        b: right-hand side vector of length N
        tau: compression tolerance for SSS construction
        block_size: SSS block size d

    Returns:
        x: solution vector
    """
    N = len(b)
    omega = np.exp(1j * np.pi / N)

    # Stage 1: Transform Toeplitz to Cauchy-like system via FFT
    # T -> C = F * Chat * D0^{-1} * F^H
    # Transform right-hand side: b_bar = F * D0^{-1} * ... * b
    C_hat = toeplitz_to_displacement(t)  # displacement representation
    b_bar = fft_transform_rhs(b, omega, N)

    # Stage 2: Build compact SSS representation of C
    # Key insight: precompute compressions from C_0 (independent of t!)
    # C_0[i,j] = 1 / (omega^{2i} - omega^{2j+1})

    # 2a. Precompute off-diagonal block compressions of C_0
    #     using divide-and-conquer with shifting strategy
    compressions = precompute_cauchy_compressions(omega, N, tau, block_size)

    # 2b. Construct SSS generators {U_i, V_i, W_i, P_i, Q_i, R_i, D_i}
    #     from precomputed compressions + actual Toeplitz data
    sss = build_sss_from_compressions(C_hat, compressions, block_size)

    # Stage 3: Solve SSS system in O(Np^2) time
    # Alternating compression and merging phases
    x_bar = sss_solve(sss, b_bar)

    # Stage 4: Recover Toeplitz solution via inverse FFT
    x = ifft_recover_solution(x_bar, omega, N)

    # Optional: iterative refinement for higher accuracy
    for _ in range(num_refinement_steps):
        r = b - toeplitz_matvec(t, x)
        dx = superfast_toeplitz_solve(t, r, tau, block_size)  # reuse precomp
        x = x + dx

    return x

def sss_solve(sss, b):
    """
    Solve SSS system Ax = b via compression + merging.
    Cost: O(Np^2) where p = max off-diagonal rank.
    """
    n = sss.num_blocks
    # Forward pass: compress from block 1 to n
    # At each step: apply unitary to introduce zeros in U_i,
    #   lower-triangularize D_i, solve for partial solution,
    #   update auxiliary variable xi
    # When k_i >= m_i: merge adjacent blocks
    # Backward pass: recover full solution from partial solutions
    pass
```

## References

- Chandrasekaran, Gu, Sun, Xia, Zhu, "A Superfast Algorithm for Toeplitz Systems of Linear Equations," SIAM J. Matrix Anal. Appl. 29(4), pp. 1247–1266, 2007
- Chandrasekaran, Dewilde, Gu, Pals, Sun, van der Veen, White, "Some fast algorithms for sequentially semiseparable representations," SIAM J. Matrix Anal. Appl. 27(2), pp. 341–364, 2005
- Martinsson, Rokhlin, Tygert, "A fast algorithm for the inversion of general Toeplitz matrices," Comput. Math. Appl. 50, pp. 741–752, 2005
- Gohberg, Kailath, Olshevsky, "Fast Gaussian elimination with partial pivoting for matrices with displacement structure," Math. Comp. 64, pp. 1557–1576, 1995

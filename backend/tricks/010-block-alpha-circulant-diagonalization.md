# 010: Block α-Circulant Diagonalization for Parallel-in-Time Computation

**Category**: parallelization
**Gain type**: efficiency
**Source**: McDonald, Pestana & Wathen "Preconditioning and iterative solution of all-at-once systems for evolutionary PDEs" (SIAM J. Sci. Comput., 2018); Lin & Hon "A block α-circulant based preconditioned MINRES method for wave equations" (arXiv:2306.03574, 2024)
**Paper**: [papers/block-alpha-circulant-preconditioner.pdf], [papers/alpha-circulant-diagonalization.pdf]
**Documented**: 2026-02-15

## Description

A **block α-circulant matrix** is a block Toeplitz matrix where the wrap-around (top-right) blocks are scaled by a parameter $\alpha \in (0,1)$, rather than being equal to the subdiagonal blocks (which would give a standard block circulant, $\alpha = 1$). This seemingly small modification has a profound consequence: the block α-circulant matrix is **diagonalizable via a scaled DFT**, where the scaling matrix $D_\alpha = \text{diag}(\alpha^{0/n}, \alpha^{1/n}, \ldots, \alpha^{(n-1)/n})$ absorbs the $\alpha$-scaling into the Fourier basis. This enables $O(n \log n)$ application of the preconditioner via FFT, with all $n$ diagonal blocks solvable **independently in parallel**.

The key insight for neural networks is that **sequential recurrences** (which arise in autoregressive models, RNNs, state-space models, and causal attention) produce block lower-triangular Toeplitz systems. The block α-circulant approximation converts this inherently sequential structure into a **parallel-in-time** one: instead of processing $n$ time steps sequentially, all time steps can be processed simultaneously after an FFT in the time dimension. The parameter $\alpha$ controls the approximation quality — as $\alpha \to 0$, the α-circulant approaches the original triangular Toeplitz, but at the cost of worse conditioning; the optimal $\alpha$ balances approximation accuracy against numerical stability.

This trick has been extensively applied to parallel-in-time (PinT) solvers for PDEs, where it enables solving all time steps simultaneously with mesh-independent convergence rates. The same principle applies to any sequential computation that can be cast as a block Toeplitz system — including SSM recurrences and autoregressive token-by-token generation.

## Mathematical Form

**Block Lower-Triangular Toeplitz System:**

An all-at-once system from a sequential recurrence with $n$ time steps and $m$-dimensional state:

$$
\mathcal{T} = \begin{bmatrix} L & & & \\ -2I_m & L & & \\ I_m & -2I_m & L & \\ & \ddots & \ddots & \ddots \\ & & I_m & -2I_m & L \end{bmatrix} \in \mathbb{R}^{mn \times mn}
$$

where $L = I_m - \frac{\tau^2}{2}\Delta_h$ encodes the spatial operator and $\tau$ is the time step. More generally, any block Toeplitz system $\mathcal{T}$ with blocks $A_0, A_1, \ldots$ on successive subdiagonals.

**Block α-Circulant Preconditioner:**

$$
\mathcal{C}_\alpha = \begin{bmatrix} A_0 & & & \alpha A_1 & -2\alpha I_m \\ -2I_m & A_0 & & & \alpha A_1 \\ A_1 & -2I_m & A_0 & & \\ & \ddots & \ddots & \ddots & \\ & & A_1 & -2I_m & A_0 \end{bmatrix}
$$

The wrap-around blocks in the top-right corner are scaled by $\alpha$. When $\alpha = 1$, this is a standard block circulant; when $\alpha = 0$, it equals $\mathcal{T}$ exactly (lower-triangular).

**Kronecker Decomposition:**

$$
\mathcal{C}_\alpha = B_1^{(\alpha)} \otimes L + B_2^{(\alpha)} \otimes (-2I_m)
$$

where:

$$
B_1^{(\alpha)} = \begin{bmatrix} 1 & & \alpha \\ 0 & 1 & \\ 1 & 0 & 1 \\ & \ddots & \ddots & \ddots \\ & & 1 & 0 & 1 \end{bmatrix}, \quad B_2^{(\alpha)} = \begin{bmatrix} 0 & & & \alpha \\ 1 & 0 & & \\ & 1 & 0 & \\ & & \ddots & \ddots \\ & & & 1 & 0 \end{bmatrix}
$$

**Scaled DFT Diagonalization:**

Both $B_1^{(\alpha)}$ and $B_2^{(\alpha)}$ are diagonalizable via a scaled Fourier basis:

$$
B_j^{(\alpha)} = D_\alpha^{-1} \mathbb{F} \Lambda_{\alpha,j} \mathbb{F}^* D_\alpha, \quad j = 1, 2
$$

where:

$$
D_\alpha = \text{diag}\left(\alpha^{(i-1)/n}\right)_{i=1}^{n}, \quad \mathbb{F} = \frac{1}{\sqrt{n}} \left[\theta_n^{(i-1)(j-1)}\right]_{i,j=1}^{n}, \quad \theta_n = \exp\left(\frac{2\pi \mathbf{i}}{n}\right)
$$

The eigenvalues are:

$$
\lambda_{1,k}^{(\alpha)} = 1 + \alpha^{2/n} \theta_n^{2(k-1)}, \quad \lambda_{2,k}^{(\alpha)} = -2\alpha^{1/n} \theta_n^{k-1}
$$

**Full Block Diagonalization:**

$$
\mathcal{C}_\alpha = \left[(D_\alpha^{-1} \mathbb{F}) \otimes I_m\right] \left[\Lambda_{\alpha,1} \otimes L + \Lambda_{\alpha,2} \otimes (-2I_m)\right] \left[(\mathbb{F}^* D_\alpha) \otimes I_m\right]
$$

Since $\Lambda_{\alpha,1} \otimes L + \Lambda_{\alpha,2} \otimes (-2I_m)$ is block diagonal with $n$ blocks of size $m \times m$:

$$
\text{Block } k: \quad \lambda_{1,k}^{(\alpha)} L - 2\lambda_{2,k}^{(\alpha)} I_m \in \mathbb{C}^{m \times m}
$$

**Preconditioner Application (3 steps, all parallelizable):**

Given vector $\mathbf{u} = [\mathbf{u}_1, \ldots, \mathbf{u}_n]^T$ with $\mathbf{u}_k \in \mathbb{R}^m$:

1. **Scaled FFT**: $\hat{\mathbf{u}} = (\mathbb{F}^* D_\alpha \otimes I_m) \mathbf{u}$ — scale each block by $\alpha^{(k-1)/n}$, then FFT across blocks
2. **Block-diagonal solve**: For each $k = 1, \ldots, n$ independently (in parallel):
   $$\hat{\mathbf{v}}_k = \left(\lambda_{1,k}^{(\alpha)} L - 2\lambda_{2,k}^{(\alpha)} I_m\right)^{-1} \hat{\mathbf{u}}_k$$
3. **Scaled IFFT**: $\mathbf{v} = (D_\alpha^{-1} \mathbb{F} \otimes I_m) \hat{\mathbf{v}}$ — IFFT across blocks, then unscale

**Absolute Value Block α-Circulant (ABAC) Preconditioner (Lin & Hon 2024):**

Since $\mathcal{C}_\alpha$ is not symmetric, it cannot directly precondition symmetric solvers (MINRES). The HPD preconditioner is constructed via a matrix square root:

$$
\mathcal{P}_\alpha = (\mathcal{C}_\alpha^{1/2})^T \mathcal{C}_\alpha^{1/2}
$$

where $\mathcal{C}_\alpha^{1/2} = \mathbf{X}^{-1} \text{diag}(\sqrt{\mu_i})_{i=1}^{nm} \mathbf{X}$ is the principal matrix square root (well-defined since $\mathcal{C}_\alpha \in \mathcal{Q}(mn)$, i.e., all eigenvalues avoid $(-\infty, 0]$).

The preconditioned matrix $\mathcal{P}_\alpha^{-1} \mathcal{Y}\mathcal{T}$ has eigenvalues clustered around $\pm 1$, giving **mesh-size-independent convergence** of MINRES.

**Key Definitions:**

- $\alpha \in (0, 1)$ — scaling parameter for the circulant wrap-around
- $n$ — number of time steps (or sequence positions)
- $m$ — spatial dimension (or hidden dimension per position)
- $L \in \mathbb{R}^{m \times m}$ — spatial operator block (HPD)
- $D_\alpha$ — diagonal scaling matrix, $D_\alpha = \text{diag}(\alpha^{(i-1)/n})$
- $\mathbb{F}$ — $n \times n$ DFT matrix
- $\theta_n = e^{2\pi \mathbf{i}/n}$ — primitive $n$-th root of unity
- $\mathcal{Q}(K)$ — set of $K \times K$ matrices with spectrum in $\mathbb{C} \setminus (-\infty, 0]$

## Complexity

| Operation | Sequential solve | Block circulant ($\alpha=1$) | Block α-circulant |
|-----------|-----------------|-------|------------|
| Total work | $O(n \cdot m^2)$ sequential | $O(nm \log n + nm^2)$ | $O(nm \log n + nm^2)$ |
| Parallel depth | $O(n)$ | $O(\log n + m^2)$ | $O(\log n + m^2)$ |
| Parallelism | 1 (inherently sequential) | $n$ blocks | $n$ blocks |
| Approximation quality | Exact | Poor (large wrap-around error) | Tunable via $\alpha$ |

**Breakdown of $O(nm \log n + nm^2)$:**
- Scaled FFT across $n$ blocks: $O(nm \log n)$ — $m$ independent FFTs of length $n$
- $n$ independent $m \times m$ block solves: $O(nm^2)$ — fully parallel across blocks
- Scaled IFFT: $O(nm \log n)$

**Convergence (from Lin & Hon 2024):**

For MINRES with the ABAC preconditioner $\mathcal{P}_\alpha$, the convergence rate is:

$$
\|\mathbf{r}_k\|_2 \leq 2 \left(\frac{\sqrt{a_1 a_4} - \sqrt{a_2 a_3}}{\sqrt{a_1 a_4} + \sqrt{a_2 a_3}}\right)^{\lfloor k/2 \rfloor} \|\mathbf{r}_0\|_2
$$

where the spectral bounds $a_1, a_2, a_3, a_4$ are **independent of mesh sizes** $h$ and $\tau$, giving linear convergence independent of problem size.

**Parallel scaling (from Goddard & Wathen 2018):**

| Processes | $n=320, \ell=768$ | $n=768, \ell=1024$ | $n=1440, \ell=1568$ |
|-----------|---|---|---|
| 1 | 77.7s | 459.1s | 2119.9s |
| 8 | 9.0s | 55.9s | 218.2s |
| 32 | 3.3s | 16.1s | 63.3s |
| Speedup (32 cores) | 23.5× | 28.5× | 33.5× |

**Memory:** $O(nm)$ — same as storing the $n$ block-sized vectors, plus $O(m^2)$ for the spatial operator factorization.

## Applicability

- **Parallel decoding in autoregressive models**: The sequential token-by-token generation in LLMs can be cast as a block lower-triangular Toeplitz system (when the transition is position-invariant). The α-circulant preconditioner enables speculative parallel generation of all tokens simultaneously, with iterative refinement to match the true sequential output
- **State-space model (SSM) parallelization**: SSM recurrences $x_k = Ax_{k-1} + Bu_k$ produce block lower-triangular Toeplitz systems. The α-circulant approach provides an alternative to the scan-based parallelization (Blelloch scan), trading exact computation for approximate parallel solves with tunable accuracy
- **Parallel-in-time training**: For sequence models trained with BPTT, the backward pass through time is a sequential block Toeplitz solve. The α-circulant preconditioner enables parallel gradient computation across time steps
- **PDE neural operators**: Neural operators that solve time-dependent PDEs (e.g., Fourier Neural Operator applied to time series) can use this for parallel-in-time inference
- **Causal token mixing with parallelism**: Causal (lower-triangular) Toeplitz token mixers (as in TNN with causal masking) can be approximately parallelized via α-circulant preconditioning, converting the sequential causal structure into a parallel circulant one

## Limitations

- **Approximation, not exact**: The α-circulant is an approximation to the original triangular Toeplitz. As $\alpha \to 0$, accuracy improves but the condition number of $\mathcal{C}_\alpha$ grows (eigenvalues approach zero), limiting numerical stability
- **Requires iterative refinement**: A single application of the α-circulant preconditioner does not give the exact answer — it must be used within an iterative solver (GMRES, MINRES, CG), typically requiring $O(1)$ to $O(\log n)$ iterations
- **Complex arithmetic**: The scaled DFT involves complex numbers even when the original system is real. The α-scaling $D_\alpha$ has irrational entries, requiring floating-point computation
- **Optimal α is problem-dependent**: The best $\alpha$ depends on the spectral properties of the spatial operator $L$. Lin & Hon (2024) provide theoretical guidance, but in practice $\alpha$ may need tuning
- **Block structure required**: The system must have block Toeplitz structure — non-uniform (content-dependent) transitions, as in standard Transformer attention, do not have this structure. Only position-invariant (Toeplitz) operations benefit
- **Not better than scan for single forward pass**: For a single forward evaluation of an SSM, the parallel scan (Blelloch) is exact and has $O(n \log n)$ work. The α-circulant approach is more naturally suited to iterative solving (e.g., Newton steps, implicit time-stepping) rather than single-pass evaluation

## Implementation Notes

```python
import torch
import torch.fft as fft
import math

class BlockAlphaCirculantPreconditioner:
    """Block α-circulant preconditioner for parallel-in-time computation.

    Converts a sequential block lower-triangular Toeplitz system
    into a parallel computation via scaled FFT + independent block solves.

    Args:
        L: (m, m) spatial operator block (HPD)
        n: number of time steps / sequence positions
        alpha: scaling parameter in (0, 1), controls approximation quality
    """

    def __init__(self, L, n, alpha=0.5):
        self.m = L.shape[0]
        self.n = n
        self.alpha = alpha

        # Precompute scaling factors: D_alpha = diag(alpha^{(i-1)/n})
        k = torch.arange(n, dtype=torch.float64)
        self.d_alpha = alpha ** (k / n)  # (n,)

        # Precompute eigenvalues of B_1^(alpha) and B_2^(alpha)
        theta = torch.exp(2j * math.pi * k / n)  # n-th roots of unity
        lam1 = 1 + alpha ** (2.0 / n) * theta ** 2  # (n,) complex
        lam2 = -2 * alpha ** (1.0 / n) * theta       # (n,) complex

        # Precompute and factorize the n diagonal blocks
        # Block k: lam1[k] * L + lam2[k] * I_m
        L_complex = L.to(torch.complex128)
        I_m = torch.eye(self.m, dtype=torch.complex128)

        # Store LU factorizations of all n blocks (can be done in parallel)
        self.block_factors = []
        for i in range(n):
            block = lam1[i] * L_complex + lam2[i] * I_m
            self.block_factors.append(torch.linalg.lu_factor(block))

    def apply(self, u):
        """Apply C_alpha^{-1} to vector u.

        u: (n, m) — n blocks of size m

        Returns: (n, m) — C_alpha^{-1} @ u
        """
        n, m = self.n, self.m

        # Step 1: Scale by D_alpha
        u_scaled = u * self.d_alpha.unsqueeze(1)  # (n, m)

        # Step 2: FFT across the time/block dimension
        u_hat = fft.fft(u_scaled.to(torch.complex128), dim=0)  # (n, m)

        # Step 3: Solve n independent m×m systems (fully parallel)
        v_hat = torch.zeros_like(u_hat)
        for i in range(n):
            v_hat[i] = torch.linalg.lu_solve(
                *self.block_factors[i], u_hat[i].unsqueeze(1)
            ).squeeze(1)

        # Step 4: IFFT across time dimension
        v_scaled = fft.ifft(v_hat, dim=0)  # (n, m)

        # Step 5: Unscale by D_alpha^{-1}
        v = v_scaled / self.d_alpha.unsqueeze(1)

        return v.real.float()


def parallel_in_time_solve(L, rhs, alpha=0.5, max_iter=10, tol=1e-6):
    """Solve a block lower-triangular Toeplitz system in parallel.

    The system arises from sequential recurrence:
    L u_k - 2 u_{k-1} + u_{k-2} = f_k

    Uses block α-circulant preconditioned GMRES.

    Args:
        L: (m, m) spatial operator
        rhs: (n, m) right-hand side
        alpha: circulant parameter
        max_iter: maximum GMRES iterations
        tol: convergence tolerance

    Returns:
        u: (n, m) solution
    """
    n, m = rhs.shape
    precond = BlockAlphaCirculantPreconditioner(L, n, alpha)

    # Simple preconditioned Richardson iteration for demonstration
    u = torch.zeros_like(rhs)
    for iteration in range(max_iter):
        # Compute residual: r = rhs - T @ u
        r = rhs.clone()
        r[0] -= L @ u[0]
        for k in range(1, n):
            r[k] -= L @ u[k] - 2 * u[k-1]
            if k >= 2:
                r[k] -= u[k-2]

        rel_res = r.norm() / rhs.norm()
        if rel_res < tol:
            break

        # Apply preconditioner: correction = C_alpha^{-1} @ r
        correction = precond.apply(r)
        u = u + correction

    return u
```

## References

- McDonald, E., Pestana, J. & Wathen, A.J. "Preconditioning and iterative solution of all-at-once systems for evolutionary partial differential equations" SIAM J. Sci. Comput. 40(2):A1012-A1033, 2018
- Lin, X.-l. & Hon, S. "A block α-circulant based preconditioned MINRES method for wave equations" Appl. Numer. Math., 2024. arXiv:2306.03574
- Goddard, A.J. & Wathen, A.J. "A Note on Parallel Preconditioning for All-at-Once Evolutionary PDEs" Electron. Trans. Numer. Anal., 2018. arXiv:1810.00615
- Gander, M.J. "50 Years of Time Parallel Time Integration" in Multiple Shooting and Time Domain Decomposition Methods, Springer, 2015
- Liu, J. & Wu, S.-L. "A Fast Block α-Circulant Preconditioner for All-at-Once Systems From Wave Equations" SIAM J. Matrix Anal. Appl. 41(4):1912-1943, 2020

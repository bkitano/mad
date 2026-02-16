# 174: FFT-Based GPU Block-Triangular Toeplitz Matvec

**Category**: kernel
**Gain type**: efficiency
**Source**: Venkat, Fernando, Henneking & Ghattas "Fast And Scalable FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices" (arXiv:2407.13066, 2024)
**Paper**: [papers/fft-block-triangular-toeplitz-gpu.pdf]
**Documented**: 2026-02-15

## Description

A block lower-triangular Toeplitz (BLTT) matrix arises whenever a linear time-invariant (LTI) system is discretized in time: the parameter-to-observable map $\mathbf{F}$ has blocks $F_{ij} = \mathbf{B}\mathbf{A}^{i-j}\mathbf{C}$ that depend only on the time-step difference $i - j$. The standard approach to multiplying $\mathbf{F}$ by a vector is to embed it inside a block circulant matrix and apply the FFT. This paper provides a highly optimized, multi-GPU implementation of this idea that achieves **>80% of peak memory bandwidth** on NVIDIA A100 GPUs, with excellent weak scaling to 48 GPUs.

The key insight for neural networks is that **any shift-invariant sequential computation** — including SSM recurrences ($x_k = Ax_{k-1} + Bu_k$, $y_k = Cx_k$), causal Toeplitz token mixing (TNN), and position-invariant causal attention — produces a block lower-triangular Toeplitz matrix-vector product when viewed as an all-at-once operation. This paper shows how to execute such products at near-peak GPU efficiency by:

1. **Reindexing** from time-outer-space-inner (TOSI) to space-outer-time-inner (SOTI) ordering, converting strided access into contiguous access
2. **Batched FFT + Strided-Batched GEMV (SBGEMV)** in Fourier space, where the block-diagonal structure of the circulant enables cuBLAS `gemvStridedBatched` at high throughput
3. **Free adjoint matvec**: Since the FFT is unitary, $\mathbf{F}^*$ matvec requires only a complex conjugate transpose before applying the same algorithm — no separate storage or setup needed
4. **Communication-aware partitioning** on multi-GPU clusters via an optimal 2D processor grid

The result is a $\sim 1000\times$ FLOP reduction over matrix-free forward/adjoint PDE solves for Hessian actions, and $>750\times$ measured efficiency gain for practical inverse problems. The algorithm achieves $O(N_m N_d N_t \log N_t)$ complexity vs $O(N_m N_d N_t^2 / 2)$ for dense matvec.

## Mathematical Form

**Block Lower-Triangular Toeplitz System:**

A discretized LTI system with $N_t$ time steps, $N_m$-dimensional parameters, and $N_d$-dimensional observations produces:

$$
\mathbf{d} = \mathbf{F}\mathbf{m}
$$

where $\mathbf{F} \in \mathbb{R}^{N_d N_t \times N_m N_t}$ is block lower-triangular Toeplitz:

$$
\mathbf{F} = \begin{bmatrix} F_{11} & 0 & \cdots & 0 \\ F_{21} & F_{11} & \cdots & 0 \\ F_{31} & F_{21} & F_{11} & \ddots \\ \vdots & \vdots & \ddots & \ddots \\ F_{N_t,1} & F_{N_t-1,1} & \cdots & F_{11} \end{bmatrix}
$$

with blocks $F_{ij} = \mathbf{B}\mathbf{A}^{i-j-1}\mathbf{C} \in \mathbb{R}^{N_d \times N_m}$.

**Circulant Embedding:**

Embed the Toeplitz matrix $\mathbf{M}_{\text{toep}} \in \mathbb{R}^{n \times n}$ (with entries $m_0, \ldots, m_{n-1}$) into the circulant matrix $\mathbf{M}_{\text{circ}} \in \mathbb{R}^{2n \times 2n}$:

$$
\mathbf{M}_{\text{circ}} = \begin{bmatrix} \mathbf{M}_{\text{toep}} & \mathbf{M}' \\ \mathbf{M}' & \mathbf{M}_{\text{toep}} \end{bmatrix}
$$

Since the DFT matrix $\mathbf{D}$ diagonalizes $\mathbf{M}_{\text{circ}}$:

$$
\mathbf{M}_{\text{circ}} = (2n)^{-1/2} \mathbf{D}^{-1} \text{diag}\left(\widehat{(\mathbf{M}_{\text{circ}})_{0:}}\right) \mathbf{D}
$$

For the matvec $\mathbf{M}_{\text{toep}}\mathbf{x}$, set $\mathbf{u} = [\mathbf{x}; \mathbf{0}]^T \in \mathbb{R}^{2n}$:

$$
\mathbf{M}_{\text{circ}}\mathbf{u} = \text{IFFT}\left(\frac{1}{\sqrt{2n}}\widehat{(\mathbf{M}_{\text{circ}})_{0:}} \odot \hat{\mathbf{u}}\right)
$$

and $\mathbf{M}_{\text{toep}}\mathbf{x}$ is the first $n$ entries of the result.

**Block Toeplitz Extension (SOTI ordering):**

In SOTI ordering, $\tilde{\mathbf{F}}$ has $N_d \times N_m$ blocks, each of which is lower-triangular Toeplitz of size $N_t \times N_t$. After circulant embedding and FFT of each block's first column:

$$
\hat{\mathbf{F}} \text{ is block-diagonal with } N_d \times N_m \text{ diagonal blocks, each of size } N_t \times N_t
$$

In TOSI ordering, $\hat{\mathbf{F}}$ consists of $N_t$ diagonal blocks of size $N_d \times N_m$. The local matvec reduces to a **Strided-Batched GEMV (SBGEMV)**:

$$
\hat{\mathbf{d}}_k = \hat{\mathbf{F}}_k \hat{\mathbf{m}}_k, \quad k = 1, \ldots, 2N_t
$$

where each $\hat{\mathbf{F}}_k \in \mathbb{C}^{N_d \times N_m}$ and $\hat{\mathbf{m}}_k \in \mathbb{C}^{N_m}$.

**Adjoint Matvec (Free):**

Since the DFT is unitary ($\mathbf{D}^{-1} = \mathbf{D}^*$):

$$
\mathbf{M}_{\text{circ}}^* = \frac{1}{\sqrt{2n}} \mathbf{D}^{-1} \text{diag}\left(\widehat{(\mathbf{M}_{\text{circ}})_{0:}}\right)^* \mathbf{D}
$$

The adjoint matvec uses the same algorithm with only a complex conjugate transpose of $\hat{\mathbf{F}}$ before the SBGEMV — no extra storage or setup required.

**Algorithm Steps (per GPU):**

1. **Broadcast** input vector to all processor rows/columns
2. **Pad** each block's local vector with zeros (size $N_t \to 2N_t$)
3. **Batched FFT** via cuFFT (along time dimension)
4. **SOTI-to-TOSI** reindex (swapaxes kernel)
5. **SBGEMV**: Apply block-diagonal matrix via `cgemvStridedBatched`
6. **TOSI-to-SOTI** reindex
7. **Batched IFFT** + unpad (extract first $N_t$ entries)
8. **Reduce** across processor row/column via NCCL

**Key Definitions:**

- $N_t$ — number of time steps (sequence length)
- $N_m$ — spatial parameter dimension (source/hidden dimension)
- $N_d$ — spatial data dimension (observation/output dimension)
- $\mathbf{F}$ — block lower-triangular Toeplitz p2o map
- $\hat{\mathbf{F}}$ — Fourier-domain block-diagonal representation
- TOSI — time-outer-space-inner ordering
- SOTI — space-outer-time-inner ordering

## Complexity

| Operation | Dense BLTT matvec | FFT-based matvec | Speedup |
|-----------|------------------|------------------|---------|
| FLOPs | $O(N_m N_d N_t^2 / 2)$ | $O(N_m N_d N_t \log N_t)$ | $O(N_t / \log N_t)$ |
| Storage (matrix) | $O(N_m N_d N_t^2 / 2)$ | $O(N_m N_d N_t)$ | $O(N_t)$ |
| Per Hessian matvec | 2 PDE solves: $O(N_t \cdot 324 N_u)$ | $8 N_m N_d N_t$ FLOPs | $\sim 1000\times$ |

**Per-GPU cost breakdown (Table 2 from paper):**

| Step | Cost | Notes |
|------|------|-------|
| FFT Matrix (setup) | $O(2n_d n_m N_t \log 2N_t)$ | One-time |
| Broadcast vector | $O((\ell + 8N_t n_m/\beta) \log c)$ | Latency + bandwidth |
| Pad vector | $O(2n_m N_t)$ | |
| FFT vector | $O(2n_m N_t \log 2N_t)$ | |
| SBGEMV | $O(n_d n_m (N_t + 1))$ | Complex arithmetic, dominant cost |
| IFFT | $O(2n_d N_t \log 2N_t)$ | |
| Unpad | $O(2n_d N_t)$ | |
| Reduce | $O((\ell + 8N_t n_d/\beta) \log c)$ | NCCL allreduce |

where $n_d = N_d/r$, $n_m = N_m/c$ are per-GPU local sizes on an $r \times c$ processor grid.

**Roofline analysis:** All major kernels achieve 75–85% of peak memory bandwidth on A100. The SBGEMV has arithmetic intensity $\approx 0.5$ FLOPs/byte (memory-bound), matching theoretical predictions.

**Scaling results (48 A100 GPUs):**
- Weak scaling efficiency: 85–87%
- Strong scaling: ~8× speedup on 48 GPUs (limited by communication)
- Overall speedup vs PDE-based Hessian matvec: $>750\times$ (measured), $\sim 10{,}000\times$ for targeted problems

**Memory:** $O(N_m N_d N_t)$ for storing the precomputed Fourier-domain matrix (vs $O(N_m N_d N_t^2/2)$ for the dense matrix).

## Applicability

- **SSM convolution kernels**: State-space models (S4, Mamba, etc.) produce causal convolution kernels $h_k = CA^k B$ that form exactly a block lower-triangular Toeplitz matrix. This GPU-optimized FFT matvec provides the fast path for computing the full sequence output $y = h * u$ in $O(N_t \log N_t)$ with near-peak GPU utilization
- **Toeplitz Neural Network (TNN) inference**: The Toeplitz token mixing layer is exactly a (block) Toeplitz matvec. For causal/autoregressive models, it becomes block lower-triangular Toeplitz. This paper's algorithm provides the GPU-optimal implementation
- **Parallel-in-time training**: When training sequence models with BPTT, the gradient computation through time is an adjoint block Toeplitz matvec — this algorithm gives it for free via the conjugate transpose trick
- **Hessian-vector products for second-order optimization**: For models with time-invariant dynamics, the Gauss-Newton Hessian is $\mathbf{H} = \mathbf{F}^*\mathbf{F} + \alpha\mathbf{R}$. Each CG iteration requires one $\mathbf{F}$ matvec and one $\mathbf{F}^*$ matvec, both handled by this algorithm
- **Multi-GPU scaling**: The 2D processor grid partitioning with NCCL-based reduction enables scaling to large clusters, relevant for long-context sequence models where $N_t$ is large

## Limitations

- **Requires shift-invariance**: The matrix must be block Toeplitz — content-dependent operations (standard attention) do not have this structure. Only works for position-invariant (convolutional/Toeplitz) token mixing
- **Memory-bound on GPU**: The SBGEMV kernel has arithmetic intensity ~0.5 FLOPs/byte, firmly in the memory-bound regime. It cannot leverage tensor cores (which require GEMM, not GEMV). For large spatial dimensions ($N_m, N_d \gg N_t$), the SBGEMV dominates and the algorithm becomes a bandwidth-limited batched GEMV
- **$2\times$ padding overhead**: The circulant embedding doubles the time dimension ($N_t \to 2N_t$), wasting memory and compute on padding. The Split FFT algorithm (trick 119) can partially mitigate this for multilevel block Toeplitz
- **One-time setup cost**: Pre-computing $\hat{\mathbf{F}}$ requires $N_m$ forward PDE solves or $N_d$ adjoint solves, plus batched FFTs. This is amortized over many matvecs but is not free
- **Communication overhead at scale**: Multi-GPU scaling is limited by the broadcast+reduce communication pattern. For 48 GPUs, weak scaling efficiency drops to ~85% due to NCCL overhead
- **Complex arithmetic**: The Fourier-domain operations require complex-valued GEMV, doubling the memory footprint and requiring complex cuBLAS routines

## Implementation Notes

```python
import torch
import torch.fft as fft

class FFTBlockToeplitzMatvec:
    """GPU-optimized FFT-based block lower-triangular Toeplitz matvec.

    Implements the algorithm from Venkat et al. (2024) for efficient
    block Toeplitz matrix-vector products on GPU.

    The block Toeplitz matrix F has blocks F_{ij} = F_{i-j+1,1}
    for i >= j, and 0 otherwise (lower-triangular).

    Args:
        blocks: (N_t, N_d, N_m) tensor — the N_t unique blocks
                F_{1,1}, F_{2,1}, ..., F_{N_t,1}
    """

    def __init__(self, blocks):
        N_t, N_d, N_m = blocks.shape
        self.N_t = N_t
        self.N_d = N_d
        self.N_m = N_m

        # Pad blocks to 2*N_t for circulant embedding
        # First column of block circulant:
        # [F_{1,1}; F_{2,1}; ...; F_{N_t,1}; 0; F_{N_t,1}; ...; F_{2,1}]
        # But for lower-triangular, the wrap-around part is zero
        padded = torch.zeros(2 * N_t, N_d, N_m,
                            dtype=blocks.dtype, device=blocks.device)
        padded[:N_t] = blocks

        # Precompute FFT of each block column (SOTI: FFT along time dim)
        # Result: (2*N_t, N_d, N_m) complex
        self.F_hat = fft.fft(padded.to(torch.complex64), dim=0)

    def forward(self, m):
        """Compute F @ m (block lower-triangular Toeplitz matvec).

        Args:
            m: (N_t, N_m) — input vector in SOTI ordering

        Returns:
            d: (N_t, N_d) — output vector
        """
        N_t = self.N_t

        # Step 1: Pad input with zeros
        m_padded = torch.zeros(2 * N_t, self.N_m,
                              dtype=m.dtype, device=m.device)
        m_padded[:N_t] = m

        # Step 2: FFT along time dimension
        m_hat = fft.fft(m_padded.to(torch.complex64), dim=0)  # (2*N_t, N_m)

        # Step 3: SBGEMV — block-diagonal multiply in Fourier space
        # For each time index k: d_hat[k] = F_hat[k] @ m_hat[k]
        # This is a batched matrix-vector product
        # F_hat: (2*N_t, N_d, N_m), m_hat: (2*N_t, N_m)
        d_hat = torch.einsum('tdn,tn->td', self.F_hat, m_hat)

        # Step 4: IFFT and unpad
        d_padded = fft.ifft(d_hat, dim=0)  # (2*N_t, N_d)
        d = d_padded[:N_t].real  # Take first N_t entries

        return d

    def adjoint(self, d):
        """Compute F* @ d (adjoint matvec, FREE via conjugate transpose).

        Args:
            d: (N_t, N_d) — input vector (data/observation)

        Returns:
            m: (N_t, N_m) — output vector (parameter space)
        """
        N_t = self.N_t

        # Step 1: Pad input
        d_padded = torch.zeros(2 * N_t, self.N_d,
                              dtype=d.dtype, device=d.device)
        d_padded[:N_t] = d

        # Step 2: FFT along time dimension
        d_hat = fft.fft(d_padded.to(torch.complex64), dim=0)

        # Step 3: Adjoint SBGEMV — conjugate transpose of F_hat
        # m_hat[k] = F_hat[k]^H @ d_hat[k]
        m_hat = torch.einsum('tdn,td->tn', self.F_hat.conj(), d_hat)

        # Step 4: IFFT and unpad
        m_padded = fft.ifft(m_hat, dim=0)
        m = m_padded[:N_t].real

        return m


# Example: SSM convolution as block Toeplitz matvec
def ssm_as_block_toeplitz(A, B, C, seq_len):
    """Construct block Toeplitz blocks from SSM parameters.

    SSM: x_k = A x_{k-1} + B u_k, y_k = C x_k
    The k-th block is F_k = C A^{k-1} B.

    Args:
        A: (d_state, d_state) — state transition
        B: (d_state, d_in) — input projection
        C: (d_out, d_state) — output projection
        seq_len: number of time steps

    Returns:
        FFTBlockToeplitzMatvec instance
    """
    d_out = C.shape[0]
    d_in = B.shape[1]

    blocks = torch.zeros(seq_len, d_out, d_in, device=A.device)
    A_power = torch.eye(A.shape[0], device=A.device)

    for k in range(seq_len):
        blocks[k] = C @ A_power @ B
        A_power = A_power @ A

    return FFTBlockToeplitzMatvec(blocks)
```

## References

- Venkat, S., Fernando, M., Henneking, S. & Ghattas, O. "Fast And Scalable FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices With Application to Linear Inverse Problems Governed by Autonomous Dynamical Systems" arXiv:2407.13066, 2024
- Venkat, S., Fernando, M., Henneking, S. & Ghattas, O. "Mixed-Precision Performance Portability of FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices" SC'25 Workshops, arXiv:2508.10202, 2025
- Qin, Z. et al. "Toeplitz Neural Network for Sequence Modeling" ICLR 2023. arXiv:2305.04749
- Gray, R.M. "Toeplitz and Circulant Matrices: A Review" Foundations and Trends in Communications and Information Theory, 2006
- GitHub implementation: https://github.com/s769/FFTMatvec

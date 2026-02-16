# 148: Structured Kernel Interpolation (SKI / KISS-GP)

**Category**: approximation
**Gain type**: efficiency
**Source**: Wilson & Nickisch (ICML 2015)
**Paper**: [papers/ski-kiss-gp-kernel-interpolation.pdf]
**Documented**: 2026-02-15

## Description

Structured Kernel Interpolation (SKI) accelerates kernel matrix-vector products (MVMs) from $O(n^2)$ to $O(n + m \log m)$ by placing inducing points on a **regular grid** and using **local cubic interpolation** to map arbitrary data points onto that grid. The grid structure enables the inducing point covariance matrix $K_{U,U}$ to be **Toeplitz** (in 1D) or **Kronecker** (in multi-D), unlocking FFT-based MVMs. The interpolation matrix $W$ is extremely sparse (only $c = 4$ nonzeros per row for cubic interpolation in 1D), making the interpolation step $O(n)$.

The key insight is that for stationary kernels $k(\mathbf{x}, \mathbf{z}) = k(\mathbf{x} - \mathbf{z})$, the kernel matrix on a regular grid is Toeplitz (1D) or block-Toeplitz-with-Toeplitz-blocks (multi-D), which can be embedded into a circulant matrix for $O(m \log m)$ MVMs via FFT. SKI decouples the data layout from the grid: it handles arbitrary (non-grid) inputs by sparse interpolation $K_{X,U} \approx W K_{U,U}$, then exploits the grid structure of $K_{U,U}$.

This is one of the most GPU-friendly fast kernel methods: the core computation is a single FFT on a regular grid (perfectly coalesced memory access, maps to cuFFT) plus sparse matrix-vector products (well-supported by cuSPARSE/custom CUDA). GPyTorch implements this as its primary scalable GP inference engine, running efficiently on GPUs for datasets with 100K+ points.

## Mathematical Form

**Core Approximation:**

Given $n$ training points $X = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ and $m$ inducing points $U = \{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$ on a regular grid:

$$
K_{X,U} \approx W K_{U,U}
$$

where $W \in \mathbb{R}^{n \times m}$ is a sparse interpolation matrix with $c$ nonzero entries per row.

The full kernel matrix is then approximated as:

$$
K_{X,X} \approx K_{\text{SKI}} = W K_{U,U} W^\top
$$

**Interpolation Matrix $W$:**

For local cubic interpolation on an equispaced 1D grid with spacing $h$, for a point $\mathbf{x}_i$ falling between grid points $\mathbf{u}_a$ and $\mathbf{u}_{a+1}$, the weights are determined by Keys' cubic convolution (Keys, 1981):

$$
W_{i,j} = \begin{cases}
\frac{3}{2}|s|^3 - \frac{5}{2}|s|^2 + 1 & \text{if } |s| \leq 1 \\
-\frac{1}{2}|s|^3 + \frac{5}{2}|s|^2 - 4|s| + 2 & \text{if } 1 < |s| \leq 2 \\
0 & \text{otherwise}
\end{cases}
$$

where $s = (\mathbf{x}_i - \mathbf{u}_j)/h$. This gives $c = 4$ nonzero entries per row (the 4 nearest grid points).

**Toeplitz Structure (1D):**

For a stationary kernel on a regular 1D grid $\{u_1, \ldots, u_m\}$ with spacing $h$:

$$
(K_{U,U})_{ij} = k(u_i - u_j) = k((i-j)h) = t_{i-j}
$$

This is Toeplitz: constant along diagonals. Embed into a circulant matrix $C$ of size $2m$:

$$
C \mathbf{v} = \text{IFFT}(\text{FFT}(\mathbf{c}) \odot \text{FFT}(\mathbf{v}))
$$

where $\mathbf{c}$ is the first column of $C$. The Toeplitz MVM is extracted from the circulant MVM.

**Kronecker Structure (multi-D):**

For a product kernel $k(\mathbf{x}, \mathbf{z}) = \prod_{p=1}^{P} k_p(\mathbf{x}^{(p)}, \mathbf{z}^{(p)})$ on a $P$-dimensional grid $U = U_1 \times \cdots \times U_P$:

$$
K_{U,U} = K_1 \otimes K_2 \otimes \cdots \otimes K_P
$$

where each $K_p \in \mathbb{R}^{m_p \times m_p}$ is the 1D kernel matrix for dimension $p$. The MVM decomposes:

$$
(K_1 \otimes \cdots \otimes K_P)\mathbf{v} = \text{vec}\left(K_P \cdots K_2 \cdot \text{mat}(\mathbf{v}) \cdot K_1^\top\right)
$$

computed via a sequence of matrix multiplications along each mode.

**Full SKI MVM:**

$$
K_{\text{SKI}} \mathbf{v} = W \underbrace{K_{U,U} \underbrace{(W^\top \mathbf{v})}_{\text{sparse MVM}}}_{\text{Toeplitz/Kronecker MVM}}
$$

Three steps: (1) sparse interpolation $\mathbf{w} = W^\top \mathbf{v}$, (2) structured MVM $\mathbf{z} = K_{U,U} \mathbf{w}$, (3) sparse interpolation back $\mathbf{y} = W \mathbf{z}$.

**Key Definitions:**

- $n$ — number of training points
- $m$ — number of grid (inducing) points, can be $m \gg n$
- $W \in \mathbb{R}^{n \times m}$ — sparse interpolation matrix ($c$ nonzeros per row)
- $K_{U,U} \in \mathbb{R}^{m \times m}$ — structured kernel matrix on grid (Toeplitz/Kronecker)
- $P$ — input dimensionality (for Kronecker decomposition)
- $h$ — grid spacing

## Complexity

| Operation | Naive | With SKI (Toeplitz) | With SKI (Kronecker) |
|-----------|-------|---------------------|----------------------|
| Kernel MVM | $O(n^2)$ | $O(n + m \log m)$ | $O(n + P m^{1+1/P})$ |
| Storage | $O(n^2)$ | $O(n + m)$ | $O(n + P m^{2/P})$ |
| CG solve ($j$ iters) | $O(jn^2)$ | $O(j(n + m\log m))$ | $O(j(n + Pm^{1+1/P}))$ |

**Memory:** $O(n + m)$ for Toeplitz, $O(n + Pm^{2/P})$ for Kronecker, vs $O(n^2)$ naive.

**Interpolation error:** For cubic interpolation with grid spacing $h$ and a kernel with bounded 4th derivative:

$$
|k(\mathbf{x}, \mathbf{z}) - k_{\text{SKI}}(\mathbf{x}, \mathbf{z})| = O(h^3)
$$

Error decays **cubically** with grid refinement — so even moderate grid sizes give excellent approximations.

**Practical regime:** For $n = 100{,}000$ points in $P = 1$ dimension with $m = 10{,}000$ grid points, SKI with Toeplitz FFT computes a kernel MVM in ~1ms on GPU vs ~10s for dense.

## Applicability

- **Kernel attention variants**: Any attention mechanism using stationary kernels (Gaussian/RBF, Matérn, Laplacian) can use SKI to accelerate the kernel MVM from $O(n^2)$ to $O(n + m \log m)$. The grid interpolation acts as a learned quantization of the feature space.
- **Gaussian process layers in neural networks**: Deep kernel learning (Wilson et al., 2016) places a GP on top of neural network features. SKI enables this to scale to large datasets during training.
- **State space model kernels**: When SSM kernels can be expressed as stationary functions of position differences, SKI provides an alternative to the Cauchy-based FKT for acceleration.
- **Scalable GP pretraining losses**: Using GP marginal likelihood as a training objective requires $K^{-1}\mathbf{y}$ and $\log|K|$ — both efficiently computable via SKI + conjugate gradients + stochastic trace estimation.
- **Kernel density estimation**: Fast KDE for t-SNE/UMAP gradient computation on regular grids with interpolation from non-grid data.

## Limitations

- **Curse of dimensionality**: The Kronecker structure requires $m = \prod_p m_p$ grid points; for $P > 4$ dimensions, this becomes impractical. SKIP (product kernel interpolation) and SimplexGP mitigate this but add complexity.
- **Stationary kernels only**: The Toeplitz/circulant structure requires translation invariance $k(\mathbf{x}, \mathbf{z}) = k(\mathbf{x} - \mathbf{z})$. Non-stationary kernels break this structure. However, the interpolation framework itself is general.
- **Grid bounds must be known**: The regular grid must cover the data range. For streaming or out-of-distribution data, the grid may need resizing.
- **Interpolation error for non-smooth kernels**: The cubic error bound $O(h^3)$ assumes sufficient smoothness. For rough kernels (Matérn-1/2), accuracy degrades and finer grids are needed.
- **Not applicable to standard softmax attention**: Like trick 147, this accelerates kernel MVMs for stationary kernels, not the exponential softmax kernel.

## Implementation Notes

```python
# SKI / KISS-GP: Structured Kernel Interpolation
# Production implementation: GPyTorch (gpytorch.kernels.GridInterpolationKernel)
import torch
import torch.fft

def ski_kernel_mvm(
    x: torch.Tensor,      # (n,) training points (1D for simplicity)
    v: torch.Tensor,      # (n,) vector to multiply
    grid: torch.Tensor,   # (m,) regular grid points
    kernel_fn,            # kernel function k(r) -> scalar
) -> torch.Tensor:
    """
    Compute K_SKI @ v = W @ K_UU @ W^T @ v in O(n + m log m).

    GPU-friendly: sparse interp is coalesced, FFT uses cuFFT.
    """
    n = x.shape[0]
    m = grid.shape[0]
    h = grid[1] - grid[0]  # grid spacing

    # Step 1: Build sparse interpolation W^T @ v  [O(n)]
    # For each x_i, find 4 nearest grid points and cubic weights
    grid_idx = ((x - grid[0]) / h).long()  # nearest lower grid index
    grid_idx = grid_idx.clamp(1, m - 3)    # ensure 4 neighbors exist

    # Cubic interpolation weights (Keys, 1981)
    s = (x - grid[grid_idx]) / h  # fractional position in [0, 1)

    w = torch.zeros(n, 4, device=x.device)
    w[:, 0] = -0.5*s**3 + s**2 - 0.5*s            # u_{j-1}
    w[:, 1] = 1.5*s**3 - 2.5*s**2 + 1              # u_j
    w[:, 2] = -1.5*s**3 + 2*s**2 + 0.5*s            # u_{j+1}
    w[:, 3] = 0.5*s**3 - 0.5*s**2                   # u_{j+2}

    # Sparse W^T @ v: scatter weighted v onto grid
    wt_v = torch.zeros(m, device=x.device)
    for k in range(4):
        idx = grid_idx - 1 + k
        wt_v.scatter_add_(0, idx, w[:, k] * v)

    # Step 2: Toeplitz MVM via circulant embedding [O(m log m)]
    # First column of Toeplitz matrix
    dists = grid - grid[0]
    t = kernel_fn(dists)  # (m,) first row of Toeplitz

    # Embed in 2m circulant matrix
    c = torch.cat([t, torch.zeros(1, device=x.device), t[1:].flip(0)])
    # FFT-based circulant MVM
    wt_v_padded = torch.cat([wt_v, torch.zeros(m, device=x.device)])
    kuu_wt_v = torch.fft.irfft(
        torch.fft.rfft(c) * torch.fft.rfft(wt_v_padded),
        n=2*m
    )[:m]

    # Step 3: Interpolation W @ (K_UU @ W^T @ v)  [O(n)]
    result = torch.zeros(n, device=x.device)
    for k in range(4):
        idx = grid_idx - 1 + k
        result += w[:, k] * kuu_wt_v[idx]

    return result

# For multi-D with Kronecker structure:
def kronecker_mvm(K_list, v, dims):
    """MVM with K_1 ⊗ K_2 ⊗ ... ⊗ K_P via mode-wise products."""
    V = v.reshape(dims)  # reshape to P-dimensional tensor
    for p in range(len(K_list)):
        # Contract along dimension p
        V = torch.tensordot(K_list[p], V, dims=([1], [p]))
        # Move contracted dim back to position p
        V = V.movedim(0, p)
    return V.reshape(-1)

# GPU notes:
# - Step 1 & 3: scatter_add_ is well-optimized on GPU (coalesced for sorted x)
# - Step 2: cuFFT runs at near-peak bandwidth for 1D FFTs
# - Kronecker MVM: sequence of GEMMs, each fully utilizing tensor cores
# - Total: 2 sparse MVPs + 1 FFT (or P GEMMs) = excellent GPU utilization
```

## References

- Wilson, A. G. & Nickisch, H. (2015). Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP). ICML 2015.
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). Deep Kernel Learning. AISTATS 2016.
- Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. NeurIPS 2018.
- Keys, R. (1981). Cubic Convolution Interpolation for Digital Image Processing. IEEE Trans. ASSP.
- Cunningham, J. P., Shenoy, K. V., & Sahani, M. (2008). Fast Gaussian Process Methods for Point Process Intensity Estimation. ICML 2008.
- Saatchi, Y. (2011). Scalable Inference for Structured Gaussian Process Models. PhD Thesis, University of Cambridge.
- Wilson, A. G. & Adams, R. P. (2013). Gaussian Process Kernels for Pattern Discovery and Extrapolation. ICML 2013.

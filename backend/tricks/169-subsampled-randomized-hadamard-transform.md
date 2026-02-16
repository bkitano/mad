# 169: Subsampled Randomized Hadamard Transform (SRHT)

**Category**: approximation
**Gain type**: efficiency
**Source**: Ailon & Chazelle (2006/2009), Tropp (2011), Lacotte, Liu, Dobriban & Pilanci (2020)
**Paper**: [papers/srht-iterative-sketching.pdf]
**Documented**: 2026-02-15

## Description

The Subsampled Randomized Hadamard Transform (SRHT) is a structured random dimensionality reduction technique that projects an $n$-dimensional vector down to $m \ll n$ dimensions in $O(n \log m)$ time, compared to $O(nm)$ for a dense Gaussian sketch. The SRHT matrix is $S = \frac{1}{\sqrt{m}} B H_n D P$, where $P$ is a random permutation, $D$ is a Rademacher diagonal, $H_n$ is the Walsh-Hadamard matrix, and $B$ is a row-sampling (subsampling) operator that selects $m$ out of $n$ rows. The construction ensures the sketch $SA$ of a data matrix $A \in \mathbb{R}^{n \times d}$ preserves spectral properties of $A$ while being much faster to compute than a dense Gaussian sketch.

The key distinction from SORF (trick 155) and Fastfood (trick 162) is the **purpose**: SORF/Fastfood are $d \times d$ structured orthogonal matrices for kernel approximation (same-dimension projection), while SRHT is an $m \times n$ structured matrix for dimensionality reduction ($m \ll n$). They share the same $HD$ core but serve complementary roles:

| | SORF (trick 155) | SRHT (this trick) |
|---|---|---|
| Shape | $d \times d$ | $m \times n$, $m \ll n$ |
| Purpose | Kernel approximation | Dimensionality reduction / sketching |
| Use in attention | Feature map $\phi(x) = f(\mathbf{W}x)$ | Sketch gradients, Hessians, data matrices |
| Preserves | Kernel inner products | Subspace embedding (spectral properties) |

The theoretical paper by Lacotte et al. (2020) proves that SRHT sketches achieve the **same asymptotic convergence rate** as ideal Haar (uniformly random orthogonal) projections in iterative Hessian sketch algorithms, and **strictly outperform** Gaussian i.i.d. projections. The convergence rate satisfies $\rho_h = \rho_g \cdot \frac{\xi(1-\xi)}{\gamma^2 + \xi - 2\gamma\xi}$ where $\rho_g$ is the Gaussian rate and the ratio $\rho_h/\rho_g < 1$ always.

For neural network training, SRHT is the foundational sketching technique behind randomized preconditioning, gradient compression, and second-order optimization methods.

## Mathematical Form

**Core Operation:**

The SRHT sketching matrix $S \in \mathbb{R}^{m \times n}$ is:

$$
S = \frac{1}{\sqrt{m}} B H_n D P
$$

where the components are applied right-to-left to a vector $x \in \mathbb{R}^n$:

1. $P \in \{0,1\}^{n \times n}$ — random permutation matrix (breaks data alignment)
2. $D \in \{-1,+1\}^{n \times n}$ — diagonal Rademacher matrix ($D_{ii} \in \{-1,+1\}$ i.i.d.)
3. $H_n \in \mathbb{R}^{n \times n}$ — unnormalized Walsh-Hadamard matrix, $H_n = \begin{bmatrix} H_{n/2} & H_{n/2} \\ H_{n/2} & -H_{n/2} \end{bmatrix}$, $H_1 = [1]$
4. $B \in \{0,1\}^{m \times n}$ — row subsampling matrix (selects $m$ rows via Bernoulli sampling with success probability $m/n$, then discards zero rows)

**Key Definitions:**

- $n$ — ambient dimension (rows of data matrix, or dimension being reduced)
- $d$ — number of features/columns in data matrix $A \in \mathbb{R}^{n \times d}$
- $m$ — sketch dimension ($m \ll n$, typically $m \asymp d \log d$ suffices)
- $\gamma := \lim_{n,d \to \infty} d/n \in (0,1)$ — aspect ratio
- $\xi := \lim_{n,m \to \infty} m/n \in (\gamma, 1)$ — sketch ratio
- $\rho_g := \gamma/\xi$ — Gaussian sketch convergence rate

**Oblivious Subspace Embedding Property:**

For any fixed matrix $A \in \mathbb{R}^{n \times d}$ with $\text{rank}(A) = d$, the SRHT $S$ satisfies: for $m = O(d \log d \cdot \log(d/\delta) / \epsilon^2)$, with probability $\geq 1 - \delta$:

$$
(1 - \epsilon) \|Ax\|^2 \leq \|SAx\|^2 \leq (1 + \epsilon) \|Ax\|^2, \quad \forall x \in \mathbb{R}^d
$$

This is the Johnson-Lindenstrauss property restricted to the column space of $A$.

**Inverse Moment Formulas (Lemma 3.2/4.3 — key new results):**

For $S$ either a Haar or SRHT matrix, and $U \in \mathbb{R}^{n \times d}$ with orthonormal columns:

$$
\theta_{1,h} := \lim_{n \to \infty} \frac{1}{d} \text{tr}\, \mathbb{E}\left[(U^\top S^\top S U)^{-1}\right] = \frac{1-\gamma}{\xi - \gamma}
$$

$$
\theta_{2,h} := \lim_{n \to \infty} \frac{1}{d} \text{tr}\, \mathbb{E}\left[(U^\top S^\top S U)^{-2}\right] = \frac{(1-\gamma)(\gamma^2 + \xi - 2\gamma\xi)}{(\xi - \gamma)^3}
$$

These formulas are identical for Haar and SRHT matrices (Lemma 4.3), proving SRHT matches the statistical quality of ideal orthogonal projections.

**Optimal IHS Convergence (Theorem 3.1 + 4.1):**

For the iterative Hessian sketch $x_{t+1} = x_t - \mu_t H_t^{-1} \nabla f(x_t)$ with refreshed SRHT embeddings $\{S_t\}$, step sizes $\mu_t = \theta_{1,h}/\theta_{2,h}$, and momentum $\beta_t = 0$:

$$
\rho_s := \left(\lim_{n \to \infty} \frac{\mathbb{E}\|\Delta_t\|^2}{\mathbb{E}\|\Delta_0\|^2}\right)^{1/t} = \rho_g \cdot \frac{\xi(1-\xi)}{\gamma^2 + \xi - 2\gamma\xi} = \rho_h
$$

where $\rho_g = \gamma/\xi$ is the Gaussian convergence rate. The ratio:

$$
\frac{\rho_h}{\rho_g} = \frac{\xi(1-\xi)}{\gamma^2 + \xi - 2\gamma\xi} < 1
$$

always, proving SRHT strictly outperforms Gaussian sketches. As $\xi \to 1$ (sketch size approaches data size), the improvement ratio approaches $1 - \xi$, giving significant acceleration.

**Complexity of IHS with SRHT (Section 5):**

For $\epsilon$-accuracy on least-squares with $A \in \mathbb{R}^{n \times d}$, sketch size $m \asymp d$:

$$
\mathcal{C}_n \asymp (nd \log d + d^3 + nd) \log(1/\epsilon)
$$

vs. preconditioned conjugate gradient:

$$
\mathcal{C}_c \asymp nd \log d + d^3 \log d + nd \log(1/\epsilon)
$$

The ratio $\mathcal{C}_n / \mathcal{C}_c \asymp 1/\log d$, so SRHT-IHS improves by a factor of $\log d$.

## Complexity

| Operation | Gaussian Sketch | Haar Sketch | SRHT |
|-----------|----------------|-------------|------|
| Form sketch $SA$ | $O(nmd)$ | $O(nm^2)$ (Gram-Schmidt) | $O(nd \log m)$ |
| Storage of $S$ | $O(nm)$ | $O(nm)$ | $O(n)$ (diagonal + permutation) |
| Convergence rate | $\rho_g = \gamma/\xi$ | $\rho_h < \rho_g$ | $\rho_s = \rho_h$ (matches Haar) |
| Optimal step size | depends on $A$ | $\theta_{1,h}/\theta_{2,h}$ (universal) | $\theta_{1,h}/\theta_{2,h}$ (universal) |

**Key insight**: SRHT achieves the same convergence rate as Haar (uniformly random orthogonal) sketches, which strictly dominate Gaussian sketches, while being $O(nd \log m)$ to apply vs $O(nm^2)$ for Haar.

**Memory:** $O(n)$ for the SRHT itself (a permutation vector + Rademacher diagonal), plus $O(md)$ for the sketch output.

**Sketch size requirement:** $m = O(d \log d)$ rows suffice for an $\epsilon$-subspace embedding (Tropp 2011), vs $m = O(d/\epsilon^2)$ for Gaussian.

## Applicability

- **Gradient/Hessian sketching for second-order optimization**: The primary application in training. For a model with $d$ parameters and loss Hessian approximated from $n$ data points, SRHT sketches the $n \times d$ Jacobian to an $m \times d$ sketch in $O(nd \log m)$ instead of $O(nmd)$, enabling Newton-type updates
- **Randomized preconditioning**: Compute a preconditioner $P$ from $SA$ (the sketch of the data matrix), then solve the preconditioned system $\min_y \|AP^{-1}y - b\|$ via conjugate gradient. SRHT makes the sketch formation fast
- **Low-rank approximation**: SRHT applied to $A$ gives $SA$ whose column space approximates the top-$m$ singular subspace of $A$. Used in randomized SVD / randomized NMF for embedding layers, LoRA initialization, etc.
- **Distributed training**: In data-parallel training, each worker can apply a local SRHT to its gradient shard before allreduce, reducing communication. Block-SRHT variants (Balabanov et al., 2022) formalize this
- **GPU implementation considerations**:
  - The FWHT is the same butterfly operation as in SORF — well-suited to shared memory, no complex arithmetic
  - The permutation $P$ is a gather operation that breaks memory coalescing — this is the main GPU concern. However, for large $n$ (data dimension), the permutation is applied once and amortized over many sketch operations
  - The subsampling $B$ is a simple index-select, which is efficient on GPU
  - For the IHS application, the sketch is refreshed at every iteration, so $S$ is applied repeatedly. Fusing $D$, $H$, $P$ into a single kernel is important
  - The resulting sketch $SA \in \mathbb{R}^{m \times d}$ is a dense matrix that feeds into standard GEMM operations on tensor cores
- **Tensor core path**: SRHT itself doesn't use tensor cores (it's FWHT + gather + subsample), but the downstream operations on the sketch (e.g., $(SA)^\top(SA)$ for the sketched Hessian, or $(SA)^\top(Sb)$ for the sketched gradient) are standard GEMMs that fully utilize tensor cores

## Limitations

- **Permutation breaks coalescing**: The random permutation $P$ involves a gather operation on GPU, similar to Fastfood's permutation. For very large $n$ (millions of rows), this can be a bandwidth bottleneck. In practice, one can omit $P$ with slight theoretical degradation (the data itself provides sufficient incoherence in many settings)
- **Power-of-2 requirement**: The Walsh-Hadamard transform requires $n = 2^k$. Non-power-of-2 dimensions must be zero-padded. For data matrices, $n$ (number of data points) is often large and arbitrary — padding can waste significant memory
- **Sketch size lower bound**: Requires $m = \Omega(d \log d)$ for subspace embedding guarantees, slightly worse than Gaussian's $m = \Omega(d/\epsilon^2)$ in the $\epsilon$-dependence for some regimes
- **Refreshing adds overhead**: In iterative algorithms (IHS), the SRHT must be redrawn each iteration. This means regenerating the Rademacher diagonal and permutation, which is cheap ($O(n)$), but the FWHT must be recomputed on the data at each step
- **Not directly applicable to attention**: Unlike SORF (which accelerates the feature map projection), SRHT is for sketching/dimensionality reduction and is used in training infrastructure (gradient compression, preconditioning) rather than in the forward pass of the model
- **Asymptotic theory**: The convergence rate results are asymptotic ($n, d, m \to \infty$ with fixed ratios). For small/moderate dimensions, finite-sample behavior may differ from the asymptotic predictions, though numerical experiments confirm good agreement even at moderate sizes ($n = 4096, d = 800$)

## Implementation Notes

```python
import torch
import math

def fwht_inplace(x):
    """Fast Walsh-Hadamard Transform (unnormalized) along last dim.

    Args:
        x: (..., n) tensor, n must be power of 2
    Returns:
        (..., n) tensor with WHT applied
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a, b = x[..., 0, :].clone(), x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        h *= 2
    return x.view(*x.shape[:-3], n)

def srht_sketch(A, m, seed=None):
    """Apply SRHT sketch to data matrix A.

    Computes S @ A where S = (1/sqrt(m)) * B * H_n * D * P
    in O(n * d * log(m)) time instead of O(n * m * d) for Gaussian.

    Args:
        A: (n, d) data matrix
        m: sketch dimension (m << n, typically m ~ d * log(d))
        seed: optional random seed for reproducibility
    Returns:
        SA: (m, d) sketched matrix
    """
    n, d = A.shape

    if seed is not None:
        torch.manual_seed(seed)

    # Pad n to next power of 2 for Hadamard
    n_pad = 1 << (n - 1).bit_length()
    if n_pad != n:
        A_pad = torch.zeros(n_pad, d, device=A.device, dtype=A.dtype)
        A_pad[:n] = A
    else:
        A_pad = A.clone()

    # Step 1: Random permutation P (gather — GPU unfriendly part)
    perm = torch.randperm(n_pad, device=A.device)
    A_pad = A_pad[perm]

    # Step 2: Rademacher diagonal D
    signs = torch.randint(0, 2, (n_pad, 1), device=A.device).float() * 2 - 1
    A_pad = A_pad * signs

    # Step 3: Walsh-Hadamard transform H_n (applied column-wise)
    # For matrix: apply FWHT to each column
    A_pad = fwht_inplace(A_pad.T).T  # transpose trick for column-wise

    # Step 4: Subsample m rows (Bernoulli or deterministic top-m)
    # Deterministic: just take first m rows (after random permutation,
    # this is equivalent to random subsampling)
    indices = torch.randperm(n_pad, device=A.device)[:m]
    SA = A_pad[indices] / math.sqrt(m)

    return SA

def iterative_hessian_sketch(A, b, m, T=20, use_srht=True):
    """Iterative Hessian Sketch for least-squares: min ||Ax - b||^2.

    Uses SRHT sketches with asymptotically optimal step sizes.

    Args:
        A: (n, d) data matrix
        b: (n,) observation vector
        m: sketch size (m >= d, typically m ~ 2d)
        T: number of iterations
        use_srht: if True, use SRHT; else use Gaussian sketch
    Returns:
        x: (d,) approximate solution
    """
    n, d = A.shape
    x = torch.zeros(d, device=A.device, dtype=A.dtype)

    # Asymptotic aspect ratios
    gamma = d / n
    xi = m / n

    # Optimal step size for SRHT/Haar (Theorem 3.1)
    theta1 = (1 - gamma) / (xi - gamma)
    theta2 = (1 - gamma) * (gamma**2 + xi - 2*gamma*xi) / (xi - gamma)**3
    mu = theta1 / theta2  # optimal step size

    # Precompute A^T b
    Atb = A.T @ b  # (d,)

    for t in range(T):
        # Sketch the data matrix: S_t @ A
        if use_srht:
            SA = srht_sketch(A, m, seed=t)  # (m, d), O(nd log m)
        else:
            S = torch.randn(m, n, device=A.device) / math.sqrt(m)
            SA = S @ A  # (m, d), O(nmd)

        # Sketched Hessian: H_t = (SA)^T (SA)
        # This is a d x d GEMM — tensor core friendly!
        H_t = SA.T @ SA  # (d, d)

        # Gradient at current point
        grad = A.T @ (A @ x) - Atb  # (d,), O(nd)

        # Newton-like update with sketched Hessian
        # Solve H_t @ delta = grad
        delta = torch.linalg.solve(H_t, grad)  # O(d^3) or O(d^2) with Cholesky

        # Update with optimal step size
        x = x - mu * delta

    return x

# GPU efficiency analysis:
#
# 1. SRHT formation (S @ A):
#    - Permutation: O(n) gather — bandwidth bound, breaks coalescing
#    - Rademacher: O(n*d) elementwise — fast
#    - FWHT: O(n*d*log n) — butterfly, shared-memory friendly
#    - Subsample: O(m*d) index-select — fast
#    Total: O(nd log n), dominated by FWHT
#
# 2. Sketched Hessian (SA)^T (SA):
#    - Standard GEMM: (d, m) @ (m, d) = (d, d)
#    - Tensor core native, high arithmetic intensity
#    - This is the operation that benefits most from the sketch
#
# 3. Comparison to Gaussian sketch:
#    - Gaussian: O(nmd) for S @ A (dense matmul)
#    - SRHT: O(nd log m) for the same
#    - Speedup: O(m / log m) ~ O(d / log d) for m ~ d
#
# 4. Practical considerations:
#    - For n < 10000, the FWHT overhead may not beat a dense GEMM
#    - For n > 100000 (typical in training), SRHT is clearly faster
#    - The permutation can be omitted in practice (slight quality loss)
#      to improve GPU coalescing
#    - Can be combined with SORF: use SRHT for gradient sketching
#      and SORF for kernel feature maps in FAVOR+
```

## References

- Lacotte, J., Liu, S., Dobriban, E., & Pilanci, M. (2020). Optimal Iterative Sketching with the Subsampled Randomized Hadamard Transform. NeurIPS 2020. arXiv:2002.00864.
- Ailon, N. & Chazelle, B. (2006). Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform. STOC 2006.
- Ailon, N. & Chazelle, B. (2009). The Fast Johnson-Lindenstrauss Transform and Approximate Nearest Neighbors. SIAM Journal on Computing, 39(1).
- Tropp, J. A. (2011). Improved Analysis of the Subsampled Randomized Hadamard Transform. Advances in Adaptive Data Analysis, 3(01n02):115–126.
- Pilanci, M. & Wainwright, M. J. (2016). Iterative Hessian Sketch: Fast and Accurate Solution Approximation for Constrained Least-Squares. JMLR, 17(1):1842–1879.
- Sarlos, T. (2006). Improved Approximation Algorithms for Large Matrices via Random Projections. FOCS 2006.
- Rokhlin, V. & Tygert, M. (2008). A Fast Randomized Algorithm for Overdetermined Linear Least-Squares Regression. PNAS, 105(36):13212–13217.
- Woodruff, D. P. (2014). Sketching as a Tool for Numerical Linear Algebra. Foundations and Trends in TCS, 10(1–2):1–157.
- Balabanov, O., Beaupere, M., Grigori, L., & Lederer, V. (2022). Block Subsampled Randomized Hadamard Transform for Low-Rank Approximation on Distributed Architectures. arXiv:2210.11295.

# 175: Gohberg-Semencul Circulant Toeplitz Inverse

**Category**: decomposition
**Gain type**: efficiency
**Source**: Gohberg & Semencul (1972); Ammar & Gader "A Variant of the Gohberg-Semencul Formula Involving Circulant Matrices" (SIAM J. Matrix Anal. Appl., 1991); Huckle "Circulant and skewcirculant matrices for solving Toeplitz matrix problems" (SIAM J. Matrix Anal. Appl., 1992)
**Paper**: [papers/huckle-circulant-skewcirculant-toeplitz.pdf]
**Documented**: 2026-02-15

## Description

The **Gohberg-Semencul (GS) formula** expresses the inverse of a nonsingular Toeplitz matrix $T_n$ as the difference of two products of triangular Toeplitz matrices, each defined by only $n$ parameters (the first and last columns of $T_n^{-1}$). This means $T_n^{-1}$ — which is a dense $n \times n$ matrix — can be represented by just $2n$ scalars instead of $n^2$, and a Toeplitz system $T_n x = b$ can be solved in $O(n \log n)$ via two FFT-based Toeplitz matvecs once the GS vectors are known.

The **circulant variant** (Ammar & Gader, 1991) replaces the upper-triangular Toeplitz factors with **circulant matrices**, exploiting the cyclic displacement structure of Toeplitz matrices. This modification yields computational savings because circulant matvecs are implemented directly via a single FFT + element-wise multiply + IFFT pipeline, avoiding the index manipulation needed for upper-triangular Toeplitz products. The circulant variant enables all four Toeplitz factors in the GS formula to be applied via FFT-based routines, making the entire inverse application pipeline consist purely of FFT operations.

For neural networks, this trick is relevant when one needs to **invert** a Toeplitz token mixer or solve Toeplitz systems arising in normalizing flows, implicit layers, or bidirectional inference. The GS formula reduces Toeplitz inversion from a $O(n^2)$ dense operation to $O(n \log n)$ using only 4 FFTs, making it practical as a differentiable layer.

## Mathematical Form

**Standard Gohberg-Semencul Formula:**

Let $T_n = (t_{i-j})_{i,j=0}^{n-1}$ be a nonsingular Toeplitz matrix. Solve the Yule-Walker system:

$$
T_n \mathbf{x} = \mathbf{e}_1, \quad \mathbf{x} = (x_0, x_1, \ldots, x_{n-1})^T
$$

$$
T_n \mathbf{y} = \mathbf{e}_n, \quad \mathbf{y} = (y_0, y_1, \ldots, y_{n-1})^T
$$

where $\mathbf{e}_1 = (1, 0, \ldots, 0)^T$ and $\mathbf{e}_n = (0, \ldots, 0, 1)^T$. Then, provided $x_0 \neq 0$:

$$
T_n^{-1} = \frac{1}{x_0}\left(L(\mathbf{x}) U(\mathbf{x}) - L(\bar{\mathbf{y}}) U(\bar{\mathbf{y}})\right)
$$

where:
- $L(\mathbf{x})$ is the lower-triangular Toeplitz matrix with first column $\mathbf{x}$
- $U(\mathbf{x})$ is the upper-triangular Toeplitz matrix with first row $\mathbf{x}^T$
- $\bar{\mathbf{y}} = (0, y_{n-1}, y_{n-2}, \ldots, y_1)^T$ is $\mathbf{y}$ reversed and shifted

Equivalently, writing out the matrix elements:

$$
(T_n^{-1})_{ij} = \frac{1}{x_0} \begin{cases}
\sum_{k=0}^{j} x_{i-j+k} x_k - \sum_{k=0}^{j} \bar{y}_{i-j+k} \bar{y}_k, & i \geq j \\
\sum_{k=0}^{i} x_k x_{j-i+k} - \sum_{k=0}^{i} \bar{y}_k \bar{y}_{j-i+k}, & i < j
\end{cases}
$$

**Circulant Variant (Ammar & Gader 1991):**

Using the **cyclic displacement** operator $\nabla_Z[A] = A - ZAZ^T$ (where $Z$ is the cyclic downshift matrix), every nonsingular Toeplitz matrix $T_n$ satisfies $\text{rank}(\nabla_Z[T_n]) = 2$. This displacement structure implies that $T_n^{-1}$ can be written as:

$$
T_n^{-1} = \frac{1}{x_0}\left(L(\mathbf{x}) C(\mathbf{x}') - L(\bar{\mathbf{y}}) C(\bar{\mathbf{y}}')\right)
$$

where:
- $L(\mathbf{x})$ and $L(\bar{\mathbf{y}})$ are **lower-triangular Toeplitz** matrices (same as before)
- $C(\mathbf{x}')$ and $C(\bar{\mathbf{y}}')$ are **circulant** matrices replacing the upper-triangular factors
- $\mathbf{x}' = (x_0, 0, x_{n-1}, x_{n-2}, \ldots, x_1)^T$ — the circulant version of $\mathbf{x}$
- $\bar{\mathbf{y}}' = (0, 0, \bar{y}_{n-1}, \bar{y}_{n-2}, \ldots, \bar{y}_1)^T$ — the circulant version of $\bar{\mathbf{y}}$

The circulant $C(\mathbf{v})$ has first column $\mathbf{v}$ and satisfies $C(\mathbf{v}) = F^* \text{diag}(\text{FFT}(\mathbf{v})) F$.

**Applying $T_n^{-1}$ to a vector (solve $T_n z = b$):**

$$
z = T_n^{-1} b = \frac{1}{x_0}\left(L(\mathbf{x}) \cdot C(\mathbf{x}') b - L(\bar{\mathbf{y}}) \cdot C(\bar{\mathbf{y}}') b\right)
$$

Each application of $C(\cdot) b$ costs $O(n \log n)$ (one FFT + multiply + IFFT).
Each application of $L(\cdot)$ to a vector also costs $O(n \log n)$ (via circulant embedding).

**Total: 4 Toeplitz/circulant matvecs = $O(n \log n)$ per solve.**

**Computing the GS Vectors ($\mathbf{x}$ and $\mathbf{y}$):**

The vectors $\mathbf{x}$ and $\mathbf{y}$ are the solutions of $T_n \mathbf{x} = \mathbf{e}_1$ and $T_n \mathbf{y} = \mathbf{e}_n$. These can be computed by:

1. **Levinson recursion**: $O(n^2)$ — computes $\mathbf{x}$ incrementally from $T_1, T_2, \ldots, T_n$
2. **Superfast algorithms**: $O(n \log^2 n)$ — via displacement rank and divide-and-conquer
3. **Preconditioned CG with circulant preconditioner**: $O(n \log n)$ per iteration, typically $O(1)$ iterations for well-conditioned Toeplitz

**Eigenvalue Bounds via Circulant/Skew-Circulant (Huckle 1992):**

From the circulant+skew-circulant splitting $T_n = C_a + S_b$ (trick 032), eigenvalue bounds for $T_n$ can be derived:

$$
\mu_1(C_e) + \mu_1(S_e) - t_0 \leq \mu_1(T_n) \leq \min\{\mu_1(S_e) + x_S^T C_e x_S,\; \mu_1(C_e) + x_C^T S_e x_C\} - t_0
$$

where $\mu_1$ denotes the minimum eigenvalue, $C_e$ and $S_e$ are the optimal circulant and skew-circulant approximations, and $x_S$, $x_C$ are their corresponding eigenvectors. These bounds are computable in $O(n \log n)$.

**Combined Lanczos Algorithm (Huckle 1992):**

Using eigenvectors of $C$ and $S$ as starting vectors for the Lanczos algorithm applied to $T_n^{-1}$:

1. Compute smallest eigenvalues and eigenvectors of $C$ and $S$: $O(n \log n)$
2. Form Rayleigh quotients $\lambda_r = \lambda_{\min}(U_r^T T U_r)$, $\lambda_a = \lambda_{\min}(U_a^T T U_a)$
3. Apply Lanczos to $A_r = (T_n - \mu_r I)^{-1}$ or $A_a = (T_n - \mu_a I)^{-1}$ with start vectors from step 1

This typically converges in **one Lanczos step** to sufficient accuracy for $\lambda_{\min}(T_n)$.

**Key Definitions:**

- $T_n$ — $n \times n$ Toeplitz matrix
- $\mathbf{x}, \mathbf{y}$ — GS vectors (first and last columns of $T_n^{-1}$)
- $L(\mathbf{v})$ — lower-triangular Toeplitz with first column $\mathbf{v}$
- $U(\mathbf{v})$ — upper-triangular Toeplitz with first row $\mathbf{v}^T$
- $C(\mathbf{v})$ — circulant matrix with first column $\mathbf{v}$
- $Z$ — cyclic downshift (permutation) matrix
- $\nabla_Z[A] = A - ZAZ^T$ — cyclic displacement operator
- $C_e, S_e$ — optimal circulant and skew-circulant approximations of $T_n$

## Complexity

| Operation | Dense inverse | GS formula solve | GS with circulant variant |
|-----------|-------------|------------------|--------------------------|
| Compute GS vectors | — | $O(n^2)$ (Levinson) | $O(n^2)$ (Levinson) |
| Store $T_n^{-1}$ | $O(n^2)$ | $O(n)$ (2 vectors) | $O(n)$ (2 vectors) |
| Apply $T_n^{-1}$ to vector | $O(n^2)$ | $O(n \log n)$ (4 FFTs) | $O(n \log n)$ (4 FFTs) |
| Eigenvalue bounds | $O(n^3)$ | $O(n \log n)$ | $O(n \log n)$ |

**Breakdown of the 4-FFT solve:**

1. $\mathbf{w}_1 = C(\mathbf{x}') \mathbf{b}$: FFT + multiply + IFFT = $O(n \log n)$
2. $\mathbf{w}_2 = L(\mathbf{x}) \mathbf{w}_1$: circulant embed + FFT + multiply + IFFT = $O(n \log n)$
3. $\mathbf{w}_3 = C(\bar{\mathbf{y}}') \mathbf{b}$: FFT + multiply + IFFT = $O(n \log n)$
4. $\mathbf{w}_4 = L(\bar{\mathbf{y}}) \mathbf{w}_3$: circulant embed + FFT + multiply + IFFT = $O(n \log n)$
5. $\mathbf{z} = (\mathbf{w}_2 - \mathbf{w}_4) / x_0$: $O(n)$

Total: ~8 FFTs of size $n$ (or $2n$ for the lower-triangular circulant embeddings).

**Memory:** $O(n)$ — only the two GS vectors ($2n$ scalars) plus workspace for FFTs.

## Applicability

- **Toeplitz system solving in sequence models**: When a Toeplitz token mixer needs to be inverted — e.g., for bidirectional inference, normalizing flows with Toeplitz Jacobians, or implicit layers — the GS formula provides $O(n \log n)$ inversion via 4 FFTs, compared to $O(n^2)$ for dense inversion or $O(n \log^2 n)$ for superfast direct solvers
- **Differentiable Toeplitz inverse layer**: The GS formula is differentiable: the GS vectors $\mathbf{x}, \mathbf{y}$ depend on the Toeplitz parameters, and their gradients can be computed via implicit differentiation of the Yule-Walker system. This enables using $T_n^{-1}$ as a learnable layer
- **Spectral analysis of learned Toeplitz mixers**: The eigenvalue bounds from the circulant/skew-circulant approximations (Huckle's Theorem 1) provide $O(n \log n)$ estimates of $\lambda_{\min}(T_n)$ and $\lambda_{\max}(T_n)$ — useful for monitoring conditioning during training
- **Circulant preconditioning for Toeplitz CG**: The GS formula shows that $T_n^{-1}$ is "almost circulant" (a sum of products involving circulants). This motivates using the circulant factor $C(\mathbf{x}')$ directly as a preconditioner, with spectrum clustered around 1 for well-conditioned Toeplitz matrices
- **Efficient Toeplitz log-determinant**: Combined with the Hutchinson trace estimator (trick 064), the GS formula enables stochastic estimation of $\log\det(T_n)$ via $\text{tr}(\log T_n) \approx \frac{1}{K}\sum_k \mathbf{z}_k^T \log(T_n) \mathbf{z}_k$, where each $\log(T_n) \mathbf{z}_k$ is computed via a polynomial of Toeplitz matvecs using the GS representation

## Limitations

- **$O(n^2)$ preprocessing**: Computing the GS vectors via Levinson recursion costs $O(n^2)$, which is the bottleneck. Superfast $O(n \log^2 n)$ algorithms exist (trick 124) but are more complex and less numerically stable
- **Requires nonsingularity and $x_0 \neq 0$**: The formula fails if $T_n$ is singular or if $x_0 = 0$ (the leading coefficient of the Yule-Walker solution). For near-singular Toeplitz matrices, the GS vectors amplify numerical errors
- **Not GPU-optimal for single solve**: The 4 sequential FFT operations have low arithmetic intensity (~0.5 FLOPs/byte each). For a single solve, the overhead of 8 FFTs may not beat a well-optimized dense solver on GPU for moderate $n$. The advantage appears for repeated solves with the same $T_n$ (amortizing the Levinson preprocessing)
- **Symmetric case only for Huckle eigenvalue bounds**: The eigenvalue bounds from the circulant+skew-circulant splitting require symmetric Toeplitz matrices. Nonsymmetric Toeplitz (as in causal token mixing) needs the non-symmetric Lanczos variant
- **No tensor core utilization**: All operations are FFT-based, which cannot use tensor cores on modern GPUs. Dense Toeplitz solvers using GEMM-based approaches may be competitive for moderate $n$ on tensor-core-rich hardware
- **Complex arithmetic for non-symmetric case**: The circulant eigenvalues are complex for non-symmetric Toeplitz matrices, requiring complex FFTs even when $T_n$ is real

## Implementation Notes

```python
import torch
import torch.fft as fft

def levinson_durbin(t, n):
    """Compute GS vectors via Levinson-Durbin recursion.

    Solves T_n x = e_1 for symmetric positive definite Toeplitz T_n.

    Args:
        t: (n,) Toeplitz coefficients [t_0, t_1, ..., t_{n-1}]

    Returns:
        x: (n,) first column of T_n^{-1}
        y: (n,) last column of T_n^{-1}
    """
    # Levinson recursion for symmetric PD Toeplitz
    x = torch.zeros(n, dtype=t.dtype, device=t.device)
    x[0] = 1.0 / t[0]

    if n == 1:
        return x, x.clone()

    # Forward recursion
    a = torch.zeros(n, dtype=t.dtype, device=t.device)  # reflection coefficients
    a[0] = -t[1] / t[0]
    x[0] = 1.0 / t[0]

    f = torch.zeros(n, dtype=t.dtype, device=t.device)  # forward predictor
    f[0] = 1.0
    f[1] = a[0]

    beta = t[0]
    alpha = t[0] + t[1] * a[0]

    for i in range(1, n - 1):
        # Compute reflection coefficient
        r = sum(t[j + 1] * f[i - j] for j in range(i + 1))
        a_new = -r / alpha

        # Update predictor
        f_new = torch.zeros_like(f)
        f_new[0] = 1.0
        for j in range(1, i + 1):
            f_new[j] = f[j] + a_new * f[i + 1 - j]
        f_new[i + 1] = a_new
        f = f_new

        alpha = alpha * (1 - a_new * a_new)

    # x = f / alpha (solution of T x = e_1)
    x[:n] = f[:n] / alpha

    # y is the reversed version for T y = e_n
    y = torch.flip(x, [0])

    return x, y


def gs_circulant_solve(t, b):
    """Solve Toeplitz system T_n z = b using Gohberg-Semencul formula
    with circulant variant.

    Args:
        t: (2n-1,) Toeplitz coefficients [t_{-(n-1)}, ..., t_0, ..., t_{n-1}]
        b: (n,) right-hand side

    Returns:
        z: (n,) solution
    """
    n = (len(t) + 1) // 2
    t_sym = t[n-1:]  # [t_0, t_1, ..., t_{n-1}]

    # Step 1: Compute GS vectors (O(n^2) via Levinson)
    x, y = levinson_durbin(t_sym, n)
    x0 = x[0]

    # Step 2: Build circulant version of x
    # x' = (x_0, 0, x_{n-1}, x_{n-2}, ..., x_1)
    x_circ = torch.zeros(n, dtype=t.dtype, device=t.device)
    x_circ[0] = x[0]
    x_circ[2:] = torch.flip(x[1:n-1], [0]) if n > 2 else torch.tensor([])

    # Build y_bar = (0, y_{n-1}, y_{n-2}, ..., y_1)
    y_bar = torch.zeros(n, dtype=t.dtype, device=t.device)
    y_bar[1:] = torch.flip(y[:n-1], [0])

    # Build circulant version of y_bar
    y_bar_circ = torch.zeros(n, dtype=t.dtype, device=t.device)
    y_bar_circ[0] = 0
    if n > 2:
        y_bar_circ[2:] = torch.flip(y_bar[1:n-1], [0])

    # Step 3: Apply GS formula via FFTs
    # z = (1/x0) * (L(x) C(x') b - L(y_bar) C(y_bar') b)

    def circulant_matvec(c_col, v):
        """Circulant matrix-vector product via FFT."""
        c_fft = fft.fft(c_col.to(torch.complex64))
        v_fft = fft.fft(v.to(torch.complex64))
        return fft.ifft(c_fft * v_fft).real

    def lower_toeplitz_matvec(first_col, v):
        """Lower-triangular Toeplitz matvec via circulant embedding."""
        m = len(v)
        # Embed in 2m circulant
        c = torch.zeros(2 * m, dtype=v.dtype, device=v.device)
        c[:m] = first_col
        v_pad = torch.zeros(2 * m, dtype=v.dtype, device=v.device)
        v_pad[:m] = v
        result = circulant_matvec(c, v_pad)
        return result[:m]

    # w1 = C(x') @ b, then L(x) @ w1
    w1 = circulant_matvec(x_circ, b)
    w2 = lower_toeplitz_matvec(x, w1)

    # w3 = C(y_bar') @ b, then L(y_bar) @ w3
    w3 = circulant_matvec(y_bar_circ, b)
    w4 = lower_toeplitz_matvec(y_bar, w3)

    # Combine
    z = (w2 - w4) / x0

    return z


def toeplitz_eigenvalue_bounds(t_sym):
    """Compute eigenvalue bounds for symmetric Toeplitz matrix
    using circulant + skew-circulant approximation (Huckle 1992).

    Args:
        t_sym: (n,) symmetric Toeplitz coefficients [t_0, t_1, ..., t_{n-1}]

    Returns:
        lower: lower bound on lambda_min(T_n)
        upper: upper bound on lambda_min(T_n) (tighter)
    """
    n = len(t_sym)

    # Optimal circulant approximation C_e
    c_e = torch.zeros(n, dtype=t_sym.dtype, device=t_sym.device)
    c_e[0] = t_sym[0]
    for k in range(1, n):
        c_e[k] = (t_sym[k] + t_sym[n - k]) / 2 if n - k < n else t_sym[k]

    # Optimal skew-circulant approximation S_e
    s_e = torch.zeros(n, dtype=t_sym.dtype, device=t_sym.device)
    s_e[0] = t_sym[0]
    for k in range(1, n):
        s_e[k] = (t_sym[k] - t_sym[n - k]) / 2 if n - k < n else t_sym[k]

    # Eigenvalues of C_e (via FFT) and S_e (via twisted FFT)
    c_eig = fft.fft(c_e.to(torch.complex64)).real
    mu1_Ce = c_eig.min().item()

    k = torch.arange(n, device=t_sym.device, dtype=torch.float32)
    twist = torch.exp(1j * torch.pi * k / n)
    s_eig = fft.fft(s_e.to(torch.complex64) * twist).real
    mu1_Se = s_eig.min().item()

    t0 = t_sym[0].item()

    # Lower bound (Theorem 1, Huckle 1992)
    lower = mu1_Ce + mu1_Se - t0

    return lower
```

## References

- Gohberg, I.C. & Semencul, A.A. "On the inversion of finite Toeplitz matrices and their continuous analogues" Mat. Issled. 7(2):201-223, 1972
- Ammar, G. & Gader, P. "A Variant of the Gohberg-Semencul Formula Involving Circulant Matrices" SIAM J. Matrix Anal. Appl. 12(3):534-540, 1991
- Huckle, T. "Circulant and skewcirculant matrices for solving Toeplitz matrix problems" SIAM J. Matrix Anal. Appl. 13(3):767-777, 1992
- Heinig, G. & Rost, K. "Algebraic Methods for Toeplitz-like Matrices and Operators" Birkhäuser, 1984
- Chan, R.H. & Ng, M.K. "Conjugate gradient methods for Toeplitz systems" SIAM Review 38(3):427-482, 1996
- Bini, D. & Pan, V. "Polynomial and Matrix Computations, Vol. 1: Fundamental Algorithms" Birkhäuser, 1994

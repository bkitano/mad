# 067: $k$-Diagonal Circulant Recursive Inverse

**Category**: decomposition
**Gain type**: efficiency
**Source**: Wang, Yu & Wang "Efficient Calculations for Inverse of $k$-diagonal Circulant Matrices and Cyclic Banded Matrices" (Applied Mathematics and Computation, 2024)
**Paper**: [papers/k-diagonal-circulant-inverse.pdf]
**Documented**: 2026-02-15

## Description

A $k$-diagonal circulant matrix ($k$-CM) is a sparse circulant matrix with only $k$ non-zero diagonals, where each row is a cyclic shift of the previous. This trick provides a **recursive formula** that uniquely determines the inverse of a $k$-CM in $O(k^3 \log n + k^4) + O(kn)$ operations, dramatically faster than both the naive $O(n^3)$ inversion and the standard FFT-based $O(n \log n)$ method when $k \ll n$.

The key insight is that the inverse of a circulant matrix is itself circulant (Proposition 2 in the paper), so one only needs to compute a **single row** (or column) of the inverse — all $n$ entries of that row — to fully characterize $M^{-1}$. The algorithm decomposes the problem into:

1. **Determinant computation** in $O(k^3 \log n)$ via matrix exponentiation of a $(k-1) \times (k-1)$ companion-like matrix $T$ raised to the power $n - 2k + 2$, using repeated squaring.
2. **First $k-1$ inverse entries** via algebraic cofactors of a structured block matrix, computable in $O(k^4)$.
3. **Remaining $n - k + 1$ entries** via a **linear recurrence** that propagates in $O(kn)$.

The crucial advantage over FFT-based inversion is that this algorithm **works over finite fields** (e.g., $\mathbb{F}_p$, $\text{GF}(2^m)$), where FFT is not applicable. It also avoids the numerical instability of complex arithmetic in floating-point FFT. For neural network applications, the sparse banded circulant structure naturally arises in depthwise/dilated convolutions with periodic boundary conditions, and the recursive inverse provides an efficient way to compute exact gradients or preconditioners for such layers.

The paper also extends the algorithm to $k$-diagonal **cyclic banded matrices** ($k$-CBM), where entries vary row-to-row (unlike circulant matrices). The $k$-CBM inverse requires $O(k^3 n + k^5) + O(kn^2)$, which is faster than Gaussian elimination's $O(kn^2)$ when the rapid representation (without explicitly computing all $n^2$ entries) suffices.

## Mathematical Form

**$k$-Diagonal Circulant Matrix ($k$-CM):**

An $n \times n$ $k$-CM has $k$ non-zero elements per row, cyclically shifted:

$$
M = \begin{bmatrix} x_k & 0 & \cdots & 0 & x_1 & x_2 & \cdots & x_{k-1} \\
x_{k-1} & x_k & 0 & \cdots & 0 & x_1 & \cdots & x_{k-2} \\
\vdots & \ddots & \ddots & & & & \ddots & \vdots \\
x_1 & x_2 & \cdots & x_k & 0 & \cdots & 0 & 0 \\
0 & x_1 & \cdots & x_{k-1} & x_k & \cdots & 0 & 0 \\
\vdots & & \ddots & & & \ddots & & \vdots \\
0 & 0 & \cdots & 0 & x_1 & x_2 & \cdots & x_k
\end{bmatrix}_{n \times n}
$$

where $x_1, x_2, \ldots, x_k$ are the $k$ non-zero elements and $x_k \neq 0$.

**Determinant via Matrix Exponentiation (Proposition 1):**

The determinant of the $k$-CM can be computed as:

$$
|M| = (-1)^{(k-i)(n-k+i)} x_k^{n-k+1} \left| D - C A^{-1} T^{n-2k+2} B \right|
$$

where:

$$
A = \begin{bmatrix} x_k & 0 & \cdots & 0 \\
x_{k-1} & x_k & & 0 \\
\vdots & & \ddots & \vdots \\
x_2 & \cdots & x_{k-1} & x_k
\end{bmatrix}_{(k-1)^2}, \quad
B = \begin{bmatrix} x_1 & x_2 & \cdots & x_{k-1} \\
0 & x_1 & \cdots & x_{k-2} \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & x_1
\end{bmatrix}_{(k-1)^2}
$$

$$
T = \begin{bmatrix} -\frac{x_{k-1}}{x_k} & 1 & 0 & \cdots & 0 \\
-\frac{x_{k-2}}{x_k} & 0 & 1 & \cdots & 0 \\
\vdots & & & \ddots & \vdots \\
-\frac{x_1}{x_k} & 0 & \cdots & 0 & 0
\end{bmatrix}_{(k-1)^2}
$$

The matrix power $T^{n-2k+2}$ is computed via **repeated squaring**:

$$
T^m = \begin{cases} T^{m/2} \cdot T^{m/2} & \text{if } m \text{ mod } 2 = 0 \\
T^{(m-1)/2} \cdot T^{(m-1)/2} \cdot T & \text{if } m \text{ mod } 2 = 1
\end{cases}
$$

This reduces the cost of the $(k-1) \times (k-1)$ matrix power from $O(k^3 n)$ to $O(k^3 \log n)$.

**Inverse via Adjugate (First $k-1$ entries):**

Since $M^{-1}$ is circulant, it suffices to find the first column $\mathbf{y} = (y_1, y_2, \ldots, y_n)^\top$ of $M^{-1}$. The first $k-1$ entries are obtained via algebraic cofactors:

$$
y_i = \frac{(-1)^{(i+1)+k(n-2k+2)} x_k^{n-2k+2}}{|M|} \begin{vmatrix} T^{n-2k+2} A_i & B \\ C_i & D \end{vmatrix} \quad (i = 1, 2, \ldots, k-1)
$$

where $A_i$ and $C_i$ are the matrices $A$ and $C$ with the $i$-th column removed.

**Inverse via Linear Recurrence (Remaining entries):**

The entries $y_k, y_{k+1}, \ldots, y_n$ satisfy the linear recurrence:

$$
y_i = -\frac{x_1 y_{i-k+1} + x_2 y_{i-k+2} + x_3 y_{i-k+3} + \cdots + x_{k-1} y_{i-1}}{x_k} \quad (i = k, k+1, \ldots, n)
$$

This propagates forward using only the $k$ non-zero coefficients, requiring $O(k)$ work per entry and $O(kn)$ total.

**Key Definitions:**

- $n$ — matrix dimension
- $k$ — number of non-zero diagonals (bandwidth)
- $x_1, \ldots, x_k$ — the $k$ non-zero elements defining the $k$-CM
- $T \in \mathbb{R}^{(k-1) \times (k-1)}$ — companion-like transition matrix
- $\mathbf{y} \in \mathbb{R}^n$ — first column of $M^{-1}$ (determines the full inverse since $M^{-1}$ is circulant)

## Complexity

| Operation | Naive (LU) | FFT-based | Recursive (this trick) |
|-----------|-----------|-----------|----------------------|
| Determinant | $O(n^3)$ | $O(n \log n)$ | $O(k^3 \log n + k^4)$ |
| Rapid representation | — | — | $O(k^3 \log n + k^4)$ |
| Full inverse | $O(n^3)$ | $O(n \log n)$ | $O(k^3 \log n + k^4) + O(kn)$ |
| Finite field support | Yes | **No** | **Yes** |

**For $k$-diagonal cyclic banded matrices ($k$-CBM):**

| Operation | Gaussian Elimination | Recursive (this trick) |
|-----------|---------------------|----------------------|
| Rapid representation | — | $O(k^3 n + k^5)$ |
| Full inverse | $O(kn^2)$ | $O(k^3 n + k^5) + O(kn^2)$ |

**When is this faster than FFT?**

The recursive method wins when $k \ll n$:
- For $k = O(1)$ (constant bandwidth): $O(\log n + n)$ vs $O(n \log n)$ — comparable
- For $k = O(\sqrt[3]{n})$: $O(n \log n)$ for both — crossover point
- The real advantage is (a) finite field applicability and (b) the "rapid representation" that avoids materializing all $n^2$ entries

**Memory:** $O(k^2)$ for the transition matrix $T$ plus $O(n)$ for the inverse column, totaling $O(k^2 + n)$.

## Applicability

- **Depthwise convolutions with periodic padding**: Neural network layers using circular/periodic boundary conditions produce $k$-CM weight matrices, where $k$ is the kernel size. The recursive inverse enables efficient computation of exact Jacobians or preconditioners for such layers
- **Circular/cyclic linear systems**: Any system $M\mathbf{x} = \mathbf{b}$ where $M$ is a banded circulant matrix (e.g., finite difference discretizations of PDEs with periodic boundary conditions) can be solved via this inverse
- **Preconditioning for iterative solvers**: The rapid representation (without full inverse materialization) can serve as a preconditioner for Krylov methods applied to circulant-plus-perturbation systems
- **Structured SSMs with banded transitions**: State space models with banded state transition matrices that have circulant structure can use this for exact inverse computation in the recurrence
- **Finite-field and cryptographic applications**: Unlike FFT-based methods, works over $\mathbb{F}_p$ and $\text{GF}(2^m)$, enabling efficient computation in lattice-based HE schemes and coding theory
- **Signal processing with cyclic convolutions**: Deconvolution of bandlimited circular convolutions benefits from the $O(kn)$ inverse construction

## Limitations

- **Requires $x_k \neq 0$**: The largest-index non-zero element must be non-zero for the recurrence to be well-defined (division by $x_k$ in the recursion). This is a non-singularity condition
- **Not faster than FFT for dense circulant**: When $k = n$ (full circulant, not banded), the method degenerates and FFT-based diagonalization is superior
- **Numerical stability of recurrence**: The linear recurrence $y_i = -\frac{1}{x_k}\sum x_j y_{i-k+j}$ can accumulate rounding errors for large $n$ in floating point. The FFT approach may be more stable for floating-point computation
- **Matrix dimension must exceed bandwidth**: Requires $n \geq 2k - 2$ for the algorithm to apply (otherwise the matrix is too small relative to its bandwidth)
- **Limited to circulant structure**: The key trick (inverse of circulant = circulant, so only one row needed) does not extend to general banded matrices. The $k$-CBM extension loses this and requires $O(kn^2)$
- **GPU parallelism**: The sequential recurrence in the final step ($y_k, \ldots, y_n$) is inherently sequential with $O(k)$ work per step, limiting GPU parallelism. A parallel scan formulation would be needed for GPU efficiency

## Implementation Notes

```python
import numpy as np

def k_cm_inverse(x, n):
    """Compute the inverse of a k-diagonal circulant matrix.

    Uses the recursive algorithm from Wang et al. (2024):
    1. Determinant via matrix exponentiation: O(k^3 log n)
    2. First k-1 entries via cofactors: O(k^4)
    3. Remaining entries via linear recurrence: O(kn)

    Args:
        x: array of k non-zero elements [x_1, x_2, ..., x_k]
        n: matrix dimension

    Returns:
        y: first column of M^{-1} (determines full inverse since M^{-1} is circulant)
    """
    k = len(x)
    assert n >= 2 * k - 2, f"Need n >= 2k-2, got n={n}, k={k}"
    assert x[-1] != 0, "x_k must be non-zero"

    # Step 1: Build transition matrix T (k-1 x k-1)
    km1 = k - 1
    T = np.zeros((km1, km1))
    for i in range(km1):
        T[i, 0] = -x[km1 - 1 - i] / x[km1]  # -x_{k-1-i}/x_k
        if i + 1 < km1:
            T[i, i + 1] = 1.0  # superdiagonal = 1

    # Step 2: Matrix power T^(n-2k+2) via repeated squaring
    power = n - 2 * k + 2
    T_pow = matrix_power(T, power)

    # Step 3: Build A, B, C, D matrices for determinant
    A = np.zeros((km1, km1))
    for i in range(km1):
        for j in range(i + 1):
            A[i, j] = x[k - 1 - (i - j)]  # lower triangular Toeplitz from x

    B = np.zeros((km1, km1))
    for i in range(km1):
        for j in range(i, km1):
            B[i, j] = x[j - i]  # upper triangular Toeplitz from x

    # Determinant: |M| = (-1)^... * x_k^(n-k+1) * |D - C A^{-1} T^{n-2k+2} B|
    # (simplified for circulant case)
    det_M = compute_determinant_k_cm(x, n, T_pow, A, B)

    # Step 4: First k-1 entries of inverse via cofactors
    y = np.zeros(n)
    y[:km1] = compute_first_entries(x, n, T_pow, A, B, det_M)

    # Step 5: Remaining entries via linear recurrence
    # y_i = -(x_1*y_{i-k+1} + x_2*y_{i-k+2} + ... + x_{k-1}*y_{i-1}) / x_k
    for i in range(km1, n):
        s = 0.0
        for j in range(km1):
            idx = (i - km1 + j) % n  # cyclic indexing
            s += x[j] * y[idx]
        y[i] = -s / x[km1]

    return y


def matrix_power(M, p):
    """Compute M^p via repeated squaring. O(d^3 log p) for d x d matrix."""
    d = M.shape[0]
    result = np.eye(d)
    base = M.copy()
    while p > 0:
        if p % 2 == 1:
            result = result @ base
        base = base @ base
        p //= 2
    return result


def k_cm_matvec(x, y_col, b):
    """Multiply k-CM (defined by x) by vector b using the inverse column.

    Since M^{-1} is circulant, M^{-1} @ b = IFFT(FFT(y_col) * FFT(b)).
    This is O(n log n) after the O(kn) inverse computation.
    """
    n = len(b)
    Y = np.fft.fft(y_col)
    B = np.fft.fft(b)
    return np.fft.ifft(Y * B).real


# For neural network applications: sparse circulant layer
import torch

class SparseCirculantLayer(torch.nn.Module):
    """Sparse (k-diagonal) circulant linear layer.

    Only k parameters per circulant block instead of n.
    Useful for depthwise convolutions with periodic padding.
    """

    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k
        # Only k learnable parameters
        self.x = torch.nn.Parameter(torch.randn(k) / n**0.5)

    def forward(self, v):
        """v: (batch, n). Returns k-CM @ v via FFT."""
        # Build full defining vector (sparse: only k non-zero)
        c = torch.zeros(self.n, device=v.device)
        # Place k elements at the right circulant positions
        c[:self.k] = self.x
        # Circulant multiply via FFT
        c_fft = torch.fft.fft(c)
        v_fft = torch.fft.fft(v, dim=-1)
        return torch.fft.ifft(c_fft * v_fft, dim=-1).real
```

## References

- Wang, C., Yu, H. & Wang, C. "Efficient Calculations for Inverse of $k$-diagonal Circulant Matrices and Cyclic Banded Matrices" Applied Mathematics and Computation, 2024. arXiv:2403.05048
- Davis, P.J. "Circulant Matrices" Wiley, 1979
- Serra-Capizzano, S. & Tablino-Possio, C. "Multigrid methods for multilevel circulant matrices" SIAM J. Scientific Computing 26, 55-85, 2004
- Gray, R.M. et al. "Toeplitz and circulant matrices: A review" Foundations and Trends in Communications and Information Theory 2, 155-239, 2006
- Chen, X.B. "A fast algorithm for computing the determinants of banded circulant matrices" Applied Mathematics and Computation 229, 201-207, 2014
- Rózsa, P. et al. "On periodic block-tridiagonal matrices" Linear Algebra and Its Applications 167, 35-52, 1992

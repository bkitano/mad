# 081: Neumann Series Approximate Inverse

**Category**: approximation
**Gain type**: efficiency
**Source**: Classical numerical linear algebra; Dubois et al. (1979); Gustafsson et al. (2017); Sao (2025)
**Paper**: [papers/neumann-series-approximate-inverse.pdf] (local path to downloaded PDF)
**Documented**: 2026-02-15

## Description

The truncated Neumann series provides an explicit, inverse-free approximation to $(I - A)^{-1}$ using only matrix-matrix products. For a matrix $A$ with spectral radius $\rho(A) < 1$, the geometric series $(I - A)^{-1} = \sum_{j=0}^{\infty} A^j$ converges, and truncating at $k$ terms gives the polynomial approximation $S_k(A) = I + A + A^2 + \cdots + A^{k-1}$. This avoids explicit matrix inversion entirely, replacing the $O(n^3)$ inverse with a sequence of matrix-matrix multiplications that can be accelerated via radix-kernel decompositions. The technique is directly applicable to resolvent computation: setting $B = I - A$ (e.g., $A = \Lambda / z$ for a diagonal SSM), the resolvent $(zI - \Lambda)^{-1}$ can be approximated without forming any inverse. Recent work by Sao (2025) shows that radix-$m$ kernels reduce the required number of matrix products from the naive $k-1$ to as few as $1.54 \log_2 k$, a 25% improvement over binary splitting.

## Mathematical Form

**Core Operation:**

For $M = I - A$ with $\rho(A) < 1$:

$$
M^{-1} = (I - A)^{-1} \approx S_k(A) = \sum_{j=0}^{k-1} A^j = I + A + A^2 + \cdots + A^{k-1}
$$

**Radix-$m$ Splitting:**

The key acceleration uses the factorization:

$$
S_{mn}(A) = S_n(A) \cdot T_m(A^n)
$$

where $T_m(B) = I + B + \cdots + B^{m-1}$ is a radix kernel. Each radix step computes:

- $A^n$ (power extraction, reusing previous computations)
- $T_m(A^n)$ (the kernel, costing $\mu_m$ products)
- $S_n \cdot T_m(A^n)$ (concatenation, 1 product)

**Binary Splitting ($m = 2$):**

$$
S_{2n}(A) = S_n(A) \cdot (I + A^n)
$$

Cost: 2 products per doubling (one for $A^{2n} = A^n \cdot A^n$, one for concatenation). Total: $2\log_2 k$ products.

**Radix-9 Kernel (Exact, 3 products):**

$$
T_9(B) = I + B + \frac{767}{800}U + \frac{15}{32}V + P
$$

where $U = B^2$, $V = U(B + 2U) = B^3 + 2B^4$, $P = (0.15B + 2U + V)(0.275B - 0.125U + 0.25V)$.

Total: $5 \log_9 k \approx 1.58 \log_2 k$ products (21% fewer than binary).

**Iterative Residual Framework:**

Maintain approximate inverse $Y_n \approx M^{-1}$ and residual $R_n = I - MY_n$:

$$
Y_{n+1} = Y_n \cdot f(R_n), \qquad R_{n+1} = I - M Y_{n+1}
$$

where $f$ is an approximate radix-$m$ kernel. The error map $E(z) = 1 - (1-z)f(z)$ satisfies $E^{[n]}(z) = c^{(m^n - 1)/(m-1)} z^{m^n} + O(z^{m^n + 1})$, showing spillover is pushed to degree $\geq m^n$ regardless of kernel coefficient $c$.

**Key Definitions:**

- $A \in \mathbb{R}^{d \times d}$ --- matrix with $\rho(A) < 1$
- $S_k(A)$ --- truncated Neumann series of degree $k-1$
- $T_m(B)$ --- radix-$m$ kernel (degree $m-1$ polynomial)
- $\mu_m$ --- minimum products to evaluate $T_m$
- $C(m) = \mu_m + 2$ --- cost per radix step (kernel + power + concatenation)

## Complexity

| Operation | Naive | Binary Splitting | Radix-9 | Radix-15 |
|-----------|-------|-----------------|---------|----------|
| Products for $S_k$ | $k - 1$ | $2\log_2 k$ | $1.58 \log_2 k$ | $1.54 \log_2 k$ |
| Coefficient | --- | $2.00$ | $1.58$ | $1.54$ |

**Per product cost:** $O(d^2)$ for dense GEMM (GPU-friendly)

**Total cost for $k$-term series:** $O(d^2 \cdot C(m) \cdot \log_m k)$ vs $O(d^3)$ for exact inverse

**Memory:** $O(d^2)$ for intermediate matrices (same as storing $A$)

## Applicability

- **Resolvent approximation in SSMs**: For DPLR matrices where $A = \Lambda - PQ^*$, the Woodbury identity reduces to diagonal resolvents; but when the diagonal is ill-conditioned near certain frequencies, a Neumann series on the perturbation $PQ^*D_z$ provides a numerically smoother alternative
- **Polynomial preconditioning**: $S_k(A)$ serves as an approximate preconditioner for iterative solvers (CG, GMRES) operating on structured matrices in SSM training
- **Log-determinant estimation**: Trace of $S_k(A)$ approximates $-\log\det(I - A)$ without forming the inverse, useful for normalizing flows and variational inference in neural network layers
- **Massive MIMO / signal detection**: Truncated Neumann series avoids cubic-cost matrix inversion in linear detectors, reducing complexity from $O(n^3)$ to $O(n^2 k)$
- **GPU-friendly**: Computation is dominated by matrix-matrix products (GEMM), which saturate GPU tensor cores efficiently

## Limitations

- Requires $\rho(A) < 1$ for convergence; does not apply when the spectral radius is large (must precondition or rescale first)
- Accuracy depends on $k$ and the spectral radius: for $\rho(A)$ close to 1, many terms are needed before the approximation is useful
- Approximate radix kernels (radix-15) introduce spillover terms at higher degrees, causing a residual floor of $\approx 10^{-6}$ instead of machine precision
- The leading coefficient ($\sim 1.54$) cannot be reduced sublinearly in $\log_2 m$ with current kernel constructions
- For small $d$ (e.g., SSM state dimensions $N = 64$), the overhead of multiple GEMMs may exceed the cost of a direct $O(N^3)$ inverse

## Implementation Notes

```python
# Binary splitting (standard approach)
def neumann_binary(A, k):
    """Compute S_k(A) = I + A + ... + A^{k-1} via binary splitting."""
    # k must be a power of 2
    S = I + A           # S_2
    A_pow = A @ A       # A^2
    while A_pow degree < k:
        S = S @ (I + A_pow)   # S_{2n} = S_n * (I + A^n)
        A_pow = A_pow @ A_pow # A^{2n}
    return S

# Radix-9 kernel (exact, 3 products for kernel)
def radix9_kernel(B):
    """Compute T_9(B) = sum_{j=0}^{8} B^j using 3 products."""
    U = B @ B                                    # B^2
    V = U @ (B + 2 * U)                         # B^3 + 2*B^4
    P = (0.15*B + 2*U + V) @ (0.275*B - 0.125*U + 0.25*V)
    return I + B + (767/800)*U + (15/32)*V + P

# Iterative approximate inverse
def neumann_inverse_approx(M, k, radix_kernel):
    """Approximate M^{-1} via Neumann iteration with radix kernel."""
    A = I - M  # or suitable splitting
    Y = I      # initial guess
    R = A      # residual = I - M*Y = A
    for _ in range(int(np.ceil(np.log(k) / np.log(m)))):
        Y = Y @ radix_kernel(R)
        R = I - M @ Y  # updated residual
    return Y
```

## References

- Sao, P. (2025). Fast Evaluation of Truncated Neumann Series by Low-Product Radix Kernels. arXiv:2602.11843.
- Gustafsson, O. et al. (2017). Approximate Neumann Series or Exact Matrix Inversion for Massive MIMO? Proc. IEEE ARITH.
- Dubois, P. F., Greenbaum, A., Rodrigue, G. H. (1979). Approximating the Inverse of a Matrix for Use in Iterative Algorithms on Vector Processors. Computing 22(3), 257-268.
- Lei, S.-F., Nakamura, T. (1992). On the Fast Evaluation of Parallel ARMA Filters and Related Matrix Power Sums. IEEE Trans. CAS-I.
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. 2nd ed., SIAM.

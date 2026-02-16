# 206: CUML Toeplitz Factor Circulant Inversion

**Category**: decomposition
**Gain type**: efficiency
**Source**: Zheng & Jiang "Quasi-cyclic displacement and inversion decomposition of a quasi-Toeplitz matrix" (AIMS Mathematics, 2022); Gohberg & Olshevsky "Circulants, displacements and decompositions of matrices" (1992); Ammar & Gader (1991)
**Paper**: [papers/quasi-cyclic-displacement-quasi-toeplitz.pdf]
**Documented**: 2026-02-16

## Description

A **Column Upper-Minus-Lower (CUML) Toeplitz matrix** is a near-Toeplitz matrix where the standard Toeplitz structure is perturbed by subtracting consecutive diagonal entries below the main diagonal. These matrices arise naturally in queueing theory (Markov chain transition matrices for waiting times) and in discretizations of differential equations with periodic-plus-shift boundary conditions.

The key trick is a **(1, −1)-cyclic displacement** operator — a generalization of the standard cyclic displacement used in the Gohberg-Semencul formula (trick 175) — that uses a *factor circulant* shift matrix $\Phi_{1,-1}$ instead of the standard cyclic shift $Z$. This displacement operator has rank 2 for CUML Toeplitz matrices, enabling the inverse to be decomposed as a **sum of products of factor (1, −1)-circulants**: specifically, "row first-minus-last right circulants" (RFMLRcircfc) and "row skew first-plus-last right circulants" (RSFPLRcircfc).

The factor circulants are diagonalizable by **twisted FFT** (FFT with phase-shifted inputs), meaning each factor circulant matvec costs $O(n \log n)$ via a single FFT with a diagonal phase pre-multiplication. The inverse is expressed as a sum of $\leq 2$ products of factor circulants, giving $O(n \log n)$ matvec application — the same asymptotic cost as the standard Gohberg-Semencul formula but for a broader class of near-Toeplitz matrices.

This extends trick 175 (Gohberg-Semencul circulant Toeplitz inverse) from standard Toeplitz to CUML Toeplitz, and differs from trick 028 (circulant cycle decomposition) which decomposes arbitrary matrices, not inverses specifically.

## Mathematical Form

**CUML Toeplitz Matrix:**

$$
T_{CUML} = \begin{pmatrix} t_0 & t_{-1} & \cdots & t_{2-n} & t_{1-n} \\ t_1 & t_0 - t_1 & \ddots & \ddots & t_{2-n} \\ t_2 & t_1 - t_2 & \ddots & \ddots & \vdots \\ \vdots & \vdots & \ddots & \ddots & t_{-1} \\ t_{n-1} & t_{n-2} - t_{n-1} & \cdots & t_1 - t_2 & t_0 - t_1 \end{pmatrix}_{n \times n}
$$

with entries:

$$
t_{ij} = \begin{cases} t_{i-j}, & j = 1 \text{ or } j > i \\ t_{i-j} - t_{i-j+1}, & 2 \leq j \leq i \end{cases}
$$

When $t_{1-n} = t_1, t_{2-n} = t_2, \ldots, t_{-1} = t_{n-1}$, $T_{CUML}$ reduces to a row first-minus-last right circulant matrix.

**(1, −1)-Cyclic Displacement Operator:**

$$
\nabla_{1,-1}(A) = A - \Phi_{1,-1} A \Phi_{1,-1}^{-1}
$$

where $\Phi_{1,-1}$ is the (1, −1)-cyclic lower shift matrix with first row $(0, \ldots, 0, 1)$:

$$
\Phi_{1,-1} = \begin{pmatrix} 0 & 0 & \cdots & 0 & 1 \\ 1 & -1 & & & 0 \\ 0 & 1 & -1 & \ddots & \vdots \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & 1 & -1 \end{pmatrix}
$$

**(1, −1)-Cyclic Displacement of $T_{CUML}$:**

$$
\nabla_{1,-1}(T_{CUML}) = \mathbf{x} \cdot \mathbf{e}_0^T + \mathbf{e}_0 \cdot \mathbf{z}^T
$$

where $\mathbf{x} = (\beta, t_1 - t_{1-n}, \ldots, t_{n-1} - t_{-1})^T$, $\mathbf{z}^T = (-\beta, t_{-1} - t_{n-1}, \ldots, t_{1-n} - t_1)$, and $\beta$ is an arbitrary complex number. The displacement rank $\tau = \text{rank}(\nabla_{1,-1}(T_{CUML})) = 2$.

**Displacement Rank Inheritance Under Inversion:**

$$
\nabla_{1,-1}(A) = -A \cdot \nabla_{1,-1}(A^{-1}) \cdot \Phi_{1,-1} A \Phi_{1,-1}^{-1}
$$

so $\text{rank}(\nabla_{1,-1}(A)) = \text{rank}(\nabla_{1,-1}(A^{-1}))$. If $A$ has low displacement rank, so does $A^{-1}$.

**Factor (1, −1)-Circulant Matrices:**

A Row First-Minus-Last Right Circulant with first column $\mathbf{w} = (w_0, w_1, \ldots, w_{n-1})^T$:

$$
\text{RFMLRcircfc}(\mathbf{w}) = \begin{pmatrix} w_0 & w_{n-1} & w_{n-2} & \cdots & w_1 \\ w_1 & w_0 - w_1 & \ddots & \ddots & \vdots \\ w_2 & w_1 - w_2 & \ddots & \ddots & w_{n-2} \\ \vdots & \vdots & \ddots & \ddots & w_{n-1} \\ w_{n-1} & w_{n-2} - w_{n-1} & \cdots & w_1 - w_2 & w_0 - w_1 \end{pmatrix}
$$

A Row Skew First-Plus-Last Right Circulant with first column $\mathbf{w}$:

$$
\text{RSFPLRcircfc}(\mathbf{w}) = \begin{pmatrix} w_0 & -w_{n-1} & -w_{n-2} & \cdots & -w_1 \\ w_1 & w_0 - w_1 & \ddots & \ddots & \vdots \\ w_2 & w_1 - w_2 & \ddots & \ddots & -w_{n-2} \\ \vdots & \vdots & \ddots & \ddots & -w_{n-1} \\ w_{n-1} & w_{n-2} - w_{n-1} & \cdots & w_1 - w_2 & w_0 - w_1 \end{pmatrix}
$$

**Key Properties:**
- $\text{RFMLRcircfc}(\mathbf{w}) \cdot \text{RFMLRcircfc}(\mathbf{a}) = \text{RFMLRcircfc}(\mathbf{a}) \cdot \text{RFMLRcircfc}(\mathbf{w})$ — commutative
- $\text{RSFPLRcircfc}(\mathbf{w}) \cdot \text{RSFPLRcircfc}(\mathbf{a}) = \text{RSFPLRcircfc}(\mathbf{a}) \cdot \text{RSFPLRcircfc}(\mathbf{w})$ — commutative
- $\text{RFMLRcircfr}(\mathbf{w}^T) = \text{RFMLRcircfc}(\tilde{\mathbf{w}})$ where $\tilde{\mathbf{w}} = (w_0, w_{n-1}, \ldots, w_1)^T$

**Main Inversion Theorem (Theorem 5):**

If $T_{CUML}$ is nonsingular and $\mathbf{c}_1, \mathbf{c}_2$ are solutions of $T_{CUML}\mathbf{c}_i = \mathbf{e}_i$ (for $i = 0$ and $i = \mathbf{e}_0$), and $\hat{\mathbf{d}}_i^T$ are solutions of $\hat{\mathbf{d}}_i^T T_{CUML} = \mathbf{e}_i^T \Phi_{1,-1}$, then:

$$
T_{CUML}^{-1} = \text{RFMLRcircfr}(\mathbf{y}_1^T) - \frac{1}{2} \sum_{i=1}^{2} \text{RFMLRcircfc}(\mathbf{c}_i) \cdot \text{RSFPLRcircfr}(\mathbf{d}_i^T)
$$

where $\mathbf{d}_i^T = \hat{\mathbf{d}}_i^T \cdot \Phi_{1,-1}^{-1}$ and $\text{RFMLRcircfr}(\mathbf{y}_1^T)$ is the row first-minus-last right circulant with first row $\mathbf{y}_1^T$.

**Alternative Form (Theorem 5, part ii):**

$$
T_{CUML}^{-1} = \text{RFMLRcircfc}(\mathbf{y}_2) - \frac{1}{2} \sum_{i=1}^{2} \text{RSFPLRcircfc}(\mathbf{c}_i) \cdot \text{RFMLRcircfr}(\mathbf{d}_i^T)
$$

**Simplified Form (Theorem 7) — when $T_{CUML}\mathbf{d} = (γ, t_{-n+1}, \ldots, t_{-1})^T$ is solvable:**

$$
T_{CUML}^{-1} = \frac{1}{2}\left[\text{RSFPLRcircfc}(\mathbf{e}_0 + \mathbf{d}) \cdot \text{RFMLRcircfc}(\mathbf{c}_2) + \text{RSFPLRcircfc}(\mathbf{c}_2) \cdot \text{RFMLRcircfc}(\mathbf{e}_0 - \mathbf{d})\right]
$$

**Key Definitions:**

- $T_{CUML} \in \mathbb{C}^{n \times n}$ — Column Upper-Minus-Lower Toeplitz matrix
- $\Phi_{1,-1}$ — (1, −1)-cyclic lower shift matrix
- $\nabla_{1,-1}(A) = A - \Phi_{1,-1} A \Phi_{1,-1}^{-1}$ — (1, −1)-cyclic displacement operator
- $\tau = \text{rank}(\nabla_{1,-1}(A))$ — (1, −1)-cyclic displacement rank
- $\text{RFMLRcircfc}(\mathbf{w})$ — row first-minus-last right circulant (factor circulant)
- $\text{RSFPLRcircfc}(\mathbf{w})$ — row skew first-plus-last right circulant (skew factor circulant)

## Complexity

| Operation | Dense inverse | Standard GS (Toeplitz) | CUML Factor Circulant |
|-----------|-------------|----------------------|----------------------|
| Compute solving vectors | $O(n^3)$ | $O(n^2)$ (Levinson) | $O(n^2)$ (solve 4 linear systems) |
| Store inverse | $O(n^2)$ | $O(n)$ (2 vectors) | $O(n)$ (4 vectors) |
| Apply $T^{-1}$ to vector | $O(n^2)$ | $O(n \log n)$ (4 FFTs) | $O(n \log n)$ (4–6 twisted FFTs) |

**Factor circulant matvec via twisted FFT:** A factor $(1, -1)$-circulant $\text{RFMLRcircfc}(\mathbf{w})$ is diagonalized by:

$$
\text{RFMLRcircfc}(\mathbf{w}) = F_n^{-1} \text{diag}(F_n P \mathbf{w}) F_n P
$$

where $P = \text{diag}(1, \omega, \omega^2, \ldots, \omega^{n-1})$ with $\omega = e^{i\pi/n}$ is the phase-twist diagonal. Each matvec costs one FFT + two element-wise multiplications + one IFFT = $O(n \log n)$.

**Total cost for inverse application:** $O(n \log n)$ — same as standard GS formula, but applicable to CUML Toeplitz matrices which are not plain Toeplitz.

**Memory:** $O(n)$ — store 4 vectors of length $n$ (the solving vectors $\mathbf{c}_1, \mathbf{c}_2, \mathbf{d}_1, \mathbf{d}_2$) plus FFT workspace.

## Applicability

- **Quasi-periodic token mixing layers**: CUML Toeplitz matrices model convolution-like operations with a systematic diagonal perturbation (the "minus-lower" part). This structure appears when a token mixer combines a shift-invariant component with a position-dependent decay, e.g., causal attention with exponential decay factors
- **Markov chain transition matrices in sequence models**: The CUML Toeplitz structure directly arises in queueing theory Markov chains; neural network layers that model state transitions with time-varying service rates produce this structure
- **Preconditioning for near-Toeplitz systems**: When a learned weight matrix is "close" to Toeplitz but not exactly Toeplitz, the CUML decomposition captures the systematic deviation and provides an $O(n \log n)$ preconditioner
- **Differentiable inverse layers**: The inversion formula is algebraically explicit and depends on 4 linear system solutions — these can be differentiated through via implicit differentiation, enabling use as a differentiable inverse layer
- **Connection to Hankel systems**: CUML Hankel matrices satisfy $H_{CUML} = T_{CUML} \hat{I}_n$ (where $\hat{I}_n$ is the reversal matrix), so the CUML Toeplitz inversion formula immediately yields CUML Hankel inversion via $H_{CUML}^{-1} = \hat{I}_n T_{CUML}^{-1}$

## Limitations

- **$O(n^2)$ preprocessing**: Computing the 4 solving vectors requires solving 4 $n \times n$ linear systems, which costs $O(n^2)$ via specialized Toeplitz-like solvers — the same bottleneck as the standard GS formula
- **Specialized matrix class**: Only applies to CUML Toeplitz matrices, not arbitrary near-Toeplitz perturbations. The diagonal perturbation must follow the specific "upper-minus-lower" pattern
- **Complex arithmetic**: The factor circulant diagonalization uses complex roots of unity ($\omega = e^{i\pi/n}$), requiring complex FFTs even for real CUML Toeplitz matrices
- **Not GPU-optimal for single application**: Like the standard GS formula, the 4–6 sequential FFT operations have low arithmetic intensity. The benefit is for repeated applications (e.g., iterative solvers using the inverse as preconditioner)
- **Numerical stability**: Forward stability is proven only for the scalar (non-block) case and requires the CUML Toeplitz matrix to be well-conditioned
- **No tensor core utilization**: All operations are FFT-based, which cannot exploit tensor cores on modern GPUs

## Implementation Notes

```python
import torch
import torch.fft as fft

def twisted_fft(x, omega_exp=None):
    """Compute FFT with phase twist for factor (1,-1)-circulant diagonalization.

    For a (1,-1)-circulant, the diagonalizing transform is F_n * P
    where P = diag(1, omega, omega^2, ..., omega^{n-1}) with omega = e^{i*pi/n}.

    Args:
        x: (n,) input vector (real or complex)
        omega_exp: precomputed phase factors (optional)

    Returns:
        (n,) complex vector = FFT(P * x)
    """
    n = x.shape[-1]
    if omega_exp is None:
        k = torch.arange(n, device=x.device, dtype=torch.float32)
        omega_exp = torch.exp(1j * torch.pi * k / n)

    x_twisted = x.to(torch.complex64) * omega_exp
    return fft.fft(x_twisted)


def twisted_ifft(X, omega_exp_conj=None):
    """Inverse twisted FFT: recover x from FFT(P * x).

    Args:
        X: (n,) frequency-domain vector
        omega_exp_conj: precomputed conjugate phase factors (optional)

    Returns:
        (n,) real or complex vector
    """
    n = X.shape[-1]
    if omega_exp_conj is None:
        k = torch.arange(n, device=X.device, dtype=torch.float32)
        omega_exp_conj = torch.exp(-1j * torch.pi * k / n)

    x_twisted = fft.ifft(X)
    return x_twisted * omega_exp_conj


def rfmlr_circfc_matvec(w, x, omega_exp=None, omega_exp_conj=None):
    """Apply RFMLRcircfc(w) @ x via twisted FFT.

    The factor (1,-1)-circulant with first column w is diagonalized by
    the twisted DFT, enabling O(n log n) matvec.

    Args:
        w: (n,) first column of RFMLRcircfc
        x: (n,) input vector

    Returns:
        y: (n,) result of RFMLRcircfc(w) @ x
    """
    n = w.shape[0]
    if omega_exp is None:
        k = torch.arange(n, device=w.device, dtype=torch.float32)
        omega_exp = torch.exp(1j * torch.pi * k / n)
    if omega_exp_conj is None:
        omega_exp_conj = omega_exp.conj()

    # Diagonalize: eigenvalues = twisted_fft(w)
    w_hat = twisted_fft(w, omega_exp)
    x_hat = twisted_fft(x, omega_exp)

    # Multiply in frequency domain
    y_hat = w_hat * x_hat

    # Inverse transform
    y = twisted_ifft(y_hat, omega_exp_conj)

    return y.real if x.is_floating_point() else y


def rsfplr_circfc_matvec(w, x, omega_exp=None, omega_exp_conj=None):
    """Apply RSFPLRcircfc(w) @ x via twisted FFT.

    The skew factor (−1,1)-circulant uses omega = e^{-i*pi/n} instead.

    Args:
        w: (n,) first column of RSFPLRcircfc
        x: (n,) input vector

    Returns:
        y: (n,) result
    """
    n = w.shape[0]
    k = torch.arange(n, device=w.device, dtype=torch.float32)
    # Skew factor uses negative twist
    omega_neg = torch.exp(-1j * torch.pi * k / n)
    omega_neg_conj = omega_neg.conj()

    w_hat = twisted_fft(w, omega_neg)
    x_hat = twisted_fft(x, omega_neg)
    y_hat = w_hat * x_hat
    y = twisted_ifft(y_hat, omega_neg_conj)

    return y.real if x.is_floating_point() else y


def cuml_toeplitz_inverse_apply(c1, c2, d1, d2, y1, b, use_simplified=False):
    """Apply T_CUML^{-1} @ b using the factor circulant decomposition.

    Uses Theorem 5: T^{-1} = RFMLRcircfr(y1^T)
      - (1/2) sum_{i=1}^{2} RFMLRcircfc(c_i) * RSFPLRcircfr(d_i^T)

    Args:
        c1, c2: (n,) solving vectors from T_CUML c_i = e_i
        d1, d2: (n,) solving vectors from d_i^T T_CUML = e_i^T Phi_{1,-1}
        y1: (n,) first row of RFMLRcircfr(y1^T) component
        b: (n,) input vector

    Returns:
        x: (n,) approximate T_CUML^{-1} @ b
    """
    n = b.shape[0]

    # Precompute phase factors
    k = torch.arange(n, device=b.device, dtype=torch.float32)
    omega_exp = torch.exp(1j * torch.pi * k / n)
    omega_exp_conj = omega_exp.conj()

    # Term 1: RFMLRcircfr(y1^T) @ b = RFMLRcircfc(y1_tilde) @ b
    y1_tilde = torch.zeros_like(y1)
    y1_tilde[0] = y1[0]
    y1_tilde[1:] = y1.flip(0)[:-1]

    term1 = rfmlr_circfc_matvec(y1_tilde, b, omega_exp, omega_exp_conj)

    # Term 2: sum of products
    # RSFPLRcircfr(d_i^T) @ b first, then RFMLRcircfc(c_i) @ result
    d1_tilde = torch.zeros_like(d1)
    d1_tilde[0] = d1[0]
    d1_tilde[1:] = d1.flip(0)[:-1]

    d2_tilde = torch.zeros_like(d2)
    d2_tilde[0] = d2[0]
    d2_tilde[1:] = d2.flip(0)[:-1]

    w1 = rsfplr_circfc_matvec(d1_tilde, b)
    prod1 = rfmlr_circfc_matvec(c1, w1, omega_exp, omega_exp_conj)

    w2 = rsfplr_circfc_matvec(d2_tilde, b)
    prod2 = rfmlr_circfc_matvec(c2, w2, omega_exp, omega_exp_conj)

    x = term1 - 0.5 * (prod1 + prod2)
    return x


# Example: Build and invert a CUML Toeplitz matrix
def demo_cuml_toeplitz():
    n = 8
    t = torch.randn(2 * n - 1)  # Toeplitz coefficients

    # Build CUML Toeplitz matrix
    T = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if j == 0 or j > i:
                T[i, j] = t[i - j + n - 1]
            else:
                T[i, j] = t[i - j + n - 1] - t[i - j + n]

    print(f"CUML Toeplitz matrix ({n}x{n}):")
    print(f"  Condition number: {torch.linalg.cond(T).item():.2f}")
    print(f"  Parameters needed for inverse: {4*n} (4 vectors)")
    print(f"  Dense inverse storage: {n*n}")
    print(f"  Matvec cost: O({n} log {n}) = O({n * int(torch.log2(torch.tensor(n*1.0)))}) vs O({n*n}) dense")
```

## References

- Zheng, Y. & Jiang, X. "Quasi-cyclic displacement and inversion decomposition of a quasi-Toeplitz matrix" AIMS Mathematics 7(7):11647-11662, 2022. doi:10.3934/math.2022649
- Gohberg, I. & Olshevsky, V. "Circulants, displacements and decompositions of matrices" J. Math. Anal. Appl. 68:730-743, 1992
- Ammar, G. & Gader, P. "A Variant of the Gohberg-Semencul Formula Involving Circulant Matrices" SIAM J. Matrix Anal. Appl. 12(3):534-540, 1991
- Gader, P. "Displacement operator based decompositions of matrices using circulants or other group matrices" Linear Algebra Appl. 139:111-131, 1990
- Kailath, T., Kung, S. & Morf, M. "Displacement ranks of matrices and linear equations" J. Math. Anal. Appl. 68:395-407, 1979
- Jiang, X.Y., Hong, K. "Algorithms for finding inverse of two patterned matrices over $\mathbb{Z}_p$" Abstr. Appl. Anal. 2014:1-6, 2014

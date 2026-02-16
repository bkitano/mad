# 086: Oscillatory Eigenvalue Stability

**Category**: stability
**Gain type**: efficiency
**Source**: Rusch and Rus (2025). Oscillatory State-Space Models (LinOSS). ICLR 2025.
**Paper**: papers/oscillatory-ssm.pdf
**Documented**: 2026-02-11

## Description

Parameterize the state-space model as a discretization of a second-order ODE system of *forced harmonic oscillators*, which guarantees that the eigenvalues of the state transition matrix lie on or within the unit circle *by construction* — with no constraints beyond requiring the diagonal frequency matrix $\mathbf{A}$ to have nonnegative entries. This directly addresses the core SSM stability problem: ensuring eigenvalues don't escape $[-1, 1]$ as sequences grow long.

The key insight: instead of constraining eigenvalues through parameterization tricks on a first-order system $\mathbf{x}_{n} = \mathbf{M}\mathbf{x}_{n-1} + \mathbf{F}_n$, start from a second-order oscillatory ODE $\mathbf{y}'' + \mathbf{A}\mathbf{y} = \mathbf{B}\mathbf{u}$ where $\mathbf{A} \geq 0$ is diagonal. The implicit discretization naturally produces a $2m \times 2m$ block transition matrix $\mathbf{M}^{IM}$ whose eigenvalues are *provably* bounded:

$$
|\lambda_j| = \sqrt{\mathbf{S}_{kk}^2(1 + \Delta t^2 \mathbf{A}_{kk})} = \sqrt{\mathbf{S}_{kk}} \leq 1
$$

where $\mathbf{S} = (I + \Delta t^2 \mathbf{A})^{-1}$. This holds for *any* positive $\mathbf{A}$ and $\Delta t > 0$ — no eigenvalue clipping, no spectral normalization, no careful initialization needed.

**Connection to the user's question about eigenvalue-preserving monoids:** The set of $2 \times 2$ block matrices of the form $\begin{bmatrix} \mathbf{S} & -\Delta t \mathbf{AS} \\ \Delta t \mathbf{S} & \mathbf{S} \end{bmatrix}$ with $\mathbf{A} \geq 0$ diagonal and $\mathbf{S} = (I + \Delta t^2 \mathbf{A})^{-1}$ does NOT form a monoid under matrix multiplication (the product of two such matrices is not generally of the same form). However, the *parallel scan associative operation* used to compute the recurrence IS a monoid: $((\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2)) = (\mathbf{a}_1 \mathbf{a}_2, \mathbf{a}_1 \mathbf{b}_2 + \mathbf{b}_1)$, and the eigenvalue bound is preserved because $|\lambda(\mathbf{M}^{n})| = |\lambda(\mathbf{M})|^n \leq 1$ for the contractive $\mathbf{M}$.

## Mathematical Form

**Continuous ODE System:**

$$
\mathbf{y}''(t) = -\mathbf{A}\mathbf{y}(t) + \mathbf{B}\mathbf{u}(t) + \mathbf{b}
$$
$$
\mathbf{x}(t) = \mathbf{C}\mathbf{y}(t) + \mathbf{D}\mathbf{u}(t)
$$

with $\mathbf{A} \in \mathbb{R}^{m \times m}$ diagonal ($\mathbf{A}_{kk} \geq 0$), $\mathbf{B} \in \mathbb{R}^{m \times p}$, auxiliary state $\mathbf{z} = \mathbf{y}'$.

**First-order form:**

$$
\mathbf{z}'(t) = -\mathbf{A}\mathbf{y}(t) + \mathbf{B}\mathbf{u}(t)
$$
$$
\mathbf{y}'(t) = \mathbf{z}(t)
$$

**Implicit (IM) Discretization:**

$$
\mathbf{x}_n = \mathbf{M}^{IM}\mathbf{x}_{n-1} + \mathbf{F}_n^{IM}
$$

where $\mathbf{x}_n = [\mathbf{z}_n, \mathbf{y}_n]^\top$ and:

$$
\mathbf{M}^{IM} = \begin{bmatrix} \mathbf{S} & -\Delta t \mathbf{A}\mathbf{S} \\ \Delta t \mathbf{S} & \mathbf{S} \end{bmatrix}, \quad \mathbf{F}_n^{IM} = \begin{bmatrix} \Delta t \mathbf{B}\mathbf{u}_n \\ 0 \end{bmatrix}
$$

with $\mathbf{S} = (I + \Delta t^2 \mathbf{A})^{-1}$ computable in $O(m)$ since $\mathbf{A}$ is diagonal.

**Implicit-Explicit (IMEX) Discretization:**

$$
\mathbf{M}^{IMEX} = \begin{bmatrix} \mathbf{I} & -\Delta t \mathbf{A} \\ \Delta t \mathbf{I} & \mathbf{I} - \Delta t^2 \mathbf{A} \end{bmatrix}
$$

This is a *symplectic* integrator — volume-preserving and energy-conserving.

**Eigenvalue Analysis (Proposition 3.1):**

For LinOSS-IM with $\mathbf{A}_{kk} \geq 0$ and $\Delta t > 0$, the eigenvalues are:

$$
\lambda_j = \frac{1}{1 + \Delta t^2 \mathbf{A}_{kk}} + i(-1)^{\lceil j/m \rceil} \Delta t \frac{\sqrt{\mathbf{A}_{kk}}}{1 + \Delta t^2 \mathbf{A}_{kk}}
$$

for $j = 1, \ldots, 2m$ with $k = j \mod m$.

**Eigenvalue magnitude:**

$$
|\lambda_j|^2 = \mathbf{S}_{kk}^2 + \Delta t^2 \mathbf{S}_{kk}^2 \mathbf{A}_{kk} = \mathbf{S}_{kk}^2(1 + \Delta t^2 \mathbf{A}_{kk}) = \mathbf{S}_{kk} \leq 1
$$

since $\mathbf{S}_{kk} = \frac{1}{1 + \Delta t^2 \mathbf{A}_{kk}} \leq 1$ for $\mathbf{A}_{kk} \geq 0$.

For LinOSS-IMEX: $|\lambda_j| = 1$ exactly (symplectic ↔ energy-conserving).

**Parameterization of $\mathbf{A}$:**

Either $\mathbf{A} = \hat{\mathbf{A}}^2$ (squared) or $\mathbf{A} = \text{ReLU}(\hat{\mathbf{A}})$, ensuring nonnegativity.

**Initialization:** $\mathbf{A}_{kk} \sim \mathcal{U}([0, 1])$, $\Delta t = 1$.

**Parallel Scan Operation:**

The recurrence $\mathbf{x}_n = \mathbf{M}\mathbf{x}_{n-1} + \mathbf{F}_n$ is computed via associative parallel scan with operation:

$$
(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) = (\mathbf{a}_1 \mathbf{a}_2, \mathbf{a}_1 \mathbf{b}_2 + \mathbf{b}_1)
$$

exploiting the $2 \times 2$ block structure of $\mathbf{M}$ (only diagonal entries in each block), this runs in $O(m)$ per step instead of $O(m^2)$.

## Complexity

| Operation | Standard SSM (diagonal $A$) | LinOSS |
|-----------|---------------------------|--------|
| State dimension for $m$ oscillators | $m$ | $2m$ |
| Forward step | $O(m)$ | $O(m)$ |
| Parallel scan | $O(m \log T)$ | $O(m \log T)$ |
| Eigenvalue constraint enforcement | Explicit (clipping/normalization) | Free (by construction) |
| Hyperparameters | Initialization-sensitive | $\Delta t$ only (robust) |

**Memory:** $O(2m)$ per timestep (doubled state for position + velocity)

**Key advantage:** No computational overhead for stability — it comes from the physics of the discretization, not from projections or constraints.

## Applicability

- **Any SSM requiring stable long-range dynamics**: LinOSS directly produces contractive state transitions from unconstrained positive parameters
- **Time-series forecasting**: Energy-conserving (IMEX) variant excels at physical systems; dissipative (IM) variant better for general sequences
- **Drop-in SSM replacement**: LinOSS outperforms Mamba, LRU, S4, S5 on long-range benchmarks (UEA classification, PPG-DaLiA, weather forecasting)
- **Hardware-efficient**: Only requires diagonal operations + parallel scan — fully compatible with existing GPU matmul and associative scan primitives

## Limitations

- State dimension is doubled ($2m$ vs $m$) — trades memory for stability guarantee
- The oscillatory inductive bias may not suit all tasks (though GLU nonlinearity between layers mitigates this)
- LinOSS-IMEX (symplectic) can produce NaN during training with explicit discretization — IM variant is more robust
- Second-order ODE structure is a strong assumption about dynamics

## Implementation Notes

```python
import torch
import torch.nn as nn

class LinOSS_IM(nn.Module):
    """Implicit LinOSS: eigenvalues bounded by construction"""
    def __init__(self, m, p, dt=1.0):
        super().__init__()
        self.m = m
        self.dt = dt
        # A_hat parameterizes A = ReLU(A_hat) ≥ 0
        self.A_hat = nn.Parameter(torch.rand(m))  # init ~ U[0,1]
        self.B = nn.Parameter(torch.randn(m, p) * 0.01)

    def get_transition(self):
        A = torch.relu(self.A_hat)  # Ensure A ≥ 0
        S = 1.0 / (1.0 + self.dt**2 * A)  # Diagonal S, O(m)
        # Eigenvalues automatically satisfy |λ| ≤ 1
        # |λ|² = S_kk ≤ 1 for all k
        return A, S

    def forward(self, u_seq):
        # u_seq: (batch, T, p)
        A, S = self.get_transition()
        dt = self.dt
        B, T, p = u_seq.shape

        # Parallel scan over (M, F) pairs
        # M = [[S, -dt*A*S], [dt*S, S]]  (block diagonal)
        # F = [[dt*B*u], [0]]
        # Use associative scan: (a1,b1)•(a2,b2) = (a1*a2, a1*b2+b1)
        # ... (implementation via existing scan primitives)
```

Code: https://github.com/tk-rusch/linoss

## References

- Rusch and Rus (2025). Oscillatory State-Space Models (LinOSS). ICLR 2025. arXiv:2410.03943.
- Rusch and Rus (2025). Learning to Dissipate Energy in Oscillatory State-Space Models (D-LinOSS). arXiv:2505.12171.
- Rusch and Mishra (2021). Coupled Oscillatory Recurrent Neural Networks.
- Smith et al. (2023). Simplified State Space Layers for Sequence Modeling (S5).

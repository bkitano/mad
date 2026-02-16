# 022: Cayley Contractive Parameterization

**Category**: stability
**Gain type**: efficiency
**Source**: Helfrich, Willmott, and Ye (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML.
**Paper**: papers/cayley-contractive-parameterization.pdf
**Documented**: 2026-02-11

## Description

Parameterize a recurrent weight matrix $W$ to be exactly orthogonal (eigenvalues on the unit circle, $|\lambda_i| = 1$) by mapping a free skew-symmetric matrix $A$ through the Cayley transform, then scaling by a diagonal sign matrix $D$ to reach the full orthogonal group. This guarantees that repeated application of $W$ over long sequences neither amplifies nor shrinks hidden state norms — directly addressing the exploding/vanishing gradient problem in SSMs and RNNs. Unlike multiplicative Stiefel-manifold updates, the Cayley parameterization is a simple rational map that stays orthogonal to machine precision under standard gradient descent, even in fp16.

The key insight for the user's question: the set of orthogonal matrices under multiplication forms a **group** (hence a monoid) where all eigenvalues lie exactly on the unit circle. By parameterizing through skew-symmetric matrices, we get a hardware-friendly route: the free parameters live in an unconstrained Euclidean space, standard matmul produces $W$, and the eigenvalue constraint $|\lambda| = 1$ is enforced *by construction* — no projection or clipping needed.

## Mathematical Form

**Core Operation:**

$$
W = (I + A)^{-1}(I - A)D
$$

where $A \in \mathbb{R}^{n \times n}$ is skew-symmetric ($A^T = -A$) and $D = \text{diag}(\pm 1)$ is a fixed diagonal sign matrix.

**Key Definitions:**

- $A \in \mathbb{R}^{n \times n}$ — skew-symmetric, $a_{ij} \in [-1, 1]$, with $\frac{n(n-1)}{2}$ free parameters
- $D \in \mathbb{R}^{n \times n}$ — fixed diagonal of $\pm 1$ entries (hyperparameter $\rho = $ number of $-1$s)
- $W \in \mathbb{R}^{n \times n}$ — resulting orthogonal matrix ($W^TW = I$)

**Why it works:**

The standard Cayley transform $W = (I+A)^{-1}(I-A)$ bijects skew-symmetric matrices to orthogonal matrices *without* $-1$ eigenvalues. The scaling matrix $D$ reflects selected eigenvalues across the imaginary axis, enabling representation of all orthogonal matrices including those with $-1$ eigenvalues.

**Theorem (Helfrich et al.):** Every orthogonal matrix $W$ can be expressed as $W = (I+A)^{-1}(I-A)D$ where $A$ is real-valued, skew-symmetric with $|a_{ij}| \leq 1$, and $D$ is diagonal with entries $\pm 1$.

**Gradient computation:**

$$
\frac{\partial L}{\partial A} = V^T - V
$$

where $V := (I + A)^{-T} \frac{\partial L}{\partial W} (D + W^T)$.

**Update rule:**

$$
A^{(k+1)} = A^{(k)} - \lambda \frac{\partial L(W(A^{(k)}))}{\partial A}
$$
$$
W^{(k+1)} = \left(I + A^{(k+1)}\right)^{-1}\left(I - A^{(k+1)}\right)D
$$

The skew-symmetry of $\frac{\partial L}{\partial A}$ ensures $A^{(k+1)}$ remains skew-symmetric, so $W^{(k+1)}$ is automatically orthogonal.

**Initialization:**

$$
A = \begin{bmatrix} B_1 & & \\ & \ddots & \\ & & B_{\lfloor n/2 \rfloor} \end{bmatrix}, \quad B_j = \begin{bmatrix} 0 & s_j \\ -s_j & 0 \end{bmatrix}
$$

with $s_j = \sqrt{\frac{1 - \cos(t_j)}{1 + \cos(t_j)}}$ and $t_j \sim \text{Uniform}[0, \frac{\pi}{2}]$, giving eigenvalues $\pm e^{it_j}$ uniformly on the right unit half-circle.

## Complexity

| Operation | Full-capacity uRNN | scoRNN (Cayley) |
|-----------|-------------------|-----------------|
| Parameters | $O(n^2)$ | $O(n^2/2)$ |
| Weight construction | $O(n^3)$ per step | $O(n^3)$ once per iteration |
| Forward pass | $O(BTn^2)$ | $O(BTn^2)$ |
| Orthogonality maintenance | Degrades with fp16 | Exact to machine precision |

**Memory:** $O(n^2/2)$ for $A$ vs $O(n^2)$ for unconstrained $W$

The weight matrix $W$ is reconstructed from $A$ only once per training iteration (not per timestep), so the $O(n^3)$ cost is amortized over $BT$ forward steps.

## Applicability

- **RNNs with long-range dependencies**: Orthogonal recurrence prevents vanishing/exploding gradients over thousands of timesteps
- **SSM state transitions**: When $|\lambda_i| = 1$ is desired (information preservation without decay)
- **Any model requiring eigenvalue-bounded transition matrices**: The Cayley map provides a hardware-friendly way to stay on the orthogonal manifold using standard matmul
- **Connection to the user's question**: The orthogonal group $O(n)$ under matrix multiplication is a monoid where eigenvalue magnitudes are *exactly* 1 — this is the strictest form of the "eigenvalues in $[-1,1]$" constraint, and it's preserved under composition by algebraic structure

## Limitations

- Eigenvalues are constrained to $|\lambda| = 1$ exactly — cannot represent controlled decay ($|\lambda| < 1$), which many SSMs need for forgetting
- The $O(n^3)$ matrix inversion cost (once per training step) can be expensive for large hidden dimensions
- For SSMs like Mamba that use diagonal state matrices, this is overparameterized — diagonal $\times$ sign matrices suffice
- Banded variants (restricting $A$ to bandwidth $\ell$) trade representational capacity for fewer parameters

## Implementation Notes

```python
import torch
import torch.nn as nn

class ScoRNN(nn.Module):
    """Scaled Cayley Orthogonal RNN"""
    def __init__(self, n, rho_frac=0.5):
        super().__init__()
        # Free parameters: upper triangle of skew-symmetric A
        self.A_params = nn.Parameter(torch.zeros(n * (n - 1) // 2))
        self.n = n
        # Fixed sign matrix D
        rho = int(n * rho_frac)
        d = torch.ones(n)
        d[:rho] = -1
        self.register_buffer('D', torch.diag(d))

    def get_W(self):
        # Build skew-symmetric A from free parameters
        A = torch.zeros(self.n, self.n, device=self.A_params.device)
        idx = torch.triu_indices(self.n, self.n, offset=1)
        A[idx[0], idx[1]] = self.A_params
        A = A - A.T  # skew-symmetric

        I = torch.eye(self.n, device=A.device)
        # W = (I + A)^{-1} (I - A) D
        W = torch.linalg.solve(I + A, I - A) @ self.D
        return W  # Guaranteed orthogonal
```

## References

- Helfrich, Willmott, and Ye (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML.
- Arjovsky, Shah, and Bengio (2016). Unitary Evolution Recurrent Neural Networks. ICML.
- Wisdom et al. (2016). Full-Capacity Unitary Recurrent Neural Networks.
- Maduranga, Helfrich, and Ye (2019). Complex Unitary Recurrent Neural Networks using Scaled Cayley Transform.

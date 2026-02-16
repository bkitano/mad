# 035: Diagonal Plus Low Rank (DPLR)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Gu et al. (S4, 2022)
**Documented**: 2026-02-10

## Description

Parameterize the state transition matrix as $A = \Lambda + PQ^*$ where $\Lambda$ is diagonal and $P, Q$ are low-rank factors. This allows the matrix exponential and convolution kernel to be computed efficiently.

## Mathematical Form

**Core Operation:**

$$
A = \Lambda + P Q^*
$$

where $\Lambda \in \mathbb{C}^{N \times N}$ is diagonal, and $P, Q \in \mathbb{C}^{N \times r}$ are low-rank factors.

**Key Definitions:**

- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ — diagonal component
- $P, Q \in \mathbb{C}^{N \times r}$ — low-rank factors (typically $r \ll N$)
- $r$ — rank of the correction (usually 1 or 2 for HiPPO)

**Efficient Operations:**

Key operations become efficient via the Woodbury identity:

- **Resolvent:** $(zI - A)^{-1}$ via Woodbury: diagonal inverse + rank-$r$ correction
  $$
  (zI - A)^{-1} = D_z - D_z P (I + Q^* D_z P)^{-1} Q^* D_z
  $$
  where $D_z = (zI - \Lambda)^{-1}$

- **Matrix exponential:** $\exp(A)$ can be approximated via diagonalization of the correction

- **Convolution kernel:** $\mathbf{K}_L = (CB, CAB, \ldots, CA^{L-1}B)$ computable in $O(N + r^2)$ per step instead of $O(N^2)$

## Complexity

| Operation | Dense $A$ | DPLR |
|-----------|-----------|------|
| Matrix-vector | $O(N^2)$ | $O(N + r^2)$ |
| Resolvent | $O(N^3)$ | $O(N + r^2)$ |
| Convolution kernel | $O(N^2 L)$ | $O((N + r^2) L)$ |

**Memory:** $O(N + Nr) = O(Nr)$ vs $O(N^2)$

Typically $r = 1$ or $2$, so complexity is effectively $O(N)$ per step.

## Applicability

State space models (S4, S5, Mamba). The DPLR structure is what makes S4's HiPPO matrix computationally tractable.

## Limitations

- Restricts the set of expressible matrices
- Not all transition dynamics can be well-approximated by diagonal + low-rank
- The low-rank correction adds implementation complexity
- Requires careful initialization to preserve HiPPO properties

## Examples

The HiPPO-LegS matrix can be decomposed as:

$$
A = -\Lambda + \mathbf{p} \mathbf{q}^\top
$$

where $\Lambda_{nk} = (2n+1)^{1/2} (2k+1)^{1/2}$ for $n > k$.

This is exactly DPLR with rank 1.

## Implementation Notes

```python
# DPLR parameterization
class DPLR:
    def __init__(self, N, r=1):
        self.Lambda = nn.Parameter(torch.randn(N))  # Diagonal
        self.P = nn.Parameter(torch.randn(N, r))    # Low-rank
        self.Q = nn.Parameter(torch.randn(N, r))    # Low-rank

    def forward(self, x):
        # A @ x = Lambda * x + P @ (Q.T @ x)
        return self.Lambda * x + self.P @ (self.Q.T @ x)
```

## References

- Gu et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR.
- Gu et al. (2022). On the Parameterization and Initialization of Diagonal State Space Models.

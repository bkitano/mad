# 080: Negative Eigenvalue Extension

**Category**: algebraic
**Gain type**: expressivity
**Source**: Grazzi et al. (ICLR 2025)
**Documented**: 2026-02-10

## Description

Multiply the beta (learning rate) parameter by 2, extending its range from $[0,1]$ to $[0,2]$. When $\beta > 1$, the state matrix eigenvalues can become negative, which is required for $\text{NC}^1$ state tracking tasks (like $S_5$ permutation composition). Without this, DeltaNet is limited to $\text{TC}^0$ expressivity.

## Mathematical Form

**Core Operation:**

**Standard DeltaNet:**
$$
\beta_t = \sigma(x_t W_\beta) \in (0, 1)
$$

This restricts state matrix $S_t$ eigenvalues to $[0, 1]$ (contraction).

**Extended DeltaNet:**
$$
\beta_t = 2 \cdot \sigma(x_t W_\beta) \in (0, 2)
$$

This allows $S_t$ eigenvalues to become **negative**.

**Key Definitions:**

- $\beta_t \in \mathbb{R}$ — learning rate / gate parameter
- $\sigma(\cdot)$ — sigmoid function
- $S_t \in \mathbb{R}^{d \times d}$ — state matrix

**Delta Rule Update:**

The state matrix evolves as:
$$
S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top
$$

With $\beta_t \in (0, 2)$, the eigenvalues of $S_t$ can satisfy $\lambda_i \in (-1, 1)$ including negative values.

**Expressivity Implications:**

| $\beta$ Range | Eigenvalues | Expressivity Class |
|---------------|-------------|-------------------|
| $(0, 1)$ | $[0, 1]$ only | $\text{TC}^0$ (abelian groups only) |
| $(0, 2)$ | $(-1, 1)$ | $\text{NC}^1$ (non-abelian groups) |

## Complexity

| Property | Standard | Extended |
|----------|----------|----------|
| Computation | Same | Same |
| Expressivity | $\text{TC}^0$ | $\text{NC}^1$ |
| State tracking | Cannot solve $S_5$ | Can solve $S_5$ |

**No additional compute cost** — just a different parameterization.

## Applicability

DeltaNet and any recurrence with a gated state update. Critical for tasks requiring non-trivial automaton simulation:

- $S_5$ (symmetric group) composition — requires negative eigenvalues
- General finite automata simulation
- Algorithmic reasoning tasks with non-abelian structure

## Limitations

- Negative eigenvalues can cause training instability if not properly controlled
- May require careful initialization and learning rate tuning
- Not all tasks benefit — abelian tasks (like modular addition) don't need this
- The extended range increases the risk of oscillatory/unstable dynamics

## Implementation Notes

```python
# Standard vs Extended DeltaNet beta
def standard_beta(x, W_beta):
    return torch.sigmoid(x @ W_beta)  # (0, 1)

def extended_beta(x, W_beta):
    return 2 * torch.sigmoid(x @ W_beta)  # (0, 2)

# Delta rule update
def delta_update(S, k, v, beta):
    # S: (d, d) state matrix
    # k, v: (d,) key and value
    error = v - S.T @ k
    S_new = S + beta * torch.outer(k, error)
    return S_new
```

## References

- Grazzi et al. (2025). Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. ICLR.

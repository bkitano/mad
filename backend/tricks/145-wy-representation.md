# 145: WY Representation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Numerical linear algebra (Bischof & Van Loan, 1987)
**Documented**: 2026-02-10

## Description

Represent a product of Householder reflections as $I + WY^\top$ where $W, Y$ are thin matrices. Instead of materializing the full $d \times d$ state matrix $S_t$ in DeltaNet, represent it implicitly via accumulated $W$ and $Y$ factors. This reduces the memory footprint from $O(d^2)$ to $O(d \cdot k)$ where $k$ is the number of updates.

## Mathematical Form

**Core Operation:**

$$
S_t = I + W_t Y_t^\top
$$

where $W_t, Y_t \in \mathbb{R}^{d \times t}$ accumulate rank-1 updates:
- $W_t = [w_1, \ldots, w_t]$
- $Y_t = [y_1, \ldots, y_t]$

**Key Definitions:**

- $S_t \in \mathbb{R}^{d \times d}$ — state matrix at time $t$
- $W_t, Y_t \in \mathbb{R}^{d \times t}$ — accumulated factors
- $k_t, v_t \in \mathbb{R}^d$ — key and value at time $t$
- $\beta_t \in \mathbb{R}$ — learning rate

**Delta Rule in WY Form:**

Each delta rule update $S_{t+1} = S_t + \beta_t k_t (v_t - S_t^\top k_t)^\top$ can be expressed as appending columns to $W$ and $Y$:

$$
S_{t+1} = I + W_{t+1} Y_{t+1}^\top
$$

where:
$$
W_{t+1} = [W_t, w_{t+1}], \quad Y_{t+1} = [Y_t, y_{t+1}]
$$

with $w_{t+1} = \beta_t k_t$ and $y_{t+1} = v_t - S_t^\top k_t$.

**Output Computation:**

To compute $S_t x$ without materializing $S_t$:
$$
S_t x = x + W_t (Y_t^\top x)
$$

This is $O(dt)$ instead of $O(d^2)$.

## Complexity

| Operation | Dense $S_t$ | WY Representation |
|-----------|-------------|-------------------|
| Memory | $O(d^2)$ | $O(d \cdot C)$ per chunk |
| Update | $O(d^2)$ | $O(d)$ (append column) |
| Query $S_t x$ | $O(d^2)$ | $O(dC)$ |
| Materialize | — | $O(d^2)$ at chunk boundary |

**Memory:** $O(d \cdot \text{chunk\_size})$ within chunk, $O(d^2)$ only at chunk boundaries

## Applicability

DeltaNet and other linear attention variants that maintain a state matrix updated via rank-1 outer products. Enables chunkwise parallel computation.

## Limitations

- The representation grows linearly with the number of updates within a chunk
- Must periodically materialize $S_t$ at chunk boundaries
- Optimal chunk size trades off memory vs. parallelism
- Only applicable to rank-1 (or low-rank) updates

## Implementation Notes

```python
# WY representation for DeltaNet
class WYState:
    def __init__(self, d):
        self.W = []  # List of w vectors
        self.Y = []  # List of y vectors
        self.d = d

    def update(self, k, v, beta, S_prev=None):
        # Compute error: v - S_t^T k
        if S_prev is not None:
            error = v - S_prev.T @ k
        else:
            # Use WY form: S^T k = k + Y (W^T k)
            WTk = sum(w * (k @ w) for w in self.W)  # Simplified
            error = v - k - sum(y * WTk_i for y, WTk_i in zip(self.Y, WTk))

        self.W.append(beta * k)
        self.Y.append(error)

    def materialize(self):
        # S = I + W Y^T
        S = torch.eye(self.d)
        for w, y in zip(self.W, self.Y):
            S += torch.outer(w, y)
        return S
```

## References

- Bischof & Van Loan (1987). The WY Representation for Products of Householder Matrices.
- Yang et al. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule.
- Sun et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States.

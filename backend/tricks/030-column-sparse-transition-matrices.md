# 030: Column-Sparse Transition Matrices (PD-SSM)

**Category**: decomposition
**Gain type**: expressivity
**Source**: IBM Research (NeurIPS 2025)
**Documented**: 2026-02-10

## Description

Replace the diagonal state transition matrix $A_t$ in selective SSMs with a column-sparse matrix where each column has exactly one non-zero complex-valued entry. This structured sparsity is strictly more expressive than diagonal matrices — it can represent permutations and routing between state dimensions — while remaining closed under matrix multiplication. Closure under multiplication means that long chains of state transitions $A_T \cdot A_{T-1} \cdots A_1$ can be efficiently computed without the product becoming dense, preserving the $O(N)$ per-step complexity of diagonal SSMs. The resulting PD-SSM can emulate automata corresponding to non-solvable groups, which diagonal SSMs provably cannot.

## Mathematical Form

**Core Operation:**

The input-dependent transition matrix is factored as:

$$
A(u_t) = P(u_t) \cdot D(u_t)
$$

**Key Definitions:**

- $P(u_t) \in \{0,1\}^{N \times N}$ — **column one-hot matrix** (permutation-like): each column has exactly one nonzero entry equal to 1. $P$ selects which state dimension each state feeds into.
- $D(u_t) \in \mathbb{C}^{N \times N}$ — **diagonal matrix** with $|D_{ii}| < 1$ (entries inside the unit circle), ensuring BIBO stability.

**Closure Property:**

If $A_1 = P_1 D_1$ and $A_2 = P_2 D_2$ are both column-sparse, then:

$$
A_2 \cdot A_1 = P_2 D_2 P_1 D_1 = P_2 P_1 \cdot (P_1^\top D_2 P_1) D_1
$$

which is again column-sparse (product of permutation-like matrices is permutation-like, product of diagonals is diagonal after reindexing).

**Cumulative Products:**

This means cumulative products $A_{t:s} = A_t \cdots A_{s+1}$ remain column-sparse, enabling efficient parallel scans:

$$
A_{t:s} = P_{t:s} \cdot D_{t:s}
$$

where $P_{t:s}$ is a permutation and $D_{t:s}$ is diagonal.

## Complexity

| Operation | Dense $A$ | Diagonal | Column-Sparse |
|-----------|-----------|----------|---------------|
| Per-step multiply | $O(N^2)$ | $O(N)$ | $O(N)$ |
| Matrix-matrix product | $O(N^3)$ | $O(N)$ | $O(N)$ |
| Memory per timestep | $O(N^2)$ | $O(N)$ | $O(N)$ |

**Memory:** $O(N)$ per timestep instead of $O(N^2)$

## Applicability

- Drop-in replacement for the diagonal $A$ matrix in Mamba, S5, or any selective SSM
- Particularly beneficial for **state tracking** tasks: finite automata, algorithmic reasoning, in-context learning of structured patterns
- Achieves near-perfect performance on finite-state automaton tracking benchmarks where diagonal SSMs fail
- Can emulate non-solvable group automata (e.g., $S_5$ permutation group), which is provably impossible for diagonal SSMs regardless of depth
- Compatible with existing parallel scan and chunkwise algorithms

## Limitations

- The column one-hot structure $P(u_t)$ requires discrete selection, which may need straight-through estimators or Gumbel-softmax for differentiable training
- More parameters per timestep than pure diagonal (need to generate $P$ and $D$ instead of just $D$)
- The expressivity advantage is most pronounced on state-tracking/algorithmic tasks; gains on standard language modeling benchmarks may be smaller
- Permutation structure means states cannot be mixed (weighted combination) — each state feeds into exactly one other state

## Implementation Notes

```python
# Column-sparse transition (PD-SSM)
def pd_ssm_step(h, x, P_logits, D):
    # P_logits: (N, N) logits for column one-hot selection
    # D: (N,) diagonal entries
    P = gumbel_softmax(P_logits, hard=True, dim=0)  # (N, N) one-hot columns
    A = P * D[None, :]  # Column-sparse matrix
    return A @ h + B @ x
```

## References

- IBM Research (2025). Efficient Transition Matrices to Enable State Tracking in State-Space Models. NeurIPS.
- Merrill et al. (2024). The Illusion of State in State-Space Models. (Theory on diagonal SSM limitations)
- Gu & Dao (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. (Baseline selective SSM)

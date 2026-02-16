# 101: Recurrence-to-Scan Reduction

**Category**: parallelization
**Gain type**: efficiency
**Source**: Blelloch (1990). Prefix Sums and Their Applications, Section 1.4.
**Paper**: papers/blelloch-prefix-sums-1993.pdf
**Documented**: 2026-02-12

## Description

A technique for converting sequential recurrences of the form $x_i = (x_{i-1} \otimes a_i) \oplus b_i$ into a parallel scan by *augmenting the state* into tuples (pairs). The key insight is that even though the recurrence appears to have two different operators ($\otimes$ for multiplication and $\oplus$ for addition), we can define a single associative binary operator $\bullet$ over pairs $c_i = [a_i, b_i]$ such that the scan of these pairs yields all recurrence values in parallel.

This is the foundational trick that enables SSMs like Mamba and S4 to be parallelized: the linear recurrence $h_t = A_t h_{t-1} + b_t$ is exactly a first-order recurrence with $\otimes$ = matrix-vector multiply and $\oplus$ = vector addition, and the pair operator becomes composition of affine transformations.

## Mathematical Form

**First-Order Recurrence:**

$$
x_i = \begin{cases} b_0 & i = 0 \\ (x_{i-1} \otimes a_i) \oplus b_i & 0 < i < n \end{cases}
$$

where $\oplus$ and $\otimes$ satisfy:
1. $\oplus$ is associative: $(a \oplus b) \oplus c = a \oplus (b \oplus c)$
2. $\otimes$ is semiassociative: there exists an associative companion operator $\odot$ such that $(a \otimes b) \otimes c = a \otimes (b \odot c)$
3. $\otimes$ distributes over $\oplus$: $a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c)$

**The Tuple Augmentation:**

Define pairs $c_i = [a_i, b_i]$ and a binary operator $\bullet$ on pairs:

$$
c_i \bullet c_j \equiv [c_{i,a} \odot c_{j,a}, \; (c_{i,b} \otimes c_{j,a}) \oplus c_{j,b}]
$$

**Theorem (Blelloch):** The operator $\bullet$ is associative. Proof:

$$
(c_i \bullet c_j) \bullet c_k = [c_{i,a} \odot c_{j,a}, \; (c_{i,b} \otimes c_{j,a}) \oplus c_{j,b}] \bullet c_k
$$
$$
= [(c_{i,a} \odot c_{j,a}) \odot c_{k,a}, \; (((c_{i,b} \otimes c_{j,a}) \oplus c_{j,b}) \otimes c_{k,a}) \oplus c_{k,b}]
$$
$$
= [c_{i,a} \odot (c_{j,a} \odot c_{k,a}), \; (c_{i,b} \otimes (c_{j,a} \odot c_{k,a})) \oplus ((c_{j,b} \otimes c_{k,a}) \oplus c_{k,b})]
$$
$$
= c_i \bullet (c_j \bullet c_k) \qquad \checkmark
$$

**Recovery of scan results:**

Define the augmented sequence $s_i = [y_i, x_i]$ where $y_i = y_{i-1} \odot a_i$ (companion scan) and $x_i$ is the desired recurrence. Then:

$$
s_0 = c_0 = [a_0, b_0]
$$
$$
s_i = s_{i-1} \bullet c_i \qquad 0 < i < n
$$

The second component of $s_i$ gives $x_i$, the recurrence solution.

**SSM Specialization ($h_t = A_t h_{t-1} + b_t$):**

Here $\otimes$ = matrix-vector multiply, $\oplus$ = vector addition, and the companion $\odot$ = matrix-matrix multiply. The tuples are $(A_t, b_t)$ and the operator is:

$$
(A_i, b_i) \bullet (A_j, b_j) = (A_j A_i, \; A_j b_i + b_j)
$$

This is composition of affine maps: if $f_i(x) = A_i x + b_i$, then $f_j \circ f_i(x) = A_j A_i x + A_j b_i + b_j$.

**Higher-Order Recurrences:**

For $m$-th order recurrences $x_i = (x_{i-1} \otimes a_{i,1}) \oplus \cdots \oplus (x_{i-m} \otimes a_{i,m}) \oplus b_i$, define the state vector:

$$
s_i = [x_i, x_{i-1}, \ldots, x_{i-m+1}]
$$

and the companion matrix:

$$
A_i = \begin{bmatrix} a_{i,1} & 1 & 0 & \cdots & 0 \\ \vdots & 0 & 1 & & \vdots \\ \vdots & \vdots & & \ddots & 0 \\ a_{i,m} & 0 & \cdots & 0 & 0 \end{bmatrix}, \quad B_i = [b_i, 0, \ldots, 0]
$$

This reduces the $m$-th order recurrence to the first-order form $s_i = (s_{i-1} \otimes_{(v)} A_i) \oplus_{(v)} B_i$, solvable in parallel.

## Complexity

| Operation | Sequential | With Scan Reduction |
|-----------|-----------|-------------------|
| First-order scalar | $O(n)$ | $O(n/p + \log p)$ |
| First-order with $d \times d$ matrices | $O(nd^2)$ | $O((n/p + \log p) \cdot d^3)$ |
| $m$-th order scalar | $O(nm)$ | $O((n/p + \log p) \cdot m^3)$ |

**Key cost:** The tuple operator $\bullet$ requires one companion multiplication ($\odot$), one mixed multiplication ($\otimes$), and one addition ($\oplus$). For matrices: $(T_\odot + T_\otimes + T_\oplus)(n/p + \log p)$.

**Memory:** $O(n)$ pairs of (matrix, vector), so $O(n(d^2 + d))$ for $d$-dimensional state.

## Applicability

- **State space models (S4, Mamba, S5)**: The linear recurrence $h_t = A_t h_{t-1} + b_t$ is exactly this pattern. Mamba uses this reduction with diagonal $A_t$ (making $\odot$ element-wise multiply, $O(d)$ instead of $O(d^3)$)
- **Linear attention**: The cumulative KV state $S_t = \lambda_t S_{t-1} + v_t k_t^T$ is a matrix-valued first-order recurrence
- **DeltaNet / RWKV**: Time-varying decay recurrences $h_t = \alpha_t h_{t-1} + \beta_t x_t$ use diagonal specialization
- **Fibonacci and polynomial evaluation**: Classical applications — $x_i = x_{i-1} + x_{i-2}$ reduces to $2 \times 2$ matrix scan
- **Tridiagonal solvers**: Backsubstitution phase is a first-order recurrence parallelizable via this technique
- **CTC loss / HMM forward-backward**: Forward variables satisfy first-order recurrences over probability vectors

## Limitations

- **Operator must satisfy the three conditions**: Associativity of $\oplus$, semiassociativity of $\otimes$, and distributivity. Non-linear recurrences (GRU, LSTM) **cannot** be reduced this way
- **Matrix multiplication overhead**: For $d \times d$ state matrices, the companion operator is $O(d^3)$, which is expensive and doesn't map well to tensor cores for small $d$ (Mamba-1 limited to $d = 16$ for this reason)
- **Increased memory**: Must store pairs $(A_t, b_t)$ for all timesteps simultaneously, requiring $O(n \cdot d^2)$ memory
- **Numerical precision**: Long chains of matrix multiplications can accumulate floating-point errors, especially in FP16
- **Not optimal in recurrence order $m$**: Parallel algorithm performs $O(m^3)$ more work per element than sequential $O(m)$

## Implementation Notes

```python
import torch

def recurrence_to_scan(A, b):
    """
    Solve h_t = A_t * h_{t-1} + b_t in parallel via tuple-augmented scan.

    A: (T, d, d) transition matrices (or (T, d) if diagonal)
    b: (T, d) input vectors
    Returns: (T, d) hidden states h_0, ..., h_{T-1}
    """
    T, d = b.shape

    # Form tuples: elements are (A_t, b_t)
    # The associative operator is composition of affine maps:
    # (A_i, b_i) * (A_j, b_j) = (A_j @ A_i, A_j @ b_i + b_j)

    # For diagonal A (Mamba-style), this simplifies to:
    # (a_i, b_i) * (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)
    # where * is element-wise multiplication

    def combine(pair_i, pair_j):
        """Associative operator on (A, b) pairs."""
        Ai, bi = pair_i
        Aj, bj = pair_j
        # Composition: f_j(f_i(x)) = A_j(A_i x + b_i) + b_j
        return (Aj * Ai, Aj * bi + bj)  # element-wise for diagonal

    # Parallel scan with the combine operator
    # (Use Blelloch up-sweep/down-sweep or hardware scan primitive)
    pairs = [(A[t], b[t]) for t in range(T)]
    scanned = parallel_inclusive_scan(pairs, combine)

    # Extract hidden states (second component of each pair)
    h = torch.stack([s[1] for s in scanned])
    return h

# The key insight: what was O(T) sequential becomes O(log T) parallel
# at the cost of 2x the work per element (companion + mixed multiply)
```

## References

- Blelloch, G.E. (1990). Prefix Sums and Their Applications. Section 1.4: Recurrence Equations.
- Gu, A., Goel, K., and Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4).
- Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Martin, E. and Cundy, C. (2018). Parallelizing Linear Recurrent Neural Nets Over Sequence Length.
- Smith, J.T.H., Warrington, A., and Linderman, S.W. (2023). Simplified State Space Layers for Sequence Modeling (S5).

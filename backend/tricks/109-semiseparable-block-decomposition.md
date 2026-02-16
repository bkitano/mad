# 109: Semiseparable Block Decomposition

**Category**: decomposition
**Gain type**: efficiency
**Source**: Dao & Gu (Mamba-2 / SSD, 2024)
**Documented**: 2026-02-10

## Description

The Structured State Space Duality (SSD) framework reveals that the output matrix $M$ of any SSM is a semiseparable matrix — a structured matrix where every submatrix below the diagonal is low-rank. This insight enables a block decomposition that combines the best of both the linear (recurrent) and quadratic (attention) computation modes. The sequence is split into blocks of size $Q$; diagonal blocks are computed via dense quadratic attention (leveraging tensor cores), while off-diagonal blocks are factorized into low-rank products connected by a shorter inter-block SSM scan. This yields a 2–8× speedup over Mamba-1's selective scan.

## Mathematical Form

**Core Operation:**

The SSM output matrix $M \in \mathbb{R}^{T \times T}$ has entries:

$$
M_{ij} = C_i^\top A_{i:j} B_j \quad \text{for } i \geq j, \quad 0 \text{ otherwise}
$$

where $A_{i:j} = A_i \cdot A_{i-1} \cdots A_{j+1}$ is the cumulative state transition.

**Key Definitions:**

- $M \in \mathbb{R}^{T \times T}$ — output (mixer) matrix
- $A_t \in \mathbb{R}^{n \times n}$ — state transition at time $t$
- $B_t, C_t \in \mathbb{R}^n$ — input/output projections
- $Q$ — block size (typically 64–256)

**1-Semiseparable Case (Scalar State):**

For scalar-identity transitions, the lower-triangular matrix simplifies to:

$$
L_{ij} = a_i \cdot a_{i-1} \cdots a_{j+1} \quad \text{(products of scalar gates)}
$$

**Block Decomposition:**

With block size $Q$:

1. **Intra-block** (diagonal blocks): Compute each $Q \times Q$ block as a small semiseparable (or full attention) matrix — uses $O(Q^2)$ matmul ops per block

2. **Inter-block states**: Contract each block into a state vector via $B$ and expand via $C$, forming a reduced sequence of $T/Q$ states

3. **Inter-block scan**: Run a linear scan (or 1-SS matmul) on the $T/Q$ compressed states — only $O(T/Q)$ sequential steps

4. **State-to-output**: Convert inter-block states back to corrections on each block's output

Steps 1, 2, 4 are embarrassingly parallel matmuls. Only step 3 is sequential but operates on a sequence $\sim 100\times$ shorter.

## Complexity

| Operation | Mamba-1 | SSD (Mamba-2) |
|-----------|---------|---------------|
| Sequential steps | $O(T)$ | $O(T/Q)$ |
| Parallel work | $O(TN)$ | $O(TQ)$ matmuls |
| **Total** | $O(T \cdot N)$ seq | $O(T \cdot Q)$ parallel + $O(\frac{T}{Q} \cdot N)$ seq |

**Memory:** $O(Q^2)$ per block for intra-block computation

In practice, the SSD algorithm is 2–8× faster than Mamba-1's optimized selective scan, primarily because it converts most FLOPs to matmul operations (up to $16\times$ faster on A100: 312 TFLOPS BF16 matmul vs 19 TFLOPS FP32 scalar).

## Applicability

- Mamba-2 and any SSM with scalar or diagonal state transitions
- The framework generalizes to any 1-semiseparable or $N$-semiseparable matrix mixer
- Connects SSMs to structured attention: the quadratic form of SSD is equivalent to masked linear attention
- Minimal implementation: ~25 lines of code for the core algorithm

## Limitations

- Requires scalar or diagonal $A$ (not full matrix transitions) for the 1-semiseparable structure
- Block size $Q$ is a hardware-dependent hyperparameter that must be tuned
- The quadratic intra-block computation means memory scales as $O(Q^2)$ per block
- Does not apply to SSMs with dense (non-structured) state transitions

## Implementation Notes

```python
# SSD block decomposition (simplified)
def ssd_forward(K, V, A, Q=64):
    T = K.shape[0]
    # Intra-block: dense Q×Q attention per block
    # Inter-block: scan over T/Q compressed states
    pass
```

## References

- Dao & Gu (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML.
- Dao & Gu (2024). State Space Duality (Mamba-2) Blog Series, Parts I–III.
- Vandebril, Van Barel, Mastronardi (2007). Matrix Computations and Semiseparable Matrices. (Mathematical foundations of semiseparable matrices)

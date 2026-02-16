# 237: Quasiseparable Matrix Mixer

**Category**: decomposition
**Gain type**: expressivity
**Source**: Hwang, Lahoti, Dao & Gu, "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers" (NeurIPS 2024)
**Paper**: papers/hydra-quasiseparable-matrix-mixer.pdf
**Documented**: 2026-02-15

## Description

The quasiseparable matrix mixer generalizes both semiseparable matrices (used in causal SSMs like Mamba-2/SSD) and low-rank matrices (used in linear attention) into a single **bidirectional** structured matrix class. The key insight is that while semiseparable matrices are inherently causal (lower triangular), quasiseparable matrices have **both** upper and lower triangular structure plus free diagonal entries — enabling principled bidirectional sequence mixing without ad-hoc forward+backward SSM combinations.

The trick decomposes a quasiseparable matrix-vector product into **two semiseparable matrix-vector products plus a diagonal term**, allowing direct reuse of existing optimized SSM kernels (like SSD/Mamba-2) for the implementation. This avoids needing any new custom CUDA kernel — the entire operation reduces to `shift(SSM_forward(X)) + flip(shift(SSM_backward(flip(X)))) + D*X`.

**Why it matters for GPU efficiency**: The decomposition maps entirely to existing hardware-optimized SSM scan kernels + elementwise ops. No new kernel fusion is needed. The shared projection layers between forward and backward passes save ~50% parameters compared to naive bidirectional SSMs (Add/Mult/Concat variants).

## Mathematical Form

**Core Operation:**

A matrix $\mathbf{M} \in \mathbb{R}^{L \times L}$ is $N$-quasiseparable if each element $m_{ij}$ satisfies:

$$
m_{ij} = \begin{cases}
\vec{c}_i^T \vec{A}_{i,j}^{\times} \vec{b}_j, & \text{if } i > j \\
\delta_i, & \text{if } i = j \\
\overleftarrow{c}_j^T \overleftarrow{A}_{j,i}^{\times} \overleftarrow{b}_i, & \text{if } i < j
\end{cases}
$$

where $\vec{A}_{i,j}^{\times} = \prod_{k=j+1}^{i} A_k$ is the cumulative state transition, and:
- $\vec{c}_i, \vec{b}_i \in \mathbb{R}^{N \times 1}$ — forward SSM input/output projections
- $\overleftarrow{c}_j, \overleftarrow{b}_i \in \mathbb{R}^{N \times 1}$ — backward SSM input/output projections
- $A_i \in \mathbb{R}^{N \times N}$ — state transition matrix (shared)
- $\delta_i \in \mathbb{R}$ — free diagonal parameters

**Key Definitions:**

- $\mathbf{M} \in \mathbb{R}^{L \times L}$ — quasiseparable mixer matrix ($L$ = sequence length)
- $N$ — quasiseparable order (SSM state dimension), controls rank of off-diagonal submatrices
- $QS(\cdot)$ — action of quasiseparable matrix on input
- $SS(\cdot)$ — action of semiseparable (causal SSM) matrix on input

**Decomposition into Two SSMs + Diagonal:**

$$
QS(\mathbf{X}) = \text{shift}(SS(\mathbf{X})) + \text{flip}(\text{shift}(SS(\text{flip}(\mathbf{X})))) + \mathbf{D}\mathbf{X}
$$

where $\text{flip}(\cdot)$ reverses the sequence, $\text{shift}(\cdot)$ right-shifts by one position (zero-padding), and $\mathbf{D} = \text{diag}(\delta_1, \ldots, \delta_L)$.

**Rank Characterization:**

- Semiseparable: any submatrix from lower triangle (including diagonal) has rank $\leq N$
- Quasiseparable: any submatrix from strictly upper or lower triangle (off-diagonal) has rank $\leq N$
- The diagonal values $\delta_i$ are **free** in quasiseparable matrices, giving strictly more expressivity

**Sequence Aligned Matrix (SAM) Property:**

Quasiseparable matrices are Sequence Aligned Matrices: each parameter maps to a specific sequence position, enabling (1) data-dependent parameterization and (2) length extendability beyond training length.

## Complexity

| Operation | Naive Dense | With QS Mixer |
|-----------|------------|---------------|
| Sequence mixing (matmul) | $O(L^2)$ | $O(L)$ via two SSM scans |
| Matrix construction | $O(L^2)$ or N/A | $O(L)$ from input projections |
| Bidirectional SSM (Add/Concat) | $O(L)$ but 2× params | $O(L)$ with shared projections |

**Memory:** $O(LN)$ for state storage — same as single SSM pass (states shared between forward/backward).

**Parameters:** Hydra adds only ~2M parameters over unidirectional Mamba (at 70M scale) due to shared $f_X$ projection layers, compared to doubling parameters for two separate SSMs.

## Applicability

- **Bidirectional sequence models**: BERT-style masked language models, ViTs, speech recognition — anywhere non-causal processing is needed
- **Drop-in replacement for attention**: Hydra outperforms BERT by 0.8 GLUE points and ViT by 2.2% Top-1 on ImageNet as a direct attention replacement
- **Unifying framework**: Provides a taxonomy of structured matrix mixers (Dense, Toeplitz, Vandermonde, Cauchy, Low-Rank, Semiseparable, Quasiseparable) with clear complexity/expressivity tradeoffs

## Limitations

- **Representation-Computation Tradeoff**: Like all structured matrix mixers, quasiseparable matrices are less expressive than dense (softmax attention) matrices. On memorization-heavy tasks with fixed model capacity, softmax attention eventually outperforms.
- **Hardware efficiency concern**: While theoretically $O(L)$, the sequential scan operations within SSD are less hardware-friendly than pure matmuls. The paper notes "structured matrices have associated sub-quadratic matrix multiplication algorithms, their implementation may not be hardware-friendly."
- **Causal (autoregressive) tasks**: For causal LM, semiseparable matrices (Mamba-2) are sufficient; the quasiseparable generalization adds overhead without benefit in strictly causal settings.
- **Throughput**: At BERT-Base scale (110M params), Hydra matches BERT's GLUE quality but throughput comparisons vs optimized Transformers are not extensively reported.

## Implementation Notes

```python
def hydra(x, A):
    """
    x: (B, L, H*P) — input sequence
    A: (H,) — SSM parameter (shared between forward/backward)
    """
    # Reverse input for backward pass
    x_b = flip(x, dim=1)

    # Project to get data-dependent SSM parameters
    # (B, L, H) for discretized A, (B, L, N) for discretized BC
    dt_f, dt_b = proj_dt(x), proj_dt(x_b)

    # Forward SSM pass (semiseparable lower triangle)
    y_f = SSD(
        x,
        discretize_A(A, dt_f),      # (B, L, H)
        discretize_bc(x, dt_f),     # (B, L, N)
    )

    # Backward SSM pass (semiseparable upper triangle via flip)
    y_b = SSD(
        x_b,
        discretize_A(A, dt_b),
        discretize_bc(x_b, dt_b),
    )

    # Combine: shift + flip(shift(backward)) + diagonal
    y_f = shift(y_f, dim=1)             # right-shift by 1
    y_b = flip(shift(y_b, dim=1), dim=1) # shift then flip back

    # QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + D*X
    y = y_f + y_b + x * repeat(proj_D(x), "B L H -> B L (H P)")

    return y
```

**Key GPU efficiency insight**: The entire quasiseparable operation decomposes into two calls to an existing SSD kernel (Mamba-2), one `flip` operation, one `shift`, and elementwise add/multiply. No new custom kernels needed — reuses the highly optimized SSD scan which already uses chunkwise parallelism with tensor core matmuls for intra-chunk computation.

**Memory Access Pattern**:
- Forward/backward SSM scans are sequential within chunks but parallel across chunks — same pattern as Mamba-2
- The flip + shift operations are coalesced memory reads with simple index arithmetic
- Diagonal term `D*X` is purely elementwise (trivially parallel)
- Arithmetic intensity is dominated by the SSD chunks (matmul-heavy), same as Mamba-2

## References

- Hwang, Lahoti, Dao & Gu, "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers" (NeurIPS 2024, arXiv:2407.09941)
- Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024)
- Bella et al., "Computations with quasiseparable polynomials and matrices" (TCS 2008)
- Pernet et al., "Exact computations with quasiseparable matrices" (ISSAC 2023)

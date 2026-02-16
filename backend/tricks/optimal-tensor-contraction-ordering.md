# Optimal Tensor Contraction Ordering

**Category**: algebraic
**Gain type**: efficiency
**Source**: Matrix chain multiplication (Cormen et al.), generalized by opt_einsum (Smith & Gray, 2018)
**Paper**: [papers/optimal-tensor-contraction-ordering.pdf]
**Documented**: 2026-02-14

## Description

Choose the optimal order of pairwise contractions in a multi-tensor einsum expression to minimize total FLOPs. This is the tensor generalization of the classical matrix chain multiplication problem: for matrices $A \in \mathbb{R}^{10 \times 100}$, $B \in \mathbb{R}^{100 \times 5}$, $C \in \mathbb{R}^{5 \times 50}$, computing $(AB)C$ costs $10 \cdot 100 \cdot 5 + 10 \cdot 5 \cdot 50 = 7{,}500$ FLOPs, while $A(BC)$ costs $100 \cdot 5 \cdot 50 + 10 \cdot 100 \cdot 50 = 75{,}000$ FLOPs — a $10\times$ difference from parenthesization alone. For tensor networks in deep learning (multi-head attention, tensor decompositions, einsum-heavy architectures), optimal contraction ordering can yield **1000$\times$+ speedups** by avoiding the creation of enormous intermediate tensors.

## Mathematical Form

**Core Problem:**

Given tensors $T_1, T_2, \ldots, T_n$ with a target contraction specified in Einstein notation:

$$
C_{i_1 \ldots i_m} = \sum_{j_1, \ldots, j_p} T_1^{(\alpha_1)} \cdot T_2^{(\alpha_2)} \cdots T_n^{(\alpha_n)}
$$

where $\alpha_k$ are the index sets of each tensor and $j_1, \ldots, j_p$ are the contracted (summed) indices.

A **contraction path** $\pi$ is a sequence of $(n-1)$ pairwise contractions. The total cost:

$$
\text{Cost}(\pi) = \sum_{s=1}^{n-1} \text{FLOPs}(\text{step}_s)
$$

where each step contracts two tensors $A_{i_1 \ldots i_a}$ and $B_{j_1 \ldots j_b}$ with cost:

$$
\text{FLOPs}(A, B) = \prod_{k \in \text{output indices}} d_k \times \prod_{k \in \text{contracted indices}} d_k
$$

**Optimal path:** $\pi^* = \arg\min_\pi \text{Cost}(\pi)$

**Key Definitions:**

- $T_i$ — input tensors with known dimension sizes
- $d_k$ — size of index $k$
- Contraction path — binary tree whose leaves are input tensors
- "Greedy" heuristic cost: at each step, choose the pair minimizing $|\text{output}| - \min(|T_i|, |T_j|)$
- "Optimal" finds exact minimum via dynamic programming (exponential in $n$)

**Matrix Chain Special Case:**

For matrices $M_1 \in \mathbb{R}^{p_0 \times p_1}, M_2 \in \mathbb{R}^{p_1 \times p_2}, \ldots, M_n \in \mathbb{R}^{p_{n-1} \times p_n}$:

$$
m[i,j] = \min_{i \leq k < j} \left\{ m[i,k] + m[k+1,j] + p_{i-1} \cdot p_k \cdot p_j \right\}
$$

Solvable in $O(n^3)$ by dynamic programming.

**Greedy Algorithm (opt_einsum):**

Three phases:
1. **Hadamard products:** Contract tensors with identical index sets (element-wise multiply)
2. **Pairwise contractions:** Greedily choose the pair with lowest cost until no shared indices remain
3. **Outer products:** Contract remaining tensors by minimizing sum of input sizes

Cost function at each greedy step:
$$
\text{cost}(A, B) = |\text{output}(A \otimes B)| - \min(|A|, |B|)
$$

where $|T|$ denotes the number of elements (product of dimension sizes).

## Complexity

| Method | Path-Finding Time | Path Quality |
|--------|------------------|-------------|
| Naive (left-to-right) | $O(1)$ | Arbitrarily bad |
| Greedy (opt_einsum) | $O(n \cdot k)$ | Good heuristic |
| Dynamic programming | $O(2^n)$ | Near-optimal |
| Exhaustive search | $O(n!)$ | Optimal |
| Hypergraph partitioning | $O(n^2 \log n)$ | State-of-the-art for large $n$ |

**Impact on FLOP count (documented examples):**

| Expression | Naive FLOPs | Optimized FLOPs | Speedup |
|------------|-------------|-----------------|---------|
| Multi-tensor contraction (8 tensors) | $8 \times 10^8$ | $8 \times 10^5$ | $1000\times$ |
| Attention score computation | $O(n^2 d h)$ | $O(n^2 h + n d h)$ | $d / (1 + d/n) \times$ |
| Tensor train contraction | $O(r^{2n})$ | $O(n r^3)$ | Exponential |

**Memory:** Optimal ordering also minimizes peak intermediate tensor size, critical for GPU memory.

## Applicability

- **Einsum-heavy architectures:** Multi-head attention, tensor decomposition layers, graph neural networks
- **Tensor networks:** Tensor train, MERA, PEPS contractions in physics-inspired models
- **Automatic differentiation:** Optimal contraction order for forward pass can differ from backward pass
- **Language models:** Attention computation where the order of $(QK^T)V$ vs $Q(K^T V)$ matters for memory
- **Frameworks:** Integrated in NumPy, PyTorch (`torch.einsum`), JAX, TensorFlow via opt_einsum backend
- **Quantum circuit simulation:** Contraction ordering is the primary determinant of simulation cost

## Limitations

- Finding the truly optimal path is NP-hard for general tensor networks ($n > 30$ tensors)
- Greedy heuristics can miss globally optimal solutions by orders of magnitude
- Path optimization adds compilation overhead — must be cached for repeated computations
- Does not account for hardware-specific costs (memory bandwidth, cache sizes, GEMM efficiency at different shapes)
- For simple 2-3 tensor contractions, the overhead of path optimization exceeds the benefit
- Different hardware may prefer different paths (e.g., memory-optimal vs. FLOP-optimal)

## Implementation Notes

```python
# Optimal tensor contraction with opt_einsum
import opt_einsum as oe
import numpy as np

# Example: multi-head attention contraction
# Q: [batch, heads, seq, d_k]
# K: [batch, heads, seq, d_k]
# V: [batch, heads, seq, d_v]

# Find optimal contraction path
expr = 'bhid,bhjd,bhjv->bhiv'  # Attention: (Q @ K^T) @ V
shapes = [(8, 12, 512, 64), (8, 12, 512, 64), (8, 12, 512, 64)]

# Compare naive vs optimized
path_naive = [(0, 1), (0, 1)]  # Left-to-right
path_opt, path_info = oe.contract_path(expr, *[np.empty(s) for s in shapes])

print(f"Naive FLOPs: {oe.helpers.compute_size_by_dict(expr, shapes)}")
print(f"Optimal path: {path_opt}")
print(f"Speedup: {path_info.speedup:.1f}x")

# Execute with optimal path
result = oe.contract(expr, Q, K, V, optimize=path_opt)

# For repeated use, cache the expression
my_expr = oe.contract_expression(expr, *shapes, optimize='optimal')
result = my_expr(Q, K, V)  # Reuses cached optimal path

# Matrix chain example: dramatic impact
# A: 10x100, B: 100x5, C: 5x50
# (AB)C = 10*100*5 + 10*5*50 = 7,500 FLOPs
# A(BC) = 100*5*50 + 10*100*50 = 75,000 FLOPs
# Optimal chooses (AB)C -> 10x cheaper!
```

## References

- Cormen, T.H. et al. Introduction to Algorithms — Matrix Chain Multiplication (Ch. 15.2).
- Smith, D.G.A. & Gray, J. (2018). opt_einsum — A Python package for optimizing contraction order for einsum-like expressions. JOSS, 3(26), 753.
- Orgler, S. & Blacher, M. (2024). Optimizing Tensor Contraction Paths: A Greedy Algorithm Approach With Improved Cost Functions. arXiv:2405.09644.
- Gray, J. & Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum, 5, 410.

# 132: Tropical Attention via Hilbert Projective Metric

**Category**: algebraic
**Gain type**: expressivity
**Source**: Hashemi, Pasque, Teska, and Yoshida (2025). Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms. NeurIPS 2025.
**Paper**: papers/tropical-attention.pdf
**Documented**: 2026-02-15

## Description

Tropical Attention replaces the softmax-normalized dot-product attention kernel with an attention mechanism that operates natively in **tropical projective space** $\mathbb{TP}^{d-1}$. Instead of computing similarity via Euclidean inner products and normalizing with the exponential map (softmax), Tropical Attention:

1. **Tropicalizes** inputs via a valuation map $\Phi_\lambda$ into tropical projective space
2. Computes attention scores using the **tropical Hilbert projective metric** $d_{\mathbb{H}}$
3. Aggregates values via **max-plus matrix-vector product** (tropical matmul)
4. Maps results back to Euclidean space via **dequantization** $\psi(z) = \exp(z)$

The result is an attention mechanism that is **piecewise-linear**, **1-Lipschitz** (non-expansive), **idempotent** in aggregation, and **scale-invariant** — properties that match the polyhedral geometry of combinatorial algorithms. This yields sharp, non-fading attention patterns that generalize out-of-distribution in length, value, and noise, unlike softmax which disperses attention as sequence length grows.

**Key connection to semiring-monoid-lifting**: Tropical Attention is a concrete, validated instantiation of the semiring-lifting idea applied to the attention mechanism. It lifts the entire attention kernel from the standard semiring $(\mathbb{R}, +, \times)$ into the tropical semiring $(\mathbb{R} \cup \{-\infty\}, \max, +)$, demonstrating that semiring substitution can yield both theoretical and practical gains.

## Mathematical Form

**Step 1 — Tropicalization Map:**

Given input $\mathbf{X} \in \mathbb{R}^{N \times d}$, compute the tropicalization:

$$
\Phi_\lambda(\mathbf{X})_i = \mathbf{U}_i - \max_{1 \leq r \leq d} \mathbf{U}_{i,r} \cdot \mathbf{1}_d, \quad \text{where } \mathbf{U} = \log(\max(\mathbf{0}, \mathbf{X}))
$$

The valuation map for each coordinate is:

$$
v_\lambda(x) = [\phi_\lambda(x)]_i = \begin{cases} \log(x) - \lambda, & x > 0 \\ -\infty, & x \leq 0 \end{cases}
$$

This maps each row into the **tropical simplex** $\Delta^{d-1} := \{z \in \mathbb{R}^d \mid \max_i z_i = \epsilon\}$, ensuring projective equivalence.

**Step 2 — Multi-Head Tropical Attention (MHTA):**

For each head $h \in [H]$ with head dimension $d_k = d/H$:

$$
\mathbf{Q}^{(h)} = \mathbf{Z} \odot \mathbf{W}_Q^{(h)\top}, \quad \mathbf{K}^{(h)} = \mathbf{Z} \odot \mathbf{W}_K^{(h)\top}, \quad \mathbf{V}^{(h)} = \mathbf{Z} \odot \mathbf{W}_V^{(h)\top}
$$

where $\odot$ denotes **max-plus matrix multiplication**: $(\mathbf{A} \odot \mathbf{B})_{ij} = \max_l \{A_{il} + B_{lj}\}$, and $\mathbf{Z} = \Phi_\lambda(\mathbf{X})$.

**Tropical attention scores** via the Hilbert projective metric:

$$
\mathbf{S}_{ij}^{(h)} = -d_{\mathbb{H}}\!\left(\mathbf{q}_i^{(h)},\, \mathbf{k}_j^{(h)}\right)
$$

where the **tropical Hilbert projective metric** is:

$$
d_{\mathbb{H}}(x, y) = \left(\max_i (x_i - y_i)\right) - \left(\min_i (x_i - y_i)\right) = \text{diam}(x \oslash y)
$$

and $x \oslash y$ is the coordinate-wise tropical quotient $(x_1 - y_1, \ldots, x_d - y_d)$.

**Step 3 — Tropical Aggregation:**

$$
\mathbf{C}_i^{(h)} = \bigoplus_{j=1}^{N} \mathbf{S}_{ij}^{(h)} \odot \mathbf{v}_j^{(h)} = \max_j \left\{ \mathbf{S}_{ij}^{(h)} + \mathbf{v}_j^{(h)} \right\}
$$

**Step 4 — Dequantization (back to Euclidean space):**

$$
\mathbf{H} = \left[\psi(\mathbf{C}^{(1)}) \| \cdots \| \psi(\mathbf{C}^{(H)})\right] \in \mathbb{R}^{N \times d}, \quad \psi(z) = \exp(z)
$$

**Why This Works — Key Properties:**

1. **Projective invariance**: $d_{\mathbb{H}}(x + c\mathbf{1}, y + c\mathbf{1}) = d_{\mathbb{H}}(x, y)$ — only relative relations matter, making it robust to distribution shifts
2. **1-Lipschitz (non-expansive)**: Every tropical linear map $A: \mathbb{T}^{d+1} \to \mathbb{T}^{m+1}$ satisfies $d_{\mathbb{H}}(Ax, Ay) \leq d_{\mathbb{H}}(x, y)$ — perturbations cannot amplify through layers
3. **Idempotence**: $\max(a, a) = a$ — the aggregation is sharp, not blurred

**Comparison with Softmax Attention:**

| Property | Softmax | Tropical |
|----------|---------|----------|
| Kernel | $\exp(\langle q, k \rangle / \tau)$ | $-d_{\mathbb{H}}(q, k)$ |
| Normalization | $\sum_j$ (sum-to-one) | None (idempotent $\max$) |
| Decision boundary | Smooth quadratic | Sharp piecewise-linear |
| Length scaling | Attention fading as $N \to \infty$ | Scale-invariant |
| Perturbation sensitivity | $e^\delta$ multiplicative | $\leq \epsilon$ additive (1-Lipschitz) |
| Temperature parameter | Required ($\tau$) | Not needed |

## Complexity

| Operation | Softmax Attention | Tropical Attention |
|-----------|------------------|-------------------|
| Score computation | $O(N^2 d)$ — dot products | $O(N^2 d)$ — Hilbert metric |
| Normalization | $O(N^2)$ — softmax | $O(0)$ — none needed |
| Aggregation | $O(N^2 d)$ — weighted sum | $O(N^2 d)$ — max-plus |
| Total | $O(N^2 d)$ | $O(N^2 d)$ |

**Memory:** Same $O(Nd + N^2)$ as standard attention.

**Practical speedups (from paper):**

| Model | CPU (ms) | GPU (ms) | Params |
|-------|----------|----------|--------|
| Vanilla UT w/ ACT | 6.285 | 0.027 | 50,242 |
| Adaptive UT w/ ACT | 7.898 | 0.018 | 50,242 |
| **Tropical Transformer** | **1.949** | **0.003** | **40,961** |

Tropical Attention is **3×–9× faster at inference** with **~20% fewer parameters** than Universal Transformer baselines.

## Applicability

- **Neural algorithmic reasoning**: Tropical Attention is natively aligned with dynamic programming and combinatorial optimization. It outperforms all baselines on PTIME, NP-hard, and NP-complete problems (Knapsack, BinPacking, ConvexHull, Floyd-Warshall, etc.).
- **Length generalization in transformers**: Because the Hilbert metric is scale-invariant and attention doesn't fade, Tropical Transformers generalize from training length 8 to test length 1024 without degradation.
- **Long-range sequence modeling**: On the Long Range Arena (LRA) benchmark, Tropical Transformer places 2nd overall (72.79% avg), competitive with specialized efficient attention methods.
- **Reasoning models**: The sharp polyhedral decision boundaries make Tropical Attention a candidate for Large Reasoning Models that must maintain precise logical structure.
- **Hybrid semiring architectures**: Can be combined with standard softmax attention — tropical attention for reasoning heads, softmax for soft matching heads.

## Limitations

- **Not validated at LLM scale**: All experiments are on small models for algorithmic tasks. Performance on autoregressive next-token prediction language tasks is unknown.
- **Hardware bottleneck**: Max-plus operations run on CUDA cores, not Tensor Cores (~16× throughput penalty). Would benefit from SIMD²-style hardware support.
- **Sparse gradients**: The $\max$ operation has sparse gradients (only the "winning" element gets gradient), which may cause training instability at scale.
- **Tropicalization overhead**: The $\log \circ \max(0, \cdot)$ valuation map and $\exp$ dequantization add computational overhead and numerical sensitivity (log of small values, exp of large values).
- **No causal masking**: The paper doesn't discuss causal (autoregressive) tropical attention, which would require masking in the max-plus aggregation.
- **Theoretical head-width bound**: A shallow MHTA approximating tropical transitive closure may require head-width proportional to $O(n^2 2^k)$ in worst case, though depth-$T$ stacks reduce this to polynomial.

## Implementation Notes

```python
import torch
import torch.nn as nn

class TropicalAttention(nn.Module):
    """
    Multi-Head Tropical Attention (MHTA).
    Replaces softmax dot-product attention with tropical Hilbert projective metric.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Tropical linear projections (applied via max-plus matmul)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_k))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_k))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_k))

    def tropicalize(self, X):
        """Map to tropical projective space: Φ_λ(X)"""
        U = torch.log(torch.clamp(X, min=1e-8))  # Valuation map
        # Project to tropical simplex: subtract max per row
        U = U - U.max(dim=-1, keepdim=True).values
        return U

    def maxplus_matmul(self, A, B):
        """Max-plus matrix multiplication: C_ij = max_k(A_ik + B_kj)"""
        # A: (..., m, k), B: (..., k, n) -> C: (..., m, n)
        return (A.unsqueeze(-1) + B.unsqueeze(-3)).max(dim=-2).values

    def hilbert_metric(self, q, k):
        """Tropical Hilbert projective metric: d_H(q, k) = max(q-k) - min(q-k)"""
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)  # (B, H, N, N, d_k)
        d_H = diff.max(dim=-1).values - diff.min(dim=-1).values
        return d_H

    def forward(self, X):
        B, N, d = X.shape

        # Step 1: Tropicalize
        Z = self.tropicalize(X)  # (B, N, d)

        # Step 2: Max-plus linear projections per head
        heads_out = []
        for h in range(self.n_heads):
            Q = self.maxplus_matmul(Z, self.W_Q[h])   # (B, N, d_k)
            K = self.maxplus_matmul(Z, self.W_K[h])   # (B, N, d_k)
            V = self.maxplus_matmul(Z, self.W_V[h])   # (B, N, d_k)

            # Step 3: Tropical attention scores via Hilbert metric
            S = -self.hilbert_metric(Q, K)  # (B, N, N), negated distance

            # Step 4: Tropical aggregation (max-plus matmul with values)
            C = (S.unsqueeze(-1) + V.unsqueeze(-3)).max(dim=-2).values

            # Step 5: Dequantize back to Euclidean space
            heads_out.append(torch.exp(C))

        # Concatenate heads
        return torch.cat(heads_out, dim=-1)  # (B, N, d)

# Key theoretical result (Theorem 3.2):
# MHTA stacks can simulate any max-plus Bellman recursion:
#   d_v(t+1) = ⊕_{u: (u,v)∈E} (w_uv ⊙ d_u(t))
# making them universal approximators of tropical circuits / DP algorithms.
```

Code: https://github.com/Baran-phys/Tropical-Attention

## References

- Hashemi, Pasque, Teska, and Yoshida (2025). Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms. NeurIPS 2025. arXiv:2505.17190.
- Joswig (2021). Essentials of Tropical Convexity. AMS.
- Maclagan and Sturmfels (2015). Introduction to Tropical Geometry. AMS.
- Akian, Gaubert, and Guterman (2012). Tropical polyhedra are equivalent to mean payoff games. Int. J. Algebra and Computation.
- Veličković et al. (2024). softmax is not enough (for sharp out-of-distribution).
- Smets, Donker, and Portegies (2024). Semiring Activation in Neural Networks. arXiv:2405.18805.

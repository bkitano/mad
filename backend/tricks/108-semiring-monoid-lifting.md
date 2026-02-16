# 108: Semiring Monoid Lifting

**Category**: algebraic
**Gain type**: expressivity
**Source**: Smets, Donker, and Portegies (2024). Semiring Activation in Neural Networks. arXiv:2405.18805.
**Paper**: papers/semiring-activation.pdf
**Documented**: 2026-02-11

## Description

Replace the standard linear+activation pattern in neural networks with *quasilinear operators* over alternative semirings — algebraic structures $(R, \oplus, \odot)$ that generalize the standard $(\mathbb{R}, +, \times)$. The tropical (max-plus) semiring, logarithmic semirings, and min-plus semiring each define different notions of "linearity" that can be computed via matrix-like operations. This directly addresses the question of whether monoids beyond standard matrix multiplication can be hardware-efficient and useful for sequence models.

**Key insight for the user's question:** A semiring $(R, \oplus, \odot)$ naturally defines a monoid via "matrix multiplication" where the inner product uses $\oplus$ instead of $+$ and $\odot$ instead of $\times$. The tropical semiring $(\mathbb{R} \cup \{-\infty\}, \max, +)$ gives "max-plus matrix multiplication" which forms a monoid where the operation is: $C_{ij} = \max_k(A_{ik} + B_{kj})$. Crucially:

1. **Standard matmul hardware can't directly compute semiring matmul** — tensor cores are wired for $(+, \times)$, so tropical operations run on general-purpose CUDA cores at ~16× lower throughput.
2. **The logarithmic semiring** $(\mathbb{R}, \oplus_\mu, +)$ where $a \oplus_\mu b = \frac{1}{\mu}\log(e^{\mu a} + e^{\mu b})$ smoothly interpolates between the standard and tropical semirings, and its backward pass is literally the softmax function — suggesting a natural connection to attention mechanisms.
3. **Any semiring gives a monoid of matrices** under the generalized matmul, but only the standard semiring maps to tensor core hardware.

## Mathematical Form

**Core Operation (Quasilinear Map):**

Given a semiring $(R, \oplus, \odot)$, the quasilinear operator $B \odot x$ for $B \in R^{m \times n}$, $x \in R^n$ is:

$$
(B \odot x)_i = \bigoplus_{j=1}^{n} b_{ij} \odot x_j
$$

This satisfies the quasilinearity property:

$$
B \odot (a \odot x \oplus b \odot y) = a \odot (B \odot x) \oplus b \odot (B \odot y)
$$

**Key Semirings:**

**Standard (linear):** $(\mathbb{R}, +, \cdot)$ — standard matrix multiplication

**Tropical (max-plus):** $R_{\max} = (\mathbb{R} \cup \{-\infty\}, \max, +)$

$$
y_i = \max_{j=1 \ldots m} \{w_{ij} + x_j\}
$$

- Additive identity: $0_R = -\infty$
- Multiplicative identity: $1_R = 0$
- Idempotent: $\max(a, a) = a$
- Backward pass: $\frac{\partial y_i}{\partial x_j} = \begin{cases} 1 & \text{if } j = \arg\max_k (w_{ik} + x_k) \\ 0 & \text{else} \end{cases}$

**Min-plus:** $R_{\min} = (\mathbb{R} \cup \{+\infty\}, \min, +)$ — isomorphic to max-plus via negation

**Logarithmic:** $R_{\log}^\mu = (\mathbb{R} \cup \{-\infty\}, \oplus_\mu, +)$ where:

$$
a \oplus_\mu b := \frac{1}{\mu} \log\left(e^{\mu a} + e^{\mu b}\right)
$$

- For $\mu > 0$: approaches $\max$ as $\mu \to \infty$
- For $\mu \to 0^+$: approaches standard addition
- Backward pass is **softmax**: $\frac{\partial y_i}{\partial x_j} = \frac{e^{\mu(x_j + w_{ij})}}{\sum_{k=1}^{n} e^{\mu(x_k + w_{ik})}}$

**Neural Network Architecture:**

Instead of: $A_L \circ \sigma \circ A_{L-1} \circ \sigma \circ \cdots \circ A_1$ (linear ops + fixed activation)

Use: $A_L \circ B_{L-1} \circ A_{L-1} \circ B_1 \circ A_1$ (linear ops + trainable semiring ops)

where $A_i$ are standard linear maps and $B_i$ are quasilinear operators over a chosen semiring.

**Fair Tropical Initialization:**

For max-plus operators $R_{\max}^m \to R_{\max}^n$:

$$
w_{ij} = \text{Unif}[-\varepsilon, \varepsilon] + \begin{cases} 0 & \text{if } i \equiv j \mod m \\ -K & \text{else} \end{cases}
$$

This ensures fair gradient distribution at initialization (each input "wins" roughly $\frac{n}{m}$ outputs).

## Complexity

| Operation | Standard Linear | Tropical (max-plus) | Logarithmic |
|-----------|----------------|--------------------|----|
| Forward (per element) | 1 multiply + 1 add | 1 add + 1 compare | 1 add + 1 exp + 1 log |
| Hardware | Tensor cores (FP16) | CUDA cores | CUDA cores |
| Relative throughput | $1\times$ | $\sim \frac{1}{16}\times$ | $\sim \frac{1}{16}\times$ |

**Memory:** Same as standard linear layers ($O(mn)$ for weight matrix)

**Critical hardware gap:** GPUs have dedicated tensor cores for $(+, \times)$ matmul at 16× throughput vs general-purpose CUDA cores. Semiring operations must run on CUDA cores, making them significantly slower in practice.

## Applicability

- **Alternative activation functions**: Tropical semiring as a trainable replacement for ReLU/max-pooling
- **Attention mechanisms**: The logarithmic semiring's backward pass *is* softmax — suggesting that attention is already computing in a log-semiring
- **Sequence models**: If parallel scans over alternative semirings could be hardware-accelerated, they would enable different monoid structures for state transitions
- **Image processing**: Max/min pooling are already tropical convolutions with fixed kernels; making them trainable yields morphological neural networks

## Limitations

- **Hardware penalty**: ~16× slower than tensor-core matmul because GPUs lack dedicated semiring acceleration — this is the fundamental bottleneck
- **Training instability**: Tropical operators have sparse gradients (only the "winning" element gets gradient), requiring careful initialization and separate learning rates
- **Not a drop-in replacement**: Requires new initialization schemes, separate optimizers for semiring parameters
- **Limited scale validation**: Only tested on small models (FashionMNIST, ConvNeXt), not at LLM scale
- **FP16 considerations**: $e^{\mu a}$ can overflow in FP16 for large $\mu$ values, requiring log-space computation

## Implementation Notes

```python
import torch
import torch.nn as nn

class TropicalMaxPlusLinear(nn.Module):
    """Max-plus semiring linear operator: y_i = max_j(w_ij + x_j)"""
    def __init__(self, in_features, out_features, K=1.0):
        super().__init__()
        # Fair tropical initialization
        W = torch.full((out_features, in_features), -K)
        for i in range(out_features):
            W[i, i % in_features] = 0.0
        noise = torch.empty_like(W).uniform_(-K/2, K/2)
        self.weight = nn.Parameter(W + noise)

    def forward(self, x):
        # y_i = max_j (w_ij + x_j)
        return (self.weight.unsqueeze(0) + x.unsqueeze(1)).max(dim=-1).values

class LogSemiringLinear(nn.Module):
    """Logarithmic semiring operator — backward pass is softmax"""
    def __init__(self, in_features, out_features, mu=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.mu = mu

    def forward(self, x):
        # a ⊕_μ b = (1/μ) log(exp(μa) + exp(μb))
        logits = self.mu * (self.weight.unsqueeze(0) + x.unsqueeze(1))
        return (1.0 / self.mu) * torch.logsumexp(logits, dim=-1)
```

Code: https://github.com/bmnsmets/semitorch

## References

- Smets, Donker, and Portegies (2024). Semiring Activation in Neural Networks. arXiv:2405.18805.
- Zhang et al. (2019). Tropical Geometry of Deep Neural Networks.
- cuASR: CUDA Algebra for Semirings. https://github.com/hpcgarage/cuASR
- Ritter and Sussner (1996). Introduction of Morphological Neural Networks.

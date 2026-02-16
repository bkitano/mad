# 024: CDVFT: Circulant and Diagonal Vector Fine-Tuning

**Category**: decomposition
**Gain type**: efficiency
**Source**: Ding et al. "Parameter-Efficient Fine-Tuning with Circulant and Diagonal Vectors" (IJCAI 2025)
**Paper**: [papers/cdvft-circulant-diagonal-peft.pdf]
**Documented**: 2026-02-15

## Description

CDVFT represents the weight change matrix $\Delta W$ during fine-tuning as a product of interleaved circulant and diagonal matrices, each defined by a single vector. This exploits the classical result of Huhtanen & Peramaki (2015) that any complex matrix $X \in \mathbb{C}^{n \times n}$ can be factored into at most $2n - 1$ alternating circulant and diagonal factors.

The key computational advantage is that **the weight change matrix $\Delta W$ is never explicitly constructed**. Instead, the forward pass applies a sequence of element-wise multiplications (for diagonal factors) and 1D FFTs (for circulant factors) directly to the input vector. This reduces computational complexity from the $O(d^2 \log(d^2))$ of FourierFT (which uses 2D FFT to reconstruct $\Delta W$) down to $O(md \log d)$ using only 1D FFTs, where $m$ is the number of factor pairs (typically $m = 2$, giving just 3 total factors).

For non-square weight matrices, CDVFT partitions the circulant matrix into blocks, converting the problem into multiple independent square circulant matrix-vector products. In practice, only $m = 2$ (two diagonal matrices and one circulant matrix) suffices, yielding 5.33$\times$ fewer parameters than LoRA and 51.81$\times$ fewer FLOPs than FourierFT on RoBERTa.

## Mathematical Form

**Huhtanen-Peramaki Factorization:**

Any matrix $X \in \mathbb{C}^{n \times n}$ can be decomposed as:

$$
X = A_{2n-1} \times C_{2n-2} \times \ldots \times C_{2j} \times A_{2j-1} \times \ldots \times A_3 \times C_2 \times A_1
$$

where for $j \in \{1, \ldots, n\}$, $A_{2j-1}$ are diagonal matrices and $C_{2j}$ are circulant matrices, with total factor count not exceeding $2n - 1$.

**CDVFT Weight Change Decomposition:**

For $m$ diagonal-circulant pairs (total $2m - 1$ factors):

$$
\Delta W = A_{2m-1} \times C_{2m-2} \times A_{2m-3} \times \cdots \times C_2 \times \text{diag}(a_1)
$$

$$
= \text{diag}(a_{2m-1}) \times \text{circ}(c_{2m-2}) \times \text{diag}(a_{2m-3}) \times \cdots \times \text{circ}(c_2) \times \text{diag}(a_1)
$$

where each factor is defined by a single $d$-dimensional vector: $a_{2j-1} \in \mathbb{R}^d$ for diagonal matrices and $c_{2j} \in \mathbb{R}^d$ for circulant matrices.

**Diagonal Matrix:**

$$
\text{diag}(a_{2j-1}) = \begin{bmatrix} a_{2j-1}^1 & 0 & \cdots & 0 \\ 0 & a_{2j-1}^2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & 0 & a_{2j-1}^d \end{bmatrix}
$$

**Circulant Matrix:**

$$
\text{circ}(c_{2j}) = \begin{bmatrix} c_{2j}^1 & c_{2j}^2 & \cdots & c_{2j}^d \\ c_{2j}^d & c_{2j}^1 & \cdots & c_{2j}^{d-1} \\ \vdots & & \ddots & \vdots \\ c_{2j}^2 & c_{2j}^3 & \cdots & c_{2j}^1 \end{bmatrix}
$$

**Forward Pass (right-to-left computation):**

Let $x \in \mathbb{R}^{d \times 1}$ be the input. Define intermediate results:

$$
y_0 = x, \quad y_{2j-1} = a_{2j-1} \odot y_{2j-2}, \quad y_{2j} = C_{2j} \times y_{2j-1}
$$

where $\odot$ denotes element-wise multiplication. The circulant-vector product uses FFT:

$$
F_{2j-1} = \text{FFT}(y_{2j-1}), \quad F_{2j} = \text{FFT}(c_{2j})
$$

$$
\hat{F} = F_{2j-1} \odot F_{2j}, \quad y_{2j} = \text{IFFT}(\hat{F})
$$

The final output:

$$
h' = h + \Delta h = W \times x + \alpha \times \Delta W \times x
$$

where $\alpha$ is a scaling hyperparameter (as in LoRA).

**Backward Pass:**

For diagonal factors:

$$
\frac{\partial \mathcal{L}}{\partial a_{2j-1}} = \frac{\partial \mathcal{L}}{\partial y_{2j-1}} \odot y_{2j-2}
$$

For circulant factors (using conjugate symmetry for real inputs):

$$
\frac{\partial \mathcal{L}}{\partial y_{2j-1}} = \text{IFFT}(\text{conj}(F_{2j}) \odot F_y)
$$

$$
\frac{\partial \mathcal{L}}{\partial c_{2j}} = \text{IFFT}(\text{conj}(F_{2j-1}) \odot F_y)
$$

where $F_y = \text{FFT}(\frac{\partial \mathcal{L}}{\partial y_{2j}})$. The conjugate trick avoids needing FFT on shifted vectors.

**Block Partition (for non-square matrices):**

For $C \in \mathbb{R}^{n \times n}$, partition into blocks of size $p$:

$$
q_1 = \lceil d_1 / p \rceil, \quad q_2 = \lceil d_2 / p \rceil
$$

$$
h = Cx = \{h_i\}_{i=0}^{q_1-1}, \quad h_i = \sum_{j=0}^{q_2-1} \text{IFFT}(\text{FFT}(c_{i,j}) \odot \text{FFT}(x_j))
$$

**Key Definitions:**

- $\Delta W \in \mathbb{R}^{d \times d}$ — weight change matrix (never explicitly constructed)
- $a_{2j-1} \in \mathbb{R}^d$ — defining vector for diagonal factor $j$
- $c_{2j} \in \mathbb{R}^d$ — defining vector for circulant factor $j$
- $m$ — number of diagonal-circulant pairs (typically $m = 2$)
- $p$ — block partition size for non-square matrices
- $\alpha$ — scaling hyperparameter (same role as in LoRA)
- $L_t$ — number of layers being fine-tuned

## Complexity

| Method | Parameters | FLOPs | Memory (Other) |
|--------|-----------|-------|----------------|
| LoRA | $2 \times d \times L_t \times r$ | $O(r(d_1 + d_2))$ | $0$ |
| FourierFT | $n \times L_t$ | $O(d^2 \log(d^2))$ | — |
| CDVFT | $(2m-1) \times d \times L_t$ | $O(md \log(d))$ | — |

**Key complexity comparison (for $m = 2$, $d = 768$, RoBERTa-base):**

| Metric | CDVFT vs LoRA | CDVFT vs FourierFT |
|--------|--------------|-------------------|
| Parameters | $5.33 \times$ fewer | Similar |
| FLOPs | Similar | $51.81 \times$ fewer |

**Memory:** Each factor is stored as a single vector of length $d$ or $p$ (for block partition), giving linear storage complexity $O((2m-1) \times d)$ per layer.

**Forward pass cost:** $m$ element-wise multiplications ($O(d)$ each) + $(m-1)$ FFT/IFFT pairs ($O(d \log d)$ each) = $O(md \log d)$.

**Backward pass cost:** Similar to forward — $2$ FFT + $1$ IFFT for forward, $2$ IFFT + $1$ FFT for backward.

## Applicability

- **Transformer fine-tuning (PEFT)**: Direct replacement for LoRA in query/value weight matrices. Demonstrated on RoBERTa (NLU), ViT (vision), and LLaMA-2-7B (instruction tuning) with competitive or better accuracy
- **Memory-constrained fine-tuning**: The linear parameter count ($O(d)$ per layer vs $O(rd)$ for LoRA) makes CDVFT ideal for fine-tuning on consumer GPUs or edge devices
- **Fourier-domain adaptation**: Part of the broader family of spectral PEFT methods (alongside FourierFT), but avoids the expensive 2D FFT reconstruction step
- **Any dense linear layer**: The circulant-diagonal factorization applies to any weight matrix; non-square matrices are handled via block partitioning
- **Multi-task fine-tuning**: The compact vector representation (3 vectors per layer for $m = 2$) enables efficient storage of many task-specific adapters

## Limitations

- **Square matrix constraint**: The interleaved circulant-diagonal product inherently produces square matrices; non-square weights require block partitioning with associated overhead and padding
- **Expressivity ceiling**: With small $m$, the factorization may not approximate arbitrary $\Delta W$ well. The paper uses $m = 2$ (3 factors), which limits the rank of representable changes compared to full fine-tuning
- **FFT overhead for small dimensions**: For very small $d$, the FFT/IFFT overhead may not be amortized, and direct matrix multiply could be faster
- **Block partition size sensitivity**: Performance on LLaMA-2-7B varies with block partition size $p$; optimal $p$ depends on the model and task (Table 3 shows accuracy can decrease for some $p$ values)
- **Limited to additive adaptation**: Like LoRA, CDVFT only learns an additive $\Delta W$; multiplicative adaptations (like OFT) require a different approach
- **Complex-valued intermediate computation**: Despite using 1D FFT (not 2D), the FFT operations still involve complex arithmetic internally

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft

class CDVFTAdapter(nn.Module):
    """CDVFT: Circulant and Diagonal Vector Fine-Tuning.

    Represents ΔW as product of interleaved circulant and diagonal
    matrices, each defined by a single learnable vector.
    Forward pass uses element-wise multiply + 1D FFT (no matrix construction).
    """

    def __init__(self, d, m=2, alpha=1.0, block_size=None):
        """
        Args:
            d: dimension of weight matrix (assumes square; use block_size for non-square)
            m: number of diagonal-circulant pairs (total 2m-1 factors)
            alpha: scaling factor (as in LoRA)
            block_size: partition size for non-square matrices (None = no partitioning)
        """
        super().__init__()
        self.d = d
        self.m = m
        self.alpha = alpha
        self.p = block_size or d

        # Learnable vectors: m diagonal vectors + (m-1) circulant vectors
        # = 2m - 1 total vectors, each of dimension p
        self.diag_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(self.p) * 0.01)
            for _ in range(m)
        ])
        self.circ_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(self.p) * 0.01)
            for _ in range(m - 1)
        ])

    def forward(self, x):
        """
        x: (batch, d) input activation
        Returns: (batch, d) delta output (add to frozen layer output)
        """
        y = x  # (batch, d)

        # Apply factors right-to-left: diag, circ, diag, circ, ..., diag
        # Factor 1: diagonal
        y = y * self.diag_vectors[0]  # element-wise multiply

        for i in range(self.m - 1):
            # Circulant factor: y = circ(c) @ y via FFT
            Y_fft = fft.fft(y, dim=-1)
            C_fft = fft.fft(self.circ_vectors[i])
            y = fft.ifft(Y_fft * C_fft, dim=-1).real

            # Diagonal factor
            y = y * self.diag_vectors[i + 1]

        return self.alpha * y


class CDVFTLinear(nn.Module):
    """Frozen linear layer + CDVFT adapter (mergeable after fine-tuning)."""

    def __init__(self, linear: nn.Linear, m=2, alpha=1.0, block_size=None):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        d = min(linear.in_features, linear.out_features)
        self.adapter = CDVFTAdapter(d, m=m, alpha=alpha, block_size=block_size)

    def forward(self, x):
        # Frozen pretrained output + learned delta
        return self.linear(x) + self.adapter(x)
```

## References

- Ding, X., Chen, L., Liao, S. & Wang, Z. "Parameter-Efficient Fine-Tuning with Circulant and Diagonal Vectors" IJCAI 2025. arXiv:2505.00580
- Huhtanen, M. & Peramaki, A. "Factoring matrices into the product of circulant and diagonal matrices" Journal of Fourier Analysis and Applications, 21:1018-1033, 2015
- Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models" ICLR 2022. arXiv:2106.09685
- Gao, Z. et al. "Parameter-efficient fine-tuning with discrete fourier transform" ICML 2024. arXiv:2405.03003
- Cheng, Y. et al. "An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections" ICCV 2015

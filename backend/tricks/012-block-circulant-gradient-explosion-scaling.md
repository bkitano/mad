# 012: Block Circulant Gradient Explosion Scaling (BCA Learning Rate Heuristic)

**Category**: stability
**Gain type**: efficiency
**Source**: Ding et al. "Block Circulant Adapter for Large Language Models" (IJCAI 2025)
**Paper**: [papers/block-circulant-adapter-llm.pdf]
**Documented**: 2026-02-15

## Description

Block circulant matrices used as fine-tuning adapters for LLMs suffer from a **gradient explosion** problem that scales linearly with the block partition size $p$. This trick provides both the theoretical analysis of *why* circulant gradients explode and a simple, effective heuristic to fix it: **scale the learning rate by $1/p$**.

The key insight, proven in Propositions 1-3 of the paper, is that the Jacobian of a circulant matrix with respect to its defining vector involves powers of the cyclic permutation matrix $\mathbf{P}$. For a single circulant matrix $\text{circ}(\mathbf{c}) \in \mathbb{R}^{n \times n}$, the gradient $\nabla f(\mathbf{c}_i) = \nabla f(\mathbf{h})^\top \mathbf{P}^i \mathbf{x}$, which is a bilinear form bounded by $\|\nabla f(\mathbf{h})\| \times \|\mathbf{x}\|$. However, summing over the cyclic permutation structure yields:

$$
\min_i\{\nabla f(\mathbf{c}_i)\} = \min_i\left\{\sum_{j=0}^{n-1} \nabla f(\mathbf{h}_j) \mathbf{x}_{j-i}\right\} \geq n \times \min\{\nabla f(\mathbf{A})\}
$$

This means circulant layer gradients are at least $n$ times larger than dense layer gradients. For **block circulant** matrices with partition size $p$, the amplification factor is $p$ — gradients are $p$ times larger than the equivalent dense matrix.

This is fundamentally different from RNN gradient explosion (which compounds through time steps). Block circulant gradient explosion is **structural**: it arises from the parameter sharing inherent in circulant structure (each weight parameter $c_i$ influences $p$ output elements through cyclic shifts), so the gradient accumulates $p$ terms. The explosion does not propagate backward through layers — it only affects the circulant parameters themselves.

The fix is correspondingly simple: divide the learning rate by the partition size $p$:

$$
\alpha \leftarrow \alpha / p
$$

This ensures that despite the $p$-fold gradient amplification, each parameter update has approximately the same magnitude as it would for a dense matrix, preventing divergence while preserving convergence speed.

## Mathematical Form

**Block Circulant Adapter Forward Pass:**

For a pretrained weight matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$, the adapter learns the weight change $\Delta \mathbf{W} = \mathbf{B}$, a block circulant matrix with partition size $p$ (block size $p$, $q = n/p$ blocks per dimension):

$$
\mathbf{h} + \Delta \mathbf{h} = \mathbf{W}\mathbf{x} + \Delta \mathbf{W}\mathbf{x} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{x}
$$

where:

$$
\mathbf{h}_i = \sum_{j=0}^{q-1} \mathbf{B}_{i,j} \mathbf{x}_j = \sum_{j=0}^{q-1} \text{IFFT}(\text{FFT}(\mathbf{c}_{i,j}) \circ \text{FFT}(\mathbf{x}_j))
$$

**Circulant Matrix as Polynomial in Permutation Matrix:**

$$
\text{circ}(\mathbf{c}) = c_0 \mathbf{I} + c_1 \mathbf{P} + c_2 \mathbf{P}^2 + \cdots + c_{n-1} \mathbf{P}^{n-1}
$$

where:

$$
\mathbf{P} = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \\ 1 & 0 & & & 0 \\ 0 & \ddots & \ddots & & \vdots \\ \vdots & & \ddots & \ddots & 0 \\ 0 & \cdots & 0 & 1 & 0 \end{bmatrix}
$$

is the cyclic permutation matrix.

**Proposition 1 — Bounded Bilinear Gradient:**

Given $\mathbf{c} \in \mathbb{R}^{n \times 1}$, let $\mathbf{h} = \text{circ}(\mathbf{c})\mathbf{x}$ and $f(\mathbf{h}): \mathbb{R}^{n \times 1} \to \mathbb{R}$ be differentiable. The Jacobian $\mathbf{J}$ of the mapping from $\mathbf{c}$ to $\mathbf{h}$ is:

$$
\mathbf{J} = [\mathbf{I}, \mathbf{P}, \mathbf{P}^2, \ldots, \mathbf{P}^{n-1}]\mathbf{x}
$$

The first-order derivative:

$$
\nabla f(\mathbf{c}) = \nabla f(\mathbf{h})^\top \mathbf{J}
$$

$$
\nabla f(\mathbf{c}_i) = \nabla f(\mathbf{h})^\top \mathbf{P}^i \mathbf{x} \leq \|\nabla f(\mathbf{h})\| \times \|\mathbf{x}\|
$$

**Proposition 2 — Circulant Gradient Amplification ($n\times$):**

For a general dense matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ with $\mathbf{g} = \mathbf{A}\mathbf{x}$ and the same loss $f$:

$$
\nabla f(\mathbf{c}_i) = \sum_{j=0}^{n-1} \nabla f(\mathbf{h}_j) \mathbf{x}_{j-i}
$$

$$
\min_i\{\nabla f(\mathbf{c}_i)\} \geq n \times \min\{\nabla f(\mathbf{A})\}
$$

The circulant gradient is at least $n$ times larger because each circulant parameter $c_i$ contributes to all $n$ outputs through cyclic shifts.

**Proposition 3 — Block Circulant Gradient Amplification ($p\times$):**

For block circulant matrix $\mathbf{B}$ with partition size $p$, each sub-matrix $\mathbf{B}_{i,j}$ has its own cyclic permutation $\mathbf{Q} \in \mathbb{R}^{p \times p}$:

$$
\nabla f(\mathbf{c}_{i,j}) = \nabla f(\mathbf{h}_i)^\top [\mathbf{I}, \mathbf{Q}, \ldots, \mathbf{Q}^{p-1}] \mathbf{x}_j
$$

$$
\min\{\nabla f(\mathbf{c}_{i,j})\} \geq p \times \min\{\nabla f(\mathbf{A})\}
$$

**Corollary 1:** Compared with gradients of a dense matrix-based linear layer, circulant matrix-based linear layer gradient explodes by $n$, and block circulant matrix-based linear layer gradient explodes by $p$.

**Learning Rate Heuristic:**

$$
\alpha \leftarrow \alpha / p
$$

This cancels the $p$-fold gradient amplification, ensuring parameter updates have the same scale as dense layer updates.

**Key Definitions:**

- $\mathbf{B} \in \mathbb{R}^{n \times n}$ — block circulant weight change matrix (the adapter)
- $p$ — partition size (each circulant sub-block is $p \times p$)
- $q = n/p$ — number of blocks per dimension
- $\mathbf{c}_{i,j} \in \mathbb{R}^{p}$ — defining vector for circulant sub-block $\mathbf{B}_{i,j}$
- $\mathbf{P} \in \mathbb{R}^{n \times n}$ — cyclic permutation matrix (for full circulant)
- $\mathbf{Q} \in \mathbb{R}^{p \times p}$ — cyclic permutation matrix (for block circulant sub-blocks)
- $\alpha$ — base learning rate (before scaling)

## Complexity

| Method | Parameters | FLOPs | Storage |
|--------|-----------|-------|---------|
| LoRA (rank $r$) | $2nr$ | $O(nr)$ | $O(nr)$ |
| VeRA | $n + r$ | $O(nr)$ | $O(nr)$ (frozen random) |
| FourierFT | $O(n)$ | $O(n^2 \log n^2)$ (2D FFT) | $O(n)$ |
| BCA (partition $p$) | $O(n^2/p)$ | $O(\frac{n^2}{p} \log p)$ (1D FFT) | $O(n^2/p)$ |

**Key complexity comparison (RoBERTa-large, $n = 1024$):**

| Setting | Params | FLOPs |
|---------|--------|-------|
| LoRA | 0.8M | 0.8G |
| VeRA | 61K | 2.5M |
| FourierFT | 49K | 3.2G |
| BCA ($p = 512$) | 3.8M | 9.0M |
| BCA ($p = 1024$) | 1.05M | 7.5M |

**Trade-off:** Larger partition $p$ reduces parameters ($n^2/p$) and FLOPs ($\frac{n^2}{p} \log p$), but increases gradient amplification ($p\times$), requiring the learning rate heuristic. When $p = n$, the adapter is a single circulant matrix with only $n$ parameters but $n\times$ gradient amplification.

**Memory:** Linear storage $O(n^2/p)$ — one $p$-dimensional vector per circulant sub-block, $q^2 = (n/p)^2$ sub-blocks total.

## Applicability

- **LLM fine-tuning (PEFT)**: Primary application — replace LoRA in query/value projection matrices. Validated on RoBERTa (GLUE benchmark) and LLaMA-2-7B (Alpaca instruction tuning, GSM8K math reasoning). Achieves competitive accuracy with 14-16x fewer parameters than LoRA/VeRA and 32x fewer FLOPs than FourierFT
- **Any Fourier-domain PEFT method**: The gradient scaling insight applies to any method using circulant matrices for adaptation, including C³A, CDVFT, and FourierFT-based approaches that operate in the spectral domain
- **Large partition sizes**: The heuristic is most critical when $p$ is large (e.g., $p = 768$ or $1024$), where without scaling the learning rate, training diverges (demonstrated in Fig. 3 of the paper)
- **Mergeable adapters**: After fine-tuning, $\Delta W = \mathbf{B}$ can be explicitly constructed and merged into the pretrained weight $\mathbf{W}$, incurring zero inference overhead

## Limitations

- **Heuristic, not optimal**: The $\alpha/p$ scaling is a sufficient condition for convergence but may be suboptimal — more sophisticated per-parameter learning rate schedules (e.g., Adam-style adaptive methods) might partially compensate without explicit scaling
- **Accuracy-compression trade-off**: Very large $p$ (e.g., $p = n$) yields maximum compression but can still degrade accuracy. On GLUE, $p = 256$ slightly outperforms $p = 768$ for RoBERTa-base, suggesting diminishing returns from extreme compression
- **Interplay with optimizers**: The paper uses Adadelta for simulations and AdamW for fine-tuning; the gradient amplification interacts with adaptive optimizers differently than with SGD, so the $\alpha/p$ heuristic may need adjustment per optimizer
- **Does not address second-order effects**: The analysis considers only first-order gradients; second-order curvature effects (Hessian structure) under circulant parameterization are not analyzed
- **Fixed partition size**: The partition size $p$ is fixed per layer; adaptive or mixed partition sizes are not explored

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft
import math

class BlockCirculantAdapter(nn.Module):
    """Block Circulant Adapter (BCA) for LLM fine-tuning.

    Key insight: circulant gradient explodes by factor p (partition size).
    Fix: scale learning rate by 1/p.

    Args:
        n: weight matrix dimension (assumes square)
        partition_size: circulant block size p (larger = more compression)
        alpha: base scaling factor
    """

    def __init__(self, n, partition_size, alpha=1.0):
        super().__init__()
        self.n = n
        self.p = partition_size
        self.q = n // partition_size  # number of blocks per dimension

        # Apply learning rate scaling heuristic: alpha <- alpha / p
        # This compensates for p-fold gradient amplification
        self.alpha = alpha / partition_size

        # Learnable circulant defining vectors: (q, q, p)
        # One p-vector per circulant sub-block
        self.c = nn.Parameter(
            torch.randn(self.q, self.q, self.p) * 0.01
        )

    def forward(self, x):
        """
        x: (batch, n) input activation
        Returns: (batch, n) adapter output (add to frozen layer output)
        """
        batch = x.shape[0]
        p, q = self.p, self.q

        # Partition input into blocks: (batch, q, p)
        x_blocks = x.view(batch, q, p)

        # FFT of input blocks: (batch, q, p)
        x_fft = fft.fft(x_blocks, dim=-1)

        # FFT of circulant defining vectors: (q, q, p)
        c_fft = fft.fft(self.c, dim=-1)

        # Block circulant multiply in Fourier domain:
        # For each output block i: h_i = sum_j IFFT(FFT(c_ij) * FFT(x_j))
        # (batch, 1, q, p) * (1, q, q, p) -> sum over input blocks
        out_fft = (x_fft.unsqueeze(1) * c_fft.unsqueeze(0)).sum(dim=2)

        # IFFT: (batch, q, p) -> (batch, n)
        out = fft.ifft(out_fft, dim=-1).real

        return self.alpha * out.reshape(batch, -1)

    @staticmethod
    def get_scaled_lr(base_lr, partition_size):
        """Compute the scaled learning rate for BCA parameters.

        The gradient of block circulant parameters is p times larger
        than dense matrix gradients (Proposition 3). Scale lr by 1/p
        to compensate.

        Args:
            base_lr: learning rate for dense/LoRA parameters
            partition_size: circulant block size p

        Returns:
            Scaled learning rate for BCA parameters
        """
        return base_lr / partition_size


class BCALinear(nn.Module):
    """Frozen linear layer + BCA adapter (mergeable after training)."""

    def __init__(self, linear: nn.Linear, partition_size, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        n = min(linear.in_features, linear.out_features)
        self.adapter = BlockCirculantAdapter(n, partition_size, alpha)

    def forward(self, x):
        return self.linear(x) + self.adapter(x)

    def merge_and_unload(self):
        """Merge adapter into frozen weight (zero inference cost)."""
        p, q = self.adapter.p, self.adapter.q
        c = self.adapter.c.data
        alpha = self.adapter.alpha

        # Build full circulant matrix from defining vectors
        idx = torch.arange(p)
        circ_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)) % p

        # Construct full block circulant matrix
        W_delta = torch.zeros(self.adapter.n, self.adapter.n)
        for i in range(q):
            for j in range(q):
                W_delta[i*p:(i+1)*p, j*p:(j+1)*p] = c[i, j][circ_idx]

        self.linear.weight.data += alpha * W_delta
        return self.linear


# Usage example with optimizer setup
def setup_bca_finetuning(model, base_lr=1e-4, partition_size=512):
    """Set up BCA fine-tuning with properly scaled learning rates."""
    bca_lr = BlockCirculantAdapter.get_scaled_lr(base_lr, partition_size)

    # Separate parameter groups
    bca_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'adapter' in name:
                bca_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': bca_params, 'lr': bca_lr},  # Scaled by 1/p
    ])
    return optimizer
```

## References

- Ding, X., Wang, M., Liao, S. & Wang, Z. "Block Circulant Adapter for Large Language Models" IJCAI 2025. arXiv:2505.00582
- Ding, C. et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" MICRO 2017. arXiv:1708.08917
- Cheng, Y. et al. "An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections" ICCV 2015
- Gao, Z. et al. "Parameter-efficient fine-tuning with discrete fourier transform" ICML 2024
- Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models" ICLR 2022
- Bengio, Y. et al. "Learning long-term dependencies with gradient descent is difficult" IEEE Trans. Neural Networks, 1994

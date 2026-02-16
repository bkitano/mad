# 018: C³A: Circular Convolution Adaptation

**Category**: decomposition
**Gain type**: efficiency
**Source**: Chen et al. "Parameter-Efficient Fine-Tuning via Circular Convolution" (arXiv 2024, HuggingFace PEFT)
**Paper**: [papers/c3a-circular-convolution-peft.pdf]
**Documented**: 2026-02-15

## Description

C³A (Circular Convolution Adaptation) replaces LoRA's low-rank decomposition $\Delta W = BA$ with a circulant matrix $\Delta W = \mathcal{C}(\Delta w)$ defined by a single learned vector $\Delta w \in \mathbb{R}^d$. The crucial insight is that **rank and parameter count are decoupled**: a circulant matrix defined by $d$ parameters can have rank up to $d$ (full rank), whereas LoRA with $d$ parameters is limited to rank $d / (d_1 + d_2) \ll d$. This enables high-rank adaptation with a small parameter footprint.

Both forward and backward passes are computed via FFT, exploiting the circulant diagonalization $\mathcal{C}(\Delta w) = F \text{diag}(\text{FFT}(\Delta w)) F^{-1}$. The forward pass reduces to $\Delta z = \text{FFT}(\text{FFT}(\Delta w) \odot \text{iFFT}(x))$, achieving $O(\frac{d_1+d_2}{p} \log b + \frac{d_1 d_2}{b})$ time complexity comparable to LoRA while delivering higher expressivity.

For non-square weight matrices ($d_1 \neq d_2$), C³A extends to **block-circular convolution**: the activation and output vectors are partitioned into blocks of size $b$ (a common divisor of $d_1$ and $d_2$), producing a block-circulant matrix with $\frac{d_1 d_2}{b^2}$ independent circulant kernels. The block size $b$ controls the parameter count ($\frac{d_1 d_2}{b}$ total) independently of the achievable rank. C³A is implemented in HuggingFace PEFT and outperforms LoRA on LLaMA-3-8B across commonsense reasoning, math, and code generation benchmarks.

## Mathematical Form

**Circular Convolution Operator:**

Given a convolution kernel $\Delta w \in \mathbb{R}^d$ (with $d_1 = d_2 = d$ for simplicity):

$$
\Delta z = \Delta w \star x = \mathcal{C}(\Delta w) x
$$

where $\star$ denotes circular convolution and $\mathcal{C}(\cdot)$ constructs the circulant matrix:

$$
\mathcal{C}(\Delta w) = \begin{bmatrix} \Delta w_1 & \Delta w_2 & \cdots & \Delta w_{d-1} & \Delta w_d \\ \Delta w_d & \Delta w_1 & \cdots & \Delta w_{d-2} & \Delta w_{d-1} \\ \vdots & & \ddots & & \vdots \\ \Delta w_3 & \Delta w_4 & \cdots & \Delta w_1 & \Delta w_2 \\ \Delta w_2 & \Delta w_3 & \cdots & \Delta w_d & \Delta w_1 \end{bmatrix}
$$

**Rank of Circulant Matrix:**

The rank of $\mathcal{C}(\Delta w)$ is:

$$
\text{rank}(\mathcal{C}(\Delta w)) = d - \text{Deg}(\gcd(f(x), x^d - 1))
$$

where $f(x) = \sum_{i=1}^d \Delta w_i x^{i-1}$ is the associated polynomial. The theoretical upper bound on rank is $d$ (full rank), achievable for generic $\Delta w$.

**FFT-Based Diagonalization:**

$$
\mathcal{C}(\Delta w) = \frac{1}{d} F \text{diag}(\Lambda) F^{-1}, \quad \Lambda = \text{FFT}(\Delta w)
$$

**Forward Pass via FFT:**

$$
\Delta w \star x = \text{FFT}(\text{FFT}(\Delta w) \odot \text{iFFT}(x))
$$

where $\odot$ is element-wise (Hadamard) product. The full adapted output:

$$
z = W_0 x + \alpha \cdot \mathcal{C}(\Delta w) x
$$

where $W_0$ is the frozen pretrained weight and $\alpha$ is a scaling factor.

**Backpropagation (via commutativity of circular convolution):**

Since $\mathcal{C}(\Delta w) x = \mathcal{C}(x) \Delta w$ (circular convolution is commutative):

$$
\frac{\partial \mathcal{L}}{\partial x} = \mathcal{C}(\Delta w) \frac{\partial \mathcal{L}}{\partial \Delta z}, \quad \frac{\partial \mathcal{L}}{\partial \Delta w} = \mathcal{C}(x) \frac{\partial \mathcal{L}}{\partial \Delta z}
$$

Both gradients are circular convolutions and computed via FFT:

$$
\frac{\partial \mathcal{L}}{\partial x} = \Delta w \star \frac{\partial \mathcal{L}}{\partial \Delta z}, \quad \frac{\partial \mathcal{L}}{\partial \Delta w} = x \star \frac{\partial \mathcal{L}}{\partial \Delta z}
$$

**Block-Circular Convolution (for $d_1 \neq d_2$):**

Partition $x = [x_1, x_2, \ldots, x_{d_2/b}]$ and $\Delta z = [\Delta z_1, \Delta z_2, \ldots, \Delta z_{d_1/b}]$ into blocks of size $b$, where $b = \text{CD}(d_1, d_2)$ is a common divisor:

$$
\Delta z_i = \sum_{j=1}^{d_2/b} \Delta w_{ij} \star x_j, \quad i \in \{1, 2, \ldots, d_1/b\}
$$

This produces a block-circulant delta weight:

$$
\mathcal{C}_{\text{blk}}(\Delta w) = \begin{bmatrix} \mathcal{C}(\Delta w_{11}) & \cdots & \mathcal{C}(\Delta w_{1, d_2/b}) \\ \vdots & \ddots & \vdots \\ \mathcal{C}(\Delta w_{d_1/b, 1}) & \cdots & \mathcal{C}(\Delta w_{d_1/b, d_2/b}) \end{bmatrix}
$$

where each $\Delta w_{ij} \in \mathbb{R}^b$ and the total parameter count is $\frac{d_1 d_2}{b}$.

**Key Definitions:**

- $\Delta w \in \mathbb{R}^d$ — circular convolution kernel (learnable)
- $\Delta w_{ij} \in \mathbb{R}^b$ — block-circulant kernels for non-square case
- $b$ — block size, common divisor of $d_1$ and $d_2$ (controls parameters)
- $\alpha$ — scaling hyperparameter
- $\mathcal{C}(\cdot)$ — circulant matrix constructor
- $\star$ — circular convolution operator

## Complexity

| Method | Time | Parameters | Auxiliary Memory |
|--------|------|-----------|-----------------|
| LoRA | $O(r(d_1 + d_2))$ | $r(d_1 + d_2)$ | $0$ |
| VeRA | $O(r_v(d_1 + d_2))$ | $r_v + d_1$ | $r_v(d_1 + d_2)$ |
| C³A | $O\left(\frac{d_1+d_2}{p} \log b + \frac{d_1 d_2}{b}\right)$ | $\frac{d_1 d_2}{b}$ | $pb$ |

where $p$ is the GPU parallelism degree for FFT.

**Rank-parameter trade-off:**

| Method | Rank | Parameters (for same rank $r$) |
|--------|------|-------------------------------|
| LoRA | $r$ | $r(d_1 + d_2)$ |
| C³A ($b = d$) | up to $d$ | $d$ |

For $d_1 = d_2 = 1024$ and $r = 8$: LoRA needs $16{,}384$ parameters for rank 8, while C³A uses $1{,}024$ parameters with rank up to $1{,}024$.

**Memory:** C³A's only auxiliary tensor is the FFT buffer of size $pb \leq \min(d_1, d_2)$, compared to VeRA's $r_v(d_1 + d_2)$ random projection matrices.

## Applicability

- **LLM fine-tuning (PEFT)**: Drop-in replacement for LoRA in query/value/key weight matrices. Validated on LLaMA-2-7B, LLaMA-3-8B, Mistral-7B/8x7B across commonsense reasoning (8 benchmarks), math (GSM8K, MATH), and code generation (HumanEval, MBPP)
- **Vision transformers**: Fine-tuning ViT on image classification with comparable accuracy to LoRA using fewer parameters
- **NLU tasks**: RoBERTa fine-tuning on GLUE benchmark with 0.018M parameters (vs 0.295M for LoRA) achieving competitive or better scores
- **Mergeable adapters**: After fine-tuning, $\Delta W = \mathcal{C}(\Delta w)$ can be explicitly constructed and merged into $W_0$, adding zero inference latency
- **Scalable to large models**: LLaMA-3-70B and Mistral-8x7B experiments show consistent gains, with the circulant inductive bias potentially acting as regularization for large models
- **Memory-constrained training**: Linear auxiliary memory ($pb$) makes C³A practical on single-GPU setups (H800 80GB) where VeRA and DoRA may OOM

## Limitations

- **Square matrix assumption**: Basic C³A requires $d_1 = d_2$; non-square matrices need the block-circular extension, which introduces the block size hyperparameter $b$
- **Circulant inductive bias**: The circular shift structure may not suit all tasks — it imposes translation equivariance on the adaptation, which may be suboptimal for tasks requiring position-specific weight changes
- **Block size selection**: The block size $b$ must be a common divisor of $d_1$ and $d_2$; the optimal $b$ varies by task and model, requiring tuning
- **FFT kernel overhead**: For very small block sizes, the per-kernel FFT/IFFT call overhead may dominate, though cuFFT batching helps
- **Not a universal approximator with finite parameters**: While rank can be high, the circulant structure constrains the space of representable $\Delta W$ — not all rank-$r$ matrices are circulant
- **Limited to additive adaptation**: Like LoRA, C³A learns an additive delta; multiplicative or orthogonal adaptations require different structures

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.fft as fft

class C3AAdapter(nn.Module):
    """C³A: Circular Convolution Adaptation.

    Replaces LoRA's low-rank BA with circulant C(Δw).
    Achieves high rank (up to d) with d parameters.
    Forward/backward via FFT: O(d log d).
    """

    def __init__(self, d1, d2, block_size=None, alpha=1.0):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.alpha = alpha

        if block_size is None:
            # Square case: single circulant kernel
            assert d1 == d2, "Need block_size for non-square"
            self.b = d1
            self.n_blocks_out = 1
            self.n_blocks_in = 1
        else:
            self.b = block_size
            self.n_blocks_out = d1 // block_size  # d1/b
            self.n_blocks_in = d2 // block_size    # d2/b

        # Learnable circulant kernels: (d1/b, d2/b, b)
        self.delta_w = nn.Parameter(
            torch.randn(self.n_blocks_out, self.n_blocks_in, self.b) * 0.01
        )

    def forward(self, x):
        """
        x: (batch, d2)
        Returns: (batch, d1) adaptation output
        """
        batch = x.shape[0]
        b = self.b

        # Partition input into blocks: (batch, d2/b, b)
        x_blocks = x.view(batch, self.n_blocks_in, b)

        # FFT of input blocks: (batch, d2/b, b)
        x_fft = fft.fft(x_blocks, dim=-1)

        # FFT of kernels: (d1/b, d2/b, b)
        w_fft = fft.fft(self.delta_w, dim=-1)

        # Block-circulant multiply:
        # (batch, 1, d2/b, b) * (1, d1/b, d2/b, b) -> sum over d2/b
        out_fft = (x_fft.unsqueeze(1) * w_fft.unsqueeze(0)).sum(dim=2)

        # IFFT: (batch, d1/b, b)
        out = fft.ifft(out_fft, dim=-1).real

        # Reshape: (batch, d1)
        return self.alpha * out.reshape(batch, self.d1)


# Key insight: rank is decoupled from parameter count
# LoRA rank r requires r(d1+d2) params; C³A achieves rank up to b
# with d1*d2/b params, where b is independent of param count.
# For d1=d2=d: d params can give rank d (full rank).

# Available in HuggingFace PEFT:
# from peft import C3AConfig, get_peft_model
# config = C3AConfig(target_modules=["q_proj", "v_proj"], block_size=768)
```

## References

- Chen, A., Cheng, J., Liu, Z., Gao, Z., Tsung, F., Li, Y. & Li, J. "Parameter-Efficient Fine-Tuning via Circular Convolution" arXiv:2407.19342, 2024. Available in HuggingFace PEFT
- Ding, C. et al. "CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices" MICRO 2017
- Golub, G.H. & Van Loan, C.F. "Matrix Computations" (4th ed.) — circulant diagonalization $\mathcal{C} = F \Lambda F^{-1}$
- Ingleton, A.W. "The rank of circulant matrices" Journal of the London Mathematical Society, 1956
- Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models" ICLR 2022
- Bamieh, B. "Discovering Transforms: A Tutorial on Circulant Matrices, Circular Convolution, and the Discrete Fourier Transform" arXiv:1805.05533, 2018

# 005: Bilinear Gating (GLU / SwiGLU)

**Category**: algebraic
**Gain type**: expressivity
**Source**: Dauphin et al. (2016), "Language Modeling with Gated Convolutional Networks"; Shazeer (2020), "GLU Variants Improve Transformer"
**Paper**: [papers/glu-variants-improve-transformer.pdf]
**Documented**: 2026-02-15

## Description

Replace the standard two-matrix feed-forward network (FFN) in Transformers with a *gated* variant: the component-wise (Hadamard) product of two parallel linear projections, one of which is passed through a nonlinear activation. This is the **Gated Linear Unit** (GLU) family. The core trick is that input-dependent gating of a linear pathway — a bilinear interaction — provides strictly more expressive feature selection than a single nonlinearity, while the parameter/FLOP budget can be held constant by shrinking the hidden dimension by a factor of $\frac{2}{3}$. The SwiGLU variant (Swish-gated) is now the default FFN in LLaMA, PaLM, Gemma, Mistral, and most modern LLMs.

## Mathematical Form

**Standard FFN (baseline):**

$$
\text{FFN}_{\text{ReLU}}(x, W_1, W_2) = \max(xW_1, 0) \, W_2
$$

where $x \in \mathbb{R}^d$, $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$.

**Gated Linear Unit (GLU):**

$$
\text{GLU}(x, W, V, b, c) = \sigma(xW + b) \otimes (xV + c)
$$

where $\sigma$ is the sigmoid function and $\otimes$ denotes element-wise (Hadamard) product.

**Bilinear variant (no activation):**

$$
\text{Bilinear}(x, W, V, b, c) = (xW + b) \otimes (xV + c)
$$

**GLU Variant FFN Layers (bias-free, as used in practice):**

$$
\text{FFN}_{\text{GLU}}(x, W, V, W_2) = (\sigma(xW) \otimes xV) \, W_2
$$

$$
\text{FFN}_{\text{Bilinear}}(x, W, V, W_2) = (xW \otimes xV) \, W_2
$$

$$
\text{FFN}_{\text{ReGLU}}(x, W, V, W_2) = (\max(0, xW) \otimes xV) \, W_2
$$

$$
\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV) \, W_2
$$

$$
\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV) \, W_2
$$

where $\text{Swish}_\beta(x) = x \sigma(\beta x)$ and $\text{GELU}(x) = x \Phi(x)$.

**Key Definitions:**

- $x \in \mathbb{R}^d$ — input hidden representation at a single position
- $W, V \in \mathbb{R}^{d \times d_{ff}'}$ — two parallel "up-projection" matrices (the gate and value paths)
- $W_2 \in \mathbb{R}^{d_{ff}' \times d}$ — down-projection matrix
- $d_{ff}' = \frac{2}{3} d_{ff}$ — reduced hidden dimension to match the FLOP budget of a standard 2-matrix FFN

**The Isoparametric Trick:**

The GLU family uses **three** weight matrices ($W$, $V$, $W_2$) instead of two ($W_1$, $W_2$). To maintain the same parameter count and compute:

$$
d_{ff}' = \frac{2}{3} d_{ff}
$$

This ensures $\text{params}(W) + \text{params}(V) + \text{params}(W_2) = 2 \cdot d \cdot \frac{2}{3} d_{ff} + \frac{2}{3} d_{ff} \cdot d = 2 d \cdot d_{ff} = \text{params}(W_1) + \text{params}(W_2)$.

**Why Bilinear Gating Helps:**

The element-wise product $\sigma(xW) \otimes xV$ creates an *input-dependent multiplicative interaction*: the gate path $\sigma(xW)$ learns *which* features to activate, while the value path $xV$ computes *what* to pass through. This is strictly more expressive than a single nonlinearity applied uniformly to all features, enabling the network to learn richer feature selection patterns.

## Complexity

| Operation | Standard FFN | GLU-variant FFN |
|-----------|-------------|-----------------|
| Parameters | $2 \cdot d \cdot d_{ff}$ | $3 \cdot d \cdot d_{ff}' = 2 \cdot d \cdot d_{ff}$ |
| FLOPs | $2 \cdot d \cdot d_{ff}$ (two matmuls) | $2 \cdot d \cdot d_{ff}' + d_{ff}'$ (two matmuls + Hadamard) $\approx 2 \cdot d \cdot d_{ff}$ |

**Memory:** Identical to standard FFN (same parameter count). Activations require storing both gate and value path outputs during backprop, but this is offset by the smaller hidden dimension.

**Key insight:** The $\frac{2}{3}$ hidden-dimension reduction is the computational trick that makes GLU variants a free lunch — better quality at the same cost.

## Applicability

- **Transformer FFN layers:** Direct drop-in replacement. SwiGLU is the default in LLaMA, LLaMA-2, PaLM, Gemma, Mistral, and most post-2022 LLMs
- **State space models:** Mamba and Mamba-2 use SwiGLU-style gating in their MLP blocks
- **Linear attention models:** GLA, RWKV-v6, and other recurrent architectures use gated FFN layers
- **Vision transformers:** Also adopted in recent ViT architectures
- **Any position-wise FFN:** The trick applies to any architecture with a feedforward sublayer

## Limitations

- Three weight matrices require three separate matmuls, which may be less efficient than two fused matmuls on some hardware (though the smaller dimension compensates)
- The activation function choice (Swish vs GELU vs sigmoid) introduces a hyperparameter, though SwiGLU and GEGLU consistently outperform others
- No theoretical explanation for *why* these variants work better — Shazeer attributes success to "divine benevolence"
- The $\frac{2}{3}$ reduction factor is specific to matching a particular baseline; custom ratios may be needed for non-standard architectures

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward network (as used in LLaMA, PaLM, etc.).

    Uses 2/3 * d_ff hidden units to match parameter count of standard FFN.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Reduce hidden dim by 2/3 to match param count
        d_ff_prime = int(2 * d_ff / 3)
        # Round to nearest multiple of 256 for hardware efficiency
        d_ff_prime = 256 * ((d_ff_prime + 255) // 256)

        self.W = nn.Linear(d_model, d_ff_prime, bias=False)  # gate path
        self.V = nn.Linear(d_model, d_ff_prime, bias=False)  # value path
        self.W2 = nn.Linear(d_ff_prime, d_model, bias=False) # down-proj

    def forward(self, x):
        # SwiGLU: Swish(xW) ⊗ xV, then project down
        gate = F.silu(self.W(x))   # Swish_1 = SiLU
        value = self.V(x)
        return self.W2(gate * value)  # Hadamard product + down-projection

# Comparison: standard FFN
class Standard_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))
```

## References

- Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2016). Language Modeling with Gated Convolutional Networks. arXiv:1612.08083.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Mnih, A. & Hinton, G. (2007). Three New Graphical Models for Statistical Language Modelling. ICML 2007.
- Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation Functions. arXiv:1710.05941.
- Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
- Chowdhery, A. et al. (2022). PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311.

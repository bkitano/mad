# 053: Generalized Kronecker Product Decomposition (GKPD)

**Category**: decomposition
**Gain type**: efficiency
**Source**: Hameed, Tahaei, Mosleh & Partovi Nia "Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition" (AAAI 2022)
**Paper**: [papers/gkpd-kronecker-cnn-compression.pdf]
**Documented**: 2026-02-15

## Description

The **Generalized Kronecker Product Decomposition** extends the classical Kronecker Product Decomposition (KPD) from matrices to multidimensional tensors, enabling efficient compression of convolutional layers in neural networks. Instead of representing a large weight tensor $\mathcal{W}$ directly, GKPD approximates it as a sum of $\hat{R}$ Kronecker products of smaller factor tensors:

$$
\mathcal{W} \approx \sum_{r=1}^{\hat{R}} \mathcal{A}_r \otimes \mathcal{B}_r
$$

The critical algorithmic insight is **Lemma 1 (the KroneckerConvolution identity)**: when $\mathcal{W} = \mathcal{A} \otimes \mathcal{B}$, the convolution $\mathcal{W} \star \mathcal{X}$ can be computed as **two sequential smaller convolutions** — first with $\mathcal{B}$, then with $\mathcal{A}$ — without ever materializing the full tensor $\mathcal{W}$. This means a single large conv layer is replaced by $\hat{R}$ pairs of smaller convolutions, reducing both memory and FLOPs.

GKPD subsumes several existing structured matrix approaches: a single Kronecker product ($\hat{R} = 1$) corresponds to separable convolutions, while full Kronecker rank $R = \min(\prod a_i, \prod b_i)$ recovers the original uncompressed tensor. The Kronecker rank $\hat{R}$ provides a smooth tradeoff between compression and expressivity.

A key theoretical contribution is proving that **Kronecker networks are universal approximators**: any shallow network with weight tensors represented as low Kronecker rank summations retains universal approximation capability, with the approximation error bounded by the tail singular values of a reshaped version of the weight tensor.

## Mathematical Form

**Multidimensional Kronecker Product (Definition):**

For tensors $\mathcal{A} \in \mathbb{R}^{a_1 \times a_2 \times \cdots \times a_N}$ and $\mathcal{B} \in \mathbb{R}^{b_1 \times b_2 \times \cdots \times b_N}$, the tensor Kronecker product $(\mathcal{A} \otimes \mathcal{B}) \in \mathbb{R}^{a_1 b_1 \times a_2 b_2 \times \cdots \times a_N b_N}$ is defined element-wise as:

$$
(\mathcal{A} \otimes \mathcal{B})_{i_1, i_2, \ldots, i_N} \triangleq \mathcal{A}_{j_1, j_2, \ldots, j_N} \, \mathcal{B}_{k_1, k_2, \ldots, k_N}
$$

where $j_n = \lfloor i_n / b_n \rfloor$ and $k_n = i_n \bmod b_n$.

**GKPD Approximation:**

Any tensor $\mathcal{W} \in \mathbb{R}^{w_1 \times w_2 \times \cdots \times w_N}$ can be decomposed as:

$$
\mathcal{W} = \sum_{r=1}^{R} \mathcal{A}_r \otimes \mathcal{B}_r, \quad R = \min(a_1 a_2 \cdots a_N, \, b_1 b_2 \cdots b_N)
$$

where $R$ is the **Kronecker rank**. For compression, we use $\hat{R} < R$ terms:

$$
\mathcal{W} \approx \sum_{r=1}^{\hat{R}} \mathcal{A}_r \otimes \mathcal{B}_r
$$

**Nearest Kronecker Product Problem:**

The optimal factors minimize:

$$
\min_{\{\mathcal{A}_r\}, \{\mathcal{B}_r\}} \left\| \mathcal{W} - \sum_{r=1}^{\hat{R}} \mathcal{A}_r \otimes \mathcal{B}_r \right\|_F^2
$$

This is solved by defining rearrangement operators $R_w$, $r_a$, $r_b$ that reshape the problem into a standard low-rank matrix approximation:

$$
\min_{\{\mathcal{A}_r\}, \{\mathcal{B}_r\}} \left\| R_w(\mathcal{W}) - \sum_{r=1}^{\hat{R}} r_a(\mathcal{A}_r) \, r_b(\mathcal{B}_r)^\top \right\|_F^2
$$

where $R_w(\mathcal{W}) \in \mathbb{R}^{N_p \times d_1 d_2 \cdots d_N}$ extracts non-overlapping patches, $r_a(\mathcal{A}) = \text{vec}(\text{unfold}(\mathcal{A}, \mathcal{I}_\mathcal{A}))$, and $r_b(\mathcal{B}) = \text{vec}(\mathcal{B})$. This has a closed-form solution via **truncated SVD** of $R_w(\mathcal{W})$.

**KroneckerConvolution Identity (Lemma 1):**

If $\mathcal{W} = \mathcal{A} \otimes \mathcal{B}$, then the multilinear map $\mathcal{W}_{i_1, i_2, \ldots, i_N} \, \mathcal{X}_{i_1+o_1, i_2+o_2, \ldots, i_N+o_N}$ can be written directly in terms of the factors:

$$
\mathcal{W}_{i_1, \ldots, i_N} \, \mathcal{X}_{i_1+o_1, \ldots, i_N+o_N} = \mathcal{A}_{j_1, \ldots, j_N} \, \mathcal{B}_{k_1, \ldots, k_N} \, \mathcal{X}_{g(j_1, k_1)+o_1, \ldots, g(j_N, k_N)+o_N}
$$

where $g(j_n, k_n) \triangleq j_n b_n + k_n$ is the re-indexing function.

**For a 2D convolution layer** with weight $\mathcal{W} \in \mathbb{R}^{F \times C \times K_h \times K_w}$ decomposed into factors $\mathcal{A} \in \mathbb{R}^{F_1 \times C_1 \times K_{h1} \times K_{w1}}$ and $\mathcal{B} \in \mathbb{R}^{F_2 \times C_2 \times K_{h2} \times K_{w2}}$, the convolution becomes:

$$
\mathcal{Y}_{f,x,y} = \sum_{\substack{i_1, j_1, c_1 \\ i_2, j_2, c_2}} \mathcal{A}_{f_1, c_1, i_1, j_1} \, \mathcal{B}_{f_2, c_2, i_2, j_2} \, \mathcal{X}_{g(c_1, c_2), \, g(i_1, i_2)+x, \, g(j_1, j_2)+y}
$$

This separates into two stages (Algorithm 1):
1. **3D convolution** with $\mathcal{B}$ (stride $(C_2, 1, 1)$) to collapse input channels
2. **Batched 2D convolution** with $\mathcal{A}$ (original stride and dilation $= \text{Shape}(\mathcal{B})$)

**Key Definitions:**

- $\mathcal{W} \in \mathbb{R}^{w_1 \times \cdots \times w_N}$ — original weight tensor, $w_n = a_n b_n$
- $\mathcal{A}_r \in \mathbb{R}^{a_1 \times \cdots \times a_N}$ — large Kronecker factor
- $\mathcal{B}_r \in \mathbb{R}^{b_1 \times \cdots \times b_N}$ — small Kronecker factor
- $\hat{R}$ — number of Kronecker summands (Kronecker rank of approximation)
- $R = \min(\prod a_n, \prod b_n)$ — full Kronecker rank
- $g(j, k) = jb + k$ — re-indexing function for Kronecker structure

**Universal Approximation (Theorem 1):**

Any shallow Kronecker network $\hat{f}_{\mathbf{W}_{\hat{R}}}$ with $n$ hidden units, Kronecker rank $\hat{R}$, and $L$-Lipschitz activation on compacta $K \subset \mathbb{R}^d$ satisfies:

$$
\sum_{r=\hat{R}+1}^{R} \sigma_r^2 < \epsilon (L \|K\|^2 \|w_2\|^2)^{-1}
$$

where $\sigma_r$ is the $r$-th singular value of the reshaped weight matrix $R_w(\mathbf{W})$. Thus if the $R - \hat{R}$ smallest singular values are small enough, the Kronecker network retains approximation capability.

## Complexity

**Memory Reduction:**

For a 2D conv with $\mathcal{W} \in \mathbb{R}^{F \times C \times K_h \times K_w}$ decomposed as $\hat{R}$ Kronecker products with $\mathcal{A} \in \mathbb{R}^{F_1 \times C_1 \times K_{h1} \times K_{w1}}$ and $\mathcal{B} \in \mathbb{R}^{F_2 \times C_2 \times K_{h2} \times K_{w2}}$:

$$
\text{Memory ratio} = \frac{F_1 C_1 K_{h1} K_{w1} \cdot F_2 C_2 K_{h2} K_{w2}}{\hat{R}(F_1 C_1 K_{h1} K_{w1} + F_2 C_2 K_{h2} K_{w2})}
$$

Memory reduction is **unconditional** — it always reduces parameters regardless of factor sizes.

**FLOPs Reduction:**

For separable $3 \times 3$ filters ($\hat{R} = 1$, $K_{h1} = K_{w1} = K_{h2} = K_{w2} = 1$ for spatial, adjusted for channel):

$$
\text{FLOPs ratio} = \frac{3 F_1 C_2}{F_1 + C_2}
$$

FLOPs reduction requires $F_1$ and $C_2$ to be sufficiently large.

| Operation | Standard Conv | KroneckerConv ($\hat{R}$ terms) |
|-----------|-------|------------|
| Parameters | $F \cdot C \cdot K_h \cdot K_w$ | $\hat{R}(F_1 C_1 K_{h1} K_{w1} + F_2 C_2 K_{h2} K_{w2})$ |
| FLOPs | $F \cdot C \cdot K_h \cdot K_w \cdot H \cdot W$ | $\hat{R}(F_2 C_2 K_{h2} C_1 H W + F_1 C_1 K_{h1} K_{w1} F_2 H W)$ |

**Practical Results (CIFAR-10):**

| Model | Params | Compression | Accuracy |
|-------|--------|-------------|----------|
| ResNet18 (baseline) | 11.17M | $1\times$ | 95.05% |
| KroneckerResNet18 | 2.2M | $5\times$ | 94.97% |
| SEResNet50 (baseline) | 21.40M | $1\times$ | 95.15% |
| KroneckerSEResNet50 | 2.30M | $9.3\times$ | 94.45% |

**Extreme compression:**
| KroneckerResNet18 | 0.48M | $23.27\times$ | 92.62% |
| KroneckerSEResNet50 | 0.29M | $73.79\times$ | 91.85% |

**ImageNet:**
| ResNet50 (baseline) | 25.6M | $1\times$ | 75.99% |
| KroneckerResNet50 | 12.0M | $2.13\times$ | 73.95% |

## Applicability

- **CNN compression for edge deployment**: Drop-in replacement for any convolutional layer; achieves $5\times$-$74\times$ compression with small accuracy drops on CIFAR-10/ImageNet
- **Fully-connected layer compression**: The matrix case ($N = 2$) directly compresses FC layers via $W \approx \sum_{r} A_r \otimes B_r$, which is equivalent to Monarch-style factorization with different factor structure
- **Transformer FFN layers**: The two large linear projections in feed-forward blocks can be compressed via the matrix GKPD
- **Transfer learning / fine-tuning**: GKPD factors can be initialized from a pretrained model's SVD, providing a strong starting point. Also works with random initialization + training from scratch
- **Model distillation alternative**: Outperforms knowledge distillation methods (Mirzadeh et al. 2020, Heo et al. 2019) at comparable compression rates on ResNet26 by 2-3.7% accuracy
- **Combining with other methods**: GKPD is orthogonal to quantization and can be combined with it for additional compression

## Limitations

- **Non-unique decomposition**: For a given compression target, different configurations of Kronecker factor sizes $\{(a_n, b_n)\}$ lead to different memory/FLOPs tradeoffs. A search over configurations is needed to find the best decomposition for each layer
- **FLOPs reduction is conditional**: Unlike memory reduction (which is unconditional), FLOPs reduction requires the factor dimensions to be sufficiently large. For very small layers, the two-stage convolution may not save FLOPs
- **Higher Kronecker rank $\hat{R}$ increases FLOPs**: Each additional Kronecker term adds a pair of convolution operations. The memory-accuracy tradeoff improves with larger $\hat{R}$, but FLOPs scale linearly with $\hat{R}$
- **SVD-based initialization requires the original model**: The optimal initialization via the Nearest Kronecker Product Problem requires access to the pretrained weight tensor. For training from scratch, random initialization works but may converge more slowly
- **Restricted to factorizable dimensions**: The tensor dimensions $w_n$ must factorize as $a_n \cdot b_n$, which may require padding if original dimensions don't factor cleanly
- **No hardware-specific optimizations**: The two-stage convolution (3D conv + batched 2D conv) may not map optimally to specialized hardware (e.g., Tensor Cores) compared to a single fused convolution

## Implementation Notes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KroneckerConv2d(nn.Module):
    """Conv2d layer compressed via GKPD.

    Replaces a single large convolution with R_hat pairs of
    (3D conv with B) + (batched 2D conv with A), implementing
    Algorithm 1 from the paper.

    Args:
        in_channels: C = C1 * C2
        out_channels: F = F1 * F2
        kernel_size: (Kh, Kw) = (Kh1*Kh2, Kw1*Kw2)
        factor_sizes: dict with keys F1, F2, C1, C2, Kh1, Kh2, Kw1, Kw2
        R_hat: number of Kronecker summands
        stride, padding: as in standard Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 factor_sizes, R_hat=1, stride=1, padding=0):
        super().__init__()
        self.R_hat = R_hat
        fs = factor_sizes
        self.F1, self.F2 = fs['F1'], fs['F2']
        self.C1, self.C2 = fs['C1'], fs['C2']
        self.Kh1, self.Kh2 = fs['Kh1'], fs['Kh2']
        self.Kw1, self.Kw2 = fs['Kw1'], fs['Kw2']
        self.stride = stride
        self.padding = padding

        # Factor tensors for each Kronecker term
        # A_r: (F1, C1, Kh1, Kw1) and B_r: (F2, C2, Kh2, Kw2)
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(self.F1, self.C1, self.Kh1, self.Kw1)
                         / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5)
            for _ in range(R_hat)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.randn(self.F2, self.C2, self.Kh2, self.Kw2)
                         / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5)
            for _ in range(R_hat)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # x: (batch, C, H, W) where C = C1 * C2
        batch, C, H, W = x.shape
        out = None

        for r in range(self.R_hat):
            # Stage 1: 3D conv with B_r
            # Unsqueeze input: (batch, 1, C, H, W)
            x_5d = x.unsqueeze(1)
            # B_r as 3D kernel: (F2, 1, C2, Kh2, Kw2) with stride (C2, 1, 1)
            B_3d = self.B[r].unsqueeze(1)
            y1 = F.conv3d(x_5d, B_3d, stride=(self.C2, 1, 1))
            # y1: (batch, F2, C1, H', W')

            # Stage 2: Batched 2D conv with A_r
            # Reshape for batched 2D conv along the F2 dimension
            _, F2, C1, H1, W1 = y1.shape
            # Merge batch and F2: (batch * F2, C1, H', W')
            y1_flat = y1.reshape(batch * F2, C1, H1, W1)
            # A_r: (F1, C1, Kh1, Kw1) - standard 2D conv
            y2 = F.conv2d(y1_flat, self.A[r],
                          stride=self.stride, padding=self.padding,
                          dilation=(self.Kh2, self.Kw2))
            # y2: (batch * F2, F1, H'', W'')
            _, F1, H2, W2 = y2.shape

            # Reshape: (batch, F2, F1, H'', W'') -> (batch, F1*F2, H'', W'')
            y2 = y2.reshape(batch, F2, F1, H2, W2)
            y2 = y2.permute(0, 2, 1, 3, 4).reshape(batch, F1 * F2, H2, W2)

            if out is None:
                out = y2
            else:
                out = out + y2

        return out + self.bias.view(1, -1, 1, 1)


def initialize_from_pretrained(kronecker_conv, weight_tensor):
    """Initialize GKPD factors from a pretrained weight tensor via SVD.

    Solves the Nearest Kronecker Product Problem by reshaping
    the weight tensor and computing truncated SVD.

    Args:
        kronecker_conv: KroneckerConv2d module
        weight_tensor: (F, C, Kh, Kw) pretrained weight
    """
    fs = kronecker_conv
    # Rearrange W into matrix form for SVD
    # R_w extracts non-overlapping patches and reshapes
    W = weight_tensor.detach().cpu()
    F_out, C_in, Kh, Kw = W.shape

    # Reshape into (N_patches, patch_size) for SVD
    W_reshaped = W.reshape(fs.F1, fs.F2, fs.C1, fs.C2,
                           fs.Kh1, fs.Kh2, fs.Kw1, fs.Kw2)
    # Rearrange to (F1*C1*Kh1*Kw1, F2*C2*Kh2*Kw2)
    W_mat = W_reshaped.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        fs.F1 * fs.C1 * fs.Kh1 * fs.Kw1,
        fs.F2 * fs.C2 * fs.Kh2 * fs.Kw2
    )

    # Truncated SVD
    U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)

    for r in range(fs.R_hat):
        # A_r from left singular vectors, B_r from right
        a_vec = U[:, r] * (S[r] ** 0.5)
        b_vec = Vh[r, :] * (S[r] ** 0.5)
        fs.A[r].data.copy_(a_vec.reshape(fs.F1, fs.C1, fs.Kh1, fs.Kw1))
        fs.B[r].data.copy_(b_vec.reshape(fs.F2, fs.C2, fs.Kh2, fs.Kw2))
```

## References

- Hameed, M.G.A., Tahaei, M.S., Mosleh, A. & Partovi Nia, V. "Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition" AAAI 2022. arXiv:2109.14710
- Van Loan, C.F. & Pitsianis, N. "Approximation with Kronecker Products" in Linear Algebra for Large Scale and Real-Time Applications, 1992
- Van Loan, C.F. "The ubiquitous Kronecker product" J. Computational and Applied Mathematics 123, 85-100, 2000
- Thakker, U., Beu, J., Gope, D., Zhou, C. & Fedorov, I. "Compressing RNNs for IoT devices by 15-38x using Kronecker Products" arXiv:1906.02876, 2019
- Dao, T., Chen, B., Sohoni, N.S., Desai, A., Poli, M., Grogan, J. & Ré, C. "Monarch: Expressive Structured Matrices for Efficient and Accurate Training" ICML 2022 (related structured matrix approach)
- Hornik, K. "Approximation capabilities of multilayer feedforward networks" Neural Networks 4(2), 251-257, 1991
- Zhou, L. "Universality of deep convolutional neural networks" Applied and Computational Harmonic Analysis 48(2), 787-794, 2020

# 004: BCCB Circulant Attention

**Category**: approximation
**Gain type**: efficiency
**Source**: Han et al. "Vision Transformers are Circulant Attention Learners" (AAAI 2026)
**Paper**: [papers/bccb-circulant-attention.pdf]
**Documented**: 2026-02-15

## Description

Self-attention in vision Transformers naturally learns attention maps that closely approximate **Block Circulant matrices with Circulant Blocks (BCCB)** — the 2D generalization of circulant matrices. This empirical observation motivates explicitly projecting the attention matrix $A = \sigma(QK^\top / \sqrt{d})$ onto the nearest BCCB matrix $\tilde{A}$, enabling $O(N \log N)$ computation via 2D FFT instead of the $O(N^2)$ dense matrix multiply.

The key insight is that a BCCB matrix-vector product is equivalent to a **2D circular cross-correlation** (i.e., a 2D global depthwise convolution with circular padding), which can be computed efficiently via the 2D DFT convolution theorem. Since the BCCB matrix is fully determined by its first row (an $H \times W$ spatial pattern), the attention mechanism reduces to computing a single "attention kernel" per head and applying it as a global convolution.

Unlike handcrafted efficient attention methods (Swin's window attention, PVT's spatial reduction) that restrict the receptive field, circulant attention preserves **global** interaction at sub-quadratic cost. The BCCB structure also enforces approximate **translation invariance** — a natural inductive bias for vision tasks where the same attention pattern should apply regardless of spatial position.

## Mathematical Form

**Standard Self-Attention:**

$$
A = QK^\top / \sqrt{d}, \quad O = \sigma(A)V
$$

where $Q, K, V \in \mathbb{R}^{N \times d}$, $N = H \times W$ is the number of spatial tokens, and $\sigma$ is Softmax.

**BCCB Matrix:**

A BCCB matrix $B \in \mathbb{R}^{N \times N}$ with $N = H \times W$ has a block circulant structure with $H \times H$ blocks, where each block $C_i$ is a $W \times W$ circulant matrix:

$$
B = \begin{pmatrix} C_0 & C_1 & \cdots & C_{H-1} \\ C_{H-1} & C_0 & \cdots & C_{H-2} \\ \vdots & \vdots & \ddots & \vdots \\ C_1 & C_2 & \cdots & C_0 \end{pmatrix}
$$

The entire BCCB matrix is fully determined by its first row $b = [c_0, c_1, \ldots, c_{HW-1}]$.

**2D DFT Diagonalization:**

BCCB multiplication equals 2D circular cross-correlation. Let $\hat{b}, \hat{x} \in \mathbb{R}^{H \times W}$ be 2D reshaped versions of $b, x$:

$$
Bx = \mathcal{F}_{2\text{D}}^{-1}\left(\overline{\mathcal{F}_{2\text{D}}(\hat{b})} \odot \mathcal{F}_{2\text{D}}(\hat{x})\right) \triangleq b \circledast x
$$

where $\mathcal{F}_{2\text{D}}$ is the 2D DFT, $\overline{(\cdot)}$ is complex conjugate, $\odot$ is Hadamard product, and $\circledast$ is the defined DFT-based multiplication.

**Orthogonal Projection onto BCCB Subspace:**

Given attention matrix $A \in \mathbb{R}^{N \times N}$, its nearest BCCB approximation is:

$$
\tilde{A} = \arg\min_{B \in \mathcal{B}} \|A - B\|_F
$$

where $\mathcal{B}$ is the BCCB subspace. Using the orthonormal basis $\{B_0, B_1, \ldots, B_{N-1}\}$ (where $B_k$ is the BCCB matrix with a one-hot first row at position $k$):

$$
\tilde{A} = \sum_{k=0}^{N-1} \frac{\langle A, B_k \rangle}{\langle B_k, B_k \rangle} B_k = \frac{1}{N} \sum_{k=0}^{N-1} \langle A, B_k \rangle B_k
$$

since $\langle B_k, B_k \rangle = N$ and $\langle B_k, B_j \rangle = 0$ for $k \neq j$.

**Efficient First-Row Computation:**

The first row $a$ of $\tilde{A}$ can be computed efficiently:

$$
a_k = \frac{1}{N\sqrt{d}} \langle QK^\top, B_k \rangle
$$

Each $B_k$ corresponds to a spatial shift $(\Delta h, \Delta w)$ where $\Delta h = \lfloor k/W \rfloor$, $\Delta w = k \bmod W$. This makes $a_k$ the circular cross-correlation of $\hat{Q}$ and $\hat{K}$:

$$
a = \frac{1}{N\sqrt{d}} \left[\mathcal{F}_{2\text{D}}^{-1}\left(\overline{\mathcal{F}_{2\text{D}}(Q)} \odot \mathcal{F}_{2\text{D}}(K)\right)\right] \cdot \mathbf{1}_{d \times 1}
$$

$$
= \frac{1}{N\sqrt{d}} (Q \circledast K) \cdot \mathbf{1}_{d \times 1}
$$

where $\mathbf{1}_{d \times 1}$ sums over the head dimension $d$.

**Full Circulant Attention:**

$$
O = \sigma(\tilde{A})V = \sigma(a) \circledast V
$$

Since $\tilde{A}$ is BCCB and $\sigma$ is applied row-wise (Softmax on each row), $\sigma(\tilde{A})$ is also BCCB, fully determined by $\sigma(a)$. The output is then computed via DFT-based multiplication.

**Token Reweighting:**

The BCCB structure constrains both row and column sums to be equal, limiting the ability to emphasize salient tokens. A post-reweighting module restores this:

$$
O = \text{CirAttn}(Q, K, V) \odot T, \quad T = \text{SiLU}(xW_T) \in \mathbb{R}^{N \times d}
$$

where $T$ is an input-dependent reweighting factor.

**Key Definitions:**

- $N = H \times W$ — number of spatial tokens (image patches)
- $\mathcal{F}_{2\text{D}}, \mathcal{F}_{2\text{D}}^{-1}$ — 2D discrete Fourier transform and inverse
- $\circledast$ — DFT-based matrix multiplication (2D circular cross-correlation)
- $\mathcal{B}$ — the BCCB matrix subspace
- $a \in \mathbb{R}^N$ — first row of projected BCCB attention matrix
- $d$ — head dimension (optimal at $d=1$ with more heads)

## Complexity

| Operation | Self-Attention | Circulant Attention |
|-----------|---------------|---------------------|
| Attention computation | $O(N^2 d)$ | $O(N \log_2 N \cdot d)$ |
| Output computation | $O(N^2 d)$ | $O(N \log_2 N \cdot d)$ |
| Total | $O(N^2 d)$ = $2N^2d$ | $O(N \log_2 N \cdot d)$ = $N(\log_2 N)(4d+2) + 4Nd$ |

**FLOPs at $224^2$ resolution:** CA-DeiT-T uses 1.2G FLOPs vs DeiT-T's 1.2G (similar at standard resolution). At $1536^2$ resolution: CA-DeiT-T requires **8x fewer FLOPs** than DeiT-T.

**Memory:** $O(N)$ for the BCCB first row vs $O(N^2)$ for the full attention matrix.

**Throughput:** Up to 7x faster inference FPS at high resolution ($1536^2$) on RTX 3090.

## Applicability

- **Vision Transformers (global attention)**: Drop-in replacement for self-attention in DeiT — CA-DeiT-T achieves 75.0% accuracy (+2.8 over DeiT-T) with same FLOPs
- **Vision Transformers (sparse attention)**: Applied to PVT — CA-PVT-S achieves 81.7% (+1.9) with comparable FLOPs
- **Vision Transformers (local attention)**: Applied to Swin Transformer — CA-Swin-T achieves 82.2% (+0.9) with same FLOPs
- **Object detection**: On COCO, CA-PVT-L achieves 45.0 box AP vs PVT-L's 42.5, with fewer FLOPs
- **Semantic segmentation**: Up to 3.7 mIoU improvement on ADE20K
- **High-resolution vision**: Particularly valuable because the $O(N \log N)$ scaling enables processing high-resolution feature maps in early stages without prohibitive cost
- **Any 2D spatial attention**: The BCCB structure naturally encodes 2D translation-invariant interactions

## Limitations

- **2D spatial structure required**: Only applies to grid-structured tokens (images, feature maps); not directly applicable to 1D sequences (use regular circulant/Toeplitz instead)
- **Translation invariance constraint**: BCCB enforces that attention patterns are shift-invariant — all queries see the same relative attention pattern (mitigated by token reweighting)
- **Column sum constraint**: Unlike standard Softmax attention where different queries can emphasize different keys, BCCB enforces equal column sums — requires the post-reweighting module $T$ to recover expressivity
- **Head dimension should be small**: The circulant attention score is a summation of $N$ query-key pairs, giving an "equivalent head dimension" of $Nd$. Best results at $d=1$ with more heads
- **Accuracy gap on NLP tasks**: Not evaluated on language — the BCCB structure is specific to 2D spatial attention patterns; 1D sequences likely don't exhibit BCCB structure
- **Softmax on BCCB**: For $\sigma(\tilde{A})$ to remain BCCB, Softmax must be applied to the first row, which is different from standard row-wise Softmax on the full attention matrix

## Implementation Notes

```python
import torch
import torch.fft as fft

class CirculantAttention(torch.nn.Module):
    """BCCB Circulant Attention for Vision Transformers.

    Replaces O(N^2) self-attention with O(N log N) DFT-based
    circulant attention by projecting onto BCCB subspace.
    """

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 1  # optimal: d=1 with more heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)

        # Token reweighting (post-reweighting)
        self.reweight = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.SiLU()
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)

        # Reshape to 2D spatial: (B, heads, H, W, head_dim)
        q_2d = q.reshape(B, self.num_heads, H, W, -1)
        k_2d = k.reshape(B, self.num_heads, H, W, -1)
        v_2d = v.reshape(B, self.num_heads, H, W, -1)

        # Step 1: Compute BCCB first row via 2D circular cross-correlation
        # a = (1/N√d) * F_2D^{-1}(conj(F_2D(Q)) ⊙ F_2D(K)) · 1_{d×1}
        q_fft = fft.fft2(q_2d, dim=(2, 3))   # (B, heads, H, W, d)
        k_fft = fft.fft2(k_2d, dim=(2, 3))

        # Cross-correlation in frequency domain, sum over head_dim
        a_fft = (q_fft.conj() * k_fft).sum(dim=-1)  # (B, heads, H, W)
        a = fft.ifft2(a_fft, dim=(2, 3)).real  # (B, heads, H, W)
        a = a * self.scale / N

        # Step 2: Apply Softmax to first row (BCCB attention weights)
        a_flat = a.reshape(B, self.num_heads, N)
        a_soft = torch.softmax(a_flat, dim=-1)
        a_soft = a_soft.reshape(B, self.num_heads, H, W)

        # Step 3: Output = σ(a) ⊛ V via 2D DFT
        a_soft_fft = fft.fft2(a_soft, dim=(2, 3)).unsqueeze(-1)
        v_fft = fft.fft2(v_2d, dim=(2, 3))
        out = fft.ifft2(a_soft_fft.conj() * v_fft, dim=(2, 3)).real

        # Reshape back: (B, N, C)
        out = out.reshape(B, self.num_heads, N, -1)
        out = out.transpose(1, 2).reshape(B, N, C)

        # Token reweighting
        T = self.reweight(x)  # (B, N, C)
        out = out * T

        return self.proj(out)
```

## References

- Han, D., Li, T., Wang, Z., Huang, G. "Vision Transformers are Circulant Attention Learners" (AAAI 2026). arXiv:2512.21542
- Davis, P.J. "Circulant Matrices" John Wiley & Sons, 1979
- Rao, Y., Zhao, W., Zhu, Z., Lu, J., Zhou, J. "Global Filter Networks for Image Classification" (NeurIPS 2021)
- Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B. "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers" arXiv:2111.13587, 2021
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H. "Training Data-Efficient Image Transformers & Distillation through Attention" (ICML 2021)

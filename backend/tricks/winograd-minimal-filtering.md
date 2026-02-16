# Winograd Minimal Filtering

**Category**: algebraic
**Gain type**: efficiency
**Source**: Winograd (1980), applied to CNNs by Lavin & Gray (2015)
**Paper**: [papers/winograd-minimal-filtering.pdf]
**Documented**: 2026-02-14

## Description

Trade multiplications for additions in convolution by computing in a transformed domain. Winograd's minimal filtering algorithm $F(m, r)$ computes $m$ outputs with an $r$-tap FIR filter using only $m + r - 1$ multiplications instead of $m \cdot r$. When nested for 2D convolutions in CNNs, this yields up to a $4\times$ reduction in arithmetic complexity. The key GPU insight: the element-wise multiplications in the transform domain can be batched across all tiles and channels into large matrix multiplications (BLAS Level-3), giving massive parallelism even at batch size 1.

## Mathematical Form

**Core Operation:**

For a 1D minimal filter $F(m, r)$ producing $m$ outputs from an $r$-tap filter:

$$
Y = A^T \left[ (Gg) \odot (B^T d) \right]
$$

where $\odot$ denotes element-wise multiplication, $g$ is the filter, $d$ is the data tile, and $B^T$, $G$, $A^T$ are fixed transform matrices.

**Key Definitions:**

- $g \in \mathbb{R}^r$ — filter (kernel) weights
- $d \in \mathbb{R}^{m+r-1}$ — input data tile
- $B^T \in \mathbb{R}^{(m+r-1) \times (m+r-1)}$ — data transform
- $G \in \mathbb{R}^{(m+r-1) \times r}$ — filter transform
- $A^T \in \mathbb{R}^{m \times (m+r-1)}$ — inverse (output) transform

**2D Nesting for CNNs:**

For $F(m \times m, r \times r)$:

$$
Y = A^T \left[ (GgG^T) \odot (B^T dB) \right] A
$$

**Conversion to GEMM (the GPU trick):**

By labeling each element of the element-wise product by index $(\xi, \nu)$, the channel summation becomes a matrix multiply:

$$
M^{(\xi, \nu)} = U^{(\xi, \nu)} V^{(\xi, \nu)}
$$

where $U^{(\xi, \nu)}_{k,c} = (GgG^T)^{(\xi,\nu)}_{k,c}$ collects filter $k$, channel $c$, and $V^{(\xi, \nu)}_{c,b} = (B^T dB)^{(\xi,\nu)}_{c,b}$ collects channel $c$, tile $b$. This yields $(m+r-1)^2$ independent GEMMs of size $K \times C \times P$ where $P$ is the number of tiles.

**Example $F(2 \times 2, 3 \times 3)$:**

$$
B^T = \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 1 & 0 & -1 \end{bmatrix}, \quad
G = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \\ 0 & 0 & 1 \end{bmatrix}, \quad
A^T = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 0 & 1 & -1 & -1 \end{bmatrix}
$$

Standard algorithm: $2 \times 2 \times 3 \times 3 = 36$ multiplications. Winograd: $4 \times 4 = 16$ multiplications. Reduction factor: $\frac{36}{16} = 2.25\times$.

## Complexity

| Operation | Direct Convolution | Winograd $F(m,r)$ |
|-----------|-------------------|-------------------|
| Multiplications (1D) | $O(mr)$ | $O(m + r - 1)$ |
| Multiplications (2D) | $O(m^2 r^2)$ | $O((m + r - 1)^2)$ |
| Total arithmetic | $O(NHWCK \cdot R^2)$ | $O(NHWCK \cdot \alpha')$ where $\alpha' = \frac{(m+r-1)^2}{m^2}$ |

**Maximum speedup:** $R^2 / \alpha'$ for filter size $R \times R$. For $3 \times 3$ filters: up to $4\times$ with $F(4 \times 4, 3 \times 3)$.

**Memory:** $O(16KC)$ workspace in transform domain (just 16MB for $K = C = 512$, fp32).

## Applicability

- CNN layers with small filters ($3 \times 3$, $5 \times 5$) — the dominant pattern in modern architectures (VGG, ResNet, etc.)
- Especially beneficial at small batch sizes where standard GEMM-based methods underperform
- Forward propagation, backward data gradient, and backward weight gradient
- Integrated in cuDNN as a selectable convolution algorithm

## Limitations

- Transform overhead grows quadratically with tile size — impractical for tiles larger than $6 \times 6$
- Numerical precision degrades with larger tiles (transform matrix magnitudes grow)
- Only applicable to small filter sizes ($3 \times 3$ to $5 \times 5$ practically)
- Not beneficial for $1 \times 1$ convolutions (which are already just matrix multiplies)
- The addition of many constant multiplications for the transforms partially offsets the multiply savings on GPUs where MUL and ADD cost the same

## Implementation Notes

```python
# Winograd F(2x2, 3x3) convolution — GPU pseudocode
# Key insight: batch the element-wise products across tiles/channels into GEMMs

def winograd_conv_F2x2_3x3(input, filters, N, C, K, H, W):
    """
    input:   [N, C, H, W]
    filters: [K, C, 3, 3]
    """
    P = N * ceil(H/2) * ceil(W/2)  # Total number of 4x4 tiles
    alpha = 4  # m + r - 1 = 2 + 3 - 1

    # Transform filters: U[xi,nu] has shape [K, C]
    # Done once, can be precomputed
    U = filter_transform(filters, G)  # -> [alpha, alpha, K, C]

    # Transform input tiles: V[xi,nu] has shape [C, P]
    V = data_transform(input, B_T)    # -> [alpha, alpha, C, P]

    # 16 batched GEMMs (one per transform-domain element)
    M = torch.zeros(alpha, alpha, K, P)
    for xi in range(alpha):
        for nu in range(alpha):
            M[xi, nu] = U[xi, nu] @ V[xi, nu]  # [K,C] x [C,P] -> [K,P]

    # Inverse transform to get output tiles
    output = inverse_transform(M, A_T)  # -> [N, K, H, W]
    return output
```

## References

- Winograd, S. (1980). Arithmetic complexity of computations.
- Lavin, A. & Gray, S. (2015). Fast Algorithms for Convolutional Neural Networks. arXiv:1509.09308.
- NVIDIA cuDNN documentation — Winograd convolution support.

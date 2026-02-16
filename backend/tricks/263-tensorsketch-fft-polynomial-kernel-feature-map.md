# 263: TensorSketch — FFT-Based Polynomial Kernel Feature Map

**Category**: approximation
**Gain type**: efficiency
**Source**: Pham & Pagh (2013) — "Fast and Scalable Estimation of Uncertainty and Fidelity in Time Series" (KDD 2013); extended version: arXiv:2505.08146
**Paper**: [papers/tensorsketch-polynomial-kernel.pdf]
**Documented**: 2026-02-16

## Description

TensorSketch is a randomized dimensionality reduction technique that computes a compact $D$-dimensional feature vector approximating the polynomial kernel $\kappa(\mathbf{x}, \mathbf{y}) = \langle \mathbf{x}, \mathbf{y} \rangle^p$ **without ever materializing the $d^p$-dimensional tensor product** $\mathbf{x}^{(p)} = \mathbf{x} \otimes \cdots \otimes \mathbf{x}$. The core insight: the Count Sketch of a tensor product equals the **convolution of Count Sketches of the factors**, and convolution is computed efficiently via FFT.

For a degree-$p$ polynomial kernel with input dimension $d$:
- **Exact feature map**: $\mathbf{x}^{(p)} \in \mathbb{R}^{d^p}$ — exponentially large, infeasible for $p > 2$ at typical head dimensions ($d = 64$–$128$).
- **TensorSketch**: $f(\mathbf{x}) \in \mathbb{R}^D$ where $D = O(d)$ — computed in $O(d + D \log D)$ time per vector, using only $O(1)$ extra space for hash functions.

**Relevance to Higher-Order Linear Attention (trick 222)**: HLA's second-order variant maintains $\mathbf{S}_t^K = \sum_{i \leq t} \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d \times d}$, which is the prefix sum of rank-1 outer products — exactly the degree-2 tensor power feature map. TensorSketch provides a principled way to compress this $d^2$-dimensional summary to $D \ll d^2$ dimensions while preserving inner products (kernel values) in expectation. For HLA's third-order variant ($d^3$ state), TensorSketch becomes essential — it reduces $d^3 = 2$M (for $d = 128$) to $D = O(d)$.

The trick is the foundation used by PolySketchFormer (trick 262) for its approximate polynomial attention.

## Mathematical Form

**Tensor power as explicit feature map:**

The $p$-th tensor power of $\mathbf{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d$ is:

$$
\mathbf{x}^{(p)} = \underbrace{\mathbf{x} \otimes \cdots \otimes \mathbf{x}}_{p \text{ times}} \in \mathbb{R}^{d^p}, \quad \mathbf{x}^{(p)}_{(i_1, \ldots, i_p)} = \prod_{j=1}^p x_{i_j}
$$

The polynomial kernel identity:

$$
\langle \mathbf{x}^{(p)}, \mathbf{y}^{(p)} \rangle = \langle \mathbf{x}, \mathbf{y} \rangle^p
$$

**Count Sketch definition:**

Given hash functions $h : [d] \to [D]$ (2-wise independent) and $s : [d] \to \{-1, +1\}$ (4-wise independent), the Count Sketch of $\mathbf{x}$ is $\mathbf{C}\mathbf{x} \in \mathbb{R}^D$ where:

$$
(\mathbf{C}\mathbf{x})_k = \sum_{i: h(i) = k} s(i) \, x_i
$$

**Properties:** $\mathbb{E}[\langle \mathbf{C}\mathbf{x}, \mathbf{C}\mathbf{y} \rangle] = \langle \mathbf{x}, \mathbf{y} \rangle$ and $\text{Var}[\langle \mathbf{C}\mathbf{x}, \mathbf{C}\mathbf{y} \rangle] \leq \frac{2}{D} \|\mathbf{x}\|_2^2 \|\mathbf{y}\|_2^2$.

**Core TensorSketch construction (degree 2):**

Given two Count Sketches $\mathbf{C}_1\mathbf{x}, \mathbf{C}_2\mathbf{x} \in \mathbb{R}^D$ using independent hash/sign families, the Count Sketch of the outer product $\mathbf{x} \otimes \mathbf{x}$ is computed via:

$$
\mathbf{C}\mathbf{x}^{(2)} = \text{FFT}^{-1}\!\left(\text{FFT}(\mathbf{C}_1\mathbf{x}) \circ \text{FFT}(\mathbf{C}_2\mathbf{x})\right) \in \mathbb{R}^D
$$

where $\circ$ denotes componentwise (Hadamard) product. This works because the composite hash function $H(i_1, i_2) = (h_1(i_1) + h_2(i_2)) \bmod D$ means the sketch entries are coefficients of a polynomial product, which is a convolution — computable via FFT.

**General degree-$p$ TensorSketch (Algorithm 1):**

Given $p$ independent Count Sketch families $(h_1, s_1), \ldots, (h_p, s_p)$:

$$
f(\mathbf{x}) = \text{FFT}^{-1}\!\left(\widehat{\mathbf{C}_1\mathbf{x}} \circ \widehat{\mathbf{C}_2\mathbf{x}} \circ \cdots \circ \widehat{\mathbf{C}_p\mathbf{x}}\right) \in \mathbb{R}^D
$$

where $\widehat{\mathbf{C}_i\mathbf{x}} = \text{FFT}(\mathbf{C}_i\mathbf{x})$.

**Unbiasedness and variance bound (Theorem 9):**

$$
\mathbb{E}\!\left[\langle \mathbf{C}\mathbf{x}^{(p)}, \mathbf{C}\mathbf{y}^{(p)} \rangle\right] = \langle \mathbf{x}, \mathbf{y} \rangle^p
$$

$$
\text{Var}\!\left[\langle \mathbf{C}\mathbf{x}^{(p)}, \mathbf{C}\mathbf{y}^{(p)} \rangle\right] \leq \frac{3^p - 1}{D} \|\mathbf{x}\|_2^{2p} \|\mathbf{y}\|_2^{2p}
$$

For normalized vectors ($\|\mathbf{x}\|_2 = \|\mathbf{y}\|_2 = 1$), the variance is simply $\frac{3^p - 1}{D}$. With $D = O(3^p / \varepsilon^2)$, the standard deviation is $O(\varepsilon)$.

**Inhomogeneous polynomial kernel:** For $\kappa(\mathbf{x}, \mathbf{y}) = (c + \langle \mathbf{x}, \mathbf{y} \rangle)^p$, append $\sqrt{c}$ as an extra coordinate to each vector: $\tilde{\mathbf{x}} = (\mathbf{x}, \sqrt{c}) \in \mathbb{R}^{d+1}$.

## Complexity

| Operation | Exact Feature Map | TensorSketch | Naive Sketch (Kar & Karnick) |
|-----------|------------------|--------------|------------------------------|
| Feature computation | $O(d^p)$ | $O(d + D \log D)$ | $O(dD)$ |
| Feature dimension | $d^p$ | $D$ | $D$ |
| Storage for randomness | N/A | $O(1)$ (hash seeds) | $O(dD)$ ($D$ random vectors) |
| Inner product approx | Exact | Unbiased, $O(3^p/D)$ variance | Unbiased, similar variance |
| Batch of $n$ vectors | $O(n d^p)$ | $O(n(d + D \log D))$ | $O(n d D)$ |

**For attention with $n$ tokens, head dim $h$, degree $p$, sketch size $D$:**

| | Standard Attn | Exact Poly Attn | TensorSketch Poly Attn |
|---|---|---|---|
| Time | $O(n^2 h)$ | $O(n h^{p+1})$ | $O(n(h + D \log D) + n D)$ |
| Memory | $O(n^2)$ | $O(h^p)$ per token | $O(D)$ per token |

**Typical values**: $h = 64, p = 2, D = 128$: TensorSketch computes $128$-dim features in $O(64 + 128 \cdot 7) \approx O(960)$ per token vs $O(64^2) = O(4096)$ for exact outer product. For $p = 4, D = 256$: $O(64 + 256 \cdot 8) \approx O(2112)$ vs $O(64^4) \approx O(17$M$)$ for exact.

## Applicability

- **Compressing higher-order linear attention state**: HLA (trick 222) maintains $\mathbf{S}_t^K \in \mathbb{R}^{d \times d}$ as a running outer-product sum. TensorSketch compresses this to a $D$-dimensional running sum: $\tilde{\mathbf{S}}_t = \sum_{i \leq t} f(\mathbf{k}_i) \in \mathbb{R}^D$, where $f$ is the degree-2 TensorSketch. The output $\mathbf{o}_t \approx f(\mathbf{q}_t)^\top \tilde{\mathbf{S}}_t$ requires only a $D$-dimensional dot product.
- **Polynomial kernel attention** (PolySketchFormer, trick 262): TensorSketch is the core subroutine for computing the approximate feature map $\phi'$.
- **Bilinear pooling in vision/multimodal**: Compact bilinear pooling (Gao et al., 2016) uses TensorSketch to compress the outer product of CNN features from $d^2$ to $D$ dimensions for fine-grained recognition.
- **Streaming/online kernel methods**: Since the sketch is a linear function of the data, it supports additive updates: $f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y})$ for Count Sketch (but NOT for TensorSketch due to the nonlinear tensor product). However, the *sum of sketches* equals the *sketch of sums* at the feature level: $\sum_i f(\mathbf{k}_i)$ is maintained incrementally.

## Limitations

- **Variance grows as $3^p$**: For degree $p$, the variance scales as $(3^p - 1)/D$, meaning sketch size must grow exponentially with $p$ to maintain accuracy. For $p = 4$: need $D = O(80/\varepsilon^2)$; for $p = 8$: $D = O(6560/\varepsilon^2)$.
- **Not a matmul**: The per-vector sketch computation involves hash-based scatter (Count Sketch), FFT, and Hadamard product — none of which map to tensor cores. However, the downstream attention computation $\phi(\mathbf{Q}) \cdot [\phi(\mathbf{K})^\top \mathbf{V}]$ IS a matmul.
- **Hash-based indexing breaks coalescing**: Count Sketch requires scatter-add indexed by hash functions, which is not coalesced memory access. For small $d$ (e.g., $d = 64$), this is fast but not tensor-core-friendly. The FFT portion maps well to GPU FFT libraries (cuFFT).
- **Non-negativity not guaranteed**: Raw TensorSketch can produce negative approximate kernel values. PolySketchFormer addresses this with self-tensoring ($r \to r^2$), but at significant cost.
- **Learned alternatives may dominate**: PolySketchFormer found that replacing random projections with learned MLPs consistently improves quality. At that point, the TensorSketch structure provides initialization/architecture guidance rather than being used directly.

## Implementation Notes

```python
import torch
import torch.fft

class TensorSketch:
    """TensorSketch for degree-p polynomial kernel approximation.

    Computes f(x) in R^D such that E[<f(x), f(y)>] = <x, y>^p
    Time: O(d + D log D) per vector
    Space: O(1) for hash functions (stored as index tensors of size d)
    """
    def __init__(self, d, D, p, device='cuda'):
        self.d = d
        self.D = D
        self.p = p
        # Pre-generate hash functions (small integer tensors)
        # h_i: [d] -> [D], s_i: [d] -> {-1, +1}
        self.h = torch.randint(0, D, (p, d), device=device)  # (p, d)
        self.s = 2 * torch.randint(0, 2, (p, d), device=device).float() - 1  # (p, d)

    def count_sketch(self, x, i):
        """Compute Count Sketch of x using i-th hash/sign family.
        x: (..., d) — batched input
        Returns: (..., D) — Count Sketch
        """
        # Scatter-add: sketch[h[i][j]] += s[i][j] * x[j]
        signed_x = x * self.s[i]  # (..., d)
        sketch = torch.zeros(*x.shape[:-1], self.D,
                            device=x.device, dtype=x.dtype)
        sketch.scatter_add_(-1, self.h[i].expand_as(signed_x), signed_x)
        return sketch

    def __call__(self, x):
        """Compute TensorSketch feature map.
        x: (..., d) — input vectors
        Returns: (..., D) — sketched features

        Algorithm:
        1. Compute p Count Sketches C_1(x), ..., C_p(x)
        2. FFT each: C_hat_i = FFT(C_i(x))
        3. Pointwise multiply: prod = C_hat_1 * ... * C_hat_p
        4. IFFT: f(x) = IFFT(prod)
        """
        # Step 1-2: Count Sketch + FFT for each of p families
        fft_product = torch.ones(*x.shape[:-1], self.D,
                                device=x.device, dtype=torch.complex64)
        for i in range(self.p):
            cs = self.count_sketch(x, i)  # (..., D)
            fft_product = fft_product * torch.fft.fft(cs)  # pointwise mult in freq domain

        # Step 3-4: IFFT
        result = torch.fft.ifft(fft_product).real  # (..., D)
        return result

# Usage for polynomial attention:
def sketched_polynomial_attention(Q, K, V, sketch_dim=128, degree=2):
    """Linear-time polynomial attention via TensorSketch.
    Q, K: (batch, n, d)  V: (batch, n, d_v)
    """
    b, n, d = Q.shape
    ts = TensorSketch(d, sketch_dim, degree, device=Q.device)

    # Compute feature maps: O(n * (d + D log D))
    phi_Q = ts(Q)  # (b, n, D)
    phi_K = ts(K)  # (b, n, D)

    # Linear attention: O(n * D * d_v)
    # KV = sum_j phi(k_j) * v_j^T  — accumulated as (b, D, d_v)
    KV = torch.bmm(phi_K.transpose(1, 2), V)  # (b, D, d_v) — GEMM!

    # Output = phi(q_i)^T @ KV for each i
    output = torch.bmm(phi_Q, KV)  # (b, n, d_v) — GEMM!

    # Normalize
    denom = phi_Q @ phi_K.sum(dim=1, keepdim=True).transpose(1, 2)  # (b, n, 1)
    output = output / (denom + 1e-6)

    return output

# GPU efficiency analysis:
# - Count Sketch: scatter_add with random indices — NOT coalesced, but d is small
# - FFT: maps to cuFFT, well-optimized for D = 128-1024
# - The TWO main GEMMs (phi_K^T @ V and phi_Q @ KV) ARE tensor-core friendly
# - Arithmetic intensity: dominated by the two matmuls, not the sketch
# - For D=128, d_v=64: KV accumulation is (n, 128)^T @ (n, 64) — large GEMM
# - Total HBM traffic: O(n * (d + D + d_v)) — linear in n, no n^2 attention matrix
```

## References

- Pham, N. & Pagh, R. (2013). Fast and Scalable Estimation of Uncertainty and Fidelity in Time Series. KDD 2013 (original TensorSketch).
- Pham, N. & Pagh, R. (2025). Tensor Sketch: Fast and Scalable Polynomial Kernel Approximation. arXiv:2505.08146 (extended JMLR version with corrected variance bounds). [https://arxiv.org/abs/2505.08146](https://arxiv.org/abs/2505.08146)
- Ahle, T. D., et al. (2020). Oblivious sketching of high-degree polynomial kernels. SODA 2020.
- Kacham, P., Mirrokni, V., & Zhong, P. (2024). PolySketchFormer (ICLR 2024) — applies TensorSketch to polynomial attention.
- Charikar, M., Chen, K., & Farach-Colton, M. (2002). Finding frequent items in data streams. (Count Sketch)
- Avron, H., Nguyen, H., & Woodruff, D. P. (2014). Subspace embeddings for the polynomial kernel. NeurIPS 2014.
- Gao, Y., et al. (2016). Compact bilinear pooling. CVPR 2016 (TensorSketch applied to vision).

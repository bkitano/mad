# 262: PolySketchFormer — Sketched Polynomial Attention

**Category**: approximation
**Gain type**: efficiency
**Source**: Kacham, Mirrokni & Zhong (2024) — "PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels" (ICLR 2024)
**Paper**: [papers/polysketchformer.pdf]
**Documented**: 2026-02-16

## Description

PolySketchFormer replaces softmax attention with **degree-$p$ polynomial attention** $\sigma(\mathbf{q}, \mathbf{k}) = \langle \mathbf{q}, \mathbf{k} \rangle^p$ and then applies **TensorSketch-based random feature maps** to reduce the $O(h^p)$-dimensional exact feature map to a compact $r$-dimensional sketch ($r \ll h^p$), achieving **linear-time attention** with provable approximation guarantees.

The key insight is three-fold:

1. **Polynomial kernels match softmax quality at degree $p \geq 4$**: Normalized polynomial attention $\frac{\langle \mathbf{q}_i', \mathbf{k}_j' \rangle^p}{1 + \sum_j \langle \mathbf{q}_i', \mathbf{k}_j' \rangle^p}$ (after layer norm) empirically matches softmax attention on language modeling for $p \geq 4$, as both interpolate between uniform and argmax distributions.

2. **Polynomial sketching preserves non-negativity**: Standard TensorSketches can produce negative entries. PolySketchFormer introduces a "self-tensoring" trick: compute a degree-$p/2$ sketch $\mathbf{M}$, then square it as $\mathbf{M}^{\otimes 2}$ to guarantee all entries of the approximate attention matrix are non-negative (since they become inner products of the form $\langle \mathbf{a}^{\otimes 2}, \mathbf{b}^{\otimes 2} \rangle \geq 0$). This costs squaring the sketch size ($r \to r^2$) but eliminates training instability from negative attention weights.

3. **Block-based lower triangular multiplication for causal masking**: For causal (autoregressive) models, the naive RNN-style prefix sum $\sum_{j \leq i} \phi(\mathbf{k}_j) \mathbf{v}_j^\top$ requires $O(n)$ sequential steps. PolySketchFormer introduces a block-based algorithm to compute $\text{lt}_\triangledown(\mathbf{A} \cdot \mathbf{B}^\top) \cdot \mathbf{C}$ without materializing $\mathbf{A} \cdot \mathbf{B}^\top$, using block prefix sums with only $O(n/b)$ sequential steps (parallelizable via Blelloch scan).

**Directly relevant to Higher-Order Linear Attention (trick 222)**: HLA maintains second-moment prefix summaries $\mathbf{S}_t^K = \sum_{i \leq t} \mathbf{k}_i \mathbf{k}_i^\top$ which is equivalent to a degree-2 polynomial kernel. PolySketchFormer provides an alternative: instead of maintaining the full $d \times d$ matrix, sketch it down to $r$ dimensions via TensorSketch, trading exactness for reduced state size from $O(d^2)$ to $O(r)$ where $r = O(p \varepsilon^{-2} \log(1/\delta))$.

## Mathematical Form

**Core Operation — Polynomial Attention:**

After layer normalization (which zero-centers queries/keys), the normalized degree-$p$ attention weight is:

$$
\mathbf{A}_{i,j}^{(p)} = \frac{\langle \mathbf{q}_i', \mathbf{k}_j' \rangle^p}{1 + \sum_{j' \in [n]} \langle \mathbf{q}_i', \mathbf{k}_j' \rangle^p}
$$

where $\mathbf{q}_i' = \mathbf{q}_i / \sqrt{\beta} + \sqrt{\alpha/(\beta h)} \cdot \mathbf{1}_h$ and $\mathbf{k}_j' = \mathbf{k}_j / \sqrt{\beta} + \sqrt{\alpha/(\beta h)} \cdot \mathbf{1}_h$ are rescaled/biased after layer norm.

**Exact feature map decomposition:**

Using the identity $\langle \mathbf{x}, \mathbf{y} \rangle^p = \langle \mathbf{x}^{\otimes p}, \mathbf{y}^{\otimes p} \rangle$:

$$
(\mathbf{Q}\mathbf{K}^\top)^p = \mathbf{Q}^{\otimes p} (\mathbf{K}^{\otimes p})^\top \cdot \mathbf{V}
$$

The exact feature map $\phi(\mathbf{x}) = \mathbf{x}^{\otimes p} \in \mathbb{R}^{h^p}$ has exponential dimension in $p$.

**Approximate feature map via polynomial sketch:**

Theorem 1.1: For $p \geq 2$ even and error $\varepsilon \in (0, 0.5)$, there exists a randomized feature mapping $\phi' : \mathbb{R}^h \to \mathbb{R}^{r^2}$ with $r = \Theta(p \varepsilon^{-2} \log(1/\delta))$ such that:

1. $\forall i, j: \langle \phi'(\mathbf{q}_i), \phi'(\mathbf{k}_j) \rangle \geq 0$ (non-negativity)
2. $\sum_{i,j} |\langle \phi'(\mathbf{q}_i), \phi'(\mathbf{k}_j) \rangle - \langle \mathbf{q}_i, \mathbf{k}_j \rangle^p|^2 \leq \varepsilon^2 \sum_{i,j} \|\mathbf{q}_i\|_2^{2p} \|\mathbf{k}_j\|_2^{2p}$
3. Computing $\phi'(\mathbf{x})$ requires only $p/2$ matrix-vector multiplications of size $h \times r$, $(p/2 - 2)$ matvecs of size $r \times r$, $(p/2 - 1)$ Hadamard products, and 1 self-Kronecker product.

**Non-negative sketch construction (Algorithm 1):**

$$
\phi'(\mathbf{x}) = \left(\mathbf{x}^{\otimes(p/2)} \mathbf{S}\right)^{\otimes 2} \in \mathbb{R}^{r^2}
$$

where $\mathbf{S} \in \mathbb{R}^{h^{p/2} \times r}$ is a polynomial sketch matrix. The computation of $\mathbf{x}^{\otimes(p/2)} \mathbf{S}$ is done recursively:

```
PolySketchWithNegativity(A, r, p):
  if p = 1: return A
  M1 = PolySketchWithNegativity(A, r, p/2)
  M2 = PolySketchWithNegativity(A, r, p/2)
  Sample Gaussian G1, G2 of size h x r
  return sqrt(1/r) * [(M1 * G1) * (M2 * G2)]  // Hadamard product

PolySketchNonNegative(A, r, p):
  M = PolySketchWithNegativity(A, r, p/2)
  return M^{otimes 2}  // self-Kronecker: guarantees non-negativity
```

**Learnable variant**: Replace random Gaussian matrices $\mathbf{G}_1, \mathbf{G}_2$ with learned MLPs $f_1(\mathbf{M}_1), f_2(\mathbf{M}_2)$ — small networks with $O(r^2)$ parameters shared across all heads. Apply $\tanh$ for stability.

**Block-based causal lower triangular multiplication:**

To compute $\text{lt}_\triangledown(\mathbf{A} \cdot \mathbf{B}^\top) \cdot \mathbf{C}$ with block size $b$ and $t = n/b$ blocks:

$$
\mathbf{H}_l = \sum_{i \in B_l} \mathbf{b}_i \mathbf{c}_i^\top, \quad \mathbf{Z}_l = \sum_{j < l} \mathbf{H}_j
$$

$$
\mathbf{P}_l = \text{lt}_\triangledown(\mathbf{A}_l \mathbf{B}_l^\top) \mathbf{C}_l \quad \text{(local block, size } b \times b \text{)}
$$

For token $i$ in block $B_l$: output $= \mathbf{a}_i^\top \mathbf{Z}_l + \mathbf{P}_l[i']$ where $i'$ is the local index.

The prefix sum $\mathbf{Z}_l$ has only $t$ sequential steps (parallelizable via Blelloch scan). The local block $\mathbf{P}_l$ uses exact polynomial attention (no sketching needed for nearby tokens, size $O(b^2)$).

**Key Definitions:**

- $\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^h$ — query and key vectors (head dimension $h$)
- $p$ — polynomial degree (even integer, typically 4 or 8)
- $r$ — sketch dimension ($r = 32$ or $64$ in practice, vs $h^p$ exact)
- $\mathbf{S} \in \mathbb{R}^{h^{p/2} \times r}$ — implicit polynomial sketch matrix (never materialized)
- $b$ — block size for causal lower triangular multiplication (typically 1024)

## Complexity

| Operation | Softmax (FlashAttn) | Polynomial (exact) | PolySketchFormer |
|-----------|--------------------|--------------------|------------------|
| Attention compute | $O(n^2 h)$ | $O(n h^{p+1})$ | $O(n b (r + h) + n r^2 h)$ |
| Memory per token | $O(n)$ (KV cache) | $O(h^p)$ | $O(r^2)$ |
| Feature map compute | N/A | $O(h^p)$ | $O(h r + r^2)$ per token |
| Causal masking | Free (in FlashAttn) | $O(n)$ sequential | $O(n/b)$ sequential |

**With $r = 32, h = 64, p = 4, b = 1024$:**
- Sketch dimension: $r^2 = 1024$ vs exact $h^4 = 16.8$M
- Per-token sketch: $O(hr + r^2) = O(3072)$ — a few small matvecs
- Total: $O(n \cdot b \cdot h + n \cdot r^2 \cdot h)$ — linear in $n$

**Empirical wall-clock**: 2x speedup over FlashAttention (block=512) at 32k context, constant throughput across all context lengths.

## Applicability

- **Drop-in replacement for softmax attention** in decoder-only language models. Polynomial attention with $p \geq 4$ matches softmax quality on perplexity and downstream tasks (HellaSwag, PIQA, Physics QA).
- **Direct complement to Higher-Order Linear Attention (HLA, trick 222)**: HLA maintains exact second-moment summaries $\mathbf{S}_t^K \in \mathbb{R}^{d \times d}$. PolySketchFormer offers an alternative where this $d^2$-dimensional state is compressed to $r^2 \ll d^2$ via sketching, at the cost of approximation error. For HLA's third-order variant ($O(d^3)$ state), sketching becomes even more attractive.
- **Long-context training**: Linear-time attention enables training on 32k+ context lengths where softmax OOMs. Training throughput (steps/sec) remains constant as context length increases.
- **Hybrid with local attention**: The "Polysketch + local" variant uses exact polynomial attention within each block and sketched attention across blocks, combining short-range precision with long-range efficiency.

## Limitations

- **Sketch size vs quality tradeoff**: $r = 32$ gives acceptable perplexity but lags softmax slightly; $r = 64$ closes the gap but reduces the speedup to ~10% over FlashAttention.
- **Quadratic sketch for non-negativity**: The self-tensoring trick ($\mathbf{M}^{\otimes 2}$) squares the sketch dimension from $r$ to $r^2$, which is the dominant cost. Without it, training diverges due to negative attention weights.
- **Random projections are GPU-unfriendly unless learned**: The random Gaussian matrices in Algorithm 1 are small ($h \times r$) matvecs — not large GEMMs. The learned variant replaces these with small MLPs, which batch better but add parameters.
- **Validated on GPT-2 scale only**: Experiments are on GPT-2 Small/Medium/Large (100M-700M parameters). Scaling behavior at 1B+ is not demonstrated.
- **TPU-optimized implementation**: The reference implementation is in JAX/Pallas on TPUs. GPU (CUDA) implementation with tensor core utilization is not provided — the 2x speedup may not transfer directly.
- **No streaming/inference mode**: The paper focuses on training throughput. For autoregressive inference, one would still need a recurrent mode (maintaining sketched prefix sums), which is not explicitly described.

## Implementation Notes

```python
# PolySketchFormer — core operations

import torch
import torch.fft

def count_sketch(x, h_func, s_func, D):
    """Count Sketch of vector x into D dimensions.
    h_func: [d] -> [D], hash function (2-wise independent)
    s_func: [d] -> {-1, +1}, sign function (4-wise independent)
    """
    sketch = torch.zeros(D, device=x.device, dtype=x.dtype)
    for i in range(x.shape[0]):
        sketch[h_func[i]] += s_func[i] * x[i]
    return sketch

def tensor_sketch_degree2(x, D, h1, h2, s1, s2):
    """Compute TensorSketch of x (x) x via FFT convolution.
    Avoids materializing d^2-dimensional outer product.
    """
    # Count Sketch with two independent hash/sign families
    Cx1 = count_sketch(x, h1, s1, D)  # D-dim
    Cx2 = count_sketch(x, h2, s2, D)  # D-dim

    # Convolution via FFT = sketch of outer product
    Cx_tensor = torch.fft.ifft(
        torch.fft.fft(Cx1) * torch.fft.fft(Cx2)
    ).real
    return Cx_tensor  # D-dim approx of d^2-dim tensor product

def polysketch_nonneg(Q, K, r, p):
    """Approximate non-negative polynomial feature map.
    Q, K: (n, h) query/key matrices
    r: sketch dimension
    p: polynomial degree (must be even)
    Returns phi_Q, phi_K: (n, r^2)
    """
    # Recursive sketch for degree p/2
    M_Q = polysketch_with_neg(Q, r, p // 2)  # (n, r)
    M_K = polysketch_with_neg(K, r, p // 2)  # (n, r)

    # Self-Kronecker for non-negativity: (n, r) -> (n, r^2)
    phi_Q = M_Q.unsqueeze(-1) * M_Q.unsqueeze(-2)  # outer product
    phi_Q = phi_Q.reshape(Q.shape[0], r * r)
    phi_K = M_K.unsqueeze(-1) * M_K.unsqueeze(-2)
    phi_K = phi_K.reshape(K.shape[0], r * r)

    return phi_Q, phi_K  # <phi_Q[i], phi_K[j]> >= 0 always

def block_causal_attention(phi_Q, phi_K, V, block_size=1024):
    """Block-based causal linear attention.
    phi_Q, phi_K: (n, r^2) sketched features
    V: (n, d_v) values
    """
    n = phi_Q.shape[0]
    t = n // block_size
    d_v = V.shape[1]
    r2 = phi_K.shape[1]
    output = torch.zeros(n, d_v, device=V.device)

    # Compute block prefix sums: H_l = sum_{i in B_l} phi_K[i] * V[i]^T
    H = torch.zeros(t, r2, d_v, device=V.device)
    for l in range(t):
        bl = slice(l * block_size, (l + 1) * block_size)
        H[l] = phi_K[bl].T @ V[bl]  # (r^2, d_v) — this is a GEMM!

    # Prefix sum Z_l = sum_{j<l} H_j  (parallelizable via scan)
    Z = torch.cumsum(H, dim=0) - H  # exclusive prefix sum

    for l in range(t):
        bl = slice(l * block_size, (l + 1) * block_size)
        # Inter-block: sketched attention to all previous blocks
        inter = phi_Q[bl] @ Z[l]  # (b, r^2) @ (r^2, d_v) = (b, d_v) — GEMM!

        # Intra-block: exact local polynomial attention (no sketch needed)
        Q_local = phi_Q[bl]  # or use exact Q for better quality
        K_local = phi_K[bl]
        local_attn = torch.tril(Q_local @ K_local.T)  # (b, b) — causal mask
        intra = local_attn @ V[bl]  # (b, d_v)

        output[bl] = inter + intra

    return output

# GPU efficiency notes:
# - H[l] computation: (block_size, r^2)^T @ (block_size, d_v) = GEMM, tensor core friendly
# - Z prefix sum: trivially parallel via Blelloch scan over t blocks
# - Inter-block: (block_size, r^2) @ (r^2, d_v) = GEMM, tensor core friendly
# - Intra-block: (block_size, block_size) local attention = FlashAttention-style tiling
# - All operations are matmuls or elementwise — maps naturally to tensor cores
# - Memory: O(n * r^2) for features + O(t * r^2 * d_v) for prefix sums
# - With r=32: r^2=1024, comparable to typical hidden dims
```

## References

- Kacham, P., Mirrokni, V., & Zhong, P. (2024). PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels. ICLR 2024. [https://arxiv.org/abs/2310.01655](https://arxiv.org/abs/2310.01655)
- Implementation: [https://github.com/google-research/google-research/tree/master/polysketchformer](https://github.com/google-research/google-research/tree/master/polysketchformer)
- Ahle, T. D., et al. (2020). Oblivious sketching of high-degree polynomial kernels. SODA 2020.
- Pham, N. & Pagh, R. (2013). Fast and scalable estimation of uncertainty and fidelity in time series. KDD 2013 (TensorSketch).
- Zhang, Y., Qin, Z., & Gu, Q. (2025). Higher-order Linear Attention. arXiv:2510.27258 (HLA — complementary approach using exact second-moment summaries instead of sketches).
- Choromanski, K., et al. (2020). Rethinking Attention with Performers. (FAVOR+ random features for exponential kernels — related but different kernel class)

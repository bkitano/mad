# 082: Nyström Landmark Attention

**Category**: approximation
**Gain type**: efficiency
**Source**: Xiong et al. (2021), "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention" (AAAI 2021)
**Paper**: [papers/nystromformer-nystrom-attention.pdf]
**Documented**: 2026-02-15

## Description

The Nyström method is a classical technique for low-rank matrix approximation that reconstructs a full $n \times n$ matrix from a small $m \times m$ sampled submatrix and its interactions with all rows/columns. The Nyströmformer adapts this to approximate the softmax attention matrix $S = \text{softmax}(QK^T/\sqrt{d_q})$ by selecting $m \ll n$ "landmark" points from queries and keys (via segment-means), computing a small $m \times m$ kernel matrix $A_S$ among landmarks, then reconstructing the full $n \times n$ attention matrix as a product of three matrices: an $n \times m$ matrix, the pseudoinverse of $A_S$, and an $m \times n$ matrix. The key algebraic insight is that the Nyström approximation is equivalent to applying a Schur complement: the residual of the Nyström approximation $S - \hat{S}$ is exactly the Schur complement of the sampled block $A_S$ in the full matrix $S$, padded by zeros. When the attention matrix is approximately low-rank (as empirically observed), this Schur complement residual is small, yielding a high-fidelity $O(n)$ approximation. A skip connection via depthwise convolution compensates for approximation error.

## Mathematical Form

**Standard Self-Attention:**

$$
S = \text{softmax}\left(\frac{QK^T}{\sqrt{d_q}}\right) \in \mathbb{R}^{n \times n}, \quad D(Q, K, V) = S \cdot V
$$

**Nyström Matrix Approximation (classical):**

Partition $S$ by sampling $m$ rows and columns:

$$
S = \begin{bmatrix} A_S & B_S \\ F_S & C_S \end{bmatrix}
$$

where $A_S \in \mathbb{R}^{m \times m}$, $B_S \in \mathbb{R}^{m \times (n-m)}$, $F_S \in \mathbb{R}^{(n-m) \times m}$, $C_S \in \mathbb{R}^{(n-m) \times (n-m)}$.

The Nyström approximation reconstructs $S$ as:

$$
\hat{S} = \begin{bmatrix} A_S \\ F_S \end{bmatrix} A_S^{+} \begin{bmatrix} A_S & B_S \end{bmatrix}
$$

where $A_S^{+}$ is the Moore-Penrose pseudoinverse.

**Connection to Schur Complement:**

The approximation error is the Schur complement of $A_S$ in $S$:

$$
S - \hat{S} = \begin{bmatrix} 0 & 0 \\ 0 & C_S - F_S A_S^{+} B_S \end{bmatrix}
$$

This is zero when $S$ has rank $\leq m$ (i.e., the Schur complement vanishes).

**Nyströmformer Adaptation (landmark-based):**

Select $m$ landmark queries $\tilde{Q} \in \mathbb{R}^{m \times d_q}$ and keys $\tilde{K} \in \mathbb{R}^{m \times d_q}$ via segment-means:

$$
\tilde{q}_j = \frac{1}{l} \sum_{i=(j-1) \times l+1}^{j \times l} q_i, \quad \tilde{k}_j = \frac{1}{l} \sum_{i=(j-1) \times l+1}^{j \times l} k_i, \quad l = n/m
$$

Compute three softmax kernel matrices:

$$
\tilde{F} = \text{softmax}\left(\frac{Q\tilde{K}^T}{\sqrt{d_q}}\right) \in \mathbb{R}^{n \times m}, \quad \tilde{A} = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_q}}\right) \in \mathbb{R}^{m \times m}, \quad \tilde{B} = \text{softmax}\left(\frac{\tilde{Q}K^T}{\sqrt{d_q}}\right) \in \mathbb{R}^{m \times n}
$$

The Nyström-approximated attention output is:

$$
\hat{S}V = \tilde{F} \cdot \tilde{A}^{+} \cdot \tilde{B} \cdot V
$$

**Iterative Moore-Penrose Pseudoinverse (avoiding SVD):**

$$
Z_{j+1} = \frac{1}{4} Z_j (13I - A_S Z_j (15I - A_S Z_j (7I - A_S Z_j)))
$$

with $Z_0 = A_S^T / (\|A_S\|_1 \|A_S\|_\infty)$, converging to $A_S^{+}$ in third-order (6 iterations suffice).

**Key Definitions:**

- $Q, K, V \in \mathbb{R}^{n \times d_q}$ — query, key, value matrices
- $n$ — sequence length (number of tokens)
- $m$ — number of landmarks ($m \ll n$, typically 32 or 64)
- $\tilde{Q}, \tilde{K} \in \mathbb{R}^{m \times d_q}$ — landmark query and key matrices (via segment-means)
- $A_S \in \mathbb{R}^{m \times m}$ — landmark kernel matrix (softmax of landmark interactions)
- $A_S^{+}$ — Moore-Penrose pseudoinverse of $A_S$
- $d_q$ — head dimension; $d_v$ — value dimension

## Complexity

| Operation | Standard Attention | Nyström Attention |
|-----------|-------------------|-------------------|
| Attention matrix | $O(n^2 d_q)$ | $O(nm d_q)$ |
| Pseudoinverse ($m \times m$) | — | $O(m^3)$ |
| Matrix products | $O(n^2 d_v)$ | $O(nm^2 + nm d_v)$ |
| **Total** | $O(n^2(d_q + d_v))$ | $O(nm^2 + nm(d_q + d_v) + m^3)$ |

When $m \ll n$: total is $O(n)$ in sequence length.

**Memory:** $O(n m + m^2 + nm + n d_v)$ vs $O(n^2 + n d_v)$ — linear vs quadratic in $n$.

**Empirical speedups** (from paper, sequence length 8192): 12.7× speed-up and 22.7× memory savings over standard attention; 1.7× memory savings and 3× speed-up over Longformer.

## Applicability

- **Long-sequence transformers**: Directly replaces $O(n^2)$ softmax attention with $O(n)$ Nyström-approximated attention for NLP, vision, and other domains
- **BERT-style masked language models**: Nyströmformer pre-trains competitively with BERT-base on GLUE and IMDB benchmarks with significant efficiency gains
- **Long Range Arena (LRA)**: Outperforms Reformer (+3.91%), Linformer (+3.36%), and Performer (+5.32%) in average accuracy across all LRA tasks
- **Any kernel attention**: Applicable whenever the attention matrix is approximately low-rank, which is common in practice (Wang et al., 2020)
- **Connection to other Schur-complement tricks**: The Nyström residual being a Schur complement means that hierarchical Nyström refinements correspond to recursive Schur complement computations (cf. HSS matrices)

## Limitations

- The softmax function is applied row-wise, making direct column/row sampling of the pre-softmax matrix $QK^T$ insufficient — the landmark approach is a compromise that avoids computing the full $QK^T$
- Segment-means landmarks assume some locality in the sequence; for highly non-local attention patterns, more sophisticated landmark selection (e.g., k-means) may be needed
- The iterative pseudoinverse requires 6 iterations of a degree-4 polynomial recurrence per forward pass, adding constant overhead
- Approximation quality degrades if the attention matrix has high effective rank (though empirically this is rare)
- Does not support causal masking natively — would require modifications for autoregressive generation
- The $m \times m$ pseudoinverse is a sequential bottleneck (though $m$ is small, typically 32-64)

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def nystrom_attention(Q, K, V, num_landmarks=64, num_pinv_iters=6):
    """Nyström-approximated self-attention with O(n) complexity.

    Args:
        Q, K: (batch, heads, seq_len, d_k)
        V:    (batch, heads, seq_len, d_v)
        num_landmarks: m, number of landmark points
        num_pinv_iters: iterations for iterative pseudoinverse
    """
    b, h, n, d = Q.shape
    m = num_landmarks
    scale = d ** -0.5

    # Segment-means to compute landmarks
    # Reshape sequence into m segments, average each
    seg_len = n // m  # l = n/m
    Q_landmarks = Q.reshape(b, h, m, seg_len, d).mean(dim=3)  # (b, h, m, d)
    K_landmarks = K.reshape(b, h, m, seg_len, d).mean(dim=3)  # (b, h, m, d)

    # Compute the three kernel matrices (after softmax)
    # F_tilde: n x m  (Q vs K_landmarks)
    # A_tilde: m x m  (Q_landmarks vs K_landmarks)
    # B_tilde: m x n  (Q_landmarks vs K)
    F_tilde = F.softmax(Q @ K_landmarks.transpose(-2, -1) * scale, dim=-1)  # (b,h,n,m)
    A_tilde = F.softmax(Q_landmarks @ K_landmarks.transpose(-2, -1) * scale, dim=-1)  # (b,h,m,m)
    B_tilde = F.softmax(Q_landmarks @ K.transpose(-2, -1) * scale, dim=-1)  # (b,h,m,n)

    # Iterative Moore-Penrose pseudoinverse of A_tilde
    # Initialize: Z_0 = A^T / (||A||_1 * ||A||_inf)
    A_t = A_tilde.transpose(-2, -1)
    norm1 = A_tilde.abs().sum(dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    norminf = A_tilde.abs().sum(dim=-2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    Z = A_t / (norm1 * norminf)

    I_m = torch.eye(m, device=Q.device).unsqueeze(0).unsqueeze(0)
    for _ in range(num_pinv_iters):
        AZ = A_tilde @ Z
        Z = 0.25 * Z @ (13 * I_m - AZ @ (15 * I_m - AZ @ (7 * I_m - AZ)))

    # Nyström approximation: S_hat @ V = F_tilde @ Z @ B_tilde @ V
    # Compute right-to-left for efficiency: (m,n) @ (n,d_v) = (m,d_v) first
    out = F_tilde @ (Z @ (B_tilde @ V))  # (b, h, n, d_v)
    return out
```

## References

- Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., & Singh, V. (2021). Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention. *AAAI 2021*. arXiv:2102.03902.
- Williams, C., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. *NeurIPS 2001*.
- Wang, S., & Zhang, Z. (2013). Improving CUR matrix decomposition and the Nyström approximation via adaptive sampling. *JMLR*, 14, 2729–2769.
- Razavi, S. A., et al. (2014). Adaptive thresholding for sparse approximation. *IEEE Signal Process. Lett.*
- Epperly, E. N. (2022). Low-Rank Approximation Toolbox: Nyström, Cholesky, and Schur. Blog post.

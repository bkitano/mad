# 154: FMMformer Near-field/Far-field Attention Decomposition

**Category**: approximation
**Gain type**: efficiency
**Source**: Nguyen, Suliafu, Osher, Chen & Wang (NeurIPS 2021)
**Paper**: [papers/fmmformer-near-far-field-attention.pdf]
**Documented**: 2026-02-15

## Description

FMMformer decomposes the attention matrix into a **sparse banded near-field component** plus a **low-rank far-field component**, directly inspired by the Fast Multipole Method's (FMM) decomposition of particle interactions. The key observation, validated empirically on trained transformer attention maps, is that attention matrices are approximately **diagonal-plus-semi-separable**: after removing a banded matrix $D$ of bandwidth $k$, the residual $A - D$ has very low numerical rank (typically $r \leq 5$ for bandwidth $k = 10$).

The core replacement is:

$$\hat{V} = (w_1 D + w_2 L)V$$

where $D$ is a banded (near-field) attention matrix computed from local token interactions, $L$ is a low-rank (far-field) attention matrix computed via kernelized linear attention, and $w_1, w_2$ are learnable blending weights. Both $DV$ and $LV$ can be computed in $O(N)$ time and memory, making FMMformers linear-complexity transformers.

The critical GPU-friendly insight is that both components map to well-optimized primitives: the banded near-field is a **sliding window attention** (contiguous memory access, no gather/scatter), and the far-field is a **linear attention** (two small matrix multiplications). No tree traversals, no irregular indexing. On LRA benchmarks, FMMformer (2-kernel + Band$_5$) achieves 60.74% average accuracy vs. softmax transformer's 58.70%, demonstrating that the decomposition can actually **improve** accuracy over full attention while being $O(N)$.

## Mathematical Form

**Standard Self-Attention:**

$$
\hat{V} = AV = \text{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right) V
$$

where $Q, K, V \in \mathbb{R}^{N \times D}$ and $A \in \mathbb{R}^{N \times N}$ is the attention matrix.

**FMMformer Decomposition:**

$$
\hat{V} = (w_1 D + w_2 L)V
$$

where $w_1, w_2 \in \mathbb{R}$ are learnable weights (enforced positive via sigmoid).

**Near-field Attention (Banded Matrix $D$):**

$$
D = \text{softmax}\left(\text{band}_k\left(\frac{QK^\top}{\sqrt{D}}\right)\right)
$$

where $\text{band}_k(\cdot)$ extracts only entries within bandwidth $k$ of the diagonal ($k \ll N$). In practice, only the $2k+1$ nonzero entries per row are computed — no need to form the full $N \times N$ matrix.

$$
D_{ij} = \begin{cases} \text{softmax}\left(\frac{q_i^\top k_j}{\sqrt{D}}\right) & \text{if } |i - j| \leq k \\ 0 & \text{otherwise} \end{cases}
$$

**Far-field Attention (Low-rank Matrix $L$):**

$$
L = \sum_{l=1}^{r} a_l b_l^\top = \sum_{l=1}^{r} \phi_l(Q) \phi_l(K)^\top
$$

where $\phi_l: \mathbb{R}^D \to \mathbb{R}^N$ are feature maps applied row-wise. The low-rank MVM is:

$$
LV = \sum_{l=1}^{r} \phi_l(Q) \left(\phi_l(K)^\top V\right)
$$

Each inner product $\phi_l(K)^\top V \in \mathbb{R}^{1 \times D_v}$ is computed first ($O(N D_v)$), then the outer expansion costs $O(N D_v)$ — total $O(rND_v)$.

**Practical Feature Maps (Kernelized Linear Attention):**

Using the generalized self-attention framework:

$$
\hat{v}_i = \frac{\sum_{j=1}^{N} k(q_i, k_j) v_j}{\sum_{j=1}^{N} k(q_i, k_j)} = \frac{\phi(q_i)^\top \sum_{j=1}^{N} \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j=1}^{N} \phi(k_j)}
$$

For rank $r$, multiple feature maps are used:
- $\phi_1(x) = \text{elu}(x) + 1$ (from linear transformer)
- $\phi_2(x) = \text{elu}(-x) + 1$
- $\phi_3(x) = \tanh(x)$

These are linearly independent for almost all $x$, ensuring the resulting $L$ has full rank $r$.

**Full FMMformer Attention:**

$$
\hat{V} = w_1 DV + w_2 \sum_{l=1}^{r} \frac{\phi_l(Q)(\phi_l(K)^\top V)}{\phi_l(Q) \phi_l(K)^\top}
$$

**Causal Masking:**

For autoregressive models, causal masking is straightforward:
- Near-field: mask out $j > i$ entries in the banded matrix (upper triangle within the band)
- Far-field: truncate the cumulative sum from $1$ to $i$ instead of $1$ to $N$

**Diagonal-Plus-Semi-Separable Structure:**

The theoretical justification comes from the observation that attention matrices satisfy a semi-separable structure. A matrix $A$ is $(p,q)$-semi-separable if:

$$
A = \text{triu}(UV^\top, 0) + \text{tril}(WZ^\top, 1)
$$

with $U, V \in \mathbb{R}^{N \times p}$ and $W, Z \in \mathbb{R}^{N \times q}$. Extending to diagonal-plus-semi-separable:

$$
A = D + \text{triu}(UV^\top, 1) + \text{tril}(WZ^\top, 1)
$$

The FMM well-separation condition formalizes this: if key vectors $\{k_j, j \in T_2\}$ are clustered around a center $k^*$ such that $|k_j - k^*| \leq \delta |q_i - k^*|$ for all $q_i \in T_1$, then the submatrix $A(T_1, T_2)$ can be approximated by a rank-$p$ matrix with $p = C|\log_\delta \epsilon|$ terms.

**Key Definitions:**

- $N$ — sequence length
- $D$ — head dimension
- $k$ — near-field bandwidth (hyperparameter, typically 5–30)
- $r$ — far-field rank (number of feature maps, typically 1–3)
- $\phi_l(\cdot)$ — feature maps for kernelized linear attention
- $w_1, w_2$ — learnable blending weights (sigmoid-gated)
- $\delta \in (0,1)$ — well-separation parameter from FMM theory

## Complexity

| Operation | Softmax Attention | FMMformer |
|-----------|-------------------|-----------|
| Attention computation | $O(N^2 D)$ | $O(N(2k+1)D + rND)$ |
| Memory | $O(N^2 + ND)$ | $O(N(2k+1) + rND)$ |
| Total (for $k, r \ll N$) | $O(N^2 D)$ | $O(ND)$ — linear |

**Detailed breakdown:**

- Near-field $DV$: $O(N \cdot (2k+1) \cdot D)$ — compute $k$ dot products per query, multiply by $V$
- Far-field $LV$: $O(r \cdot N \cdot D)$ — $r$ passes of linear attention, each $O(ND)$
- Blending: $O(ND)$ — elementwise weighted sum
- Total: $O(N(2k+1+r)D)$ which is $O(ND)$ for fixed $k, r$

**Empirical wall-clock**: From Figure 6 in the paper, on NVIDIA 3090Ti:
- At $N = 2^{14}$: FMMformer (rank 3 + band$_{30}$) is ~10$\times$ faster than softmax attention
- At $N = 2^{16}$: FMMformer is ~40$\times$ faster (softmax becomes memory-bound)
- Memory: FMMformer uses ~100$\times$ less GPU memory at $N = 2^{16}$

## Applicability

- **Long-context language modeling**: FMMformer achieves 36.11 test perplexity on WikiText-103 (vs. 34.29 for softmax, 38.40 for linear), closing the gap while maintaining $O(N)$ complexity. The near-field component captures the crucial local attention pattern that linear transformers miss.
- **Long Range Arena**: FMMformer achieves 60.74% average accuracy across all 5 LRA tasks (ListOps, Text, Retrieval, Image, Pathfinder), outperforming both softmax transformers (58.70%) and linear transformers (54.67%). This is significant because it demonstrates the decomposition improves expressivity, not just efficiency.
- **Hybrid architectures**: The banded + low-rank pattern is exactly the structure used in modern hybrid architectures like Mamba-2 (SSD) + sliding window attention. FMMformer provides the theoretical FMM justification for why this works.
- **Vision transformers**: The near-field captures spatial locality while far-field captures global context — a natural fit for image tokens arranged on a grid.
- **Streaming/online inference**: The banded near-field requires only the last $k$ KV pairs in cache, and the linear attention far-field maintains a fixed-size state. Total KV cache: $O(kD + rD)$ instead of $O(ND)$.

## Limitations

- **Near-field bandwidth $k$ is a hyperparameter**: The optimal $k$ depends on the task. WikiText-103 results show $k=20$ significantly outperforms $k=5$ (36.43 vs 37.29 PPL), but larger $k$ increases the constant factor. In practice, $k=5$–$30$ works well.
- **Far-field approximation quality**: The kernelized linear attention approximation using simple feature maps ($\text{elu}+1$, $\tanh$) is a coarse approximation of softmax attention. The paper uses only $r \leq 3$ kernels, which limits expressivity of the far-field. More sophisticated feature maps (FAVOR+, etc.) could improve this at additional cost.
- **Not a drop-in replacement**: Requires retraining from scratch. The banded matrix $D$ uses a different softmax normalization (only over the band) than full attention, and the far-field uses a different kernel entirely. Pre-trained attention weights cannot be directly transferred.
- **Banded near-field assumes sequential token ordering**: For tasks where relevant tokens are not positionally close (e.g., code with distant variable references), the banded pattern may miss important interactions. Graph-structured near-fields could help but break the simple GPU-friendly structure.
- **Gap to softmax on language modeling**: On WikiText-103, FMMformer still trails softmax by ~2 PPL points (36.11 vs 34.29). The paper suggests larger bandwidth and more kernels can close this gap, but at the cost of increasing the constant factor.

## Implementation Notes

```python
# FMMformer: Near-field + Far-field Attention
# GPU-friendly: banded attention is a sliding window (FlashAttention-compatible),
# far-field is linear attention (two small GEMMs).
import torch
import torch.nn.functional as F

class FMMformerAttention(torch.nn.Module):
    """
    FMMformer attention: O(N) time and memory.
    Near-field: banded softmax attention (sliding window)
    Far-field: kernelized linear attention with r feature maps
    """
    def __init__(self, d_model, n_heads, bandwidth=5, n_kernels=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bandwidth = bandwidth
        self.n_kernels = n_kernels

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        # Learnable blending weights (sigmoid-gated for positivity)
        self.w1_logit = torch.nn.Parameter(torch.tensor(0.0))
        self.w2_logit = torch.nn.Parameter(torch.tensor(0.0))

    def feature_map(self, x, kernel_idx):
        """Feature maps for far-field linear attention."""
        if kernel_idx == 0:
            return F.elu(x) + 1      # phi_1
        elif kernel_idx == 1:
            return F.elu(-x) + 1     # phi_2
        elif kernel_idx == 2:
            return torch.tanh(x)      # phi_3
        else:
            raise ValueError(f"Unknown kernel index {kernel_idx}")

    def near_field_attention(self, Q, K, V):
        """
        Banded (sliding window) softmax attention.
        O(N * bandwidth * D) — maps to FlashAttention with window mask.
        """
        N, D = Q.shape[-2], Q.shape[-1]
        k = self.bandwidth
        scale = D ** -0.5

        # Compute local attention scores within bandwidth
        # For each query i, attend to keys [max(0,i-k), min(N-1,i+k)]
        # This is a sliding window — highly GPU-efficient
        output = torch.zeros_like(V)
        for offset in range(-k, k + 1):
            # Shifted dot products — vectorized over all positions
            if offset >= 0:
                scores = (Q[:, :, :N-offset] * K[:, :, offset:]).sum(-1) * scale
                attn = torch.softmax(scores, dim=-1)  # normalize per-row
                output[:, :, :N-offset] += attn.unsqueeze(-1) * V[:, :, offset:]
            else:
                scores = (Q[:, :, -offset:] * K[:, :, :N+offset]).sum(-1) * scale
                attn = torch.softmax(scores, dim=-1)
                output[:, :, -offset:] += attn.unsqueeze(-1) * V[:, :, :N+offset]

        return output  # In practice, use FlashAttention with window mask

    def far_field_attention(self, Q, K, V):
        """
        Kernelized linear attention with r feature maps.
        O(r * N * D) — two GEMMs per kernel.
        """
        output = torch.zeros_like(V)
        for l in range(self.n_kernels):
            phi_Q = self.feature_map(Q, l)  # (B, H, N, D)
            phi_K = self.feature_map(K, l)  # (B, H, N, D)

            # Linear attention: phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
            KV = torch.einsum('bhnd,bhnv->bhdv', phi_K, V)  # (B,H,D,D_v)
            K_sum = phi_K.sum(dim=-2)  # (B, H, D)

            numer = torch.einsum('bhnd,bhdv->bhnv', phi_Q, KV)  # (B,H,N,D_v)
            denom = torch.einsum('bhnd,bhd->bhn', phi_Q, K_sum)  # (B,H,N)

            output += numer / (denom.unsqueeze(-1) + 1e-6)

        return output

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.W_q(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        w1 = torch.sigmoid(self.w1_logit)
        w2 = torch.sigmoid(self.w2_logit)

        near = self.near_field_attention(Q, K, V)
        far = self.far_field_attention(Q, K, V)

        output = w1 * near + w2 * far  # (B, H, N, D_head)
        output = output.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.W_o(output)

# GPU efficiency analysis:
# Near-field (banded):
#   - Sliding window attention: coalesced memory access
#   - Maps directly to FlashAttention-2 with window_size=k
#   - Arithmetic intensity: same as FlashAttention within the window
#   - Tensor core utilization: full (it's a sequence of small matmuls)
#
# Far-field (linear attention):
#   - Two GEMMs per kernel: phi(K)^T @ V  and  phi(Q) @ result
#   - Shapes: (D, N) @ (N, D_v) = (D, D_v) — small matmul, compute-bound
#   - Feature map application: elementwise (elu, tanh) — bandwidth-bound
#   - Total: r * 2 GEMMs + r elementwise ops
#
# Blending: single elementwise weighted sum — negligible
#
# For FlashAttention integration:
#   Use flash_attn with sliding_window=bandwidth for near-field
#   Use standard einsum for far-field (it's already O(N))
```

## References

- Nguyen, T. M., Suliafu, V., Osher, S., Chen, L., & Wang, B. (2021). FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention. NeurIPS 2021. arXiv:2108.02347.
- Greengard, L. & Rokhlin, V. (1987). A fast algorithm for particle simulations. J. Comput. Phys., 73(2):325-348.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Choromanski, K., et al. (2021). Rethinking Attention with Performers. ICLR 2021.
- Bebendorf, M. (2008). Hierarchical Matrices. Springer.
- Tay, Y., et al. (2021). Long Range Arena: A Benchmark for Efficient Transformers. ICLR 2021.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.

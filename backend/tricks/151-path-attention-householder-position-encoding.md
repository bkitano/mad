# 151: PaTH Attention — Position Encoding via Accumulated Householder Transformations

**Category**: decomposition
**Gain type**: expressivity
**Source**: Yang, Shen, Wen, Tan, Mishra, Ren, Panda, Kim (2026) — NeurIPS 2025
**Paper**: [papers/path-attention-householder-position.pdf]
**Documented**: 2026-02-15

## Description

PaTH (position encoding with accumulated Householder transformations) replaces RoPE's static, data-independent rotation matrices with **data-dependent** cumulative products of Householder-like transformations $\mathbf{H}_t = I - \beta_t \mathbf{w}_t \mathbf{w}_t^\top$ applied to the key/query bilinear form. Whereas RoPE computes attention logits as $\mathbf{q}_i^\top \mathbf{R}^{i-j} \mathbf{k}_j$ with a fixed block-diagonal rotation $\mathbf{R}$, PaTH computes:

$$
A_{ij} \propto \exp\!\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^{i} \mathbf{H}_s\right) \mathbf{q}_i\right)
$$

where each $\mathbf{H}_s$ depends on the input token $\mathbf{x}_s$. This makes the relative position encoding **input-adaptive**: the transformation between positions $j$ and $i$ depends on the tokens in between, not just the distance $i - j$.

PaTH is the **softmax-attention analog of DeltaProduct**: unrolling the DeltaProduct RNN recurrence yields the same cumulative Householder product structure. This connection gives PaTH transformers the state-tracking capabilities of expressive linear RNNs while retaining the associative-recall strengths of softmax attention. A one-layer PaTH transformer with two attention heads can solve NC$^1$-complete problems under AC$^0$ reductions — beyond the TC$^0$ complexity class of standard (RoPE) transformers.

The paper develops a FlashAttention-style blockwise algorithm using the UT transform and masked UT representations, with a Triton implementation available in the flash-linear-attention library.

## Mathematical Form

**Core Attention Score:**

$$
A_{ij} \propto \exp\!\left(\mathbf{k}_j^\top \left(\prod_{t=j+1}^{i} \mathbf{H}_t\right) \mathbf{q}_i\right)
$$

where:
- $\mathbf{H}_t = I - \beta_t \mathbf{w}_t \mathbf{w}_t^\top \in \mathbb{R}^{d \times d}$ — data-dependent Householder-like transformation
- $\beta_t = 2 \times \text{sigmoid}(\mathbf{u}^\top \mathbf{x}_t + b) \in (0, 2)$ — input-dependent scalar
- $\mathbf{w}_t \in \mathbb{R}^d$ — direction vector, computed via a low-rank linear layer + short convolution + $L_2$ normalization of $\mathbf{x}_t$

**Connection to Linear RNNs (DeltaProduct):**

The DeltaProduct RNN output unrolls to:

$$
\text{RNN: } \mathbf{o}_t = \sum_{j=1}^t \mathbf{v}_j \mathbf{k}_j^\top \left(\prod_{s=j+1}^t \mathbf{H}_s\right) \mathbf{q}_t
$$

PaTH applies softmax normalization to the same expression:

$$
\text{PaTH: } \mathbf{o}_t = \frac{1}{Z_t} \sum_{j=1}^t \mathbf{v}_j \exp\!\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^t \mathbf{H}_s\right) \mathbf{q}_i\right)
$$

**UT Transform for Product Computation (within a block of $B$ tokens):**

For a sequence of $L$ transformations $\mathbf{H}_t = I - \beta_t \mathbf{w}_t \mathbf{w}_t^\top$, the cumulative product is compactly expressed as:

$$
\mathbf{P} := \prod_{t=0}^{L-1} \mathbf{H}_t = I - \mathbf{W}^\top \mathbf{T}^{-1} \mathbf{W} \in \mathbb{R}^{d \times d}
$$

where:

$$
\mathbf{T}^{-1} := \left(I + \text{strictLower}(\mathbf{D}\mathbf{W}\mathbf{W}^\top)\right)^{-1} \mathbf{D} \in \mathbb{R}^{L \times L}
$$

- $\mathbf{W} = [\mathbf{w}_0, \ldots, \mathbf{w}_{L-1}]^\top \in \mathbb{R}^{L \times d}$
- $\mathbf{D} = \text{diag}(\beta_0, \ldots, \beta_{L-1}) \in \mathbb{R}^{L \times L}$

**Masked UT for Arbitrary Sub-intervals $[s_0, e_0]$:**

$$
\prod_{t=s_0}^{e_0} \mathbf{H}_t = I - (\mathbf{W} \odot \mathbf{M}_{s_0}^L)^\top \mathbf{T}^{-1} (\mathbf{W} \odot \mathbf{M}_{e_0}^R)
$$

where binary masks $\mathbf{M}_{s_0}^L$ and $\mathbf{M}_{e_0}^R$ select appropriate rows. This enables reusing a single global $\mathbf{T}^{-1}$ across all query-key pairs.

**Full Attention Matrix (within a block):**

$$
\tilde{\mathbf{A}} = \text{lower}(\mathbf{Q}\mathbf{K}^\top) - \text{lower}(\mathbf{Q}\mathbf{W}^\top)\;\mathbf{T}^{-1}\;\text{strictLower}(\mathbf{W}\mathbf{K}^\top)
$$

**Boundary-Adjusted Queries and Keys (for blockwise processing):**

$$
\overleftarrow{\mathbf{Q}}_{[i]} = \mathbf{Q}_{[i]} - \text{lower}(\mathbf{Q}_{[i]}\mathbf{W}_{[i]}^\top)\;\mathbf{T}_{[i]}^{-1}\;\mathbf{W}_{[i]} \in \mathbb{R}^{B \times d}
$$

$$
\overrightarrow{\mathbf{K}}_{[i]} = \mathbf{K}_{[i]} - \left(\mathbf{T}_{[i]}^{-1}\;\text{strictLower}(\mathbf{W}_{[i]}\mathbf{K}_{[i]}^\top)\right)^\top \mathbf{W}_{[i]} \in \mathbb{R}^{B \times d}
$$

Cross-block attention uses these adjusted representations with streaming cumulative products $\mathbf{P}_{[i]}$ updated right-to-left.

**In-place KV Cache Update (for inference):**

$$
\mathbf{k}_i^{(t)} \leftarrow (I - \beta_t \mathbf{w}_t \mathbf{w}_t^\top) \mathbf{k}_i^{(t-1)} \quad \text{for all } i < t
$$

This rank-1 update to all cached keys eliminates storing $\{\mathbf{w}_t\}$ vectors and makes decoding equivalent to standard softmax attention with FlashDecoding/PagedAttention.

**Key Definitions:**

- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$ — query, key, value matrices
- $\mathbf{W} \in \mathbb{R}^{L \times d}$ — stacked Householder direction vectors
- $\mathbf{T}^{-1} \in \mathbb{R}^{B \times B}$ — UT transform matrix (computed per block)
- $\mathbf{P}_{[i]} \in \mathbb{R}^{d \times d}$ — cumulative Householder product at block boundary $i$
- $B$ — block size (typically $\approx d$, e.g., 64)

## Complexity

| Operation | Standard Attention (RoPE) | PaTH Attention |
|-----------|--------------------------|----------------|
| Attention logits (per block pair) | $O(B^2 d)$ | $O(B^2 d)$ |
| Position preprocessing (per block) | $O(Bd)$ (rotation) | $O(B^3 + B^2 d)$ (UT transform) |
| Total preprocessing | $O(Ld)$ | $O(LB^2 + LBd)$ |
| Cross-block query update | — | $O(Bd^2)$ per block |
| **Total attention cost** | $O(L^2 d)$ | $O(L^2 d + Ld^2/B)$ |
| Inference (per-step KV update) | $O(d)$ | $O(Ld)$ (rank-1 update to all keys) |

**Memory:** $O(Ld)$ for $\mathbf{W}$ vectors (same order as $\mathbf{Q}, \mathbf{K}$); $O(B^2)$ for $\mathbf{T}^{-1}$ per block.

**When $B \approx d$:** Preprocessing is $O(LB^2 + LBd) = O(Ld^2)$, and the cross-block term adds $O(Ld^2/B) = O(Ld)$ per query block traversal. Total complexity matches standard attention's $O(L^2 d)$ with quadratic scaling in sequence length.

**Wall-clock (H100, batch=32, 32 heads, $d$=64):**
- PaTH-triton incurs a modest slowdown vs FlashAttention-triton (roughly 1.2-1.5$\times$ slower)
- PaTH outperforms FoX-triton at all sequence lengths tested (2K-16K)
- Further kernel-level optimizations (e.g., ThunderKittens) expected to close the gap

## Applicability

- **Transformer position encoding:** Drop-in replacement for RoPE that adds data-dependent positional information. Compatible with standard softmax attention infrastructure (FlashAttention, FlashDecoding, PagedAttention)
- **State-tracking tasks:** Solves flip-flop language modeling (FFLM) with 0% error where RoPE gets 6.9% error; solves $A_5$ word problems with fewer layers than baselines
- **Long-context modeling:** PaTH generalizes to 32K tokens (trained on 4K); PaTH-FoX generalizes to 64K tokens on code, books, and conversational domains
- **Conversion from RoPE:** Pretrained RoPE transformers can be distilled into PaTH with continued pretraining (100M tokens MSE distillation + 3B tokens KL fine-tuning), yielding gains on math/coding benchmarks
- **PaTH-FoX variant:** Combines PaTH with Forgetting Transformer's data-dependent additive logit modification for further gains, especially on length extrapolation
- **Context parallelism:** Compatible with Ring Attention / distributed context-parallel strategies — each device computes local $\overleftarrow{\mathbf{Q}}$, $\overrightarrow{\mathbf{K}}$, and $\mathbf{P}^{(d)}$, then passes transformed KV + cumulative product during ring communication

## Limitations

- **Numerical precision:** Cumulative Householder products can become unstable under BF16; $\beta$ must be clipped to prevent reaching 2 (which would cause eigenvalues $> 1$ and divergence)
- **Speed comparisons limited to $d = 64$:** Larger head dimensions increase the $O(Bd^2)$ cross-block query update cost; scaling behavior for $d = 128$ or $d = 256$ not yet benchmarked
- **No rotational structure:** A single Householder reflection cannot model rotations (which require products of 2+ reflections). This means PaTH does not subsume RoPE's geometric inductive biases for rotation-based relative position
- **Inference overhead:** Each new token requires a rank-1 update to all cached keys ($O(Ld)$ per token), unlike RoPE which requires no KV cache modification. For very long sequences, this may dominate
- **Additional parameters:** Small number of extra parameters for the $\mathbf{w}_t$ projection (low-rank linear + short convolution + normalization)

## Implementation Notes

```python
# PaTH Attention blockwise algorithm (simplified pseudocode)
# Based on Yang et al. (2026), Section 3.3

def path_attention_blockwise(Q, K, V, W, beta, B):
    """
    Q, K, V: (L, d) - query, key, value
    W: (L, d) - Householder direction vectors (L2-normalized)
    beta: (L,) - scaling factors in (0, 2)
    B: int - block size (typically ~ d)
    """
    L, d = Q.shape
    n_blocks = L // B

    # Reshape into blocks
    Q_b = Q.reshape(n_blocks, B, d)
    K_b = K.reshape(n_blocks, B, d)
    V_b = V.reshape(n_blocks, B, d)
    W_b = W.reshape(n_blocks, B, d)
    beta_b = beta.reshape(n_blocks, B)

    O = torch.zeros_like(V)

    for i in range(n_blocks):
        # === INTRA-BLOCK: UT Transform (Eq. 10-11 analog) ===
        D = torch.diag(beta_b[i])  # (B, B)
        # Tensor-core matmul: (B, d) @ (d, B) -> (B, B)
        L_mat = torch.tril(D @ W_b[i] @ W_b[i].T, diagonal=-1)
        # Forward substitution: O(B^2), cheap
        T_inv = torch.linalg.solve_triangular(
            torch.eye(B) + L_mat, D, upper=False
        )

        # Boundary-adjusted Q and K (Eq. in Section 3.3)
        # Tensor-core matmuls
        Q_tilde = Q_b[i] - torch.tril(Q_b[i] @ W_b[i].T) @ T_inv @ W_b[i]
        K_tilde = K_b[i] - (T_inv @ torch.tril(W_b[i] @ K_b[i].T, -1)).T @ W_b[i]

        # Intra-block attention (causal, with Householder correction)
        A_intra = torch.tril(Q_b[i] @ K_b[i].T) \
                  - torch.tril(Q_b[i] @ W_b[i].T) @ T_inv \
                    @ torch.tril(W_b[i] @ K_b[i].T, -1)

        # === CROSS-BLOCK: right-to-left scan ===
        # Load Q_tilde[i] into SRAM
        # For j = i-1, ..., 0 (right-to-left):
        #   Compute logits: A_cross = Q_tilde[i] @ K_tilde[j].T
        #   Accumulate output with online softmax
        #   Update: Q_tilde[i] <- Q_tilde[i] @ P[j].T
        #   where P[j] = W[j]^T T_inv[j] W[j] (block product matrix)

        # Combine intra + cross with online softmax normalization
        O[i] = ...  # FlashAttention-style accumulation

    return O.reshape(L, d)


# Efficient inference: in-place KV cache update
def path_kv_cache_update(K_cache, w_t, beta_t):
    """
    Update all cached keys when a new token arrives.
    K_cache: (t-1, d) - previously cached keys
    w_t: (d,) - current Householder direction (normalized)
    beta_t: scalar in (0, 2)
    """
    # Rank-1 update: k_i <- (I - beta * w w^T) k_i for all i
    # Vectorized: K_cache -= beta * (K_cache @ w).unsqueeze(-1) * w
    proj = K_cache @ w_t           # (t-1,) — dot products
    K_cache -= beta_t * proj.unsqueeze(-1) * w_t.unsqueeze(0)
    return K_cache
```

**Implementation available at:** https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/path_attn

**GPU efficiency notes:**
- All dominant operations (logit computation, UT transform matmuls) map to tensor cores
- Forward substitution for $\mathbf{T}^{-1}$ is $O(B^2)$, negligible when $B \leq 128$
- Blockwise structure preserves FlashAttention's IO-awareness (SRAM tiling)
- Right-to-left cross-block scan adds one $d \times d$ matmul per block traversal (for query update via $\mathbf{P}_{[j]}$)

## References

- Yang, S., Shen, Y., Wen, K., Tan, S., Mishra, M., Ren, L., Panda, R., & Kim, Y. (2026). PaTH Attention: Position Encoding via Accumulating Householder Transformations. NeurIPS 2025. arXiv:2505.16381.
- Schlag, I., Irie, K., & Schmidhuber, J. (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. ICLR 2025. arXiv:2502.10297.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. arXiv:2406.06484.
- Joffrain, T., Low, T. M., Quintana-Ortí, E. S., van de Geijn, R. A., & Van Zee, F. G. (2006). Accumulating Householder Transformations, Revisited. ACM Trans. Math. Softw., 32(2), 169–179.
- Lin, Z., Nikishin, E., He, X., & Courville, A. (2025). Forgetting Transformer: Softmax Attention with a Forget Gate. ICLR 2025.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. SIAM J. Sci. Stat. Comput., 8(1), 2–13.

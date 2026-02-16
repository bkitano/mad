# 177: GLA Secondary Chunking — Log-Space Gating for Tensor Core Chunkwise Attention

**Category**: kernel
**Gain type**: efficiency
**Source**: Yang, Wang, Shen, Panda & Kim (2024) — ICML 2024 (arXiv:2312.06635)
**Paper**: [papers/gla-hardware-efficient-training.pdf]
**Documented**: 2026-02-15

## Description

Gated Linear Attention (GLA) introduces a **data-dependent 2D forget gate** $G_t \in (0,1)^{d_k \times d_v}$ into the linear recurrence, parameterized as a low-rank outer product $G_t = \alpha_t^\top \mathbf{1}$ where $\alpha_t \in (0,1)^{d_k}$ is input-dependent. This creates the recurrence:

$$
S_t = \text{Diag}(\alpha_t) \, S_{t-1} + k_t^\top v_t
$$

The chunkwise-parallel form of GLA computes the intra-chunk attention-like matrix $P$ via cumulative products of gate values. However, the cumulative product $b_t = \prod_{j=1}^{t} \alpha_j$ can explode or vanish, forcing computation in **log space**:

$$
P_{ij} = \sum_{k=1}^{d} Q_{ik} K_{jk} \exp(\log B_{ik} - \log B_{jk}), \quad i \geq j
$$

This log-space computation **cannot use standard half-precision matmuls** because it involves element-wise exponentials interleaved with dot products. Standard tensor cores expect pure matrix multiplications, not element-wise nonlinearities.

**The secondary chunking trick** solves this by introducing a **second level of chunking within each chunk**: each primary chunk of size $C$ is subdivided into sub-chunks of size $c$. The key insight is that **inter-sub-chunk interactions** can be expressed as standard matmuls (half-precision, tensor-core friendly), while only **intra-sub-chunk interactions** (the $c \times c$ pink tiles) require log-space computation in full precision:

- **Inter-sub-chunk tiles** (orange in Figure 3): The gate factors between sub-chunks can be factored out as diagonal rescalings applied to standard Q and K, yielding $P_{[i][j]} = (\tilde{Q}_{[i]} \odot \Lambda_{[i]}) (\tilde{K}_{[j]} \odot \Gamma_{[j]} \odot \frac{b_{jC}}{b_{(j+1)C}})^\top \odot M$, which is a standard matmul after absorbing the gate scalings.
- **Intra-sub-chunk tiles** (pink): Only these $c \times c$ blocks require the expensive log-space Eq. 4, computed in full precision.

Since sub-chunk size $c \ll C$, the vast majority of FLOPs are in the tensor-core-friendly inter-sub-chunk matmuls.

## Mathematical Form

**GLA Recurrence:**

$$
S_t = (\alpha_t^\top \mathbf{1}) \odot S_{t-1} + k_t^\top v_t = \text{Diag}(\alpha_t) \, S_{t-1} + k_t^\top v_t
$$

where $\alpha_t = \sigma((x_t W_{\alpha_1} W_{\alpha_2} + b_\alpha) / \tau)^{1/\tau} \in (0,1)^{d_k}$, parameterized via low-rank projection with temperature $\tau = 16$.

**Chunkwise form — inter-chunk recurrence:**

$$
S_{[i+1]} = (\gamma_{i+1}^\top \mathbf{1}) \odot S_{[i]} + (K_{[i+1]} \odot \Gamma_{[i+1]})^\top V_{[i+1]}
$$

where:
- $\gamma_{i+1} \in \mathbb{R}^{d_k}$ is the cumulative gate over the full chunk
- $\Gamma_{[i+1]} \in \mathbb{R}^{C \times d_k}$ encodes decay from each position to the chunk boundary

**Intra-chunk parallel output:**

$$
O_{[i+1]}^{\text{inter}} = (Q_{[i+1]} \odot \Lambda_{[i+1]}) \, S_{[i]}
$$

$$
O_{[i+1]}^{\text{intra}} = P \, V_{[i+1]}, \quad \text{where } P \in \mathbb{R}^{C \times C}
$$

$$
O_{[i+1]} = O_{[i+1]}^{\text{inter}} + O_{[i+1]}^{\text{intra}}
$$

**The attention-like matrix $P$ (in log space for stability):**

$$
P_{ij} = \sum_{k=1}^{d_k} Q_{ik} K_{jk} \exp(\log B_{ik} - \log B_{jk}), \quad i \geq j
$$

where $B_{ik} = \prod_{j=1}^{i} \alpha_{jk}$ is the cumulative gate product (computed via $\log B_{ik} = \sum_{j=1}^{i} \log \alpha_{jk}$).

**Secondary chunking decomposition of $P$:**

Divide the $C \times C$ matrix $P$ into $(C/c) \times (C/c)$ blocks of size $c \times c$:

For **inter-sub-chunk** block $(i, j)$ with $i > j$ (sub-chunk $i$ attending to sub-chunk $j$):

$$
P_{[i][j]} = \left(\tilde{Q}_{[i]} \odot \Lambda_{[i]}\right) \left(\tilde{K}_{[j]} \odot \Gamma_{[j]} \odot \frac{b_{jc}}{b_{(j+1)c}}\right)^\top \in \mathbb{R}^{c \times c}
$$

This is a **standard matmul** of shape $(c \times d_k) \times (d_k \times c)$ — fully tensor-core compatible at half-precision.

For **intra-sub-chunk** block $(i, i)$ (diagonal blocks):

$$
P_{[i][i],mn} = \sum_{k=1}^{d_k} Q_{[i],mk} K_{[i],nk} \exp(\log B_{[i],mk} - \log B_{[i],nk}), \quad m \geq n
$$

This must use **full precision** and the log-space formulation. Only $(C/c)$ such blocks exist, each of size $c \times c$.

**Key Definitions:**

- $C$ — primary chunk size (e.g., 64 or 128)
- $c$ — secondary sub-chunk size (e.g., 16, must be a multiple of tensor core tile size)
- $N_s = C/c$ — number of sub-chunks per chunk
- $\alpha_t \in (0,1)^{d_k}$ — per-position, per-head-dim forget gate (data-dependent)
- $B \in (0,1)^{L \times d_k}$ — cumulative gate product matrix
- $\Lambda_{[i]} \in \mathbb{R}^{c \times d_k}$ — gate encoding from chunk start to each position within sub-chunk $i$
- $\Gamma_{[i]} \in \mathbb{R}^{c \times d_k}$ — gate encoding from each position to sub-chunk boundary

## Complexity

**FLOPs decomposition within a primary chunk:**

| Component | FLOPs | Precision | Tensor Cores |
|-----------|-------|-----------|-------------|
| Inter-sub-chunk matmuls ($P_{[i][j]}$) | $O(N_s^2 \cdot c^2 \cdot d_k)$ = $O(C^2 d_k)$ | FP16/BF16 | Yes |
| Intra-sub-chunk (diagonal, log-space) | $O(N_s \cdot c^2 \cdot d_k)$ = $O(C \cdot c \cdot d_k)$ | FP32 | No |
| Inter-chunk state update | $O(C \cdot d_k \cdot d_v)$ | FP16/BF16 | Yes |
| Inter-chunk output $O^{\text{inter}}$ | $O(C \cdot d_k \cdot d_v)$ | FP16/BF16 | Yes |

**Ratio of tensor-core FLOPs:** The inter-sub-chunk matmuls dominate at $O(C^2 d_k)$ vs. $O(Ccd_k)$ for intra-sub-chunk. With $C = 128, c = 16$: ratio is $C/c = 8$, so $\sim$87.5% of intra-chunk FLOPs use tensor cores.

**Comparison to alternatives:**

| Method | Tensor core usage | State materialization | IO cost |
|--------|------------------|----------------------|---------|
| Recurrent form | None (element-wise) | All timesteps in HBM | $O(L \cdot d_k \cdot d_v)$ |
| Parallel scan | None (state per step) | All timesteps in HBM | $O(L \cdot d_k \cdot d_v)$ |
| Chunkwise (no secondary) | Partial | Per-chunk in SRAM | $O(N_c \cdot C \cdot (d_k + d_v))$ |
| **GLA secondary chunking** | **87%+** | Per-chunk in SRAM | $O(N_c \cdot C \cdot (d_k + d_v))$ |

**Wall-clock throughput (H100, 1.3B model, sequence length 2048, batch 8):**

| Model | Throughput (Ktok/s) | Memory (GB) |
|-------|--------------------|----|
| Transformer++ (FlashAttention-2) | ~35 | ~8 |
| Mamba | ~30 | ~8 |
| **GLA (FlashLinearAttention)** | **~45** | **~6** |

GLA achieves higher throughput than both Transformer++ and Mamba, with lower memory, due to efficient tensor core utilization via secondary chunking.

**Memory — materialization vs. recomputation:**

| Strategy | Memory overhead | Speed |
|----------|---------------|-------|
| Materialization (store all $S_{[n]}$) | $+10\text{--}20\%$ | Fastest (parallel bwd) |
| Recomputation (recompute $S_{[n]}$ in bwd) | Baseline | $\sim$2% slower |

The paper defaults to recomputation: discard $S_{[n]}$ after forward, recompute during backward. The 2% runtime overhead is negligible vs. the 10-20% memory savings.

## Applicability

- **GLA Transformer (primary):** The paper's main application. GLA at 340M/1.3B parameters matches or exceeds Transformer++, RetNet, and Mamba on language modeling benchmarks. Especially strong on recall-intensive tasks (FDA, SWDE, SQUAD) due to its larger state dimension.

- **All linear RNNs with data-dependent matrix gates:** The secondary chunking technique applies whenever the gate is a matrix (not just scalar). This includes: DFW (Schlag et al., 2021), GateLoop (Katsch, 2023), HGRN/HGRN-2, RWKV-6, and any model with $G_t = \alpha_t^\top \beta_t$ parameterization. These are listed in Table 1 of the paper.

- **mLSTM with exponential gates:** mLSTM's exponential input gate creates similar numerical stability issues. The log-space secondary chunking approach can be adapted, though the TFLA paper (trick 158) notes complications with max-state tracking for exponential gates.

- **Complementary to TFLA (trick 158):** TFLA adds a second level of tiling over the *sequence dimension* to decouple chunk size from SRAM. GLA's secondary chunking adds a second level over the *attention matrix* to enable tensor cores despite log-space gates. For a model like mLSTM with both data-dependent gates and large chunks, both tricks could be applied.

- **FlashLinearAttention library:** Implemented in the FLA library (https://github.com/sustcsonglin/flash-linear-attention) as fused Triton kernels, serving as the basis for GLA, RetNet, and other linear attention variants.

## Limitations

- **Low-rank gate parameterization:** GLA uses $G_t = \alpha_t^\top \mathbf{1}$ (rank-1 outer product), which is less expressive than a full $d_k \times d_v$ gate matrix. The paper notes marginal improvement from $\alpha_t^\top \beta_t$ (rank-1 with two vectors) and opts for the simpler form.

- **Intra-sub-chunk still uses full precision:** The diagonal $c \times c$ blocks cannot use tensor cores and require FP32 element-wise operations in log space. With $c = 16$, these blocks are small, but they break the pure-matmul kernel pattern.

- **Fixed gate parameterization:** The low-rank bottleneck $\alpha_t = \sigma(x_t W_{\alpha_1} W_{\alpha_2} / \tau)^{1/\tau}$ with $W_{\alpha_1} \in \mathbb{R}^{d \times 16}, W_{\alpha_2} \in \mathbb{R}^{16 \times d_k}$ adds two small extra projections. The temperature $\tau = 16$ encourages slow forgetting.

- **Bounded memory capacity:** Unlike softmax attention's unbounded KV cache, GLA's $d_k \times d_v$ state has fixed capacity. For recall-intensive tasks, this limits performance. The paper mitigates this by using 4 heads with $d_k = d/2, d_v = d$, but it still lags softmax on some recall benchmarks.

- **Backward pass has 3-pass structure:** The backward with materialization requires: (1) sequential forward to store $S_{[n]}$, (2) sequential backward to accumulate $dS$, then (3) parallel gradient computation. The sequential passes are a bottleneck at large chunk counts.

## Implementation Notes

```python
# GLA secondary chunking — core idea from Listing 1 and Alg. 3/5
# Shows how inter-sub-chunk matmuls use tensor cores while
# intra-sub-chunk uses log-space full precision

def gla_chunkwise_forward(Q, K, V, alpha, C, c):
    """
    Q, K: (L, d_k), V: (L, d_v), alpha: (L, d_k) — log forget gates
    C: primary chunk size, c: secondary sub-chunk size
    """
    L, d_k = Q.shape
    d_v = V.shape[1]
    N = L // C  # number of primary chunks

    # Precompute cumulative log-gates B[i] = cumsum(log(alpha))
    log_alpha = torch.log(alpha)  # (L, d_k)

    S = torch.zeros(d_k, d_v)  # inter-chunk state
    O = torch.empty_like(V)

    for chunk in range(N):
        r = slice(chunk * C, (chunk + 1) * C)
        bq, bk, bv = Q[r], K[r], V[r]
        ba = log_alpha[r]  # (C, d_k)

        # Cumulative log-gate within chunk
        bb = torch.cumsum(ba, dim=0)  # (C, d_k)

        # === Inter-chunk contribution (tensor core matmul) ===
        # Q * exp(cumulative_gate) @ S
        Lambda = torch.exp(bb)  # (C, d_k)
        o_inter = (bq * Lambda) @ S  # (C, d_k) @ (d_k, d_v) = (C, d_v)

        # === Intra-chunk: secondary chunking ===
        o_intra = torch.zeros(C, d_v)
        N_s = C // c  # number of sub-chunks

        for i in range(N_s):  # query sub-chunk
            qi = bq[i*c:(i+1)*c]      # (c, d_k)
            bi = bb[i*c:(i+1)*c]      # (c, d_k)

            for j in range(i + 1):  # key sub-chunk (causal: j <= i)
                kj = bk[j*c:(j+1)*c]  # (c, d_k)
                vj = bv[j*c:(j+1)*c]  # (c, d_v)
                bj = bb[j*c:(j+1)*c]  # (c, d_k)

                if i == j:
                    # INTRA-sub-chunk: log-space, full precision (no tensor core)
                    p = torch.zeros(c, c)
                    for m in range(c):
                        for n in range(m + 1):
                            p[m, n] = (qi[m] * kj[n] *
                                      torch.exp(bi[m] - bj[n])).sum()
                    o_intra[i*c:(i+1)*c] += p @ vj
                else:
                    # INTER-sub-chunk: standard matmul, FP16 TENSOR CORE!
                    # Factor gate scalings into Q and K
                    q_scaled = qi * torch.exp(bi)          # (c, d_k)
                    k_scaled = kj * torch.exp(-bj)         # (c, d_k)
                    p = q_scaled @ k_scaled.T               # (c, c) — TENSOR CORE
                    o_intra[i*c:(i+1)*c] += p @ vj          # (c, d_v) — TENSOR CORE

        O[r] = o_inter + o_intra

        # Update inter-chunk state
        Gamma = torch.exp(bb[-1:] - bb)  # decay to chunk boundary
        gamma = torch.exp(bb[-1])         # total chunk decay
        S = (gamma.unsqueeze(1) * S) + (bk * Gamma).T @ bv

    return O

# Key GPU properties:
# 1. Inter-sub-chunk matmuls (i != j): (c x d_k) @ (d_k x c) — WGMMA/MMA
# 2. Intra-sub-chunk (i == j): element-wise in FP32, only c x c = 16x16
# 3. With C=128, c=16: 8x8=64 blocks total, 8 diagonal (FP32), 28 lower-tri (TC)
#    → 77% of blocks use tensor cores, but blocks further from diagonal
#    are computed first, so effective TC utilization is even higher
# 4. Gate gradient dlog(alpha_t) = q_t ⊙ dq_t - k_t ⊙ dk_t (closed form!)
#    No need to materialize d x d state for gate gradients
```

**Closed-form gate gradient (critical for memory efficiency):**

Prior work (Mao, 2022) claimed that computing $d\alpha_t$ requires materializing the full $d_k \times d_v$ state $S_t$ in HBM. The GLA paper derives a **closed-form** gradient that avoids this:

$$
d\log b_t = k_t \odot dk_t - q_t \odot dq_t
$$

$$
d\log \alpha_t = \sum_{t \leq i \leq L} d\log b_i
$$

This is simply a suffix sum of element-wise products — computed as a reverse cumulative sum, which is $O(L \cdot d_k)$ with no extra memory. This avoids the $O(L \cdot d_k \cdot d_v)$ cost of materializing all hidden states.

## References

- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- Dao, T. & Gu, A. (2024). Transformers are SSMs. ICML 2024.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention. NeurIPS 2025. arXiv:2503.14376.
- Hua, W., Dai, Z., Liu, H., & Le, Q. (2022). Transformer Quality in Linear Time. ICML 2022.
- Sun, Y. et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621.
- Katsch, T. (2023). GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling. arXiv:2311.01927.
- Mao, H. H. (2022). Fine-Tuning Pre-Trained Transformers into Decaying Fast Weights. EMNLP 2022.

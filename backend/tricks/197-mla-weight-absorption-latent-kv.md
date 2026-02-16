# 197: MLA Weight Absorption — Latent KV Attention Without Decompression

**Category**: efficiency
**Gain type**: efficiency
**Source**: DeepSeek-AI — "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (arXiv 2405.04434, Jun 2024)
**Paper**: [papers/deepseek-v2-mla.pdf]
**Documented**: 2026-02-16

## Description

Multi-Head Attention (MHA) in standard Transformers caches separate key and value vectors for each head at each position during autoregressive inference, resulting in KV cache of size $2 n_h d_h l$ elements per token (where $n_h$ is the number of heads, $d_h$ the head dimension, $l$ the number of layers). This becomes the primary memory bottleneck for long-context and high-throughput inference.

**Multi-Head Latent Attention (MLA)** addresses this by jointly compressing keys and values into a single low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ where $d_c \ll n_h d_h$. The naive implementation would decompress $\mathbf{c}_t^{KV}$ back to full keys $\mathbf{k}_t^C$ and values $\mathbf{v}_t^C$ via up-projection matrices $W^{UK}$ and $W^{UV}$ before computing attention. This decompression step adds both compute and memory overhead during inference.

The **weight absorption trick** eliminates this decompression entirely. By exploiting the associativity of matrix multiplication, the up-projection matrices $W^{UK}$ and $W^{UV}$ can be **absorbed** (pre-multiplied) into the query projection $W^{UQ}$ and output projection $W^O$ respectively. The attention computation then operates directly on the compressed latent vectors $\mathbf{c}_t^{KV}$, never materializing the full-size keys or values.

This yields:
- **93.3% KV cache reduction** compared to standard MHA (from $2 n_h d_h$ to $d_c + d_h^R$ per token per layer)
- **5.76× maximum generation throughput** compared to DeepSeek 67B dense model
- **Better quality than MHA** — MLA outperforms MHA on hard benchmarks even with dramatically smaller cache
- **Equivalent to GQA with only 2.25 groups** in cache size, but with MHA-level or better quality

The trick is now the default in SGLang and other inference engines serving DeepSeek models.

## Mathematical Form

**Standard MHA (baseline):**

$$
\mathbf{q}_t = W^Q \mathbf{h}_t, \quad \mathbf{k}_t = W^K \mathbf{h}_t, \quad \mathbf{v}_t = W^V \mathbf{h}_t
$$

Split into $n_h$ heads: $\mathbf{q}_{t,i}, \mathbf{k}_{t,i}, \mathbf{v}_{t,i} \in \mathbb{R}^{d_h}$

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h}}\right) \mathbf{v}_{j,i}
$$

KV cache per token: $2 n_h d_h$ elements. For DeepSeek-V2 with $n_h = 128$, $d_h = 128$: **32,768 elements/token/layer**.

**MLA — Low-Rank Joint KV Compression:**

**Down-projection** (compress to latent):

$$
\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
$$

**Up-projection** (decompress to keys and values):

$$
\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h n_h}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h n_h}
$$

where $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the down-projection, $W^{UK} \in \mathbb{R}^{d_h n_h \times d_c}$ and $W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ are up-projections.

KV cache per token: only $d_c$ elements for the compressed content (plus $d_h^R$ for decoupled RoPE keys). For DeepSeek-V2 with $d_c = 512$, $d_h^R = 64$: **576 elements/token/layer** (vs. 32,768 for MHA).

**The Weight Absorption Trick:**

The naive attention score computation for head $i$ requires decompressing $\mathbf{c}_t^{KV}$:

$$
\text{score}_{t,j,i} = \frac{(\mathbf{q}_{t,i}^C)^T \, \underbrace{W_i^{UK} \, \mathbf{c}_j^{KV}}_{\mathbf{k}_{j,i}^C}}{\sqrt{d_h + d_h^R}}
$$

where $W_i^{UK} \in \mathbb{R}^{d_h \times d_c}$ is the per-head slice of the up-projection.

**Key observation:** By associativity of matrix multiplication, we can absorb $W_i^{UK}$ into the query:

$$
(\mathbf{q}_{t,i}^C)^T W_i^{UK} \, \mathbf{c}_j^{KV} = \underbrace{(W_i^{UK,T} \, \mathbf{q}_{t,i}^C)^T}_{\tilde{\mathbf{q}}_{t,i}^C \in \mathbb{R}^{d_c}} \mathbf{c}_j^{KV}
$$

Or equivalently, absorb $W^{UK}$ into $W^{UQ}$:

$$
\tilde{W}^Q = W^{UK,T} W^{UQ} \in \mathbb{R}^{d_c \times d_c'}
$$

Now the attention score is computed directly between the absorbed query $\tilde{\mathbf{q}}_{t,i}^C \in \mathbb{R}^{d_c}$ and the cached latent $\mathbf{c}_j^{KV} \in \mathbb{R}^{d_c}$ — **no decompression of $\mathbf{c}_j^{KV}$ needed**.

Similarly, for the output computation:

$$
\mathbf{o}_{t,i} = \sum_j \alpha_{t,j,i} \, \underbrace{W_i^{UV} \, \mathbf{c}_j^{KV}}_{\mathbf{v}_{j,i}^C}
$$

By absorbing $W^{UV}$ into $W^O$:

$$
\mathbf{u}_t = W^O \left[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}\right] = W^O \, \text{diag}(W_1^{UV}, \ldots, W_{n_h}^{UV}) \left[\sum_j \alpha_j \mathbf{c}_j^{KV}; \ldots\right]
$$

$$
\tilde{W}^O = W^O \cdot \text{BlockDiag}(W_1^{UV}, \ldots, W_{n_h}^{UV})
$$

The absorbed output projection $\tilde{W}^O$ operates directly on the weighted sums of latent vectors.

**Decoupled RoPE handling:**

RoPE (Rotary Position Embedding) is position-sensitive and cannot be absorbed because it's applied after projection and before the attention score. MLA uses a **decoupled** RoPE strategy with additional small query/key projections:

$$
\mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q) \in \mathbb{R}^{d_h^R n_h}, \quad \mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_h^R}
$$

The full per-head query and key are:

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \quad \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]
$$

The RoPE key $\mathbf{k}_t^R$ is shared across all heads (like multi-query attention) and is small ($d_h^R = 64$), adding minimal cache overhead.

**Full inference computation (with absorption):**

$$
\text{score}_{t,j,i} = \frac{\tilde{\mathbf{q}}_{t,i}^{C,T} \, \mathbf{c}_j^{KV} + \mathbf{q}_{t,i}^{R,T} \, \mathbf{k}_j^R}{\sqrt{d_h + d_h^R}}
$$

$$
\mathbf{o}_{t,i} = \sum_{j=1}^t \text{Softmax}_j(\text{score}_{t,j,i}) \, \mathbf{c}_j^{KV}
$$

$$
\mathbf{u}_t = \tilde{W}^O \, [\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]
$$

## Complexity

| Metric | MHA | GQA ($n_g$ groups) | MQA | MLA (with absorption) |
|--------|-----|-----|-----|----------------------|
| KV cache per token | $2 n_h d_h$ | $2 n_g d_h$ | $2 d_h$ | $d_c + d_h^R$ |
| For DeepSeek-V2 | 32,768 | varies | 256 | **576** |
| Quality vs MHA | baseline | moderate | weak | **stronger** |

**KV cache reduction:** $d_c + d_h^R = 512 + 64 = 576$ vs. $2 \times 128 \times 128 = 32{,}768$ for MHA → **93.3% reduction**.

**Inference compute with absorption:**

| Operation | Without absorption | With absorption |
|-----------|-------------------|-----------------|
| Score computation | $O(n_h d_h)$ per (query, key) pair after decompressing $\mathbf{c}^{KV}$ via $W^{UK} \in \mathbb{R}^{n_h d_h \times d_c}$ | $O(d_c)$ per (query, key) pair — directly dot product latent vectors |
| Value aggregation | Decompress via $W^{UV}$, then weight | Weight latent vectors directly, apply absorbed $\tilde{W}^O$ |
| Memory bandwidth | Load $n_h d_h$ per cached token | Load $d_c + d_h^R$ per cached token |

**Memory bandwidth saving during generation:** The inference bottleneck is loading KV cache from HBM. With absorption, each token loads $576 \times 2 = 1{,}152$ bytes (FP16) instead of $32{,}768 \times 2 = 65{,}536$ bytes — **57× less HBM bandwidth** per cached token accessed.

**Throughput:** DeepSeek-V2 achieves **5.76× maximum generation throughput** (tokens/sec) compared to DeepSeek 67B dense, primarily due to the dramatically reduced KV cache memory and bandwidth.

## Applicability

- **Autoregressive inference for large Transformers:** The primary use case. MLA with absorption enables long-context generation (128K tokens) with dramatically reduced memory. Deployed in DeepSeek-V2 (236B params, 21B activated) and DeepSeek-V3.

- **High-throughput serving:** Smaller KV cache means larger batch sizes fit in GPU memory, directly increasing serving throughput. The 57× bandwidth reduction per token is critical for memory-bound decode.

- **Any attention variant with low-rank KV structure:** The absorption trick applies whenever keys and values are generated via low-rank projections from a shared latent. This pattern could be combined with linear attention (where the "KV cache" is a matrix state) — the down-projection would compress the state update.

- **Chunkwise attention (TFLA connection):** For TFLA-style chunkwise parallel attention, the inter-chunk state $C_k \in \mathbb{R}^{d_q \times d_v}$ is analogous to a matrix-valued KV cache. If $d_q$ and $d_v$ are large, a low-rank compression of $C_k$ with weight absorption could reduce the HBM cost of materializing chunk boundary states.

- **Training:** MLA is also used during training (with low-rank query compression for activation memory reduction). The absorption trick is specific to inference (where you want to avoid decompressing cached latents), but the low-rank structure benefits training memory too.

## Limitations

- **RoPE incompatibility requires decoupling:** The core limitation that motivates the decoupled RoPE design. Because RoPE applies a position-dependent rotation $R_t$ between the projection and the dot product, $W^{UK}$ cannot be absorbed when RoPE is applied to the compressed keys: $\mathbf{q}^T R_t W^{UK} \mathbf{c}^{KV} \neq (W^{UK,T} R_t^T \mathbf{q})^T \mathbf{c}^{KV}$ unless $R_t$ commutes with $W^{UK}$ (it doesn't). The decoupled RoPE keys $\mathbf{k}_t^R$ add $d_h^R$ elements per token to the cache.

- **Absorbed weight matrices are larger:** $\tilde{W}^Q$ has dimensions $d_c \times d_c'$ per head (compared to $d_h \times d$ for standard $W^Q$). The one-time cost of the absorbed projection at each generation step is a GEMM of size $(d_c \times d_c') \times (d_c' \times 1)$ per head — slightly larger than standard query projection, but this is negligible compared to the bandwidth savings from smaller KV cache.

- **Training-inference asymmetry:** During training, the model uses the decomposed formulation (separate down-projection, up-projection, and RoPE). At inference time, the absorbed formulation is used. This requires a weight transformation step when converting from training to inference checkpoints.

- **Not directly applicable to linear attention states:** Linear attention's state $C_t = \sum_j k_j v_j^T$ is not generated by a single low-rank projection from a latent vector — it's accumulated over time. Absorption applies to the projection step, not the accumulation step.

- **Head dimension mismatch:** The absorbed query $\tilde{\mathbf{q}}_{t,i}^C \in \mathbb{R}^{d_c}$ has dimension $d_c$ (e.g., 512) rather than $d_h$ (e.g., 128). The attention score dot product is over $d_c + d_h^R = 576$ dimensions, which is larger than standard $d_h = 128$. This increases per-score compute but is overwhelmingly compensated by the cache savings.

## Implementation Notes

```python
# MLA Weight Absorption: Training vs. Inference formulation

# === TRAINING (standard formulation) ===
class MLATraining(nn.Module):
    def __init__(self, d, n_h, d_h, d_c, d_c_q, d_h_R):
        # Down-projections (compress)
        self.W_DKV = nn.Linear(d, d_c, bias=False)      # h -> c_KV
        self.W_DQ = nn.Linear(d, d_c_q, bias=False)      # h -> c_Q
        # Up-projections (decompress)
        self.W_UK = nn.Linear(d_c, n_h * d_h, bias=False) # c_KV -> keys
        self.W_UV = nn.Linear(d_c, n_h * d_h, bias=False) # c_KV -> values
        self.W_UQ = nn.Linear(d_c_q, n_h * d_h, bias=False) # c_Q -> queries
        # Decoupled RoPE projections
        self.W_QR = nn.Linear(d_c_q, n_h * d_h_R, bias=False)
        self.W_KR = nn.Linear(d, d_h_R, bias=False)       # shared key
        # Output
        self.W_O = nn.Linear(n_h * d_h, d, bias=False)

    def forward(self, h):
        # Compress
        c_KV = self.W_DKV(h)       # (T, d_c)
        c_Q = self.W_DQ(h)         # (T, d_c_q)
        # Decompress
        k_C = self.W_UK(c_KV)      # (T, n_h * d_h)
        v_C = self.W_UV(c_KV)      # (T, n_h * d_h)
        q_C = self.W_UQ(c_Q)       # (T, n_h * d_h)
        # RoPE (decoupled)
        q_R = apply_rope(self.W_QR(c_Q))  # (T, n_h * d_h_R)
        k_R = apply_rope(self.W_KR(h))    # (T, d_h_R)
        # Concatenate and compute attention
        q = concat(q_C, q_R, dim=-1)      # per head: d_h + d_h_R
        k = concat(k_C, k_R, dim=-1)      # per head: d_h + d_h_R
        # ... standard multi-head attention with q, k, v_C ...

# === INFERENCE (with weight absorption) ===
class MLAInference(nn.Module):
    def __init__(self, training_model):
        # Absorb W_UK into W_UQ: query operates in latent space
        # W_tilde_Q = W_UK^T @ W_UQ  (per head)
        # Shape: (d_c, d_h) @ (d_h, d_c_q) -> (d_c, d_c_q) per head
        self.W_absorbed_Q = absorb_weights(
            training_model.W_UK, training_model.W_UQ)

        # Absorb W_UV into W_O: output operates on latent sums
        # W_tilde_O = W_O @ BlockDiag(W_UV_1, ..., W_UV_nh)
        self.W_absorbed_O = absorb_weights(
            training_model.W_O, training_model.W_UV)

        # RoPE projections unchanged
        self.W_QR = training_model.W_QR
        self.W_KR = training_model.W_KR
        self.W_DQ = training_model.W_DQ

    def generate_step(self, h_t, kv_cache):
        """
        Single autoregressive step.
        kv_cache stores (c_KV, k_R) per position — NOT full keys/values.
        """
        # Compress query
        c_Q = self.W_DQ(h_t)                     # (1, d_c_q)

        # Absorbed query: operates in d_c-dimensional latent space
        q_absorbed = self.W_absorbed_Q(c_Q)       # (1, n_h, d_c)

        # RoPE query (decoupled, unchanged)
        q_R = apply_rope(self.W_QR(c_Q))           # (1, n_h, d_h_R)

        # New KV: just compress and cache the latent + RoPE key
        c_KV_new = self.W_DKV(h_t)                 # (1, d_c) — CACHED
        k_R_new = apply_rope(self.W_KR(h_t))       # (1, d_h_R) — CACHED

        # Attention scores directly on latents (NO decompression!)
        # score = q_absorbed^T @ c_KV + q_R^T @ k_R
        scores = (q_absorbed @ kv_cache.c_KV.T      # (1, n_h, T) content
                + q_R @ kv_cache.k_R.T)              # (1, n_h, T) position

        attn_weights = softmax(scores / sqrt(d_h + d_h_R))

        # Output: weighted sum of latents, then absorbed output projection
        o = attn_weights @ kv_cache.c_KV  # (1, n_h, d_c) — latent space
        u = self.W_absorbed_O(o)           # (1, d) — NO W_UV decompression
        return u
```

**GPU efficiency analysis:**

1. **Memory bandwidth is the bottleneck for decode:** During autoregressive generation, each step loads the entire KV cache. With absorption, each cached token requires $d_c + d_h^R = 576$ elements instead of $2 n_h d_h = 32{,}768$ — a **57× reduction in HBM bandwidth** per token, directly translating to higher generation throughput.

2. **All operations are dense matmuls:** The absorbed query projection, attention score computation, and absorbed output projection are all standard GEMM/GEMV operations mapping to tensor cores.

3. **Compatible with FlashAttention/FlashDecoding:** The attention computation on the latent vectors uses standard dot-product attention with modified head dimension ($d_c + d_h^R$). Existing optimized attention kernels work directly.

4. **Batch size scaling:** The KV cache reduction enables **larger batch sizes** (more concurrent requests) in GPU memory, which is critical for inference serving throughput — the 5.76× throughput gain comes primarily from this.

5. **One-time weight absorption cost:** The absorption $\tilde{W}^Q = W^{UK,T} W^{UQ}$ is computed once at model load time and stored. No runtime overhead.

## References

- DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
- DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150.
- Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023.
- Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing.

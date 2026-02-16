# 254: RAT — Intra-Chunk Recurrence with Inter-Chunk Softmax Attention

**Category**: parallelization
**Gain type**: efficiency
**Source**: Wei, Yadav, Pascanu & Gulcehre (2025). RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling. NeurIPS 2025. arXiv:2507.04416.
**Paper**: papers/rat-chunk-recurrence-attention.pdf
**Documented**: 2026-02-16

## Description

Recurrent models compress all history into a fixed-size state, which degrades memory for long sequences and limits precise information retrieval. Full softmax attention preserves all tokens but costs $O(T^2)$. RAT proposes a clean **intermediate architecture** that gets the best of both: partition the sequence into chunks of size $L$, apply a **lightweight linear recurrence within each chunk** (intra-chunk RNN) to compress local context, then use **softmax attention across chunk-level representations** (inter-chunk attention) for precise long-range retrieval.

The critical parallelization trick is the **separation of concerns**:
1. **Intra-chunk recurrence** is applied independently per chunk, reducing scan depth from $O(\log T)$ to $O(\log L)$. Since chunks are independent, all $C = T/L$ chunks can be processed in parallel.
2. **Inter-chunk attention** operates on only $C$ chunk-level key-value pairs, reducing the attention cost from $O(T^2)$ to $O(T \cdot C) = O(T^2/L)$.

By adjusting $L$, RAT **interpolates continuously between full attention ($L=1$) and pure RNN ($L=T$)**. With $L=16$, the model achieves a $7\times$ training speedup at 100K sequence length and $9\times$ generation speedup at position 4K, while matching full attention quality on short- and long-context benchmarks at 1.3B parameters. The maximum throughput improvement is $10\times$ at 1.3B and $10.8\times$ at 13B (31,170 vs. 3,052 tokens/sec and 5,749 vs. 534 tokens/sec respectively), because the KV cache stores only $C$ chunk-level vectors instead of $T$ token-level vectors.

Crucially, RAT requires **no custom CUDA kernels** — intra-chunk recurrence uses PyTorch's `associative_scan`, and inter-chunk attention uses standard FlashAttention with flex attention for causal masking.

## Mathematical Form

**Intra-Chunk Linear Recurrence:**

Divide sequence of length $T$ into $C$ chunks of size $L$, where $T = C \cdot L$. Index tokens as $(c, l)$ for chunk $c$ and position $l$ within chunk.

$$
\tilde{v}_{c,l} = g_{c,l} \odot \tilde{v}_{c,l-1} + (1 - g_{c,l}) \odot v_{c,l} \quad \text{(value accumulation)}
$$

$$
\tilde{k}_{c,l} = g_{c,l} \odot \tilde{k}_{c,l-1} + (1 - g_{c,l}) \odot k_{c,l} \quad \text{(key accumulation)}
$$

where $g_{c,l} \in \mathbb{R}^D$ is a per-dimension forget gate computed as $g_{c,l} = \sigma(W_g x_{c,l})$, and $\odot$ denotes element-wise multiplication.

**Key Definitions:**

- $v_{c,l}, k_{c,l}, q_{c,l} \in \mathbb{R}^D$ — value, key, query projections of token $(c,l)$
- $g_{c,l} \in \mathbb{R}^D$ — sigmoid forget gate (per-dimension)
- $z_{c,l} \in \mathbb{R}^D$ — sigmoid output gate
- $\tilde{v}_{c,l}, \tilde{k}_{c,l} \in \mathbb{R}^D$ — gated recurrent summaries of value and key within chunk $c$
- $\bar{K}_{:,c-1}, \bar{V}_{:,c-1}$ — stacked chunk-level keys and values from all preceding chunks
- $L$ — chunk size (hyperparameter; $L=16$ recommended)
- $C = T / L$ — number of chunks

**Inter-Chunk Softmax Attention:**

For each query $q_{c,l}$, attend over all preceding chunk-level keys $\bar{K}_{:,c-1}^{\top}$ (from completed chunks) and the current chunk's gated key $\tilde{k}_{c,l}$:

$$
y_{c,l} = f\!\left(\left[q_{c,l} \bar{K}_{:,c-1}^{\top}; q_{c,l} \tilde{k}_{c,l}^{\top}\right]\right) \left[\bar{V}_{:,c-1}; \tilde{v}_{c,l}\right]
$$

where $f(\cdot)$ denotes causal masking followed by softmax normalization. The current chunk's contribution is separated to handle the causal structure properly.

**Output Gating:**

$$
\mathbf{y}_{c,l} = z_{c,l} \odot y_{c,l}
$$

**Causal Masking (Online Softmax):**

To handle the fact that within a chunk, each token's gated key $\tilde{k}_{c,l}$ varies (due to causal gating), the inter-chunk attention is split into two terms computed separately with adjusted softmax denominators:

$$
y_{c,l} = \frac{e^{\max_1} \cdot f(q_{c,l} \bar{K}_{:,c-1}^{\top}) \bar{V}_{:,c-1} + e^{\max_2} \cdot f(q_{c,l} \tilde{k}_{c,l}^{\top}) \tilde{v}_{c,l}}{e^{\max_1} \cdot \text{denom}_1 + e^{\max_2} \cdot \text{denom}_2}
$$

The first term (attention over completed chunks) is batch-parallelizable using standard FlashAttention; the second term (current chunk) is a simple einsum.

## Complexity

| Operation | Full Attention | Pure RNN | RAT (chunk $L$) |
|-----------|---------------|----------|-----------------|
| Training FLOPs/token | $O(T \cdot D)$ | $O(D)$ | $O(C \cdot D) = O(T/L \cdot D)$ |
| KV cache (inference) | $O(T \cdot D)$ | $O(D)$ | $O(C \cdot D) = O(T/L \cdot D)$ |
| Parallel depth (training) | $O(1)$ | $O(\log T)$ via scan | $O(\log L)$ for scan |
| Generation latency | $O(T)$ | $O(1)$ | $O(C)$ |

**Memory (KV cache):** $O(T \cdot D / L)$ — an $L\times$ reduction. For $L=16$, this is a $16\times$ reduction, which enables much higher maximum throughput before hitting memory limits.

**Parameter count:** $\sim 4D^2$ per RAT block ($D^2$ output projection, $3D^2$ for Q/K/V in attention, $2D^2$ for gates — using shared Q/K projections and low-rank gates for efficiency). Comparable to $\sim 4D^2$ for a standard attention block.

## Applicability

- **LLM pretraining**: Validated at 1.3B scale on 100B tokens from FineWeb-Edu. RAT(L=16) achieves 7.67 validation PPL vs. 7.61 for full attention, with 10$\times$ higher maximum throughput
- **Long-context understanding**: Outperforms attention on several LongBench tasks (QA, summarization) due to the structural bias from chunk-level compression that is more robust to distractor information
- **Retrieval tasks**: With $L=16$, RAT matches attention on most needle-in-haystack retrieval tasks (single/multi-key). Performance degrades for UUID-based exact matching at high difficulty
- **Hybrid architectures**: RAT + sliding window attention (SWA) interleaving yields the best overall results — RAT handles long-range efficiently while SWA handles local interactions precisely
- **Context parallelism**: Intra-chunk recurrence is chunk-independent, allowing distribution across GPUs without ring communication for the recurrence. Only the inter-chunk attention requires cross-GPU communication
- **Inference-heavy workloads**: The $L\times$ KV cache reduction is critical for high-throughput serving (31K vs. 3K tokens/sec at 1.3B)

## Limitations

- **Not validated beyond 1.3B**: The paper trains up to 1.3B parameters; 7B/14B scaling results are throughput-only (no accuracy evaluation at those scales)
- **Retrieval ceiling**: For tasks requiring exact token retrieval (UUID matching), chunk-level compression inherently loses precision. $L=64$ significantly underperforms attention on multi-key retrieval
- **Perplexity gap**: RAT(L=16) at 1.3B achieves 7.67 PPL vs. 7.61 for attention (a small but consistent gap). The gap narrows with longer training context but doesn't fully close
- **Length generalization**: Like attention, RAT with RoPE may struggle to generalize beyond training sequence lengths. Using NoPE or chunk-level RoPE partially mitigates this
- **No custom kernel**: While being an advantage for ease of implementation, the lack of a fused kernel means the intra-chunk scan and inter-chunk attention are separate kernel launches. A fused RAT kernel could further improve throughput
- **Chunk size sensitivity**: $L$ is a fixed hyperparameter. Different layers or heads might benefit from different chunk sizes, which is left as future work

## GPU Efficiency Analysis

**Memory Access Pattern**: Intra-chunk recurrence operates on contiguous $(B, L, D)$ tensors per chunk — well-suited for coalesced access. Inter-chunk attention operates on $(B, C, D)$ chunk-level KV pairs, which are much smaller than full $(B, T, D)$ KV matrices. The reduced working set fits better in GPU caches.

**Tensor Core Utilization**: The inter-chunk attention is a standard softmax attention over $C$ chunk-level vectors, directly using FlashAttention's tensor core-optimized kernels. The intra-chunk recurrence is an element-wise gated scan (no tensor cores), but with only $L=16$ steps the scan depth is $\log_2(16) = 4$, making it very fast.

**Parallelism**: All $C$ chunks process their intra-chunk recurrence independently — full SM saturation for $C \geq 100$ (sequence length $\geq 1600$ with $L=16$). The inter-chunk attention parallelizes across batch and heads as usual.

**Practical Throughput (Measured on H100 GH200)**:
- Training: RAT(L=16) is $7\times$ faster than FlashAttention at 100K sequence length
- Generation: $9\times$ faster at position 4K, $10\times$ at longer positions
- Maximum throughput: 31,170 tokens/sec (RAT, 1.3B) vs. 3,052 (Attention, 1.3B) — a $10.2\times$ improvement
- At 13B: 5,749 vs. 534 tokens/sec — a $10.8\times$ improvement

**No Custom Kernels Required**: Implementation uses `torch._associative_scan` for recurrence and `flex_attention` for causal masking in inter-chunk attention. Fully compatible with `torch.compile`.

## Implementation Notes

```python
# RAT: Intra-chunk recurrence + inter-chunk softmax attention
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

class RAT(nn.Module):
    def __init__(self, d_model: int, chunk_size: int = 16):
        super().__init__()
        self.L = chunk_size
        self.d = d_model

        # Shared q/k projection (saves parameters)
        self.W_qk = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Gates (low-rank for efficiency)
        self.W_g = nn.Linear(d_model, d_model, bias=False)  # forget gate
        self.W_z = nn.Linear(d_model, d_model, bias=False)  # output gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        L = self.L
        C = T // L  # number of chunks

        # Project to q, k, v
        qk = self.W_qk(x)
        q, k = qk, qk  # shared projections
        v = self.W_v(x)

        # Compute gates
        g = sigmoid(self.W_g(x))  # forget gate (B, T, D)
        z = sigmoid(self.W_z(x))  # output gate (B, T, D)

        # Reshape into chunks: (B, C, L, D)
        k_chunks = k.view(B, C, L, D)
        v_chunks = v.view(B, C, L, D)
        q_chunks = q.view(B, C, L, D)
        g_chunks = g.view(B, C, L, D)
        z_chunks = z.view(B, C, L, D)

        # --- Intra-chunk recurrence (parallel across chunks) ---
        # Gated EMA: v_tilde[l] = g[l] * v_tilde[l-1] + (1-g[l]) * v[l]
        # This is a linear recurrence, parallelizable via associative scan
        # Each chunk is independent → all C chunks run in parallel
        k_tilde = gated_scan(g_chunks, k_chunks)  # (B, C, L, D)
        v_tilde = gated_scan(g_chunks, v_chunks)  # (B, C, L, D)

        # Extract chunk-level summaries (last position of each chunk)
        K_bar = k_tilde[:, :, -1, :]  # (B, C, D) — chunk-level keys
        V_bar = v_tilde[:, :, -1, :]  # (B, C, D) — chunk-level values

        # --- Inter-chunk softmax attention ---
        # For each query q[c,l], attend to:
        #   1. K_bar[:c-1], V_bar[:c-1] (completed chunks) — standard causal attn
        #   2. k_tilde[c,l], v_tilde[c,l] (current chunk, causally gated)
        # Use online softmax to combine both terms

        # Term 1: attention over completed chunks (use FlashAttention)
        q_flat = q.view(B, C, L, D)
        # This can use flex_attention with block causal mask
        attn_out_cross = flash_chunk_attention(q_flat, K_bar, V_bar)  # (B, C, L, D)

        # Term 2: current chunk self-attention via gated k/v
        attn_out_self = torch.einsum('bcld,bcld->bcl', q_flat, k_tilde)
        attn_out_self = attn_out_self.unsqueeze(-1) * v_tilde  # simplified

        # Combine with online softmax normalization
        y = combine_online_softmax(attn_out_cross, attn_out_self)

        # Output gating
        y = z_chunks * y
        y = y.view(B, T, D)

        return self.W_o(y)


def gated_scan(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Parallel gated scan: y[l] = g[l] * y[l-1] + (1-g[l]) * x[l]
    g, x: (B, C, L, D)
    Runs L-length scans independently for each of B*C*D elements.
    With L=16, scan depth = log2(16) = 4 steps.
    """
    # Rearrange to (B*C*D, L) for batched scalar scan
    a = g          # coefficients
    b = (1 - g) * x  # inputs
    # Associative scan with op: (a1,b1)*(a2,b2) = (a2*a1, a2*b1+b2)
    _, result = torch._associative_scan(
        lambda p, q: (q[0] * p[0], q[0] * p[1] + q[1]),
        (a, b),
        dim=2  # scan over L dimension
    )
    return result
```

## References

- Wei, X., Yadav, A., Pascanu, R. & Gulcehre, C. "RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling." NeurIPS 2025. arXiv:2507.04416.
- Feng, L. et al. "Were RNNs All We Needed?" arXiv:2410.01201, 2024.
- Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality." arXiv:2405.21060, 2024.
- Orvieto, A. et al. "Resurrecting Recurrent Neural Networks for Long Sequences." ICML 2023.
- Yang, S., Kautz, J. & Hatamizadeh, A. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025.
- Hutchins, D. et al. "Block-Recurrent Transformers." NeurIPS 2022.

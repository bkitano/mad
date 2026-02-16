# 251: RetNet Multi-Scale Retention

**Category**: parallelization
**Gain type**: efficiency
**Source**: Sun, Dong, Huang et al., "Retentive Network: A Successor to Transformer for Large Language Models" (2023)
**Paper**: papers/retnet-multi-scale-retention.pdf
**Documented**: 2026-02-16

## Description

Multi-Scale Retention (MSR) is the core mechanism of RetNet that achieves the "impossible triangle" of training parallelism, low-cost $O(1)$ inference, and strong performance simultaneously. The key trick is that retention admits three mathematically equivalent computation forms:

1. **Parallel form**: Computes a causal attention-like matrix with exponential decay masking, enabling full GPU parallelism during training via a single matmul.
2. **Recurrent form**: Updates a fixed-size $d \times d$ state matrix with scalar decay, enabling $O(1)$ per-step inference with no KV cache growth.
3. **Chunkwise recurrent form**: Hybrid that applies the parallel form within fixed-size chunks and the recurrent form across chunks, enabling linear-complexity long-sequence training.

The multi-scale aspect assigns different exponential decay rates $\gamma$ to different heads, allowing some heads to capture short-range patterns (fast decay) and others to capture long-range dependencies (slow decay). GroupNorm is used across heads to handle the different variance scales introduced by this multi-scale design.

The mathematical derivation connects diagonalized linear recurrences (state space models) to attention via complex exponential position embeddings (xPos), removing the softmax bottleneck and enabling the dual recurrent/parallel view.

## Mathematical Form

**Core Operation — Retention:**

Starting from the linear recurrence with diagonalizable state matrix $A = \Lambda(\gamma e^{i\theta})\Lambda^{-1}$:

$$
s_n = A s_{n-1} + K_n^\top v_n, \quad o_n = Q_n s_n
$$

After absorbing $\Lambda$ into projections and simplifying $\gamma$ as a scalar:

$$
o_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m
$$

where $\dagger$ denotes conjugate transpose.

**Parallel Representation (Training):**

$$
Q = (XW_Q) \odot \Theta, \quad K = (XW_K) \odot \bar{\Theta}, \quad V = XW_V
$$

$$
\Theta_n = e^{in\theta}, \quad D_{nm} = \begin{cases} \gamma^{n-m}, & n \geq m \\ 0, & n < m \end{cases}
$$

$$
\text{Retention}(X) = (QK^\top \odot D) V
$$

The matrix $D \in \mathbb{R}^{|x| \times |x|}$ combines causal masking and exponential decay into one element-wise mask applied to the $QK^\top$ matrix, analogous to softmax attention but without the softmax.

**Recurrent Representation (Inference):**

$$
S_n = \gamma S_{n-1} + K_n^\top V_n, \quad S_n \in \mathbb{R}^{d_k \times d_v}
$$

$$
\text{Retention}(X_n) = Q_n S_n
$$

Per-step cost is $O(d_k \cdot d_v)$ with a fixed-size state matrix $S_n$ — no KV cache growth.

**Chunkwise Recurrent Representation (Long-Sequence Training):**

For chunk $[i]$ of size $B$:

$$
Q_{[i]} = Q_{Bi:B(i+1)}, \quad K_{[i]} = K_{Bi:B(i+1)}, \quad V_{[i]} = V_{Bi:B(i+1)}
$$

$$
R_i = K_{[i]}^\top (V_{[i]} \odot \zeta) + \gamma^B R_{i-1}, \quad \zeta_j = \gamma^{B-j-1}
$$

$$
\text{Retention}(X_{[i]}) = \underbrace{(Q_{[i]} K_{[i]}^\top \odot D) V_{[i]}}_{\text{Inner-Chunk (parallel)}} + \underbrace{(Q_{[i]} R_{i-1}) \odot \xi}_{\text{Cross-Chunk (recurrent)}}, \quad \xi_{ij} = \gamma^{j+1}
$$

**Gated Multi-Scale Retention (MSR):**

$$
\gamma = 1 - 2^{-5-\text{arange}(0, h)} \in \mathbb{R}^h
$$

$$
\text{head}_i = \text{Retention}(X, \gamma_i)
$$

$$
Y = \text{GroupNorm}_h(\text{Concat}(\text{head}_1, \ldots, \text{head}_h))
$$

$$
\text{MSR}(X) = (\text{swish}(XW_G) \odot Y) W_O
$$

**Key Definitions:**

- $X \in \mathbb{R}^{|x| \times d_{\text{model}}}$ — input sequence
- $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d}$, $W_V \in \mathbb{R}^{d_{\text{model}} \times 2d}$, $W_G, W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ — learnable projections
- $S_n \in \mathbb{R}^{d_k \times d_v}$ — recurrent state matrix (fixed size, $O(1)$ inference)
- $D \in \mathbb{R}^{|x| \times |x|}$ — causal decay mask combining masking and exponential decay
- $\gamma \in (0, 1)$ — per-head scalar decay rate
- $\theta \in \mathbb{R}^d$ — complex rotation angle (xPos-style position embedding)
- $h$ — number of retention heads
- $B$ — chunk size for chunkwise representation

**Retention Score Normalization:**

Three normalization factors stabilize numerical flow without affecting outputs (due to GroupNorm's scale-invariance):

1. Scale $QK^\top$ by $1/\sqrt{d}$
2. Normalize $D$: $\tilde{D}_{nm} = D_{nm} / \sqrt{\sum_{i=1}^{n} D_{ni}}$
3. Normalize retention scores: $\tilde{R}_{nm} = R_{nm} / \max(|\sum_{i=1}^{n} R_{ni}|, 1)$

## Complexity

| Operation | Transformer | RetNet (Parallel) | RetNet (Chunkwise) | RetNet (Recurrent) |
|-----------|------------|-------------------|--------------------|--------------------|
| Training (per step) | $O(Nd)$ | $O(Nd)$ | $O(N d(B+h))$ | N/A (sequential) |
| Inference (per step) | $O(N)$ (KV cache read) | N/A | N/A | $O(1)$ (fixed state) |
| Memory (long sequence) | $O(N^2)$ | $O(N^2)$ | $O(N)$ | $O(d^2)$ constant |

**Memory:** Parallel form uses $O(N^2)$ like attention (but no softmax). Chunkwise form uses $O(NB)$ where $B$ is chunk size. Recurrent form uses $O(d_k \cdot d_v)$ constant memory.

**Measured throughput (A100-80GB, sequence length 8192):**

| Model Size | Transformer (wps) | Trm+FlashAttn (wps) | RetNet (wps) |
|------------|-------------------|---------------------|--------------|
| 1.3B | 10832.4 | 63965.2 | 73344.8 |
| 2.7B | 5186.0 | 34990.2 | 38921.2 |
| 6.7B | 2754.4 | 16230.1 | 17458.6 |

**Inference (6.7B, 8K sequence):** 8.4x faster decoding, 70% memory reduction, 15.6x lower latency vs Transformer.

## Applicability

- **Drop-in Transformer replacement**: RetNet follows the same stacked block layout (MSR + FFN) as Transformers, making it straightforward to substitute. Scaling behavior is competitive at 1.3B to 6.7B parameters.
- **Long-sequence training**: The chunkwise form enables linear memory complexity for arbitrarily long sequences, unlike quadratic attention.
- **Low-latency inference**: The recurrent form provides $O(1)$ per-step inference with constant memory, eliminating the growing KV cache bottleneck.
- **Batch-insensitive serving**: RetNet's inference latency is nearly constant across batch sizes and input lengths, unlike Transformers where latency grows with both.
- **Foundation for GLA/DeltaNet/RWKV**: The retention mechanism is the basis for subsequent architectures like Gated Linear Attention (GLA), which adds input-dependent gating to the decay factor.

## Limitations

- **Fixed (non-input-dependent) decay**: The scalar $\gamma$ per head is fixed at initialization and does not adapt to input content. GLA and Mamba address this with input-dependent gates, improving expressivity.
- **No softmax — different attention pattern**: Removing softmax means retention scores are not probability-normalized, which can affect the model's ability to form sharp, sparse attention patterns.
- **Parallel form is still $O(N^2)$**: The parallel representation computes the full $QK^\top$ matrix, so it doesn't reduce asymptotic training cost vs attention (though it avoids softmax overhead).
- **Chunkwise implementation not fully optimized**: Paper uses vanilla PyTorch without kernel fusion or FlashAttention-style tiling. Actual GPU utilization could be improved with fused kernels (e.g., FLA library).
- **Complex exponential positions**: Uses complex-valued $e^{i\theta}$ position encoding, adding implementation complexity. In practice, this can be simplified to real-valued cosine/sine components.
- **GroupNorm adds overhead**: Required to handle multi-scale variance differences, but adds a normalization step not present in standard attention.

## Implementation Notes

```python
import torch
import torch.nn.functional as F

def parallel_retention(q, k, v, gamma, head_dim):
    """
    Parallel form of retention — used during training.
    q, k: [B, H, T, d_k] with xPos applied
    v:    [B, H, T, d_v]
    gamma: [H] per-head decay rate
    """
    T = q.shape[2]
    # Build causal decay mask D
    positions = torch.arange(T, device=q.device)
    # D[n,m] = gamma^(n-m) if n >= m, else 0
    D = gamma.view(-1, 1, 1) ** (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(min=0)
    D = D * (positions.unsqueeze(1) >= positions.unsqueeze(0))  # causal mask

    # Retention = (Q K^T ⊙ D) V
    retention = (q @ k.transpose(-1, -2)) * D.unsqueeze(0)  # [B, H, T, T]
    output = retention @ v  # [B, H, T, d_v]
    return output

def recurrent_retention(q, k, v, gamma, past_kv):
    """
    Recurrent form — used during inference.
    q, k: [B, H, 1, d_k]   (single step)
    v:    [B, H, 1, d_v]
    past_kv: [B, H, d_k, d_v]  (fixed-size state)
    """
    # S_n = gamma * S_{n-1} + k^T v
    current_kv = gamma * past_kv + k.transpose(-1, -2) @ v
    # output = q @ S_n
    output = q @ current_kv
    return output, current_kv

def chunkwise_retention(q, k, v, gamma, chunk_size, past_kv):
    """
    Chunkwise form — used for long-sequence training.
    Uses parallel form within chunks, recurrent form across chunks.
    """
    # Inner-chunk: parallel retention within chunk
    inner_output = parallel_retention(q, k, v, gamma, q.shape[-1])

    # Cross-chunk: propagate state from previous chunk
    decay = gamma ** torch.arange(1, chunk_size + 1, device=q.device)
    cross_output = (q @ past_kv) * decay.unsqueeze(-1)

    # Update chunk state for next chunk
    chunk_decay = gamma ** torch.arange(chunk_size - 1, -1, -1, device=q.device)
    current_kv = k.transpose(-1, -2) @ (v * chunk_decay.unsqueeze(-1))
    current_kv = current_kv + (gamma ** chunk_size) * past_kv

    return inner_output + cross_output, current_kv

# Multi-Scale Retention: different gamma per head
# gamma = 1 - 2^{-5-arange(0,h)} gives heads with decay
# rates ranging from ~0.97 (fast decay) to ~1.0 (slow decay)
```

## References

- Sun, Dong, Huang, Ma, Xia, Xue, Wang & Wei. "Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621, 2023.
- Code: https://aka.ms/retnet
- Sun, Dong, Patra et al. "A Length-Extrapolatable Transformer." (xPos) ACL 2022.
- Yang, Wang, Yu et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. (extends retention with input-dependent gating)
- Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality." ICML 2024. (unifies retention and SSMs)

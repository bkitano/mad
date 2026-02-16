# 261: Decay/Gate Factor Absorption into Q/K Projections

**Category**: kernel
**Gain type**: efficiency
**Source**: Multiple — RetNet (Sun et al., 2023), GLA (Yang et al., 2024), Mamba-2/SSD (Dao & Gu, 2024), Lightning Attention-2 (Qin et al., 2024), KDA (He et al., 2025)
**Paper**: papers/gla-hardware-efficient-training.pdf
**Documented**: 2026-02-16

## Description

In gated linear attention and retention-style models, the recurrent state evolves as:

$$
S_t = G_t \odot S_{t-1} + k_t^\top v_t, \quad o_t = q_t S_t
$$

where $G_t$ is a decay/gate factor (scalar, diagonal, or low-rank). When this recurrence is unrolled into a parallel (attention-like) form, the attention matrix $P$ acquires **position-dependent scaling factors** from the cumulative gate products:

$$
P_{ij} = q_i^\top \left(\frac{b_i}{b_j}\right) k_j, \quad i \geq j
$$

where $b_t = \prod_{s=1}^{t} \alpha_s$ is the cumulative gate. Naively, this element-wise ratio $b_i / b_j$ is entangled with the dot product, preventing the use of standard half-precision matrix multiplication on tensor cores.

**The decay absorption trick** factors these cumulative gate terms into modified query and key vectors:

$$
\tilde{Q} = Q \odot \Lambda, \quad \tilde{K} = K \odot \Gamma
$$

where $\Lambda$ and $\Gamma$ encode the decay from the chunk boundary to each position. After absorption, the gated attention becomes a **standard matrix multiplication**:

$$
P = \tilde{Q} \tilde{K}^\top \odot M
$$

where $M$ is a simple causal mask (or causal decay mask for scalar gates). This matmul maps directly to tensor core WMMA/MMA instructions at half precision, converting what was a non-standard element-wise-interleaved computation into a highly optimized GEMM.

This absorption pattern appears independently across virtually all modern chunkwise linear attention implementations, but the specific factorization of $\Lambda$ and $\Gamma$ depends on whether the gate is scalar (RetNet), per-dimension diagonal (GLA), or tied to the key projection (KDA/DPLR).

## Mathematical Form

**General Gated Linear Attention — Chunkwise Parallel Form:**

For chunk $[n]$ of size $C$, positions $nC+1$ to $(n+1)C$:

$$
O_{[n]}^{\text{intra}} = P_{[n]} V_{[n]}, \quad O_{[n]}^{\text{inter}} = \tilde{Q}_{[n]} S_{[n-1]}
$$

**Without absorption** (naive, non-tensor-core):

$$
P_{[n],ij} = \sum_{k=1}^{d_k} Q_{[n],ik} K_{[n],jk} \exp(\log B_{ik} - \log B_{jk}), \quad i \geq j
$$

This requires element-wise `exp()` interleaved with the dot product — **not a matmul**.

**With absorption** (tensor-core friendly):

Define the gate encodings relative to chunk boundaries:

$$
\Lambda_{[n],j} = \frac{b_{nC+j}}{b_{nC}} = \prod_{s=1}^{j} \alpha_{nC+s}, \quad \Gamma_{[n],j} = \frac{b_{(n+1)C}}{b_{nC+j}} = \prod_{s=j+1}^{C} \alpha_{nC+s}
$$

Absorb into Q and K:

$$
\tilde{Q}_{[n]} = Q_{[n]} \odot \Lambda_{[n]}, \quad \tilde{K}_{[n]} = K_{[n]} \odot \Gamma_{[n]}
$$

Then:

$$
P_{[n]} = \tilde{Q}_{[n]} \tilde{K}_{[n]}^\top \odot M
$$

where $M$ is a causal mask. This is a **standard $(C \times d_k) \times (d_k \times C)$ matmul** followed by element-wise masking — fully tensor-core compatible.

**Inter-chunk state update (also uses absorption):**

$$
S_{[n+1]} = (\gamma_{[n+1]}^\top \mathbf{1}) \odot S_{[n]} + \tilde{K}_{[n+1]}^\top V_{[n+1]}
$$

where $\gamma_{[n+1]} = \prod_{s=1}^{C} \alpha_{(n+1)C+s}$ is the total decay over the chunk. The $\tilde{K}^\top V$ update is again a standard matmul.

**Inter-chunk output:**

$$
O_{[n]}^{\text{inter}} = \tilde{Q}_{[n]} S_{[n-1]}
$$

Again a standard $(C \times d_k) \times (d_k \times d_v)$ matmul.

**Model-Specific Instantiations:**

| Model | Gate Type | $\Lambda$ | $\Gamma$ | Mask $M$ |
|-------|-----------|-----------|-----------|----------|
| RetNet | Scalar $\gamma$ | $\gamma^j$ | $\gamma^{C-j}$ | $\gamma^{i-j}$ (causal decay) |
| GLA | Diagonal $\alpha_t \in (0,1)^{d_k}$ | $\prod_{s=1}^{j} \alpha_s$ | $\prod_{s=j+1}^{C} \alpha_s$ | Binary causal |
| Mamba-2/SSD | Scalar $\gamma_t$ | $\prod_{s=1}^{j} \gamma_s$ | $\prod_{s=j+1}^{C} \gamma_s$ | Binary causal |
| KDA (DPLR) | Tied $\alpha_t \in (0,1)^{d_k}$ | $\prod \alpha_s$ | $\prod \alpha_s$ | Binary causal |
| LA-2 | Scalar $\lambda$ | $\text{diag}(\lambda, \ldots, \lambda^B)$ | $\lambda^{B-j}$ | $\lambda^{i-j}$ (causal decay) |

**Key Definitions:**

- $\Lambda_{[n]} \in \mathbb{R}^{C \times d_k}$ — query-side absorption: cumulative decay from chunk start to each position
- $\Gamma_{[n]} \in \mathbb{R}^{C \times d_k}$ — key-side absorption: cumulative decay from each position to chunk end
- $\gamma_{[n]} \in \mathbb{R}^{d_k}$ — total decay over chunk $[n]$: $\gamma = \Lambda_C = \Gamma_0^{-1}$
- $\tilde{Q}, \tilde{K}$ — gate-absorbed query and key matrices
- $M \in \mathbb{R}^{C \times C}$ — post-absorption mask (binary causal or weighted causal decay)
- $C$ — chunk size, $d_k$ — key dimension

## Complexity

| Operation | Without Absorption | With Absorption |
|-----------|-------------------|-----------------|
| Intra-chunk attention | $O(C^2 d_k)$ element-wise (no TC) | $O(C^2 d_k)$ matmul (**tensor core**) |
| Absorption overhead | — | $O(C \cdot d_k)$ element-wise (negligible) |
| Inter-chunk output | $O(C \cdot d_k \cdot d_v)$ matmul | $O(C \cdot d_k \cdot d_v)$ matmul (same) |
| State update | $O(C \cdot d_k \cdot d_v)$ matmul | $O(C \cdot d_k \cdot d_v)$ matmul (same) |

**Arithmetic intensity improvement:**

Without absorption, the intra-chunk computation mixes element-wise operations (exp, multiply) with reductions (dot products), resulting in memory-bound execution. With absorption, the dominant computation is a $(C \times d_k) \times (d_k \times C)$ GEMM with arithmetic intensity $O(d_k)$ FLOPs/byte — **compute-bound** on modern GPUs.

For $C = 128, d_k = 64$: the GEMM has 128 × 64 × 128 ≈ 1M FLOPs over 2 × 128 × 64 × 2 ≈ 32KB of BF16 data = 32 FLOPs/byte, well above the ~10 FLOPs/byte needed to saturate H100 tensor cores.

**Memory:** The absorption adds $O(C \cdot d_k)$ temporary storage for $\tilde{Q}$ and $\tilde{K}$ per chunk, which is negligible compared to the $C \times d_v$ output tile. In practice, $\Lambda$ and $\Gamma$ are computed on-chip from the log-gates via `cumsum` and `exp` and never stored to HBM.

## Applicability

- **All chunkwise linear attention implementations:** This is the standard approach used in every production chunkwise kernel. The FLA library, Lightning Attention, and Mamba-2's SSD all use some form of decay absorption.

- **Enables secondary chunking in GLA:** When gates are per-dimension ($\alpha_t \in (0,1)^{d_k}$), the absorption enables inter-sub-chunk tiles to use tensor cores (see trick 177). The absorption is applied at the sub-chunk level: $\tilde{Q}_{[i]} = Q_{[i]} \odot \Lambda_{[i]}$, $\tilde{K}_{[j]} = K_{[j]} \odot \Gamma_{[j]} \odot (b_{jc} / b_{(j+1)c})$.

- **Unifies RetNet, GLA, and Mamba-2:** Despite different gate parameterizations, all three use the same $\tilde{Q}\tilde{K}^\top$ matmul pattern after absorption. This allows a single Triton/CUDA kernel template to serve multiple architectures by varying only the $\Lambda, \Gamma, M$ construction.

- **Critical for mixed-precision training:** By isolating the gate factors (which may require FP32 for numerical stability of cumsum/exp) from the matmul (which runs in BF16 on tensor cores), absorption enables **mixed-precision within a single kernel**: gates are computed in FP32, absorbed into Q/K, then cast to BF16 for the GEMM.

## Limitations

- **Per-dimension gates still need log-space for diagonal blocks:** When $\alpha_t$ varies per dimension (GLA), the absorption works perfectly for inter-sub-chunk tiles, but diagonal (intra-sub-chunk) tiles still require log-space computation because the gate ratio $b_i/b_j$ cannot be factored into a Q-side and K-side component when $i$ and $j$ are in the same sub-chunk and the gate is per-dimension. This is exactly why GLA needs secondary chunking (trick 177).

- **Absorption changes the numerical conditioning:** Multiplying Q by $\Lambda$ (cumulative decay) can make early positions in a chunk very small (for fast-decaying heads) or late positions very large. This can affect the numerical stability of the subsequent matmul in low precision. RetNet mitigates this with GroupNorm; GLA uses the secondary chunking approach.

- **Cannot absorb full-matrix gates:** If the gate is a full matrix $A_t \in \mathbb{R}^{d \times d}$ (not diagonal), absorption would require $\tilde{Q} = Q A_t^{-1}$ which is itself a matmul, not an element-wise operation. The trick only works for scalar or diagonal gates.

- **Adds pre-processing overhead:** The cumsum + exp to compute $\Lambda, \Gamma$ is an extra sequential step before the matmul. For very small chunks, this overhead can be non-negligible. However, for typical chunk sizes ($C \geq 64$), the quadratic matmul FLOPs dominate.

## Implementation Notes

```python
import torch

def absorb_decay_into_qk(Q, K, log_alpha, chunk_size):
    """
    Absorb cumulative decay factors into Q and K for tensor-core matmul.

    Args:
        Q: (B, H, L, d_k) — queries
        K: (B, H, L, d_k) — keys
        log_alpha: (B, H, L, d_k) — log forget gates (per-position, per-dim)
        chunk_size: int — primary chunk size C

    Returns:
        Q_tilde: (B, H, L, d_k) — decay-absorbed queries
        K_tilde: (B, H, L, d_k) — decay-absorbed keys
        gamma: (B, H, N, d_k) — per-chunk total decay for state update
    """
    B, H, L, d_k = Q.shape
    C = chunk_size
    N = L // C

    # Reshape into chunks
    log_a = log_alpha.view(B, H, N, C, d_k)

    # Cumulative log-gate within each chunk: [0, a_1, a_1+a_2, ...]
    log_cumsum = torch.cumsum(log_a, dim=3)  # (B, H, N, C, d_k)

    # Lambda: decay from chunk start to position j
    # Lambda_j = exp(sum_{s=1}^{j} log alpha_s) = exp(log_cumsum_j)
    Lambda = torch.exp(log_cumsum)  # (B, H, N, C, d_k)

    # Gamma: decay from position j to chunk end
    # Gamma_j = exp(sum_{s=j+1}^{C} log alpha_s) = exp(total - log_cumsum_j)
    total_decay = log_cumsum[:, :, :, -1:, :]  # (B, H, N, 1, d_k)
    Gamma = torch.exp(total_decay - log_cumsum)  # (B, H, N, C, d_k)

    # Absorb into Q and K
    Q_chunks = Q.view(B, H, N, C, d_k)
    K_chunks = K.view(B, H, N, C, d_k)

    Q_tilde = (Q_chunks * Lambda).view(B, H, L, d_k)  # element-wise, O(L*d_k)
    K_tilde = (K_chunks * Gamma).view(B, H, L, d_k)   # element-wise, O(L*d_k)

    # Total chunk decay for inter-chunk state update
    gamma = torch.exp(total_decay.squeeze(3))  # (B, H, N, d_k)

    return Q_tilde, K_tilde, gamma


def chunkwise_attention_with_absorption(Q_tilde, K_tilde, V, gamma, S_prev, C):
    """
    After absorption, the intra-chunk attention is a standard matmul.

    Q_tilde, K_tilde: (B, H, C, d_k) — per-chunk, gate-absorbed
    V: (B, H, C, d_v) — values
    S_prev: (B, H, d_k, d_v) — inter-chunk state
    """
    # INTRA-CHUNK: standard tensor-core matmul!
    P = Q_tilde @ K_tilde.transpose(-1, -2)  # (C, C) — TENSOR CORE ✓
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(C, C, device=P.device))
    P = P * causal_mask
    O_intra = P @ V  # (C, d_v) — TENSOR CORE ✓

    # INTER-CHUNK: also a standard matmul
    O_inter = Q_tilde @ S_prev  # (C, d_v) — TENSOR CORE ✓

    # STATE UPDATE: standard matmul with decay
    S_new = gamma.unsqueeze(-1) * S_prev + K_tilde.transpose(-1, -2) @ V

    return O_intra + O_inter, S_new


# SCALAR GATE EXAMPLE (RetNet):
# For RetNet with fixed scalar decay gamma per head:
#   Lambda = diag(gamma, gamma^2, ..., gamma^C)
#   Gamma  = diag(gamma^{C-1}, gamma^{C-2}, ..., 1)
#   M[i,j] = gamma^{i-j} for i >= j  (causal decay mask)
#   Q_tilde = Q * [gamma, gamma^2, ..., gamma^C]
#   K_tilde = K * [gamma^{C-1}, ..., gamma, 1]
#   P = Q_tilde @ K_tilde.T * M  (matmul + Hadamard with decay mask)
#
# This is exactly the RetNet parallel form (trick 251),
# and Lightning Attention-2's intra-block computation (trick 217).
#
# KEY INSIGHT: The absorption trick is what CONVERTS a gated recurrence
# into a standard attention-like matmul. Without it, the exponential
# decay factors would be interleaved with the dot product, preventing
# tensor core utilization. With it, the gate computation is separated
# into O(C*d_k) element-wise pre-processing, and the attention itself
# is a pure GEMM — the most optimized operation on any GPU.
```

## References

- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635. (Section 4.3, Figure 3, Listing 1 — explicit absorption of $\Lambda, \Gamma$ into Q, K for secondary chunking)
- Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621. (Section 2 — implicit absorption via xPos embeddings applied to Q and K)
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality. ICML 2024. (SSD chunkwise form uses the same Q/K absorption pattern)
- Qin, Z., Sun, W., Li, D., et al. (2024). Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models. arXiv:2401.04658. (Intra-block uses $\Lambda$-absorbed Q with KV state, Eq. 5-7)
- He, B., Hofstätter, S., Bernstein, D., & Zhai, S. (2025). Kimi-Audio / KDA: Constrained DPLR with tied $a = \beta k$ absorption. arXiv:2510.26692. (Table 7 — the constrained DPLR form eliminates one absorption step by tying $a$ to $k$)

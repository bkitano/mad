# 190: SageAttention2 — Per-Thread Mixed-Precision Tiled Attention

**Category**: kernel
**Gain type**: efficiency
**Source**: Zhang, Huang, Zhang, Wei, Zhu & Chen (2025) — ICML 2025
**Paper**: [papers/sageattention2-mixed-precision-tiled-attention.pdf]
**Documented**: 2026-02-16

## Description

Standard FlashAttention computes both matmuls ($QK^\top$ and $\tilde{P}V$) in FP16 on tensor cores. SageAttention2 introduces a **mixed-precision tiled attention kernel** that quantizes $Q, K$ to INT4 and $\tilde{P}, V$ to FP8 (E4M3), achieving up to **3× speedup** over FlashAttention2 on RTX 4090 and matching FlashAttention3(fp8) speed on H100 while delivering significantly higher accuracy.

The key innovations address three challenges of aggressive quantization within the FlashAttention tiling framework:

1. **Per-thread quantization**: Instead of per-block or per-tensor quantization, SageAttention2 assigns quantization groups that align with the MMA instruction's thread-to-data mapping. Each GPU thread holds elements that share a single quantization scale, achieving per-token-level accuracy with zero dequantization overhead. For INT4 `mma.m16n8k64`, this creates 32 quantization groups for $Q$ (32× finer than per-block) and 4 groups for $K$ (4× finer) within a 128×64 tile.

2. **Q+K smoothing**: Channel-wise outliers in $Q$ and $K$ destroy INT4 accuracy. SageAttention2 subtracts the per-block channel mean from both $Q$ and $K$ before quantization: $\gamma(Q_i) = Q_i - \bar{q}_i$, $\gamma(K_j) = K_j - \bar{k}$. The correction term $\Delta S_{ij} = \bar{q}_i \gamma(K_j)^\top$ is a rank-1 vector that adds a constant bias per row of $S$, which vanishes after softmax. This makes the dynamic range uniform and dramatically improves quantization accuracy (cosine similarity 99.46% vs 80.04% without smoothing).

3. **Two-level FP32 accumulation for $\tilde{P}V$**: The FP8 `mma(f32.f8.f8.f32)` accumulator on Ada/Hopper is actually FP22 (1 sign + 8 exponent + 13 mantissa), not true FP32. This causes accuracy loss when accumulating many $\tilde{P}V$ tiles. SageAttention2 uses an inner FP22 accumulator $R_{ij}$ that accumulates over a small number of KV tiles ($b_k = 64$), then periodically flushes to an outer FP32 accumulator $O_{ij}$, confining errors to the block range.

**Relevance to TFLA/chunkwise linear attention**: The per-thread quantization and two-level accumulation strategies are directly applicable to TFLA's inner tiled matmuls ($Q^{(k)}K^{(k)\top}$ and $S^{(k)}V^{(k)}$). Since TFLA already tiles along the sequence dimension at two levels, adding mixed-precision quantization at the inner tile level would compound the arithmetic intensity gains with reduced precision throughput (2× for FP8, 4× for INT4 tensor cores).

## Mathematical Form

**Standard attention (FlashAttention tiling):**

$$
S_{ij} = Q_i K_j^\top / \sqrt{d}, \quad \tilde{P}_{ij} = \exp(S_{ij} - m_{ij}), \quad O_i = \text{diag}(\ell_{i})^{-1} \sum_j \tilde{P}_{ij} V_j
$$

**Per-thread quantization of $QK^\top$:**

Each block $Q_i \in \mathbb{R}^{b_q \times d}$ is split into $c_w$ warp segments $Q_w$. For the INT4 `mma.m16n8k64` instruction, each thread handles a specific subset of elements. The quantization group index for thread $n$ is:

$$
i_{\delta q} = \lfloor n \cdot 8 \cdot c_w / b_q \rfloor
$$

$$
q_i[i_{\delta q}] = \{8 \times (n\%8) + \lfloor n \cdot \frac{c_w}{b_q} \rfloor \cdot \frac{b_q}{c_w}\}, \quad n \in [0, N]
$$

$$
\delta_Q[i_{\delta q}] = \frac{\max(|Q[q_i[i_{\delta q}]]|)}{7}, \quad \hat{Q}[q_i[i_{\delta q}]] = \left\lfloor \frac{Q[q_i[i_{\delta q}]]}{\delta_Q[i_{\delta q}]} \right\rceil
$$

Similarly for $K$ with groups indexed by $i_{\delta k} = \lfloor n \cdot 4/b_k \rfloor$.

**Key property**: Each thread performs dequantization with exactly one $\delta_Q$ and one $\delta_K$ scale, so:

$$
S_{ij}[st : st + c_w] = \psi^{-1}_{\delta_Q, \delta_K}\left(\text{Matmul}(\hat{Q}_i[st : st + c_w], \hat{K}_j^\top)\right) + \text{GEMV}(\bar{q}_i, K_j^\top)
$$

where $\psi^{-1}_{\delta_Q, \delta_K}(\hat{Q}\hat{K}^\top) = \hat{Q}\hat{K}^\top \times \delta_Q \times \delta_K$ is element-wise rescaling.

**Q+K Smoothing:**

$$
\gamma(Q_i) = Q_i - \bar{q}_i, \quad \gamma(K_j) = K_j - \bar{k}
$$

where $\bar{q}_i = \text{mean}(Q_i)$ is a $1 \times d$ vector (mean along token axis), $\bar{k} = \text{mean}(K)$ is a $1 \times d$ vector (global mean). Then:

$$
Q_i K_j^\top = \gamma(Q_i)\gamma(K_j)^\top + \underbrace{\bar{q}_i \gamma(K_j)^\top}_{\Delta S_{ij}} + \underbrace{\gamma(Q_i)\bar{k}^\top + \bar{q}_i \bar{k}^\top}_{b}
$$

The term $b$ is an $N \times 1$ vector (constant per row) that cancels after softmax. $\Delta S_{ij} = \bar{q}_i \gamma(K_j)^\top$ is a GEMV computed once and added back.

**Two-level accumulation for $\tilde{P}V$:**

$$
R_{ij}(\text{FP22}) = \text{Matmul}(\tilde{P}_{ij} \times 448, V_j) \quad \text{(cast to FP8 E4M3)}
$$

$$
O_{ij}(\text{FP32}) = \text{diag}(e^{m_{i,j-1} - m_{ij}}) \cdot O_{i,j-1}(\text{FP32}) + R_{ij}(\text{FP22})
$$

The inner accumulator $R_{ij}$ uses the hardware FP22 precision over a small number of tiles (e.g., $b_k = 64$ tokens). The outer accumulator $O_{ij}$ uses true FP32, preventing error buildup across the full KV sequence.

**Key Definitions:**

- $b_q, b_k, b_v$ — block sizes for $Q$, $K$, $V$ tiles (typically 128, 64, 64)
- $c_w$ — number of warps per SM (typically 4)
- $\delta_Q, \delta_K$ — per-thread quantization scales for $Q$, $K$
- $\delta_P = 1/448$ — static FP8 scale for $\tilde{P}$ (since $\tilde{P} \in [0,1]$, E4M3 range $[-448, 448]$)
- $\delta_V$ — per-channel scale for $V$
- $\bar{q}_i, \bar{k}$ — smoothing mean vectors

## Complexity

| Operation | FlashAttention2 (FP16) | SageAttention2 (INT4+FP8) |
|-----------|----------------------|--------------------------|
| $QK^\top$ matmul | FP16 tensor cores | INT4 tensor cores (**4× throughput**) |
| $\tilde{P}V$ matmul | FP16 tensor cores | FP8 tensor cores (**2× throughput**) |
| Smoothing overhead | — | 3.7% of kernel time |
| Per-thread quant overhead | — | 0.35% of kernel time |
| Two-level accum overhead | — | 0% (register-level) |

**Throughput (TOPS on RTX4090, headdim=128, causal):**

| Method | 1K | 4K | 8K | 16K | 32K |
|--------|----|----|----|----|-----|
| FlashAttention2 | 78 | 150 | 193 | 244 | 310 |
| SageAttention2-4b | 88 | 293 | 400 | 519 | 326 |
| SageAttention2-8b | 78 | 218 | 310 | 391 | 326 |

**Speedup summary (Table 9):**

| GPU | SageAttention2 vs FlashAttention2 |
|-----|----------------------------------|
| RTX 3090 | 1.97× |
| RTX 4090 | 2.93× |
| L40 | 2.60× |
| L20 | 2.46× |
| H100 | 2.61× |
| H20 | 3.12× |

**Memory:** Same as FlashAttention2 — $O(N)$ for mask storage, no additional memory for quantization scales (computed on-the-fly in registers).

## Applicability

- **Softmax attention (primary):** Drop-in replacement for FlashAttention in any transformer. Validated on Llama3.1, GLM4, CogVideoX, HunyuanVideo, Flux, Stable-Diffusion, TIMM.

- **Tiled linear attention (TFLA):** The per-thread quantization and two-level accumulation are directly applicable to TFLA's inner tile matmuls. The $QK^\top$ in TFLA's intra-chunk computation can use INT4, and the $SV$ accumulation can use FP8 with two-level buffering. The smoothing technique for $Q$ and $K$ is independent of the attention mechanism (softmax vs linear).

- **Training acceleration (potential):** SageAttention2 is demonstrated for inference. For training, the backward pass requires higher precision for gradient computation, but FP8 forward passes with FP16 backward passes (mixed-precision training) could use this technique for the forward attention.

- **Linear RNNs with gated attention:** mLSTM and GLA both compute $QK^\top$-like products within their chunkwise parallel kernels. Per-thread INT4 quantization of these products could accelerate the compute-bound intra-chunk phase.

## Limitations

- **INT4 `mma.m16n8k64` only available on Ada/Hopper+:** Not available on A100 (Ampere), which only has INT8 tensor cores. The 8-bit variant (SageAttn2-8b) works on older GPUs but with less speedup.

- **Accuracy degrades for some models:** Per-thread INT4 achieves 99.45% cosine similarity on CogVideoX but is not perfect. Some sensitive applications may notice quality differences (Table 5 shows INT4+smoothing slightly degrades Llama3.1 Lambda accuracy).

- **FP22 accumulator discovery is hardware-specific:** The two-level accumulation addresses an undocumented hardware behavior (FP22 instead of FP32 in `mma.f32.f8.f8.f32`). Future hardware may not have this issue, making the technique unnecessary.

- **Smoothing adds a preprocessing step:** The $\bar{q}_i$ and $\bar{k}$ computation requires a reduction over the channel dimension before quantization, adding 3.7% overhead.

- **Not validated for linear attention training:** All experiments are inference-only. Applying INT4 quantization during training backward passes is an open question.

## Implementation Notes

```python
# SageAttention2 kernel pseudocode (Algorithm 1 from the paper)
def sage_attention2(Q_fp16, K_fp16, V_fp16, b_q=128, b_k=64, c_w=4):
    """
    Mixed-precision tiled attention with per-thread INT4/FP8 quantization.
    """
    N, d = Q_fp16.shape

    # === PREPROCESSING (on-chip, fused with attention kernel) ===
    # Step 1: Smooth K globally
    k_bar = K_fp16.mean(dim=0)  # 1 x d
    K_smooth = K_fp16 - k_bar

    # Divide into Q blocks and K blocks
    T_m = N // b_q  # number of Q blocks
    T_n = N // b_k  # number of K blocks

    for i in range(T_m):  # PARALLEL over SMs
        Q_i = Q_fp16[i*b_q:(i+1)*b_q]

        # Step 2: Smooth Q per-block
        q_bar_i = Q_i.mean(dim=0)  # 1 x d
        Q_smooth_i = Q_i - q_bar_i

        # Step 3: Per-thread INT4 quantization (zero overhead)
        # Each thread in mma.m16n8k64 gets its own scale
        delta_Q, Q_hat = per_thread_quantize_int4(Q_smooth_i)  # 32 groups
        delta_K_list = []

        O_i = zeros(b_q, d, dtype=fp32)  # outer accumulator
        m_i = full(b_q, -inf)
        l_i = zeros(b_q)

        for j in range(T_n):  # LOOP over KV blocks
            K_j = K_smooth[j*b_k:(j+1)*b_k]
            V_j = V_fp16[j*b_k:(j+1)*b_k]

            # Per-thread INT4 quantization of K_j
            delta_K, K_hat = per_thread_quantize_int4(K_j)  # 4 groups

            # INT4 tensor core matmul + dequantize (in registers)
            S_ij = dequant(matmul_int4(Q_hat, K_hat.T), delta_Q, delta_K)
            S_ij += q_bar_i @ K_j.T  # GEMV correction (smoothing)

            # Online softmax
            m_new = max(m_i, rowmax(S_ij))
            P_ij = exp(S_ij - m_new)
            l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)

            # FP8 quantization of P_ij (static scale 1/448)
            P_hat = quantize_fp8_e4m3(P_ij * 448)

            # Per-channel FP8 quantization of V_j
            delta_V, V_hat = per_channel_quantize_fp8(V_j)

            # FP8 tensor core matmul -> FP22 inner accumulator
            R_ij = matmul_fp8(P_hat, V_hat)  # FP22 accumulator

            # Two-level accumulation: flush FP22 -> FP32
            O_i = diag(exp(m_i - m_new)) @ O_i + R_ij  # FP32

            m_i = m_new

        # Final normalization and output
        O_i = diag(l_i) ** -1 @ O_i
        # Write to HBM

    return O
```

**GPU efficiency analysis:**

1. **All dominant operations are tensor core matmuls:** INT4 `mma.m16n8k64` for $QK^\top$ (4× throughput vs FP16), FP8 `mma.f32.f8.f8.f32` for $\tilde{P}V$ (2× throughput vs FP16).

2. **Zero dequantization overhead:** Per-thread quantization aligns scales with the MMA thread-to-data mapping, so each thread uses a single scalar scale — no gather/scatter needed.

3. **Memory access pattern unchanged:** Same tiling and HBM access pattern as FlashAttention2 — coalesced loads of $Q$, $K$, $V$ blocks, tiled computation in SRAM.

4. **Smoothing is fused:** The mean subtraction and GEMV correction are fused into the main kernel, adding only 3.7% overhead.

5. **Peak performance: 481 TOPS on RTX4090** (3× FlashAttention2's 163 TOPS).

## References

- Zhang, J., Huang, H., Zhang, P., Wei, J., Zhu, J., & Chen, J. (2025). SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization. ICML 2025. arXiv:2411.10958.
- Zhang, J., Wei, J., Zhang, P., Chen, J., & Zhu, J. (2025). SageAttention: Accurate 8-bit Attention for Plug-and-Play Inference Acceleration. ICLR 2025.
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS 2024.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- NVIDIA. CUTLASS: CUDA Templates for Linear Algebra Subroutines and Solvers. https://github.com/NVIDIA/cutlass.

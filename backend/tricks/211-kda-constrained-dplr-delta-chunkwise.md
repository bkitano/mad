# 211: KDA Constrained DPLR Delta Rule Chunkwise Algorithm

**Category**: decomposition
**Gain type**: efficiency
**Source**: Kimi Team / Moonshot AI (2025) — Technical Report
**Paper**: [papers/kimi-linear-delta-attention.pdf]
**Documented**: 2026-02-15

## Description

The gated delta rule is one of the most expressive linear attention update rules, combining a Householder-style rank-1 state correction with multiplicative decay. In its general Diagonal-Plus-Low-Rank (DPLR) form $\boldsymbol{S}_t = (\boldsymbol{D} - \boldsymbol{a}_t\boldsymbol{b}_t^\top)\boldsymbol{S}_{t-1} + \boldsymbol{k}_t\boldsymbol{v}_t^\top$, the transition matrix has independent diagonal $\boldsymbol{D}$ and low-rank vectors $\boldsymbol{a}_t, \boldsymbol{b}_t$. While expressive, this general DPLR requires **4 secondary-level chunk matmuls** and **3 additional inter/intra-chunk matmuls** in the chunkwise parallel algorithm — making it roughly 2× slower than simpler gated linear attention.

**Kimi Delta Attention (KDA)** introduces a *constrained* DPLR variant that **ties** $\boldsymbol{a} = \beta\boldsymbol{k}$ and $\boldsymbol{b} = \boldsymbol{k} \odot \boldsymbol{\alpha}$ (where $\boldsymbol{\alpha}$ is a per-channel decay gate), reducing the transition matrix to:

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

This constraint:
1. **Eliminates secondary chunking**: The cumulative decay $1/\Gamma$ reciprocal (needed for general DPLR) never appears because the diagonal decay $\boldsymbol{\alpha}$ and the rank-1 correction share the same key vector, avoiding numerical instability.
2. **Removes 3 matmuls** from the intra-chunk and inter-chunk computations.
3. **Enables full half-precision tensor core utilization** — no log-domain secondary chunking needed (unlike GLA).

The result: KDA's chunkwise kernel is **~2× faster** than general DPLR at equal sequence lengths, while the Kimi Linear architecture (3:1 KDA-to-MLA hybrid) achieves **6.3× faster decoding** than full MLA at 1M context length, with **better quality** across pretraining, SFT, and RL benchmarks.

## Mathematical Form

**Gated DeltaNet (GDN) — scalar decay:**

$$
\boldsymbol{S}_t = \alpha_t (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

where $\alpha_t \in [0, 1]$ is a scalar (per-head) forget gate and $\beta_t \in [0, 1]$ is the learning rate gate.

**KDA — channel-wise (fine-grained) decay:**

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \operatorname{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top \in \mathbb{R}^{d_k \times d_v}
$$

$$
\boldsymbol{o}_t = \boldsymbol{S}_t^\top \boldsymbol{q}_t \in \mathbb{R}^{d_v}
$$

where $\boldsymbol{\alpha}_t \in [0, 1]^{d_k}$ is a per-channel decay vector (not scalar).

**Equivalence to constrained DPLR:**

Rewriting KDA as $\boldsymbol{S}_t = (\boldsymbol{D} - \boldsymbol{a}_t \boldsymbol{b}_t^\top)\boldsymbol{S}_{t-1} + \boldsymbol{k}_t \boldsymbol{v}_t^\top$ with:

$$
\boldsymbol{D} = \operatorname{Diag}(\boldsymbol{\alpha}_t), \quad \boldsymbol{a}_t = \beta_t \boldsymbol{k}_t, \quad \boldsymbol{b}_t = \boldsymbol{k}_t \odot \boldsymbol{\alpha}_t
$$

The constraint $\boldsymbol{a} \propto \boldsymbol{k}$ and $\boldsymbol{b} \propto \boldsymbol{k} \odot \boldsymbol{\alpha}$ is what enables the algorithmic simplification.

**Chunkwise-parallel formulation:**

Divide sequence of length $L$ into chunks of size $C$. For chunk $[t]$:

**WY representation** (packing rank-1 updates into compact form):

$$
\boldsymbol{P}_{[t]}^r = \operatorname{Diag}(\boldsymbol{\gamma}_{[t]}^r) - \sum_{i=1}^{r} \operatorname{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \boldsymbol{k}_{[t]}^i \boldsymbol{w}_{[t]}^{i\top}
$$

$$
\boldsymbol{H}_{[t]}^r = \sum_{i=1}^{r} \operatorname{Diag}\left(\boldsymbol{\gamma}_{[t]}^{i \to r}\right) \boldsymbol{k}_{[t]}^i \boldsymbol{u}_{[t]}^{i\top}
$$

where auxiliary vectors $\boldsymbol{w}$ and $\boldsymbol{u}$ are computed via recurrence:

$$
\boldsymbol{w}_{[t]}^r = \beta_{[t]}^r \left(\operatorname{Diag}(\boldsymbol{\gamma}_{[t]}^0) \boldsymbol{k}_{[t]}^r - \sum_{i=1}^{r-1} \boldsymbol{w}_{[t]}^i \left(\boldsymbol{k}_{[t]}^{i\top} \operatorname{Diag}\left(\boldsymbol{\gamma}_{[t]}^{i \to r}\right) \boldsymbol{k}_{[t]}^r\right)\right)
$$

$$
\boldsymbol{u}_{[t]}^r = \beta_{[t]}^r \left(\boldsymbol{v}_{[t]}^r - \sum_{i=1}^{r-1} \boldsymbol{u}_{[t]}^i \left(\boldsymbol{k}_{[t]}^{i\top} \operatorname{Diag}\left(\boldsymbol{\gamma}_{[t]}^{i \to r}\right) \boldsymbol{k}_{[t]}^r\right)\right)
$$

**UT transform** (converting triangular solve to matmul):

$$
\boldsymbol{M}_{[t]} = \left(\boldsymbol{I} + \text{StrictTril}\left(\operatorname{Diag}(\beta_{[t]}) \left(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \boldsymbol{K}_{[t]}\right) \left(\frac{\boldsymbol{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}}\right)^\top\right)\right)^{-1} \operatorname{Diag}(\beta_{[t]})
$$

$$
\boldsymbol{W}_{[t]} = \boldsymbol{M}_{[t]} \left(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \boldsymbol{K}_{[t]}\right), \quad \boldsymbol{U}_{[t]} = \boldsymbol{M}_{[t]} \boldsymbol{V}_{[t]}
$$

**KDA simplification vs. general DPLR:**

In general DPLR, the intra-chunk computation requires the reciprocal of the cumulative decay $1/\Gamma$, which can be numerically unstable and necessitates secondary chunking in log-space (as in GLA). By constraining $\boldsymbol{a} = \beta\boldsymbol{k}$ and $\boldsymbol{b} = \boldsymbol{k} \odot \boldsymbol{\alpha}$, KDA:

- Reduces the number of secondary-level matmul accumulations from **4 to 2**
- Eliminates **3 additional matmuls** in inter/intra-chunk output computation
- Avoids the $1/\Gamma$ reciprocal entirely

**Inter-chunk state update:**

$$
\boldsymbol{S}_{[t+1]} = \operatorname{Diag}(\boldsymbol{\gamma}_{[t]}^C) \boldsymbol{S}_{[t]} + \left(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \boldsymbol{K}_{[t]}\right)^\top (\boldsymbol{U}_{[t]} - \boldsymbol{W}_{[t]} \boldsymbol{S}_{[t]}) \in \mathbb{R}^{d_k \times d_v}
$$

**Intra-chunk output:**

$$
\boldsymbol{O}_{[t]} = \underbrace{\left(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \boldsymbol{Q}_{[t]}\right) \boldsymbol{S}_{[t]}}_{\text{inter chunk}} + \underbrace{\text{Tril}\left(\left(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \boldsymbol{Q}_{[t]}\right) \left(\frac{\boldsymbol{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}}\right)^\top\right) (\boldsymbol{U}_{[t]} - \boldsymbol{W}_{[t]} \boldsymbol{S}_{[t]})}_{\text{intra chunk}}
$$

**Key Definitions:**

- $\boldsymbol{S}_t \in \mathbb{R}^{d_k \times d_v}$ — matrix-valued recurrent state (typically $128 \times 128$)
- $\boldsymbol{\alpha}_t \in [0, 1]^{d_k}$ — per-channel decay gate
- $\beta_t \in [0, 1]$ — scalar learning rate gate
- $\boldsymbol{\gamma}_{[t]}^{i \to j} = \prod_{m=i}^{j} \boldsymbol{\alpha}_{[t]}^m$ — cumulative decay from position $i$ to $j$ in chunk
- $C$ — chunk size (typically 64)
- $\boldsymbol{M}_{[t]}$ — UT transform matrix (lower-triangular inverse, computed via forward substitution)

## Complexity

**FLOPs per sequence (single head, chunk size $C = 64$):**

$$
\text{FLOPs}_{\text{KDA}}(T; C, d_h) = 6T d_h^2 + 3T C d_h + T C^2
$$

vs. full softmax attention:

$$
\text{FLOPs}_{\text{Attn}}(T; d_h) = 2T^2 d_h
$$

| Operation | General DPLR | KDA (Constrained) | Savings |
|-----------|-------------|-------------------|---------|
| Secondary chunk matmuls | 4 | **2** | 2× fewer |
| Inter/intra-chunk matmuls | 6+ | **3** | ~2× fewer |
| Numerical stability ops | Log-domain secondary chunking | **Not needed** | Eliminated |
| Total kernel time (vs DPLR) | baseline | **~2× faster** | Fig 2 in paper |

**Memory:** Fixed recurrent state $d_k \times d_v = 128 \times 128 = 16K$ per head, independent of sequence length. vs. MLA KV-cache which grows as $O(N)$.

**Wall-clock performance (48B MoE, 3B active, H100):**

| Metric | Full MLA | KDA (Kimi Linear) | Improvement |
|--------|---------|-------------------|-------------|
| Prefill 128K | baseline | ~comparable | ~1× |
| Prefill 512K | baseline | **2.3× faster** | 2.3× |
| Prefill 1M | baseline | **2.9× faster** | 2.9× |
| Decoding TPOT 128K | baseline | **1.8× faster** | 1.8× |
| Decoding TPOT 512K | baseline | **2.2× faster** | 2.2× |
| Decoding TPOT 1M | ~11.5ms | **~1.84ms** | **6.3× faster** |
| KV cache (1M context) | ~75% of memory | **~25% of memory** | 75% reduction |

## Applicability

- **Drop-in replacement for softmax attention in LLMs**: Kimi Linear demonstrates that a 3:1 hybrid (3 KDA layers per 1 MLA layer) outperforms pure MLA across pretraining, SFT, and RL at 48B scale (1.4T tokens). This is the first linear attention to **beat** full attention under fair comparison.

- **Long-context generation (agentic, code, RL)**: The fixed-size recurrent state ($d_k \times d_v$ per head) provides constant memory during autoregressive generation, enabling 1M+ context without KV-cache blowup. Critical for agentic workloads with extended trajectories.

- **Extension of DeltaNet/GDN**: KDA directly extends Gated DeltaNet by replacing the scalar decay $\alpha_t$ with per-channel $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$, adding fine-grained forgetting analogous to how GLA extends RetNet. Any system running GDN can upgrade to KDA.

- **Positional encoding via decay**: KDA serves as a learnable, data-dependent multiplicative positional encoding — replacing RoPE entirely. The paper shows NoPE + KDA outperforms RoPE on long-context tasks.

- **vLLM integration**: Open-source KDA kernels integrate with vLLM inference framework, requiring no modification to KV-cache management or scheduling.

## Limitations

- **Recall capacity bounded by state size**: The recurrent state $\boldsymbol{S} \in \mathbb{R}^{d_k \times d_v}$ has finite capacity. Pure KDA still underperforms full attention on exact copying and fine-grained retrieval over very long contexts — hence the hybrid design with periodic MLA layers.

- **Chunk size sensitivity**: The UT transform within each chunk involves a lower-triangular matrix inverse computed via forward substitution — an $O(C^2)$ sequential operation. Larger chunks increase parallelism but also increase this sequential cost.

- **Per-channel decay adds parameters**: The decay $\boldsymbol{\alpha}_t$ is parameterized via a low-rank projection $f(\boldsymbol{W}_\alpha^\uparrow \boldsymbol{W}_\alpha^\downarrow \boldsymbol{x}_t)$, adding parameters relative to GDN's scalar decay. This is a minor overhead but increases the projection FLOPs.

- **Training still requires chunkwise parallel algorithm**: Unlike softmax attention which parallelizes trivially across the sequence, KDA requires the chunkwise algorithm with inter-chunk sequential state passing. This limits training parallelism to chunk-level granularity.

- **Not yet proven beyond 48B scale**: While scaling law experiments (653M–1.7B) show favorable curves, the largest validated model is 48B (3B active). Behavior at 70B+ dense models is unknown.

## Implementation Notes

```python
# KDA chunkwise forward — PyTorch pseudocode (Listing 8b from paper)
# Contrast with general DPLR (Listing 8a): fewer matmuls, no 1/Gamma

def chunk_kda(q, k, v, alpha, beta, chunk_size):
    """
    Kimi Delta Attention chunkwise parallel forward.

    Key simplifications vs general DPLR:
    - Lines 14-15: Only 2 Aqk accumulations (vs 4 for DPLR)
    - Lines 26,29: Only 2 state update matmuls (vs 5 for DPLR)
    - No secondary chunking needed (no 1/Gamma instability)
    """
    B, H, T, K = q.shape
    BT = chunk_size
    NT, S = T // BT, v.shape[-1]

    # Reshape into chunks
    q, k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d',
                                        c=BT), [q, k, v])

    # Cumulative decay (log-space for stability)
    gc = g.cumsum(-2)  # g = log(alpha)

    # Allocate accumulators
    Aqk, Akk = (torch.zeros(B, H, NT, BT, BT) for _ in range(2))

    for i in range(BT):
        k_i, q_i = k[:, :, :, i, None], q[:, :, :, i, None]
        g_i = gc[:, :, :, i:i+1, :]

        mask = (torch.arange(BT) <= i)[..., None]
        s1_i = (g_i - gc).exp().where(mask, 0)
        s2_i = (gc - g_i).exp()

        # KEY: Only 2 matmul accumulations (vs 4 for DPLR)
        Aqk[:, :, :, i, :] = (q_i * k + s1_i).sum(-1)  # ← simplified
        Akk[:, :, :, i, :] = (k_i * k + s2_i).sum(-1)  # ← simplified

    # UT transform via lower-triangular inverse
    A = Akk.masked_fill_(torch.triu(torch.ones(BT, BT), diagonal=0), 0)
    A[:, :, :, :, :i:i] = A[:, :, :, :, :i:i] + (A[:, :, :, :, :i].clone().sum(-1))
    mask = torch.triu(torch.ones(BT, BT), diagonal=-1)

    for i in range(0, NT):
        q_i, k_i, g_i, u_i, w_i = (x[:, :, i] for x in
            (q, k, gc, u, w))

        # Inter-chunk state contribution
        o[:, :, i] = (q_i * gc_i.exp()) @ S + Aqk @ (u_i - w_i @ S)

        # State update: only 2 outer products (vs 5 for DPLR)
        decay = (gc_i[:, :, -1, :, None] - gc_i).exp()
        S = S * gc_i[:, :, -1, :].exp()            # diagonal decay
        S += (k_i * decay).transpose(-1, -2) @ v_i  # rank-1 correction

    return o, S
```

**GPU efficiency analysis:**

1. **All dominant operations are matmuls**: The chunkwise algorithm's core operations — $QK^\top$ within chunks, $Q \cdot S$ for inter-chunk, and the UT transform — are all matrix-matrix multiplications that map to WGMMA/MMA tensor core instructions on H100/A100.

2. **Half-precision throughout**: Unlike GLA which requires log-domain secondary chunking in full precision (preventing FP16 tensor core use), KDA's constrained form avoids the $1/\Gamma$ reciprocal, enabling **full FP16/BF16 tensor core utilization**.

3. **Fixed state size = no KV-cache growth**: During generation, the recurrent state $\boldsymbol{S} \in \mathbb{R}^{128 \times 128}$ per head is constant regardless of sequence length. At 1M context: MLA needs ~75% of GPU memory for KV-cache; KDA needs effectively 0%.

4. **Arithmetic intensity**: The chunkwise parallel computation has arithmetic intensity proportional to chunk size $C$. With $C = 64$ and $d_k = d_v = 128$, the kernel is compute-bound on H100.

5. **Open-source kernels**: Available at https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda with vLLM integration.

## References

- Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. Technical Report. arXiv:2510.26692.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. arXiv:2111.00396.
- Bischof, C. & Van Loan, C. (1987). The WY Representation for Products of Householder Matrices. SIAM J. Sci. Stat. Comput.
- Code: https://github.com/MoonshotAI/Kimi-Linear
- Kernels: https://github.com/fla-org/flash-linear-attention

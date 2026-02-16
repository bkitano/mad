# 258: mLSTM Max-State Exponential Gate Stabilization

**Category**: stability
**Gain type**: efficiency
**Source**: Beck, Pöppel, Spanring et al. (2024) — xLSTM: Extended Long Short-Term Memory. NeurIPS 2024. arXiv:2405.04517. Milakov & Gimelshein (2018) — Online Normalizer Calculation for Softmax.
**Paper**: [papers/xlstm-exponential-gating.pdf]
**Documented**: 2026-02-16

## Description

Exponential gating replaces the sigmoid activation of LSTM gates with an exponential function: $i_t = \exp(\tilde{i}_t)$ and optionally $f_t = \exp(\tilde{f}_t)$, enabling unbounded gate magnitudes. This dramatically increases the dynamic range of memory storage and erasure — a single time step can write with strength $\exp(10) \approx 22000$ or forget with strength $\exp(-10) \approx 0.00005$. However, naively computing $f_t \cdot c_{t-1} + i_t \cdot z_t$ with exponential gates causes **immediate numerical overflow** since the accumulated products of exponential forget gates grow without bound.

The **max-state stabilization trick** (Milakov & Gimelshein, 2018, extended to gated recurrences) introduces a running maximum state $m_t$ that tracks the log-scale of the dominant gate contribution:

$$
m_t = \max\!\left(\log f_t + m_{t-1}, \, \log i_t\right)
$$

The stabilized gates $f'_t$ and $i'_t$ are then computed by subtracting $m_t$ before exponentiation:

$$
f'_t = \exp\!\left(\log f_t + m_{t-1} - m_t\right), \quad i'_t = \exp\!\left(\log i_t - m_t\right)
$$

This is mathematically equivalent to the original computation — replacing $f_t$ by $f'_t$ and $i_t$ by $i'_t$ changes neither the network output nor the gradients — but ensures all intermediate quantities remain $\leq 1$, preventing overflow. The trick is directly analogous to the "safe softmax" technique (subtract the max before exponentiating), generalized to sequential gated recurrences where the max evolves over time.

This stabilization is **essential** for both sLSTM and mLSTM in xLSTM. Without it, training produces NaN immediately. It also interacts critically with chunkwise-parallel formulations: in the parallel form, the stabilization becomes a row-wise max-subtraction on a log-space gate matrix, which is analogous to the online softmax trick used in FlashAttention.

## Mathematical Form

**Exponential Gating in sLSTM/mLSTM:**

The cell state update with exponential gates:

$$
c_t = f_t \, c_{t-1} + i_t \, z_t
$$

where $f_t = \sigma(\tilde{f}_t)$ or $\exp(\tilde{f}_t)$ (forget gate), $i_t = \exp(\tilde{i}_t)$ (input gate), and $z_t$ is the cell input.

**Problem:** After $T$ steps, the effective scale of $c_t$ involves $\prod_{s=1}^{T} f_s + \sum_s i_s \prod_{j>s} f_j$. With exponential gates, these products can reach $\exp(\sum_s \tilde{f}_s) \sim \exp(1000)$ — immediate overflow in any finite precision.

**Stabilizer State (Eq. 15 in the paper):**

$$
m_t = \max\!\left(\log f_t + m_{t-1}, \, \log i_t\right)
$$

**Stabilized Input Gate (Eq. 16):**

$$
i'_t = \exp\!\left(\log i_t - m_t\right) = \exp\!\left(\tilde{i}_t - m_t\right)
$$

**Stabilized Forget Gate (Eq. 17):**

$$
f'_t = \exp\!\left(\log f_t + m_{t-1} - m_t\right)
$$

**Proof of equivalence:** Since $m_t \geq \log f_t + m_{t-1}$ and $m_t \geq \log i_t$, both $f'_t \leq 1$ and $i'_t \leq 1$. The cell state becomes:

$$
c'_t = f'_t \, c'_{t-1} + i'_t \, z_t
$$

where $c'_t = c_t / \exp(m_t)$ is the rescaled cell state. The output $h_t = o_t \cdot \psi(c_t)$ can be recovered via $h_t = o_t \cdot \psi(\exp(m_t) \cdot c'_t)$. For mLSTM, which uses a normalizer state $n_t$ with the same exponential gate structure, the division $\bar{h}_t = C_t q_t / \max(|n_t^\top q_t|, 1)$ cancels the $\exp(m_t)$ factor, so the stabilized computation produces the correct output directly.

**Extension to mLSTM Matrix Memory:**

For mLSTM with matrix memory $C_t \in \mathbb{R}^{d \times d}$:

$$
C_t = f_t \, C_{t-1} + i_t \, v_t \, k_t^\top
$$

$$
n_t = f_t \, n_{t-1} + i_t \, k_t
$$

$$
\bar{h}_t = C_t \, q_t \,/\, \max\{|n_t^\top q_t|, 1\}
$$

The stabilizer state $m_t$ is the same scalar, and the normalizer $n_t$ is stabilized identically. The key insight is that $m_t$ is **per-head scalar**, not per-dimension, so it adds negligible overhead.

**Parallel Form — Chunkwise Stabilization:**

In the parallel (chunkwise) formulation of mLSTM, the intra-chunk gate matrix $D^{(k)} \in \mathbb{R}^{L \times L}$ has entries:

$$
D^{(k)}_{ij} = \begin{cases} \exp\!\left(\sum_{s=j+1}^{i} \log f_s + \log i_j\right) & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}
$$

Naively, the exponent $\sum_{s=j+1}^{i} \log f_s + \log i_j$ can be very large. The stabilization becomes a **row-wise max-subtraction**:

$$
\tilde{D}^{(k)}_{ij} = \exp\!\left(\sum_{s=j+1}^{i} \log f_s + \log i_j - \max_j\left(\sum_{s=j+1}^{i} \log f_s + \log i_j\right)\right)
$$

This is exactly the online softmax / safe-exp pattern applied to the gate matrix, ensuring all entries of $\tilde{D}^{(k)}$ are in $[0, 1]$.

**Key Definitions:**

- $m_t \in \mathbb{R}$ — stabilizer state (per-head scalar, tracks log-scale of dominant contribution)
- $f_t = \exp(\tilde{f}_t)$ or $\sigma(\tilde{f}_t)$ — forget gate (exponential or sigmoid)
- $i_t = \exp(\tilde{i}_t)$ — input gate (always exponential for enhanced dynamic range)
- $c'_t$ — stabilized cell state ($c_t / \exp(m_t)$)
- $D^{(k)} \in \mathbb{R}^{L \times L}$ — intra-chunk gate matrix in parallel form

## Complexity

| Operation | Without Stabilization | With Stabilization |
|-----------|----------------------|-------------------|
| Recurrent: per-step overhead | 0 | $O(1)$ max + 2 exp |
| Parallel: gate matrix | $O(L^2)$ exp (overflow!) | $O(L^2)$ exp + $O(L)$ row-max |
| Numerical range | $\exp(\sum \log f) \to \infty$ | All values in $[0, 1]$ |
| Precision requirements | FP64 (still overflows) | FP32 or BF16 (stable) |

**Memory:** $O(1)$ per head — just the scalar $m_t$. Negligible compared to the $O(d^2)$ matrix state.

**FLOPs:** The max and subtraction add $O(1)$ per time step in recurrent mode, $O(L)$ per chunk row in parallel mode. Both are negligible compared to the $O(d^2)$ matrix update.

**Interaction with TFLA (trick 158):** In TFLA's tiled computation, the max-state must be tracked across tiles within a chunk. The TFLA paper notes this as a complication for the exponential-gate mLSTM variant: "the max state tracks the maximum of the forget and input gates over time and is used to stabilize the exponential input gate." For the sigmoid-gate variant (mLSTMsig), no max-state tracking is needed, and TFLA's loop fusion is simpler. This is why TFLA's fastest kernels use sigmoid gates.

## Applicability

- **sLSTM (primary):** The stabilization was first introduced for the scalar-memory sLSTM with exponential gating and memory mixing. Essential for training stability.

- **mLSTM (primary):** Extended to the matrix-memory mLSTM, where the normalizer state $n_t$ provides automatic scale cancellation. Used in all xLSTM language modeling experiments (125M–1.3B parameters, 300B tokens).

- **Any exponential-gated recurrence:** The technique applies to any linear recurrence with $\exp(\cdot)$ gate activations, including exponential-gated variants of GLA, RetNet, or RWKV. The key requirement is that the gates appear multiplicatively in the recurrence.

- **Chunkwise-parallel linear attention with exponential gates:** In the parallel form, the stabilization becomes row-wise max-subtraction on the log-space gate matrix — the same pattern as online softmax (trick 083). This connects to GLA's secondary chunking (trick 177): GLA uses sigmoid gates ($\alpha_t \in (0,1)$) and works in log-space via cumulative sums; mLSTM uses exponential gates ($i_t \in (0, \infty)$) and requires the additional max-state for stabilization.

- **TFLA tiled kernels:** The TFLA paper's mLSTM implementation directly uses this stabilization within the tiled forward and backward passes. The max-state must be carried across tiles, adding complexity to the tiling strategy.

- **Online softmax connection:** The stabilizer state $m_t = \max(\log f_t + m_{t-1}, \log i_t)$ is the sequential analog of the running max in online softmax (Milakov & Gimelshein, 2018). In softmax attention, $m = \max_j(q \cdot k_j)$ prevents overflow in $\exp(q \cdot k_j - m)$. Here, $m_t$ prevents overflow in $\exp(\text{cumulative log-gate} - m_t)$. The parallel form's row-wise max is exactly the same as FlashAttention's block-wise max tracking.

## Limitations

- **Sequential bottleneck in max-state:** The recurrence $m_t = \max(\log f_t + m_{t-1}, \log i_t)$ is inherently sequential — $m_t$ depends on $m_{t-1}$. In the parallel/chunkwise form, the max must be computed as a prefix-max scan, which is parallelizable ($O(\log L)$ depth) but adds a scan dependency not present in sigmoid-gated models.

- **Complicates TFLA loop fusion:** As noted in TFLA (trick 158), the max-state tracking prevents efficient fusing of the inner KV loop with the output accumulation loop. The sigmoid-gate variant avoids this, achieving better kernel throughput. This is a fundamental tension: exponential gates are more expressive but harder to tile efficiently.

- **Forget gate initialization sensitivity:** The xLSTM paper finds that forget gate bias initialization in $[3, 6]$ is critical for good performance. Too small → fast forgetting, too large → vanishing input gate gradients. This initialization interacts with the stabilizer state in non-obvious ways.

- **Per-head scalar limitation:** The max-state $m_t$ is a single scalar per head. If different dimensions of the state have vastly different scales, a single max may over-suppress some dimensions. This is mitigated by the mLSTM's normalizer state $n_t$ but could be a concern for models with per-dimension exponential gates (e.g., full-matrix GLA with $\exp(\cdot)$ gates).

- **Backward pass complexity:** The gradient of the max function introduces non-smooth behavior. In practice, the straight-through estimator or the log-sum-exp smooth approximation is used, but this adds implementation complexity to the backward kernel.

## Implementation Notes

```python
# Max-state stabilization for exponential gating
# From xLSTM paper Equations 15-17

def stabilized_exponential_gating(f_log, i_log, z, c_prev, m_prev):
    """
    Stabilized sLSTM cell update with exponential gating.

    Args:
        f_log: (B, D) — log forget gate pre-activation (log f_t)
        i_log: (B, D) — log input gate pre-activation (log i_t = i_tilde)
        z: (B, D) — cell input
        c_prev: (B, D) — previous (stabilized) cell state
        m_prev: (B, D) — previous stabilizer state

    Returns:
        c_new: (B, D) — new stabilized cell state
        m_new: (B, D) — new stabilizer state
    """
    # Stabilizer state: track the log-scale maximum
    # m_t = max(log(f_t) + m_{t-1}, log(i_t))
    m_new = torch.maximum(f_log + m_prev, i_log)

    # Stabilized gates: subtract m_t before exponentiating
    # All values now <= 1 (since m_t >= both arguments)
    f_stable = torch.exp(f_log + m_prev - m_new)  # <= 1
    i_stable = torch.exp(i_log - m_new)             # <= 1

    # Standard LSTM update with stabilized gates
    c_new = f_stable * c_prev + i_stable * z

    return c_new, m_new


def stabilized_mlstm_cell(q, k, v, f_log, i_log, C_prev, n_prev, m_prev):
    """
    Stabilized mLSTM cell with matrix memory.

    Args:
        q, k, v: (B, D) — query, key, value
        f_log, i_log: (B,) — scalar log-gates per head
        C_prev: (B, D, D) — previous matrix memory
        n_prev: (B, D) — previous normalizer state
        m_prev: (B,) — previous stabilizer state

    Returns:
        h: (B, D) — output
        C_new, n_new, m_new — updated states
    """
    # Stabilizer state (scalar per head)
    m_new = torch.maximum(f_log + m_prev, i_log)

    # Stabilized gates
    f_stable = torch.exp(f_log + m_prev - m_new)
    i_stable = torch.exp(i_log - m_new)

    # Matrix memory update: C_t = f'_t * C_{t-1} + i'_t * v_t k_t^T
    C_new = f_stable.unsqueeze(-1).unsqueeze(-1) * C_prev + \
            i_stable.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bd,be->bde', v, k)

    # Normalizer update: n_t = f'_t * n_{t-1} + i'_t * k_t
    n_new = f_stable.unsqueeze(-1) * n_prev + i_stable.unsqueeze(-1) * k

    # Output: h_t = C_t q_t / max(|n_t^T q_t|, 1)
    h_unnorm = torch.einsum('bde,be->bd', C_new, q)
    denom = torch.clamp(torch.abs(torch.einsum('bd,bd->b', n_new, q)), min=1.0)
    h = h_unnorm / denom.unsqueeze(-1)

    return h, C_new, n_new, m_new


def parallel_mlstm_gate_matrix_stabilized(f_log, i_log, L):
    """
    Compute stabilized gate matrix D for chunkwise-parallel mLSTM.
    Analogous to online softmax / safe-exp pattern.

    Args:
        f_log: (L,) — log forget gates within chunk
        i_log: (L,) — log input gates within chunk

    Returns:
        D: (L, L) — stabilized gate matrix, all entries in [0, 1]
    """
    # Cumulative log-forget from position j+1 to i
    cum_f = torch.cumsum(f_log, dim=0)  # (L,)

    # Log-scale gate matrix: D_log[i,j] = cum_f[i] - cum_f[j] + i_log[j]
    D_log = cum_f.unsqueeze(1) - cum_f.unsqueeze(0) + i_log.unsqueeze(0)

    # Causal mask
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    D_log = D_log.masked_fill(~causal, -float('inf'))

    # Row-wise max stabilization (online softmax pattern!)
    D_max = D_log.max(dim=1, keepdim=True).values  # (L, 1)
    D = torch.exp(D_log - D_max)  # All entries in [0, 1]

    # The max values must be tracked for combining with inter-chunk terms
    # (same as FlashAttention's block-wise max tracking)
    return D, D_max

# KEY GPU INSIGHT:
# 1. Recurrent form: m_t update is O(1) per step — just a max + 2 exps
#    No extra memory, no extra kernel launches
# 2. Parallel form: row-wise max on D_log is O(L) per row
#    This is the SAME pattern as online softmax in FlashAttention
#    Can be fused into the matmul kernel (compute D_log tile, max, exp, matmul)
# 3. The sigmoid-gate variant (f_t = sigmoid) avoids max-state entirely
#    because sigmoid outputs are bounded [0,1] — no overflow possible
#    This is why TFLA's fastest kernels use mLSTMsig
# 4. Trade-off: exp gates have higher expressivity (unbounded strength)
#    but sigmoid gates have simpler/faster kernels
```

## References

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS 2024. arXiv:2405.04517.
- Milakov, M. & Gimelshein, N. (2018). Online Normalizer Calculation for Softmax. arXiv:1805.02867.
- Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels. NeurIPS 2025. arXiv:2503.14376.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. Neural Computation 12(10):2451–2471.

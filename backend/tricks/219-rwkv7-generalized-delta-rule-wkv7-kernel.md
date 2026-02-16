# 219: RWKV-7 Generalized Delta Rule with WKV7 Kernel

**Category**: parallelization
**Gain type**: expressivity
**Source**: Peng, Zhang, Goldstein et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution" (2025)
**Paper**: papers/rwkv7-goose-dynamic-state.pdf
**Documented**: 2026-02-15

## Description

RWKV-7 generalizes the delta rule for matrix-valued recurrent states, introducing a **non-diagonal, input-dependent transition matrix** that enables vector-valued gating, per-channel in-context learning rates, and decoupled removal/replacement keys. The resulting state update is still a **diagonal-plus-rank-one** matrix times the previous state, which preserves the structure needed for efficient parallelization via chunkwise training.

The core innovation is the state evolution:

$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1}\left(\text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t)\right) + v_t^T \hat{k}_t
$$

where: $w_t$ is a **vector-valued, data-dependent decay** (not scalar), $\hat{\kappa}_t$ is a normalized removal key, $a_t$ is a per-channel in-context learning rate, and $\hat{k}_t$ is a replacement key distinct from the removal key. This transition matrix approximates a scaled Householder-like matrix with eigenvalues in $[-1, 1]$, enabling the model to decay information in **all subspaces** (including negative eigenvalues) while remaining stable.

The key theoretical result is that RWKV-7 can solve problems **outside TC$^0$** — specifically recognizing all regular languages with a constant number of layers — surpassing Transformers under standard complexity conjectures. This comes from the "copy" state transition enabled by generalized eigenvalues.

The WKV7 CUDA kernel achieves **3× faster** training than RWKV-6 and **linear scaling** in sequence length, being significantly faster than Flash Attention v3 for sequences beyond 4K on H100 GPUs, while using constant memory at inference.

## Mathematical Form

**Weight preparation (per-token projections):**

$$
x_t^{\square} = \text{lerp}(x_t, x_{t-1}, \mu_\square) \quad \square \in \{r, k, v, d, a, g\}
$$

$$
a_t = \sigma(\text{loramlp}_a(\text{Identity}, x_t^a, \text{bias=True})) \qquad \text{(in-context learning rate, per-channel)}
$$

$$
k_t = x_t^k W_k, \quad \kappa_t = k_t \odot \xi, \quad \hat{\kappa}_t = \kappa_t / \|\kappa_t\|_2 \qquad \text{(removal key, normalized)}
$$

$$
\hat{k}_t = k_t \odot \text{lerp}(1, a_t, \alpha) \qquad \text{(replacement key)}
$$

$$
w_t = \exp(-e^{-0.5} \cdot d_t.\text{float}.\sigma()) \qquad \text{(decay, bounded in } (\exp(-e^{-0.5}), 1))
$$

**Core WKV state evolution:**

$$
\boldsymbol{wkv}_0 = \boldsymbol{0}
$$

$$
\boldsymbol{wkv}_t = \boldsymbol{wkv}_{t-1}\left(\text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t)\right) + v_t^T \cdot \hat{k}_t
$$

where $\boldsymbol{wkv}_t \in \mathbb{R}^{(D/h) \times (D/h)}$ is the per-head state matrix (e.g., $64 \times 64$).

**Transition matrix analysis:**

$$
G_t = \text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t) = \left(\boldsymbol{I} - \hat{\kappa}_t^T\left(\frac{a_t}{w_t} \odot \hat{\kappa}_t\right)\right) \text{diag}(w_t)
$$

$$
\approx \left(\boldsymbol{I} - 2\hat{\kappa}_t^T \hat{\kappa}_t\right) \text{diag}(w_t)
$$

This approximates a Householder reflection times a diagonal decay, giving eigenvalues in $[-1, 1]$ (not just $[0, 1]$ like prior models). This enables the critical "copy" operation for state tracking.

**Parallel form (attention-like):**

$$
\boldsymbol{wkv}_t = \sum_{i=1}^{t} \left(v_i^T \hat{k}_i \prod_{j=i+1}^{t} \left(\text{diag}(w_j) - \hat{\kappa}_j^T (a_j \odot \hat{\kappa}_j)\right)\right)
$$

**WKV output (bonus + attention result):**

$$
u_t = \left(r_t \cdot (\rho \odot \hat{k}_t)^T\right) v_t \qquad \text{(bonus: current-token self-attention)}
$$

$$
p_t = \text{LayerNorm}(r_t \boldsymbol{wkv}_t^T) + u_t \qquad \text{(attention result)}
$$

$$
o_t = (g_t \odot p_t) W_o \qquad \text{(gated output)}
$$

**Key Definitions:**

- $\boldsymbol{S}_t \in \mathbb{R}^{(D/h) \times (D/h)}$ — per-head state matrix (fast weights / key-value memory)
- $w_t \in \mathbb{R}^{D/h}$ — vector-valued, data-dependent decay ("in-context weight decay")
- $a_t \in (0, 1)^{D/h}$ — per-channel in-context learning rate
- $\hat{\kappa}_t \in \mathbb{R}^{D/h}$ — normalized removal key (what to erase from state)
- $\hat{k}_t \in \mathbb{R}^{D/h}$ — replacement key (what to write into state)
- $v_t \in \mathbb{R}^{D/h}$ — value vector
- $r_t \in \mathbb{R}^{D/h}$ — receptance (query analog)
- $h$ — number of heads, $D/h$ = head dimension (typically 64)

## Complexity

| Operation | Softmax Attention | RWKV-7 (recurrent) | RWKV-7 (chunkwise) |
|-----------|------------------|---------------------|---------------------|
| Time per token (inference) | $O(N)$ (KV cache) | $O(D/h)$ = $O(1)$ w.r.t. $N$ | — |
| Training (full sequence) | $O(N^2 D)$ | $O(N (D/h)^2 h)$ = $O(N D^2/h)$ | $O(N D^2/h)$ parallel |
| Memory (inference) | $O(N D)$ (KV cache) | $O(h (D/h)^2)$ = $O(D^2/h)$ constant | — |
| Memory (training) | $O(N D)$ | $O(N D)$ | $O(N D)$ |

**Measured speed (H100 SXM, batch=8, D=4096, head_dim=64):**

| Sequence length | RWKV-7 bf16 | RWKV-6 | Flash Attention v3 |
|----------------|-------------|--------|-------------------|
| 4K | ~5 ms | ~12 ms | ~5 ms |
| 8K | ~10 ms | ~25 ms | ~15 ms |
| 16K fwd+bwd | ~30 ms | ~100 ms | ~55 ms |

RWKV-7 is ~3× faster than RWKV-6 and crosses over Flash Attention v3 around 4-8K sequence length.

## Applicability

- **Language modeling**: 2.9B model achieves 3B SoTA on multilingual benchmarks, competitive with attention-based models trained on far more tokens.
- **State tracking**: Can recognize all regular languages in a constant number of layers (beyond TC$^0$), unlike Transformers and diagonal-only linear RNNs.
- **Long-context inference**: Constant memory and time per token during autoregressive generation (no KV cache growth).
- **Multimodal**: VisualRWKV-7 demonstrated on vision-language tasks.
- **Drop-in replacement**: Compatible with the Flash Linear Attention (fla) library for larger head dimensions.
- **Upgrading existing models**: RWKV-5/6 checkpoints can be upgraded to RWKV-7 without pretraining from scratch.

## Limitations

- **Still sequential for non-chunkwise training**: The diagonal-plus-rank-one structure admits parallelization, but the optimized CUDA kernel (WKV7) currently processes timesteps sequentially within chunks. The fla library provides a chunkwise variant for larger head dims.
- **Head dimension sensitivity**: Official CUDA kernels optimized for head_dim=64. Efficiency drops for head_dim=128+.
- **More memory than RWKV-6**: Requires 18 variable-equivalents of memory (vs. 10 for RWKV-6 and Flash Attention v3) due to additional vectors ($\hat{\kappa}$, $a \odot \hat{\kappa}$, etc.).
- **Training stability**: Restricts eigenvalue range to $(\exp(-e^{-0.5}), 1)$ for large-scale models to avoid training instabilities with negative eigenvalues.
- **Numerical precision**: WKV7 kernel sensitive to precision; bf16 kernel is faster but fp32 kernel was used for released models.
- **Not tensor-core-dominated**: The per-timestep recurrence involves matrix-vector products (state × removal_key outer product), not large GEMMs. The arithmetic intensity is lower than chunkwise approaches that batch into matmuls.

## Implementation Notes

```python
# RWKV-7 WKV7 Kernel — Core State Evolution (PyTorch reference)
# From Appendix G/H of the paper

def wkv7_forward(r, w, k, v, a, b, state):
    """
    r: [B, T, H, N] — receptance (query)
    w: [B, T, H, N] — decay (pre-exponentiated: w0, then w = exp(-exp(w0)))
    k: [B, T, H, N] — key
    v: [B, T, H, N] — value
    a: [B, T, H, N] — in-context learning rate * removal_key  (= iclr * removal_k)
    b: [B, T, H, N] — replacement key (= k̂)
    state: [B, H, N, N] — initial wkv state (64×64 per head)
    """
    B, T, H, N = r.shape
    w = torch.exp(-torch.exp(w))  # decay in (exp(-e^{-0.5}), 1)
    out = torch.zeros((B, T, H, N))

    for t in range(T):
        # State transition: S_t = S_{t-1} * diag(w_t)
        #                        - S_{t-1} @ removal_k_t @ (iclr_t * removal_k_t).T
        #                        + v_t @ replacement_k_t.T
        state = (
            state * w[:, t, :, None, :]           # diagonal decay
            + torch.einsum('bhik,bhk,bhj->bhij',  # rank-1 removal
                           state, a[:, t, :], b[:, t, :])
            + torch.einsum('bhj,bhi->bhij',       # rank-1 addition
                           k[:, t, :], v[:, t, :])
        )
        # Output: r @ state^T → [B, H, N]
        out[:, t, :] = torch.einsum('bhj,bhij->bhi', r[:, t, :], state)

    return out, state

# Key GPU optimization (actual CUDA kernel):
# - State (64×64) kept in registers/SRAM across all T timesteps
# - w, k, v, a, b streamed from HBM one timestep at a time
# - Rank-1 updates fused with diagonal scaling
# - Output written once per timestep
# - Chunks of 16 timesteps for intermediate state checkpointing
```

## References

- Peng, Zhang, Goldstein et al. "RWKV-7 'Goose' with Expressive Dynamic State Evolution." arXiv:2503.14456, 2025. COLM 2025.
- GitHub: https://github.com/RWKV/RWKV-LM
- Models: https://huggingface.co/RWKV
- Yang & Zhang. "Flash Linear Attention." https://github.com/sustcsonglin/flash-linear-attention, 2024.
- Schlag, Irie & Schmidhuber. "Linear Transformers Are Secretly Fast Weight Programmers." ICML 2021.
- Yang, Goldstein, Peng, Alcaide, Poli & Song. "Parallelizing Linear Transformers with the Delta Rule." arXiv:2406.06484, 2024.
- Peng et al. "Eagle and Finch: RWKV with Matrix-Valued States." EMNLP Findings 2024.

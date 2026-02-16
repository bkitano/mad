# 254: Diffusion Forcing Parallel Sampler for Recurrent-Depth Models

**Category**: parallelization
**Gain type**: efficiency
**Source**: Geiping, Yang & Su. "Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models." ELLIS/MPI/CMU. arXiv:2510.14961, October 2025. NeurIPS 2025.
**Paper**: papers/diffusion-forcing-parallel-recurrent-depth.pdf
**Documented**: 2026-02-16

## Description

Recurrent-depth (looped/universal) transformers apply the same transformer block $r$ times at each token position before advancing to the next token. This creates an $O(r)$ sequential bottleneck during autoregressive generation: for each new token, you must run $r$ full forward passes before decoding. This trick **parallelizes generation across the sequence dimension** by advancing the wavefront diagonally — computing one recurrence step per token position per forward pass, so that after $r$ forward passes the model has produced $r$ tokens instead of just 1.

The key insight is that recurrent-depth models are structurally equivalent to **latent-space diffusion models**: starting from random noise $\mathbf{s}_0$, the recurrent block iteratively "denoises" the latent state until it converges to a decodable token. This connection enables applying **diffusion forcing** — a technique from diffusion models where the sampler advances along both the sequence dimension and the denoising dimension simultaneously, producing token drafts from intermediate (partially converged) recurrence iterates.

The sampler achieves **4-5x speedup** on A100 GPUs over standard autoregressive generation with only ~1% accuracy trade-off, applied zero-shot to existing pretrained 3.5B recurrent-depth models without any fine-tuning. It outperforms both adaptive compute and speculative decoding baselines.

## Mathematical Form

**Recurrent-Depth Model Architecture:**

A recurrent-depth transformer has three blocks:
- Prelude $\mathcal{P}$: embeds input tokens into latent space
- Recurrent block $\mathcal{R}$: applied $r$ times to refine latent state
- Coda $\mathcal{C}$: projects final latent state to token logits

$$
\mathbf{e} = \mathcal{P}(\mathbf{x})
$$

$$
\mathbf{s}_0 \sim \mathcal{N}(0, \sigma^2 I)
$$

$$
\mathbf{s}_i = \mathcal{R}(\mathbf{e}, \mathbf{s}_{i-1}) \quad \text{for } i \in \{1, \ldots, r\}
$$

$$
\mathbf{p} = \mathcal{C}(\mathbf{s}_r)
$$

**Sequential Baseline Cost:**

To generate $w$ tokens: total forward passes = $(r+1) \cdot w$ (each token needs $r$ recurrence steps + 1 decode).

**Diffusion Forcing Sampler:**

Instead of fully converging at each position, advance diagonally:

At each sampler step $t = 1, \ldots, T$:

1. Compute input conditioning with momentum:
$$
\mathbf{e} = \eta \, \mathbf{e}_{\text{prev}} + (1 - \eta) \, \mathcal{P}(y_{\text{current}})
$$

2. Optionally inject noise for stability:
$$
\mathbf{z}' = (1 - \beta_t) \mathbf{z} + \beta_t \, \mathbf{z}_{\text{noise}} \quad \text{where } \mathbf{z}_{\text{noise}} = \text{InitState}(1, \alpha)
$$

3. Apply $r'$ inner recurrence steps ($r' \ll r$, typically $r' = 2$-$4$):
$$
\mathbf{z} \leftarrow \mathcal{R}(\mathbf{e}, \mathbf{z}) \quad \text{for } j = 1, \ldots, r'
$$

4. Decode draft token:
$$
\mathbf{p} \leftarrow \mathcal{C}(\mathbf{z}), \quad y \sim \text{Sample}(\mathbf{p})
$$

5. Freeze converged positions and advance wavefront.

**Adaptive Exit Criterion:**

Freeze position $i$ when the normalized latent distance falls below threshold $\varepsilon$:

$$
\delta_i = \frac{\|\mathbf{z}_i - \mathbf{z}_{\text{prev},i}\|_2}{\|\mathbf{z}_i\|_2}, \quad k^* = \max\{k : \delta_j < \varepsilon \text{ for all } j \leq k\}
$$

All positions up to $k^*$ are frozen and committed to the KV cache.

**Noise Schedule:**

$\beta_t$ is scheduled linearly, decreasing over recurrence steps so later iterates are less noisy:

$$
\beta_t = \beta_{\max} \cdot \frac{r - t}{r}
$$

**FLOP Cost:**

For generating $w$ tokens with non-adaptive sampler:

$$
\text{Total FLOPs} = \left(r + \frac{r}{r'}\right) f \cdot w
$$

vs. $(r+1) f \cdot w$ for baseline, where $f$ is the FLOP cost of one forward pass through $\mathcal{R}$. The FLOP overhead is offset by parallelization gains since the wavefront of $W$ tokens is processed simultaneously.

## Complexity

| Operation | Sequential AR | Diffusion Forcing Sampler |
|-----------|---------------|---------------------------|
| Forward passes per token | $r + 1$ | $r' + 1$ (amortized, $r' \ll r$) |
| Tokens per forward pass | $1$ | $W$ (wavefront, $W \leq 128$) |
| Wall-clock for $w$ tokens | $O(w \cdot r)$ | $O(w \cdot r / W)$ |
| Total FLOPs | $(r+1)fw$ | $(r + r/r')fw$ (slightly more) |

**Memory:** With KV cache sharing, memory equals that of a fixed-depth transformer — only the most recent KV state per position is stored ($r$-fold reduction vs. storing all recurrence KV states).

## Applicability

- **Recurrent-depth / looped / universal transformers**: Any model that applies transformer blocks repeatedly at each position (e.g., Huginn, Coconut, RingFormer)
- **Requirements for the sampler** (no retraining needed):
  1. **Input injection**: Recurrent block must be conditioned on input embeddings $\mathbf{e}$ (not just previous hidden state)
  2. **Robust recurrence**: Intermediate iterates must be approximately decodable (not just the final iterate)
  3. **KV cache sharing**: Different recurrence depths should share KV cache (fungible KV states)
- **Inference-only optimization**: Applied at generation time, no training changes needed
- Works with existing 3.5B recurrent-depth checkpoints (Huginn-0125)

## Limitations

- **Inference only**: Does not speed up training (training is already parallel over sequence positions via teacher forcing)
- **Slight FLOP overhead**: Total FLOPs increase by factor $(r + r/r') / (r+1) \approx 1 + 1/r'$, offset by parallelism
- **Not applicable to standard (non-recurrent) transformers**: Requires depth-recurrence as the computational primitive
- **Accuracy-speed trade-off**: ~1% accuracy loss at 5x speedup; configurable via $\varepsilon$, $r'$, $\beta_t$, $\eta$
- **Wavefront size limited by GPU memory**: Optimal $W = 64$-$128$ on A100, hardware-dependent
- **Causality constraint**: Information propagates left-to-right; late-changing conditioning from earlier tokens can require adaptive stalling
- **Batch size 1 only**: Current evaluation at batch size 1; extensions to larger batches not yet explored

## Implementation Notes

```python
# Simplified diffusion forcing sampler for recurrent-depth models
def diffusion_forcing_generate(model, prompt_tokens, max_new_tokens,
                                r_inner=4, r_total=32, epsilon=0.03,
                                eta=0.1, beta_max=0.0, wavefront_max=128):
    """
    model: recurrent-depth transformer with .prelude(), .recur(), .coda()
    r_inner: inner recurrence steps per sampler iteration (r')
    r_total: total recurrence budget per token (r)
    """
    y_frozen = prompt_tokens
    y_current = prompt_tokens
    z = init_random_state(alpha=1.0)  # Random latent for first new position

    for t in range(1, r_total * max_new_tokens):
        # 1. Compute conditioning with momentum
        e = eta * e_prev + (1 - eta) * model.prelude(y_current)

        # 2. Optional noise injection (linear schedule)
        beta_t = beta_max * (r_total - t % r_total) / r_total
        z_noise = init_random_state(alpha=1.0)
        z = (1 - beta_t) * z + beta_t * z_noise

        # 3. Inner recurrence (r' steps, all parallelizable across wavefront)
        for j in range(r_inner):
            z = model.recur(z, e)  # Single forward pass over wavefront W positions

        # 4. Decode token drafts
        logits = model.coda(z)
        y = sample(logits)
        y_current = concat(y_frozen, y)

        # 5. Freeze converged positions (adaptive exit)
        # Compare latent states to previous iteration
        delta = norm(z - z_prev) / norm(z)
        k_star = max_contiguous_below_threshold(delta, epsilon)
        y_frozen = y_current[:k_star]

        # Commit frozen tokens to KV cache, remove from active wavefront
        if len(y_frozen) - len(prompt_tokens) >= max_new_tokens:
            break

        # Append new random latent state for next position
        z = concat(z[:active_positions], init_random_state(alpha=1.0))

    return y_frozen

# GPU efficiency notes:
# - Each "model.recur(z, e)" processes W token positions in parallel
#   via standard transformer attention over the wavefront
# - All operations are matmuls (attention QKV projections, FFN)
#   -> full tensor core utilization
# - KV cache sharing means memory = fixed-depth transformer
# - On A100: ~182 tokens/sec vs ~36 tokens/sec baseline (5.05x speedup)
# - Wavefront W=64-128 saturates A100 SMs at batch_size=1
```

**Hardware considerations:**
- Memory-bound at batch size 1 (I/O cost of loading model params dominates)
- Wavefront parallelism turns single-token decode into multi-token prefill-like computation
- Shared KV cache means memory is independent of recurrence depth $r$
- All operations are standard transformer matmuls — fully tensor-core compatible
- On A100-40GB: 4.36x (GSM8K), 4.73x (MATH500), 4.81x (HumanEval), 4.59x (MBPP) speedup

## References

- Geiping, Yang & Su, "Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models," arXiv:2510.14961, 2025.
- Geiping et al., "Scaling Up Masked Diffusion Models on Text," 2025 (Huginn model).
- Chen et al., "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion," 2024.
- Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models," 2025.
- Schwarzschild et al., "Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks," 2021.

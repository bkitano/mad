# 226: SPAM — Spike-Aware Adam with Momentum Reset

**Category**: stability
**Gain type**: efficiency
**Source**: Huang et al., "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training" (ICLR 2025)
**Paper**: papers/spam-spike-aware-adam.pdf
**Documented**: 2026-02-15

## Description

Gradient spikes — sudden bursts where individual gradient magnitudes exceed their running average by $50\text{--}1000\times$ — are a pervasive phenomenon during LLM training. These spikes corrupt Adam's exponential moving averages (first and second moments), with the contamination persisting for $50\text{--}200+$ steps due to the slow decay rates ($\beta_1 = 0.9$, $\beta_2 = 0.999$). SPAM introduces two complementary mechanisms to neutralize spike damage: **periodic momentum reset** to flush corrupted state, and **spike-aware gradient clipping** that preserves directional information while controlling magnitude. Together, these yield consistent perplexity improvements of $0.5\text{--}1.5$ points across model scales from 60M to 1B parameters, while also enabling a memory-efficient sparse momentum variant.

## Mathematical Form

**Gradient Spike Score (GSS):**

$$
\text{GSS}(g_i) = \frac{|g_i|}{\frac{1}{T+1} \sum_{j=0}^{T} |g_j|}
$$

A gradient $g_i$ is classified as spiked if $\text{GSS}(g_i) > \theta$ (default $\theta = 5000$).

**Efficient On-the-Fly Spike Detection (using Adam's second moment):**

$$
\mathcal{G} = \left\{ g_i \;\middle|\; \frac{g_i^2}{V_i} > \theta \right\}
$$

where $V_i$ is Adam's second moment estimate (a running average of $g_i^2$). This avoids storing full gradient history — the ratio $g_i^2 / V_i$ approximates the GSS.

**Spike-Aware Clipping:**

For each detected spike gradient $g_i \in \mathcal{G}$:

$$
g_i \leftarrow \text{sign}(g_i) \cdot \sqrt{\theta \cdot V_i}
$$

This rescales the gradient to a manageable magnitude while preserving its sign (directional information).

**Momentum Reset:**

Every $\Delta T$ steps, reset both Adam moments:

$$
m_t \leftarrow 0, \quad v_t \leftarrow 0
$$

followed by $N$ steps of cosine learning rate warmup to stabilize the optimizer after reset.

**Full SPAM Update (Algorithm 1):**

Given parameters $\theta_t$, gradient $g_t$, learning rate $\alpha_t$:

1. **Detect spikes**: $\mathcal{G}_t = \{i : g_{t,i}^2 / v_{t-1,i} > \theta\}$
2. **Clip spikes**: $g_{t,i} \leftarrow \text{sign}(g_{t,i}) \cdot \sqrt{\theta \cdot v_{t-1,i}}$ for $i \in \mathcal{G}_t$
3. **Update moments**: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $\;v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4. **Adam step**: $\theta_t = \theta_{t-1} - \alpha_t \cdot m_t / (\sqrt{v_t} + \epsilon)$
5. **If** $t \bmod \Delta T = 0$: reset $m_t \leftarrow 0$, $v_t \leftarrow 0$; apply $N$-step cosine warmup

**Key Definitions:**

- $g_t \in \mathbb{R}^d$ — gradient at step $t$
- $m_t, v_t \in \mathbb{R}^d$ — first and second moment estimates
- $\theta$ — GSS threshold for spike detection (default: 5000)
- $\Delta T$ — momentum reset interval (default: 500 steps)
- $N$ — cosine warmup steps after reset (default: 150)

## Complexity

| Operation | Naive (Adam) | With SPAM |
|-----------|-------------|-----------|
| Per-step compute | $O(d)$ | $O(d)$ (same — only adds elementwise comparisons) |
| Memory (full) | $2d$ (two moments) | $2d$ (identical to Adam) |
| Memory (sparse, $p\%$) | $2d$ | $2 \cdot p\% \cdot d$ (sparse momentum variant) |

**Memory:** With sparse momentum ($d\% = 0.25\%$), SPAM reduces optimizer memory from $2d$ to $\approx 0.005d$ while outperforming GaLore and Adam-Mini.

**Overhead:** Negligible — spike detection is a single elementwise division and comparison against the already-computed $v_t$. No additional kernel launches required.

## GPU Efficiency Analysis

**Memory Access Pattern:** Fully coalesced — all operations (spike detection, clipping, moment update) are elementwise on contiguous parameter tensors. Identical memory access pattern to standard Adam.

**Parallelism:** Embarrassingly parallel across all $d$ parameters. No inter-parameter dependencies. Maps directly to existing Adam CUDA kernels with minimal modification (add spike check + periodic reset logic).

**Arithmetic Intensity:** Same as Adam — dominated by the optimizer step, which is memory-bandwidth-bound. SPAM adds ~3 elementwise ops per parameter (division, comparison, conditional multiply), negligible vs. the Adam update itself.

**Tensor Core Usage:** N/A — optimizer steps are elementwise, not matmul. Same as Adam.

**Integration:** Drop-in replacement for Adam. Can be fused into existing fused Adam kernels (e.g., `apex.optimizers.FusedAdam`) by adding the spike check inside the same kernel. Zero additional kernel launches.

## Applicability

- **All transformer LLMs**: Validated on LLaMA 60M–1B, Pythia-70M architectures
- **Pre-training and fine-tuning**: Consistent gains on both; fine-tuning LLaMA2-7B on Commonsense170K shows 64.4→66.7 avg accuracy
- **Memory-constrained training**: Sparse momentum variant matches or exceeds GaLore, LoRA, Adam-Mini at equivalent memory budgets
- **Mixed-precision training**: Especially beneficial in BF16 where gradient representation range is limited
- **Time series forecasting**: Also validated beyond NLP (Appendix G)

## Limitations

- **Hyperparameter sensitivity**: Reset interval $\Delta T$ and threshold $\theta$ require tuning — too-frequent resets ($\Delta T < 100$) or too-low threshold ($\theta < 100$) degrade performance
- **Warmup overhead**: The $N$-step cosine warmup after each reset temporarily reduces effective learning rate
- **Root cause not addressed**: SPAM treats the symptom (corrupted moments) rather than the cause (why spikes occur — likely related to edge-of-stability dynamics)
- **Sparse momentum randomness**: Random subset selection for sparse momentum works best empirically, but lacks theoretical justification for why gradient-magnitude or weight-magnitude selection strategies perform worse

## Implementation Notes

```python
# Core SPAM logic (simplified, fused into Adam step)
def spam_step(params, grads, m, v, lr, beta1, beta2, eps,
              theta=5000, step=0, delta_T=500, warmup_N=150):
    # 1. Spike-aware clipping (elementwise, no extra memory)
    spike_mask = (grads ** 2) / (v + eps) > theta
    clipped_grads = torch.where(
        spike_mask,
        torch.sign(grads) * torch.sqrt(theta * (v + eps)),
        grads
    )

    # 2. Standard Adam moment update
    m.mul_(beta1).add_(clipped_grads, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(clipped_grads, clipped_grads, value=1 - beta2)

    # 3. Adam parameter update
    denom = v.sqrt().add_(eps)
    params.addcdiv_(m, denom, value=-lr)

    # 4. Periodic momentum reset
    if step > 0 and step % delta_T == 0:
        m.zero_()
        v.zero_()
        # Apply cosine warmup for next N steps externally
```

## References

- Huang et al., "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training", ICLR 2025. arXiv:2501.06842
- Code: https://github.com/TianjinYellow/SPAM-Optimizer.git
- Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
- Shazeer & Stern, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost", ICML 2018
- Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", ICML 2024

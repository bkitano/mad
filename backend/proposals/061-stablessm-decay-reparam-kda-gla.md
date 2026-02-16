---
status: ongoing
priority: high
created: 2026-02-15
based_on: stablessm-gradient-balanced-reparameterization (233), kda-constrained-dplr-delta-chunkwise (211), gla-secondary-chunking-log-space-gating (177), gated-deltanet-chunkwise-wy-gating (203), qk-norm-softmax-capping-llm-stability (215)
experiment_number: 061
experiment_log: experiment-log-061.md
---

# Gradient-Balanced Decay Reparameterization for KDA and Gated Linear Attention

## Hypothesis

Replacing the standard sigmoid parameterization of per-channel decay gates $\boldsymbol{\alpha}_t = \sigma(\boldsymbol{W}_\alpha \boldsymbol{x}_t)$ in KDA and GLA with StableSSM's **gradient-balanced reparameterization** $\boldsymbol{\alpha}_t = 1 - 1/(a \cdot f(\boldsymbol{x}_t)^2 + b)$ will:

1. Enable **$2$–$5\times$ higher peak learning rates** without NaN divergence during pretraining
2. Achieve **$0.3$–$0.8\%$ lower final perplexity** at fixed compute budget (due to faster convergence from balanced gradient landscape)
3. Add **zero wall-clock overhead** (the reparameterization is a cheap elementwise function replacing the existing sigmoid)

The core insight: sigmoid gates exhibit the same **gradient imbalance** as SSM eigenvalues — channels with $\alpha \to 1$ (long memory) have vanishing gradients through $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \to 0$, while channels with $\alpha \to 0$ (short memory) also have vanishing gradients. The "useful" gradient region is concentrated around $\alpha \approx 0.5$, creating an optimization bottleneck analogous to StableSSM's curse of memory.

## Background

### The decay gate gradient problem in linear RNNs

KDA, GLA, and Gated DeltaNet all use per-channel or per-head decay gates that control how quickly the hidden state forgets past information:

$$
\boldsymbol{S}_t = \text{Diag}(\boldsymbol{\alpha}_t) \boldsymbol{S}_{t-1} + (\text{input update})
$$

where $\boldsymbol{\alpha}_t \in [0, 1]^{d_k}$ determines the memory retention per channel. These gates are typically parameterized as:

$$
\boldsymbol{\alpha}_t = \sigma(\boldsymbol{W}_\alpha \boldsymbol{x}_t + \boldsymbol{b}_\alpha)
$$

**Problem 1: Sigmoid saturation.** The sigmoid function $\sigma(z) = 1/(1+e^{-z})$ has derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which vanishes for $|z| \gg 0$. For channels that learn long-range dependencies ($\alpha \to 1$, i.e., $z \gg 0$), the gradient through the gate is exponentially suppressed. This is the same "curse of memory" that StableSSM identified for SSM eigenvalues.

**Problem 2: Gradient imbalance across channels.** In a well-trained model, different channels specialize to different time scales — some channels retain information for hundreds of tokens ($\alpha \approx 0.99$), others forget quickly ($\alpha \approx 0.5$). The sigmoid parameterization creates a **gradient imbalance**: the long-memory channels get orders of magnitude smaller gradients than the short-memory channels, making it harder for the optimizer to adjust them.

**Problem 3: Interaction with learning rate.** The gradient imbalance forces the use of conservative learning rates. If the learning rate is too high, the short-memory channels (which receive large gradients) diverge, even though the long-memory channels need larger updates to learn. This constrains the maximum useful learning rate.

### StableSSM's insight

StableSSM proved that the **optimal** mapping from trainable weight $w$ to eigenvalue $\lambda$ minimizes the worst-case gradient-over-weight ratio. For continuous-time SSMs with eigenvalues in $(-\infty, 0)$:

$$
f^*(w) = -\frac{1}{aw^2 + b}
$$

This achieves a gradient scale of $|f'(w)|/f(w)^2 = 2a|w|$ — **linear in $|w|$**, meaning all modes receive proportionally balanced gradients. The key property is that the gradient doesn't vanish or explode for any mode.

### Adaptation to decay gates

For decay gates in $[0, 1]$, we need a discrete-time version. StableSSM provides:

$$
f^*(w) = 1 - \frac{1}{aw^2 + b}
$$

which maps $\mathbb{R} \to (-1, 1)$ with gradient scale $2a|w|$ — the same balanced property. We restrict to $[0, 1]$ by taking $\alpha = \max(0, f^*(w))$ or by using $\alpha = 1 - 1/(aw^2 + b)$ with appropriate $(a, b)$ to ensure the range is $[0, 1)$.

### What ReGLA does differently

ReGLA (NAACL 2025) addresses gate saturation with a **composite gating** mechanism that adds a "refining module" — essentially an additive correction to the saturated gate. This adds parameters and a small compute cost. Our approach is fundamentally different: we change the parameterization function itself (sigmoid → gradient-balanced), adding zero parameters and zero compute, while achieving the same goal of preventing gradient vanishing at saturation.

### Gap being filled

- **StableSSM** was validated on data-independent SSM eigenvalues (S4D, S5, LRU) — not on data-dependent gates
- **ReGLA** addresses gate saturation via architectural modification — not via reparameterization
- **No existing work** applies gradient-balanced reparameterization to the per-channel data-dependent decay gates used in KDA, GLA, or Gated DeltaNet

## Related Work

- **StableSSM** (Wang & Li, ICML 2024): Proved the optimal eigenvalue reparameterization for diagonal SSMs and demonstrated 1000× higher stable learning rates. Applied to S4D, S5, LRU — all with **data-independent** eigenvalues. Our work extends this to **data-dependent** gates where the weight-to-eigenvalue mapping includes a learned projection from the input.
- **ReGLA** (Lu et al., NAACL 2025): Addressed gate saturation in GLA via a composite gating module. Adds parameters ($d \times d_k$ for the refining gate) and FLOPs. Our approach achieves similar gradient flow benefits with zero parameter/compute overhead by changing only the activation function.
- **Gated Attention** (Qwen3-Next, NeurIPS 2025 Best Paper): Applied post-attention sigmoid gating to softmax attention. Uses standard sigmoid — could also benefit from gradient-balanced reparameterization.
- **LRU** (Orvieto et al., ICML 2023): Used exponential parameterization for recurrent eigenvalues. StableSSM showed the "best" parameterization improves over exponential.
- **Kimi Linear** (Moonshot AI, 2025): Introduced KDA with per-channel decay via low-rank projection + sigmoid. Does not explore alternative parameterizations for the decay gate.

No directly related work found combining gradient-balanced reparameterization with data-dependent per-channel decay gates in chunkwise linear RNNs.

## Mathematical Formulation

**Standard sigmoid decay gate (baseline):**

$$
\boldsymbol{\alpha}_t = \sigma(\boldsymbol{z}_t), \quad \boldsymbol{z}_t = \boldsymbol{W}_\alpha^{\uparrow} \boldsymbol{W}_\alpha^{\downarrow} \boldsymbol{x}_t + \boldsymbol{b}_\alpha \in \mathbb{R}^{d_k}
$$

Gradient through gate:

$$
\frac{\partial \alpha_{t,i}}{\partial z_{t,i}} = \sigma(z_{t,i})(1 - \sigma(z_{t,i}))
$$

For $\alpha_{t,i} \to 1$ ($z_{t,i} \to +\infty$): $\frac{\partial \alpha}{\partial z} \to 0$ (vanishing gradient for long-memory channels).

For $\alpha_{t,i} \to 0$ ($z_{t,i} \to -\infty$): $\frac{\partial \alpha}{\partial z} \to 0$ (vanishing gradient for short-memory channels).

**Proposed gradient-balanced decay gate:**

$$
\boldsymbol{\alpha}_t = \phi(\boldsymbol{z}_t), \quad \phi(z) = \max\left(0, \; 1 - \frac{1}{a z^2 + b}\right)
$$

with $a = 1, b = 1$ (ensuring $\phi(0) = 0$ and $\phi(z) \to 1$ as $|z| \to \infty$).

Gradient:

$$
\frac{\partial \phi}{\partial z} = \frac{2az}{(az^2 + b)^2}
$$

**Gradient scale comparison** (the key metric from StableSSM):

For sigmoid, the gradient-over-weight ratio $|\sigma'(z)|/\sigma(z)^2 = (1 - \sigma(z))/\sigma(z)$, which diverges as $\sigma \to 0$ and vanishes as $\sigma \to 1$.

For the balanced function, $|\phi'(z)|/\phi(z)^2 = 2a|z|$ — linear in $|z|$, bounded and balanced.

**Integration into KDA:**

$$
\boldsymbol{S}_t = (\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \text{Diag}(\phi(\boldsymbol{z}_t)) \boldsymbol{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top
$$

Everything else in the chunkwise algorithm remains identical. The cumulative decay in log-space becomes:

$$
\log \gamma_{[t]}^{i \to j} = \sum_{m=i}^{j} \log \phi(z_{[t]}^m)
$$

which is well-defined since $\phi(z) \in [0, 1)$ and $\log \phi(z) \in (-\infty, 0)$.

**Key Variables:**

- $\boldsymbol{z}_t \in \mathbb{R}^{d_k}$ — pre-activation for decay gate (output of learned projection)
- $\phi : \mathbb{R} \to [0, 1)$ — gradient-balanced activation function
- $a, b > 0$ — reparameterization hyperparameters (default: $a = 1, b = 1$)
- $\boldsymbol{\alpha}_t = \phi(\boldsymbol{z}_t) \in [0, 1)^{d_k}$ — per-channel decay

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA / KDA (unchanged architecture) |
| Modification | Replace $\sigma$ with $\phi$ in decay gate only |
| Layers | $L = 12$–$24$ |
| Hidden dim | $d_{\text{model}} = 768$–$2048$ |
| Heads | $H = 8$–$16$ |
| Head dim | $d_k = d_v = 64$–$128$ |
| Chunk size | $C = 64$ |

### Baseline

1. **Standard sigmoid GLA/KDA**: Current default parameterization with $\boldsymbol{\alpha}_t = \sigma(\boldsymbol{z}_t)$. Complexity: $O(T C d_k^2)$ per layer.
2. **ReGLA composite gating**: Alternative gate saturation fix using a refining module. Adds parameters and FLOPs.
3. **Exponential reparameterization**: $\boldsymbol{\alpha}_t = 1 - e^{-\text{softplus}(z_t)}$, analogous to S4's eigenvalue parameterization.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Max stable LR | $\geq 2\times$ baseline | Highest LR without NaN over 10K steps |
| Perplexity (350M, 15B tokens) | $\leq$ baseline $- 0.3$ | WikiText-103 validation |
| Training throughput | $= 1.00\times$ baseline | Tokens/sec on H100 (must be identical) |
| Gate gradient variance | $\leq 0.5\times$ baseline | Std of $\|\partial L / \partial z_t\|$ across channels |
| Convergence speed | $\geq 1.1\times$ faster | Steps to reach baseline's final loss |

### Estimated Compute

**MVE**: < 10 minutes on single GPU (synthetic task + tiny model)
**Ablation sweep** ($a, b$ hyperparameters): ~8 GPU-hours on A100
**Full 350M pretraining comparison**: ~100 GPU-hours on H100
**Total**: ~110 GPU-hours (medium scale)

## Expected Outcome

**If hypothesis is correct:**

- Maximum stable learning rate increases by $2$–$5\times$ (e.g., from $3 \times 10^{-4}$ to $1 \times 10^{-3}$) because the gradient-balanced parameterization prevents any channel from receiving disproportionately large gradients.
- Final perplexity improves by $0.3$–$0.8\%$ at fixed compute budget, because the model converges faster with a more balanced optimization landscape.
- The gradient variance across channels (measured as $\text{Var}_{i}[\|\partial L / \partial z_{t,i}\|]$) decreases by $> 50\%$, confirming the gradient balancing effect.
- Zero throughput difference — the reparameterization is a single elementwise operation replacing another elementwise operation (sigmoid).

**If hypothesis is wrong:**

- **Scenario A: Data-dependent gates don't suffer from gradient imbalance.** If the learned projection $\boldsymbol{z}_t = \boldsymbol{W}_\alpha \boldsymbol{x}_t$ naturally keeps gates away from saturation, the sigmoid's flat regions are never reached and the reparameterization provides no benefit. **Learn**: Measure the empirical gate distribution — if gates cluster near $0.5$, the saturation problem doesn't exist for data-dependent gates.
- **Scenario B: The balanced parameterization's range is wrong for GLA/KDA.** The $\phi$ function approaches 1 more slowly than sigmoid, which might prevent channels from learning near-perfect retention ($\alpha \approx 0.999$). **Learn**: Compare the empirical gate distributions and check if long-memory channels are underrepresented.
- **Scenario C: Adam already compensates.** Adam's per-parameter learning rate adaptation may already compensate for gradient imbalance across channels, making the reparameterization redundant. **Learn**: Test with SGD (which doesn't adapt) to isolate the reparameterization's effect.

## Minimum Viable Experiment

### Setup

- **Model**: 2-layer GLA, $d = 64$, $d_k = d_v = 32$, 4 heads (~120K params)
- **Task**: Multi-Query Associative Recall (MQAR) with variable association lengths
  - Short associations: key-value pairs 5 tokens apart
  - Long associations: key-value pairs 50 tokens apart
  - This creates channels with different required memory lengths, exercising the gradient imbalance
- **Data**: 10K synthetic MQAR samples, sequence length $T = 128$
- **Training**: SGD with momentum (not Adam, to expose gradient imbalance) at multiple learning rates
- **Compute**: Single GPU, < 10 minutes

### Success Criteria

- With **gradient-balanced $\phi$**: model achieves $> 85\%$ MQAR accuracy at **peak learning rate** $\text{LR}_{\text{max}}$
- With **sigmoid**: model diverges (NaN) at $\text{LR}_{\text{max}}$ or achieves $< 70\%$ accuracy
- The gap must be visible with SGD; with Adam, both should work but $\phi$ should converge faster (within $1.2\times$ fewer steps)

### Failure Criteria

- If both sigmoid and $\phi$ achieve the same accuracy at the same learning rate with SGD: the gradient imbalance is not a real bottleneck for this architecture.
- If $\phi$ achieves worse accuracy than sigmoid at any learning rate: the parameterization's range or gradient profile is fundamentally wrong for data-dependent gates.

### Why This Test Is Sufficient

- MQAR requires learning multi-scale memory (short and long associations), directly exercising the gradient imbalance between channels with different decay rates.
- SGD (no adaptive LR) isolates the reparameterization's effect from optimizer-level compensation.
- If the gradient balancing helps at toy scale with SGD, it will help more at large scale where gradient imbalance compounds over many layers and long training runs.

## Memory Access Pattern Analysis

**Reparameterization function itself:**
- **Sigmoid**: $\sigma(z) = 1/(1 + e^{-z})$ — 1 exp + 1 add + 1 div per element
- **Balanced $\phi$**: $\phi(z) = \max(0, 1 - 1/(az^2 + b))$ — 1 mul + 1 add + 1 div + 1 sub + 1 max per element
- Both are elementwise, fully coalesced, and negligible compared to the $O(C^2 d_k)$ matmuls in the chunk

**Impact on chunkwise kernel:**
- The cumulative log-decay $\log \gamma$ is computed from $\log \alpha_t = \log \phi(z_t)$. The balanced function $\phi$ is smooth and positive on the domain where we take the log, so this is numerically identical to the sigmoid case.
- No change to memory access patterns, tiling, or tensor core usage in the chunkwise kernel.

## Parallelism Analysis

- **No change to parallelism**: The reparameterization is applied elementwise to the pre-activation $\boldsymbol{z}_t$ before feeding into the chunkwise kernel. The kernel itself is unchanged.
- **No warp divergence**: The $\max(0, \cdot)$ in $\phi$ could theoretically cause divergence, but in practice the gate is always positive after initial training.
- **Tensor core mapping**: Unchanged — all dominant ops remain matmuls.

## Theoretical Analysis

**Gradient scale comparison (the key analysis from StableSSM):**

| Reparameterization | $\phi(z)$ | $\|\phi'(z)\| / \phi(z)^2$ | Behavior at saturation |
|---|---|---|---|
| Sigmoid | $\sigma(z)$ | $(1 - \sigma(z))/\sigma(z)$ | $\to 0$ as $z \to +\infty$ (long memory) |
| Softplus-based | $1 - e^{-\text{softplus}(z)}$ | Bounded but exponentially varying | Better than sigmoid but still imbalanced |
| **Gradient-balanced** | $1 - 1/(z^2 + 1)$ | $2\|z\|$ | **Linear in $\|z\|$** — maximally balanced |

**Effect on optimization:**

With sigmoid, channels at $\alpha = 0.99$ (long memory, $z \approx 4.6$) receive gradient $\sigma'(4.6) \approx 0.01$ — $100\times$ smaller than channels at $\alpha = 0.5$ ($z = 0$, $\sigma'(0) = 0.25$).

With the balanced function, channels at $\alpha = 0.99$ ($z \approx 10$) receive gradient $\phi'(10) = 20/(101)^2 \approx 0.002$, while channels at $\alpha = 0.5$ ($z = 1$) receive $\phi'(1) = 2/4 = 0.5$. The ratio is $250\times$, which is worse in absolute terms but the **gradient-over-weight ratio** $|\phi'|/\phi^2 = 2|z|$ is linear — the key StableSSM insight is that this ratio, not the absolute gradient, determines optimization stability.

## Risks & Limitations

1. **Data-dependent gates may not need reparameterization.** Unlike SSM eigenvalues (which are fixed parameters), decay gates are functions of the input. The input dependence may naturally regularize the gate distribution, preventing saturation. The MVE tests this directly.

2. **The balanced function's approach to 1 is slower than sigmoid.** Sigmoid reaches $0.99$ at $z \approx 4.6$; the balanced function reaches $0.99$ at $z = 10$. This means the network needs larger pre-activations to achieve near-perfect retention, which could stress the linear projection weights.

3. **Interaction with log-space cumulative decay.** GLA's secondary chunking uses $\log \alpha$ in FP32 for numerical stability. The balanced function $\phi(z) = 1 - 1/(z^2 + 1)$ is smooth and positive, so $\log \phi$ is well-defined. However, for small $z$ where $\phi(z) \to 0$, $\log \phi(z) \to -\infty$ — same behavior as sigmoid.

4. **Hyperparameter sensitivity.** The choice of $(a, b)$ in $\phi(z) = 1 - 1/(az^2 + b)$ affects the gate's range and gradient profile. We may need to sweep these, adding to compute cost. Default: $a = 1, b = 1$.

5. **Adam compensation.** Modern training uses Adam with per-parameter adaptive learning rates. Adam's second moment $v_t$ naturally scales down the learning rate for parameters with large gradients, partially compensating for gradient imbalance. The reparameterization's benefit may be smaller with Adam than with SGD. However, StableSSM showed benefits even with Adam at scale.

## Follow-up Experiments

1. **Scale to 1.3B–7B**: If the MVE and 350M experiments show benefits, scale to 1.3B+ to validate at the regime where gradient imbalance compounds over many layers.

2. **Combine with ReGLA's composite gating**: The reparameterization (changing $\sigma \to \phi$) and ReGLA's refining module are orthogonal modifications. Test if they combine for additional benefit or if one subsumes the other.

3. **Apply to Gated DeltaNet's scalar gate**: Gated DeltaNet uses a scalar (per-head) gate $g_t = \sigma(w_g^\top x_t)$. The same reparameterization applies and may be even more impactful since a single scalar gate controls all channels simultaneously.

4. **Learning rate warmup interaction**: The balanced parameterization may change the optimal warmup schedule. Test with shorter warmup to see if the balanced gradients allow faster ramp-up.

5. **Gate distribution analysis**: Measure the empirical distribution of $\alpha_t$ across channels throughout training. Compare sigmoid vs. balanced: does the balanced version develop a wider spread of time scales?

6. **Apply to RWKV-7's per-channel decay**: RWKV-7 uses a vector-valued decay $w_t$ in $[-1, 1]^{D/h}$. The balanced reparameterization could stabilize the training of channels near the stability boundary ($|w| \to 1$).

## Human Review

(To be filled by reviewer)

## References

- Wang, S. & Li, Q. (2024). StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization. ICML 2024. arXiv:2311.14495.
- Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692.
- Yang, S. et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024. arXiv:2312.06635.
- Lu, P. et al. (2025). ReGLA: Refining Gated Linear Attention. NAACL 2025.
- Yang, S. et al. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.
- Orvieto, A. et al. (2023). Resurrecting Recurrent Neural Networks for Long Sequences. ICML 2023.

# 262: TaperNorm — Gated Normalization Removal with Weight Folding

**Category**: stability
**Gain type**: efficiency
**Source**: Kanavalau, Amo Alonso & Lall (2026) — "Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference", Stanford University. arXiv:2602.10408
**Paper**: [papers/tapernorm-gated-normalization-removal.pdf]
**Documented**: 2026-02-16

## Description

TaperNorm is a drop-in replacement for RMSNorm/LayerNorm that **starts as standard normalization** during early training and **smoothly transitions to a fixed linear/affine map** that can be folded into adjacent weight matrices at inference. Unlike DyT (trick #241) and Derf (trick #260), which replace normalization with a nonlinear elementwise function that remains in the inference graph, TaperNorm's converged form is a simple diagonal scaling $x \mapsto c \cdot D_{\tilde\gamma} \cdot x$ — a **linear** map that can be fully absorbed into adjacent linear projections, completely removing the layer from the computation graph.

**The key insight — scale anchoring:** The paper identifies the fundamental role of output normalization in pre-norm transformers: it acts as a **0-homogeneous map** that removes the radial component of the gradient at the output. Without this scale anchor, cross-entropy loss encourages unbounded growth of logit magnitudes — a phenomenon the authors call **"logit chasing"**. A final normalization layer prevents this by making the loss invariant to the norm of the pre-logit representation ($\|h\|_2$).

**The tapering mechanism:** A single global gate $g(k) \in [0, 1]$ shared across all layers and tokens controls a convex combination of:
1. A standard normalization branch (RMSNorm or LayerNorm)
2. A sample-independent scaling branch ($c \cdot D_{\tilde\gamma}$)

During warmup ($g = 1$), the model trains with standard normalization. Exponential moving averages (EMAs) of the normalization statistics are accumulated to calibrate the scaling branch. After warmup, $g$ is cosine-decayed to 0, at which point per-token statistics vanish. At convergence ($g = 0$), each TaperNorm layer becomes a fixed diagonal scaling that can be folded into adjacent linear projections.

**Fixed-target scale anchoring loss:** To enable removal of even the final normalization layer (which provides the implicit scale anchor), the paper introduces an auxiliary loss $\mathcal{L}_{\text{aux}} = \lambda \, \mathbb{E}_{b,t}(s(h_{b,t}) - s_{\text{tgt}})^2$ that penalizes deviations of per-token scale statistics from a frozen EMA target. This provides an explicit radial restoring force, replacing the implicit anchoring of the final normalization layer.

**GPU efficiency — real wall-clock gains:** Unlike DyT/Derf which show no measurable speedup over optimized LN, TaperNorm provides **actual inference throughput gains** by completely eliminating normalization layers from the inference graph:
- **1.08–1.15×** throughput improvement with unfused TaperNorm (diagonal scaling as explicit ops)
- **1.13–1.22×** throughput improvement with fused TaperNorm (scalings folded into adjacent projections)

These gains come from eliminating per-token reductions entirely — the folded model has **fewer layers** in the computation graph and **fewer kernel launches**.

## Mathematical Form

**Standard RMSNorm:**

$$
\text{RMSNorm}(h) = \frac{h}{r(h)} D_\gamma, \quad r(h) = \sqrt{\|h\|_2^2/d + \varepsilon}
$$

where $h \in \mathbb{R}^{1 \times d}$ is a token vector, $D_\gamma = \text{diag}(\gamma)$ with learnable $\gamma \in \mathbb{R}^d$.

---

**TaperNorm (RMSNorm variant):**

$$
\text{TaperNorm}(h; g) = g \cdot \frac{h}{r(h)} D_\gamma + (1 - g) \cdot c \, h \, D_{\tilde\gamma}
$$

where:
- $g \in [0, 1]$ — single global gate, shared across all layers and tokens
- $c \in \mathbb{R}$ — per-layer scalar calibrated from EMA statistics
- $\tilde\gamma \in \mathbb{R}^d$ — per-feature gain (remains trainable after taper start), initialized to $\gamma$
- $D_{\tilde\gamma} = \text{diag}(\tilde\gamma)$ — diagonal scaling matrix

For $g = 1$: recovers standard RMSNorm. For $g = 0$: reduces to $c \, h \, D_{\tilde\gamma}$, a sample-independent linear map.

---

**Gate schedule:**

$$
g(k) = \begin{cases}
1 & k \leq k_{\text{warmup}} \quad \text{(gate warmup, accumulate EMAs)} \\
\frac{1}{2}\left(1 + \cos\left(\frac{\pi(k - k_{\text{warmup}})}{k_{\text{taper}} - k_{\text{warmup}}}\right)\right) & k_{\text{warmup}} < k \leq k_{\text{taper}} \\
0 & k > k_{\text{taper}} \quad \text{(fold into weights)}
\end{cases}
$$

---

**Calibrating the scaling branch ($c^\star$):**

At the warmup boundary, $c$ is set to match the normalization branch in a least-squares sense:

$$
c^\star = \arg\min_c \, \mathbb{E}\left[\left\|\frac{h}{r(h)} D_\gamma - c \, h \, D_\gamma\right\|_2^2\right] = \frac{\mathbb{E}\left[\|h D_\gamma\|_2^2 / r(h)\right]}{\mathbb{E}\left[\|h D_\gamma\|_2^2\right]}
$$

This minimizes the distribution shift when the gate begins to decay.

---

**Weight folding at inference ($g = 0$):**

When $g = 0$, TaperNorm becomes $h \mapsto c \, h \, D_{\tilde\gamma}$. If the next operation is a linear projection $h W$:

$$
c \, h \, D_{\tilde\gamma} \, W = h \, (c \, D_{\tilde\gamma} \, W) = h \, W'
$$

where $W' = c \, D_{\tilde\gamma} \, W$ is precomputed once. The TaperNorm layer **disappears from the inference graph**.

The same folding applies to attention QKV and MLP input projections: for a projection $W_Q$, the fused weight is $W'_Q = c \, D_{\tilde\gamma} \, W_Q$.

---

**Fixed-target scale anchoring loss:**

$$
\mathcal{L}_{\text{aux}} = \lambda \, \mathbb{E}_{b,t}\left(s(h_{b,t}) - s_{\text{tgt}}\right)^2
$$

where $s(h) = r(h)$ for RMSNorm or $s(h) = \sigma_h$ for LayerNorm, and $s_{\text{tgt}}$ is a bias-corrected EMA of $\mathbb{E}_{b,t}[s(h_{b,t})]$ frozen at taper start. This prevents logit chasing when the final normalization is also removed.

---

**Scale anchoring theory — why normalization matters:**

*Proposition 4.1*: Final normalization removes the radial gradient. For logits $z = \text{Norm}_{\text{final}}(h) W_{\text{out}}$ and any differentiable loss $\ell(z, y)$:

$$
\langle \nabla_h \ell(z, y), h \rangle = 0 \quad (h \neq 0)
$$

This means gradient steps cannot change $\|h\|_2$ — the norm is anchored.

*Proposition 4.2*: Without the final norm, cross-entropy pushes norms upward. If the correct class margin $m = z_y - \max_{j \neq y} z_j > 0$:

$$
\langle \nabla_h \ell(z, y), h \rangle \leq -(1 - \text{softmax}(z)_y) \cdot m < 0
$$

so a gradient step along $-\nabla_h \ell$ increases $\|h\|_2^2$ — this is **logit chasing**.

## Complexity

| Operation | RMSNorm | TaperNorm (training) | TaperNorm (inference, folded) |
|-----------|---------|---------------------|-------------------------------|
| Forward pass | $O(BTC)$ with $C$-reduction | $O(BTC)$ with $C$-reduction + elementwise blend | **$O(0)$** — removed from graph |
| Per-token statistics | $T$ reductions ($r(h)$) | $T$ reductions (during warmup/taper) | **None** |
| Learnable parameters | $C$ ($\gamma$) | $C + 2$ ($\gamma, \tilde\gamma, c$) per layer | Folded into adjacent $W$ |
| Kernel launches | 1 per norm layer | 1 per norm layer (during training) | **0** — layer eliminated |
| Weight matrix overhead | None | None | One-time precomputation of $W' = c D_{\tilde\gamma} W$ |

**Memory:** Training memory is identical to standard normalization. At inference, memory **decreases** because TaperNorm layers are removed.

**Wall-clock throughput (H100 80GB, bf16, 30M param model):**

| Batch $B$ | Seq $T$ | RMSNorm baseline | TaperNorm (fused) | Speedup |
|-----------|---------|------------------|-------------------|---------|
| 1 | 128 | 40.7 k tok/s | 48.7 k tok/s | **1.20×** |
| 1 | 256 | 79.3 k tok/s | 96.8 k tok/s | **1.22×** |
| 1 | 512 | 163.2 k tok/s | 195.5 k tok/s | **1.20×** |
| 4 | 128 | 149.3 k tok/s | 180.4 k tok/s | **1.21×** |
| 4 | 256 | 290.5 k tok/s | 344.9 k tok/s | **1.19×** |
| 4 | 512 | 546.3 k tok/s | 615.9 k tok/s | **1.13×** |

Gains are larger when normalization is a larger fraction of total compute (shorter sequences, smaller batches — i.e., the inference-latency regime).

## Applicability

- **LLM inference optimization:** Primary use case. After training, fold all internal normalization layers into adjacent projections for 13–22% throughput gain. Particularly impactful for latency-sensitive serving (small batch, short sequence).

- **Pre-training from scratch:** TaperNorm can be used during pre-training on TinyStories and GPT-2 scale models. Internal-Taper (+aux) matches RMSNorm baselines within 0.7–1.5% relative validation loss across 1M–30M parameter models.

- **Fine-tuning existing models:** TaperNorm can be applied to existing GPT-2 family models via fine-tuning. Internal-TaperLN matches or outperforms the FakeLN baseline (Baroni et al., 2025) across GPT-2 S/M/L/XL on OpenWebText, The Pile, and Pile-filtered.

- **Mechanistic interpretability:** Norm-free models have simpler computation graphs, aiding circuit-level analysis (no nonlinear normalization to model).

- **Custom hardware / in-memory computing:** Eliminating normalization removes collective reduction operations that are expensive on DIMC and other in-memory computing architectures.

## Limitations

- **Scale validated only to 30M (pre-training) and GPT-2 XL (fine-tuning):** Not yet validated on 7B+ pre-training from scratch. The tapering schedule and $\alpha_0$ calibration may need adjustment at larger scales.

- **Requires careful gate schedule:** The warmup period must be long enough to calibrate EMAs. The cosine decay period affects how smoothly the transition occurs. Misaligned schedules could cause training instability.

- **Final normalization keeps implicit role:** The default configuration (Internal-Taper) keeps the final RMSNorm/LayerNorm as a scale anchor. Removing it (All-Taper) requires the auxiliary scale loss and shows slightly higher variance across seeds.

- **Not applicable during training for speedup:** The speedup is inference-only. During training, TaperNorm adds overhead (EMA tracking, gate blending) compared to standard normalization.

- **All-Taper can be unstable at some scales:** For 9M parameters, one of 6 seeds produced an unusually high validation loss in All-Taper mode, suggesting the fully norm-free regime is less robust.

## Implementation Notes

```python
import torch
import torch.nn as nn
import math

class TaperNorm(nn.Module):
    """
    TaperNorm — gated transition from RMSNorm to foldable linear scaling.

    Training: starts as RMSNorm (g=1), decays to diagonal scaling (g=0).
    Inference: when g=0, fold c * diag(gamma_tilde) into adjacent W.

    GPU notes:
    - During training: same cost as RMSNorm + one extra elementwise blend
    - At inference (g=0, folded): ZERO cost — layer removed from graph
    - Weight folding: W' = c * diag(gamma_tilde) @ W — one-time precompute
    - Eliminates per-token reductions entirely at inference
    - 1.13-1.22x throughput gain on H100 in last-token logits mode
    """
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.d = d
        self.eps = eps

        # Normalization branch parameters (standard RMSNorm)
        self.gamma = nn.Parameter(torch.ones(d))

        # Scaling branch parameters
        self.gamma_tilde = nn.Parameter(torch.ones(d))
        self.c = 1.0  # calibrated from EMAs at warmup boundary

        # EMA tracking for calibration
        self.register_buffer('ema_num', torch.zeros(1))   # EMA of ||h*gamma||^2/r(h)
        self.register_buffer('ema_den', torch.zeros(1))   # EMA of ||h*gamma||^2
        self.register_buffer('c_frozen', torch.ones(1))
        self.register_buffer('calibrated', torch.tensor(False))

    def rmsnorm(self, h):
        """Standard RMSNorm: h / r(h) * gamma"""
        rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return h / rms * self.gamma

    def scale_branch(self, h):
        """Sample-independent scaling: c * h * gamma_tilde"""
        return self.c_frozen * h * self.gamma_tilde

    def forward(self, h, gate):
        """
        h: (B, T, d) input tensor
        gate: scalar in [0, 1], from gate schedule
        """
        if gate == 0.0:
            # Fully tapered — just diagonal scaling (can be folded)
            return self.scale_branch(h)
        elif gate == 1.0:
            return self.rmsnorm(h)
        else:
            # Convex combination during tapering
            return gate * self.rmsnorm(h) + (1 - gate) * self.scale_branch(h)

    def calibrate(self):
        """Call at warmup boundary to set c* and copy gamma -> gamma_tilde."""
        self.c_frozen.fill_(self.ema_num / (self.ema_den + self.eps))
        self.gamma_tilde.data.copy_(self.gamma.data)
        self.calibrated.fill_(True)


def gate_schedule(step, warmup_end, taper_end):
    """
    Cosine decay gate schedule.
    g=1 during warmup, cosine decay to 0 after warmup.
    """
    if step <= warmup_end:
        return 1.0
    elif step >= taper_end:
        return 0.0
    else:
        progress = (step - warmup_end) / (taper_end - warmup_end)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def fold_tapernorm_into_projection(tapernorm, W):
    """
    Fold converged TaperNorm (g=0) into adjacent weight matrix.

    TaperNorm(h) = c * h * gamma_tilde
    Next layer: TaperNorm(h) @ W = c * h * diag(gamma_tilde) @ W

    Returns W' = c * diag(gamma_tilde) @ W
    The TaperNorm layer can then be removed from the graph.
    """
    c = tapernorm.c_frozen.item()
    gamma = tapernorm.gamma_tilde.data  # (d,)
    # W is (d, d_out): scale each row of W by c * gamma[i]
    W_folded = c * gamma.unsqueeze(1) * W
    return W_folded


# Scale anchoring auxiliary loss
class ScaleAnchorLoss(nn.Module):
    """
    Fixed-target scale loss to prevent logit chasing.
    L_aux = lambda * E_{b,t}[(s(h) - s_tgt)^2]
    """
    def __init__(self, lam=0.01):
        super().__init__()
        self.lam = lam
        self.register_buffer('s_tgt', torch.ones(1))
        self.register_buffer('frozen', torch.tensor(False))

    def forward(self, h):
        """h: (B, T, d) pre-logit hidden states"""
        # s(h) = RMS norm of each token
        s = torch.sqrt(h.pow(2).mean(dim=-1) + 1e-8)  # (B, T)
        if not self.frozen:
            # During warmup: update EMA target
            with torch.no_grad():
                self.s_tgt.lerp_(s.mean(), 0.01)
            return torch.tensor(0.0, device=h.device)
        else:
            return self.lam * ((s - self.s_tgt) ** 2).mean()
```

## GPU Efficiency Analysis

**Memory Access Pattern:**
- During training: identical to RMSNorm — one reduction over $d$ per token
- At inference (folded): **no memory access for normalization** — the layer is gone
- Weight folding is a one-time $O(d^2)$ precomputation per projection matrix

**Parallelism:**
- Training: same parallelism profile as RMSNorm (per-token reduction)
- Inference: **improved** — fewer sequential kernel launches in the forward pass
- Folded model has strictly fewer operations than the normalized model

**Arithmetic Intensity:**
- Training: identical to RMSNorm + negligible gate blending cost
- Inference: **strictly higher** than normalized model — same matmul compute with fewer elementwise/reduction ops → better compute utilization

**Hardware:**
- No tensor core compatibility issues — folding doesn't change matmul shapes
- Reduces kernel launch overhead — fewer layers in the graph
- Particularly beneficial for latency-sensitive inference (autoregressive decoding)
- Gains scale inversely with sequence length (normalization is a larger fraction of cost at short sequences)

## References

- Kanavalau, A., Amo Alonso, C., & Lall, S. (2026). Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference. arXiv:2602.10408.
- Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). Transformers without Normalization. CVPR 2025. arXiv:2503.10622.
- Baroni, L., Khara, G., Schaeffer, J., Subkhankulov, M., & Heimersheim, H. (2025). Transformers don't need layernorm at inference time. arXiv:2507.02559.
- Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML 2020.
- Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. ICML 2021.

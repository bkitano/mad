---
status: ongoing
priority: high
created: 2026-02-15
based_on: post-attention-sigmoid-gating, linear-attention-approximation, cosine-reweighted-linear-attention, input-dependent-gating, bilinear-gating-glu, kernel-fusion
experiment_number: 009
---

# Post-Sigmoid Gating for Linear Attention and SSM Readout

## Hypothesis

Applying post-readout sigmoid gating — the NeurIPS 2025 Best Paper technique for softmax attention — to **linear attention** and **SSM output readout** will break the low-rank bottleneck inherent in these architectures' output projections, improving quality by 5–15% (perplexity reduction) with $< 2\%$ latency overhead, and the benefit will be *larger* for linear attention / SSMs than for softmax attention because these models already suffer from a more severe information bottleneck (fixed $d \times d$ state vs. softmax's $T \times T$ attention matrix).

## Background

**The low-rank bottleneck in softmax attention** (solved by Qiu et al., 2025): In standard multi-head attention with GQA, the composition $W_V^k W_O^k$ forms a single low-rank linear mapping. Post-attention sigmoid gating breaks this bottleneck by inserting a position-dependent nonlinearity between the attention output and the output projection. This was shown to improve quality, eliminate attention sinks, and enable stable training at scale.

**The *worse* low-rank bottleneck in linear attention and SSMs**: Linear attention computes $o_t = q_t^\top S_t$ where $S_t = \sum_{j \leq t} k_j v_j^\top \in \mathbb{R}^{d \times d}$. The state $S_t$ is a rank-bounded matrix (rank $\leq t$, but practically $\ll d$ due to redundancy), and the readout $o_t = q_t^\top S_t$ is a *single linear operation* on this compressed state. Similarly, SSMs compute $y_t = C_t^\top h_t$ where $h_t \in \mathbb{R}^n$ is the hidden state. In both cases, the output is a linear function of a compressed representation — suffering from the **same low-rank bottleneck** that sigmoid gating fixes in softmax attention, but arguably worse because:

1. The state $S_t$ has bounded capacity ($d^2$ parameters compressing $T$ tokens)
2. The readout is always linear ($q^\top S$), with no nonlinearity
3. There's no attention normalization (no softmax) to create sharp selection

**The gap**: No existing work applies post-readout sigmoid gating to linear attention or SSMs. The original paper (Qiu et al., 2025) only evaluated on softmax attention. All 8 existing proposals focus on improving the *state transition* or *state structure* of SSMs/linear attention — none address the readout bottleneck.

**Why this is promising**: The gating is orthogonal to all other improvements. It can be composed with any proposal (004, 006, 007, 008) as a drop-in addition. The cost is negligible ($O(nd)$ per step vs. $O(nd^2)$ or $O(n^2 d)$ for the main computation), and the implementation requires only a single linear projection + sigmoid + elementwise multiply.

## Mathematical Formulation

### Standard Linear Attention Readout

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t} \in \mathbb{R}^d
$$

where $S_t = \sum_{j \leq t} \phi(k_j) v_j^\top \in \mathbb{R}^{d \times d}$ and $z_t = \sum_{j \leq t} \phi(k_j) \in \mathbb{R}^d$.

The final output passes through an output projection: $\hat{o}_t = o_t W_O$.

**Bottleneck**: The composition $[\phi(q)^\top \cdot S_t] \cdot W_O$ is linear end-to-end. The readout can only extract information that lies in the column space of $S_t$, and $W_O$ cannot recover information lost by this linear projection.

### Proposed: Gated Linear Attention Readout

$$
\hat{o}_t = \left(o_t \odot \sigma(x_t W_g)\right) W_O
$$

where:
- $o_t \in \mathbb{R}^d$ — linear attention output (per head, after normalization)
- $x_t \in \mathbb{R}^{d_{\text{model}}}$ — input hidden state (pre-norm, same as used for Q/K/V projections)
- $W_g \in \mathbb{R}^{d_{\text{model}} \times d}$ — learnable gate projection (per head)
- $\sigma(\cdot)$ — sigmoid activation
- $\odot$ — elementwise multiplication

**Key insight**: The sigmoid gate $\sigma(x_t W_g)$ is a function of the *current input* $x_t$, not the attention output $o_t$. This means:
1. It provides a position/content-dependent nonlinear mask on the readout
2. It can suppress dimensions of $o_t$ that are irrelevant for the current position
3. It breaks the linearity of $[\phi(q)^\top S_t] W_O$, allowing $W_O$ to operate on a nonlinearly filtered signal

### SSM Readout (analogous formulation)

For SSMs with readout $y_t = C_t^\top h_t$:

$$
\hat{y}_t = \left(C_t^\top h_t \odot \sigma(x_t W_g)\right) W_O
$$

Or equivalently, for multi-head SSMs:

$$
\hat{y}_t^{(k)} = \left(C_k^\top h_t^{(k)}\right) \odot \sigma(x_t W_g^{(k)})
$$

This is a direct analog: the sigmoid gate filters the SSM output before it contributes to the residual stream.

### Why This Matters More for Linear Attention / SSMs

**Softmax attention**: The attention matrix $\text{softmax}(QK^\top / \sqrt{d})$ is already nonlinear and input-dependent. The gate adds additional nonlinearity on top of an already expressive mechanism.

**Linear attention**: The readout $\phi(q)^\top S$ is purely linear. The gate is the *first and only* nonlinearity in the readout path. Its relative contribution is therefore larger.

**Quantitative prediction**: If the gate improves softmax attention perplexity by $X\%$ (Qiu et al. report ~1.5% for 15B models), we predict $1.5X$–$3X$ improvement for linear attention at matched scale, because the linear readout has strictly less expressive power pre-gating.

### Key Variables

- $x_t \in \mathbb{R}^{d_{\text{model}}}$ — input at position $t$
- $o_t \in \mathbb{R}^d$ — attention/SSM output per head
- $W_g \in \mathbb{R}^{d_{\text{model}} \times d}$ — gate projection (per head)
- $d_{\text{model}}$ — model dimension
- $d$ — head dimension
- $H$ — number of heads

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Base model | Linear Attention (cosFormer or GLA) + SSM (Mamba-2) |
| Gating position | Post-readout, pre-output-projection |
| Gate activation | Sigmoid (following Qiu et al.) |
| Gate initialization | Zero-init $W_g$ (gate starts at $\sigma(0) = 0.5$) |
| Layers | $L = 12$ |
| Hidden dim | $d_{\text{model}} = 768$ |
| Heads | $H = 12$ |
| Head dim | $d = 64$ |

### Experiment Matrix

Test the gate on **three base architectures** to isolate its effect:

| Base Architecture | Gate? | Expected Outcome |
|-------------------|-------|------------------|
| cosFormer (linear attention) | No | Baseline |
| cosFormer + sigmoid gate | Yes | Improved perplexity |
| GLA (gated linear attention) | No | Baseline |
| GLA + sigmoid gate | Yes | Improved perplexity |
| Mamba-2 (diagonal SSM) | No | Baseline |
| Mamba-2 + sigmoid gate | Yes | Improved perplexity |
| Softmax attention | No | Reference |
| Softmax + sigmoid gate | Yes | Reference (reproduces Qiu et al.) |

### Baseline

1. **cosFormer** (no gate): $O(Td^2)$ — linear attention with cosine reweighting
2. **GLA** (no gate): $O(Td^2)$ — gated linear attention (already has input-dependent decay, but no readout gate)
3. **Mamba-2** (no gate): $O(Tn)$ — diagonal SSM with SSD algorithm
4. **Softmax + FlashAttention** (no gate): $O(T^2 d)$ — gold standard for quality

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity improvement | $> 3\%$ reduction vs. ungated | WikiText-103, 125M param models |
| MQAR accuracy | $> 5\%$ improvement at 4 KV pairs | Multi-Query Associative Recall |
| Attention sink reduction | $< 0.5\times$ first-token attention mass | Measure $\sum_j S_{1j}$ / $\sum_{ij} S_{ij}$ |
| Throughput overhead | $< 3\%$ latency increase | Timed forward pass, A100 |
| Gate sparsity | Gate values $< 0.1$ for $> 50\%$ of entries | Histogram of $\sigma(x_t W_g)$ |

### Estimated Compute

**MVE**: ~10 minutes on single GPU (~$0.50)
**Small-scale**: 8 GPU-hours on A100 (~$32)
**Full-scale**: 48 GPU-hours on A100 (~$200)

## Expected Outcome

**If hypothesis is correct:**

1. **Perplexity**: Gated cosFormer achieves $3$–$8\%$ lower perplexity than ungated cosFormer on WikiText-103 at 125M params. This is a larger relative improvement than the ~1.5% seen for softmax attention because the linear readout is a tighter bottleneck.

2. **MQAR recall**: Gated linear attention improves recall by $5$–$15\%$ at 4+ KV pairs because the gate enables the model to sharpen readout to specific stored associations rather than producing a blurred linear combination.

3. **Attention sinks**: In linear attention, the state $S_t$ accumulates contributions from all past tokens. Without gating, early tokens with large key norms dominate. The sigmoid gate suppresses these stale contributions at readout, functioning as an "attention sink eliminator" just as it does in softmax attention.

4. **SSM benefit**: Mamba-2 with gated readout shows modest improvement ($1$–$3\%$ perplexity) because SSM readout $C^\top h$ already has some position-dependence through $C = C(x_t)$. The gate adds additional nonlinearity but the marginal benefit is smaller.

5. **Composability**: The gate composes cleanly with other improvements. Gated cosFormer + log-linear hierarchy (proposal 008) should show compounding benefits.

**If hypothesis is wrong:**

- **Scenario A**: Gate improves softmax but not linear attention
  - **Learn**: The bottleneck in linear attention is in the *state capacity* ($d \times d$ matrix), not the readout path. The gate cannot compensate for insufficient state expressivity.
  - **Insight**: Focus future work on state structure (proposals 005, 006) rather than readout modifications.

- **Scenario B**: Gate helps linear attention but not SSMs
  - **Learn**: SSM readout $C(x_t)^\top h_t$ already has sufficient nonlinearity via input-dependent $C$. Linear attention's purely linear readout $q^\top S$ is the true bottleneck.
  - **Insight**: The gate specifically addresses the *linearity* of the readout, not just the rank constraint.

- **Scenario C**: Gate helps everywhere but the improvement is $< 1\%$
  - **Learn**: The readout bottleneck exists but is dominated by other factors (state capacity, kernel quality). The gate is necessary but not sufficient.
  - **Next step**: Combine with state-level improvements and measure if benefits compound.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer cosFormer with and without sigmoid gate ($d_{\text{model}} = 64$, $H = 4$, $d_k = 16$, ~80K params)
- **Task**: **Multi-Query Associative Recall (MQAR)** — store 4 KV pairs, then query 2 of them. Sequence length $T = 64$, vocabulary size 16.
- **Data**: 10K synthetic sequences
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- Gated cosFormer achieves $> 75\%$ accuracy on MQAR with 4 KV pairs at $d_k = 16$
- Ungated cosFormer achieves $< 55\%$ accuracy on the same task
- The improvement persists across 3 random seeds (not a lucky initialization artifact)
- Training is stable (no NaN/Inf) and wall-clock time increases $< 5\%$

### Failure Criteria
- Gated and ungated cosFormer perform within $3\%$ of each other on MQAR — the gate doesn't help
- The gate causes training instability (NaN/Inf or loss divergence)
- The gate adds $> 10\%$ wall-clock overhead (implementation issue)

### Why This Test Is Sufficient
- **MQAR tests readout precision**: The task requires extracting specific KV associations from the compressed state $S_t$. A sharper, gated readout should improve selection accuracy, while a purely linear readout averages over stored associations.
- **Small $d_k = 16$ stresses the bottleneck**: With tiny head dimension, the state $S_t \in \mathbb{R}^{16 \times 16}$ is severely capacity-limited. The gate must do more work to extract useful information, making the effect more visible.
- **cosFormer is the cleanest test**: cosFormer has no input-dependent decay or gating in its state update (unlike GLA or Mamba), so the post-readout gate is the *only* nonlinearity. This isolates the gate's contribution cleanly.
- **If the gate helps at $d_k = 16$, it generalizes**: The bottleneck becomes milder at larger $d_k$ but doesn't disappear. Success at the smallest scale implies benefit at all scales.

### Implementation Sketch

```python
import torch
import torch.nn as nn

class GatedLinearAttention(nn.Module):
    """cosFormer with post-readout sigmoid gating."""

    def __init__(self, d_model, n_heads, d_k, use_gate=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.use_gate = use_gate

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)

        if use_gate:
            self.W_gate = nn.Linear(d_model, n_heads * d_k, bias=False)
            nn.init.zeros_(self.W_gate.weight)  # Start as identity

    def forward(self, x):
        B, T, D = x.shape
        Q = torch.relu(self.W_Q(x)).view(B, T, self.n_heads, self.d_k)
        K = torch.relu(self.W_K(x)).view(B, T, self.n_heads, self.d_k)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k)

        # cosFormer: cosine-reweighted linear attention
        positions = torch.arange(T, device=x.device).float()
        cos_pos = torch.cos(positions * 3.14159 / (2 * T))
        sin_pos = torch.sin(positions * 3.14159 / (2 * T))

        Q_cos = Q * cos_pos[None, :, None, None]
        Q_sin = Q * sin_pos[None, :, None, None]
        K_cos = K * cos_pos[None, :, None, None]
        K_sin = K * sin_pos[None, :, None, None]

        # Causal linear attention (cumulative state)
        # S_cos = cumsum(K_cos^T V), S_sin = cumsum(K_sin^T V)
        S_cos = torch.cumsum(
            torch.einsum('bthd,bthe->bthde', K_cos, V), dim=1
        )
        S_sin = torch.cumsum(
            torch.einsum('bthd,bthe->bthde', K_sin, V), dim=1
        )
        o = torch.einsum('bthd,bthde->bthe', Q_cos, S_cos) + \
            torch.einsum('bthd,bthde->bthe', Q_sin, S_sin)

        # === THE TRICK: Post-readout sigmoid gate ===
        if self.use_gate:
            gate = torch.sigmoid(
                self.W_gate(x).view(B, T, self.n_heads, self.d_k)
            )
            o = o * gate

        o = o.reshape(B, T, -1)
        return self.W_O(o)
```

## Theoretical Analysis

### Complexity Comparison

| Operation | cosFormer | cosFormer + Gate | Mamba-2 | Mamba-2 + Gate |
|-----------|-----------|------------------|---------|----------------|
| State update | $O(Td^2)$ | $O(Td^2)$ | $O(Tn)$ | $O(Tn)$ |
| Readout | $O(Td)$ | $O(Td)$ | $O(Tn)$ | $O(Tn)$ |
| Gate projection | — | $O(Td \cdot d_{\text{model}})$ | — | $O(Tn \cdot d_{\text{model}})$ |
| Output projection | $O(Td \cdot d_{\text{model}})$ | $O(Td \cdot d_{\text{model}})$ | Same | Same |
| **Overhead** | — | **$< 2\%$** | — | **$< 2\%$** |

The gate projection $O(Td \cdot d_{\text{model}})$ is dominated by the state update cost $O(Td^2)$ when $d > d_{\text{model}}/d = d_{\text{model}} / H$. For typical values ($d = 64$, $d_{\text{model}} = 768$, $H = 12$), the gate costs $64 \times 768 \approx 50K$ FLOPs per position vs. $64^2 \times 2 \approx 8K$ for the state update per cosFormer stream — so the gate is ~$6\times$ the readout cost but comparable to Q/K/V projection costs that already exist. Net overhead is $< 2\%$ of total layer cost.

### Expressivity Analysis

**Without gate**: The linear attention output at position $t$ is:

$$
o_t = \phi(q_t)^\top S_t = \phi(q_t)^\top \left(\sum_{j \leq t} \phi(k_j) v_j^\top\right)
$$

This is a *bilinear form* in $q_t$ and $\{k_j, v_j\}_{j \leq t}$ — linear in $q_t$ for fixed state.

**With gate**: The gated output is:

$$
\hat{o}_t = \left(\phi(q_t)^\top S_t\right) \odot \sigma(x_t W_g)
$$

This is a *multiplicative interaction* between the linear attention output and an input-dependent gate. The gate introduces a **data-dependent rank modulation**: it can suppress irrelevant dimensions of $o_t$ (pushing them toward 0) and amplify relevant ones (keeping them near their attention-derived values).

**Connection to GLU / bilinear gating**: The gated readout $o \odot \sigma(g)$ is structurally identical to the GLU mechanism $h \odot \sigma(g)$ used in Transformer FFN layers (SwiGLU, GEGLU). Just as GLU improves FFN expressivity by introducing multiplicative interactions, the readout gate improves attention expressivity by the same mechanism. This is a principled transfer of a proven technique to a new setting.

### Information-Theoretic Argument

The mutual information between the readout $o_t$ and the target $y_t$ is bounded by:

$$
I(o_t; y_t) \leq I(S_t, q_t; y_t)
$$

Without gating, $o_t = q^\top S$ is a deterministic linear function, so $I(o_t; y_t) = I(q^\top S; y_t)$. With gating, $\hat{o}_t = (q^\top S) \odot \sigma(g(x_t))$ can potentially extract more information because the nonlinear gate can implement position-dependent feature selection that the linear readout cannot.

## Risks & Limitations

### Risk 1: Redundancy with Existing Gating
- **Issue**: GLA (Gated Linear Attention) already has input-dependent decay gates. Mamba-2 has input-dependent $\Delta, B, C$. The post-readout gate may be redundant.
- **Mitigation**: Test on cosFormer (no existing gates) first to isolate the readout gate's contribution. Then test on GLA/Mamba-2 to measure marginal benefit.
- **Prediction**: Largest benefit for cosFormer (no competing gates), moderate for GLA, smallest for Mamba-2 (most existing gating).

### Risk 2: Gate Collapse
- **Issue**: The sigmoid gate may learn to be uniformly $\approx 0.5$ (no selectivity) or uniformly $\approx 0$ (killing the signal).
- **Mitigation**: Zero-initialize $W_g$ (starts at $\sigma(0) = 0.5$, a benign scaling). Monitor gate histogram during training. If collapse occurs, try learnable bias initialization.
- **Detection**: Plot histogram of gate values at checkpoints; healthy gates should have bimodal distribution (near 0 and near 1).

### Risk 3: Interaction with Normalization
- **Issue**: Linear attention uses division by $z_t = \phi(q)^\top z$ for normalization. The gate applied *after* normalization could interact poorly (scaling normalized values).
- **Mitigation**: Test gating both before and after normalization:
  - **Post-norm gate**: $\hat{o}_t = (o_t / z_t) \odot \sigma(g_t)$ (gate on normalized output)
  - **Pre-norm gate**: $\hat{o}_t = (o_t \odot \sigma(g_t)) / (z_t \odot \sigma(g_t'))$ (gate on raw output, separate normalization gate)
- **Ablation**: Compare both positions; pre-norm may be more stable.

### Risk 4: Kernel Fusion Complexity
- **Issue**: For maximum efficiency, the gate should be fused with the readout kernel. This requires custom Triton/CUDA code.
- **Mitigation**: Start with unfused PyTorch implementation for MVE. The gate is a simple elementwise operation and should fuse trivially with most kernels.
- **Note**: FlashLinearAttention already fuses the readout; adding a sigmoid gate is a 2-line change to the fused kernel.

## Follow-up Experiments

### If Successful:
1. **Compose with log-linear attention (proposal 008)**: Apply the gate to cos-LogLinear attention — does it improve quality *per state*?
2. **Compose with Monarch-Gated SSM (proposal 006)**: Apply the gate to the Monarch SSM readout — does it complement the state-level coordinate mixing?
3. **Headwise vs. elementwise gate granularity**: Test a single scalar gate per head (cheapest) vs. full elementwise (most expressive)
4. **Gate-based pruning at inference**: If many gate values are near 0, skip those state dimensions entirely at inference for speedup
5. **Extension to Gated DeltaNet**: Apply to DeltaNet's $o_t = S_t v_t$ readout (proposal 001/002 base)

### If Unsuccessful:
1. **Ablate gate position**: Test pre-output, mid-layer, and post-output positions
2. **Replace sigmoid with other activations**: Test SiLU, GELU, or learnable activation
3. **Analyze gate statistics**: If gates are uniform, the readout bottleneck may not exist for the tested tasks
4. **Compare with alternative nonlinearities**: Test quadratic readout $o_t \odot o_t$ (parameter-free nonlinearity) vs. learned gate

## Connection to Existing Proposals

- **Orthogonal to all existing proposals**: This modifies the *readout* path, not the state transition or state structure. It can be composed with proposals 001–008 as a drop-in addition.
- **Strongest synergy with 008 (cos-LogLinear)**: Both target linear attention quality improvement. cos-LogLinear adds state capacity; this proposal adds readout expressivity. Together: more states + sharper readout.
- **Weakest overlap with 007 (OscGate-SSM)**: OscGate-SSM already has extensive input-dependent gating in the state transition. A readout gate adds marginal benefit on top of rich state-level gating.
- **Fills a unique gap**: Only proposal addressing the readout/output path of efficient sequence models. All others modify the state transition or state structure.

---
status: ongoing
priority: high
created: 2026-02-15
based_on: semiring-monoid-lifting, recurrence-to-scan-reduction, blelloch-work-efficient-scan, online-softmax, input-dependent-gating, linear-attention-approximation
experiment_number: 014
experiment_log: experiment-log-014.md
---

# Log-Semiring SSM: Softmax-Native Parallel Scan via Logarithmic Semiring Recurrence

## Hypothesis

Replacing the standard $(\mathbb{R}, +, \times)$ semiring in SSM parallel scans with the **logarithmic semiring** $(\mathbb{R}, \text{logsumexp}, +)$ produces a recurrent model whose hidden state natively computes **softmax-weighted attention** over the input history — unifying the expressivity of softmax attention with the $O(T)$ sequential cost and $O(\log T)$ parallel depth of SSMs, without any kernel approximation.

## Background

### The Linear Attention Approximation Problem

Linear attention replaces $\text{softmax}(QK^\top)V$ with $\phi(Q)(\phi(K)^\top V)$, achieving $O(Td^2)$ cost via the associative scan over the state $S_t = \lambda_t S_{t-1} + \phi(k_t) v_t^\top$. The fundamental problem: **no feature map $\phi$ can exactly reproduce softmax attention**. All linear attention variants (FAVOR+, cosFormer, Hedgehog, etc.) are approximations that lose the sharp, sparse attention patterns that make softmax effective.

### The Key Insight: Softmax IS a Semiring Backward Pass

The documented trick **semiring-monoid-lifting** reveals a remarkable fact: the backward pass of the logarithmic semiring $(\mathbb{R}, \oplus_\mu, +)$ where $a \oplus_\mu b = \frac{1}{\mu} \log(e^{\mu a} + e^{\mu b})$ **is exactly the softmax function**:

$$
\frac{\partial (a \oplus_\mu b)}{\partial a} = \frac{e^{\mu a}}{e^{\mu a} + e^{\mu b}} = \text{softmax}_\mu([a, b])_1
$$

This means: if we define an SSM recurrence whose "addition" operation is $\text{logsumexp}$ and whose "multiplication" is standard addition, the resulting scan computes something analogous to softmax-weighted aggregation — **exactly**, not approximately.

### Why This Is Different from All Existing Proposals

- **Proposals 008, 009** (cosine-log-linear, post-sigmoid gating): modify the *feature map* or *readout* of standard linear attention — still approximate softmax
- **Proposal 005** (segmented HSS linear attention): changes the *state structure* but keeps the standard $(+, \times)$ semiring
- **All existing SSMs** (Mamba, S4, DeltaNet): operate in the standard semiring — their scan combines states via addition and multiplication

This proposal changes the **algebraic foundation** of the scan itself. The recurrence operates in log-space, and the "sum" over history is a logsumexp — mathematically equivalent to a softmax-weighted combination.

### The Hardware Challenge

The semiring-monoid-lifting trick documents a critical limitation: semiring operations are $\sim 16\times$ slower than tensor-core matmul because GPUs lack dedicated hardware for $(\text{logsumexp}, +)$. Our proposal addresses this via:

1. **Log-space arithmetic**: All computation stays in log-space, avoiding exp/log until the final readout. The scan operator is $\text{logsumexp}$ (implemented as $\max + \log(1 + e^{-|\cdot|})$, numerically stable)
2. **Online softmax fusion**: The documented online-softmax trick shows how to compute softmax in a single streaming pass with $O(1)$ state — we extend this to the parallel scan setting
3. **Scalar state per head**: Like Mamba-2, each head maintains a scalar log-space state, enabling element-wise scans that parallelize trivially

## Mathematical Formulation

**Standard SSM Recurrence (standard semiring):**

$$
h_t = a_t \cdot h_{t-1} + b_t \quad \text{over } (\mathbb{R}, +, \times)
$$

**Log-Semiring SSM Recurrence (proposed):**

$$
\ell_t = a_t + \ell_{t-1} \;\oplus_\mu\; b_t \quad \text{over } (\mathbb{R}, \oplus_\mu, +)
$$

where $\oplus_\mu$ is the log-addition: $x \oplus_\mu y = \frac{1}{\mu} \log(e^{\mu x} + e^{\mu y})$.

**Expanding the Recurrence:**

At temperature $\mu = 1$ (dropping $\mu$ for clarity):

$$
\ell_t = \text{logsumexp}(a_t + \ell_{t-1}, \; b_t)
$$

Unrolling:

$$
\ell_t = \text{logsumexp}\left(\sum_{s=j+1}^{t} a_s + b_j \;\Big|\; j = 0, 1, \ldots, t\right)
$$

This is the log-partition function of the distribution:

$$
p(j | t) = \text{softmax}\left(\sum_{s=j+1}^{t} a_s + b_j\right)_j = \frac{\exp\left(\sum_{s=j+1}^t a_s + b_j\right)}{\sum_{j'=0}^{t} \exp\left(\sum_{s=j'+1}^t a_s + b_{j'}\right)}
$$

**Interpretation:** The hidden state $\ell_t$ is the log-normalizer of a softmax distribution over the input history, where:
- $b_j = q^\top k_j / \sqrt{d}$ plays the role of the attention logit at position $j$
- $a_s$ plays the role of a **cumulative position-dependent decay** (analogous to ALiBi or relative position bias)

**Multi-Dimensional Extension (Vector State):**

For a $d$-dimensional state, operate element-wise in log-space:

$$
\ell_{t,i} = \text{logsumexp}(a_{t,i} + \ell_{t-1,i}, \; b_{t,i}) \quad \text{for } i = 1, \ldots, d
$$

**Full Architecture (Log-Semiring Attention SSM):**

Given input $x_t \in \mathbb{R}^D$:

1. **Projections**: $q_t, k_t \in \mathbb{R}^d$, $v_t \in \mathbb{R}^D$, $\alpha_t \in \mathbb{R}^d$ from linear projections of $x_t$

2. **Log-space input**: For each head dimension $i$:
$$
b_{t,i} = q_{t,i} \cdot k_{t,i} / \sqrt{d}
$$
$$
a_{t,i} = -\text{softplus}(w_\alpha^\top x_t + c_\alpha)_i \in (-\infty, 0)
$$

The decay $a_{t,i} < 0$ ensures the contribution of past tokens diminishes (input-dependent forgetting).

3. **Log-semiring scan**: Parallel scan with operator:
$$
(\ell_1, a_1) \bullet (\ell_2, a_2) = (\text{logsumexp}(a_2 + \ell_1, \ell_2), \; a_1 + a_2)
$$

This operator is **associative** (proof follows from associativity of logsumexp and addition).

4. **Readout via log-space softmax**:

To compute the output $y_t$, we need the softmax-weighted sum of values:

$$
y_t = \sum_{j=0}^{t} p(j|t) \cdot v_j
$$

We maintain a second set of log-space states for the numerator, for each value dimension $m$:

$$
n_{t,i,m} = \text{logsumexp}(a_{t,i} + n_{t-1,i,m}, \; b_{t,i} + \log |v_{t,m}|)
$$

with signs tracked separately. Then:

$$
y_{t,m} = \sum_{i=1}^{d} \exp(n_{t,i,m} - \ell_{t,i}) \cdot \text{sign}_{t,i,m}
$$

**Key Variables:**
- $\ell_t \in \mathbb{R}^d$ — log-normalizer state (log-partition function per head dim)
- $n_t \in \mathbb{R}^{d \times D}$ — log-numerator states
- $a_t \in \mathbb{R}^d$ — input-dependent log-decay (negative, controls forgetting)
- $b_t \in \mathbb{R}^d$ — attention logits ($q \cdot k / \sqrt{d}$)
- $v_t \in \mathbb{R}^D$ — value vectors
- $\mu > 0$ — temperature parameter (learnable or fixed)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Log-Semiring SSM (LogSSM) |
| Layers | $L = 12$ |
| Model dim | $D = 768$ |
| Heads | $H = 12$ |
| Head dim | $d = 64$ |
| Temperature | $\mu = 1$ (fixed initially) |
| Decay parameterization | Input-dependent via softplus |
| Scan algorithm | Blelloch parallel scan with logsumexp operator |

### Baseline

1. **Mamba-2** (diagonal SSM, standard semiring): $O(Tn)$ per layer
2. **Linear Attention** (standard semiring, $O(Td^2)$): RetNet / GLA style
3. **cosFormer** (cosine-reweighted linear attention): $O(Td^2)$
4. **Softmax Transformer**: $O(T^2 d)$ per layer (gold standard for quality)

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $> 0.8\times$ Mamba-2 tokens/sec | Timed forward pass |
| Memory | $< 1.5\times$ Mamba-2 | Peak GPU (extra state for numerator tracking) |
| MQAR accuracy | $> 90\%$ at 4 KV pairs | Multi-Query Associative Recall |
| Needle-in-haystack | $> 85\%$ at $T = 4096$ | Retrieval accuracy |
| Perplexity | $\leq$ softmax Transformer | WikiText-103, 380M params |

### Estimated Compute

**MVE**: ~8 minutes on single GPU (~$0.40)
**Small-scale**: 4 GPU-hours on A100 (~$16)
**Full-scale**: 48 GPU-hours on A100 (~$200)

## Expected Outcome

**If hypothesis is correct:**
- LogSSM achieves MQAR accuracy comparable to softmax Transformer ($> 90\%$ at 4 KV pairs) where linear attention variants achieve $< 70\%$
- Needle-in-haystack retrieval matches softmax quality — the log-semiring scan natively computes sharp attention over the full history
- Throughput is within $20\%$ of Mamba-2 — the logsumexp operation is slightly more expensive than multiply-add but the scan structure is identical
- Perplexity matches or beats softmax Transformer at matched parameters — the model can learn both sharp (high $\mu$) and diffuse (low $\mu$) attention patterns

**If hypothesis is wrong:**
- If MQAR doesn't improve over linear attention: the scalar log-space state per head may be too compressed to track multiple keys simultaneously. This motivates a vector-valued log-semiring state (log-space matrix, exponentially more expensive).
- If throughput is much worse: the logsumexp operation's lack of tensor-core support is fatal. This establishes a quantitative "hardware tax" for non-standard semirings and motivates custom hardware kernels.
- If training is unstable: log-space gradients may explode when $\ell_t$ values become very large or very small. This motivates the online-softmax normalization trick applied to the scan.

## Minimum Viable Experiment

### Setup
- **Model**: 2 layers, $D = 64$, $d = 16$, $H = 4$, ~80K params
- **Task**: Selective copying — given a sequence like `[a, _, b, _, c, QUERY:2] → b`, retrieve the token at a queried position. This requires sharp attention (selecting exactly one past token), which is the core capability of softmax that linear attention lacks.
- **Data**: 5K synthetic sequences of length 32, with 8 tokens to remember and 1 query
- **Compute**: Single GPU, $< 8$ minutes

### Success Criteria
- LogSSM achieves $> 90\%$ accuracy on selective copying at length 32
- Standard linear attention (RetNet-style) achieves $< 60\%$ on the same task
- Diagonal SSM (Mamba-style) achieves $< 70\%$ on the same task
- Training converges within 500 steps (no instability)

### Failure Criteria
- If LogSSM cannot beat linear attention on selective copying — a task specifically designed to require sharp attention — the log-semiring scan does not provide the hypothesized softmax-like sharpness
- If training diverges or produces NaN — the log-space arithmetic is numerically broken and needs fundamental fixes before proceeding

### Why This Test Is Sufficient
- Selective copying is the **minimal task that separates softmax from linear attention** — it requires attending to exactly one position with high precision
- The log-semiring scan is mathematically equivalent to computing softmax over the history, so it should solve this task as easily as a Transformer
- If the mechanism works at $d = 16$, scaling to $d = 64$ adds capacity (more heads) not qualitatively new behavior
- The numerics of logsumexp scanning are fully exercised at small scale — if they work for 32 steps, they work for 4096 steps (the online-softmax trick ensures bounded intermediate values)

## Theoretical Analysis

**Complexity comparison:**

| Operation | Softmax Attn | Linear Attn | Log-Semiring SSM |
|-----------|-------------|-------------|-----------------|
| Forward (per step) | $O(Td)$ | $O(d^2)$ | $O(dD)$ |
| Forward (full seq) | $O(T^2d)$ | $O(Td^2)$ | $O(TdD)$ |
| Parallel depth | $O(1)$ ‡ | $O(\log T)$ | $O(\log T)$ |
| Memory (state) | $O(T)$ | $O(d^2)$ | $O(dD)$ |
| Attention sharpness | Exact softmax | Approximate | Exact softmax |
| Decoding cost | $O(Td)$ per token | $O(d^2)$ per token | $O(dD)$ per token |

‡ Softmax attention is fully parallel but $O(T^2)$ work.

**Key insight:** LogSSM achieves the same $O(\log T)$ parallel depth as linear attention SSMs but with exact softmax-like attention patterns (no feature map approximation). The cost is $O(TdD)$ — same order as linear attention when $D \sim d$.

**Crossover vs. softmax Transformer:**

LogSSM is preferred when $T > D$ (sequence length exceeds model dimension), which is common for long-context tasks. At $T = 4096$, $D = 768$: LogSSM does $4096 \times 64 \times 768 \approx 200M$ operations vs. softmax's $4096^2 \times 64 \approx 1B$ operations — a $5\times$ advantage.

**Expressivity analysis:**

The log-semiring scan computes:

$$
p(j | t) = \frac{\exp\left(\sum_{s=j+1}^t a_s + b_j\right)}{\sum_{j'} \exp\left(\sum_{s=j'+1}^t a_s + b_{j'}\right)}
$$

This is a **causal softmax attention** distribution where:
- $b_j = q_t \cdot k_j / \sqrt{d}$ is the content-based attention score
- $\sum_{s=j+1}^t a_s$ is the cumulative positional decay (like ALiBi with input-dependent slopes)

Crucially, this is **not** an approximation — it is exactly the softmax attention distribution factored through a recurrent scan. The only limitation is that $q_t$ is shared across all $j$ (the query doesn't change per key), which is inherent to any recurrent formulation.

**Associativity proof for the scan operator:**

The operator $(\ell_1, a_1) \bullet (\ell_2, a_2) = (\text{logsumexp}(a_2 + \ell_1, \ell_2), a_1 + a_2)$ is associative because:

$((\ell_1, a_1) \bullet (\ell_2, a_2)) \bullet (\ell_3, a_3)$
$= (\text{lse}(a_2 + \ell_1, \ell_2), a_1 + a_2) \bullet (\ell_3, a_3)$
$= (\text{lse}(a_3 + \text{lse}(a_2 + \ell_1, \ell_2), \ell_3), a_1 + a_2 + a_3)$
$= (\text{lse}(a_2 + a_3 + \ell_1, a_3 + \ell_2, \ell_3), a_1 + a_2 + a_3)$

$(\ell_1, a_1) \bullet ((\ell_2, a_2) \bullet (\ell_3, a_3))$
$= (\ell_1, a_1) \bullet (\text{lse}(a_3 + \ell_2, \ell_3), a_2 + a_3)$
$= (\text{lse}(a_2 + a_3 + \ell_1, \text{lse}(a_3 + \ell_2, \ell_3)), a_1 + a_2 + a_3)$
$= (\text{lse}(a_2 + a_3 + \ell_1, a_3 + \ell_2, \ell_3), a_1 + a_2 + a_3)$

Both sides are equal. $\checkmark$

## Risks & Limitations

1. **No tensor-core acceleration**: The logsumexp operation runs on CUDA cores, not tensor cores. At $\sim 16\times$ lower throughput, this may negate the algorithmic advantage. Mitigation: the scan is element-wise (no matmul needed), so the comparison is CUDA-core element-wise ops vs. tensor-core matmul, a smaller gap than the $16\times$ figure for full matrix ops.

2. **Sign tracking for values**: The log-semiring naturally handles positive quantities. Tracking signed values (needed for $v_t$ with negative entries) requires maintaining separate positive/negative accumulators or using a signed-log representation, adding complexity and memory.

3. **Fixed query limitation**: In the recurrent formulation, $b_j = q_j \cdot k_j$ uses the query at time $j$, not the query at time $t$. This means the attention pattern is "causal content-based" but not identical to standard causal attention where $b_{t,j} = q_t \cdot k_j$. The model must learn to encode "what I want to attend to later" into $b_j$ at the time of input. This is analogous to the same limitation in all linear attention / SSM models.

4. **Temperature sensitivity**: The temperature $\mu$ controls sharpness. Too high $\mu$ → hard attention (sparse gradients). Too low $\mu$ → uniform attention (poor retrieval). May need annealing or per-head learning.

5. **Numerical stability at long sequences**: Even with logsumexp (which is numerically stable per operation), accumulating over very long sequences may lead to large magnitude differences that stress BF16 precision. The online-softmax trick from FlashAttention provides the mitigation pattern.

6. **Memory overhead from numerator tracking**: Tracking $d \times D$ log-numerator states is $D \times$ more expensive than the $d$-dimensional log-normalizer. For $D = 768$, $d = 64$, this is $12\times$ more state memory than the normalizer alone.

## Follow-up Experiments

1. **Learnable temperature $\mu$ per head**: Allow each attention head to learn its own sharpness, enabling some heads to do sharp retrieval ($\mu \gg 1$) and others to do diffuse aggregation ($\mu \ll 1$).

2. **Tropical limit ($\mu \to \infty$)**: In the hard-attention limit, logsumexp becomes max, and the scan becomes $\ell_t = \max(a_t + \ell_{t-1}, b_t)$. This is a **max-plus scan** — even simpler, and computes hard attention (winner-take-all). Test whether this extreme gives useful sparse attention patterns.

3. **Hybrid LogSSM + diagonal layers**: Alternate log-semiring layers (for attention-like retrieval) with standard-semiring diagonal layers (for per-coordinate gating and decay), combining the strengths of both.

4. **Chunkwise log-semiring scan**: Apply the chunkwise parallel scan trick where intra-chunk computation uses dense logsumexp and inter-chunk uses the scan. The intra-chunk computation is a softmax attention matrix — recovering exact chunked softmax attention as a special case.

5. **Connection to FlashAttention**: The online-softmax trick in FlashAttention is mathematically equivalent to computing the log-normalizer incrementally. LogSSM generalizes this from "online softmax over a sequence" to "parallel scan over log-semiring tuples," potentially enabling a FlashAttention-like kernel for recurrent softmax attention.

## References

- Smets, Donker, and Portegies (2024). Semiring Activation in Neural Networks. arXiv:2405.18805.
- Blelloch, G.E. (1990). Prefix Sums and Their Applications.
- Milakov & Gimelshein (2018). Online normalizer calculation for softmax.
- Katharopoulos et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.
- Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
- Gu & Dao (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2).

# 238: Titans Chunkwise Momentum Neural Memory

**Category**: parallelization
**Gain type**: expressivity
**Source**: Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test Time" (Google Research, ICML 2025)
**Paper**: papers/titans-test-time-memorization.pdf
**Documented**: 2026-02-15

## Description

Titans introduces a **chunkwise-parallelizable deep neural memory** module that learns to memorize at test time via online gradient descent with momentum and weight decay. The key computational trick is **tensorizing mini-batch gradient descent into batched matmuls** so that the inherently sequential memory update can be parallelized across chunks of the sequence, making it trainable with hardware-efficient operations on GPUs.

The core idea: the memory module $\mathcal{M}_t$ (a small MLP) is updated at each timestep by taking a gradient step on an associative memory loss $\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2$. Tokens that are "surprising" (high loss gradient) get stored more aggressively. This combines three mechanisms absent from standard linear attention/SSMs:

1. **Momentum** (past surprise): a linear recurrence $S_t = \eta_t S_{t-1} - \theta_t \nabla\ell$ that can be parallelized via **associative scan** within each chunk
2. **Weight decay** (forgetting): $(1 - \alpha_t)\mathcal{M}_{t-1}$ gating, generalizing the decay in GLA/Mamba-2
3. **Deep non-linearity**: Using $L_\mathcal{M} \geq 2$ MLP layers instead of a single linear map, giving strictly more expressive memory

**Why it matters for GPU efficiency**: For the linear memory case ($L_\mathcal{M} = 1$, i.e. $\mathcal{M}_t = W_t$), the gradient computation and weight updates collapse to **batched matrix multiplications** over sequence chunks. The momentum term becomes a standard parallel associative scan. All operations map directly to tensor core matmuls. The cross-chunk state passing uses the same pattern as chunkwise linear attention.

## Mathematical Form

**Core Memory Update (with momentum and weight decay):**

$$
\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t
$$

$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)
$$

where:
- $\mathcal{M}_t$ — neural memory parameters at time $t$
- $S_t$ — momentum (accumulated "past surprise")
- $\alpha_t \in [0, 1]$ — data-dependent forgetting gate
- $\eta_t$ — data-dependent momentum decay
- $\theta_t$ — data-dependent learning rate
- $\nabla \ell(\mathcal{M}_{t-1}; x_t)$ — "momentary surprise" (gradient of associative memory loss)

**Associative Memory Loss:**

$$
\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2
$$

where $\mathbf{k}_t = x_t W_K$ and $\mathbf{v}_t = x_t W_V$ are key-value projections.

**Linear Memory Case ($\mathcal{M}_t = W_t$):**

For a single linear layer, the gradient is:

$$
\nabla \ell(W_0; x_t) = (W_0 x_t - x_t x_t^\top) x_t^\top
$$

**Chunkwise Parallel Form (within a chunk of size $b$):**

The memory update over a full chunk $t = 1, \ldots, b$ can be written as:

$$
\mathcal{M}_t = \beta_t \mathcal{M}_0 - \sum_{i=1}^{t} \theta_i \frac{\beta_t}{\beta_i} \nabla\ell(\mathcal{M}_{t'}; x_i)
$$

where $\beta_i = \prod_{j=1}^{i} (1 - \alpha_j)$ and $t' = t - \text{mod}(t, b)$.

**Key Tensorization:**

$$
\nabla\ell(W_0; x_i) = \Theta_b \mathbf{B}_b (W_0 X - X) X^\top
$$

where $\Theta_b = \text{diag}(\theta_1, \ldots, \theta_b)$ and $\mathbf{B}_b$ encodes the $\beta_t / \beta_i$ ratios. This is computable via **batched matmul** operations.

**Momentum as Parallel Scan:**

The momentum recurrence $S_t = \eta_t S_{t-1} - \theta_t u_t$ (where $u_t = \nabla\ell$) is a **linear recurrence** with input-dependent transition $\eta_t$, directly parallelizable via **associative scan** (same as GLA/Mamba).

**Memory Retrieval:**

$$
y_t = \mathcal{M}_t^*(\mathbf{q}_t)
$$

where $\mathbf{q}_t = x_t W_Q$ and $\mathcal{M}_t^*$ denotes the forward pass **without** weight update (inference-only read).

## Complexity

| Operation | Standard Attention | Titans Linear Memory | Titans Deep Memory ($L_\mathcal{M}$ layers) |
|-----------|-------------------|---------------------|---------------------------------------------|
| Per-token update | N/A (no state) | $O(d_{in}^2)$ matmul | $O(L_\mathcal{M} \cdot d_{in}^2)$ |
| Sequence total | $O(N^2 d)$ | $O(N d_{in}^2)$ | $O(N L_\mathcal{M} d_{in}^2)$ |
| Chunkwise parallel | N/A | $O(N/b \cdot b^2 d + N/b \cdot d^2 b)$ | Same per MLP layer |

**Memory:** $O(d_{in}^2)$ per-head state (the weight matrix $W_t$), constant w.r.t. sequence length — same scaling as linear attention.

**Throughput:** Linear scaling in tokens/sec vs sequence length (Figure 8 in paper). At 760M params, ~30K tokens/sec. Slightly slower than Mamba-2 (~37K tokens/sec) due to deeper memory but comparable to Gated DeltaNet.

## Applicability

- **Language modeling**: Outperforms all modern recurrent models (Mamba, Mamba-2, GLA, DeltaNet, Gated DeltaNet, TTT) at 340M/400M/760M scales on perplexity and commonsense reasoning benchmarks
- **Long-context**: Scales to >2M tokens on needle-in-a-haystack tasks, outperforming GPT-4 and Llama 3.1-8B on BABILong benchmark
- **Hybrid architectures**: Three variants (MAC, MAG, MAL) combine neural memory with sliding-window attention for different efficiency/quality tradeoffs
- **Time series forecasting**: State-of-the-art on ETT, ECL, Traffic, Weather benchmarks
- **Beyond TC$^0$**: Theoretically more expressive than Transformers, diagonal linear RNNs, and DeltaNet for state tracking tasks

## Limitations

- **Deep memory throughput cost**: Each additional MLP layer in the memory ($L_\mathcal{M}$) reduces training throughput roughly linearly. $L_\mathcal{M} = 2$ is the sweet spot in practice (sufficient for non-linear approximation without excessive overhead).
- **Non-linear memory breaks full parallelism**: For deep memory ($L_\mathcal{M} \geq 2$), the gradient computation through non-linear layers requires sequential forward passes across chunks. Only the linear case ($L_\mathcal{M} = 1$) fully tensorizes into batched matmuls.
- **Kernel optimization gap**: The paper notes neural memory is "slightly slower than Mamba2 and Gated DeltaNet, mainly due to (1) having deep memory and (2) highly optimized kernel in the implementation of Mamba2." Custom CUDA kernels could close this gap.
- **Simplification tradeoff**: Making $\alpha, \theta, \eta$ functions of chunks (not tokens) accelerates training but loses token-level expressivity. The paper uses per-token control in experiments.
- **Memory capacity**: Fixed-size matrix memory still has capacity limits. On very long sequences with highly diverse information, the model must decide what to forget.

## Implementation Notes

```python
# Titans Neural Memory - Chunkwise Parallel Training (Linear Case)
def titans_memory_chunk(X, W0, W_K, W_V, alpha, theta, eta, chunk_size):
    """
    X: (B, N, d_in) input sequence
    W0: (d_in, d_in) initial memory weights
    W_K, W_V: (d_in, d_in) key/value projections
    alpha, theta, eta: (B, N) per-token gates
    """
    B, N, d = X.shape
    n_chunks = N // chunk_size

    # Project to keys and values
    K = X @ W_K  # (B, N, d)
    V = X @ W_V  # (B, N, d)

    M = W0  # Initial memory state
    outputs = []

    for c in range(n_chunks):
        # Get chunk slice
        X_c = X[:, c*chunk_size:(c+1)*chunk_size]  # (B, b, d)
        K_c = K[:, c*chunk_size:(c+1)*chunk_size]
        V_c = V[:, c*chunk_size:(c+1)*chunk_size]
        alpha_c = alpha[:, c*chunk_size:(c+1)*chunk_size]
        theta_c = theta[:, c*chunk_size:(c+1)*chunk_size]
        eta_c = eta[:, c*chunk_size:(c+1)*chunk_size]

        # === INTRA-CHUNK: Batched matmul for gradients ===
        # Gradient: (M @ K_c^T - V_c^T) @ K_c  -- batched matmul!
        residual = M @ K_c.transpose(-1, -2) - V_c.transpose(-1, -2)
        grad_batch = residual @ K_c  # (B, d, d) per chunk token

        # Weight decay + gradient with learned gates
        # Theta_b @ B_b @ grad_batch -> tensorized via matmul
        Theta_diag = torch.diag_embed(theta_c)  # (B, b, b)
        beta = torch.cumprod(1 - alpha_c, dim=1)  # (B, b)
        B_ratio = beta.unsqueeze(-1) / beta.unsqueeze(-2)  # (B, b, b)
        weighted_grad = Theta_diag @ B_ratio @ grad_batch

        # === INTRA-CHUNK: Momentum via parallel associative scan ===
        # S_t = eta_t * S_{t-1} - theta_t * u_t
        # This is a linear recurrence -> parallel scan!
        S = parallel_scan(eta_c, -theta_c * grad_batch)

        # Update memory with decay + momentum
        M = beta[:, -1:, None] * M + S[:, -1]

        # === RETRIEVAL: Forward pass without weight update ===
        Q_c = X_c @ W_Q  # (B, b, d)
        y_c = M @ Q_c.transpose(-1, -2)  # Read from memory
        outputs.append(y_c.transpose(-1, -2))

    return torch.cat(outputs, dim=1)
```

**Key GPU Efficiency Observations:**

1. **Matmul-dominant**: The gradient computation $(W_0 X - X)X^\top$ is a batched GEMM, perfectly suited for tensor cores
2. **Parallel scan for momentum**: The $S_t = \eta_t S_{t-1} - \theta_t u_t$ recurrence uses the same associative scan as GLA/Mamba, reusing existing optimized kernels
3. **Chunk-level parallelism**: All tokens within a chunk compute gradients simultaneously; only cross-chunk state passing is sequential
4. **Memory access**: The memory matrix $W_t \in \mathbb{R}^{d \times d}$ is small enough to stay in shared memory for typical head dimensions ($d \leq 128$), enabling IO-efficient updates
5. **Arithmetic intensity**: The $d \times d$ matmuls within each chunk have high arithmetic intensity ($O(d)$ FLOPs per byte), saturating compute rather than being memory-bound

## References

- Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test Time" (Google Research, arXiv:2501.00663, ICML 2025)
- Yu Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (TTT, 2024)
- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (GLA, 2023)
- Yang, Kautz & Hatamizadeh, "Gated Delta Networks" (ICLR 2025)
- Dao & Gu, "Transformers are SSMs" (Mamba-2/SSD, ICML 2024)

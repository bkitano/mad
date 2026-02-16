# 246: Sequence Accumulation with Pipeline Parallelism (SAPP)

**Category**: parallelization
**Gain type**: efficiency
**Source**: Sun et al. (2025), "Sequence Accumulation and Beyond: Infinite Context Length on Single GPU and Large Clusters" (AAAI-25)
**Paper**: [papers/sequence-accumulation-pipeline-parallelism.pdf]
**Documented**: 2026-02-15

## Description

Sequence Accumulation (SA) is a technique that enables **constant-memory training of linear sequence models on arbitrarily long contexts** by splitting the input into fixed-length sub-sequences and sequentially accumulating the recurrent state ($d \times d$ matrix) across chunks. Unlike sequence parallelism or context parallelism which shard the sequence across GPUs (requiring expensive all-to-all communication), SA keeps the entire sequence on a single device and processes chunks one at a time, carrying forward only the compact state matrix.

SAPP (Sequence Accumulation with Pipeline Parallelism) extends SA to multi-GPU training by replacing the micro-batch dimension of pipeline parallelism with the sub-sequence dimension. Each model shard processes sub-sequences sequentially on its own GPU, accumulating states locally **without any inter-device communication in the sequence dimension**. This makes SAPP compatible with existing distributed training strategies (DP, TP, CP) with zero additional synchronization overhead for the sequence dimension.

The technique is applicable to all linear sequence modeling methods that follow the unified recurrence $M_t = \Theta_t \diamond M_{t-1} + \hat{M}_t$, including linear attention, SSMs, and linear RNNs.

## Mathematical Form

**Unified Linear Recurrence:**

All supported models follow:

$$
\hat{M}_t = f(K_t^\top, V_t), \quad M_t = \Theta_t \diamond M_{t-1} + \hat{M}_t
$$

where $M_t \in \mathcal{R}^{d \times d}$ is the memory state, $\Theta_t$ is a coefficient (scalar, vector, or matrix), and $\diamond$ denotes matrix or Hadamard product.

**Sequence Accumulation (Forward Pass):**

Given input $X \in \mathcal{R}^{N \times d}$, split into $T$ sub-sequences $\{X_1, \ldots, X_T\}$ each of length $S = N/T$:

$$
Q_t = X_t W_Q, \quad K_t = X_t W_K, \quad V_t = X_t W_V
$$

$$
\hat{M}_t = f(K_t^\top, V_t)
$$

$$
M_t = \Theta_t \diamond M_{t-1} + \hat{M}_t, \quad M_0 = 0 \in \mathcal{R}^{d \times d}
$$

$$
O_t = Q_t M_t
$$

Each sub-sequence $t$ computes its local output using the accumulated state $M_t$, which is carried forward to the next sub-sequence. Only $M_t$ (size $d \times d$) persists between chunks.

**Sequence Accumulation (Backward Pass):**

$$
dQ_t = dO_t M_t^\top, \quad d\hat{M}_t = Q_t^\top dO_t
$$

$$
dM_t = \Theta_t \diamond dM_{t+1} + d\hat{M}_t, \quad dM_{T+1} = 0
$$

$$
dK_t = V_t dM_t^\top, \quad dV_t = K_t dM_t
$$

The backward pass runs in reverse order ($t = T, \ldots, 1$), accumulating gradient states $dM_t$ with the **same** constant memory cost.

**SAPP Integration with Pipeline Parallelism:**

In standard PP with $P$ accelerators and $B$ micro-batches, SAPP replaces micro-batches with $S$ sub-sequences. The data flow on device $p$ is:

$$
F_{p,0} \to F_{p,1} \to \cdots \to F_{p,S-1}
$$

where $F_{p,t}$ is the forward pass of sub-sequence $t$ on device $p$. Memory states accumulate locally:

$$
M_t^{(p)} = \Theta_t \diamond M_{t-1}^{(p)} + \hat{M}_t^{(p)}
$$

No inter-device communication is needed for the sequence dimension — only the standard PP send/receive of layer activations occurs between devices.

## Complexity

| Operation | Standard (full sequence) | With SA |
|-----------|-------------------------|---------|
| Memory (activations) | $O(N \cdot d)$ | $O(S \cdot d)$ = $O(d)$ per state |
| Memory (state) | $O(N \cdot d^2)$ total | $O(d^2)$ constant |
| Forward FLOPs | $O(N \cdot d^2)$ | $O(N \cdot d^2)$ (same) |
| Communication (SAPP) | $O(N)$ per sync | $O(1)$ in seq. dim. |

**Memory:** $O(d^2)$ constant for the accumulated state, independent of context length $N$. The sub-sequence length $S$ determines the per-chunk activation memory.

**Throughput:** SA on a single GPU with 1B params processes 2K-1024K context lengths with constant ~33GB memory. Standard PP OOMs at 32K context.

## Applicability

**Directly applicable to all unified linear recurrence models:**

| Model | Recurrence | $\Theta_t$ |
|-------|-----------|------------|
| Linear Attention | $KV_t = KV_{t-1} + K_t^\top V_t$ | — (no decay) |
| Lightning Attention | $KV_t = \lambda KV_{t-1} + K_t^\top V_t$ | $\lambda \in \mathcal{R}$ |
| RetNet | $KV_t = \lambda KV_{t-1} + K_t^\top V_t$ | $\lambda \in \mathcal{R}$ |
| GLA | $KV_t = \text{diag}(\lambda_t) KV_{t-1} + K_t^\top V_t$ | $\lambda_t \in \mathcal{R}^d$ |
| DeltaNet | $KV_t = (I - \lambda_t K_t^\top K_t) KV_{t-1} + \lambda_t K_t^\top V_t$ | input-dependent |
| Mamba2 | $KV_t = \lambda_t KV_{t-1} + K_t^\top V_t$ | $\lambda_t \in \mathcal{R}$ |
| RWKV-6 | $KV_t = \text{diag}(\lambda_t) KV_{t-1} + K^\top V_t$ | $\lambda_t \in \mathcal{R}^d$ |

**Training scenarios:**
- Single GPU with extremely long contexts (100K-1M+ tokens)
- Multi-GPU with pipeline parallelism for large models + long contexts
- Compatible with DP, TP, CP as orthogonal parallelism dimensions

## Limitations

- **Sequential across sub-sequences**: SA processes sub-sequences one at a time on a single device. This is inherently sequential in the time dimension — unlike parallel scan which achieves $O(\log T)$ span. SA trades temporal parallelism for constant memory.
- **Throughput vs. memory tradeoff**: While memory stays constant, throughput on a single GPU shows only modest increases with longer context (since more sub-sequences means more sequential steps). The benefit is enabling training at all, not speeding it up.
- **Not a replacement for parallel scan**: For contexts that fit in memory, parallel scan (chunkwise) is faster. SA is specifically for when the full sequence **does not fit** in memory.
- **Bubble time in SAPP**: Like standard pipeline parallelism, SAPP has bubble time (idle GPU cycles). The bubble ratio decreases as the number of sub-sequences grows: bubble$\%$ = $\frac{P-1}{S+P-1}$.
- **No gradient truncation**: Unlike truncated BPTT, the backward pass processes all sub-sequences without truncation. This is correct but means full backprop cost scales linearly with $T$.

## GPU Efficiency Analysis

**Memory access pattern**: Highly favorable — each sub-sequence is processed with standard matmul operations ($Q_t, K_t, V_t$ projections, state accumulation). The state $M_t$ is $d \times d$ and stored in HBM, read/written once per sub-sequence.

**Parallelism**: Within each sub-sequence, standard parallelism applies (tensor cores for matmuls). Across sub-sequences, computation is sequential but each step saturates the GPU.

**Tensor core utilization**: Excellent — the core operations are matmuls ($X_t W_Q$, $Q_t M_t$, $K_t^\top V_t$) that map directly to tensor cores. The state accumulation $M_t = \Theta \diamond M_{t-1} + \hat{M}_t$ is a simple matrix add/multiply.

**Practical results (A100 80GB):**
- SA single GPU: constant ~33GB for 2K-1024K context with 1B model
- SAPP 4 GPUs: constant ~18.2GB for 2K-8K context with 7B model
- Standard PP: OOM at 32K context
- Benchmark performance: comparable loss to baseline (3.710 vs 3.712) with 8x fewer iterations needed per effective batch

## Implementation Notes

```python
# Sequence Accumulation — Forward Pass
def sa_forward(X, W_Q, W_K, W_V, theta, sub_seq_len):
    """
    X: input tensor [N, d]
    Returns: output [N, d], accumulated states for backward
    """
    N, d = X.shape
    T = N // sub_seq_len
    M = torch.zeros(d, d, device=X.device)  # constant memory state
    outputs = []
    states = []  # store M_t in HBM for backward pass

    for t in range(T):
        X_t = X[t * sub_seq_len : (t+1) * sub_seq_len]
        Q_t = X_t @ W_Q
        K_t = X_t @ W_K
        V_t = X_t @ W_V

        M_hat_t = K_t.T @ V_t           # outer product update
        M = theta * M + M_hat_t          # accumulate state
        O_t = Q_t @ M                    # compute output

        outputs.append(O_t)
        states.append(M.clone())          # store in HBM for backward

    return torch.cat(outputs, dim=0), states

# Sequence Accumulation — Backward Pass
def sa_backward(dO, states, Q, K, V, theta):
    """Backward pass: reverse sequential accumulation of gradients."""
    T = len(states)
    d = states[0].shape[0]
    dM = torch.zeros(d, d, device=dO.device)

    for t in range(T-1, -1, -1):
        dO_t = dO[t]
        dM_hat_t = Q[t].T @ dO_t
        dM = theta * dM + dM_hat_t       # accumulate gradient state
        dK_t = V[t] @ dM.T
        dV_t = K[t] @ dM
        # ... accumulate parameter gradients

    return dQ, dK, dV
```

## References

- Sun, W., Liu, Y., Tang, X., & Mo, X. (2025). "Sequence Accumulation and Beyond: Infinite Context Length on Single GPU and Large Clusters." Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25), pp. 20725-20733.
- Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020.
- Qin, Z., et al. (2024a). "Unlocking the Secrets of Linear Complexity Sequence Model from A Unified Perspective." arXiv:2405.17383.
- Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." NeurIPS 2019.
- Sun, W., et al. (2024a). "Linear Attention Sequence Parallelism." arXiv:2404.02882.

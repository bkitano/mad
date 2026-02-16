# 198: Striped Attention — Causal Ring Attention Load Balancing via Interleaved Partitioning

**Category**: parallelization
**Gain type**: efficiency
**Source**: Brandon, Nrusimha, Qian, Ankner, Jin, Song & Ragan-Kelley (2023) — arXiv:2311.09431
**Paper**: [papers/striped-attention-ring-causal.pdf]
**Documented**: 2026-02-15

## Description

Ring Attention distributes causal self-attention across $N$ GPUs by partitioning the sequence into $N$ contiguous blocks of size $c = n_\text{seq}/N$ tokens. Each GPU holds one Q block stationary while K/V blocks rotate around the ring over $N$ rounds. The critical problem: **causal masking creates severe workload imbalance**. Device 0 (earliest tokens) has nearly all blocks fully masked out, while device $N-1$ (latest tokens) must compute fully unmasked blocks. Since Ring Attention's per-round latency is determined by the *slowest* device, the entire system runs at the pace of the fully-unmasked workload — gaining no benefit from causal masking on $N-1$ out of $N$ rounds.

**Striped Attention** fixes this with one simple change: instead of assigning *contiguous* subsequences to devices, it assigns *interleaved stripes*. Device $j$ receives tokens $\{j, j+N, j+2N, \ldots\}$. Because each device now holds tokens distributed uniformly throughout the sequence, approximately half of each device's query/key interactions are masked on every round — eliminating the imbalance.

This works because attention is *permutation equivariant* in its token dimension: the output at position $i$ depends only on the set of (Q, K, V) values, not their physical ordering in memory. By permuting the input before the first layer and keeping it permuted throughout, the striping is free (no per-layer communication cost). Position encodings (e.g., RoPE) are applied using original sequence positions, so the model sees the correct causal structure.

## Mathematical Form

**Stripe permutation:** Given $N$ devices, token $i$ is assigned to device $j = i \bmod N$.

$$
\sigma(i) = i \bmod N
$$

Device $j$ holds tokens $\{j, j+N, j+2N, \ldots, j + (c-1)N\}$ where $c = n_\text{seq}/N$.

**Causal mask in striped layout:** For blocks $Q_j$ (on device $j$) and $K_k, V_k$ (received from device $k$), the mask function is:

$$
\text{GetMask}_\text{Striped}(j, k) = \begin{cases}
\text{MASK}[x, y] = -\infty & \text{for } y \leq x, & \text{if } j < k \\
\text{MASK}[x, y] = -\infty & \text{for } y < x, & \text{if } j \geq k
\end{cases}
$$

This is the key insight: in the striped layout, *every* block pair has an approximately upper-triangular mask (roughly half masked), unlike Ring Attention where some block pairs are fully unmasked and others fully masked.

**Per-device workload:** When computing attention between $Q_j$ and $K_k, V_k$:

$$
\text{Work}(j, k) = \begin{cases}
\frac{c(c+1)}{2} & \text{if } j \geq k \\
\frac{c(c-1)}{2} & \text{if } j < k
\end{cases}
$$

This is approximately $\frac{c^2}{2}$ for every block pair, versus Ring Attention where some devices see workload $c^2$ and others $0$.

**Ring Attention's worst-case workload per round:** In standard Ring Attention, on round $i > 0$ there exists a device $j$ whose $(j, k)$ block pair is fully unmasked ($\text{Work} = c^2$) and another device whose pair is fully masked ($\text{Work} = 0$). Since latency = max(Work), causal masking provides no benefit after round 0.

**Striped Attention's workload per round:** Every device on every round has $\text{Work} \approx c^2/2$, so the $2\times$ potential savings from causal masking is realized on *every* round.

## Complexity

**Per-round latency (the quantity that determines wall-clock time):**

| Method | Best-case per-round work | Worst-case per-round work | Speedup |
|--------|------------------------|-------------------------|---------|
| Ring Attention | $\frac{c^2}{2}$ (round 0 only) | $c^2$ (rounds 1..N-1) | 1.0$\times$ |
| **Striped Attention** | $\frac{c(c-1)}{2}$ | $\frac{c(c+1)}{2}$ | **$\to 2\times$** |

**Idle fraction (fraction of total work that is wasted):**

| Method | Idle fraction | Asymptotic ($P \to \infty$) |
|--------|--------------|---------------------------|
| Ring Attention (contiguous) | $\frac{P^2 - P}{2P^2}$ | $\frac{1}{2}$ |
| **Striped Attention** | $\frac{1}{2P}$ (P even), 0 (P odd) | $0$ |

**Communication:** Identical to Ring Attention — $N$ rounds of P2P send/recv of K/V blocks, each of size $c \times d$. The striping adds no extra communication.

**Memory:** Identical to Ring Attention — $O(c \cdot d)$ per device for Q, K, V, plus softmax statistics.

**Measured speedups (A100 80GB GPUs, 256K sequence length):**

| Model | GPUs | Ring Attn | Striped Attn | Speedup |
|-------|------|-----------|-------------|---------|
| 1B | 8 (1,8) | baseline | 1.45$\times$ | 1.45$\times$ |
| 3B | 8 (2,4) | baseline | 1.42$\times$ | 1.42$\times$ |
| 7B | 8 (4,2) | baseline | 1.31$\times$ | 1.31$\times$ |

**TPU v4 (16 chips, 786K sequence length):**

| Model | Ring Attn | Striped Attn | Speedup |
|-------|-----------|-------------|---------|
| 1B | baseline | 1.65$\times$ | 1.65$\times$ |

## Applicability

- **All causal (autoregressive) transformer training with Ring Attention.** Drop-in replacement — only changes the token-to-device assignment and mask function. No changes to the attention kernel itself.

- **Works with any tile-skipping attention kernel:** FlashAttention's tile-level mask checking naturally benefits — tiles that are fully masked are skipped. Striped Attention ensures every device sees a balanced mix of skip-able and compute-required tiles.

- **Compatible with model parallelism:** Can be combined with tensor model parallelism on orthogonal dimensions. Tested with (model_parallel, seq_parallel) mesh configurations from (1,2) to (4,4).

- **Particularly relevant to LASP-2 context:** For hybrid linear+softmax models using LASP-2H (trick 176), the softmax attention layers use Ring Attention. Applying Striped Attention to those layers eliminates the causal workload imbalance, improving overall hybrid training throughput.

- **Bidirectional attention:** No benefit — bidirectional attention has no masking imbalance. Striped Attention specifically targets the causal case.

## Limitations

- **Requires global permutation before embedding layer:** Tokens must be physically reordered. For RoPE-based models, the position IDs and loss target token IDs must also be permuted. This is a one-time setup cost at data loading, not per-layer.

- **Tile-size dependent speedup:** With very large tiles (e.g., 4096$\times$4096 on A100), the smallest block size in Ring Attention has only a few tiles, limiting the benefit. On TPUs with smaller tiles (2048$\times$2048), Striped Attention can skip tiles even at the smallest block size, yielding larger speedups.

- **Does not eliminate communication:** Communication volume and pattern are identical to Ring Attention. The improvement comes purely from better compute utilization (less idle time due to masking imbalance).

- **Slightly unequal work between devices:** The diagonal blocks differ by $c$ operations ($\frac{c(c+1)}{2}$ vs $\frac{c(c-1)}{2}$), but this imbalance is $O(c)$ versus $O(c^2)$ total work — negligible as block size grows.

## Implementation Notes

```python
# Striped Attention — the key changes from Ring Attention
# 1. Permute input tokens before first layer
# 2. Modify the mask function

def stripe_permutation(tokens, position_ids, num_devices):
    """Permute tokens so device j gets tokens {j, j+N, j+2N, ...}"""
    N = num_devices
    seq_len = len(tokens)
    # Interleave: token i goes to device (i % N), local index (i // N)
    perm = []
    for device in range(N):
        perm.extend(range(device, seq_len, N))
    return tokens[perm], position_ids[perm]

def get_mask_striped(j, k, block_size):
    """
    Mask for Striped Attention.
    Unlike Ring Attention where mask depends on block ordering,
    Striped Attention always produces ~upper-triangular masks.

    j: device holding Q block (original positions j, j+N, j+2N, ...)
    k: device whose K/V block is being processed
    """
    mask = torch.zeros(block_size, block_size)
    if j < k:
        # Q positions are "earlier" on average — mask where y <= x
        # (in original sequence, Q[x] = x*N+j, K[y] = y*N+k, causal: Q >= K)
        # x*N+j >= y*N+k  =>  x >= y + (k-j)/N  =>  for k>j: x > y or (x==y and j>=k)
        mask = torch.where(torch.arange(block_size)[:,None] < torch.arange(block_size)[None,:] + 1,
                          0.0, float('-inf'))  # upper tri + diagonal masked
    else:  # j >= k
        mask = torch.where(torch.arange(block_size)[:,None] < torch.arange(block_size)[None,:],
                          0.0, float('-inf'))  # strictly upper tri masked
    return mask

# Ring Attention loop is UNCHANGED — only GetMask differs
def striped_attention(Q_blocks, K_blocks, V_blocks, N):
    """Same ring loop as Ring Attention, different mask function."""
    Out = [torch.zeros_like(Q_blocks[j]) for j in range(N)]
    stats = [init_softmax_stats() for _ in range(N)]

    for i in range(N):  # N communication rounds
        for j in range(N):  # all devices in parallel
            k = (j - i) % N
            mask = get_mask_striped(j, k, block_size)
            # AccumulateAttentionFragment with online softmax rescaling
            Out[j], stats[j] = accumulate_attention(
                Out[j], Q_blocks[j], K_blocks[k], V_blocks[k], mask, stats[j]
            )
        # Ring-rotate K/V blocks (overlapped with compute)
        K_blocks = ring_send_recv(K_blocks)
        V_blocks = ring_send_recv(V_blocks)

    return normalize(Out, stats)

# GPU Efficiency Analysis:
# - Zero extra communication cost over Ring Attention
# - All operations remain matmuls (tensor core friendly)
# - Better SM utilization: no devices are idle due to fully-masked blocks
# - Tile skipping is uniformly distributed across all devices and rounds
# - Memory access pattern unchanged (coalesced block reads)
# - The only "cost" is permuting token IDs at data loading time
```

**GPU efficiency properties:**

1. **No extra kernel launches:** Same ring loop, same number of rounds, same FlashAttention kernel calls.
2. **Better tensor core utilization:** Every device processes roughly $c^2/2$ FLOPs per round instead of some doing $c^2$ and others $0$. This means all SMs are busy on every round.
3. **Coalesced memory access preserved:** Within each block, tokens are contiguous in memory (they're just different tokens). FlashAttention's tiling pattern is unchanged.
4. **Arithmetic intensity unchanged:** Same FLOPs per byte as Ring Attention, but fewer total FLOPs wasted.
5. **Communication-computation overlap unchanged:** P2P send/recv of K/V blocks still overlaps with FlashAttention computation, exactly as in Ring Attention.

## References

- Brandon, W., Nrusimha, A., Qian, K., Ankner, Z., Jin, T., Song, Z., & Ragan-Kelley, J. (2023). Striped Attention: Faster Ring Attention for Causal Transformers. arXiv:2311.09431.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- Code: https://github.com/exists-forall/striped_attention/

# 186: DeepSpeed-Ulysses All-to-All Sequence Parallelism

**Category**: parallelization
**Gain type**: efficiency
**Source**: Jacobs, Tanaka, Zhang, Zhang, Song, Rajbhandari & He (2023) — arXiv:2309.14509
**Paper**: [papers/deepspeed-ulysses-all-to-all-sp.pdf]
**Documented**: 2026-02-15

## Description

When training transformers on long sequences distributed across $P$ GPUs, existing sequence parallelism (SP) methods either incur communication volume that grows linearly with sequence length $N$ regardless of $P$ (Megatron-LM SP), or require ring-style P2P communication with $O(P)$ latency steps (Ring Attention / ColAI-SP). DeepSpeed-Ulysses introduces a fundamentally different communication pattern: **All-to-All collectives that repartition between sequence-parallel and head-parallel layouts**.

The key insight is a **duality between sequence dimension and attention head dimension**. Before attention, each GPU holds $N/P$ tokens for all $H$ heads. An All-to-All transposes this to: each GPU holds all $N$ tokens for $H/P$ heads. Attention is then computed locally on each GPU over the full sequence length but only for its assigned head subset. A second All-to-All transposes back to sequence-parallel layout for the MLP and other layers.

This achieves **constant communication volume per link** of $O(N \cdot d / P)$ when $N$ and $P$ are scaled proportionally — the aggregate volume is $O(M/P)$ where $M = N \cdot h$ is the total message size, compared to $O(M)$ for Megatron-LM SP. The communication volume is $P\times$ smaller than Megatron-LM's AllGather + ReduceScatter pattern.

**Relevance to LASP-2 (trick 176):** DeepSpeed-Ulysses and LASP-2 represent two complementary approaches to sequence parallelism:
- **Ulysses** repartitions across heads via All-to-All — each GPU sees the full sequence for a head subset → works for any attention (softmax, linear, sparse)
- **LASP-2** exploits linear attention's fixed-size state $M_t \in \mathbb{R}^{d \times d}$ via AllGather — communication is sequence-length-independent → only works for linear attention

For hybrid linear+softmax models (LASP-2H), Ulysses's All-to-All is an alternative to LASP-2H's AllGather-on-KV for the softmax layers. USP (arXiv:2405.07719) combines both in a 2D mesh.

## Mathematical Form

**Setup:** Sequence of length $N$ distributed across $P$ GPUs. Each GPU $p$ holds local chunk $X_p \in \mathbb{R}^{(N/P) \times d}$. Model has $H$ attention heads with head dimension $h_s = d/H$.

**Step 1 — Local QKV projection (sequence-parallel, all GPUs):**

$$
Q_p, K_p, V_p = X_p W_Q, \; X_p W_K, \; X_p W_V \quad \in \mathbb{R}^{(N/P) \times d}
$$

Each GPU holds $N/P$ tokens across all $H$ heads. Reshape to $(N/P, H, h_s)$.

**Step 2 — All-to-All: sequence-parallel → head-parallel:**

$$
\hat{Q}_p, \hat{K}_p, \hat{V}_p = \texttt{AllToAll}(Q_p, K_p, V_p)
$$

After All-to-All, GPU $p$ holds all $N$ tokens but only for heads $\{p \cdot H/P, \ldots, (p+1) \cdot H/P - 1\}$:

$$
\hat{Q}_p \in \mathbb{R}^{N \times (H/P) \times h_s}
$$

The All-to-All sends/receives $3 \cdot N \cdot h_s / P$ elements per GPU pair, for an aggregate per-link volume of $3Nh/P$.

**Step 3 — Local attention (full sequence, head subset):**

$$
\hat{O}_p = \text{Attention}(\hat{Q}_p, \hat{K}_p, \hat{V}_p) \in \mathbb{R}^{N \times (H/P) \times h_s}
$$

This is a standard attention computation (softmax, linear, sparse — any variant) over the **full sequence** for $H/P$ heads. Compatible with FlashAttention-2 for the local kernel.

**Step 4 — All-to-All: head-parallel → sequence-parallel:**

$$
O_p = \texttt{AllToAll}(\hat{O}_p) \in \mathbb{R}^{(N/P) \times d}
$$

Per-link volume: $Nh/P$. Total aggregate volume for both All-to-All ops: $4Nh/P$.

**Key Definitions:**

- $P$ — number of GPUs in the sequence parallel group
- $N$ — total sequence length
- $H$ — number of attention heads
- $h_s = d/H$ — per-head dimension
- $d$ — hidden dimension (model width)
- $M = Nh$ — total activation message size

**Constraint:** $P \leq H$ (parallelism degree cannot exceed head count). With GQA ($H_{kv} < H_q$), $P \leq H_{kv}$ for the KV All-to-All.

## Complexity

**Communication cost per transformer layer:**

| Method | Pattern | Per-link volume | Total aggregate | Latency steps |
|--------|---------|-----------------|-----------------|---------------|
| Megatron-LM SP | 2× AllGather + 2× ReduceScatter | $M$ | $4M$ | $4 \cdot O(\log P)$ |
| Ring Attention (ColAI-SP) | $2(P-1)$ × P2P | $M/P$ | $2M(P-1)/P$ | $2(P-1)$ |
| **DS-Ulysses** | **2× All-to-All** | $M/P$ | $4M/P$ | $2$ |
| LASP-2 (linear attn only) | 2× AllGather | $BHd^2$ | $2BHd^2$ | $2$ |

Key comparison:
- Megatron-LM: per-link volume = $M = Nh$, **independent of $P$** → does not benefit from more GPUs
- DS-Ulysses: per-link volume = $M/P = Nh/P$, **scales with $P$** → adding GPUs reduces per-link communication
- LASP-2: volume = $BHd^2$, **independent of both $N$ and $P$** → best for linear attention

**Computation cost (per device):**

| Component | Cost |
|-----------|------|
| QKV projection | $O((N/P) \cdot d^2)$ |
| Attention (FlashAttention on full seq, $H/P$ heads) | $O(N^2 \cdot h_s \cdot H/P)$ |
| Output projection + MLP | $O((N/P) \cdot d^2)$ |

**Memory:** Each GPU stores activations for $N/P$ tokens (sequence-parallel phases) or $N$ tokens for $H/P$ heads (head-parallel phase). Combined with ZeRO-3, model states are also partitioned across both data-parallel and sequence-parallel ranks.

**Wall-clock performance (A100 GPUs):**

| Model | GPUs | DS-Ulysses TFLOPS/GPU | Megatron-LM TFLOPS/GPU | Speedup |
|-------|------|----------------------|----------------------|---------|
| GPT-7B (32K) | 32 | ~175 | ~100 | 1.75× |
| GPT-7B (128K) | 32 | ~165 | ~85 | 1.94× |
| GPT-30B (64K) | 64 | ~140 | ~60 | 2.33× |
| GPT-1.2B (1M) | 64 | ~95 | OOM | ∞ |

Sustained throughput: >175 TFLOPS/GPU (>54% of A100 peak) on 1.2B model at 1M tokens.

## Applicability

- **Any attention mechanism:** Softmax, linear, sparse, cross-attention — the All-to-All repartitioning is attention-agnostic. Each GPU runs standard local attention over full sequence for its head subset.

- **FlashAttention compatible:** The local attention kernel on each GPU operates on full-length sequences with $H/P$ heads, which maps directly to FlashAttention-2 with no modification.

- **Combines with ZeRO-3:** SP ranks join data-parallel ranks in the ZeRO partition group, enabling simultaneous scaling of sequence length and model size. Memory for model states is partitioned across $\text{DP} \times \text{SP}$ ranks.

- **Hybrid with LASP-2 (via USP):** In hybrid linear+softmax models, Ulysses can handle softmax layers while LASP-2 handles linear attention layers. USP (arXiv:2405.07719) formalizes this as a 2D $(P_\text{ulysses} \times P_\text{ring})$ mesh.

- **Multi-dimensional models (video/image DiT):** DSP (arXiv:2403.10266) extends Ulysses to dynamically switch the parallel dimension across spatial/temporal axes.

## Limitations

- **Head count constraint:** Parallelism degree $P \leq H$. With GQA (e.g., 8 KV heads), $P \leq 8$ for the Ulysses dimension. This is the primary scaling bottleneck — USP addresses it by combining with Ring Attention for the remaining factor.

- **Communication volume scales with $N$:** Unlike LASP-2 where AllGather volume is $O(d^2)$ independent of sequence length, Ulysses's All-to-All volume is $O(Nh/P)$ — it grows linearly with $N$. For very long sequences (>1M tokens), this becomes a bandwidth bottleneck even at large $P$.

- **All-to-All latency:** All-to-All requires all-pairs communication, which stresses bisection bandwidth. On hierarchical networks (intra-node NVLink + inter-node IB), All-to-All across nodes can be significantly slower than intra-node AllGather. LASP-2's AllGather of tiny $d^2$ states is much more network-friendly.

- **Quadratic attention cost remains:** Each GPU computes attention over the full $N$-length sequence (for $H/P$ heads). The $O(N^2)$ cost per head is unchanged — Ulysses provides memory distribution and communication reduction, not computational reduction.

- **No communication-computation overlap:** The two All-to-All operations are synchronization barriers — computation cannot proceed until the All-to-All completes. Unlike LASP-2 where AllGather overlaps with intra-chunk computation, or FLUX (trick 049) where AllGather is fused into the GEMM kernel.

## Implementation Notes

```python
# DeepSpeed-Ulysses Forward Pass — Core Algorithm
# Each GPU p in SP group of size P executes this

def ulysses_attention_forward(X_local, W_Q, W_K, W_V, W_O, sp_group):
    """
    X_local: (N/P, d) — local sequence chunk on this GPU
    Returns: O_local: (N/P, d) — local output chunk
    """
    P = get_world_size(sp_group)
    H = num_heads
    h_s = d // H  # per-head dim

    # Step 1: Local QKV projection (sequence-parallel, all GPUs)
    Q = X_local @ W_Q  # (N/P, d) = (N/P, H, h_s)
    K = X_local @ W_K  # (N/P, d)
    V = X_local @ W_V  # (N/P, d)

    # Reshape for All-to-All: (N/P, H, h_s) -> ready to scatter across heads
    Q = Q.view(N // P, H, h_s)
    K = K.view(N // P, H, h_s)
    V = V.view(N // P, H, h_s)

    # Step 2: All-to-All — sequence-parallel -> head-parallel
    # Before: GPU p has tokens [p*N/P : (p+1)*N/P] for ALL H heads
    # After:  GPU p has ALL N tokens for heads [p*H/P : (p+1)*H/P]
    Q_head = all_to_all(Q, scatter_dim=1, gather_dim=0, group=sp_group)
    K_head = all_to_all(K, scatter_dim=1, gather_dim=0, group=sp_group)
    V_head = all_to_all(V, scatter_dim=1, gather_dim=0, group=sp_group)
    # Q_head: (N, H/P, h_s) — full sequence, head subset

    # Step 3: Local attention — full sequence, H/P heads
    # This is a STANDARD attention call — works with FlashAttention-2!
    O_head = flash_attention(Q_head, K_head, V_head)  # (N, H/P, h_s)

    # Step 4: All-to-All — head-parallel -> sequence-parallel
    O_local = all_to_all(O_head, scatter_dim=0, gather_dim=1, group=sp_group)
    # O_local: (N/P, H, h_s) = (N/P, d) — back to local chunk

    # Step 5: Output projection (sequence-parallel)
    O_local = O_local.view(N // P, d) @ W_O  # (N/P, d)

    return O_local

# Key GPU efficiency properties:
# 1. Only 2 All-to-All collectives per attention layer
# 2. Per-link volume = Nh/P — scales inversely with P
# 3. Local attention kernel is standard FlashAttention (tensor-core friendly)
# 4. All-to-All is well-optimized in NCCL for NVSwitch topologies
# 5. Compatible with ZeRO-3 for model state partitioning
# 6. No code changes to attention kernel — only communication wrappers
```

**GPU efficiency analysis:**

1. **All-to-All on NVSwitch:** Modern DGX nodes with NVSwitch provide full-bisection bandwidth for All-to-All within a node (900 GB/s on H100 NVLink). For $P \leq 8$ (one node), the All-to-All latency is dominated by bandwidth, not network topology. For $P > 8$ (multi-node), inter-node IB (400 Gbps) becomes the bottleneck.

2. **Tensor core utilization:** The local attention computation is a standard FlashAttention-2 call over full $N$-length sequences. This maps perfectly to tensor cores (WGMMA on H100) with the same arithmetic intensity as single-GPU attention.

3. **Memory access pattern:** The All-to-All transpose between sequence and head dimensions is a bulk data movement — each GPU sends $N \cdot h_s / P$ contiguous elements to each peer. This is bandwidth-optimal with coalesced memory access.

4. **Comparison with LASP-2 for linear attention:** For linear attention models, LASP-2's AllGather of $d^2$ bytes is strictly better than Ulysses's All-to-All of $Nh/P$ bytes (since $d^2 \ll Nh/P$ for long sequences). However, Ulysses is attention-agnostic and doesn't require the right-product kernel trick.

## References

- Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, S. L., Rajbhandari, S., & He, Y. (2023). DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. arXiv:2309.14509.
- Fang, J. & Zhao, S. (2024). USP: A Unified Sequence Parallelism Approach for Long Context Generative AI. arXiv:2405.07719.
- Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2023). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023.
- Sun, W., Lan, D., Zhong, Y., Qu, X., & Cheng, Y. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.

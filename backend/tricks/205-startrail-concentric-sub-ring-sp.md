# 205: StarTrail — Concentric Sub-Ring Sequence Parallelism

**Category**: parallelization
**Gain type**: efficiency
**Source**: Liu, Wang, Cheng, Zhao, Wang, Zhao, Demmel & You (2025) — arXiv:2407.00611 (NeurIPS 2025)
**Paper**: [papers/startrail-concentric-ring-sp.pdf]
**Documented**: 2026-02-15

## Description

StarTrail introduces a **team-based concentric multi-ring** communication pattern that reduces the P2P communication volume of Ring Attention by up to 75%. The core insight is a divide-and-conquer decomposition of the single global ring into multiple smaller sub-rings connected by intra-team collective communication.

In standard Ring Attention with $P$ GPUs, each device must see all $P-1$ other KV partitions via sequential P2P steps. StarTrail instead:

1. **Groups GPUs into teams of size $C$**: Each team of $C$ GPUs performs an initial **AllGather** of Q, K, V within the team (fast intra-node NVLink). After this, every GPU in the team holds $C$ chunks of Q, K, V data.

2. **Forms $P/C^2$ concurrent sub-rings**: GPUs with the same local rank across different teams form a sub-ring. Each sub-ring passes KV blocks of size $CN/P$ (i.e., $C \times$ the original per-GPU chunk) through only $P/C^2 - 1$ P2P steps instead of $P - 1$.

3. **ReduceScatter to combine**: After the sub-ring iterations, each team member holds $1/C$ of the overall attention result. A ReduceScatter within the team combines these via online softmax accumulation.

The net effect: the P2P communication volume drops by factor $C$, and the number of P2P rounds drops by factor $C^2$. The intra-team collective communication (AllGather + ReduceScatter) is cheap because it runs on fast NVLink within a single node, while the expensive inter-node P2P traffic is dramatically reduced.

## Mathematical Form

**Setup:** $P$ GPUs, team size $C$ (where $1 \leq C \leq \sqrt{P}$). Sequence of length $N$ split into $P$ chunks, each of size $N/P$.

**Phase 1 — Preprocessing (intra-team AllGather):**

Each team of $C$ GPUs gathers Q, K, V within the team:

$$
Q_{\text{team}}, K_{\text{team}}, V_{\text{team}} = \texttt{AllGather\_QKV}(Q_i, K_i, V_i, \text{team\_group})
$$

After this, each GPU holds $C$ chunks: $Q_{\text{team}} \in \mathbb{R}^{CN/P \times d}$, etc.

**Phase 2 — Ring phase (inter-team P2P sub-rings):**

GPUs with matching local rank across teams form a sub-ring of size $P/C^2$. Each sub-ring passes KV blocks of size $CN/P$:

$$
\text{Number of P2P iterations} = \frac{P}{C^2} - 1
$$

At each iteration, device $i$ computes attention between its $Q_{\text{team}}$ and the received $K_{\text{current}}, V_{\text{current}}$:

$$
O_i, lse_i = \texttt{forward\_iteration}(lse_i, O_i, Q_{\text{team}}, K_{\text{current}}, V_{\text{current}})
$$

using online softmax accumulation (same as FlashAttention / BurstAttention).

**Phase 3 — Postprocessing (intra-team ReduceScatter):**

$$
O_{\text{final}} = \texttt{ReduceScatter\_combine}(lse, O, \text{team\_group})
$$

Each GPU gets its final $N/P$-length output, combining the partial results from all $C$ team members.

**Key Definitions:**

- $P$ — total number of GPUs
- $C$ — team size (StarTrail parallel size), $1 \leq C \leq \sqrt{P}$
- $N$ — total sequence length
- $B$ — batch size
- $H$ — model hidden dimension
- $d$ — head dimension
- $W$ — inter-GPU bandwidth
- $L$ — inter-GPU latency

**Communication volume analysis:**

Ring Attention total communication:

$$
V_{\text{Ring}} = (P-1) \cdot \frac{2BNH}{PW} + (P-1) \cdot L
$$

StarTrail collective (AllGather + ReduceScatter):

$$
V_{\text{collective}} = \frac{4BNH(C-1)}{PW}
$$

StarTrail P2P:

$$
V_{\text{P2P}} = \left(\frac{P}{C^2} - 1\right) \cdot \frac{2CBNH}{PW} + \left(\frac{P}{C^2} - 1\right) \cdot L = \frac{(P - C^2) \cdot 2BNH}{CPW} + \left(\frac{P}{C^2} - 1\right) \cdot L
$$

**P2P volume ratio (StarTrail / Ring Attention):**

$$
\frac{V_{\text{StarTrail,P2P}}}{V_{\text{Ring,P2P}}} \approx \frac{1}{C}
$$

For $C = 2$: 50% P2P reduction. For $C = 4$: 75% P2P reduction.

**Backward pass:** Structured as key/value outer loop, query inner loop. KV gradients ($dK_i, dV_i$) stay fixed on their home GPUs. Query gradients ($dQ$) circulate through the sub-rings. This avoids the need to communicate KV gradients, matching FlashAttention's backward loop structure.

## Complexity

**Communication cost per Transformer block:**

| Method | P2P Volume | Collective Volume | P2P Steps | Total Steps |
|--------|-----------|------------------|-----------|-------------|
| Ring Attention | $\frac{2BNH(P-1)}{PW}$ | $0$ | $P-1$ | $P-1$ |
| DeepSpeed Ulysses | $0$ | $\frac{4BNH(P-1)}{PW}$ | $0$ | $2$ |
| **StarTrail-$C$** | $\frac{2BNH(P-C^2)}{CPW}$ | $\frac{4BNH(C-1)}{PW}$ | $\frac{P}{C^2}-1$ | $\frac{P}{C^2}+1$ |

**Latency reduction:** Ring Attention has $P-1$ latency-bearing steps. StarTrail-$C$ has $P/C^2 - 1$ latency-bearing P2P steps plus 2 collective steps. For $P = 64, C = 4$: Ring = 63 steps, StarTrail = 3 P2P + 2 collective = 5 steps.

**Computation-to-communication ratio improvement:** StarTrail's communication volume per iteration is $C \times$ larger (chunks are $C \times$ bigger), while computation per iteration is $C^2 \times$ larger (each team member computes attention for $C$ query chunks against $C$ KV chunks). This $C \times$ higher compute-to-comm ratio makes overlap more effective.

**Memory overhead (peak memory):**

$$
PM_{\text{Ring}} = M_{m+o} + (Y + 4)A
$$

$$
PM_{\text{Star}} = M_{m+o} + (Y + 3C + 1)A
$$

where $A = BNH/P$ is one activation slice, $Y$ is number of layers, $M_{m+o}$ is model+optimizer memory. The extra memory is $(3C - 3)A$ for buffering the team's Q, K, V. For $C = 4$: ~13% more memory than Ring Attention (for the 30B model example in the paper).

## Applicability

- **Softmax attention at scale:** Tested on GPT-3B, GPT-7B (up to 512K tokens on 64 H100s) and DiT-1B (up to 512K on 32 A100s). NeurIPS 2025 paper.

- **Heterogeneous interconnect topologies:** The key advantage is localizing P2P traffic. Teams can be placed within a single node (fast NVLink at 900 GB/s on H100) while only the reduced inter-team P2P goes over slower inter-node links (InfiniBand/Ethernet). Ring Attention forces all GPUs into one ring regardless of topology.

- **Complementary to LASP-2 for hybrid models:** For hybrid linear+softmax architectures (LASP-2H), StarTrail handles the softmax attention layers with reduced inter-node communication, while LASP-2 (trick 176) handles linear attention layers via AllGather on $d \times d$ states. The team AllGather in StarTrail naturally mirrors LASP-2's AllGather pattern.

- **Compatible with DeepSpeed Ulysses:** StarTrail is orthogonal to head-sharding methods. The paper notes these can be combined: DeepSpeed Ulysses handles head parallelism, StarTrail handles sequence parallelism within each head group.

- **Strong and weak scaling:** Shows improvements at 8, 16, 32, and 64 GPUs in both scaling regimes.

## Limitations

- **Softmax attention only:** The ring P2P pattern passes full K/V tensors, which is inherent to softmax attention. For linear attention, LASP-2's $d \times d$ state AllGather is strictly superior.

- **Extra memory for team buffers:** The AllGather within teams requires $3(C-1)$ extra activation slices. For $C = 4$ this is up to 30% more memory for smaller models. The relative overhead shrinks for larger models (7.9% for 7B model).

- **Optimal $C$ depends on hardware:** The best team size depends on the compute-to-communication ratio, which varies with GPU generation, interconnect, and sequence length. On H100 with InfiniBand, $C = 2$ is often optimal; on A100 with Ethernet, $C = 4$ is better. Requires tuning per cluster.

- **P2P latency not eliminated:** Still has $P/C^2 - 1$ sequential P2P steps, each incurring network latency. For very large $P$, this remains a bottleneck. When $C = \sqrt{P}$, the P2P phase vanishes entirely but collective communication grows.

- **Team size constrained:** $C$ must be at most $\sqrt{P}$, and practically should divide $P$ evenly. For $P = 8$, only $C \in \{1, 2\}$ are valid. The scheme works best when $P$ is large relative to $C^2$.

## Implementation Notes

```python
# StarTrail Attention Forward (Algorithm 1)
# Each GPU executes this

def startrail_forward(x, query_fn, key_fn, value_fn,
                       rank, world_size, team_size):
    """
    x: (N/P, H) — local input on this GPU
    team_size: C — the StarTrail parallel size
    Returns: O_final: (N/P, d) — local attention output
    """
    P = world_size
    C = team_size

    # Compute local Q, K, V
    Q_local = query_fn(x)   # (N/P, d)
    K_local = key_fn(x)     # (N/P, d)
    V_local = value_fn(x)   # (N/P, d)

    # === Phase 1: Preprocessing — intra-team AllGather ===
    # Teams are groups of C GPUs (typically within same node)
    team_group = get_team_process_group(rank, C)
    Q_team = all_gather(Q_local, team_group)  # (CN/P, d)
    K_team = all_gather(K_local, team_group)  # (CN/P, d)
    V_team = all_gather(V_local, team_group)  # (CN/P, d)

    # Launch async P2P send of K_team, V_team to next sub-ring neighbor
    r_send, r_recv = get_P2P_targets(rank, world_size, C)
    async_send(K_team, V_team, dst=r_send)
    async_recv(K_next, V_next, src=r_recv)

    # === Phase 2: Ring phase — sub-ring P2P iterations ===
    O = zeros_like(Q_team)     # (CN/P, d)
    lse = zeros(Q_team.shape[0])  # log-sum-exp accumulators

    num_iterations = P // (C * C)  # P/C^2 iterations

    for i in range(num_iterations):
        if i != 0:
            wait(async_recv)
            K_current, V_current = K_next, V_next
        else:
            K_current, V_current = K_team, V_team

        # Launch async P2P for next iteration
        if i < num_iterations - 1:
            async_send(K_current, V_current, dst=r_send)
            async_recv(K_next, V_next, src=r_recv)

        # Compute local attention with online softmax
        # Uses FlashAttention kernel internally
        lse, O = forward_iteration(lse, O, Q_team, K_current, V_current)

    # === Phase 3: Postprocessing — intra-team ReduceScatter ===
    # Each team member computed 1/C of total result; combine via lse
    O_final = reduce_scatter_combine(lse, O, team_group)

    return O_final  # (N/P, d)

# Key GPU efficiency properties:
# 1. Intra-team AllGather runs on NVLink (900 GB/s) — fast
# 2. Inter-team P2P reduced by C^2x (only P/C^2 - 1 rounds)
# 3. Each P2P round has C^2x more compute — better overlap
# 4. Double-buffer async P2P hides remaining latency
# 5. Internal attention uses FlashAttention (tensor core friendly)
# 6. ZigZag dataloader balances causal masking load across GPUs
```

**GPU efficiency analysis:**

1. **Topology-aware communication:** The critical insight is placing teams within NVLink-connected nodes. Intra-team AllGather at 900 GB/s (H100 NVSwitch) or 600 GB/s (A100 NVLink) is 5-10x faster than inter-node P2P over InfiniBand/Ethernet. StarTrail concentrates the expensive operation (attention redistribution) on fast links and minimizes slow inter-node traffic.

2. **Better compute-to-communication ratio:** Each sub-ring iteration processes $C \times$ more data ($CN/P$ tokens of KV) with $C^2 \times$ more FLOPs (team Q of size $CN/P$ against team KV of size $CN/P$). The communication per iteration is only $C \times$ more. This $C \times$ improvement in arithmetic intensity makes it easier to fully overlap communication behind computation.

3. **Latency reduction:** For $P = 64, C = 4$: Ring Attention has 63 sequential P2P steps. StarTrail has only 3 P2P steps ($P/C^2 - 1 = 64/16 - 1 = 3$). Even if each StarTrail step is $C \times$ larger in volume, the latency savings ($63L$ vs $5L$) are significant on high-latency interconnects.

4. **Measured performance:**
   - Up to 2.114x throughput over Ring Attention on GPT models (H100 cluster, 64 GPUs, 128K sequence)
   - Up to 2.143x on DiT models (A100 cluster, 32 GPUs, 512K sequence)
   - 7.9-30.79% more GPU memory (acceptable tradeoff for models > 3B)

## References

- Liu, Z., Wang, S., Cheng, S., Zhao, Z., Wang, K., Zhao, X., Demmel, J., & You, Y. (2025). StarTrail: Concentric Ring Sequence Parallelism for Efficient Near-Infinite-Context Transformer Model Training. NeurIPS 2025. arXiv:2407.00611.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.
- Jacobs, S.A., et al. (2023). DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. arXiv:2309.14509.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- Sun, W., et al. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
- Fang, J. & Zhao, S. (2024). USP: A Unified Sequence Parallelism Approach for Long Context Generative AI. arXiv:2405.07719.

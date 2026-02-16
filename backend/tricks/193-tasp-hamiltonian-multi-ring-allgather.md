# 193: TASP — Topology-Aware Multi-Ring AllGather via Hamiltonian Decomposition

**Category**: parallelization
**Gain type**: efficiency
**Source**: Wang, Hong, Li, Xu, Wang, Dai & Wang (2025) — arXiv:2509.26541
**Paper**: [papers/tasp-topology-aware-sequence-parallelism.pdf]
**Documented**: 2026-02-15

## Description

Ring Attention distributes long sequences across $n$ GPUs and uses a **Ring AllGather** communication primitive: KV blocks circulate in a single ring, with one GPU sending to the next each iteration. On modern accelerator topologies (NVSwitch on H100, full-mesh on MI300X), this ring pattern utilizes only $\frac{1}{n-1}$ of available communication links at each step — approximately **14.3%** for 8 GPUs. The remaining links sit idle, making communication the dominant bottleneck when the compute-to-communication ratio (CCR) drops below 1.0.

TASP (Topology-Aware Sequence Parallelism) resolves this mismatch through two decompositions:

1. **Topology Decomposition**: The fully-connected accelerator topology (modeled as a complete directed graph $K_n$) is decomposed into $n-1$ **edge-disjoint Hamiltonian cycles** using the classical Hamiltonian decomposition of complete directed graphs. Each cycle is a ring that visits all $n$ GPUs exactly once, and no two cycles share any edge — so all $n-1$ rings can transfer data **simultaneously without interference**.

2. **Primitive Decomposition**: The standard Ring AllGather (which transfers one KV block per ring step over a single ring) is decomposed into a **Multi-Ring AllGather** where $n-1$ concurrent ring-style transfers happen in parallel across the $n-1$ orthogonal ring datapaths. Each query chunk is assigned multiple KV blocks, and each KV block circulates on its own dedicated ring.

The result: **all $n(n-1)$ communication links are utilized at every step**, achieving $(n-1)\times$ the bandwidth of standard Ring Attention. For 8 GPUs, this means 7 concurrent rings instead of 1 — a theoretical 7× communication bandwidth improvement.

## Mathematical Form

**Topology as a directed graph:**

$$
G = (V, E), \quad V = \{R_0, R_1, \ldots, R_{n-1}\}
$$

where $R_i$ is the $i$-th GPU and $E$ contains all directed communication links. For a fully-connected topology (NVSwitch or full-mesh):

$$
G = K_n, \quad |E| = n(n-1)
$$

**Hamiltonian decomposition of $K_n$:**

A ring datapath $D \subseteq G$ is a directed Hamiltonian cycle — it visits every vertex exactly once. Two ring datapaths $D_i = (V, E_i)$ and $D_j = (V, E_j)$ are **orthogonal** if:

$$
E_i \cap E_j = \emptyset
$$

The complete directed graph $K_n$ can be decomposed into exactly $n-1$ edge-disjoint Hamiltonian cycles (for $n \geq 8$, $n \mod 4 = 0$):

$$
K_n = D_0 \cup D_1 \cup \cdots \cup D_{n-2}, \quad E_i \cap E_j = \emptyset \;\forall\; i \neq j
$$

This is computed in $O(n)$ time via Algorithm 1 (based on classical constructions from combinatorics).

**Multi-Ring AllGather:**

In standard Ring AllGather with $n$ GPUs, each iteration $t$ transfers one KV block per GPU along a single ring $D_0$:

$$
\text{Ring AllGather: } 1 \text{ ring} \times (n-1) \text{ iterations} \times 1 \text{ block/iter}
$$

In TASP's Multi-Ring AllGather, all $m = n-1$ rings operate concurrently. Each query chunk is split into $m$ sub-chunks, and each sub-chunk's corresponding KV blocks circulate on a dedicated ring:

$$
\text{Multi-Ring AllGather: } m \text{ rings} \times (n-1) \text{ iterations} \times 1 \text{ block/ring/iter}
$$

Total KV blocks transferred per iteration: $m = n-1$ (vs. 1 for standard Ring AllGather).

**Chunk placement for causal masking (Zig-zag TASP):**

Each KV chunk $t[i, j]$ assigned to accelerator $j$ and circulated via ring $i$ is split into two halves from opposite ends of the sequence:

$$
t[i, j][0] = KV\left[\frac{(2nj + i)S}{2n(n-1)}, \frac{(2nj + i + 1)S}{2n(n-1)}\right)
$$

$$
t[i, j][1] = KV\left[S - \frac{(2nj + i + 1)S}{2n(n-1)}, S - \frac{(2nj + i)S}{2n(n-1)}\right)
$$

where $S$ is the total sequence length. This ensures **100% load balance** under causal masking: at every iteration, each GPU computes attention over exactly half of the transferred KV tokens (one half falls within the causal mask, the other half falls outside).

**Properties preserved:**

- **Accessibility**: After $n-1$ iterations, every $Q_i$ has attended to every $KV_j$ (each Hamiltonian cycle ensures full circulation).
- **Zero-copy**: Each KV tensor exists in exactly one copy at any time (transferred along exactly one ring).
- **Load balance**: Zig-zag placement ensures uniform computation per GPU per iteration.

**Key Definitions:**

- $n$ — number of accelerators (GPUs)
- $K_n$ — complete directed graph on $n$ vertices
- $D_i$ — the $i$-th Hamiltonian cycle (ring datapath)
- $m = n-1$ — number of orthogonal ring datapaths
- $S$ — total sequence length
- $CCR^B$ — baseline compute-to-communication ratio (lower = more communication-bound)

## Complexity

**Communication link utilization per iteration:**

| Method | Active links/iter | Total links | Utilization |
|--------|-------------------|-------------|-------------|
| Ring Attention | $n$ | $n(n-1)$ | $\frac{1}{n-1}$ |
| **TASP (Multi-Ring)** | $n(n-1)$ | $n(n-1)$ | **100%** |

For $n = 8$: Ring Attention uses 14.3% of links; TASP uses 100%.

**Effective communication bandwidth:**

$$
BW_{\text{Ring}} = \frac{BW_{\text{link}}}{1} \quad \text{(one ring active)}
$$

$$
BW_{\text{TASP}} = (n-1) \times BW_{\text{link}} \quad \text{(all rings active)}
$$

where $BW_{\text{link}}$ is the per-link bandwidth. TASP achieves $(n-1)\times$ the effective bandwidth.

**Multi-node decomposition** — for $u$ nodes of $m$ GPUs each:

The $(m - K_m - m)^u$-decomposition uses Hamiltonian **paths** (not cycles) to account for heterogeneous intra-node (NVLink) vs. inter-node (InfiniBand) bandwidth:

- Intra-node: $m-1$ Hamiltonian cycles from $K_m$ decomposition → bandwidth $\approx 64$ GB/s per link (H100)
- Inter-node: $m$ Hamiltonian paths connecting nodes via IB NICs → bandwidth $\approx 50$ GB/s per link (H100)
- Only a 28% bandwidth gap (vs. >9× gap with $K_{m \times u}$-decomposition that treats all links equally)

**Computation overhead:**

| Component | Cost | Notes |
|-----------|------|-------|
| Routing table precomputation | $O(n^2)$ | One-time, before forward pass |
| KV concatenation for FlashAttention | 1–5% of $t_{\text{comp}}$ | Concatenating $m$ KV blocks before kernel call |
| CPU overhead (kernel launch) | Negligible | Overlapped with GPU computation |

## Wall-Clock Performance

**Single-node (8 GPUs, CCR $\geq$ threshold):**

| Platform | Mask | Avg Speedup | Max Speedup |
|----------|------|-------------|-------------|
| H100 NVSwitch | Full | 1.05× | 2.31× |
| H100 NVSwitch | Causal | 1.08× | 2.25× |
| MI300X full-mesh | Full | 1.57× | 3.58× |
| MI300X full-mesh | Causal | 1.68× | 3.58× |

TASP benefits more on MI300X because its lower intra-node bandwidth (vs. NVSwitch) makes it more communication-bound.

**Communication time speedup (MI300X, batch=48):**

| Seq Length | 10K | 20K | 40K | 50K | 100K |
|------------|-----|-----|-----|-----|------|
| CCR | 0.39 | 0.65 | 0.80 | 0.98 | 1.17 |
| Speedup | 2.4× | 1.8× | 1.5× | 1.3× | 1.1× |

Speedup is highest when CCR is lowest (most communication-bound).

**Multi-node (H100, full-mask attention):**

| Config | Decomposition | Avg Speedup |
|--------|---------------|-------------|
| 2 nodes (16 GPUs) | $K_{8 \times 2}$ | 1.43× |
| 2 nodes (16 GPUs) | $(8-K_8-8)^2$ | 1.20× |
| 4 nodes (32 GPUs) | $K_{8 \times 4}$ | 1.27× |
| 4 nodes (32 GPUs) | $(8-K_8-8)^4$ | 1.20× |

The $(m-K_m-m)^u$ decomposition scales better across nodes due to balanced NVLink/IB utilization.

## Applicability

- **Any Ring Attention variant**: TASP is a drop-in replacement for the Ring AllGather communication primitive. Works with standard Ring Attention, Zigzag Ring Attention, Striped Attention, and any other ring-based SP method. No changes to the attention kernel itself.

- **Softmax attention focus**: Designed for full (softmax) attention where KV blocks must be communicated. The communication optimization directly reduces the $O(N \cdot d / P)$ KV transfer time per ring iteration by utilizing all links simultaneously.

- **Complementary to LASP-2 and ZeCO**: For hybrid linear+softmax models (LASP-2H), TASP can handle the softmax attention layers (which still need ring-style KV exchange) while LASP-2/ZeCO handles the linear attention layers (which use state-based communication). The two approaches operate on orthogonal communication patterns.

- **Hardware-agnostic with topology-specific optimization**: Works on NVSwitch (H100), full-mesh (MI300X), multi-node IB clusters. The Hamiltonian decomposition adapts to the specific topology graph.

- **FlashAttention compatible**: The local attention kernel on each GPU is standard FlashAttention-2 — TASP only modifies which KV blocks arrive and when, not how attention is computed.

- **Inference prefill**: Particularly relevant for long-context inference prefill where CCR < 1.0 is common (short batch dimension, long sequence).

## Limitations

- **Diminishing returns at high CCR**: When compute dominates communication ($CCR^B > 1.5$), TASP's speedup vanishes because communication is already fully hidden behind computation. This occurs for very long sequences or large batch sizes where attention FLOPs dominate.

- **More KV chunks to manage**: In a $K_8$-decomposition, each GPU handles 7 KV chunks per ring iteration (vs. 1 for standard Ring Attention). This requires concatenating KV blocks before calling FlashAttention, adding 1–5% computation overhead.

- **AllToAll implementation dependency**: The $K_{m \times u}$-decomposition for multi-node uses NCCL's AllToAll primitive, which is well-optimized. But the $(m-K_m-m)^u$-decomposition requires batched SendRecv (custom scheduling), which has higher CPU overhead and less low-level optimization than native NCCL collectives.

- **Memory for extra KV buffers**: Each GPU needs receive buffers for KV blocks from all $n-1$ concurrent rings. For 8 GPUs, this is 7× the buffer memory vs. standard Ring Attention (which needs only 1 extra KV buffer). For large KV blocks, this can increase peak memory.

- **Requires $n \mod 4 = 0$**: The Hamiltonian decomposition algorithm requires $n \geq 8$ and $n \mod 4 = 0$. This covers the common cases (8, 16, 32, 64) but excludes odd GPU counts or small configurations.

## Implementation Notes

```python
# TASP Forward Pass — Algorithm 3 (simplified)
# Uses pre-computed routing tables from Hamiltonian decomposition

def tasp_forward(q, k, v, world_size, rank):
    """
    q: (seq_local, d) — local query chunk on this GPU
    k, v: (seq_local, d) — local KV chunk on this GPU
    Returns: output, logsumexp
    """
    n = world_size
    m = n - 1  # number of concurrent rings

    # Step 0: Pre-compute routing tables (one-time)
    # gen_hamilton_circle returns the Hamiltonian decomposition of K_n
    looping = gen_hamilton_circle(n)       # m Hamilton cycles
    out_mapping = cal_out_mapping(looping) # where to send at each step
    in_mapping = cal_in_mapping(looping)   # where to receive at each step

    # Step 1: Stack KV blocks for all m rings
    # In TASP, each GPU holds m KV sub-chunks (from zig-zag placement)
    # this_kv[i] = (k_i, v_i) for ring i
    this_kv = [stack(k[i], v[i]) for i in range(m)]
    next_kv = [empty_like(this_kv[0]) for _ in range(m)]

    # Step 2: Iterative ring attention with multi-ring communication
    output = None
    lse = None

    for step in range(n - 1):
        # Extract current KV blocks from all m rings
        k_cat = concat([this_kv[i][0] for i in range(m)], dim=0)  # seqlen
        v_cat = concat([this_kv[i][1] for i in range(m)], dim=0)

        # Launch async AllToAll for NEXT iteration's KV blocks
        if step < n - 2:
            for i in range(m):
                send_buf = this_kv[out_mapping[i][step]]
                recv_buf = next_kv[in_mapping[i][step]]
                all_to_all_async(send_buf, recv_buf)
                # All m rings transfer concurrently — 100% link utilization!

        # Compute local FlashAttention with concatenated KV
        out_curr, lse_curr = flash_attention(q, k_cat, v_cat)

        # Online softmax accumulation (merge with previous iterations)
        output, lse = update_out_lse(output, lse, out_curr, lse_curr)

        # Wait for communication, swap buffers
        if step < n - 2:
            wait_all()
            this_kv, next_kv = next_kv, this_kv

    return output, lse

def gen_hamilton_circle(n):
    """
    Decompose K_n into n-1 edge-disjoint Hamiltonian cycles.
    Returns: looping matrix of shape (n-1, n) where looping[i][j]
             gives the next GPU in ring i starting from GPU j.
    Time complexity: O(n)
    """
    # Based on classical construction (Algorithm 1 in paper):
    # 1. Generate n-2 Hamiltonian paths via GetPath
    # 2. Close paths into cycles via GenCycles
    # 3. Apply rotational shifts to ensure edge-disjointness
    k = n // 4 - 1
    shifts = [0, 1, k+1, 4*k+2, 2*k+2, 3, ...]  # pattern from paper
    init_cycles = gen_cycles(get_path(n - 2))
    rotated = apply_shifts(init_cycles, shifts, n)
    return close_into_cycles(rotated)

# Key GPU efficiency properties:
# 1. ALL n(n-1) links active at every step (vs n for Ring Attention)
# 2. Communication bandwidth: (n-1)x improvement over Ring Attention
# 3. Zig-zag placement ensures 100% load balance with causal masking
# 4. Routing tables pre-computed — zero CPU overhead during attention
# 5. FlashAttention kernel unchanged — only communication wrapper changes
# 6. Works with NCCL AllToAll for K_{mxu} decomposition (multi-node)
# 7. Speedup grows as CCR decreases (most benefit when comm-bound)
```

**GPU efficiency analysis:**

1. **Full NVSwitch utilization**: On H100 with NVSwitch4 (3.6 TB/s bisection bandwidth), standard Ring Attention uses ~514 GB/s (one ring direction). TASP uses all 7 concurrent rings for ~3.6 TB/s — a 7× improvement in effective bandwidth.

2. **Tensor core utilization**: Attention computation is unchanged — standard FlashAttention-2 on concatenated KV blocks. The concatenation introduces a small overhead (1–5%) from memory copies but all matmuls map to WGMMA/MMA as before.

3. **Memory access pattern**: The multi-ring AllToAll sends/receives contiguous KV blocks. Each GPU receives $m$ blocks from $m$ different peers simultaneously — all transfers are to/from contiguous buffers in GPU memory, so memory access is coalesced.

4. **Communication-computation overlap**: Same as standard Ring Attention: FlashAttention computation for iteration $t$ overlaps with AllToAll communication for iteration $t+1$. With TASP, the communication finishes $(n-1)\times$ faster, so the overlap condition ($t_{\text{comp}} \geq t_{\text{comm}}$) is satisfied at much shorter sequence lengths.

5. **Multi-node bandwidth balance**: The $(m-K_m-m)^u$ decomposition ensures intra-node NVLink ($\sim$64 GB/s per link) and inter-node IB ($\sim$50 GB/s per link) are loaded proportionally — only 28% bandwidth gap vs. >9× gap with naive $K_{m \times u}$ decomposition.

## References

- Wang, Y., Hong, K., Li, X., Xu, Y., Wang, W., Dai, G., & Wang, Y. (2025). TASP: Topology-Aware Sequence Parallelism. arXiv:2509.26541.
- Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.
- Brandon, W., Nrusimha, A., Qian, K., et al. (2023). Striped Attention: Faster Ring Attention for Causal Transformers. arXiv:2311.09431.
- Jacobs, S. A., et al. (2023). DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. arXiv:2309.14509.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- NVIDIA (2023). Megatron Context Parallelism. NVIDIA NeMo documentation.
- Thakur, R., Rabenseifner, R., & Gropp, W. (2005). Optimization of Collective Communication Operations in MPICH. Int. J. High Performance Computing Applications, 19(1):49–66.

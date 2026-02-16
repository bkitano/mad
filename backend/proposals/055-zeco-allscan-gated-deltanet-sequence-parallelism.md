---
status: ongoing
priority: high
created: 2026-02-16
based_on: zeco-all-scan-fused-gather-prefix-sum (192), gated-deltanet-chunkwise-wy-gating (203), fused-chunkwise-ssd-atomic-state-passing (182), lasp2-allgather-sequence-parallelism (176), deepspeed-ulysses-all-to-all-sequence-parallelism (186), chunkwise-parallel-scan (026)
experiment_number: 055
experiment_log: experiment-log-055.md
---

# ZeCO All-Scan Sequence Parallelism for Gated DeltaNet with WY State Factorization

## Hypothesis

Extending ZeCO's All-Scan collective — which achieves $P$-independent communication volume for diagonal-decay linear RNNs — to **Gated DeltaNet's Householder-gated transition** by transmitting a factored WY representation of the inter-device state correction (two low-rank matrices $W, Y \in \mathbb{R}^{d_k \times C}$ plus a diagonal decay vector $\gamma \in \mathbb{R}^{d_k}$) instead of the full $d_v \times d_k$ state matrix will achieve near-ZeCO communication efficiency for Gated DeltaNet multi-GPU training, with communication volume $O(d_k \cdot C + d_v \cdot C + d_k)$ per device (independent of $P$), enabling Gated DeltaNet to scale to 64+ GPUs with $<15\%$ throughput degradation compared to data parallelism.

## Background

### The sequence parallelism gap for Gated DeltaNet

ZeCO (trick 192) achieves revolutionary communication efficiency for GLA/Mamba-2 by exploiting a key property of diagonal-decay recurrences:

$$
S_{(p-1)L+nC} = (\hat{\gamma}_{[n]}^\top \mathbf{1}) \odot S_{(p-1)L} + S_{[n]}
$$

The correction from the previous device's state $S_{(p-1)L}$ is just an **elementwise multiply** by the cumulative decay $\hat{\gamma}_{[n]}$. This means each device only needs to receive one $d_k \times d_v$ matrix (the predecessor's final state) and apply a cheap elementwise operation.

**Gated DeltaNet breaks this structure.** Its recurrence is:

$$
\mathbf{S}_t = \alpha_t (\mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

The transition $\alpha_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ is a **gated Householder reflection** — not diagonal. After $C$ steps within a chunk, the cumulative transition is:

$$
\mathbf{F}_{[t]}^C = \gamma_{[t]}^C \tilde{\mathbf{P}}_{[t]}^C = \gamma_{[t]}^C (\mathbf{I} - \tilde{\mathbf{W}}_{[t]}^\top \mathbf{K}_{[t]})
$$

where $\tilde{\mathbf{W}}_{[t]} \in \mathbb{R}^{C \times d_k}$ and $\mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k}$ are the WY factors, and $\gamma_{[t]}^C$ is the scalar cumulative gating product.

The correction from the previous device's state is therefore:

$$
S_{(p-1)L+nC} = \gamma_{[n]}^C \cdot S_{(p-1)L} \cdot \underbrace{(\mathbf{I} - \tilde{\mathbf{W}}_{1:n}^\top \mathbf{K}_{1:n})}_{:= \mathbf{P}_{\text{cum}}} + S_{[n]}^{\text{local}}
$$

This requires multiplying $S_{(p-1)L}$ by a **dense $d_k \times d_k$ matrix** $\mathbf{P}_{\text{cum}}$ — not just an elementwise decay. Naively transmitting this matrix costs $O(d_k^2)$ communication, much larger than ZeCO's $O(d_k \times d_v)$.

### The WY factorization opportunity

However, $\mathbf{P}_{\text{cum}} = \mathbf{I} - \tilde{\mathbf{W}}_{\text{cum}}^\top \mathbf{K}_{\text{cum}}$ is always represented in WY form: the product of $C \cdot N$ Householder reflections across all chunks on a device. The cumulative WY factors $\tilde{\mathbf{W}}_{\text{cum}} \in \mathbb{R}^{(C \cdot N) \times d_k}$ and $\mathbf{K}_{\text{cum}} \in \mathbb{R}^{(C \cdot N) \times d_k}$ are too large to transmit. But we can **re-compress** them.

**Key insight**: The product $\mathbf{P}_{\text{cum}}$ of $C \cdot N$ rank-1 updates can be exactly represented as $\mathbf{I} - \mathbf{W}_{\text{dev}}^\top \mathbf{K}_{\text{dev}}$ where $\mathbf{W}_{\text{dev}}, \mathbf{K}_{\text{dev}} \in \mathbb{R}^{r \times d_k}$ with $r = \min(C \cdot N, d_k)$. In practice, $r = d_k$ (the reflections span the full key space after enough steps). But we can use a **low-rank approximation** of $\mathbf{P}_{\text{cum}}$ with rank $r \ll d_k$, transmitting only $O(r \cdot d_k)$ instead of $O(d_k^2)$.

For the practical setting where $d_k = 128$ and $C = 64$: even the exact WY transmission ($r = d_k$) costs $2 \times d_k \times d_k = 32$K elements — the same as ZeCO's $d_k \times d_v$ for GLA with $d_v = d_k$. **The WY factorization naturally limits communication cost.**

### Why this gives real GPU speedup

1. **Would I bet $100 this is faster than LASP-2 for Gated DeltaNet at 32+ GPUs?** Yes — LASP-2 AllGathers $P$ copies of the full $d_v \times d_k$ state matrix. Our approach transmits one WY-factored state per device in a P2P chain, reducing volume by $P\times$ (same advantage as ZeCO over LASP-2 for GLA).

2. **Can I sketch the CUDA kernel in 5 minutes?** Yes — the All-Scan pipeline sends/receives $\tilde{\mathbf{W}}_{\text{dev}} \in \mathbb{R}^{d_k \times d_k}$ and $\gamma_{\text{dev}} \in \mathbb{R}^{d_k}$ via NCCL P2P send/recv. The receiving device applies the correction: $S_{\text{corrected}} = \gamma_{\text{dev}} \odot (S_{\text{recv}} - \tilde{\mathbf{W}}_{\text{dev}}^\top (\mathbf{K}_{\text{dev}} S_{\text{recv}}^{\top})^{\top}) + S_{\text{local}}$. This is two small matmuls ($d_k \times d_k$ @ $d_k \times d_v$) plus elementwise ops — negligible compute.

3. **Does it reduce HBM bandwidth or increase compute utilization?** Yes — by reducing inter-GPU communication volume by $P\times$, the communication-computation overlap gap shrinks. At 64 GPUs, LASP-2 spends 35.7ms on communication (Table in ZeCO paper) while ZeCO spends 7.65ms. Our approach should achieve comparable to ZeCO since the message size is similar.

## Related Work

- **[ZeCO (Chou et al., 2025)](https://arxiv.org/abs/2507.01004)**: Introduced All-Scan for diagonal-decay linear attention (GLA, RetNet, Mamba-2). Achieves $P$-independent communication volume. **Does not handle non-diagonal transitions** (Householder products in DeltaNet). Our approach extends All-Scan to the WY-factored transition.
- **[LASP-2 (Sun et al., 2025)](https://arxiv.org/abs/2502.07563)**: AllGather-based SP for linear attention. Works for any recurrence but communication grows linearly with $P$. Our approach achieves $P$-independent communication for Gated DeltaNet specifically.
- **[Gated DeltaNet (Yang et al., ICLR 2025)](https://arxiv.org/abs/2412.06464)**: Introduced the gated delta rule with chunkwise WY-based training. Single-GPU only — no SP implementation. Our approach is the first multi-GPU SP for Gated DeltaNet.
- **[DeltaNet SP (Yang et al., NeurIPS 2024)](https://arxiv.org/abs/2406.06484)**: Parallelized DeltaNet over sequence length but focused on single-node chunkwise parallelism, not multi-GPU SP across nodes.
- **Proposal 047 (LASP2-TFLA)**: Combines LASP-2 with TFLA tiling for GLA. Uses AllGather, not All-Scan. Does not address Gated DeltaNet.
- **Proposal 049 (DHelix-strand-interleaved)**: Focuses on microbatch overlap for distributed training. Orthogonal to our communication primitive optimization.

**Gap**: No existing work provides $P$-independent communication SP for models with non-diagonal state transitions (DeltaNet, Gated DeltaNet, DeltaProduct).

## Mathematical Formulation

### Gated DeltaNet Inter-Device State Correction

**Per-device local computation** (device $p$, $N$ chunks of size $C$):

Each device computes its local chunk states $\mathbf{S}_{[n]}^{\text{local}}$ and cumulative transition factors using the Gated DeltaNet chunkwise algorithm (trick 203). At the end, device $p$ has:

- $\mathbf{S}_{\text{final}}^{\text{local}} \in \mathbb{R}^{d_v \times d_k}$ — final local state (without inter-device correction)
- $\gamma_{\text{dev}} = \prod_{n=1}^{N} \gamma_{[n]}^C \in \mathbb{R}^{d_k}$ — cumulative gating over all local chunks (elementwise product of per-chunk decays, computed in log-space for stability)
- $\tilde{\mathbf{W}}_{\text{dev}} \in \mathbb{R}^{d_k \times d_k}$, $\mathbf{K}_{\text{dev}} \in \mathbb{R}^{d_k \times d_k}$ — cumulative WY factors representing the product of all Householder reflections across the device

The cumulative WY factors are built incrementally: after processing chunk $n$, the device-level WY is updated by "appending" chunk $n$'s WY factors $(\tilde{\mathbf{W}}_{[n]}, \mathbf{K}_{[n]})$ to the device-level accumulation.

**Device-level WY merging:**

Given two consecutive WY products $\mathbf{P}_1 = \mathbf{I} - \mathbf{W}_1^\top \mathbf{K}_1$ and $\mathbf{P}_2 = \mathbf{I} - \mathbf{W}_2^\top \mathbf{K}_2$, their product is:

$$
\mathbf{P}_2 \mathbf{P}_1 = \mathbf{I} - \begin{pmatrix} \mathbf{W}_2 \\ \mathbf{W}_1 - \mathbf{K}_2^\top \mathbf{W}_2 \cdot \mathbf{W}_1 \end{pmatrix}^\top \begin{pmatrix} \mathbf{K}_2 \\ \mathbf{K}_1 \end{pmatrix}
$$

This grows the WY representation by concatenation. To keep the size bounded at $d_k \times d_k$, we periodically "compress" via QR factorization of $\mathbf{K}_{\text{dev}}$ — since the WY product of $>d_k$ reflections in $\mathbb{R}^{d_k}$ has rank at most $d_k$.

**All-Scan with WY Factors:**

Device $p$ sends to device $p+1$:
- $\mathbf{S}_{\text{final}}^{\text{local}} \in \mathbb{R}^{d_v \times d_k}$ — the local final state
- $\gamma_{\text{dev}} \in \mathbb{R}^{d_k}$ — cumulative gating
- $\tilde{\mathbf{W}}_{\text{dev}}, \mathbf{K}_{\text{dev}} \in \mathbb{R}^{d_k \times d_k}$ — WY factors

Device $p+1$ receives $S_{\text{recv}}, \gamma_{\text{recv}}, \mathbf{W}_{\text{recv}}, \mathbf{K}_{\text{recv}}$ and computes:

$$
S_{pL} = \gamma_{\text{dev}}^{(p+1)} \odot \left( S_{\text{recv}} \cdot (\mathbf{I} - \mathbf{W}_{\text{recv}}^\top \mathbf{K}_{\text{recv}}) \right) + S_{\text{local}}^{(p+1)}
$$

Wait — this is not quite right. The global state at device $p$'s boundary $S_{pL}$ should incorporate the contributions of all previous devices. The correction involves applying device $p+1$'s full transition to the received state:

$$
S_{pL} = \mathbf{F}_{\text{dev}}^{(p+1)} \cdot S_{(p-1)L} + S_{\text{local}}^{(p+1)}
$$

where $\mathbf{F}_{\text{dev}}^{(p+1)} = \text{diag}(\gamma_{\text{dev}}^{(p+1)}) \cdot (\mathbf{I} - {\tilde{\mathbf{W}}_{\text{dev}}^{(p+1)}}^\top \mathbf{K}_{\text{dev}}^{(p+1)})$.

The All-Scan scan operation combines the received global state with the local transition:

$$
S_{\text{send}} = \text{diag}(\gamma_{\text{dev}}) \cdot S_{\text{recv}} \cdot (\mathbf{I} - \tilde{\mathbf{W}}_{\text{dev}}^\top \mathbf{K}_{\text{dev}}) + S_{\text{local}}
$$

This is a matmul $S_{\text{recv}} \cdot (\mathbf{I} - \tilde{\mathbf{W}}_{\text{dev}}^\top \mathbf{K}_{\text{dev}})$ which expands to:

$$
S_{\text{send}} = \text{diag}(\gamma_{\text{dev}}) \cdot (S_{\text{recv}} - S_{\text{recv}} \cdot \tilde{\mathbf{W}}_{\text{dev}}^\top \mathbf{K}_{\text{dev}}) + S_{\text{local}}
$$

The key computation $S_{\text{recv}} \cdot \tilde{\mathbf{W}}_{\text{dev}}^\top$ is a $(d_v \times d_k) \times (d_k \times d_k) = d_v \times d_k$ matmul — a GEMM on tensor cores.

### Communication Volume Analysis

| Component | Size | Bytes (BF16) |
|-----------|------|-------------|
| $S_{\text{local}}$ | $d_v \times d_k$ | $2 d_v d_k$ |
| $\gamma_{\text{dev}}$ | $d_k$ | $2 d_k$ |
| $\tilde{\mathbf{W}}_{\text{dev}}$ | $d_k \times d_k$ | $2 d_k^2$ |
| $\mathbf{K}_{\text{dev}}$ | $d_k \times d_k$ | $2 d_k^2$ |
| **Total per head** | — | $2(d_v d_k + 2 d_k^2 + d_k)$ |

For $d_k = d_v = 128$: Total = $2(128^2 + 2 \times 128^2 + 128) = 2(16384 + 32768 + 128) = 98,560$ bytes $\approx 96$ KB per head.

**Comparison:**

| Method | Comm. per head per device | At $P=64$, $H=32$ |
|--------|--------------------------|-------------------|
| LASP-2 (AllGather) | $2P \cdot d_v \cdot d_k$ | $2 \times 64 \times 128^2 = 2$ MB |
| ZeCO (All-Scan, GLA) | $2 \cdot d_v \cdot d_k$ | $2 \times 128^2 = 32$ KB |
| **Ours (All-Scan, Gated DeltaNet)** | $2(d_v d_k + 2d_k^2 + d_k)$ | **96 KB** |

Our method communicates $3\times$ more than ZeCO for GLA (because of the extra WY factors), but still $P\times$ less than LASP-2. At $P = 64$: ours is $21\times$ less than LASP-2.

### Key Variables

- $P$ — number of GPUs
- $L$ — local sequence length per GPU
- $N = L/C$ — chunks per GPU
- $C$ — chunk size (64)
- $d_k, d_v$ — key/value head dimensions (128)
- $H$ — number of heads
- $\gamma_{\text{dev}} \in \mathbb{R}^{d_k}$ — cumulative gating across device
- $\tilde{\mathbf{W}}_{\text{dev}}, \mathbf{K}_{\text{dev}} \in \mathbb{R}^{d_k \times d_k}$ — device-level WY factors

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Gated DeltaNet 1.3B |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Heads | $H = 16$ |
| Key/value dim | $d_k = d_v = 128$ |
| Chunk size | $C = 64$ |
| GPUs | 8–128 H100 SXM |
| Interconnect | NVLink (intra-node), InfiniBand (inter-node) |

### Baseline

1. **LASP-2 adapted for Gated DeltaNet**: AllGather the full $d_v \times d_k$ state plus $d_k \times d_k$ WY factors from all $P$ devices; compute prefix scan locally. Communication: $O(P \cdot (d_v d_k + d_k^2))$.
2. **Data parallelism (DP)**: Each GPU processes independent sequences. No sequence-level communication. This is the throughput ceiling.
3. **ZeCO for GLA** (if applicable): ZeCO on an equivalent GLA model as a reference for the best possible communication efficiency with diagonal transitions.

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Comm. volume per device | $O(d_v d_k + d_k^2)$ ($P$-indep) | nccl profiling |
| Throughput at 64 GPUs | $> 80\%$ of DP baseline | Tokens/sec per GPU |
| Throughput vs LASP-2 at 64 GPUs | $> 1.5\times$ | Tokens/sec per GPU |
| Numerical accuracy | Bit-exact to single-GPU | Compare output tensors |
| All-Scan latency | $< 2\times$ ZeCO-GLA | Wall-clock ms |

### Estimated Compute

**MVE**: 4 GPU-hours on 8× H100 node
- Implement WY-factored All-Scan as a standalone collective
- Microbenchmark: measure latency vs LASP-2 AllGather at various $P$
- Verify numerical correctness against single-GPU reference

**Phase 1**: 64 GPU-hours on 64× H100
- Full Gated DeltaNet 1.3B training with WY-All-Scan SP
- Compare throughput vs LASP-2 and DP baselines
- Profile communication/computation overlap

**Phase 2**: 256 GPU-hours on 128× H100
- Scale to 128 GPUs, 1M total sequence length
- Full perplexity comparison to verify training quality

## Expected Outcome

**If hypothesis is correct:**
- Communication volume per device is constant ($\sim 96$ KB/head), independent of $P$
- At 64 GPUs: throughput within 15% of DP (vs LASP-2's ~45% gap, extrapolating from ZeCO paper)
- At 128 GPUs: throughput within 20% of DP (LASP-2 degrades to ~30% of DP)
- Gated DeltaNet becomes viable for long-context pretraining (1M+ tokens) across 64+ GPUs
- The WY-factored All-Scan adds ~$3\times$ the latency of ZeCO's scalar All-Scan (due to $3\times$ message size), but this is still hidden behind intra-chunk computation

**If hypothesis is wrong:**
- **Scenario A**: The WY merging across chunks is numerically unstable. Accumulated Householder products across $N \cdot P$ steps could drift. **What we learn**: Gated DeltaNet's non-diagonal transition introduces numerical challenges absent in diagonal models. **Mitigation**: Re-orthogonalize the WY factors periodically (every device boundary) via QR factorization of $\mathbf{K}_{\text{dev}}$.
- **Scenario B**: The $3\times$ larger message size makes All-Scan latency-bound at high $P$. The P2P pipeline doesn't hide the extra latency. **What we learn**: The WY factors dominate communication for large $P$. **Mitigation**: Use low-rank approximation of $\tilde{\mathbf{W}}_{\text{dev}}, \mathbf{K}_{\text{dev}}$ (rank $r < d_k$) to trade communication for approximation error.
- **Scenario C**: The scan operation (applying WY correction to received state) adds too much compute at the pipeline boundary. **What we learn**: The $d_v \times d_k \times d_k$ matmul per pipeline stage is non-negligible. **Mitigation**: Pipeline the WY correction computation itself using the $K$-block splitting from ZeCO.

## Minimum Viable Experiment

### Setup
- **System**: 8× H100 node (single NVLink domain)
- **Test**: Standalone All-Scan collective microbenchmark (no full model)
- **Shapes**: $d_k = d_v = 128$, $H = 16$, varying $P \in \{2, 4, 8\}$
- **Comparison**: LASP-2 AllGather of equivalent data volume
- **Compute**: < 10 minutes wall-clock

### Protocol
1. Generate random state matrices $S \in \mathbb{R}^{d_v \times d_k}$, WY factors $W, K \in \mathbb{R}^{d_k \times d_k}$, and decay $\gamma \in \mathbb{R}^{d_k}$ on each device
2. Run WY-factored All-Scan: P2P send/recv + matmul correction
3. Run LASP-2 AllGather + local prefix scan as baseline
4. Compare latency and verify numerical equivalence

### Success Criteria
- WY-All-Scan latency at $P=8$ is $< 2\times$ ZeCO-GLA All-Scan latency (message is $3\times$ larger, but pipeline amortizes)
- WY-All-Scan latency at $P=8$ is $< 0.5\times$ LASP-2 AllGather latency
- Numerical output matches single-device prefix scan to BF16 precision ($< 10^{-3}$ relative error)
- WY correction matmul overhead $< 0.5$ ms on H100 per pipeline stage

### Failure Criteria
- WY-All-Scan is slower than LASP-2 at $P=8$ → message size is too large for the P2P pipeline to amortize
- Numerical error $> 10^{-2}$ → WY accumulation is unstable
- WY correction matmul $> 2$ ms → compute overhead dominates

### Why This Test Is Sufficient
- The All-Scan collective is the core novelty. If it's fast and correct at $P=8$, scaling to $P=64$ only improves the relative advantage over LASP-2 (whose communication grows with $P$ while ours doesn't)
- The matmul shapes ($d_k = 128$) are identical at any scale — the correction cost per pipeline stage doesn't change with $P$
- Numerical stability can be checked with synthetic data; it doesn't require full model training

## Memory Access Pattern Analysis

**Communication is P2P coalesced**: Each device sends one contiguous buffer of $\sim 96$ KB to its successor. NCCL P2P send/recv over NVLink achieves near-peak bandwidth for buffers $> 64$ KB.

**WY correction is a GEMM**: The scan operation $S_{\text{recv}} \cdot \tilde{\mathbf{W}}^\top$ is a $(d_v \times d_k) \times (d_k \times d_k)$ matmul — maps directly to tensor cores. For $d_v = d_k = 128$: $128 \times 128 \times 128 = 2M$ FLOPs, which takes $\sim 1\mu$s on H100 FP8 WGMMA.

**Pipeline blocks reuse cache**: ZeCO's $K$-block splitting of the state tensor ensures that when device $p+1$ receives block $k$, it's processed immediately while block $k+1$ is in transit — maximizing NVLink utilization.

## Parallelism Analysis

**SM saturation**: The All-Scan is a communication + small GEMM operation running on 1-2 SMs. Meanwhile, 130+ SMs are busy with intra-chunk diagonal attention (the dominant compute). Full overlap.

**No warp divergence**: All operations are standard matmuls and elementwise ops.

**Tensor core mapping**: WY correction GEMM maps to WGMMA. The elementwise decay $\gamma \odot S$ uses FMA pipeline.

**No sequential bottleneck beyond P2P chain**: The All-Scan pipeline has $O(P)$ latency steps, but each step is a single P2P transfer ($\sim 96$ KB at NVLink 900 GB/s = $\sim 0.1\mu$s) plus a small GEMM ($\sim 1\mu$s). Total pipeline latency at $P=64$: $\sim 64 \times 1.1\mu$s $= 70\mu$s — negligible.

## Theoretical Analysis

| Operation | LASP-2 | ZeCO (GLA) | Ours (Gated DeltaNet) |
|-----------|--------|-----------|----------------------|
| Comm. volume/device | $O(P \cdot H \cdot d_v d_k)$ | $O(H \cdot d_v d_k)$ | $O(H \cdot (d_v d_k + 2d_k^2))$ |
| Scan compute/device | $O(P \cdot H \cdot d_v d_k)$ | $O(H \cdot N \cdot d_v d_k)$ | $O(H \cdot N \cdot d_v d_k^2)$ |
| Pipeline latency | $O(\log P)$ | $O(P / K)$ | $O(P / K)$ |
| Scales with $P$? | Yes (comm.) | No (comm.) | **No (comm.)** |

The extra $d_k$ factor in scan compute ($d_v d_k^2$ vs $d_v d_k$) comes from the WY correction matmul. For $d_k = 128$, $N = 128$: this is $N \cdot d_v \cdot d_k^2 = 128 \times 128 \times 128^2 = 256$M FLOPs per head — about 0.5ms on H100 at 500 TFLOPS, or ~2% of total layer compute.

## Risks & Limitations

1. **WY accumulation numerical stability**: Composing $C \cdot N$ Householder reflections across an entire device's local sequence may lead to numerical drift. Each Gated DeltaNet step applies $\alpha_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ — the accumulated product needs the WY representation to be maintained accurately. **Mitigation**: Use FP32 for WY accumulation; re-orthogonalize $\mathbf{K}_{\text{dev}}$ at device boundaries.

2. **Backward pass complexity**: The backward pass through the All-Scan requires reverse-direction P2P (device $p+1 \to p$). The gradient of the WY correction matmul adds extra GEMM operations. **Mitigation**: Cache the WY factors and received states during forward; reuse in backward.

3. **Applicable only to WY-structured transitions**: Our approach requires the state transition to be expressible in WY form. This covers DeltaNet, Gated DeltaNet, and DeltaProduct, but not arbitrary non-diagonal transitions. **Acceptable**: These are the primary architectures of interest.

4. **$3\times$ communication overhead vs ZeCO-GLA**: For architectures where GLA suffices, there's no reason to use this method. It's specifically for when Gated DeltaNet's superior expressivity (state-tracking, non-abelian group simulation) is needed.

5. **NVLink-optimized**: On InfiniBand (inter-node), the 96 KB message may be too small for efficient RDMA. **Mitigation**: Batch across heads before sending — $H \times 96$ KB $= 1.5$ MB for $H = 16$, a more reasonable RDMA payload.

## Follow-up Experiments

1. **TASP multi-ring All-Scan**: Combine with trick 193 (TASP Hamiltonian multi-ring) to utilize multiple NVLink rings simultaneously for the All-Scan, reducing pipeline latency by the number of available Hamiltonian cycles.

2. **Low-rank WY approximation**: Use rank-$r$ approximation of $\tilde{\mathbf{W}}_{\text{dev}}, \mathbf{K}_{\text{dev}}$ with $r < d_k$ to reduce communication further, at the cost of approximate state correction. Test quality degradation vs communication savings.

3. **ZeCO + atomic fusion**: Combine the ZeCO All-Scan inter-device communication with the intra-device atomic state-passing fusion (trick 182), creating a single-launch kernel that handles both intra-device chunk-to-chunk atomics AND inter-device P2P All-Scan in one persistent kernel.

4. **Hybrid Gated DeltaNet + Attention SP**: For hybrid architectures (Gated DeltaNet-H2 with sliding window attention), use our WY-All-Scan for the linear layers and Ring Attention or Striped Attention (trick 198) for the attention layers.

5. **DeltaProduct extension**: DeltaProduct uses $n_h$ Householder reflections per step, making the WY factors grow faster ($n_h \times C$ reflections per chunk). Test whether the WY compression is still communication-efficient for $n_h = 4$.

## Human Review

(To be filled by reviewer)

## References

- Chou, Y., et al. (2025). ZeCO: Zero Communication Overhead Sequence Parallelism for Linear Attention. arXiv:2507.01004.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
- Yang, S., Wang, B., Zhang, Y., et al. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. arXiv:2406.06484.
- Sun, W., et al. (2025). LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. arXiv:2502.07563.
- Astra, R., Dao, T., & Hoque, A. (2026). Accelerating Mamba2 with Kernel Fusion. PyTorch Blog.
- Bischof, C. H. & Van Loan, C. (1985). The WY Representation for Products of Householder Matrices. SIAM J. Sci. Stat. Comput.

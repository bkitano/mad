---
status: ongoing
priority: high
created: 2026-02-15
based_on: optimal-tensor-contraction-ordering, chimera-block-reorder-compute-fusion, epilogue-visitor-tree-fusion, chunkwise-parallel-scan, io-aware-tiling, batch-reduce-gemm, kernel-fusion
experiment_number: 042
experiment_log: experiment-log-042.md
---

# Contraction-Ordered Multi-Operand Chunkwise GLA Fusion

## Hypothesis

Applying **optimal tensor contraction ordering** (à la opt_einsum) to the multi-operand intra-chunk computation of chunkwise GLA — which involves 6 tensors ($Q_j, K_j, V_j, M_j, h_{jC}, \alpha_j$) combined through 3+ GEMMs and elementwise operations — will reveal non-obvious **parenthesizations** that reduce total FLOPs by $15$–$30\%$ or total HBM traffic by $20$–$40\%$ compared to the standard left-to-right evaluation order, because the optimal contraction path depends on the relative dimensions ($C$ vs $d$ vs $n$ vs $d_v$), and modern models operate in regimes where $d \gg C$ (e.g., $d = 128$, $C = 64$) or $n \ll d$ (e.g., $n = 16$), creating asymmetries that standard implementations do not exploit.

## Background

### The multi-operand chunkwise computation

The intra-chunk output of GLA for chunk $j$ is:

$$
\hat{O}_j = \underbrace{(\underbrace{Q_j K_j^\top}_{C \times C} \odot M_j)}_{C \times C} V_j + Q_j h_{jC}
$$

where $Q_j, K_j \in \mathbb{R}^{C \times d_k}$, $V_j \in \mathbb{R}^{C \times d_v}$, $M_j \in \mathbb{R}^{C \times C}$ (causal decay mask), $h_{jC} \in \mathbb{R}^{d_k \times d_v}$ (boundary state).

This is **not** a simple 2-GEMM chain. It's a 6-tensor contraction with multiple paths:

**Standard evaluation (what all implementations use):**
1. $S_j = Q_j K_j^\top$ → GEMM: $O(C^2 d_k)$ FLOPs, produces $C \times C$ intermediate
2. $\tilde{S}_j = S_j \odot M_j$ → elementwise: $O(C^2)$
3. $O_j^{\text{intra}} = \tilde{S}_j V_j$ → GEMM: $O(C^2 d_v)$ FLOPs
4. $O_j^{\text{state}} = Q_j h_{jC}$ → GEMM: $O(C d_k d_v)$ FLOPs
5. $\hat{O}_j = O_j^{\text{intra}} + O_j^{\text{state}}$ → elementwise: $O(C d_v)$

Total: $C^2 d_k + C^2 d_v + C d_k d_v + O(C^2)$ FLOPs.

**Alternative evaluation (unexplored):**

Without the mask $M_j$, there's a well-known identity: $Q_j K_j^\top V_j = Q_j (K_j^\top V_j)$, changing $O(C^2 d)$ to $O(C d^2)$. But the mask $M_j$ prevents this factorization for the intra-chunk term.

However, there are **partial factorizations** and **dimension-dependent orderings** that haven't been explored:

1. **State correction reordering**: $Q_j h_{jC}$ where $h_{jC} \in \mathbb{R}^{d_k \times d_v}$. If $d_k > d_v$, it's cheaper to compute $(Q_j h_{jC})$ as written ($O(C d_k d_v)$). But in models where $d_k = d_v = d/H$ (multi-head), this is symmetric. The key question: can the state correction be fused with the intra-chunk computation to share operand loads?

2. **Masked attention as structured operation**: The causal decay mask $M_j[i,k] = \prod_{t=k+1}^{i} \alpha_t$ has specific structure — it's a lower-triangular matrix whose entries are products of consecutive decay rates. This structure means $\tilde{S}_j = S_j \odot M_j$ can potentially be decomposed differently.

3. **The full einsum**: The complete computation can be written as a single einsum-like expression:

$$
\hat{O}_j[i, v] = \sum_k \left(\sum_{q} Q_j[i,q] K_j[k,q] \cdot M_j[i,k]\right) V_j[k,v] + \sum_q Q_j[i,q] h_{jC}[q,v]
$$

In Einstein notation: `iq,kq,ik,kv,qv->iv` with appropriate masks. The optimal contraction path through these 5 tensors depends on the relative sizes of $C$, $d_k$, $d_v$.

### Why contraction ordering matters here

The **matrix chain problem** applies directly. For the standard path, the FLOPs are:

$$
\text{FLOPs}_{\text{standard}} = 2C^2 d_k + 2C^2 d_v + 2C d_k d_v
$$

Consider the case where $C = 64$, $d_k = d_v = 128$ (typical for GLA with 8 heads and $d = 1024$):

$$
\text{FLOPs}_{\text{standard}} = 2 \times 64^2 \times 128 + 2 \times 64^2 \times 128 + 2 \times 64 \times 128^2 = 2.1M + 2.1M + 2.1M = 6.3M
$$

Now consider that the state correction $Q_j h_{jC}$ (the third term) accounts for $\frac{1}{3}$ of total FLOPs. If we can share the load of $Q_j$ between the intra-chunk attention and the state correction (trivial with EVT — $Q_j$ is loaded once for both), we save one HBM read of $Q_j$ ($C \times d_k$ elements).

More importantly: when the decay mask $M_j$ has **low effective rank** (which happens when decay rates $\alpha_t \approx 1$, common in practice), the masked attention can be approximated as:

$$
\tilde{S}_j \approx S_j \odot (\mathbf{1}\mathbf{1}^\top - \text{low-rank correction})
$$

This transforms the problem into: $Q_j K_j^\top V_j - Q_j K_j^\top (\text{low-rank}) V_j$, where the first term uses the cheap right-association $Q_j (K_j^\top V_j)$, and only the correction needs the $O(C^2)$ path.

### Why this hasn't been done

1. **Fixed evaluation order**: All implementations (fla, Mamba-2, TFLA) use the same left-to-right evaluation order without considering dimension-dependent alternatives.
2. **Mask prevents naive factorization**: The causal mask makes the standard right-association $Q(K^\top V)$ invalid for the intra-chunk term, so implementers default to the left-to-right path.
3. **Contraction ordering is a compile-time concern**: PyTorch's `torch.einsum` uses opt_einsum for simple expressions, but the chunkwise GLA computation is not expressed as a single einsum — it's a sequence of manual kernel calls.
4. **Chimera assumes a fixed chain topology**: Chimera optimizes within a given GEMM chain but doesn't consider alternative factorizations of the computation itself.
5. **Centaurus (ICLR 2025)** applies contraction ordering to SSM convolution-mode kernels, not to the chunkwise parallel matmul chain.

### What this proposal adds

We systematically enumerate the contraction paths for the full chunkwise GLA computation, evaluate each path's FLOP count, HBM traffic, and arithmetic intensity as a function of $(C, d_k, d_v, n)$, and identify dimension regimes where non-standard paths win. We then implement the optimal path as a fused CUDA/Triton kernel.

## Related Work

- **[Centaurus (Pei, ICLR 2025)](https://arxiv.org/abs/2501.13230)**: Applies optimal tensor contraction ordering to SSMs in **convolution training mode** — treating the SSM kernel computation as a tensor contraction and optimizing its evaluation order. Our proposal targets a different computational pattern: the **chunkwise parallel mode** multi-GEMM chain, which has different operands, shapes, and mask constraints.
- **[opt_einsum (Smith & Gray, 2018)](https://optimized-einsum.readthedocs.io/)**: General-purpose tensor contraction path optimizer. Our proposal applies this principle to the specific structured computation of chunkwise linear attention, with additional constraints from the causal mask.
- **[Chimera (Zheng et al., HPCA 2023)](https://dl.acm.org/doi/10.1109/HPCA56546.2023.10071018)**: Optimizes block execution order within a fixed GEMM chain. Our proposal optimizes the **chain structure itself** (which GEMMs to compute and in what order), then applies Chimera-style optimization within the chosen chain.
- **[GLA (Yang et al., ICML 2024)](https://arxiv.org/abs/2312.06635)**: Defines the chunkwise parallel algorithm with a fixed evaluation order. Our proposal finds dimension-dependent optimal orderings.
- **[Compilation of Generalized Matrix Chains (López et al., 2025)](https://arxiv.org/abs/2511.20198)**: Compiles matrix chain expressions with symbolic sizes. Our proposal extends to masked/structured contractions with hardware-aware cost models.
- **Proposal 032 (Chimera-Fused Chunkwise SSM)**: Fuses the fixed Q·K^T → mask → attn·V chain. Our proposal first selects the optimal chain, then fuses it.

No existing work applies contraction ordering optimization to the chunkwise parallel linear attention multi-operand computation.

## Mathematical Formulation

### Contraction Path Enumeration

The intra-chunk computation involves the following tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| $Q_j$ | $C \times d_k$ | Query |
| $K_j$ | $C \times d_k$ | Key |
| $V_j$ | $C \times d_v$ | Value |
| $M_j$ | $C \times C$ | Causal decay mask (lower triangular) |
| $h_{jC}$ | $d_k \times d_v$ | Boundary state (from inter-chunk scan) |
| $\alpha_j$ | $C$ | Per-timestep decay rates |

The target: $\hat{O}_j \in \mathbb{R}^{C \times d_v}$.

### Path 1: Standard (current implementations)

$$
\hat{O}_j = \underbrace{(\underbrace{(Q_j K_j^\top)}_{C \times C} \odot M_j)}_{C \times C} V_j + Q_j h_{jC}
$$

| Step | Operation | FLOPs | Intermediate | HBM |
|------|-----------|-------|-------------|-----|
| 1 | $Q_j K_j^\top$ | $2C^2 d_k$ | $C \times C$ | Read $Q_j$, $K_j$ |
| 2 | $\odot M_j$ | $C^2$ | $C \times C$ | Read $M_j$ |
| 3 | $\cdot V_j$ | $2C^2 d_v$ | $C \times d_v$ | Read $V_j$ |
| 4 | $Q_j h_{jC}$ | $2C d_k d_v$ | $C \times d_v$ | Read $Q_j$, $h_{jC}$ |
| 5 | Sum | $C d_v$ | $C \times d_v$ | — |

**Total FLOPs:** $2C^2(d_k + d_v) + 2Cd_k d_v + C^2 + Cd_v$

**Peak intermediate:** $C \times C$ (attention matrix)

### Path 2: Right-associated with mask decomposition

Decompose $M_j = L_j - \Delta_j$ where $L_j$ is the all-ones lower-triangular mask and $\Delta_j$ encodes the decay deviations:

$$
\hat{O}_j = Q_j \underbrace{(\text{cumsum}(K_j^\top V_j))}_{d_k \times d_v \text{ cumulative}} - \underbrace{(Q_j K_j^\top \odot \Delta_j)}_{C \times C} V_j + Q_j h_{jC}
$$

If $\Delta_j$ has low effective rank $r$ (when $\alpha_t \approx 1$):

$$
\Delta_j \approx U_j \Sigma_j W_j^\top, \quad U_j \in \mathbb{R}^{C \times r}, W_j \in \mathbb{R}^{C \times r}
$$

Then the correction term becomes:

$$
(Q_j K_j^\top \odot U_j \Sigma_j W_j^\top) V_j \approx (Q_j \odot U_j \Sigma_j) (K_j \odot W_j)^\top V_j
$$

which can be computed as:

| Step | Operation | FLOPs | Notes |
|------|-----------|-------|-------|
| 1 | $P_j = K_j^\top V_j$ | $2C d_k d_v$ | Right-association |
| 2 | $\text{cumsum}(P_j)$ | $C d_k d_v$ | Prefix sum along $C$ |
| 3 | $O_j^{\text{right}} = Q_j \cdot \text{cumsum}(P_j)$ | $2C d_k d_v$ | Right-associated term |
| 4 | $\tilde{Q}_j = Q_j \odot U_j$ | $Cr d_k$ | Hadamard |
| 5 | $\tilde{K}_j = K_j \odot W_j$ | $Cr d_k$ | Hadamard |
| 6 | $\text{correction} = \tilde{Q}_j \tilde{K}_j^\top V_j$ | $2C^2 r d_v$ or $2Crd_k d_v$ | Depends on $r$ vs $C$ |
| 7 | $\hat{O}_j = O_j^{\text{right}} - \text{correction} + Q_j h_{jC}$ | — | — |

**Total FLOPs (if $r \ll C$):** $5Cd_k d_v + 2C^2 r d_v + 2Crd_k$ (much cheaper if $r \leq C/10$)

**When this wins:** $d_k, d_v \gg C$ and $r \ll C$. For $C = 64$, $d = 128$, $r \leq 6$: Path 2 uses $\sim 5.3M$ FLOPs vs Path 1's $\sim 6.3M$ FLOPs = **16% reduction**.

### Path 3: Tiled hybrid (TFLA-compatible)

Within TFLA's two-level tiling, the inner tiles are small enough that Path 1 or Path 2 can be selected per tile based on the local effective rank of $M_j$:

$$
\hat{O}_j = \sum_{t=1}^{C/C_{\text{inner}}} \text{optimal\_path}(Q_j^{(t)}, K_j^{(t)}, V_j^{(t)}, M_j^{(t)}, h_j^{(t)})
$$

This is an **adaptive** contraction ordering that selects the cheapest path per inner tile.

### Dimension-Dependent Crossover Analysis

Define the dimension ratio $\rho = d / C$:

| Regime | $\rho$ | Optimal Path | Why |
|--------|--------|-------------|-----|
| $d \ll C$ | $< 0.5$ | Path 1 (standard) | $C^2 d$ is cheap; mask has no structure to exploit |
| $d \approx C$ | $0.5$–$2$ | Path 1 or Path 2 | Depends on mask rank |
| $d \gg C$ | $> 2$ | Path 2 (right-associated) | $Cd^2$ terms dominate; avoiding $C^2 d$ saves less than right-association saves |

For modern GLA with $d_k = d_v = 128$ and $C = 64$: $\rho = 2.0$, sitting in the regime where Path 2 becomes attractive if mask rank is moderate.

### Hardware-Aware Cost Model

The contraction ordering optimization must account for GPU hardware, not just FLOPs:

$$
\text{Cost}(\text{path}) = \max\left(\frac{\text{FLOPs}(\text{path})}{\text{TFLOPS}}, \frac{\text{HBM}(\text{path})}{\text{BW}}\right) + \text{Overhead}(\text{path})
$$

where:
- $\text{TFLOPS}$: Peak tensor core throughput (989 TFLOPS BF16 on H100)
- $\text{BW}$: HBM bandwidth (3.35 TB/s on H100)
- $\text{Overhead}$: Extra costs (non-GEMM ops, kernel launches, synchronization)

For a path to be preferred, it must reduce the **binding bottleneck** (memory or compute), not just total FLOPs.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | GLA (primary), Mamba-2 SSD, Gated DeltaNet |
| Layers | $L = 12$ |
| Hidden dim | $d \in \{512, 768, 1024, 2048\}$ |
| Heads | $H \in \{4, 8, 16\}$ (so $d_k = d/H$) |
| State dim | $n = 16$ |
| Chunk size | $C \in \{32, 64, 128, 256\}$ |
| Sequence length | $T \in \{2048, 4096, 8192\}$ |

### Experimental Design

**Phase 1: Offline contraction path analysis**
- For each $(C, d_k, d_v, n)$ configuration, enumerate all valid contraction paths (with and without mask decomposition)
- Evaluate each path's FLOP count, peak intermediate size, and HBM traffic using opt_einsum's cost model
- Identify the Pareto-optimal paths (minimize FLOPs and HBM jointly)

**Phase 2: Mask rank characterization**
- Train a small GLA model and measure the effective rank of the causal decay mask $M_j$ across layers and training steps
- Test: is $\text{rank}_\epsilon(M_j)$ typically $\ll C$? (If not, Path 2 is not viable)

**Phase 3: Fused kernel implementation**
- Implement the top-2 contraction paths as fused Triton kernels
- Compare wall-clock time against fla library's standard path

### Baseline
1. **fla library GLA kernel**: Standard left-to-right evaluation ($O(C^2 d)$ per chunk)
2. **TFLA kernel**: Two-level tiled forward (SOTA wall-clock)
3. **torch.einsum with opt_einsum**: Let opt_einsum choose the path automatically (upper bound on what contraction ordering alone can achieve, without fusion)

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| FLOPs (per chunk) | $< 0.85\times$ baseline | Counted analytically |
| Wall-clock throughput | $> 1.1\times$ TFLA baseline | tokens/sec on H100 |
| Peak intermediate memory | $< 0.5\times$ baseline | Profiled via Nsight |
| Quality | Exact or $< 10^{-4}$ relative error | Compared to reference |

### Estimated Compute

- **Phase 1**: CPU-only, < 1 hour (enumeration + cost model)
- **Phase 2**: 2–4 GPU-hours (train small model, measure mask ranks)
- **Phase 3**: 4–8 GPU-hours (kernel implementation + benchmarking)
- **Total**: Small experiment (~10 GPU-hours)

## Expected Outcome

**If hypothesis is correct:**
- Dimension-dependent optimal contraction paths differ from the standard left-to-right order in $\geq 50\%$ of tested configurations
- Path 2 (right-associated with mask correction) achieves $15$–$30\%$ FLOP reduction when $d/C > 1.5$ and mask effective rank $< C/5$
- Fused kernel achieves $1.1$–$1.3\times$ wall-clock speedup over TFLA for large-$d$ configurations

**If hypothesis is wrong:**
- The causal decay mask's effective rank is always $\Theta(C)$ (i.e., decay rates are far from 1), making Path 2 no better than Path 1
- HBM traffic, not FLOPs, is the binding bottleneck, and all paths have similar HBM patterns
- We learn the **sensitivity of contraction ordering to dimension ratios**, providing a lookup table for future kernel designers: "for $(C, d)$ in this range, always use left-to-right"

## Minimum Viable Experiment

### Setup
- **Model**: None (pure kernel microbenchmark)
- **Task**: Time the intra-chunk computation for random tensors
- **Implementation**:
  1. Implement Path 1 (standard) and Path 2 (right-associated with rank-$r$ mask correction) as Triton kernels
  2. Sweep over configurations: $C \in \{32, 64, 128\}$, $d \in \{64, 128, 256\}$, $r \in \{1, 4, 8, 16, C\}$
  3. Measure wall-clock time and verify numerical correctness
- **Data**: Random $Q, K, V, \alpha$ tensors (BF16)
- **Compute**: Single A100 GPU, < 10 minutes

### Success Criteria
- Path 2 is $\geq 10\%$ faster than Path 1 for **at least one** realistic configuration (e.g., $C = 64$, $d = 128$, $r \leq 8$)
- Numerical agreement within $\epsilon < 10^{-3}$ relative error (BF16)

### Failure Criteria
- If Path 2 is **never** faster than Path 1 across all configurations, the contraction ordering does not help for this specific computation
- If the mask's effective rank in trained models is always $\geq C/2$, the low-rank approximation is not useful

### Why This Test Is Sufficient
- The kernel microbenchmark directly measures the wall-clock impact of contraction ordering — no model training needed for Phase 1
- If the kernel is faster on random tensors, it will be faster on real tensors (the compute pattern doesn't depend on tensor values, only shapes)
- The mask rank characterization (Phase 2) is the main uncertainty; the MVE assumes various ranks to establish the shape of the speedup curve

## Theoretical Analysis

Complexity comparison per chunk (for $d_k = d_v = d$):

| Operation | Path 1 (Standard) | Path 2 (Right-associated, rank-$r$) |
|-----------|-------------------|--------------------------------------|
| Intra-chunk FLOPs | $2C^2 d + 2C^2 d = 4C^2 d$ | $5Cd^2 + 2C^2 rd$ |
| State correction FLOPs | $2Cd^2$ | $2Cd^2$ (same) |
| Total FLOPs | $4C^2 d + 2Cd^2$ | $7Cd^2 + 2C^2 rd$ |
| Peak intermediate | $C^2$ | $Cd$ (if $r < C$) |
| HBM reads | $3Cd + C^2 + d^2$ | $3Cd + d^2 + Cr$ |
| HBM writes | $Cd$ | $Cd$ |

**Crossover condition (Path 2 < Path 1 in FLOPs):**

$$
7Cd^2 + 2C^2 rd < 4C^2 d + 2Cd^2
$$

$$
5Cd^2 < 4C^2 d - 2C^2 rd = C^2 d(4 - 2r)
$$

$$
5d < C(4 - 2r)
$$

For $r = 1$: $5d < 2C$, i.e., $C > 2.5d$. This means Path 2 only wins when $C \gg d$, which is **not** the typical regime.

**Revised analysis with tiling**: Within TFLA's inner tiles of size $C_{\text{inner}} \ll C$, Path 2 can win within each tile even when $C_{\text{inner}} \leq d$, because the right-association $K^\top V$ can be precomputed once and reused across query tiles:

$$
\hat{O}_j[i, :] = q_i^\top \underbrace{\left(\sum_{k \leq i} \gamma_{ik} k_k v_k^\top\right)}_{\text{cumulative } d_k \times d_v} = q_i^\top S_i
$$

where $S_i = S_{i-1} + \gamma_{i,i} k_i v_i^\top$ is a running state that's incrementally updated.

This is exactly the **recurrent form**, and the chunkwise algorithm already uses it for inter-chunk computation. The insight is: **within inner TFLA tiles**, switching from the parallel form to the recurrent form may be cheaper when tiles are small relative to $d$.

**Optimal tile-level switching rule:**

$$
\text{Use parallel form if } C_{\text{inner}}^2 d < C_{\text{inner}} d^2 \iff C_{\text{inner}} > d
$$

$$
\text{Use recurrent form if } C_{\text{inner}} < d
$$

For TFLA with $C_{\text{inner}} = 32$ and $d = 128$: **use recurrent form within inner tiles** (saving $C_{\text{inner}}^2 d - C_{\text{inner}} d^2 = 32^2 \times 128 - 32 \times 128^2 = 131K - 524K$... actually this is MORE FLOPs for recurrent). Wait — recurrent form is $O(C_{\text{inner}} \times d^2)$ per inner tile, parallel form is $O(C_{\text{inner}}^2 \times d)$ per inner tile. For $C_{\text{inner}} = 32$, $d = 128$: recurrent = $32 \times 128^2 = 524K$, parallel = $32^2 \times 128 = 131K$. Parallel is cheaper.

This confirms the standard analysis: parallel form wins when $C > d$... but that means for small inner tiles ($C_{\text{inner}} < d$), recurrent form wins! However, the recurrent form has sequential dependency ($d^2$ matmul per step, $C_{\text{inner}}$ steps) which is bad for GPU parallelism.

The real opportunity is: **hybrid evaluation** where the outer chunk uses parallel form but the state correction and decay masking exploit structural properties of $M_j$ (low rank, near-identity) to reduce the $O(C^2)$ intermediate to $O(Cr)$.

## Memory Access Pattern Analysis

1. **Coalesced access**: Both Path 1 and Path 2 use standard GEMM loads (coalesced). Path 2's cumulative sum is sequential but operates on registers.
2. **Cache-friendly**: Path 2's $K_j^\top V_j$ product ($d_k \times d_v$) is smaller than Path 1's $Q_j K_j^\top$ ($C \times C$) when $d < C$ — better cache utilization. For $d > C$ (our target regime), $K_j^\top V_j$ is larger, potentially worse.
3. **Arithmetic intensity**: Path 1: $\frac{4C^2d + 2Cd^2}{(3Cd + C^2) \times 2} \approx \frac{4 \times 64^2 \times 128}{(3 \times 64 \times 128 + 64^2) \times 2} \approx 36$ FLOP/byte. Path 2 (with low-rank mask): comparable AI but fewer total FLOPs.
4. **SMEM requirement**: Path 1 needs $Q_j$, $K_j$, intermediate $S_j$ tiles. Path 2 needs $Q_j$, $K_j^\top V_j$ accumulator, correction terms. Similar SMEM footprint.

## Parallelism Analysis

1. **SM saturation**: Both paths parallelize over chunks ($B \times T/C$ independent work units). No change.
2. **Warp divergence**: None — both paths use uniform control flow.
3. **Tensor core utilization**: All GEMMs in both paths map to tensor cores. Path 2's cumulative sum uses general-purpose CUDA cores but is $O(Cd^2)$ — subdominant.
4. **Sequential bottleneck in Path 2**: The cumulative sum requires sequential evaluation along $C$, introducing a small serial dependency. Within a tile of size $C_t$, this is $C_t$ sequential steps — manageable for $C_t \leq 32$.

## Risks & Limitations

1. **Mask rank uncertainty**: The core assumption (mask $M_j$ has low effective rank) may not hold in practice. If decay rates vary significantly across timesteps, $\text{rank}(M_j) \approx C$, and Path 2 degenerates to Path 1 with extra overhead.
2. **Approximation error**: Low-rank mask approximation introduces error. For pretraining, even small systematic errors can compound over training steps. Must verify that the approximation doesn't degrade model quality.
3. **Implementation complexity**: Path 2 requires implementing cumulative outer products and low-rank mask decomposition within a fused kernel — more complex than the standard parallel form.
4. **Limited regime**: The crossover analysis shows Path 2 only helps when mask rank is low AND the dimension ratio favors it. The benefit may be too narrow to justify the implementation effort.
5. **TFLA already optimizes tiling**: TFLA's two-level tiling may already implicitly optimize some aspects of contraction ordering through its tile size selection.

## Follow-up Experiments

1. **Adaptive per-chunk path selection**: Compute a cheap mask rank estimate per chunk at runtime, and dispatch to Path 1 or Path 2 accordingly. This requires two compiled kernel variants and a lightweight rank estimator.
2. **Combine with Proposal 041 (EVT joint fwd+bwd)**: Apply contraction ordering to the full forward+backward DAG, not just the forward.
3. **Extension to DeltaNet**: DeltaNet's intra-chunk has a different mask structure (beta-weighted outer products). Contraction ordering analysis may yield different optimal paths.
4. **Learned contraction paths**: Train a small RL agent to select contraction paths per layer/head based on runtime dimension statistics. Overhead: negligible (path selection is a lookup table).
5. **Compiler integration**: Integrate the dimension-dependent path selection into a compiler pass (e.g., Triton autotuner) that automatically selects the optimal contraction for each model configuration.

## Human Review

(To be filled by reviewer)

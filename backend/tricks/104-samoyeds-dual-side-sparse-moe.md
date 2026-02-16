# 104: Samoyeds: Dual-Side Structured Sparse MoE Kernel

**Category**: kernel
**Gain type**: efficiency
**Source**: Wu, Gu, Shi, Yao & Guan "Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores" (EuroSys 2025)
**Paper**: [papers/samoyeds-moe-sparse-tensor-cores.pdf]
**Documented**: 2025-06-15

## Description

Samoyeds is the first system to exploit **both** weight sparsity and activation sparsity simultaneously in Mixture-of-Experts (MoE) layers using Sparse Tensor Cores (SpTCs). The key insight is that MoE layers have two independent sources of sparsity that can be co-exploited:

1. **Weight sparsity** (static): Model parameters can be pruned to 2:4 element-wise structured sparsity, directly accelerable by SpTCs.
2. **Activation sparsity** (dynamic): The MoE routing mechanism assigns each token to only a subset of experts, creating vector-wise column sparsity in activation matrices — entire columns are zero because the corresponding tokens were not routed to that expert.

Previous approaches (cuSPARSELt, VENOM, vLLM-DS) only exploit one side of this sparsity. Samoyeds introduces a **dual-side sparse data format** that encodes both patterns simultaneously and a corresponding **sparse-sparse matrix multiplication kernel** on SpTCs. The format integrates 2:4 element-wise sparsity for weights with vector-wise column sparsity for activations, enabling the `mma.sp` PTX instruction to skip both zero weights and zero activation columns.

The system also addresses critical GPU kernel engineering challenges: a 3-step hierarchical tiling strategy across global memory → shared memory → registers, a data-stationary optimization for output matrices with intermediate register shuffling, a custom metadata packing scheme for 2-bit SpTC metadata, and an optimized output layout that eliminates redundant memory transfers for sparse outputs.

## Mathematical Form

**MoE Layer Computation:**

For an MoE layer with $E$ experts, input tokens $\mathbf{x} \in \mathbb{R}^{T \times d}$, and a router that assigns each token to $k$ of $E$ experts:

$$
\text{MoE}(\mathbf{x}) = \sum_{e=1}^{E} g_e(\mathbf{x}) \cdot \text{Expert}_e(\mathbf{x}_e)
$$

where $g_e$ are gating scores and $\mathbf{x}_e$ is the subset of tokens routed to expert $e$.

**Per-Expert Linear Layer (Dense):**

Each expert $e$ computes $\mathbf{y}_e = \mathbf{x}_e W_e$ where $W_e \in \mathbb{R}^{d \times d'}$. On SpTCs, this is restructured as:

$$
\mathbf{y}_e = (W_e^T \mathbf{x}_e^T)^T
$$

**Dual-Side Sparsity Pattern:**

The weight matrix $W_e$ has **2:4 element-wise** + **vector-wise** structured sparsity:
- **Element-wise (2:4)**: In every group of 4 elements along a row, 2 are zero. This aligns with the SpTC's `mma.sp` instruction.
- **Vector-wise**: Within blocks of size $M \times V$, only $N$ out of $V$ "Sub-Rows" (vectors of SpTC units) are retained. This provides an additional layer of coarser sparsity beyond 2:4.

The activation matrix $\mathbf{x}_e$ has **column-wise** sparsity via a **selection array** (SEL):
- SEL records which columns (tokens) are routed to expert $e$
- Only $\text{len}_d \leq T$ columns have nonzero entries; the rest are skipped entirely

**Combined Sparsity Ratio:**

$$
\rho_{\text{total}} = \rho_{\text{weight}} \times \rho_{\text{activation}} = \left(\frac{N}{V} \times \frac{1}{2}\right) \times \frac{\text{len}_d}{T}
$$

where $\frac{1}{2}$ comes from 2:4 sparsity, $\frac{N}{V}$ from vector-wise pruning, and $\frac{\text{len}_d}{T}$ from routing sparsity.

**Dual-Side Sparse Format Encoding:**

Weight matrix $W \in \mathbb{R}^{m \times k}$ is encoded as three components:
- **Data matrix**: $\frac{m}{M} \times \frac{k}{2}$ — compressed nonzero values (2:4 compressed)
- **Indices matrix**: $\frac{m}{M} \times \frac{k}{2}$ — relative positions of retained Sub-Rows within blocks
- **Metadata matrix**: $\frac{m}{M} \times \frac{k}{2}$ — 2-bit per element, encoding positions of nonzeros within each group of 4 (SpTC hardware format)

Activation matrix $\mathbf{x}$ is encoded with:
- **Sparse column data**: only the $\text{len}_d$ nonzero columns
- **SEL array**: mapping from compact indices to original column positions

**Key Definitions:**

- $M$ — block height for vector-wise sparsity
- $V$ — number of Sub-Rows per block (vector-wise granularity)
- $N$ — number of retained Sub-Rows per block ($N < V$)
- SEL $\in \mathbb{Z}^{\text{len}_d}$ — selection array mapping routed token indices
- `mma.sp` — PTX instruction for sparse matrix multiply-accumulate on SpTCs

## Complexity

| Operation | Dense (cuBLAS) | Sparse-Dense (cuSPARSELt) | Samoyeds (Sparse-Sparse) |
|-----------|---------------|--------------------------|--------------------------|
| Weight FLOPS | $2mkn$ | $mkn$ (2:4 → 2× speedup) | $\frac{N}{V} \cdot mkn$ (vector+2:4) |
| Activation reads | $kn$ elements | $kn$ elements | $k \cdot \text{len}_d$ elements |
| Weight memory | $mk$ | $\frac{mk}{2}$ + metadata | $\frac{m}{M} \cdot \frac{k}{2}$ + indices + metadata |
| Output memory | $mn$ (dense) | $mn$ (dense) | $m \cdot \text{len}_d$ (sparse layout) |

**Kernel-level speedup:** Up to 1.99× over VENOM (best prior structured sparse kernel), up to 18.76× over Sputnik (unstructured), up to 5.44× over cuBLAS (dense).

**MoE layer speedup:** 1.42× average over Transformers framework, up to 2.36×. Up to 1.58× over vLLM-DS.

**End-to-end model speedup:** 1.42× average over Transformers, 1.30× over vLLM-DS across MiniCPM-MoE, OpenMoE-34B, Mixtral-8×7B, Mixtral-8×22B, Qwen2-MoE, DeepSeek-MoE.

**Memory:** Maximum batch size increased by 4.41× on average vs. Transformers framework (e.g., OpenMoE-34B: 3→56 batch size, 18.67× increase).

## Applicability

- **MoE transformer inference**: All MoE LLMs with routing-induced activation sparsity (Mixtral, DeepSeek-MoE, Qwen2-MoE, MiniCPM-MoE, OpenMoE)
- **Any model with dual sparsity**: Systems where both weights and activations exhibit structured sparsity patterns
- **Gate/up/down projections**: The three linear layers in each MoE expert (`gate_proj`, `up_proj`, `down_proj`) are primary targets
- **Hardware compatibility**: NVIDIA Ampere (A100), Ada Lovelace (4070S, 4090), Hopper (H100) GPUs with SpTCs. Also AMD MI300 (CDNA3) with sparse ALU
- **Composable with pruning methods**: Weight sparsity is applied offline using WoodFisher or SparseGPT; Samoyeds is purely an inference-time acceleration system

## Limitations

- **Inference only**: The dual-side sparse kernel is designed for inference; training still uses dense or single-side sparse approaches
- **Tiling sensitivity**: Performance depends on tiling size matching hardware (L2 cache, SM count); suboptimal on hardware different from the tuning target (4070S)
- **Padding overhead**: When token count per expert doesn't align with tile size, padding is needed — models with many experts (small expert size) suffer more
- **Small matrix sizes**: When $m$ or $n \leq 256$, Samoyeds slightly underperforms VENOM due to limited parallelism and initialization overhead
- **Accuracy depends on pruning method**: The sparse format is orthogonal to pruning; accuracy depends on the pruning algorithm used (WoodFisher, SparseGPT, etc.)
- **Vector-wise sparsity reduces expressivity**: The additional coarse-grained vector-wise sparsity (beyond 2:4) may degrade model quality for some configurations

## Implementation Notes

```python
# Pseudocode for Samoyeds dual-side sparse data format construction
# and kernel execution scheme

# === 1. Weight encoding (offline, during model pruning) ===
def encode_weight_samoyeds(W, M, V, N):
    """Encode weight matrix with 2:4 element-wise + vector-wise sparsity.

    Args:
        W: dense weight matrix [m, k]
        M: block height
        V: Sub-Rows per block
        N: retained Sub-Rows per block (N < V)
    Returns:
        data: compressed nonzero values [m/M, k/2]
        indices: Sub-Row positions [m/M, k/2]
        metadata: 2-bit SpTC metadata [m/M, k/2]
    """
    # Step 1: Apply vector-wise pruning — keep N of V Sub-Rows per block
    # Step 2: Apply 2:4 element-wise pruning within retained Sub-Rows
    # Step 3: Compress into SpTC-compatible format:
    #   - data: only nonzero values (after 2:4 pruning)
    #   - indices: which Sub-Rows were kept (for decoding)
    #   - metadata: 2-bit per element encoding position in group of 4
    pass

# === 2. Activation encoding (online, per-batch) ===
def encode_activation_samoyeds(x, routing_indices):
    """Encode activation with column-wise routing sparsity.

    Args:
        x: input activations [seq_len, hidden_dim]
        routing_indices: which tokens routed to this expert
    Returns:
        sparse_x: compacted activations [len_d, hidden_dim]
        sel: selection array mapping back to original positions
    """
    sel = routing_indices  # which columns are active
    sparse_x = x[sel]     # gather only routed tokens
    return sparse_x, sel

# === 3. Kernel execution (Algorithm 1 from paper) ===
# The Samoyeds kernel performs C = A × B where:
#   A = encoded sparse weight (data + indices + metadata)
#   B = sparse activation (compacted columns + SEL)
#
# Key optimizations in the kernel:
# 1. Three-step tiling: Global→Shared→Register hierarchy
#    - Thread block: m_b × n_b tile of output C
#    - Warp: m_w × n_w sub-tile
#    - SpTC: m_i × n_i (e.g., 16×8 for mma.sp.m16n8k32)
#
# 2. Data stationary for output C:
#    - C stays in registers throughout computation
#    - Intermediate register C_IR used for shuffling when
#      Sub-Row index changes (every V/K_b iterations)
#
# 3. Pipeline: fetch (cp.async) overlapped with compute (mma.sp)
#
# 4. Metadata packing: 2-bit metadata packed into 32-bit registers
#    with custom mapping: [row_idx%8×2 + col_idx//8, col_idx%8 + row_idx//8×8]
#    ensuring 32-bit aligned loads
```

## References

- Wu, C., Gu, Q., Shi, H., Yao, J. & Guan, H. "Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores" (EuroSys 2025). arXiv:2503.10725
- Castro, R.L., et al. "VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores" (SC 2023). Baseline structured sparse kernel.
- Gale, T., Zaharia, M., Young, C. & Elsen, E. "Sputnik: Sparse GPU Kernels for Deep Learning" (SC 2020). Unstructured sparse baseline.
- Gale, T., Narayanan, D., Young, C. & Zaharia, M. "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" (MLSys 2023). Block-sparse MoE baseline.
- Mishra, A., et al. "Accelerating Sparse Deep Neural Networks" (2021). arXiv:2104.08378. NVIDIA 2:4 sparsity foundation.
- Frantar, E. & Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" (ICML 2023). Pruning method compatible with Samoyeds.

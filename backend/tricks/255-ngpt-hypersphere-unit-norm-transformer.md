# 255: nGPT — Hypersphere Unit-Norm Transformer

**Category**: stability
**Gain type**: efficiency
**Source**: Loshchilov, Hsieh, Sun & Ginsburg (2024) — "nGPT: Normalized Transformer with Representation Learning on the Hypersphere", ICLR 2025. arXiv:2410.01131
**Paper**: [papers/ngpt-normalized-transformer-hypersphere.pdf]
**Documented**: 2026-02-16

## Description

nGPT constrains **all** vectors in the Transformer — embeddings, hidden states, weight matrix rows/columns, queries, keys — to lie on a unit-norm hypersphere. This single geometric constraint eliminates the need for LayerNorm/RMSNorm, weight decay, and learning rate warmup, while providing inherent numerical stability: dot products become cosine similarities bounded in $[-1, 1]$, preventing the unbounded logit growth that causes attention entropy collapse and loss spikes.

The key insight is that the Transformer can be viewed as a **variable-metric optimizer on the hypersphere**: each attention/MLP block proposes a "gradient direction" (displacement on the sphere), and the network learns per-dimension **eigen learning rates** $\boldsymbol{\alpha}$ that control how much of each block's suggestion is incorporated. After each update, a normalization step (SLERP approximated by LERP + renormalize) projects the hidden state back onto the sphere — a Riemannian retraction step.

This achieves **4–20× faster convergence** (in training steps to same validation loss) compared to standard GPT, with the speedup increasing with sequence length. The stability comes from bounding all intermediate representations, which eliminates the root causes of gradient explosion: unconstrained norm growth in residual streams and attention logit drift.

## Mathematical Form

**Standard Transformer (Pre-LN):**

$$
\boldsymbol{h} \leftarrow \boldsymbol{h} + \text{ATTN}(\text{RMSNorm}(\boldsymbol{h}))
$$
$$
\boldsymbol{h} \leftarrow \boldsymbol{h} + \text{MLP}(\text{RMSNorm}(\boldsymbol{h}))
$$

**Normalized Transformer (nGPT):**

$$
\boldsymbol{h}_A = \text{Norm}(\text{ATTN}(\boldsymbol{h}))
$$
$$
\boldsymbol{h} \leftarrow \text{Norm}(\boldsymbol{h} + \boldsymbol{\alpha}_A \odot (\boldsymbol{h}_A - \boldsymbol{h}))
$$
$$
\boldsymbol{h}_M = \text{Norm}(\text{MLP}(\boldsymbol{h}))
$$
$$
\boldsymbol{h} \leftarrow \text{Norm}(\boldsymbol{h} + \boldsymbol{\alpha}_M \odot (\boldsymbol{h}_M - \boldsymbol{h}))
$$

where $\text{Norm}(\boldsymbol{x}) = \boldsymbol{x} / \|\boldsymbol{x}\|_2$ (unit normalization, no learnable scale).

**Key Definitions:**

- $\boldsymbol{h} \in \mathbb{R}^{d_{\text{model}}}$ — hidden state, constrained to $\|\boldsymbol{h}\|_2 = 1$
- $\boldsymbol{\alpha}_A, \boldsymbol{\alpha}_M \in \mathbb{R}_{\geq 0}^{d_{\text{model}}}$ — learnable eigen learning rates (per-dimension, per-layer)
- $\boldsymbol{h}_A, \boldsymbol{h}_M$ — normalized outputs of attention and MLP blocks

**SLERP Motivation:**

The update $\boldsymbol{h} \leftarrow \text{Norm}(\boldsymbol{h} + \boldsymbol{\alpha}(\boldsymbol{b} - \boldsymbol{h}))$ approximates the geodesic interpolation (SLERP) on the hypersphere:

$$
\text{SLERP}(\boldsymbol{a}, \boldsymbol{b}; \alpha) = \frac{\sin((1-\alpha)\theta)}{\sin(\theta)}\boldsymbol{a} + \frac{\sin(\alpha\theta)}{\sin(\theta)}\boldsymbol{b}
$$

where $\theta = \arccos(\boldsymbol{a} \cdot \boldsymbol{b})$. For small $\alpha$ (the typical regime — learned values are ~0.2–0.3), LERP + renormalize closely approximates SLERP.

**QK Normalization in nGPT:**

Queries and keys are additionally normalized after projection:

$$
\boldsymbol{q} \leftarrow \text{Norm}(\boldsymbol{q}) \cdot \boldsymbol{s}_{qk}, \quad \boldsymbol{k} \leftarrow \text{Norm}(\boldsymbol{k}) \cdot \boldsymbol{s}_{qk}
$$

where $\boldsymbol{s}_{qk} \in \mathbb{R}^{d_k}$ are trainable scaling factors (initialized at $d_k^{1/4}$). The softmax scaling changes from $1/\sqrt{d_k}$ to $\sqrt{d_k}$ since normalized QK dot products have variance $1/d_k$ instead of $1$.

**MLP Scaling:**

$$
\boldsymbol{u} \leftarrow \boldsymbol{u} \cdot \boldsymbol{s}_u, \quad \boldsymbol{\nu} \leftarrow \boldsymbol{\nu} \cdot \boldsymbol{s}_\nu \sqrt{d_{\text{model}}}
$$

where $\boldsymbol{s}_u, \boldsymbol{s}_\nu$ compensate for magnitude information lost by normalization.

**Logit Scaling:**

$$
\boldsymbol{z} \leftarrow \boldsymbol{z} \cdot \boldsymbol{s}_z
$$

where $\boldsymbol{s}_z \in \mathbb{R}^V$ is a per-vocabulary trainable temperature, since logits are cosine similarities in $[-1, 1]$ and need rescaling for proper softmax temperature.

## Complexity

| Operation | Standard GPT | nGPT |
|-----------|-------------|------|
| Forward pass FLOPs | $O(d^2)$ per layer | $O(d^2)$ per layer (same) |
| Normalization | RMSNorm: $O(d)$ per layer | Unit norm: $O(d)$ per layer |
| Weight normalization | None during forward | $O(d^2)$ per weight matrix (post-step) |
| Training steps to target loss | $S$ | $S/4$ to $S/20$ |

**Memory:** Same parameter count. Additional overhead: $2 \times L$ eigen learning rate vectors ($\boldsymbol{\alpha}_A, \boldsymbol{\alpha}_M \in \mathbb{R}^d$ per layer) and scaling factors ($\boldsymbol{s}_{qk}, \boldsymbol{s}_u, \boldsymbol{s}_\nu, \boldsymbol{s}_z$) — negligible vs. weight matrices.

**Wall-clock overhead:** ~60–80% per-step overhead for weight normalization (at 4K–8K context). This is amortized by the 4–20× fewer steps needed to reach target loss. Overhead decreases at larger model sizes where matmuls dominate.

## Applicability

- **Transformer LLM pretraining** at 0.5B–1B scale (validated). The 4–20× convergence speedup more than compensates for per-step overhead.
- **Long-context training**: Speedup increases with sequence length (4× at 1K, 10× at 4K, 20× at 8K context).
- Naturally extends to **encoder-decoder** and **cross-attention** architectures.
- Eliminates hyperparameter tuning for weight decay and LR warmup.
- Compatible with **RoPE** positional embeddings (handles longer contexts without modification per ablation studies).
- The **eigen learning rate** concept could be applied to SSMs and linear attention architectures.

## Limitations

- **Per-step overhead**: Weight normalization after each optimizer step adds 60–80% wall-clock time at small scale. For very large models, this overhead shrinks (matmul-dominated).
- **Requires full retraining**: Cannot be applied as a drop-in replacement to existing checkpoints — the geometry is fundamentally different.
- **Post-step normalization is a global sync**: All weight matrices must be normalized after each optimizer step, adding a sequential dependency. This is an elementwise operation though (divide rows by their norms), so it maps well to GPU.
- **Scaling beyond 1B**: The paper only validates at 0.5B and 1B parameters. Scaling behavior at 7B+ is unknown.
- **Interaction with quantization**: Constraining weights to unit norm may interact poorly with post-training quantization schemes that assume unconstrained weight distributions.

## Implementation Notes

```python
# Core nGPT update rule (per layer)
def ngpt_layer_update(h, block_output, alpha):
    """
    h: hidden state on unit sphere, shape [B, T, d]
    block_output: raw output from ATTN or MLP block
    alpha: eigen learning rate, shape [d] (per-dimension, learnable)
    """
    # Normalize block output to unit sphere
    h_block = F.normalize(block_output, dim=-1)

    # LERP on sphere (approximates SLERP for small alpha)
    h_new = h + alpha * (h_block - h)

    # Project back to sphere (retraction step)
    h_new = F.normalize(h_new, dim=-1)
    return h_new

# Post-optimizer-step weight normalization
def normalize_weights(model):
    """Normalize all weight matrices along embedding dimension."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # Normalize each row (embedding dimension) to unit norm
            param.data = F.normalize(param.data, dim=-1)

# Recipe to convert baseline Transformer to nGPT:
# 1. Remove all RMSNorm/LayerNorm layers
# 2. Normalize all weight matrices along embedding dim after each step
# 3. Replace residual add with: h = Norm(h + alpha * (h_block - h))
# 4. Add learnable scaling factors s_qk, s_u, s_v, s_z
# 5. Change softmax scale from 1/sqrt(d_k) to sqrt(d_k)
# 6. Remove weight decay and LR warmup
# 7. Initialize alpha ~ 0.05 (order of 1/n_layers)
```

## GPU Efficiency Analysis

**Memory Access Pattern:**
- Unit normalization is elementwise (reduction + divide) — same pattern as RMSNorm, fully coalesced
- Weight normalization is a per-row reduction — maps to standard row-wise kernel, coalesced
- No irregular memory access patterns introduced

**Parallelism:**
- All normalization operations are embarrassingly parallel across batch, sequence, and embedding dimensions
- No sequential dependencies beyond standard Transformer
- Weight normalization can be fused with optimizer step

**Tensor Core Utilization:**
- Core matmuls (QKV projection, attention, MLP) are identical to standard Transformer — full tensor core utilization
- The normalization steps are elementwise/reduction ops, not matmul — same as existing RMSNorm

**Arithmetic Intensity:**
- Slightly lower per-step (extra normalizations are memory-bound ops)
- But 4–20× fewer steps dramatically reduces total HBM traffic over full training

## References

- Loshchilov, Hsieh, Sun & Ginsburg. "nGPT: Normalized Transformer with Representation Learning on the Hypersphere." ICLR 2025. arXiv:2410.01131
- NVIDIA implementation: https://github.com/NVIDIA/ngpt
- Salimans & Kingma (2016). Weight Normalization.
- Henry et al. (2020). QK Normalization in attention.
- Wang & Isola (2020). Representation learning on hypersphere.
- Kosson et al. (2023). Rotation-based weight analysis.

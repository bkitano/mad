---
status: ongoing
priority: high
created: 2026-02-15
based_on: two-four-structured-sparsity, permutation-augmented-structured-sparsity, monarch-matrix-factorization, input-dependent-gating, cayley-contractive-parameterization, io-aware-tiling, kernel-fusion
experiment_number: 010
---

# Sparse Monarch SSM: 2:4 Structured Sparsity with Permutation-Augmented Expressivity

## Hypothesis

Applying **2:4 structured sparsity** to the block matrices within Monarch-factored SSM state transitions, combined with **permutation-augmented structured sparsity (PA-DST)** to restore the expressivity lost from pruning, will yield a state transition that is:

1. **$2\times$ faster per step** than dense Monarch (via NVIDIA Sparse Tensor Cores)
2. **Equally expressive** as dense Monarch (PA-DST's learned permutation recovers the lost rank)
3. **Still sub-quadratic**: $O(n\sqrt{n} / 2)$ per step — half the cost of dense Monarch, still cheaper than dense $O(n^2)$

The key insight is that Monarch's block-diagonal structure naturally decomposes into small matrices ($\sqrt{n} \times \sqrt{n}$, typically $16 \times 16$) that are the **ideal granularity** for 2:4 sparsity — small enough that 50% sparsity doesn't catastrophically reduce expressivity, but large enough that Sparse Tensor Cores provide real throughput gains. PA-DST's learned permutations then "undo" the damage from pruning by reorienting coordinates between blocks.

## Background

### The Monarch Bottleneck

Proposal 006 (Monarch-Gated State Transition) introduces Monarch-factored state transitions: $M(x_t) = P_b^\top L(x_t) P_b R(x_t)$, where $L$ and $R$ are block-diagonal with blocks of size $\sqrt{n} \times \sqrt{n}$. This achieves $O(n\sqrt{n})$ per step — a significant improvement over dense $O(n^2)$, but still $\sqrt{n} \times$ more expensive than diagonal SSMs.

For $n = 256$ (a typical state dimension), the Monarch cost is $16 \times$ more per step than diagonal. Proposal 006's analysis estimates the effective overhead is $\sim 8\times$ after accounting for BMM efficiency. **Can we cut this in half?**

### 2:4 Sparsity on Small Blocks

NVIDIA's Ampere+ architecture provides hardware-native 2:4 structured sparsity support: out of every 4 contiguous weight elements, exactly 2 must be zero, enabling $2\times$ throughput on Sparse Tensor Cores. This has been successfully applied to Transformer weight matrices ($d \times d$ with $d \geq 768$), but applying it to **small block matrices** ($16 \times 16$ or $32 \times 32$) within structured factorizations is unexplored.

**Why small blocks are actually ideal for 2:4 sparsity**:
- Large matrices have complex weight distributions where 50% pruning causes significant accuracy loss
- Small blocks have simpler structure (fewer important singular values), so 50% pruning is less damaging
- The permutation between blocks (Monarch's $P_b$) already provides coordinate mixing, so intra-block sparsity has a smaller relative impact

### PA-DST: Restoring What Sparsity Takes Away

Permutation-Augmented Structured Sparsity (PA-DST) adds a learned permutation $P$ such that $W' = S \cdot P$, where $S$ is the structured-sparse matrix. The permutation reorients coordinates to align important weights with the non-zero pattern, restoring dense-level expressivity. Critically, **the permutation can be absorbed at inference time as a zero-cost index remapping**.

For Monarch blocks, this means: apply 2:4 sparsity to each block $L_i$, $R_i$, but learn block-local permutations $P_i^L$, $P_i^R$ that optimize the alignment between the block's important directions and the 2:4 pattern.

## Mathematical Formulation

### Dense Monarch Transition (from Proposal 006)

$$
M(x_t) = P_b^\top \cdot L(x_t) \cdot P_b \cdot R(x_t)
$$

with:
$$
L(x_t) = \text{blkdiag}\left(\alpha_1(x_t) L_1, \ldots, \alpha_{\sqrt{n}}(x_t) L_{\sqrt{n}}\right)
$$
$$
R(x_t) = \text{blkdiag}\left(\beta_1(x_t) R_1, \ldots, \beta_{\sqrt{n}}(x_t) R_{\sqrt{n}}\right)
$$

where $L_i, R_i \in \mathbb{R}^{\sqrt{n} \times \sqrt{n}}$ are fixed orthogonal blocks (Cayley-parameterized), and $\alpha_i, \beta_i \in (0, 1)$ are input-dependent scalar gates.

### Proposed: Sparse-Monarch Transition

**Step 1**: Apply 2:4 structured sparsity to each block, with learned permutations:

$$
L_i^{\text{sparse}} = \text{Mask}_{2:4}\left(L_i \cdot P_i^L\right)
$$

$$
R_i^{\text{sparse}} = \text{Mask}_{2:4}\left(R_i \cdot P_i^R\right)
$$

where:
- $\text{Mask}_{2:4}(\cdot)$ retains the 2 largest-magnitude entries per group of 4 contiguous elements
- $P_i^L, P_i^R \in S_{\sqrt{n}}$ are learned block-local permutations (trained via Gumbel-Sinkhorn, fixed at inference)
- $L_i, R_i$ are the underlying dense blocks (Cayley-parameterized for orthogonality)

**Step 2**: Absorb permutations at inference:

At inference, the permutations are folded into the block structure:

$$
\tilde{L}_i = \text{Mask}_{2:4}\left(L_i \cdot P_i^L\right) = S_i^L \quad \text{(sparse matrix, stored in 2:4 format)}
$$

The permutation $P_i^L$ is no longer needed — it was used only during training to learn the optimal 2:4 mask alignment.

**Step 3**: The full sparse-Monarch transition:

$$
M^{\text{sparse}}(x_t) = P_b^\top \cdot L^{\text{sparse}}(x_t) \cdot P_b \cdot R^{\text{sparse}}(x_t)
$$

with:

$$
L^{\text{sparse}}(x_t) = \text{blkdiag}\left(\alpha_1(x_t) \tilde{L}_1, \ldots, \alpha_{\sqrt{n}}(x_t) \tilde{L}_{\sqrt{n}}\right)
$$

### Key Variables

- $n$ — state dimension (perfect square, e.g., $n = 256$)
- $b = \sqrt{n}$ — block count and block size (e.g., $b = 16$)
- $L_i, R_i \in \mathbb{R}^{b \times b}$ — dense block matrices
- $\tilde{L}_i, \tilde{R}_i$ — 2:4 sparse versions of blocks
- $P_i^L, P_i^R \in S_b$ — block-local permutations (PA-DST)
- $P_b \in \{0,1\}^{n \times n}$ — Monarch stride permutation (fixed)
- $\alpha_i, \beta_i \in (0, 1)$ — input-dependent scalar gates

### Training Procedure

**Phase 1: Dense pre-training** (optional warmup, ~10% of total steps)
- Train dense Monarch blocks $L_i, R_i$ without sparsity
- Initialize block-local permutations $P_i^L = P_i^R = I$

**Phase 2: PA-DST learning** (~20% of total steps)
- Learn permutations $P_i^L, P_i^R$ via Gumbel-Sinkhorn relaxation
- Gradually anneal Sinkhorn temperature $\tau: 1.0 \to 0.1$
- Simultaneously fine-tune $L_i, R_i$ (the blocks co-adapt with permutations)

**Phase 3: Sparse fine-tuning** (~70% of total steps)
- Fix permutations (project to hard permutation via Hungarian)
- Apply 2:4 mask: $\tilde{L}_i = \text{Mask}_{2:4}(L_i \cdot P_i^L)$
- Fine-tune remaining non-zero weights with straight-through gradient estimator for masked weights
- Absorb permutations into index layout for inference

### Stability Guarantee

With 2:4 sparsity, each block $\tilde{L}_i$ has 50% of entries zeroed. Is contractivity preserved?

**Claim**: If the dense block $L_i$ is orthogonal (Cayley-parameterized, $\|L_i\| = 1$), the sparse block satisfies $\|\tilde{L}_i\| \leq 1$ because zeroing entries can only reduce the spectral norm.

**More precisely**: For any matrix $A$ and sparsity mask $M \in \{0, 1\}^{b \times b}$:

$$
\|A \odot M\|_2 \leq \|A\|_2 \cdot \sqrt{\max_j \|M_{:,j}\|_0} = \|A\|_2 \cdot \sqrt{b/2}
$$

Wait — this is an overestimate. The Frobenius norm satisfies $\|A \odot M\|_F \leq \|A\|_F$, but the spectral norm may increase for specific masks. Let me reconsider.

**Corrected analysis**: For 2:4 structured sparsity applied to an orthogonal matrix, the spectral norm of the masked matrix is $\leq 1$ because:
- $\|A \odot M\|_2 \leq \|A\|_2 = 1$ only if $M$ is a submatrix selector (rows/columns), not for arbitrary elementwise masks
- For general elementwise masks, $\|A \odot M\|_2$ can exceed $\|A\|_2$

**Resolution**: Rely on the input-dependent scalar gates $\alpha_i, \beta_i \in (0, 1)$ for contractivity:

$$
\|M^{\text{sparse}}(x_t)\| \leq \max_i \alpha_i \cdot \max_j \beta_j \cdot \|\tilde{L}\|_{\max} \cdot \|\tilde{R}\|_{\max}
$$

If $\max \alpha_i \cdot \max \beta_j < 1 / (\|\tilde{L}\|_{\max} \cdot \|\tilde{R}\|_{\max})$, the transition is contractive. Since the gates are sigmoid-bounded and the sparse blocks have bounded norm ($\|\tilde{L}_i\|_2 \leq \sqrt{b}$ in the worst case for a $b \times b$ matrix with $b/2$ nonzeros per row), we need $\alpha_i \beta_j < 1/b$. For $b = 16$, this means gates must be $< 0.25$ on average — achievable with bias initialization of $W_g$.

**Practical approach**: Add spectral normalization to the sparse blocks during training:

$$
\tilde{L}_i \leftarrow \tilde{L}_i / \max(1, \|\tilde{L}_i\|_2)
$$

This is a single SVD per block (only $16 \times 16$, negligible cost) and ensures $\|\tilde{L}_i\|_2 \leq 1$.

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Sparse-Monarch-Gated SSM |
| Layers | $L = 12$ |
| Hidden dim | $d = 768$ |
| State dim | $n = 256$ ($16 \times 16$ blocks) |
| Block size | $b = \sqrt{n} = 16$ |
| Sparsity | 2:4 structured (50%) per block |
| PA-DST | Block-local Gumbel-Sinkhorn permutations |
| Gate params | $2\sqrt{n} = 32$ per head |

### Baseline

1. **Dense Monarch SSM** (Proposal 006): $O(n\sqrt{n})$ per step — no sparsity
2. **Mamba-2** (diagonal): $O(n)$ per step — diagonal, no coordinate mixing
3. **Naively sparse Monarch** (2:4 without PA-DST): Same sparsity but no learned permutation alignment
4. **Unstructured 50% sparse Monarch**: Random magnitude pruning to 50%, without hardware acceleration

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Per-step throughput | $> 1.5\times$ dense Monarch | Timed forward pass, Sparse Tensor Cores |
| State tracking ($S_5$) | $> 80\%$ accuracy | Permutation group composition |
| Quality gap vs. dense | $< 2\%$ perplexity diff | WikiText-103, 380M param |
| PA-DST benefit | $> 10\%$ accuracy gain over naive sparse | $S_5$ task comparison |
| Memory savings | $> 30\%$ reduction in block storage | Parameter count (2:4 compressed format) |

### Estimated Compute

**MVE**: ~15 minutes on single A100 (~$0.75)
**Small-scale**: 8 GPU-hours on A100 (~$32)
**Full-scale**: 48 GPU-hours on A100 (~$200)

## Expected Outcome

**If hypothesis is correct:**

1. **Throughput**: Sparse-Monarch SSM achieves $1.5$–$1.8\times$ throughput vs. dense Monarch on A100 (NVIDIA's documented 2:4 sparsity provides $\sim 2\times$ on matmul; overhead from Sinkhorn training and spectral norm reduces this to $1.5$–$1.8\times$ net).

2. **Expressivity preservation**: With PA-DST, the sparse-Monarch SSM achieves $> 80\%$ accuracy on $S_5$ permutation composition (vs. $> 85\%$ for dense Monarch), closing most of the gap. Without PA-DST (naive 2:4 pruning), accuracy drops to $< 60\%$ — demonstrating PA-DST's critical role.

3. **Quality**: On WikiText-103 at 380M params, sparse-Monarch SSM perplexity is within $2\%$ of dense Monarch. The permutation-augmented sparsity recovers nearly all the expressivity while halving the block computation.

4. **Effective complexity**: The sparse-Monarch SSM operates at effective cost $O(n\sqrt{n} / 2) \approx O(n^{1.25})$, placing it between diagonal ($O(n)$) and dense Monarch ($O(n^{1.5})$) on the expressivity-efficiency Pareto frontier.

**If hypothesis is wrong:**

- **Scenario A**: 2:4 sparsity on $16 \times 16$ blocks is too aggressive
  - **Learn**: Small blocks have insufficient redundancy for 50% pruning. The important singular values are spread across all entries.
  - **Fix**: Use larger state dim ($n = 1024$, blocks = $32 \times 32$) where redundancy increases, or use less aggressive sparsity (4:8).

- **Scenario B**: PA-DST permutations don't help (naive sparse ≈ PA-DST sparse)
  - **Learn**: Block-local permutations are too small ($16!$ options) to meaningfully reorient the pruning mask. The Monarch stride permutation $P_b$ already provides sufficient coordinate diversity.
  - **Fix**: Try inter-block permutations (reorder which state dimensions map to which block) instead of intra-block permutations.

- **Scenario C**: Sparse Tensor Core speedup doesn't materialize for BMM
  - **Learn**: NVIDIA's 2:4 support is optimized for large GEMMs, not batched small-block multiplies. The overhead of loading block metadata dominates.
  - **Fix**: Fuse multiple block multiplies into a single large sparse GEMM via block-diagonal packing.

## Minimum Viable Experiment

### Setup
- **Model**: 2-layer Sparse-Monarch SSM ($d = 64$, $n = 64$, blocks = $8 \times 8$, ~120K params)
- **Task**: $S_5$ permutation group composition — given a sequence of $S_5$ generators, predict the resulting permutation
- **Data**: 10K synthetic sequences of length 20
- **Compute**: Single A100 GPU, $< 15$ minutes
- **Comparison**: 4 models — dense Monarch, naive 2:4 sparse, PA-DST 2:4 sparse, diagonal SSM

### Success Criteria
- PA-DST sparse Monarch achieves $> 70\%$ accuracy on $S_5$ composition (length 20)
- Dense Monarch achieves $> 80\%$ (upper bound reference)
- Naive 2:4 sparse Monarch achieves $< 55\%$ (to demonstrate PA-DST's value)
- Diagonal SSM achieves $< 45\%$ (expected, cannot do permutation routing)
- Forward pass of sparse Monarch is at least $1.2\times$ faster than dense (even without custom Sparse Tensor Core kernel, PyTorch sparse should show some speedup)

### Failure Criteria
- PA-DST sparse Monarch performs no better than naive sparse ($< 5\%$ gap) — permutations don't help at this block size
- All sparse variants perform $< 50\%$ — 2:4 sparsity is too destructive for $8 \times 8$ blocks
- Dense Monarch also fails ($< 60\%$) at $n = 64$ — block size too small for $S_5$ (need to increase $n$, not a sparsity issue)

### Why This Test Is Sufficient
- **$S_5$ is the canonical coordinate-mixing test**: It requires the transition matrix to implement permutations — the exact capability that sparsity threatens and PA-DST aims to restore
- **Small blocks ($8 \times 8$) are the hardest case**: If PA-DST works for $8 \times 8$ blocks, it will work even better for $16 \times 16$ (more redundancy, more room for the permutation to optimize)
- **The ranking (PA-DST > naive > diagonal) is the key signal**: We don't need state-of-the-art accuracy, just clear evidence that PA-DST recovers expressivity from sparsity
- **15 minutes for 4 models**: Fast enough to iterate on hyperparameters (Sinkhorn temperature, training schedule)

### Implementation Sketch

```python
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

class SparseMonarchBlock(nn.Module):
    """Single Monarch block with 2:4 structured sparsity + PA-DST."""

    def __init__(self, block_size, use_pa_dst=True, use_sparsity=True):
        super().__init__()
        self.block_size = block_size
        self.use_pa_dst = use_pa_dst
        self.use_sparsity = use_sparsity

        # Dense block (Cayley-parameterized for orthogonality)
        self.skew = nn.Parameter(torch.randn(block_size, block_size) * 0.1)

        # PA-DST: block-local permutation (Gumbel-Sinkhorn)
        if use_pa_dst:
            self.perm_logits = nn.Parameter(torch.zeros(block_size, block_size))

    def get_dense_block(self):
        """Cayley transform: W = (I + A)^{-1}(I - A) for skew-symmetric A."""
        A = self.skew - self.skew.T  # Make skew-symmetric
        I = torch.eye(self.block_size, device=A.device)
        return torch.linalg.solve(I + A, I - A)

    def get_permutation(self, tau=1.0):
        """Gumbel-Sinkhorn permutation."""
        if not self.use_pa_dst:
            return torch.eye(self.block_size, device=self.skew.device)
        noise = -torch.log(-torch.log(torch.rand_like(self.perm_logits) + 1e-20) + 1e-20)
        log_alpha = (self.perm_logits + noise) / tau
        for _ in range(20):  # Sinkhorn iterations
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def apply_24_mask(self, W):
        """Apply 2:4 structured sparsity mask."""
        if not self.use_sparsity:
            return W
        b = self.block_size
        W_flat = W.reshape(-1, 4)  # Group into 4s
        _, indices = W_flat.abs().topk(2, dim=-1)  # Keep top 2
        mask = torch.zeros_like(W_flat)
        mask.scatter_(1, indices, 1.0)
        return (W_flat * mask).reshape(b, b)

    def forward(self):
        """Return the sparse block matrix."""
        W = self.get_dense_block()
        if self.use_pa_dst:
            P = self.get_permutation()
            W = W @ P
        W_sparse = self.apply_24_mask(W)
        # Spectral normalize for stability
        W_sparse = W_sparse / max(1.0, torch.linalg.norm(W_sparse, ord=2).item())
        return W_sparse
```

## Theoretical Analysis

### Complexity Comparison

| Operation | Diagonal SSM | Dense Monarch | Sparse Monarch | Dense $O(n^2)$ |
|-----------|-------------|---------------|----------------|----------------|
| Per-step cost | $O(n)$ | $O(n\sqrt{n})$ | $O(n\sqrt{n}/2)$ | $O(n^2)$ |
| Parameters (transition) | $O(n)$ | $O(n\sqrt{n})$ | $O(n\sqrt{n}/2)$ | $O(n^2)$ |
| Memory (compressed) | $O(n)$ | $O(n\sqrt{n})$ | $O(n\sqrt{n} \cdot 0.56)$ | $O(n^2)$ |
| GPU utilization | Low (element-wise) | High (BMM) | High (Sparse TC) | High (GEMM) |
| Coordinate mixing | None | Full (via $P_b$) | Partial + PA-DST | Full |

For $n = 256$:
- Diagonal: $256$ FLOPs/step
- Dense Monarch: $256 \times 16 = 4096$ FLOPs/step
- Sparse Monarch: $\approx 2048$ FLOPs/step (2× speedup from Sparse TC)
- Dense: $256^2 = 65536$ FLOPs/step

**Crossover with dense**: Sparse Monarch is faster than dense when $n\sqrt{n}/2 < n^2$, i.e., $\sqrt{n} > 2$ — always true for $n \geq 4$.

**Effective complexity**: $O(n^{1.25})$ for sparse Monarch vs. $O(n^{1.5})$ for dense Monarch.

### Memory Analysis (2:4 Compressed Format)

NVIDIA's 2:4 format stores the non-zero values (50% of entries) plus a 2-bit index per group of 4, yielding 44% memory savings for 16-bit weights:

- Dense block ($16 \times 16$, fp16): $256 \times 2 = 512$ bytes
- Sparse block (2:4, fp16): $128 \times 2 + 64 \times 0.5 = 288$ bytes (44% savings)

For $n = 256$ with $2 \times 16$ blocks (L and R):
- Dense: $2 \times 16 \times 512 = 16384$ bytes
- Sparse: $2 \times 16 \times 288 = 9216$ bytes (44% savings)

### PA-DST Expressivity Recovery

PA-DST theory (from the trick documentation) shows that the **effective rank growth** of a structured-sparse layer is restored from stalled growth to dense-like growth after a warm-up period of $\lceil d_0 / r_{\text{struct}} \rceil$ layers, where $d_0$ is the initial rank deficit and $r_{\text{struct}}$ is the structural rank per layer.

For 2:4 sparse $16 \times 16$ blocks:
- Without PA-DST: Each block has rank $\leq 8$ (at most 8 non-zero rows effectively), so the block can represent at most rank-8 transformations
- With PA-DST: The permutation reorients the important $r$ singular vectors to align with the non-zero pattern, enabling effective rank up to $\min(16, r_{\text{task}})$

**Prediction**: For $S_5$ tracking (requires rank-5 permutation representation), PA-DST should fully recover expressivity since $5 < 8$ (the sparse block's maximum effective rank exceeds the task requirement).

## Risks & Limitations

### Risk 1: Block Size Too Small for 2:4 Benefit
- **Issue**: NVIDIA's Sparse Tensor Core support is optimized for matrices $\geq 16 \times 16$. For $8 \times 8$ blocks (MVE), the overhead of sparse metadata may dominate.
- **Mitigation**: MVE uses $8 \times 8$ for proof of concept (slower is OK). Full experiment uses $16 \times 16$ (ideal for Sparse TC).
- **Fallback**: Pack multiple small blocks into a single large sparse operation via block-diagonal layout.

### Risk 2: Gumbel-Sinkhorn Training Instability
- **Issue**: Learning permutations for $16 \times 16$ blocks via Gumbel-Sinkhorn adds training complexity. The loss landscape over the Birkhoff polytope may have bad local optima.
- **Mitigation**:
  - Warm start: Train dense for 10% of steps before enabling permutation learning
  - Anneal temperature slowly: $\tau = 1.0 \to 0.1$ over 20% of training
  - Fall back to random fixed permutation if Sinkhorn doesn't converge
- **Alternative**: Use straight-through estimator instead of Sinkhorn for simplicity

### Risk 3: Spectral Norm Growth Under Sparsity
- **Issue**: 2:4 masking of an orthogonal matrix may increase the spectral norm, breaking the contractivity guarantee from Cayley parameterization.
- **Mitigation**: Explicit spectral normalization after masking ($\tilde{L}_i / \max(1, \|\tilde{L}_i\|_2)$). This is cheap for $16 \times 16$ matrices.
- **Theory**: Cayley parameterization ensures the *dense* block is orthogonal. The sparse block is sub-orthogonal (some singular values reduced, none increased above the original max). So spectral norm $\leq 1$ should hold... but this needs verification.

### Risk 4: Interaction with Input-Dependent Gates
- **Issue**: The input-dependent scalar gates $\alpha_i, \beta_i$ multiply each block. If the gates are near 1.0, they don't compensate for the spectral norm increase from sparsity.
- **Mitigation**: Initialize gate bias to produce $\alpha, \beta \approx 0.5$ (safe contractivity margin). The gates learn to open as needed.

### Risk 5: Training-Inference Gap
- **Issue**: Training uses soft Sinkhorn permutations; inference uses hard permutations. The gap between soft and hard may cause accuracy drop.
- **Mitigation**: Gradually anneal temperature during training until the soft permutation is nearly hard. Use straight-through estimator in the final phase.

## Follow-up Experiments

### If Successful:
1. **Sparse-Monarch$^2$**: Apply 2:4 sparsity to stacked Monarch factors (Monarch$^2$), which can represent DFT/DCT. Tests if PA-DST recovers transform-level expressivity.
2. **Dynamic sparsity patterns**: Instead of fixed 2:4 masks, learn input-dependent masks via Gumbel-softmax — different tokens prune different entries.
3. **Compose with OscGate-SSM (Proposal 007)**: Replace the diagonal blocks in the oscillatory $2 \times 2$ structure with sparse-Monarch blocks — combining oscillatory stability with sub-quadratic coordinate mixing.
4. **Scale to 1B+ params**: Test if 2:4 sparsity on Monarch blocks provides compound savings at scale (memory + compute).
5. **Extend to GQA-style state sharing**: Share sparse blocks across multiple SSM heads (analogous to grouped query attention), reducing parameters further.

### If Unsuccessful:
1. **Try 4:8 sparsity**: Less aggressive pruning (25% sparse) that preserves more block structure.
2. **Increase block size**: Use $n = 1024$ with $32 \times 32$ blocks — more redundancy for sparsity.
3. **Unstructured pruning comparison**: If 2:4 doesn't help, try magnitude pruning to 50% without hardware constraints — isolates whether the issue is sparsity level or structure.
4. **Block-level expert routing**: Instead of pruning within blocks, route tokens to a subset of blocks (MoE-style) — achieves sparsity at the block level, not within blocks.

## Connection to Existing Proposals

- **Direct extension of 006 (Monarch-Gated SSM)**: This proposal takes proposal 006's Monarch transition and makes it faster via structured sparsity. Proposal 006 lists "2:4 sparsity on Monarch blocks" as follow-up item #2 — this proposal fully develops that idea with PA-DST recovery.
- **Uses stability from 006/007**: Inherits Cayley parameterization for orthogonal blocks and sigmoid gating for contractivity.
- **Complementary to 009 (Post-Sigmoid Gating)**: Proposal 009 improves the readout; this proposal improves the state transition. Both can be applied simultaneously.
- **Unique contribution**: Only proposal exploring hardware-accelerated structured sparsity within algebraically-structured matrix factorizations. Bridges the GPU kernel optimization literature (2:4 sparsity) with the structured SSM literature (Monarch factorization).

---
status: completed
priority: medium
created: 2026-02-15
based_on: 
experiment_number: 005
results_file: 005_results.md
---

# Segmented-HSS Linear Attention: Variable-Length Hierarchical Attention

- segmented-scan
- hierarchically-semiseparable-hss-matrices
- linear-attention-approximation
- telescopic-decomposition-hss
- recurrence-to-scan-reduction

## Hypothesis

Combining segmented scan (variable-length batching) with HSS (hierarchically semiseparable) matrix structure for linear attention will enable:
1. **O(n log n)** complexity for linear attention updates (vs O(n²) for full attention, O(nd²) for naive linear attention)
2. **Variable-length batching** without padding waste (critical for training efficiency)
3. **Hierarchical attention patterns** that naturally capture multi-scale dependencies

**Key insight**: Linear attention materializes a state matrix $S_t \in \mathbb{R}^{d \times d}$ that accumulates key-value products. This matrix is typically dense (O(d²) storage/update). BUT: if we parameterize $S_t$ as HSS, we get O(d log d) storage + update via telescopic decomposition.

## Background

**Current landscape:**

- **Full attention**: $O(T^2 d)$ time, $O(T^2)$ memory → prohibitive for long sequences
- **Linear attention**: $O(Td^2)$ time, $O(d^2)$ memory → efficient but loses softmax expressivity
- **FlashAttention**: $O(T^2 d)$ time but IO-optimized → still quadratic complexity
- **Log-linear attention**: $O(T \log T)$ via hierarchical masking → maintains quadratic structure, just more efficient

**Gap**: No architecture exploits HSS structure for linear attention state matrices.

**Why this matters**:
- Linear attention is parallelizable (unlike SSMs) but has large $d$ overhead
- Variable-length sequences waste compute with padding (up to 50% in practice)
- Hierarchical structure matches multi-scale nature of language (words → phrases → sentences)

### Linear Attention Background

Standard attention:
$$
\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V
$$

Linear attention (feature map approximation):
$$
\text{Attn}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

Recurrent form (enables O(Td²) complexity):
$$
S_t = S_{t-1} + \phi(k_t) \phi(v_t)^T \in \mathbb{R}^{d \times d}
$$
$$
z_t = z_{t-1} + \phi(k_t) \in \mathbb{R}^{d}
$$
$$
o_t = \frac{\phi(q_t)^T S_t}{\phi(q_t)^T z_t}
$$

**Cost**: $O(d^2)$ per step for $S_t$ update; $O(Td^2)$ total.

### HSS Matrix Background

A matrix $M \in \mathbb{R}^{n \times n}$ is **rank-$r$ semiseparable** if all off-diagonal blocks have rank $\leq r$.

**HSS structure**: Recursively partition into $2 \times 2$ blocks:
$$
M = \begin{bmatrix} D_{11} & U_1 W_{12} V_2^T \\ U_2 W_{21} V_1^T & D_{22} \end{bmatrix}
$$

where:
- $D_{11}, D_{22}$ are smaller HSS matrices (recursive)
- $U_i, V_i \in \mathbb{R}^{n/2 \times r}$ — basis matrices
- $W_{ij} \in \mathbb{R}^{r \times r}$ — coupling matrices

**Storage**: $O(nr \log n)$ instead of $O(n^2)$ for dense.

**Key operations**:
- Matrix-vector multiply: $O(nr \log n)$
- Inversion: $O(nr^2 \log n)$
- Matrix functions $f(M)$: $O(nr^2 \log^2 n)$ via telescopic decomposition

## Mathematical Formulation

### Proposed Segmented-HSS Linear Attention

**Step 1**: Parameterize state matrix $S_t$ in HSS form.

At depth $\ell = 0$ (leaves), binary tree nodes correspond to dimensions $[i, i+1)$:
$$
D_i = s_i \in \mathbb{R} \quad \text{(scalars)}
$$

At depth $\ell > 0$, internal nodes:
$$
S^{(\ell)} = \begin{bmatrix} S^{(\ell-1)}_L & U_L W V_R^T \\ U_R W^T V_L^T & S^{(\ell-1)}_R \end{bmatrix}
$$

where $U_L, V_L, U_R, V_R \in \mathbb{R}^{d/2 \times r}$, $W \in \mathbb{R}^{r \times r}$.

**Step 2**: Update HSS structure with rank-1 addition $\phi(k_t) \phi(v_t)^T$.

Rank-1 update to HSS matrix:
$$
S_t = S_{t-1} + \phi(k_t) \phi(v_t)^T
$$

**Key insight**: HSS matrices are closed under low-rank updates via telescopic recompression.

**Algorithm** (simplified):
1. Decompose rank-1 update as $\phi(k_t) = \sum_{i} k_i e_i$ (in leaf basis)
2. Propagate update up binary tree, accumulating at each level
3. Recompress: If rank exceeds $r$, apply truncated SVD to $U_L W V_R^T$

**Complexity**: $O(r^2 \log d)$ per update (vs $O(d^2)$ for dense).

**Step 3**: Query HSS matrix for attention output.

$$
o_t = S_t \phi(q_t)
$$

Via HSS matrix-vector multiply: $O(r d \log d)$.

**Step 4**: Segmented scan for variable-length batching.

Batch contains sequences of lengths $L_1, \ldots, L_B$. Use segmented scan to:
- Reset state $S_t$ at segment boundaries
- Process all sequences in parallel
- Avoid padding waste

**Segment flag**: $f_t \in \{0, 1\}$ where $f_t = 1$ indicates start of new sequence.

**Segmented scan operator**:
$$
S_t = \begin{cases} \phi(k_t) \phi(v_t)^T & \text{if } f_t = 1 \\ S_{t-1} + \phi(k_t) \phi(v_t)^T & \text{if } f_t = 0 \end{cases}
$$

Implemented via scan with binary operator:
$$
(S, f) \oplus (S', f') = \begin{cases} (S', f') & \text{if } f' = 1 \\ (S + S', 0) & \text{if } f' = 0 \end{cases}
$$

### Standard Linear Attention (for comparison)

**Recurrent form**:
$$
S_t = S_{t-1} + k_t v_t^T \quad \in \mathbb{R}^{d \times d}
$$
$$
o_t = \frac{q_t^T S_t}{q_t^T z_t} \quad \in \mathbb{R}^{d}
$$

**Complexity**: $O(d^2)$ per step, $O(Td^2)$ total.

**Batching**: Requires padding to max length $L_{\max}$; wasted compute on padding tokens.

**Storage**: $O(Bd^2)$ for $B$ sequences (stores full $d \times d$ matrix per sequence).

### Key Variables

- $T$ — total sequence length (sum across batch)
- $d$ — model dimension
- $r$ — HSS rank (typically $r \ll d$, e.g., $r = 16$-$64$)
- $B$ — batch size
- $\phi(\cdot)$ — feature map (e.g., $\text{elu}(x) + 1$ or random Fourier features)

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Segmented-HSS Linear Attention |
| Layers | $L = 12$ |
| Model dimension | $d = 512$ |
| HSS rank | $r = 32$ |
| Attention heads | $H = 8$ |
| Head dimension | $d_h = d/H = 64$ |
| Feature map | $\phi(x) = \text{elu}(x) + 1$ |

### Baseline

**Primary**: Standard linear attention (dense state matrix)
- Complexity: $O(Td^2)$
- Memory: $O(Bd^2)$

**Secondary**: FlashAttention-2 (softmax attention with IO optimization)
- Complexity: $O(T^2 d)$
- Memory: $O(Td)$ (with recomputation)

**Tertiary**: Vanilla transformer (for quality reference)
- Complexity: $O(T^2 d)$
- Memory: $O(T^2)$

### Training Details

- **HSS initialization**:
  - Diagonal blocks $D_i$: Glorot uniform
  - Basis matrices $U, V$: Glorot uniform
  - Coupling matrices $W$: Identity (initially identity mapping)

- **Rank adaptation**: Start with $r=16$, increase to $r=32$ if accuracy lags baseline

- **Recompression**: Truncated SVD when rank exceeds $1.5r$; keep top $r$ singular values

- **Optimizer**: AdamW, lr=3e-4, warmup=10k steps, cosine decay

- **Sequence lengths**: Variable from 128 to 2048 (no padding)

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Accuracy | $\geq 0.95 \times$ linear attention | Test perplexity on WikiText-103 |
| Throughput | $\geq 1.5 \times$ linear attention | Tokens/sec on A100 (variable-length batches) |
| Memory | $\leq 0.5 \times$ linear attention | Peak GPU memory per sequence |
| Quality vs FlashAttn | $\geq 0.90$ | Perplexity ratio (HSS-LinAttn / FlashAttn) |

### Estimated Compute

**Medium**: ~120 GPU-hours on A100
- WikiText-103 training: 3 seeds × 2 models (HSS, baseline) = 6 runs
- ~20 GPU-hours per run (12-layer models, smaller scale than GPT)

## Expected Outcome

**If hypothesis is correct:**

1. **Memory efficiency**: $0.3\times$-$0.5\times$ memory usage vs dense linear attention
   - **Why**: $O(r d \log d)$ storage vs $O(d^2)$; for $d=512$, $r=32$: $\sim 0.1\times$ storage

2. **Throughput improvement**: $1.5\times$-$2\times$ tokens/sec on variable-length batches
   - **Why**: Segmented scan eliminates padding waste (up to 50% in practice)

3. **Quality preservation**: Perplexity within $5\%$ of dense linear attention
   - **Why**: HSS approximation is controlled (rank $r$ tunable); higher $r$ → better approximation

4. **Hierarchical attention patterns**: Visualization shows multi-scale structure (local + global)
   - **Why**: HSS binary tree naturally captures nested dependencies

**If hypothesis is wrong:**

- **Scenario A**: HSS rank-1 updates accumulate error, quality degrades
  - **Learn**: Recompression strategy is insufficient; need periodic full SVD or higher rank
  - **Fix**: Increase $r$ (e.g., $r=64$), or add periodic exact state recomputation

- **Scenario B**: HSS matrix-vector multiply is slower than dense due to tree traversal overhead
  - **Learn**: $O(r d \log d)$ constant is larger than $O(d^2)$ for $d=512$
  - **Fix**: Optimize tree implementation (flatten to avoid recursion), or batch multiple queries

- **Scenario C**: Segmented scan overhead dominates (segment boundaries destroy parallelism)
  - **Learn**: Too many short sequences in batch; segmented scan parallelism degrades
  - **Mitigation**: Pack similar-length sequences in batches; avoid high segment frequency

## Minimum Viable Experiment

**CRITICAL**: Validate HSS update mechanism before full training.

### Setup
- **Model**: Single HSS linear attention layer ($d=64$, $r=8$, ~10K params)
- **Task**: **Hierarchical Copying**
  - Input: Nested structure `[A [B C] D [E [F G] H]]` (8 tokens, 3 hierarchy levels)
  - Target: Copy input with delays: level-1 (immediate), level-2 (+4 steps), level-3 (+8 steps)
  - Example: Input at $t=0$, output level-1 at $t=0$, level-2 at $t=4$, level-3 at $t=8$
- **Why this task**: Requires multi-scale memory (local + global); HSS structure should excel
- **Data**: 5K synthetic sequences
- **Compute**: Single GPU, $< 5$ minutes

### Success Criteria
- **Perfect copying**: 100% accuracy on training set (HSS can represent nested dependencies)
- **Hierarchical state structure**: Visualize $S_T$ (final state); confirm block structure matches input hierarchy
- **Memory efficiency**: HSS state uses $< 0.2 \times$ memory vs dense $d \times d$ matrix

### Failure Criteria
- Accuracy $< 90\%$ (HSS cannot represent hierarchical dependencies)
- HSS state structure is unstructured (blocks are dense, not low-rank)
- Memory usage $\geq 0.5 \times$ dense (HSS overhead dominates)

### Why This Test Is Sufficient
- **Hierarchical structure validation**: If HSS can copy nested inputs, the binary tree structure is working
- **Update mechanism validation**: Perfect copying requires accurate rank-1 updates across many steps
- **Memory efficiency validation**: $d=64$, $r=8$ should show clear storage advantage

**Decision rule**:
- ✅ All success criteria met → Proceed to WikiText-103 training
- ❌ Any failure criterion → Debug HSS update (check recompression, tree propagation) before scaling

## Theoretical Analysis

### Complexity Comparison

| Operation | Dense LinAttn | HSS LinAttn | FlashAttention |
|-----------|---------------|-------------|----------------|
| State update | $O(d^2)$ | $O(r^2 \log d)$ | N/A (no state) |
| Query | $O(d^2)$ | $O(rd \log d)$ | $O(T d)$ (per query) |
| Total (training) | $O(Td^2)$ | $O(Trd \log d)$ | $O(T^2 d)$ |
| Memory per seq | $O(d^2)$ | $O(rd \log d)$ | $O(Td)$ (with recomp) |
| Batching overhead | $O(BL_{\max}d^2)$ | $O(Trd \log d)$ | $O(BL_{\max}^2 d)$ |

**Key insight**: Segmented scan eliminates $L_{\max}$ (max length) from complexity; use actual lengths $\sum L_i = T$.

**Crossover point**: HSS faster when $r^2 \log d < d^2$, i.e., $r < d / \sqrt{\log d}$.
- For $d=512$: $r < 512 / \sqrt{9} \approx 170$ ✓ (our $r=32$ is well below threshold)

### Memory Analysis

**Dense linear attention**:
$$
\text{Memory} = B \times d^2 = B \times 512^2 = 262144 B \text{ floats}
$$

**HSS linear attention**:
$$
\text{Memory} = B \times (2d + 2rd \log_2 d) \approx B \times (1024 + 2 \times 32 \times 512 \times 9) = B \times 295936 \text{ floats}
$$

Wait, this is LARGER! Issue: $r d \log d$ overhead dominates for small $d$.

**Correction**: For large $d$ (e.g., $d=2048$):
$$
\text{Dense} = B \times 2048^2 = 4194304 B
$$
$$
\text{HSS} = B \times (4096 + 2 \times 32 \times 2048 \times 11) = B \times 1454080 \approx 0.35 \times \text{dense}
$$

**Insight**: HSS advantage grows with $d$; may need $d \geq 1024$ to see significant memory savings.

### Accuracy Analysis

**HSS approximation error**: Bounded by truncation rank $r$.

Given true state $S_t^{\text{true}} = \sum_{i=1}^t k_i v_i^T$, HSS approximation $S_t^{\text{HSS}}$ satisfies:
$$
\|S_t^{\text{true}} - S_t^{\text{HSS}}\|_F \leq \epsilon_t
$$

where $\epsilon_t$ depends on recompression frequency and rank $r$.

**Practical**: If recompression keeps top $r$ singular values, error is:
$$
\epsilon_t \leq \sum_{j > r} \sigma_j
$$

where $\sigma_j$ are singular values of accumulated updates.

**Key question**: Does $\epsilon_t$ grow unboundedly with $t$, or does recompression control it?

**Hypothesis**: For linear attention, key-value products $k_i v_i^T$ are approximately low-rank (rank $\ll d$), so truncation error is small.

## Risks & Limitations

### Risk 1: HSS Overhead Dominates for Moderate $d$

- **Issue**: For $d=512$, $r=32$, $O(rd \log d)$ may not be significantly smaller than $O(d^2)$
- **Mitigation**: Test on larger models ($d=2048$) where asymptotic advantage is clearer
- **Fallback**: Use HSS only for $d \geq 1024$; dense for smaller dimensions

### Risk 2: Recompression Frequency Unclear

- **Issue**: Too frequent → expensive SVDs; too rare → rank explosion, quality loss
- **Mitigation**: Adaptive recompression: monitor rank, recompress when rank $> 1.5r$
- **Ablation**: Test recompression every $k$ steps for $k \in \{10, 50, 100\}$

### Risk 3: Segmented Scan Implementation Complexity

- **Issue**: Efficient GPU segmented scan requires careful kernel engineering
- **Mitigation**: Use existing implementations (e.g., CUB library, PyTorch compile)
- **Fallback**: Sequential processing of segments (loses parallelism but still avoids padding)

### Risk 4: HSS May Not Match Attention Structure

- **Issue**: Linear attention state $S_t = \sum k_i v_i^T$ may not be well-approximated by HSS
- **Mitigation**: Ablation study: measure rank of $S_t$ at different training steps; if rank $\gg r$, HSS is wrong structure
- **Insight**: If rank is high, consider alternative structures (e.g., Monarch, group-and-shuffle)

### Risk 5: Quality Gap vs Softmax Attention

- **Issue**: Linear attention (with or without HSS) is known to underperform softmax on some tasks
- **Mitigation**: This is a known limitation of linear attention; HSS structure shouldn't worsen it
- **Expectation**: HSS should match dense linear attention quality (within 5%); gap vs FlashAttention is expected

## Follow-up Experiments

### If Successful (meets target metrics):

1. **Scale to larger models** ($d=2048$, 1.3B params)
   - Test if memory advantage compounds at scale
   - Compare to Mamba/RWKV at similar parameter counts

2. **Hybrid HSS-softmax attention**
   - Use HSS linear attention for long-range (global) context
   - Use local softmax attention for short-range (within window)
   - Combine via mixture or hierarchical architecture

3. **Learned HSS structure**
   - Make binary tree structure learnable (which dimensions to group)
   - Hypothesis: Task-specific hierarchies emerge (e.g., syntactic groupings for language)

4. **Apply to other sequence models**
   - Use HSS for SSM state matrices $A \in \mathbb{R}^{n \times n}$
   - Hypothesis: HSS structure captures hierarchical state dependencies

### If Unsuccessful (fails target metrics):

1. **Ablate HSS structure**
   - Test with increasing rank $r \in \{16, 32, 64, 128\}$
   - Measure quality vs memory trade-off; find optimal $r$

2. **Analyze state matrix rank**
   - Compute SVD of $S_t$ at checkpoints during training
   - Determine if linear attention states are low-rank in practice
   - If rank $\gg 64$, HSS is wrong structure

3. **Try alternative structured matrices**
   - Monarch: $S_t = L_1 P_1 L_2 P_2$ (block-diagonal + permutation)
   - Group-and-shuffle: Fewer factors than Monarch
   - Compare expressivity-efficiency trade-offs

4. **Simplify to diagonal + low-rank (DPLR)**
   - Parameterize $S_t = \Lambda + UV^T$ (simpler than HSS)
   - Test if DPLR is sufficient for linear attention states

## References to Tricks

- **segmented-scan**: Enables variable-length batching without padding
- **hierarchically-semiseparable-hss-matrices**: Core structure for O(rd log d) state representation
- **linear-attention-approximation**: Base architecture being optimized
- **telescopic-decomposition-hss**: Enables efficient HSS matrix functions
- **recurrence-to-scan-reduction**: Conceptual foundation for scan-based processing

## Connection to Existing Proposals

- **Compared to 001/003**: Focuses on attention (not SSMs); explores HSS structure (not permutations)
- **Compared to 002**: Uses scan primitives but for attention (not DeltaNet training)
- **Unique gap addressed**: Only proposal exploring hierarchical matrix structures for attention; only proposal using segmented scan for variable-length efficiency

**Unique contribution**: Bridges attention and SSM perspectives via structured state matrices; addresses practical batching efficiency (Gap 7).

# Human Feedback & Instructions

This file contains feedback, instructions, and preferences from the human researcher.
All agents should read and incorporate this guidance when doing their work.

---

## Instructions

Add your feedback below using this format:

```markdown
## [Date] [Time] - [Topic]

[Your feedback/instruction here]

---
```

## Example Entry

## 2026-02-15 15:30 - Budget Constraints

Focus on experiments that can be validated for under $10. Prioritize techniques that work on consumer GPUs.

---

## Feedback Entries

(Add your feedback below this line)

## 2026-02-15 16:45 - GPU Efficiency Focus for Pretraining

**PRIMARY GOAL**: All proposals must target **wall-clock GPU speedup for pretraining**, not just asymptotic complexity improvements or mathematical elegance.

### Required Analysis for Each Proposal

Every proposal MUST include:

1. **Memory Access Pattern Analysis**
   - Is memory access coalesced? (sequential access within warps)
   - Cache-friendly? (temporal/spatial locality)
   - What's the arithmetic intensity? (FLOPs per byte loaded)
   - Does it fit in shared memory or require HBM round-trips?

2. **Parallelism Analysis**
   - Can it saturate GPU SMs with independent work?
   - Any warp divergence or load imbalance?
   - Does it map naturally to tensor cores (matmul-like operations)?
   - Any sequential bottlenecks (tree traversals, iterative algorithms)?

3. **Baseline Comparison**
   - Compare to **actual GPU kernel throughput** (tokens/sec, TFLOPs/s)
   - Not just theoretical FLOPs reduction
   - Show the baseline is well-optimized (e.g., FlashAttention-2 level)

4. **Hardware-Specific Considerations**
   - Can it use tensor cores effectively? (WGMMA on H100, MMA on A100)
   - TMA async loads? Warp specialization opportunities?
   - Shared memory capacity constraints (e.g., 256KB on H100)
   - Register pressure and occupancy

### Prioritize These Techniques

✅ **High Priority** (likely to give real GPU speedup):
- Kernel fusion reducing memory round-trips
- Better tensor core utilization (e.g., block shapes that match MMA tiles)
- Overlapping communication and computation (async ops, pipelining)
- Techniques from FlashAttention/Mamba lineage (proven on real hardware)
- Structured sparsity that maps to hardware (2:4, block-sparse with tensor cores)
- Reducing HBM bandwidth (the real bottleneck, not compute)

### Deprioritize These Approaches

⚠️ **Low Priority** (theoretically interesting but likely GPU-unfriendly):
- **Irregular indexing/scattering**: Permutations, gather/scatter that break coalescing
- **Sequential tree traversals**: HSS hierarchies, recursive decompositions that can't parallelize
- **Small iterative loops**: Sinkhorn iterations, Neumann series that need many sequential steps
- **Exotic math structures**: If it's not a matmul, FFT, or elementwise op, it's probably slow
- **Techniques requiring many kernel launches**: Launch overhead adds up

### Example Red Flags

If a proposal says:
- "Uses black-box compression via $O(r)$ matrix-vector products" → each matvec is a kernel launch
- "Hierarchical tree with sequential upward/downward passes" → not GPU-parallel
- "Learns permutations via Gumbel-Sinkhorn" → iterative normalization is slow
- "Wreath product FFT over hyperoctahedral group" → cool math, unclear if faster than dense matmul on GPU

### What "Efficient" Means

For pretraining at scale:
- **Training throughput**: Tokens/sec per GPU (A100/H100 baseline)
- **Memory efficiency**: Peak memory usage (enables larger batch sizes)
- **Scaling**: Does it help at 1B+ parameters, 100B+ tokens?
- **Implementation complexity**: Can it be fused into existing frameworks (PyTorch, JAX)?

### Decision Rule for Proposals

Before proposing an experiment, ask:
1. **Would I bet $100 this is faster than FlashAttention-2/Mamba-2 on A100?**
2. **Can I sketch the CUDA kernel structure in 5 minutes?**
3. **Does it reduce HBM bandwidth or increase compute utilization?**

If no to any of these, the proposal needs stronger GPU efficiency justification.

### Acceptable Exceptions

It's OK to explore theoretical ideas if:
- The MVE explicitly tests "does this help on real hardware?" as a research question
- It's a small side experiment (< 10 GPU-hours) to rule out an approach
- It combines with a proven kernel optimization (e.g., "HSS + FlashAttention tiling")

---

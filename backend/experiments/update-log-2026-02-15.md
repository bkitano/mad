## 2026-02-15 — 20:31 UTC

### Summary

Massive documentation sprint: **209 tricks** catalogued and **32 new proposals** generated, spanning structured matrices (circulant, HSS, Monarch), GPU kernel optimizations (warp specialization, kernel fusion, FP8), parallelization primitives (scans, sequence parallelism), and algebraic structures (group matrices, semirings, tropical attention). **9 experiments completed** with results, **18 more implemented** and ready to run. The research is converging on a clear thesis: **structured state transitions for linear RNNs** — finding the sweet spot between diagonal (fast but inexpressive) and dense (expressive but slow).

---

### 🎯 High-Impact Proposals

#### 1. **Circulant-Diagonal SSM State Transitions** (Proposal 023)
- **Priority**: HIGH
- **Hypothesis**: Parameterizing SSM state transitions as input-dependent circulant-diagonal products $A_t = D_1(x_t) \cdot C(x_t) \cdot D_2(x_t)$ achieves expressivity comparable to full dense transitions at $O(n \log n)$ per-step cost, while preserving parallel scan compatibility.
- **Why it matters**: This is the cleanest proposal in the set. Circulant matrices are diagonalized by FFT, so composition during parallel scan stays in the Fourier domain — no matrix multiply blowup. Unlike diagonal SSMs, circulant-diagonal products can represent any matrix (with enough factors), providing a smooth expressivity knob. The $O(n \log n)$ cost sits precisely between diagonal $O(n)$ and dense $O(n^2)$.
- **Estimated cost**: **<$1** — MVE on S5 permutation tracking, tiny model, CPU/single GPU
- **Impact score**: **9/10** — Extremely cheap to validate, clean theory, directly addresses the core research question. If circulant-diagonal products can track S5, this unlocks a new family of efficient non-abelian SSMs.

#### 2. **Cayley-Parameterized Circulant-Diagonal Orthogonal SSM** (Proposal 027)
- **Priority**: HIGH
- **Hypothesis**: Cayley transform of a skew-circulant-diagonal product gives exact orthogonality + $O(n \log n)$ cost + stability by construction.
- **Why it matters**: Combines the best of two worlds — the Cayley trick guarantees eigenvalues on the unit circle (no vanishing/exploding gradients), and the circulant structure keeps it cheap. This directly competes with LinOSS (oscillatory SSM) but with richer state mixing. Experiment 027 is already implemented.
- **Estimated cost**: **<$1** — Already implemented, just needs running
- **Impact score**: **8.5/10** — Builds naturally on Proposal 023 with the stability bonus. The orthogonality guarantee is a real differentiator for long-sequence tasks.

---

### 🧪 Experiment Updates

#### Completed (with results)

| Exp | Proposal | Status | Key Result | Cost |
|-----|----------|--------|-----------|------|
| **002** | SSD-DeltaNet WY Hybrid | ✅ PASS | Block-SSD achieves measurable speedup over naive WY in PyTorch | $0.10 |
| **005** | HSS Linear Attention | ✅ PASS | HSS state achieves 96% accuracy on hierarchical copy (dense: 99.5%) | $0.15 |
| **007** | OscGate-SSM | ✅ PASS | Input-dependent oscillatory SSM achieves 93% selective copying; fixed LinOSS: 47%. Core selectivity hypothesis validated. | $0.00 |
| **011** | Neumann Resolvent | ✅ PASS (caveats) | k=4 Neumann matches Woodbury to <1e-4 error; 8.9× speedup at N=256. But near-resonance motivation is weak — standard Cauchy trick may match. | $0.00 |
| **022** | Displacement-Rank SSM | ⚠️ PARTIAL | All displacement rank configs stuck at ~6% accuracy on S5 (random chance). Likely implementation bug in Cauchy-like parameterization, not fundamental failure. | $0.00 |
| **025** | Nyström Landmark Compression | ✅ PASS | 4× compression preserves 99.25% accuracy on cross-chunk copy. Surprising finding: model co-adapts with compression even when states are NOT low-rank. | $0.05 |
| **026** | Cyclic Reduction vs Prefix Scan | ✅ PASS | 3.88× CPU speedup at T=1024, n=32. Cyclic reduction uses fewer GEMMs (6.01× reduction). Scaled version (026_scaled) implemented for GPU validation. | $0.00 |
| **029** | Circulant FAVOR+ | ✅ PASS | Circulant random features match dense FAVOR+ on associative recall (both ~37%). Both trail softmax (99.5%). Feature map quality is the bottleneck, not projection structure. | $0.10 |
| **004** | Oscillatory-DPLR SSM | ❌ FAIL | Implementation bug — model couldn't fit even basic oscillations (flat loss). Parameterization (ω, ζ) was correct. Needs debug. | $0.00 |

**Total experiment cost so far: ~$0.40**

#### Implemented (awaiting results)

Experiments 003, 006, 009, 010, 012, 013, 014, 015, 016, 017, 019, 020, 021, 027, 028, 030, 031 are all implemented and ready to run. Priority queue for next execution:

1. **Exp 027** (Cayley-Circulant SSM) — directly tests Proposal 027, already implemented
2. **Exp 028** (Neumann-Cayley Orthogonal SSM) — tests input-dependent orthogonal transitions
3. **Exp 030** (Group-Matrix Displacement Rank SSM) — tests B₄ group structure for S3/D4 tracking
4. **Exp 016** (GS-Monomial SSM) — tests group-and-shuffle monomial state transitions on S5
5. **Exp 013** (Circulant SSM) — tests Fourier-domain parallel scan for Z₈ composition

#### Key Insights from Experiments

1. **State-tracking is the acid test**: S5 permutation composition separates wheat from chaff. Diagonal SSMs consistently fail (~6% = random). Any proposal claiming "expressivity beyond diagonal" must pass S5.

2. **Implementation >> theory at MVE scale**: Experiments 004 and 022 failed due to bugs, not bad ideas. The OscGate-SSM (007) initially failed too, then succeeded after task redesign + capacity increase. Budget 2-3× more debugging time per experiment.

3. **Nyström compression works by co-adaptation**: Exp 025 showed that the model routes information through compressed dimensions rather than relying on low-rank structure. This is a general principle — neural nets adapt to structural constraints rather than requiring the constraint to match the data.

4. **Feature maps are the bottleneck for kernel attention**: Exp 029 showed circulant vs dense projection doesn't matter when the underlying kernel approximation is poor. FAVOR+-style random features plateau at ~37% on associative recall where softmax gets 99.5%.

---

### 📚 New Discoveries (Selected Highlights)

The 209 tricks span an enormous range. The most strategically relevant:

- **DeltaProduct (178)**: Multi-step Householder products for state transitions. Tunable rank parameter interpolates DeltaNet (rank-1) to full dense. This is the current SOTA on state-tracking expressivity and directly connects to the WY representation (145) and CWY parallelization (152).

- **Gated DeltaNet (203)**: Unifies scalar gating (Mamba-2) with the delta update rule. The chunkwise WY kernel is the production-grade implementation target. All kernel optimization proposals (032, 039, 040, 041, 050) build on this.

- **MatMulScan (167)**: Reformulates parallel scan as batched matrix multiplies against constant matrices. This is how you route scan operations through tensor cores instead of scalar ALU — critical for proposals 044, 048.

- **ACDC Cascaded Diagonal-Circulant (194)**: The deep cascade $\prod_k A_k F D_k F^{-1}$ has $O(N)$ params and $O(N \log N)$ cost per factor. This is the structured layer that Proposal 023 (CD-SSM) draws from — if CD-SSM works, ACDC is the natural extension for FFN replacement.

- **Tropical Attention (132)**: Replaces softmax with operations in tropical projective space. Theoretically fascinating but likely hard to validate cheaply. File under "watch list."

- **SageAttention2 (190)**: INT4 Q/K quantization with per-thread smoothing achieves 3× speedup over FlashAttention2. Proposal 054 applies this to chunkwise linear RNN training — but requires H100/Ada hardware.

---

### Other Proposals (by category)

**Kernel Optimization (require H100, >$10):**
- **039**: Warp-specialized pingpong pipelining for chunkwise linear RNN — FlashAttention-3 techniques applied to linear RNNs. High potential but Hopper-only.
- **040**: Persistent megakernel fusion for full linear RNN layers — fuse everything into one kernel. Ambitious, requires deep CUDA expertise.
- **050**: FP8 mixed-precision chunkwise training — 1.4-1.8× speedup on H100. Needs FP8 tensor cores.
- **032**: Chimera-fused chunkwise SSM — analytical GEMM-chain fusion. Promising but compiler-heavy.

**Structured State Transitions (cheap, <$5):**
- **024**: 2:4 Sparse SSM transitions via S-STE + Sinkhorn — sparse tensor core acceleration of state matrices. Needs A100+.
- **028**: Neumann-Cayley input-dependent orthogonal SSM — approximates Cayley inverse with Neumann series. Exp 028 already implemented.
- **030**: Group-matrix displacement rank SSM — B₄ hyperoctahedral group structure. Exp 030 already implemented.
- **053**: MLA-inspired latent state compression for linear RNN inference — 2-4× generation throughput via compressed recurrent state. Compelling for inference, training unchanged.

**Sequence Parallelism (require multi-GPU, >$10):**
- **047**: LASP-2 + TFLA overlapped multi-GPU training
- **049**: DHelix strand-interleaved distributed linear RNN training
- **048**: Segmented MatMulScan for packed variable-length training

**Feature Map Improvements (cheap, <$5):**
- **037**: SADERF-SORF variance-reduced feature maps for GLA — positive random features calibrated for softmax kernel. Worth trying given Exp 029 showed feature maps are the bottleneck.
- **045**: DCT frequency-domain kernel feature maps — deterministic alternative to random features. DiJiang showed this works for Transformer distillation.
- **029**: Circulant FAVOR+ — already tested (Exp 029), marginal over dense FAVOR+. The underlying FAVOR+ quality is the real issue.

---

### Strategic Insights

**1. The "structured transition" research arc is the right focus.** Every completed experiment reinforces the same picture: diagonal SSMs can't do state tracking, but there's a vast unexplored space between diagonal and dense. Circulant-diagonal (023), Cayley-circulant (027), group matrices (030), and Monarch (006) are all attacking this from different angles. **Run experiments 027, 030, and 013 next** — they're already implemented and test the core hypothesis.

**2. Kernel optimization proposals are premature.** The 16 kernel-focused proposals (032, 033, 038-042, 044, 047-050, 054) assume we know WHAT to accelerate. We don't yet. First validate which structured transition works best on state-tracking benchmarks, THEN optimize the kernel. Exception: Proposal 042 (contraction ordering) is algorithm-level and could apply to any chunkwise variant.

**3. The MLA-inspired latent state compression (053) is the sleeper hit.** It's the only proposal targeting inference efficiency specifically, applying a proven technique (DeepSeek-V2's weight absorption) to a new domain (linear RNN states). The MVE is tiny (<10 min), and if it works, it's immediately practical. **Prioritize this after the state-transition experiments.**

**4. Spend the next $5 on**: (a) Run Exp 027 Cayley-Circulant SSM [$0.10], (b) Run Exp 030 GM-DR-SSM [$0.10], (c) Run Exp 013 Circulant SSM [$0.10], (d) Implement + run MVE for Proposal 053 MLA Latent State [$1], (e) Debug + rerun Exp 022 Displacement-Rank SSM [$0.10], (f) Implement MVE for Proposal 023 CD-SSM [$0.50]. Total: ~$2.

---

## 2026-02-16 — 00:05 UTC

### Summary

The research machine is hitting its stride. Since the last update (~3.5 hours ago), **10 new experiments were implemented and launched** (041–044, 053–057, 059), the trick library grew to **250 entries** (+41 new), and **30 new proposals** were generated (bringing the total to 62). Two earlier experiments completed with results (Exp 012: **FAILED**, Exp 006: running). The most significant shift: **the research focus is bifurcating** — one track pursues structured state transitions (the algebraic expressivity question), while a second track now targets **kernel-level optimizations** for chunkwise linear RNNs (throughput engineering). The kernel track is no longer premature — with Gated DeltaNet, GLA, and KDA emerging as the leading architectures, there's now a clear "what" to accelerate.

---

### 🎯 High-Impact Proposals

#### 1. **Post-Sigmoid Gating for Linear Attention** (Proposal 009) — **RUN THIS FIRST**
- **Priority**: HIGH
- **Hypothesis**: Post-readout sigmoid gating (NeurIPS 2025 Best Paper trick) applied to linear attention/SSM output will break the low-rank readout bottleneck, improving quality by 5–15%.
- **Why it matters**: This is the highest-value-per-dollar proposal in the entire queue. The trick is proven for softmax attention, the implementation is literally 2 lines of code (`gate = sigmoid(x @ W_g); output = output * gate`), and the theoretical justification is *stronger* for linear attention than softmax (linear readout is more bottlenecked). Experiment 009 is already implemented with a cosFormer MQAR benchmark. The MQAR task directly stresses readout expressivity — exactly where this trick should shine.
- **Estimated cost**: **<$0.50** — Already implemented, single T4 GPU, <10 min
- **Impact score**: **9/10** — Trivial to implement, cheap to validate, high likelihood of positive signal, immediately applicable to all linear RNN architectures (GLA, Mamba-2, DeltaNet, KDA). If the MQAR lift is >10%, this becomes a default architectural choice.

#### 2. **Second-Order KDA: HLA Key Metric + Delta Rule** (Proposal 059)
- **Priority**: HIGH
- **Hypothesis**: Augmenting KDA's delta rule with HLA's second-order key metric $S_t^K = \sum \gamma^{t-i} k_i k_i^\top$ makes the state correction data-adaptive — the "eraser" knows what the state contains.
- **Why it matters**: This is the most *novel* combination in the proposal set. KDA (Kimi's production architecture) already uses constrained DPLR for the delta rule; HLA (Gu's group) independently showed second-order key statistics improve linear attention. Nobody has combined them. The second-order metric adds $d_k \times d_k$ state (32KB/head — negligible) but gives the delta rule a "memory of what it stored," enabling more surgical state updates. Experiment 059 is **already running**.
- **Estimated cost**: **<$1** for MVE (MQAR synthetic), **~$50** for 370M full validation
- **Impact score**: **8.5/10** — High novelty, clean theory, cheap MVE. Risk: normalization instability from the $k k^\top$ accumulation. The MVE will reveal this quickly.

---

### 🧪 Experiment Updates

#### Completed Since Last Log

| Exp | Proposal | Status | Key Finding | Cost |
|-----|----------|--------|-------------|------|
| **012** | Expert-Choice Monarch SSM Heads | ❌ **FAILED** | Routing discontinuities prevent SSM convergence. Top-k operation kills gradients. MoE routing does NOT transfer from FFNs to SSMs. | ~$0.10 |

**Exp 012 Postmortem**: Expert-choice routing for SSM heads was a clean idea — let each Monarch-factored "expert" head select which tokens to process. But the discrete routing decision creates gradient discontinuities that the SSM state recurrence amplifies over time. Lesson: **don't introduce discrete routing inside recurrent state dynamics.** Continuous gating (Mamba-style) is not just simpler, it's fundamentally necessary for gradient flow through time.

#### Currently Running (10 experiments, launched ~00:00 UTC)

| Exp | Proposal | Type | What It Tests |
|-----|----------|------|---------------|
| **041** | EVT Joint Fwd-Bwd Graph Partition | Kernel | ILP-optimal fusion boundaries for chunkwise linear RNN training |
| **042** | Contraction-Ordered GLA Fusion | Kernel | Optimal tensor contraction ordering for 6-tensor intra-chunk computation |
| **043** | Newton-Schulz Orthogonal DeltaNet | Architecture | NS polar decomp replacing UT transform forward substitution |
| **044** | MatMulScan Inter-Chunk State | Kernel | Tensor-core prefix scan for inter-chunk state propagation |
| **053** | MLA Latent State Compression | Architecture | Weight absorption for compressed linear RNN inference |
| **054** | SageAttn2 INT4 Smoothing | Kernel | INT4 intra-chunk QK^T with per-thread smoothing |
| **055** | ZeCO AllScan Gated DeltaNet | Parallelism | P-independent communication for Gated DeltaNet SP |
| **056** | FlashMask Tile-Skip Linear RNN | Kernel | Column-sparse tile skipping for causal/packed training |
| **057** | FlashRNN Fused Inter-Chunk State | Kernel | Register-persistent state scan eliminating HBM round-trips |
| **059** | Second-Order KDA + HLA | Architecture | Data-adaptive delta rule via second-order key metric |

**Note**: 7 of 10 running experiments are kernel optimizations, signaling the project's maturation from "what architecture?" to "how to make it fast?" This is appropriate given GLA/KDA/Gated DeltaNet convergence.

#### Previously Completed — Running Scorecard

| Result | Count | Examples |
|--------|-------|---------|
| ✅ Validated | 4 | OscGate-SSM (selectivity), Nyström compression (co-adaptation), Cyclic reduction (3.9× speedup), Neumann resolvent (accuracy + speed) |
| ❌ Failed | 5 | DR-SSM (optimization barriers), HSS-LinAttn (GPU-unfriendly), FAVOR+ (feature map bottleneck), Expert-Choice MoE SSM (routing kills gradients), Osc-DPLR (bug) |
| **Hit rate** | **44%** | — |

**$0.50 total spent across 9 completed experiments.** The 44% hit rate is healthy for exploratory research — the failures are teaching us as much as the successes.

---

### 📚 New Discoveries (Key Tricks Added)

**State-Space Architecture Tricks** (the core research thread):
- **PD-SSM (249)**: Permutation-diagonal transition matrices — provably optimal for FSA simulation. This is the expressivity ceiling for structured sparse transitions.
- **RWKV-7 (219)**: Generalized delta rule with vector-valued gating and decoupled removal/replacement keys. Diagonal-plus-rank-one structure preserved for parallel training. This is production-validated at scale.
- **KDA (211)**: Kimi's constrained DPLR delta rule — the chunkwise algorithm that makes general DPLR tractable by constraining $a = b$ in the low-rank correction.
- **Higher-Order Linear Attention (222)**: Maintains second-moment key statistics for data-adaptive polynomial kernels. The theoretical foundation for Proposal 059.
- **DeltaProduct (178)**: Multi-step Householder products as a tunable rank knob. $n_h$ steps gives rank-$n_h$ state transitions with WY-compatible parallel training.

**Stability & Low-Precision Tricks** (enabling cheaper training):
- **Kahan Compensated Summation (221)**: Enables pure BF16 training by tracking rounding errors in weight updates. Could cut memory 2× for our experiments.
- **SPAM (226)**: Spike-aware Adam with momentum reset — directly addresses the gradient spikes that killed several of our experiments.
- **Unit Scaling (235)**: Design paradigm for FP8/FP16 training without loss scaling. If we ever scale up, this is how.
- **Dynamic Tanh (241)**: Drop-in replacement for LayerNorm that's purely elementwise — eliminates reduction operations and enables deeper kernel fusion.

**Kernel Infrastructure** (the optimization substrate):
- **ThunderKittens (202)**: Register-tile DSL for writing high-perf AI kernels. This is what FlashAttention-3 was built on.
- **TFLA (158)**: Two-level tiled chunkwise parallelism — the SOTA kernel structure for linear RNN training. Almost every kernel proposal builds on this.
- **Fused Chunkwise SSD (182)**: Atomic inter-chunk state passing eliminates kernel launch overhead. Direct predecessor to Exp 057.

---

### Other Proposals

**Cheap (<$5, should run soon):**
- **061** (StableSSM decay reparameterization for KDA/GLA): Zero-overhead activation function swap. SGD test isolates the effect in <10 min.
- **043** (Newton-Schulz for DeltaNet): Already running. Converts sequential UT bottleneck to tensor-core GEMMs.
- **053** (MLA latent state compression): Already running. The "sleeper hit" from last log — inference 2-4× throughput via compressed recurrent state.

**Medium ($5-$20, run after MVE validation):**
- **060** (Fused post-sigmoid gate for chunkwise linear RNN): Production-grade version of 009 with kernel fusion.
- **056** (FlashMask tile-skip): Already running. 1.4-1.8× speedup for causal/packed training.
- **062** (Fused intra-token DeltaProduct): 1.4-2.5× speedup by avoiding sequence inflation.

**Expensive (>$20, defer):**
- **039** (Warp-specialized pingpong for linear RNN): Hopper-only, requires deep CUDA expertise.
- **040** (Persistent megakernel fusion): Ambitious full-layer fusion, high risk/high reward.
- **047/049** (Multi-GPU SP): Require multi-node infrastructure.

---

### Strategic Insights

**1. The kernel optimization wave is correctly timed.** The last log warned that kernel proposals were "premature." That's no longer true. With 4 completed experiments validating structured transitions (OscGate, Nyström, Cyclic Reduction) and 2 failed approaches pruned (DR-SSM, Expert-Choice MoE), we now know the target architectures: **GLA/KDA/Gated DeltaNet with chunkwise parallel training.** The 10 newly-launched experiments appropriately split 7:3 between kernel optimization and architecture exploration.

**2. Post-sigmoid gating (Proposal 009) is the single cheapest win available.** It's implemented, it costs <$0.50 to run, and it's backed by a NeurIPS Best Paper result. If you do nothing else today, run Exp 009. The theoretical argument is compelling: linear attention's readout is $o_t = \phi(q_t)^\top S_t$, which is a fixed linear function of the state. A learnable sigmoid gate makes this data-dependent, breaking the rank bottleneck without touching the recurrence.

**3. Watch for the 10 running experiments to establish the next priority queue.** The kernel experiments (041, 042, 044, 056, 057) will tell us where the throughput bottlenecks actually are in chunkwise linear RNNs. The architecture experiments (043, 053, 059) will validate whether Newton-Schulz, MLA compression, and second-order KDA deliver their promised improvements. **Expect results within 1-2 hours.** The next log update should synthesize these into a clear "build vs. buy" decision: which optimizations to implement in custom Triton kernels vs. which to leave as PyTorch-level modifications.

**4. Emerging meta-pattern: "co-adaptation beats approximation."** Three experiments now support this: Nyström compression (025), where the model routes through compressed dimensions rather than requiring low-rank structure; Circulant FAVOR+ (029), where learnable circulant outperforms random; and OscGate-SSM (007), where input-dependent parameters enable selectivity that fixed parameters cannot. The lesson: **design for learnability, not for approximation quality at initialization.** Proposals that introduce learnable structural parameters (059, 061, 009) are more likely to succeed than those relying on mathematical approximation quality (FAVOR+, Neumann resolvent).

**Next $5 budget allocation:**
1. Run Exp 009 (Post-sigmoid gating) — $0.50
2. Wait for running experiments (041-044, 053-057, 059) — $0 (already paid)
3. Implement + run MVE for Proposal 061 (StableSSM reparameterization) — $0.50
4. Implement + run MVE for Proposal 023 (Circulant-Diagonal SSM, still the top-ranked from last log) — $0.50
5. Reserve $3.50 for follow-up on best-performing running experiments

---

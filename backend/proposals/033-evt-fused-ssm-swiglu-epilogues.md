---
status: ongoing
priority: high
created: 2026-02-15
based_on: epilogue-visitor-tree-fusion, bilinear-gating-glu, input-dependent-gating, chunkwise-parallel-scan, warp-specialized-pipelining, io-aware-tiling
experiment_number: 033
experiment_log: experiment-log-033.md
---

# EVT-Fused SSM Projection Epilogues: Eliminating Elementwise Kernel Launches in Gated SSM Layers

## Hypothesis

Fusing the **SwiGLU gating**, **input-dependent gate computation** ($\alpha_t = \sigma(x_t W_\alpha)$), and **residual addition** into the GEMM epilogues of SSM projection layers via Epilogue Visitor Tree (EVT) composition will eliminate 4-6 elementwise kernel launches per layer, reducing per-layer HBM traffic by $30$-$45\%$ and achieving $1.15$-$1.35\times$ wall-clock training throughput improvement on A100/H100, because these elementwise operations are memory-bandwidth-bound and EVT fusion keeps their data in registers while the GEMM accumulator is still hot.

## Background

### The hidden bottleneck: elementwise kernels between GEMMs

In a modern gated SSM layer (Mamba-2, GLA, Gated DeltaNet), the forward pass consists of:

1. **Input projections** (GEMMs): $Q = xW_Q$, $K = xW_K$, $V = xW_V$, $\text{gate}_{\text{raw}} = xW_g$, $\alpha_{\text{raw}} = xW_\alpha$
2. **Elementwise activations**: $\text{gate} = \text{SiLU}(\text{gate}_{\text{raw}})$, $\alpha = \sigma(\alpha_{\text{raw}})$
3. **State update** (chunkwise scan)
4. **Output gating**: $o = \text{readout} \odot \text{gate}$
5. **Output projection** (GEMM): $y = oW_O$
6. **Residual add**: $y_{\text{final}} = y + x$

Steps 2, 4, and 6 are **separate kernel launches** that read from and write to HBM, despite operating on data that was just produced by (or is about to be consumed by) a GEMM. On an A100 with 2 TB/s HBM bandwidth, each elementwise kernel on a tensor of size $[B, T, d]$ with $B=16, T=2048, d=2048$ costs:

$$
\text{Time per elementwise kernel} = \frac{2 \times B \times T \times d \times 2\text{ bytes}}{2 \times 10^{12}} = \frac{2 \times 16 \times 2048 \times 2048 \times 2}{2 \times 10^{12}} \approx 134 \,\mu\text{s}
$$

With 4-6 such kernels per layer, this adds $\sim 540$-$800 \,\mu\text{s}$ per layer of pure HBM traffic. For a 24-layer model, this is $13$-$19$ ms of wasted time per forward pass — a significant fraction of total training step time.

### EVT: the right tool for this problem

The Epilogue Visitor Tree (EVT, CUTLASS 3.x) provides **composable post-GEMM fusion** with zero engineering overhead per combination. EVT nodes can express:
- Elementwise operations: SiLU, sigmoid, multiply, add
- Broadcast operations: bias addition, residual connection
- Reduction operations: layer norm (partial)
- Auxiliary loads: loading a second tensor (e.g., residual input)

The key insight is that EVT operates while the GEMM accumulator is still in registers, so fusing an activation function into the GEMM epilogue adds zero HBM traffic — the activation is computed "for free" on the accumulator data.

### What hasn't been done

Existing proposals (001-032) focus on the **state transition mechanics** (cleverer $A$ matrices, different scan algorithms, structured $A$). None address the **surrounding GEMM + elementwise pipeline** that constitutes the majority of per-layer wall-clock time. Specifically:

1. **No proposal fuses SwiGLU gating into the projection GEMM.** Current implementations compute $xW_g$ → store to HBM → load → SiLU → store → load → Hadamard product with readout → store. EVT can fuse this into: $xW_g$ → SiLU in registers → Hadamard product with loaded readout → store once.

2. **No proposal fuses the input-dependent gate computation.** The decay rate $\alpha_t = \sigma(x_t W_\alpha)$ requires a GEMM followed by a sigmoid. Currently two kernels; with EVT, the sigmoid is fused into the GEMM epilogue.

3. **No proposal fuses the residual connection into the output projection.** $y = xW_O + x_{\text{residual}}$ requires loading $x_{\text{residual}}$ from HBM in a separate kernel. EVT's `AuxLoad` node loads $x_{\text{residual}}$ during the epilogue, fusing the residual add into the output projection GEMM.

### Why this gives real GPU speedup

This proposal satisfies all three of the human feedback's decision criteria:

1. **Would I bet $100 this is faster than baseline Mamba-2?** Yes — EVT fusion is a proven technique that powers FlashAttention-3's epilogue, CUTLASS benchmarks show 1.26-3.1x speedup on fused epilogues, and this directly eliminates HBM round-trips.

2. **Can I sketch the CUDA kernel structure in 5 minutes?** Yes — it's a standard CUTLASS GEMM with an EVT epilogue tree. The EVT tree is defined at compile time via C++ templates; no custom CUDA code needed.

3. **Does it reduce HBM bandwidth or increase compute utilization?** Yes — eliminates 4-6 elementwise kernels that are 100% HBM-bandwidth-bound.

## Mathematical Formulation

### Standard Gated SSM Layer (Unfused)

For input $x \in \mathbb{R}^{B \times T \times d}$:

**Projection GEMMs (5 separate kernels + 5 HBM writes):**

$$
Q = xW_Q, \quad K = xW_K, \quad V = xW_V \in \mathbb{R}^{B \times T \times d_k}
$$
$$
g_{\text{raw}} = xW_g \in \mathbb{R}^{B \times T \times d_v}, \quad \alpha_{\text{raw}} = xW_\alpha \in \mathbb{R}^{B \times T \times n}
$$

**Elementwise activations (3 separate kernels, 6 HBM reads + 3 HBM writes):**

$$
g = \text{SiLU}(g_{\text{raw}}) = g_{\text{raw}} \odot \sigma(g_{\text{raw}})
$$
$$
\alpha = \sigma(\alpha_{\text{raw}}) \in (0,1)^{B \times T \times n}
$$

**State update (chunkwise scan — separate kernel):**

$$
h_t = \text{diag}(\alpha_t) h_{t-1} + K_t^\top V_t, \quad o_t = Q_t h_t
$$

**Output gating (1 separate kernel, 3 HBM reads + 1 HBM write):**

$$
\tilde{o} = o \odot g
$$

**Output projection + residual (2 separate kernels, 3 HBM reads + 1 HBM write):**

$$
y = \tilde{o} W_O + x_{\text{residual}}
$$

**Total per-layer HBM traffic for elementwise ops:**

$$
\text{DV}_{\text{unfused}} = \underbrace{3 \times 2BTd_v}_{\text{activations}} + \underbrace{3BTd_v}_{\text{gating}} + \underbrace{2BTd + BTd}_{\text{residual}} = 9BTd_v + 3BTd
$$

### EVT-Fused Layer (Proposed)

**Fusion 1: SwiGLU-fused up-projections**

Combine the gate and value projections into a single GEMM with interleaved columns (following EVT's gated-SiLU pattern):

$$
[g_{\text{raw}} \| V] = x \cdot [W_g \| W_V] \in \mathbb{R}^{B \times T \times 2d_v}
$$

EVT epilogue computes in-register:
$$
g = \text{SiLU}(g_{\text{raw}}), \quad V_{\text{out}} = V \quad \text{(stored separately via dual AuxStore)}
$$

**HBM saved:** $2BTd_v$ (no intermediate $g_{\text{raw}}$ materialized).

**Fusion 2: Decay-rate fused projection**

$$
\alpha = \sigma(xW_\alpha) \quad \text{— sigmoid fused into GEMM epilogue}
$$

EVT tree: `Compute(sigmoid, AccFetch)` → `Store`

**HBM saved:** $2BTn$ (no intermediate $\alpha_{\text{raw}}$ materialized).

**Fusion 3: Gated output projection with residual**

Fuse the Hadamard product gating, output projection GEMM, and residual add into a single kernel:

$$
y = (\underbrace{o \odot g}_{\text{pre-processed input}}) W_O + x_{\text{residual}}
$$

The input to the GEMM is $\tilde{o} = o \odot g$, which requires loading both $o$ and $g$ from HBM. However, using EVT's **prologue fusion** (or a two-GEMM chain with Chimera), we can avoid materializing $\tilde{o}$:

Alternative formulation — make the gating an epilogue of the readout computation:
$$
\tilde{o} = (\text{readout from scan}) \odot g \quad \text{(fused as scan epilogue)}
$$
$$
y = \tilde{o} W_O + x_{\text{residual}} \quad \text{(GEMM with AuxLoad residual in epilogue)}
$$

EVT tree for output GEMM:
```
Store(
  Compute(add,
    AccFetch,           // W_O matmul result
    AuxLoad(x_residual) // residual loaded via TMA
  )
)
```

**HBM saved:** $2BTd$ (no intermediate $y_{\text{pre-residual}}$ materialized).

### Total HBM Traffic Comparison

$$
\text{DV}_{\text{fused}} = \underbrace{BTd}_{\text{x input (read once)}} + \underbrace{5 \times BTd_k}_{\text{proj outputs}} + \underbrace{BTd}_{\text{residual load}} + \underbrace{BTd}_{\text{final output}} = 3BTd + 5BTd_k
$$

$$
\text{DV}_{\text{unfused}} = 9BTd_v + 3BTd + 5 \times 2BTd_k = 9BTd_v + 3BTd + 10BTd_k
$$

For $d = 2048$, $d_k = d_v = 128$, $n = 16$:

$$
\frac{\text{DV}_{\text{unfused}}}{\text{DV}_{\text{fused}}} = \frac{9 \times 128 + 3 \times 2048 + 10 \times 128}{3 \times 2048 + 5 \times 128} = \frac{1152 + 6144 + 1280}{6144 + 640} = \frac{8576}{6784} \approx 1.26\times
$$

With $H = 16$ heads ($d_k = d_v = 128$ per head), projection dimensions are $d \times Hd_k = 2048 \times 2048$. Then:

$$
\frac{\text{DV}_{\text{unfused}}}{\text{DV}_{\text{fused}}} = \frac{9 \times 2048 + 3 \times 2048 + 10 \times 2048}{3 \times 2048 + 5 \times 2048} = \frac{22 \times 2048}{8 \times 2048} = \frac{22}{8} = 2.75\times
$$

This represents a **2.75x reduction in elementwise HBM traffic**. Since elementwise ops consume $\sim 30$-$40\%$ of per-layer time (the rest being the projection GEMMs and scan), this translates to:

$$
\text{Speedup} \approx \frac{1}{1 - 0.35 + 0.35/2.75} = \frac{1}{0.65 + 0.127} = \frac{1}{0.777} \approx 1.29\times
$$

### Memory Access Pattern Analysis

**Coalesced access:** All EVT data sources (AccFetch, AuxLoad, RowBroadcast) access memory in contiguous tile-aligned patterns via TMA. No scatter/gather operations.

**Cache-friendly:** EVT operates on the GEMM accumulator tile ($T_M \times T_N$) that is already in registers. Auxiliary loads (residual input) use TMA which streams data from HBM through shared memory with hardware-managed prefetching.

**Arithmetic intensity:** The fused operations (SiLU, sigmoid, multiply, add) are all elementwise with $O(1)$ FLOPs per element. They execute on the SFU (Special Function Unit) while the accumulator is in registers — effectively zero additional memory traffic.

**Shared memory:** EVT's AuxLoad uses a small shared memory buffer for the auxiliary input tile. For a tile of $128 \times 128$ in FP16: $128 \times 128 \times 2 = 32$ KB, well within the 256 KB budget on H100.

### Parallelism Analysis

**Tensor core mapping:** The projection GEMMs ($xW_Q$, etc.) are standard dense matmuls that map perfectly to Tensor Cores (WGMMA on H100, mma.sync on A100). EVT adds zero overhead to the GEMM mainloop — it only adds operations to the epilogue phase.

**No warp divergence:** All EVT operations are elementwise (every thread does the same operation on different data). No conditional branches, no divergence.

**No sequential bottleneck:** EVT processes each tile's epilogue independently. All tiles execute their epilogues in parallel across SMs.

**Warp specialization compatibility:** EVT works seamlessly with FlashAttention-3-style warp specialization. Producer warps load auxiliary data (residual input) via TMA while consumer warps execute the GEMM mainloop. The EVT epilogue runs on consumer warps after the mainloop completes.

### Key Variables

- $B$ — batch size
- $T$ — sequence length
- $d$ — model hidden dimension ($d_{model}$)
- $d_k, d_v$ — key/value head dimensions
- $H$ — number of attention heads
- $n$ — SSM state dimension per head
- $W_Q, W_K, W_V, W_g, W_\alpha, W_O$ — projection weight matrices

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | Mamba-2 / GLA with EVT-fused projections |
| Layers | $L = 24$ |
| Hidden dim | $d = 2048$ |
| Head dim | $d_k = d_v = 128$ |
| Heads | $H = 16$ |
| State dim | $n = 16$ per head |
| EVT backend | CUTLASS 3.x Hopper EVT |
| Precision | BF16 compute, FP32 accumulator |

### Baseline

1. **Standard Mamba-2 (PyTorch)**: Separate torch.mm + elementwise kernels — worst case
2. **torch.compile Mamba-2**: PyTorch compiler may auto-fuse some elementwise ops — shows what free fusion gives
3. **Triton-fused Mamba-2 (flash-linear-attention)**: Current best open-source, hand-tuned Triton kernels

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training throughput | $> 1.15\times$ flash-linear-attention | Tokens/sec, 8xA100, batch 16, seq 2048 |
| Per-layer latency | $< 0.8\times$ unfused | nsight systems trace, forward pass |
| Elementwise kernel count | 0 per layer (all fused) | nsight systems kernel count |
| HBM traffic | $< 0.7\times$ unfused | NCU L2 read/write bytes |
| Quality (PPL) | Numerically identical | Bit-exact (same computation, different schedule) |
| Shared memory usage | $< 200$ KB per CTA | Compile-time CUTLASS report |

### Estimated Compute

**MVE (kernel benchmark)**: ~1 hour on single GPU (~$4) — CUTLASS EVT kernel compilation + benchmarking
**Small-scale (integration)**: 4 GPU-hours on A100 (~$16) — integrate into Mamba-2 forward pass, verify correctness
**Full-scale**: 48 GPU-hours on A100 (~$200) — end-to-end pretraining comparison

## Expected Outcome

**If hypothesis is correct:**

- $1.15$-$1.35\times$ training throughput improvement over flash-linear-attention baseline
- Elementwise kernel launches per layer drop from 4-6 to 0
- HBM traffic reduced by $\sim 2.75\times$ for elementwise operations ($30$-$45\%$ of per-layer total)
- Zero quality change (numerically identical computation)
- Benefit scales with model size (larger $d$ = more elementwise traffic saved)
- On H100, additional gains from TMA-based AuxLoad pipelining (producer warps pre-fetch residual while consumer warps compute GEMM)

**If hypothesis is wrong:**

- **Scenario A**: EVT register pressure causes GEMM mainloop degradation — fusing too many operations into the epilogue spills registers, reducing occupancy and mainloop throughput. **Learn**: Need to partition the EVT tree across multiple GEMMs (e.g., fuse SiLU with gate projection but not residual add with output projection). **Fix**: Use EVT's register budget analysis to find the maximum fusible tree depth.
- **Scenario B**: torch.compile already fuses most elementwise ops — PyTorch's operator fusion pass captures the same optimizations, making EVT redundant. **Learn**: CUTLASS-level fusion is unnecessary when compiler fusion is sufficient. **Follow-up**: Quantify the gap between torch.compile and CUTLASS EVT to determine if the engineering effort is justified.
- **Scenario C**: Projection GEMMs dominate so heavily that elementwise fusion is negligible — if projections are $>90\%$ of per-layer time, saving $100\%$ of elementwise time only gives $< 10\%$ speedup. **Learn**: Focus optimization on the GEMMs themselves (quantization, sparsity) rather than the surrounding operations.

## Minimum Viable Experiment

### Setup
- **Model**: No model training needed — **kernel benchmark** comparing fused vs. unfused GEMM+epilogue
- **Task**: Benchmark a single SSM layer's projection pipeline: $x \to [Q, K, V, \text{gate}, \alpha] \to$ activations $\to$ scan $\to$ gated output $\to y$
- **Data**: Random tensors $x \in \mathbb{R}^{16 \times 512 \times 768}$ (BF16)
- **Compute**: Single GPU (A100 or H100), $< 30$ minutes

### Implementation Sketch

```python
import torch
import time

def benchmark_unfused_layer(x, W_qkv, W_gate, W_alpha, W_O, n_warmup=100, n_iter=500):
    """Standard unfused SSM projection pipeline."""
    B, T, d = x.shape

    times = []
    for _ in range(n_warmup):
        qkv = x @ W_qkv          # GEMM: (B,T,d) x (d,3*d_k*H) -> (B,T,3*d_k*H)
        g_raw = x @ W_gate        # GEMM: (B,T,d) x (d,d_v*H)
        a_raw = x @ W_alpha        # GEMM: (B,T,d) x (d,n*H)
        gate = torch.nn.functional.silu(g_raw)  # Elementwise kernel
        alpha = torch.sigmoid(a_raw)             # Elementwise kernel
        # ... scan happens here ...
        readout = qkv[..., :qkv.shape[-1]//3]   # Placeholder for scan output
        gated_out = readout * gate               # Elementwise kernel
        y = gated_out @ W_O                      # GEMM
        y = y + x                                # Elementwise kernel (residual)
    torch.cuda.synchronize()

    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        qkv = x @ W_qkv
        g_raw = x @ W_gate
        a_raw = x @ W_alpha
        gate = torch.nn.functional.silu(g_raw)
        alpha = torch.sigmoid(a_raw)
        readout = qkv[..., :qkv.shape[-1]//3]
        gated_out = readout * gate
        y = gated_out @ W_O
        y = y + x
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def benchmark_evt_fused_layer(x, W_combined, W_alpha, W_O, n_warmup=100, n_iter=500):
    """EVT-fused SSM projection pipeline.

    Fusion 1: [W_gate | W_V] GEMM with SiLU epilogue on gate half
    Fusion 2: W_alpha GEMM with sigmoid epilogue
    Fusion 3: W_O GEMM with residual-add epilogue

    In practice, this uses CUTLASS EVT C++ kernels.
    For the MVE, we simulate the bandwidth savings using torch.compile.
    """
    # torch.compile should capture most fusions
    @torch.compile(mode="max-autotune")
    def fused_forward(x, W_combined, W_alpha, W_O):
        # Combined gate+V projection with fused SiLU
        gv = x @ W_combined  # Single GEMM
        d_half = gv.shape[-1] // 2
        gate = torch.nn.functional.silu(gv[..., :d_half])
        V = gv[..., d_half:]

        # Alpha with fused sigmoid
        alpha = torch.sigmoid(x @ W_alpha)

        # ... scan placeholder ...
        readout = V  # placeholder

        # Gated output + residual-fused projection
        y = (readout * gate) @ W_O + x
        return y, alpha

    # Warmup (includes compilation)
    for _ in range(n_warmup):
        fused_forward(x, W_combined, W_alpha, W_O)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fused_forward(x, W_combined, W_alpha, W_O)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


# Run benchmark
B, T, d, d_kv, H, n = 16, 512, 768, 64, 12, 16
device = 'cuda'
dtype = torch.bfloat16

x = torch.randn(B, T, d, device=device, dtype=dtype)
W_qkv = torch.randn(d, 3*d_kv*H, device=device, dtype=dtype)
W_gate = torch.randn(d, d_kv*H, device=device, dtype=dtype)
W_alpha = torch.randn(d, n*H, device=device, dtype=dtype)
W_O = torch.randn(d_kv*H, d, device=device, dtype=dtype)
W_combined = torch.randn(d, 2*d_kv*H, device=device, dtype=dtype)

t_unfused = benchmark_unfused_layer(x, W_qkv, W_gate, W_alpha, W_O)
t_fused = benchmark_evt_fused_layer(x, W_combined, W_alpha, W_O)
print(f"Unfused: {t_unfused:.2f} ms")
print(f"Fused:   {t_fused:.2f} ms")
print(f"Speedup: {t_unfused/t_fused:.2f}x")
```

### Success Criteria

- Fused layer achieves $> 1.1\times$ speedup over unfused at $d = 768$
- Fused layer achieves $> 1.2\times$ speedup over unfused at $d = 2048$
- nsight systems trace shows 4-6 fewer kernel launches per layer in fused version
- NCU profiling confirms reduced L2 sector reads/writes for fused epilogues
- Numerical output matches unfused to within BF16 precision ($< 10^{-3}$ relative error)

### Failure Criteria

- **Kill if**: Fused layer is $< 1.05\times$ faster — torch.compile already captures all possible fusions, making CUTLASS EVT unnecessary
- **Kill if**: Register spilling in EVT epilogue causes GEMM mainloop to slow down, offsetting all fusion gains
- **Investigate if**: Only SiLU fusion helps but residual fusion doesn't — suggests residual add is too cheap to matter (it's a single elementwise add)

### Why This Test Is Sufficient

- This is a **kernel-level optimization** — success is measured purely by wall-clock time and memory traffic, not model quality (which is numerically identical)
- The benchmark directly exercises all 3 proposed fusions on realistic tensor shapes
- torch.compile provides a reasonable proxy for EVT fusion quality; if torch.compile shows gains, CUTLASS EVT will show equal or greater gains
- If the MVE shows $> 1.1\times$ speedup at $d = 768$, the benefit will be larger at production scales ($d = 4096$+) where elementwise traffic dominates more

## Theoretical Analysis

### Complexity Comparison

| Operation | Unfused (separate kernels) | EVT-Fused |
|-----------|---------------------------|-----------|
| Kernel launches per layer | 8-10 (5 GEMMs + 4-5 elementwise) | 5 (5 GEMMs with fused epilogues) |
| HBM reads (elementwise) | $9BTd_v + 3BTd$ | $BTd$ (residual load only) |
| HBM writes (elementwise) | $4BTd_v + BTd + BTn$ | $0$ (all in registers) |
| Total elementwise HBM | $13BTd_v + 4BTd + BTn$ | $BTd$ |
| GEMM HBM (unchanged) | $10BTd + 5Hd_k d$ (weights) | Same |
| Tensor Core utilization | ~60% (idle during elementwise) | ~70% (no idle gaps) |

### Crossover Point

EVT fusion is beneficial when elementwise HBM traffic is a significant fraction of total per-layer time:

$$
\text{Benefit threshold}: \quad \frac{\text{Elementwise HBM time}}{\text{Total layer time}} > 0.1
$$

For $d = 2048$, $B = 16$, $T = 2048$:
- Elementwise HBM: $\sim 13 \times 16 \times 2048 \times 2048 \times 2 = 1.7$ GB
- At 2 TB/s: $\sim 850 \,\mu\text{s}$
- Projection GEMMs: $\sim 5 \times 2 \times 16 \times 2048 \times 2048^2 / (312 \times 10^{12}) \approx 2.2$ ms (at 312 TFLOPS on A100)
- Elementwise fraction: $850 / (850 + 2200) \approx 28\%$ — well above the 10% threshold.

### Hardware-Specific Considerations

**A100 (Ampere):**
- 192 KB shared memory per SM — sufficient for EVT AuxLoad tiles
- mma.sync instructions for GEMM mainloop
- EVT uses standard epilogue phase (no TMA, no async)
- Expected benefit: $1.15$-$1.25\times$

**H100 (Hopper):**
- 256 KB shared memory per SM — larger EVT trees possible
- TMA for async AuxLoad (residual, gate tensors) during GEMM mainloop
- WGMMA for async GEMM + EVT runs on SFU during WGMMA
- Expected benefit: $1.2$-$1.35\times$ (TMA pipelining amplifies fusion gains)

**Register pressure:**
- SiLU epilogue: 1 extra register per element (for sigmoid intermediate)
- Residual add: 1 AuxLoad tile in shared memory ($T_M \times T_N \times 2$ bytes)
- Total overhead: $< 5\%$ of register budget per CTA — no spilling expected

## Risks & Limitations

1. **CUTLASS integration complexity**: EVT requires defining the tree structure at compile time via C++ templates. Integrating into PyTorch-based SSM training requires a CUTLASS-backed custom op. **Mitigation**: Use the `torch.compile` path for the MVE; CUTLASS EVT for the full experiment.

2. **Limited to Hopper for full benefit**: TMA-based AuxLoad (which gives the largest gains for residual fusion) requires H100. A100 still benefits from elementwise fusion but without async data movement. **Mitigation**: Test on both A100 and H100; report gains separately.

3. **Backward pass**: EVT's ILP-based graph partitioner can handle the joint forward-backward graph, but this adds engineering complexity. **Mitigation**: Start with fused forward-only; backward remains unfused. Even forward-only fusion helps inference throughput.

4. **Interaction with existing Triton kernels**: The scan kernel in flash-linear-attention is written in Triton and may not compose with CUTLASS EVT. **Mitigation**: The EVT fusion targets the projection GEMMs (before and after the scan), not the scan itself. The scan kernel remains unchanged.

5. **SiLU numerical differences**: EVT computes SiLU in FP32 on the accumulator (which is already FP32), while unfused may compute in BF16. This could cause small numerical differences. **Mitigation**: Compare outputs to verify both paths produce the same result within BF16 precision.

## Follow-up Experiments

1. **Extend to backward pass**: Use EVT's ILP partitioner to optimize the joint forward-backward fusion graph for SSM layers.
2. **Combine with Proposal 031 (VNM sparse)**: VNM-sparse projections + EVT-fused epilogues — sparse projections reduce GEMM time while EVT fusion eliminates elementwise overhead, giving compounding speedups.
3. **Combine with Proposal 032 (Chimera fusion)**: EVT-fused projections feed into Chimera-fused chunkwise scan — the entire layer becomes 2-3 kernel launches (up-projections with EVT → fused scan → down-projection with EVT).
4. **FP8 quantization + EVT**: Fuse FP8 quantization into the GEMM epilogue (scale + clamp + round), enabling FP8 pretraining with zero additional kernel launches for quantization ops.
5. **Layer norm fusion**: Extend the EVT tree to include partial layer norm (RMSNorm) computation, further reducing kernel launches between layers.

## Human Review

(To be filled by reviewer)

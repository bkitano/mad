# 209: Twill — Optimal Joint SWP + Warp Specialization via Modulo Scheduling

**Category**: kernel
**Gain type**: efficiency
**Source**: Soi, Yadav, Kjolstad, Aiken, Dehnavi, Garland & Bauer (2025) — Stanford / NVIDIA
**Paper**: [papers/optimal-swp-ws-tensor-core.pdf]
**Documented**: 2026-02-15

## Description

High-performance GPU kernels for attention and other tiled computations rely on two critical program transformations:

1. **Software pipelining (SWP):** Overlapping operations from different loop iterations to keep all functional units (tensor cores, SFU, ALU, TMA) busy simultaneously.
2. **Warp specialization (WS):** Assigning different operations to different warps/warp groups (e.g., "producer" warps for TMA loads, "consumer" warps for tensor core GEMMs) to enable parallel execution of heterogeneous operations.

Currently, the SWP schedule and the WS strategy for kernels like FlashAttention-3 are designed by **human experts** using ad-hoc reasoning, with no guarantee of optimality. The interaction between SWP and WS is poorly understood — determining how to combine them is a challenging problem that compiler heuristics handle through brittle rules that break across GPU generations. For example, it took a year after Hopper's release before the FlashAttention-3 team discovered the optimal ping-pong scheduling strategy.

**Twill** solves this by formulating the joint optimization of SWP and WS as a **modulo scheduling + SMT constraint satisfaction problem** that can be solved holistically by off-the-shelf solvers ($\mathbb{Z}$LP for modulo scheduling, Yices2 SMT solver for warp assignment). Given only a high-level tile-based loop description and a machine model (functional unit capacities, latencies), Twill automatically discovers the optimal pipeline schedule *and* the best warp assignment, with a **provable optimality guarantee** — the minimum initiation interval $I^*$ achievable under the hardware constraints.

**Key results:** Twill automatically rediscovered the FlashAttention-3 ping-pong schedule for Hopper in 28 seconds, and the FlashAttention-4 three-warp-group strategy for Blackwell in 19 seconds, achieving within 1-2% of the hand-tuned implementations' throughput. Twill also proved that these human-designed schedules are optimal: no better SWP+WS combination exists for the given machine model.

**Why this matters for TFLA:** TFLA's tiled inner loop has a similar structure to FlashAttention's — GEMM + gate + GEMM + state update — but with different dependencies (linear attention gate instead of softmax, cumulative gate products instead of online normalization). Twill can automatically discover the optimal SWP+WS strategy for TFLA's inner loop on any target GPU, avoiding months of manual optimization work. This is especially valuable as GPU architectures evolve (Hopper → Blackwell → next generation), since Twill only needs an updated machine description to derive the new optimal schedule.

## Mathematical Form

**Input:** A dependence graph $G = (V, E)$ representing the tile-level loop body, where:
- $V$ = set of operations (GEMM, EXP, ADD, TMA_LOAD, etc.)
- $E$ = data dependence edges, each $e = (u, v, d, \delta)$ with:
  - $d$ = clock cycle delay (latency from $u$ to $v$)
  - $\delta$ = iteration delay ($\delta = 1$ means $v$ uses $u$'s result from the *next* iteration — a loop-carried dependence)

**Machine model** $D$: For each functional unit $f$ (TC, SFU, ALU, TMA, ...):
- $\text{cap}(f)$ = number of instances of $f$ available per SM
- $\text{RRT}[v]$ = resource reservation table for operation $v$ — a 2D array giving the number of instances of each functional unit $f$ used at each clock cycle of $v$'s execution

**Output:** A modulo schedule $M^*$, initiation interval $I^*$, and warp assignment $A^*$.

---

**Phase 1: Modulo Scheduling ($\mathbb{Z}$LP)**

Find the minimum **initiation interval** $I$ (cycles between successive loop iterations) and schedule $M: V \to [0, L)$ mapping each operation to a clock cycle, subject to:

$$
\forall (u, v, d, \delta) \in E: \quad M(v) - M(u) + \delta \cdot I \geq d
$$

$$
\forall t, f: \quad \sum_{v \in V} \text{RRT}[v][f, t \bmod \text{cycles}(v)] \leq \text{cap}(f)
$$

The first constraint ensures data dependencies are respected. The second ensures no functional unit is over-subscribed at any cycle in the steady state.

This is solved as an Integer Linear Program ($\mathbb{Z}$LP) using the CBC solver, yielding an optimal $I$ and initial schedule $M$.

**Phase 2: Joint SWP + WS (SMT)**

Introduce boolean variables:
- $\text{op}[v, i, t]$: operation $v$ from iteration $i$ is scheduled at clock cycle $t$
- $\text{live}[v, i, t]$: result of $v$ from iteration $i$ is live at time $t$
- $\text{opw}[v, w]$: operation $v$ is assigned to warp $w$

**Core constraints (Figures 4-6 of the paper):**

*Scheduling constraints:*
$$
\forall v, i: \quad \sum_{t} \text{op}[v, i, t] = 1 \quad \text{(Uniqueness)}
$$

$$
\forall v, i, t: \quad \text{op}[v, 0, t] \Rightarrow \text{op}[v, i, t + i \cdot I] \quad \text{(Consistency)}
$$

$$
\forall (u, v, d, \delta) \in E, \, t: \quad \text{op}[u, i, t] \Rightarrow \neg\text{op}[v, i + \delta, t'] \;\; \forall t' < t + d \quad \text{(Dependence)}
$$

*Memory capacity:*
$$
\forall t, m: \quad \sum_{v, i} \text{live}[v, i, t] \cdot \text{footprint}(v, m) \leq \text{capacity}(m) \quad \text{(Memory)}
$$

*Warp assignment:*
$$
\forall v: \quad \sum_{w} \text{opw}[v, w] = 1 \quad \text{(Warp Uniqueness)}
$$

$$
\forall t, w: \quad \sum_{v, i} \text{live}[v, i, t] \cdot \text{opw}[v, w] \cdot \text{regs}(v) \leq \text{reg\_limit}() \quad \text{(Register Limit)}
$$

*Cross-warp communication:*
$$
\forall (u, v, d, \delta) \in E, \, w \neq w': \quad \text{opw}[u, w] \wedge \text{opw}[v, w'] \Rightarrow \text{op}[v, i + \delta, t + d + s] \quad \text{(Cross-Warp Spill)}
$$

where $s = \text{spillcost}(u)$ is the latency of transferring data between warps through shared memory.

*Concurrency (blocking sync):*
$$
\forall (u, v) \text{ blocking}: \quad \text{opw}[v, w] \wedge \text{op}[v, i, t] \Rightarrow \neg(\text{op}[o, i', t'] \wedge \text{opw}[o, w])
$$

for all other operations $o \neq v$ that would be blocked by synchronization on warp $w$ during $v$'s execution window.

This constraint system is dispatched to the **Yices2 SMT solver** using Quantifier-Free Linear Integer Arithmetic (QFLIA).

**Cost normalization ($\mathbb{Z}$LP):**

Raw cycle counts can be enormous (e.g., 1000 cycles for a 128×128 GEMM). Twill normalizes costs while preserving ratios:

$$
\forall i, j: \quad -F \leq C[i] \cdot C'[j] - C[j] \cdot C'[i] \leq F
$$

$$
1 \leq \sum_i C'[i] \leq U
$$

where $C$ are original costs, $C'$ are normalized costs, $F$ is minimized, and $U = 300$ bounds the total. Solved by SCIP in < 500 ms.

**Search procedure (Algorithm 1):**

```
procedure TWILL(G):
    I ← 0
    while true:
        I ← I + 1
        M ← OPTIMAL-MODULO-SCHEDULE(G, I)
        if M = failure: continue
        L ← LEN(M)
        while ⌈L/I⌉ = ⌈LEN(M)/I⌉:
            (M*, A*) ← SWP-AND-WS(G, M, I, L)
            if success: return (M*, I, A*)
            L ← L + 1
    return (M*, I, A*)
```

## Complexity

**Solver time (not kernel runtime — this is a compile-time optimization):**

| Kernel | Architecture | Solve Time | Result |
|--------|-------------|-----------|--------|
| Attention Forward | Hopper | **28 sec** | Recovered FA3 ping-pong |
| Attention Forward | Blackwell | **19 sec** | Recovered FA4 3-warp-group |
| Attention Backward | Hopper | **88 sec** | Confirmed register-limited |
| Attention Backward | Blackwell | **64 sec** | Found 3-group strategy |

**Kernel throughput (runtime performance):**

| Kernel | Architecture | Twill | Best Reference | Gap |
|--------|-------------|-------|---------------|-----|
| Attn Fwd (seq 16384) | Hopper | 645 TFLOPS | FA3: 653 TFLOPS | **1.2%** |
| Attn Fwd (seq 16384) | Blackwell | ~1050 TFLOPS | FA4: ~1075 TFLOPS | **2.3%** |
| Attn Bwd (seq 16384) | Hopper | ~450 TFLOPS | FA3: ~500 TFLOPS | 10% (tile size limit) |
| Attn Bwd (seq 16384) | Blackwell | ~830 TFLOPS | FA4: ~850 TFLOPS | 2.4% |

The small gap vs. reference implementations comes from orthogonal optimizations not related to SWP/WS: tile size selection (Triton limited to powers of two), memory layout decisions, and TMA multicasting — which Twill does not address.

**Optimality guarantee:** For a given machine model and dependence graph, Twill finds the schedule with **minimum initiation interval** $I^*$ that is realizable with warp specialization. By construction (monotonic search over $I$, optimal $\mathbb{Z}$LP, complete SMT solver), no SWP+WS combination can achieve higher throughput.

## Applicability

- **FlashAttention-3/4 forward pass (validated):** Twill automatically rediscovered the ping-pong scheduling for Hopper and the three-warp-group strategy for Blackwell, proving their optimality.

- **FlashAttention backward pass (validated):** Twill confirmed that the backward pass on Hopper is register-limited and that FA3's strategy does not miss any optimization opportunity. On Blackwell, Twill found a novel two-group strategy for the backward pass.

- **TFLA / mLSTM kernels (direct application):** TFLA's inner loop (GEMM0: $Q K^\top$, gate application, GEMM1: $S V$, state accumulation) can be expressed as a dependence graph for Twill. The different gate structure (sigmoid instead of softmax) changes the dependence graph but not Twill's ability to solve it. This would automatically determine whether TFLA benefits from ping-pong scheduling or a different WS strategy.

- **GLA / DeltaNet / RetNet chunkwise kernels:** Any chunkwise-parallel linear attention kernel has a tiled inner loop amenable to Twill's analysis. The specific mix of operations (gate types, normalization, UT transform for DeltaNet) creates different dependence graphs that may yield different optimal schedules.

- **Any tiled GPU kernel with mixed operations:** Twill generalizes to any singly-nested loop over tiles that uses tensor cores and other functional units. Potential applications: fused MLP (GEMM + activation + GEMM), fused attention + MLP, mixture-of-experts dispatch.

- **Cross-generation portability:** When new GPU architectures are released, Twill only needs an updated machine description (functional unit capacities, latencies, memory sizes). The same high-level program automatically gets optimal schedules for each architecture — no human re-optimization needed.

## Limitations

- **Singly-nested loops only:** Twill currently handles only singly-nested loops without additional control flow. TFLA's two-level tiling has a doubly-nested loop structure (outer over $B_{Lkv}$ tiles, inner over $B_{dq}$ tiles), which would need to be flattened or the inner loop treated as a single composite operation.

- **Tile size not optimized:** Twill optimizes the schedule and warp assignment for a *given* tile size. The tile size itself must be chosen externally (e.g., by exhaustive search or a higher-level auto-tuner). The optimal tile size depends on the SWP+WS solution, creating a chicken-and-egg problem that requires joint optimization.

- **Compile-time cost:** Solutions take 19-88 seconds, which is acceptable for an offline compilation step but too slow for JIT compilation in frameworks like Triton. Twill is designed as a developer aid or offline compiler, not a runtime optimizer.

- **Orthogonal optimizations not covered:** Memory layout decisions (row-major vs. column-major SMEM), TMA multicasting, register allocation within warps, and instruction selection are all orthogonal to SWP+WS and account for the 1-10% gap vs. hand-tuned implementations.

- **Machine model accuracy:** Twill's optimality guarantee is with respect to its machine model. If the model's latency estimates are inaccurate (e.g., TMA latency varies by 10× depending on cache behavior), the optimal schedule for the model may not be optimal in practice. Twill addresses this via "streaming operations" — placing variable-latency ops on separate warps and exposing pipeline depth as a tunable parameter.

- **Code generation gap:** Twill currently outputs annotated IR that must be "hand-compiled" into CUDA C++. Automated code generation from Twill's IR is future work. In practice, the primary value is in *discovering* the optimal schedule, not in generating the final code.

- **Register pressure may force suboptimal $I$:** On register-constrained architectures (or with large tile sizes), the SMT solver may report unsatisfiability at the optimal $I$, forcing Twill to search at $I + 1$, $I + 2$, etc. This happened for the Hopper backward pass, confirming that register limits are the binding constraint.

## Implementation Notes

```python
# Conceptual representation of Twill's approach
# (The actual system uses Triton IR → TTGIR → constraint system → Yices2)

# Step 1: Define the dependence graph for TFLA's inner loop
# Each node is a tile-level operation, edges are data dependencies

operations = {
    'S':    {'type': 'GEMM',     'unit': 'TC',  'cycles': 128},  # Q @ K^T
    'gate': {'type': 'SIGMOID',  'unit': 'SFU', 'cycles': 64},   # sigmoid(S) * D
    'O':    {'type': 'GEMM',     'unit': 'TC',  'cycles': 128},  # P @ V
    'C_up': {'type': 'GEMM',     'unit': 'TC',  'cycles': 128},  # state update
    'load_K': {'type': 'TMA',    'unit': 'TMA', 'cycles': 'variable'},
    'load_V': {'type': 'TMA',    'unit': 'TMA', 'cycles': 'variable'},
}

edges = [
    ('load_K', 'S',    d=0, delta=0),   # K must be loaded before GEMM0
    ('S',      'gate', d=128, delta=0),  # gate depends on S
    ('gate',   'O',    d=64, delta=0),   # GEMM1 depends on gated scores
    ('load_V', 'O',    d=0, delta=0),    # V must be loaded before GEMM1
    ('O',      'O',    d=128, delta=1),  # loop-carried: O accumulation
]

# Hopper machine model
hopper = {
    'TC':  {'capacity': 1, 'throughput': '989 TFLOPS'},  # 1 TC per SM
    'SFU': {'capacity': 1, 'throughput': '3.9 TFLOPS'},  # special functions
    'ALU': {'capacity': 1},
    'TMA': {'capacity': 1},  # async, variable latency
    'registers': 65536,      # per SM (256 KB)
    'smem': 228 * 1024,      # 228 KB per SM
}

# Step 2: Twill's optimization
# Phase 1: Find minimum initiation interval via ILP
I_opt = modulo_schedule_ilp(operations, edges, hopper)
# For TFLA-like loop: I_opt ≈ 2 cycles (normalized)
# Meaning: one new iteration can start every 2 cycles

# Phase 2: Find realizable schedule via SMT
M_star, A_star = smt_joint_optimization(
    operations, edges, hopper, I=I_opt
)
# A_star might assign:
#   Warp 0: load_K, load_V (producer/TMA)
#   Warp 1: S, gate, O (consumer/compute) — if single warp suffices
#   OR
#   Warp 1: S, O (TC operations)
#   Warp 2: gate (SFU operations) — ping-pong style

# Step 3: Verify optimality
# If SMT returns SAT at I_opt: schedule is provably optimal
# If UNSAT: try I_opt + 1, and the result is optimal at that I
```

**What Twill discovers for FlashAttention on Hopper (Figure 7 case study):**

1. **SWP component:** Extract the first GEMM0 into the prologue. In the steady state, pipeline iterations with $I = 2$ normalized cycles: GEMM0 of iteration $j+1$ overlaps with softmax of iteration $j$.

2. **WS component:** Two consumer warp groups alternate ("ping-pong"): while WG1 does GEMMs, WG2 does softmax, and vice versa. One producer warp group handles TMA loads.

3. **Optimality proof:** The TC capacity constraint forces $I \geq 2$ (two GEMMs per iteration, one TC per SM). Since Twill achieves $I = 2$, no better schedule exists.

**GPU efficiency analysis:**

1. **Provable optimality:** The only system that can guarantee the discovered schedule achieves maximum throughput for the given machine model. This eliminates months of manual trial-and-error.

2. **Architecture-portable:** Same TFLA dependence graph, different machine model → optimal schedule for each GPU generation. No human re-optimization when moving from Hopper to Blackwell.

3. **Reveals binding constraints:** When the SMT solver reports UNSAT at a given $I$, the unsatisfiable core identifies exactly which constraint (register pressure, functional unit capacity, cross-warp communication) prevents the optimal schedule, guiding hardware-aware algorithm design.

4. **Composition with other tricks:** Twill's schedule can be combined with orthogonal optimizations: ThunderKittens' bank-conflict-free layouts (trick 202), TFLA's two-level tiling (trick 158), FlashMask's tile skipping (trick 191), etc.

5. **Practical solve times:** 19-88 seconds per kernel — fast enough for offline compilation, making it practical for production kernel development.

## References

- Soi, R., Yadav, R., Kjolstad, F., Aiken, A., Dehnavi, M. M., Garland, M., & Bauer, M. (2025). Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs. arXiv:2512.18134.
- Shah, J., et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. arXiv:2407.08608 (trick 208).
- Dao, T., Shah, J., Zoudari, T., Hoehnerbach, M., & Thakkar, V. (2025). Flash Attention 4. https://github.com/Dao-AILab/flash-attention.
- Chen, H., et al. (2025). Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References. arXiv:2510.14719 (trick 183).
- Lam, M. (1988). Software Pipelining: An Effective Scheduling Technique for VLIW Machines. PLDI 1988.
- Barrett, C. & Tinelli, C. (2018). Satisfiability Modulo Theories. Springer.
- Cheng, Y., et al. (2025). PipeThreader: Software-Defined Pipelining for Efficient DNN Execution. OSDI 2025.
- Beck, M., et al. (2025). Tiled Flash Linear Attention. arXiv:2503.14376 (trick 158).

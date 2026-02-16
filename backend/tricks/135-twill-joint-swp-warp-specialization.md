# 135: Twill: Joint Software Pipelining and Warp Specialization via Constraint Solving

**Category**: kernel
**Gain type**: efficiency
**Source**: Soi, Yadav, Kjolstad, Aiken (Stanford), Mehri Dehnavi, Garland, Bauer (NVIDIA) — "Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs" (arXiv 2512.18134, Dec 2025)
**Paper**: [papers/twill-optimal-swp-warp-specialization.pdf]
**Documented**: 2026-02-15

## Description

Twill is a compiler system that automatically discovers **provably optimal** software pipelining (SWP) schedules and warp specialization (WS) strategies for GPU kernels targeting Tensor Core (TC) architectures. The key insight is that SWP and WS — previously treated as independent optimization steps — are deeply coupled, and determining them jointly as a single optimization problem yields significantly better results than any sequential or heuristic approach.

**Software pipelining (SWP)** is a classic compiler transformation that overlaps instructions from different loop iterations to maximize functional unit utilization. For GPU GEMM kernels, this means overlapping TMA loads (data movement), Tensor Core operations (compute), and exponentials/softmax (SFU operations) across iterations so that no functional unit is ever idle. The throughput of a pipelined loop is determined by the **initiation interval** $I$ — the number of cycles between starting consecutive iterations. Minimizing $I$ maximizes throughput.

**Warp specialization (WS)** assigns different operations to different warps within a thread block. This is necessary on modern TC GPUs because: (1) TC operations require multiple warps (warp groups) to cooperatively issue; (2) modulo schedules demand large working sets that exceed per-thread register limits; (3) TMA operations have highly variable latencies (orders of magnitude range); and (4) TC and TMA have asynchronous interfaces that cause blocking synchronization to stall concurrent operations on the same warp.

Prior to Twill, SWP schedules and WS strategies for kernels like FlashAttention were developed by human experts over months of manual effort, with no guarantee of optimality. Twill formulates the joint problem as a **Satisfiability Modulo Theories (SMT) constraint satisfaction problem** and solves it with off-the-shelf solvers (Yices2 for SMT, SCIP for $\mathbb{Z}$LP), producing schedules that are guaranteed optimal for a given machine model. Twill automatically rediscovers the expert-designed FlashAttention 3 (Hopper) and FlashAttention 4 (Blackwell) schedules from first principles in 19–88 seconds.

## Mathematical Form

**Input program:**

The input is a loop described as a dependence graph $G = (V, E)$ where $V$ is the set of tile-level instructions (GEMM, EXP, TMA\_LOAD, TMA\_STORE, etc.) and $E$ defines data dependencies. Each edge $e = (u, v, d, \delta) \in E$ indicates that $v$ must be issued at least $d$ clock cycles after $u$, and $\delta \geq 0$ is the iteration delay ($\delta = 1$ means the dependence is loop-carried).

**Resource Reservation Tables (RRTs):**

Each instruction $v$ has an RRT $\text{RRT}[v]$, a 2D integer array where rows are clock cycles and columns are functional unit types (TC, SFU, ALU, TMA, etc.). Entry $\text{RRT}[v][f, c]$ gives the number of instances of functional unit $f$ occupied at cycle $c$ during $v$'s execution.

**Machine description:**

$D = \{(f, \text{cap}(f)) : f \in \text{FunctionalUnits}\}$ specifies the capacity (number of instances) of each functional unit. E.g., on Hopper: $D = \{\text{TC}: 1, \text{SFU}: 1\}$ (one TC unit and one SFU unit per SM).

**Modulo scheduling:**

The goal is to find:
- **Initiation interval** $I$: the steady-state throughput (iterations per $I$ cycles)
- **Modulo schedule** $M: V \to \{0, \ldots, L-1\}$: maps each instruction to its issue cycle within one iteration
- **Schedule length** $L = |M|$: total cycles for one iteration

The pipelined loop overlaps $\lceil L/I \rceil$ copies of $M$, each offset by $I$ cycles:

$$
\text{Prologue: } (\lceil L/I \rceil - 1) \cdot I \text{ cycles}, \quad \text{Steady state: repeats every } I \text{ cycles}, \quad \text{Epilogue: drains}
$$

**Constraint formulation (Figure 4 from paper):**

Define boolean variable $\text{op}[v, i, t] = 1$ iff operation $v$ from iteration $i$ is scheduled at clock cycle $t$ in the straight-line program $Q^*$:

**Uniqueness:** Each operation scheduled exactly once:
$$
\forall v, i: \quad \sum_{t} \text{op}[v, i, t] = 1
$$

**Consistency:** Operations maintain modulo structure:
$$
\forall v, i \in [1, \lceil L/I \rceil), t: \quad \text{op}[v, 0, t] \Rightarrow \text{op}[v, i, t + i \cdot I]
$$

**Completion:** All operations finish within schedule length:
$$
\forall v, i, t: \quad t + \text{cycles}(v) > T \Rightarrow \neg\text{op}[v, i, t]
$$

**Dependence:** Data dependencies respected:
$$
\forall i, t, (u, v, d, \delta) \in E, t' \in [0, t+d): \quad \text{op}[u, i, t] \Rightarrow \neg\text{op}[v, i + \delta, t']
$$

**Capacity:** Functional unit capacity not exceeded:
$$
\forall t, f: \quad \sum_{v, i, c \in [0, \text{cycles}(v))} \text{op}[v, i, t-c] \cdot \text{RRT}[v][f, c] \leq \text{cap}(f)
$$

**Memory-aware constraints (Figure 5):**

Define $\text{live}[v, i, t] = 1$ when the result of the $i$-th instance of $v$ is live at time $t$:

$$
\forall t, m: \quad \sum_{v, i} \text{live}[v, i, t] \cdot \text{footprint}(v, m) \leq \text{capacity}(m)
$$

This ensures the working set fits in on-chip memory (registers + shared memory).

**Warp assignment constraints (Figure 6):**

Define $\text{opw}[v, w] = 1$ iff operation $v$ is assigned to warp $w$:

**Warp uniqueness:**
$$
\forall v: \quad \sum_{w} \text{opw}[v, w] = 1
$$

**Variable latency:** Operations with variable latency (TMA) are assigned to a designated warp $W_{vl}$:
$$
\forall v: \quad \text{variable\_latency}(v) \Leftrightarrow \text{opw}[v, W_{vl}]
$$

**Register limit:** Per-warp register budget:
$$
\forall t, w: \quad \sum_{v, i} \text{live}[v, i, t] \cdot \text{opw}[v, w] \cdot \text{regs}(v) \leq \text{reg\_limit}()
$$

**Cross-warp spill:** When producer and consumer are on different warps, add spill delay:
$$
\forall (u, v, d, \delta) \in E, t, i, w, w' \neq w, s: \quad \text{op}[u, i, t] \wedge \text{opw}[u, w] \wedge \text{opw}[v, w'] \Rightarrow \neg\text{op}[v, i + \delta, t + d + s]
$$

where $s$ is the spill cost (cycles to transfer through shared memory).

**Concurrency:** Blocking synchronization prevents concurrent operations on the same warp:
$$
\forall (u, v, \_, \_) \in E, t, w, i, o \neq v: \quad \text{op}[v, i, t] \wedge \text{opw}[v, w] \wedge \text{blocking}(u, v) \Rightarrow \forall t', t' \in [t - \text{cycles}(o) + 1, t]: \neg(\text{op}[o, i', t'] \wedge \text{opw}[o, w])
$$

**Cost normalization ($\mathbb{Z}$LP):**

Raw cycle counts (e.g., 128×128×128 GEMM ≈ 1000 cycles) make the SMT problem intractable. Twill normalizes costs by solving:

$$
\forall i, j: \quad -F \leq C[i] \cdot C'[j] - C[j] \cdot C'[i] \leq F
$$

$$
1 \leq \sum_i C'[i] \leq U
$$

where $C$ are original cycle counts, $C'$ are normalized counts, $F$ bounds the ratio distortion, and $U$ controls resolution. This $\mathbb{Z}$LP (solved by SCIP) produces small integers whose ratios approximate the original ratios.

**Key Definitions:**

- $I$ — Initiation interval; the number of cycles between the start of consecutive loop iterations in the pipelined schedule. Smaller $I$ = higher throughput
- $L$ — Schedule length; total cycles for a single iteration. $\lceil L/I \rceil$ iterations are in-flight simultaneously
- Modulo schedule $M$ — Assignment of each instruction to a cycle within one iteration, repeating every $I$ cycles in steady state
- RRT — Resource Reservation Table; specifies which functional units an instruction occupies at each cycle of its execution
- $\mathbb{Z}$LP — Integer Linear Program; used for both cost normalization and initial modulo schedule computation
- SMT — Satisfiability Modulo Theories; the constraint theory used to jointly solve for schedule + warp assignment (specifically QFLIA — quantifier-free linear integer arithmetic)

## Complexity

| Approach | Optimality Guarantee | Time to Solution | Human Effort |
|----------|---------------------|------------------|--------------|
| Manual expert design (FA3, FA4) | None | Months | Very high |
| Heuristic compilers (Triton, Tawa, Cypress) | None | Seconds | Low |
| Twill (joint SWP + WS) | **Optimal** for given $G$, $D$ | 19–88 seconds | **None** |

**Performance on Flash Attention (NVIDIA H100 SXM, Blackwell B200):**

| Architecture | Kernel | Twill TFLOPS/s | Best Expert TFLOPS/s | Gap |
|-------------|--------|---------------|---------------------|-----|
| Hopper (H100) | FA Forward (seq=16384) | ~645 | ~650 (FA3) | <1% |
| Hopper (H100) | FA Backward (seq=16384) | ~415 | ~460 (FA3) | ~11%* |
| Blackwell (B200) | FA Forward (seq=16384) | ~960 | ~980 (FA4/cuDNN) | ~2% |
| Blackwell (B200) | FA Backward (seq=16384) | ~820 | ~860 (FA4) | ~5%* |

*\*Backward gap is primarily due to orthogonal optimizations (tile size, memory layout, instruction selection) rather than SWP/WS strategy.*

**Key discovery: Twill automatically rediscovers:**

1. **FA3 Hopper forward**: Extracts first GEMM into the loop prologue (SWP) + ping-pong scheduling between sub-tiles (WS) — exactly matching the manually designed FA3 pipeline
2. **FA4 Blackwell forward**: A novel 3-warp-group strategy with separate groups for GEMMs, softmax, and accumulator rescaling — a strategy that does not correspond to conventional "loader/compute" warp roles and was independently discovered by Twill and FA4 experts

## Applicability

- **Fused Multi-Head Attention (forward + backward)**: The primary evaluation target. Twill derives optimal pipelines for FlashAttention on both Hopper and Blackwell, the most performance-critical kernel in transformer inference and training
- **Any tile-based iterative GPU kernel**: Twill operates on dependence graphs extracted from tile-based IRs (Triton TTGIR). Any kernel expressible as a singly-nested loop with tile-level operations — GEMM chains, convolutions, linear recurrences — can be optimized
- **Cross-architecture portability**: The same high-level program produces different optimal schedules for Hopper vs. Blackwell by simply changing the machine description $D$. As new architectures (Blackwell Ultra, Rubin) emerge, only $D$ needs updating
- **Kernel development acceleration**: Eliminates months of manual SWP/WS tuning for new GPU architectures. Useful as an offline compilation tool or developer aid for performance-critical kernels
- **SSM and linear attention kernels**: Any kernel with the structure "iterate: load tile → compute (TC) → apply element-wise ops (SFU) → accumulate → store" can benefit from Twill's joint optimization

## Limitations

- **Singly-nested loops only**: Twill currently supports only singly-nested loops without additional control flow. Hierarchical reduction techniques from the SWP literature could extend this, but are left as future work
- **Static tile size**: Twill optimizes the schedule for a given tile size but does not determine the optimal tile size itself. A higher-level auto-tuning system must select tile dimensions
- **Solution time**: 19–88 seconds for FlashAttention-class kernels. Acceptable for offline compilation but not for interactive development. More complex programs (larger $G$, larger $L/I$) may take longer
- **Orthogonal optimizations not addressed**: Memory layout conversions, instruction selection, synchronization placement, and data layout transformations are handled separately. Twill's schedule may not compile efficiently through existing backends (Triton had issues; the paper uses hand-compiled CUDA C++)
- **Cost model approximation**: RRTs and cycle counts are estimated from documentation or measurement. The cost normalization introduces small ratio distortions. Variable-latency operations (TMA) are handled heuristically by offloading to dedicated warps
- **SMT scalability**: The constraint system is exponential in the sum of edge delays $\sum d$. Cost normalization mitigates this, but very large dependence graphs may be intractable

## Implementation Notes

```python
# Pseudocode for Twill's search procedure (Algorithm 1 from paper)

def twill(G):
    """
    Find optimal SWP schedule M* and WS assignment A* for dependence graph G.

    G = (V, E): V = set of tile-level instructions, E = dependencies
    Each v in V has: RRT[v], cycles(v), regs(v), footprint(v)
    Each e in E: (u, v, d, delta) — u->v with delay d, iteration delay delta
    """
    I = 0  # initiation interval (to be minimized)

    while True:
        I += 1

        # Step 1: Find optimal modulo schedule with initiation interval I
        # Solved as ZLP (Integer Linear Program) using CBC/SCIP solver
        M = optimal_modulo_schedule(G, I)
        if M is None:  # failure: no valid schedule at this I
            continue

        L = schedule_length(M)

        # Step 2: Search over schedule lengths that don't change ceil(L/I)
        while ceil(L / I) == ceil(schedule_length(M) / I):
            # Construct straight-line program Q from M
            # Q has ceil(L/I) overlapped copies of M, offset by I each
            Q = construct_straight_line_program(M, I, L)

            # Step 3: Solve joint SWP + WS as SMT constraints
            # Uses Yices2 solver with QFLIA (quantifier-free linear integer arithmetic)
            constraints = []

            # --- Modulo scheduling constraints ---
            constraints += uniqueness_constraints(Q)      # each op scheduled once
            constraints += consistency_constraints(Q, I)  # modulo structure
            constraints += completion_constraints(Q)      # finish within T
            constraints += dependence_constraints(Q, G)   # respect data deps
            constraints += capacity_constraints(Q, G, D)  # functional unit limits

            # --- Memory constraints ---
            constraints += liveness_constraints(Q, G)     # track live variables
            constraints += memory_capacity_constraints(Q) # fit in registers/SMEM

            # --- Warp assignment constraints ---
            constraints += warp_uniqueness_constraints(Q)     # each op -> 1 warp
            constraints += variable_latency_constraints(Q)    # TMA -> dedicated warp
            constraints += register_limit_constraints(Q)      # per-warp reg budget
            constraints += cross_warp_spill_constraints(Q, G) # spill delay penalty
            constraints += concurrency_constraints(Q, G)      # no blocking conflicts

            (M_star, A_star) = smt_solve(constraints)  # Yices2 SMT solver

            if (M_star, A_star) is not None:  # SUCCESS
                return (M_star, I, A_star)

            L += 1  # try longer schedule length

    # Output: M* (modulo schedule), I (initiation interval), A* (warp assignments)
    # The schedule specifies for each instruction:
    #   - Which cycle to issue it (within steady state)
    #   - Which warp(s) execute it
    # This is sufficient to generate pipelined, warp-specialized code


def cost_normalization(C, U=300):
    """
    Normalize cycle counts to small integers preserving ratios.
    Solved as ZLP using SCIP solver.

    C: list of original cycle counts
    U: upper bound on sum of normalized counts (controls resolution)
    Returns C': normalized counts where C[i]/C[j] ≈ C'[i]/C'[j]
    """
    # Minimize F subject to:
    #   -F <= C[i]*C'[j] - C[j]*C'[i] <= F   for all i,j
    #   1 <= sum(C') <= U
    #   C'[i] >= 1 (integer)
    F, C_prime = solve_zlp(C, U)
    return C_prime


# Example: Simplified Flash Attention dependence graph
# (from Figure 1 of the paper)
#
# Instructions:
#   S = gemm(Q, K[i])     — uses TC, takes 1 cycle (normalized)
#   P = exp(S)             — uses SFU, takes 1 cycle (normalized)
#   O += gemm(P, V[i-1])  — uses TC, takes 1 cycle (loop-carried)
#
# Dependencies:
#   S -> P: d=1, delta=0  (within iteration)
#   P -> O: d=1, delta=0  (within iteration)
#   O -> O: d=1, delta=1  (loop-carried: accumulate across iterations)
#
# Twill finds I=2 (one iteration every 2 cycles):
#   Cycle 0: TC=S0        (GEMM for Q*K)
#   Cycle 1:              (TC idle — exposed latency)
#   Cycle 2: TC=S1, SFU=P0  (next GEMM overlapped with exp)
#   Cycle 3: TC=O0        (accumulate GEMM overlapped)
#   Cycle 4: TC=S2, SFU=P1  (steady state repeats from here)
#   ...
#
# Warp assignment (Hopper):
#   Warp group 0: TC operations (S, O)
#   Warp group 1: SFU operations (P) + TMA loads
#   This is exactly the FA3 ping-pong scheduling strategy!
```

## References

- Soi, R., Yadav, R., Kjolstad, F., Aiken, A., Mehri Dehnavi, M., Garland, M., Bauer, M. "Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs." arXiv:2512.18134, Dec 2025.
- Dao, T., Haziza, D., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024.
- Shah, J., Bikshandi, G., et al. "FlashAttention-4." NVIDIA, 2025.
- Chen, H., Fan, A., Collins, B., et al. "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References." arXiv:2510.14719, 2025.
- Bauer, M., Cook, H., Khailany, B. "CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization." SC 2011.
- Bauer, M., Treichler, S., Aiken, A. "Singe: Leveraging Warp Specialization for High Performance on GPUs." PPoPP 2014.
- Rau, B. R. "Iterative Modulo Scheduling: An Algorithm for Software Pipelining Loops." MICRO 1994.

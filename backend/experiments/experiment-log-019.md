# Experiment Log - 019: Capacitance-Coupled Multi-Scale SSM

## [2026-02-15 17:00] Experiment 019: Capacitance-Coupled Multi-Scale SSM

### Selected Proposal
- **ID**: 019-capacitance-coupled-multi-scale-ssm
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Tests a principled multi-scale decomposition where SSM blocks at different timescales are coupled through a small capacitance matrix. The nested periodic detection task directly exercises the core hypothesis.

### Implementation Plan
1. Implement MC-SSM model with k=4 scale blocks, capacitance coupling matrix
2. Implement monolithic SSM baseline (same total state dim n=32)
3. Implement uncoupled multi-scale baseline (C=0)
4. Create synthetic nested periodic detection data generator
5. Write training script with proper logging
6. Run all three models and compare
7. Report results vs success criteria

### [17:00] Attempt: Initial Implementation
**Goal**: Implement complete MVE with model, data, and training

**Design Decisions**:
- Task: Nested periodic detection — signal = sin(2πf1·t) + 0.5·sin(2πf2·t) + 0.25·sin(2πf3·t) + noise
  - f1=1/8 (fast), f2=1/32 (medium), f3=1/128 (slow)
  - 8 classes: all 2^3 combinations of which frequencies are present
  - Model must classify which frequencies are present in each window
- Model: 2-layer MC-SSM, d=64, n=32 total state (k=4 scales, n_i=8), ~90K params
- Timescales: geometric spacing Δt_i = Δt_min · ρ^(i-1), ρ=4
- Capacitance matrix: k×k with diagonal (input-dependent gating) + off-diagonal (non-positive coupling)
- Training: CrossEntropy loss, 10K sequences, length 512

**Actions**:
- Creating all files...

(Log continues below as work progresses)

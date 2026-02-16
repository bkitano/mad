# Experiment Log — 014: Log-Semiring SSM Scan

## [2026-02-15 17:00] Experiment 014: Log-Semiring SSM

### Selected Proposal
- **ID**: 014-log-semiring-ssm-scan
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: Novel algebraic foundation — replaces (R, +, ×) semiring with (R, logsumexp, +). Should natively compute softmax-weighted attention over history without approximation. Selective copying task is the minimal test separating softmax from linear attention.

### Implementation Plan
1. Implement selective copying data generator (synthetic)
2. Implement LogSSM model with log-semiring parallel scan
3. Implement Linear Attention baseline (RetNet-style)
4. Implement Diagonal SSM baseline (Mamba-style)
5. Write training script with accuracy tracking
6. Run all three models, compare accuracy
7. Report results vs success criteria

### [17:00] Starting Implementation
**Goal**: Create full MVE codebase
**Actions**: Creating directory structure, models, data gen, training script

---

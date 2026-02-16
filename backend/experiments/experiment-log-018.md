# Experiment Log 018: Hutchinson Trace-Guided Adaptive Rank for DPLR SSMs

## [2026-02-16 00:00] Experiment 018: Hutchinson Adaptive Rank DPLR SSM

### Selected Proposal
- **ID**: 018-hutchinson-adaptive-rank-dplr-ssm
- **Priority**: medium
- **Estimated cost**: $0.33
- **Reasoning**: Tests whether adaptive rank allocation via Hutchinson trace estimates can make DPLR SSMs more parameter-efficient. Clear MVE with synthetic sCIFAR task.

### Implementation Plan
1. Implement DPLR SSM model with configurable per-layer rank
2. Implement power-series log-det importance scoring
3. Implement SVD-based rank truncation
4. Implement training loop with 4 phases: warmup → measure → prune → fine-tune
5. Run baselines (fixed r=4, fixed r=8) and adaptive model
6. Compare results against success criteria

### MVE Details (from proposal)
- **Model**: 4-layer DPLR SSM, n=32, d=64, r_max=8, ~80K params
- **Task**: Sequential CIFAR-10 (sCIFAR) — length-1024 sequences
- **Procedure**:
  1. Train all 4 layers at r=8 for 2K steps
  2. Compute importance scores via power-series log-det
  3. Print rank distribution
  4. Truncate to adaptive ranks with budget R_total=16 (avg r=4)
  5. Fine-tune for 2K more steps
  6. Compare against fixed r=4 (4K steps) and fixed r=8 (4K steps)
- **Success criteria**:
  - max importance / min importance > 2
  - Adaptive (budget 16) ≥ 95% of fixed r=8 (budget 32) accuracy
  - Adaptive (budget 16) > fixed r=4 (budget 16) by >1% accuracy

---

## [2026-02-16 00:01] Starting Implementation

### Creating code structure
- code/018/models/__init__.py
- code/018/models/dplr_ssm.py — DPLR SSM model
- code/018/train.py — Training script with all phases
- code/018/config.yaml — Experiment configuration
- code/018/modal_config.py — Modal deployment
- code/018/pyproject.toml — Dependencies
- code/018/README.md — Instructions

---

## [2026-02-16 00:05] Implementation Complete

### Files Created
1. **models/dplr_ssm.py** (~400 lines): Full DPLR SSM implementation with:
   - `DPLRSSMLayer`: Complex-valued A = Λ + PQ* with ZOH discretization and sequential recurrence
   - `compute_importance_logdet()`: Power-series log-det importance scoring (O(k*r^3) per freq)
   - `compute_importance_hutchinson()`: Hutchinson trace estimator for Frobenius norm (O(nr) per freq)
   - `truncate_rank()`: SVD-based rank reduction with sqrt-singular-value split
   - `DPLRSSMBlock`: SSM + LayerNorm + MLP with residual connections
   - `DPLRSSMModel`: Full model with input projection, blocks, pooling, classification head

2. **train.py** (~430 lines): Training script with:
   - sCIFAR-10 data loading (torchvision) with synthetic fallback
   - Step-based training with wandb logging
   - 4-phase adaptive procedure: warmup → measure → prune → fine-tune
   - Baselines: fixed r=4 and fixed r=8
   - Success/failure criteria evaluation

3. **config.yaml**: MVE configuration (4 layers, n=32, d=64, r_max=8, 2K+2K steps)
4. **modal_config.py**: Modal T4 GPU deployment with wandb secret
5. **pyproject.toml**: Dependencies (torch, torchvision, wandb, etc.)
6. **README.md**: Setup and run instructions

### Design Decisions
- Used sequential recurrence (not parallel scan) for simplicity — acceptable for seq_len=1024 in MVE
- Input dim = 3 (RGB channels) rather than 1 (grayscale) for sCIFAR — follows standard practice
- Power-series log-det as default importance method (cheaper than Hutchinson: O(r^3) vs O(nr))
- Global average pooling over sequence for classification
- Cosine annealing LR scheduler with lower LR for fine-tuning phase (0.5x ratio)

---

## [2026-02-16 00:10] Deployment to Modal

### Attempt: Deploy to Modal
**Goal**: Submit experiment to Modal T4 GPU
**Command**: `uv run modal run --detach modal_config.py --config config.yaml`

**Result**: ✅ Success
**Modal App URL**: https://modal.com/apps/bkitano/main/ap-XKXnVzISuNLGETKW4Rbgap
**Image Build**: im-D2gS4874dczO0zA6ufERIe (built in 108.68s)
**GPU**: T4 x 1
**Timeout**: 3600s (1 hour)

Job is running in detached mode. Will monitor via Modal dashboard and wandb.


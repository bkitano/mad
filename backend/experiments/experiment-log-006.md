# Experiment Log - MVE 006: Monarch-Gated State Transition SSM

## [2026-02-15] Experiment 006: Monarch-Gated State Transition SSM

### Selected Proposal
- **ID**: 006-monarch-gated-state-transition
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: Tests whether Monarch-factored state transitions enable coordinate mixing that diagonal SSMs cannot achieve. S5 permutation composition is the canonical test for this capability. The BMM structure of Monarch matrices enables efficient GPU computation.

### Implementation Plan
1. Implement S5 permutation group data generator (10 transpositions, 120 permutation classes)
2. Implement Monarch-Gated SSM with Cayley-parameterized orthogonal blocks and input-dependent gating
3. Implement Diagonal SSM baseline for comparison
4. Create training script that trains both models and measures accuracy + speed
5. Deploy to Modal and evaluate against success criteria

### Implementation Notes

**S5 Task Design**:
- 10 transpositions as generators: (0,1), (0,2), ..., (3,4)
- 120 output classes (all S5 permutations)
- Fixed-length-20 sequences for fair train/test comparison
- Online data generation to prevent memorization

**Monarch-Gated SSM Architecture**:
- State dim n=64, block size b=8 (8x8 blocks)
- Cayley-parameterized orthogonal blocks L_i, R_i (batched, pre-computed once per forward)
- Input-dependent scalar gates alpha_i, beta_i per block via sigmoid
- Stride permutation P_b for inter-block mixing
- Efficient: uses einsum for block-diagonal application (O(n*sqrt(n)) not O(n^2))
- Stability: contractive by construction (orthogonal blocks * gates < 1)

**Model Configuration**:
- 2 layers, d_model=64, state_dim=64
- Both models share same architecture except SSM layer type
- Monarch: 106,136 params, Diagonal: 110,328 params

### Attempt Log

#### [2026-02-15 22:00] Implementation Complete

**Goal**: Implement full MVE 006 including models, data generator, training script

**Files created**:
- `code/006/tasks/s5.py` - S5 permutation group data generator
- `code/006/models/monarch_ssm.py` - Monarch-Gated SSM + Diagonal SSM baseline
- `code/006/train/run_config.py` - Training script with wandb logging
- `code/006/train/modal_config.py` - Modal deployment config
- `code/006/config.yaml` - Experiment configuration
- `code/006/pyproject.toml` - Dependencies

**Smoke tests**: ✅ All pass (data gen, forward pass, backward pass, training loop)

---

#### [2026-02-15 22:09] v1 Modal Deployment

**Modal App URL**: https://modal.com/apps/bkitano/main/ap-UtgpGetkc6tPf1tuEU8GyP
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/nbb4vjhb

**Result**: ❌ Severe overfitting

**Issue**: Monarch model reached 97.8% TRAIN accuracy but only ~2% TEST accuracy. Complete memorization.

**Root cause analysis**:
- 10K samples with variable-length sequences (10-50 transpositions)
- 120 output classes → only ~67 train samples per class on average
- Model memorizes specific input sequences rather than learning the group operation
- Train/test distribution mismatch due to variable sequence lengths

---

#### [2026-02-15 22:30] v1.5 Optimization: Batched Cayley Transform

**Goal**: Optimize Monarch model to pre-compute orthogonal blocks once per forward call

**Bug found**: Original implementation called Cayley transform (matrix solve) for each block at each timestep. With b=8 blocks, 2 factors (L,R), seq_len=50: that's 800 Cayley transforms per sample per layer.

**Fix**:
- Changed from `nn.ModuleList[CayleyOrthogonal]` to batched `nn.Parameter` tensors
- Single batched `cayley_transform_batched()` call for all blocks
- Pre-compute L_Q, R_Q once per layer forward, pass to all timesteps
- Use einsum for efficient block-diagonal multiplication

**Speed improvement**: 2.23x Monarch/Diagonal ratio on CPU (well under 3x target)

**Modal App URL**: https://modal.com/apps/bkitano/main/ap-Xfilcttl7V0zmpv38rI1yp
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/usdsa1c2

**Result**: ❌ Still overfitting (same root cause, but optimized code runs faster)

---

#### [2026-02-15 23:15] v2: Online Data Generation Fix

**Goal**: Fix memorization by using online (infinite) data generation

**Key changes**:
1. **Online training data**: Fresh random batches generated every epoch (S5OnlineDataset)
2. **Fixed sequence length**: All sequences use exactly 20 transpositions (no length variability)
3. **Deterministic test set**: Fixed 2000-sample test set for consistent evaluation
4. **Shorter sequences**: seq_len=20 instead of 10-50 (easier to generalize)
5. **Lower learning rate**: 1e-3 instead of 3e-3 (more stable optimization)

**Rationale**: With infinite training data, the model MUST learn the general group composition operation to achieve good accuracy — it can't memorize individual sequences.

**Modal App URL**: https://modal.com/apps/bkitano/main/ap-5TI1yPMQZUPyMw60K1Osqs
**GPU**: T4 (1 GPU)
**Status**: Running (detached)

**Awaiting results...**

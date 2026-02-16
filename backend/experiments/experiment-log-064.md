# Experiment Log - 064: Residual KDA with Channel-Wise Auxiliary Decay

## [2026-02-16 00:00] Experiment 064: Residual KDA

### Selected Proposal
- **ID**: 064-residual-kda-channel-wise-auxiliary-decay
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: Tests whether channel-wise decay in the auxiliary residual state improves recall over scalar decay. Directly builds on KDA (proven at 48B scale) and RLA (proven for scalar decay). Simple to implement since the residual state uses the same KDA kernel structure.

### Implementation Plan
1. Implement unified KDA/RKDA model with three modes (none, scalar, channel residual)
2. Reuse MQAR data generator from code/059
3. Create training script that runs all three variants sequentially
4. Deploy on Modal T4 GPU

### [00:00] Implementation Attempt 1
**Goal**: Create all code files based on code/059 patterns
**Actions**:
- Created code/064/ directory structure
- Implemented models/rkda.py with unified KDALayer supporting three residual modes
- Created data/generate.py (MQAR dataset generator)
- Created train.py with training loop, evaluation, and success criteria checking
- Created config.yaml matching proposal specs (2 layers, d=128, d_k=d_v=64, 2 heads)
- Created modal_config.py for T4 GPU deployment
- Created pyproject.toml and README.md

**Key design decisions**:
- Single KDALayer class handles all three variants via `residual_mode` parameter
  - "none": Standard KDA (no residual state)
  - "scalar": Scalar-decay residual (W_alpha_R_scalar produces scalar per head, expanded to all channels)
  - "channel": Channel-wise-decay residual (W_alpha_R produces per-channel decay vector)
- Residual computed BEFORE primary state update (uses S_{t-1}, per proposal)
- Keys normalized before delta rule for stability (matching code/059)
- Residual clipping threshold c=1.0 (per proposal)

**Result**: ✅ Files created

### [00:05] Local Sanity Check
**Goal**: Verify model builds and forward pass works
**Actions**:
- Ran forward pass for all three variants (none, scalar, channel)
- Verified parameter counts:
  - KDA (none): 247,300 params
  - KDA + scalar residual: 248,332 params
  - RKDA (channel): 280,840 params
- Ran full training step (forward + backward + optimizer step)
- Loss: 4.8094 (expected for random init on 64-class problem)
- All gradients computed successfully

**Result**: ✅ All variants build and train correctly

**Bugs encountered**: None

### [00:10] Modal Deployment
**Goal**: Deploy experiment on Modal T4 GPU
**Command**: `uv run modal run --detach modal_config.py --config config.yaml`
**Modal App URL**: https://modal.com/apps/bkitano/main/ap-avE5YteQhr7HcZkIJqILXX
**Status**: Running (detached)
**GPU**: T4 x 1
**Expected duration**: ~5-10 minutes (3 variants × ~150 epochs max with early stopping)

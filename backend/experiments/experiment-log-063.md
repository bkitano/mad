# Experiment Log 063: MFA-Style Shared Latent Projections for Linear RNN Training

## [2026-02-16 00:00] Experiment 063: MFA-Shared-Latent-Projections-Linear-RNN

### Selected Proposal
- **ID**: 063-mfa-shared-latent-projections-linear-rnn
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: Novel application of MFA-style shared latent projections to linear RNNs. Pure GEMM operations (no FFT/permutations), high priority in proposal system. Tests whether shared key structure is compatible with gated linear attention recall.

### Implementation Plan
1. Implement MFA-GLA model with shared down-projections + per-head latent rotations
2. Implement standard GLA baseline for comparison
3. Reuse MQAR data generation from code/059 pattern
4. Write training script that runs both variants, measures accuracy + timing
5. Deploy to Modal with T4 GPU
6. Evaluate against success criteria

### Key Design Decisions
- **MVE spec from proposal**: 2-layer GLA, d=128, C=64, m=4 MFA heads vs n=2 standard heads (~100K params)
- **Task**: MQAR with 8 KV pairs, length 128, 10K sequences
- **Success criteria**:
  - MFA variant >= 90% MQAR accuracy matching standard baseline
  - Forward pass wall-clock <= baseline (no slowdown)
  - Both converge in same number of training steps

### [00:00] Starting Implementation
**Goal**: Set up complete experiment directory with all code files

### [00:05] Attempt: Create directory and implement models
**Goal**: Implement both GLA baseline and MFA-GLA models
**Actions**:
- Created code/063/ with models/, data/ subdirectories
- Cleaned up leftover files from previous experiment 063 (DyT normalization - now replaced)
- Implemented `models/gla_baseline.py`: Standard GLA with independent per-head projections
  - n=2 heads, d_head=32, state S in R^{32x32}
  - Standard recurrent scan: S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
- Implemented `models/mfa_gla.py`: MFA-GLA with shared latent projections
  - Key design: shared S_q, S_k, S_v ∈ R^{128×64} down-projections
  - Per-head Q_c ∈ R^{C×C} = R^{64×64} (rotation in latent space, NOT C→d_k)
  - Per-head V_c ∈ R^{C×d_v} = R^{64×32} (value projection)
  - Keys shared across heads (latent MQA) - k stays in R^C
  - State S^(h) ∈ R^{C×d_v} = R^{64×32} per head
  - Decay gate in key dimension (C)
  - m=4 MFA heads (2× standard)

**Design Decision**: Q_c maps C→C (not C→d_k) because q and k must be in the same space for state readout (o_t = S^T @ q_t). The key dimension for the state IS C (the latent dim).

**Result**: ✅ Success

**Bugs encountered**:
- Bug 1: pyproject.toml used invalid build backend `setuptools.backends._legacy:_Backend`
  - Fix: Changed to `setuptools.build_meta`
- Bug 2: setuptools complained about "Multiple top-level packages discovered" (models, data)
  - Fix: Added `[tool.setuptools.packages.find]` with explicit includes

### [00:10] Attempt: Local verification
**Goal**: Verify forward + backward pass correctness
**Actions**:
- Tested model instantiation: GLA 131,968 params, MFA-GLA 247,040 params
- Tested forward pass shapes: both produce (B, T, vocab_size)
- Tested backward pass with gradients: both compute valid gradients
- CPU training too slow for full test (seq_len=128 with sequential scan) — will test on GPU

**Result**: ✅ Success (models instantiate, forward/backward work correctly)

### [00:15] Deploying to Modal
**Goal**: Submit training job to Modal T4 GPU
**Command**: `uv run modal run --detach modal_config.py --config config.yaml`

---

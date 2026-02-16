# Experiment Log: 003 - DPLR Column-Sparse Cauchy Kernel

## [2026-02-15 14:00] Experiment 003: DPLR Column-Sparse SSM

### Selected Proposal
- **ID**: 003-dplr-column-sparse-cauchy-kernel
- **Priority**: medium
- **Estimated cost**: $0.40
- **Reasoning**: Tests whether fixed permutation P applied to DPLR state matrix provides measurable benefit on parity task requiring inter-dimension coupling. Quick MVE (~5 min) with clear binary outcome.

### Human Feedback Considerations
- Human feedback deprioritizes "irregular indexing/scattering: Permutations, gather/scatter that break coalescing"
- However, this MVE is acceptable as a "small side experiment (<10 GPU-hours) to rule out an approach"
- This experiment will help determine if the permutation idea has merit before investing more resources

### Implementation Plan
1. Implement DPLR-CS SSM model with 4 permutation types (identity, cyclic, bit-reversal, learned)
2. Implement parity task data generator
3. Write training script with proper logging
4. Run all 4 variants and compare accuracy
5. Report results against success/failure criteria

### [14:00] Starting Implementation
**Goal**: Create all necessary files for the MVE

**Actions**:
- Created `code/003/models/dplr_cs_ssm.py` — DPLR-CS SSM model with 4 permutation types
  - `DPLR_CS_SSM`: Core SSM layer with A = P(Lambda + pq^T)P^T
  - `DPLRCSModel`: Full model wrapper with input proj, SSM, pooling, classifier
  - Permutation types: identity, cyclic, bit_reversal, learned (Sinkhorn + straight-through)
  - Bilinear (Tustin) discretization for stability
- Created `code/003/train.py` — Training script
  - Parity data generation (binary XOR task)
  - Train/eval loops with gradient clipping
  - Multi-seed evaluation (3 seeds default)
  - Automatic success/failure criteria checking
- Created `config.yaml`, `pyproject.toml`, `README.md`

**Result**: ✅ All files created

### [14:10] Environment Check
- PyTorch 2.10.0+cu128 available (CPU mode — no CUDA device found)
- CPU is fine for this MVE scale (~5K params, 10K samples)
- All dependencies available

### [14:10] Running Experiment
**Goal**: Run all 4 permutation variants with 3 seeds each
**Command**: `cd code/003 && python train.py --n_seeds 3`

---

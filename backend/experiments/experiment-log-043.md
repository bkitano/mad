# Experiment Log 043: Newton-Schulz Orthogonal State Projection for DeltaNet

## [2026-02-16 00:00] Experiment 043: Newton-Schulz Orthogonal DeltaNet Transition

### Selected Proposal
- **ID**: 043-newton-schulz-orthogonal-deltanet-transition
- **Priority**: high
- **Estimated cost**: $0.17
- **Reasoning**: This proposal replaces the sequential UT transform forward substitution in DeltaNet's chunkwise training with Newton-Schulz polar orthogonalization. The key insight is that the UT forward substitution is a sequential O(C^2) bottleneck that can't use tensor cores, while Newton-Schulz uses pure GEMMs. This directly addresses the GPU efficiency focus from human feedback.

### Implementation Plan
1. Implement S3 (symmetric group of 3 elements = 6 permutations) token system and dataset
2. Implement chunked DeltaNet with UT transform (baseline)
3. Implement chunked DeltaNet with Newton-Schulz orthogonalization (proposed)
4. Training script comparing both approaches on S3 composition task
5. Config, Modal deployment, wandb logging
6. Run and report results

### [00:00] Implementation: S3 Token System
**Goal**: Create S3 group (6 elements) for permutation composition task
**Actions**:
- S3 has 6 elements: e, (12), (13), (23), (123), (132)
- Presented as: identity, 3 transpositions, 2 3-cycles
- Non-abelian, simplest non-abelian group
- Creating tasks/s3/tokens.py and tasks/s3/dataset.py

### [00:05] Implementation: Chunked DeltaNet Models
**Goal**: Implement both UT transform and Newton-Schulz variants
**Actions**:
- Both models use chunk-based processing (chunk_size=32)
- UT variant: computes T via forward substitution of (I+L)
- NS variant: computes orthogonal projection via Newton-Schulz iterations
- Track orthogonality error ||I - X_q X_q^T||_F during training

### [00:10] Implementation: Training and Config
**Goal**: Complete training pipeline with wandb integration
**Actions**:
- Training script runs both models (UT baseline + NS proposed)
- Logs train/val loss, accuracy, orthogonality error
- Microbenchmark for per-chunk kernel timing comparison
- Modal deployment config for T4 GPU

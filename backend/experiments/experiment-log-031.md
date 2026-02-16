# Experiment Log - 031: V:N:M Sparse SSM Projections with S-STE Training

## [2026-02-15] Experiment 031: VNM Sparse SSM Projections

### Selected Proposal
- **ID**: 031-vnm-sparse-ssm-projections-sste
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: VNM structured sparsity targeting projection matrices (the actual computational bottleneck at ~75% of FLOPs) in gated SSMs. High-priority because it addresses real hardware speedup via Sparse Tensor Cores with 60-75% sparsity, uses S-STE for from-scratch training, and SwiGLU gating compensates for sparsity loss.

### Implementation Plan
1. Implement VNMSparseLinear module with two-level pruning (column selection + 2:4 S-STE)
2. Implement simplified GatedSSM (Mamba-2 style) with VNM-sparse projections
3. Implement MQAR (Multi-Query Associative Recall) synthetic data generator
4. Implement training script with 4 baselines: Dense, 2:4, VNM (67%), VNM (75%)
5. Add iso-parameter dense baseline (smaller d_model matching VNM param count)
6. Run all configurations and compare against success criteria

### MVE Specifications (from proposal)
- **Model**: 2-layer Mamba-2 with VNM-sparse projections
- **Dims**: d_model=128, H=4, d_k=32, n=16
- **Params**: ~200K
- **Task**: MQAR, 4 KV pairs, seq_len=64, vocab=16
- **Data**: 10K synthetic sequences
- **Sparsity levels**: Dense, 2:4 (50%), V:2:6 (67%), V:2:8 (75%)

### Success Criteria
- VNM-sparse (75%) achieves >80% accuracy on MQAR at 4 KV pairs
- Dense achieves >95% accuracy (sanity check)
- 2:4 sparse achieves >90% accuracy
- Iso-parameter dense (<70%) shows sparse > small-dense
- S-STE mask flip rate converges within 500 training steps

---

### Implementation Log

#### [Start] Creating directory structure and writing files
**Goal**: Set up code/031 with all necessary files
**Actions**: Creating models, data generator, training script, config, pyproject.toml

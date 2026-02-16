# Experiment Log 064-GHLA: Gated Second-Order Linear Attention

## [2026-02-16] Experiment 064-GHLA: Gated Second-Order Linear Attention

### Selected Proposal
- **ID**: 064-gated-second-order-linear-attention-chunkwise
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Novel combination of GLA's data-dependent gating with HLA's second-order accumulator. Clear MVE with synthetic MQAR task. Tests whether gated second-order attention improves associative recall over first-order GLA and ungated HLA.

**Note**: code/064/ directory was already occupied by a different experiment (Residual KDA), so this experiment uses code/064_ghla/.

### Key Architecture Details (from proposal)
- **GHLA recurrence**:
  - S_t^K = Diag(alpha_t^K) * S_{t-1}^K * Diag(alpha_t^K) + k_t @ k_t^T  (d_k x d_k)
  - C_t^{QV} = Diag(alpha_t^C) * C_{t-1}^{QV} + q_t @ v_t^T  (d_k x d_v)
  - G_t = Diag(alpha_t^K) * G_{t-1} + k_t @ (k_t^T @ Diag(alpha_t^C) @ C_{t-1}^{QV})
  - o_t = q_t^T @ (S_t^K @ C_t^{QV} - G_t)
- **Gates**: alpha_t^K, alpha_t^C in (0,1)^{d_k}, data-dependent via sigmoid
- **MVE model**: 2 layers, d_model=64, d_k=16, d_v=32, 2 heads, ~100K params
- **Task**: MQAR with 10K sequences, length 128, 8 key-value pairs
- **Success**: GHLA > 90% accuracy where GLA < 70%, GHLA > HLA by 5%, GHLA >= HLA-decay

### Implementation Plan
1. Implement GHLA model with gated second-order recurrence
2. Implement baselines: GLA (first-order), HLA (ungated), HLA-decay (fixed decay)
3. Create MQAR synthetic data generator
4. Write training script with wandb logging
5. Create Modal deployment config
6. Run all 4 models, compare MQAR accuracy

---

### Implementation Attempt 1
**Goal**: Create all code files for GHLA MVE
**Actions**:
- Created code/064_ghla/ directory structure (models/, data/)
- Implemented models/ghla.py: GHLA with two-sided diagonal gating on S_t^K
  - Key design: gate_outer = aK_t * aK_t^T for two-sided gating
  - Careful ordering: G_t uses C_{t-1}^{QV} (before current update)
  - Gate bias initialized to 3.0 (sigmoid(3) ~ 0.95 = slow forgetting)
- Implemented models/gla.py: First-order GLA baseline
- Implemented models/hla.py: HLA ungated + HLA-decay (gamma=0.99)
- Created data/mqar.py: MQAR data generator with configurable KV pairs
- Created train.py: Runs all 4 variants sequentially with wandb logging
- Created config.yaml: 2L, d=64, dk=16, dv=32, 2H, 8 KV pairs, seq_len=128
- Created modal_config.py: T4 GPU deployment
- Created pyproject.toml and README.md

**Result**: ✅ Files created

---

### Local Sanity Check
**Goal**: Verify model builds and forward pass works
**Actions**:
- Ran forward pass for all 4 variants
- Verified parameter counts:
  - GHLA: 78,976 params
  - GLA: 74,816 params
  - HLA: 70,656 params
  - HLA-decay: 70,656 params
- MQAR data generation: 8 answer positions per sample (correct)
- Forward pass produces correct logits shape: (4, 128, 32)
- Backward pass: loss=3.4775, all gradients computed successfully

**Result**: ✅ All variants build, forward pass, and backward pass work correctly

**Bugs encountered**: None

---

### Modal Deployment
**Goal**: Deploy experiment on Modal T4 GPU
**Command**: `uv run modal run --detach modal_config.py --config config.yaml`
**Modal App URL**: https://modal.com/apps/bkitano/main/ap-ATL9VgbwKRgFN0AA0y3Tna
**Status**: Running (detached)
**GPU**: T4 x 1
**Expected duration**: ~10-20 minutes (4 variants × ~200 epochs max with early stopping)

### Run 1 Results (8 KV pairs — too easy)
**Wandb URL**: https://wandb.ai/bkitano/mad-architecture-search/runs/g8894h1z
**Duration**: ~8 minutes total

| Variant | Params | Test Acc | Best Val | Time |
|---------|--------|----------|----------|------|
| GLA | 74,816 | 1.0000 | 1.0000 | 45.7s |
| HLA | 70,656 | 0.9996 | 0.9995 | 130.9s |
| HLA-decay | 70,656 | 0.9992 | 0.9986 | 144.5s |
| GHLA | 78,976 | 0.9986 | 0.9980 | 175.4s |

**Analysis**: Task too easy — all models achieve >99% accuracy within 2-3 epochs. GLA hits 100% first, suggesting the first-order state is sufficient for 8 KV pairs with this model size. Need to increase difficulty.

**Decision**: Re-run with harder settings:
- Increase num_kv_pairs from 8 to 32 (much harder)
- Increase vocab_size from 32 to 64 (more values to distinguish)
- Increase seq_len from 128 to 256 (longer sequences)
- This should push GLA below 70% while GHLA may still learn

---

### Run 2: Harder MQAR (32 KV pairs)
**Goal**: Increase task difficulty to differentiate models
**Config changes**: 32 KV pairs, vocab_size=128, seq_len=256, 12K total samples
**Command**: `uv run modal run --detach modal_config.py --config config_hard.yaml`
**Modal App URL**: https://modal.com/apps/bkitano/main/ap-iUOZ4xZIYddomy8yIrp7Qe
**Status**: Running (detached)

Waiting for results...


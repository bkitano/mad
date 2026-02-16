# Experiment Log 059: Second-Order KDA (SO-KDA)

## [2026-02-16 00:00] Experiment 059: Second-Order KDA with HLA Key Metric

### Selected Proposal
- **ID**: 059-second-order-kda-hla-key-metric
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: High-priority proposal combining KDA's delta rule with HLA's second-order key metric. Clear MVE with synthetic MQAR task. Tests whether data-adaptive removal direction improves associative recall.

### Key Idea
Replace KDA's rank-1 removal term (beta_t * k_t * k_t^T) with an adapted removal using a running key covariance matrix M_t. The adapted removal key is k_tilde = M_t @ k_t / ||M_t @ k_t||, which weights removal by historical key importance.

### MVE Setup
- **Model**: 2-layer, d=128, H=4, d_k=d_v=32 (~200K params)
- **Task**: MQAR with 16 key-value pairs, vocab_size=64, T=512
- **Data**: 10K training samples, synthetic
- **Baselines**: Standard KDA, GLA (no delta rule)
- **Success**: SO-KDA > 95% accuracy; KDA ~85%; GLA ~70%
- **Also test**: Length generalization at T=1024

### Implementation Plan
1. Implement MQAR data generator for 16 KV pairs, T=512, vocab=64
2. Implement GLA model (diagonal gating, no delta rule)
3. Implement standard KDA model (delta rule with k_t removal)
4. Implement SO-KDA model (delta rule with M_t-adapted removal)
5. Training script with wandb logging, runs all 3 variants
6. Modal deployment config
7. Run experiment and report results

---

## [2026-02-16 00:05] Implementation: Directory Structure & Data Generator

### Goal: Create code/059/ with full directory structure and MQAR data generator


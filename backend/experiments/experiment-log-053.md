# Experiment Log 053: MLA-Inspired Latent State Compression for Linear RNN Inference

## [2026-02-16 00:00] Experiment 053: MLA Latent State Compression

### Selected Proposal
- **ID**: 053-mla-latent-state-linear-rnn-inference
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Tests whether linear RNN hidden states (d_k x d_v matrices) can be compressed into a low-rank latent vector d_c using SVD, and whether readout can be done via weight absorption in the compressed space without decompression - analogous to DeepSeek-V2 MLA for Transformer KV caches.

### MVE Overview
- **Model**: 2-layer GLA, d=128, d_k=64, d_v=128, 2 heads (~200K params)
- **Task**: Language modeling on synthetic data (WikiText-2-like token sequences)
- **Latent dims tested**: d_c in {8, 16, 32, 64}
- **Protocol**:
  1. Train small GLA model to convergence
  2. Collect hidden states S_t on validation data
  3. Compute SVD of empirical state covariance to check effective rank
  4. Initialize W_down, W_up from top-d_c SVD components
  5. Measure readout error: ||q_t^T S_t - q_tilde^T c_t||^2 / ||q_t^T S_t||^2
  6. Run compressed inference and measure perplexity

### Success Criteria
- Effective rank of states << min(d_k, d_v) (top-16 SVs capture >90% energy)
- d_c=32 achieves <5% relative readout error
- Compressed inference perplexity within 2 points of full inference
- Per-step latency measurably decreases

### Implementation Plan
1. Implement minimal GLA model with recurrent inference mode
2. Implement latent state compression module (SVD-based initialization, weight absorption)
3. Write training script: Phase 1 (train GLA), Phase 2 (analyze states + compressed eval)
4. Deploy on Modal

---

## [2026-02-16 00:01] Implementation: GLA Model

### Attempt: Implement minimal GLA model
**Goal**: Create a small GLA model for language modeling that exposes hidden states during inference.

**Design decisions**:
- GLA head: maintains S_t = diag(alpha_t) * S_{t-1} + k_t * v_t^T
- Readout: o_t = q_t^T * S_t
- Use sigmoid for gating alpha_t (element-wise forget gate)
- For training, use sequential recurrence (this is MVE, not optimized)
- For inference analysis, expose the full state S_t for SVD analysis
- Use synthetic data (random token sequences) to avoid dataset download overhead

**Actions**:
- Creating models/__init__.py
- Creating models/gla.py with GLAHead, GLALayer, GLAModel classes

---

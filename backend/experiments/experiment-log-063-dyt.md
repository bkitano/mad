# Experiment Log 063-DyT: DyT-Fused Normalization-Free Chunkwise Linear RNN

## [2026-02-16] Experiment 063: DyT vs RMSNorm in GLA

### Selected Proposal
- **ID**: 063-dyt-fused-normalization-free-chunkwise-linear-rnn
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: Tests whether Dynamic Tanh (DyT) can replace RMSNorm in chunkwise linear RNN architectures (GLA) without quality degradation. This is a prerequisite for the full fusion experiment which could yield 5-15% throughput improvements.

### Implementation Plan
1. Implement DyT normalization module (elementwise tanh with learnable alpha, gamma, beta)
2. Implement RMSNorm normalization module (standard baseline)
3. Implement 2-layer GLA model with configurable normalization (Pre-LN/Peri-LN x RMSNorm/DyT = 4 variants)
4. Create synthetic language modeling data (Zipf + bigram patterns)
5. Write training script with wandb logging and all 4 variant comparisons
6. Deploy to Modal and monitor results

### Implementation Details
**Model architecture**: 2-layer GLA, d=64, d_k=32, d_v=64, 2 heads, d_ff=128
**Parameters per variant**: ~156K
**Data**: Synthetic LM with vocab=256, seq_len=128, 10K train + 1K val samples
**Training**: 30 epochs, batch_size=64, lr=3e-4, cosine schedule

### Variants
1. `preln_rmsnorm` - Pre-LN RMSNorm (baseline)
2. `preln_dyt` - Pre-LN DyT
3. `periln_rmsnorm` - Peri-LN RMSNorm
4. `periln_dyt` - Peri-LN DyT (full proposal)

### Success Criteria
1. DyT variants match RMSNorm within 0.5 perplexity
2. Peri-LN DyT has best stability (lowest gradient norm variance)
3. No NaN/Inf in any variant
4. DyT activation distributions approximate RMSNorm outputs

### Implementation Steps Completed
- Created code/063/ with models/normalization.py, models/gla.py, train.py
- All 4 variants tested locally (forward + backward passes verified)
- modal_config.py created for Modal deployment

### Modal Deployment
- **Command**: `uv run modal run --detach modal_config.py --config config.yaml`
- **Modal App URL**: https://modal.com/apps/bkitano/main/ap-kKB7jAbwX8pLJb865IIP0z
- **GPU**: T4
- **Status**: Running

### Bugs/Issues
1. **Build system error**: Initial pyproject.toml used `setuptools.backends._legacy:_Backend` which doesn't exist.
   - Fix: Changed to `setuptools.build_meta`

### Awaiting Results
- Job submitted to Modal, awaiting completion
- Will check wandb for metrics when done

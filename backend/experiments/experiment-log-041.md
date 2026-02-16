# Experiment Log 041

## [2026-02-16 00:00] Experiment 041: EVT Joint Forward-Backward Graph Partitioning for Linear RNN Training

### Selected Proposal
- **ID**: 041-evt-joint-fwd-bwd-graph-partitioning-linear-rnn
- **Priority**: high
- **Estimated cost**: $0.33
- **Reasoning**: This is a pure systems/kernel optimization targeting 25%+ HBM traffic reduction in chunkwise linear RNN training by fusing forward+backward GEMM chains. High priority because it directly targets GPU wall-clock speedup for pretraining - aligned with human feedback requirements.

### Implementation Plan
1. Implement a tiny GLA (Gated Linear Attention) model: 1 layer, d=64, n=16, C=32, ~50K params
2. Write baseline separate forward/backward Triton kernels for intra-chunk computation
3. Write fused forward+backward Triton kernel that keeps Q, K, V in registers across fwd→bwd boundary
4. Create benchmarking harness measuring:
   - HBM traffic (proxy via memory profiling)
   - Wall-clock speedup
   - Numerical agreement (bit-exact comparison)
5. Train on synthetic autoregressive task (T=256, 1K random sequences)
6. Run on Modal with A100 GPU

### [00:00] Attempt: Setting up directory structure and project files
**Goal**: Create all project scaffolding
**Actions**:
- Creating code/041/ directory with models/, config, training scripts
- Setting up pyproject.toml with Triton dependency
- Creating modal_config.py

**Result**: In progress...

# Experiment Log - 013: Circulant SSM: Fourier-Domain Parallel Scan

## [2026-02-15] Experiment 013: Circulant SSM with Fourier-Domain Scan

### Selected Proposal
- **ID**: 013-circulant-ssm-fourier-domain-scan
- **Priority**: high
- **Estimated cost**: $0.40
- **Reasoning**: Circulant SSMs diagonalize via FFT in Fourier domain, enabling element-wise parallel scans while providing full coordinate mixing. The Z8 cyclic group task is the canonical test case since circulant matrices represent exactly the cyclic convolution algebra.

### Implementation Plan
1. Implement CirculantSSM model with Fourier-domain parallel scan
2. Implement DiagonalSSM baseline for comparison
3. Implement Z8 cyclic group composition data generator
4. Write training script with proper metric logging
5. Run both models and compare results
6. Check success criteria: Circ-SSM >90% acc, Diag-SSM <60% acc, throughput ratio >0.5x

### MVE Specifications (from proposal)
- **Model**: 2 layers, d=64, n=64, ~100K params
- **Task**: Z8 composition (cumulative mod-8 addition on sequences)
- **Data**: 10K synthetic sequences, length 16-64
- **Compute**: Single GPU, <5 minutes
- **Success criteria**:
  - Circ-SSM >90% accuracy on Z8 at seq len 32
  - Diagonal SSM <60% accuracy on same task
  - Forward throughput Circ-SSM >0.5x diagonal
  - Numerical error ||h_spatial - IFFT(h_scan)||_inf < 1e-4 in FP32

---

### [Starting] Implementation

**Goal**: Create complete MVE implementation

**Actions**:
- Creating directory structure at code/013/
- Implementing CirculantSSM model with Fourier-domain operations
- Implementing DiagonalSSM baseline
- Creating Z8 data generator
- Writing training script

(Log continues below as implementation proceeds...)

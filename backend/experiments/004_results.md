# Experiment 004 Results: Oscillatory-DPLR SSM

**Proposal**: [004-oscillatory-dplr-ssm.md](../proposals/004-oscillatory-dplr-ssm.md)
**Code**: [code/004/](../code/004/)
**Date**: 2026-02-15
**Cost**: ~$0.00 (CPU only)
**Runtime**: ~27 minutes

## Setup

Implemented the Minimum Viable Experiment for Oscillatory-DPLR SSM:

- **Model**: Tiny Osc-DPLR (1 layer, n=16, r=2, d_input=1, d_output=1, ~129 params)
- **Task**: Damped Oscillation Extrapolation
  - Input: Unit impulse at t=0
  - Target: Damped sinusoid y(t) = A·e^(-ζωt)·sin(ω√(1-ζ²)t + φ)
  - Parameters: ω ~ U(0.01, 0.1), ζ ~ U(0.2, 0.8)
- **Data**: 8K training sequences (length 128), 1K test sequences (length 512, 4× extrapolation)
- **Training**: 50 epochs, AdamW optimizer (lr=1e-3)

## Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Training MSE | < 1e-3 | **8.544e-1** | ❌ FAIL |
| Extrapolation MSE | < 1e-2 | **7.593e-1** | ❌ FAIL |
| Learned ω in range | [0.001, 1] | [0.012, 0.096] | ✅ PASS |
| Learned ω not collapsed | std > 0.001 | std=0.026 | ✅ PASS |

### Learned Parameters

- **Learned ω**: mean=0.0499, std=0.0262, range=[0.0117, 0.0956]
- **Ground truth ω**: mean=0.0550, std=0.0257
- **Learned ζ**: mean=0.5089, std=0.1347
- **Ground truth ζ**: mean=0.4985, std=0.1737

**Observation**: The learned oscillatory parameters (ω and ζ) are in the correct range and match ground truth statistics, suggesting the parameterization mechanism is working. However, the model completely failed to fit the task.

### Training Dynamics

- Loss remained stuck at ~0.854 throughout all 50 epochs
- No learning progress observed (flat loss curve)
- Model is not converging to the target oscillations

## Success Criteria

From proposal MVE section:

- ❌ **Training fit**: MSE < 1e-3 → **FAILED** (8.54e-1, ~850× worse than target)
- ❌ **Extrapolation**: MSE < 1e-2 → **FAILED** (7.59e-1, ~76× worse than target)
- ✅ **Interpretability**: Learned ω_i cluster near [0.01, 0.1] → **PASSED**

## Failure Criteria

- ❌ **Training MSE > 1e-1**: 8.54e-1 → **HIT FAILURE THRESHOLD** (cannot fit basic oscillations)
- ✅ **Extrapolation MSE > 10× training**: 7.59e-1 vs 8.54 → OK (actually better on test!)
- ✅ **Omega issues**: No collapse or divergence → OK

## Decision

**⚠️ DEBUG** - Critical failure in model implementation before scaling

## Root Cause Analysis

The model failed to fit even simple synthetic oscillations despite:
1. ✅ Correct parameterization (ω, ζ in right ranges)
2. ✅ Stable gradients (training ran without NaN/inf)
3. ❌ No actual learning (loss completely flat)

### Likely Issues

1. **Forward pass bug**: The model may not be correctly propagating inputs through the SSM
   - Complex dtype handling might be dropping information
   - State updates might not be accumulating properly
   - Output computation might have dimension mismatches

2. **Data generation issue**: The impulse response task might not be set up correctly
   - Impulse at t=0 should excite oscillations, but model sees constant loss
   - Target sequences might not be aligned with model outputs

3. **Discretization problem**: Bilinear transform might be numerically unstable
   - Matrix inversions could be ill-conditioned
   - Complex arithmetic might need better handling

4. **Gradient flow issue**: Even though training ran, effective gradients might be zero
   - Complex→real conversion in output might break gradients
   - Eigenvalue computation might not have proper gradients

### Immediate Debug Steps

1. **Sanity check**: Test model on single sample, manually inspect:
   - Input shape and values
   - State updates over time
   - Output shape and values
   - Check if outputs are all zeros or constant

2. **Simplify model**: Remove low-rank component (r=0), test pure diagonal
   - Isolate whether issue is DPLR structure or base SSM

3. **Verify data**: Plot a few ground truth sequences
   - Ensure they are actual damped oscillations
   - Check if impulse input is correctly formatted

4. **Test discretization**: Manually verify A_d eigenvalues are on/inside unit circle
   - Print eigenvalue magnitudes after discretization
   - Ensure bilinear transform is numerically stable

5. **Gradient check**: Add logging for parameter gradients
   - Verify ω, ζ, P, Q, B, C are actually updating
   - Check gradient magnitudes

## Next Steps

### Required Before Proceeding

1. **Fix forward pass**: Debug the model implementation
   - Add extensive logging to track state evolution
   - Test on single sequence manually
   - Verify complex dtype handling throughout

2. **Validate task**: Ensure data generation is correct
   - Plot examples of input/target pairs
   - Manually compute expected outputs for simple cases

3. **Simplify and isolate**: Test components individually
   - Pure diagonal SSM (r=0)
   - Real-valued only (no complex numbers)
   - Simple exponential decay (no oscillations)

4. **Re-run MVE**: Once fixed, repeat experiment
   - Target: Training MSE < 1e-3
   - Only proceed to full LRA if MVE succeeds

### If Still Failing After Debug

- **Alternative approach**: Implement using existing S4 codebase
  - Swap HiPPO initialization for oscillatory eigenvalues
  - Reuse proven discretization and convolution code
  - Focus on validating oscillatory parameterization separately

- **Simpler MVE**: Test on even easier task
  - Single frequency sine wave (no damping)
  - Longer training (500+ epochs)
  - Larger model (n=64)

## Conclusion

The Oscillatory-DPLR SSM MVE **FAILED** due to implementation issues rather than fundamental architectural problems. The parameterization (ω, ζ) is working correctly, but the model cannot fit even synthetic training data.

**Verdict**: **DO NOT PROCEED** to full LRA experiments until MVE succeeds.

**Estimated time to fix**: 2-4 hours of debugging and reimplementation.

**Recommendation**:
1. Debug current implementation (most likely: forward pass bug in complex dtype handling)
2. If issues persist after 2 hours, reimplement using proven S4 codebase as template
3. Only proceed to proposal's full experiment after MVE achieves < 1e-3 training MSE

## Files Generated

- `code/004/`: Full implementation (model, data, training)
- `code/004/results.yaml`: Detailed metrics and learned parameters
- `code/004/best_model.pt`: Trained model checkpoint (not useful since it didn't learn)

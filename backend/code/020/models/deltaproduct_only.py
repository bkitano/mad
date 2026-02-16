"""
DeltaProduct-Only Baseline for Experiment 007.

This is the OH-DeltaProduct with use_oscillatory=False (no oscillatory component).
Only the Householder reflections are active as state transitions.

Without the oscillatory contraction, the state update is:
  h_t = H_t · h_{t-1} + B_t · x_t
where H_t = Π_j (I - β_{t,j} k_{t,j} k_{t,j}^T)

Expected behavior:
  - Achieves > 90% accuracy on S3 (Householder products can represent O(n))
  - BUT may have NaN risk with β ∈ (0,2) without oscillatory anchoring stability
"""

from models.oh_deltaproduct import OHDeltaProductClassifier


def DeltaProductOnlyClassifier(
    vocab_size: int,
    d_model: int,
    m: int,
    num_classes: int,
    n_h: int = 2,
    num_layers: int = 1,
    beta_range: str = 'full',
    dropout: float = 0.05,
    **kwargs,
):
    """
    DeltaProduct-only model (no oscillatory component).
    """
    return OHDeltaProductClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        m=m,
        num_classes=num_classes,
        n_h=n_h,
        num_layers=num_layers,
        dt=0.1,  # Unused since oscillatory is disabled
        omega_max=100.0,
        use_oscillatory=False,  # No oscillatory rotation
        beta_range=beta_range,
        dropout=dropout,
    )

"""
LinOSS-Only Baseline for Experiment 007.

This is the OH-DeltaProduct with n_h=0 (no Householder reflections).
Only the oscillatory rotation-contraction component is active.

Expected behavior on S3 task: FAIL (< 40% accuracy)
Reason: Oscillatory dynamics are diagonal (abelian), cannot track non-abelian S3.
LinOSS ∈ TC^0, but S3 composition requires NC^1 (non-abelian state manipulation).
"""

from models.oh_deltaproduct import OHDeltaProductClassifier


def LinOSSOnlyClassifier(
    vocab_size: int,
    d_model: int,
    m: int,
    num_classes: int,
    num_layers: int = 1,
    dt: float = 0.1,
    omega_max: float = 100.0,
    dropout: float = 0.05,
    **kwargs,
):
    """
    LinOSS-only model (n_h=0, oscillatory only).
    This is just OHDeltaProductClassifier with n_h=0.
    """
    return OHDeltaProductClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        m=m,
        num_classes=num_classes,
        n_h=0,  # No Householder reflections
        num_layers=num_layers,
        dt=dt,
        omega_max=omega_max,
        use_oscillatory=True,
        beta_range='full',
        dropout=dropout,
    )

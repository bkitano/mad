"""
Models for MVE 058: DSM-Fused Linear RNN Projection Chain

This module implements unfused and fused projection chain variants
for benchmarking the hypothesis that fusing input projection GEMMs
with their downstream activations reduces HBM round-trips and wall-clock time.
"""

from .projection_chain import (
    UnfusedProjectionChain,
    FusedProjectionChain,
    FullyFusedProjectionChain,
    UnfusedOutputChain,
    FusedOutputChain,
)

__all__ = [
    "UnfusedProjectionChain",
    "FusedProjectionChain",
    "FullyFusedProjectionChain",
    "UnfusedOutputChain",
    "FusedOutputChain",
]

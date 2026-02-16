"""
MVE 039: Warp-Specialized Pingpong Pipelining for Chunkwise Linear RNN Kernels

This module contains:
- chunkwise_gla.py: Triton kernels for chunkwise GLA forward pass
  - Baseline: Naive sequential implementation
  - Pipelined: Software-pipelined with double-buffering and overlapped ops
  - PyTorch reference: Pure PyTorch for correctness verification
"""

from models.chunkwise_gla import (
    pytorch_chunkwise_gla_forward,
    triton_chunkwise_gla_baseline,
    triton_chunkwise_gla_pipelined,
)

__all__ = [
    "pytorch_chunkwise_gla_forward",
    "triton_chunkwise_gla_baseline",
    "triton_chunkwise_gla_pipelined",
]

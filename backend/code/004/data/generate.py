"""
Synthetic data generation for Damped Oscillation Extrapolation task.

Task (from proposal MVE section):
- Input: Random impulse u_t = δ_{t=0} (unit impulse at t=0)
- Target: Generate damped sinusoid y_t = A e^{-ζωt} sin(ωt√(1-ζ²) + φ)
- Parameters: ω ~ U(0.01, 0.1), ζ ~ U(0.2, 0.8), random amplitude A and phase φ
- Train on T=128, test on T=512 (4× extrapolation)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def generate_damped_oscillation_sequence(
    seq_len: int,
    omega: float,
    zeta: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dt: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single damped oscillation sequence.

    Args:
        seq_len: Sequence length
        omega: Natural frequency (oscillation rate)
        zeta: Damping ratio (0=undamped, 1=critically damped)
        amplitude: Amplitude of oscillation
        phase: Phase offset
        dt: Time step

    Returns:
        u: Input impulse (seq_len, 1) - impulse at t=0
        y: Target damped sinusoid (seq_len, 1)
    """
    # Time vector
    t = torch.arange(seq_len, dtype=torch.float32) * dt

    # Input: unit impulse at t=0
    u = torch.zeros(seq_len, 1)
    u[0, 0] = 1.0

    # Target: damped sinusoid
    # y(t) = A * e^(-ζωt) * sin(ω√(1-ζ²)t + φ)
    damped_freq = omega * np.sqrt(1 - zeta**2)
    y = amplitude * torch.exp(-zeta * omega * t) * torch.sin(damped_freq * t + phase)
    y = y.unsqueeze(-1)  # (seq_len, 1)

    return u, y


class DampedOscillationDataset(Dataset):
    """
    Dataset of synthetic damped oscillation sequences.

    Each sample is a random damped oscillation with:
    - ω ~ U(0.01, 0.1)
    - ζ ~ U(0.2, 0.8)
    - A ~ U(0.5, 2.0)
    - φ ~ U(0, 2π)
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        omega_range: Tuple[float, float] = (0.01, 0.1),
        zeta_range: Tuple[float, float] = (0.2, 0.8),
        amplitude_range: Tuple[float, float] = (0.5, 2.0),
        dt: float = 0.01,
        seed: int = None,
    ):
        """
        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of each sequence
            omega_range: (min, max) for natural frequency
            zeta_range: (min, max) for damping ratio
            amplitude_range: (min, max) for amplitude
            dt: Time step
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.omega_range = omega_range
        self.zeta_range = zeta_range
        self.amplitude_range = amplitude_range
        self.dt = dt

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Pre-generate all sequences
        self.inputs = []
        self.targets = []
        self.params = []  # Store (omega, zeta) for analysis

        for _ in range(num_samples):
            # Sample random parameters
            omega = np.random.uniform(*omega_range)
            zeta = np.random.uniform(*zeta_range)
            amplitude = np.random.uniform(*amplitude_range)
            phase = np.random.uniform(0, 2 * np.pi)

            # Generate sequence
            u, y = generate_damped_oscillation_sequence(
                seq_len, omega, zeta, amplitude, phase, dt
            )

            self.inputs.append(u)
            self.targets.append(y)
            self.params.append((omega, zeta))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]:
        """
        Returns:
            u: Input impulse (seq_len, 1)
            y: Target damped oscillation (seq_len, 1)
            params: (omega, zeta) for this sequence
        """
        return self.inputs[idx], self.targets[idx], self.params[idx]


def generate_damped_oscillation_dataset(
    num_train: int = 8000,
    num_val: int = 1000,
    num_test: int = 1000,
    train_len: int = 128,
    test_len: int = 512,
    batch_size: int = 32,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate train/val/test dataloaders for damped oscillation task.

    Args:
        num_train: Number of training sequences
        num_val: Number of validation sequences
        num_test: Number of test sequences
        train_len: Sequence length for training (default: 128)
        test_len: Sequence length for testing (default: 512, 4× extrapolation)
        batch_size: Batch size
        **kwargs: Additional arguments for DampedOscillationDataset

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = DampedOscillationDataset(
        num_samples=num_train,
        seq_len=train_len,
        seed=42,
        **kwargs,
    )

    val_dataset = DampedOscillationDataset(
        num_samples=num_val,
        seq_len=train_len,
        seed=43,
        **kwargs,
    )

    # Test dataset uses longer sequences for extrapolation
    test_dataset = DampedOscillationDataset(
        num_samples=num_test,
        seq_len=test_len,
        seed=44,
        **kwargs,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for simplicity in MVE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader

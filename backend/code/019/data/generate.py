"""
Nested Periodic Detection Dataset
From Proposal 019 MVE.

Task: Classify which frequencies are present in a noisy multi-frequency signal.

Signal:
  x_t = a1*sin(2π*f1*t) + a2*sin(2π*f2*t) + a3*sin(2π*f3*t) + noise

Frequencies: f1=1/8 (fast), f2=1/32 (medium), f3=1/128 (slow)
Amplitudes: a1=1.0, a2=0.5, a3=0.25 (when present)
Labels: 8 classes (2^3 combinations of which frequencies are present)

This task directly exercises multi-scale modeling:
- Scale 1 (fast) should capture f1=1/8 (period=8 steps)
- Scale 2 (medium) should capture f2=1/32 (period=32 steps)
- Scale 3 (slow) should capture f3=1/128 (period=128 steps)
- Scale 4 (coupling) should integrate information across scales
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Frequencies for the nested periodic detection task
FREQUENCIES = [1/8, 1/32, 1/128]  # fast, medium, slow
AMPLITUDES = [1.0, 0.5, 0.25]     # decreasing amplitude for slower components
N_FREQ = len(FREQUENCIES)
N_CLASSES = 2 ** N_FREQ  # 8 classes (all combos of present/absent)


def label_to_freq_mask(label: int) -> list:
    """Convert label index to binary mask of which frequencies are present."""
    mask = []
    for i in range(N_FREQ):
        mask.append((label >> i) & 1)
    return mask


def freq_mask_to_label(mask: list) -> int:
    """Convert binary frequency mask to label index."""
    label = 0
    for i, m in enumerate(mask):
        label += m * (2 ** i)
    return label


class NestedPeriodicDataset(Dataset):
    """
    Synthetic dataset for nested periodic detection.

    Each sample is a time series with 0-3 sinusoidal components at different
    frequencies, plus Gaussian noise. The model must classify which frequencies
    are present.
    """
    def __init__(self, n_samples: int = 10000, seq_len: int = 512,
                 noise_std: float = 0.3, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.noise_std = noise_std

        rng = np.random.RandomState(seed)

        # Time axis
        t = np.arange(seq_len, dtype=np.float32)  # (seq_len,)

        # Generate samples
        self.signals = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
        self.labels = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            # Random label (which frequencies are present)
            label = rng.randint(0, N_CLASSES)
            mask = label_to_freq_mask(label)

            # Generate signal
            signal = np.zeros(seq_len, dtype=np.float32)
            for f_idx in range(N_FREQ):
                if mask[f_idx]:
                    # Add sinusoidal component with random phase
                    phase = rng.uniform(0, 2 * np.pi)
                    signal += AMPLITUDES[f_idx] * np.sin(
                        2 * np.pi * FREQUENCIES[f_idx] * t + phase
                    )

            # Add noise
            signal += rng.randn(seq_len).astype(np.float32) * noise_std

            self.signals[i, :, 0] = signal
            self.labels[i] = label

        # Convert to tensors
        self.signals = torch.from_numpy(self.signals)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def create_dataloaders(n_train: int = 8000, n_val: int = 1000, n_test: int = 1000,
                       seq_len: int = 512, noise_std: float = 0.3,
                       batch_size: int = 64, num_workers: int = 0):
    """Create train/val/test dataloaders for nested periodic detection."""
    train_dataset = NestedPeriodicDataset(
        n_samples=n_train, seq_len=seq_len, noise_std=noise_std, seed=42
    )
    val_dataset = NestedPeriodicDataset(
        n_samples=n_val, seq_len=seq_len, noise_std=noise_std, seed=123
    )
    test_dataset = NestedPeriodicDataset(
        n_samples=n_test, seq_len=seq_len, noise_std=noise_std, seed=456
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Quick test of data generation."""
    dataset = NestedPeriodicDataset(n_samples=100, seq_len=512)
    print(f"Dataset size: {len(dataset)}")
    print(f"Signal shape: {dataset.signals.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    print(f"Label distribution: {torch.bincount(dataset.labels, minlength=N_CLASSES)}")
    print(f"Signal range: [{dataset.signals.min():.2f}, {dataset.signals.max():.2f}]")

    # Check class balance
    for label in range(N_CLASSES):
        mask = label_to_freq_mask(label)
        count = (dataset.labels == label).sum().item()
        print(f"  Label {label} (f1={mask[0]}, f2={mask[1]}, f3={mask[2]}): {count} samples")

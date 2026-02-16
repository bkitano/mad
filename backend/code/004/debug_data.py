"""Debug data generation to understand the loss scale."""

import torch
import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
# import matplotlib.pyplot as plt
from data.generate import generate_damped_oscillation_sequence, generate_damped_oscillation_dataset

print("=" * 80)
print("DEBUG: Analyzing data and loss scale")
print("=" * 80)

# Generate a few samples
samples = []
for i in range(5):
    omega = np.random.uniform(0.01, 0.1)
    zeta = np.random.uniform(0.2, 0.8)
    u, y = generate_damped_oscillation_sequence(128, omega, zeta, amplitude=np.random.uniform(0.5, 2.0), dt=0.01)
    samples.append((u, y, omega, zeta))

    print(f"\nSample {i+1}: ω={omega:.4f}, ζ={zeta:.4f}")
    print(f"  Target y: mean={y.mean():.6f}, std={y.std():.6f}, max={y.abs().max():.6f}")

# Compute MSE if we predict zero
zero_mse_list = []
for u, y, _, _ in samples:
    zero_mse = ((y - 0)**2).mean()
    zero_mse_list.append(zero_mse.item())

print(f"\nMSE if predict 0: {np.mean(zero_mse_list):.6f} ± {np.std(zero_mse_list):.6f}")

# Compute MSE if we predict global mean
mean_y = torch.cat([y for _, y, _, _ in samples]).mean()
print(f"Global mean of y: {mean_y:.6f}")

mean_mse_list = []
for u, y, _, _ in samples:
    mean_mse = ((y - mean_y)**2).mean()
    mean_mse_list.append(mean_mse.item())

print(f"MSE if predict global mean: {np.mean(mean_mse_list):.6f} ± {np.std(mean_mse_list):.6f}")

# Now test on actual dataset
print("\n" + "=" * 80)
print("Testing on actual dataloader")
print("=" * 80)

train_loader, _, _ = generate_damped_oscillation_dataset(
    num_train=100, num_val=10, num_test=10,
    train_len=128, test_len=512, batch_size=32
)

# Get first batch
for u_batch, y_batch, params in train_loader:
    print(f"Batch shape: u={u_batch.shape}, y={y_batch.shape}")
    print(f"u stats: mean={u_batch.mean():.6f}, std={u_batch.std():.6f}")
    print(f"y stats: mean={y_batch.mean():.6f}, std={y_batch.std():.6f}, max={y_batch.abs().max():.6f}")

    # MSE with different predictions
    zero_mse = ((y_batch - 0)**2).mean()
    mean_mse = ((y_batch - y_batch.mean())**2).mean()

    print(f"MSE if predict 0: {zero_mse:.6f}")
    print(f"MSE if predict mean: {mean_mse:.6f}")

    break

# Plot a few examples - skipped (no matplotlib)

print("\n" + "=" * 80)
print("Analysis complete")
print("=" * 80)

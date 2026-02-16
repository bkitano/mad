"""Quick debug script to understand why the model isn't learning."""

import torch
import numpy as np
from models import OscillatoryDPLRSSM
from data.generate import generate_damped_oscillation_sequence

# Create a single sample
print("=" * 80)
print("DEBUG: Testing Oscillatory-DPLR SSM on single sample")
print("=" * 80)

# Generate one damped oscillation
seq_len = 128
omega = 0.05  # mid-range frequency
zeta = 0.5    # mid-range damping
u, y_target = generate_damped_oscillation_sequence(seq_len, omega, zeta, amplitude=1.0, phase=0.0, dt=0.01)

print(f"\nInput (impulse):")
print(f"  Shape: {u.shape}")
print(f"  First 5 timesteps: {u[:5, 0].tolist()}")
print(f"  Sum: {u.sum().item()}")

print(f"\nTarget (damped oscillation):")
print(f"  Shape: {y_target.shape}")
print(f"  First 10 timesteps: {y_target[:10, 0].tolist()}")
print(f"  Mean: {y_target.mean().item():.6f}, Std: {y_target.std().item():.6f}")
print(f"  Min: {y_target.min().item():.6f}, Max: {y_target.max().item():.6f}")

# Create model
model = OscillatoryDPLRSSM(n=16, r=2, d_input=1, d_output=1, dt=0.01)

print(f"\n Model:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"  Initial ω: {model.get_omega().detach().numpy()}")
print(f"  Initial ζ: {model.get_zeta().detach().numpy()}")

# Forward pass
u_batch = u.unsqueeze(0)  # (1, seq_len, 1)
y_pred = model(u_batch)

print(f"\nModel output:")
print(f"  Shape: {y_pred.shape}")
print(f"  First 10 timesteps: {y_pred[0, :10, 0].detach().tolist()}")
print(f"  Mean: {y_pred.mean().item():.6f}, Std: {y_pred.std().item():.6f}")
print(f"  Min: {y_pred.min().item():.6f}, Max: {y_pred.max().item():.6f}")

# Check if output is all zeros or constant
if torch.allclose(y_pred, torch.zeros_like(y_pred), atol=1e-6):
    print("\n⚠️  WARNING: Model output is all zeros!")
elif y_pred.std() < 1e-6:
    print(f"\n⚠️  WARNING: Model output is constant: {y_pred.mean().item()}")
else:
    print("\n✓ Model output is non-trivial")

# Compute loss
loss = torch.nn.functional.mse_loss(y_pred[0], y_target)
print(f"\nLoss (MSE): {loss.item():.6f}")

# Expected loss if model outputs zeros
zero_loss = torch.nn.functional.mse_loss(torch.zeros_like(y_target), y_target)
print(f"Loss if output=0: {zero_loss.item():.6f}")

# Expected loss if model outputs mean
mean_loss = torch.nn.functional.mse_loss(torch.full_like(y_target, y_target.mean()), y_target)
print(f"Loss if output=mean: {mean_loss.item():.6f}")

if abs(loss.item() - zero_loss.item()) < 1e-6:
    print("\n🔴 ISSUE: Model is outputting zeros!")
elif abs(loss.item() - mean_loss.item()) < 1e-6:
    print("\n🔴 ISSUE: Model is outputting constant (mean)!")
else:
    print("\n✓ Model output is different from trivial baselines")

# Test gradient flow
print("\n" + "=" * 80)
print("Testing gradient flow")
print("=" * 80)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(5):
    optimizer.zero_grad()
    y_pred = model(u_batch)
    loss = torch.nn.functional.mse_loss(y_pred[0], y_target)
    loss.backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    optimizer.step()

    print(f"Step {step}: Loss={loss.item():.6f}, Grad norms: {grad_norms}")

print("\n" + "=" * 80)
print("DEBUG complete - check output above for issues")
print("=" * 80)

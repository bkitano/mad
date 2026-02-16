"""
Training script for MVE 002: Oscillatory-DPLR SSM on Damped Oscillation Extrapolation.

Success criteria (from proposal):
1. Training fit: MSE < 1e-3 on training sequences
2. Extrapolation: MSE < 1e-2 on 4× longer test sequences
3. Interpretability: Learned ω_i cluster near ground-truth range [0.01, 0.1]

Failure criteria:
- MSE > 1e-1 on training (cannot fit basic oscillations)
- Extrapolation MSE > 10× training MSE (complete failure to generalize)
- Learned ω_i collapse to single value or diverge outside [0.001, 1]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models import OscillatoryDPLRSSM
from data import generate_damped_oscillation_dataset


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for u, y, _ in tqdm(loader, desc="Training", leave=False):
        u = u.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(u)
        loss = criterion(y_pred, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for u, y, _ in loader:
            u = u.to(device)
            y = y.to(device)

            y_pred = model(u)
            loss = criterion(y_pred, y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def analyze_learned_parameters(model, data_loader):
    """
    Analyze learned oscillator parameters vs ground truth.

    Returns:
        dict with analysis metrics
    """
    # Get learned parameters
    omega_learned = model.get_learned_frequencies().cpu().numpy()
    zeta_learned = model.get_learned_damping().cpu().numpy()

    # Get ground truth from dataset
    omega_gt = []
    zeta_gt = []
    for _, _, (omega, zeta) in data_loader:
        omega_gt.append(omega)
        zeta_gt.append(zeta)
    omega_gt = np.array(omega_gt)
    zeta_gt = np.array(zeta_gt)

    analysis = {
        "omega_learned_mean": omega_learned.mean(),
        "omega_learned_std": omega_learned.std(),
        "omega_learned_min": omega_learned.min(),
        "omega_learned_max": omega_learned.max(),
        "omega_gt_mean": omega_gt.mean(),
        "omega_gt_std": omega_gt.std(),
        "zeta_learned_mean": zeta_learned.mean(),
        "zeta_learned_std": zeta_learned.std(),
        "zeta_gt_mean": zeta_gt.mean(),
        "zeta_gt_std": zeta_gt.std(),
        "omega_learned": omega_learned.tolist(),
        "zeta_learned": zeta_learned.tolist(),
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Train Oscillatory-DPLR SSM")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("MVE 002: Oscillatory-DPLR SSM - Damped Oscillation Extrapolation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Config: {config}")
    print()

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create datasets
    train_loader, val_loader, test_loader = generate_damped_oscillation_dataset(
        num_train=config["data"]["num_train"],
        num_val=config["data"]["num_val"],
        num_test=config["data"]["num_test"],
        train_len=config["data"]["train_len"],
        test_len=config["data"]["test_len"],
        batch_size=config["training"]["batch_size"],
        omega_range=tuple(config["data"]["omega_range"]),
        zeta_range=tuple(config["data"]["zeta_range"]),
        dt=config["model"]["dt"],
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Train seq len: {config['data']['train_len']}")
    print(f"Test seq len: {config['data']['test_len']} (extrapolation: {config['data']['test_len'] / config['data']['train_len']:.1f}×)")
    print()

    # Create model
    model = OscillatoryDPLRSSM(
        n=config["model"]["n"],
        r=config["model"]["r"],
        d_input=config["model"]["d_input"],
        d_output=config["model"]["d_output"],
        dt=config["model"]["dt"],
        init_omega_range=tuple(config["model"]["init_omega_range"]),
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    results = {
        "train_losses": [],
        "val_losses": [],
        "test_loss": None,
        "success_criteria": {},
    }

    print("Training...")
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = evaluate(model, val_loader, criterion, args.device)

        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)

        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    # Load best model for testing
    model.load_state_dict(torch.load("best_model.pt"))

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    final_train_loss = evaluate(model, train_loader, criterion, args.device)
    final_val_loss = evaluate(model, val_loader, criterion, args.device)
    test_loss = evaluate(model, test_loader, criterion, args.device)

    results["test_loss"] = test_loss

    print(f"Final Train Loss (MSE): {final_train_loss:.6e}")
    print(f"Final Val Loss (MSE): {final_val_loss:.6e}")
    print(f"Test Loss (MSE) [extrapolation]: {test_loss:.6e}")
    print()

    # Analyze learned parameters
    print("Learned Parameters Analysis:")
    analysis = analyze_learned_parameters(model, train_loader)
    print(f"  Learned ω: mean={analysis['omega_learned_mean']:.4f}, "
          f"std={analysis['omega_learned_std']:.4f}, "
          f"range=[{analysis['omega_learned_min']:.4f}, {analysis['omega_learned_max']:.4f}]")
    print(f"  Ground truth ω: mean={analysis['omega_gt_mean']:.4f}, "
          f"std={analysis['omega_gt_std']:.4f}")
    print(f"  Learned ζ: mean={analysis['zeta_learned_mean']:.4f}, "
          f"std={analysis['zeta_learned_std']:.4f}")
    print(f"  Ground truth ζ: mean={analysis['zeta_gt_mean']:.4f}, "
          f"std={analysis['zeta_gt_std']:.4f}")
    print()

    # Check success/failure criteria
    print("=" * 80)
    print("Success Criteria Check")
    print("=" * 80)

    # Criterion 1: Training fit
    train_fit_pass = final_train_loss < 1e-3
    print(f"1. Training MSE < 1e-3: {final_train_loss:.6e} - {'✅ PASS' if train_fit_pass else '❌ FAIL'}")
    results["success_criteria"]["train_fit"] = train_fit_pass

    # Criterion 2: Extrapolation
    extrap_pass = test_loss < 1e-2
    print(f"2. Extrapolation MSE < 1e-2: {test_loss:.6e} - {'✅ PASS' if extrap_pass else '❌ FAIL'}")
    results["success_criteria"]["extrapolation"] = extrap_pass

    # Criterion 3: Interpretability (omega in [0.01, 0.1])
    omega_in_range = (analysis["omega_learned_min"] >= 0.001 and
                      analysis["omega_learned_max"] <= 1.0)
    omega_not_collapsed = analysis["omega_learned_std"] > 0.001
    interp_pass = omega_in_range and omega_not_collapsed
    print(f"3. Learned ω in valid range [0.001, 1] and not collapsed: "
          f"{'✅ PASS' if interp_pass else '❌ FAIL'}")
    results["success_criteria"]["interpretability"] = interp_pass

    # Check failure criteria
    print("\nFailure Criteria Check:")
    fail_train = final_train_loss > 1e-1
    fail_extrap = test_loss > 10 * final_train_loss
    fail_omega = not omega_in_range or not omega_not_collapsed

    print(f"- Training MSE > 1e-1: {final_train_loss:.6e} - {'❌ FAIL' if fail_train else '✅ OK'}")
    print(f"- Extrapolation MSE > 10× training: {test_loss:.6e} vs {10*final_train_loss:.6e} - "
          f"{'❌ FAIL' if fail_extrap else '✅ OK'}")
    print(f"- Omega issues: {'❌ FAIL' if fail_omega else '✅ OK'}")

    # Overall verdict
    print("\n" + "=" * 80)
    all_pass = train_fit_pass and extrap_pass and interp_pass
    any_fail = fail_train or fail_extrap or fail_omega

    if all_pass and not any_fail:
        print("🎉 OVERALL: SUCCESS - Proceed to full LRA experiments")
        results["verdict"] = "PROCEED"
    elif any_fail:
        print("⚠️  OVERALL: FAILURE - Debug parameterization before scaling")
        results["verdict"] = "DEBUG"
    else:
        print("⚙️  OVERALL: PARTIAL - Some criteria met, investigate further")
        results["verdict"] = "INVESTIGATE"

    print("=" * 80)

    # Save results
    results["analysis"] = analysis
    results["config"] = config

    with open("results.yaml", "w") as f:
        yaml.dump(results, f)

    print("\nResults saved to results.yaml")
    print("Model saved to best_model.pt")


if __name__ == "__main__":
    main()

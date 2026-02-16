"""
CLI launcher for MVE 001: D4 State Tracking

Usage:
    # Run locally
    uv run python -m train.launch --config configs/cs_neg_deltanet.yaml --local

    # Run on Modal (detached)
    uv run python -m train.launch --config configs/cs_neg_deltanet.yaml

    # Run on Modal (wait for completion)
    uv run python -m train.launch --config configs/cs_neg_deltanet.yaml --no-detach
"""

import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description="Launch MVE 001 training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--local", action="store_true", help="Run locally instead of on Modal")
    parser.add_argument("--no-detach", action="store_true", help="Wait for Modal job to complete")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)

    if args.local:
        # Run locally with accelerate
        cmd = [
            "uv", "run", "accelerate", "launch",
            "-m", "train.run_config",
            "--config", config_path,
        ]
    else:
        # Run on Modal
        cmd = ["uv", "run", "modal", "run"]
        if not args.no_detach:
            cmd.append("--detach")
        cmd.extend(["-m", "train.modal_config", "--config", config_path])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

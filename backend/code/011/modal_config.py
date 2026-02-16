"""
Modal deployment for MVE 011: Neumann Resolvent Kernel Accuracy Test

This MVE tests the accuracy and speed of Neumann series approximation
for DPLR SSM resolvent computation. No training involved - purely a
kernel accuracy and numerical stability benchmark.

Usage:
    modal run --detach modal_config.py --config config.yaml
"""

import argparse
from pathlib import Path
from modal import App, Image, Volume
import yaml
import os
import sys


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Parse config at module load time
N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 600

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv)

if not _is_modal_container and _has_config_arg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    deployment_config = config.get("deployment", {})

    N_GPUS = deployment_config.get("n_gpus", 1)
    GPU_TYPE = deployment_config.get("gpu_type", "T4")
    TIMEOUT = deployment_config.get("timeout_seconds", 600)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
    )
    .add_local_dir(Path(__file__).parent, remote_path="/root/app", copy=True)
)

volume = Volume.from_name("mve-011-results", create_if_missing=True)

app = App("mve-011-neumann-resolvent")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote experiment."""
    with open(config, "r") as f:
        config_content = f.read()

    run_experiment.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    timeout=TIMEOUT,
    volumes={"/results": volume},
)
def run_experiment(config_content: str):
    """Run the Neumann resolvent kernel accuracy experiment."""
    import subprocess
    import tempfile

    os.chdir("/root/app")

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    cmd = ["python", "run_experiment.py", "--config", config_path]
    print(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")
    exit_code = process.wait()

    # Copy results to volume
    import shutil
    results_src = "/root/app/results.json"
    results_dst = "/results/results.json"
    if os.path.exists(results_src):
        shutil.copy2(results_src, results_dst)
        volume.commit()
        print(f"\nResults saved to Modal volume at {results_dst}")

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

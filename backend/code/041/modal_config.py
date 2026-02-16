"""
Modal deployment for MVE 041: EVT Joint Forward-Backward Graph Partitioning

This runs the benchmarking and training script on a GPU via Modal.

Usage:
    uv run modal run --detach modal_config.py --config config.yaml
"""

import argparse
from pathlib import Path
from modal import App, Image, Secret, Volume
import os
import sys


N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 3600

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv)

if not _is_modal_container and _has_config_arg:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    deployment_config = config.get("deployment", {})
    N_GPUS = deployment_config.get("n_gpus", 1)
    GPU_TYPE = deployment_config.get("gpu_type", "T4")
    TIMEOUT = deployment_config.get("timeout_seconds", 3600)


image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
        "triton>=3.0.0",
        "wandb>=0.17.6",
    )
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/root/app",
        copy=True,
    )
)

volume = Volume.from_name("mve-041-results", create_if_missing=True)

app = App("mve-041-evt-joint-fwd-bwd")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote training function."""
    with open(config, "r") as f:
        config_content = f.read()

    train_with_config.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    secrets=[Secret.from_name("wandb-secret")],
    timeout=TIMEOUT,
    volumes={"/results": volume},
)
def train_with_config(config_content: str):
    """Run training with the provided config content."""
    import subprocess
    import tempfile

    os.chdir("/root/app")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    cmd = ["python", "train.py", "--config", config_path]
    print(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")
    exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

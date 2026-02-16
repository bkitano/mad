"""
Modal deployment for MVE 057: FlashRNN-Style Fused Inter-Chunk State Recurrence

This is a kernel microbenchmark comparing:
1. Baseline: Sequential inter-chunk scan with HBM round-trips
2. Proposed: FlashRNN-style fused scan with state in registers

Usage:
    uv run modal run --detach modal_config.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from modal import App, Image, Secret, Volume


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Parse config at module load time to configure Modal resources
N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 1800

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(
    arg.startswith("--config=") for arg in sys.argv
)

if not _is_modal_container and _has_config_arg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    deployment_config = config.get("deployment", {})

    N_GPUS = deployment_config.get("n_gpus", 1)
    GPU_TYPE = deployment_config.get("gpu_type", "T4")
    TIMEOUT = deployment_config.get("timeout_seconds", 1800)

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
        Path(__file__).parent, remote_path="/root/app", copy=True
    )
)

volume = Volume.from_name("mve-057-results", create_if_missing=True)

app = App("mve-057-flashrnn-fused-scan")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote benchmark function."""
    with open(config, "r") as f:
        config_content = f.read()

    run_benchmark_remote.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    secrets=[Secret.from_name("wandb-secret")],
    timeout=TIMEOUT,
    volumes={"/root/results": volume},
)
def run_benchmark_remote(config_content: str):
    """Run the kernel benchmark on Modal GPU."""
    import subprocess
    import tempfile

    os.chdir("/root/app")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    cmd = ["python", "train.py", "--config", config_path]

    print(f"Running command: {' '.join(cmd)}")
    print(f"GPU: {GPU_TYPE}, N_GPUs: {N_GPUS}")
    print()

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")
    exit_code = process.wait()

    # Commit results to volume
    volume.commit()

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

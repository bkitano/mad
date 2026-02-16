"""
Modal deployment for MVE 008: Cosine-Reweighted Log-Linear Attention

GPU Selection Guide:
- T4: Default for MVEs, small models (< 100M params), short training (< 1 hour)
- A100: Larger models (100M-1B params), longer training
- H100: Very large models (> 1B params)

Usage:
    modal run --detach modal_config.py --config config.yaml
"""

import argparse
from pathlib import Path
from modal import App, Image, Secret, Volume
import yaml
import os
import sys


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Parse config at module load time to configure Modal resources
N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 3600

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
    TIMEOUT = deployment_config.get("timeout_seconds", 3600)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
        "tqdm>=4.65.0",
        "wandb>=0.17.6",
    )
    .add_local_dir(
        Path(__file__).parent, remote_path="/root/app", copy=True
    )
)

volume = Volume.from_name("mve-008-results", create_if_missing=True)

app = App("mve-008-cos-log-linear")


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

    def _exec_subprocess(cmd: list[str]):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                print(line.decode(), end="")
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

    cmd = ["python", "train.py", "--config", config_path]

    print(f"Running command: {' '.join(cmd)}")
    _exec_subprocess(cmd)

    # Copy results to volume
    import shutil
    if os.path.exists("results.json"):
        shutil.copy("results.json", "/results/results.json")
        volume.commit()
        print("Results saved to volume.")

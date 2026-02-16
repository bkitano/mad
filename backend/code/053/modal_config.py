"""
Modal deployment for MVE 053: MLA-Inspired Latent State Compression

GPU Selection Guide:
- T4: Default for MVEs, small models (< 100M params), short training (< 1 hour)
- A100: Larger models (100M-1B params), longer training, or when T4 is too slow
- H100: Very large models (> 1B params) or when A100 is insufficient

Usage:
    modal run --detach modal_config.py --config config.yaml
"""

import argparse
from pathlib import Path
from modal import App, Image, Secret, Volume
import yaml


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Parse config at module load time to configure Modal resources
import os
import sys

N_GPUS = 1
GPU_TYPE = "T4"  # Default to T4 for MVEs
TIMEOUT = 3600  # 1 hour

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
    .add_local_dir(Path(__file__).parent, remote_path="/root/app", copy=True)
    .run_commands("pip install -e /root/app")
)

volume = Volume.from_name("mve-053-results", create_if_missing=True)

app = App("mve-053-latent-state-compression")


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
    volumes={"/root/results": volume},
)
def train_with_config(config_content: str):
    """Run training with the provided config content."""
    import subprocess
    import os
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

    results_file = "/root/app/results.json"
    if os.path.exists(results_file):
        shutil.copy(results_file, "/root/results/results.json")
        print("Results copied to volume")

"""
Modal deployment for MVE 029: Circulant FAVOR+ Linear Attention

Runs four attention variants on associative recall task:
1. Dense FAVOR+ (baseline random features)
2. C-FAVOR+ (circulant random features via FFT)
3. ReLU linear attention (no feature map baseline)
4. Softmax attention (quality ceiling)

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


# Parse config at module load time to configure Modal resources
N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 1800  # 30 minutes

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
    TIMEOUT = deployment_config.get("timeout_seconds", 1800)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
        "tqdm>=4.65.0",
    )
    .add_local_dir(Path(__file__).parent, remote_path="/root/app", copy=True)
)

volume = Volume.from_name("mve-029-results", create_if_missing=True)

app = App("mve-029-circulant-favor-plus")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote training function."""
    with open(config, "r") as f:
        config_content = f.read()

    train_remote.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    timeout=TIMEOUT,
    volumes={"/results": volume},
)
def train_remote(config_content: str):
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
    output_lines = []
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            decoded = line.decode()
            print(decoded, end="")
            output_lines.append(decoded)

    exit_code = process.wait()

    # Save results to volume
    results_text = "".join(output_lines)
    with open("/results/029_output.txt", "w") as f:
        f.write(results_text)

    volume.commit()

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

    print("\nResults saved to /results/029_output.txt")

"""
Modal deployment for MVE 058: DSM-Fused Linear RNN Projection Chain

This is a microbenchmark experiment — no training, just CUDA timing of
projection chain variants (unfused vs fused).

GPU Selection:
- A100: Needed for bf16 support and good GEMM performance
- H100: Ideal (has DSM support) but A100 is sufficient for the EVT fusion test

Usage:
    uv run modal run --detach modal_config.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

from modal import App, Image, Secret, Volume

# Parse config at module load time to configure Modal resources
N_GPUS = 1
GPU_TYPE = "A100"
TIMEOUT = 1800  # 30 minutes

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(
    arg.startswith("--config=") for arg in sys.argv
)

if not _is_modal_container and _has_config_arg:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    deployment_config = config.get("deployment", {})
    N_GPUS = deployment_config.get("n_gpus", 1)
    GPU_TYPE = deployment_config.get("gpu_type", "A100")
    TIMEOUT = deployment_config.get("timeout_seconds", 1800)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
        "wandb>=0.17.6",
    )
    .add_local_file(
        Path(__file__).parent / "train.py",
        remote_path="/root/app/train.py",
        copy=True,
    )
    .add_local_dir(
        Path(__file__).parent / "models",
        remote_path="/root/app/models",
        copy=True,
    )
)

volume = Volume.from_name("mve-058-results", create_if_missing=True)

app = App("mve-058-dsm-fused-projection-chain")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote benchmark function."""
    with open(config, "r") as f:
        config_content = f.read()

    run_benchmark.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    secrets=[Secret.from_name("wandb-secret")],
    timeout=TIMEOUT,
    volumes={"/root/results": volume},
)
def run_benchmark(config_content: str):
    """Run the projection chain microbenchmark on Modal GPU."""
    import subprocess
    import tempfile

    os.chdir("/root/app")

    # Write config to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/root/app"
    ) as f:
        f.write(config_content)
        config_path = f.name

    # Run the benchmark
    cmd = ["python", "train.py", "--config", config_path]
    print(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")

    exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

    # Commit the volume to persist results
    volume.commit()
    print("\nResults saved to Modal volume 'mve-058-results'")

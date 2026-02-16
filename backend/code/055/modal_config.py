"""
Modal deployment for MVE 055: WY-All-Scan Microbenchmark

This runs a multi-GPU benchmark comparing WY-All-Scan, LASP-2, and ZeCO-GLA
communication primitives for sequence parallelism.

Requires multiple GPUs on a single node with NVLink for P2P communication.

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

N_GPUS = 8          # Default: 8 GPUs for full microbenchmark
GPU_TYPE = "A100"    # A100 for NVLink P2P support
TIMEOUT = 1800       # 30 minutes max

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv)

if not _is_modal_container and _has_config_arg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    deployment_config = config.get("deployment", {})

    N_GPUS = deployment_config.get("n_gpus", 8)
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
    .add_local_dir(Path(__file__).parent, remote_path="/root/app", copy=True)
)

volume = Volume.from_name("mve-055-results", create_if_missing=True)

app = App("mve-055-wy-allscan-sp")


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
    volumes={"/results": volume},
)
def run_benchmark(config_content: str):
    """Run the multi-GPU benchmark with the provided config."""
    import subprocess
    import os
    import tempfile

    os.chdir("/root/app")

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    def _exec_subprocess(cmd: list[str]):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                print(line.decode(), end="")
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

    # Run benchmarks at different GPU counts: P=2, P=4, P=8
    # Only run up to N_GPUS
    gpu_counts = [p for p in [2, 4, 8] if p <= int(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").count(",") + 1)]

    # If we have fewer GPUs than 8, adjust
    import torch
    n_available = torch.cuda.device_count()
    gpu_counts = [p for p in [2, 4, 8] if p <= n_available]

    print(f"Available GPUs: {n_available}")
    print(f"Will benchmark P = {gpu_counts}")
    print(f"GPU: {torch.cuda.get_device_name(0) if n_available > 0 else 'None'}")

    for n_gpus in gpu_counts:
        print(f"\n{'#'*70}")
        print(f"# Running benchmark with P={n_gpus} GPUs")
        print(f"{'#'*70}\n")

        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            "--master_addr=localhost",
            "--master_port=29500",
            "benchmark.py",
            "--config", config_path,
        ]

        print(f"Command: {' '.join(cmd)}")
        try:
            _exec_subprocess(cmd)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Benchmark with P={n_gpus} failed with exit code {e.returncode}")
            continue

    print("\nAll benchmarks completed!")

    # Commit volume to persist results
    volume.commit()

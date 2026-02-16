"""
Modal deployment for MVE 002: SSD-DeltaNet Benchmark

This is a pure forward-pass throughput benchmark (no training).
Runs on T4 GPU — only needs ~5 minutes.

Usage:
    modal run --detach modal_config.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

from modal import App, Image, Volume

# Parse config at module load time for Modal resource configuration
N_GPUS = 1
GPU_TYPE = "T4"
TIMEOUT = 1800  # 30 minutes

_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv)

if not _is_modal_container and _has_config_arg:
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
    )
    .add_local_dir(Path(__file__).parent, remote_path="/root/app", copy=True)
)

volume = Volume.from_name("mve-002-results", create_if_missing=True)

app = App("mve-002-ssd-deltanet")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote benchmark."""
    with open(config, "r") as f:
        config_content = f.read()

    run_benchmark.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    timeout=TIMEOUT,
    volumes={"/results": volume},
)
def run_benchmark(config_content: str):
    """Run the SSD-DeltaNet benchmark on GPU."""
    import subprocess
    import tempfile

    os.chdir("/root/app")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    cmd = ["python", "train.py", "--config", config_path]
    print(f"Running: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")
    exit_code = process.wait()

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(cmd))

    # Copy results to volume
    results_src = "/root/app/benchmark_results.json"
    if os.path.exists(results_src):
        import shutil
        shutil.copy(results_src, "/results/benchmark_results.json")
        volume.commit()
        print(f"Results saved to Modal volume: mve-002-results/benchmark_results.json")

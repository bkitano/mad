"""
Artifact upload helper — uploads experiment artifacts to Hugging Face Hub.

Tar-gzips the experiment directory and uploads to a configured HF dataset repo.
Returns a direct download URL.
"""

import io
import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi


def upload_artifacts(experiment_id: str, experiments_dir: Path) -> str:
    """
    Tar-gz experiments/{experiment_id}/, upload to Hugging Face Hub,
    return a direct download URL.

    Args:
        experiment_id: e.g. "042" or "042-r2"
        experiments_dir: base experiments directory (e.g. /workspace/experiments)

    Returns:
        Direct download URL string for the artifacts tarball

    Raises:
        ValueError: if required env vars not set
        Exception: on upload failure
    """
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")

    if not hf_token or not repo_id:
        raise ValueError("HF_TOKEN and HF_REPO_ID env vars required")

    # Create tarball in memory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        src = experiments_dir / experiment_id
        if src.exists():
            tar.add(src, arcname=experiment_id)
    buf.seek(0)

    path_in_repo = f"experiments/{experiment_id}/artifacts.tar.gz"

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"

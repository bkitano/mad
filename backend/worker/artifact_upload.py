"""
Artifact upload helper — uploads experiment artifacts to Supabase Storage.

Tar-gzips the experiment directory and uploads to configured Supabase bucket.
Returns a 7-day signed URL for download.
"""

import io
import os
import tarfile
from pathlib import Path

from supabase import create_client

SIGNED_URL_EXPIRY = 60 * 60 * 24 * 7  # 7 days


def upload_artifacts(experiment_id: str, experiments_dir: Path) -> str:
    """
    Tar-gz experiments/{experiment_id}/, upload to Supabase Storage,
    return a 7-day signed URL.

    Args:
        experiment_id: e.g. "042" or "042-r2"
        experiments_dir: base experiments directory (e.g. /workspace/experiments)

    Returns:
        Signed URL string for downloading the artifacts tarball

    Raises:
        ValueError: if required env vars not set or directory doesn't exist
        Exception: on Supabase upload failure
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    bucket = os.environ.get("MAD_SUPABASE_BUCKET", "mad-experiments")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY env vars required")

    client = create_client(supabase_url, supabase_key)

    # Create tarball in memory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        src = experiments_dir / experiment_id
        if src.exists():
            tar.add(src, arcname=experiment_id)
    buf.seek(0)

    # Upload to Supabase Storage
    object_path = f"experiments/{experiment_id}/artifacts.tar.gz"
    client.storage.from_(bucket).upload(
        object_path,
        buf.getvalue(),
        {"content-type": "application/gzip", "upsert": "true"},
    )

    # Create and return signed URL
    result = client.storage.from_(bucket).create_signed_url(object_path, SIGNED_URL_EXPIRY)
    return result["signedURL"]

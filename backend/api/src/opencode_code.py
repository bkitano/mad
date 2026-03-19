"""
Fetch experiment code from OpenCode API on the worker that ran the experiment.

Used when viewing code during an active experiment (worker still running).
OpenCode's /file and /file/content endpoints operate on the workspace filesystem.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OpenCodeProxyError(Exception):
    """Connection/HTTP error reaching the worker's opencode server."""
    pass


class OpenCodeNoFiles(Exception):
    """Proxy succeeded but the code directory is empty or doesn't exist yet."""
    pass


# Directories to skip when listing workspace root — these are infrastructure,
# not experiment code written by the agent.
_SKIP_DIRS = {"wandb", "proposals", "experiments", "code", ".opencode", "__pycache__", ".git"}


def get_code_from_opencode(
    experiment_id: str,
    opencode_url: str,
) -> tuple[dict, dict[str, str]]:
    """
    Fetch experiment code from worker's OpenCode API.

    The agent writes code directly into the workspace root (/workspace/),
    not into code/{experiment_id}/.  We list the root and skip known
    infrastructure directories (wandb/, proposals/, etc.).

    Returns (manifest_info, files).
    Raises OpenCodeProxyError on connection/HTTP failure.
    Raises OpenCodeNoFiles when the workspace has no code files yet.
    """
    url = opencode_url.rstrip("/")

    timeout = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=5.0)
    with httpx.Client(timeout=timeout) as http:
        # List the workspace root — opencode runs with cwd=/workspace
        try:
            resp = http.get(f"{url}/file", params={"path": "."})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(
                "opencode file listing failed: url=%s experiment=%s error=%s: %s",
                url, experiment_id, type(exc).__name__, exc,
            )
            raise OpenCodeProxyError(f"{type(exc).__name__}: {exc}") from exc

        nodes = data if isinstance(data, list) else data.get("children") or data.get("nodes") or []
        file_list: list[tuple[str, int]] = []

        def walk(ns: list, prefix: str, current_path: str) -> None:
            """Recurse into directories via /file requests, skipping infrastructure dirs."""
            for node in ns:
                if not isinstance(node, dict):
                    continue
                node_type = node.get("type", "file")
                name = node.get("name") or (node.get("path") or "").split("/")[-1]
                if not name or name in (".", ".."):
                    continue
                rel = f"{prefix}{name}" if prefix else name
                if node_type == "directory":
                    # Skip infrastructure directories at root level
                    if not prefix and name in _SKIP_DIRS:
                        continue
                    subpath = f"{current_path}/{name}".lstrip("/")
                    try:
                        r = http.get(f"{url}/file", params={"path": subpath})
                        r.raise_for_status()
                        subnodes = r.json()
                        subnodes = subnodes if isinstance(subnodes, list) else subnodes.get("children") or subnodes.get("nodes") or []
                        walk(subnodes, f"{rel}/", subpath)
                    except Exception as exc:
                        logger.warning(
                            "opencode directory walk failed: url=%s path=%s error=%s: %s",
                            url, subpath, type(exc).__name__, exc,
                        )
                else:
                    sz = node.get("size") or node.get("length") or 0
                    file_list.append((rel, sz))

        walk(nodes, "", ".")

        if not file_list:
            logger.info(
                "opencode workspace has no code files: url=%s experiment=%s raw_response=%s",
                url, experiment_id, data,
            )
            raise OpenCodeNoFiles(
                f"Worker workspace is empty — the agent may not have written code yet"
            )

        files: dict[str, str] = {}
        for rel_path, _ in file_list:
            try:
                cr = http.get(f"{url}/file/content", params={"path": rel_path})
                cr.raise_for_status()
                content_data = cr.json()
                if isinstance(content_data, str):
                    content = content_data
                elif isinstance(content_data, dict):
                    content = content_data.get("content") or content_data.get("text") or ""
                else:
                    content = ""
                files[rel_path] = content
            except Exception as exc:
                logger.warning(
                    "opencode file content failed: url=%s path=%s error=%s: %s",
                    url, rel_path, type(exc).__name__, exc,
                )
                continue

        if not files:
            raise OpenCodeNoFiles(
                f"Found {len(file_list)} file entries but failed to read any content"
            )

        total_bytes = sum(len(c.encode("utf-8")) for c in files.values())
        manifest_info = {
            "total_files": len(files),
            "total_size_bytes": total_bytes,
            "files": [{"path": p, "size": len(c.encode("utf-8"))} for p, c in files.items()],
        }
        return (manifest_info, files)

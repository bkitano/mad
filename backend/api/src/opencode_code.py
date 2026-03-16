"""
Fetch experiment code from OpenCode API on the worker that ran the experiment.

Used when viewing code during an active experiment (worker still running).
OpenCode's /file and /file/content endpoints operate on the workspace filesystem.
"""

from __future__ import annotations

import httpx


def get_code_from_opencode(
    experiment_id: str,
    opencode_url: str,
) -> tuple[dict, dict[str, str]] | None:
    """
    Fetch experiment code from worker's OpenCode API.
    Returns (manifest_info, files) or None on any error.
    """
    url = opencode_url.rstrip("/")
    code_prefix = f"code/{experiment_id}/"

    with httpx.Client(timeout=15.0) as http:
        # OpenCode must use cwd=WORKSPACE (set in worker) so path code/exp_id resolves correctly
        try:
            resp = http.get(f"{url}/file", params={"path": code_prefix.rstrip("/")}, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        nodes = data if isinstance(data, list) else data.get("children") or data.get("nodes") or []
        file_list: list[tuple[str, int]] = []

        def walk(ns: list, prefix: str, current_path: str) -> None:
            """OpenCode returns flat listings; recurse into directories via another /file request."""
            for node in ns:
                if not isinstance(node, dict):
                    continue
                node_type = node.get("type", "file")
                name = node.get("name") or (node.get("path") or "").split("/")[-1]
                if not name or name in (".", ".."):
                    continue
                rel = f"{prefix}{name}" if prefix else name
                if node_type == "directory":
                    subpath = f"{current_path}/{name}".lstrip("/")
                    try:
                        r = http.get(f"{url}/file", params={"path": subpath}, timeout=10.0)
                        r.raise_for_status()
                        subnodes = r.json()
                        subnodes = subnodes if isinstance(subnodes, list) else subnodes.get("children") or subnodes.get("nodes") or []
                        walk(subnodes, f"{rel}/", subpath)
                    except Exception:
                        pass
                else:
                    sz = node.get("size") or node.get("length") or 0
                    file_list.append((rel, sz))

        base_path = code_prefix.rstrip("/")
        walk(nodes, "", base_path)

        if not file_list:
            return None

        files: dict[str, str] = {}
        for rel_path, _ in file_list:
            full_path = f"{code_prefix}{rel_path}"
            try:
                cr = http.get(f"{url}/file/content", params={"path": full_path}, timeout=10.0)
                cr.raise_for_status()
                content_data = cr.json()
                # OpenCode FileContent: {content: string} or {text: string} or raw string
                if isinstance(content_data, str):
                    content = content_data
                elif isinstance(content_data, dict):
                    content = content_data.get("content") or content_data.get("text") or ""
                else:
                    content = ""
                files[rel_path] = content
            except Exception:
                continue

        if not files:
            return None

        total_bytes = sum(len(c.encode("utf-8")) for c in files.values())
        manifest_info = {
            "total_files": len(files),
            "total_size_bytes": total_bytes,
            "files": [{"path": p, "size": len(c.encode("utf-8"))} for p, c in files.items()],
        }
        return (manifest_info, files)

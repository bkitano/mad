"""Volume inspection tools — used by the API endpoints and the chat agent.

Each function operates on a named Modal Volume and returns plain JSON-serializable
data.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import modal


BINARY_EXTS = {
    ".bin", ".pt", ".pth", ".safetensors", ".npy", ".npz", ".pkl",
    ".gz", ".tar", ".zip", ".7z", ".xz",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf",
    ".so", ".o", ".a", ".dll", ".dylib", ".exe",
    ".mp4", ".mov", ".webm", ".wav", ".mp3",
}

MAX_GREP_FILE_BYTES = 1_000_000


def list_volumes() -> list[dict]:
    out: list[dict] = []
    for vol in modal.Volume.objects.list():
        if not vol.name:
            continue
        info = vol.info()
        out.append({
            "name": vol.name,
            "volume_id": vol.object_id,
            "created_at": info.created_at.isoformat() if info.created_at else None,
            "created_by": info.created_by,
        })
    return out


def list_files(volume_name: str, path: str = "/") -> list[dict]:
    vol = modal.Volume.from_name(volume_name)
    entries: list[dict] = []
    for entry in vol.listdir(path, recursive=False):
        entries.append({
            "path": entry.path,
            "type": "directory" if entry.type == modal.volume.FileEntryType.DIRECTORY else "file",
        })
    return entries


def read_file(volume_name: str, path: str, max_bytes: int | None = 200_000) -> dict:
    vol = modal.Volume.from_name(volume_name)
    buf = bytearray()
    truncated = False
    for chunk in vol.read_file(path):
        buf.extend(chunk)
        if max_bytes is not None and len(buf) > max_bytes:
            buf = buf[:max_bytes]
            truncated = True
            break
    data = bytes(buf)
    try:
        return {
            "path": path,
            "encoding": "utf-8",
            "content": data.decode("utf-8"),
            "truncated": truncated,
        }
    except UnicodeDecodeError:
        return {
            "path": path,
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
            "truncated": truncated,
        }


def read_notebook(volume_name: str, path: str) -> dict:
    vol = modal.Volume.from_name(volume_name)
    buf = bytearray()
    for chunk in vol.read_file(path):
        buf.extend(chunk)
    nb = json.loads(bytes(buf).decode("utf-8"))

    parts: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        ct = cell.get("cell_type", "")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        parts.append(f"## Cell {i} [{ct}]")
        parts.append(src.rstrip() or "<empty>")
        if ct == "code":
            for out in cell.get("outputs", []):
                text: str | None = None
                if isinstance(out.get("text"), (list, str)):
                    text = out["text"]
                    if isinstance(text, list):
                        text = "".join(text)
                elif isinstance(out.get("data"), dict):
                    tp = out["data"].get("text/plain")
                    if isinstance(tp, list):
                        text = "".join(tp)
                    elif isinstance(tp, str):
                        text = tp
                if text:
                    parts.append("[out] " + text.rstrip())
        parts.append("")
    return {"path": path, "content": "\n".join(parts)}


def grep(
    volume_name: str,
    pattern: str,
    path: str = "/",
    max_matches: int = 200,
) -> dict:
    vol = modal.Volume.from_name(volume_name)
    rx = re.compile(pattern)
    matches: list[dict] = []
    files_scanned = 0

    for entry in vol.listdir(path, recursive=True):
        if len(matches) >= max_matches:
            break
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            continue
        ext = os.path.splitext(entry.path)[1].lower()
        if ext in BINARY_EXTS:
            continue
        try:
            buf = bytearray()
            for chunk in vol.read_file(entry.path):
                buf.extend(chunk)
                if len(buf) > MAX_GREP_FILE_BYTES:
                    break
            text = bytes(buf).decode("utf-8", errors="ignore")
        except Exception:
            continue
        files_scanned += 1
        for ln, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                matches.append({"path": entry.path, "line": ln, "text": line[:300]})
                if len(matches) >= max_matches:
                    break

    return {
        "matches": matches,
        "files_scanned": files_scanned,
        "truncated": len(matches) >= max_matches,
    }


# ---- Chat-agent tool schemas + dispatch --------------------------------------

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the immediate contents of a path on the volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path within the volume. Defaults to '/'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the volume. Returns up to max_bytes of UTF-8 text (or base64 if binary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path within the volume."},
                    "max_bytes": {"type": "integer", "description": "Truncation cap. Defaults to 200000."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": "Read a Jupyter notebook (.ipynb), returning cell sources plus text outputs. Image data is stripped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to a .ipynb file on the volume."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursively search the volume for a Python regex pattern. Skips binary files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to '/'."},
                    "max_matches": {"type": "integer", "description": "Defaults to 200."},
                },
                "required": ["pattern"],
            },
        },
    },
]


def dispatch_tool(name: str, volume_name: str, args: dict[str, Any]) -> Any:
    args = dict(args or {})
    args["volume_name"] = volume_name
    if name == "list_files":
        return list_files(**args)
    if name == "read_file":
        return read_file(**args)
    if name == "read_notebook":
        return read_notebook(**args)
    if name == "grep":
        return grep(**args)
    raise ValueError(f"unknown tool: {name}")

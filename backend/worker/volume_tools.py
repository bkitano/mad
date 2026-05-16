"""Volume inspection tools — shared by the MCP server and the chat agent.

Each function operates on a named Modal Volume and returns plain JSON-serializable
data. The chat agent (volume_chat endpoint) calls these directly in-process; the
MCP server (mcp_server endpoint) wraps them with @mcp.tool() for external clients.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import modal


# Files we skip when walking a volume for grep / search.
BINARY_EXTS = {
    ".bin", ".pt", ".pth", ".safetensors", ".npy", ".npz", ".pkl",
    ".gz", ".tar", ".zip", ".7z", ".xz",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf",
    ".so", ".o", ".a", ".dll", ".dylib", ".exe",
    ".mp4", ".mov", ".webm", ".wav", ".mp3",
}

# Per-file read cap during grep — protects us from giant logs eating memory.
MAX_GREP_FILE_BYTES = 1_000_000


def list_volumes() -> list[dict]:
    """List every Modal volume in this workspace."""
    out: list[dict] = []
    for vol in modal.Volume.objects.list():
        if not vol.name:
            continue
        info = vol.info()
        out.append(
            {
                "name": vol.name,
                "created_at": info.created_at.isoformat() if info.created_at else None,
            }
        )
    return out


def list_files(volume_name: str, path: str = "/") -> list[dict]:
    """List the immediate contents of `path` on the named volume."""
    vol = modal.Volume.from_name(volume_name)
    entries: list[dict] = []
    for entry in vol.listdir(path, recursive=False):
        entries.append(
            {
                "path": entry.path,
                "type": "directory" if entry.type == modal.volume.FileEntryType.DIRECTORY else "file",
            }
        )
    return entries


def read_file(volume_name: str, path: str, max_bytes: int = 200_000) -> dict:
    """Read a file from a volume. Returns UTF-8 text or base64 if binary."""
    vol = modal.Volume.from_name(volume_name)
    buf = bytearray()
    truncated = False
    for chunk in vol.read_file(path):
        buf.extend(chunk)
        if len(buf) > max_bytes:
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
    """Read a .ipynb and render cell sources + text outputs. Image data stripped."""
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
    """Recursive regex search across a volume. Skips binary files."""
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
# These are the tools we hand to the OpenRouter chat completions API. The
# `volume_name` is injected by dispatch_tool() — the model never has to pass it.

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the immediate contents of a path on the volume. Use '/' for the volume root. Returns each entry's full path and whether it is a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path within the volume. Defaults to '/'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the volume. Returns up to max_bytes of UTF-8 text (or base64 if the file is binary).",
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
            "description": "Read a Jupyter notebook (.ipynb), returning cell sources plus text outputs. Image data is stripped — use this instead of read_file for notebooks.",
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
            "description": "Recursively search the volume for a Python regex pattern. Skips binary files. Best way to find logs, metrics, wandb URLs, error messages, hyperparameters, etc. Returns at most max_matches matches.",
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
    """Run a tool call from the chat agent. Injects volume_name automatically."""
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

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


def list_volumes(_user_id: str = "") -> list[dict]:
    # If user_id provided, only return volumes owned by that user
    owned_names: set[str] | None = None
    if _user_id:
        import volume_store
        owned_names = set(volume_store.list_user_volumes(_user_id))

    out: list[dict] = []
    for vol in modal.Volume.objects.list():
        if not vol.name:
            continue
        if owned_names is not None and vol.name not in owned_names:
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
    """Dispatch for single-volume mode (volume_name auto-injected)."""
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


# ---- Global mode: volume_name is explicit on each tool -----------------------
# Used when the chat agent operates across all volumes (no pre-selected volume).

_VOL_NAME_PARAM = {"type": "string", "description": "Name of the Modal volume."}

GLOBAL_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_volumes",
            "description": "List all available Modal volumes with their names and creation dates.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the immediate contents of a path on a volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_name": _VOL_NAME_PARAM,
                    "path": {"type": "string", "description": "Absolute path within the volume. Defaults to '/'."},
                },
                "required": ["volume_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from a volume. Returns up to max_bytes of UTF-8 text (or base64 if binary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_name": _VOL_NAME_PARAM,
                    "path": {"type": "string", "description": "Absolute path within the volume."},
                    "max_bytes": {"type": "integer", "description": "Truncation cap. Defaults to 200000."},
                },
                "required": ["volume_name", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": "Read a Jupyter notebook (.ipynb), returning cell sources plus text outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_name": _VOL_NAME_PARAM,
                    "path": {"type": "string", "description": "Absolute path to a .ipynb file on the volume."},
                },
                "required": ["volume_name", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursively search a volume for a Python regex pattern. Skips binary files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_name": _VOL_NAME_PARAM,
                    "pattern": {"type": "string", "description": "Python regex."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to '/'."},
                    "max_matches": {"type": "integer", "description": "Defaults to 200."},
                },
                "required": ["volume_name", "pattern"],
            },
        },
    },
    # -- Sandbox tools --
    {
        "type": "function",
        "function": {
            "name": "create_sandbox",
            "description": "Create a new sandbox with a volume attached (creates the volume if it doesn't exist). Returns sandbox URLs for interactive access. Use this to spin up compute for running code or experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_name": {"type": "string", "description": "Volume name to attach. Created if it doesn't exist."},
                    "github_repo": {"type": "string", "description": "GitHub repo to clone. Defaults to 'bkitano/mad-experiments-template'."},
                    "github_ref": {"type": "string", "description": "Branch/ref. Defaults to 'main'."},
                    "gpu": {"type": "string", "description": "GPU type: T4, L4, A10G, L40S, A100, H100. Defaults to T4."},
                },
                "required": ["volume_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminate_sandbox",
            "description": "Terminate a running sandbox by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID to terminate."},
                },
                "required": ["sandbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_volume",
            "description": "Rename a volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current volume name."},
                    "new_name": {"type": "string", "description": "New volume name."},
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sandboxes",
            "description": "List all running sandboxes with their OpenCode URLs, Jupyter URLs, and attached volume names.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_sandbox",
            "description": "Send a message to an OpenCode agent running in a sandbox and wait for its full response. Use this to ask the sandbox agent to run code, edit files, execute experiments, etc. Can take up to 5 minutes for complex tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID to send the message to."},
                    "message": {"type": "string", "description": "The instruction or question to send to the OpenCode agent."},
                    "session_id": {"type": "string", "description": "Optional session ID to continue a conversation. Omit to create a new session."},
                },
                "required": ["sandbox_id", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_sandbox_async",
            "description": "Send a task to an OpenCode sandbox without waiting for a response. Use for long-running tasks. Returns the session_id so you can check back later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID."},
                    "message": {"type": "string", "description": "The instruction to send."},
                    "session_id": {"type": "string", "description": "Optional session ID to continue a conversation."},
                },
                "required": ["sandbox_id", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sandbox_sessions",
            "description": "List active sessions on a sandbox. Shows what conversations/tasks are running.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID."},
                },
                "required": ["sandbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sandbox_files",
            "description": "List files in a running sandbox's workspace (live filesystem, not volume snapshot).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID."},
                    "path": {"type": "string", "description": "Directory path relative to workspace root. Defaults to '.'."},
                },
                "required": ["sandbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_sandbox_file",
            "description": "Read a file from a running sandbox's live workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The sandbox ID."},
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                },
                "required": ["sandbox_id", "path"],
            },
        },
    },
]


# ---- Sandbox / OpenCode interaction tools ------------------------------------

import httpx


MODAL_CREATE_SANDBOX_URL = os.environ.get(
    "MODAL_CREATE_SANDBOX_URL",
    "https://miravoice--mad-sandbox-worker-create-sandbox-worker.modal.run",
)


def create_sandbox(
    volume_name: str,
    github_repo: str = "bkitano/mad-experiments-template",
    github_ref: str = "main",
    gpu: str = "",
    _user_id: str = "",
) -> dict:
    """Create a sandbox by calling the Modal endpoint."""
    payload: dict = {
        "volume_name": volume_name,
        "github_repo": github_repo,
        "github_ref": github_ref,
        "gpu": gpu,
    }
    if _user_id:
        payload["user_id"] = _user_id
    resp = httpx.post(
        MODAL_CREATE_SANDBOX_URL,
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Register volume ownership
    if _user_id:
        import volume_store
        vol = data.get("volume_name") or volume_name
        volume_store.register_volume(vol, _user_id)
    return data


def terminate_sandbox(sandbox_id: str, _user_id: str = "") -> dict:
    """Terminate a sandbox by ID."""
    sb = modal.Sandbox.from_id(sandbox_id)
    # Verify ownership if user_id provided
    if _user_id:
        try:
            tags = sb.get_tags()
            owner = tags.get("user_id", "")
            if owner and owner != _user_id:
                return {"error": "Not your sandbox."}
        except Exception:
            pass
    sb.terminate()
    return {"sandbox_id": sandbox_id, "status": "terminated"}


def rename_volume(old_name: str, new_name: str, _user_id: str = "") -> dict:
    """Rename a Modal volume."""
    if _user_id:
        import volume_store
        if not volume_store.user_owns_volume(old_name, _user_id):
            return {"error": "You don't own that volume."}
    modal.Volume.rename(old_name, new_name)
    if _user_id:
        import volume_store
        volume_store.rename_volume(old_name, new_name)
    return {"old_name": old_name, "new_name": new_name, "status": "renamed"}


def list_sandboxes(_user_id: str = "") -> list[dict]:
    """List running sandboxes, optionally filtered by user."""
    sandbox_app = modal.App.lookup("mad-sandbox-worker", create_if_missing=True)
    sandboxes = []
    for sb in modal.Sandbox.list(app_id=sandbox_app.app_id):
        try:
            tags = sb.get_tags()
            vol_name = tags.get("volume_name", "")
            owner = tags.get("user_id", "")
        except Exception:
            vol_name = ""
            owner = ""
        # Filter by user if provided
        if _user_id and owner and owner != _user_id:
            continue
        try:
            tunnels = sb.tunnels()
            opencode_url = tunnels[4096].url if 4096 in tunnels else None
            jupyter_url = tunnels[8888].url if 8888 in tunnels else None
        except Exception:
            opencode_url = None
            jupyter_url = None
        sandboxes.append({
            "sandbox_id": sb.object_id,
            "opencode_url": opencode_url,
            "jupyter_url": jupyter_url,
            "volume_name": vol_name,
        })
    return sandboxes


def _get_opencode_url(sandbox_id: str) -> str:
    """Resolve a sandbox_id to its OpenCode tunnel URL."""
    sb = modal.Sandbox.from_id(sandbox_id)
    tunnels = sb.tunnels()
    if 4096 not in tunnels:
        raise ValueError(f"Sandbox {sandbox_id} has no OpenCode tunnel")
    return tunnels[4096].url


def send_to_sandbox(sandbox_id: str, message: str, session_id: str | None = None) -> dict:
    """Send a message to an OpenCode sandbox and wait for the full response."""
    url = _get_opencode_url(sandbox_id)

    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=300.0, write=5.0, pool=5.0)) as http:
        # Create or reuse session
        if not session_id:
            resp = http.post(f"{url}/session", json={})
            resp.raise_for_status()
            session_id = resp.json()["id"]

        # Send message (sync — waits for response)
        resp = http.post(
            f"{url}/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": message}]},
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response parts
        response_parts = data.get("response", {}).get("parts", []) if isinstance(data.get("response"), dict) else []
        text_parts = [p.get("text", "") for p in response_parts if p.get("type") == "text"]
        text = "\n".join(text_parts) if text_parts else str(data)

        return {
            "sandbox_id": sandbox_id,
            "session_id": session_id,
            "response": text,
        }


def send_to_sandbox_async(sandbox_id: str, message: str, session_id: str | None = None) -> dict:
    """Send a task to an OpenCode sandbox (fire-and-forget). Returns immediately."""
    url = _get_opencode_url(sandbox_id)

    with httpx.Client(timeout=10.0) as http:
        if not session_id:
            resp = http.post(f"{url}/session", json={})
            resp.raise_for_status()
            session_id = resp.json()["id"]

        http.post(
            f"{url}/session/{session_id}/prompt_async",
            json={"parts": [{"type": "text", "text": message}]},
        )

        return {
            "sandbox_id": sandbox_id,
            "session_id": session_id,
            "status": "sent",
        }


def get_sandbox_sessions(sandbox_id: str) -> list[dict]:
    """List active sessions on an OpenCode sandbox."""
    url = _get_opencode_url(sandbox_id)
    with httpx.Client(timeout=10.0) as http:
        resp = http.get(f"{url}/session")
        resp.raise_for_status()
        return resp.json()


def list_sandbox_files(sandbox_id: str, path: str = ".") -> list[dict]:
    """List files in a live sandbox's workspace."""
    url = _get_opencode_url(sandbox_id)
    with httpx.Client(timeout=10.0) as http:
        resp = http.get(f"{url}/file", params={"path": path})
        resp.raise_for_status()
        data = resp.json()
        # Normalize response format
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("children") or data.get("nodes") or data.get("list") or []
        return []


def read_sandbox_file(sandbox_id: str, path: str) -> dict:
    """Read a file from a live sandbox's workspace."""
    url = _get_opencode_url(sandbox_id)
    with httpx.Client(timeout=10.0) as http:
        resp = http.get(f"{url}/file/content", params={"path": path})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, str):
            return {"path": path, "content": data}
        if isinstance(data, dict):
            return {"path": path, "content": data.get("content") or data.get("text") or str(data)}
        return {"path": path, "content": str(data)}


# ---- Dispatch ----------------------------------------------------------------


def dispatch_global_tool(name: str, args: dict[str, Any], user_id: str = "") -> Any:
    """Dispatch for global mode — volume_name comes from the LLM's tool call args.

    When user_id is provided, tools that create/list/modify resources will scope
    to that user (ownership registration, filtering, permission checks).
    """
    args = dict(args or {})
    # Volume tools — inject _user_id for tools that support it
    if name == "list_volumes":
        return list_volumes(_user_id=user_id)
    if name == "list_files":
        return list_files(**args)
    if name == "read_file":
        return read_file(**args)
    if name == "read_notebook":
        return read_notebook(**args)
    if name == "grep":
        return grep(**args)
    # Sandbox tools
    if name == "create_sandbox":
        return create_sandbox(**args, _user_id=user_id)
    if name == "terminate_sandbox":
        return terminate_sandbox(**args, _user_id=user_id)
    if name == "rename_volume":
        return rename_volume(**args, _user_id=user_id)
    if name == "list_sandboxes":
        return list_sandboxes(_user_id=user_id)
    if name == "send_to_sandbox":
        return send_to_sandbox(**args)
    if name == "send_to_sandbox_async":
        return send_to_sandbox_async(**args)
    if name == "get_sandbox_sessions":
        return get_sandbox_sessions(**args)
    if name == "list_sandbox_files":
        return list_sandbox_files(**args)
    if name == "read_sandbox_file":
        return read_sandbox_file(**args)
    raise ValueError(f"unknown tool: {name}")

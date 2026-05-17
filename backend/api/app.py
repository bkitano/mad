"""
FastAPI service for MAD sandbox and volume management.

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
from typing import Optional

import httpx as _httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MAD API",
    description="Volume and sandbox management for MAD experiments",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Constants -----------------------------------------------------------------

SANDBOX_APP_NAME = "mad-sandbox-worker"
OPENCODE_PORT = 4096
JUPYTER_PORT = 8888

MODAL_CREATE_SANDBOX_URL = os.environ.get(
    "MODAL_CREATE_SANDBOX_URL",
    "https://miravoice--mad-sandbox-worker-create-sandbox-worker.modal.run",
)

# Volume chat config
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_CHAT_MODEL = "glm-5.1"
MAX_AGENT_STEPS = 12
CHAT_SESSIONS_DICT = "mad-volume-chat-sessions"

SYSTEM_PROMPT_TMPL = (
    "You are a helpful assistant with tool-access to files on the Modal volume "
    "'{volume_name}'. Use the tools to explore the volume and answer questions "
    "about the experiment(s) that ran on it.\n\n"
    "Guidelines:\n"
    "- Prefer `grep` and targeted `read_file` calls over walking the directory tree.\n"
    "- Use `read_notebook` for .ipynb files — never `read_file`.\n"
    "- When the user asks about training results, look for wandb run URLs, "
    "metric logs, and notebook outputs.\n"
    "- Be concise. Quote short excerpts rather than dumping whole files."
)


# -- Pydantic models ----------------------------------------------------------


class RenameVolumeRequest(BaseModel):
    old_name: str
    new_name: str


class DeleteVolumeRequest(BaseModel):
    volume_name: str


class TerminateSandboxRequest(BaseModel):
    sandbox_id: str


class VolumeChatRequest(BaseModel):
    volume_name: str
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None


# -- Volume endpoints ----------------------------------------------------------


@app.get("/volumes")
def list_volumes():
    """List all mad-sandbox volumes."""
    import volume_tools

    return {"volumes": volume_tools.list_volumes()}


@app.get("/volumes/ls")
def volume_ls(volume_name: str, path: str = "/"):
    """List files in a volume at the given path."""
    import volume_tools

    entries = volume_tools.list_files(volume_name, path)
    return {"volume_name": volume_name, "path": path, "entries": entries}


@app.get("/volumes/read")
def volume_read(volume_name: str, path: str = Query(...)):
    """Read a file from a volume. Returns the file content as text."""
    import volume_tools

    result = volume_tools.read_file(volume_name, path, max_bytes=None)
    return {
        "volume_name": volume_name,
        "path": result["path"],
        "encoding": result["encoding"],
        "content": result["content"],
    }


@app.post("/volumes/rename")
def rename_volume(payload: RenameVolumeRequest):
    """Rename a volume."""
    import modal as _modal

    _modal.Volume.rename(payload.old_name, payload.new_name)
    return {
        "old_name": payload.old_name,
        "new_name": payload.new_name,
        "status": "renamed",
    }


@app.post("/volumes/delete")
def delete_volume(payload: DeleteVolumeRequest):
    """Delete a volume by name."""
    import modal as _modal

    _modal.Volume.objects.delete(payload.volume_name)
    return {
        "volume_name": payload.volume_name,
        "status": "deleted",
    }


# -- Volume chat ---------------------------------------------------------------


def _serialize_assistant_message(msg) -> dict:
    """Strip the OpenAI ChatCompletionMessage to what the API expects on replay."""
    out: dict = {"role": "assistant"}
    if msg.content:
        out["content"] = msg.content
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


@app.post("/volumes/chat")
def volume_chat(req: VolumeChatRequest):
    """Chat with the contents of a Modal volume via an LLM + tool-use loop."""
    import uuid as _uuid

    import modal as _modal
    from openai import OpenAI
    import volume_tools

    api_key = os.environ.get("OPENCODE_GO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENCODE_GO_API_KEY missing from environment.")

    sessions = _modal.Dict.from_name(CHAT_SESSIONS_DICT, create_if_missing=True)
    session_id = req.session_id or _uuid.uuid4().hex
    history: list[dict] = sessions.get(session_id, [])

    if not history:
        history.append({
            "role": "system",
            "content": SYSTEM_PROMPT_TMPL.format(volume_name=req.volume_name),
        })
    history.append({"role": "user", "content": req.message})

    model = req.model or os.environ.get("OPENCODE_MODEL_DEFAULT") or DEFAULT_CHAT_MODEL
    client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)

    final_text = ""
    for _ in range(MAX_AGENT_STEPS):
        resp = client.chat.completions.create(
            model=model,
            messages=history,
            tools=volume_tools.TOOLS_SCHEMA,
        )
        msg = resp.choices[0].message
        history.append(_serialize_assistant_message(msg))

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = volume_tools.dispatch_tool(tc.function.name, req.volume_name, args)
                content = json.dumps(result, default=str)
            except Exception as e:
                content = json.dumps({"error": f"{type(e).__name__}: {e}"})
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
    else:
        for h in reversed(history):
            if h.get("role") == "assistant" and h.get("content"):
                final_text = h["content"]
                break

    sessions[session_id] = history

    return {
        "session_id": session_id,
        "response": final_text,
        "model": model,
        "volume_name": req.volume_name,
    }


@app.post("/volumes/chat/reset")
def volume_chat_reset(session_id: str = Query(...)):
    """Drop a chat session's history."""
    import modal as _modal

    sessions = _modal.Dict.from_name(CHAT_SESSIONS_DICT, create_if_missing=True)
    sessions.pop(session_id, None)
    return {"session_id": session_id, "status": "reset"}


# -- Sandbox endpoints ---------------------------------------------------------


@app.get("/sandboxes")
def list_sandboxes():
    """List all running sandboxes for this app."""
    import modal as _modal

    sandbox_app = _modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True)
    sandboxes = []
    for sb in _modal.Sandbox.list(app_id=sandbox_app.app_id):
        try:
            tunnels = sb.tunnels()
            opencode_url = tunnels[OPENCODE_PORT].url if OPENCODE_PORT in tunnels else None
            jupyter_url = tunnels[JUPYTER_PORT].url if JUPYTER_PORT in tunnels else None
        except Exception:
            opencode_url = None
            jupyter_url = None
        try:
            tags = sb.get_tags()
            vol_name = tags.get("volume_name", "")
        except Exception:
            vol_name = ""
        sandboxes.append({
            "sandbox_id": sb.object_id,
            "opencode_url": opencode_url,
            "jupyter_url": jupyter_url,
            "volume_name": vol_name,
        })
    return {"sandboxes": sandboxes}


@app.post("/sandboxes/create")
async def create_sandbox_proxy(payload: dict = {}):
    """Proxy to the Modal create_sandbox_worker endpoint."""
    async with _httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(MODAL_CREATE_SANDBOX_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


@app.post("/sandboxes/terminate")
def terminate_sandbox(payload: TerminateSandboxRequest):
    """Terminate a running sandbox by its ID."""
    import modal as _modal

    sandbox = _modal.Sandbox.from_id(payload.sandbox_id)
    sandbox.terminate()
    return {
        "sandbox_id": payload.sandbox_id,
        "status": "terminated",
    }

"""
FastAPI service for MAD sandbox and volume management.

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure api/ is on sys.path so `import volume_tools` resolves to the local module
# regardless of whether we're run as `uvicorn api.app:app` or `uvicorn app:app`.
_API_DIR = str(Path(__file__).resolve().parent)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

import json
import logging
import os
from pathlib import Path as _DotenvPath
from typing import Optional

# Auto-load .env from the api/ directory
from dotenv import load_dotenv
load_dotenv(_DotenvPath(__file__).resolve().parent / ".env")

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


@app.on_event("startup")
async def startup_voice_server():
    """Start the Pipecat voice WebSocket server alongside FastAPI."""
    import asyncio
    try:
        from voice import start_voice_server
        asyncio.create_task(start_voice_server())
    except Exception as e:
        logger.warning(f"Voice server failed to start: {e}")


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
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
MAX_AGENT_STEPS = 12
CHAT_SESSIONS_DICT = "mad-volume-chat-sessions"

SYSTEM_PROMPT = (
    "You are a helpful assistant with tool-access to Modal volumes. "
    "Use `list_volumes` to discover available volumes, then use the other tools "
    "to explore files and answer questions about the experiment(s) stored on them.\n\n"
    "Guidelines:\n"
    "- Start with `list_volumes` if you don't know which volume to look at.\n"
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
    message: str
    volume_name: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None


# -- Volume endpoints ----------------------------------------------------------


@app.get("/volumes")
def list_volumes():
    """List all mad-sandbox volumes."""
    import volume_tools  # noqa: F811

    return {"volumes": volume_tools.list_volumes()}


@app.get("/volumes/ls")
def volume_ls(volume_name: str, path: str = "/"):
    """List files in a volume at the given path."""
    import volume_tools  # noqa: F811

    entries = volume_tools.list_files(volume_name, path)
    return {"volume_name": volume_name, "path": path, "entries": entries}


@app.get("/volumes/read")
def volume_read(volume_name: str, path: str = Query(...)):
    """Read a file from a volume. Returns the file content as text."""
    import volume_tools  # noqa: F811

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
    # DeepSeek thinking models require reasoning_content to be passed back
    if getattr(msg, "reasoning_content", None):
        out["reasoning_content"] = msg.reasoning_content
    # Always set content (even if empty string) — DeepSeek requires content or tool_calls
    out["content"] = msg.content or ""
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
    """Streaming chat with volume contents via an LLM + tool-use loop.

    Returns an SSE stream with events:
      - {"type": "session_id", "session_id": "..."}
      - {"type": "thinking", "content": "..."}
      - {"type": "tool_call", "name": "...", "arguments": {...}}
      - {"type": "tool_result", "name": "...", "summary": "..."}
      - {"type": "content", "content": "..."}
      - {"type": "done"}
      - {"type": "error", "content": "..."}
    """
    import uuid as _uuid

    import modal as _modal
    from openai import OpenAI
    import volume_tools  # noqa: F811

    from fastapi.responses import StreamingResponse

    api_key = os.environ.get("OPENCODE_GO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENCODE_GO_API_KEY missing from environment.")

    sessions = _modal.Dict.from_name(CHAT_SESSIONS_DICT, create_if_missing=True)
    session_id = req.session_id or _uuid.uuid4().hex
    history: list[dict] = sessions.get(session_id, [])

    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})
    history.append({"role": "user", "content": req.message})

    model = req.model or os.environ.get("OPENCODE_MODEL_DEFAULT") or DEFAULT_CHAT_MODEL
    client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)
    tools_schema = volume_tools.GLOBAL_TOOLS_SCHEMA

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate():
        yield _sse({"type": "session_id", "session_id": session_id})

        try:
            for _ in range(MAX_AGENT_STEPS):
                stream = client.chat.completions.create(
                    model=model,
                    messages=history,
                    tools=tools_schema,
                    stream=True,
                )

                # Accumulate the streamed response
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls_acc: dict[int, dict] = {}  # index -> {id, name, arguments}

                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Reasoning/thinking tokens
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_parts.append(rc)
                        yield _sse({"type": "thinking", "content": rc})

                    # Content tokens
                    if delta.content:
                        content_parts.append(delta.content)
                        yield _sse({"type": "content", "content": delta.content})

                    # Tool call deltas
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

                # Build the assistant message for history
                full_content = "".join(content_parts)
                full_reasoning = "".join(reasoning_parts)
                assistant_msg: dict = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    assistant_msg["reasoning_content"] = full_reasoning
                if tool_calls_acc:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls_acc.values()
                    ]
                history.append(assistant_msg)

                # No tool calls — we're done
                if not tool_calls_acc:
                    break

                # Execute tool calls
                for tc in tool_calls_acc.values():
                    yield _sse({"type": "tool_call", "name": tc["name"], "arguments": tc["arguments"]})
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                        result = volume_tools.dispatch_global_tool(tc["name"], args)
                        result_str = json.dumps(result, default=str)
                        # Send a short summary to the client
                        summary = result_str[:200] + ("..." if len(result_str) > 200 else "")
                        yield _sse({"type": "tool_result", "name": tc["name"], "summary": summary})
                    except Exception as e:
                        result_str = json.dumps({"error": f"{type(e).__name__}: {e}"})
                        yield _sse({"type": "tool_result", "name": tc["name"], "summary": result_str})
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

        except Exception as e:
            yield _sse({"type": "error", "content": f"{type(e).__name__}: {e}"})

        sessions[session_id] = history
        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

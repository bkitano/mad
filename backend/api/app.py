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
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MAD API",
    description="Volume and sandbox management for MAD experiments",
    version="0.3.0",
)

from starlette.types import ASGIApp, Receive, Scope, Send


class WebSocketCORSFix:
    """Allow all WebSocket connections regardless of origin.

    Starlette's CORSMiddleware rejects WebSocket upgrades when the origin
    doesn't match. This middleware accepts all WebSocket connections before
    CORS gets a chance to reject them.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            # Strip origin check — allow all WS connections
            await self.app(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(WebSocketCORSFix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice chat via Pipecat pipeline."""
    await websocket.accept()
    logger.info("[voice] WebSocket accepted")
    try:
        from voice import run_voice_pipeline
        await run_voice_pipeline(websocket)
    except Exception as e:
        logger.error(f"Voice pipeline error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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
    "You are a helpful assistant that can browse experiment volumes AND orchestrate "
    "live sandboxes running OpenCode agents.\n\n"
    "## Capabilities\n"
    "- **Volumes**: Browse stored experiment data (list_volumes, list_files, read_file, grep, read_notebook)\n"
    "- **Sandboxes**: Manage running compute instances (list_sandboxes, send_to_sandbox, send_to_sandbox_async)\n"
    "- **Live files**: Read files from running sandboxes (list_sandbox_files, read_sandbox_file)\n\n"
    "## Guidelines\n"
    "- Use `list_volumes` or `list_sandboxes` first to discover what's available.\n"
    "- For read-only questions about past experiments, use volume tools (grep, read_file).\n"
    "- For active work (running code, editing files, executing experiments), use `send_to_sandbox` "
    "to delegate to the OpenCode agent in a sandbox.\n"
    "- Use `send_to_sandbox_async` for long-running tasks, then check back with sandbox file tools.\n"
    "- Prefer `grep` and targeted reads over walking directory trees.\n"
    "- Use `read_notebook` for .ipynb files on volumes.\n"
    "- Be concise. Summarize results rather than dumping raw output."
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
def volume_chat(request: Request):
    """Streaming chat endpoint compatible with the Vercel AI SDK UI Message Stream protocol.

    Accepts: { "messages": [...] }  (from useChat)
    Returns: SSE stream with x-vercel-ai-ui-message-stream: v1
    """
    from openai import OpenAI
    import volume_tools  # noqa: F811
    import chat_store

    from fastapi.responses import StreamingResponse

    import asyncio
    body = asyncio.get_event_loop().run_until_complete(request.json())

    api_key = os.environ.get("OPENCODE_GO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENCODE_GO_API_KEY missing from environment.")

    # Get or create session
    session_id = body.get("session_id", "")
    if not session_id or not chat_store.get_session(session_id):
        session = chat_store.create_session(session_type="text")
        session_id = session["id"]

    # Load history from DB
    history: list[dict] = chat_store.get_history_for_llm(session_id)
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Extract the latest user message from the AI SDK messages array
    messages = body.get("messages", [])
    user_text = ""
    if messages:
        last_msg = messages[-1]
        if "parts" in last_msg:
            text_parts = [p["text"] for p in last_msg["parts"] if p.get("type") == "text"]
            user_text = "\n".join(text_parts)
        else:
            user_text = last_msg.get("content", "")
        if user_text:
            history.append({"role": "user", "content": user_text})
            chat_store.add_message(session_id, "user", user_text)
            # Auto-title from first user message
            if len([m for m in history if m["role"] == "user"]) == 1:
                chat_store.auto_title(session_id, user_text)

    model = body.get("model") or os.environ.get("OPENCODE_MODEL_DEFAULT") or DEFAULT_CHAT_MODEL
    client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)
    tools_schema = volume_tools.GLOBAL_TOOLS_SCHEMA

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate():
        msg_id = f"msg_{_uuid.uuid4().hex[:12]}"

        yield _sse({"type": "start", "messageId": msg_id})
        yield _sse({"type": "start-step"})

        try:
            text_block_id = 0
            reasoning_block_id = 0

            for step in range(MAX_AGENT_STEPS):
                stream = client.chat.completions.create(
                    model=model,
                    messages=history,
                    tools=tools_schema,
                    stream=True,
                )

                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls_acc: dict[int, dict] = {}
                in_text = False
                in_reasoning = False

                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Reasoning tokens
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        if not in_reasoning:
                            r_id = f"reasoning_{reasoning_block_id}"
                            yield _sse({"type": "reasoning-start", "id": r_id})
                            in_reasoning = True
                        reasoning_parts.append(rc)
                        yield _sse({"type": "reasoning-delta", "id": r_id, "delta": rc})

                    # Content tokens
                    if delta.content:
                        if in_reasoning:
                            yield _sse({"type": "reasoning-end", "id": r_id})
                            in_reasoning = False
                            reasoning_block_id += 1
                        if not in_text:
                            t_id = f"text_{text_block_id}"
                            yield _sse({"type": "text-start", "id": t_id})
                            in_text = True
                        content_parts.append(delta.content)
                        yield _sse({"type": "text-delta", "id": t_id, "delta": delta.content})

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

                # Close open blocks
                if in_reasoning:
                    yield _sse({"type": "reasoning-end", "id": r_id})
                    reasoning_block_id += 1
                if in_text:
                    yield _sse({"type": "text-end", "id": t_id})
                    text_block_id += 1

                # Build history entry
                full_content = "".join(content_parts)
                full_reasoning = "".join(reasoning_parts)
                assistant_msg: dict = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    assistant_msg["reasoning_content"] = full_reasoning
                if tool_calls_acc:
                    assistant_msg["tool_calls"] = [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls_acc.values()
                    ]
                history.append(assistant_msg)

                # No tool calls — done
                if not tool_calls_acc:
                    break

                # Execute tools and stream results
                for tc in tool_calls_acc.values():
                    yield _sse({"type": "tool-input-start", "toolCallId": tc["id"], "toolName": tc["name"]})
                    yield _sse({"type": "tool-input-available", "toolCallId": tc["id"], "toolName": tc["name"], "input": json.loads(tc["arguments"] or "{}")})

                    try:
                        args = json.loads(tc["arguments"] or "{}")
                        result = volume_tools.dispatch_global_tool(tc["name"], args)
                        result_json = json.loads(json.dumps(result, default=str))
                    except Exception as e:
                        result_json = {"error": f"{type(e).__name__}: {e}"}
                    result_str = json.dumps(result_json, default=str)

                    yield _sse({"type": "tool-output-available", "toolCallId": tc["id"], "output": result_json})

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

                yield _sse({"type": "finish-step"})
                if tool_calls_acc:
                    yield _sse({"type": "start-step"})

        except Exception as e:
            yield _sse({"type": "error", "errorText": f"{type(e).__name__}: {e}"})

        # Save assistant response to DB
        final_content = ""
        for h in reversed(history):
            if h.get("role") == "assistant" and h.get("content"):
                final_content = h["content"]
                break
        if final_content:
            chat_store.add_message(session_id, "assistant", final_content)

        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
            "x-chat-session-id": session_id,
        },
    )


@app.get("/chats")
def list_chats(limit: int = Query(50, ge=1, le=200)):
    """List recent chat sessions."""
    import chat_store
    return chat_store.list_sessions(limit=limit)


@app.get("/chats/{session_id}")
def get_chat(session_id: str):
    """Get a chat session with its messages."""
    import chat_store
    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = chat_store.get_messages(session_id)
    return {"session": session, "messages": messages}


@app.delete("/chats/{session_id}")
def delete_chat(session_id: str):
    """Delete a chat session and all its messages."""
    import chat_store
    if not chat_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


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

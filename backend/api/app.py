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
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import get_current_user

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

import asyncio
from contextlib import asynccontextmanager

SANDBOX_POLL_INTERVAL = 300  # 5 minutes


async def _poll_sandbox_liveness():
    """Background loop: check if 'active' sandboxes are actually still running."""
    while True:
        await asyncio.sleep(SANDBOX_POLL_INTERVAL)
        try:
            import modal as _modal
            import usage_store

            active_ids = usage_store.get_all_active_sandbox_ids()
            if not active_ids:
                continue

            closed = 0
            for sandbox_id in active_ids:
                try:
                    sb = _modal.Sandbox.from_id(sandbox_id)
                    rc = sb.poll()
                    if rc is not None:
                        # Sandbox has exited
                        usage_store.record_sandbox_stop(sandbox_id)
                        closed += 1
                except Exception:
                    # Sandbox doesn't exist anymore (deleted, expired, etc.)
                    usage_store.record_sandbox_stop(sandbox_id)
                    closed += 1

            if closed:
                logger.info(f"[poller] Closed {closed} stale sandbox session(s)")
        except Exception as e:
            logger.error(f"[poller] Error: {e}")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_poll_sandbox_liveness())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="MAD API",
    description="Volume and sandbox management for MAD experiments",
    version="0.3.0",
    lifespan=lifespan,
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket, token: str = ""):
    """WebSocket endpoint for voice chat via Pipecat pipeline."""
    await websocket.accept()
    logger.info("[voice] WebSocket accepted")

    # Try to resolve user_id from token query param
    user_id = ""
    if token:
        try:
            async with _httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{os.environ.get('SUPABASE_URL', '')}/auth/v1/user",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": os.environ.get("SUPABASE_KEY", ""),
                    },
                )
            if resp.status_code == 200:
                user_id = resp.json().get("id", "")
        except Exception:
            pass
    logger.info(f"[voice] user_id={user_id or '(anonymous)'}")

    try:
        from voice import run_voice_pipeline
        await run_voice_pipeline(websocket, user_id=user_id)
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

# Abuse prevention defaults (no plan system yet)
MAX_SANDBOXES_PER_USER = 3
ALLOWED_GPUS = {"T4", "L4", "A10G", ""}  # empty string = CPU-only
MAX_TIMEOUT_HOURS = 6

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
    "- **Live files**: Read files from running sandboxes (list_sandbox_files, read_sandbox_file)\n"
    "- **Jupyter kernels**: Execute code directly on sandbox kernels (list_jupyter_kernels, execute_in_jupyter, run_notebook_in_jupyter)\n\n"
    "## Guidelines\n"
    "- Use `list_volumes` or `list_sandboxes` first to discover what's available.\n"
    "- For read-only questions about past experiments, use volume tools (grep, read_file).\n"
    "- For active work (running code, editing files, executing experiments), use `send_to_sandbox` "
    "to delegate to the OpenCode agent in a sandbox.\n"
    "- For direct code execution (quick computations, inspecting data, checking GPU status, etc.), "
    "use `execute_in_jupyter`. The kernel persists state between calls.\n"
    "- To run an entire notebook end-to-end, use `run_notebook_in_jupyter`.\n"
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
async def list_volumes(user: dict = Depends(get_current_user)):
    """List volumes owned by the authenticated user."""
    import volume_tools  # noqa: F811
    import volume_store

    owned_names = set(volume_store.list_user_volumes(user["id"]))
    all_volumes = volume_tools.list_volumes()
    return {"volumes": [v for v in all_volumes if v["name"] in owned_names]}


@app.get("/volumes/ls")
async def volume_ls(volume_name: str, path: str = "/", user: dict = Depends(get_current_user)):
    """List files in a volume at the given path."""
    import volume_tools  # noqa: F811
    import volume_store

    if not volume_store.user_owns_volume(volume_name, user["id"]):
        raise HTTPException(status_code=403, detail="Not your volume")
    entries = volume_tools.list_files(volume_name, path)
    return {"volume_name": volume_name, "path": path, "entries": entries}


@app.get("/volumes/read")
async def volume_read(volume_name: str, path: str = Query(...), user: dict = Depends(get_current_user)):
    """Read a file from a volume. Returns the file content as text."""
    import volume_tools  # noqa: F811
    import volume_store

    if not volume_store.user_owns_volume(volume_name, user["id"]):
        raise HTTPException(status_code=403, detail="Not your volume")
    result = volume_tools.read_file(volume_name, path, max_bytes=None)
    return {
        "volume_name": volume_name,
        "path": result["path"],
        "encoding": result["encoding"],
        "content": result["content"],
    }


@app.post("/volumes/rename")
async def rename_volume(payload: RenameVolumeRequest, user: dict = Depends(get_current_user)):
    """Rename a volume."""
    import modal as _modal
    import volume_store

    if not volume_store.user_owns_volume(payload.old_name, user["id"]):
        raise HTTPException(status_code=403, detail="Not your volume")
    await _modal.Volume.rename.aio(payload.old_name, payload.new_name)
    volume_store.rename_volume(payload.old_name, payload.new_name)
    return {
        "old_name": payload.old_name,
        "new_name": payload.new_name,
        "status": "renamed",
    }


@app.post("/volumes/delete")
async def delete_volume(payload: DeleteVolumeRequest, user: dict = Depends(get_current_user)):
    """Delete a volume by name."""
    import modal as _modal
    import volume_store

    if not volume_store.user_owns_volume(payload.volume_name, user["id"]):
        raise HTTPException(status_code=403, detail="Not your volume")
    _modal.Volume.objects.delete(payload.volume_name)
    volume_store.delete_volume(payload.volume_name)
    return {
        "volume_name": payload.volume_name,
        "status": "deleted",
    }


# -- Volume chat ---------------------------------------------------------------


@app.post("/volumes/chat")
async def volume_chat(request: Request, user: dict = Depends(get_current_user)):
    """Streaming chat endpoint over the Vercel AI SDK UI Message Stream protocol.

    Every block-level UI event (user submission, completed text block,
    completed reasoning block, finished tool call, step boundary) is written
    to `chat_events` at the same moment its corresponding SSE event is
    yielded. The LLM context for the next step — including across reloads —
    is always rebuilt from that event log, so any session is resumable and
    the persisted view is identical to the live view.
    """
    from openai import OpenAI
    import volume_tools  # noqa: F811
    import chat_store

    from fastapi.responses import StreamingResponse

    body = await request.json()

    api_key = os.environ.get("OPENCODE_GO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENCODE_GO_API_KEY missing from environment.")

    user_id = user["id"]

    # Resolve session. If the client supplies an id we honor it (so the first
    # send and reload point at the same row); if it doesn't exist we create it
    # with that id, owned by this user. Mismatched owner is a 403.
    session_id = (body.get("session_id") or "").strip()
    if session_id:
        existing = chat_store.get_session(session_id)
        if existing:
            if existing.get("user_id") and existing["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Not your session")
        else:
            chat_store.create_session(user_id=user_id, session_id=session_id)
    else:
        session = chat_store.create_session(user_id=user_id)
        session_id = session["id"]

    # Rebuild LLM history from the event log (resumability lives here)
    prior_events = chat_store.list_events(session_id)
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.extend(chat_store.events_to_llm_history(prior_events))

    # Persist the new user submission as one event before we start streaming
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")
    last_msg = messages[-1]
    if "parts" in last_msg:
        user_text = "\n".join(p.get("text", "") for p in last_msg["parts"] if p.get("type") == "text")
    else:
        user_text = last_msg.get("content", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="user message has no text")

    user_message_id = last_msg.get("id") or f"msg_{_uuid.uuid4().hex[:12]}"
    chat_store.append_event(
        session_id, user_message_id, "user",
        {"type": "text", "text": user_text},
    )
    had_prior_user = any(e["role"] == "user" for e in prior_events)
    if not had_prior_user:
        chat_store.auto_title(session_id, user_text)
    history.append({"role": "user", "content": user_text})

    model = body.get("model") or os.environ.get("OPENCODE_MODEL_DEFAULT") or DEFAULT_CHAT_MODEL
    client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)
    tools_schema = volume_tools.GLOBAL_TOOLS_SCHEMA

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate():
        assistant_message_id = f"msg_{_uuid.uuid4().hex[:12]}"

        yield _sse({"type": "start", "messageId": assistant_message_id})
        yield _sse({"type": "start-step"})

        try:
            text_block_id = 0
            reasoning_block_id = 0

            for step in range(MAX_AGENT_STEPS):
                # Persist the step boundary so the reload view contains the same
                # step-start parts the AI SDK produces from the SSE start-step
                # event during live streaming.
                chat_store.append_event(
                    session_id, assistant_message_id, "assistant",
                    {"type": "step-start"},
                )

                stream = client.chat.completions.create(
                    model=model,
                    messages=history,
                    tools=tools_schema,
                    stream=True,
                )

                step_text: list[str] = []        # all text emitted this step (for LLM history)
                step_reasoning: list[str] = []   # all reasoning emitted this step
                current_text: list[str] = []     # text block currently being streamed
                current_reasoning: list[str] = []
                tool_calls_acc: dict[int, dict] = {}
                in_text = False
                in_reasoning = False
                t_id = ""
                r_id = ""

                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        if not in_reasoning:
                            r_id = f"reasoning_{reasoning_block_id}"
                            yield _sse({"type": "reasoning-start", "id": r_id})
                            in_reasoning = True
                            current_reasoning = []
                        current_reasoning.append(rc)
                        step_reasoning.append(rc)
                        yield _sse({"type": "reasoning-delta", "id": r_id, "delta": rc})

                    if delta.content:
                        if in_reasoning:
                            yield _sse({"type": "reasoning-end", "id": r_id})
                            chat_store.append_event(
                                session_id, assistant_message_id, "assistant",
                                {"type": "reasoning", "text": "".join(current_reasoning)},
                            )
                            in_reasoning = False
                            reasoning_block_id += 1
                        if not in_text:
                            t_id = f"text_{text_block_id}"
                            yield _sse({"type": "text-start", "id": t_id})
                            in_text = True
                            current_text = []
                        current_text.append(delta.content)
                        step_text.append(delta.content)
                        yield _sse({"type": "text-delta", "id": t_id, "delta": delta.content})

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

                if in_reasoning:
                    yield _sse({"type": "reasoning-end", "id": r_id})
                    chat_store.append_event(
                        session_id, assistant_message_id, "assistant",
                        {"type": "reasoning", "text": "".join(current_reasoning)},
                    )
                    reasoning_block_id += 1
                if in_text:
                    yield _sse({"type": "text-end", "id": t_id})
                    chat_store.append_event(
                        session_id, assistant_message_id, "assistant",
                        {"type": "text", "text": "".join(current_text)},
                    )
                    text_block_id += 1

                # Build the OpenAI message for this step and append to history
                assistant_msg: dict = {"role": "assistant", "content": "".join(step_text)}
                if step_reasoning:
                    assistant_msg["reasoning_content"] = "".join(step_reasoning)
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

                if not tool_calls_acc:
                    break

                # Execute tools, stream their input/output, and persist each as one event
                for tc in tool_calls_acc.values():
                    try:
                        tool_input = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        tool_input = {"_raw_arguments": tc["arguments"]}

                    yield _sse({"type": "tool-input-start", "toolCallId": tc["id"], "toolName": tc["name"]})
                    yield _sse({
                        "type": "tool-input-available",
                        "toolCallId": tc["id"],
                        "toolName": tc["name"],
                        "input": tool_input,
                    })

                    import concurrent.futures
                    TOOL_TIMEOUT = 120
                    error_text: str | None = None
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(
                                volume_tools.dispatch_global_tool, tc["name"], tool_input, user_id,
                            )
                            result = future.result(timeout=TOOL_TIMEOUT)
                        result_json = json.loads(json.dumps(result, default=str))
                    except concurrent.futures.TimeoutError:
                        result_json = {"error": f"Tool '{tc['name']}' timed out after {TOOL_TIMEOUT}s"}
                        error_text = result_json["error"]
                    except Exception as e:
                        result_json = {"error": f"{type(e).__name__}: {e}"}
                        error_text = result_json["error"]

                    yield _sse({"type": "tool-output-available", "toolCallId": tc["id"], "output": result_json})

                    tool_part = {
                        "type": f"tool-{tc['name']}",
                        "toolCallId": tc["id"],
                        "state": "output-error" if error_text else "output-available",
                        "input": tool_input,
                        "output": result_json,
                    }
                    if error_text:
                        tool_part["errorText"] = error_text
                    chat_store.append_event(session_id, assistant_message_id, "assistant", tool_part)

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result_json, default=str),
                    })

                yield _sse({"type": "finish-step"})
                yield _sse({"type": "start-step"})

        except Exception as e:
            yield _sse({"type": "error", "errorText": f"{type(e).__name__}: {e}"})

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
async def list_chats(limit: int = Query(50, ge=1, le=200), user: dict = Depends(get_current_user)):
    """List recent chat sessions."""
    import chat_store
    return chat_store.list_sessions(limit=limit, user_id=user["id"])


@app.get("/chats/{session_id}")
async def get_chat(session_id: str, user: dict = Depends(get_current_user)):
    """Get a chat session as AI SDK UIMessages built from the event log.

    The returned `messages` array has the exact shape `useChat` puts into its
    `messages` state, so the frontend can call `setMessages(messages)` and
    render with the same code path used for live streaming.
    """
    import chat_store
    session = chat_store.get_session(session_id, user_id=user["id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    events = chat_store.list_events(session_id)
    messages = chat_store.events_to_ui_messages(events)
    return {"session": session, "messages": messages}


@app.delete("/chats/{session_id}")
async def delete_chat(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat session and all its messages."""
    import chat_store
    if not chat_store.delete_session(session_id, user_id=user["id"]):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# -- Sandbox helpers -----------------------------------------------------------

import time as _time
import uuid as _uuid

# Simple per-user rate limiter for sandbox creation: {user_id: last_create_timestamp}
_sandbox_create_timestamps: dict[str, float] = {}
SANDBOX_CREATE_COOLDOWN_SECONDS = 30


def _count_user_sandboxes(user_id: str) -> int:
    """Count running sandboxes owned by a user."""
    import modal as _modal

    try:
        sandbox_app = _modal.App.lookup(SANDBOX_APP_NAME)
    except _modal.exception.NotFoundError:
        return 0
    count = 0
    for sb in _modal.Sandbox.list(app_id=sandbox_app.app_id):
        try:
            tags = sb.get_tags()
            if tags.get("user_id") == user_id:
                count += 1
        except Exception:
            pass
    return count


# -- Sandbox endpoints ---------------------------------------------------------


@app.get("/sandboxes")
async def list_sandboxes(user: dict = Depends(get_current_user)):
    """List running sandboxes for the authenticated user."""
    import modal as _modal

    user_id = user["id"]
    try:
        sandbox_app = await _modal.App.lookup.aio(SANDBOX_APP_NAME)
    except _modal.exception.NotFoundError:
        return {"sandboxes": []}
    sandboxes = []
    async for sb in _modal.Sandbox.list.aio(app_id=sandbox_app.app_id):
        try:
            tags = sb.get_tags()
            vol_name = tags.get("volume_name", "")
            owner = tags.get("user_id", "")
        except Exception:
            vol_name = ""
            owner = ""
        # Only show sandboxes owned by this user (or untagged legacy ones)
        if owner and owner != user_id:
            continue
        try:
            tunnels = sb.tunnels()
            opencode_url = tunnels[OPENCODE_PORT].url if OPENCODE_PORT in tunnels else None
            jupyter_url = tunnels[JUPYTER_PORT].url if JUPYTER_PORT in tunnels else None
        except Exception:
            opencode_url = None
            jupyter_url = None
        created_at = int(tags.get("created_at", "0"))
        timeout_seconds = int(tags.get("timeout_seconds", str(6 * 3600)))
        expires_at = (created_at + timeout_seconds) if created_at else 0
        sandboxes.append({
            "sandbox_id": sb.object_id,
            "opencode_url": opencode_url,
            "jupyter_url": jupyter_url,
            "volume_name": vol_name,
            "gpu": tags.get("gpu", ""),
            "cpu": float(tags.get("cpu", "4")),
            "memory": int(tags.get("memory", "8192")),
            "expires_at": expires_at,
        })
    return {"sandboxes": sandboxes}


@app.post("/sandboxes/create")
async def create_sandbox_proxy(payload: dict = {}, user: dict = Depends(get_current_user)):
    """Proxy to the Modal create_sandbox_worker endpoint."""
    import volume_store

    user_id = user["id"]

    # Rate limit: one create per cooldown period
    now = _time.time()
    last = _sandbox_create_timestamps.get(user_id, 0)
    if now - last < SANDBOX_CREATE_COOLDOWN_SECONDS:
        wait = int(SANDBOX_CREATE_COOLDOWN_SECONDS - (now - last))
        raise HTTPException(status_code=429, detail=f"Please wait {wait}s before creating another sandbox")

    # GPU restriction
    gpu = payload.get("gpu", "") or ""
    # Strip multi-GPU suffix like "T4:4" → "T4"
    gpu_type = gpu.split(":")[0] if gpu else ""
    if gpu_type not in ALLOWED_GPUS:
        raise HTTPException(status_code=403, detail=f"GPU '{gpu_type}' is not available on your plan. Allowed: {', '.join(g for g in ALLOWED_GPUS if g) or 'CPU only'}")

    # Timeout cap
    timeout_hours = payload.get("timeout_hours", 6)
    if timeout_hours > MAX_TIMEOUT_HOURS:
        payload["timeout_hours"] = MAX_TIMEOUT_HOURS

    # Concurrent sandbox cap
    running = _count_user_sandboxes(user_id)
    if running >= MAX_SANDBOXES_PER_USER:
        raise HTTPException(status_code=429, detail=f"You already have {running} running sandbox(es). Max is {MAX_SANDBOXES_PER_USER}. Terminate one first.")

    # Tag the sandbox with the user_id so we can filter later
    payload["user_id"] = user_id
    _sandbox_create_timestamps[user_id] = now
    async with _httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(MODAL_CREATE_SANDBOX_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Register volume ownership
    vol_name = data.get("volume_name")
    if vol_name:
        volume_store.register_volume(vol_name, user_id)

    # Track usage
    import usage_store
    sandbox_id = data.get("sandbox_id", "")
    if sandbox_id:
        gpu_count_val = int(gpu.split(":")[1]) if ":" in gpu else 1
        cpu_val = payload.get("cpu", 4.0)
        memory_val = payload.get("memory", 8192)
        usage_store.record_sandbox_start(
            user_id, sandbox_id, gpu=gpu_type, gpu_count=gpu_count_val,
            cpu=cpu_val, memory_mb=memory_val,
        )

    return data


@app.post("/sandboxes/terminate")
async def terminate_sandbox(payload: TerminateSandboxRequest, user: dict = Depends(get_current_user)):
    """Terminate a running sandbox by its ID."""
    import modal as _modal

    sandbox = await _modal.Sandbox.from_id.aio(payload.sandbox_id)
    # Verify ownership
    try:
        tags = sandbox.get_tags()
        owner = tags.get("user_id", "")
        if owner and owner != user["id"]:
            raise HTTPException(status_code=403, detail="Not your sandbox")
    except HTTPException:
        raise
    except Exception:
        pass  # Legacy sandbox without tags — allow termination
    await sandbox.terminate.aio()

    # Record stop for usage tracking
    import usage_store
    usage_store.record_sandbox_stop(payload.sandbox_id)

    return {
        "sandbox_id": payload.sandbox_id,
        "status": "terminated",
    }


# -- Usage endpoints -----------------------------------------------------------


@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    """Return compute usage summary for the authenticated user."""
    import usage_store
    from datetime import datetime, timezone

    # Active sessions
    active = usage_store.get_active_sessions(user["id"])

    # Usage this calendar month
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    summary = usage_store.get_usage_summary(user["id"], since=month_start)

    return {
        "active_sandboxes": active,
        "period_start": month_start.isoformat(),
        "total_gpu_seconds": summary["total_seconds"],
        "by_gpu": summary["by_gpu"],
    }

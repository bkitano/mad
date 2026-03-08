"""
OpenCode adapter with a query() interface matching claude_agent_sdk.

Uses the OpenCode HTTP API directly (no opencode_ai SDK — it targets a
different spec version). Requires OpenCode server running locally.

Usage:
    # Start OpenCode (TUI or headless):
    #   opencode serve            (headless, default port 4096)
    #   opencode                  (TUI mode — also exposes the API)
    #
    # Optional env var to override server URL:
    #   export OPENCODE_BASE_URL=http://localhost:4096

    from agents.opencode_query import query, OpenCodeAgentOptions

    async for message in query(
        prompt="...",
        options=OpenCodeAgentOptions(model="sonnet", system_prompt="..."),
    ):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Optional

import httpx

BASE_URL = os.environ.get("OPENCODE_BASE_URL", "http://localhost:4096")

# ── Request queue (throttle to 1 concurrent request) ────────────────────────
# Modal is throttled to 1 concurrent request; this semaphore serialises callers.
# All agents share this single semaphore across the process.
_SEMAPHORE = asyncio.Semaphore(1)

# ── Normalized message types (matching claude_agent_sdk interface) ──────────

@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list[TextBlock]


@dataclass
class ToolCallMessage:
    tool_name: str
    tool_input: dict
    part_id: str = ""

@dataclass
class ToolResultMessage:
    tool_name: str
    output: str
    part_id: str = ""

@dataclass
class ResultMessage:
    result: str


# ── Options ─────────────────────────────────────────────────────────────────

@dataclass
class OpenCodeAgentOptions:
    """Drop-in replacement for ClaudeAgentOptions (subset of fields)."""
    model: str = "sonnet"
    system_prompt: str = ""
    cwd: str = ""
    # Kept for API compatibility with claude_agent_sdk callers; ignored here
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "acceptEdits"


# ── Model name → (providerID, modelID) ──────────────────────────────────────

_MODEL_MAP: dict[str, tuple[str, str]] = {
    # Default aliases → Anthropic
    "opus":   ("anthropic", "claude-sonnet-4-5"),
    "sonnet": ("anthropic", "claude-sonnet-4-5"),
    "haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    # Modal GLM-5 (if MODAL_GLM5_TOKEN is set)
    "glm5":   ("opencode-go", "glm5"),
    # opencode provider fallback models
    "big-pickle":   ("opencode", "big-pickle"),
    "gpt-5-nano":   ("opencode", "gpt-5-nano"),
    "minimax":      ("opencode", "minimax-m2.5-free"),
    "trinity":      ("opencode", "trinity-large-preview-free"),
    # anthropic models (if ANTHROPIC_API_KEY is set)
    "claude-opus":   ("anthropic", "claude-opus-4-5"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-6"),
    "claude-haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    # openai models (if OPENAI_API_KEY is set)
    "gpt-5.2":        ("openai", "gpt-5.2"),
}


# ── SSE helpers ──────────────────────────────────────────────────────────────

async def _sse_reader(http: httpx.AsyncClient, queue: asyncio.Queue) -> None:
    """Read the /event SSE stream and put parsed event dicts onto queue."""
    try:
        async with http.stream("GET", "/event", timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                await queue.put(("line", line))
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        await queue.put(("error", exc))
    finally:
        await queue.put(("done", None))


# ── Main query() function ────────────────────────────────────────────────────

async def query(
    prompt: str,
    options: Optional[OpenCodeAgentOptions] = None,
) -> AsyncGenerator[AssistantMessage | ToolCallMessage | ToolResultMessage | ResultMessage, None]:
    """
    Async generator: sends `prompt` to OpenCode and yields normalized messages.

    Streams text deltas via the SSE event bus, stops when the session idles.
    """
    if options is None:
        options = OpenCodeAgentOptions()

    # provider_id, model_id = _MODEL_MAP.get(
    #     options.model,
    #     ("anthropic", options.model),  # pass through raw model IDs
    # )

    async with _SEMAPHORE:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as http:
            # 1. Create a fresh session
            resp = await http.post("/session", json={})
            resp.raise_for_status()
            session_id: str = resp.json()["id"]

            # 2. Build the message body
            body: dict = {
                "parts": [{"type": "text", "text": prompt}],
                # "model": {"providerID": provider_id, "modelID": model_id},
            }
            if options.system_prompt:
                body["system"] = options.system_prompt

            # 3. Start SSE reader in background
            queue: asyncio.Queue = asyncio.Queue()
            sse_task = asyncio.create_task(_sse_reader(http, queue))

            # 4. Fire prompt asynchronously (returns 204 immediately)
            pr = await http.post(
                f"/session/{session_id}/prompt_async",
                json=body,
                timeout=30.0,
            )
            pr.raise_for_status()

            # 5. Consume SSE events, yield text deltas, stop on idle/error
            seen_text_len: dict[str, int] = {}

            try:
                while True:
                    kind, value = await queue.get()

                    if kind == "done":
                        break

                    if kind == "error":
                        raise value

                    line: str = value
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    props = event.get("properties", {})

                    if etype == "message.part.updated":
                        part = props.get("part", {})
                        if part.get("sessionID") != session_id:
                            continue
                        part_type = part.get("type", "")
                        part_id: str = part.get("id", "default")

                        if part_type == "text":
                            full_text: str = part.get("text", "") or ""
                            prev = seen_text_len.get(part_id, 0)
                            delta = full_text[prev:]
                            if delta:
                                seen_text_len[part_id] = len(full_text)
                                yield AssistantMessage(content=[TextBlock(text=delta)])

                        elif part_type == "tool":
                            tool_name = part.get("tool", "")
                            state = part.get("state", {})
                            status = state.get("status", "")
                            tool_input = state.get("input", {})
                            # Yield tool call when we first see this part
                            call_key = f"{part_id}:call"
                            if call_key not in seen_text_len:
                                seen_text_len[call_key] = 1
                                yield ToolCallMessage(
                                    tool_name=tool_name,
                                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                                    part_id=part_id,
                                )
                            # Yield tool result when completed
                            result_key = f"{part_id}:result"
                            if status == "completed" and result_key not in seen_text_len:
                                seen_text_len[result_key] = 1
                                output = state.get("output", "") or state.get("text", "") or ""
                                yield ToolResultMessage(
                                    tool_name=tool_name,
                                    output=output[:2000] if isinstance(output, str) else str(output)[:2000],
                                    part_id=part_id,
                                )

                    elif etype == "session.idle":
                        if props.get("sessionID") == session_id:
                            yield ResultMessage(result="done")
                            break

                    elif etype == "session.error":
                        if props.get("sessionID") == session_id:
                            raise RuntimeError(f"OpenCode session error: {props}")

            finally:
                sse_task.cancel()
                try:
                    await sse_task
                except asyncio.CancelledError:
                    pass

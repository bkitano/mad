"""
OpenCode adapter with a query() interface.

Uses the OpenCode HTTP API directly. Requires OpenCode server running locally.

Usage:
    from agents.opencode_query import query, OpenCodeAgentOptions

    async for event in query(prompt="...", options=OpenCodeAgentOptions()):
        print(event)  # raw SSE event dict
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


# ── Options ─────────────────────────────────────────────────────────────────

@dataclass
class OpenCodeAgentOptions:
    """Drop-in replacement for ClaudeAgentOptions (subset of fields)."""
    model: str = "sonnet"
    system_prompt: str = ""
    cwd: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "acceptEdits"


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
) -> AsyncGenerator[dict, None]:
    """
    Async generator: sends `prompt` to OpenCode and yields raw SSE event dicts.

    Yields every event that belongs to this session, stopping on session.idle.
    Each yielded dict has the shape: {"type": str, "properties": dict}.
    """
    if options is None:
        options = OpenCodeAgentOptions()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as http:
        # 1. Create a fresh session
        resp = await http.post("/session", json={})
        resp.raise_for_status()
        session_id: str = resp.json()["id"]

        # 2. Build the message body
        body: dict = {
            "parts": [{"type": "text", "text": prompt}],
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

        # 5. Consume SSE events, yield raw events for this session
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

                # Skip noisy delta events
                if etype == "message.part.delta":
                    continue

                # Filter to events for this session
                session_match = (
                    props.get("sessionID") == session_id
                    or props.get("part", {}).get("sessionID") == session_id
                )
                if not session_match:
                    continue

                yield event

                if etype == "session.idle":
                    break

                if etype == "session.error":
                    raise RuntimeError(f"OpenCode session error: {props}")

        finally:
            sse_task.cancel()
            try:
                await sse_task
            except asyncio.CancelledError:
                pass

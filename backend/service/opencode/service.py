"""
OpencodeService — manages the opencode server subprocess, SSE event forwarder,
and provides a query() method for sending prompts.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import AsyncGenerator
from typing import Optional

import httpx

from service.client import ExperimentClient
from service.opencode.types import (
    Event,
    EventMessagePartUpdated,
    OpenCodeAgentOptions,
    TextPart,
    ToolPart,
    parse_event,
)


def _log(msg: str):
    print(f"[opencode-service] {msg}", flush=True)


class OpencodeService:
    """Manages the opencode server subprocess and SSE event forwarder.

    Holds a mutable `metadata` dict that is attached to every forwarded event,
    so callers can set experiment_id / parent_id at any point and all
    subsequent events will include them.
    """

    def __init__(
        self,
        port: int,
        client: ExperimentClient,
        agent_id: str,
    ):
        self.url = f"http://127.0.0.1:{port}"
        self.client = client
        self.agent_id = agent_id
        self.metadata: dict = {}  # e.g. {"experiment_id": ..., "parent_id": ...}
        self.is_started: bool = False
        self._proc: Optional[subprocess.Popen] = None
        self._stop = asyncio.Event()
        self._forwarder: Optional[asyncio.Task] = None
        self._session_id: Optional[str] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the opencode server subprocess and the SSE forwarder task."""
        _log("Starting opencode serve...")
        self._proc = subprocess.Popen(
            ["opencode", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(2)
        if self._proc.poll() is not None:
            output = self._proc.stdout.read().decode() if self._proc.stdout else ""
            raise RuntimeError(f"opencode serve exited immediately: {output}")
        _log("opencode serve is running")

        self._stop.clear()
        self._forwarder = asyncio.create_task(self._event_forwarder())
        self.is_started = True

    def start_forwarder_only(self) -> None:
        """Start just the SSE forwarder (when the subprocess is managed externally)."""
        self._stop.clear()
        self._forwarder = asyncio.create_task(self._event_forwarder())
        self.is_started = True

    async def stop(self) -> None:
        """Shut down the forwarder and opencode server."""
        self._stop.set()
        self.is_started = False
        if self._forwarder:
            self._forwarder.cancel()
            try:
                await self._forwarder
            except asyncio.CancelledError:
                pass
        if self._proc:
            _log("Stopping opencode serve...")
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def wait_until_ready(self, timeout_s: float = 30.0) -> None:
        """Block until opencode HTTP server is ready."""
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.url}/session", timeout=2.0)
                if r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.5)
        raise RuntimeError(f"opencode not ready after {timeout_s}s; last_err={last_err!r}")

    # ── query ────────────────────────────────────────────────────────────

    async def query(
        self,
        prompt: str,
        options: Optional[OpenCodeAgentOptions] = None,
    ) -> AsyncGenerator[Event, None]:
        """Send a prompt to the opencode server and yield typed Event objects.

        Yields every event belonging to the created session, stopping on
        session.idle. Requires the service to be started first.
        """
        if not self.is_started:
            raise RuntimeError("OpencodeService is not started — call start() first")

        if options is None:
            options = OpenCodeAgentOptions()

        async with httpx.AsyncClient(base_url=self.url, timeout=60.0) as http:
            # 1. Create a fresh session
            resp = await http.post("/session", json={})
            resp.raise_for_status()
            session_id: str = resp.json()["id"]
            self._session_id = session_id

            # 2. Build the message body
            body: dict = {
                "parts": [{"type": "text", "text": prompt}],
            }
            if options.system_prompt:
                body["system"] = options.system_prompt

            # 3. Start SSE reader in background
            queue: asyncio.Queue = asyncio.Queue()
            sse_task = asyncio.create_task(self._sse_reader(http, queue))

            # 4. Fire prompt asynchronously (returns 204 immediately)
            pr = await http.post(
                f"/session/{session_id}/prompt_async",
                json=body,
                timeout=30.0,
            )
            pr.raise_for_status()

            # 5. Consume SSE events, yield typed events for this session
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
                        raw = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = raw.get("type")
                    props = raw.get("properties", {})

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

                    try:
                        event = parse_event(raw)
                    except Exception:
                        continue

                    yield event

                    if etype == "session.idle":
                        break

                    if etype == "session.error":
                        raise RuntimeError(f"OpenCode session error: {props}")

            finally:
                self._session_id = None
                sse_task.cancel()
                try:
                    await sse_task
                except asyncio.CancelledError:
                    pass

    async def send_message(self, text: str) -> None:
        """Inject a message into the active OpenCode session."""
        session_id = self._session_id  # capture before any await
        if not session_id:
            raise RuntimeError("No active OpenCode session")
        async with httpx.AsyncClient(base_url=self.url, timeout=30.0) as http:
            resp = await http.post(
                f"/session/{session_id}/prompt_async",
                json={"parts": [{"type": "text", "text": text}]},
            )
            resp.raise_for_status()
        _log(f"Injected message into session {session_id} (len={len(text)})")

    @staticmethod
    async def _sse_reader(http: httpx.AsyncClient, queue: asyncio.Queue) -> None:
        """Read the /event SSE stream and put raw lines onto queue."""
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

    # ── SSE forwarder ────────────────────────────────────────────────────

    @staticmethod
    def _summarize(event: Event) -> str:
        """Extract a short summary string from a typed event."""
        if isinstance(event, EventMessagePartUpdated):
            part = event.part
            if isinstance(part, TextPart):
                return part.text[:500].replace("\n", " ")
            if isinstance(part, ToolPart):
                return f"{part.tool} ({part.state.status})"
        return event.type

    async def _event_forwarder(self) -> None:
        """Persistent SSE listener that forwards opencode events to the MAD service.

        Parses each SSE line into a typed Event, extracts a summary, and
        forwards to the MAD API with experiment_id/parent_id from self.metadata.
        """
        while not self._stop.is_set():
            try:
                async with httpx.AsyncClient(base_url=self.url, timeout=None) as http:
                    async with http.stream("GET", "/event") as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if self._stop.is_set():
                                break
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if not data_str:
                                continue
                            try:
                                raw = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Skip noisy deltas before parsing
                            if raw.get("type") == "message.part.delta":
                                continue

                            try:
                                event = parse_event(raw)
                            except Exception:
                                continue

                            summary = self._summarize(event)
                            _log(f"  {event.type}: {summary[:200]}")

                            # Pull experiment_id from metadata (falls back to event props)
                            props = raw.get("properties", {})
                            experiment_id = self.metadata.get(
                                "experiment_id", props.get("experimentID")
                            )
                            parent_id = self.metadata.get("parent_id")

                            try:
                                self.client.emit_event(
                                    event.type,
                                    summary[:500],
                                    experiment_id=experiment_id,
                                    agent=self.agent_id,
                                    details=props,
                                    parent_id=parent_id,
                                )
                            except Exception:
                                pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._stop.is_set():
                    break
                _log(f"SSE connection lost ({e}), reconnecting in 1s...")
                await asyncio.sleep(1)

    # ── grace period helper ──────────────────────────────────────────────

    async def grace_period(self, seconds: int, details: dict | None = None) -> None:
        """Keep the forwarder running for a grace period after experiment completion."""
        if details:
            self.client.emit_event(
                "worker.grace_period",
                f"Experiment done, opencode staying up for {seconds}s grace period",
                agent=self.agent_id,
                experiment_id=self.metadata.get("experiment_id"),
                parent_id=self.metadata.get("parent_id"),
                details=details,
            )
            _log(f"Grace period {seconds}s — forwarder still active")
        await asyncio.sleep(seconds)

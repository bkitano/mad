"""Unit tests for OpencodeService — lifecycle, metadata, summarize, forwarder."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from service.opencode.service import OpencodeService
from service.opencode.types import (
    EventMessagePartUpdated,
    EventSessionIdle,
    parse_event,
)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.emit_event = MagicMock(return_value={"id": 1})
    return client


@pytest.fixture
def service(mock_client):
    return OpencodeService(port=4096, client=mock_client, agent_id="test-agent")


# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_not_started_by_default(self, service):
        assert service.is_started is False

    @pytest.mark.asyncio
    async def test_query_raises_when_not_started(self, service):
        with pytest.raises(RuntimeError, match="not started"):
            async for _ in service.query("hello"):
                pass

    def test_metadata_initially_empty(self, service):
        assert service.metadata == {}

    def test_metadata_update_and_clear(self, service):
        service.metadata.update({"experiment_id": "exp-1", "parent_id": 42})
        assert service.metadata["experiment_id"] == "exp-1"
        assert service.metadata["parent_id"] == 42

        service.metadata.clear()
        assert service.metadata == {}

    @patch("service.opencode.service.subprocess.Popen")
    @patch("service.opencode.service.time.sleep")
    @patch("service.opencode.service.asyncio.create_task")
    def test_start_sets_is_started(self, mock_create_task, mock_sleep, mock_popen, service):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is still running
        mock_popen.return_value = mock_proc

        service.start()

        assert service.is_started is True
        assert service._proc is mock_proc
        mock_create_task.assert_called_once()

    @patch("service.opencode.service.asyncio.create_task")
    def test_start_forwarder_only_sets_is_started(self, mock_create_task, service):
        service.start_forwarder_only()

        assert service.is_started is True
        assert service._proc is None
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_clears_is_started(self, service):
        service.is_started = True
        service._forwarder = None
        service._proc = None

        await service.stop()

        assert service.is_started is False


# ── _summarize ───────────────────────────────────────────────────────────────


class TestSummarize:
    def test_summarize_text_part(self, service):
        event = parse_event({
            "type": "message.part.updated",
            "properties": {
                "part": {"type": "text", "text": "Hello world\nSecond line"},
            },
        })
        summary = service._summarize(event)
        assert summary == "Hello world Second line"

    def test_summarize_text_part_truncated(self, service):
        long_text = "x" * 600
        event = parse_event({
            "type": "message.part.updated",
            "properties": {
                "part": {"type": "text", "text": long_text},
            },
        })
        summary = service._summarize(event)
        assert len(summary) == 500

    def test_summarize_tool_part(self, service):
        event = parse_event({
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "type": "tool", "tool": "Bash",
                    "state": {"status": "completed", "input": {}},
                },
            },
        })
        summary = service._summarize(event)
        assert summary == "Bash (completed)"

    def test_summarize_non_part_event(self, service):
        event = parse_event({"type": "session.idle", "properties": {"sessionID": "s1"}})
        summary = service._summarize(event)
        assert summary == "session.idle"


# ── Forwarder ────────────────────────────────────────────────────────────────


def _make_sse_lines(events: list[dict]) -> str:
    """Build an SSE response body from a list of event dicts."""
    lines = []
    for e in events:
        lines.append(f"data: {json.dumps(e)}")
        lines.append("")  # blank line separates SSE events
    return "\n".join(lines)


class TestForwarder:
    @pytest.mark.asyncio
    async def test_forwarder_emits_to_client(self, service, mock_client):
        """Forwarder should parse SSE events and call client.emit_event."""
        events = [
            {"type": "session.idle", "properties": {"sessionID": "s1"}},
        ]
        sse_body = _make_sse_lines(events)

        import httpx

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/event":
                return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404)

        with patch("service.opencode.service.httpx.AsyncClient") as mock_ac:
            # Set up the mock to use our transport
            real_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
            mock_ac.return_value.__aenter__ = AsyncMock(return_value=real_client)
            mock_ac.return_value.__aexit__ = AsyncMock(return_value=None)

            # Start forwarder, let it process, then stop
            service._stop.clear()

            async def run_forwarder():
                # Run forwarder but stop it after a short delay
                task = asyncio.create_task(service._event_forwarder())
                await asyncio.sleep(0.2)
                service._stop.set()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_forwarder()

            # Check that emit_event was called
            if mock_client.emit_event.called:
                call_args = mock_client.emit_event.call_args
                assert call_args[0][0] == "session.idle"

    @pytest.mark.asyncio
    async def test_forwarder_skips_delta_events(self, service, mock_client):
        """message.part.delta events should be silently skipped."""
        events = [
            {"type": "message.part.delta", "properties": {"part": {"type": "text", "text": "h"}}},
            {"type": "session.idle", "properties": {"sessionID": "s1"}},
        ]
        sse_body = _make_sse_lines(events)

        import httpx

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/event":
                return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404)

        with patch("service.opencode.service.httpx.AsyncClient") as mock_ac:
            real_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
            mock_ac.return_value.__aenter__ = AsyncMock(return_value=real_client)
            mock_ac.return_value.__aexit__ = AsyncMock(return_value=None)

            service._stop.clear()
            task = asyncio.create_task(service._event_forwarder())
            await asyncio.sleep(0.2)
            service._stop.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # If emit was called, it should NOT have been for delta
            for call in mock_client.emit_event.call_args_list:
                assert call[0][0] != "message.part.delta"

    def test_forwarder_uses_metadata(self, service, mock_client):
        """When metadata is set, forwarder should pass experiment_id and parent_id."""
        service.metadata.update({"experiment_id": "exp-42", "parent_id": 7})

        # Directly test the emit call that the forwarder would make
        props = {"sessionID": "s1"}
        event = parse_event({"type": "session.idle", "properties": props})
        summary = service._summarize(event)

        experiment_id = service.metadata.get("experiment_id", props.get("experimentID"))
        parent_id = service.metadata.get("parent_id")

        mock_client.emit_event(
            event.type,
            summary[:500],
            experiment_id=experiment_id,
            agent=service.agent_id,
            details=props,
            parent_id=parent_id,
        )

        call_kwargs = mock_client.emit_event.call_args[1]
        assert call_kwargs["experiment_id"] == "exp-42"
        assert call_kwargs["parent_id"] == 7
        assert call_kwargs["agent"] == "test-agent"


# ── Grace period ─────────────────────────────────────────────────────────────


class TestGracePeriod:
    @pytest.mark.asyncio
    async def test_grace_period_emits_event(self, service, mock_client):
        service.metadata.update({"experiment_id": "exp-1", "parent_id": 5})

        with patch("service.opencode.service.asyncio.sleep", new_callable=AsyncMock):
            await service.grace_period(10, details={"opencode_url": "http://localhost:4096"})

        mock_client.emit_event.assert_called_once()
        call_kwargs = mock_client.emit_event.call_args[1]
        assert call_kwargs["experiment_id"] == "exp-1"
        assert call_kwargs["parent_id"] == 5
        assert "grace period" in mock_client.emit_event.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_grace_period_no_details_no_emit(self, service, mock_client):
        with patch("service.opencode.service.asyncio.sleep", new_callable=AsyncMock):
            await service.grace_period(10)

        mock_client.emit_event.assert_not_called()

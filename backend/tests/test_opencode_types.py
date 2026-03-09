"""Unit tests for service.opencode.types — parse_event, parse_part, discriminators."""

import pytest
from pydantic import ValidationError

from service.opencode.types import (
    EventMessagePartUpdated,
    EventSessionError,
    EventSessionIdle,
    EventSessionStatus,
    TextPart,
    ToolPart,
    StepStartPart,
    AgentPart,
    CompactionPart,
    parse_event,
    parse_part,
)


# ── parse_event ─────────────────────────────────────────────────────────────


class TestParseEvent:
    def test_session_idle(self):
        e = parse_event({"type": "session.idle", "properties": {"sessionID": "s1"}})
        assert isinstance(e, EventSessionIdle)
        assert e.type == "session.idle"
        assert e.properties["sessionID"] == "s1"

    def test_session_status(self):
        e = parse_event({
            "type": "session.status",
            "properties": {"sessionID": "s1", "status": {"type": "busy"}},
        })
        assert isinstance(e, EventSessionStatus)

    def test_session_error(self):
        e = parse_event({
            "type": "session.error",
            "properties": {"sessionID": "s1", "error": {"name": "UnknownError", "data": {"message": "boom"}}},
        })
        assert isinstance(e, EventSessionError)

    def test_message_part_updated_text(self):
        e = parse_event({
            "type": "message.part.updated",
            "properties": {
                "part": {"type": "text", "id": "p1", "sessionID": "s1", "messageID": "m1", "text": "hello"},
                "delta": "hello",
            },
        })
        assert isinstance(e, EventMessagePartUpdated)
        assert e.delta == "hello"
        part = e.part
        assert isinstance(part, TextPart)
        assert part.text == "hello"

    def test_message_part_updated_tool(self):
        e = parse_event({
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "type": "tool", "id": "t1", "sessionID": "s1", "messageID": "m1",
                    "callID": "c1", "tool": "Bash",
                    "state": {"status": "completed", "input": {"command": "ls"}, "output": "file.txt", "title": "ls", "metadata": {}, "time": {"start": 1, "end": 2}},
                },
            },
        })
        assert isinstance(e, EventMessagePartUpdated)
        part = e.part
        assert isinstance(part, ToolPart)
        assert part.tool == "Bash"
        assert part.state.status == "completed"
        assert part.state.output == "file.txt"

    def test_unknown_event_type_raises(self):
        with pytest.raises(ValidationError):
            parse_event({"type": "totally.unknown", "properties": {}})

    def test_missing_type_raises(self):
        with pytest.raises((ValidationError, KeyError)):
            parse_event({"properties": {}})

    def test_missing_properties_raises(self):
        with pytest.raises(ValidationError):
            parse_event({"type": "session.idle"})

    def test_all_known_event_types(self):
        """Smoke test: every known event type can be parsed with minimal properties."""
        event_types = [
            "server.instance.disposed", "installation.updated", "installation.update-available",
            "lsp.client.diagnostics", "lsp.updated",
            "message.updated", "message.removed",
            "message.part.updated", "message.part.delta", "message.part.removed",
            "permission.updated", "permission.replied",
            "session.status", "session.idle", "session.compacted",
            "file.edited", "todo.updated", "command.executed",
            "session.created", "session.updated", "session.deleted", "session.diff",
            "session.error",
            "file.watcher.updated", "vcs.branch.updated",
            "tui.prompt.append", "tui.command.execute", "tui.toast.show",
            "pty.created", "pty.updated", "pty.exited", "pty.deleted",
            "server.connected",
        ]
        for etype in event_types:
            props = {}
            # message.part.updated needs a valid part in properties
            if etype == "message.part.updated":
                props = {"part": {"type": "text"}}
            e = parse_event({"type": etype, "properties": props})
            assert e.type == etype


# ── parse_part ──────────────────────────────────────────────────────────────


class TestParsePart:
    def test_text_part(self):
        p = parse_part({"type": "text", "text": "hello world"})
        assert isinstance(p, TextPart)
        assert p.text == "hello world"

    def test_text_part_minimal(self):
        p = parse_part({"type": "text"})
        assert isinstance(p, TextPart)
        assert p.text == ""

    def test_tool_part(self):
        p = parse_part({
            "type": "tool",
            "tool": "Read",
            "state": {"status": "running", "input": {"file": "a.py"}, "time": {"start": 100}},
        })
        assert isinstance(p, ToolPart)
        assert p.tool == "Read"
        assert p.state.status == "running"

    def test_tool_part_error_state(self):
        p = parse_part({
            "type": "tool",
            "tool": "Bash",
            "state": {"status": "error", "input": {}, "error": "command failed", "time": {"start": 1, "end": 2}},
        })
        assert isinstance(p, ToolPart)
        assert p.state.status == "error"
        assert p.state.error == "command failed"

    def test_step_start(self):
        p = parse_part({"type": "step-start", "snapshot": "snap1"})
        assert isinstance(p, StepStartPart)
        assert p.snapshot == "snap1"

    def test_agent_part(self):
        p = parse_part({"type": "agent", "name": "coder"})
        assert isinstance(p, AgentPart)
        assert p.name == "coder"

    def test_compaction_part(self):
        p = parse_part({"type": "compaction", "auto": True})
        assert isinstance(p, CompactionPart)
        assert p.auto is True

    def test_unknown_part_type_raises(self):
        with pytest.raises(ValidationError):
            parse_part({"type": "nonexistent"})

    def test_all_part_types_minimal(self):
        """Smoke test: every part type can be parsed with minimal fields."""
        part_types = [
            ("text", {}),
            ("reasoning", {}),
            ("file", {}),
            ("tool", {"state": {"status": "pending", "input": {}}}),
            ("step-start", {}),
            ("step-finish", {}),
            ("snapshot", {}),
            ("patch", {}),
            ("agent", {}),
            ("retry", {}),
            ("compaction", {}),
            ("subtask", {}),
        ]
        for ptype, extra in part_types:
            p = parse_part({"type": ptype, **extra})
            assert p.type == ptype

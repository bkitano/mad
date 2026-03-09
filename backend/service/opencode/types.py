"""
Pydantic models for opencode SSE events.

Generated from the opencode TypeScript SDK types:
https://github.com/anomalyco/opencode/blob/dev/packages/sdk/js/src/gen/types.gen.ts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Query options (non-pydantic, kept as dataclass) ──────────────────────────


@dataclass
class OpenCodeAgentOptions:
    """Options for the opencode query() interface."""
    model: str = "sonnet"
    system_prompt: str = ""
    cwd: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "acceptEdits"


# ── Shared / nested models ──────────────────────────────────────────────────


class FileDiff(BaseModel):
    file: str
    before: str
    after: str
    additions: int
    deletions: int


class TimeRange(BaseModel):
    start: float
    end: Optional[float] = None
    compacted: Optional[float] = None


class TokenUsage(BaseModel):
    input: int
    output: int
    reasoning: int
    cache: dict[str, int] = {}


# ── Messages ────────────────────────────────────────────────────────────────


class UserMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["user"]
    time: dict[str, Any]
    summary: Optional[dict[str, Any]] = None
    agent: str = ""
    model: Optional[dict[str, str]] = None
    system: Optional[str] = None
    tools: Optional[dict[str, bool]] = None


class AssistantMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["assistant"]
    time: dict[str, Any]
    error: Optional[dict[str, Any]] = None
    parentID: str = ""
    modelID: str = ""
    providerID: str = ""
    mode: str = ""
    path: Optional[dict[str, str]] = None
    summary: Optional[bool] = None
    cost: float = 0
    tokens: Optional[TokenUsage] = None
    finish: Optional[str] = None


Message = Annotated[
    Union[UserMessage, AssistantMessage],
    Field(discriminator="role"),
]


# ── Parts ────────────────────────────────────────────────────────────────────


class TextPart(BaseModel):
    type: Literal["text"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    text: str = ""
    synthetic: Optional[bool] = None
    ignored: Optional[bool] = None
    time: Optional[TimeRange] = None
    metadata: Optional[dict[str, Any]] = None


class ReasoningPart(BaseModel):
    type: Literal["reasoning"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    text: str = ""
    metadata: Optional[dict[str, Any]] = None
    time: Optional[TimeRange] = None


class FilePart(BaseModel):
    type: Literal["file"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    mime: str = ""
    filename: Optional[str] = None
    url: str = ""
    source: Optional[dict[str, Any]] = None


class ToolState(BaseModel):
    status: str  # "pending" | "running" | "completed" | "error"
    input: dict[str, Any] = {}
    output: Optional[str] = None
    error: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    time: Optional[TimeRange] = None
    raw: Optional[str] = None
    attachments: Optional[list[dict[str, Any]]] = None


class ToolPart(BaseModel):
    type: Literal["tool"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    callID: str = ""
    tool: str = ""
    state: ToolState
    metadata: Optional[dict[str, Any]] = None


class StepStartPart(BaseModel):
    type: Literal["step-start"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    snapshot: Optional[str] = None


class StepFinishPart(BaseModel):
    type: Literal["step-finish"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    reason: str = ""
    snapshot: Optional[str] = None
    cost: float = 0
    tokens: Optional[TokenUsage] = None


class SnapshotPart(BaseModel):
    type: Literal["snapshot"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    snapshot: str = ""


class PatchPart(BaseModel):
    type: Literal["patch"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    hash: str = ""
    files: list[str] = []


class AgentPart(BaseModel):
    type: Literal["agent"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    name: str = ""
    source: Optional[dict[str, Any]] = None


class RetryPart(BaseModel):
    type: Literal["retry"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    attempt: int = 0
    error: Optional[dict[str, Any]] = None
    time: Optional[dict[str, Any]] = None


class CompactionPart(BaseModel):
    type: Literal["compaction"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    auto: bool = False


class SubtaskPart(BaseModel):
    type: Literal["subtask"]
    id: str = ""
    sessionID: str = ""
    messageID: str = ""
    prompt: str = ""
    description: str = ""
    agent: str = ""


Part = Annotated[
    Union[
        TextPart,
        ReasoningPart,
        FilePart,
        ToolPart,
        StepStartPart,
        StepFinishPart,
        SnapshotPart,
        PatchPart,
        AgentPart,
        RetryPart,
        CompactionPart,
        SubtaskPart,
    ],
    Field(discriminator="type"),
]


# ── Permission ──────────────────────────────────────────────────────────────


class Permission(BaseModel):
    id: str = ""
    type: str = ""
    pattern: Optional[Union[str, list[str]]] = None
    sessionID: str = ""
    messageID: str = ""
    callID: Optional[str] = None
    title: str = ""
    metadata: dict[str, Any] = {}
    time: Optional[dict[str, Any]] = None


# ── Session ─────────────────────────────────────────────────────────────────


class Session(BaseModel):
    id: str
    projectID: str = ""
    directory: str = ""
    parentID: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    share: Optional[dict[str, str]] = None
    title: str = ""
    version: str = ""
    time: dict[str, Any] = {}
    revert: Optional[dict[str, Any]] = None


class SessionStatus(BaseModel):
    type: str  # "idle" | "retry" | "busy"
    attempt: Optional[int] = None
    message: Optional[str] = None
    next: Optional[float] = None


# ── Todo ────────────────────────────────────────────────────────────────────


class Todo(BaseModel):
    id: str = ""
    content: str = ""
    status: str = ""
    priority: str = ""


# ── Pty ─────────────────────────────────────────────────────────────────────


class Pty(BaseModel):
    id: str = ""
    title: str = ""
    command: str = ""
    args: list[str] = []
    cwd: str = ""
    status: str = ""  # "running" | "exited"
    pid: int = 0


# ── Events ──────────────────────────────────────────────────────────────────
#
# Each event has a literal `type` discriminator and a `properties` dict.
# We model the full Event union from the opencode SDK.


class EventServerInstanceDisposed(BaseModel):
    type: Literal["server.instance.disposed"]
    properties: dict[str, Any]


class EventInstallationUpdated(BaseModel):
    type: Literal["installation.updated"]
    properties: dict[str, Any]


class EventInstallationUpdateAvailable(BaseModel):
    type: Literal["installation.update-available"]
    properties: dict[str, Any]


class EventLspClientDiagnostics(BaseModel):
    type: Literal["lsp.client.diagnostics"]
    properties: dict[str, Any]


class EventLspUpdated(BaseModel):
    type: Literal["lsp.updated"]
    properties: dict[str, Any]


class EventMessageUpdated(BaseModel):
    type: Literal["message.updated"]
    properties: dict[str, Any]


class EventMessageRemoved(BaseModel):
    type: Literal["message.removed"]
    properties: dict[str, Any]


class EventMessagePartUpdated(BaseModel):
    type: Literal["message.part.updated"]
    properties: dict[str, Any]

    @property
    def part(self) -> Part:
        return _PART_ADAPTER.validate_python(self.properties.get("part", {}))

    @property
    def delta(self) -> Optional[str]:
        return self.properties.get("delta")


class EventMessagePartDelta(BaseModel):
    type: Literal["message.part.delta"]
    properties: dict[str, Any]


class EventMessagePartRemoved(BaseModel):
    type: Literal["message.part.removed"]
    properties: dict[str, Any]


class EventPermissionUpdated(BaseModel):
    type: Literal["permission.updated"]
    properties: dict[str, Any]


class EventPermissionReplied(BaseModel):
    type: Literal["permission.replied"]
    properties: dict[str, Any]


class EventSessionStatus(BaseModel):
    type: Literal["session.status"]
    properties: dict[str, Any]


class EventSessionIdle(BaseModel):
    type: Literal["session.idle"]
    properties: dict[str, Any]


class EventSessionCompacted(BaseModel):
    type: Literal["session.compacted"]
    properties: dict[str, Any]


class EventFileEdited(BaseModel):
    type: Literal["file.edited"]
    properties: dict[str, Any]


class EventTodoUpdated(BaseModel):
    type: Literal["todo.updated"]
    properties: dict[str, Any]


class EventCommandExecuted(BaseModel):
    type: Literal["command.executed"]
    properties: dict[str, Any]


class EventSessionCreated(BaseModel):
    type: Literal["session.created"]
    properties: dict[str, Any]


class EventSessionUpdated(BaseModel):
    type: Literal["session.updated"]
    properties: dict[str, Any]


class EventSessionDeleted(BaseModel):
    type: Literal["session.deleted"]
    properties: dict[str, Any]


class EventSessionDiff(BaseModel):
    type: Literal["session.diff"]
    properties: dict[str, Any]


class EventSessionError(BaseModel):
    type: Literal["session.error"]
    properties: dict[str, Any]


class EventFileWatcherUpdated(BaseModel):
    type: Literal["file.watcher.updated"]
    properties: dict[str, Any]


class EventVcsBranchUpdated(BaseModel):
    type: Literal["vcs.branch.updated"]
    properties: dict[str, Any]


class EventTuiPromptAppend(BaseModel):
    type: Literal["tui.prompt.append"]
    properties: dict[str, Any]


class EventTuiCommandExecute(BaseModel):
    type: Literal["tui.command.execute"]
    properties: dict[str, Any]


class EventTuiToastShow(BaseModel):
    type: Literal["tui.toast.show"]
    properties: dict[str, Any]


class EventPtyCreated(BaseModel):
    type: Literal["pty.created"]
    properties: dict[str, Any]


class EventPtyUpdated(BaseModel):
    type: Literal["pty.updated"]
    properties: dict[str, Any]


class EventPtyExited(BaseModel):
    type: Literal["pty.exited"]
    properties: dict[str, Any]


class EventPtyDeleted(BaseModel):
    type: Literal["pty.deleted"]
    properties: dict[str, Any]


class EventServerConnected(BaseModel):
    type: Literal["server.connected"]
    properties: dict[str, Any]


# ── Event union ─────────────────────────────────────────────────────────────

Event = Annotated[
    Union[
        EventServerInstanceDisposed,
        EventInstallationUpdated,
        EventInstallationUpdateAvailable,
        EventLspClientDiagnostics,
        EventLspUpdated,
        EventMessageUpdated,
        EventMessageRemoved,
        EventMessagePartUpdated,
        EventMessagePartDelta,
        EventMessagePartRemoved,
        EventPermissionUpdated,
        EventPermissionReplied,
        EventSessionStatus,
        EventSessionIdle,
        EventSessionCompacted,
        EventFileEdited,
        EventTodoUpdated,
        EventCommandExecuted,
        EventSessionCreated,
        EventSessionUpdated,
        EventSessionDeleted,
        EventSessionDiff,
        EventSessionError,
        EventFileWatcherUpdated,
        EventVcsBranchUpdated,
        EventTuiPromptAppend,
        EventTuiCommandExecute,
        EventTuiToastShow,
        EventPtyCreated,
        EventPtyUpdated,
        EventPtyExited,
        EventPtyDeleted,
        EventServerConnected,
    ],
    Field(discriminator="type"),
]


# ── Parsing helper ──────────────────────────────────────────────────────────

from pydantic import TypeAdapter

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)
_PART_ADAPTER: TypeAdapter[Part] = TypeAdapter(Part)


def parse_event(data: dict[str, Any]) -> Event:
    """Parse a raw SSE event dict into a typed Event model.

    Raises pydantic.ValidationError if the event doesn't match any known type.
    """
    return _EVENT_ADAPTER.validate_python(data)


def parse_part(data: dict[str, Any]) -> Part:
    """Parse a raw part dict into a typed Part model."""
    return _PART_ADAPTER.validate_python(data)

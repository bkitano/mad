"""Chat persistence backed by an append-only event log in Supabase.

The single source of truth is `chat_events`. Each row is one UI part
(text block, reasoning block, tool call, step boundary, user submission)
attached to a session and a logical message_id. Grouping rows by
message_id in seq order yields the AI SDK UIMessage[] structure the
frontend renders, byte-for-byte identical between live streaming and
historical reload. The same rows are projected into OpenAI/DeepSeek
chat-completion history when we resume a conversation.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras


def _dsn() -> str:
    return (
        f"host={os.environ['PGHOST']} "
        f"port={os.environ['PGPORT']} "
        f"dbname={os.environ['PGDATABASE']} "
        f"user={os.environ['PGUSER']} "
        f"password={os.environ['PGPASSWORD']} "
        f"sslmode=require"
    )


@contextmanager
def _conn():
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# -- Sessions ------------------------------------------------------------------


def create_session(
    title: str = "New chat",
    session_type: str = "text",
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    sid = session_id or uuid.uuid4().hex
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO chat_sessions (id, title, type, user_id) VALUES (%s, %s, %s, %s) RETURNING *",
                (sid, title, session_type, user_id),
            )
            return dict(cur.fetchone())


def list_sessions(limit: int = 50, user_id: str | None = None) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if user_id:
                cur.execute(
                    "SELECT id, title, type, created_at, updated_at FROM chat_sessions "
                    "WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                    (user_id, limit),
                )
            else:
                cur.execute(
                    "SELECT id, title, type, created_at, updated_at FROM chat_sessions "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (limit,),
                )
            return [dict(r) for r in cur.fetchall()]


def get_session(session_id: str, user_id: str | None = None) -> dict | None:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if user_id:
                cur.execute("SELECT * FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            else:
                cur.execute("SELECT * FROM chat_sessions WHERE id = %s", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = now() WHERE id = %s",
                (title, session_id),
            )


def delete_session(session_id: str, user_id: str | None = None) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            else:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            return cur.rowcount > 0


def auto_title(session_id: str, first_user_message: str) -> None:
    title = first_user_message[:80].strip()
    if len(first_user_message) > 80:
        title += "..."
    update_session_title(session_id, title)


# -- Events --------------------------------------------------------------------


def append_event(session_id: str, message_id: str, role: str, event: dict) -> int:
    """Append one UI part to the session log. Returns the new seq."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_events (session_id, message_id, seq, role, event)
                VALUES (
                    %s, %s,
                    COALESCE((SELECT MAX(seq) + 1 FROM chat_events WHERE session_id = %s), 0),
                    %s, %s::jsonb
                )
                RETURNING seq
                """,
                (session_id, message_id, session_id, role, json.dumps(event)),
            )
            seq = cur.fetchone()[0]
            cur.execute("UPDATE chat_sessions SET updated_at = now() WHERE id = %s", (session_id,))
            return seq


def list_events(session_id: str) -> list[dict]:
    """Return all events for a session ordered by seq.

    `event` values are returned as already-decoded dicts.
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, message_id, seq, role, event, created_at "
                "FROM chat_events WHERE session_id = %s ORDER BY seq",
                (session_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if isinstance(r.get("event"), str):
                    r["event"] = json.loads(r["event"])
            return rows


# -- Projections ---------------------------------------------------------------


def events_to_ui_messages(events: list[dict]) -> list[dict]:
    """Group rows by message_id (first-seen seq order) into AI SDK UIMessages.

    Output shape per message:
        {"id": str, "role": "user"|"assistant", "parts": list[UIMessagePart]}

    UIMessagePart is exactly the JSON stored in the `event` column, which is
    what `@ai-sdk/react`'s `useChat` puts in `messages[*].parts`.
    """
    messages: list[dict] = []
    by_id: dict[str, dict] = {}
    for row in events:
        mid = row["message_id"]
        msg = by_id.get(mid)
        if msg is None:
            msg = {"id": mid, "role": row["role"], "parts": []}
            by_id[mid] = msg
            messages.append(msg)
        msg["parts"].append(row["event"])
    return messages


def events_to_llm_history(events: list[dict]) -> list[dict]:
    """Project the event log into OpenAI chat-completion messages.

    User messages become a single `{role: "user", content}` entry.
    Assistant message parts are split on `step-start` boundaries — each step
    becomes one `{role: "assistant", ...}` message (with text, reasoning,
    and tool_calls fields populated as needed), followed by one
    `{role: "tool", tool_call_id, content}` per finished tool call so the
    model can see its prior results.
    """
    ui_messages = events_to_ui_messages(events)
    history: list[dict] = []

    for m in ui_messages:
        if m["role"] == "user":
            text = "\n".join(p.get("text", "") for p in m["parts"] if p.get("type") == "text")
            history.append({"role": "user", "content": text})
            continue

        # assistant: partition parts by step-start markers
        steps: list[list[dict]] = [[]]
        for p in m["parts"]:
            if p.get("type") == "step-start":
                if steps[-1]:
                    steps.append([])
                continue
            steps[-1].append(p)
        if not steps[-1]:
            steps.pop()

        for step_parts in steps:
            content_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            tool_calls: list[dict] = []
            tool_outputs: list[tuple[str, Any]] = []
            for p in step_parts:
                t = p.get("type", "")
                if t == "text":
                    content_chunks.append(p.get("text", ""))
                elif t == "reasoning":
                    reasoning_chunks.append(p.get("text", ""))
                elif t.startswith("tool-"):
                    tool_name = t[len("tool-"):]
                    tool_calls.append({
                        "id": p["toolCallId"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(p.get("input") or {}),
                        },
                    })
                    if "output" in p:
                        tool_outputs.append((p["toolCallId"], p["output"]))
                    elif "errorText" in p:
                        tool_outputs.append((p["toolCallId"], {"error": p["errorText"]}))

            assistant_msg: dict = {"role": "assistant", "content": "".join(content_chunks)}
            if reasoning_chunks:
                assistant_msg["reasoning_content"] = "".join(reasoning_chunks)
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            history.append(assistant_msg)

            for tool_id, output in tool_outputs:
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(output, default=str),
                })

    return history

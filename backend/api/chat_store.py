"""Chat session persistence backed by Supabase Postgres."""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

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


def create_session(session_type: str = "text", title: str = "New chat", user_id: str | None = None) -> dict:
    sid = uuid.uuid4().hex
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
                    "SELECT id, title, type, created_at, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                    (user_id, limit),
                )
            else:
                cur.execute(
                    "SELECT id, title, type, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT %s",
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
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE id = %s",
                (title, datetime.now(timezone.utc), session_id),
            )


def touch_session(session_id: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET updated_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), session_id),
            )


def delete_session(session_id: str, user_id: str | None = None) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
            else:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            return cur.rowcount > 0


# -- Messages ------------------------------------------------------------------


def add_message(session_id: str, role: str, content: str = "", parts: list | None = None) -> dict:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content, parts) VALUES (%s, %s, %s, %s) RETURNING *",
                (session_id, role, content, json.dumps(parts) if parts else None),
            )
            touch_session(session_id)
            return dict(cur.fetchone())


def get_messages(session_id: str) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content, parts, created_at FROM chat_messages WHERE session_id = %s ORDER BY created_at",
                (session_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if isinstance(r.get("parts"), str):
                    r["parts"] = json.loads(r["parts"])
            return rows


def get_history_for_llm(session_id: str) -> list[dict]:
    """Return messages in the format the LLM expects (role + content)."""
    messages = get_messages(session_id)
    history = []
    for m in messages:
        entry: dict = {"role": m["role"], "content": m["content"]}
        history.append(entry)
    return history


def auto_title(session_id: str, first_user_message: str) -> None:
    """Set session title from the first user message (truncated)."""
    title = first_user_message[:80].strip()
    if len(first_user_message) > 80:
        title += "..."
    update_session_title(session_id, title)

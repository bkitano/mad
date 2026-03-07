"""
Postgres database layer for experiments, events, and claims.

Requires POSTGRES_URL env var (Supabase connection string).
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.extras


def _get_conn():
    url = os.environ.get("POSTGRES_URL")
    if not url:
        raise RuntimeError("POSTGRES_URL not configured")
    return psycopg2.connect(url)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(sql: str, params: tuple = ()) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_one(sql: str, params: tuple = ()) -> Optional[dict]:
    rows = _fetch(sql, params)
    return rows[0] if rows else None


def _execute(sql: str, params: tuple = ()) -> int:
    """Execute a write query, return rowcount."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rowcount = cur.rowcount
        conn.commit()
        return rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """No-op — tables created via migrations/001_experiments_and_events.sql."""
    pass


# ── Experiments ──────────────────────────────────────────────────────────────


def create_experiment(
    experiment_id: str,
    proposal_id: str,
    agent_id: str = "",
    config: Optional[dict] = None,
    cost_estimate: Optional[float] = None,
) -> dict:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO experiments
                   (id, proposal_id, status, agent_id, config, cost_estimate)
                   VALUES (%s, %s, 'created', %s, %s, %s)""",
                (
                    experiment_id,
                    proposal_id,
                    agent_id,
                    json.dumps(config or {}),
                    cost_estimate,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_experiment(experiment_id)


def get_experiment(experiment_id: str) -> Optional[dict]:
    return _fetch_one("SELECT * FROM experiments WHERE id = %s", (experiment_id,))


def list_experiments(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    if status:
        return _fetch(
            "SELECT * FROM experiments WHERE status = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (status, limit, offset),
        )
    return _fetch(
        "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    )


def update_experiment(experiment_id: str, **fields) -> Optional[dict]:
    if not fields:
        return get_experiment(experiment_id)

    # Serialize dict fields to JSON
    for json_field in ("config", "results"):
        if json_field in fields and isinstance(fields[json_field], dict):
            fields[json_field] = json.dumps(fields[json_field])

    set_clause = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [experiment_id]

    _execute(f"UPDATE experiments SET {set_clause} WHERE id = %s", tuple(values))
    return get_experiment(experiment_id)


# ── Events ───────────────────────────────────────────────────────────────────


def emit_event(
    event_type: str,
    summary: str,
    experiment_id: Optional[str] = None,
    agent: str = "",
    details: Optional[dict] = None,
    parent_id: Optional[int] = None,
) -> dict:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO events (type, experiment_id, agent_id, summary, details, parent_id)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (
                    event_type,
                    experiment_id,
                    agent,
                    summary,
                    json.dumps(details) if details else None,
                    parent_id,
                ),
            )
            row = dict(cur.fetchone())
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_events(
    experiment_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    parent_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    conditions: list[str] = []
    params: list[Any] = []

    if experiment_id:
        conditions.append("experiment_id = %s")
        params.append(experiment_id)
    if event_type:
        conditions.append("type = %s")
        params.append(event_type)
    if since:
        conditions.append("created_at >= %s")
        params.append(since)
    if parent_id is not None:
        conditions.append("parent_id = %s")
        params.append(parent_id)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.extend([limit, offset])

    return _fetch(
        f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
        tuple(params),
    )


# ── Claims ───────────────────────────────────────────────────────────────────

CLAIM_STALE_MINUTES = 30


def _clean_stale_claims(conn) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=CLAIM_STALE_MINUTES)).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM claims WHERE status = 'active' AND heartbeat_at < %s",
            (cutoff,),
        )
        return cur.rowcount


def claim_proposal(proposal_id: str, agent_id: str) -> bool:
    conn = _get_conn()
    try:
        _clean_stale_claims(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(
                    """INSERT INTO claims (proposal_id, agent_id, status)
                       VALUES (%s, %s, 'active')""",
                    (proposal_id, agent_id),
                )
                conn.commit()
                return True
            except psycopg2.IntegrityError:
                conn.rollback()
                # Check if same agent already holds it (idempotent re-claim)
                cur.execute(
                    "SELECT agent_id, status FROM claims WHERE proposal_id = %s",
                    (proposal_id,),
                )
                row = cur.fetchone()
                if row and row["agent_id"] == agent_id and row["status"] == "active":
                    return True
                return False
    finally:
        conn.close()


def heartbeat_claim(proposal_id: str, agent_id: str, details: Optional[str] = None) -> bool:
    return _execute(
        "UPDATE claims SET heartbeat_at = now(), details = COALESCE(%s, details) "
        "WHERE proposal_id = %s AND agent_id = %s AND status = 'active'",
        (details, proposal_id, agent_id),
    ) > 0


def release_claim(proposal_id: str, agent_id: str, status: str = "completed") -> bool:
    return _execute(
        "DELETE FROM claims WHERE proposal_id = %s AND agent_id = %s",
        (proposal_id, agent_id),
    ) > 0


def list_claims(status: Optional[str] = "active") -> list[dict]:
    conn = _get_conn()
    try:
        _clean_stale_claims(conn)
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()

    if status:
        return _fetch(
            "SELECT * FROM claims WHERE status = %s ORDER BY claimed_at DESC",
            (status,),
        )
    return _fetch("SELECT * FROM claims ORDER BY claimed_at DESC")


def is_proposal_claimed(proposal_id: str) -> bool:
    conn = _get_conn()
    try:
        _clean_stale_claims(conn)
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()

    row = _fetch_one(
        "SELECT 1 FROM claims WHERE proposal_id = %s AND status = 'active'",
        (proposal_id,),
    )
    return row is not None

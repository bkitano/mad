"""
Domain stores for experiments, events, proposals, and claims.

Each store takes a DatabaseManager instance and encapsulates
all CRUD operations for its domain.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from service.db import DatabaseManager

CLAIM_STALE_MINUTES = 30


class ExperimentsStore:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(
        self,
        experiment_id: str,
        proposal_id: str,
        agent_id: str = "",
        cost_estimate: Optional[float] = None,
    ) -> dict:
        self.db._execute(
            """INSERT INTO experiments
               (id, proposal_id, status, agent_id, cost_estimate)
               VALUES (%s, %s, 'created', %s, %s)""",
            (experiment_id, proposal_id, agent_id, cost_estimate),
        )
        return self.get(experiment_id)

    def get(self, experiment_id: str) -> Optional[dict]:
        return self.db._fetch_one("SELECT * FROM experiments WHERE id = %s", (experiment_id,))

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if status:
            return self.db._fetch(
                "SELECT * FROM experiments WHERE status = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (status, limit, offset),
            )
        return self.db._fetch(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )

    def update(self, experiment_id: str, **fields) -> Optional[dict]:
        if not fields:
            return self.get(experiment_id)

        for json_field in ("config", "results"):
            if json_field in fields and isinstance(fields[json_field], dict):
                fields[json_field] = json.dumps(fields[json_field])

        set_clause = ", ".join(f"{k} = %s" for k in fields)
        values = list(fields.values()) + [experiment_id]
        self.db._execute(f"UPDATE experiments SET {set_clause} WHERE id = %s", tuple(values))
        return self.get(experiment_id)


class EventsStore:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def emit(
        self,
        event_type: str,
        summary: str,
        experiment_id: Optional[str] = None,
        agent: str = "",
        details: Optional[dict] = None,
        parent_id: Optional[int] = None,
    ) -> dict:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
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
            return dict(cur.fetchone())

    def get(self, event_id: int) -> Optional[dict]:
        return self.db._fetch_one("SELECT * FROM events WHERE id = %s", (event_id,))

    def list(
        self,
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

        return self.db._fetch(
            f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            tuple(params),
        )


class ProposalsStore:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(
        self,
        filename: str,
        title: str,
        content: str,
        experiment_number: Optional[int] = None,
        status: str = "draft",
        priority: Optional[str] = None,
        hypothesis: Optional[str] = None,
        based_on: Optional[str] = None,
    ) -> dict:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute("DELETE FROM proposals WHERE filename = %s", (filename,))
            cur.execute(
                """INSERT INTO proposals
                   (filename, experiment_number, title, status, priority, hypothesis, based_on, content)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (filename, experiment_number, title, status, priority, hypothesis, based_on, content),
            )
            return dict(cur.fetchone())

    def list(self, status: Optional[str] = None) -> list[dict]:
        cols = "filename, experiment_number, title, status, priority, created, based_on, results_file, content"
        if status:
            return self.db._fetch(
                f"SELECT {cols} FROM proposals WHERE lower(status) = lower(%s)"
                " ORDER BY experiment_number NULLS LAST, filename",
                (status,),
            )
        return self.db._fetch(
            f"SELECT {cols} FROM proposals ORDER BY experiment_number NULLS LAST, filename"
        )

    def get(self, proposal_id: str) -> Optional[dict]:
        fname = proposal_id if proposal_id.endswith(".md") else f"{proposal_id}.md"
        return self.db._fetch_one(
            "SELECT * FROM proposals WHERE filename = %s OR filename LIKE %s"
            " ORDER BY filename LIMIT 1",
            (fname, f"{proposal_id}%"),
        )

    # ── Claims ───────────────────────────────────────────────────────────────

    def _clean_stale_claims(self, conn) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=CLAIM_STALE_MINUTES)).isoformat()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM claims WHERE status = 'active' AND heartbeat_at < %s",
                (cutoff,),
            )
            return cur.rowcount

    def claim(self, proposal_id: str, agent_id: str) -> bool:
        with self.db.get_connection() as conn:
            self._clean_stale_claims(conn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                try:
                    cur.execute(
                        """INSERT INTO claims (proposal_id, agent_id, status)
                           VALUES (%s, %s, 'active')""",
                        (proposal_id, agent_id),
                    )
                    return True
                except psycopg2.IntegrityError:
                    conn.rollback()
                    cur.execute(
                        "SELECT agent_id, status FROM claims WHERE proposal_id = %s",
                        (proposal_id,),
                    )
                    row = cur.fetchone()
                    if row and row["agent_id"] == agent_id and row["status"] == "active":
                        return True
                    return False

    def heartbeat_claim(self, proposal_id: str, agent_id: str, details: Optional[str] = None) -> bool:
        return self.db._execute(
            "UPDATE claims SET heartbeat_at = now(), details = COALESCE(%s, details) "
            "WHERE proposal_id = %s AND agent_id = %s AND status = 'active'",
            (details, proposal_id, agent_id),
        ) > 0

    def release_claim(self, proposal_id: str, agent_id: str, status: str = "completed") -> bool:
        return self.db._execute(
            "DELETE FROM claims WHERE proposal_id = %s AND agent_id = %s",
            (proposal_id, agent_id),
        ) > 0

    def list_claims(self, status: Optional[str] = "active") -> list[dict]:
        with self.db.get_connection() as conn:
            self._clean_stale_claims(conn)

        if status:
            return self.db._fetch(
                "SELECT * FROM claims WHERE status = %s ORDER BY claimed_at DESC",
                (status,),
            )
        return self.db._fetch("SELECT * FROM claims ORDER BY claimed_at DESC")

    def is_claimed(self, proposal_id: str) -> bool:
        with self.db.get_connection() as conn:
            self._clean_stale_claims(conn)

        row = self.db._fetch_one(
            "SELECT 1 FROM claims WHERE proposal_id = %s AND status = 'active'",
            (proposal_id,),
        )
        return row is not None

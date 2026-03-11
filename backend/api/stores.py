"""
Domain stores for experiments, events, and proposals.

Each store takes a DatabaseManager instance and encapsulates
all CRUD operations for its domain.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from api.db import DatabaseManager


class ExperimentsStore:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(
        self,
        experiment_id: str,
        proposal_id: str,
        cost_estimate: Optional[float] = None,
        worker_id: Optional[str] = None,
        run_number: int = 1,
    ) -> dict:
        self.db._execute(
            """INSERT INTO experiments
               (id, proposal_id, status, cost_estimate, worker_id, run_number)
               VALUES (%s, %s, 'created', %s, %s, %s)""",
            (experiment_id, proposal_id, cost_estimate, worker_id, run_number),
        )
        return self.get(experiment_id)

    def get(self, experiment_id: str) -> Optional[dict]:
        return self.db._fetch_one("SELECT * FROM experiments WHERE id = %s", (experiment_id,))

    def get_next_run_number(self, base_id: str) -> int:
        """Return MAX(run_number) + 1 across all reruns of base_id."""
        row = self.db._fetch_one(
            "SELECT COALESCE(MAX(run_number), 0) + 1 AS next FROM experiments WHERE id = %s OR id LIKE %s",
            (base_id, f"{base_id}-r%"),
        )
        return row["next"] if row else 2

    def list(
        self,
        status: Optional[str] = None,
        proposal_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = %s")
            params.append(status)
        if proposal_id:
            conditions.append("proposal_id = %s")
            params.append(proposal_id)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.extend([limit, offset])
        return self.db._fetch(
            f"SELECT * FROM experiments{where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            tuple(params),
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
        details: Optional[dict] = None,
        parent_id: Optional[int] = None,
        worker_id: Optional[str] = None,
    ) -> dict:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(
                """INSERT INTO events (type, experiment_id, summary, details, parent_id, worker_id)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (
                    event_type,
                    experiment_id,
                    summary,
                    json.dumps(details) if details else None,
                    parent_id,
                    worker_id,
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
        proposal_id: str,
        title: str,
        content: str,
        priority: Optional[str] = None,
        hypothesis: Optional[str] = None,
        based_on: Optional[str] = None,
    ) -> dict:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute("DELETE FROM proposals WHERE proposal_id = %s", (proposal_id,))
            cur.execute(
                """INSERT INTO proposals
                   (proposal_id, title, priority, hypothesis, based_on, content)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (proposal_id, title, priority, hypothesis, based_on, content),
            )
            return dict(cur.fetchone())

    def list(self) -> list[dict]:
        return self.db._fetch(
            "SELECT * FROM proposals ORDER BY proposal_id"
        )

    def get(self, proposal_id: str) -> Optional[dict]:
        return self.db._fetch_one(
            "SELECT * FROM proposals WHERE proposal_id = %s",
            (proposal_id,),
        )


HEARTBEAT_TTL_MINUTES = 3  # Mark stale if no heartbeat for this long


class WorkersStore:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def register(
        self,
        worker_id: str,
        opencode_url: str,
        function_call_id: Optional[str] = None,
        timeout_hours: float = 8,
    ) -> dict:
        self.db._execute(
            """INSERT INTO workers (worker_id, opencode_url, function_call_id, status, timeout_hours)
               VALUES (%s, %s, %s, 'ready', %s)
               ON CONFLICT (worker_id) DO UPDATE SET
                   opencode_url = EXCLUDED.opencode_url,
                   function_call_id = EXCLUDED.function_call_id,
                   status = 'ready',
                   timeout_hours = EXCLUDED.timeout_hours,
                   last_heartbeat = now()""",
            (worker_id, opencode_url, function_call_id, timeout_hours),
        )
        return self.get(worker_id)

    def heartbeat(self, worker_id: str) -> Optional[dict]:
        self.db._execute(
            "UPDATE workers SET last_heartbeat = now(), status = 'ready' WHERE worker_id = %s",
            (worker_id,),
        )
        return self.get(worker_id)

    def get(self, worker_id: str) -> Optional[dict]:
        return self.db._fetch_one("SELECT * FROM workers WHERE worker_id = %s", (worker_id,))

    def list(self, include_stopped: bool = False) -> list[dict]:
        # Mark workers as stale if heartbeat is too old
        self.db._execute(
            f"""UPDATE workers SET status = 'stale'
                WHERE status = 'ready'
                AND last_heartbeat < now() - interval '{HEARTBEAT_TTL_MINUTES} minutes'""",
        )
        if include_stopped:
            return self.db._fetch("SELECT * FROM workers ORDER BY registered_at DESC")
        return self.db._fetch(
            "SELECT * FROM workers WHERE status != 'stopped' ORDER BY registered_at DESC"
        )

    def remove(self, worker_id: str) -> Optional[dict]:
        row = self.get(worker_id)
        if row:
            self.db._execute(
                "UPDATE workers SET status = 'stopped' WHERE worker_id = %s",
                (worker_id,),
            )
        return row

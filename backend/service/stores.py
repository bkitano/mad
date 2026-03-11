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

from service.db import DatabaseManager


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


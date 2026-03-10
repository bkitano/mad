"""
Postgres database layer for experiments, events, and claims.

Requires PG* env vars (Supabase connection string).
"""

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Optional, Union

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool


class DatabaseManager:
    """Manages database connections and operations via a thread-safe pool."""

    DEFAULT_POOL_MIN_SIZE = 1
    DEFAULT_POOL_MAX_SIZE = 10
    CLAIM_STALE_MINUTES = 30

    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        dbname: Optional[str] = None,
        pool_min_size: Optional[int] = None,
        pool_max_size: Optional[int] = None,
    ) -> None:
        self.connection_params = {
            "user": user or os.environ["PGUSER"],
            "password": password or os.environ["PGPASSWORD"],
            "host": host or os.environ["PGHOST"],
            "port": port or os.environ.get("PGPORT", "6543"),
            "dbname": dbname or os.environ.get("PGDATABASE", "postgres"),
        }
        self._pool: Optional[ThreadedConnectionPool] = None
        self._pool_lock = Lock()
        self._pool_min_size = pool_min_size or self.DEFAULT_POOL_MIN_SIZE
        self._pool_max_size = pool_max_size or self.DEFAULT_POOL_MAX_SIZE

    def _ensure_pool(self) -> ThreadedConnectionPool:
        if self._pool is None or self._pool.closed:
            with self._pool_lock:
                if self._pool is None or self._pool.closed:
                    self._pool = ThreadedConnectionPool(
                        self._pool_min_size,
                        self._pool_max_size,
                        **self.connection_params,
                    )
        return self._pool

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Yield a pooled database connection with automatic commit/rollback."""
        pool = self._ensure_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)

    def execute_query(
        self, query: str, params: Optional[tuple] = None, fetch: bool = True
    ) -> Union[list[dict], int, None]:
        """Execute a query and return results.

        For SELECT (fetch=True): returns list of dicts.
        For INSERT/UPDATE/DELETE (fetch=False): returns rowcount.
        """
        with self.get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cursor:
            cursor.execute(query, params)
            if fetch and cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            elif not fetch:
                return cursor.rowcount
            return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fetch(self, sql: str, params: tuple = ()) -> list[dict]:
        return self.execute_query(sql, params, fetch=True) or []

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._fetch(sql, params)
        return rows[0] if rows else None

    def _execute(self, sql: str, params: tuple = ()) -> int:
        result = self.execute_query(sql, params, fetch=False)
        return result if isinstance(result, int) else 0

    # ── Proposals ────────────────────────────────────────────────────────────

    def create_proposal(
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
        with self.get_connection() as conn, conn.cursor(
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

    # ── Experiments ──────────────────────────────────────────────────────────

    def create_experiment(
        self,
        experiment_id: str,
        proposal_id: str,
        agent_id: str = "",
        cost_estimate: Optional[float] = None,
    ) -> dict:
        self._execute(
            """INSERT INTO experiments
               (id, proposal_id, status, agent_id, cost_estimate)
               VALUES (%s, %s, 'created', %s, %s)""",
            (experiment_id, proposal_id, agent_id, cost_estimate),
        )
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        return self._fetch_one("SELECT * FROM experiments WHERE id = %s", (experiment_id,))

    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if status:
            return self._fetch(
                "SELECT * FROM experiments WHERE status = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (status, limit, offset),
            )
        return self._fetch(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )

    def update_experiment(self, experiment_id: str, **fields) -> Optional[dict]:
        if not fields:
            return self.get_experiment(experiment_id)

        for json_field in ("config", "results"):
            if json_field in fields and isinstance(fields[json_field], dict):
                fields[json_field] = json.dumps(fields[json_field])

        set_clause = ", ".join(f"{k} = %s" for k in fields)
        values = list(fields.values()) + [experiment_id]
        self._execute(f"UPDATE experiments SET {set_clause} WHERE id = %s", tuple(values))
        return self.get_experiment(experiment_id)

    # ── Events ───────────────────────────────────────────────────────────────

    def emit_event(
        self,
        event_type: str,
        summary: str,
        experiment_id: Optional[str] = None,
        agent: str = "",
        details: Optional[dict] = None,
        parent_id: Optional[int] = None,
    ) -> dict:
        with self.get_connection() as conn, conn.cursor(
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

    def list_events(
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

        return self._fetch(
            f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            tuple(params),
        )

    # ── Claims ───────────────────────────────────────────────────────────────

    def _clean_stale_claims(self, conn) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=self.CLAIM_STALE_MINUTES)).isoformat()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM claims WHERE status = 'active' AND heartbeat_at < %s",
                (cutoff,),
            )
            return cur.rowcount

    def claim_proposal(self, proposal_id: str, agent_id: str) -> bool:
        with self.get_connection() as conn:
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
        return self._execute(
            "UPDATE claims SET heartbeat_at = now(), details = COALESCE(%s, details) "
            "WHERE proposal_id = %s AND agent_id = %s AND status = 'active'",
            (details, proposal_id, agent_id),
        ) > 0

    def release_claim(self, proposal_id: str, agent_id: str, status: str = "completed") -> bool:
        return self._execute(
            "DELETE FROM claims WHERE proposal_id = %s AND agent_id = %s",
            (proposal_id, agent_id),
        ) > 0

    def list_claims(self, status: Optional[str] = "active") -> list[dict]:
        with self.get_connection() as conn:
            self._clean_stale_claims(conn)

        if status:
            return self._fetch(
                "SELECT * FROM claims WHERE status = %s ORDER BY claimed_at DESC",
                (status,),
            )
        return self._fetch("SELECT * FROM claims ORDER BY claimed_at DESC")

    def is_proposal_claimed(self, proposal_id: str) -> bool:
        with self.get_connection() as conn:
            self._clean_stale_claims(conn)

        row = self._fetch_one(
            "SELECT 1 FROM claims WHERE proposal_id = %s AND status = 'active'",
            (proposal_id,),
        )
        return row is not None


# ── Default instance & module-level aliases ──────────────────────────────────
# Lazily initialized so import doesn't fail if env vars aren't set yet.

_default: Optional[DatabaseManager] = None


def _get_default() -> DatabaseManager:
    global _default
    if _default is None:
        _default = DatabaseManager()
    return _default


def init_db() -> None:
    """No-op — tables created via migrations."""
    pass


# Expose all domain methods as module-level functions for backward compat.
def _fetch(sql, params=()):
    return _get_default()._fetch(sql, params)

def _fetch_one(sql, params=()):
    return _get_default()._fetch_one(sql, params)

def _execute(sql, params=()):
    return _get_default()._execute(sql, params)

def create_proposal(*a, **kw):
    return _get_default().create_proposal(*a, **kw)

def create_experiment(*a, **kw):
    return _get_default().create_experiment(*a, **kw)

def get_experiment(*a, **kw):
    return _get_default().get_experiment(*a, **kw)

def list_experiments(*a, **kw):
    return _get_default().list_experiments(*a, **kw)

def update_experiment(*a, **kw):
    return _get_default().update_experiment(*a, **kw)

def emit_event(*a, **kw):
    return _get_default().emit_event(*a, **kw)

def list_events(*a, **kw):
    return _get_default().list_events(*a, **kw)

def claim_proposal(*a, **kw):
    return _get_default().claim_proposal(*a, **kw)

def heartbeat_claim(*a, **kw):
    return _get_default().heartbeat_claim(*a, **kw)

def release_claim(*a, **kw):
    return _get_default().release_claim(*a, **kw)

def list_claims(*a, **kw):
    return _get_default().list_claims(*a, **kw)

def is_proposal_claimed(*a, **kw):
    return _get_default().is_proposal_claimed(*a, **kw)

"""Usage tracking for sandbox compute and volume storage."""

import os
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


# -- Sandbox usage -------------------------------------------------------------


def record_sandbox_start(
    user_id: str, sandbox_id: str, gpu: str = "", gpu_count: int = 1,
    cpu: float = 4.0, memory_mb: int = 8192,
) -> int:
    """Record that a sandbox was created. Returns the usage row ID."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sandbox_usage (user_id, sandbox_id, gpu, gpu_count, cpu, memory_mb) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (user_id, sandbox_id, gpu, gpu_count, cpu, memory_mb),
            )
            return cur.fetchone()[0]


def record_sandbox_stop(sandbox_id: str) -> bool:
    """Mark a sandbox as stopped. Returns True if a row was updated."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sandbox_usage SET stopped_at = %s WHERE sandbox_id = %s AND stopped_at IS NULL",
                (datetime.now(timezone.utc), sandbox_id),
            )
            return cur.rowcount > 0


def get_all_active_sandbox_ids() -> list[str]:
    """Return sandbox IDs for all sessions that haven't stopped yet (all users)."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT sandbox_id FROM sandbox_usage WHERE stopped_at IS NULL")
            return [row[0] for row in cur.fetchall()]


def get_active_sessions(user_id: str) -> list[dict]:
    """Return sandbox sessions that haven't stopped yet."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, sandbox_id, gpu, gpu_count, started_at "
                "FROM sandbox_usage WHERE user_id = %s AND stopped_at IS NULL "
                "ORDER BY started_at DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def get_usage_summary(user_id: str, since: datetime | None = None) -> dict:
    """Compute total GPU-seconds used since a given time.

    Returns {total_seconds, by_gpu: {gpu_type: seconds}}.
    For running sandboxes, counts up to now.
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            where = "user_id = %s"
            params: list = [user_id]
            if since:
                where += " AND COALESCE(stopped_at, now()) >= %s"
                params.append(since)

            cur.execute(
                f"SELECT gpu, gpu_count, "
                f"  EXTRACT(EPOCH FROM COALESCE(stopped_at, now()) - "
                f"    GREATEST(started_at, %s)) as seconds "
                f"FROM sandbox_usage WHERE {where}",
                [since or datetime(2000, 1, 1, tzinfo=timezone.utc)] + params,
            )

            total = 0.0
            by_gpu: dict[str, float] = {}
            for row in cur.fetchall():
                secs = max(0, float(row["seconds"])) * row["gpu_count"]
                total += secs
                key = row["gpu"] or "cpu"
                by_gpu[key] = by_gpu.get(key, 0) + secs

            return {"total_seconds": total, "by_gpu": by_gpu}

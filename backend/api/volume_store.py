"""Volume ownership tracking backed by Supabase Postgres."""

import os
from contextlib import contextmanager

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


def register_volume(volume_name: str, user_id: str) -> None:
    """Record that a volume belongs to a user. Idempotent (ON CONFLICT ignores)."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_volumes (volume_name, user_id) VALUES (%s, %s) ON CONFLICT (volume_name) DO NOTHING",
                (volume_name, user_id),
            )


def list_user_volumes(user_id: str) -> list[str]:
    """Return volume names owned by this user."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT volume_name FROM user_volumes WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            return [row[0] for row in cur.fetchall()]


def user_owns_volume(volume_name: str, user_id: str) -> bool:
    """Check if a user owns a specific volume."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_volumes WHERE volume_name = %s AND user_id = %s",
                (volume_name, user_id),
            )
            return cur.fetchone() is not None


def rename_volume(old_name: str, new_name: str) -> None:
    """Update the volume name in the ownership table."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_volumes SET volume_name = %s WHERE volume_name = %s",
                (new_name, old_name),
            )


def delete_volume(volume_name: str) -> None:
    """Remove ownership record for a deleted volume."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_volumes WHERE volume_name = %s",
                (volume_name,),
            )

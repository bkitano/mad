"""
Postgres connection pool manager.

Requires PG* env vars (Supabase connection string).
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock
from typing import Optional, Union

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool


class DatabaseManager:
    """Manages database connections and operations via a thread-safe pool."""

    DEFAULT_POOL_MIN_SIZE = 1
    DEFAULT_POOL_MAX_SIZE = 10

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

    def _fetch(self, sql: str, params: tuple = ()) -> list[dict]:
        return self.execute_query(sql, params, fetch=True) or []

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._fetch(sql, params)
        return rows[0] if rows else None

    def _execute(self, sql: str, params: tuple = ()) -> int:
        result = self.execute_query(sql, params, fetch=False)
        return result if isinstance(result, int) else 0

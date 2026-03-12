-- Migration: Create workers table for persistent worker registry with heartbeat TTL

CREATE TABLE IF NOT EXISTS workers (
    worker_id       TEXT PRIMARY KEY,
    opencode_url    TEXT NOT NULL,
    function_call_id TEXT,
    status          TEXT NOT NULL DEFAULT 'starting',  -- starting|ready|stale|stopped
    timeout_hours   NUMERIC NOT NULL DEFAULT 8,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_heartbeat ON workers(last_heartbeat);

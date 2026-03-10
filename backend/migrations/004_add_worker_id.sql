-- Migration: Add worker_id to experiments and events
--
-- worker_id references the worker registry (POST /workers/register).
-- Unlike the old agent_id, this is a real entity that maps to an opencode URL.

ALTER TABLE experiments ADD COLUMN IF NOT EXISTS worker_id TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS worker_id TEXT;

CREATE INDEX IF NOT EXISTS idx_experiments_worker ON experiments(worker_id);
CREATE INDEX IF NOT EXISTS idx_events_worker ON events(worker_id);

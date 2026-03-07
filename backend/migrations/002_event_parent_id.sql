-- Migration: Add parent_id to events for graph-like event chains.
--
-- Events can now reference a parent event, enabling reactive observer patterns:
-- e.g. an error event spawns a debugger agent, whose events are children of the original.

ALTER TABLE events ADD COLUMN IF NOT EXISTS parent_id BIGINT REFERENCES events(id);

CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_id);

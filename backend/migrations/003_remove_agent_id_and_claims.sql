-- Migration: Remove agent_id columns and claims table
--
-- agent_id was a phantom entity — generated at runtime, never validated,
-- never stored as a first-class record. Removing it simplifies the schema.
-- Claims table is also removed as it depended on agent_id for ownership.

-- ── Drop claims table ────────────────────────────────────────────────────────

DROP TABLE IF EXISTS claims;

-- ── Drop agent_id columns ────────────────────────────────────────────────────

ALTER TABLE experiments DROP COLUMN IF EXISTS agent_id;
ALTER TABLE events DROP COLUMN IF EXISTS agent_id;

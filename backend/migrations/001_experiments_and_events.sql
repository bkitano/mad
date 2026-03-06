-- Migration: Create experiments, events, and claims tables in Postgres (Supabase)
--
-- experiments: one-to-many from proposals, tracks each experiment run
-- events: generic event log with type field, supports Supabase Realtime SSE
-- claims: distributed work coordination
--
-- proposal_id stores the filename stem (e.g. "042-monarch-gated-state-transition")
-- matching proposals.filename minus the .md suffix.

-- ── Experiments ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,                  -- e.g. "042" or "042-r2" for reruns
    proposal_id     TEXT NOT NULL,                     -- filename stem, e.g. "042-monarch-gated-state-transition"
    status          TEXT NOT NULL DEFAULT 'created',   -- created|code_ready|submitted|running|completed|failed|cancelled
    agent_id        TEXT,

    -- Code tracking
    code_dir        TEXT,
    code_hash       TEXT,

    -- Modal execution
    modal_job_id    TEXT,
    modal_url       TEXT,

    -- Weights & Biases
    wandb_run_id    TEXT,
    wandb_url       TEXT,

    -- Config & results
    config          JSONB DEFAULT '{}',
    results         JSONB,

    -- Error tracking
    error           TEXT,
    error_class     TEXT,                              -- infra|code_bug|timeout|rate_limit|data
    retry_count     INT DEFAULT 0,

    -- Cost
    cost_estimate   NUMERIC,
    cost_actual     NUMERIC,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    submitted_at    TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_experiments_proposal ON experiments(proposal_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);

-- ── Events ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS events (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    type            TEXT NOT NULL,                     -- experiment.created|experiment.completed|worker.started|system.info|...
    experiment_id   TEXT,                              -- nullable: not all events are experiment-scoped
    agent_id        TEXT,
    summary         TEXT NOT NULL,
    details         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_experiment ON events(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);

-- ── Claims ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS claims (
    proposal_id     TEXT PRIMARY KEY,                  -- filename stem
    agent_id        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',    -- active|completed|failed|abandoned
    claimed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    heartbeat_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    details         JSONB
);

CREATE INDEX IF NOT EXISTS idx_claims_agent ON claims(agent_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);

-- ── Auto-update updated_at on experiments ───────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS experiments_updated_at ON experiments;
CREATE TRIGGER experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ── Enable Supabase Realtime on events table ────────────────────────────────
-- Run after migration:
-- ALTER PUBLICATION supabase_realtime ADD TABLE events;

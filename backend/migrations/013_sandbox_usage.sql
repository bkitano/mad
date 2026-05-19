-- Track sandbox compute sessions for usage metering.
CREATE TABLE IF NOT EXISTS sandbox_usage (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    sandbox_id  TEXT NOT NULL,
    gpu         TEXT NOT NULL DEFAULT '',
    gpu_count   INT NOT NULL DEFAULT 1,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    stopped_at  TIMESTAMPTZ
);

CREATE INDEX idx_sandbox_usage_user_id ON sandbox_usage (user_id);
CREATE INDEX idx_sandbox_usage_sandbox_id ON sandbox_usage (sandbox_id);

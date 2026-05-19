-- Periodic snapshots of volume sizes for storage billing.
CREATE TABLE IF NOT EXISTS volume_snapshots (
    id           SERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL,
    volume_name  TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    file_count   INT NOT NULL DEFAULT 0,
    measured_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_volume_snapshots_user_id ON volume_snapshots (user_id);
CREATE INDEX idx_volume_snapshots_measured_at ON volume_snapshots (measured_at);

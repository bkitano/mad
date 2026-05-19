-- Track volume ownership per user.
-- Modal volumes are global; this table maps them to the user who created them.
CREATE TABLE IF NOT EXISTS user_volumes (
    volume_name TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_volumes_user_id ON user_volumes (user_id);

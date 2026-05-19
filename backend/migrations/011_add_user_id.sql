-- Add user_id to chat_sessions for multi-tenancy
ALTER TABLE chat_sessions ADD COLUMN user_id text;

-- Index for fast user-scoped queries
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions (user_id);

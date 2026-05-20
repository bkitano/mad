-- Rebuild chat persistence: a single append-only event log is the source of
-- truth for everything rendered in the UI and everything fed back to the LLM.
--
-- One row per UI part (text block, reasoning block, tool call, step boundary,
-- user submission). Grouping rows by message_id, in seq order, yields the
-- exact AI SDK UIMessage[] structure the frontend renders.

DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS chat_events CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;

CREATE TABLE chat_sessions (
    id          text PRIMARY KEY,
    title       text NOT NULL DEFAULT 'New chat',
    type        text NOT NULL DEFAULT 'text',
    user_id     text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_sessions_user_id     ON chat_sessions (user_id);
CREATE INDEX idx_chat_sessions_updated_at  ON chat_sessions (updated_at DESC);

CREATE TABLE chat_events (
    id          bigserial PRIMARY KEY,
    session_id  text NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    message_id  text NOT NULL,
    seq         integer NOT NULL,
    role        text NOT NULL,
    event       jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);

CREATE INDEX idx_chat_events_session_seq ON chat_events (session_id, seq);

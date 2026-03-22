-- Migration: Create tricks table for storing mathematical/algorithmic tricks

CREATE TABLE IF NOT EXISTS tricks (
    id              INTEGER PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,
    gain_type       TEXT NOT NULL,
    source          TEXT,
    paper           TEXT,
    documented      DATE,
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tricks_category ON tricks(category);
CREATE INDEX IF NOT EXISTS idx_tricks_gain_type ON tricks(gain_type);
CREATE INDEX IF NOT EXISTS idx_tricks_slug ON tricks(slug);

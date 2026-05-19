-- Add CPU and memory columns to sandbox_usage for accurate billing.
ALTER TABLE sandbox_usage ADD COLUMN cpu REAL NOT NULL DEFAULT 4.0;
ALTER TABLE sandbox_usage ADD COLUMN memory_mb INT NOT NULL DEFAULT 32768;
